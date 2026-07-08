"""Unit tests: model shapes, differentiability, anti-collapse machinery."""
import numpy as np
import torch

from src.models import (
    DDIMSampler, DDPMScheduler, Decoder1, Decoder2CVAE, LatentDiffusionMLP,
    MathFeatureExtractor, Normalization, RotorPINN, TSJEPA, epps_pulley,
    estimate_mi,
)

T = 512


def test_tsjepa_loss_and_collapse_metrics():
    m = TSJEPA(patch_size=16, num_layers=2, dim_feedforward=64, sigreg_coef=0.1)
    x = torch.randn(8, 4, T)
    out = m.loss(x)
    assert torch.isfinite(out["total"])
    assert {"pred", "var", "cov", "sigreg"} <= set(out)
    z = m.get_z_macro(x)
    assert z.shape == (8, 128)
    cm = m.collapse_metrics(z)
    assert cm["effective_rank"] > 1


def test_epps_pulley_orders_distributions():
    g = torch.Generator().manual_seed(0)
    z_gauss = torch.randn(512, 16, generator=g)
    z_exp = torch.distributions.Exponential(1.0).sample((512, 16)) - 1.0
    assert epps_pulley(z_gauss) < epps_pulley(z_exp)


def test_decoders_shapes_and_grad():
    d1 = Decoder1(seq_length=T)
    z = torch.randn(4, 128, requires_grad=True)
    y = d1(z)
    assert y.shape == (4, 4, T)
    y.sum().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()

    d2 = Decoder2CVAE(seq_length=T, num_classes=7, beta_tc=1.0, latent_dim=16)
    res = torch.randn(4, 4, T)
    zm = torch.randn(4, 128)
    lab = torch.tensor([0, 1, 2, 3])
    out = d2.loss(res, zm, lab)
    assert torch.isfinite(out["total"]) and "tc" in out
    s = d2.sample(zm, lab)
    assert s.shape == (4, 4, T)


def test_ddim_sampler():
    torch.manual_seed(0)
    model = LatentDiffusionMLP(z_dim=16, num_classes=7)
    sch = DDPMScheduler(num_train_timesteps=100)
    ddim = DDIMSampler(sch)
    z = ddim.sample(model, n=6, z_dim=16, device="cpu", n_steps=10,
                    label=torch.zeros(6, dtype=torch.long))
    assert z.shape == (6, 16) and torch.isfinite(z).all()
    ts = ddim.schedule(50, 10)
    assert ts[0] == 49 and ts[-1] == 0 and all(a > b for a, b in zip(ts, ts[1:]))


def test_rotor_pinn_forward_and_residual_grad():
    norm = Normalization(torch.zeros(10), torch.ones(10), torch.zeros(4), torch.ones(4))
    pinn = RotorPINN(norm, hidden_layers=(32, 32))
    x = torch.randn(16, 10, dtype=torch.float64, requires_grad=True)
    acc_n, forces = pinn(x)
    r = pinn.compute_residuals(x, acc_n, forces)
    assert r.shape == (16, 4)
    r.pow(2).mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_feature_extractor_grad_flow():
    fe = MathFeatureExtractor(in_channels=12, fft_bins=64, wavelet_levels=3)
    x = torch.randn(2, 400, 12, requires_grad=True)
    v = fe(x)
    assert v.shape == (2, fe.output_dim)
    v.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_mine_sanity_correlated_gaussians():
    rng = np.random.default_rng(0)
    n, rho = 4000, 0.9
    x = rng.normal(size=(n, 1))
    y = rho * x + np.sqrt(1 - rho**2) * rng.normal(size=(n, 1))
    true_mi = -0.5 * np.log(1 - rho**2)          # 0.83 nats
    est = estimate_mi(torch.from_numpy(x), torch.from_numpy(y), epochs=400, seed=0)
    assert abs(est - true_mi) < 0.25, (est, true_mi)
    # independent variables -> near zero
    est0 = estimate_mi(torch.from_numpy(x), torch.from_numpy(rng.normal(size=(n, 1))),
                       epochs=400, seed=0)
    assert est0 < 0.1, est0
