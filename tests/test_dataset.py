"""P1.2 acceptance: determinism, contract compliance, spectral signatures."""
import numpy as np
import pytest
import torch

from src.configs import DataCfg
from src.data import get_dataset
from src.data.mafaulda_synthetic import generate

SMOKE = DataCfg(T=1024, speeds_hz=(16.0,), n_train=6, n_val=4, n_test=4,
                version="testtmp", seed=0)


@pytest.fixture(scope="module")
def bundle():
    return get_dataset(SMOKE)


def test_determinism(tmp_path):
    cfg1 = DataCfg(T=512, speeds_hz=(16.0,), n_train=3, n_val=2, n_test=2,
                   version="det1", root=str(tmp_path / "a"))
    cfg2 = DataCfg(T=512, speeds_hz=(16.0,), n_train=3, n_val=2, n_test=2,
                   version="det1", root=str(tmp_path / "b"))
    d1, d2 = generate(cfg1, verbose=False), generate(cfg2, verbose=False)
    a = torch.load(d1 / "train.pth", weights_only=True)
    b = torch.load(d2 / "train.pth", weights_only=True)
    assert torch.equal(a["raw"], b["raw"]) and torch.equal(a["label"], b["label"])


def test_contract_and_shapes(bundle):
    a = bundle.arrays("train")
    n_cls = len(SMOKE.classes)
    assert a["raw"].shape == (6 * n_cls, 4, 1024)
    assert a["jitter_phys"] is not None and bundle.meta["has_ground_truth"]
    batch = next(iter(bundle.loader("val", batch_size=8, shuffle=True)))
    assert batch.raw.shape[1:] == (4, 1024) and batch.label.dtype == torch.long
    # normalized healthy stats near (0, 1)
    h = bundle.arrays("train", classes=[0])["raw"]
    assert abs(h.mean().item()) < 0.15 and 0.7 < h.std().item() < 1.3


def _spectrum(x: torch.Tensor, fs: int):
    mag = torch.fft.rfft(x.double(), dim=-1).abs().mean(dim=(0, 1)).numpy()
    freqs = np.fft.rfftfreq(x.shape[-1], 1 / fs)
    return freqs, mag


def test_spectral_signatures():
    # 1X = 16 Hz needs fine frequency resolution: T=8192 -> 6.1 Hz/bin
    cfg = DataCfg(T=8192, speeds_hz=(16.0,), n_train=3, n_val=1, n_test=1,
                  version="testspec", seed=0)
    b = get_dataset(cfg)
    names = list(cfg.classes)

    def band_energy(ci, lo, hi):
        x = b.arrays("train", classes=[ci])["raw_phys"]
        freqs, mag = _spectrum(x, cfg.fs)
        return mag[(freqs >= lo) & (freqs < hi)].sum()

    # class 1 (imbalance) boosts the 1X line (~16 Hz) vs healthy
    e1 = band_energy(names.index("imbalance_uni"), 10, 24)
    e0 = band_energy(0, 10, 24)
    assert e1 > 3 * e0, (e1, e0)
    # class 3 (BPFO) adds energy in the 2-3 kHz resonance band
    e3 = band_energy(names.index("bpfo_impulsive"), 2000, 3000)
    e0b = band_energy(0, 2000, 3000)
    assert e3 > 3 * e0b, (e3, e0b)
    # class 4 (misalignment) shows 2X (~32 Hz) energy above healthy
    e4 = band_energy(names.index("misalign_cont"), 26, 42)
    e0c = band_energy(0, 26, 42)
    assert e4 > 2 * e0c, (e4, e0c)


def test_fewshot_sets(bundle):
    fs = bundle.fewshot_fault_sets(3, seed=1)
    assert set(fs.keys()) == set(range(1, len(SMOKE.classes)))
    assert fs[1]["raw_phys"].shape == (3, 4, 1024)


@pytest.mark.parametrize("speed", [12.0, 16.0, 20.0])
def test_kinematics_consistency(speed):
    """Synchronously-derived vel/pos from clean acc keep healthy residuals small
    at every operating speed (window covers >= 1 revolution)."""
    from src.data.kinematics import derive_kinematics
    from src.physics import FaultKnobs, RotorParams, residuals, simulate

    p = RotorParams()
    T, fs = 4167, 50_000
    omega = np.array([2 * np.pi * speed])
    out = simulate(p, omega, FaultKnobs.healthy(1), T=T, fs=fs)
    acc = torch.from_numpy(out["acc"]).permute(0, 2, 1)     # (1, 4, T)
    vel, pos = derive_kinematics(acc, fs=fs, omega=torch.from_numpy(omega))
    t = torch.arange(T, dtype=torch.float64) / fs
    r = residuals(pos.permute(0, 2, 1), vel.permute(0, 2, 1), acc.permute(0, 2, 1),
                  torch.from_numpy(omega).view(1, 1), t.view(1, -1), p)
    F0 = p.M1 * omega[0] ** 2 * p.E1
    rel = r.abs().max().item() / F0
    assert rel < 0.05, f"speed {speed}: derived-kinematics residual rel={rel:.3f}"
