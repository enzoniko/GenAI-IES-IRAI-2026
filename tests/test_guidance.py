"""P3.3 acceptance: guidance densities fit, differentiate, and order correctly
on designed toy data (bimodal favors GMM/Flow over Gaussian)."""
import torch

from src.configs import GuidanceCfg
from src.oracle import FlowGuidance, GMMGuidance, GaussianGuidance, make_guidance


def _bimodal(n, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(n // 2, d, generator=g) * 0.3 + 3.0
    b = torch.randn(n - n // 2, d, generator=g) * 0.3 - 3.0
    return torch.cat([a, b])


def test_gaussian_penalty_differentiable():
    V = torch.randn(64, 8)
    gd = GaussianGuidance().fit(V)
    v = torch.randn(4, 8, requires_grad=True)
    p = gd.penalty(v)
    p.backward()
    assert v.grad is not None and torch.isfinite(v.grad).all()


def test_gmm_and_flow_beat_gaussian_on_bimodal():
    Vtr, Vte = _bimodal(400, seed=0), _bimodal(200, seed=1)
    ll = {}
    for est in (GaussianGuidance(), GMMGuidance(max_k=4),
                FlowGuidance(epochs=200, min_fit_samples=20)):
        est.fit(Vtr, seed=0)
        ll[est.kind] = est.held_out_ll(Vte)
    assert ll["gmm"] > ll["gaussian"] + 1.0, ll
    assert ll["flow"] > ll["gaussian"] + 1.0, ll


def test_gmm_selects_two_components():
    gm = GMMGuidance(max_k=4).fit(_bimodal(400))
    assert gm.chosen_k >= 2


def test_flow_fallback_few_shot():
    fl = FlowGuidance(min_fit_samples=20).fit(_bimodal(5))
    assert fl.fallback is not None
    v = torch.randn(3, 8, requires_grad=True)
    fl.penalty(v).backward()
    assert torch.isfinite(v.grad).all()


def test_fit_n5_no_crash_all_estimators():
    V = torch.randn(5, 8)
    for kind in ("gaussian", "gmm", "flow"):
        est = make_guidance(kind, GuidanceCfg())
        est.fit(V)
        assert torch.isfinite(est.log_prob(torch.randn(2, 8))).all()


def test_penalty_gradients_all_estimators():
    Vtr = _bimodal(200)
    for kind in ("gaussian", "gmm", "flow"):
        est = make_guidance(kind, GuidanceCfg(flow_epochs=50))
        est.fit(Vtr)
        v = torch.zeros(2, 8, requires_grad=True)
        est.penalty(v).backward()
        assert v.grad is not None and torch.isfinite(v.grad).all(), kind
