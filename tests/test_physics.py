"""P1.1 acceptance: simulator satisfies its own healthy equations, matches the
closed-form frequency response, and natural frequencies sit in [80, 400] Hz."""
import numpy as np
import pytest

from src.physics import (
    FaultKnobs, RotorParams, frequency_response_x2, natural_frequencies_hz,
    residuals, sample_knobs, set_speed_coupling, simulate,
)

FS, T = 50_000, 3014


@pytest.fixture(scope="module")
def healthy_sim():
    p = RotorParams()
    omega = np.array([2 * np.pi * 16.0])
    out = simulate(p, omega, FaultKnobs.healthy(1), T=T, fs=FS)
    return p, omega, out


def test_natural_frequencies_in_band():
    f = natural_frequencies_hz(RotorParams())
    assert np.all(f > 80) and np.all(f < 400), f


def test_healthy_residuals_near_zero(healthy_sim):
    p, omega, out = healthy_sim
    t = np.arange(T) / FS
    r = residuals(out["pos"], out["vel"], out["acc"], omega[:, None], t[None, :], p)
    # residual scale relative to the forcing amplitude
    F0 = p.M1 * omega[0] ** 2 * p.E1
    rel = np.abs(r).max() / F0
    assert rel < 1e-6, f"healthy residual relative magnitude {rel:.2e}"


def test_steady_state_matches_frequency_response(healthy_sim):
    p, omega, out = healthy_sim
    x2 = out["pos"][0, :, 0]
    amp_sim = (x2.max() - x2.min()) / 2
    amp_ref = np.abs(frequency_response_x2(p, omega[0]))
    assert abs(amp_sim - amp_ref) / amp_ref < 0.01, (amp_sim, amp_ref)


def test_fault_residuals_dominate_healthy(healthy_sim):
    p, omega, out = healthy_sim
    t = np.arange(T) / FS
    r_h = residuals(out["pos"], out["vel"], out["acc"], omega[:, None], t[None, :], p)
    rng = np.random.default_rng(0)
    for cls in ("imbalance_uni", "bpfo_impulsive", "misalign_cont"):
        knobs, _ = sample_knobs(cls, 1, rng)
        set_speed_coupling(knobs, 16.0)
        o = simulate(p, omega, knobs, T=T, fs=FS)
        r_f = residuals(o["pos"], o["vel"], o["acc"], omega[:, None], t[None, :], p)
        ratio = np.abs(r_f).max() / max(np.abs(r_h).max(), 1e-30)
        assert ratio > 100, f"{cls}: fault/healthy residual ratio {ratio:.1f}"


def test_simulation_determinism():
    p = RotorParams()
    omega = np.array([2 * np.pi * 16.0])
    knobs, _ = sample_knobs("imbalance_uni", 1, np.random.default_rng(7))
    a = simulate(p, omega, knobs, T=256, fs=FS)["acc"]
    b = simulate(p, omega, knobs, T=256, fs=FS)["acc"]
    assert np.array_equal(a, b)
