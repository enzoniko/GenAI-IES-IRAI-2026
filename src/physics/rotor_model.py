"""Single source of truth for the mafaulda_synthetic rotor physics.

A 2-mass, 2-plane lumped rotor-bearing model in deviation coordinates around
static equilibrium (gravity cancels exactly by construction):

    M2*x2'' + D2*x2' + K1x(t)*x2 + Kc*(x2 - x3) = F_ub_x(t) + F_2x_x(t)
    M2*y2'' + D2*y2' + K1y(t)*y2 + Kc*(y2 - y3) = F_ub_y(t) + F_2x_y(t)
    M3*x3'' + D3*x3' + K2*x3    + Kc*(x3 - x2) = 0
    M3*y3'' + D3*y3' + K2*y3    + Kc*(y3 - y2) = F_imp(t)

Healthy regime: kappa=1, phi0=0, no 2X forcing, isotropic constant K1, no
impacts. The PINN's physics residuals are THESE equations with the healthy
forcing assumption — both the simulator and the PINN loss import them from
here, so agreement is guaranteed by construction, not by hope.

Observables (the 4 "accelerometer" channels): x2'', y2'', x3'', y3''.
"""
from __future__ import annotations

from dataclasses import dataclass, replace as dc_replace

import numpy as np


@dataclass(frozen=True)
class RotorParams:
    M1: float = 50.0        # rotor mass (drives unbalance force amplitude)
    M2: float = 3.5         # underhang bearing lumped mass
    M3: float = 3.5         # overhang bearing lumped mass
    D2: float = 300.0       # light damping so structural modes are visible
    D3: float = 300.0
    K1: float = 3.4635e6
    K2: float = 3.8127e6
    Kc: float = 1.0e6       # shaft coupling stiffness
    E1: float = 5.0e-6      # nominal eccentricity [m]

    @staticmethod
    def variant(name: str) -> "RotorParams":
        p = RotorParams()
        if name == "A":
            return p
        if name == "B":  # detuned "different machine" for cross-dataset E4
            return dc_replace(p, M2=4.5, M3=2.8, D2=420.0, D3=250.0,
                              K1=2.6e6, K2=4.9e6, Kc=1.4e6, E1=6.5e-6)
        raise ValueError(f"Unknown rotor variant '{name}'")


def natural_frequencies_hz(p: RotorParams) -> np.ndarray:
    """Undamped natural frequencies of one plane (x), in Hz."""
    K = np.array([[p.K1 + p.Kc, -p.Kc], [-p.Kc, p.K2 + p.Kc]])
    M = np.diag([p.M2, p.M3])
    evals = np.linalg.eigvals(np.linalg.solve(M, K))
    return np.sort(np.sqrt(np.real(evals))) / (2 * np.pi)


def frequency_response_x2(p: RotorParams, omega: float) -> complex:
    """Closed-form steady-state complex amplitude of x2 under unit-kappa
    unbalance forcing at rotation speed `omega` (rad/s)."""
    F0 = p.M1 * omega**2 * p.E1
    A = np.array([
        [p.K1 + p.Kc - p.M2 * omega**2 + 1j * omega * p.D2, -p.Kc],
        [-p.Kc, p.K2 + p.Kc - p.M3 * omega**2 + 1j * omega * p.D3],
    ])
    X = np.linalg.solve(A, np.array([F0, 0.0]))
    return X[0]


# ---------------------------------------------------------------------------
# Fault knobs consumed by the simulator (all healthy by default)
# ---------------------------------------------------------------------------
@dataclass
class FaultKnobs:
    """Per-window arrays of shape (B,). Defaults reproduce the healthy regime."""
    kappa: np.ndarray            # unbalance multiplier (healthy = 1)
    phi0: np.ndarray             # unbalance phase offset (healthy = 0)
    two_x_frac: np.ndarray       # 2X forcing fraction of nominal unbalance force
    k1y_scale: np.ndarray        # static stiffness anisotropy on K1y (healthy = 1)
    drop_depth: np.ndarray       # intermittent K1y dropout depth in [0,1)
    drop_duty: np.ndarray        # fraction of each rotation the dropout is active
    imp_amp: np.ndarray          # impact force amplitude [N] (healthy = 0)
    imp_freq: np.ndarray         # impact repetition frequency [Hz] (BPFO)
    imp_res: np.ndarray          # excited resonance [Hz]
    imp_decay: np.ndarray        # impact decay rate [1/s]

    @staticmethod
    def healthy(B: int) -> "FaultKnobs":
        z = np.zeros(B)
        return FaultKnobs(
            kappa=np.ones(B), phi0=z.copy(), two_x_frac=z.copy(),
            k1y_scale=np.ones(B), drop_depth=z.copy(), drop_duty=z.copy(),
            imp_amp=z.copy(), imp_freq=np.full(B, 48.0),
            imp_res=np.full(B, 2500.0), imp_decay=np.full(B, 1000.0),
        )


def _forces(t: float, p: RotorParams, omega: np.ndarray, k: FaultKnobs):
    """External forcing and instantaneous K1y at absolute time t. Vectorized (B,)."""
    F0 = p.M1 * omega**2 * p.E1
    wt = omega * t
    f_ub_x = F0 * k.kappa * np.cos(wt + k.phi0)
    f_ub_y = F0 * k.kappa * np.sin(wt + k.phi0)
    f2x = k.two_x_frac * F0
    f_ub_x = f_ub_x + f2x * np.cos(2 * wt)
    f_ub_y = f_ub_y + f2x * np.sin(2 * wt)
    # intermittent stiffness dropout gated on rotation phase
    phase = (wt / (2 * np.pi)) % 1.0
    gate = (phase < k.drop_duty).astype(np.float64)
    k1y = p.K1 * k.k1y_scale * (1.0 - k.drop_depth * gate)
    # bearing impact train (nearest previous impact dominates: decay*T >> 1)
    ph = np.mod(t, 1.0 / k.imp_freq)
    f_imp = k.imp_amp * np.exp(-k.imp_decay * ph) * np.sin(2 * np.pi * k.imp_res * ph)
    return f_ub_x, f_ub_y, f_imp, k1y


def _deriv(t: float, s: np.ndarray, p: RotorParams, omega: np.ndarray, k: FaultKnobs):
    """State derivative. s: (B, 8) = [x2,y2,x3,y3,vx2,vy2,vx3,vy3]."""
    x2, y2, x3, y3 = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
    vx2, vy2, vx3, vy3 = s[:, 4], s[:, 5], s[:, 6], s[:, 7]
    f_ub_x, f_ub_y, f_imp, k1y = _forces(t, p, omega, k)
    ax2 = (-p.D2 * vx2 - p.K1 * x2 - p.Kc * (x2 - x3) + f_ub_x) / p.M2
    ay2 = (-p.D2 * vy2 - k1y * y2 - p.Kc * (y2 - y3) + f_ub_y) / p.M2
    ax3 = (-p.D3 * vx3 - p.K2 * x3 - p.Kc * (x3 - x2)) / p.M3
    ay3 = (-p.D3 * vy3 - p.K2 * y3 - p.Kc * (y3 - y2) + f_imp) / p.M3
    return np.stack([vx2, vy2, vx3, vy3, ax2, ay2, ax3, ay3], axis=1)


def _solve_2x2(a, b, c, d, f1, f2):
    """Vectorized complex 2x2 solve: [[a,b],[c,d]] @ [X1,X2] = [f1,f2]."""
    det = a * d - b * c
    return (d * f1 - b * f2) / det, (a * f2 - c * f1) / det


def _steady_state_init(p: RotorParams, omega: np.ndarray, k: FaultKnobs, t0: float):
    """Closed-form LTI steady state at time t0 for the harmonic forcing
    components (1X unbalance with kappa/phi0, 2X misalignment forcing, static
    K1y anisotropy). Impacts/dropouts are aperiodic and rely on warmup.
    Eliminates the startup transient that would otherwise corrupt the
    synchronous kinematics recovery downstream."""
    F0 = p.M1 * omega**2 * p.E1
    s0 = np.zeros((omega.shape[0], 8))
    k1y_static = p.K1 * k.k1y_scale
    for (k1_eff, amp_1x, amp_2x) in (
        # plane x: cos forcing -> complex amp = F; plane y: sin -> -i * F
        (np.full_like(omega, p.K1), F0 * k.kappa * np.exp(1j * k.phi0), k.two_x_frac * F0),
        (k1y_static, -1j * F0 * k.kappa * np.exp(1j * k.phi0), -1j * k.two_x_frac * F0),
    ):
        plane = 0 if k1_eff is not k1y_static else 1
        x2c = np.zeros_like(omega, dtype=complex)
        x3c = np.zeros_like(omega, dtype=complex)
        v2c = np.zeros_like(omega, dtype=complex)
        v3c = np.zeros_like(omega, dtype=complex)
        for mult, amp in ((1.0, amp_1x), (2.0, amp_2x)):
            w = mult * omega
            X2, X3 = _solve_2x2(
                k1_eff + p.Kc - p.M2 * w**2 + 1j * w * p.D2, -p.Kc * np.ones_like(w),
                -p.Kc * np.ones_like(w), p.K2 + p.Kc - p.M3 * w**2 + 1j * w * p.D3,
                amp, np.zeros_like(w),
            )
            ph = np.exp(1j * w * t0)
            x2c, x3c = x2c + X2 * ph, x3c + X3 * ph
            v2c, v3c = v2c + 1j * w * X2 * ph, v3c + 1j * w * X3 * ph
        s0[:, 0 + plane] = np.real(x2c)   # x2 or y2
        s0[:, 2 + plane] = np.real(x3c)   # x3 or y3
        s0[:, 4 + plane] = np.real(v2c)
        s0[:, 6 + plane] = np.real(v3c)
    return s0


def simulate(
    p: RotorParams,
    omega: np.ndarray,          # (B,) rad/s
    knobs: FaultKnobs,
    T: int,
    fs: int,
    warmup_s: float = 0.05,
    oversample: int = 2,
) -> dict[str, np.ndarray]:
    """Batched fixed-step RK4 integration.

    Windows are keyphasor-aligned: t=0 (start of the recorded window) has
    rotation phase 0, matching the PINN's healthy-forcing time reference.
    Integration starts from the harmonic steady state at negative time; the
    warmup covers the aperiodic fault content (impacts, dropouts).

    Returns dict with pos, vel, acc arrays of shape (B, T, 4), float64.
    """
    B = omega.shape[0]
    dt = 1.0 / (fs * oversample)
    n_warm = int(round(warmup_s * fs * oversample))
    n_rec = T * oversample
    s = _steady_state_init(p, omega, knobs, t0=-n_warm * dt)
    pos = np.empty((B, T, 4))
    vel = np.empty((B, T, 4))
    acc = np.empty((B, T, 4))
    t = -n_warm * dt
    out_i = 0
    for step in range(n_warm + n_rec):
        if step >= n_warm and (step - n_warm) % oversample == 0:
            d = _deriv(t, s, p, omega, knobs)
            pos[:, out_i] = s[:, :4]
            vel[:, out_i] = s[:, 4:]
            acc[:, out_i] = d[:, 4:]
            out_i += 1
        k1 = _deriv(t, s, p, omega, knobs)
        k2 = _deriv(t + dt / 2, s + dt / 2 * k1, p, omega, knobs)
        k3 = _deriv(t + dt / 2, s + dt / 2 * k2, p, omega, knobs)
        k4 = _deriv(t + dt, s + dt * k3, p, omega, knobs)
        s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
    assert out_i == T
    return {"pos": pos, "vel": vel, "acc": acc}


# ---------------------------------------------------------------------------
# Healthy-physics residuals (shared by simulator tests and the PINN loss)
# ---------------------------------------------------------------------------
def residuals(pos, vel, acc, omega, t, p: RotorParams):
    """Residuals of the HEALTHY equations (kappa=1, phi0=0, no impacts,
    isotropic K1). Works on numpy arrays and torch tensors alike.

    pos, vel, acc: (..., 4) with channel order [x2, y2, x3, y3] (pos/vel) and
                   [x2'', y2'', x3'', y3''] (acc)
    omega, t:      broadcastable to (...)
    Returns (..., 4) residual array [r1, r2, r3, r4] in force units [N].
    """
    xp = np if isinstance(pos, np.ndarray) else __import__("torch")
    x2, y2, x3, y3 = pos[..., 0], pos[..., 1], pos[..., 2], pos[..., 3]
    vx2, vy2, vx3, vy3 = vel[..., 0], vel[..., 1], vel[..., 2], vel[..., 3]
    ax2, ay2, ax3, ay3 = acc[..., 0], acc[..., 1], acc[..., 2], acc[..., 3]
    F0 = p.M1 * omega**2 * p.E1
    wt = omega * t
    r1 = p.M2 * ax2 + p.D2 * vx2 + p.K1 * x2 + p.Kc * (x2 - x3) - F0 * xp.cos(wt)
    r2 = p.M2 * ay2 + p.D2 * vy2 + p.K1 * y2 + p.Kc * (y2 - y3) - F0 * xp.sin(wt)
    r3 = p.M3 * ax3 + p.D3 * vx3 + p.K2 * x3 + p.Kc * (x3 - x2)
    r4 = p.M3 * ay3 + p.D3 * vy3 + p.K2 * y3 + p.Kc * (y3 - y2)
    if xp is np:
        return np.stack([r1, r2, r3, r4], axis=-1)
    return xp.stack([r1, r2, r3, r4], dim=-1)
