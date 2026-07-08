"""Geometry-dial fault classes for mafaulda_synthetic.

Each class = a physical mechanism + a SAMPLING LAW over its fault parameters.
The sampling law is what shapes the class's cluster geometry in the
physics-informed embedding space (the independent variable of hypothesis H):

  0 healthy          nominal, tiny param jitter            -> tight unimodal
  1 imbalance_uni    kappa ~ N(6, 0.3) trunc               -> unimodal near-Gaussian (low G)
  2 imbalance_bi     (kappa,phi0) ~ half N((4,0)) + half N((9,pi/2))  -> bimodal (high G)
  3 bpfo_impulsive   impact amp ~ LogNormal                -> heavy-tailed impulsive (high G)
  4 misalign_cont    severity s ~ U(0.05,0.5) (K1y aniso + 2X) -> elongated continuum
  5 looseness_skew   dropout duty ~ Beta(2,8)              -> skewed unimodal
  6 combo_ring       phi0 ~ U(0,2pi), kappa fixed 6        -> ring manifold (high G)

Classes 1 and 2 share the mechanism and differ ONLY in the sampling law —
the matched pair at the heart of the hypothesis test. Do not "improve"
class 2's realism.
"""
from __future__ import annotations

import numpy as np

from .rotor_model import FaultKnobs

CLASS_NAMES = (
    "healthy", "imbalance_uni", "imbalance_bi", "bpfo_impulsive",
    "misalign_cont", "looseness_skew", "combo_ring",
)


def sample_knobs(class_name: str, B: int, rng: np.random.Generator) -> tuple[FaultKnobs, dict]:
    """Draw per-window fault parameters for `class_name`.

    Returns (knobs, params_dict) where params_dict holds the ground-truth
    sampled parameters (persisted with the dataset for later analysis).
    """
    k = FaultKnobs.healthy(B)
    params: dict[str, np.ndarray] = {}

    if class_name == "healthy":
        pass  # nominal; measurement noise supplies within-class variation

    elif class_name == "imbalance_uni":
        kappa = rng.normal(6.0, 0.3, B)
        k.kappa = np.clip(kappa, 1.5, None)
        params["kappa"] = k.kappa

    elif class_name == "imbalance_bi":
        pick = rng.random(B) < 0.5
        kappa = np.where(pick, rng.normal(4.0, 0.25, B), rng.normal(9.0, 0.25, B))
        phi0 = np.where(pick, rng.normal(0.0, 0.05, B), rng.normal(np.pi / 2, 0.05, B))
        k.kappa = np.clip(kappa, 1.5, None)
        k.phi0 = phi0
        params["kappa"], params["phi0"], params["mode"] = k.kappa, phi0, pick.astype(int)

    elif class_name == "bpfo_impulsive":
        k.imp_amp = rng.lognormal(mean=np.log(2.0), sigma=0.5, size=B)
        params["imp_amp"] = k.imp_amp

    elif class_name == "misalign_cont":
        s = rng.uniform(0.05, 0.5, B)
        k.k1y_scale = 1.0 + s
        k.two_x_frac = 1.5 * s
        params["severity"] = s

    elif class_name == "looseness_skew":
        duty = rng.beta(2.0, 8.0, B)
        k.drop_duty = duty
        k.drop_depth = np.full(B, 0.4)
        params["duty"] = duty

    elif class_name == "combo_ring":
        k.kappa = np.full(B, 6.0)
        k.phi0 = rng.uniform(0.0, 2 * np.pi, B)
        params["phi0"] = k.phi0

    else:
        raise ValueError(f"Unknown fault class '{class_name}'")

    # BPFO repetition frequency tracks rotation speed (set later, needs omega);
    # imp_freq is refreshed by the dataset generator via `set_speed_coupling`.
    return k, params


def set_speed_coupling(k: FaultKnobs, speed_hz: float) -> None:
    """Couple speed-dependent knobs: BPFO = n_rollers/2 * f_rot * (1 - d/D cos(theta))."""
    n_rollers, geom = 8.0, 0.25
    k.imp_freq = np.full_like(k.imp_freq, (n_rollers / 2.0) * speed_hz * (1.0 - geom))
