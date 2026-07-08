"""Typed, frozen configuration schema for the mafaulda_synthetic extension.

Single source of truth for every knob. Composed per experiment via YAML +
dot-overrides (see loader.py). No module-level mutable state anywhere.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data / benchmark
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DataCfg:
    name: str = "mafaulda_synthetic"      # "mafaulda_synthetic" | "mafaulda"
    variant: str = "A"                    # "A" nominal machine, "B" detuned (E4)
    version: str = "v1"
    root: str = "data/processed-mafaulda-synthetic"
    fs: int = 50_000                      # Hz, mirrors MaFaulDa
    T: int = 4167                         # one revolution at the slowest speed (12 Hz):
                                          #   guarantees >=1 full rev per window at every
                                          #   speed, required by synchronous kinematics
    speeds_hz: tuple[float, ...] = (12.0, 16.0, 20.0)
    classes: tuple[str, ...] = (
        "healthy", "imbalance_uni", "imbalance_bi", "bpfo_impulsive",
        "misalign_cont", "looseness_skew", "combo_ring",
    )
    n_train: int = 600                    # windows per class per speed
    n_val: int = 200
    n_test: int = 200
    noise_std_frac: float = 0.05          # broadband noise, fraction of healthy clean RMS
    hum_amp_frac: float = 0.10            # 5 kHz hum amplitude, fraction of healthy clean RMS
    hum_freq_hz: float = 5000.0
    seed: int = 0
    batch_size: int = 32


# ---------------------------------------------------------------------------
# PINN (Phase 0)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PinnCfg:
    hidden_layers: tuple[int, ...] = (128, 128)
    activation: str = "elu"
    dropout: float = 0.0
    learnable_params: bool = False        # physics params known for synthetic
    lr: float = 1e-3
    epochs: int = 200
    batch_size: int = 8192                # pointwise samples per batch
    early_stop_patience: int = 20
    lambda_force: float = 1e-3            # small-norm penalty on unmeasured forces
    relobralo_alpha: float = 0.5125
    relobralo_rho: float = 0.2332
    relobralo_temperature: float = 1.4198


# ---------------------------------------------------------------------------
# Representation (Phase 1)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class JepaCfg:
    d_model: int = 128
    patch_size: int = 25
    nhead: int = 4
    num_layers: int = 4
    dim_feedforward: int = 512
    ema_decay: float = 0.99
    var_coef: float = 1.0                 # VICReg variance term on z_macro (anti-collapse)
    cov_coef: float = 0.05                # VICReg covariance term
    sigreg_coef: float = 0.0              # A2 arm: >0 enables SIGReg (Epps-Pulley) term
    lr: float = 1e-3
    epochs: int = 60
    batch_size: int = 32
    early_stop_patience: int = 15


@dataclass(frozen=True)
class Dec1Cfg:
    lr: float = 1e-3
    epochs: int = 60
    batch_size: int = 32
    early_stop_patience: int = 15


@dataclass(frozen=True)
class Dec2Cfg:
    latent_dim: int = 64
    label_embed_dim: int = 16
    beta_kl: float = 0.01
    beta_tc: float = 0.0                  # E1 arm: >0 enables beta-TCVAE total-correlation penalty
    lr: float = 1e-3
    epochs: int = 60
    batch_size: int = 16
    early_stop_patience: int = 15


# ---------------------------------------------------------------------------
# Generation (Phase 2)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LdmCfg:
    time_dim: int = 64
    hidden: int = 256
    num_train_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    class_cond: bool = False              # E2 arm
    lr: float = 1e-3
    epochs: int = 100
    batch_size: int = 64
    early_stop_patience: int = 15


@dataclass(frozen=True)
class SdeditCfg:
    t0_frac: float = 0.5                  # fraction of the TRAINING schedule (t0 = 500)
    n_infer: int = 50                     # DDIM-subsampled inference steps over [0, t0]
    guidance_scale: float = 1.0
    guidance_interval: int = 1            # apply guidance every N inference steps (D1)
    clip_tau: float = 10.0                # max L2 norm of the VJP gradient
    record_trace: bool = True


@dataclass(frozen=True)
class GuidanceCfg:
    kind: str = "gaussian"                # "gaussian" | "gmm" | "flow"
    pca_dim: int = 32                     # densities fitted in PCA-whitened Phi space
    gmm_max_k: int = 4                    # k selected by BIC
    flow_layers: int = 5
    flow_hidden: int = 64
    flow_epochs: int = 300
    flow_lr: float = 1e-3
    fit_seed: int = 0


@dataclass(frozen=True)
class OracleCfg:
    mode: str = "pinn"                    # "pinn" | "raw" (raw = DIAGNOSTIC ONLY, tagged bypass)
    fft_bins: int = 256
    wavelet_levels: int = 4


@dataclass(frozen=True)
class EvalCfg:
    n_gen_per_class: int = 200
    probes: tuple[str, ...] = ("logistic", "knn", "svm")
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    mmd_max_samples: int = 200


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RunCfg:
    data: DataCfg = field(default_factory=DataCfg)
    pinn: PinnCfg = field(default_factory=PinnCfg)
    jepa: JepaCfg = field(default_factory=JepaCfg)
    dec1: Dec1Cfg = field(default_factory=Dec1Cfg)
    dec2: Dec2Cfg = field(default_factory=Dec2Cfg)
    ldm: LdmCfg = field(default_factory=LdmCfg)
    sdedit: SdeditCfg = field(default_factory=SdeditCfg)
    guidance: GuidanceCfg = field(default_factory=GuidanceCfg)
    oracle: OracleCfg = field(default_factory=OracleCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    seed: int = 0
    device: str = "auto"                  # "auto" | "cpu" | "cuda"
    smoke: bool = False
    results_root: str = "results"
    experiment: str = "default"


# ---------------------------------------------------------------------------
# (De)serialization helpers
# ---------------------------------------------------------------------------
def to_dict(cfg: Any) -> Any:
    if dataclasses.is_dataclass(cfg):
        return {f.name: to_dict(getattr(cfg, f.name)) for f in dataclasses.fields(cfg)}
    if isinstance(cfg, (list, tuple)):
        return [to_dict(v) for v in cfg]
    return cfg


def _coerce(value: Any, target_type: Any) -> Any:
    """Coerce YAML/CLI values into the dataclass field type."""
    origin = getattr(target_type, "__origin__", None)
    if origin is tuple:
        args = target_type.__args__
        elem = args[0] if args else str
        return tuple(_coerce(v, elem) for v in value)
    if dataclasses.is_dataclass(target_type):
        return from_dict(target_type, value)
    if target_type is float and value is not None:
        return float(value)
    if target_type is int and value is not None and not isinstance(value, bool):
        return int(value)
    if target_type is bool and isinstance(value, str):
        return value.lower() in ("1", "true", "yes", "on")
    return value


def from_dict(cls: type, d: dict) -> Any:
    """Build a (nested) frozen dataclass from a plain dict, coercing types."""
    kwargs = {}
    fields = {f.name: f for f in dataclasses.fields(cls)}
    for key, value in (d or {}).items():
        if key not in fields:
            raise KeyError(f"Unknown config key '{key}' for {cls.__name__}")
        f = fields[key]
        ftype = f.type if not isinstance(f.type, str) else _resolve_type(cls, f.name)
        kwargs[key] = _coerce(value, ftype)
    return cls(**kwargs)


def _resolve_type(cls: type, name: str) -> Any:
    import typing
    hints = typing.get_type_hints(cls)
    return hints[name]


def replace(cfg: Any, path: str, value: Any) -> Any:
    """Return a new config with dotted `path` (e.g. 'jepa.lr') replaced by `value`."""
    parts = path.split(".")
    if len(parts) == 1:
        ftype = _resolve_type(type(cfg), parts[0])
        return dataclasses.replace(cfg, **{parts[0]: _coerce(value, ftype)})
    child = getattr(cfg, parts[0])
    new_child = replace(child, ".".join(parts[1:]), value)
    return dataclasses.replace(cfg, **{parts[0]: new_child})
