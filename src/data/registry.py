"""Dataset registry: the ONLY place `cfg.data.name` is interpreted."""
from __future__ import annotations

from src.configs import DataCfg


def get_dataset(cfg: DataCfg, auto_generate: bool = True):
    if cfg.name == "mafaulda_synthetic":
        from .mafaulda_synthetic import SyntheticBundle
        return SyntheticBundle(cfg, auto_generate=auto_generate)
    if cfg.name == "mafaulda":
        from .mafaulda_real import RealBundle
        return RealBundle(cfg)
    if cfg.name == "mafaulda_stub":
        from .mafaulda_real import StubBundle
        return StubBundle(cfg)
    raise ValueError(f"Unknown dataset '{cfg.name}'")
