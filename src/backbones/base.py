"""SynthesisBackbone protocol + the per-step guidance trace (D3)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import torch


@dataclass
class StepTrace:
    """Per-step log of the guided reverse process (feeds D3)."""
    t: list[int] = field(default_factory=list)
    penalty: list[float] = field(default_factory=list)
    grad_norm: list[float] = field(default_factory=list)
    z_snapshots: list[torch.Tensor] = field(default_factory=list)

    def as_dict(self):
        return {"t": self.t, "penalty": self.penalty, "grad_norm": self.grad_norm}


@runtime_checkable
class SynthesisBackbone(Protocol):
    """Common contract (P3.4). `guidance` is any GuidanceDensity (or None);
    every backbone must consume it — diffusion backbones as VJP gradient
    guidance, the CVAE via density-weighted resampling of its outputs."""
    name: str

    def prepare(self, cfg, bundle, device) -> None: ...

    def synthesize(self, target_class: int, n: int, guidance=None,
                   source: dict | None = None, sdedit_overrides: dict | None = None
                   ) -> dict: ...
