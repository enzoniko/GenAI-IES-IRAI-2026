"""Config loading, smoke shrinking, seeding, run directories, manifests."""
from __future__ import annotations

import dataclasses
import json
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .schema import RunCfg, from_dict, replace, to_dict


def load_config(yaml_path: str | None = None, overrides: list[str] | None = None) -> RunCfg:
    """Build RunCfg from an optional YAML file plus `a.b=c` dot-overrides."""
    if yaml_path:
        with open(yaml_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        cfg = from_dict(RunCfg, raw)
    else:
        cfg = RunCfg()
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"Override must be key.path=value, got '{ov}'")
        key, val = ov.split("=", 1)
        cfg = replace(cfg, key.strip(), yaml.safe_load(val))
    if cfg.smoke:
        cfg = apply_smoke(cfg)
    return cfg


def apply_smoke(cfg: RunCfg) -> RunCfg:
    """Shrink every knob so the whole pipeline runs in minutes on CPU."""
    for path, val in [
        ("data.T", 1024), ("data.speeds_hz", [16.0]),
        ("data.n_train", 16), ("data.n_val", 8), ("data.n_test", 8),
        ("data.version", "smoke"), ("data.batch_size", 8),
        ("pinn.epochs", 3), ("pinn.batch_size", 2048),
        ("jepa.epochs", 3), ("jepa.batch_size", 8), ("jepa.num_layers", 2),
        ("jepa.dim_feedforward", 128),
        ("dec1.epochs", 3), ("dec1.batch_size", 8),
        ("dec2.epochs", 3), ("dec2.batch_size", 8), ("dec2.latent_dim", 16),
        ("ldm.epochs", 5), ("ldm.batch_size", 16),
        ("sdedit.n_infer", 10),
        ("guidance.pca_dim", 8), ("guidance.flow_epochs", 30),
        ("eval.n_gen_per_class", 8), ("eval.seeds", [0]), ("eval.mmd_max_samples", 32),
    ]:
        cfg = replace(cfg, path, val)
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(cfg_device: str) -> torch.device:
    if cfg_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg_device)


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


class RunDir:
    """Per-run output directory with a manifest (`run.json`).

    Manifest fields: resolved config, git sha, seed, start/end time, outputs
    (arbitrary JSON-serializable metrics/paths), and a `bypass` tag that is
    forced true whenever oracle.mode == 'raw' so the aggregator can refuse to
    mix diagnostic runs into headline tables.
    """

    def __init__(self, cfg: RunCfg, name: str, root: str | None = None):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        base = Path(root or cfg.results_root) / cfg.experiment
        self.path = base / f"{name}_{stamp}_s{cfg.seed}"
        self.path.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self.manifest: dict[str, Any] = {
            "name": name,
            "config": to_dict(cfg),
            "git_sha": git_sha(),
            "seed": cfg.seed,
            "bypass": cfg.oracle.mode == "raw",
            "started": stamp,
            "outputs": {},
        }
        self.flush()

    def log(self, **outputs: Any) -> None:
        self.manifest["outputs"].update(_jsonable(outputs))
        self.flush()

    def flush(self) -> None:
        with open(self.path / "run.json", "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2, default=str)

    def file(self, rel: str) -> Path:
        p = self.path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def find_latest_run(results_root: str, experiment: str, name_prefix: str) -> Path | None:
    """Locate the newest run dir whose name starts with `name_prefix`."""
    base = Path(results_root) / experiment
    if not base.exists():
        return None
    candidates = sorted(p for p in base.iterdir() if p.is_dir() and p.name.startswith(name_prefix))
    return candidates[-1] if candidates else None
