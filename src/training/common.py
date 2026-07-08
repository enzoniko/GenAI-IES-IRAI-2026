"""Shared training utilities + artifact registry (stage chaining)."""
from __future__ import annotations

import json
from pathlib import Path

from src.configs import RunCfg


class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-5):
        self.patience, self.min_delta = patience, min_delta
        self.best = None
        self.counter = 0
        self.stop = False

    def __call__(self, val_loss: float) -> bool:
        """Returns True if this is a new best."""
        if self.best is None or val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.stop = True
        return False


def _artifacts_path(cfg: RunCfg) -> Path:
    p = Path(cfg.results_root) / cfg.experiment / "artifacts.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def register_artifact(cfg: RunCfg, name: str, path: str | Path) -> None:
    p = _artifacts_path(cfg)
    d = json.loads(p.read_text()) if p.exists() else {}
    d[name] = str(path)
    p.write_text(json.dumps(d, indent=2))


def get_artifact(cfg: RunCfg, name: str) -> Path:
    p = _artifacts_path(cfg)
    if not p.exists():
        raise FileNotFoundError(f"No artifacts registered for experiment '{cfg.experiment}'")
    d = json.loads(p.read_text())
    if name not in d:
        raise KeyError(f"Artifact '{name}' not registered; available: {list(d)}")
    return Path(d[name])
