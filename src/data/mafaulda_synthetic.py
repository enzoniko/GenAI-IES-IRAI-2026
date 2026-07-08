"""mafaulda_synthetic: generator + DatasetBundle.

Windows are INDEPENDENT draws (fresh fault parameters + fresh noise per
window, keyphasor-aligned), so there is no temporal leakage between splits by
construction. Tensors are stored in PHYSICAL units; normalization to model
units (z-score on healthy-train stats) happens in the bundle.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.configs import DataCfg
from src.physics import FaultKnobs, RotorParams, sample_knobs, set_speed_coupling, simulate
from .contract import Batch, WindowDataset, collate, make_loader

SPLITS = ("train", "val", "test")
GENERATOR_VERSION = 3  # bump whenever simulator/measurement semantics change


def _dir(cfg: DataCfg) -> Path:
    return Path(cfg.root) / cfg.version / cfg.variant


def _cfg_hash(cfg: DataCfg) -> str:
    from src.configs import to_dict
    payload = {"gen_version": GENERATOR_VERSION, **to_dict(cfg)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def _measure(clean: np.ndarray, ref_rms: np.ndarray, cfg: DataCfg,
             rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """clean (B,T,4) -> (raw, jitter). Noise level is anchored to HEALTHY rms
    so noise carries no class information."""
    B, T, C = clean.shape
    t = np.arange(T) / cfg.fs
    noise = rng.normal(0.0, 1.0, clean.shape) * (cfg.noise_std_frac * ref_rms)
    phase = rng.uniform(0, 2 * np.pi, (B, 1, C)) + np.arange(C) * np.pi / 8
    hum = (cfg.hum_amp_frac * ref_rms) * np.sin(
        2 * np.pi * cfg.hum_freq_hz * t[None, :, None] + phase)
    jitter = noise + hum
    return clean + jitter, jitter


def generate(cfg: DataCfg, verbose: bool = True) -> Path:
    """Generate (or reuse) the processed dataset. Deterministic given cfg."""
    out = _dir(cfg)
    meta_path = out / "metadata.json"
    h = _cfg_hash(cfg)
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("cfg_hash") == h:
            if verbose:
                print(f"[data] reusing {out} (hash {h})")
            return out
    out.mkdir(parents=True, exist_ok=True)
    p = RotorParams.variant(cfg.variant)

    # reference healthy RMS per channel (16 Hz calibration batch, no noise)
    cal = simulate(p, np.full(4, 2 * np.pi * 16.0), FaultKnobs.healthy(4),
                   T=cfg.T, fs=cfg.fs)
    ref_rms = np.sqrt((cal["acc"] ** 2).mean(axis=(0, 1)))  # (4,)

    n_by_split = {"train": cfg.n_train, "val": cfg.n_val, "test": cfg.n_test}
    healthy_train_raw = []
    for split in SPLITS:
        raws, cleans, labels, omegas, speeds = [], [], [], [], []
        params_all: dict[str, dict] = {}
        for speed in cfg.speeds_hz:
            for ci, cls in enumerate(cfg.classes):
                B = n_by_split[split]
                key = f"{split}|{speed}|{cls}|{cfg.seed}|{cfg.variant}"
                rng = np.random.default_rng(
                    int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little"))
                knobs, pr = sample_knobs(cls, B, rng)
                set_speed_coupling(knobs, speed)
                # +-0.5% speed jitter: realistic for a fixed-speed rig, and small
                # enough that resonance-curvature does not swamp the DESIGNED
                # geometry of tight classes (measured at +-2%: imbalance_uni's
                # (kappa, omega) manifold curvature dominated its G_y)
                omega = 2 * np.pi * speed * (1 + rng.uniform(-0.005, 0.005, B))
                chunks = []
                for s0 in range(0, B, 512):
                    sl = slice(s0, min(s0 + 512, B))
                    sub = type(knobs)(**{f: getattr(knobs, f)[sl] for f in knobs.__dataclass_fields__})
                    chunks.append(simulate(p, omega[sl], sub, T=cfg.T, fs=cfg.fs)["acc"])
                clean = np.concatenate(chunks, axis=0)                     # (B,T,4)
                raw, _ = _measure(clean, ref_rms, cfg, rng)
                raws.append(raw.astype(np.float32))
                cleans.append(clean.astype(np.float32))
                labels.append(np.full(B, ci, dtype=np.int64))
                omegas.append(omega.astype(np.float32))
                speeds.append(np.full(B, speed, dtype=np.float32))
                params_all[f"{cls}@{speed}"] = {k: v.tolist() for k, v in pr.items()}
                if split == "train" and cls == "healthy":
                    healthy_train_raw.append(raw)
                if verbose:
                    print(f"[data] {split} {speed:g}Hz {cls}: {B} windows")
        torch.save({
            "raw": torch.from_numpy(np.concatenate(raws)),
            "clean": torch.from_numpy(np.concatenate(cleans)),
            "label": torch.from_numpy(np.concatenate(labels)),
            "omega": torch.from_numpy(np.concatenate(omegas)),
            "speed_hz": torch.from_numpy(np.concatenate(speeds)),
        }, out / f"{split}.pth")
        (out / f"{split}_params.json").write_text(json.dumps(params_all))

    ht = np.concatenate(healthy_train_raw)  # (N,T,4)
    meta = {
        "cfg_hash": h, "fs": cfg.fs, "T": cfg.T, "channels": 4,
        "class_names": list(cfg.classes), "speeds_hz": list(cfg.speeds_hz),
        "variant": cfg.variant, "ref_rms": ref_rms.tolist(),
        "norm_mean": ht.mean(axis=(0, 1)).tolist(),
        "norm_std": ht.std(axis=(0, 1)).tolist(),
        "has_ground_truth": True,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return out


class SyntheticBundle:
    """DatasetBundle over the generated tensors (in-memory)."""

    def __init__(self, cfg: DataCfg, auto_generate: bool = True):
        self.cfg = cfg
        root = _dir(cfg)
        if not (root / "metadata.json").exists() or \
           json.loads((root / "metadata.json").read_text()).get("cfg_hash") != _cfg_hash(cfg):
            if not auto_generate:
                raise FileNotFoundError(f"Dataset missing/stale at {root}; run generate-data.")
            generate(cfg)
        self.meta = json.loads((root / "metadata.json").read_text())
        self._data = {s: torch.load(root / f"{s}.pth", weights_only=True) for s in SPLITS}
        m, sd = self.meta["norm_mean"], self.meta["norm_std"]
        self._mean = torch.tensor(m).view(1, 1, 4)
        self._std = torch.tensor(sd).view(1, 1, 4)

    # -- normalization between physical and model units ---------------------
    def normalize(self, x_phys: torch.Tensor) -> torch.Tensor:
        """(..., 4, T) physical -> model units."""
        m = self._mean.view(1, 4, 1).to(x_phys.device, x_phys.dtype)
        s = self._std.view(1, 4, 1).to(x_phys.device, x_phys.dtype)
        return (x_phys - m) / s

    def denormalize(self, x_norm: torch.Tensor) -> torch.Tensor:
        m = self._mean.view(1, 4, 1).to(x_norm.device, x_norm.dtype)
        s = self._std.view(1, 4, 1).to(x_norm.device, x_norm.dtype)
        return x_norm * s + m

    # -- selection -----------------------------------------------------------
    def _mask(self, d, classes, speeds):
        m = torch.ones_like(d["label"], dtype=torch.bool)
        if classes is not None:
            m &= torch.isin(d["label"], torch.tensor(classes))
        if speeds is not None:
            m &= torch.isin(d["speed_hz"], torch.tensor(speeds, dtype=torch.float32))
        return m

    def arrays(self, split: str, classes=None, speeds=None) -> dict:
        d = self._data[split]
        m = self._mask(d, classes, speeds)
        raw_p = d["raw"][m].transpose(1, 2)      # (N, 4, T) physical
        clean_p = d["clean"][m].transpose(1, 2)
        return {
            "raw": self.normalize(raw_p), "clean": self.normalize(clean_p),
            "raw_phys": raw_p, "clean_phys": clean_p,
            "jitter_phys": raw_p - clean_p,
            "label": d["label"][m], "omega": d["omega"][m],
            "speed_hz": d["speed_hz"][m],
        }

    def loader(self, split: str, batch_size: int, shuffle: bool = False,
               classes=None, speeds=None):
        a = self.arrays(split, classes, speeds)
        ds = WindowDataset(a["raw"], a["clean"], a["label"], a["omega"], a["speed_hz"])
        return make_loader(ds, batch_size, shuffle, seed=self.cfg.seed)

    def fewshot_fault_sets(self, n_per_class: int, seed: int) -> dict:
        """Few-shot REAL fault samples for density fitting (from val split)."""
        rng = np.random.default_rng(seed)
        out = {}
        d = self._data["val"]
        for ci in range(1, len(self.meta["class_names"])):
            idx = torch.nonzero(d["label"] == ci).squeeze(1).numpy()
            pick = rng.choice(idx, size=min(n_per_class, len(idx)), replace=False)
            pick_t = torch.from_numpy(pick)
            out[ci] = {
                "raw_phys": d["raw"][pick_t].transpose(1, 2),
                "omega": d["omega"][pick_t],
            }
        return out
