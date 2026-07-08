"""Real MaFaulDa adapter implementing the same DatasetBundle contract.

NOT exercised in the synthetic phase. Ported from the legacy loader so the
`dataset=mafaulda` flag resolves end-to-end; `has_ground_truth=False` makes
ground-truth-dependent experiments (A1/A3, open-set distance-to-truth) skip
automatically. A StubBundle (shapes only) backs the contract test without
real data on disk.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import torch

from src.configs import DataCfg
from .contract import WindowDataset, make_loader

# reduced 4-class mapping from the legacy pipeline (see legacy/ for the full map)
LABEL_MAPPING = {
    "normal": 0,
    "imbalance_fault_20g": 1,
    "vertical_misalignment_fault_1.27mm": 2,
    "overhang_ball_fault_20g": 3,
}


class RealBundle:
    def __init__(self, cfg: DataCfg):
        self.cfg = cfg
        root = Path(cfg.root)
        traces, labels, omegas = [], [], []
        for category, ci in LABEL_MAPPING.items():
            yf = sorted(glob.glob(str(root / f"Y_{category}_trainingset.pth")))
            xf = sorted(glob.glob(str(root / f"X_{category}_trainingset.pth")))
            if not yf or not xf:
                raise FileNotFoundError(f"Missing processed tensors for '{category}' in {root}")
            y = torch.load(yf[0], weights_only=True).float()          # (N, T, 4)
            x = torch.load(xf[0], weights_only=True).float()
            traces.append(y)
            labels.append(torch.full((y.shape[0],), ci, dtype=torch.long))
            omegas.append(x[:, 0, 8].float())
        raw = torch.cat(traces)                                       # (N, T, 4) physical
        self._label = torch.cat(labels)
        self._omega = torch.cat(omegas)
        mean = raw[self._label == 0].mean(dim=(0, 1))
        std = raw[self._label == 0].std(dim=(0, 1))
        self._mean, self._std = mean.view(1, 1, 4), std.view(1, 1, 4)
        self._raw = raw
        # interleaved split (legacy convention) to avoid temporal leakage
        N = raw.shape[0]
        idx = torch.arange(N)
        self._splits = {
            "train": idx[(idx % 5 != 0) & (idx % 5 != 1)],
            "val": idx[idx % 5 == 1],
            "test": idx[idx % 5 == 0],
        }
        self.meta = {
            "fs": cfg.fs, "T": raw.shape[1], "channels": 4,
            "class_names": list(LABEL_MAPPING.keys()),
            "speeds_hz": list(cfg.speeds_hz),
            "norm_mean": mean.tolist(), "norm_std": std.tolist(),
            "has_ground_truth": False,
        }

    def normalize(self, x_phys):
        m = self._mean.view(1, 4, 1).to(x_phys.device, x_phys.dtype)
        s = self._std.view(1, 4, 1).to(x_phys.device, x_phys.dtype)
        return (x_phys - m) / s

    def denormalize(self, x_norm):
        m = self._mean.view(1, 4, 1).to(x_norm.device, x_norm.dtype)
        s = self._std.view(1, 4, 1).to(x_norm.device, x_norm.dtype)
        return x_norm * s + m

    def arrays(self, split, classes=None, speeds=None):
        idx = self._splits[split]
        lab = self._label[idx]
        m = torch.ones_like(lab, dtype=torch.bool)
        if classes is not None:
            m &= torch.isin(lab, torch.tensor(classes))
        idx = idx[m]
        raw_p = self._raw[idx].transpose(1, 2)
        return {
            "raw": self.normalize(raw_p), "clean": self.normalize(raw_p),
            "raw_phys": raw_p, "clean_phys": raw_p, "jitter_phys": None,
            "label": self._label[idx], "omega": self._omega[idx],
            "speed_hz": self._omega[idx] / (2 * np.pi),
        }

    def loader(self, split, batch_size, shuffle=False, classes=None, speeds=None):
        a = self.arrays(split, classes, speeds)
        ds = WindowDataset(a["raw"], a["clean"], a["label"], a["omega"], a["speed_hz"])
        return make_loader(ds, batch_size, shuffle, seed=self.cfg.seed)

    def fewshot_fault_sets(self, n_per_class, seed):
        rng = np.random.default_rng(seed)
        out = {}
        a = self.arrays("val")
        for ci in sorted(set(a["label"].tolist()) - {0}):
            idx = torch.nonzero(a["label"] == ci).squeeze(1).numpy()
            pick = torch.from_numpy(rng.choice(idx, min(n_per_class, len(idx)), replace=False))
            out[ci] = {"raw_phys": a["raw_phys"][pick], "omega": a["omega"][pick]}
        return out


class StubBundle:
    """Shape-only stand-in used by the contract test (no real data needed)."""

    def __init__(self, cfg: DataCfg):
        self.cfg = cfg
        g = torch.Generator().manual_seed(0)
        N, T = 24, cfg.T
        self._raw = torch.randn(N, 4, T, generator=g)
        self._label = torch.arange(N) % 4
        self._omega = torch.full((N,), 2 * np.pi * 16.0)
        self.meta = {
            "fs": cfg.fs, "T": T, "channels": 4,
            "class_names": ["normal", "c1", "c2", "c3"],
            "speeds_hz": [16.0], "norm_mean": [0.0] * 4, "norm_std": [1.0] * 4,
            "has_ground_truth": False,
        }

    def normalize(self, x):
        return x

    def denormalize(self, x):
        return x

    def arrays(self, split, classes=None, speeds=None):
        lab = self._label
        m = torch.ones_like(lab, dtype=torch.bool)
        if classes is not None:
            m &= torch.isin(lab, torch.tensor(classes))
        return {
            "raw": self._raw[m], "clean": self._raw[m],
            "raw_phys": self._raw[m], "clean_phys": self._raw[m], "jitter_phys": None,
            "label": lab[m], "omega": self._omega[m],
            "speed_hz": self._omega[m] / (2 * np.pi),
        }

    def loader(self, split, batch_size, shuffle=False, classes=None, speeds=None):
        a = self.arrays(split, classes, speeds)
        ds = WindowDataset(a["raw"], a["clean"], a["label"], a["omega"], a["speed_hz"])
        return make_loader(ds, batch_size, shuffle)

    def fewshot_fault_sets(self, n_per_class, seed):
        out = {}
        a = self.arrays("val")
        for ci in (1, 2, 3):
            m = a["label"] == ci
            out[ci] = {"raw_phys": a["raw_phys"][m][:n_per_class], "omega": a["omega"][m][:n_per_class]}
        return out
