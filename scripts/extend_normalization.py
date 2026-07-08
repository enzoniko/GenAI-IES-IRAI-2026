#!/usr/bin/env python3
"""Extend normalization metadata from 8 to 10 features.

This script backs up results/normalization_metadata.pth, scans all X_*.pth
training tensors under data/processed-mafaulda/16hz, computes global omega/time
min/max, and appends them to X_min/X_max.
"""

# pyright: reportAny=false, reportUnknownVariableType=false, reportUnusedCallResult=false, reportPrivateImportUsage=false

from __future__ import annotations

import glob
import shutil
from pathlib import Path
from typing import cast

import torch


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed-mafaulda" / "16hz"
RESULTS_DIR = ROOT / "results"
METADATA_PATH = RESULTS_DIR / "normalization_metadata.pth"
BACKUP_PATH = RESULTS_DIR / "normalization_metadata_backup.pth"


def _load_tensor_dict(path: Path) -> dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict in {path}, got {type(obj)!r}")
    return cast(dict[str, torch.Tensor], obj)


def _ensure_backup(src: Path, dst: Path) -> None:
    if dst.exists():
        print(f"Backup already exists: {dst}")
        return
    _ = shutil.copy2(src, dst)
    print(f"Created backup: {dst}")


def main() -> None:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Missing metadata file: {METADATA_PATH}")

    metadata = _load_tensor_dict(METADATA_PATH)
    x_min = metadata["X_min"]
    x_max = metadata["X_max"]
    y_min = metadata["y_min"]
    y_max = metadata["y_max"]

    print(f"Old shapes: X_min={tuple(x_min.shape)}, X_max={tuple(x_max.shape)}")
    print(f"y shapes: y_min={tuple(y_min.shape)}, y_max={tuple(y_max.shape)}")

    if x_min.shape != x_max.shape:
        raise ValueError("X_min and X_max shapes differ")
    if x_min.ndim != 1:
        raise ValueError(f"Expected 1D X bounds, got {x_min.shape}")

    if x_min.shape[0] == 10:
        print("Metadata already extended to 10 features; no update needed.")
        return
    if x_min.shape[0] != 8:
        raise ValueError(f"Expected X_min/X_max shape (8,) before extension, got {tuple(x_min.shape)}")

    _ensure_backup(METADATA_PATH, BACKUP_PATH)
    if not BACKUP_PATH.exists():
        raise RuntimeError("Backup creation failed; aborting")

    x_files = sorted(glob.glob(str(DATA_DIR / "X_*.pth")))
    print("Discovered X files:")
    for path in x_files:
        print(f"  - {path}")
    if not x_files:
        raise FileNotFoundError(f"No X_*.pth files found in {DATA_DIR}")

    omega_values: list[torch.Tensor] = []
    time_values: list[torch.Tensor] = []
    for file_path in x_files:
        x = torch.load(file_path, map_location="cpu", weights_only=True)
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected tensor in {file_path}, got {type(x)!r}")
        if x.ndim != 3 or x.shape[-1] != 10:
            raise ValueError(f"Expected [N, T, 10] tensor in {file_path}, got {tuple(x.shape)}")
        omega_values.append(x[:, :, 8].reshape(-1).to(dtype=torch.double))
        time_values.append(x[:, :, 9].reshape(-1).to(dtype=torch.double))

    omega_min = omega_values[0].min()
    omega_max = omega_values[0].max()
    time_min = time_values[0].min()
    time_max = time_values[0].max()
    for omega_chunk, time_chunk in zip(omega_values[1:], time_values[1:], strict=True):
        omega_min = torch.minimum(omega_min, omega_chunk.min())
        omega_max = torch.maximum(omega_max, omega_chunk.max())
        time_min = torch.minimum(time_min, time_chunk.min())
        time_max = torch.maximum(time_max, time_chunk.max())

    out_dtype = x_min.dtype
    x_min_new = torch.cat(
        [x_min.clone(), omega_min.to(dtype=out_dtype).view(1), time_min.to(dtype=out_dtype).view(1)],
        dim=0,
    )
    x_max_new = torch.cat(
        [x_max.clone(), omega_max.to(dtype=out_dtype).view(1), time_max.to(dtype=out_dtype).view(1)],
        dim=0,
    )

    updated = {
        "X_min": x_min_new,
        "X_max": x_max_new,
        "y_min": y_min,
        "y_max": y_max,
    }
    torch.save(updated, METADATA_PATH)

    print(f"New shapes: X_min={tuple(x_min_new.shape)}, X_max={tuple(x_max_new.shape)}")
    print(f"Omega range: [{omega_min.item()}, {omega_max.item()}]")
    print(f"Time range: [{time_min.item()}, {time_max.item()}]")
    print(f"Updated metadata: {METADATA_PATH}")


if __name__ == "__main__":
    main()
