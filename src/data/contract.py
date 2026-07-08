"""Dataset contract every downstream module depends on (and nothing else)."""
from __future__ import annotations

from typing import NamedTuple, Protocol, runtime_checkable

import torch
from torch.utils.data import DataLoader, Dataset


class Batch(NamedTuple):
    raw: torch.Tensor            # (B, 4, T) float32, normalized model units
    clean: torch.Tensor          # (B, 4, T) envelope ground truth (== raw for real data)
    label: torch.Tensor          # (B,) long
    omega: torch.Tensor          # (B,) float32, rad/s
    speed_hz: torch.Tensor       # (B,) float32


class WindowDataset(Dataset):
    def __init__(self, raw, clean, label, omega, speed_hz):
        self.raw, self.clean, self.label = raw, clean, label
        self.omega, self.speed_hz = omega, speed_hz

    def __len__(self):
        return self.raw.shape[0]

    def __getitem__(self, i):
        return self.raw[i], self.clean[i], self.label[i], self.omega[i], self.speed_hz[i]


def collate(items) -> Batch:
    cols = list(zip(*items))
    return Batch(
        raw=torch.stack(cols[0]), clean=torch.stack(cols[1]),
        label=torch.stack(cols[2]), omega=torch.stack(cols[3]),
        speed_hz=torch.stack(cols[4]),
    )


def make_loader(ds: WindowDataset, batch_size: int, shuffle: bool, seed: int = 0) -> DataLoader:
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate, generator=gen if shuffle else None)


@runtime_checkable
class DatasetBundle(Protocol):
    """What every dataset implementation must provide.

    meta keys: channels, T, fs, class_names, speeds_hz, norm_mean, norm_std,
    has_ground_truth (bool — False for real data: `clean`==`raw`, no jitter,
    no fault params; experiments that need ground truth must check this).
    """
    meta: dict

    def loader(self, split: str, batch_size: int, shuffle: bool = False,
               classes: list[int] | None = None,
               speeds: list[float] | None = None) -> DataLoader: ...

    def arrays(self, split: str, classes: list[int] | None = None,
               speeds: list[float] | None = None) -> dict: ...

    def fewshot_fault_sets(self, n_per_class: int, seed: int) -> dict: ...
