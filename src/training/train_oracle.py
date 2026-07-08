"""P1.4/P1.5 support: build the PhysicsEmbedding from the trained PINN,
fit its PCA on train-split features, and cache per-split feature tensors
(reused by geometry analysis, guidance fitting, and evaluation)."""
from __future__ import annotations

import torch

from src.configs import RunCfg, RunDir, resolve_device, seed_everything
from src.data import get_dataset
from src.models import RotorPINN
from src.oracle import PhysicsEmbedding
from src.physics import RotorParams
from .common import get_artifact, register_artifact


def build_embedding(cfg: RunCfg, bundle, device="cpu") -> PhysicsEmbedding:
    pinn = None
    if cfg.oracle.mode == "pinn":
        params = RotorParams.variant(cfg.data.variant if cfg.data.name == "mafaulda_synthetic" else "A")
        pinn = RotorPINN.load(get_artifact(cfg, "pinn"), params=params,
                              hidden_layers=cfg.pinn.hidden_layers,
                              activation=cfg.pinn.activation).to(device)
    return PhysicsEmbedding(
        pinn, cfg.oracle.mode, bundle.meta["fs"],
        torch.tensor(bundle.meta["norm_mean"]), torch.tensor(bundle.meta["norm_std"]),
        fft_bins=cfg.oracle.fft_bins, wavelet_levels=cfg.oracle.wavelet_levels,
    ).to(device)


@torch.no_grad()
def compute_features(emb: PhysicsEmbedding, bundle, split: str, device,
                     batch: int = 32) -> dict:
    a = bundle.arrays(split)
    feats = []
    N = a["raw"].shape[0]
    for s0 in range(0, N, batch):
        x = a["raw"][s0:s0 + batch].to(device)
        om = a["omega"][s0:s0 + batch].to(device)
        feats.append(emb(x, om).cpu())
    return {"phi": torch.cat(feats), "label": a["label"],
            "speed_hz": a["speed_hz"], "omega": a["omega"]}


def train_oracle(cfg: RunCfg, bundle=None, run: RunDir | None = None) -> RunDir:
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    bundle = bundle or get_dataset(cfg.data)
    run = run or RunDir(cfg, "oracle")
    emb = build_embedding(cfg, bundle, device)
    cache = {}
    for split in ("train", "val", "test"):
        cache[split] = compute_features(emb, bundle, split, device)
        p = run.file(f"features_{split}.pth")
        torch.save(cache[split], p)
        register_artifact(cfg, f"features_{split}", p)
    emb.fit_pca(cache["train"]["phi"], cfg.guidance.pca_dim)
    ckpt = run.file("oracle.pth")
    emb.save(ckpt)
    register_artifact(cfg, "oracle", ckpt)
    run.log(feature_dim=emb.output_dim, pca_dim=cfg.guidance.pca_dim,
            bypass=cfg.oracle.mode == "raw", ckpt=str(ckpt))
    return run


def load_oracle(cfg: RunCfg, bundle, device="cpu") -> PhysicsEmbedding:
    pinn = None
    if cfg.oracle.mode == "pinn":
        params = RotorParams.variant(cfg.data.variant if cfg.data.name == "mafaulda_synthetic" else "A")
        pinn = RotorPINN.load(get_artifact(cfg, "pinn"), params=params,
                              hidden_layers=cfg.pinn.hidden_layers,
                              activation=cfg.pinn.activation)
    emb = PhysicsEmbedding.load(get_artifact(cfg, "oracle"), pinn)
    return emb.to(device)


def load_features(cfg: RunCfg, split: str) -> dict:
    return torch.load(get_artifact(cfg, f"features_{split}"), weights_only=True)
