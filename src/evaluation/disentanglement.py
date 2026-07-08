"""Block A tooling: MINE I(z; jitter) (A1) and spectral leakage (A3).
Requires ground truth — callers must check bundle.meta['has_ground_truth']."""
from __future__ import annotations

import numpy as np
import torch
from scipy.signal import coherence

from src.models import estimate_mi
from .fidelity import signal_features


def mi_z_jitter(encode_fn, bundle, device="cpu", split: str = "val",
                mi_epochs: int = 300, seed: int = 0) -> float:
    """I(z_macro ; jitter-summary) in nats. encode_fn: (B,4,T)->(B,d)."""
    a = bundle.arrays(split)
    with torch.no_grad():
        zs = []
        for s0 in range(0, a["raw"].shape[0], 32):
            zs.append(encode_fn(a["raw"][s0:s0 + 32].to(device)).cpu())
        Z = torch.cat(zs)
    jit = bundle.normalize(a["jitter_phys"])
    J = torch.from_numpy(signal_features(jit, fft_bins=32, device=device))
    # compact both sides via PCA to keep MINE well-posed
    def pca16(M):
        M = M - M.mean(0)
        U, S, Vt = torch.linalg.svd(M.double(), full_matrices=False)
        return (M.double() @ Vt[:16].T).float()
    return estimate_mi(pca16(Z), pca16(J), epochs=mi_epochs, seed=seed, device=device)


def spectral_leakage(envelope: torch.Tensor, jitter: torch.Tensor, fs: int) -> dict:
    """A3: mean magnitude-squared coherence between the decoded envelope and
    the jitter residual, overall and in the jitter band (>4 kHz)."""
    env = envelope.detach().cpu().numpy()
    jit = jitter.detach().cpu().numpy()
    cohs, cohs_hf = [], []
    for i in range(min(env.shape[0], 16)):
        for c in range(env.shape[1]):
            f, C = coherence(env[i, c], jit[i, c], fs=fs, nperseg=512)
            cohs.append(C.mean())
            cohs_hf.append(C[f > 4000].mean() if np.any(f > 4000) else np.nan)
    return {"coherence_mean": float(np.nanmean(cohs)),
            "coherence_hf_mean": float(np.nanmean(cohs_hf))}
