"""Cluster geometry (C1): standard indices + the non-Gaussianity scores G_y —
the independent variable of hypothesis H. Three defensible statistics are
reported (proposal threat #4): Henze-Zirkler, Mardia (skew/kurtosis), and
the Epps-Pulley energy statistic (the SIGReg objective)."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    calinski_harabasz_score, davies_bouldin_score, silhouette_score,
)

from src.models import epps_pulley


def cluster_indices(V: np.ndarray, labels: np.ndarray) -> dict:
    return {
        "silhouette": float(silhouette_score(V, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(V, labels)),
        "davies_bouldin": float(davies_bouldin_score(V, labels)),
    }


def _standardize(V: np.ndarray) -> np.ndarray:
    mu = V.mean(axis=0)
    cov = np.cov(V.T)
    # ridge scaled to the covariance magnitude keeps tiny-n cells invertible
    ridge = max(1e-12, 1e-3 * np.trace(cov) / V.shape[1])
    if V.shape[0] <= V.shape[1] + 1:
        ridge = max(ridge, 0.1 * np.trace(cov) / V.shape[1])
    cov = cov + ridge * np.eye(V.shape[1])
    L = np.linalg.cholesky(np.linalg.inv(cov))
    return (V - mu) @ L


def henze_zirkler(V: np.ndarray) -> float:
    """HZ statistic (larger = less Gaussian)."""
    n, d = V.shape
    Z = _standardize(V)
    b = (n * (2 * d + 1) / 4.0) ** (1.0 / (d + 4)) / np.sqrt(2.0)
    D = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    Di = (Z**2).sum(1)
    t1 = np.exp(-(b**2) / 2.0 * D).mean()
    t2 = 2.0 * (1 + b**2) ** (-d / 2.0) * np.exp(
        -(b**2) / (2.0 * (1 + b**2)) * Di).mean()
    t3 = (1 + 2 * b**2) ** (-d / 2.0)
    return float(n * (t1 - t2 + t3))


def mardia(V: np.ndarray) -> dict:
    n, d = V.shape
    Z = _standardize(V)
    G = Z @ Z.T
    b1 = (G**3).mean()                       # multivariate skewness
    b2 = (np.diag(G) ** 2).mean()            # multivariate kurtosis
    b2_expected = d * (d + 2)
    return {"mardia_skew": float(b1),
            "mardia_kurt_excess": float(b2 - b2_expected)}


def gng_scores(V: np.ndarray, max_samples: int = 500, seed: int = 0) -> dict:
    """All three non-Gaussianity statistics for one class cluster (n, d)."""
    rng = np.random.default_rng(seed)
    if len(V) > max_samples:
        V = V[rng.choice(len(V), max_samples, replace=False)]
    Z = _standardize(V)
    out = {"henze_zirkler": henze_zirkler(V)}
    out.update(mardia(V))
    out["epps_pulley"] = float(epps_pulley(torch.from_numpy(Z), n_proj=32))
    return out


def geometry_table(V: np.ndarray, labels: np.ndarray, speeds: np.ndarray,
                   seed: int = 0, min_n: int = 8) -> list[dict]:
    """Per (class, speed) cell: G_y statistics; plus global cluster indices
    per speed. V is expected in the whitened PCA embedding space."""
    rows = []
    for sp in np.unique(speeds):
        m_sp = speeds == sp
        idx = cluster_indices(V[m_sp], labels[m_sp])
        for c in np.unique(labels):
            m = m_sp & (labels == c)
            if m.sum() < min_n:
                continue
            row = {"class": int(c), "speed": float(sp), **gng_scores(V[m], seed=seed)}
            row.update({f"global_{k}": v for k, v in idx.items()})
            rows.append(row)
    return rows
