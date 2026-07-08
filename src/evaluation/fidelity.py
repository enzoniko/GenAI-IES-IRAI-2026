"""Fidelity metrics (P3.5): TSTR (3 probes, NO pseudo-labels, signal space),
per-class MMD, and the Delta-Silhouette transfer score (C2).

Protocol fixes vs the withdrawn paper:
  * generated labels come from the generation condition, never NearestCentroid;
  * classifiers consume a FIXED, backbone-agnostic feature map of raw signal
    windows (MathFeatureExtractor on the 4 channels), not latent embeddings.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.models import MathFeatureExtractor

_extractor_cache: dict[tuple, MathFeatureExtractor] = {}


@torch.no_grad()
def signal_features(x: torch.Tensor, fft_bins: int = 128, device="cpu") -> np.ndarray:
    """Fixed evaluation feature map: (N, 4, T) normalized windows -> (N, F)."""
    key = (fft_bins,)
    if key not in _extractor_cache:
        _extractor_cache[key] = MathFeatureExtractor(in_channels=4, fft_bins=fft_bins,
                                                     wavelet_levels=4)
    fe = _extractor_cache[key].to(device)
    out = []
    for s0 in range(0, x.shape[0], 64):
        out.append(fe(x[s0:s0 + 64].to(device).permute(0, 2, 1)).cpu())
    return torch.cat(out).numpy()


def _probe(name: str, seed: int):
    if name == "logistic":
        return LogisticRegression(max_iter=2000, random_state=seed)
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=1)
    if name == "svm":
        return SVC(kernel="rbf", C=1.0, random_state=seed)
    raise ValueError(name)


def tstr(gen_feats: np.ndarray, gen_labels: np.ndarray,
         real_train_feats: np.ndarray, real_train_labels: np.ndarray,
         real_test_feats: np.ndarray, real_test_labels: np.ndarray,
         probes=("logistic", "knn", "svm"), seed: int = 0) -> dict:
    """Train-on-synthetic-test-on-real vs oracle, per probe + per-class recall."""
    out = {}
    classes = np.unique(real_test_labels)
    for probe in probes:
        sc_g = StandardScaler().fit(gen_feats)
        clf = _probe(probe, seed).fit(sc_g.transform(gen_feats), gen_labels)
        pred = clf.predict(sc_g.transform(real_test_feats))
        tstr_acc = float((pred == real_test_labels).mean())
        recalls = {int(c): float((pred[real_test_labels == c] == c).mean())
                   for c in classes}
        sc_r = StandardScaler().fit(real_train_feats)
        oracle = _probe(probe, seed).fit(sc_r.transform(real_train_feats),
                                         real_train_labels)
        opred = oracle.predict(sc_r.transform(real_test_feats))
        oracle_acc = float((opred == real_test_labels).mean())
        out[probe] = {
            "tstr_acc": tstr_acc, "oracle_acc": oracle_acc,
            "ratio": tstr_acc / oracle_acc if oracle_acc > 1e-12 else 0.0,
            "per_class_recall": recalls,
        }
    return out


def mmd_rbf(X: np.ndarray, Y: np.ndarray, max_samples: int = 200,
            seed: int = 0) -> float:
    """RBF-kernel MMD^2 with median-heuristic bandwidth (standardized inputs)."""
    rng = np.random.default_rng(seed)
    if len(X) > max_samples:
        X = X[rng.choice(len(X), max_samples, replace=False)]
    if len(Y) > max_samples:
        Y = Y[rng.choice(len(Y), max_samples, replace=False)]
    stacked = np.vstack([X, Y])
    sc = StandardScaler().fit(stacked)
    Xs, Ys = sc.transform(X), sc.transform(Y)
    Z = np.vstack([Xs, Ys])
    d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
    sigma2 = max(med / 2.0, 1e-12)

    def k(A, B):
        dd = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        return np.exp(-dd / (2 * sigma2)).mean()

    return float(max(k(Xs, Xs) - 2 * k(Xs, Ys) + k(Ys, Ys), 0.0))


def per_class_mmd(gen_feats, gen_labels, real_feats, real_labels,
                  max_samples=200, seed=0) -> dict[int, float]:
    out = {}
    for c in np.unique(real_labels):
        g = gen_feats[gen_labels == c]
        r = real_feats[real_labels == c]
        if len(g) < 2 or len(r) < 2:
            continue
        out[int(c)] = mmd_rbf(g, r, max_samples, seed)
    return out


def delta_silhouette_transfer(gen_feats, gen_labels, real_test_feats,
                              real_test_labels, seed: int = 0) -> dict:
    """C2: cluster-quality gain on REAL data from a projection learned on
    SYNTHETIC-only data (LDA as the metric-learning probe)."""
    base = float(silhouette_score(StandardScaler().fit_transform(real_test_feats),
                                  real_test_labels))
    n_comp = min(len(np.unique(gen_labels)) - 1, gen_feats.shape[1])
    lda = LinearDiscriminantAnalysis(n_components=n_comp)
    sc = StandardScaler().fit(gen_feats)
    lda.fit(sc.transform(gen_feats), gen_labels)
    proj = lda.transform(sc.transform(real_test_feats))
    after = float(silhouette_score(proj, real_test_labels))
    return {"silhouette_raw": base, "silhouette_synth_projected": after,
            "delta_silhouette": after - base}
