from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler


def _as_2d_array(x):
    arr = np.asarray(x)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {arr.shape}")
    return arr


def compute_tstr(z_real, labels_real, z_generated, labels_generated=None, n_folds=5):
    z_real = _as_2d_array(z_real)
    z_generated = _as_2d_array(z_generated)
    labels_real = np.asarray(labels_real)

    scaler = StandardScaler()
    z_real_scaled = scaler.fit_transform(z_real)
    z_generated_scaled = scaler.transform(z_generated)

    if labels_generated is None:
        centroid_model = NearestCentroid()
        centroid_model.fit(z_real_scaled, labels_real)
        labels_generated = centroid_model.predict(z_generated_scaled)
    else:
        labels_generated = np.asarray(labels_generated)

    lr_tstr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    lr_tstr.fit(z_generated_scaled, labels_generated)
    tstr_accuracy = float(lr_tstr.score(z_real_scaled, labels_real))

    lr_oracle = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    oracle_accuracy = float(
        cross_val_score(lr_oracle, z_real_scaled, labels_real, cv=n_folds, scoring="accuracy").mean()
    )

    ratio = float(tstr_accuracy / oracle_accuracy) if oracle_accuracy > 1e-12 else 0.0
    return {
        "tstr_accuracy": tstr_accuracy,
        "oracle_accuracy": oracle_accuracy,
        "ratio": ratio,
    }


def compute_mmd(X, Y, kernel="rbf", sigma=None):
    if kernel != "rbf":
        raise ValueError("Only the 'rbf' kernel is supported")

    X = _as_2d_array(X)
    Y = _as_2d_array(Y)

    scaler = StandardScaler()
    stacked = np.vstack([X, Y])
    stacked_scaled = scaler.fit_transform(stacked)
    X_scaled = stacked_scaled[: len(X)]
    Y_scaled = stacked_scaled[len(X) :]

    if sigma is None:
        if len(X_scaled) < 2:
            # Fall back to median heuristic on Y (subsample to avoid O(N²) memory)
            n_sub = min(100, len(Y_scaled))
            rng = np.random.default_rng(42)
            Y_sub = Y_scaled[rng.choice(len(Y_scaled), n_sub, replace=False)]
            diff = Y_sub[:, None, :] - Y_sub[None, :, :]
            dists = np.sqrt(np.sum(diff * diff, axis=2))
            mask = ~np.eye(n_sub, dtype=bool)
            median = np.median(dists[mask]) if np.any(mask) else 1.0
            sigma = float(median if median > 1e-12 else 1.0)
        else:
            diff = X_scaled[:, None, :] - X_scaled[None, :, :]
            dists = np.sqrt(np.sum(diff * diff, axis=2))
            mask = ~np.eye(len(X_scaled), dtype=bool)
            median = np.median(dists[mask]) if np.any(mask) else 1.0
            sigma = float(median if median > 1e-12 else 1.0)

    sigma = float(sigma)
    if sigma <= 0:
        sigma = 1.0

    def _rbf_mean(A, B):
        diff = A[:, None, :] - B[None, :, :]
        sq_dist = np.sum(diff * diff, axis=2)
        kernel_mat = np.exp(-sq_dist / (2.0 * sigma * sigma))
        return float(kernel_mat.mean())

    mmd2 = _rbf_mean(X_scaled, X_scaled) - 2.0 * _rbf_mean(X_scaled, Y_scaled) + _rbf_mean(Y_scaled, Y_scaled)
    return float(max(mmd2, 0.0))


def evaluate_sdedit_quality(z_real, labels_real, z_generated, labels_generated):
    tstr_result = compute_tstr(z_real, labels_real, z_generated, labels_generated, n_folds=5)
    mmd_value = compute_mmd(z_real, z_generated)
    return {**tstr_result, "mmd": mmd_value}
