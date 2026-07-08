"""C3 — the primary test of hypothesis H: correlate per-(class, speed)
non-Gaussianity G_y with synthesis fidelity, per backbone and per estimator,
with bootstrap CIs and the PRE-REGISTERED decision threshold."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# Pre-registered falsification threshold (research proposal, Sec. "Hypothesis"):
# H is rejected for a condition if |rho| < 0.4 or the sign is inconsistent
# across estimators/backbones/speeds.
RHO_THRESHOLD = 0.4


def _boot_ci(x, y, fn, n_boot=1000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    n = len(x)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(x[idx])) < 2:
            continue
        v = fn(x[idx], y[idx])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return (np.nan, np.nan)
    return (float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2))))


def correlate_geometry_fidelity(
    cells: pd.DataFrame,
    gng_col: str = "henze_zirkler",
    fidelity_col: str = "mmd",
    group_cols: tuple = ("backbone", "estimator"),
) -> pd.DataFrame:
    """cells: long-format rows with columns [backbone, estimator, class, speed,
    seed, <gng_col>, <fidelity_col>]. Fidelity is averaged over seeds per cell,
    then correlated against G_y across (class, speed) cells within each group."""
    rows = []
    for keys, g in cells.groupby(list(group_cols)):
        cell = g.groupby(["class", "speed"]).agg(
            gng=(gng_col, "mean"), fid=(fidelity_col, "mean")).reset_index()
        x, y = cell["gng"].values, cell["fid"].values
        if len(x) < 3:
            continue
        pear = stats.pearsonr(x, y)
        spear = stats.spearmanr(x, y)
        kend = stats.kendalltau(x, y)
        lo, hi = _boot_ci(x, y, lambda a, b: stats.spearmanr(a, b).statistic)
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row.update({
            "n_cells": len(x),
            "pearson_r": float(pear.statistic), "pearson_p": float(pear.pvalue),
            "spearman_rho": float(spear.statistic), "spearman_p": float(spear.pvalue),
            "kendall_tau": float(kend.statistic),
            "spearman_ci_lo": lo, "spearman_ci_hi": hi,
            "passes_threshold": bool(abs(spear.statistic) >= RHO_THRESHOLD),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def invariance_grade(corr_df: pd.DataFrame, rho_col: str = "spearman_rho") -> dict:
    """Grade the backbone-invariance outcome per the proposal:
    (a) comparable sign+strength across backbones; (b) same sign, different
    magnitude; (c) holds on one backbone only."""
    per_bb = corr_df.groupby("backbone")[rho_col].mean()
    signs = np.sign(per_bb.values)
    strong = np.abs(per_bb.values) >= RHO_THRESHOLD
    if strong.all() and len(set(signs)) == 1:
        grade = "a"
    elif len(set(signs)) == 1 and strong.any():
        grade = "b"
    elif strong.any():
        grade = "c"
    else:
        grade = "rejected"
    return {"grade": grade, "per_backbone_rho": per_bb.to_dict(),
            "threshold": RHO_THRESHOLD}
