"""Metric sanity: TSTR==oracle when generated==real; MMD->0 for identical
distributions; G_ng orders Gaussian < bimodal; correlation machinery."""
import numpy as np
import pandas as pd
import torch

from src.evaluation import (
    correlate_geometry_fidelity, gng_scores, invariance_grade, mmd_rbf,
    per_class_mmd, signal_features, tstr,
)


def _toy_feats(n_per=60, seed=0):
    rng = np.random.default_rng(seed)
    feats, labels = [], []
    for c in range(3):
        feats.append(rng.normal(loc=3 * c, scale=1.0, size=(n_per, 10)))
        labels.append(np.full(n_per, c))
    return np.vstack(feats), np.concatenate(labels)


def test_tstr_equals_oracle_when_gen_is_real():
    Xtr, ytr = _toy_feats(seed=0)
    Xte, yte = _toy_feats(seed=1)
    out = tstr(Xtr, ytr, Xtr, ytr, Xte, yte)
    for probe in ("logistic", "knn", "svm"):
        assert abs(out[probe]["ratio"] - 1.0) < 1e-9
        assert out[probe]["oracle_acc"] > 0.9


def test_mmd_zero_for_identical_and_positive_for_shifted():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(150, 8))
    Y = rng.normal(size=(150, 8))
    Z = rng.normal(loc=2.0, size=(150, 8))
    assert mmd_rbf(X, Y) < 0.02
    assert mmd_rbf(X, Z) > 10 * max(mmd_rbf(X, Y), 1e-6)
    pc = per_class_mmd(X, np.zeros(150), Z, np.zeros(150))
    assert pc[0] > 0.05


def test_gng_orders_gaussian_below_bimodal():
    rng = np.random.default_rng(0)
    gauss = rng.normal(size=(400, 8))
    bimodal = np.vstack([rng.normal(-3, 0.4, (200, 8)), rng.normal(3, 0.4, (200, 8))])
    g1, g2 = gng_scores(gauss), gng_scores(bimodal)
    assert g2["henze_zirkler"] > g1["henze_zirkler"]
    assert g2["epps_pulley"] > g1["epps_pulley"]


def test_signal_features_shape():
    x = torch.randn(6, 4, 512)
    f = signal_features(x, fft_bins=32)
    assert f.shape[0] == 6 and np.isfinite(f).all()


def test_correlation_and_grading():
    rows = []
    rng = np.random.default_rng(0)
    for bb in ("ours", "cvae"):
        for est in ("gaussian",):
            for c in range(1, 7):
                for sp in (12.0, 16.0):
                    gng = float(c)
                    for seed in range(3):
                        rows.append({"backbone": bb, "estimator": est, "class": c,
                                     "speed": sp, "seed": seed,
                                     "henze_zirkler": gng,
                                     "mmd": 0.1 * gng + rng.normal(0, 0.02)})
    df = correlate_geometry_fidelity(pd.DataFrame(rows))
    assert (df["spearman_rho"] > 0.8).all()
    assert invariance_grade(df)["grade"] == "a"
