#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from datetime import datetime
from pathlib import Path

# MUST SET ORACLE_MODE BEFORE IMPORTING PriorWorkOracle — constructor reads cfg at init time
import src.configs as cfg
cfg.ORACLE_MODE = "raw"

from src.models.oracles import PriorWorkOracle

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/processed-mafaulda/16hz"
RESULTS_DIR = ROOT / "results"
EVIDENCE_DIR = ROOT / ".sisyphus/evidence"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

TEST_FILES = [
    ("Y_normal_testset.pth",                              "X_normal_testset.pth",                              0),
    ("Y_imbalance_fault_20g_testset.pth",                 "X_imbalance_fault_20g_testset.pth",                 1),
    ("Y_vertical_misalignment_fault_1.27mm_testset.pth",  "X_vertical_misalignment_fault_1.27mm_testset.pth",  2),
    ("Y_overhang_ball_fault_20g_testset.pth",             "X_overhang_ball_fault_20g_testset.pth",             3),
]

BATCH_SIZE = 64
PINN_BASELINE_SILHOUETTE = 0.0789
PINN_BASELINE_KNN = 0.7783


def extract_embeddings(oracle, device, batch_size=64):
    all_embeddings = []
    all_labels = []
    class_counts = {}

    for y_fname, x_fname, label in TEST_FILES:
        y_path = DATA_DIR / y_fname
        x_path = DATA_DIR / x_fname

        if not y_path.exists():
            raise FileNotFoundError(f"Y file not found: {y_path}")
        if not x_path.exists():
            raise FileNotFoundError(f"X file not found: {x_path}")

        Y = torch.load(y_path, map_location="cpu", weights_only=True)
        X = torch.load(x_path, map_location="cpu", weights_only=True)

        print(f"  Class {label} ({y_fname}): Y={tuple(Y.shape)}, X={tuple(X.shape)}")

        assert Y.ndim == 3 and Y.shape[-1] == 4, f"Expected Y (N,T,4), got {Y.shape}"
        assert X.ndim == 3 and X.shape[-1] == 10, f"Expected X (N,T,10), got {X.shape}"

        N = Y.shape[0]
        class_counts[label] = N

        # Y stored as (N, T, 4) — oracle forward expects (N, 4, T)
        Y_t = Y.permute(0, 2, 1).float()
        # omega: col 8 of X, one scalar per window
        omega = X[:, 0, 8].float()

        embeddings_list = []
        oracle.eval()
        with torch.no_grad():
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                y_batch = Y_t[start:end].to(device)
                omega_batch = omega[start:end].to(device)
                emb = oracle(y_batch, omega=omega_batch)
                embeddings_list.append(emb.cpu())

        class_embs = torch.cat(embeddings_list, dim=0).numpy()
        all_embeddings.append(class_embs)
        all_labels.extend([label] * N)

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.array(all_labels)

    return embeddings, labels, class_counts


def compute_metrics(embeddings, labels):
    n_samples, n_features = embeddings.shape
    n_pca = min(50, n_samples, n_features)
    print(f"\nFitting PCA({n_pca}) on {embeddings.shape} embeddings (capped by n_samples={n_samples})...")
    pca = PCA(n_components=n_pca, random_state=42)
    pca_embs = pca.fit_transform(embeddings)
    explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA({n_pca}) explained variance: {explained:.4f}")

    print("Computing silhouette score...")
    sil = silhouette_score(pca_embs, labels)
    print(f"  silhouette_pca50 = {sil:.4f}")

    n_cv = max(2, min(5, int(np.bincount(labels.astype(int)).min())))
    n_fold_train = n_samples - (n_samples // n_cv)
    n_neighbors = min(5, n_fold_train)
    print(f"Computing {n_cv}-fold kNN (k={n_neighbors}) accuracy...")
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_scores = cross_val_score(knn, pca_embs, labels, cv=n_cv, scoring="accuracy")
    knn_mean = knn_scores.mean()
    print(f"  kNN_{n_cv}fold_pca50  = {knn_mean:.4f}  (per-fold: {[f'{s:.4f}' for s in knn_scores]})")

    return pca_embs, sil, knn_mean, explained, n_pca, n_cv, n_neighbors


def make_scatter(pca_embs, labels, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    class_names = ["Normal", "Imbalance 20g", "Vert.Misalign 1.27mm", "Overhang Ball 20g"]

    for cls_idx in range(4):
        mask = labels == cls_idx
        ax.scatter(
            pca_embs[mask, 0], pca_embs[mask, 1],
            c=colors[cls_idx], label=class_names[cls_idx],
            alpha=0.5, s=10, linewidths=0
        )

    ax.set_title("Raw Bypass Clustering — PCA 2D Scatter\n(umap-learn not installed; using PCA components 1–2)", fontsize=11)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(markerscale=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nPCA scatter saved to: {out_path}")


def build_evidence(sil, knn_mean, class_counts, embeddings_shape, explained):
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_n = embeddings_shape[0]
    dim = embeddings_shape[1]

    sil_vs = "BETTER" if sil > PINN_BASELINE_SILHOUETTE else ("EQUAL" if abs(sil - PINN_BASELINE_SILHOUETTE) < 1e-4 else "WORSE")
    knn_vs = "BETTER" if knn_mean > PINN_BASELINE_KNN else ("EQUAL" if abs(knn_mean - PINN_BASELINE_KNN) < 1e-4 else "WORSE")
    viable = "YES" if sil > 0.05 else "NO"

    counts_str = ", ".join(f"{k}: {v}" for k, v in sorted(class_counts.items()))

    return f"""=== Task 8: Raw Bypass Clustering Evaluation ===
Date: {date_str}
ORACLE_MODE: raw

Embeddings: ({total_n}, {dim}) for {total_n} test windows across 4 classes
Class counts: {{{counts_str}}}
PCA(50) explained variance: {explained:.4f}

Results:
  silhouette_pca50    = {sil:.4f}
  kNN_5fold_pca50     = {knn_mean:.4f}

Baseline comparison (PINN mode from prior work):
  silhouette_pca50    = {PINN_BASELINE_SILHOUETTE}
  kNN_5fold_pca50     = {PINN_BASELINE_KNN}

Verdict:
  Raw bypass viable for SDEdit guidance: {viable} (threshold: silhouette > 0.05)
  Raw bypass silhouette vs PINN silhouette: {sil_vs}
  Raw bypass kNN vs PINN kNN: {knn_vs}

Notes:
  umap-learn not installed — visualization saved as PCA 2D scatter instead.
  Scatter plot: results/umap_raw_bypass_clustering.png
"""


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"ORACLE_MODE: {cfg.ORACLE_MODE}")

    print("\nInstantiating PriorWorkOracle in raw mode...")
    oracle = PriorWorkOracle(in_channels=4).to(device)
    oracle.eval()
    print(f"  embed_dim = {oracle.embed_dim}")

    print("\nLoading test data and extracting embeddings...")
    embeddings, labels, class_counts = extract_embeddings(oracle, device, batch_size=BATCH_SIZE)
    print(f"\nTotal embeddings: {embeddings.shape}")

    if not np.isfinite(embeddings).all():
        n_bad = int((~np.isfinite(embeddings)).sum())
        print(f"WARNING: {n_bad} non-finite values in embeddings — replacing with 0")
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)

    pca_embs, sil, knn_mean, explained, n_pca, n_cv, n_neighbors = compute_metrics(embeddings, labels)

    scatter_path = RESULTS_DIR / "umap_raw_bypass_clustering.png"
    make_scatter(pca_embs, labels, scatter_path)

    evidence = build_evidence(sil, knn_mean, class_counts, embeddings.shape, explained)
    print("\n" + evidence)

    evidence_path = EVIDENCE_DIR / "task-8-raw-bypass-clustering.txt"
    evidence_path.write_text(evidence)
    print(f"Evidence saved to: {evidence_path}")

    print("\n=== SUMMARY ===")
    print(f"  silhouette_pca50 = {sil:.4f}  (baseline: {PINN_BASELINE_SILHOUETTE})")
    print(f"  kNN_5fold_pca50  = {knn_mean:.4f}  (baseline: {PINN_BASELINE_KNN})")
    viable = sil > 0.05
    print(f"  Raw bypass VIABLE: {'YES' if viable else 'NO'} (silhouette > 0.05 threshold)")


if __name__ == "__main__":
    main()
