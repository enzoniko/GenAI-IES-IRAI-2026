"""
T10: PINN Variant Comparison + UMAP Generation + Best Model Selection

Variant A: from-scratch PINN (T9, results/pinn.pth)
Variant B: old weights (T5 evaluated, metrics pulled from evidence files)

Generates:
  .sisyphus/evidence/task-10-variant-comparison.txt
  .sisyphus/evidence/task-10-best-umap.png   (150 DPI preview)
  assets/fig_pinn_umap_mafaulda.pdf           (300 DPI, vector)
"""

import sys
import os

sys.path.insert(0, ".")

import io
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from src.models.pinn import ConfigurablePINN
from src.models.feature_extractors import MathFeatureExtractor

# ─── Paths ────────────────────────────────────────────────────────────────────
PINN_CKPT   = "results/pinn.pth"
DATA_DIR    = "data/processed-mafaulda/16hz"
EVIDENCE_DIR = ".sisyphus/evidence"
ASSETS_DIR  = "assets"

os.makedirs(ASSETS_DIR, exist_ok=True)
os.makedirs(EVIDENCE_DIR, exist_ok=True)

# ─── Load PINN ────────────────────────────────────────────────────────────────
print("Loading PINN checkpoint ...")
ckpt = torch.load(PINN_CKPT, map_location="cpu")
pinn = ConfigurablePINN()
pinn.load_state_dict(ckpt["model_state_dict"])
pinn.eval()

X_min_ckpt = ckpt["X_min"].double()   # shape (10,)
X_max_ckpt = ckpt["X_max"].double()   # shape (10,)
y_min_ckpt = ckpt["y_min"].double()   # shape (4,)
y_max_ckpt = ckpt["y_max"].double()   # shape (4,)

print(f"  strategy: {ckpt.get('strategy')}, best_epoch: {ckpt.get('best_epoch')}")
print(f"  X_min: {X_min_ckpt.tolist()}")
print(f"  X_max: {X_max_ckpt.tolist()}")

# ─── Load Data ────────────────────────────────────────────────────────────────
DATA_FILES = [
    ("X_normal_trainingset.pth",                              0, "Normal"),
    ("X_imbalance_fault_20g_trainingset.pth",                 1, "Imbalance"),
    ("X_vertical_misalignment_fault_1.27mm_trainingset.pth",  2, "Vert.Misalign."),
    ("X_overhang_ball_fault_20g_trainingset.pth",             3, "Overhang.Ball"),
]

print("\nLoading data ...")
Xs, labels = [], []
for fname, lbl, cname in DATA_FILES:
    fpath = os.path.join(DATA_DIR, fname)
    X = torch.load(fpath, map_location="cpu").double()
    print(f"  {cname}: shape={X.shape}, dtype={X.dtype}")
    Xs.append(X)
    labels.extend([lbl] * X.shape[0])

X_all = torch.cat(Xs, dim=0)  # (N, T, 10)
y_all = np.array(labels)
N, T, C = X_all.shape
print(f"\nCombined X_all: {X_all.shape}  N={N}  T={T}  C={C}")

# ─── Normalize ────────────────────────────────────────────────────────────────
# x_norm = (x - x_min) / (x_max - x_min + 1e-10)
X_min_b = X_min_ckpt.view(1, 1, 10)
X_max_b = X_max_ckpt.view(1, 1, 10)
X_norm = (X_all - X_min_b) / (X_max_b - X_min_b + 1e-10)
print(f"Normalized range: [{X_norm.min().item():.4f}, {X_norm.max().item():.4f}]")

# ─── PINN forward + compute_residuals in batches ──────────────────────────────
print("\nRunning PINN forward pass ...")
X_flat = X_norm.view(N * T, 10)   # (N*T, 10)

PINN_BATCH = 8192
n_batches = (N * T + PINN_BATCH - 1) // PINN_BATCH

residuals_list = []
forces_list    = []

with torch.no_grad():
    for i in range(n_batches):
        s = i * PINN_BATCH
        e = min(s + PINN_BATCH, N * T)
        x_b = X_flat[s:e]

        pred_b = pinn(x_b)  # (B, 4) — also sets pinn.fA/fB/fC/fD

        r1, r2, r3, r4, _rM1, _rM2 = pinn.compute_residuals(
            x_b, pred_b,
            X_max=X_max_ckpt, X_min=X_min_ckpt,
            y_max=y_max_ckpt, y_min=y_min_ckpt,
        )
        res_b    = torch.cat([r1, r2, r3, r4], dim=1)                     # (B, 4)
        forces_b = torch.cat([pinn.fA, pinn.fB, pinn.fC, pinn.fD], dim=1) # (B, 4)

        residuals_list.append(res_b)
        forces_list.append(forces_b)

        if (i + 1) % 20 == 0 or (i + 1) == n_batches:
            pct = (i + 1) / n_batches * 100
            print(f"  batch {i+1}/{n_batches}  ({pct:.0f}%)")

residuals = torch.cat(residuals_list, dim=0)  # (N*T, 4)
forces    = torch.cat(forces_list,    dim=0)  # (N*T, 4)
print(f"Residuals: {residuals.shape}, mean_abs={residuals.abs().mean().item():.2f}")

# ─── Stack 8-channel physics signal ───────────────────────────────────────────
physics = torch.cat([residuals, forces], dim=1)  # (N*T, 8)
physics_seq = physics.view(N, T, 8)               # (N, T, 8)
print(f"physics_seq: {physics_seq.shape}")

# ─── MathFeatureExtractor → 2240-dim embeddings ───────────────────────────────
print("\nExtracting math features ...")
feat_extractor = MathFeatureExtractor(in_channels=8)
feat_extractor.eval()

feat_list = []
FEAT_BATCH = 16
n_feat_batches = (N + FEAT_BATCH - 1) // FEAT_BATCH

with torch.no_grad():
    for i in range(n_feat_batches):
        s = i * FEAT_BATCH
        e = min(s + FEAT_BATCH, N)
        feats = feat_extractor(physics_seq[s:e])  # (B, 2240)
        feat_list.append(feats)
        if (i + 1) % 5 == 0 or (i + 1) == n_feat_batches:
            print(f"  batch {i+1}/{n_feat_batches}")

features_raw = torch.cat(feat_list, dim=0).numpy()  # (N, 2240)
print(f"features_raw: {features_raw.shape}  range=[{features_raw.min():.4f}, {features_raw.max():.4f}]")

# Replace NaN/Inf with 0
n_bad = np.sum(~np.isfinite(features_raw))
if n_bad > 0:
    print(f"  WARNING: {n_bad} non-finite values, replacing with 0")
    features_raw = np.nan_to_num(features_raw, nan=0.0, posinf=0.0, neginf=0.0)

# ─── Silhouette (raw) ─────────────────────────────────────────────────────────
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_raw)

sil_raw = silhouette_score(features_scaled, y_all)
print(f"\nSilhouette (raw 2240): {sil_raw:.4f}")

# ─── PCA(50) ──────────────────────────────────────────────────────────────────
pca = PCA(n_components=50, random_state=42)
features_pca50 = pca.fit_transform(features_scaled)
print(f"PCA50 explained variance: {pca.explained_variance_ratio_.sum():.6f}")

sil_pca50 = silhouette_score(features_pca50, y_all)
print(f"Silhouette (PCA50): {sil_pca50:.4f}")

knn = KNeighborsClassifier(n_neighbors=5)
cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
knn_acc = cross_val_score(knn, features_pca50, y_all, cv=cv, scoring="accuracy").mean()
print(f"kNN accuracy (PCA50): {knn_acc:.4f}")

# ─── UMAP ─────────────────────────────────────────────────────────────────────
print("\nRunning UMAP ...")
reducer  = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
embedding = reducer.fit_transform(features_pca50)
print(f"UMAP embedding: {embedding.shape}")

# ─── Best Variant Selection ────────────────────────────────────────────────────
# Variant B from T5 evidence
sil_B_pca50  = -0.203745
knn_B_pca50  =  0.800786

print(f"\n=== Variant A (T9, from-scratch) ===")
print(f"  Silhouette_raw2240: {sil_raw:.4f}")
print(f"  Silhouette_pca50:   {sil_pca50:.4f}")
print(f"  kNN_accuracy_pca50: {knn_acc:.4f}")

print(f"\n=== Variant B (T5 old weights) ===")
print(f"  Silhouette_pca50:   {sil_B_pca50:.4f}")
print(f"  kNN_accuracy_pca50: {knn_B_pca50:.4f}")

if sil_pca50 >= sil_B_pca50:
    best_variant = "A"
    justification = (
        f"Variant A (from-scratch T9) has higher Silhouette ({sil_pca50:.4f} vs {sil_B_pca50:.4f}). "
        "Old weights exhibit omega mismatch (153 vs 101 rad/s) causing residuals ~1e9, "
        "making their kNN accuracy (0.801) spurious."
    )
    best_label = f"from-scratch (T9, results/pinn.pth)"
else:
    best_variant = "B"
    justification = (
        f"Variant B (old weights) has higher Silhouette ({sil_B_pca50:.4f} vs {sil_pca50:.4f}). "
        "However note: old-weight residuals explode 1e9x on 16Hz due to omega mismatch. "
        "Both variants show weak clustering — fallback framing applies."
    )
    best_label = "old weights (T5, legacy checkpoint)"

print(f"\nBEST VARIANT: {best_variant}  ({best_label})")

# ─── Figure: colorblind-friendly palette ─────────────────────────────────────
CLASS_META = [
    (0, "Normal",         "#1f77b4"),
    (1, "Imbalance",      "#ff7f0e"),
    (2, "Vert.Misalign.", "#2ca02c"),
    (3, "Overhang.Ball",  "#d62728"),
]

def make_umap_figure(emb, y, class_meta, title, figsize=(3.5, 3.5)):
    fig, ax = plt.subplots(figsize=figsize)
    for lbl, cname, color in class_meta:
        mask = (y == lbl)
        ax.scatter(
            emb[mask, 0], emb[mask, 1],
            label=cname, c=color,
            s=8, alpha=0.7, rasterized=True,
        )
    ax.set_title(title, fontsize=7, pad=4)
    ax.set_xlabel("UMAP-1", fontsize=7)
    ax.set_ylabel("UMAP-2", fontsize=7)
    ax.tick_params(labelsize=6)
    leg = ax.legend(fontsize=6, markerscale=1.5, loc="best",
                    framealpha=0.8, edgecolor="gray")
    fig.tight_layout(pad=0.5)
    return fig

TITLE = "Physics-Informed Feature Space (PINN Residuals, PCA-50, UMAP)"

# PNG preview (150 DPI)
fig_png = make_umap_figure(embedding, y_all, CLASS_META, TITLE)
png_path = os.path.join(EVIDENCE_DIR, "task-10-best-umap.png")
fig_png.savefig(png_path, dpi=150, bbox_inches="tight")
print(f"\nSaved PNG preview: {png_path}")
plt.close(fig_png)

# PDF paper-ready (300 DPI, vector)
fig_pdf = make_umap_figure(embedding, y_all, CLASS_META, TITLE)
pdf_path = os.path.join(ASSETS_DIR, "fig_pinn_umap_mafaulda.pdf")
fig_pdf.savefig(pdf_path, dpi=300, bbox_inches="tight", format="pdf")
print(f"Saved PDF paper-ready: {pdf_path}")
plt.close(fig_pdf)

# ─── Evidence file ─────────────────────────────────────────────────────────────
evidence_txt = f"""=== T10: PINN Variant Comparison ===
Variant A: from-scratch (T9, 250 epochs, results/pinn.pth)
  Silhouette_raw2240: {sil_raw:.4f}
  Silhouette_pca50: {sil_pca50:.4f}
  kNN_accuracy_pca50: {knn_acc:.4f}

Variant B: old weights as-is (T5 eval)
  Silhouette_pca50: {sil_B_pca50:.4f}
  kNN_accuracy_pca50: {knn_B_pca50:.4f}

BEST VARIANT: {best_variant}
JUSTIFICATION: {justification}
FALLBACK STATUS: Active -- 'physics-informed representation' framing (no clean clusters)

Figures:
  PNG preview: .sisyphus/evidence/task-10-best-umap.png
  PDF paper:   assets/fig_pinn_umap_mafaulda.pdf
"""

evidence_path = os.path.join(EVIDENCE_DIR, "task-10-variant-comparison.txt")
with io.open(evidence_path, "w", encoding="utf-8") as f:
    f.write(evidence_txt)
print(f"\nSaved evidence: {evidence_path}")
print("\n" + evidence_txt)
print("T10 complete.")
