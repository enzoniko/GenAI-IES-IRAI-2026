"""
Task 14 - Phase 1 Validation: Full reconstruction pipeline on MaFaulDa 16Hz data.
Computes per-class metrics and generates figures for the IEEE paper.
NO training. Inference only.
"""

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

REPO = r"D:\WORK\GenAI\GenAI-IES-IRAI-2026"
sys.path.insert(0, REPO)

from src.models.ts_jepa import TSJEPA
from src.models.decoder1 import Decoder1
from src.models.decoder2_cvae import Decoder2CVAE

# ─── Paths ───────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(REPO, "results")
BASE_DATA   = os.path.join(REPO, "data", "processed-mafaulda", "16hz")
SYN_DIR     = os.path.join(REPO, "results-synthetic")
EVIDENCE    = os.path.join(REPO, ".sisyphus", "evidence")
ASSETS      = os.path.join(REPO, "assets")
os.makedirs(EVIDENCE, exist_ok=True)
os.makedirs(ASSETS,   exist_ok=True)

Y_FILES = {
    0: os.path.join(BASE_DATA, "Y_normal_trainingset.pth"),
    1: os.path.join(BASE_DATA, "Y_imbalance_fault_20g_trainingset.pth"),
    2: os.path.join(BASE_DATA, "Y_vertical_misalignment_fault_1.27mm_trainingset.pth"),
    3: os.path.join(BASE_DATA, "Y_overhang_ball_fault_20g_trainingset.pth"),
}
CLASS_NAMES = {
    0: "Normal",
    1: "Imbalance_20g",
    2: "Vert_Misalign_1.27mm",
    3: "Overhang_Ball_20g",
}
PLOT_LABELS = {
    0: "Normal",
    1: "Imbalance 20g",
    2: "Vert. Misalign. 1.27mm",
    3: "Overhang Ball 20g",
}

device = torch.device("cpu")

# ─── Load normalization metadata ──────────────────────────────────────────────
print("Loading normalization metadata...")
meta = torch.load(os.path.join(RESULTS_DIR, "normalization_metadata.pth"), weights_only=False)
y_min = meta["y_min"].to(device)  # shape [4]
y_max = meta["y_max"].to(device)  # shape [4]
y_range = (y_max - y_min)         # shape [4]
print(f"  y_min: {y_min.tolist()}")
print(f"  y_max: {y_max.tolist()}")

# ─── Load models ──────────────────────────────────────────────────────────────
print("Loading TS-JEPA...")
tsjepa = TSJEPA(in_channels=4)
ckpt = torch.load(os.path.join(RESULTS_DIR, "ts_jepa.pth"), map_location=device, weights_only=False)
# Handle various checkpoint formats
if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    tsjepa.load_state_dict(ckpt["model_state_dict"])
elif isinstance(ckpt, dict) and "state_dict" in ckpt:
    tsjepa.load_state_dict(ckpt["state_dict"])
elif isinstance(ckpt, dict) and any(k.startswith("tokenizer") or k.startswith("context_encoder") for k in ckpt.keys()):
    tsjepa.load_state_dict(ckpt)
else:
    tsjepa.load_state_dict(ckpt)
tsjepa.eval()
tsjepa.to(device)
print("  TS-JEPA loaded OK")

print("Loading Decoder1...")
decoder1 = Decoder1(d_model=128, seq_length=3014, out_channels=4)
ckpt = torch.load(os.path.join(RESULTS_DIR, "decoder1.pth"), map_location=device, weights_only=False)
if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    decoder1.load_state_dict(ckpt["model_state_dict"])
elif isinstance(ckpt, dict) and "state_dict" in ckpt:
    decoder1.load_state_dict(ckpt["state_dict"])
elif isinstance(ckpt, dict) and any(k.startswith("fc_proj") or k.startswith("decoder_cnn") for k in ckpt.keys()):
    decoder1.load_state_dict(ckpt)
else:
    decoder1.load_state_dict(ckpt)
decoder1.eval()
decoder1.to(device)
print("  Decoder1 loaded OK")

print("Loading Decoder2CVAE...")
decoder2 = Decoder2CVAE(seq_length=3014, in_channels=4, context_dim=128, latent_dim=64, num_classes=4)
ckpt = torch.load(os.path.join(RESULTS_DIR, "decoder2.pth"), map_location=device, weights_only=False)
if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    decoder2.load_state_dict(ckpt["model_state_dict"])
elif isinstance(ckpt, dict) and "state_dict" in ckpt:
    decoder2.load_state_dict(ckpt["state_dict"])
elif isinstance(ckpt, dict) and any(k.startswith("encoder") or k.startswith("decoder") for k in ckpt.keys()):
    decoder2.load_state_dict(ckpt)
else:
    decoder2.load_state_dict(ckpt)
decoder2.eval()
decoder2.to(device)
print("  Decoder2CVAE loaded OK")

# ─── Per-class inference ──────────────────────────────────────────────────────
print("\nRunning inference per class...")

metrics = {}
# Store first-sample data for plots
plot_data = {}  # cls -> {"y_norm", "envelope_norm", "full_recon_norm"}

BATCH_SIZE = 16

for cls_id in range(4):
    print(f"\n  Class {cls_id} ({CLASS_NAMES[cls_id]})...")
    y_raw = torch.load(Y_FILES[cls_id], weights_only=True)  # (N, 3014, 4)
    # Permute: (N, 3014, 4) → (N, 4, 3014)
    y_raw = y_raw.permute(0, 2, 1).to(device)  # (N, 4, 3014)
    N = y_raw.shape[0]

    # Normalize: y_norm = (y - y_min) / (y_max - y_min)
    # y_min shape [4], y shape (N, 4, T) → broadcast y_min[:, None]
    y_norm = (y_raw - y_min[:, None]) / y_range[:, None]  # (N, 4, 3014)

    all_envelope_norm = []
    all_full_recon_norm = []
    label_tensor = torch.full((BATCH_SIZE,), cls_id, dtype=torch.long, device=device)

    with torch.no_grad():
        for start in range(0, N, BATCH_SIZE):
            end = min(start + BATCH_SIZE, N)
            y_batch = y_norm[start:end]   # (B, 4, 3014)
            b = y_batch.shape[0]
            lbl = label_tensor[:b]

            z_macro     = tsjepa.get_z_macro(y_batch)          # (B, 128)
            envelope    = decoder1(z_macro)                     # (B, 4, 3014)
            jitter      = decoder2.sample(z_macro, lbl)        # (B, 4, 3014)
            full_recon  = envelope + jitter                    # (B, 4, 3014)

            all_envelope_norm.append(envelope.cpu())
            all_full_recon_norm.append(full_recon.cpu())

    all_envelope_norm  = torch.cat(all_envelope_norm, dim=0)   # (N, 4, 3014)
    all_full_recon_norm = torch.cat(all_full_recon_norm, dim=0) # (N, 4, 3014)
    y_norm_cpu = y_norm.cpu()

    # Compute RMSE over all N, 4, T
    envelope_rmse = torch.sqrt(torch.mean((all_envelope_norm - y_norm_cpu) ** 2)).item()
    full_rmse     = torch.sqrt(torch.mean((all_full_recon_norm - y_norm_cpu) ** 2)).item()
    improvement   = (envelope_rmse - full_rmse) / envelope_rmse * 100.0

    metrics[cls_id] = {
        "envelope_RMSE": envelope_rmse,
        "full_RMSE": full_rmse,
        "improvement_pct": improvement,
    }
    print(f"    envelope_RMSE = {envelope_rmse:.4f}")
    print(f"    full_RMSE     = {full_rmse:.4f}")
    print(f"    improvement   = {improvement:.2f}%")

    # Save first sample for plotting
    plot_data[cls_id] = {
        "y_norm":         y_norm_cpu[0],            # (4, 3014)
        "envelope_norm":  all_envelope_norm[0],     # (4, 3014)
        "full_recon_norm": all_full_recon_norm[0],  # (4, 3014)
    }

# ─── Reconstruction overlay plot (4 subplots) ────────────────────────────────
print("\nGenerating reconstruction overlay plot...")

T_SHOW = 200
CHAN   = 0   # channel 0

fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
colors = plt.cm.tab10.colors

for i, cls_id in enumerate(range(4)):
    ax = axes[i]
    pd = plot_data[cls_id]
    t = np.arange(T_SHOW)

    orig     = pd["y_norm"][CHAN, :T_SHOW].numpy()
    envelope = pd["envelope_norm"][CHAN, :T_SHOW].numpy()
    full_rec = pd["full_recon_norm"][CHAN, :T_SHOW].numpy()

    ax.plot(t, orig,     color=colors[0], lw=1.2,  label="Original",         zorder=3)
    ax.plot(t, envelope, color=colors[1], lw=1.2, ls="--", label="Envelope (D1)",  zorder=2)
    ax.plot(t, full_rec, color=colors[2], lw=1.2, ls="--", label="Full Recon (D1+D2)", zorder=1)
    ax.set_ylabel(f"Norm. Amp.", fontsize=9)
    ax.set_title(f"Class {cls_id}: {PLOT_LABELS[cls_id]}", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Time step", fontsize=10)
fig.suptitle("Phase 1 Reconstruction: Original vs Envelope vs Full Reconstruction (ch0, first 200 steps)",
             fontsize=11, y=1.01)
fig.tight_layout()

overlay_path = os.path.join(EVIDENCE, "task-14-reconstruction-overlay.png")
fig.savefig(overlay_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {overlay_path}")

# ─── Twin plot ────────────────────────────────────────────────────────────────
print("\nGenerating twin plot...")

# Find best synthetic image for left panel
syn_candidates = [
    "umap_latent_space.png",
    "evaluation_class_0.png",
    "umap_sdedit_trajectory_class_1.png",
]
syn_img_path = None
for cand in syn_candidates:
    p = os.path.join(SYN_DIR, cand)
    if os.path.exists(p):
        syn_img_path = p
        print(f"  Using synthetic image: {cand}")
        break

fig_twin, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(7, 3.5))

# LEFT panel: synthetic oscillator (from T8)
if syn_img_path is not None:
    syn_img = mpimg.imread(syn_img_path)
    ax_left.imshow(syn_img, aspect="auto")
    ax_left.axis("off")
    ax_left.set_title("(a) Synthetic Oscillator\n(Phase 1, T8)", fontsize=9)
else:
    ax_left.set_facecolor("#f0f0f0")
    ax_left.text(0.5, 0.5, "Synthetic results\n(see results-synthetic/)",
                 ha="center", va="center", fontsize=9, transform=ax_left.transAxes,
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax_left.set_title("(a) Synthetic Oscillator\n(Phase 1, T8)", fontsize=9)
    ax_left.set_xticks([]); ax_left.set_yticks([])

# RIGHT panel: MaFaulDa reconstruction overlay (4 classes, channel 0, 200 steps)
ax_right.set_prop_cycle(color=[colors[0], colors[1], colors[2], colors[3]])
t = np.arange(T_SHOW)
for cls_id in range(4):
    pd = plot_data[cls_id]
    orig_ch0 = pd["y_norm"][CHAN, :T_SHOW].numpy()
    full_ch0 = pd["full_recon_norm"][CHAN, :T_SHOW].numpy()
    lbl = PLOT_LABELS[cls_id]
    c = colors[cls_id]
    ax_right.plot(t, orig_ch0,  color=c, lw=1.0, alpha=0.8, label=f"{lbl}")
    ax_right.plot(t, full_ch0,  color=c, lw=1.0, ls="--", alpha=0.6)

ax_right.set_xlabel("Time step", fontsize=9)
ax_right.set_ylabel("Normalised amplitude", fontsize=9)
ax_right.set_title("(b) MaFaulDa 16 Hz\n(Full Recon., ch0, 200 steps)", fontsize=9)
ax_right.legend(fontsize=7, loc="upper right", framealpha=0.7)
ax_right.grid(True, alpha=0.3)
ax_right.tick_params(labelsize=8)

fig_twin.tight_layout(pad=1.0)

# Save preview PNG (150 DPI)
preview_path = os.path.join(EVIDENCE, "task-14-twin-plot-preview.png")
fig_twin.savefig(preview_path, dpi=150, bbox_inches="tight")
print(f"  Saved preview: {preview_path}")

# Save paper PDF (300 DPI, vector)
pdf_path = os.path.join(ASSETS, "fig_phase1_validation.pdf")
fig_twin.savefig(pdf_path, dpi=300, bbox_inches="tight", format="pdf")
print(f"  Saved PDF: {pdf_path}")
plt.close(fig_twin)

# ─── Write metrics file ───────────────────────────────────────────────────────
print("\nWriting metrics file...")

VAL_ELBO        = 0.0066
SILHOUETTE_PCA50 = 0.0789

lines = [
    "Task 14 - Phase 1 Validation Metrics",
    "=====================================",
    f"Class 0 (Normal):             envelope_RMSE={metrics[0]['envelope_RMSE']:.4f}, full_RMSE={metrics[0]['full_RMSE']:.4f}, improvement={metrics[0]['improvement_pct']:.1f}%",
    f"Class 1 (Imbalance_20g):      envelope_RMSE={metrics[1]['envelope_RMSE']:.4f}, full_RMSE={metrics[1]['full_RMSE']:.4f}, improvement={metrics[1]['improvement_pct']:.1f}%",
    f"Class 2 (Vert_1.27mm):        envelope_RMSE={metrics[2]['envelope_RMSE']:.4f}, full_RMSE={metrics[2]['full_RMSE']:.4f}, improvement={metrics[2]['improvement_pct']:.1f}%",
    f"Class 3 (Overhang_20g):       envelope_RMSE={metrics[3]['envelope_RMSE']:.4f}, full_RMSE={metrics[3]['full_RMSE']:.4f}, improvement={metrics[3]['improvement_pct']:.1f}%",
    "",
    f"val_ELBO (T13): {VAL_ELBO}",
    f"Silhouette_pca50 (T10): {SILHOUETTE_PCA50}",
    "",
    f"Twin plot: assets/fig_phase1_validation.pdf",
]

metrics_path = os.path.join(EVIDENCE, "task-14-phase1-metrics.txt")
with open(metrics_path, "w", encoding="utf-8", errors="replace") as f:
    f.write("\n".join(lines) + "\n")
print(f"  Saved: {metrics_path}")

print("\n=== Task 14 DONE ===")
print("Evidence files:")
print(f"  {metrics_path}")
print(f"  {overlay_path}")
print(f"  {preview_path}")
print(f"  {pdf_path}")
