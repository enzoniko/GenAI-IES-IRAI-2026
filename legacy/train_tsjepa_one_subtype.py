"""
Custom TS-JEPA training script: ONE subtype per fault class (4 classes total).
- Loads exactly 4 .pth file pairs (one per class)
- Computes and saves normalization metadata
- Trains TS-JEPA via train_phase1_tsjepa (max 20 epochs)
- Saves checkpoint to results/ts_jepa.pth
- Generates UMAP scatter and Silhouette score
- Writes evidence files
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for Windows
import matplotlib.pyplot as plt

# Project imports
from src.models.ts_jepa import TSJEPA
from src.pipelines.train_phase1 import train_phase1_tsjepa
from src.data.mafaulda_dataset import BatchTuple, mafaulda_collate_fn
import src.configs as cfg

# ─────────────────────────────────────────────
# 1. ONE-SUBTYPE FILE MAP
# ─────────────────────────────────────────────
ONE_SUBTYPE_FILES = {
    0: {  # Normal
        'Y': 'data/processed-mafaulda/16hz/Y_normal_trainingset.pth',
        'X': 'data/processed-mafaulda/16hz/X_normal_trainingset.pth',
    },
    1: {  # Imbalance 20g
        'Y': 'data/processed-mafaulda/16hz/Y_imbalance_fault_20g_trainingset.pth',
        'X': 'data/processed-mafaulda/16hz/X_imbalance_fault_20g_trainingset.pth',
    },
    2: {  # Vertical misalignment 1.27mm
        'Y': 'data/processed-mafaulda/16hz/Y_vertical_misalignment_fault_1.27mm_trainingset.pth',
        'X': 'data/processed-mafaulda/16hz/X_vertical_misalignment_fault_1.27mm_trainingset.pth',
    },
    3: {  # Overhang ball 20g
        'Y': 'data/processed-mafaulda/16hz/Y_overhang_ball_fault_20g_trainingset.pth',
        'X': 'data/processed-mafaulda/16hz/X_overhang_ball_fault_20g_trainingset.pth',
    }
}

# ─────────────────────────────────────────────
# 2. DATASET
# ─────────────────────────────────────────────
class OneSubtypeDataset(Dataset):
    """In-memory dataset for one-subtype-per-class training."""
    def __init__(self, y_list, omega_list, label_list):
        # y_list: list of tensors (N_i, 4, 3014) — already normalized
        # omega_list: list of 1-D tensors (N_i,)
        # label_list: list of 1-D long tensors (N_i,)
        self.all_y = torch.cat(y_list, dim=0)       # (total_N, 4, 3014)
        self.all_omega = torch.cat(omega_list, dim=0)  # (total_N,)
        self.all_labels = torch.cat(label_list, dim=0)  # (total_N,)

    def __len__(self):
        return self.all_y.shape[0]

    def __getitem__(self, idx):
        trace = self.all_y[idx]
        label = self.all_labels[idx]
        omega = self.all_omega[idx].item()
        # Contract: (raw, clean, label, omega)
        return trace, trace, label, omega


# ─────────────────────────────────────────────
# 3. LOAD DATA + COMPUTE NORMALIZATION
# ─────────────────────────────────────────────
def load_data_and_compute_norm():
    print("=" * 60)
    print("Loading one-subtype-per-class data ...")
    print("=" * 60)

    y_raw_list = []   # unnormalized Y tensors
    x_raw_list = []   # X tensors (for X_min/X_max on cols 0-7)
    omega_list = []
    label_list = []

    for label_idx, paths in ONE_SUBTYPE_FILES.items():
        y_path = paths['Y']
        x_path = paths['X']

        assert os.path.exists(y_path), f"Y file missing: {y_path}"
        assert os.path.exists(x_path), f"X file missing: {x_path}"

        # Y: (N, 3014, 4) → transpose → (N, 4, 3014)
        y_tensor = torch.load(y_path, map_location='cpu', weights_only=True)
        y_tensor = y_tensor.transpose(1, 2).float()  # (N, 4, 3014)

        # X: (N, 3014, 10)
        x_tensor = torch.load(x_path, map_location='cpu', weights_only=True).float()

        # Omega from col 8
        omega_vals = x_tensor[:, 0, 8].float()  # (N,)

        print(f"  Class {label_idx}: Y={y_tensor.shape}, X={x_tensor.shape}, omega mean={omega_vals.mean():.2f}")

        y_raw_list.append(y_tensor)
        x_raw_list.append(x_tensor)
        omega_list.append(omega_vals)
        label_list.append(torch.full((y_tensor.shape[0],), label_idx, dtype=torch.long))

    # ── Global per-channel Y min/max across ALL classes ──
    all_y_combined = torch.cat(y_raw_list, dim=0)  # (total_N, 4, 3014)
    # min/max over batch and time dimensions, per channel
    # shape: (4,)
    y_min = all_y_combined.min(dim=0)[0].min(dim=-1)[0]   # per channel
    y_max = all_y_combined.max(dim=0)[0].max(dim=-1)[0]

    # ── Global per-feature X min/max across ALL classes (cols 0-7) ──
    all_x_combined = torch.cat(x_raw_list, dim=0)  # (total_N, 3014, 10)
    X_min = all_x_combined[:, :, :8].reshape(-1, 8).min(dim=0)[0]  # (8,)
    X_max = all_x_combined[:, :, :8].reshape(-1, 8).max(dim=0)[0]

    print(f"\ny_min (per channel): {y_min.tolist()}")
    print(f"y_max (per channel): {y_max.tolist()}")
    print(f"X_min (cols 0-7):    {X_min.tolist()}")
    print(f"X_max (cols 0-7):    {X_max.tolist()}")

    # ── Save normalization metadata ──
    os.makedirs('results', exist_ok=True)
    norm_meta = {'y_min': y_min, 'y_max': y_max, 'X_min': X_min, 'X_max': X_max}
    torch.save(norm_meta, 'results/normalization_metadata.pth')
    print("Saved: results/normalization_metadata.pth")

    # ── Normalize Y tensors ──
    y_min_f = y_min.view(1, 4, 1)
    y_max_f = y_max.view(1, 4, 1)

    y_norm_list = []
    for y_tensor in y_raw_list:
        y_norm = (y_tensor - y_min_f) / (y_max_f - y_min_f + 1e-12)
        y_norm_list.append(y_norm)

    return y_norm_list, omega_list, label_list, all_y_combined.shape[0]


# ─────────────────────────────────────────────
# 4. MAIN TRAINING
# ─────────────────────────────────────────────
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load and normalize
    y_norm_list, omega_list, label_list, total_n = load_data_and_compute_norm()
    print(f"\nTotal training windows: {total_n}")

    # Build dataset + 80/20 interleaved split
    dataset = OneSubtypeDataset(y_norm_list, omega_list, label_list)
    N = len(dataset)
    k = 5  # every 5th sample → val (20%)
    val_indices   = list(range(0, N, k))
    train_indices = [i for i in range(N) if i % k != 0]

    train_ds = Subset(dataset, train_indices)
    val_ds   = Subset(dataset, val_indices)

    batch_size = cfg.PHASE1_TRAIN_SETTINGS['batch_size']
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=mafaulda_collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=mafaulda_collate_fn)

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    print(f"Batch size: {batch_size}, Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ── Build model ──
    ts_jepa = TSJEPA(in_channels=4).to(device)
    print(f"\nTS-JEPA model: {sum(p.numel() for p in ts_jepa.parameters()):,} parameters")

    # ── Intercept stdout to capture epoch losses ──
    # We'll monkey-patch train_phase1_tsjepa to capture epoch logs
    epoch_logs = []
    original_print = print  # keep original

    # Run training with custom max_epochs=20
    MAX_EPOCHS = 20
    print(f"\n{'='*60}")
    print(f"Starting TS-JEPA training: max_epochs={MAX_EPOCHS}")
    print(f"{'='*60}")

    # Capture training by running with a simple wrapper that also records losses
    # We call train_phase1_tsjepa but additionally track the first/last epoch loss
    # by hooking into stdout via sys.stdout redirect

    import io

    class TeeOutput:
        def __init__(self, stream1, stream2):
            self.stream1 = stream1
            self.stream2 = stream2
        def write(self, text):
            self.stream1.write(text)
            self.stream2.write(text)
        def flush(self):
            self.stream1.flush()
            self.stream2.flush()

    log_capture = io.StringIO()
    tee = TeeOutput(sys.stdout, log_capture)
    sys.stdout = tee

    try:
        ts_jepa = train_phase1_tsjepa(ts_jepa, train_loader, val_loader, MAX_EPOCHS, device)
    finally:
        sys.stdout = sys.__stdout__  # restore even on exception

    training_log = log_capture.getvalue()

    # ── Save checkpoint ──
    os.makedirs('results', exist_ok=True)
    torch.save(ts_jepa.state_dict(), cfg.JEPA_MODEL_PATH)
    print(f"Saved checkpoint: {cfg.JEPA_MODEL_PATH}")

    # ── Parse epoch losses from log ──
    lines = training_log.splitlines()
    epoch_lines = [l for l in lines if 'TS-JEPA Train Loss' in l]
    print(f"\nCaptured {len(epoch_lines)} epoch log lines")
    if epoch_lines:
        print(f"  First epoch: {epoch_lines[0]}")
        print(f"  Last  epoch: {epoch_lines[-1]}")

    # Extract first/last train loss for convergence check
    def extract_train_loss(line):
        # Format: "Epoch X/Y, TS-JEPA Train Loss: T, Val Loss: V"
        try:
            t_part = line.split('Train Loss:')[1].split(',')[0].strip()
            return float(t_part)
        except Exception:
            return None

    first_loss = extract_train_loss(epoch_lines[0]) if epoch_lines else None
    final_loss = extract_train_loss(epoch_lines[-1]) if epoch_lines else None
    converged  = (final_loss is not None and first_loss is not None and
                  final_loss < 0.9 * first_loss)

    print(f"\nFirst epoch train loss: {first_loss}")
    print(f"Final epoch train loss: {final_loss}")
    print(f"Converged (>10% drop):  {converged}")

    # ── UMAP + Silhouette ──
    print(f"\n{'='*60}")
    print("Encoding all samples for UMAP ...")
    print(f"{'='*60}")

    ts_jepa.eval()
    all_z = []
    all_labels_list = []
    full_loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=mafaulda_collate_fn)

    with torch.no_grad():
        for batch in full_loader:
            raw, _, labels = batch
            raw = raw.to(device)
            z = ts_jepa.get_z_macro(raw)
            all_z.append(z.cpu())
            all_labels_list.append(labels)

    all_z_np     = torch.cat(all_z, dim=0).numpy()            # (N, d_model)
    all_labels_np = torch.cat(all_labels_list, dim=0).numpy() # (N,)

    print(f"z_macro shape: {all_z_np.shape}")
    print(f"z_macro range: min={all_z_np.min():.4f}, max={all_z_np.max():.4f}")

    # Silhouette
    from sklearn.metrics import silhouette_score
    sil_score = silhouette_score(all_z_np, all_labels_np, random_state=42)
    print(f"Silhouette score: {sil_score:.4f}")

    # UMAP
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
        embedding = reducer.fit_transform(all_z_np)
        print(f"UMAP embedding shape: {embedding.shape}")

        # Plot
        fig, ax = plt.subplots(figsize=(9, 7))
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
        class_names = ['Normal (0)', 'Imbalance 20g (1)', 'Vert. Misalign. 1.27mm (2)', 'Overhang Ball 20g (3)']

        for cls_idx in range(4):
            mask = all_labels_np == cls_idx
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       c=colors[cls_idx], label=class_names[cls_idx],
                       s=25, alpha=0.85)

        ax.set_title(f"UMAP of TS-JEPA z_macro  |  Silhouette={sil_score:.3f}", fontsize=13)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.legend(fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()

        os.makedirs('.sisyphus/evidence', exist_ok=True)
        umap_path = '.sisyphus/evidence/task-6-tsjepa-umap.png'
        plt.savefig(umap_path, dpi=150)
        plt.close()
        print(f"Saved UMAP: {umap_path}")
    except ImportError as e:
        print(f"UMAP not available: {e}")
        umap_path = None

    # ── Evidence files ──
    os.makedirs('.sisyphus/evidence', exist_ok=True)

    # Training log
    training_txt_path = '.sisyphus/evidence/task-6-tsjepa-training.txt'
    with open(training_txt_path, 'w', encoding='utf-8') as f:
        f.write("=== TS-JEPA ONE-SUBTYPE-PER-CLASS TRAINING LOG ===\n\n")
        f.write(f"Device: {device}\n")
        f.write(f"Max epochs: {MAX_EPOCHS}\n")
        f.write(f"Total samples: {N}  (train={len(train_ds)}, val={len(val_ds)})\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Checkpoint: {cfg.JEPA_MODEL_PATH}\n\n")
        f.write("--- Epoch-by-epoch log ---\n")
        for line in epoch_lines:
            f.write(line + "\n")
        f.write(f"\n--- Summary ---\n")
        f.write(f"First epoch train loss: {first_loss}\n")
        f.write(f"Final epoch train loss: {final_loss}\n")
        f.write(f"Converged (>10% drop):  {converged}\n")
    print(f"Saved training log: {training_txt_path}")

    # Metrics file
    metrics_txt_path = '.sisyphus/evidence/task-6-tsjepa-metrics.txt'
    with open(metrics_txt_path, 'w', encoding='utf-8') as f:
        f.write("=== TS-JEPA ENCODING METRICS ===\n\n")
        f.write(f"z_macro shape: {list(all_z_np.shape)}\n")
        f.write(f"z_macro min:   {float(all_z_np.min()):.6f}\n")
        f.write(f"z_macro max:   {float(all_z_np.max()):.6f}\n")
        f.write(f"z_macro mean:  {float(all_z_np.mean()):.6f}\n")
        f.write(f"z_macro std:   {float(all_z_np.std()):.6f}\n")
        f.write(f"\nSilhouette score (random_state=42): {sil_score:.6f}\n")
        f.write(f"\nClass distribution:\n")
        for cls_idx in range(4):
            count = int((all_labels_np == cls_idx).sum())
            f.write(f"  Class {cls_idx}: {count} samples\n")
    print(f"Saved metrics: {metrics_txt_path}")

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"  Checkpoint:  {cfg.JEPA_MODEL_PATH}")
    print(f"  Norm meta:   results/normalization_metadata.pth")
    print(f"  Training log:{training_txt_path}")
    print(f"  Metrics:     {metrics_txt_path}")
    if umap_path:
        print(f"  UMAP:        {umap_path}")
    print(f"  Silhouette:  {sil_score:.4f}")
    print(f"  Converged:   {converged}  (first={first_loss:.4f}, final={final_loss:.4f})")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
