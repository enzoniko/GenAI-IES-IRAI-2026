"""
Task 11 — Decoder 1 (Deterministic Envelope) Training
Trains Decoder 1 to reconstruct signal envelope from z_macro using frozen TS-JEPA.
Uses one-subtype-per-class MaFaulDa 16Hz data.
"""
import os
import sys
import io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import src.configs as cfg
from src.models.ts_jepa import TSJEPA
from src.models.decoder1 import Decoder1
from src.pipelines.train_phase1 import EarlyStopping

# ──────────────────────────────────────────────────────────────────────────────
# Tee logger: write stdout to both console and string buffer
# ──────────────────────────────────────────────────────────────────────────────
class TeeOutput:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            if hasattr(s, 'encoding') and s.encoding and s.encoding.lower() not in ('utf-8', 'utf8'):
                data_safe = data.encode(s.encoding, errors='replace').decode(s.encoding)
                s.write(data_safe)
            else:
                s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

log_buffer = io.StringIO()
sys.stdout = TeeOutput(sys.__stdout__, log_buffer)


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = "data/processed-mafaulda/16hz"
EVIDENCE_DIR = ".sisyphus/evidence"
os.makedirs(EVIDENCE_DIR, exist_ok=True)
os.makedirs("results", exist_ok=True)

Y_FILES = [
    ("Y_normal_trainingset.pth",                                   0),
    ("Y_imbalance_fault_20g_trainingset.pth",                      1),
    ("Y_vertical_misalignment_fault_1.27mm_trainingset.pth",       2),
    ("Y_overhang_ball_fault_20g_trainingset.pth",                  3),
]

MAX_EPOCHS   = min(cfg.PHASE1_TRAIN_SETTINGS['max_epochs'], 50)
BATCH_SIZE   = 32
LR           = cfg.PHASE1_TRAIN_SETTINGS['learning_rate']
ES_PATIENCE  = cfg.PHASE1_TRAIN_SETTINGS['early_stop_patience']
SEQ_LENGTH   = cfg.SEQ_LENGTH  # 3014


# ──────────────────────────────────────────────────────────────────────────────
# 1. Load normalization metadata
# ──────────────────────────────────────────────────────────────────────────────
print("[1] Loading normalization metadata …")
norm_meta = torch.load(cfg.NORM_METADATA_PATH, map_location='cpu')
y_min = norm_meta['y_min'].float()   # shape [4]
y_max = norm_meta['y_max'].float()   # shape [4]
print(f"    y_min: {y_min.tolist()}")
print(f"    y_max: {y_max.tolist()}")


# ──────────────────────────────────────────────────────────────────────────────
# 2. Load Y files, normalize, transpose → (N, 4, 3014)
# ──────────────────────────────────────────────────────────────────────────────
print("[2] Loading Y files …")
all_y      = []
all_labels = []

for fname, label in Y_FILES:
    fpath = os.path.join(DATA_DIR, fname)
    y_raw = torch.load(fpath, map_location='cpu').float()   # (N, T, 4) or (N, 4, T)
    # Ensure (N, T, 4) → transpose to (N, 4, T)
    if y_raw.ndim == 3 and y_raw.shape[2] == 4:
        y_raw = y_raw.permute(0, 2, 1)   # (N, 4, T)
    assert y_raw.shape[1] == 4, f"Expected 4 channels, got {y_raw.shape}"
    assert y_raw.shape[2] == SEQ_LENGTH, f"Expected seq_len={SEQ_LENGTH}, got {y_raw.shape[2]}"
    print(f"    {fname}: {y_raw.shape}, label={label}")
    # Normalize: y_norm = (y - y_min) / (y_max - y_min + 1e-8)
    # y_min/y_max shape [4] → broadcast over (N, 4, T)
    y_min_bc = y_min.view(1, 4, 1)
    y_max_bc = y_max.view(1, 4, 1)
    y_norm = (y_raw - y_min_bc) / (y_max_bc - y_min_bc + 1e-8)
    all_y.append(y_norm)
    all_labels.append(torch.full((y_norm.shape[0],), label, dtype=torch.long))

all_y      = torch.cat(all_y, dim=0)       # (284, 4, 3014)
all_labels = torch.cat(all_labels, dim=0)  # (284,)
print(f"    Total: {all_y.shape}, labels={all_labels.shape}")
print(f"    y_norm range: [{all_y.min():.4f}, {all_y.max():.4f}]")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Interleaved 80/20 train/val split
# ──────────────────────────────────────────────────────────────────────────────
print("[3] Creating 80/20 interleaved train/val split …")
N = all_y.shape[0]
# Interleaved: every 5th sample → val (indices 4, 9, 14, …)
val_mask  = torch.zeros(N, dtype=torch.bool)
val_mask[4::5] = True
train_mask = ~val_mask

y_train      = all_y[train_mask]
labels_train = all_labels[train_mask]
y_val        = all_y[val_mask]
labels_val   = all_labels[val_mask]

print(f"    Train: {y_train.shape[0]} | Val: {y_val.shape[0]}")

# DataLoader: returns (y_norm, placeholder_omega, label) to match train_phase1_decoder1 API
train_ds = TensorDataset(y_train, torch.zeros(y_train.shape[0], 1), labels_train)
val_ds   = TensorDataset(y_val,   torch.zeros(y_val.shape[0],   1), labels_val)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Load frozen TS-JEPA
# ──────────────────────────────────────────────────────────────────────────────
print("[4] Loading frozen TS-JEPA …")
ts_jepa = TSJEPA(in_channels=4).to(DEVICE)
ts_jepa.load_state_dict(torch.load(cfg.JEPA_MODEL_PATH, map_location=DEVICE))
ts_jepa.eval()
for p in ts_jepa.parameters():
    p.requires_grad = False

n_frozen = sum(1 for p in ts_jepa.parameters() if not p.requires_grad)
n_total  = sum(1 for p in ts_jepa.parameters())
print(f"    TS-JEPA params: {n_total} total, {n_frozen} frozen (all must be frozen)")
assert n_frozen == n_total, "ERROR: Not all TS-JEPA params are frozen!"

# Smoke test: verify z_macro shape
sample_batch = next(iter(train_loader))
y_sample, _, _ = sample_batch
y_sample = y_sample.to(DEVICE)
with torch.no_grad():
    z_macro_sample = ts_jepa.get_z_macro(y_sample)
print(f"    Smoke test z_macro shape: {z_macro_sample.shape}  (expected [batch, 128])")
assert z_macro_sample.shape[1] == 128, f"z_macro dim mismatch: {z_macro_sample.shape}"


# ──────────────────────────────────────────────────────────────────────────────
# 5. Instantiate Decoder 1
# ──────────────────────────────────────────────────────────────────────────────
print("[5] Instantiating Decoder 1 …")
decoder1 = Decoder1(d_model=128, seq_length=SEQ_LENGTH, out_channels=4).to(DEVICE)

# Smoke test forward
with torch.no_grad():
    recon_smoke = decoder1(z_macro_sample)
print(f"    Smoke test decoder output shape: {recon_smoke.shape}  (expected [batch, 4, {SEQ_LENGTH}])")
assert recon_smoke.shape == (y_sample.shape[0], 4, SEQ_LENGTH), \
    f"Output shape mismatch: {recon_smoke.shape}"

n_dec_params = sum(p.numel() for p in decoder1.parameters())
print(f"    Decoder 1 params: {n_dec_params:,}")


# ──────────────────────────────────────────────────────────────────────────────
# 6. Train Decoder 1
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n[6] Training Decoder 1 — max_epochs={MAX_EPOCHS}, batch_size={BATCH_SIZE}, lr={LR}")
print(f"    early_stop_patience={ES_PATIENCE}")

optimizer    = optim.Adam(decoder1.parameters(), lr=LR)
scheduler    = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
early_stop   = EarlyStopping(patience=ES_PATIENCE)
criterion    = nn.MSELoss()

epoch_train_losses = []
epoch_val_losses   = []
epoch1_loss        = None

for epoch in range(MAX_EPOCHS):
    decoder1.train()
    train_loss = 0.0
    for batch in train_loader:
        raw, _, _ = batch
        raw = raw.to(DEVICE)
        with torch.no_grad():
            z_macro = ts_jepa.get_z_macro(raw)
        optimizer.zero_grad()
        recon = decoder1(z_macro)
        loss  = criterion(recon, raw)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    decoder1.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            raw, _, _ = batch
            raw = raw.to(DEVICE)
            z_macro = ts_jepa.get_z_macro(raw)
            recon   = decoder1(z_macro)
            val_loss += criterion(recon, raw).item()

    t_loss = train_loss / len(train_loader)
    v_loss = val_loss   / len(val_loader)
    epoch_train_losses.append(t_loss)
    epoch_val_losses.append(v_loss)

    if epoch == 0:
        epoch1_loss = t_loss

    print(f"  Epoch {epoch+1:3d}/{MAX_EPOCHS}  train={t_loss:.6f}  val={v_loss:.6f}")

    scheduler.step(v_loss)
    early_stop(v_loss)
    if early_stop.early_stop:
        print(f"  Early stopping at epoch {epoch+1}.")
        break

final_epoch_loss = epoch_train_losses[-1]
loss_drop_pct    = (epoch1_loss - final_epoch_loss) / (epoch1_loss + 1e-12) * 100
print(f"\n  Epoch 1 train loss : {epoch1_loss:.6f}")
print(f"  Final train loss   : {final_epoch_loss:.6f}")
print(f"  Loss drop          : {loss_drop_pct:.1f}%  (requirement >=30%)")
assert loss_drop_pct >= 30.0, f"Loss drop {loss_drop_pct:.1f}% < 30% requirement!"


# ──────────────────────────────────────────────────────────────────────────────
# 7. Save checkpoint
# ──────────────────────────────────────────────────────────────────────────────
print("\n[7] Saving Decoder 1 checkpoint …")
torch.save(decoder1.state_dict(), cfg.DEC1_MODEL_PATH)
print(f"    Saved to {cfg.DEC1_MODEL_PATH}")


# ──────────────────────────────────────────────────────────────────────────────
# 8. Evaluation — RMSE per class + reconstruction plot
# ──────────────────────────────────────────────────────────────────────────────
print("\n[8] Evaluating — RMSE per class …")
CLASS_NAMES = ["Normal", "Imbalance_20g", "Vert_Misalign_1.27mm", "Overhang_Ball_20g"]

decoder1.eval()
ts_jepa.eval()

# Collect one sample per class from val set, compute RMSE across all val samples per class
class_sq_errors = {c: [] for c in range(4)}
class_samples   = {c: None for c in range(4)}  # (y_norm, y_recon) for plotting

with torch.no_grad():
    for y_batch, _, lbl_batch in val_loader:
        y_batch   = y_batch.to(DEVICE)
        lbl_batch = lbl_batch.to(DEVICE)
        z_macro   = ts_jepa.get_z_macro(y_batch)
        recon     = decoder1(z_macro)
        sq_err    = (recon - y_batch) ** 2  # (B, 4, T)
        mse_per_sample = sq_err.mean(dim=(1, 2))  # (B,)
        for i in range(y_batch.shape[0]):
            c = lbl_batch[i].item()
            class_sq_errors[c].append(mse_per_sample[i].item())
            if class_samples[c] is None:
                class_samples[c] = (
                    y_batch[i].cpu(),
                    recon[i].cpu()
                )

rmse_per_class = {}
for c in range(4):
    if class_sq_errors[c]:
        rmse_per_class[c] = float(np.sqrt(np.mean(class_sq_errors[c])))
    else:
        rmse_per_class[c] = float('nan')
    print(f"    Class {c} ({CLASS_NAMES[c]}): RMSE = {rmse_per_class[c]:.6f}")


# ──────────────────────────────────────────────────────────────────────────────
# 9. Reconstruction plot — overlay decoder vs original, 4 channels × 4 classes
# ──────────────────────────────────────────────────────────────────────────────
print("\n[9] Generating reconstruction plot …")
CHANNEL_NAMES = ["CH0", "CH1", "CH2", "CH3"]
t = np.arange(SEQ_LENGTH)

fig, axes = plt.subplots(4, 4, figsize=(20, 14))
fig.suptitle("Decoder 1 Reconstruction vs Original (normalized)\nOne sample per class, all 4 channels", fontsize=13)

for col, c in enumerate(range(4)):
    y_orig, y_recon = class_samples[c]
    for row, ch in enumerate(range(4)):
        ax = axes[row, col]
        ax.plot(t, y_orig[ch].numpy(),  color='steelblue',  alpha=0.7, linewidth=0.8, label='Original')
        ax.plot(t, y_recon[ch].numpy(), color='darkorange', alpha=0.9, linewidth=0.8, linestyle='--', label='Decoder 1')
        if row == 0:
            ax.set_title(f"{CLASS_NAMES[c]}\nRMSE={rmse_per_class[c]:.4f}", fontsize=8)
        if col == 0:
            ax.set_ylabel(CHANNEL_NAMES[ch], fontsize=8)
        ax.tick_params(labelsize=6)
        if row == 0 and col == 0:
            ax.legend(fontsize=6, loc='upper right')

plt.tight_layout()
recon_plot_path = os.path.join(EVIDENCE_DIR, "task-11-reconstruction.png")
plt.savefig(recon_plot_path, dpi=150)
plt.close()
print(f"    Saved to {recon_plot_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 10. Training log
# ──────────────────────────────────────────────────────────────────────────────
print("\n[10] Writing training log …")
log_path = os.path.join(EVIDENCE_DIR, "task-11-decoder1-training.txt")

log_lines = []
log_lines.append("=" * 70)
log_lines.append("Task 11 — Decoder 1 Training Log")
log_lines.append("=" * 70)
log_lines.append(f"Device          : {DEVICE}")
log_lines.append(f"Max epochs      : {MAX_EPOCHS}")
log_lines.append(f"Batch size      : {BATCH_SIZE}")
log_lines.append(f"Learning rate   : {LR}")
log_lines.append(f"ES patience     : {ES_PATIENCE}")
log_lines.append(f"Total windows   : {N}  (train={y_train.shape[0]}, val={y_val.shape[0]})")
log_lines.append(f"Decoder1 params : {n_dec_params:,}")
log_lines.append("")
log_lines.append("--- Loss Curve ---")
for i, (tl, vl) in enumerate(zip(epoch_train_losses, epoch_val_losses)):
    log_lines.append(f"  Epoch {i+1:3d}: train={tl:.6f}  val={vl:.6f}")
log_lines.append("")
log_lines.append(f"Epoch 1 train loss : {epoch1_loss:.6f}")
log_lines.append(f"Final train loss   : {final_epoch_loss:.6f}")
log_lines.append(f"Loss drop          : {loss_drop_pct:.1f}%")
log_lines.append("")
log_lines.append("--- RMSE per Class (normalized signal domain) ---")
for c in range(4):
    log_lines.append(f"  Class {c} ({CLASS_NAMES[c]}): {rmse_per_class[c]:.6f}")
log_lines.append("")
log_lines.append(f"Checkpoint saved to: {cfg.DEC1_MODEL_PATH}")
log_lines.append(f"Reconstruction plot: {recon_plot_path}")

log_text = "\n".join(log_lines)
with open(log_path, "w") as f:
    f.write(log_text)
print(f"    Saved to {log_path}")
print("\n" + log_text)

print("\n[DONE] Task 11 complete.")
