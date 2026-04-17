"""
Task 15 — Baseline Diffusion Experiments
Train VanillaDDPM and LabelConditionedDDPM on z_macro latent space.

Same training setup as T12 (LDM):
  - epochs=50, lr=1e-3, Adam, batch_size=32
  - DDIM-style x0-prediction sampling with clamp [-3.5, 3.5], every-5th-step

Outputs:
  - results/baseline_vanilla.pth
  - results/baseline_label.pth
  - .sisyphus/evidence/task-15-baseline-training.txt
"""

import sys
import io
import os
import time

import torch
import torch.optim as optim
import numpy as np

import src.configs as cfg
from src.models.ts_jepa import TSJEPA
from src.models.baselines import VanillaDDPM, LabelConditionedDDPM, _move_scheduler_to

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_16HZ = "data/processed-mafaulda/16hz"
Y_FILES = [
    (os.path.join(DATA_16HZ, "Y_normal_trainingset.pth"),                             0),
    (os.path.join(DATA_16HZ, "Y_imbalance_fault_20g_trainingset.pth"),                1),
    (os.path.join(DATA_16HZ, "Y_vertical_misalignment_fault_1.27mm_trainingset.pth"), 2),
    (os.path.join(DATA_16HZ, "Y_overhang_ball_fault_20g_trainingset.pth"),            3),
]
NORM_PATH      = "results/normalization_metadata.pth"
JEPA_PATH      = "results/ts_jepa.pth"
VANILLA_PATH   = "results/baseline_vanilla.pth"
LABEL_PATH     = "results/baseline_label.pth"
EVIDENCE_DIR   = ".sisyphus/evidence"
LOG_PATH       = os.path.join(EVIDENCE_DIR, "task-15-baseline-training.txt")

os.makedirs(EVIDENCE_DIR, exist_ok=True)
os.makedirs("results", exist_ok=True)

# ─── Tee output to file ───────────────────────────────────────────────────────
class TeeOutput:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams: s.flush()

log_buf = io.StringIO()
orig_stdout = sys.stdout
sys.stdout = TeeOutput(orig_stdout, log_buf)

def log(msg=""):
    print(msg)

# ─── Hyperparams (EXACT match to T12 for fairness) ────────────────────────────
N_EPOCHS   = 50
LR         = 1e-3
BATCH_SIZE = 32
X0_CLAMP   = 3.5
SKIP_STEPS = 5  # use every 5th timestep → 200 steps total

# ─── Device ───────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log(f"Device: {device}")

# ─── Step 1: Load normalization metadata ──────────────────────────────────────
log("\n=== Loading normalization metadata ===")
norm_meta = torch.load(NORM_PATH, map_location='cpu', weights_only=True)
y_min = norm_meta['y_min'].float()  # [4]
y_max = norm_meta['y_max'].float()  # [4]
log(f"y_min: {y_min.tolist()}")
log(f"y_max: {y_max.tolist()}")

# ─── Step 2: Load frozen TS-JEPA ──────────────────────────────────────────────
log("\n=== Loading frozen TS-JEPA ===")
ts_jepa = TSJEPA(in_channels=4).to(device)
state = torch.load(JEPA_PATH, map_location=device, weights_only=True)
ts_jepa.load_state_dict(state)
for param in ts_jepa.parameters():
    param.requires_grad = False
ts_jepa.eval()
total_params = sum(p.numel() for p in ts_jepa.parameters())
log(f"TS-JEPA loaded: {total_params:,} params, all frozen")

# ─── Step 3: Encode all Y-files offline to z_macro ───────────────────────────
log("\n=== Encoding training data to z_macro (offline) ===")

all_z_macro = []
all_labels  = []

for y_path, label in Y_FILES:
    Y_raw = torch.load(y_path, map_location='cpu', weights_only=True)  # (N, T, 4)
    log(f"  {os.path.basename(y_path)}: shape={tuple(Y_raw.shape)}, label={label}")

    # Normalize: (Y - y_min) / (y_max - y_min + 1e-8), broadcast on last dim
    y_norm = (Y_raw.float() - y_min) / (y_max - y_min + 1e-8)  # (N, T, 4)

    # Transpose: (N, T, 4) → (N, 4, T) for Conv1d in TSJEPA
    y_t = y_norm.permute(0, 2, 1).float().to(device)  # (N, 4, T)

    with torch.no_grad():
        z = ts_jepa.get_z_macro(y_t)  # (N, 128)

    log(f"    z_macro: shape={tuple(z.shape)}, min={z.min().item():.4f}, max={z.max().item():.4f}, finite={z.isfinite().all().item()}")
    all_z_macro.append(z.cpu())
    all_labels.append(torch.full((len(z),), label, dtype=torch.long))

z_macro_all = torch.cat(all_z_macro, dim=0)   # (284, 128)
labels_all  = torch.cat(all_labels,  dim=0)   # (284,)
log(f"\nTotal z_macro dataset: {z_macro_all.shape}, labels: {labels_all.shape}")
log(f"z_macro global: mean={z_macro_all.mean().item():.4f}, std={z_macro_all.std().item():.4f}")
log(f"z_macro range:  min={z_macro_all.min().item():.4f},  max={z_macro_all.max().item():.4f}")

# ─── DataLoaders ──────────────────────────────────────────────────────────────
ds_vanilla = torch.utils.data.TensorDataset(z_macro_all)
loader_vanilla = torch.utils.data.DataLoader(ds_vanilla, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

ds_label = torch.utils.data.TensorDataset(z_macro_all, labels_all)
loader_label = torch.utils.data.DataLoader(ds_label, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

# ─── DDIM-style sampling (x0-prediction, subsampled timesteps) ────────────────
def sample_ddim(model, scheduler, n_samples, z_dim=128, device_str='cpu',
                x0_clamp=3.5, skip_steps=5, class_label=None):
    """
    DDIM-style reverse diffusion with x0-prediction and clamping.
    Works for both VanillaDDPM and LabelConditionedDDPM.
    """
    dev = torch.device(device_str)
    _move_scheduler_to(scheduler, dev)
    model.eval()

    T = scheduler.num_train_timesteps
    # Every skip_steps-th timestep: [0, 5, 10, ..., 995]
    timesteps_list = list(range(0, T, skip_steps))

    with torch.no_grad():
        z = torch.randn(n_samples, z_dim, device=dev)

        for idx in reversed(range(len(timesteps_list))):
            t_val = timesteps_list[idx]
            t_tensor = torch.full((n_samples,), t_val, dtype=torch.long, device=dev)

            # Forward pass (with or without class label)
            if class_label is not None:
                if isinstance(class_label, int):
                    lbl = torch.full((n_samples,), class_label, dtype=torch.long, device=dev)
                else:
                    lbl = class_label.to(dev)
                noise_pred = model(z, t_tensor, lbl)
            else:
                noise_pred = model(z, t_tensor)

            # Predict x0 from current noisy sample
            sqrt_alpha_bar_t  = scheduler.sqrt_alphas_cumprod[t_val]
            sqrt_1m_alpha_bar = scheduler.sqrt_one_minus_alphas_cumprod[t_val]
            x0_pred = (z - sqrt_1m_alpha_bar * noise_pred) / (sqrt_alpha_bar_t + 1e-8)
            x0_pred = x0_pred.clamp(-x0_clamp, x0_clamp)

            if idx > 0:
                t_prev = timesteps_list[idx - 1]
                alpha_bar_prev = scheduler.alphas_cumprod[t_prev]
                # DDIM update: z_{t_prev} = sqrt(abar_prev)*x0 + sqrt(1-abar_prev)*eps
                direction = torch.sqrt(1.0 - alpha_bar_prev) * noise_pred
                z = torch.sqrt(alpha_bar_prev) * x0_pred + direction
            else:
                z = x0_pred  # final step: return x0 estimate

    return z

# ─── Step 4: Train VanillaDDPM ────────────────────────────────────────────────
log(f"\n=== VanillaDDPM Training: {N_EPOCHS} epochs, lr={LR}, batch_size={BATCH_SIZE} ===")

vanilla = VanillaDDPM(z_dim=128, time_dim=64).to(device)
vanilla_params = sum(p.numel() for p in vanilla.parameters())
log(f"Parameters: {vanilla_params:,}")
assert vanilla_params == 181568, f"FAIL: Expected 181,568 params, got {vanilla_params:,}"
log(f"[OK] Parameter count matches expected 181,568")

opt_vanilla = optim.Adam(vanilla.parameters(), lr=LR)

vanilla_losses = []
t0 = time.time()

for epoch in range(N_EPOCHS):
    vanilla.train()
    ep_loss = 0.0
    for (zb,) in loader_vanilla:
        zb = zb.to(device)
        opt_vanilla.zero_grad()
        loss = vanilla.get_loss(zb)
        loss.backward()
        opt_vanilla.step()
        ep_loss += loss.item()
    avg = ep_loss / len(loader_vanilla)
    vanilla_losses.append(avg)

    if (epoch + 1) % 5 == 0 or epoch == 0:
        elapsed = time.time() - t0
        log(f"  Epoch {epoch+1:3d}/{N_EPOCHS} | loss={avg:.6f} | elapsed={elapsed:.1f}s")

vanilla_drop_pct = (vanilla_losses[0] - vanilla_losses[-1]) / (vanilla_losses[0] + 1e-12) * 100
log(f"\nVanillaDDPM Training complete.")
log(f"  Epoch 1:  loss={vanilla_losses[0]:.6f}")
log(f"  Epoch 25: loss={vanilla_losses[24]:.6f}")
log(f"  Epoch 50: loss={vanilla_losses[49]:.6f}")
log(f"  Loss drop: {vanilla_drop_pct:.1f}%")
assert vanilla_drop_pct >= 30, f"FAIL: VanillaDDPM loss dropped only {vanilla_drop_pct:.1f}% (need >=30%)"
log(f"[OK] Loss drop criterion met: {vanilla_drop_pct:.1f}% >= 30%")

# Save checkpoint
torch.save(vanilla.state_dict(), VANILLA_PATH)
vanilla_kb = os.path.getsize(VANILLA_PATH) // 1024
log(f"Checkpoint saved: {VANILLA_PATH} ({vanilla_kb} KB)")

# Sample validation — use DDIM-style x0-prediction (NOT naive 1000-step)
log(f"\n=== VanillaDDPM: Sampling 10 unconditional samples ===")
vanilla.eval()
_move_scheduler_to(vanilla.scheduler, device)
z_vanilla = sample_ddim(vanilla, vanilla.scheduler, n_samples=10,
                        z_dim=128, device_str=str(device),
                        x0_clamp=X0_CLAMP, skip_steps=SKIP_STEPS)
vanilla_finite = z_vanilla.isfinite().all().item()
vanilla_range_min = z_vanilla.min().item()
vanilla_range_max = z_vanilla.max().item()
log(f"  Generated: shape={tuple(z_vanilla.shape)}")
log(f"  Range: [{vanilla_range_min:.4f}, {vanilla_range_max:.4f}]")
log(f"  All finite: {vanilla_finite}")
assert vanilla_finite, "FAIL: VanillaDDPM samples contain non-finite values"
log(f"[OK] VanillaDDPM sampling validated")

# ─── Step 5: Train LabelConditionedDDPM ──────────────────────────────────────
log(f"\n=== LabelConditionedDDPM Training: {N_EPOCHS} epochs, lr={LR}, batch_size={BATCH_SIZE} ===")

label_cond = LabelConditionedDDPM(z_dim=128, time_dim=64, num_classes=4).to(device)
label_params = sum(p.numel() for p in label_cond.parameters())
log(f"Parameters: {label_params:,}")
assert label_params == 181824, f"FAIL: Expected 181,824 params, got {label_params:,}"
log(f"[OK] Parameter count matches expected 181,824")

opt_label = optim.Adam(label_cond.parameters(), lr=LR)

label_losses = []
t0 = time.time()

for epoch in range(N_EPOCHS):
    label_cond.train()
    ep_loss = 0.0
    for zb, lb in loader_label:
        zb = zb.to(device)
        lb = lb.to(device)
        opt_label.zero_grad()
        loss = label_cond.get_loss(zb, lb)
        loss.backward()
        opt_label.step()
        ep_loss += loss.item()
    avg = ep_loss / len(loader_label)
    label_losses.append(avg)

    if (epoch + 1) % 5 == 0 or epoch == 0:
        elapsed = time.time() - t0
        log(f"  Epoch {epoch+1:3d}/{N_EPOCHS} | loss={avg:.6f} | elapsed={elapsed:.1f}s")

label_drop_pct = (label_losses[0] - label_losses[-1]) / (label_losses[0] + 1e-12) * 100
log(f"\nLabelConditionedDDPM Training complete.")
log(f"  Epoch 1:  loss={label_losses[0]:.6f}")
log(f"  Epoch 25: loss={label_losses[24]:.6f}")
log(f"  Epoch 50: loss={label_losses[49]:.6f}")
log(f"  Loss drop: {label_drop_pct:.1f}%")
assert label_drop_pct >= 30, f"FAIL: LabelConditionedDDPM loss dropped only {label_drop_pct:.1f}% (need >=30%)"
log(f"[OK] Loss drop criterion met: {label_drop_pct:.1f}% >= 30%")

# Save checkpoint
torch.save(label_cond.state_dict(), LABEL_PATH)
label_kb = os.path.getsize(LABEL_PATH) // 1024
log(f"Checkpoint saved: {LABEL_PATH} ({label_kb} KB)")

# Sample validation — 10 samples per class (40 total), DDIM-style
log(f"\n=== LabelConditionedDDPM: Sampling 10 per class (40 total) ===")
label_cond.eval()
_move_scheduler_to(label_cond.scheduler, device)

all_gen = []
for cls_id in range(4):
    z_cls = sample_ddim(label_cond, label_cond.scheduler, n_samples=10,
                        z_dim=128, device_str=str(device),
                        x0_clamp=X0_CLAMP, skip_steps=SKIP_STEPS, class_label=cls_id)
    all_gen.append(z_cls)
    log(f"  Class {cls_id}: range=[{z_cls.min().item():.4f}, {z_cls.max().item():.4f}], finite={z_cls.isfinite().all().item()}")

z_label_all = torch.cat(all_gen, dim=0)  # (40, 128)
label_finite = z_label_all.isfinite().all().item()
label_range_min = z_label_all.min().item()
label_range_max = z_label_all.max().item()
log(f"  Total: shape={tuple(z_label_all.shape)}")
log(f"  Range: [{label_range_min:.4f}, {label_range_max:.4f}]")
log(f"  All finite: {label_finite}")
assert label_finite, "FAIL: LabelConditionedDDPM samples contain non-finite values"
log(f"[OK] LabelConditionedDDPM sampling validated")

# ─── Step 6: Write structured evidence file ───────────────────────────────────
log("\n=== Writing evidence file ===")

evidence_lines = [
    "=== T15: Baseline Diffusion Experiments ===",
    f"Data: 284 z_macro vectors (encoded offline), z_dim=128, classes=4",
    f"Training config: epochs={N_EPOCHS}, lr={LR}, Adam, batch_size={BATCH_SIZE}",
    "",
    "=== VanillaDDPM ===",
    f"Parameters: {vanilla_params:,}",
    f"Epoch 1: loss={vanilla_losses[0]:.4f}",
    f"Epoch 25: loss={vanilla_losses[24]:.4f}",
    f"Epoch 50: loss={vanilla_losses[49]:.4f}",
    f"Loss drop: {vanilla_drop_pct:.1f}%",
    f"Checkpoint: {VANILLA_PATH} ({vanilla_kb} KB)",
    f"Sampling: 10 unconditional samples, range=[{vanilla_range_min:.2f}, {vanilla_range_max:.2f}], finite={vanilla_finite}",
    "",
    "=== LabelConditionedDDPM ===",
    f"Parameters: {label_params:,}",
    f"Epoch 1: loss={label_losses[0]:.4f}",
    f"Epoch 25: loss={label_losses[24]:.4f}",
    f"Epoch 50: loss={label_losses[49]:.4f}",
    f"Loss drop: {label_drop_pct:.1f}%",
    f"Checkpoint: {LABEL_PATH} ({label_kb} KB)",
    f"Sampling: 40 samples (10 per class), range=[{label_range_min:.2f}, {label_range_max:.2f}], finite={label_finite}",
    "",
    "=== Full VanillaDDPM loss curve ===",
]
for i, l in enumerate(vanilla_losses):
    evidence_lines.append(f"  Epoch {i+1:3d}: {l:.6f}")

evidence_lines += [
    "",
    "=== Full LabelConditionedDDPM loss curve ===",
]
for i, l in enumerate(label_losses):
    evidence_lines.append(f"  Epoch {i+1:3d}: {l:.6f}")

evidence_lines += [
    "",
    "=== Validation ===",
    f"[OK] VanillaDDPM loss drop {vanilla_drop_pct:.1f}% >= 30%",
    f"[OK] LabelConditionedDDPM loss drop {label_drop_pct:.1f}% >= 30%",
    f"[OK] VanillaDDPM samples: finite={vanilla_finite}, range=[{vanilla_range_min:.2f}, {vanilla_range_max:.2f}]",
    f"[OK] LabelConditionedDDPM samples: finite={label_finite}, range=[{label_range_min:.2f}, {label_range_max:.2f}]",
    "ALL CHECKS PASSED",
]

evidence_text = "\n".join(evidence_lines)

# Restore stdout before writing
sys.stdout = orig_stdout
full_log = log_buf.getvalue()

# Write evidence file (structured format per spec)
with open(LOG_PATH, 'w', encoding='utf-8') as f:
    f.write(evidence_text)
    f.write("\n\n=== Full Training Log ===\n")
    f.write(full_log)

print(f"Evidence saved: {LOG_PATH}")
print(f"VanillaDDPM: {vanilla_params:,} params, loss drop={vanilla_drop_pct:.1f}%")
print(f"LabelConditioned: {label_params:,} params, loss drop={label_drop_pct:.1f}%")
print("DONE — Task 15 complete.")
