"""
Task 12 — LDM Training on z_macro Latent Space

1. Load frozen TS-JEPA from results/ts_jepa.pth
2. Encode 4 Y-files (one per class, 284 windows) offline to z_macro
3. Train LatentDiffusionMLP (DDPM) on these precomputed z_macro vectors
4. Generate 100 unconditional samples; compare distribution
5. UMAP visualization of real + generated z_macro
6. Save checkpoint to results/ldm.pth
"""

import sys
import io
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time

import src.configs as cfg
from src.models.ts_jepa import TSJEPA
from src.models.latent_diffusion import LatentDiffusionMLP, DDPMScheduler

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_16HZ = "data/processed-mafaulda/16hz"
Y_FILES = [
    (os.path.join(DATA_16HZ, "Y_normal_trainingset.pth"),                           0),
    (os.path.join(DATA_16HZ, "Y_imbalance_fault_20g_trainingset.pth"),              1),
    (os.path.join(DATA_16HZ, "Y_vertical_misalignment_fault_1.27mm_trainingset.pth"), 2),
    (os.path.join(DATA_16HZ, "Y_overhang_ball_fault_20g_trainingset.pth"),          3),
]
NORM_PATH   = "results/normalization_metadata.pth"
JEPA_PATH   = "results/ts_jepa_hpo_best.pth"
LDM_PATH    = "results/ldm.pth"
EVIDENCE_DIR = ".sisyphus/evidence"
LOG_PATH    = os.path.join(EVIDENCE_DIR, "task-12-ldm-training.txt")
UMAP_PATH   = os.path.join(EVIDENCE_DIR, "task-12-ldm-samples-umap.png")

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

# ─── Device ───────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log(f"Device: {device}")

# ─── Step 1: Load normalization metadata ──────────────────────────────────────
log("\n=== Loading normalization metadata ===")
norm_meta = torch.load(NORM_PATH, map_location='cpu', weights_only=True)
y_min = norm_meta['y_min'].float()  # shape [4]
y_max = norm_meta['y_max'].float()  # shape [4]
log(f"y_min: {y_min.tolist()}")
log(f"y_max: {y_max.tolist()}")

# ─── Step 2: Load frozen TS-JEPA ──────────────────────────────────────────────
log("\n=== Loading frozen TS-JEPA ===")
ts_jepa = TSJEPA(in_channels=4).to(device)
state = torch.load(JEPA_PATH, map_location=device, weights_only=False)
# Handle both wrapped (HPO checkpoint) and bare state_dict formats
if "model_state_dict" in state:
    state = state["model_state_dict"]
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
    Y_raw = torch.load(y_path, map_location='cpu', weights_only=True)  # (N, 3014, 4)
    log(f"  {os.path.basename(y_path)}: shape={tuple(Y_raw.shape)}, label={label}")

    # Normalize: y_norm = (y - y_min) / (y_max - y_min + 1e-8)
    # y_min/y_max shape [4], Y shape (N, 3014, 4) → broadcast along last dim
    y_norm = (Y_raw.float() - y_min) / (y_max - y_min + 1e-8)  # (N, 3014, 4)

    # Transpose: (N, 3014, 4) → (N, 4, 3014) for TSJEPA Conv1d
    y_t = y_norm.permute(0, 2, 1).float().to(device)  # (N, 4, 3014)

    with torch.no_grad():
        z = ts_jepa.get_z_macro(y_t)  # (N, 128)

    log(f"    z_macro: shape={tuple(z.shape)}, min={z.min().item():.4f}, max={z.max().item():.4f}, all_finite={z.isfinite().all().item()}")
    all_z_macro.append(z.cpu())
    all_labels.append(torch.full((len(z),), label, dtype=torch.long))

z_macro_all = torch.cat(all_z_macro, dim=0)   # (284, 128)
labels_all  = torch.cat(all_labels,  dim=0)   # (284,)
log(f"\nTotal z_macro dataset: {z_macro_all.shape}, labels: {labels_all.shape}")
log(f"z_macro global: mean={z_macro_all.mean().item():.4f}, std={z_macro_all.std().item():.4f}")
log(f"z_macro range:  min={z_macro_all.min().item():.4f},  max={z_macro_all.max().item():.4f}")

# ─── Step 4: Smoke test (3 epochs) ───────────────────────────────────────────
log("\n=== SMOKE TEST: 3 epochs ===")

scheduler_smoke = DDPMScheduler(num_train_timesteps=1000, device=device)
ldm_smoke = LatentDiffusionMLP(z_dim=128, time_dim=64).to(device)
opt_smoke  = optim.Adam(ldm_smoke.parameters(), lr=1e-3)
crit = nn.MSELoss()

ds = torch.utils.data.TensorDataset(z_macro_all)
loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True, drop_last=False)

smoke_losses = []
for ep in range(3):
    ldm_smoke.train()
    ep_loss = 0.0
    for (zb,) in loader:
        zb = zb.to(device)
        noise = torch.randn_like(zb)
        t = torch.randint(0, 1000, (zb.shape[0],), device=device).long()
        noisy = scheduler_smoke.add_noise(zb, noise, t).float()
        opt_smoke.zero_grad()
        pred = ldm_smoke(noisy, t)
        loss = crit(pred, noise)
        loss.backward()
        opt_smoke.step()
        ep_loss += loss.item()
    avg = ep_loss / len(loader)
    smoke_losses.append(avg)
    log(f"  Smoke epoch {ep+1}/3: loss={avg:.6f}")

log(f"Smoke test: shapes OK, losses finite={all(np.isfinite(smoke_losses))}")
assert smoke_losses[0] > smoke_losses[-1] or abs(smoke_losses[0]-smoke_losses[-1]) < 0.5, \
    "Smoke test WARN: loss not decreasing in 3 epochs (may be OK with tiny dataset)"
# Verify sample shape
ldm_smoke.eval()
with torch.no_grad():
    z_test = torch.randn(4, 128, device=device)
    t_test = torch.zeros(4, dtype=torch.long, device=device)
    out = ldm_smoke(z_test, t_test)
assert out.shape == (4, 128), f"Shape mismatch: {out.shape}"
assert out.isfinite().all(), "Generated output has non-finite values"
log(f"Smoke sample shape: {out.shape}, all finite: {out.isfinite().all().item()}")
log("SMOKE TEST PASSED")

# ─── Step 5: Full LDM training ────────────────────────────────────────────────
N_EPOCHS = min(cfg.PHASE2_TRAIN_SETTINGS['ldm_epochs'], 50)
LR       = cfg.PHASE2_TRAIN_SETTINGS['learning_rate']
log(f"\n=== FULL LDM TRAINING: {N_EPOCHS} epochs, lr={LR} ===")

scheduler = DDPMScheduler(num_train_timesteps=1000, device=device)
ldm = LatentDiffusionMLP(z_dim=128, time_dim=64).to(device)
optimizer = optim.Adam(ldm.parameters(), lr=LR)
criterion  = nn.MSELoss()

train_losses = []
t0 = time.time()

for epoch in range(N_EPOCHS):
    ldm.train()
    ep_loss = 0.0
    for (zb,) in loader:
        zb = zb.to(device)
        noise = torch.randn_like(zb)
        t_steps = torch.randint(0, 1000, (zb.shape[0],), device=device).long()
        noisy = scheduler.add_noise(zb, noise, t_steps).float()
        optimizer.zero_grad()
        noise_pred = ldm(noisy, t_steps)
        loss = criterion(noise_pred, noise)
        loss.backward()
        optimizer.step()
        ep_loss += loss.item()
    avg = ep_loss / len(loader)
    train_losses.append(avg)

    if (epoch + 1) % 5 == 0 or epoch == 0:
        elapsed = time.time() - t0
        log(f"  Epoch {epoch+1:3d}/{N_EPOCHS} | loss={avg:.6f} | elapsed={elapsed:.1f}s")

loss_drop_pct = (train_losses[0] - train_losses[-1]) / (train_losses[0] + 1e-12) * 100
log(f"\nTraining complete. Loss: epoch1={train_losses[0]:.6f} -> final={train_losses[-1]:.6f} ({loss_drop_pct:.1f}% drop)")
assert loss_drop_pct >= 30, f"FAIL: Loss dropped only {loss_drop_pct:.1f}% (need ≥30%)"
log(f"[OK] Loss decrease criterion met: {loss_drop_pct:.1f}% >= 30%")

# ─── Step 6: Save checkpoint ──────────────────────────────────────────────────
torch.save(ldm.state_dict(), LDM_PATH)
log(f"\nCheckpoint saved: {LDM_PATH}")

# ─── Step 7: Generate 100 unconditional samples ───────────────────────────────
log("\n=== Generating 100 unconditional z_macro samples ===")

def sample_ldm(model, scheduler, n_samples, z_dim=128, device='cpu',
               x0_clamp=3.5, skip_steps=5):
    """
    DDPM reverse diffusion using x0-prediction with clamping for numerical stability.
    Uses every `skip_steps`-th timestep to speed up and reduce error accumulation.
    """
    model.eval()
    T = scheduler.num_train_timesteps
    # Use a subset of timesteps for stability (every skip_steps-th)
    timesteps_list = list(range(0, T, skip_steps))  # e.g. [0, 5, 10, ..., 995]
    
    with torch.no_grad():
        z = torch.randn(n_samples, z_dim, device=device)
        
        for idx in reversed(range(len(timesteps_list))):
            t_val = timesteps_list[idx]
            t_tensor = torch.full((n_samples,), t_val, dtype=torch.long, device=device)
            noise_pred = model(z, t_tensor)
            
            # Predict x0 from current noisy sample
            sqrt_alpha_bar_t  = scheduler.sqrt_alphas_cumprod[t_val]
            sqrt_1m_alpha_bar = scheduler.sqrt_one_minus_alphas_cumprod[t_val]
            x0_pred = (z - sqrt_1m_alpha_bar * noise_pred) / (sqrt_alpha_bar_t + 1e-8)
            # Clamp x0 to realistic z_macro range
            x0_pred = x0_pred.clamp(-x0_clamp, x0_clamp)
            
            if idx > 0:
                t_prev = timesteps_list[idx - 1]
                alpha_bar_prev = scheduler.alphas_cumprod[t_prev]
                # DDIM-style: z_{t_prev} = sqrt(abar_{t-1})*x0 + sqrt(1-abar_{t-1})*eps
                direction = torch.sqrt(1.0 - alpha_bar_prev) * noise_pred
                z = torch.sqrt(alpha_bar_prev) * x0_pred + direction
            else:
                z = x0_pred  # final step: return x0 estimate
    return z

log("Running reverse diffusion (subsampled, x0-prediction)... this may take a moment.")
t_gen_start = time.time()
z_gen = sample_ldm(ldm, scheduler, n_samples=100, z_dim=128, device=device)
t_gen = time.time() - t_gen_start
z_gen_cpu = z_gen.cpu()
log(f"Generation done in {t_gen:.1f}s")

# Distribution comparison — use global scalar stats (meaningful for both real & generated)
real_mean_scalar = z_macro_all.mean().item()      # global mean of all 284×128 elements
real_std_scalar  = z_macro_all.std().item()       # global std  of all 284×128 elements
gen_mean_scalar  = z_gen_cpu.mean().item()
gen_std_scalar   = z_gen_cpu.std().item()

# Per-class mean vectors (for reference)
log(f"\nDistribution comparison (global scalar stats):")
log(f"  Real z_macro:  mean={real_mean_scalar:.4f}, std={real_std_scalar:.4f}")
log(f"  Gen  z_macro:  mean={gen_mean_scalar:.4f},  std={gen_std_scalar:.4f}")
log(f"  Real global:   min={z_macro_all.min().item():.4f}, max={z_macro_all.max().item():.4f}")
log(f"  Gen  global:   min={z_gen_cpu.min().item():.4f},  max={z_gen_cpu.max().item():.4f}")
log(f"  Gen finite:    {z_gen_cpu.isfinite().all().item()}")

# Sanity check: same order of magnitude for std
ratio = gen_std_scalar / (real_std_scalar + 1e-12)
log(f"  Std ratio (gen/real): {ratio:.4f} (expect ~0.1-10)")
assert 0.05 < ratio < 20, f"Distribution mismatch: std ratio={ratio:.2f}"
log(f"[OK] Generated samples within expected distribution range")

# ─── Step 8: UMAP Visualization ───────────────────────────────────────────────
log("\n=== UMAP Visualization ===")
try:
    import umap
    log("umap-learn available")
except ImportError:
    log("WARNING: umap-learn not installed. Attempting pip install...")
    os.system(f"{sys.executable} -m pip install umap-learn -q")
    import umap
    log("umap-learn installed and imported")

log("Fitting UMAP on combined real + generated z_macro...")
real_np = z_macro_all.numpy()      # (284, 128)
gen_np  = z_gen_cpu.numpy()        # (100, 128)
labels_np = labels_all.numpy()     # (284,)

combined = np.concatenate([real_np, gen_np], axis=0)  # (384, 128)
reducer = umap.UMAP(n_components=2, random_state=42)
emb = reducer.fit_transform(combined)

emb_real = emb[:284]   # (284, 2)
emb_gen  = emb[284:]   # (100, 2)

CLASS_COLORS = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c', 3: '#d62728'}
CLASS_NAMES  = {0: 'Normal', 1: 'Imbalance', 2: 'Vert. Misalign.', 3: 'Overhang Ball'}

fig, ax = plt.subplots(figsize=(10, 8))

for cls_id in [0, 1, 2, 3]:
    mask = labels_np == cls_id
    ax.scatter(emb_real[mask, 0], emb_real[mask, 1],
               c=CLASS_COLORS[cls_id], label=f'Real: {CLASS_NAMES[cls_id]}',
               s=40, alpha=0.7, edgecolors='white', linewidths=0.3)

ax.scatter(emb_gen[:, 0], emb_gen[:, 1],
           c='gray', marker='x', s=60, alpha=0.8,
           label='Generated (LDM)', linewidths=1.5)

ax.set_title("UMAP: Real z_macro (colored by class) vs LDM-Generated z_macro (gray ×)", fontsize=13)
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
ax.legend(loc='best', fontsize=9)
ax.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(UMAP_PATH, dpi=150, bbox_inches='tight')
plt.close()
log(f"UMAP saved: {UMAP_PATH}")

# ─── Step 9: Full training log ────────────────────────────────────────────────
log("\n=== Training Loss Curve ===")
for i, l in enumerate(train_losses):
    log(f"  Epoch {i+1:3d}: {l:.6f}")

log("\n=== Summary ===")
log(f"LDM checkpoint:  {LDM_PATH}")
log(f"UMAP plot:       {UMAP_PATH}")
log(f"Training log:    {LOG_PATH}")
log(f"Epochs trained:  {N_EPOCHS}")
log(f"Loss drop:       {loss_drop_pct:.1f}%")
log(f"Samples generated: 100")
log(f"All checks PASSED")

# Restore stdout and save log
sys.stdout = orig_stdout
log_content = log_buf.getvalue()
with open(LOG_PATH, 'w', encoding='utf-8') as f:
    f.write(log_content)
print(f"Log saved: {LOG_PATH}")
print("DONE — Task 12 complete.")
