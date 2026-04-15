"""
tools/plot_imbalance_phase2.py
--------------------------------
Standalone Phase 2 tool: trains a lightweight Latent Diffusion Model on the
frozen TS-JEPA latent space, calibrates Oracle target distributions, and runs
physics-guided SDEdit for a chosen MaFaulDa imbalance class.

Outputs (saved to --outdir, default: results-comparison/):
  mafaulda_umap_latent_space_imbalance.png        — UMAP of full dataset coloured by class
  mafaulda_umap_sdedit_trajectory_imbalance.png   — UMAP + SDEdit trajectory Healthy→Imbalance
  mafaulda_sdedit_counterfactual_imbalance.png    — 4-channel overlay healthy vs counterfactual

Usage (from project root):
    python -m tools.plot_imbalance_phase2
    python -m tools.plot_imbalance_phase2 --target-class 35 --ldm-epochs 10
    python -m tools.plot_imbalance_phase2 --target-class 38 --outdir results-comparison

Imbalance label mapping:
    35=6g  36=10g  37=15g  38=20g  39=25g  40=30g  41=35g
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.cm as cm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import src.configs as cfg
from src.models import TSJEPA, Decoder1, Decoder2CVAE, LatentDiffusionMLP, DDPMScheduler, PriorWorkOracle
from src.data import get_dataloaders
from src.pipelines.train_phase1 import EarlyStopping

IMBALANCE_NAMES = {
    35: "Imbalance 6g", 36: "Imbalance 10g", 37: "Imbalance 15g",
    38: "Imbalance 20g", 39: "Imbalance 25g", 40: "Imbalance 30g",
    41: "Imbalance 35g",
}

# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_phase1_models(device, seq_length):
    ts_jepa  = TSJEPA(in_channels=4).to(device)
    decoder1 = Decoder1(out_channels=4, seq_length=seq_length).to(device)
    decoder2 = Decoder2CVAE(in_channels=4, seq_length=seq_length).to(device)

    for name, model, path in [
        ("TS-JEPA",   ts_jepa,   cfg.JEPA_MODEL_PATH),
        ("Decoder 1", decoder1,  cfg.DEC1_MODEL_PATH),
        ("Decoder 2", decoder2,  cfg.DEC2_MODEL_PATH),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint for {name} not found at '{path}'.")
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        print(f"[Load] {name}  ← {path}")

    return ts_jepa, decoder1, decoder2


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight LDM training
# ─────────────────────────────────────────────────────────────────────────────

def train_ldm(ts_jepa, ldm, scheduler, train_loader, val_loader, device, epochs):
    print(f"\n[LDM] Training for {epochs} epochs …")
    optimizer = optim.Adam(ldm.parameters(), lr=cfg.PHASE2_TRAIN_SETTINGS['learning_rate'])
    lr_sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    stopper   = EarlyStopping(patience=5)
    criterion = nn.MSELoss()
    ts_jepa.eval()

    for epoch in range(epochs):
        ldm.train()
        t_loss = 0.0
        for batch in train_loader:
            raw, _, _ = batch
            raw = raw.to(device)
            with torch.no_grad():
                z = ts_jepa.get_z_macro(raw)
            noise = torch.randn_like(z)
            t_steps = torch.randint(0, scheduler.num_train_timesteps, (raw.shape[0],), device=device).long()
            noisy   = scheduler.add_noise(z, noise, t_steps).float()
            optimizer.zero_grad()
            pred = ldm(noisy, t_steps)
            loss = criterion(pred, noise)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()

        ldm.eval()
        v_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                raw, _, _ = batch
                raw = raw.to(device)
                z   = ts_jepa.get_z_macro(raw)
                noise  = torch.randn_like(z)
                t_steps = torch.randint(0, scheduler.num_train_timesteps, (raw.shape[0],), device=device).long()
                noisy   = scheduler.add_noise(z, noise, t_steps).float()
                v_loss += criterion(ldm(noisy, t_steps), noise).item()

        t_avg = t_loss / len(train_loader)
        v_avg = v_loss / len(val_loader)
        if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  train={t_avg:.4f}  val={v_avg:.4f}")
        lr_sched.step(v_avg)
        stopper(v_avg)
        if stopper.early_stop:
            print(f"  Early stopping at epoch {epoch+1}.")
            break

    print("[LDM] Training complete.")
    return ldm


# ─────────────────────────────────────────────────────────────────────────────
# Oracle calibration
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_oracle(ts_jepa, oracle, val_loader, device):
    print("\n[Oracle] Calibrating target distributions …")
    ts_jepa.eval(); oracle.eval()
    class_emb = {}; class_cnt = {}

    with torch.no_grad():
        for batch in val_loader:
            raw, _, labels = batch
            raw = raw.to(device)
            z_phys = oracle(raw, omega=batch.omega.to(device))
            for i in range(len(labels)):
                c = labels[i].item()
                if c not in class_emb:
                    class_emb[c] = torch.zeros_like(z_phys[i])
                    class_cnt[c] = 0
                class_emb[c] += z_phys[i]
                class_cnt[c] += 1

    for c in class_emb:
        oracle.set_target_distribution(c, class_emb[c] / class_cnt[c])
        print(f"  Class {c:2d}  ({class_cnt[c]} windows)")

    print("[Oracle] Done.")
    return oracle


# ─────────────────────────────────────────────────────────────────────────────
# Healthy trace extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_healthy(val_loader, device):
    for batch in val_loader:
        raw, _, label = batch
        idx = (label == 0).nonzero(as_tuple=True)[0]
        if len(idx) > 0:
            omega_batch = getattr(batch, 'omega', None)
            trace = raw[idx[0]:idx[0]+1].to(device)
            omega = omega_batch[idx[0]:idx[0]+1].to(device) if omega_batch is not None else None
            return trace, omega
    raise ValueError("No healthy (class 0) trace found in the validation set.")


# ─────────────────────────────────────────────────────────────────────────────
# Guided SDEdit
# ─────────────────────────────────────────────────────────────────────────────

def run_sdedit(ts_jepa, decoder1, decoder2, ldm, oracle, scheduler,
               healthy_trace, omega, target_class, device,
               num_inference_steps, guidance_scale, strength):
    print(f"\n[SDEdit] Healthy → Class {target_class}  "
          f"({IMBALANCE_NAMES.get(target_class, target_class)})")

    for m in [ts_jepa, decoder1, decoder2, ldm, oracle]:
        m.eval()

    t_start = int(num_inference_steps * strength)

    with torch.no_grad():
        z_start = ts_jepa.get_z_macro(healthy_trace).float()

    noise = torch.randn_like(z_start)
    z_t   = scheduler.add_noise(z_start, noise, torch.tensor([t_start], device=device).long())

    target_dist = oracle.get_target_distribution(target_class).to(device)
    mse = nn.MSELoss()
    z_trajectory = [z_start.detach().cpu()]

    for t in reversed(range(0, t_start)):
        z_trajectory.append(z_t.detach().cpu())
        t_tensor = torch.tensor([t], device=device).long()

        with torch.no_grad():
            uncond_pred = ldm(z_t, t_tensor)

        z_t = z_t.detach().requires_grad_(True)
        pred_trace = decoder1(z_t)
        pred_emb   = oracle(pred_trace, omega=omega)
        penalty    = mse(pred_emb.squeeze(0), target_dist)
        grad       = torch.autograd.grad(penalty, z_t)[0]
        z_t        = z_t.detach()

        z_t = scheduler.step(uncond_pred, t, z_t)
        z_t = z_t - guidance_scale * grad

        if t % 100 == 0:
            print(f"  step {t:4d}/{num_inference_steps}  penalty={penalty.item():.4f}")

    label_tensor = torch.tensor([target_class], dtype=torch.long, device=device)
    with torch.no_grad():
        final_macro  = decoder1(z_t)
        jitter       = decoder2.sample(z_t, label_tensor)
        counterfactual = final_macro + jitter

    z_trajectory.append(z_t.detach().cpu())
    return z_start, z_t, counterfactual, z_trajectory


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_umap_background(ts_jepa, val_loader, device, outdir, suffix="imbalance"):
    """UMAP of the entire val set coloured by class label."""
    try:
        import umap as umap_lib
    except ImportError:
        print("[UMAP] umap-learn not installed. Skipping.")
        return

    ts_jepa.eval()
    all_z, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            raw, _, labels = batch
            z = ts_jepa.get_z_macro(raw.to(device))
            all_z.append(z.cpu())
            all_labels.append(labels)

    all_z      = torch.cat(all_z, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    reducer   = umap_lib.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(all_z)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1],
                          c=all_labels, cmap='tab20', s=15, alpha=0.8)
    plt.colorbar(scatter, label='Fault Class ID')
    plt.title("UMAP Projection of TS-JEPA z_macro Latent Space")
    plt.xlabel("UMAP 1"); plt.ylabel("UMAP 2")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    save_path = os.path.join(outdir, f"mafaulda_umap_latent_space_{suffix}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plot] UMAP background → {save_path}")


def plot_sdedit_trajectory(ts_jepa, val_loader, z_trajectory, target_class,
                           device, outdir, suffix="imbalance"):
    try:
        import umap as umap_lib
    except ImportError:
        print("[UMAP] umap-learn not installed. Skipping.")
        return

    ts_jepa.eval()
    all_z, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            raw, _, labels = batch
            z = ts_jepa.get_z_macro(raw.to(device))
            all_z.append(z.cpu())
            all_labels.append(labels)

    all_z      = torch.cat(all_z, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    z_traj_arr = torch.cat(z_trajectory, dim=0).numpy()
    combined   = np.concatenate([all_z, z_traj_arr], axis=0)

    print("[UMAP] Fitting on combined background + trajectory …")
    reducer           = umap_lib.UMAP(n_components=2, random_state=42)
    combined_emb      = reducer.fit_transform(combined)

    bg_emb   = combined_emb[:len(all_z)]
    traj_emb = combined_emb[len(all_z):]

    plt.figure(figsize=(10, 8))
    plt.scatter(bg_emb[:, 0], bg_emb[:, 1],
                c=all_labels, cmap='viridis', s=15, alpha=0.3)

    colors = cm.Reds(np.linspace(0.2, 1.0, len(traj_emb)))
    plt.scatter(traj_emb[1:, 0], traj_emb[1:, 1],
                c=colors[1:], s=40, edgecolors='none',
                label='SDEdit Guided Path')
    plt.scatter(traj_emb[0, 0],  traj_emb[0, 1],
                color='black',   s=150, marker='X',  label='Start (Healthy)')
    plt.scatter(traj_emb[-1, 0], traj_emb[-1, 1],
                color='darkred', s=200, marker='*',
                label=f'End ({IMBALANCE_NAMES.get(target_class, f"Class {target_class}")})')

    plt.title(f"UMAP SDEdit Trajectory: Healthy → {IMBALANCE_NAMES.get(target_class, target_class)}")
    plt.xlabel("UMAP 1"); plt.ylabel("UMAP 2")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc="lower right")
    plt.tight_layout()
    save_path = os.path.join(outdir, f"mafaulda_umap_sdedit_trajectory_{suffix}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plot] SDEdit trajectory → {save_path}")


def plot_counterfactual(healthy_trace, counterfactual, target_class, outdir, suffix="imbalance"):
    seq_len = healthy_trace.shape[-1]
    t_ax    = torch.linspace(0, seq_len / cfg.SAMPLING_RATE, seq_len).numpy()

    fig, axes = plt.subplots(4, 1, figsize=(10, 12))
    name = IMBALANCE_NAMES.get(target_class, f"Class {target_class}")
    fig.suptitle(f"Phase 2: Physics-Guided Counterfactual\nHealthy → {name}", fontsize=16)

    healthy_np = healthy_trace[0].cpu().numpy()
    cf_np      = counterfactual[0].cpu().numpy()

    for c in range(4):
        axes[c].plot(t_ax, healthy_np[c], label='Original Healthy',
                     color='blue', alpha=0.5)
        axes[c].plot(t_ax, cf_np[c], label=f'Counterfactual ({name})',
                     color='red', alpha=0.8, linestyle='dashed')
        axes[c].set_ylabel(f"Ch {c+1}")
        if c == 0:
            axes[c].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(outdir, f"mafaulda_sdedit_counterfactual_{suffix}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Counterfactual → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 2 plots (UMAP + SDEdit trajectory) for MaFaulDa imbalance classes."
    )
    parser.add_argument("--target-class", type=int, default=35,
                        help="Imbalance label to target (default 35 = 6g).")
    parser.add_argument("--ldm-epochs",   type=int, default=15,
                        help="LDM training epochs (default 15; increase for better quality).")
    parser.add_argument("--num-samples",  type=int, default=5000,
                        help="Dataset samples to load.")
    parser.add_argument("--num-inference-steps", type=int,
                        default=cfg.SDEDIT_GUIDANCE_SETTINGS['num_inference_steps'])
    parser.add_argument("--guidance-scale", type=float,
                        default=cfg.SDEDIT_GUIDANCE_SETTINGS['guidance_scale'])
    parser.add_argument("--strength",    type=float,
                        default=cfg.SDEDIT_GUIDANCE_SETTINGS['strength'])
    parser.add_argument("--outdir",      type=str, default="results-comparison",
                        help="Output directory for plots.")
    parser.add_argument("--suffix",      type=str, default="imbalance",
                        help="File-name suffix (e.g. 'imbalance' → mafaulda_umap_latent_space_imbalance.png).")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"[Device] {device}")
    os.makedirs(args.outdir, exist_ok=True)

    # ── Sequence length auto-discovery ──────────────────────────────────────
    meta_path = os.path.join(cfg.DATA_DIR_PROCESSED, "metadata.json")
    seq_length = cfg.SEQ_LENGTH
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            seq_length = json.load(f).get("seq_length", cfg.SEQ_LENGTH)
    print(f"[Config] seq_length={seq_length}")

    # ── Phase 1 models ───────────────────────────────────────────────────────
    ts_jepa, decoder1, decoder2 = load_phase1_models(device, seq_length)

    # ── Data loaders ─────────────────────────────────────────────────────────
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=cfg.PHASE1_TRAIN_SETTINGS['batch_size'],
        num_samples=args.num_samples,
        val_split=0.2,
    )
    if val_loader is None:
        print("[ERROR] Dataloader failed. Check DATA_DIR_PROCESSED.")
        sys.exit(1)

    # ── Phase 2 modules ──────────────────────────────────────────────────────
    ldm       = LatentDiffusionMLP(z_dim=cfg.JEPA_CONFIG['d_model'], time_dim=64).to(device)
    oracle    = PriorWorkOracle().to(device)
    scheduler = DDPMScheduler(
        num_train_timesteps=args.num_inference_steps, device=device
    )

    # ── LDM training ─────────────────────────────────────────────────────────
    ldm = train_ldm(ts_jepa, ldm, scheduler, train_loader, val_loader,
                    device, epochs=args.ldm_epochs)

    # ── Oracle calibration ───────────────────────────────────────────────────
    oracle = calibrate_oracle(ts_jepa, oracle, val_loader, device)

    # ── Healthy trace ────────────────────────────────────────────────────────
    healthy_trace, healthy_omega = extract_healthy(val_loader, device)
    print(f"[Healthy] omega={healthy_omega.item():.2f} rad/s")

    # ── SDEdit ───────────────────────────────────────────────────────────────
    z_start, z_end, counterfactual, z_trajectory = run_sdedit(
        ts_jepa, decoder1, decoder2, ldm, oracle, scheduler,
        healthy_trace, healthy_omega,
        target_class=args.target_class,
        device=device,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        strength=args.strength,
    )

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_umap_background(ts_jepa, val_loader, device, args.outdir, suffix=args.suffix)
    plot_sdedit_trajectory(ts_jepa, val_loader, z_trajectory,
                           args.target_class, device, args.outdir, suffix=args.suffix)
    plot_counterfactual(healthy_trace, counterfactual,
                        args.target_class, args.outdir, suffix=args.suffix)

    print("\nDone.")


if __name__ == "__main__":
    main()
