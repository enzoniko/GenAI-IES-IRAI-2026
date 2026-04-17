"""
main_synthetic.py — Paper-Aligned Synthetic Pipeline
=====================================================
Runs the full Phase 0 -> 1 -> 2 pipeline on the 1D damped-oscillator
synthetic dataset (paper Eq. 4).

Reduced epochs for quick validation:
  Phase 0 PINN      : PINN_EPOCHS  = 50
  Phase 1 TS-JEPA   : TSJEPA_EPOCHS= 20
  Phase 2 LDM       : LDM_EPOCHS   = 20

All outputs go to results-synthetic/  (never overwrites results/).

Usage:
  python main_synthetic.py
  python main_synthetic.py --pinn_epochs 50 --tsjepa_epochs 20 --ldm_epochs 20
"""

# ── Override cfg paths BEFORE any other imports that cache them ───────────────
import src.configs as cfg
import os

SYNTHETIC_RESULTS_DIR = "results-synthetic"
SYNTHETIC_DATA_DIR    = "data/processed-synthetic"

os.makedirs(SYNTHETIC_RESULTS_DIR, exist_ok=True)
os.makedirs(SYNTHETIC_DATA_DIR,    exist_ok=True)

cfg.RESULTS_DIR        = SYNTHETIC_RESULTS_DIR
cfg.NORM_METADATA_PATH = os.path.join(SYNTHETIC_RESULTS_DIR, "normalization_metadata.pth")
cfg.PINN_MODEL_PATH    = os.path.join(SYNTHETIC_RESULTS_DIR, "pinn.pth")
cfg.JEPA_MODEL_PATH    = os.path.join(SYNTHETIC_RESULTS_DIR, "ts_jepa.pth")
cfg.DEC1_MODEL_PATH    = os.path.join(SYNTHETIC_RESULTS_DIR, "decoder1.pth")
cfg.DEC2_MODEL_PATH    = os.path.join(SYNTHETIC_RESULTS_DIR, "decoder2.pth")
cfg.LDM_MODEL_PATH     = os.path.join(SYNTHETIC_RESULTS_DIR, "ldm.pth")

# ── Standard imports (after cfg override) ────────────────────────────────────
import sys
import io
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from src.data.paper_synthetic_oscillator import (
    generate_and_save_tensors,
    get_synthetic_dataloaders,
    SyntheticOscillatorDataset,
    OMEGA_0,
    CLASS_NAMES,
)
from main_phase0 import step2_train_pinn_oracle, evaluate_on_test_set
from src.pipelines.train_phase1 import (
    train_phase1_tsjepa,
    train_phase1_decoder1,
    extract_residuals_and_train_decoder2,
    plot_umap,
)
from src.pipelines.run_sdedit_phase2 import train_latent_diffusion
from src.models import (
    TSJEPA,
    Decoder1,
    Decoder2CVAE,
    LatentDiffusionMLP,
    DDPMScheduler,
)

# ── Default epoch budgets ─────────────────────────────────────────────────────
PINN_EPOCHS   = 50
TSJEPA_EPOCHS = 20
LDM_EPOCHS    = 20
BATCH_SIZE    = 32

EVIDENCE_DIR = ".sisyphus/evidence"
os.makedirs(EVIDENCE_DIR, exist_ok=True)


# ── Evidence file helper ──────────────────────────────────────────────────────
class TeeBuffer:
    """Write to both the REAL stdout and an in-memory buffer.

    Must be constructed BEFORE sys.stdout is replaced so it can hold a
    reference to the original stream (avoids infinite recursion).
    """
    def __init__(self, real_stdout):
        self._real = real_stdout
        self._buf  = io.StringIO()

    def write(self, msg):
        self._real.write(msg)
        self._buf.write(msg)

    def flush(self):
        self._real.flush()

    def getvalue(self):
        return self._buf.getvalue()


def save_evidence(filename, content):
    path = os.path.join(EVIDENCE_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  [evidence] Saved -> {path}")


# ── Phase 0: PINN ─────────────────────────────────────────────────────────────
def run_synthetic_phase0(pinn_epochs, batch_size):
    """Train PINN on synthetic normal-class (class 0) data.

    NOTE: The existing PINN uses 6-DOF rotor equations internally.
    For the synthetic oscillator these physics residuals are approximate,
    but the data-driven component still trains and the normalization metadata
    is correctly computed for use by downstream modules (Oracle, MaFaulDa loader).
    """
    print("\n" + "=" * 68)
    print("SYNTHETIC PHASE 0: PINN Training (normal-class healthy windows)")
    print("=" * 68)

    model_weights, X_max, X_min, y_max, y_min = step2_train_pinn_oracle(
        processed_data_dir=SYNTHETIC_DATA_DIR,
        output_model_path=cfg.PINN_MODEL_PATH,
        epochs=pinn_epochs,
        batch_size=batch_size,
        num_samples=None,
        patience=10,
        min_delta=1e-4,
    )
    evaluate_on_test_set(
        processed_data_dir=SYNTHETIC_DATA_DIR,
        model_weights=model_weights,
        X_max=X_max, X_min=X_min,
        y_max=y_max, y_min=y_min,
        batch_size=batch_size,
    )
    return model_weights, X_max, X_min, y_max, y_min


# ── Phase 1: TS-JEPA + Decoders ───────────────────────────────────────────────
def run_synthetic_phase1(tsjepa_epochs, batch_size, device):
    """Train TS-JEPA encoder + Decoder1 + Decoder2 on all synthetic classes."""
    print("\n" + "=" * 68)
    print("SYNTHETIC PHASE 1: TS-JEPA + Decoders Training")
    print("=" * 68)

    train_loader, val_loader, seq_len = get_synthetic_dataloaders(
        data_dir=SYNTHETIC_DATA_DIR,
        batch_size=batch_size,
        val_split=0.2,
    )
    print(f"  Seq length: {seq_len} | Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    ts_jepa  = TSJEPA(in_channels=4).to(device)
    decoder1 = Decoder1(out_channels=4, seq_length=seq_len).to(device)
    decoder2 = Decoder2CVAE(in_channels=4, seq_length=seq_len).to(device)

    ts_jepa  = train_phase1_tsjepa(ts_jepa, train_loader, val_loader, tsjepa_epochs, device)
    decoder1 = train_phase1_decoder1(ts_jepa, decoder1, train_loader, val_loader, tsjepa_epochs, device)
    decoder2 = extract_residuals_and_train_decoder2(
        ts_jepa, decoder1, decoder2, train_loader, val_loader, tsjepa_epochs, device
    )

    # Save checkpoints to results-synthetic/
    torch.save(ts_jepa.state_dict(),  cfg.JEPA_MODEL_PATH)
    torch.save(decoder1.state_dict(), cfg.DEC1_MODEL_PATH)
    torch.save(decoder2.state_dict(), cfg.DEC2_MODEL_PATH)
    print(f"Phase 1 checkpoints saved to {SYNTHETIC_RESULTS_DIR}/")

    # UMAP plot
    try:
        plot_umap(ts_jepa, val_loader, device)
    except Exception as e:
        print(f"  [UMAP] skipped: {e}")

    return ts_jepa, decoder1, decoder2, train_loader, val_loader, seq_len


# ── Phase 2: LDM Training ─────────────────────────────────────────────────────
def run_synthetic_phase2_ldm(ts_jepa, train_loader, val_loader, ldm_epochs, device):
    """Train the Latent Diffusion Model on synthetic z_macro embeddings."""
    print("\n" + "=" * 68)
    print("SYNTHETIC PHASE 2: LDM Training on z_macro")
    print("=" * 68)

    ldm = LatentDiffusionMLP(
        z_dim=int(cfg.JEPA_CONFIG['d_model']), time_dim=64
    ).to(device)
    scheduler = DDPMScheduler(
        num_train_timesteps=int(cfg.SDEDIT_GUIDANCE_SETTINGS['num_inference_steps']),
        device=device,
    )

    # Freeze TS-JEPA
    for p in ts_jepa.parameters():
        p.requires_grad = False

    ldm = train_latent_diffusion(
        ts_jepa, ldm, scheduler,
        train_loader, val_loader, device, epochs=ldm_epochs
    )

    torch.save(ldm.state_dict(), cfg.LDM_MODEL_PATH)
    print(f"LDM checkpoint saved to {cfg.LDM_MODEL_PATH}")

    return ldm, scheduler


# ── LDM sampling helper ───────────────────────────────────────────────────────
def _sample_ldm_latents(ldm, scheduler, n_samples, device, n_steps=100):
    """Generate z_macro samples from pure Gaussian noise using N reverse steps.

    Uses a strided subset of the 1000-step schedule (every 10 steps) to keep
    sampling fast while still traversing the full noise range.
    """
    ldm.eval()
    z_dim   = cfg.JEPA_CONFIG['d_model']
    stride  = max(1, scheduler.num_train_timesteps // n_steps)
    t_steps = list(range(scheduler.num_train_timesteps - 1, -1, -stride))

    z_t = torch.randn(n_samples, int(z_dim), device=device)

    with torch.no_grad():
        for t in t_steps:
            t_tensor = torch.tensor([t] * n_samples, device=device).long()
            noise_pred = ldm(z_t.float(), t_tensor)
            z_t = scheduler.step(noise_pred, t, z_t)

    return z_t.cpu().numpy()


# ── Fairness metrics ──────────────────────────────────────────────────────────
def compute_fairness(ts_jepa, ldm, scheduler,
                     real_train_loader, device,
                     n_generated=200, n_ldm_steps=100):
    """Compute oracle and TSTR accuracy.

    Oracle (upper bound)
    --------------------
    Logistic Regression trained on real z_macro features, evaluated via
    5-fold cross-validation on the same real data.

    TSTR (Train on Synthetic, Test on Real)
    ----------------------------------------
    1. Sample `n_generated` z_macro vectors from the LDM.
    2. Assign pseudo-labels using the nearest real cluster centroid.
    3. Train LR on generated z_macro + pseudo-labels.
    4. Test the LR on real z_macro + true labels.
    """
    print("\n--- Fairness Evaluation ---")

    # 1. Collect real z_macro and true labels
    ts_jepa.eval()
    all_z_real, all_labels_real = [], []
    with torch.no_grad():
        for batch in real_train_loader:
            raw, _, labels = batch
            raw = raw.to(device)
            z   = ts_jepa.get_z_macro(raw)
            all_z_real.append(z.cpu().numpy())
            all_labels_real.append(labels.numpy())

    z_real      = np.concatenate(all_z_real,    axis=0)  # (N, 128)
    labels_real = np.concatenate(all_labels_real, axis=0) # (N,)

    n_classes = len(np.unique(labels_real))
    print(f"  Real data: {len(z_real)} samples, {n_classes} classes")

    # 2. Oracle accuracy (5-fold cross-val on real data)
    scaler_oracle = StandardScaler()
    z_real_scaled = scaler_oracle.fit_transform(z_real)
    lr_oracle = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    cv_scores  = cross_val_score(lr_oracle, z_real_scaled, labels_real, cv=5, scoring='accuracy')
    oracle_acc = float(cv_scores.mean())
    oracle_std = float(cv_scores.std())
    print(f"  Oracle accuracy (5-fold CV): {oracle_acc:.4f} +/- {oracle_std:.4f}")

    # 3. Generate z_macro from LDM
    print(f"  Sampling {n_generated} latents from LDM ({n_ldm_steps} steps)...")
    z_gen = _sample_ldm_latents(ldm, scheduler, n_generated, device, n_steps=n_ldm_steps)

    # 4. Assign pseudo-labels by nearest class centroid
    centroids = np.stack(
        [z_real[labels_real == c].mean(axis=0) for c in range(n_classes)],
        axis=0
    )  # (n_classes, 128)
    dists = np.linalg.norm(
        z_gen[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2
    )  # (n_gen, n_classes)
    pseudo_labels = np.argmin(dists, axis=1)

    unique, counts = np.unique(pseudo_labels, return_counts=True)
    print(f"  Pseudo-label distribution: { {int(u): int(c) for u, c in zip(unique, counts)} }")

    # 5. TSTR: train on generated, test on real
    scaler_tstr = StandardScaler()
    z_gen_scaled = scaler_tstr.fit_transform(z_gen)
    lr_tstr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    lr_tstr.fit(z_gen_scaled, pseudo_labels)

    z_real_in_gen_space = scaler_tstr.transform(z_real)
    tstr_preds = lr_tstr.predict(z_real_in_gen_space)
    tstr_acc   = float((tstr_preds == labels_real).mean())
    print(f"  TSTR accuracy (generated -> real): {tstr_acc:.4f}")

    ratio = tstr_acc / oracle_acc if oracle_acc > 1e-9 else 0.0
    print(f"  Ratio (TSTR/oracle): {ratio:.4f}  [acceptable: 0.30–0.95]")

    assessment = (
        "PASS (acceptable range 0.30–0.95)"
        if 0.30 <= ratio <= 0.95
        else f"WARN (outside range: ratio={ratio:.4f})"
    )
    print(f"  Assessment: {assessment}")

    return {
        'oracle_accuracy':     oracle_acc,
        'oracle_std':          oracle_std,
        'tstr_accuracy':       tstr_acc,
        'ratio':               ratio,
        'assessment':          assessment,
        'n_real':              int(len(z_real)),
        'n_generated':         n_generated,
        'n_classes':           n_classes,
        'pseudo_label_counts': {int(u): int(c) for u, c in zip(unique, counts)},
    }


# ── Main orchestrator ─────────────────────────────────────────────────────────
def run_synthetic_pipeline(pinn_epochs   = PINN_EPOCHS,
                            tsjepa_epochs = TSJEPA_EPOCHS,
                            ldm_epochs    = LDM_EPOCHS,
                            batch_size    = BATCH_SIZE):

    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps'  if torch.backends.mps.is_available() else 'cpu'
    )
    print(f"Device: {device}")
    print(f"Epochs -> PINN: {pinn_epochs}, TS-JEPA: {tsjepa_epochs}, LDM: {ldm_epochs}")

    # ── Step 0: Generate synthetic data ──────────────────────────────────────
    osc_check_log = io.StringIO()
    print("\n" + "=" * 68)
    print("GENERATING 1D OSCILLATOR SYNTHETIC DATA")
    print("=" * 68)

    generate_and_save_tensors(
        output_dir=SYNTHETIC_DATA_DIR,
        train_per_class=50,
        test_per_class=10,
        verbose=True,
    )
    osc_log_lines = [
        "paper_synthetic_oscillator.py – oscillator check",
        f"Generated at: {datetime.datetime.now().isoformat()}",
        "",
        "ODE:  m*x'' + c*x' + k*x = A*sin(ω₀*t)",
        "Healthy params: m=1, c=0.5, k=4.0, A=1.0, ω₀=1.0",
        "Fault classes:",
        "  0 – Healthy:              k=4.0 (baseline)",
        "  1 – Stiffness reduction:  k->0.5k=2.0 (crack analog)",
        "  2 – Damping increase:     c->2c=1.0   (lubrication analog)",
        "  3 – Forcing perturbation: A->2A=2.0   (imbalance analog)",
        "",
        f"SEQ_LENGTH = {cfg.SEQ_LENGTH}  @ TARGET_HZ = {cfg.TARGET_HZ} Hz",
        f"Total duration per window ≈ {cfg.SEQ_LENGTH / cfg.TARGET_HZ:.1f} s",
        "Samples per class: 50 train + 10 test = 60 total",
        "Y format: (N, T, 4) — 4-channel acceleration (ch1=true, ch2-4=+noise)",
        "X format: (N, T, 10) — [vel×4, pos×4, omega=1.0, time]",
        "",
    ]
    # Verify a sample window
    from src.data.paper_synthetic_oscillator import _solve_window, FAULT_PARAMS
    for fc in range(4):
        x, xdot, xddot, t = _solve_window(fc, window_idx=0)
        osc_log_lines.append(
            f"  Class {fc}: xddot range=[{xddot.min():.4f}, {xddot.max():.4f}]"
            f"  x range=[{x.min():.4f}, {x.max():.4f}]"
        )
    save_evidence("task-8-oscillator-check.txt", "\n".join(osc_log_lines))

    # ── Step 1: Phase 0 – PINN ────────────────────────────────────────────────
    old_stdout = sys.stdout
    p0_buf = TeeBuffer(old_stdout)
    sys.stdout = p0_buf
    try:
        model_weights, X_max, X_min, y_max, y_min = run_synthetic_phase0(
            pinn_epochs, batch_size
        )
    finally:
        sys.stdout = old_stdout

    # ── Step 2: Phase 1 – TS-JEPA + Decoders ─────────────────────────────────
    p1_buf = TeeBuffer(old_stdout)
    sys.stdout = p1_buf
    try:
        ts_jepa, decoder1, decoder2, train_loader, val_loader, seq_len = \
            run_synthetic_phase1(tsjepa_epochs, batch_size, device)
    finally:
        sys.stdout = old_stdout

    # ── Step 3: Phase 2 – LDM ────────────────────────────────────────────────
    p2_buf = TeeBuffer(old_stdout)
    sys.stdout = p2_buf
    try:
        ldm, scheduler = run_synthetic_phase2_ldm(
            ts_jepa, train_loader, val_loader, ldm_epochs, device
        )
    finally:
        sys.stdout = old_stdout

    # Combine pipeline logs
    pipeline_log = (
        f"=== Synthetic Pipeline Run — {datetime.datetime.now().isoformat()} ===\n\n"
        "=== PHASE 0 OUTPUT ===\n" + p0_buf.getvalue() +
        "\n=== PHASE 1 OUTPUT ===\n" + p1_buf.getvalue() +
        "\n=== PHASE 2 OUTPUT ===\n" + p2_buf.getvalue()
    )
    save_evidence("task-8-synthetic-pipeline.txt", pipeline_log)

    # ── Step 4: Fairness evaluation ───────────────────────────────────────────
    fairness_buf = TeeBuffer(old_stdout)
    sys.stdout = fairness_buf
    try:
        metrics = compute_fairness(
            ts_jepa, ldm, scheduler,
            real_train_loader=train_loader,
            device=device,
            n_generated=200,
            n_ldm_steps=100,
        )
    finally:
        sys.stdout = old_stdout
    print(fairness_buf.getvalue(), end="")

    fairness_log_lines = [
        f"=== Fairness Report — {datetime.datetime.now().isoformat()} ===",
        "",
        "Method : TSTR in TS-JEPA latent space",
        "         Oracle  = 5-fold CV on real oscillator z_macro",
        "         TSTR    = LR trained on LDM-generated z_macro (pseudo-labeled",
        "                   by nearest real cluster centroid), tested on real",
        f"",
        f"Real samples   : {metrics['n_real']}",
        f"Generated      : {metrics['n_generated']}  (LDM, 100 denoising steps)",
        f"Classes        : {metrics['n_classes']}",
        f"",
        f"Oracle accuracy (5-fold CV) : {metrics['oracle_accuracy']:.4f}  +/- {metrics['oracle_std']:.4f}",
        f"TSTR accuracy              : {metrics['tstr_accuracy']:.4f}",
        f"Ratio (TSTR / oracle)      : {metrics['ratio']:.4f}",
        f"Acceptable range           : 0.30 – 0.95",
        f"Assessment                 : {metrics['assessment']}",
        "",
        f"Pseudo-label distribution  : {metrics['pseudo_label_counts']}",
        "",
        fairness_buf.getvalue(),
    ]
    save_evidence("task-8-fairness.txt", "\n".join(fairness_log_lines))

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("SYNTHETIC PIPELINE COMPLETE")
    print(f"  results-synthetic/  : {os.listdir(SYNTHETIC_RESULTS_DIR)}")
    print(f"  Oracle accuracy     : {metrics['oracle_accuracy']:.4f}")
    print(f"  TSTR accuracy       : {metrics['tstr_accuracy']:.4f}")
    print(f"  Ratio               : {metrics['ratio']:.4f}  -> {metrics['assessment']}")
    print("=" * 68)

    return metrics


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Synthetic oscillator pipeline (Phase 0->1->2)"
    )
    parser.add_argument("--pinn_epochs",   type=int, default=PINN_EPOCHS,
                        help="PINN training epochs (default 50)")
    parser.add_argument("--tsjepa_epochs", type=int, default=TSJEPA_EPOCHS,
                        help="TS-JEPA + decoder epochs (default 20)")
    parser.add_argument("--ldm_epochs",    type=int, default=LDM_EPOCHS,
                        help="LDM training epochs (default 20)")
    parser.add_argument("--batch_size",    type=int, default=BATCH_SIZE,
                        help="Mini-batch size (default 32)")
    args = parser.parse_args()

    run_synthetic_pipeline(
        pinn_epochs   = args.pinn_epochs,
        tsjepa_epochs = args.tsjepa_epochs,
        ldm_epochs    = args.ldm_epochs,
        batch_size    = args.batch_size,
    )
