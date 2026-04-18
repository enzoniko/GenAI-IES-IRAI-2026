"""
T13: Standalone training script for Decoder 2 (CVAE) on residual signals.
Conditions on z_macro. Residuals computed in normalized domain.
"""
import sys
import os
import io
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import src.configs as cfg
from src.models.ts_jepa import TSJEPA
from src.models.decoder1 import Decoder1
from src.models.decoder2_cvae import Decoder2CVAE

# ── Configuration ─────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_EPOCHS = 50
PATIENCE = 15
LR = 1e-3
BATCH_SIZE = 16
BETA_KL = 0.01   # start conservative; task says increase if KL=0

Y_FILES = [
    ('data/processed-mafaulda/16hz/Y_normal_trainingset.pth',                                0),
    ('data/processed-mafaulda/16hz/Y_imbalance_fault_20g_trainingset.pth',                   1),
    ('data/processed-mafaulda/16hz/Y_vertical_misalignment_fault_1.27mm_trainingset.pth',    2),
    ('data/processed-mafaulda/16hz/Y_overhang_ball_fault_20g_trainingset.pth',               3),
]

EVIDENCE_DIR = '.sisyphus/evidence'
os.makedirs(EVIDENCE_DIR, exist_ok=True)
os.makedirs('results', exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
class TeeOutput:
    """Write to both stdout and a string buffer."""
    def __init__(self, original):
        self.original = original
        self.buf = io.StringIO()

    def write(self, text):
        self.original.write(text)
        self.buf.write(text)

    def flush(self):
        self.original.flush()

    def getvalue(self):
        return self.buf.getvalue()


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def kl_loss_fn(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


# ── Step 1: Load & normalize data ─────────────────────────────────────────────
def load_data():
    meta = torch.load('results/normalization_metadata.pth', weights_only=False)
    y_min = meta['y_min'].float()  # [4]
    y_max = meta['y_max'].float()  # [4]

    all_Y = []
    all_labels = []
    counts = []
    for fpath, label in Y_FILES:
        Y = torch.load(fpath, weights_only=False).float()
        # Shape check: should be (N, T, 4) -> transpose to (N, 4, T)
        if Y.ndim == 3 and Y.shape[-1] == 4:
            Y = Y.permute(0, 2, 1)   # (N, 4, T)
        elif Y.ndim == 3 and Y.shape[1] == 4:
            pass  # already (N, 4, T)
        else:
            raise ValueError(f"Unexpected Y shape {Y.shape} in {fpath}")

        # Normalize to [0, 1]
        Y_norm = (Y - y_min[None, :, None]) / (y_max[None, :, None] - y_min[None, :, None] + 1e-8)
        n = Y_norm.shape[0]
        counts.append(n)
        all_Y.append(Y_norm)
        all_labels.append(torch.full((n,), label, dtype=torch.long))
        print(f"  Loaded label={label}: {n} windows, shape {Y_norm.shape}, file={os.path.basename(fpath)}")

    Y_all = torch.cat(all_Y, dim=0)     # (N_total, 4, T)
    L_all = torch.cat(all_labels, dim=0)  # (N_total,)
    print(f"  Total: {Y_all.shape[0]} windows, shape {Y_all.shape}, counts={counts}")
    return Y_all, L_all, counts


# ── Step 2: Train/val split (interleaved 80/20) ────────────────────────────────
def split_data(Y_all, L_all):
    N = Y_all.shape[0]
    val_idx = list(range(4, N, 5))
    train_idx = [i for i in range(N) if i not in set(val_idx)]
    return (Y_all[train_idx], L_all[train_idx],
            Y_all[val_idx],   L_all[val_idx])


# ── Step 3: Build z_macro & residuals dataset ──────────────────────────────────
def build_residual_dataset(Y_tensor, L_tensor, tsjepa, dec1):
    """Precompute (z_macro, residual, label) for all samples."""
    dataset = []
    BS = 32
    N = Y_tensor.shape[0]
    with torch.no_grad():
        for start in range(0, N, BS):
            y_b = Y_tensor[start:start+BS].to(DEVICE)
            l_b = L_tensor[start:start+BS]
            z_macro = tsjepa.get_z_macro(y_b)        # (B, 128)
            recon   = dec1(z_macro)                   # (B, 4, T)
            residual = y_b - recon                    # (B, 4, T)
            for i in range(y_b.shape[0]):
                dataset.append((z_macro[i].cpu(), residual[i].cpu(), l_b[i]))
    return dataset


def make_batches(dataset, batch_size, shuffle=True):
    import random
    indices = list(range(len(dataset)))
    if shuffle:
        random.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start+batch_size]
        z_m  = torch.stack([dataset[i][0] for i in batch_idx])
        res  = torch.stack([dataset[i][1] for i in batch_idx])
        lbl  = torch.stack([dataset[i][2] for i in batch_idx])
        yield z_m.to(DEVICE), res.to(DEVICE), lbl.to(DEVICE)


# ── Step 4: Training ───────────────────────────────────────────────────────────
def compute_loss(dec2, z_m, res, lbl, beta_kl):
    recon_res, mu, logvar = dec2(res, z_m, lbl)
    recon_loss = nn.functional.mse_loss(recon_res, res)
    kl_raw     = kl_loss_fn(mu, logvar)
    kl_loss    = kl_raw / res.size(0)
    total      = recon_loss + beta_kl * kl_loss
    return total, recon_loss.item(), kl_loss.item()


def evaluate(dec2, val_dataset, beta_kl):
    dec2.eval()
    tot_loss = tot_recon = tot_kl = 0.0
    n_batches = 0
    with torch.no_grad():
        for z_m, res, lbl in make_batches(val_dataset, BATCH_SIZE, shuffle=False):
            total, recon, kl = compute_loss(dec2, z_m, res, lbl, beta_kl)
            tot_loss  += total.item()
            tot_recon += recon
            tot_kl    += kl
            n_batches += 1
    return tot_loss/n_batches, tot_recon/n_batches, tot_kl/n_batches


def train(dec2, train_dataset, val_dataset, beta_kl):
    optimizer = optim.Adam(dec2.parameters(), lr=LR)
    best_val_elbo = float('inf')
    best_state = None
    patience_counter = 0
    best_epoch = 0

    epoch_log = []  # list of (epoch, t_total, t_recon, t_kl, v_elbo)

    for epoch in range(1, MAX_EPOCHS + 1):
        dec2.train()
        n_batches = 0
        run_total = run_recon = run_kl = 0.0

        for z_m, res, lbl in make_batches(train_dataset, BATCH_SIZE):
            optimizer.zero_grad()
            total, recon, kl = compute_loss(dec2, z_m, res, lbl, beta_kl)
            total.backward()
            optimizer.step()
            run_total  += total.item()
            run_recon  += recon
            run_kl     += kl
            n_batches  += 1

        t_total = run_total / n_batches
        t_recon = run_recon / n_batches
        t_kl    = run_kl   / n_batches

        v_elbo, v_recon, v_kl = evaluate(dec2, val_dataset, beta_kl)

        print(f"Epoch {epoch:3d}/{MAX_EPOCHS}: total={t_total:.4f}, recon={t_recon:.4f}, "
              f"kl={t_kl:.4f} | val_elbo={v_elbo:.4f}")
        epoch_log.append((epoch, t_total, t_recon, t_kl, v_elbo))

        # Early stopping on val ELBO
        if v_elbo < best_val_elbo - 1e-5:
            best_val_elbo = v_elbo
            best_state = {k: v.cpu().clone() for k, v in dec2.state_dict().items()}
            patience_counter = 0
            best_epoch = epoch
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch} (patience={PATIENCE})")
                break

    # Restore best
    if best_state is not None:
        dec2.load_state_dict(best_state)
    print(f"  Best checkpoint: epoch {best_epoch}, val_elbo={best_val_elbo:.4f}")
    return epoch_log, best_epoch, best_val_elbo


# ── Step 5: Reconstruction comparison plot ────────────────────────────────────
def make_comparison_plot(Y_all, L_all, tsjepa, dec1, dec2, y_min, y_max):
    """Plot original vs envelope vs total_recon for 2 samples per class."""
    dec2.eval()
    fig, axes = plt.subplots(4 * 4, 3, figsize=(18, 4 * 4 * 1.8))
    # 4 classes x 4 channels; 3 columns: original, dec1_envelope, total_recon

    T = Y_all.shape[-1]
    t = range(T)
    row = 0
    with torch.no_grad():
        for cls in range(4):
            cls_idx = (L_all == cls).nonzero(as_tuple=True)[0]
            # pick 2 samples (first and mid)
            sample_indices = [cls_idx[0].item(), cls_idx[len(cls_idx)//2].item()]
            for s_idx in sample_indices[:1]:  # 1 sample per class for clarity
                y_sample = Y_all[s_idx:s_idx+1].to(DEVICE)   # (1, 4, T)
                lbl_t    = L_all[s_idx:s_idx+1].to(DEVICE)
                z_macro  = tsjepa.get_z_macro(y_sample)
                dec1_out = dec1(z_macro)
                residual = y_sample - dec1_out
                jitter   = dec2.sample(z_macro, lbl_t)
                total_recon = dec1_out + jitter

                for ch in range(4):
                    ax = axes[row]
                    orig  = y_sample[0, ch].cpu().numpy()
                    env   = dec1_out[0, ch].cpu().numpy()
                    total = total_recon[0, ch].cpu().numpy()

                    ax[0].plot(t, orig, color='tab:blue', lw=0.5, alpha=0.8, label='Original')
                    ax[0].plot(t, env,  color='tab:orange', lw=0.8, ls='--', label='Dec1 envelope')
                    if row == 0 and ch == 0: ax[0].legend(fontsize=6)
                    ax[0].set_ylabel(f'cls={cls} ch={ch}', fontsize=7)
                    ax[0].set_title('Original vs Envelope', fontsize=7)
                    ax[0].tick_params(labelsize=6)

                    ax[1].plot(t, residual[0, ch].cpu().numpy(), color='tab:red', lw=0.5, label='True residual')
                    ax[1].set_title('Residual', fontsize=7)
                    ax[1].tick_params(labelsize=6)

                    ax[2].plot(t, orig,  color='tab:blue', lw=0.5, alpha=0.5, label='Original')
                    ax[2].plot(t, total, color='tab:green', lw=0.7, ls='-', label='Total recon')
                    if row == 0: ax[2].legend(fontsize=6)
                    ax[2].set_title('Total Recon', fontsize=7)
                    ax[2].tick_params(labelsize=6)

                    row += 1

    # Hide unused axes
    for r in range(row, len(axes)):
        for a in axes[r]: a.axis('off')

    plt.suptitle('T13: Decoder2 CVAE - Reconstruction Comparison\n(Envelope + Jitter vs Original)',
                 fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(EVIDENCE_DIR, 'task-13-reconstruction-comparison.png')
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"  Plot saved: {out_path}")
    return out_path


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    original_stdout = sys.stdout
    tee = TeeOutput(original_stdout)
    sys.stdout = tee

    try:
        print("=" * 60)
        print("T13: Decoder 2 CVAE Training")
        print(f"Device: {DEVICE}")
        print("=" * 60)

        # Load normalization
        meta = torch.load('results/normalization_metadata.pth', weights_only=False)
        y_min = meta['y_min'].float()
        y_max = meta['y_max'].float()

        # Load data
        print("\n[1] Loading & normalizing data...")
        Y_all, L_all, counts = load_data()
        N_total = Y_all.shape[0]
        print(f"  Total windows: {N_total}, counts per class: {counts}")

        # Load frozen models
        print("\n[2] Loading frozen TS-JEPA and Decoder1...")
        tsjepa = TSJEPA(in_channels=4).to(DEVICE)
        tsjepa.load_state_dict(torch.load('results/ts_jepa.pth', map_location=DEVICE, weights_only=True))
        tsjepa.eval()
        for p in tsjepa.parameters(): p.requires_grad = False
        print(f"  TS-JEPA loaded, params={count_params(tsjepa):,} (all frozen)")

        SEQ_LEN = Y_all.shape[-1]
        dec1 = Decoder1(d_model=128, seq_length=SEQ_LEN, out_channels=4).to(DEVICE)
        dec1.load_state_dict(torch.load('results/decoder1.pth', map_location=DEVICE, weights_only=True))
        dec1.eval()
        for p in dec1.parameters(): p.requires_grad = False
        print(f"  Decoder1 loaded, params={count_params(dec1):,} (all frozen)")

        # Instantiate Decoder2 CVAE
        dec2 = Decoder2CVAE(
            seq_length=SEQ_LEN,
            in_channels=4,
            context_dim=128,
            latent_dim=64,
            num_classes=4,
            label_embed_dim=16
        ).to(DEVICE)
        print(f"  Decoder2CVAE initialized, params={count_params(dec2):,}")
        print(f"  latent_dim=64, context_dim=128, seq_length={SEQ_LEN}")

        # Build residual datasets
        print("\n[3] Building residual datasets (precomputing z_macro + residuals)...")
        Y_train, L_train, Y_val, L_val = split_data(Y_all, L_all)
        print(f"  Train: {len(Y_train)} windows, Val: {len(Y_val)} windows")

        train_dataset = build_residual_dataset(Y_train, L_train, tsjepa, dec1)
        val_dataset   = build_residual_dataset(Y_val,   L_val,   tsjepa, dec1)
        print(f"  Residual dataset: train={len(train_dataset)}, val={len(val_dataset)}")

        # Train CVAE
        print(f"\n[4] Training CVAE (max_epochs={MAX_EPOCHS}, patience={PATIENCE}, beta_kl={BETA_KL})...")
        epoch_log, best_epoch, best_val_elbo = train(dec2, train_dataset, val_dataset, BETA_KL)

        # Check KL
        _, best_recon, best_kl = evaluate(dec2, val_dataset, BETA_KL)
        kl_status = "NOT COLLAPSED" if best_kl > 1e-4 else "WARNING: KL ~0"
        print(f"\n  Final KL (val): {best_kl:.6f} -> {kl_status}")

        # If KL collapsed, retry with higher beta
        if best_kl <= 1e-4:
            print("  KL collapsed! Retraining with beta_kl=0.1...")
            dec2_v2 = Decoder2CVAE(
                seq_length=SEQ_LEN, in_channels=4, context_dim=128,
                latent_dim=64, num_classes=4, label_embed_dim=16
            ).to(DEVICE)
            epoch_log2, best_epoch2, best_val_elbo2 = train(dec2_v2, train_dataset, val_dataset, 0.1)
            _, best_recon2, best_kl2 = evaluate(dec2_v2, val_dataset, 0.1)
            if best_kl2 > best_kl:
                print(f"  Beta=0.1 improved KL: {best_kl:.6f} -> {best_kl2:.6f}")
                dec2 = dec2_v2
                epoch_log = epoch_log2
                best_epoch = best_epoch2
                best_val_elbo = best_val_elbo2
                best_recon = best_recon2
                best_kl = best_kl2
                BETA_KL_USED = 0.1
                kl_status = "NOT COLLAPSED" if best_kl > 1e-4 else "WARNING: KL ~0"
            else:
                BETA_KL_USED = BETA_KL
        else:
            BETA_KL_USED = BETA_KL

        # Save checkpoint
        print("\n[5] Saving checkpoint...")
        torch.save(dec2.state_dict(), 'results/decoder2.pth')
        sz_mb = os.path.getsize('results/decoder2.pth') / 1e6
        print(f"  Saved: results/decoder2.pth ({sz_mb:.2f}MB)")

        # Validation: reload and verify
        chk = torch.load('results/decoder2.pth', map_location='cpu', weights_only=True)
        print(f"  Checkpoint reload: OK, {len(chk)} keys")

        # Generate comparison plot
        print("\n[6] Generating reconstruction comparison plot...")
        make_comparison_plot(Y_all, L_all, tsjepa, dec1, dec2, y_min, y_max)

        # Build evidence file
        print("\n[7] Writing evidence file...")
        lines = []
        lines.append("=== T13: Decoder 2 CVAE Training ===\n")
        lines.append(f"Architecture: Decoder2CVAE, params={count_params(dec2):,}\n")
        lines.append(f"  EncoderCVAE + DecoderCVAE (latent_dim=64, context_dim=128, num_classes=4)\n")
        lines.append(f"Data: {N_total} windows ({'+'.join(str(c) for c in counts)}), "
                     f"z_jitter_dim=64, z_macro_dim=128\n")
        lines.append(f"  Labels: 0=normal, 1=imbalance_20g, 2=vmisalign_1.27mm, 3=overhang_ball_20g\n")
        lines.append(f"  SEQ_LENGTH={SEQ_LEN}\n")
        lines.append(f"Split: train={len(train_dataset)}, val={len(val_dataset)} (interleaved 80/20)\n")
        lines.append(f"Hyperparams: lr={LR}, batch_size={BATCH_SIZE}, beta_kl={BETA_KL_USED}, "
                     f"patience={PATIENCE}\n")
        lines.append(f"\nTraining log (selected epochs):\n")
        display_epochs = set([1, 10, 25, 50, best_epoch])
        for ep, t_tot, t_rec, t_kl, v_elbo in epoch_log:
            if ep in display_epochs or ep == epoch_log[-1][0]:
                lines.append(f"  Epoch {ep:3d}: total={t_tot:.4f}, recon={t_rec:.4f}, "
                              f"kl={t_kl:.4f} | val_elbo={v_elbo:.4f}\n")

        lines.append(f"\nFull training log:\n")
        for ep, t_tot, t_rec, t_kl, v_elbo in epoch_log:
            lines.append(f"  Epoch {ep:3d}: total={t_tot:.4f}, recon={t_rec:.4f}, "
                         f"kl={t_kl:.4f} | val_elbo={v_elbo:.4f}\n")

        lines.append(f"\nFinal (best checkpoint, epoch {best_epoch}):\n")
        lines.append(f"  ELBO: {best_val_elbo:.4f}\n")
        lines.append(f"  Recon loss: {best_recon:.4f}\n")
        lines.append(f"  KL divergence: {best_kl:.6f}\n")
        lines.append(f"  KL status: {kl_status}\n")
        lines.append(f"\nCheckpoint: results/decoder2.pth ({sz_mb:.2f}MB)\n")

        evidence_path = os.path.join(EVIDENCE_DIR, 'task-13-decoder2-training.txt')
        with io.open(evidence_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print(f"  Evidence saved: {evidence_path}")

        print("\n=== T13 COMPLETE ===")
        print(f"  KL status: {kl_status}")
        print(f"  best_epoch={best_epoch}, val_elbo={best_val_elbo:.4f}, kl={best_kl:.6f}")

    finally:
        sys.stdout = original_stdout


if __name__ == '__main__':
    main()
