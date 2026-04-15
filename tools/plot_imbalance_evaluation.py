"""
tools/plot_imbalance_evaluation.py
-----------------------------------
Standalone evaluation tool: loads frozen Phase 1 checkpoints and produces
multivariate evaluation plots for MaFaulDa imbalance classes only.

Usage (from project root):
    python -m tools.plot_imbalance_evaluation
    python -m tools.plot_imbalance_evaluation --label 35   # specific imbalance subtype
    python -m tools.plot_imbalance_evaluation --outdir results-comparison

Imbalance label mapping (from mafaulda_dataset.py):
    35 -> imbalance_fault_6g
    36 -> imbalance_fault_10g
    37 -> imbalance_fault_15g
    38 -> imbalance_fault_20g
    39 -> imbalance_fault_25g
    40 -> imbalance_fault_30g
    41 -> imbalance_fault_35g
"""

import argparse
import os
import sys

import torch
import matplotlib.pyplot as plt

# --- Ensure the project root is on sys.path when run as a module
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import src.configs as cfg
from src.models import TSJEPA, Decoder1, Decoder2CVAE
from src.data import get_dataloaders

# Human-readable names for the imbalance sub-types
IMBALANCE_NAMES = {
    35: "Imbalance 6g",
    36: "Imbalance 10g",
    37: "Imbalance 15g",
    38: "Imbalance 20g",
    39: "Imbalance 25g",
    40: "Imbalance 30g",
    41: "Imbalance 35g",
}

# Labels to evaluate by default (one representative subtype + healthy baseline)
DEFAULT_TARGET_LABELS = [0, 35]


def load_models(device: torch.device):
    """Load all three Phase 1 frozen checkpoints."""
    # We need the sequence length to instantiate the decoders.
    # Auto-discover it from the dataset metadata if available.
    meta_path = os.path.join(cfg.DATA_DIR_PROCESSED, "metadata.json")
    if os.path.exists(meta_path):
        import json
        with open(meta_path) as f:
            meta = json.load(f)
        seq_length = meta.get("seq_length", cfg.SEQ_LENGTH)
    else:
        seq_length = cfg.SEQ_LENGTH
    print(f"[Loader] Using seq_length={seq_length}")

    ts_jepa = TSJEPA(in_channels=4).to(device)
    decoder1 = Decoder1(out_channels=4, seq_length=seq_length).to(device)
    decoder2 = Decoder2CVAE(in_channels=4, seq_length=seq_length).to(device)

    for name, model, path in [
        ("TS-JEPA",   ts_jepa,   cfg.JEPA_MODEL_PATH),
        ("Decoder 1", decoder1,  cfg.DEC1_MODEL_PATH),
        ("Decoder 2", decoder2,  cfg.DEC2_MODEL_PATH),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Checkpoint for {name} not found at '{path}'. "
                "Run train_phase1 first."
            )
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        print(f"[Loader] Loaded {name} from {path}")

    return ts_jepa, decoder1, decoder2, seq_length


def collect_samples(val_loader, target_labels):
    """Iterate the val_loader once and collect one sample per target label."""
    samples = {l: None for l in target_labels}
    for batch in val_loader:
        raws, cleans, labels = batch
        for i in range(len(labels)):
            label = labels[i].item()
            if label in samples and samples[label] is None:
                samples[label] = (raws[i : i + 1], cleans[i : i + 1])
        if all(v is not None for v in samples.values()):
            break

    missing = [l for l, v in samples.items() if v is None]
    if missing:
        print(f"[WARNING] Could not find samples for labels: {missing}")
    return samples


def plot_evaluation(ts_jepa, decoder1, decoder2, samples, device, outdir):
    """Replicate the evaluate_pipeline plot logic for the given samples."""
    os.makedirs(outdir, exist_ok=True)

    metric_titles = [
        "Dec1 Recon (TS-JEPA Filter)",
        "True High-Freq Residual",
        "CVAE Generated Jitter",
        "Final Superposition",
    ]

    for idx_label, sample_data in samples.items():
        if sample_data is None:
            print(f"  Skipping label {idx_label} (no sample found).")
            continue

        raw, clean = sample_data
        raw, clean = raw.to(device), clean.to(device)
        label_tensor = torch.tensor([idx_label], dtype=torch.long, device=device)

        with torch.no_grad():
            z_macro = ts_jepa.get_z_macro(raw)
            recon_raw = decoder1(z_macro)
            residual = raw - recon_raw
            sampled_jitter = decoder2.sample(z_macro, label_tensor)
            final_synthetic_trace = recon_raw + sampled_jitter

        seq_len = raw.shape[-1]
        t = torch.linspace(0, seq_len / cfg.SAMPLING_RATE, seq_len).numpy()

        name = IMBALANCE_NAMES.get(idx_label, f"Label {idx_label}")
        fig, axes = plt.subplots(4, 4, figsize=(24, 16))
        fig.suptitle(f"Multivariate Analysis: {name} (Label {idx_label})", fontsize=20)

        for c in range(4):
            orig_c         = raw[0, c].cpu().numpy()
            recon_dec1_c   = recon_raw[0, c].cpu().numpy()
            res_c          = residual[0, c].cpu().numpy()
            synth_jitter_c = sampled_jitter[0, c].cpu().numpy()
            synth_final_c  = final_synthetic_trace[0, c].cpu().numpy()

            # Col 0: Decoder 1 reconstruction
            axes[c, 0].plot(t, orig_c, alpha=0.5, label="True Target")
            axes[c, 0].plot(t, recon_dec1_c, alpha=0.8, linestyle="--",
                            color="orange", label="Dec1 Recon")
            axes[c, 0].set_ylabel(f"Channel {c + 1}", fontsize=14)
            if c == 0:
                axes[c, 0].set_title(metric_titles[0], fontsize=14)
            axes[c, 0].legend()

            # Col 1: True high-freq residual
            axes[c, 1].plot(t, res_c, color="red", alpha=0.6)
            if c == 0:
                axes[c, 1].set_title(metric_titles[1], fontsize=14)

            # Col 2: CVAE jitter vs true residual
            axes[c, 2].plot(t, res_c, color="red", alpha=0.2, label="True Res")
            axes[c, 2].plot(t, synth_jitter_c, color="green", alpha=0.7,
                            label="CVAE Jitter")
            if c == 0:
                axes[c, 2].set_title(metric_titles[2], fontsize=14)
            axes[c, 2].legend()

            # Col 3: Final superposition
            axes[c, 3].plot(t, orig_c, alpha=0.3, label="True Tr")
            axes[c, 3].plot(t, synth_final_c, color="purple", linestyle="--",
                            alpha=0.8, label="Synth")
            if c == 0:
                axes[c, 3].set_title(metric_titles[3], fontsize=14)
            axes[c, 3].legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(outdir, f"mafaulda_evaluation_imbalance_label{idx_label}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot Phase 1 evaluation for MaFaulDa imbalance classes only."
    )
    parser.add_argument(
        "--labels", nargs="+", type=int, default=DEFAULT_TARGET_LABELS,
        metavar="LABEL",
        help=(
            "Label IDs to evaluate (default: [0, 35]). "
            "Imbalance IDs: 35=6g, 36=10g, 37=15g, 38=20g, 39=25g, 40=30g, 41=35g"
        ),
    )
    parser.add_argument(
        "--outdir", type=str, default="results-comparison",
        help="Directory to save the evaluation plots (default: results-comparison).",
    )
    parser.add_argument(
        "--num-samples", type=int, default=5000,
        help="Max dataset samples to load (increase if target labels not found).",
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"[Device] {device}")

    # Load frozen checkpoints
    ts_jepa, decoder1, decoder2, seq_length = load_models(device)

    # Build validation data loader (no training needed)
    _, val_loader, _ = get_dataloaders(
        batch_size=32,
        num_samples=args.num_samples,
        val_split=0.2,
    )
    if val_loader is None:
        print("[ERROR] Could not build dataloader. Check DATA_DIR_PROCESSED in configs.py")
        sys.exit(1)

    print(f"\n[Eval] Searching for samples with labels: {args.labels}")
    samples = collect_samples(val_loader, args.labels)

    print(f"\n[Plot] Generating evaluation figures → {args.outdir}/")
    plot_evaluation(ts_jepa, decoder1, decoder2, samples, device, args.outdir)

    print("\nDone.")


if __name__ == "__main__":
    main()
