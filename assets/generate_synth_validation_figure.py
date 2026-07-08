#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportAny=false
"""Generate the synthetic validation overview figure."""

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import torch

import src.configs as cfg


DEVICE = torch.device("cpu")
SEQ_LENGTH = cfg.SEQ_LENGTH
CHANNEL = 0
STEP = 3
OUT_PATH = ROOT / "assets" / "synth_validation_overview.png"
UMAP_PATH = ROOT / "assets" / "synth_umap_sdedit_trajectory_imbalance.png"
SYNTH_DIR = ROOT / "data" / "processed-synthetic"


def load_class(module_name: str, relative_path: str, class_name: str):
    module_path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


TSJEPA = load_class("ts_jepa_local", "src/models/ts_jepa.py", "TSJEPA")
Decoder1 = load_class("decoder1_local", "src/models/decoder1.py", "Decoder1")
Decoder2CVAE = load_class("decoder2_cvae_local", "src/models/decoder2_cvae.py", "Decoder2CVAE")


def load_models():
    jepa = TSJEPA(patch_size=50).to(DEVICE)
    jepa.load_state_dict(torch.load(cfg.JEPA_MODEL_PATH, map_location=DEVICE, weights_only=True))
    jepa.eval()

    dec1 = Decoder1(seq_length=SEQ_LENGTH).to(DEVICE)
    dec1.load_state_dict(torch.load(cfg.DEC1_MODEL_PATH, map_location=DEVICE, weights_only=True))
    dec1.eval()

    dec2 = Decoder2CVAE(seq_length=SEQ_LENGTH).to(DEVICE)
    dec2.load_state_dict(torch.load(cfg.DEC2_MODEL_PATH, map_location=DEVICE, weights_only=True))
    dec2.eval()

    return jepa, dec1, dec2


def load_synthetic_trace(name: str) -> torch.Tensor:
    tensor = torch.load(SYNTH_DIR / name, map_location=DEVICE, weights_only=True).float()
    if tensor.ndim != 3:
        raise ValueError(f"Expected 3D tensor for {name}, got shape {tuple(tensor.shape)}")
    tensor = tensor.transpose(1, 2).contiguous()
    if tensor.shape[1] != 4 or tensor.shape[2] != SEQ_LENGTH:
        raise ValueError(f"Unexpected transposed shape for {name}: {tuple(tensor.shape)}")
    return tensor


def to_numpy(trace: torch.Tensor) -> np.ndarray:
    return trace.detach().cpu().numpy()


def style_axis(ax, title: str) -> None:
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("Sample index", fontsize=7)
    ax.set_ylabel("Amplitude", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=6, framealpha=0.75)


def main() -> None:
    torch.manual_seed(42)

    y_healthy = load_synthetic_trace("Y_healthy_testset.pth")
    y_stiff = load_synthetic_trace("Y_stiffness_reduction_testset.pth")
    y_healthy_1 = y_healthy[:1]
    y_stiff_1 = y_stiff[:1]

    jepa, dec1, dec2 = load_models()

    with torch.no_grad():
        z_macro = jepa.get_z_macro(y_healthy_1)
        env = dec1(z_macro)
        residual = y_healthy_1 - env
        label_healthy = torch.zeros(1, dtype=torch.long, device=DEVICE)
        label_stiff = torch.ones(1, dtype=torch.long, device=DEVICE)
        jitter_cvae = dec2.sample(z_macro, label_healthy)
        gen_jitter = dec2.sample(z_macro, label_stiff)
        gen_trace = env + gen_jitter

    t = np.arange(SEQ_LENGTH)[::STEP]
    healthy_np = to_numpy(y_healthy_1[0, CHANNEL])[::STEP]
    stiff_np = to_numpy(y_stiff_1[0, CHANNEL])[::STEP]
    env_np = to_numpy(env[0, CHANNEL])[::STEP]
    residual_np = to_numpy(residual[0, CHANNEL])[::STEP]
    jitter_np = to_numpy(jitter_cvae[0, CHANNEL])[::STEP]
    gen_np = to_numpy(gen_trace[0, CHANNEL])[::STEP]

    fig, axes = plt.subplots(2, 2, figsize=(12, 5))
    fig.subplots_adjust(wspace=0.35, hspace=0.52)

    ax = axes[0, 0]
    ax.plot(t, healthy_np, color="#1f77b4", lw=1.0, label="Synthetic healthy")
    ax.plot(t, env_np, color="#d62728", lw=1.0, ls="--", label="Dec₁ envelope")
    style_axis(ax, "(a) Dec₁ Envelope Reconstruction")

    ax = axes[0, 1]
    ax.plot(t, residual_np, color="#7f7f7f", lw=1.0, label="Real residual")
    ax.plot(t, jitter_np, color="#1f77b4", lw=1.0, ls="--", label="CVAE sample")
    style_axis(ax, "(b) Dec₂ High-Frequency Jitter")

    ax = axes[1, 0]
    ax.plot(t, healthy_np, color="#1f77b4", lw=1.0, label="Synthetic healthy")
    ax.plot(t, stiff_np, color="#ff7f0e", lw=1.0, label="Stiffness-reduction fault")
    ax.plot(t, gen_np, color="#d62728", lw=1.0, ls="--", label="CVAE-guided synthesis")
    style_axis(ax, "(c) Synthetic Fault: Healthy vs. Real vs. Synthesis")

    ax = axes[1, 1]
    ax.imshow(mpimg.imread(UMAP_PATH), aspect="auto")
    ax.set_title("(d) SDEdit Latent Trajectory (UMAP)", fontsize=8)
    ax.axis("off")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
