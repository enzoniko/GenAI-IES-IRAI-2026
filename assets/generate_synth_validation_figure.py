#!/usr/bin/env python3
"""
generate_synth_validation_figure.py
Produces assets/synth_validation_overview.png — 2x2 IEEE validation figure.
All panels use synthetic damped-oscillator data (NOT MaFaulDa).

Data format: Y_*.pth files are (N, T, C)=(N, 3014, 4), already normalized to ~[-2,2].
Models expect (N, C, T) = (N, 4, 3014) — transpose before passing.
JEPA checkpoint uses patch_size=50 (overrides src/configs.py which says 25).

Panels:
  (a) Dec1 envelope reconstruction on synthetic healthy
  (b) Dec2 CVAE high-frequency jitter on synthetic healthy
  (c) Synthetic healthy vs real stiffness-reduction fault vs model reconstruction
  (d) SDEdit latent trajectory UMAP (embeds existing PNG)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import importlib.util

import src.configs as cfg


def load_class(module_name, file_name, class_name):
    module_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'src', 'models', file_name
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


TSJEPA = load_class('ts_jepa', 'ts_jepa.py', 'TSJEPA')
Decoder1 = load_class('decoder1', 'decoder1.py', 'Decoder1')
Decoder2CVAE = load_class('decoder2_cvae', 'decoder2_cvae.py', 'Decoder2CVAE')

DEVICE = torch.device('cpu')
torch.manual_seed(42)

# ── Load models (patch_size=50 per saved checkpoint — do NOT use configs.JEPA_CONFIG which says 25)
jepa = TSJEPA(patch_size=50).to(DEVICE)
jepa.load_state_dict(torch.load(cfg.JEPA_MODEL_PATH, map_location=DEVICE, weights_only=True))
jepa.eval()

dec1 = Decoder1().to(DEVICE)
dec1.load_state_dict(torch.load(cfg.DEC1_MODEL_PATH, map_location=DEVICE, weights_only=True))
dec1.eval()

dec2 = Decoder2CVAE().to(DEVICE)
dec2.load_state_dict(torch.load(cfg.DEC2_MODEL_PATH, map_location=DEVICE, weights_only=True))
dec2.eval()

# ── Load synthetic data
# Y_* files: shape (N, T=3014, C=4), already normalized — NO extra normalization needed.
# Transpose to (N, C=4, T=3014) for model input.
y_healthy_raw = torch.load(
    'data/processed-synthetic/Y_healthy_testset.pth',
    map_location=DEVICE, weights_only=True
).float()
y_healthy = y_healthy_raw.transpose(1, 2)  # (10, 4, 3014)
y_healthy_1 = y_healthy[:1]               # (1, 4, 3014)

y_stiff_raw = torch.load(
    'data/processed-synthetic/Y_stiffness_reduction_testset.pth',
    map_location=DEVICE, weights_only=True
).float()
y_stiff = y_stiff_raw.transpose(1, 2)     # (10, 4, 3014)
y_stiff_1 = y_stiff[:1]                   # (1, 4, 3014)

# ── Model inference
with torch.no_grad():
    # Encode healthy trace
    z_healthy = jepa.get_z_macro(y_healthy_1)          # (1, 128)
    env_healthy = dec1(z_healthy)                        # (1, 4, 3014)
    residual_healthy = y_healthy_1 - env_healthy         # (1, 4, 3014)
    label_0 = torch.zeros(1, dtype=torch.long)
    jitter_cvae = dec2.sample(z_healthy, label_0)        # (1, 4, 3014)

    # Encode stiffness fault — reconstruct to show pipeline output
    z_stiff = jepa.get_z_macro(y_stiff_1)               # (1, 128)
    env_stiff = dec1(z_stiff)                            # (1, 4, 3014)
    jitter_stiff = dec2.sample(z_stiff, label_0)         # (1, 4, 3014)
    recon_stiff = env_stiff + jitter_stiff               # (1, 4, 3014)

# ── Plot
CH = 0    # channel to display
STEP = 3  # subsample every 3rd point for clean rendering
T = np.arange(0, 3014, STEP)


def to_np(tensor, ch=CH, step=STEP):
    """Extract channel ch from (1, C, T) tensor and subsample."""
    return tensor[0, ch].numpy()[::step]


fig, axes = plt.subplots(2, 2, figsize=(12, 5))
fig.subplots_adjust(wspace=0.35, hspace=0.55)

# ── (a) Dec1 Envelope Reconstruction
ax = axes[0, 0]
ax.plot(T, to_np(y_healthy_1), lw=1.0, color='#2c7bb6', label='Healthy trace', alpha=0.85)
ax.plot(T, to_np(env_healthy), lw=1.2, color='#d7191c', linestyle='--',
        label='Dec$_1$ envelope', alpha=0.92)
ax.set_title('(a) Dec$_1$ Envelope Reconstruction', fontsize=8, fontweight='bold', pad=4)
ax.set_xlabel('Sample index', fontsize=7)
ax.set_ylabel('Norm. amplitude', fontsize=7)
ax.tick_params(labelsize=6)
ax.legend(fontsize=6, loc='upper right', framealpha=0.7)

# ── (b) Dec2 High-Frequency Jitter
ax = axes[0, 1]
ax.plot(T, to_np(residual_healthy), lw=1.0, color='#555555', label='Real residual', alpha=0.85)
ax.plot(T, to_np(jitter_cvae), lw=1.2, color='#1a9641', linestyle='--',
        label='CVAE sample', alpha=0.92)
ax.set_title('(b) Dec$_2$ High-Frequency Jitter', fontsize=8, fontweight='bold', pad=4)
ax.set_xlabel('Sample index', fontsize=7)
ax.set_ylabel('Norm. amplitude', fontsize=7)
ax.tick_params(labelsize=6)
ax.legend(fontsize=6, loc='upper right', framealpha=0.7)

# ── (c) Healthy vs Real Stiffness Fault vs Model Reconstruction
ax = axes[1, 0]
ax.plot(T, to_np(y_healthy_1), lw=1.0, color='#2c7bb6', label='Healthy', alpha=0.85)
ax.plot(T, to_np(y_stiff_1), lw=1.0, color='#1a9641', label='Real stiffness fault', alpha=0.85)
ax.plot(T, to_np(recon_stiff), lw=1.2, color='#d7191c', linestyle='--',
        label='Model reconstruction', alpha=0.92)
ax.set_title('(c) Synthetic: Healthy / Fault / Reconstruction', fontsize=8, fontweight='bold', pad=4)
ax.set_xlabel('Sample index', fontsize=7)
ax.set_ylabel('Norm. amplitude', fontsize=7)
ax.tick_params(labelsize=6)
ax.legend(fontsize=6, loc='upper right', framealpha=0.7)

# ── (d) SDEdit UMAP Trajectory (embed existing PNG)
ax = axes[1, 1]
umap_src = 'assets/synth_umap_sdedit_trajectory_imbalance.png'
umap_img = mpimg.imread(umap_src)
ax.imshow(umap_img, aspect='auto')
ax.set_title('(d) SDEdit Latent Trajectory (UMAP)', fontsize=8, fontweight='bold', pad=4)
ax.axis('off')

# ── Save
out_path = 'assets/synth_validation_overview.png'
fig.savefig(out_path, dpi=180, bbox_inches='tight')
print(f"[OK] Saved: {out_path}")
plt.close()
