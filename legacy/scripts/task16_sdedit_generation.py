# pyright: reportUnusedImport=false, reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportMissingParameterType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportAny=false, reportPrivateImportUsage=false, reportUnusedCallResult=false, reportUnannotatedClassAttribute=false, reportImplicitOverride=false, reportAssignmentType=false, reportGeneralTypeIssues=false, reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false, reportReturnType=false, reportMissingTypeArgument=false, reportDeprecated=false

"""
Task 16 - Physics-guided SDEdit counterfactual generation on MaFaulDa 16Hz.

Uses the frozen Phase 0/1/2 checkpoints, generates 40 fault counterfactuals per
class from healthy z_macro seeds, evaluates them, writes evidence artifacts, and
appends a short learning note.
"""

from __future__ import annotations

import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal as scipy_signal
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import rbf_kernel

REPO = r"D:\WORK\GenAI\GenAI-IES-IRAI-2026"
sys.path.insert(0, REPO)

from src.models.ts_jepa import TSJEPA
from src.models.latent_diffusion import DDPMScheduler, LatentDiffusionMLP
from src.models.decoder1 import Decoder1
from src.models.decoder2_cvae import Decoder2CVAE
from src.models.oracles import PriorWorkOracle


BASE = os.path.join(REPO, "data", "processed-mafaulda", "16hz")
RESULTS = os.path.join(REPO, "results")
EVIDENCE = os.path.join(REPO, ".sisyphus", "evidence")
ASSETS = os.path.join(REPO, "assets")
NOTEPAD = os.path.join(REPO, ".sisyphus", "notepads", "irai-pipeline-results", "learnings.md")

Y_FILES = {
    0: os.path.join(BASE, "Y_normal_trainingset.pth"),
    1: os.path.join(BASE, "Y_imbalance_fault_20g_trainingset.pth"),
    2: os.path.join(BASE, "Y_vertical_misalignment_fault_1.27mm_trainingset.pth"),
    3: os.path.join(BASE, "Y_overhang_ball_fault_20g_trainingset.pth"),
}
X_FILES = {k: v.replace("Y_", "X_") for k, v in Y_FILES.items()}

CLASS_NAMES = {
    0: "Normal",
    1: "Imbalance",
    2: "Vert.Mis.",
    3: "Overhang",
}

FAULT_CLASSES = [1, 2, 3]
N_PER_CLASS = 40
DEFAULT_T_START = 400
DEFAULT_GUIDANCE_SCALE = 1.0
TIMESTEP_STRIDE = 5
X0_CLAMP = 3.5
RBF_SIGMA = 1.0
SEED = 42


@dataclass
class RealClassData:
    y_norm: torch.Tensor
    z_macro: torch.Tensor
    omega: torch.Tensor
    oracle_embed_mean: torch.Tensor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs() -> None:
    os.makedirs(EVIDENCE, exist_ok=True)
    os.makedirs(ASSETS, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)


def load_state_dict_flexible(model: torch.nn.Module, path: str, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_models(device: torch.device):
    tsjepa = load_state_dict_flexible(TSJEPA(in_channels=4), os.path.join(RESULTS, "ts_jepa.pth"), device)
    tsjepa.requires_grad_(False)

    ldm = load_state_dict_flexible(LatentDiffusionMLP(z_dim=128), os.path.join(RESULTS, "ldm.pth"), device)

    decoder1 = load_state_dict_flexible(
        Decoder1(d_model=128, seq_length=3014, out_channels=4),
        os.path.join(RESULTS, "decoder1.pth"),
        device,
    )
    decoder1.requires_grad_(False)

    decoder2 = load_state_dict_flexible(
        Decoder2CVAE(latent_dim=64, context_dim=128, num_classes=4, seq_length=3014),
        os.path.join(RESULTS, "decoder2.pth"),
        device,
    )
    decoder2.requires_grad_(False)

    oracle = PriorWorkOracle().to(device)
    oracle.eval()
    oracle.requires_grad_(False)

    return tsjepa, ldm, decoder1, decoder2, oracle


def normalize_y(y_raw: torch.Tensor, y_min: torch.Tensor, y_max: torch.Tensor) -> torch.Tensor:
    return (y_raw - y_min[None, :, None]) / (y_max - y_min + 1e-8)[None, :, None]


def omega_from_x(x: torch.Tensor) -> torch.Tensor:
    return x[:, 0, 8].float()


def compute_real_class_data(
    tsjepa: TSJEPA,
    oracle: PriorWorkOracle,
    y_min: torch.Tensor,
    y_max: torch.Tensor,
    device: torch.device,
) -> Tuple[Dict[int, RealClassData], torch.Tensor, torch.Tensor]:
    class_data: Dict[int, RealClassData] = {}
    real_z_parts = []
    real_label_parts = []

    for cls_idx in range(4):
        y_raw = torch.load(Y_FILES[cls_idx], map_location="cpu", weights_only=False).permute(0, 2, 1).float()
        x_raw = torch.load(X_FILES[cls_idx], map_location="cpu", weights_only=False).float()
        omega = omega_from_x(x_raw)
        y_norm = normalize_y(y_raw, y_min.cpu(), y_max.cpu())

        with torch.no_grad():
            z = tsjepa.get_z_macro(y_norm.to(device)).cpu()

        oracle_embeds = []
        batch_size = 8
        with torch.no_grad():
            for start in range(0, y_norm.shape[0], batch_size):
                end = min(start + batch_size, y_norm.shape[0])
                oracle_embeds.append(
                    oracle(
                        y_norm[start:end].to(device),
                        omega=omega[start:end].to(device),
                    ).cpu()
                )
        oracle_embeds = torch.cat(oracle_embeds, dim=0)
        oracle_embed_mean = oracle_embeds.mean(dim=0)
        oracle.set_target_distribution(cls_idx, oracle_embed_mean.to(device))

        class_data[cls_idx] = RealClassData(
            y_norm=y_norm,
            z_macro=z,
            omega=omega,
            oracle_embed_mean=oracle_embed_mean,
        )
        real_z_parts.append(z)
        real_label_parts.append(torch.full((z.shape[0],), cls_idx, dtype=torch.long))

    real_z = torch.cat(real_z_parts, dim=0)
    real_labels = torch.cat(real_label_parts, dim=0)
    return class_data, real_z, real_labels


def make_scheduler(device: torch.device) -> DDPMScheduler:
    return DDPMScheduler(num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02, device=str(device))


def ddim_step(
    scheduler: DDPMScheduler,
    z_t: torch.Tensor,
    eps_pred: torch.Tensor,
    t: int,
    prev_t: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    alpha_bar_t = scheduler.alphas_cumprod[t].to(z_t.device)
    sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)

    x0_pred = (z_t - sqrt_one_minus_alpha_bar_t * eps_pred) / sqrt_alpha_bar_t.clamp_min(1e-8)
    x0_pred = torch.clamp(x0_pred, -X0_CLAMP, X0_CLAMP)

    if prev_t < 0:
        return x0_pred, x0_pred

    alpha_bar_prev = scheduler.alphas_cumprod[prev_t].to(z_t.device)
    z_prev = torch.sqrt(alpha_bar_prev) * x0_pred + torch.sqrt(1.0 - alpha_bar_prev) * eps_pred
    return z_prev, x0_pred


def oracle_guidance_gradient(
    z_t: torch.Tensor,
    target_class: int,
    omega_value: float,
    decoder1: Decoder1,
    oracle: PriorWorkOracle,
) -> Tuple[torch.Tensor, float]:
    z_in = z_t.detach().clone().requires_grad_(True)
    pred_trace = decoder1(z_in)
    omega_tensor = torch.tensor([omega_value], device=z_in.device, dtype=pred_trace.dtype)
    pred_embed = oracle(pred_trace, omega=omega_tensor)
    target_embed = oracle.get_target_distribution(target_class).unsqueeze(0).to(z_in.device)
    loss = F.mse_loss(pred_embed, target_embed)
    grad = torch.autograd.grad(loss, z_in, retain_graph=False, create_graph=False)[0]
    grad = torch.nan_to_num(grad)
    grad_norm = grad.norm(dim=1, keepdim=True).clamp_min(1e-6)
    grad = grad / grad_norm
    return grad.detach(), float(loss.item())


def mean_target_gradient(z_t: torch.Tensor, target_mean: torch.Tensor) -> Tuple[torch.Tensor, float]:
    diff = z_t - target_mean.to(z_t.device)
    loss = (diff.pow(2).mean()).item()
    grad = 2.0 * diff / diff.shape[1]
    grad = grad / grad.norm(dim=1, keepdim=True).clamp_min(1e-6)
    return grad.detach(), float(loss)


def probe_oracle_guidance(
    healthy_z: torch.Tensor,
    target_class: int,
    omega_value: float,
    decoder1: Decoder1,
    oracle: PriorWorkOracle,
) -> bool:
    try:
        grad, loss = oracle_guidance_gradient(healthy_z, target_class, omega_value, decoder1, oracle)
        return torch.isfinite(grad).all().item() and math.isfinite(loss)
    except Exception:
        return False


def run_sdedit(
    z_start: torch.Tensor,
    target_class: int,
    omega_value: float,
    ldm: LatentDiffusionMLP,
    decoder1: Decoder1,
    oracle: PriorWorkOracle,
    scheduler: DDPMScheduler,
    target_mean_z: torch.Tensor,
    t_start: int,
    guidance_scale: float,
    use_oracle_guidance: bool,
    store_trajectory: bool,
) -> Tuple[torch.Tensor, List[torch.Tensor], List[float]]:
    noise = torch.randn_like(z_start)
    z_t = scheduler.add_noise(z_start, noise, torch.tensor([t_start], device=z_start.device, dtype=torch.long))

    ddim_timesteps = list(range(0, t_start + 1, TIMESTEP_STRIDE))
    if ddim_timesteps[-1] != t_start:
        ddim_timesteps.append(t_start)

    trajectory: List[torch.Tensor] = [z_start.detach().cpu()] if store_trajectory else []
    penalties: List[float] = []

    for index in range(len(ddim_timesteps) - 1, -1, -1):
        t = ddim_timesteps[index]
        prev_t = ddim_timesteps[index - 1] if index > 0 else -1
        t_tensor = torch.tensor([t], device=z_start.device, dtype=torch.long)

        with torch.no_grad():
            eps_pred = ldm(z_t, t_tensor)

        z_prev, _ = ddim_step(scheduler, z_t, eps_pred, t, prev_t)

        if use_oracle_guidance:
            grad, penalty = oracle_guidance_gradient(z_t, target_class, omega_value, decoder1, oracle)
        else:
            grad, penalty = mean_target_gradient(z_t, target_mean_z)

        penalties.append(penalty)
        z_t = z_prev - guidance_scale * 0.05 * grad
        z_t = torch.clamp(z_t, -X0_CLAMP, X0_CLAMP)

        if store_trajectory:
            trajectory.append(z_t.detach().cpu())

    return z_t.detach(), trajectory, penalties


def decode_signal(z_final: torch.Tensor, target_class: int, decoder1: Decoder1, decoder2: Decoder2CVAE) -> torch.Tensor:
    label_tensor = torch.tensor([target_class], device=z_final.device, dtype=torch.long)
    with torch.no_grad():
        env = decoder1(z_final)
        jitter = decoder2.sample(z_final, label_tensor)
        return env + jitter


def mmd_score(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> float:
    gamma = 1.0 / (2.0 * sigma * sigma)
    xx = rbf_kernel(x, x, gamma=gamma).mean()
    yy = rbf_kernel(y, y, gamma=gamma).mean()
    xy = rbf_kernel(x, y, gamma=gamma).mean()
    return float(xx + yy - 2.0 * xy)


def save_trajectory_figure(real_z: torch.Tensor, real_labels: torch.Tensor, trajectories: Dict[int, List[torch.Tensor]]) -> None:
    import umap

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    real_umap = reducer.fit_transform(real_z.numpy())

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))
    for ax_idx, target_cls in enumerate(FAULT_CLASSES):
        ax = axes[ax_idx]
        for cls_idx in range(4):
            mask = real_labels.numpy() == cls_idx
            ax.scatter(
                real_umap[mask, 0],
                real_umap[mask, 1],
                c=colors[cls_idx],
                s=10,
                alpha=0.28,
                label=CLASS_NAMES[cls_idx] if ax_idx == 0 else "",
            )

        traj = trajectories[target_cls]
        traj_z = torch.cat(traj, dim=0).numpy()
        traj_umap = reducer.transform(traj_z)
        for step in range(len(traj_umap) - 1):
            alpha = 0.25 + 0.75 * (step / max(len(traj_umap) - 1, 1))
            ax.plot(traj_umap[step:step + 2, 0], traj_umap[step:step + 2, 1], color="black", alpha=alpha, linewidth=1.2)
        ax.scatter(traj_umap[0, 0], traj_umap[0, 1], c="gray", s=65, marker="o", zorder=5, label="Start" if ax_idx == 0 else "")
        ax.scatter(traj_umap[-1, 0], traj_umap[-1, 1], c="black", s=70, marker="*", zorder=5, label="End" if ax_idx == 0 else "")
        ax.set_title(f"Target: {CLASS_NAMES[target_cls]}", fontsize=10)
        ax.set_xlabel("UMAP-1", fontsize=9)
        ax.set_ylabel("UMAP-2", fontsize=9)

    axes[0].legend(fontsize=7, markerscale=1.1, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(EVIDENCE, "task-16-sdedit-trajectory.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig_pdf, axes_pdf = plt.subplots(3, 1, figsize=(3.5, 6.2))
    for ax_idx, target_cls in enumerate(FAULT_CLASSES):
        ax = axes_pdf[ax_idx]
        for cls_idx in range(4):
            mask = real_labels.numpy() == cls_idx
            ax.scatter(real_umap[mask, 0], real_umap[mask, 1], c=colors[cls_idx], s=5, alpha=0.22)
        traj = trajectories[target_cls]
        traj_umap = reducer.transform(torch.cat(traj, dim=0).numpy())
        ax.plot(traj_umap[:, 0], traj_umap[:, 1], color="black", linewidth=1.0)
        ax.scatter(traj_umap[0, 0], traj_umap[0, 1], c="gray", s=28, marker="o")
        ax.scatter(traj_umap[-1, 0], traj_umap[-1, 1], c="black", s=35, marker="*")
        ax.set_title(CLASS_NAMES[target_cls], fontsize=8)
        ax.set_xlabel("UMAP-1", fontsize=7)
        ax.set_ylabel("UMAP-2", fontsize=7)
        ax.tick_params(labelsize=7)
    fig_pdf.tight_layout()
    fig_pdf.savefig(os.path.join(ASSETS, "fig_sdedit_trajectory.pdf"), dpi=300, bbox_inches="tight", format="pdf")
    plt.close(fig_pdf)


def save_signal_comparison_figure(generated: Dict[int, Dict[str, torch.Tensor]], class_data: Dict[int, RealClassData]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.0))
    for col, target_cls in enumerate(FAULT_CLASSES):
        y_real = class_data[target_cls].y_norm[0, 0, :].numpy()
        y_gen = generated[target_cls]["signals"][0, 0, :].numpy()

        t = np.arange(200)
        ax_t = axes[0, col]
        ax_t.plot(t, y_real[:200], color="#1f77b4", linewidth=1.0, alpha=0.8, label="Real")
        ax_t.plot(t, y_gen[:200], color="#d62728", linewidth=1.0, alpha=0.8, linestyle="--", label="Generated")
        ax_t.set_title(CLASS_NAMES[target_cls], fontsize=10)
        ax_t.set_xlabel("Time step", fontsize=9)
        if col == 0:
            ax_t.set_ylabel("Amplitude (norm.)", fontsize=9)
            ax_t.legend(fontsize=8)

        f_real, psd_real = scipy_signal.welch(y_real, fs=16.0, nperseg=256)
        f_gen, psd_gen = scipy_signal.welch(y_gen, fs=16.0, nperseg=256)
        ax_f = axes[1, col]
        ax_f.semilogy(f_real, psd_real, color="#1f77b4", linewidth=1.0, alpha=0.8)
        ax_f.semilogy(f_gen, psd_gen, color="#d62728", linewidth=1.0, alpha=0.8, linestyle="--")
        ax_f.set_xlabel("Frequency (Hz)", fontsize=9)
        if col == 0:
            ax_f.set_ylabel("PSD", fontsize=9)

    fig.suptitle("Generated vs Real Fault Signals (channel 0)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(EVIDENCE, "task-16-signal-comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig_pdf, axes_pdf = plt.subplots(2, 3, figsize=(3.5, 3.6))
    for col, target_cls in enumerate(FAULT_CLASSES):
        y_real = class_data[target_cls].y_norm[0, 0, :].numpy()
        y_gen = generated[target_cls]["signals"][0, 0, :].numpy()
        t = np.arange(120)
        axes_pdf[0, col].plot(t, y_real[:120], color="#1f77b4", linewidth=0.8)
        axes_pdf[0, col].plot(t, y_gen[:120], color="#d62728", linewidth=0.8, linestyle="--")
        axes_pdf[0, col].set_title(CLASS_NAMES[target_cls], fontsize=7)
        axes_pdf[0, col].tick_params(labelsize=6)
        f_real, psd_real = scipy_signal.welch(y_real, fs=16.0, nperseg=256)
        f_gen, psd_gen = scipy_signal.welch(y_gen, fs=16.0, nperseg=256)
        axes_pdf[1, col].semilogy(f_real, psd_real, color="#1f77b4", linewidth=0.8)
        axes_pdf[1, col].semilogy(f_gen, psd_gen, color="#d62728", linewidth=0.8, linestyle="--")
        axes_pdf[1, col].tick_params(labelsize=6)
    fig_pdf.tight_layout()
    fig_pdf.savefig(os.path.join(ASSETS, "fig_signal_comparison.pdf"), dpi=300, bbox_inches="tight", format="pdf")
    plt.close(fig_pdf)


def append_learning_note(summary: str) -> None:
    with open(NOTEPAD, "a", encoding="utf-8", errors="replace") as f:
        f.write(summary)


def main() -> None:
    set_seed(SEED)
    ensure_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    meta = torch.load(os.path.join(RESULTS, "normalization_metadata.pth"), map_location="cpu", weights_only=False)
    y_min = meta["y_min"].float().to(device)
    y_max = meta["y_max"].float().to(device)

    tsjepa, ldm, decoder1, decoder2, oracle = load_models(device)
    scheduler = make_scheduler(device)

    class_data, real_z, real_labels = compute_real_class_data(tsjepa, oracle, y_min, y_max, device)

    healthy_probe = class_data[0].z_macro[0:1].to(device)
    oracle_guidance_worked = probe_oracle_guidance(
        healthy_probe,
        target_class=1,
        omega_value=float(class_data[1].omega.mean().item()),
        decoder1=decoder1,
        oracle=oracle,
    )
    guidance_mode = "oracle-vjp" if oracle_guidance_worked else "latent-mean-fallback"
    print(f"Guidance mode: {guidance_mode}")

    generated: Dict[int, Dict[str, torch.Tensor]] = {}
    trajectories: Dict[int, List[torch.Tensor]] = {}
    penalty_summaries: Dict[int, Tuple[float, float]] = {}

    for target_cls in FAULT_CLASSES:
        z_samples = []
        signal_samples = []
        penalty_first = None
        penalty_last = None
        target_mean_z = class_data[target_cls].z_macro.mean(dim=0, keepdim=True).to(device)
        omega_value = float(class_data[target_cls].omega.mean().item())

        for sample_idx in range(N_PER_CLASS):
            z_start = class_data[0].z_macro[sample_idx % class_data[0].z_macro.shape[0]].unsqueeze(0).to(device)
            z_final, step_trajectory, penalties = run_sdedit(
                z_start=z_start,
                target_class=target_cls,
                omega_value=omega_value,
                ldm=ldm,
                decoder1=decoder1,
                oracle=oracle,
                scheduler=scheduler,
                target_mean_z=target_mean_z,
                t_start=DEFAULT_T_START,
                guidance_scale=DEFAULT_GUIDANCE_SCALE,
                use_oracle_guidance=oracle_guidance_worked,
                store_trajectory=(sample_idx == 0),
            )

            signal = decode_signal(z_final, target_cls, decoder1, decoder2)
            z_samples.append(z_final.cpu())
            signal_samples.append(signal.cpu())

            if sample_idx == 0:
                trajectories[target_cls] = step_trajectory
                if penalties:
                    penalty_first = penalties[0]
                    penalty_last = penalties[-1]

        generated[target_cls] = {
            "z_macro": torch.cat(z_samples, dim=0),
            "signals": torch.cat(signal_samples, dim=0),
        }
        penalty_summaries[target_cls] = (float(penalty_first or 0.0), float(penalty_last or 0.0))

    gen_z = torch.cat([generated[c]["z_macro"] for c in FAULT_CLASSES], dim=0)
    gen_labels = torch.cat([torch.full((N_PER_CLASS,), c, dtype=torch.long) for c in FAULT_CLASSES])

    mixed_z = torch.cat([real_z, gen_z], dim=0).numpy()
    mixed_labels = torch.cat([real_labels, gen_labels], dim=0).numpy()
    pca = PCA(n_components=min(50, mixed_z.shape[0], mixed_z.shape[1]), random_state=42)
    mixed_pca = pca.fit_transform(mixed_z)
    silhouette_mixed = float(silhouette_score(mixed_pca, mixed_labels))

    mmd_per_class = {}
    for cls_idx in FAULT_CLASSES:
        mmd_per_class[cls_idx] = mmd_score(
            generated[cls_idx]["z_macro"].numpy(),
            class_data[cls_idx].z_macro.numpy(),
            sigma=RBF_SIGMA,
        )

    save_trajectory_figure(real_z, real_labels, trajectories)
    save_signal_comparison_figure(generated, class_data)

    torch.save(
        {
            "generated": {c: generated[c] for c in FAULT_CLASSES},
            "real_z": real_z,
            "real_labels": real_labels,
            "fault_classes": FAULT_CLASSES,
            "t_start": DEFAULT_T_START,
            "guidance_scale": DEFAULT_GUIDANCE_SCALE,
            "N_per_class": N_PER_CLASS,
            "guidance_mode": guidance_mode,
            "oracle_penalties": penalty_summaries,
        },
        os.path.join(RESULTS, "sdedit_counterfactuals.pth"),
    )

    lines = [
        "Task 16 - SDEdit Counterfactual Generation",
        "===========================================",
        f"N samples per class: {N_PER_CLASS}",
        f"t_start: {DEFAULT_T_START}",
        f"guidance_scale: {DEFAULT_GUIDANCE_SCALE}",
        f"device: {device}",
        "",
        "Generated z_macro stats:",
    ]
    for cls_idx in FAULT_CLASSES:
        z = generated[cls_idx]["z_macro"]
        lines.append(
            f"  Class {cls_idx} ({CLASS_NAMES[cls_idx]}): min={z.min().item():.4f}, max={z.max().item():.4f}, mean={z.mean().item():.4f}"
        )
    lines.extend(
        [
            "",
            f"Silhouette (mixed real+generated, PCA50): {silhouette_mixed:.4f}",
            "Per-class MMD (generated vs real):",
            f"  Class 1 (Imbalance): {mmd_per_class[1]:.4f}",
            f"  Class 2 (Vert.Mis.): {mmd_per_class[2]:.4f}",
            f"  Class 3 (Overhang):  {mmd_per_class[3]:.4f}",
            "",
            f"Oracle guidance: {guidance_mode}; penalty first->last class1={penalty_summaries[1][0]:.4f}->{penalty_summaries[1][1]:.4f}, class2={penalty_summaries[2][0]:.4f}->{penalty_summaries[2][1]:.4f}, class3={penalty_summaries[3][0]:.4f}->{penalty_summaries[3][1]:.4f}",
            "Figures: task-16-sdedit-trajectory.png, task-16-signal-comparison.png",
            "Paper figs: assets/fig_sdedit_trajectory.pdf, assets/fig_signal_comparison.pdf",
        ]
    )
    evidence_path = os.path.join(EVIDENCE, "task-16-sdedit-generation.txt")
    with open(evidence_path, "w", encoding="utf-8", errors="replace") as f:
        f.write("\n".join(lines) + "\n")

    note = (
        "\n## [2026-04-17] T16: SDEdit Generation\n"
        f"- t_start={DEFAULT_T_START}, guidance_scale={DEFAULT_GUIDANCE_SCALE}, N={N_PER_CLASS} per class\n"
        f"- Oracle guidance: {guidance_mode}\n"
        f"- Silhouette mixed: {silhouette_mixed:.4f}\n"
        f"- MMD per class: cls1={mmd_per_class[1]:.4f}, cls2={mmd_per_class[2]:.4f}, cls3={mmd_per_class[3]:.4f}\n"
        f"- Generated signal stats: normalized outputs stayed within [{min(generated[c]['signals'].min().item() for c in FAULT_CLASSES):.4f}, {max(generated[c]['signals'].max().item() for c in FAULT_CLASSES):.4f}] with stable decoded envelopes+jitter\n"
        f"- Counterfactuals saved: {os.path.join('results', 'sdedit_counterfactuals.pth')}\n"
        f"- Oracle penalty summaries: cls1 {penalty_summaries[1][0]:.4f}->{penalty_summaries[1][1]:.4f}, cls2 {penalty_summaries[2][0]:.4f}->{penalty_summaries[2][1]:.4f}, cls3 {penalty_summaries[3][0]:.4f}->{penalty_summaries[3][1]:.4f}\n"
    )
    append_learning_note(note)

    print("Saved:")
    print(f"  {os.path.join(RESULTS, 'sdedit_counterfactuals.pth')}")
    print(f"  {evidence_path}")
    print(f"  {os.path.join(EVIDENCE, 'task-16-sdedit-trajectory.png')}")
    print(f"  {os.path.join(EVIDENCE, 'task-16-signal-comparison.png')}")
    print(f"  {os.path.join(ASSETS, 'fig_sdedit_trajectory.pdf')}")
    print(f"  {os.path.join(ASSETS, 'fig_signal_comparison.pdf')}")


if __name__ == "__main__":
    main()
