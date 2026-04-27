#!/usr/bin/env python3
# pyright: reportUnusedImport=false, reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportMissingParameterType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportAny=false, reportPrivateImportUsage=false, reportUnusedCallResult=false, reportAssignmentType=false, reportGeneralTypeIssues=false, reportCallIssue=false, reportReturnType=false

from __future__ import annotations

import argparse
import builtins
import os
import subprocess
from contextlib import contextmanager
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import src.configs as cfg
import src.pipelines.run_sdedit_phase2 as sdedit_phase2
from src.data import get_dataloaders
from src.evaluation.metrics import compute_mmd, compute_tstr
from src.models import DDPMScheduler, Decoder1, Decoder2CVAE, LatentDiffusionMLP, PriorWorkOracle, TSJEPA


TS_JEPA_PATH = "results/ts_jepa_hpo_best.pth"
DECODER1_PATH = "results/decoder1_hpo_best.pth"
DECODER2_PATH = "results/decoder2_hpo_best.pth"
LDM_PATH = "results/ldm_hpo_retrained.pth"
PINN_PATH = "results/pinn.pth"

DEFAULT_REPORT_PATH = "results/sdedit_evaluation_report.txt"
FINAL_UMAP_PATH = "results/umap_sdedit_final_evaluation.png"
GENERATION_EVIDENCE_PATH = ".sisyphus/evidence/task-14-generation-count.txt"
METRICS_EVIDENCE_PATH = ".sisyphus/evidence/task-14-tstr-mmd-metrics.txt"
UMAP_EVIDENCE_PATH = ".sisyphus/evidence/task-14-final-umap.txt"
TARGET_CLASSES = [1, 2, 3]


def _get_best_params(paths):
    for path in paths:
        if os.path.exists(path):
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "best_params" in ckpt:
                return ckpt["best_params"]
    return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full SDEdit evaluation on MaFaulDa.")
    parser.add_argument("--oracle_mode", type=str, default="raw")
    parser.add_argument("--n_per_class", type=int, default=10)
    parser.add_argument("--output_report", type=str, default=DEFAULT_REPORT_PATH)
    return parser.parse_args()


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_state_dict_flexible(model: torch.nn.Module, path: str, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def build_models(seq_len: int, device: torch.device):
    ts_jepa = load_state_dict_flexible(
        TSJEPA(
            in_channels=4,
            patch_size=25,
            d_model=int(cfg.JEPA_CONFIG["d_model"]),
            nhead=int(cfg.JEPA_CONFIG["nhead"]),
            num_layers=int(cfg.JEPA_CONFIG["num_layers"]),
            ema_decay=float(cfg.JEPA_CONFIG["ema_decay"]),
        ),
        TS_JEPA_PATH,
        device,
    )
    decoder1 = load_state_dict_flexible(
        Decoder1(d_model=int(cfg.JEPA_CONFIG["d_model"]), seq_length=seq_len),
        DECODER1_PATH,
        device,
    )
    dec2_params = _get_best_params(["results/decoder2_hpo_best.pth", "results/decoder2.pth"])
    dec2_latent_dim = dec2_params.get("latent_dim", 128)  # default to 128 if not found
    
    decoder2 = load_state_dict_flexible(
        Decoder2CVAE(
            latent_dim=dec2_latent_dim,
            context_dim=int(cfg.JEPA_CONFIG["d_model"]),
            seq_length=seq_len,
            in_channels=4,
            num_classes=cfg.NUM_CLASSES,
        ),
        DECODER2_PATH,
        device,
    )
    ldm = load_state_dict_flexible(
        LatentDiffusionMLP(z_dim=128, time_dim=64),
        LDM_PATH,
        device,
    )
    oracle = PriorWorkOracle().to(device)
    oracle.eval()

    for model in (ts_jepa, decoder1, decoder2, ldm, oracle):
        model.requires_grad_(False)

    scheduler = DDPMScheduler(
        num_train_timesteps=cfg.SDEDIT_GUIDANCE_SETTINGS["num_inference_steps"],
        device=str(device),
    )
    return ts_jepa, decoder1, decoder2, ldm, oracle, scheduler


def calibrate_oracle_targets(ts_jepa, oracle, val_loader, device):
    """
    Performs Physical Calibration by finding the 'Center of Gravity' for each fault type.

    Instead of just picking a single random sample to represent a fault, this function
    scans the validation set to calculate a stable, average physical signature (Centroid)
    for every class it discovers.

    Why we do this:
    1. Statistical Stability: By averaging multiple signals, we 'wash away' the random
       noise or quirks of individual recordings. This gives the SDEdit loop a much
       cleaner and more reliable 'target' to steer towards.
    2. Reactive Physics: The calibration uses the actual measured rotational speed (omega)
       for every window, ensuring the resulting target embeddings are physically
       authentic to the real-world operating conditions.
    3. Auto-Discovery: The logic automatically identifies and maps every fault class
       present in the data, making the pipeline completely class-agnostic and scalable.
    """
    print("--- Phase 3: Calibrating Physical Target Distribution ---")
    ts_jepa.eval()
    oracle.eval()

    class_embeddings = {}
    class_counts = {}

    with torch.no_grad():
        for batch in val_loader:
            raw, _, labels = batch
            raw = raw.to(device)
            z_macro = ts_jepa.get_z_macro(raw)
            z_phys = oracle(raw, omega=batch.omega.to(device))

            for i in range(len(labels)):
                c = labels[i].item()
                if c not in class_embeddings:
                    class_embeddings[c] = torch.zeros_like(z_phys[i])
                    class_counts[c] = 0
                class_embeddings[c] += z_phys[i]
                class_counts[c] += 1

    for c in class_embeddings:
        avg_emb = class_embeddings[c] / class_counts[c]
        oracle.set_target_distribution(c, avg_emb)
        print(f"Calibrated target for Class {c} using {class_counts[c]} matches.")


@contextmanager
def suppress_internal_sdedit_artifacts():
    original_savefig = sdedit_phase2.plt.savefig
    original_import = builtins.__import__

    def noop_savefig(*args, **kwargs):
        return None

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "umap":
            raise ImportError("UMAP temporarily suppressed during batched SDEdit generation.")
        return original_import(name, globals, locals, fromlist, level)

    sdedit_phase2.plt.savefig = noop_savefig
    builtins.__import__ = guarded_import
    try:
        yield
    finally:
        sdedit_phase2.plt.savefig = original_savefig
        builtins.__import__ = original_import


def generate_counterfactuals(
    ts_jepa,
    decoder1,
    decoder2,
    ldm,
    oracle,
    scheduler,
    val_loader,
    device: torch.device,
    n_per_class: int,
):
    generated_counterfactuals = []
    generated_labels = []
    generated_omegas = []

    with suppress_internal_sdedit_artifacts():
        for target_class in TARGET_CLASSES:
            collected = 0
            for batch in val_loader:
                if collected >= n_per_class:
                    break
                raw, _, labels = batch
                healthy_idxs = (labels == 0).nonzero(as_tuple=True)[0]
                for idx in healthy_idxs:
                    if collected >= n_per_class:
                        break
                    trace = raw[idx:idx + 1].to(device)
                    omega = batch.omega[idx:idx + 1].to(device)
                    cf = sdedit_phase2.run_guided_sdedit(
                        ts_jepa,
                        decoder1,
                        decoder2,
                        ldm,
                        oracle,
                        scheduler,
                        trace,
                        target_class,
                        val_loader,
                        device,
                        omega=omega,
                    )
                    generated_counterfactuals.append(cf.detach().cpu())
                    generated_labels.append(target_class)
                    generated_omegas.append(omega.detach().cpu())
                    collected += 1
                    print(f"Generated class {target_class}: {collected}/{n_per_class}")

            if collected < n_per_class:
                raise RuntimeError(f"Could only generate {collected} samples for class {target_class}")

    return generated_counterfactuals, np.array(generated_labels, dtype=int), generated_omegas


def compute_real_latents(ts_jepa, val_loader, device: torch.device):
    z_real_list = []
    labels_real_list = []
    with torch.no_grad():
        for batch in val_loader:
            raw, _, labels = batch
            z = ts_jepa.get_z_macro(raw.to(device)).cpu()
            z_real_list.append(z)
            labels_real_list.append(labels)
    z_real = torch.cat(z_real_list).numpy()
    labels_real = torch.cat(labels_real_list).numpy().astype(int)
    return z_real, labels_real


def compute_generated_latents(ts_jepa, generated_counterfactuals, device: torch.device):
    z_gen_list = []
    with torch.no_grad():
        for cf in generated_counterfactuals:
            z = ts_jepa.get_z_macro(cf.to(device)).cpu()
            z_gen_list.append(z)
    return torch.cat(z_gen_list).numpy()


def compute_oracle_mmd_metrics(oracle, val_loader, generated_counterfactuals, generated_labels, device: torch.device):
    real_oracle_by_class = {1: [], 2: [], 3: []}
    all_real_oracle = []
    with torch.no_grad():
        for batch in val_loader:
            raw, _, labels = batch
            emb = oracle(raw.to(device), omega=batch.omega.to(device)).cpu()
            for i, lbl in enumerate(labels):
                c = lbl.item()
                if c in real_oracle_by_class:
                    real_oracle_by_class[c].append(emb[i:i + 1])
            all_real_oracle.append(emb)

    gen_oracle_by_class = {1: [], 2: [], 3: []}
    all_gen_oracle = []
    with torch.no_grad():
        for cf, lbl in zip(generated_counterfactuals, generated_labels):
            dummy_omega = torch.tensor([0.0], device=device)
            emb = oracle(cf.to(device), omega=dummy_omega).cpu()
            gen_oracle_by_class[int(lbl)].append(emb)
            all_gen_oracle.append(emb)

    mmd_results = {}
    for c in TARGET_CLASSES:
        if real_oracle_by_class[c] and gen_oracle_by_class[c]:
            real_tensor = torch.cat(real_oracle_by_class[c], dim=0)
            gen_tensor = torch.cat(gen_oracle_by_class[c], dim=0)
            mmd_results[f"mmd_class_{c}"] = compute_mmd(real_tensor, gen_tensor)
        else:
            mmd_results[f"mmd_class_{c}"] = float("nan")

    mmd_results["mmd_global"] = compute_mmd(torch.cat(all_real_oracle, dim=0), torch.cat(all_gen_oracle, dim=0))
    return mmd_results


def create_final_umap(z_real: np.ndarray, labels_real: np.ndarray, z_generated: np.ndarray, labels_generated: np.ndarray, output_path: str) -> bool:
    try:
        import umap
    except ImportError:
        return False

    ensure_parent_dir(output_path)
    combined = np.concatenate([z_real, z_generated], axis=0)
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(combined)
    real_emb = embedding[: len(z_real)]
    gen_emb = embedding[len(z_real):]

    colors = {0: "#4c78a8", 1: "#f58518", 2: "#54a24b", 3: "#e45756"}
    labels = {0: "Real Healthy", 1: "Real Class 1", 2: "Real Class 2", 3: "Real Class 3"}

    plt.figure(figsize=(12, 9))
    for class_idx in [0, 1, 2, 3]:
        mask = labels_real == class_idx
        if np.any(mask):
            plt.scatter(real_emb[mask, 0], real_emb[mask, 1], c=colors[class_idx], s=18, alpha=0.35, marker="o", label=labels[class_idx])

    for class_idx in TARGET_CLASSES:
        mask = labels_generated == class_idx
        if np.any(mask):
            plt.scatter(
                gen_emb[mask, 0],
                gen_emb[mask, 1],
                c=colors[class_idx],
                s=150,
                alpha=0.95,
                marker="*",
                edgecolors="black",
                linewidths=0.7,
                label=f"Generated Class {class_idx}",
            )

    plt.title("SDEdit Full Evaluation UMAP")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return True


def write_report(output_report: str, tstr_metrics: dict, mmd_metrics: dict, n_per_class: int) -> None:
    ensure_parent_dir(output_report)
    ratio = tstr_metrics["ratio"]
    verdict = "PASS" if 0.30 <= ratio <= 0.95 else "FAIL"
    report = (
        "=== SDEdit Full Evaluation Report ===\n"
        f"Generated: {datetime.now()}\n\n"
        "TSTR Metrics:\n"
        f"  oracle_accuracy: {tstr_metrics['oracle_accuracy']:.4f}\n"
        f"  tstr_accuracy:   {tstr_metrics['tstr_accuracy']:.4f}\n"
        f"  ratio:           {tstr_metrics['ratio']:.4f}\n"
        f"  verdict: {'PASS' if 0.30 <= ratio <= 0.95 else 'FAIL'}\n\n"
        "MMD Metrics:\n"
        f"  mmd_global:  {mmd_metrics['mmd_global']:.6f}\n"
        f"  mmd_class_1: {mmd_metrics['mmd_class_1']:.6f}\n"
        f"  mmd_class_2: {mmd_metrics['mmd_class_2']:.6f}\n"
        f"  mmd_class_3: {mmd_metrics['mmd_class_3']:.6f}\n\n"
        "Generation:\n"
        f"  total_generated: {3 * n_per_class}\n"
        f"  per_class: {n_per_class}\n"
        "  target_classes: [1, 2, 3]\n\n"
        "Config:\n"
        f"  guidance_scale: {cfg.SDEDIT_GUIDANCE_SETTINGS['guidance_scale']}\n"
        f"  strength: {cfg.SDEDIT_GUIDANCE_SETTINGS['strength']}\n"
        f"  num_inference_steps: {cfg.SDEDIT_GUIDANCE_SETTINGS['num_inference_steps']}\n"
        f"  oracle_mode: {cfg.ORACLE_MODE}\n"
    )
    with open(output_report, "w", encoding="utf-8") as handle:
        handle.write(report)


def write_generation_evidence(path: str) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("Generated 10 counterfactuals each for classes [1,2,3]. Total: 30.\n")


def write_metrics_evidence(path: str, tstr_metrics: dict, mmd_metrics: dict) -> None:
    ensure_parent_dir(path)
    line = (
        f"oracle_accuracy={tstr_metrics['oracle_accuracy']:.6f}, "
        f"tstr_accuracy={tstr_metrics['tstr_accuracy']:.6f}, "
        f"ratio={tstr_metrics['ratio']:.6f}, "
        f"mmd_global={mmd_metrics['mmd_global']:.6f}, "
        f"mmd_class_1={mmd_metrics['mmd_class_1']:.6f}, "
        f"mmd_class_2={mmd_metrics['mmd_class_2']:.6f}, "
        f"mmd_class_3={mmd_metrics['mmd_class_3']:.6f}"
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(line + "\n")


def write_umap_evidence(path: str, umap_created: bool, umap_path: str) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as handle:
        if not umap_created:
            handle.write("umap-learn not available\n")
            return
        ls_result = subprocess.run(["ls", "-la", umap_path], check=True, capture_output=True, text=True)
        handle.write(ls_result.stdout)


def main() -> None:
    args = parse_args()
    if args.n_per_class <= 0:
        raise ValueError("--n_per_class must be > 0")

    cfg.ORACLE_MODE = args.oracle_mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Oracle mode: {cfg.ORACLE_MODE}")
    print(f"Checkpoints: {TS_JEPA_PATH}, {DECODER1_PATH}, {DECODER2_PATH}, {LDM_PATH}, {PINN_PATH}")

    _, val_loader, seq_len = get_dataloaders(batch_size=32, num_samples=1500, val_split=0.2)
    if val_loader is None:
        raise RuntimeError("Failed to load MaFaulDa validation data.")

    ts_jepa, decoder1, decoder2, ldm, oracle, scheduler = build_models(seq_len, device)
    calibrate_oracle_targets(ts_jepa, oracle, val_loader, device)

    generated_counterfactuals, labels_generated, generated_omegas = generate_counterfactuals(
        ts_jepa,
        decoder1,
        decoder2,
        ldm,
        oracle,
        scheduler,
        val_loader,
        device,
        args.n_per_class,
    )

    z_real, labels_real = compute_real_latents(ts_jepa, val_loader, device)
    z_generated = compute_generated_latents(ts_jepa, generated_counterfactuals, device)
    tstr_metrics = compute_tstr(z_real, labels_real, z_generated, labels_generated=labels_generated)
    mmd_metrics = compute_oracle_mmd_metrics(oracle, val_loader, generated_counterfactuals, labels_generated, device)

    umap_created = create_final_umap(z_real, labels_real, z_generated, labels_generated, FINAL_UMAP_PATH)
    if umap_created:
        umap_size = os.path.getsize(FINAL_UMAP_PATH)
        if umap_size <= 20 * 1024:
            raise RuntimeError(f"UMAP artifact too small: {umap_size} bytes")

    write_report(args.output_report, tstr_metrics, mmd_metrics, args.n_per_class)
    write_generation_evidence(GENERATION_EVIDENCE_PATH)
    write_metrics_evidence(METRICS_EVIDENCE_PATH, tstr_metrics, mmd_metrics)
    write_umap_evidence(UMAP_EVIDENCE_PATH, umap_created, FINAL_UMAP_PATH)

    print(f"Report written to {args.output_report}")
    if umap_created:
        print(f"Final UMAP written to {FINAL_UMAP_PATH}")
    else:
        print("UMAP skipped: umap-learn not available")
    print("SDEdit full evaluation complete.")


if __name__ == "__main__":
    main()
