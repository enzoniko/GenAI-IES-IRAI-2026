# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportMissingParameterType=false, reportUnknownParameterType=false, reportUntypedBaseClass=false, reportUnannotatedClassAttribute=false, reportPrivateLocalImportUsage=false

import src.configs as cfg
import os
import sys
import types

cfg.RESULTS_DIR = "results-simple-synthetic"
cfg.NORM_METADATA_PATH = "results-simple-synthetic/normalization_metadata.pth"
cfg.PINN_MODEL_PATH = "results-simple-synthetic/pinn.pth"
cfg.JEPA_MODEL_PATH = "results-simple-synthetic/ts_jepa.pth"
cfg.DEC1_MODEL_PATH = "results-simple-synthetic/decoder1.pth"
cfg.DEC2_MODEL_PATH = "results-simple-synthetic/decoder2.pth"
cfg.LDM_MODEL_PATH = "results-simple-synthetic/ldm.pth"
cfg.NUM_CLASSES = 3
cfg.SEQ_LENGTH = 5000
cfg.SAMPLING_RATE = 50000
cfg.ORACLE_MODE = "raw"
cfg.JEPA_CONFIG = {**cfg.JEPA_CONFIG, "patch_size": 50}

import math
import traceback
import numpy as np
import torch
import torch.nn as nn


def _install_torchaudio_stub():
    if "torchaudio.functional" in sys.modules:
        return
    torchaudio_module = types.ModuleType("torchaudio")
    functional_module = types.ModuleType("torchaudio.functional")

    def _lfilter_stub(*args, **kwargs):
        raise RuntimeError("torchaudio is not installed; lfilter is unavailable in this environment.")

    setattr(functional_module, "lfilter", _lfilter_stub)
    setattr(torchaudio_module, "functional", functional_module)
    sys.modules["torchaudio"] = torchaudio_module
    sys.modules["torchaudio.functional"] = functional_module


_install_torchaudio_stub()

from src.data.synthetic_dataset import get_dataloaders
from src.evaluation.metrics import compute_mmd, compute_tstr
from src.models.decoder1 import Decoder1
from src.models.decoder2_cvae import Decoder2CVAE
from src.models.latent_diffusion import DDPMScheduler, LatentDiffusionMLP
from src.models.oracles import PriorWorkOracle
from src.models.ts_jepa import TSJEPA
from src.pipelines.run_sdedit_phase2 import run_guided_sdedit, train_latent_diffusion
from src.pipelines.train_phase1 import (
    extract_residuals_and_train_decoder2,
    train_phase1_decoder1,
    train_phase1_tsjepa,
)


RESULTS_DIR = cfg.RESULTS_DIR
EVIDENCE_DIR = ".sisyphus/evidence"
TSJEPA_EPOCHS = 15
LDM_EPOCHS = 15
BATCH_SIZE = 32
NUM_SAMPLES = 600


class SimpleSHOPINN(nn.Module):
    def __init__(self, sample_rate: int = 50000):
        super().__init__()
        self.m = 1.0
        self.c = 0.01
        self.k = (2.0 * math.pi * 50.0) ** 2
        self.f1 = 50.0
        self.f2 = 100.0
        self.A1 = 1.0
        self.A2 = 0.5
        self.dt = 1.0 / sample_rate

    def _integrate_trapezoidal(self, signal: torch.Tensor) -> torch.Tensor:
        integral = torch.zeros_like(signal)
        increments = 0.5 * (signal[..., 1:] + signal[..., :-1]) * self.dt
        integral[..., 1:] = torch.cumsum(increments, dim=-1)
        return integral

    def forward(self, acceleration: torch.Tensor):
        velocity = self._integrate_trapezoidal(acceleration)
        position = self._integrate_trapezoidal(velocity)
        t = torch.arange(
            acceleration.shape[-1], device=acceleration.device, dtype=acceleration.dtype
        ) * self.dt
        forcing = self.A1 * torch.sin(2.0 * math.pi * self.f1 * t) + self.A2 * torch.sin(
            2.0 * math.pi * self.f2 * t
        )
        forcing = forcing.view(1, 1, -1)
        residual = self.m * acceleration + self.c * velocity + self.k * position - forcing
        return residual, velocity, position


def _sample_ldm_latents(ldm, scheduler, n_samples, device, n_steps=100):
    ldm.eval()
    z_dim = int(cfg.JEPA_CONFIG["d_model"])
    stride = max(1, scheduler.num_train_timesteps // n_steps)
    t_steps = list(range(scheduler.num_train_timesteps - 1, -1, -stride))
    z_t = torch.randn(n_samples, z_dim, device=device)

    with torch.no_grad():
        for t in t_steps:
            t_tensor = torch.tensor([t] * n_samples, device=device).long()
            noise_pred = ldm(z_t.float(), t_tensor)
            z_t = scheduler.step(noise_pred, t, z_t)

    return z_t.cpu().numpy()


def compute_phase0_residual(train_loader, device):
    pinn = SimpleSHOPINN(sample_rate=cfg.SAMPLING_RATE).to(device)
    channel_sse = torch.zeros(4, dtype=torch.float64, device=device)
    total_points = 0

    pinn.eval()
    with torch.no_grad():
        for raw, _, labels in train_loader:
            mask = labels == 0
            if mask.sum() == 0:
                continue
            healthy_raw = raw[mask].to(device)
            residual, _, _ = pinn(healthy_raw)
            channel_sse += residual.double().pow(2).sum(dim=(0, 2))
            total_points += healthy_raw.shape[0] * healthy_raw.shape[-1]

    if total_points == 0:
        raise RuntimeError("No healthy samples found for Phase 0 residual computation.")

    per_channel_mse = channel_sse / total_points
    pinn_residual_mse = float(per_channel_mse.mean().item())
    torch.save({"pinn_residual_mse": pinn_residual_mse}, cfg.PINN_MODEL_PATH)
    return pinn_residual_mse, per_channel_mse.detach().cpu().numpy().tolist()


def compute_phase1_metrics(ts_jepa, decoder1, decoder2, val_loader, device):
    mse_loss = nn.MSELoss()
    z_list = []
    dec1_losses = []
    dec2_elbos = []

    ts_jepa.eval()
    decoder1.eval()
    decoder2.eval()

    with torch.no_grad():
        for raw, _, labels in val_loader:
            raw = raw.to(device)
            labels = labels.to(device)

            z_macro = ts_jepa.get_z_macro(raw)
            z_list.append(z_macro.cpu())

            recon_raw = decoder1(z_macro)
            dec1_losses.append(float(mse_loss(recon_raw, raw).item()))

            residual = raw - recon_raw
            recon_residual, mu, logvar = decoder2(residual, z_macro, labels)
            recon_loss = mse_loss(recon_residual, residual)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / raw.size(0)
            dec2_elbos.append(float((recon_loss + kl_loss).item()))

    all_z = torch.cat(z_list, dim=0)
    sigma_z = float(all_z.std(unbiased=False).item())
    dec1_mse = float(np.mean(dec1_losses))
    dec2_elbo = float(np.mean(dec2_elbos))
    return sigma_z, dec1_mse, dec2_elbo


def calibrate_oracle(oracle, val_loader, device):
    oracle.eval()
    with torch.no_grad():
        for class_idx in range(cfg.NUM_CLASSES):
            class_embs = []
            for raw, _, labels in val_loader:
                mask = labels == class_idx
                if mask.sum() == 0:
                    continue
                emb = oracle(raw[mask].to(device), omega=None)
                class_embs.append(emb)
            if class_embs:
                avg_emb = torch.cat(class_embs, dim=0).mean(0)
                oracle.set_target_distribution(class_idx, avg_emb)


def get_healthy_reference(val_loader, device):
    for raw, _, labels in val_loader:
        healthy_mask = labels == 0
        if healthy_mask.any():
            return raw[healthy_mask][0:1].to(device)
    raise RuntimeError("No healthy validation sample found for guided SDEdit.")


def compute_fairness_metrics(ts_jepa, ldm, scheduler, val_loader, device):
    ts_jepa.eval()
    z_real_list = []
    labels_real_list = []

    with torch.no_grad():
        for raw, _, labels in val_loader:
            z = ts_jepa.get_z_macro(raw.to(device))
            z_real_list.append(z.cpu().numpy())
            labels_real_list.append(labels.numpy())

    z_real = np.concatenate(z_real_list, axis=0)
    labels_real = np.concatenate(labels_real_list, axis=0)
    z_gen = _sample_ldm_latents(ldm, scheduler, n_samples=200, device=device, n_steps=100)
    tstr_result = compute_tstr(z_real, labels_real, z_gen)
    mmd_val = compute_mmd(z_real, z_gen)
    return z_real, labels_real, z_gen, tstr_result, mmd_val


def write_evidence(metrics, sdedit_logs):
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    evidence_path = os.path.join(EVIDENCE_DIR, "simple-synthetic-metrics.txt")
    lines = [
        "=== Simple Synthetic Pipeline Results ===",
        "Dataset: VibrationDataset (Healthy/Imbalance/Outer-Race)",
        "Physics: m*a + c*v + k*x = A1*sin(2pi*f1*t) + A2*sin(2pi*f2*t)",
        "         m=1.0, c=0.01, k=98696.0, f1=50Hz, f2=100Hz, A1=1.0, A2=0.5",
        "",
        "Phase 0 (SHO PINN physics residual):",
        f"  pinn_residual_mse = {metrics['pinn_residual_mse']:.6f}",
        "",
        "Phase 1 (TS-JEPA + Decoders):",
        f"  sigma_z    = {metrics['sigma_z']:.6f}  (z_macro std across val set)",
        f"  dec1_mse   = {metrics['dec1_mse']:.6f}  (Decoder1 val MSE)",
        f"  dec2_elbo  = {metrics['dec2_elbo']:.6f}  (Decoder2 val ELBO)",
        "",
        "Phase 2 (LDM + TSTR):",
        f"  oracle_accuracy = {metrics['oracle_accuracy']:.6f}",
        f"  tstr_accuracy   = {metrics['tstr_accuracy']:.6f}",
        f"  tstr_ratio      = {metrics['tstr_ratio']:.6f}  (TSTR / Oracle)",
        f"  mmd             = {metrics['mmd']:.6f}",
    ]
    if sdedit_logs:
        lines.extend(["", "Guided SDEdit status:", *[f"  {line}" for line in sdedit_logs]])
    with open(evidence_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return evidence_path


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(EVIDENCE_DIR, exist_ok=True)

    device = torch.device("cpu")
    print(f"Device: {device}")
    print(f"Torch threads: {torch.get_num_threads()}")
    print(f"Results dir: {RESULTS_DIR}")

    train_loader, val_loader = get_dataloaders(
        batch_size=BATCH_SIZE,
        num_samples=NUM_SAMPLES,
        seq_length=cfg.SEQ_LENGTH,
        sample_rate=cfg.SAMPLING_RATE,
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    print("\n=== Phase 0: Simple SHO PINN residual ===")
    pinn_residual_mse, per_channel_mse = compute_phase0_residual(train_loader, device)
    print(f"pinn_residual_mse = {pinn_residual_mse:.6f}")
    print(f"per_channel_residual_mse = {[round(x, 6) for x in per_channel_mse]}")

    print("\n=== Phase 1: TS-JEPA + Decoder1 + Decoder2 ===")
    ts_jepa = TSJEPA(in_channels=4, patch_size=50).to(device)
    decoder1 = Decoder1(out_channels=4, seq_length=cfg.SEQ_LENGTH).to(device)
    decoder2 = Decoder2CVAE(in_channels=4, seq_length=cfg.SEQ_LENGTH).to(device)

    ts_jepa = train_phase1_tsjepa(ts_jepa, train_loader, val_loader, TSJEPA_EPOCHS, device)
    decoder1 = train_phase1_decoder1(ts_jepa, decoder1, train_loader, val_loader, TSJEPA_EPOCHS, device)
    decoder2 = extract_residuals_and_train_decoder2(
        ts_jepa, decoder1, decoder2, train_loader, val_loader, TSJEPA_EPOCHS, device
    )

    torch.save(ts_jepa.state_dict(), cfg.JEPA_MODEL_PATH)
    torch.save(decoder1.state_dict(), cfg.DEC1_MODEL_PATH)
    torch.save(decoder2.state_dict(), cfg.DEC2_MODEL_PATH)

    sigma_z, dec1_mse, dec2_elbo = compute_phase1_metrics(
        ts_jepa, decoder1, decoder2, val_loader, device
    )
    print(f"sigma_z = {sigma_z:.6f}")
    print(f"dec1_mse = {dec1_mse:.6f}")
    print(f"dec2_elbo = {dec2_elbo:.6f}")

    print("\n=== Phase 2: Latent diffusion + guided SDEdit ===")
    ldm = LatentDiffusionMLP(z_dim=int(cfg.JEPA_CONFIG["d_model"]), time_dim=64).to(device)
    scheduler = DDPMScheduler(
        num_train_timesteps=int(cfg.SDEDIT_GUIDANCE_SETTINGS["num_inference_steps"]),
        device=device,
    )
    for param in ts_jepa.parameters():
        param.requires_grad = False
    ldm = train_latent_diffusion(ts_jepa, ldm, scheduler, train_loader, val_loader, device, epochs=LDM_EPOCHS)
    torch.save(ldm.state_dict(), cfg.LDM_MODEL_PATH)

    oracle = PriorWorkOracle().to(device)
    calibrate_oracle(oracle, val_loader, device)
    healthy_raw = get_healthy_reference(val_loader, device)

    sdedit_logs = []
    for class_idx in [1, 2]:
        try:
            _ = run_guided_sdedit(
                ts_jepa,
                decoder1,
                decoder2,
                ldm,
                oracle,
                scheduler,
                healthy_raw,
                class_idx,
                val_loader,
                device,
                omega=None,
                num_inference_steps=10,
                guidance_scale=0.4203,
                strength=0.1062,
            )
            status = f"class {class_idx}: success"
        except Exception as exc:
            status = f"class {class_idx}: failed - {exc}"
            print(status)
            print(traceback.format_exc())
        sdedit_logs.append(status)

    _, _, _, tstr_result, mmd_val = compute_fairness_metrics(ts_jepa, ldm, scheduler, val_loader, device)
    oracle_accuracy = float(tstr_result["oracle_accuracy"])
    tstr_accuracy = float(tstr_result["tstr_accuracy"])
    tstr_ratio = float(tstr_result["ratio"])
    mmd = float(mmd_val)

    print("\n=== Metrics Summary ===")
    print(f"sigma_z = {sigma_z:.6f}")
    print(f"Dec1 MSE = {dec1_mse:.6f}")
    print(f"Dec2 ELBO = {dec2_elbo:.6f}")
    print(f"Oracle acc = {oracle_accuracy:.6f}")
    print(f"TSTR acc = {tstr_accuracy:.6f}")
    print(f"TSTR/Oracle ratio = {tstr_ratio:.6f}")
    print(f"MMD = {mmd:.6f}")

    metrics = {
        "pinn_residual_mse": pinn_residual_mse,
        "sigma_z": sigma_z,
        "dec1_mse": dec1_mse,
        "dec2_elbo": dec2_elbo,
        "oracle_accuracy": oracle_accuracy,
        "tstr_accuracy": tstr_accuracy,
        "tstr_ratio": tstr_ratio,
        "mmd": mmd,
    }
    evidence_path = write_evidence(metrics, sdedit_logs)

    print("\n=== Artifacts ===")
    for artifact in [
        cfg.PINN_MODEL_PATH,
        cfg.JEPA_MODEL_PATH,
        cfg.DEC1_MODEL_PATH,
        cfg.DEC2_MODEL_PATH,
        cfg.LDM_MODEL_PATH,
        evidence_path,
    ]:
        print(artifact)


if __name__ == "__main__":
    main()
