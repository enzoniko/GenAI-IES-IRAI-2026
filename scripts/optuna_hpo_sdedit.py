#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import optuna

from src.evaluation.hpo_utils import (
    create_or_load_study,
    get_data_loaders,
    log_trial_result,
)
from src.evaluation.metrics import compute_mmd
from src.models.ts_jepa import TSJEPA
from src.models.decoder1 import Decoder1
from src.models.decoder2_cvae import Decoder2CVAE
from src.models.latent_diffusion import LatentDiffusionMLP, DDPMScheduler
from src.models.oracles import PriorWorkOracle
from src.pipelines.run_sdedit_phase2 import run_guided_sdedit
import src.configs as cfg


def _freeze(model):
    for param in model.parameters():
        param.requires_grad = False
    return model


def _load_checkpoint(model, paths, device):
    for path in paths:
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=device, weights_only=False)
            state = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state, strict=False)
            print(f"Loaded {model.__class__.__name__} from {path}")
            return model
    print(f"WARNING: No checkpoint found for {model.__class__.__name__}, using random weights.")
    return model


def _get_best_params(paths):
    for path in paths:
        if os.path.exists(path):
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "best_params" in ckpt:
                print(f"Loaded best_params from {path}")
                return ckpt["best_params"]
    return {}


def _build_models(device):
    dec2_params = _get_best_params(["results/decoder2_hpo_best.pth", "results/decoder2.pth"])
    dec2_latent_dim = dec2_params.get("latent_dim", 128)  # default to 128 if not found
    
    ts_jepa = _freeze(_load_checkpoint(
        TSJEPA(
            d_model=cfg.JEPA_CONFIG["d_model"],
            in_channels=4,
            patch_size=cfg.JEPA_CONFIG["patch_size"],
            nhead=cfg.JEPA_CONFIG["nhead"],
            num_layers=cfg.JEPA_CONFIG["num_layers"],
            ema_decay=cfg.JEPA_CONFIG["ema_decay"],
        ).to(device),
        ["results/ts_jepa_hpo_best.pth", "results/ts_jepa.pth"],
        device,
    ))

    decoder1 = _freeze(_load_checkpoint(
        Decoder1(d_model=int(cfg.JEPA_CONFIG["d_model"]), seq_length=cfg.SEQ_LENGTH, out_channels=4).to(device),
        ["results/decoder1_hpo_best.pth", "results/decoder1.pth"],
        device,
    ))

    decoder2 = _freeze(_load_checkpoint(
        Decoder2CVAE(
            seq_length=cfg.SEQ_LENGTH,
            in_channels=4,
            context_dim=int(cfg.JEPA_CONFIG["d_model"]),
            latent_dim=dec2_latent_dim,
            num_classes=cfg.NUM_CLASSES,
        ).to(device),
        ["results/decoder2_hpo_best.pth", "results/decoder2.pth"],
        device,
    ))

    ldm = _freeze(_load_checkpoint(
        LatentDiffusionMLP(z_dim=int(cfg.JEPA_CONFIG["d_model"])).to(device),
        ["results/ldm_hpo_retrained.pth", "results/ldm.pth"],
        device,
    ))

    oracle = _freeze(_load_checkpoint(
        PriorWorkOracle(in_channels=4).to(device),
        [cfg.PINN_MODEL_PATH],
        device,
    ))

    scheduler = DDPMScheduler(num_train_timesteps=1000, device=str(device))
    return ts_jepa, decoder1, decoder2, ldm, oracle, scheduler


def _calibrate_oracle(oracle, ts_jepa, decoder1, val_loader, device):
    oracle.eval()
    with torch.no_grad():
        for class_idx in range(cfg.NUM_CLASSES):
            class_embs = []
            for batch in val_loader:
                raw   = batch[0].to(device)
                omega = batch.omega.to(device)
                label = batch[2]
                mask  = (label == class_idx)
                if mask.sum() == 0:
                    continue
                emb = oracle(raw[mask], omega=omega[mask])
                class_embs.append(emb)
            if class_embs:
                mean_emb = torch.cat(class_embs).mean(0)
                oracle.set_target_distribution(class_idx, mean_emb)


def _get_healthy_sample(val_loader, device):
    for batch in val_loader:
        raw   = batch[0]
        omega = batch.omega
        label = batch[2]
        mask  = (label == 0)
        if mask.sum() > 0:
            idx = mask.nonzero(as_tuple=True)[0][0]
            return raw[idx : idx + 1].to(device), float(omega[idx])
    return None, None


def _collect_real_oracle_embeddings(oracle, val_loader, device):
    oracle.eval()
    embs = []
    with torch.no_grad():
        for batch in val_loader:
            raw   = batch[0].to(device)
            omega = batch.omega.to(device)
            emb   = oracle(raw, omega=omega)
            embs.append(emb.cpu().numpy())
    return np.vstack(embs)


def make_objective(device):
    def objective(trial):
        guidance_scale      = trial.suggest_float("guidance_scale", 0.01, 5.0, log=True)
        strength            = trial.suggest_float("strength", 0.1, 0.9)
        num_inference_steps = trial.suggest_categorical("num_inference_steps", [100, 250, 500, 1000])

        ts_jepa, decoder1, decoder2, ldm, oracle, scheduler = _build_models(device)

        _, val_loader = get_data_loaders(32)
        if val_loader is None:
            raise optuna.TrialPruned()

        _calibrate_oracle(oracle, ts_jepa, decoder1, val_loader, device)

        healthy_trace, healthy_omega = _get_healthy_sample(val_loader, device)
        if healthy_trace is None:
            raise optuna.TrialPruned()

        final_counterfactual = run_guided_sdedit(
            ts_jepa, decoder1, decoder2, ldm, oracle, scheduler,
            healthy_trace,
            target_class_idx=1,
            val_loader=val_loader,
            device=device,
            omega=healthy_omega,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
        )

        with torch.no_grad():
            generated_emb = oracle(final_counterfactual, omega=healthy_omega)

        real_embs = _collect_real_oracle_embeddings(oracle, val_loader, device)
        generated_np = generated_emb.detach().cpu().numpy()
        mmd = compute_mmd(generated_np, real_embs)

        log_trial_result(None, trial, {"mmd": mmd})
        return float(mmd)

    return objective


def main():
    parser = argparse.ArgumentParser(description="SDEdit HPO")
    parser.add_argument("--n_trials",   type=int, default=20)
    parser.add_argument("--storage",    type=str, default="results/optuna_studies.db")
    parser.add_argument("--study_name", type=str, default="sdedit-hpo")
    parser.add_argument("--dry_run",    action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print("SDEdit HPO search space:")
        print("  guidance_scale: [0.01, 5.0] (log scale)")
        print("  strength: [0.1, 0.9]")
        print("  num_inference_steps: [100, 250, 500, 1000]")
        print("Objective: minimize MMD(oracle(generated), oracle(real_val))")
        print("Oracle calibration: run oracle on val_loader per class → set_target_distribution()")
        print("Requires checkpoints: ts_jepa, decoder1, decoder2, ldm, pinn (oracle)")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    study = create_or_load_study(args.study_name, "minimize", args.storage)
    study.optimize(make_objective(device), n_trials=args.n_trials)

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best MMD: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")


if __name__ == "__main__":
    main()
