#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn as nn
import optuna

from src.evaluation.hpo_utils import (
    create_or_load_study,
    get_data_loaders,
    log_trial_result,
)
from src.models.ts_jepa import TSJEPA
from src.models.decoder1 import Decoder1
from src.models.decoder2_cvae import Decoder2CVAE
import src.configs as cfg


def _load_tsjepa(device):
    model = TSJEPA(
        d_model=cfg.JEPA_CONFIG["d_model"],
        in_channels=4,
        patch_size=cfg.JEPA_CONFIG["patch_size"],
        nhead=cfg.JEPA_CONFIG["nhead"],
        num_layers=cfg.JEPA_CONFIG["num_layers"],
        ema_decay=cfg.JEPA_CONFIG["ema_decay"],
    ).to(device)
    for path in ["results/ts_jepa_hpo_best.pth", "results/ts_jepa.pth"]:
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=device, weights_only=False)
            state = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state, strict=False)
            print(f"Loaded TS-JEPA from {path}")
            break
    for param in model.parameters():
        param.requires_grad = False
    return model


def _load_decoder1(device):
    model = Decoder1(
        d_model=int(cfg.JEPA_CONFIG["d_model"]),
        seq_length=cfg.SEQ_LENGTH,
        out_channels=4,
    ).to(device)
    for path in ["results/decoder1_hpo_best.pth", "results/decoder1.pth"]:
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=device, weights_only=False)
            state = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state, strict=False)
            print(f"Loaded Decoder1 from {path}")
            break
    for param in model.parameters():
        param.requires_grad = False
    return model


def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def make_objective(ts_jepa, decoder1, device):
    recon_criterion = nn.MSELoss()

    def objective(trial):
        lr         = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        epochs     = trial.suggest_categorical("epochs", [50, 100, 150])
        latent_dim = trial.suggest_categorical("latent_dim", [32, 64, 128])
        beta_kl    = trial.suggest_float("beta_kl", 0.001, 0.1, log=True)

        train_loader, val_loader = get_data_loaders(batch_size)
        if train_loader is None:
            raise optuna.TrialPruned()

        decoder2 = Decoder2CVAE(
            seq_length=cfg.SEQ_LENGTH,
            in_channels=4,
            context_dim=int(cfg.JEPA_CONFIG["d_model"]),
            latent_dim=latent_dim,
            num_classes=cfg.NUM_CLASSES,
        ).to(device)
        optimizer = torch.optim.Adam(decoder2.parameters(), lr=lr)

        ts_jepa.eval()
        decoder1.eval()
        decoder2.train()
        for epoch in range(epochs):
            for batch in train_loader:
                raw   = batch[0].to(device)
                label = batch[2].to(device)
                with torch.no_grad():
                    z_macro  = ts_jepa.get_z_macro(raw)
                    recon_d1 = decoder1(z_macro)
                    residual = raw - recon_d1
                optimizer.zero_grad()
                recon_residual, mu, logvar = decoder2(residual, z_macro, label)
                recon_loss = recon_criterion(recon_residual, residual)
                kl_loss    = kl_divergence(mu, logvar)
                loss = recon_loss + beta_kl * (kl_loss / raw.size(0))
                loss.backward()
                optimizer.step()

            if epoch % 10 == 0:
                val_elbo = _eval_decoder2(ts_jepa, decoder1, decoder2, val_loader, device, recon_criterion, beta_kl)
                trial.report(val_elbo, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                decoder2.train()

        val_elbo = _eval_decoder2(ts_jepa, decoder1, decoder2, val_loader, device, recon_criterion, beta_kl)
        log_trial_result(None, trial, {"val_elbo": val_elbo})
        return val_elbo

    return objective


def _eval_decoder2(ts_jepa, decoder1, decoder2, val_loader, device, recon_criterion, beta_kl):
    decoder2.eval()
    total = 0.0
    with torch.no_grad():
        for batch in val_loader:
            raw   = batch[0].to(device)
            label = batch[2].to(device)
            z_macro  = ts_jepa.get_z_macro(raw)
            recon_d1 = decoder1(z_macro)
            residual = raw - recon_d1
            recon_residual, mu, logvar = decoder2(residual, z_macro, label)
            recon_loss = recon_criterion(recon_residual, residual)
            kl_loss    = kl_divergence(mu, logvar)
            total += (recon_loss + beta_kl * (kl_loss / raw.size(0))).item()
    return total / max(len(val_loader), 1)


def retrain_best(ts_jepa, decoder1, best_params, device):
    train_loader, _ = get_data_loaders(best_params["batch_size"])
    if train_loader is None:
        print("No data found — skipping retrain.")
        return
    recon_criterion = nn.MSELoss()
    decoder2 = Decoder2CVAE(
        seq_length=cfg.SEQ_LENGTH,
        in_channels=4,
        context_dim=int(cfg.JEPA_CONFIG["d_model"]),
        latent_dim=best_params["latent_dim"],
        num_classes=cfg.NUM_CLASSES,
    ).to(device)
    optimizer = torch.optim.Adam(decoder2.parameters(), lr=best_params["lr"])
    ts_jepa.eval()
    decoder1.eval()
    decoder2.train()
    for _ in range(best_params["epochs"]):
        for batch in train_loader:
            raw   = batch[0].to(device)
            label = batch[2].to(device)
            with torch.no_grad():
                z_macro  = ts_jepa.get_z_macro(raw)
                recon_d1 = decoder1(z_macro)
                residual = raw - recon_d1
            optimizer.zero_grad()
            recon_residual, mu, logvar = decoder2(residual, z_macro, label)
            recon_loss = recon_criterion(recon_residual, residual)
            kl_loss    = kl_divergence(mu, logvar)
            loss = recon_loss + best_params["beta_kl"] * (kl_loss / raw.size(0))
            loss.backward()
            optimizer.step()
    os.makedirs("results", exist_ok=True)
    torch.save(
        {"model_state_dict": decoder2.state_dict(), "best_params": best_params},
        "results/decoder2_hpo_best.pth",
    )
    print("Saved: results/decoder2_hpo_best.pth")


def main():
    parser = argparse.ArgumentParser(description="Decoder2 CVAE HPO")
    parser.add_argument("--n_trials",   type=int, default=20)
    parser.add_argument("--storage",    type=str, default="results/optuna_studies.db")
    parser.add_argument("--study_name", type=str, default="decoder2-hpo")
    parser.add_argument("--dry_run",    action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print("Decoder2 CVAE HPO search space:")
        print("  lr: [1e-5, 1e-2] (log scale)")
        print("  batch_size: [16, 32, 64]")
        print("  epochs: [50, 100, 150]")
        print("  latent_dim: [32, 64, 128]")
        print("  beta_kl: [0.001, 0.1] (log scale)")
        print("  (hidden_dim/num_layers/dropout excluded — no such constructor args in Decoder2CVAE)")
        print("Objective: minimize validation ELBO (recon_loss + beta_kl * KL / batch_size)")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ts_jepa  = _load_tsjepa(device)
    decoder1 = _load_decoder1(device)
    study = create_or_load_study(args.study_name, "minimize", args.storage)
    study.optimize(make_objective(ts_jepa, decoder1, device), n_trials=args.n_trials)

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best val ELBO: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")

    retrain_best(ts_jepa, decoder1, study.best_params, device)


if __name__ == "__main__":
    main()
