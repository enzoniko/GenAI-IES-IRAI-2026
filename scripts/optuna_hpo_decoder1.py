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


def make_objective(ts_jepa, device):
    criterion = nn.MSELoss()

    def objective(trial):
        lr         = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        epochs     = trial.suggest_categorical("epochs", [50, 100, 150])

        train_loader, val_loader = get_data_loaders(batch_size)
        if train_loader is None:
            raise optuna.TrialPruned()

        decoder = Decoder1(
            d_model=int(cfg.JEPA_CONFIG["d_model"]),
            seq_length=cfg.SEQ_LENGTH,
            out_channels=4,
        ).to(device)
        optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)

        ts_jepa.eval()
        decoder.train()
        for epoch in range(epochs):
            for batch in train_loader:
                raw = batch[0].to(device)
                with torch.no_grad():
                    z_macro = ts_jepa.get_z_macro(raw)
                optimizer.zero_grad()
                recon = decoder(z_macro)
                loss = criterion(recon, raw)
                loss.backward()
                optimizer.step()

            if epoch % 10 == 0:
                val_loss = _eval_decoder1(ts_jepa, decoder, val_loader, device, criterion)
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                decoder.train()

        val_loss = _eval_decoder1(ts_jepa, decoder, val_loader, device, criterion)
        log_trial_result(None, trial, {"val_mse": val_loss})
        return val_loss

    return objective


def _eval_decoder1(ts_jepa, decoder, val_loader, device, criterion):
    decoder.eval()
    total = 0.0
    with torch.no_grad():
        for batch in val_loader:
            raw = batch[0].to(device)
            z_macro = ts_jepa.get_z_macro(raw)
            recon = decoder(z_macro)
            total += criterion(recon, raw).item()
    return total / max(len(val_loader), 1)


def retrain_best(ts_jepa, best_params, device):
    train_loader, _ = get_data_loaders(best_params["batch_size"])
    if train_loader is None:
        print("No data found — skipping retrain.")
        return
    criterion = nn.MSELoss()
    decoder = Decoder1(
        d_model=int(cfg.JEPA_CONFIG["d_model"]),
        seq_length=cfg.SEQ_LENGTH,
        out_channels=4,
    ).to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=best_params["lr"])
    ts_jepa.eval()
    decoder.train()
    retrain_epochs = best_params["epochs"]
    for _ in range(retrain_epochs):
        for batch in train_loader:
            raw = batch[0].to(device)
            with torch.no_grad():
                z_macro = ts_jepa.get_z_macro(raw)
            optimizer.zero_grad()
            recon = decoder(z_macro)
            loss = criterion(recon, raw)
            loss.backward()
            optimizer.step()
    os.makedirs("results", exist_ok=True)
    torch.save(
        {"model_state_dict": decoder.state_dict(), "best_params": best_params},
        "results/decoder1_hpo_best.pth",
    )
    print("Saved: results/decoder1_hpo_best.pth")


def main():
    parser = argparse.ArgumentParser(description="Decoder1 HPO")
    parser.add_argument("--n_trials",   type=int, default=20)
    parser.add_argument("--storage",    type=str, default="results/optuna_studies.db")
    parser.add_argument("--study_name", type=str, default="decoder1-hpo")
    parser.add_argument("--dry_run",    action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print("Decoder1 HPO search space:")
        print("  lr: [1e-5, 1e-2] (log scale)")
        print("  batch_size: [16, 32, 64]")
        print("  epochs: [50, 100, 150]")
        print("  (Decoder1 is a fixed CNN — no hidden_dim/num_layers/dropout constructor args)")
        print("  (d_model=128, seq_length=3014, out_channels=4 fixed)")
        print("Objective: minimize validation reconstruction MSE")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ts_jepa = _load_tsjepa(device)
    study = create_or_load_study(args.study_name, "minimize", args.storage)
    study.optimize(make_objective(ts_jepa, device), n_trials=args.n_trials)

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best val MSE: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")

    retrain_best(ts_jepa, study.best_params, device)


if __name__ == "__main__":
    main()
