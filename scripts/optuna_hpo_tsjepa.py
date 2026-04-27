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
    evaluate_z_macro_health,
    get_data_loaders,
    log_trial_result,
    check_convergence,
)
from src.models.ts_jepa import TSJEPA
import src.configs as cfg


def make_objective(device):
    criterion = nn.MSELoss()

    def objective(trial):
        lr         = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        epochs     = trial.suggest_categorical("epochs", [50, 100, 150])
        patch_size = trial.suggest_categorical("patch_size", [25, 50, 100])
        ema_decay  = trial.suggest_float("ema_decay", 0.95, 0.999)

        train_loader, val_loader = get_data_loaders(batch_size)
        if train_loader is None:
            raise optuna.TrialPruned()

        model = TSJEPA(
            d_model=cfg.JEPA_CONFIG["d_model"],
            in_channels=4,
            patch_size=patch_size,
            nhead=cfg.JEPA_CONFIG["nhead"],
            num_layers=cfg.JEPA_CONFIG["num_layers"],
            ema_decay=ema_decay,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        model.train()
        for epoch in range(epochs):
            for batch in train_loader:
                raw = batch[0].to(device)
                optimizer.zero_grad()
                pred_unobs, target_unobs = model(raw)
                loss = criterion(pred_unobs, target_unobs)
                loss.backward()
                optimizer.step()
                model.update_ema()

            if epoch % 10 == 0:
                health = evaluate_z_macro_health(model, val_loader, device)
                trial.report(health, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                model.train()

        final_health = evaluate_z_macro_health(model, val_loader, device)
        log_trial_result(None, trial, {"z_macro_mean_std": final_health})
        return final_health

    return objective


def retrain_best(best_params, device):
    train_loader, val_loader = get_data_loaders(best_params["batch_size"])
    if train_loader is None:
        print("No data found — skipping retrain.")
        return
    criterion = nn.MSELoss()
    model = TSJEPA(
        d_model=cfg.JEPA_CONFIG["d_model"],
        in_channels=4,
        patch_size=best_params["patch_size"],
        nhead=cfg.JEPA_CONFIG["nhead"],
        num_layers=cfg.JEPA_CONFIG["num_layers"],
        ema_decay=best_params["ema_decay"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])
    retrain_epochs = 50
    model.train()
    for _ in range(retrain_epochs):
        for batch in train_loader:
            raw = batch[0].to(device)
            optimizer.zero_grad()
            pred_unobs, target_unobs = model(raw)
            loss = criterion(pred_unobs, target_unobs)
            loss.backward()
            optimizer.step()
            model.update_ema()
    os.makedirs("results", exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "best_params": best_params},
        "results/ts_jepa_hpo_best.pth",
    )
    print("Saved: results/ts_jepa_hpo_best.pth")


def main():
    parser = argparse.ArgumentParser(description="TS-JEPA HPO")
    parser.add_argument("--n_trials",   type=int, default=30)
    parser.add_argument("--storage",    type=str, default="results/optuna_studies.db")
    parser.add_argument("--study_name", type=str, default="tsjepa-hpo")
    parser.add_argument("--dry_run",    action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print("TS-JEPA HPO search space:")
        print("  lr: [1e-5, 1e-2] (log scale)")
        print("  batch_size: [16, 32, 64]")
        print("  epochs: [50, 100, 150]")
        print("  patch_size: [25, 50, 100]")
        print("  ema_decay: [0.95, 0.999]")
        print("  (mask_ratio excluded — hardcoded as num_mask = N // 2 in ts_jepa.forward)")
        print("  (d_model=128, num_layers=4 fixed — architecture constraints)")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    study = create_or_load_study(args.study_name, "maximize", args.storage)
    study.optimize(make_objective(device), n_trials=args.n_trials)

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best value (z_macro mean per-dim std): {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")
    verdict = "HEALTHY" if study.best_value > 0.01 else "STILL COLLAPSED"
    print(f"z_macro health: {verdict}")

    retrain_best(study.best_params, device)


if __name__ == "__main__":
    main()
