#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path

import optuna
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.hpo_utils import create_or_load_study
from src.models.pinn import ConfigurablePINN
from src.models.relobralo_loss import ReLoBRaLoLoss
import src.configs as cfg


ROOT = Path(__file__).resolve().parents[1]
X_PATH = ROOT / "data/processed-mafaulda/16hz/X_normal_trainingset.pth"
Y_PATH = ROOT / "data/processed-mafaulda/16hz/Y_normal_trainingset.pth"


def load_data(batch_size: int) -> tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    X = torch.load(X_PATH, map_location="cpu", weights_only=True)
    y = torch.load(Y_PATH, map_location="cpu", weights_only=True)

    X_flat = X.reshape(-1, 10)
    y_flat = y.reshape(-1, 4)

    metadata_path = ROOT / "results/normalization_metadata.pth"
    metadata = torch.load(metadata_path, map_location="cpu", weights_only=True)
    X_min = metadata["X_min"].reshape(-1).double()
    X_max = metadata["X_max"].reshape(-1).double()
    y_min = metadata["y_min"].reshape(-1).double()
    y_max = metadata["y_max"].reshape(-1).double()

    X_norm = (X_flat.double() - X_min) / (X_max - X_min + 1e-8)
    y_norm = (y_flat.double() - y_min) / (y_max - y_min + 1e-8)

    n_train = int(0.8 * X_norm.shape[0])
    train_loader = DataLoader(TensorDataset(X_norm[:n_train], y_norm[:n_train]), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_norm[n_train:], y_norm[n_train:]), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, X_min, X_max, y_min, y_max


def build_model() -> ConfigurablePINN:
    arch = cfg.PINN_POINTWISE_ARCH
    return ConfigurablePINN(
        unmeasured_net_config=arch["unmeasured_net_config"],
        acceleration_net_config=arch["acceleration_net_config"],
        param_init_config=arch["param_init_config"],
        enable_mass_constraints=arch["enable_mass_constraints"],
    ).double()


def make_objective(device):
    loss_fn = ReLoBRaLoLoss(
        alpha=cfg.RELOBRALO_DEFAULT["alpha"],
        rho=cfg.RELOBRALO_DEFAULT["rho"],
        temperature=cfg.RELOBRALO_DEFAULT["temperature"],
        enable_mass_constraints=cfg.PINN_POINTWISE_ARCH["enable_mass_constraints"],
    )

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
        trial_epochs = trial.suggest_categorical("epochs", [50, 100, 200])
        no_denorm = trial.suggest_categorical("no_denorm", [True, False])

        train_loader, val_loader, X_min, X_max, y_min, y_max = load_data(batch_size)
        model = build_model().to(device)
        optimizer = Adam(model.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        y_min_arg = None if no_denorm else y_min.to(device)
        y_max_arg = None if no_denorm else y_max.to(device)
        X_min = X_min.to(device)
        X_max = X_max.to(device)

        best_val_loss = float("inf")
        model.train()
        for epoch in range(trial_epochs):
            for xb, yb in train_loader:
                xb = xb.to(device).double()
                yb = yb.to(device).double()
                loss_components = loss_fn(model, xb, yb, X_max, X_min, y_max_arg, y_min_arg)
                total_loss = loss_fn.step(loss_components, optimizer, model, xb, yb)
                if not torch.isfinite(total_loss):
                    raise optuna.TrialPruned()

            model.eval()
            val_total = 0.0
            val_batches = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device).double()
                    yb = yb.to(device).double()
                    loss_components = loss_fn(model, xb, yb, X_max, X_min, y_max_arg, y_min_arg)
                    weights = loss_fn.running_weights
                    if weights is None:
                        weights = torch.ones(len(loss_components), dtype=torch.float64, device=device)
                    weighted_val = torch.sum(torch.stack([weights[i] * loss_components[i] for i in range(len(loss_components))]))
                    val_total += float(weighted_val.detach().item())
                    val_batches += 1

            val_loss = val_total / max(val_batches, 1)
            scheduler.step(val_loss)
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
            best_val_loss = min(best_val_loss, val_loss)
            model.train()

        return best_val_loss

    return objective


def main():
    parser = argparse.ArgumentParser(description="PINN HPO")
    parser.add_argument("--n_trials", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    study = create_or_load_study("pinn-hpo", "minimize", "results/optuna_studies.db")
    study.optimize(make_objective(device), n_trials=args.n_trials)

    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value (val_loss): {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")
    print(f"Summary: pinn-hpo minimized val_loss to {study.best_value:.6f}")


if __name__ == "__main__":
    main()
