"""Standalone point-wise PINN training script matching prior-paper setup."""

# pyright: reportMissingTypeArgument=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportAny=false, reportUnusedCallResult=false, reportPrivateImportUsage=false, reportImplicitStringConcatenation=false, reportIndexIssue=false, reportExplicitAny=false, reportUnknownMemberType=false

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from src.configs import PINN_POINTWISE_ARCH, PINN_POINTWISE_SETTINGS, RELOBRALO_DEFAULT
from src.models.pinn import ConfigurablePINN
from src.models.relobralo_loss import ReLoBRaLoLoss


ROOT = Path(__file__).resolve().parents[1]
X_PATH = ROOT / "data/processed-mafaulda/16hz/X_normal_trainingset.pth"
Y_PATH = ROOT / "data/processed-mafaulda/16hz/Y_normal_trainingset.pth"
METADATA_PATH = ROOT / "results/normalization_metadata.pth"
CHECKPOINT_PATH = ROOT / "results/pinn_pointwise.pth"

PRINT_EVERY = 100
PARAM_NAMES = ("M1", "M2", "M3", "D1", "D2", "D3", "K1", "K2", "E1")


def load_pointwise_data(max_samples: int, batch_size: int) -> tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load normal-only data, flatten to points, subsample, normalize, and split."""
    X = torch.load(X_PATH, map_location="cpu", weights_only=True)
    y = torch.load(Y_PATH, map_location="cpu", weights_only=True)

    if X.shape[-1] != 10:
        raise ValueError(f"Expected 10 input features, got {X.shape[-1]}")
    if y.shape[-1] != 4:
        raise ValueError(f"Expected 4 target features, got {y.shape[-1]}")

    X_flat = X.reshape(-1, 10)
    y_flat = y.reshape(-1, 4)

    if X_flat.shape[0] < max_samples:
        raise ValueError(f"Requested max_samples={max_samples}, but only {X_flat.shape[0]} points are available")

    idx = torch.randperm(X_flat.shape[0])[:max_samples]
    X_sub = X_flat[idx]
    y_sub = y_flat[idx]

    metadata = torch.load(METADATA_PATH, map_location="cpu", weights_only=True)
    X_min = metadata["X_min"].reshape(-1).double()
    X_max = metadata["X_max"].reshape(-1).double()
    y_min = metadata["y_min"].reshape(-1).double()
    y_max = metadata["y_max"].reshape(-1).double()

    if X_min.numel() != 10 or X_max.numel() != 10:
        raise ValueError("Normalization metadata must contain 10-feature X_min/X_max tensors")
    if y_min.numel() != 4 or y_max.numel() != 4:
        raise ValueError("Normalization metadata must contain 4-feature y_min/y_max tensors")

    X_norm = (X_sub.double() - X_min) / (X_max - X_min + 1e-8)
    y_norm = (y_sub.double() - y_min) / (y_max - y_min + 1e-8)

    n_train = int(0.8 * max_samples)
    X_train, y_train = X_norm[:n_train], y_norm[:n_train]
    X_val, y_val = X_norm[n_train:], y_norm[n_train:]

    train_loader = DataLoader(
        TensorDataset(X_train.double(), y_train.double()),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val.double(), y_val.double()),
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, X_min, X_max, y_min, y_max


def build_model() -> ConfigurablePINN:
    """Construct the pointwise PINN using configured architecture."""
    arch = cast(dict[str, Any], PINN_POINTWISE_ARCH)
    return ConfigurablePINN(
        unmeasured_net_config=arch["unmeasured_net_config"],
        acceleration_net_config=arch["acceleration_net_config"],
        param_init_config=arch["param_init_config"],
        enable_mass_constraints=arch["enable_mass_constraints"],
    ).double()


def format_physical_params(model: ConfigurablePINN) -> str:
    values = []
    for name in PARAM_NAMES:
        raw_value = getattr(model, name).detach().reshape(-1)[0].item()
        values.append(f"{name}={raw_value:.6f}")
    return " | ".join(values)


def report_non_physical_params(model: ConfigurablePINN) -> None:
    negatives = []
    for name in PARAM_NAMES:
        value = getattr(model, name).detach().reshape(-1)[0].item()
        if value < 0:
            negatives.append(f"{name}={value:.6f}")

    if negatives:
        print("Non-physical negative parameters detected (expected/acceptable per prior paper):")
        print("  " + ", ".join(negatives))
    else:
        print("No negative physical parameters detected at training end.")


def save_checkpoint(
    model: ConfigurablePINN,
    X_min: torch.Tensor,
    X_max: torch.Tensor,
    y_min: torch.Tensor,
    y_max: torch.Tensor,
    epoch: int,
    val_loss: float,
    train_loss: float,
) -> None:
    checkpoint = {
        "model_state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        "X_min": X_min.detach().cpu().clone(),
        "X_max": X_max.detach().cpu().clone(),
        "y_min": y_min.detach().cpu().clone(),
        "y_max": y_max.detach().cpu().clone(),
        "epoch": epoch,
        "val_loss": val_loss,
        "train_loss": train_loss,
    }
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, CHECKPOINT_PATH)


def main() -> None:
    torch.manual_seed(42)

    settings = cast(dict[str, Any], PINN_POINTWISE_SETTINGS)
    arch = cast(dict[str, Any], PINN_POINTWISE_ARCH)
    epochs = int(settings["epochs"])
    batch_size = int(settings["batch_size"])
    lr = float(settings["learning_rate"])
    max_samples = int(settings["max_samples"])
    patience = int(settings["early_stop_patience"])
    min_delta = float(settings["early_stop_min_delta"])
    lr_patience = int(settings["lr_scheduler_patience"])
    lr_factor = float(settings["lr_scheduler_factor"])

    print("Training point-wise PINN on normal-class data only")
    print(
        "Hyperparameters: "
        f"batch_size={batch_size}, lr={lr:.4f}, max_samples={max_samples}, "
        f"patience={patience}, init={arch['unmeasured_net_config']['init_method']}, "
        f"dropout={arch['unmeasured_net_config']['dropout_rate']:.3f}"
    )

    train_loader, val_loader, X_min, X_max, y_min, y_max = load_pointwise_data(
        max_samples=max_samples,
        batch_size=batch_size,
    )

    model = build_model()
    loss_fn = ReLoBRaLoLoss(
        alpha=RELOBRALO_DEFAULT["alpha"],
        rho=RELOBRALO_DEFAULT["rho"],
        temperature=RELOBRALO_DEFAULT["temperature"],
        enable_mass_constraints=arch["enable_mass_constraints"],
    )
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, patience=lr_patience, factor=lr_factor)

    X_min = X_min.double()
    X_max = X_max.double()
    y_min = y_min.double()
    y_max = y_max.double()

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_total = 0.0
        train_batches = 0

        for xb, yb in train_loader:
            xb = xb.double()
            yb = yb.double()

            loss_components = loss_fn(model, xb, yb, X_max.double(), X_min.double(), y_max.double(), y_min.double())
            total_loss = loss_fn.step(loss_components, optimizer, model, xb, yb)

            if not torch.isfinite(total_loss):
                raise RuntimeError(f"Non-finite training loss encountered at epoch {epoch}")

            train_total += float(total_loss.detach().item())
            train_batches += 1

        if train_batches == 0:
            raise RuntimeError("No training batches were processed")

        avg_train_loss = train_total / train_batches

        model.eval()
        val_total = 0.0
        val_batches = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.double()
                yb = yb.double()

                loss_components = loss_fn(model, xb, yb, X_max.double(), X_min.double(), y_max.double(), y_min.double())
                if any(not torch.isfinite(component) for component in loss_components):
                    raise RuntimeError(f"Non-finite validation loss component encountered at epoch {epoch}")

                weights = loss_fn.running_weights
                if weights is None:
                    weights = torch.ones(len(loss_components), dtype=torch.float64)

                weighted_val = torch.sum(torch.stack([weights[i] * loss_components[i] for i in range(len(loss_components))]))
                if not torch.isfinite(weighted_val):
                    raise RuntimeError(f"Non-finite validation total encountered at epoch {epoch}")

                val_total += float(weighted_val.detach().item())
                val_batches += 1

        if val_batches == 0:
            raise RuntimeError("No validation batches were processed")

        avg_val_loss = val_total / val_batches
        scheduler.step(avg_val_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])

        if epoch % PRINT_EVERY == 0 or epoch == 1:
            print(
                f"epoch={epoch:05d} | train_loss={avg_train_loss:.6f} | "
                f"val_loss={avg_val_loss:.6f} | lr={current_lr:.6e}"
            )
            print(format_physical_params(model))

        if avg_val_loss < (best_val_loss - min_delta):
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            save_checkpoint(model, X_min, X_max, y_min, y_max, epoch, best_val_loss, avg_train_loss)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch} after {patience} epochs without improvement.")
            break

    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Checkpoint saved to: {CHECKPOINT_PATH}")
    print("Final physical parameters:")
    print(format_physical_params(model))
    report_non_physical_params(model)


if __name__ == "__main__":
    main()
