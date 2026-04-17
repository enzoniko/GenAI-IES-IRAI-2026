"""Task 9 PINN training on MaFaulDa 16Hz one-subtype-per-class data."""

# pyright: reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportMissingTypeArgument=false, reportPrivateImportUsage=false, reportAny=false, reportUnknownArgumentType=false, reportArgumentType=false, reportUnusedCallResult=false, reportUnusedImport=false, reportUnusedVariable=false, reportUnannotatedClassAttribute=false, reportImplicitOverride=false, reportExplicitAny=false

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, Subset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.configs as cfg
from src.models.feature_extractors import MathFeatureExtractor
from src.models.pinn import ConfigurablePINN
from src.models.relobralo_loss import ReLoBRaLoLoss


RESULTS_DIR = ROOT / "results"
EVIDENCE_DIR = ROOT / ".sisyphus" / "evidence"
DEFAULT_NORM_PATH = RESULTS_DIR / "normalization_metadata.pth"
LEGACY_WEIGHTS_PATH = ROOT / "previous-work-pinn" / "export_relobralo" / "weights.pth"
SCRATCH_CKPT_PATH = RESULTS_DIR / "pinn_scratch.pth"
FINETUNE_CKPT_PATH = RESULTS_DIR / "pinn_finetune.pth"
BEST_CKPT_PATH = RESULTS_DIR / "pinn.pth"
TRAINING_EVIDENCE_PATH = EVIDENCE_DIR / "task-9-pinn-training.txt"
DECISION_EVIDENCE_PATH = EVIDENCE_DIR / "task-9-pinn-decision.txt"
LOG_EPOCHS = {1, 10, 50, 100, 250, 500}
LOSS_KEYS = ["data", "phys_res1", "phys_res2", "phys_res3", "phys_res4", "phys_mass1", "phys_mass2"]
CLASS_NAMES = {
    0: "normal",
    1: "imbalance_fault_20g",
    2: "vertical_misalignment_fault_1.27mm",
    3: "overhang_ball_fault_20g",
}
SEED = 42
ONE_SUBTYPE_FILES = {
    0: {
        "X": ROOT / "data" / "processed-mafaulda" / "16hz" / "X_normal_trainingset.pth",
        "Y": ROOT / "data" / "processed-mafaulda" / "16hz" / "Y_normal_trainingset.pth",
    },
    1: {
        "X": ROOT / "data" / "processed-mafaulda" / "16hz" / "X_imbalance_fault_20g_trainingset.pth",
        "Y": ROOT / "data" / "processed-mafaulda" / "16hz" / "Y_imbalance_fault_20g_trainingset.pth",
    },
    2: {
        "X": ROOT / "data" / "processed-mafaulda" / "16hz" / "X_vertical_misalignment_fault_1.27mm_trainingset.pth",
        "Y": ROOT / "data" / "processed-mafaulda" / "16hz" / "Y_vertical_misalignment_fault_1.27mm_trainingset.pth",
    },
    3: {
        "X": ROOT / "data" / "processed-mafaulda" / "16hz" / "X_overhang_ball_fault_20g_trainingset.pth",
        "Y": ROOT / "data" / "processed-mafaulda" / "16hz" / "Y_overhang_ball_fault_20g_trainingset.pth",
    },
}


class XYWindowDataset(Dataset):
    def __init__(self, X: torch.Tensor, Y: torch.Tensor, labels: torch.Tensor):
        self.X = X
        self.Y = Y
        self.labels = labels

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx], self.labels[idx]


@dataclass
class LoadedDataset:
    dataset: XYWindowDataset
    full_loader: DataLoader
    train_loader: DataLoader
    val_loader: DataLoader
    labels_np: np.ndarray
    x_bounds_full: dict[str, torch.Tensor]
    y_bounds: dict[str, torch.Tensor]
    counts_by_class: dict[int, int]
    omega_stats: dict[str, float]
    time_stats: dict[str, float]


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_norm_metadata(norm_path: Path) -> dict[str, torch.Tensor]:
    metadata = torch.load(norm_path, map_location="cpu", weights_only=True)
    required = {"X_min", "X_max", "y_min", "y_max"}
    missing = required.difference(metadata)
    if missing:
        raise KeyError(f"Normalization metadata missing keys: {sorted(missing)}")
    return {key: metadata[key].double() for key in required}


def load_dataset(batch_size: int, norm_path: Path) -> LoadedDataset:
    metadata = _load_norm_metadata(norm_path)
    X_min8 = metadata["X_min"]
    X_max8 = metadata["X_max"]
    y_min = metadata["y_min"]
    y_max = metadata["y_max"]

    X_chunks: list[torch.Tensor] = []
    Y_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    counts_by_class: dict[int, int] = {}

    omega_all: list[torch.Tensor] = []
    time_all: list[torch.Tensor] = []

    for label, paths in ONE_SUBTYPE_FILES.items():
        X_raw = torch.load(paths["X"], map_location="cpu", weights_only=True).double()
        Y_raw = torch.load(paths["Y"], map_location="cpu", weights_only=True).double()
        labels = torch.full((X_raw.shape[0],), label, dtype=torch.long)

        counts_by_class[label] = int(X_raw.shape[0])
        omega_all.append(X_raw[:, :, 8].reshape(-1))
        time_all.append(X_raw[:, :, 9].reshape(-1))

        X_chunks.append(X_raw)
        Y_chunks.append(Y_raw)
        label_chunks.append(labels)

    X = torch.cat(X_chunks, dim=0)
    Y = torch.cat(Y_chunks, dim=0)
    labels = torch.cat(label_chunks, dim=0)

    omega_values = torch.cat(omega_all)
    time_values = torch.cat(time_all)
    omega_min = omega_values.min().view(1).double()
    omega_max = omega_values.max().view(1).double()
    time_min = time_values.min().view(1).double()
    time_max = time_values.max().view(1).double()

    X_min_full = torch.cat([X_min8, omega_min, time_min], dim=0)
    X_max_full = torch.cat([X_max8, omega_max, time_max], dim=0)

    X_norm = X.clone().double()
    X_norm[:, :, :8] = (X_norm[:, :, :8] - X_min8.view(1, 1, 8)) / (X_max8 - X_min8 + 1e-12).view(1, 1, 8)
    X_norm[:, :, 8] = (X_norm[:, :, 8] - omega_min.item()) / (omega_max.item() - omega_min.item() + 1e-12)
    X_norm[:, :, 9] = (X_norm[:, :, 9] - time_min.item()) / (time_max.item() - time_min.item() + 1e-12)

    Y_norm = (Y - y_min.view(1, 1, 4)) / (y_max - y_min + 1e-12).view(1, 1, 4)

    dataset = XYWindowDataset(X_norm, Y_norm, labels)

    train_indices: list[int] = []
    val_indices: list[int] = []
    for label in sorted(CLASS_NAMES):
        label_idx = torch.where(labels == label)[0].tolist()
        val_indices.extend(label_idx[::5])
        train_indices.extend(label_idx[i] for i in range(len(label_idx)) if i % 5 != 0)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    full_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return LoadedDataset(
        dataset=dataset,
        full_loader=full_loader,
        train_loader=train_loader,
        val_loader=val_loader,
        labels_np=labels.numpy(),
        x_bounds_full={"X_min": X_min_full, "X_max": X_max_full},
        y_bounds={"y_min": y_min, "y_max": y_max},
        counts_by_class=counts_by_class,
        omega_stats={
            "min": float(omega_min.item()),
            "max": float(omega_max.item()),
            "mean": float(torch.cat([chunk[:, 0, 8] for chunk in X_chunks]).mean().item()),
        },
        time_stats={
            "min": float(time_min.item()),
            "max": float(time_max.item()),
        },
    )


def _safe_component_stats(loss_components: list[torch.Tensor]) -> dict[str, float]:
    return {key: float(loss_components[idx].detach().item()) for idx, key in enumerate(LOSS_KEYS)}


def _avg_dicts(records: list[dict[str, float]]) -> dict[str, float]:
    if not records:
        return {key: float("nan") for key in LOSS_KEYS}
    return {key: float(np.mean([record[key] for record in records])) for key in LOSS_KEYS}


def build_model() -> ConfigurablePINN:
    pinn_config = cfg.PINN_ARCH_DEFAULT
    return ConfigurablePINN(
        unmeasured_net_config=pinn_config["unmeasured_net_config"],
        acceleration_net_config=pinn_config["acceleration_net_config"],
        param_init_config=pinn_config["param_init_config"],
        enable_mass_constraints=pinn_config["enable_mass_constraints"],
    )


def _map_legacy_state_dict(raw_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    mapped: dict[str, torch.Tensor] = {}
    for key, value in raw_state.items():
        new_key = key.replace(".model.6.", ".model.4.").replace(".model.3.", ".model.2.")
        mapped[new_key] = value
    return mapped


def load_legacy_weights_for_current_model(model: ConfigurablePINN, checkpoint_path: Path) -> None:
    raw_state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(raw_state, dict):
        raise TypeError(f"Unexpected legacy checkpoint type: {type(raw_state)!r}")
    model.load_state_dict(_map_legacy_state_dict(raw_state), strict=True)


def train_strategy(
    strategy_name: str,
    loaded: LoadedDataset,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    checkpoint_path: Path,
    init_legacy_weights: bool = False,
) -> dict[str, Any]:
    model = build_model().to(device).double()
    if init_legacy_weights:
        load_legacy_weights_for_current_model(model, LEGACY_WEIGHTS_PATH)

    loss_method = ReLoBRaLoLoss(enable_mass_constraints=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=25,
        min_lr=1e-6,
    )

    X_min = loaded.x_bounds_full["X_min"].to(device)
    X_max = loaded.x_bounds_full["X_max"].to(device)
    y_min = loaded.y_bounds["y_min"].to(device)
    y_max = loaded.y_bounds["y_max"].to(device)

    best_val_loss = float("inf")
    best_epoch = 0
    best_state_dict: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    early_stop_patience = 80
    min_delta = 1e-4
    epochs_without_improvement = 0
    diverged_early = False
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        train_component_records: list[dict[str, float]] = []
        train_total_losses: list[float] = []
        invalid_epoch = False

        for xb, yb, _ in loaded.train_loader:
            xb = xb.to(device=device, dtype=torch.float64)
            yb = yb.to(device=device, dtype=torch.float64)

            xb_flat = xb.reshape(-1, xb.shape[-1])
            yb_flat = yb.reshape(-1, yb.shape[-1])
            loss_components = loss_method(model, xb_flat, yb_flat, X_max, X_min, y_max, y_min)

            if any(not torch.isfinite(component) for component in loss_components):
                invalid_epoch = True
                break

            total_loss = loss_method.step(loss_components, optimizer, model, xb_flat, yb_flat)
            if not torch.isfinite(total_loss):
                invalid_epoch = True
                break

            train_total_losses.append(float(total_loss.detach().item()))
            train_component_records.append(_safe_component_stats(loss_components))

        if invalid_epoch or not train_total_losses:
            if epoch <= 50:
                diverged_early = True
            print(f"[{strategy_name}] epoch {epoch}: divergence detected, stopping strategy.")
            break

        model.eval()
        val_component_records: list[dict[str, float]] = []
        val_total_losses: list[float] = []
        with torch.no_grad():
            for xb, yb, _ in loaded.val_loader:
                xb = xb.to(device=device, dtype=torch.float64)
                yb = yb.to(device=device, dtype=torch.float64)

                xb_flat = xb.reshape(-1, xb.shape[-1])
                yb_flat = yb.reshape(-1, yb.shape[-1])
                loss_components = loss_method(model, xb_flat, yb_flat, X_max, X_min, y_max, y_min)
                if any(not torch.isfinite(component) for component in loss_components):
                    invalid_epoch = True
                    break

                weights = [loss_method.current_weights[key] for key in LOSS_KEYS]
                weighted_val = sum(weights[idx] * float(loss_components[idx].item()) for idx in range(len(LOSS_KEYS)))
                val_total_losses.append(weighted_val)
                val_component_records.append(_safe_component_stats(loss_components))

        if invalid_epoch or not val_total_losses:
            if epoch <= 50:
                diverged_early = True
            print(f"[{strategy_name}] epoch {epoch}: validation divergence detected, stopping strategy.")
            break

        avg_train_total = float(np.mean(train_total_losses))
        avg_val_total = float(np.mean(val_total_losses))
        avg_train_components = _avg_dicts(train_component_records)
        avg_val_components = _avg_dicts(val_component_records)

        scheduler.step(avg_val_total)
        current_lr = float(optimizer.param_groups[0]["lr"])

        if epoch in LOG_EPOCHS or epoch == epochs:
            print(
                f"[{strategy_name}] epoch {epoch:04d} | train={avg_train_total:.6f} | val={avg_val_total:.6f} | lr={current_lr:.2e}"
            )
            print(
                f"[{strategy_name}]   train components: "
                + ", ".join(f"{key}={avg_train_components[key]:.6f}" for key in LOSS_KEYS)
            )
            print(
                f"[{strategy_name}]   val components:   "
                + ", ".join(f"{key}={avg_val_components[key]:.6f}" for key in LOSS_KEYS)
            )

        history.append(
            {
                "epoch": epoch,
                "train_total": avg_train_total,
                "val_total": avg_val_total,
                "lr": current_lr,
                "train_components": avg_train_components,
                "val_components": avg_val_components,
            }
        )

        if avg_val_total < (best_val_loss - min_delta):
            best_val_loss = avg_val_total
            best_epoch = epoch
            best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stop_patience:
            print(f"[{strategy_name}] early stopping at epoch {epoch}.")
            break

    duration_sec = time.perf_counter() - started

    if best_state_dict is None:
        return {
            "strategy": strategy_name,
            "status": "failed",
            "diverged_early": diverged_early,
            "history": history,
            "duration_sec": duration_sec,
            "checkpoint_path": str(checkpoint_path),
        }

    checkpoint = {
        "model_state_dict": best_state_dict,
        "X_min": loaded.x_bounds_full["X_min"].cpu(),
        "X_max": loaded.x_bounds_full["X_max"].cpu(),
        "y_min": loaded.y_bounds["y_min"].cpu(),
        "y_max": loaded.y_bounds["y_max"].cpu(),
        "strategy": strategy_name,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "duration_sec": duration_sec,
    }
    torch.save(checkpoint, checkpoint_path)

    return {
        "strategy": strategy_name,
        "status": "ok",
        "diverged_early": diverged_early,
        "history": history,
        "duration_sec": duration_sec,
        "checkpoint_path": str(checkpoint_path),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }


def evaluate_checkpoint(checkpoint_path: Path, loaded: LoadedDataset, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_model().to(device).double()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    extractor = MathFeatureExtractor(in_channels=4).to(device).double().eval()
    X_min = checkpoint["X_min"].to(device)
    X_max = checkpoint["X_max"].to(device)
    y_min = checkpoint["y_min"].to(device)
    y_max = checkpoint["y_max"].to(device)

    residual_feature_batches: list[np.ndarray] = []
    residual_abs_means: list[np.ndarray] = []

    with torch.no_grad():
        for xb, _yb, _ in loaded.full_loader:
            xb = xb.to(device=device, dtype=torch.float64)
            batch_size, seq_len, _ = xb.shape

            xb_flat = xb.reshape(-1, xb.shape[-1])
            pred = model(xb_flat)
            residuals = model.compute_residuals(xb_flat, pred, X_max, X_min, y_max, y_min)[:4]
            residual_matrix = torch.cat(residuals, dim=1)
            residual_seq = residual_matrix.reshape(batch_size, seq_len, 4)
            features = extractor(residual_seq).cpu().numpy()
            residual_feature_batches.append(features)
            residual_abs_means.append(residual_seq.abs().mean(dim=(0, 1)).cpu().numpy())

    residual_features = np.concatenate(residual_feature_batches, axis=0)
    labels = loaded.labels_np
    pca = PCA(n_components=min(50, residual_features.shape[0], residual_features.shape[1]), random_state=42)
    scaled_features = StandardScaler().fit_transform(residual_features)
    residual_features_pca = pca.fit_transform(scaled_features)
    silhouette = float(silhouette_score(residual_features_pca, labels, random_state=42))

    knn_pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    knn_scores = cross_val_score(knn_pipeline, residual_features_pca, labels, cv=cv, scoring="accuracy", n_jobs=1)

    return {
        "feature_shape": list(residual_features.shape),
        "pca50_shape": list(residual_features_pca.shape),
        "silhouette_pca50": silhouette,
        "knn5_cv_accuracy_pca50": float(knn_scores.mean()),
        "knn5_cv_accuracy_pca50_std": float(knn_scores.std()),
        "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "residual_abs_component_means": np.mean(np.stack(residual_abs_means, axis=0), axis=0).tolist(),
    }


def write_evidence(
    loaded: LoadedDataset,
    scratch_result: dict[str, Any],
    scratch_eval: dict[str, Any] | None,
    finetune_result: dict[str, Any] | None,
    finetune_eval: dict[str, Any] | None,
    winner: dict[str, Any],
) -> None:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "dataset": {
            "counts_by_class": {CLASS_NAMES[key]: value for key, value in loaded.counts_by_class.items()},
            "total_windows": int(len(loaded.dataset)),
            "train_windows": int(len(loaded.train_loader.dataset)),
            "val_windows": int(len(loaded.val_loader.dataset)),
            "omega_stats": loaded.omega_stats,
            "time_stats": loaded.time_stats,
            "x_min_full": loaded.x_bounds_full["X_min"].tolist(),
            "x_max_full": loaded.x_bounds_full["X_max"].tolist(),
            "y_min": loaded.y_bounds["y_min"].tolist(),
            "y_max": loaded.y_bounds["y_max"].tolist(),
        },
        "scratch": {
            "result": scratch_result,
            "evaluation": scratch_eval,
        },
        "finetune": {
            "result": finetune_result,
            "evaluation": finetune_eval,
        },
        "winner": winner,
    }

    lines: list[str] = []
    lines.append("Task 9 PINN Training Evidence")
    lines.append("")
    lines.append("Dataset")
    lines.append(json.dumps(payload["dataset"], indent=2))
    lines.append("")

    for key in ("scratch", "finetune"):
        lines.append(f"Strategy: {key}")
        lines.append(json.dumps(payload[key]["result"], indent=2))
        if payload[key]["evaluation"] is not None:
            lines.append(json.dumps(payload[key]["evaluation"], indent=2))
        lines.append("")

    lines.append("Winner")
    lines.append(json.dumps(winner, indent=2))
    TRAINING_EVIDENCE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_decision(winner: dict[str, Any], scratch_attempted_only: bool, scratch_diverged: bool) -> None:
    silhouette = winner["evaluation"]["silhouette_pca50"]
    if silhouette > 0.3:
        decision = "Success: residual-feature clustering is sufficiently separated (Silhouette > 0.3)."
    else:
        decision = "Fallback: Use physics-informed representation for transfer classification (no clean clustering claim)."

    notes = [
        "Task 9 PINN Decision",
        "",
        f"winning_strategy: {winner['strategy']}",
        f"checkpoint: {winner['checkpoint_path']}",
        f"best_epoch: {winner['best_epoch']}",
        f"best_val_loss: {winner['best_val_loss']:.6f}",
        f"silhouette_pca50: {winner['evaluation']['silhouette_pca50']:.6f}",
        f"knn5_cv_accuracy_pca50: {winner['evaluation']['knn5_cv_accuracy_pca50']:.6f}",
        f"decision: {decision}",
    ]
    if scratch_attempted_only:
        notes.append("strategy_a_status: skipped (scratch training remained stable; T5 already showed legacy omega mismatch)")
    elif scratch_diverged:
        notes.append("strategy_a_status: attempted because scratch diverged within first 50 epochs")
    DECISION_EVIDENCE_PATH.write_text("\n".join(notes) + "\n", encoding="utf-8")


def select_winner(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [candidate for candidate in candidates if candidate.get("evaluation") is not None]
    if not successful:
        raise RuntimeError("No successful PINN strategy produced evaluation metrics.")
    successful.sort(
        key=lambda item: (
            item["evaluation"]["silhouette_pca50"],
            item["evaluation"]["knn5_cv_accuracy_pca50"],
            -item["best_val_loss"],
        ),
        reverse=True,
    )
    return successful[0]


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 9 PINN training on 16Hz MaFaulDa")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=cfg.PHASE0_TRAIN_SETTINGS["learning_rate"])
    parser.add_argument("--norm-path", type=Path, default=DEFAULT_NORM_PATH)
    args = parser.parse_args()

    if args.epochs > 1000:
        raise ValueError("Task 9 forbids training beyond 1000 epochs.")

    ensure_dirs()
    set_seed(SEED)
    device = get_device()
    print(f"Device: {device}")
    print(f"Epoch cap override: {args.epochs}")
    loaded = load_dataset(batch_size=args.batch_size, norm_path=args.norm_path)
    print(f"Loaded {len(loaded.dataset)} windows across classes: {loaded.counts_by_class}")
    print(f"Train windows: {len(loaded.train_loader.dataset)} | Val windows: {len(loaded.val_loader.dataset)}")

    scratch_result = train_strategy(
        strategy_name="scratch",
        loaded=loaded,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        checkpoint_path=SCRATCH_CKPT_PATH,
        init_legacy_weights=False,
    )
    scratch_eval = None
    if scratch_result["status"] == "ok":
        scratch_eval = evaluate_checkpoint(SCRATCH_CKPT_PATH, loaded, device)
        scratch_result.update({
            "best_epoch": scratch_result["best_epoch"],
            "best_val_loss": scratch_result["best_val_loss"],
        })

    finetune_result = None
    finetune_eval = None
    should_try_finetune = scratch_result["status"] != "ok" or scratch_result["diverged_early"]
    if should_try_finetune:
        finetune_result = train_strategy(
            strategy_name="finetune",
            loaded=loaded,
            device=device,
            epochs=args.epochs,
            learning_rate=args.learning_rate * 0.1,
            checkpoint_path=FINETUNE_CKPT_PATH,
            init_legacy_weights=True,
        )
        if finetune_result["status"] == "ok":
            finetune_eval = evaluate_checkpoint(FINETUNE_CKPT_PATH, loaded, device)
            finetune_result.update({
                "best_epoch": finetune_result["best_epoch"],
                "best_val_loss": finetune_result["best_val_loss"],
            })
    else:
        print("[finetune] skipped: scratch training stayed finite through the first 50 epochs and T5 already flagged legacy omega mismatch.")

    candidates: list[dict[str, Any]] = []
    if scratch_eval is not None:
        scratch_result["evaluation"] = scratch_eval
        candidates.append(scratch_result)
    if finetune_eval is not None and finetune_result is not None:
        finetune_result["evaluation"] = finetune_eval
        candidates.append(finetune_result)

    winner = select_winner(candidates)
    winner_path = Path(winner["checkpoint_path"])
    shutil.copyfile(winner_path, BEST_CKPT_PATH)

    write_evidence(loaded, scratch_result, scratch_eval, finetune_result, finetune_eval, winner)
    write_decision(winner, scratch_attempted_only=not should_try_finetune, scratch_diverged=scratch_result["diverged_early"])

    print("Task 9 complete.")
    print(f"  scratch checkpoint: {SCRATCH_CKPT_PATH}")
    if finetune_result is not None and finetune_result.get("status") == "ok":
        print(f"  finetune checkpoint: {FINETUNE_CKPT_PATH}")
    print(f"  best checkpoint: {BEST_CKPT_PATH}")
    print(f"  winner: {winner['strategy']}")
    print(f"  silhouette_pca50: {winner['evaluation']['silhouette_pca50']:.6f}")
    print(f"  knn5_cv_accuracy_pca50: {winner['evaluation']['knn5_cv_accuracy_pca50']:.6f}")


if __name__ == "__main__":
    main()
