"""24.4 Hz PINN experiment: fresh normalization, scratch, and fine-tune."""

# pyright: reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportMissingTypeArgument=false, reportPrivateImportUsage=false, reportAny=false, reportUnknownArgumentType=false, reportArgumentType=false, reportUnusedCallResult=false, reportUnusedImport=false, reportUnusedVariable=false, reportUnannotatedClassAttribute=false, reportImplicitOverride=false, reportExplicitAny=false

from __future__ import annotations

import argparse
import json
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
from src.data.clean_mafaulda_processor import CleanMaFaulDaProcessor
from src.models.feature_extractors import MathFeatureExtractor
from src.models.pinn import ConfigurablePINN
from src.models.relobralo_loss import ReLoBRaLoLoss


RESULTS_DIR = ROOT / "results"
EVIDENCE_DIR = ROOT / ".sisyphus" / "evidence"
PROCESSED_DIR_24HZ = ROOT / "data" / "processed-mafaulda" / "24hz"
NORMALIZATION_24HZ_PATH = RESULTS_DIR / "normalization_metadata_24hz.pth"
LEGACY_WEIGHTS_PATH = ROOT / "previous-work-pinn" / "export_relobralo" / "weights.pth"
LEGACY_NORMALIZATION_PATH = ROOT / "previous-work-pinn" / "export_relobralo" / "normalization.npz"
FINETUNE_CKPT_PATH = RESULTS_DIR / "pinn_finetuned_24hz.pth"
SCRATCH_CKPT_PATH = RESULTS_DIR / "pinn_scratch_24hz.pth"
EVIDENCE_PATH = EVIDENCE_DIR / "task-exp-24hz-pinn.txt"
TARGET_HZ = 24.4
LOG_EPOCHS = {1, 10, 50, 100, 250, 500, 1000}
LOSS_KEYS = ["data", "phys_res1", "phys_res2", "phys_res3", "phys_res4", "phys_mass1", "phys_mass2"]
BASELINE_16HZ = {
    "silhouette_pca50": 0.0789,
    "knn_pca50": 0.7783,
}
CLASS_NAMES = {
    0: "normal",
    1: "imbalance_fault_20g",
    2: "vertical_misalignment_fault_1.27mm",
    3: "overhang_ball_fault_20g",
}
SEED = 42
ONE_SUBTYPE_FILES = {
    0: {
        "X": PROCESSED_DIR_24HZ / "X_normal_trainingset.pth",
        "Y": PROCESSED_DIR_24HZ / "Y_normal_trainingset.pth",
    },
    1: {
        "X": PROCESSED_DIR_24HZ / "X_imbalance_fault_20g_trainingset.pth",
        "Y": PROCESSED_DIR_24HZ / "Y_imbalance_fault_20g_trainingset.pth",
    },
    2: {
        "X": PROCESSED_DIR_24HZ / "X_vertical_misalignment_fault_1.27mm_trainingset.pth",
        "Y": PROCESSED_DIR_24HZ / "Y_vertical_misalignment_fault_1.27mm_trainingset.pth",
    },
    3: {
        "X": PROCESSED_DIR_24HZ / "X_overhang_ball_fault_20g_trainingset.pth",
        "Y": PROCESSED_DIR_24HZ / "Y_overhang_ball_fault_20g_trainingset.pth",
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
    seq_length: int


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


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)


def ensure_prerequisites() -> None:
    required_paths = [
        LEGACY_WEIGHTS_PATH,
        LEGACY_NORMALIZATION_PATH,
        ROOT / "data" / "raw-mafaulda" / "normal",
        ROOT / "data" / "raw-mafaulda" / "imbalance" / "20g",
        ROOT / "data" / "raw-mafaulda" / "vertical-misalignment" / "1.27mm",
        ROOT / "data" / "raw-mafaulda" / "overhang" / "ball_fault" / "20g",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required paths: {missing}")


def processed_files_exist() -> bool:
    return all(paths[side].exists() for paths in ONE_SUBTYPE_FILES.values() for side in ("X", "Y"))


def ensure_processed_data() -> None:
    if processed_files_exist():
        print(f"Processed 24 Hz tensors already exist in {PROCESSED_DIR_24HZ}")
        return

    processor = CleanMaFaulDaProcessor(
        raw_data_dir=cfg.DATA_DIR_RAW,
        processed_data_dir=cfg.DATA_DIR_PROCESSED,
        cutoff_hz=cfg.CUTOFF_HZ,
        strategy=cfg.SIGNAL_PROCESSING_STRATEGY,
    )
    processor.run(
        target_hz=TARGET_HZ,
        training_windows=cfg.PHASE0_TRAIN_SETTINGS["training_windows"],
        test_windows=cfg.PHASE0_TRAIN_SETTINGS["test_windows"],
    )

    missing = [str(paths[side]) for paths in ONE_SUBTYPE_FILES.values() for side in ("X", "Y") if not paths[side].exists()]
    if missing:
        raise FileNotFoundError(f"Expected processed 24 Hz tensors missing after processing: {missing}")


def compute_and_save_normalization() -> dict[str, torch.Tensor]:
    X_normal = torch.load(ONE_SUBTYPE_FILES[0]["X"], map_location="cpu", weights_only=True).double()
    Y_normal = torch.load(ONE_SUBTYPE_FILES[0]["Y"], map_location="cpu", weights_only=True).double()

    metadata = {
        "X_min": X_normal[:, :, :8].reshape(-1, 8).min(dim=0).values,
        "X_max": X_normal[:, :, :8].reshape(-1, 8).max(dim=0).values,
        "y_min": Y_normal.reshape(-1, 4).min(dim=0).values,
        "y_max": Y_normal.reshape(-1, 4).max(dim=0).values,
    }
    torch.save(metadata, NORMALIZATION_24HZ_PATH)
    return metadata


def _load_norm_metadata(norm_path: Path) -> dict[str, torch.Tensor]:
    metadata = torch.load(norm_path, map_location="cpu", weights_only=True)
    required = {"X_min", "X_max", "y_min", "y_max"}
    missing = required.difference(metadata)
    if missing:
        raise KeyError(f"Normalization metadata missing keys: {sorted(missing)}")
    loaded = {key: metadata[key].double() for key in required}
    if loaded["X_min"].shape != torch.Size([8]) or loaded["X_max"].shape != torch.Size([8]):
        raise ValueError("Expected X_min/X_max shape [8] for 24 Hz normalization metadata.")
    if loaded["y_min"].shape != torch.Size([4]) or loaded["y_max"].shape != torch.Size([4]):
        raise ValueError("Expected y_min/y_max shape [4] for 24 Hz normalization metadata.")
    return loaded


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

    seq_length: int | None = None

    for label, paths in ONE_SUBTYPE_FILES.items():
        X_raw = torch.load(paths["X"], map_location="cpu", weights_only=True).double()
        Y_raw = torch.load(paths["Y"], map_location="cpu", weights_only=True).double()
        if seq_length is None:
            seq_length = int(X_raw.shape[1])
        if X_raw.shape[1] != seq_length or Y_raw.shape[1] != seq_length:
            raise ValueError("All 24 Hz tensors must share the same sequence length.")

        labels = torch.full((X_raw.shape[0],), label, dtype=torch.long)
        counts_by_class[label] = int(X_raw.shape[0])
        omega_all.append(X_raw[:, :, 8].reshape(-1))
        time_all.append(X_raw[:, :, 9].reshape(-1))

        X_chunks.append(X_raw)
        Y_chunks.append(Y_raw)
        label_chunks.append(labels)

    if seq_length is None:
        raise RuntimeError("No 24 Hz tensors were loaded.")

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
        seq_length=seq_length,
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
        raise RuntimeError(f"Strategy '{strategy_name}' failed to produce a checkpoint.")

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


def build_comparison_rows(scratch_eval: dict[str, Any], finetune_eval: dict[str, Any]) -> list[dict[str, float | str]]:
    return [
        {
            "strategy": "16hz_scratch_baseline",
            "silhouette_pca50": BASELINE_16HZ["silhouette_pca50"],
            "knn_pca50": BASELINE_16HZ["knn_pca50"],
        },
        {
            "strategy": "24hz_scratch",
            "silhouette_pca50": scratch_eval["silhouette_pca50"],
            "knn_pca50": scratch_eval["knn5_cv_accuracy_pca50"],
        },
        {
            "strategy": "24hz_finetune",
            "silhouette_pca50": finetune_eval["silhouette_pca50"],
            "knn_pca50": finetune_eval["knn5_cv_accuracy_pca50"],
        },
    ]


def write_evidence(
    loaded: LoadedDataset,
    scratch_result: dict[str, Any],
    scratch_eval: dict[str, Any],
    finetune_result: dict[str, Any],
    finetune_eval: dict[str, Any],
) -> None:
    comparison_rows = build_comparison_rows(scratch_eval, finetune_eval)
    winner = max(
        [
            {"strategy": "scratch", "eval": scratch_eval},
            {"strategy": "finetune", "eval": finetune_eval},
        ],
        key=lambda item: (item["eval"]["silhouette_pca50"], item["eval"]["knn5_cv_accuracy_pca50"]),
    )

    lines: list[str] = []
    lines.append("24.4 Hz PINN Experiment Evidence")
    lines.append("")
    lines.append("Dataset info")
    lines.append(json.dumps({
        "target_hz": TARGET_HZ,
        "seq_length": loaded.seq_length,
        "counts_by_class": {CLASS_NAMES[key]: value for key, value in loaded.counts_by_class.items()},
        "total_windows": int(len(loaded.dataset)),
        "train_windows": int(len(loaded.train_loader.dataset)),
        "val_windows": int(len(loaded.val_loader.dataset)),
        "omega_stats": loaded.omega_stats,
        "time_stats": loaded.time_stats,
        "normalization_path": str(NORMALIZATION_24HZ_PATH),
        "x_min_full": loaded.x_bounds_full["X_min"].tolist(),
        "x_max_full": loaded.x_bounds_full["X_max"].tolist(),
        "y_min": loaded.y_bounds["y_min"].tolist(),
        "y_max": loaded.y_bounds["y_max"].tolist(),
    }, indent=2))
    lines.append("")
    lines.append("Strategy results")
    lines.append("")
    lines.append("finetune")
    lines.append(json.dumps({"result": finetune_result, "evaluation": finetune_eval}, indent=2))
    lines.append("")
    lines.append("scratch")
    lines.append(json.dumps({"result": scratch_result, "evaluation": scratch_eval}, indent=2))
    lines.append("")
    lines.append("Comparison vs 16 Hz baseline")
    lines.append("strategy | silhouette_pca50 | kNN_pca50")
    lines.append("--- | ---: | ---:")
    for row in comparison_rows:
        lines.append(f"{row['strategy']} | {row['silhouette_pca50']:.6f} | {row['knn_pca50']:.6f}")
    lines.append("")
    lines.append("Winner")
    winner_silhouette = float(winner["eval"]["silhouette_pca50"])
    winner_knn = float(winner["eval"]["knn5_cv_accuracy_pca50"])
    lines.append(json.dumps({
        "strategy": winner["strategy"],
        "silhouette_pca50": winner_silhouette,
        "knn_pca50": winner_knn,
        "beats_16hz_baseline": bool(
            winner_silhouette > BASELINE_16HZ["silhouette_pca50"] and winner_knn > BASELINE_16HZ["knn_pca50"]
        ),
    }, indent=2))
    EVIDENCE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_checkpoint_for_validation(checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    required = {"model_state_dict", "X_min", "X_max", "y_min", "y_max", "strategy", "best_epoch", "best_val_loss"}
    missing = required.difference(checkpoint)
    if missing:
        raise KeyError(f"Checkpoint {checkpoint_path} missing keys: {sorted(missing)}")
    model = build_model().double()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return checkpoint


def run_check_only() -> None:
    ensure_prerequisites()
    for paths in ONE_SUBTYPE_FILES.values():
        torch.load(paths["X"], map_location="cpu", weights_only=True)
        torch.load(paths["Y"], map_location="cpu", weights_only=True)
    _load_norm_metadata(NORMALIZATION_24HZ_PATH)
    load_checkpoint_for_validation(FINETUNE_CKPT_PATH)
    load_checkpoint_for_validation(SCRATCH_CKPT_PATH)
    print("CHECK PASSED")


def main() -> None:
    parser = argparse.ArgumentParser(description="24.4 Hz PINN fine-tune vs scratch experiment")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=cfg.PHASE0_TRAIN_SETTINGS["learning_rate"])
    parser.add_argument("--norm-path", type=Path, default=NORMALIZATION_24HZ_PATH)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    if args.epochs > 1000:
        raise ValueError("24 Hz experiment forbids training beyond 1000 epochs.")

    if args.check_only:
        run_check_only()
        return

    ensure_dirs()
    ensure_prerequisites()
    set_seed(SEED)
    device = get_device()

    print(f"Device: {device}")
    print(f"Target Hz: {TARGET_HZ}")
    print(f"Epoch cap override: {args.epochs}")

    ensure_processed_data()
    compute_and_save_normalization()
    loaded = load_dataset(batch_size=args.batch_size, norm_path=args.norm_path)
    print(f"Loaded {len(loaded.dataset)} windows across classes: {loaded.counts_by_class}")
    print(f"Train windows: {len(loaded.train_loader.dataset)} | Val windows: {len(loaded.val_loader.dataset)}")
    print(f"SEQ_LENGTH (from data): {loaded.seq_length}")

    finetune_result = train_strategy(
        strategy_name="finetune",
        loaded=loaded,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate * 0.1,
        checkpoint_path=FINETUNE_CKPT_PATH,
        init_legacy_weights=True,
    )
    finetune_eval = evaluate_checkpoint(FINETUNE_CKPT_PATH, loaded, device)
    print(f"[finetune] Silhouette_pca50={finetune_eval['silhouette_pca50']:.6f} | kNN_pca50={finetune_eval['knn5_cv_accuracy_pca50']:.6f}")

    set_seed(SEED)
    scratch_result = train_strategy(
        strategy_name="scratch",
        loaded=loaded,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        checkpoint_path=SCRATCH_CKPT_PATH,
        init_legacy_weights=False,
    )
    scratch_eval = evaluate_checkpoint(SCRATCH_CKPT_PATH, loaded, device)
    print(f"[scratch] Silhouette_pca50={scratch_eval['silhouette_pca50']:.6f} | kNN_pca50={scratch_eval['knn5_cv_accuracy_pca50']:.6f}")

    write_evidence(loaded, scratch_result, scratch_eval, finetune_result, finetune_eval)

    print("24.4 Hz PINN experiment complete.")
    print(f"  normalization: {NORMALIZATION_24HZ_PATH}")
    print(f"  finetune checkpoint: {FINETUNE_CKPT_PATH}")
    print(f"  scratch checkpoint: {SCRATCH_CKPT_PATH}")
    print(f"  evidence: {EVIDENCE_PATH}")


if __name__ == "__main__":
    main()
