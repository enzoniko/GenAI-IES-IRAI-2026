"""Evaluate the point-wise PINN checkpoint against the windowed baseline."""

from __future__ import annotations

import json
import sys
from typing import TypedDict, cast
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.train_task9_pinn as task9
from src.configs import PINN_POINTWISE_ARCH
from src.models.pinn import ConfigurablePINN


BASELINE_SILHOUETTE = 0.0789
BASELINE_KNN = 0.7783
CHECKPOINT_PATH = ROOT / "results" / "pinn_pointwise.pth"
OUTPUT_PATH = ROOT / "results" / "pinn_pointwise_evaluation.json"
TRAINING_APPROACH = "point-wise normal-only 1000 samples"
EVAL_NORM_PATH = ROOT / "results" / "normalization_metadata_backup.pth"
PARAM_NAMES = ("M1", "M2", "M3", "D1", "D2", "D3", "K1", "K2", "E1")


class MLPConfig(TypedDict):
    hidden_layers: list[int]
    activation: str
    dropout_rate: float
    init_method: str


class ParamInitValues(TypedDict):
    M1: float
    M2: float
    M3: float
    D1: float
    D2: float
    D3: float
    K1: float
    K2: float
    E1: float


class ParamInitConfig(TypedDict):
    method: str
    values: ParamInitValues


class ArchConfig(TypedDict):
    unmeasured_net_config: MLPConfig
    acceleration_net_config: MLPConfig
    param_init_config: ParamInitConfig
    enable_mass_constraints: bool


class CheckpointDict(TypedDict):
    model_state_dict: dict[str, torch.Tensor]


class EvalMetrics(TypedDict):
    silhouette_pca50: float
    knn5_cv_accuracy_pca50: float


def build_pointwise_model() -> ConfigurablePINN:
    arch = cast(ArchConfig, cast(object, PINN_POINTWISE_ARCH))
    return ConfigurablePINN(
        unmeasured_net_config=arch["unmeasured_net_config"],
        acceleration_net_config=arch["acceleration_net_config"],
        param_init_config=arch["param_init_config"],
        enable_mass_constraints=arch["enable_mass_constraints"],
    )


def main() -> None:
    task9.build_model = build_pointwise_model

    device = task9.get_device()
    loaded = task9.load_dataset(batch_size=8, norm_path=EVAL_NORM_PATH)
    metrics_obj = task9.evaluate_checkpoint(CHECKPOINT_PATH, loaded, device)
    metrics = cast(EvalMetrics, cast(object, metrics_obj))

    checkpoint_obj = cast(object, torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False))
    if not isinstance(checkpoint_obj, dict):
        raise TypeError(f"Unexpected checkpoint type: {type(checkpoint_obj)!r}")
    checkpoint_dict = cast(dict[str, object], checkpoint_obj)
    state_dict_obj = checkpoint_dict.get("model_state_dict")
    if not isinstance(state_dict_obj, dict):
        raise TypeError("Checkpoint missing model_state_dict dictionary")
    checkpoint = cast(CheckpointDict, cast(object, {"model_state_dict": state_dict_obj}))
    state_dict = checkpoint["model_state_dict"]
    physical_params = {name: float(state_dict[name].reshape(-1)[0].item()) for name in PARAM_NAMES}

    new_silhouette = float(metrics["silhouette_pca50"])
    new_knn = float(metrics["knn5_cv_accuracy_pca50"])
    silhouette_delta = new_silhouette - BASELINE_SILHOUETTE
    knn_delta = new_knn - BASELINE_KNN
    go_no_go = "GO" if new_silhouette > BASELINE_SILHOUETTE else "NO-GO"
    go_no_go_reason = (
        f"GO because silhouette_pca50 improved from {BASELINE_SILHOUETTE:.4f} to {new_silhouette:.4f}."
        if go_no_go == "GO"
        else f"NO-GO because silhouette_pca50 did not exceed baseline {BASELINE_SILHOUETTE:.4f} (got {new_silhouette:.4f})."
    )

    payload = {
        "baseline_silhouette_pca50": BASELINE_SILHOUETTE,
        "baseline_knn5_cv_accuracy_pca50": BASELINE_KNN,
        "new_silhouette_pca50": new_silhouette,
        "new_knn5_cv_accuracy_pca50": new_knn,
        "silhouette_delta": silhouette_delta,
        "knn_delta": knn_delta,
        "go_no_go": go_no_go,
        "go_no_go_reason": go_no_go_reason,
        "pinn_checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
        "training_approach": TRAINING_APPROACH,
        "physical_params": physical_params,
        "old_metrics": {
            "silhouette_pca50": BASELINE_SILHOUETTE,
            "knn5_cv_accuracy_pca50": BASELINE_KNN,
        },
        "new_metrics": metrics,
        "deltas": {
            "silhouette_pca50": silhouette_delta,
            "knn5_cv_accuracy_pca50": knn_delta,
        },
        "decision": {
            "go_no_go": go_no_go,
            "reason": go_no_go_reason,
            "threshold": f"silhouette_pca50 > {BASELINE_SILHOUETTE:.4f}",
        },
    }

    _ = OUTPUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
