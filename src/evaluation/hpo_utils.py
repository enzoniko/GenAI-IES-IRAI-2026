import os
import sys
import optuna
import torch
import numpy as np


def create_or_load_study(
    study_name: str,
    direction: str,
    storage_path: str = "results/optuna_studies.db",
) -> optuna.Study:
    os.makedirs(os.path.dirname(os.path.abspath(storage_path)), exist_ok=True)
    storage = f"sqlite:///{os.path.abspath(storage_path)}"
    return optuna.create_study(
        study_name=study_name,
        direction=direction,
        storage=storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
    )


def log_trial_result(study, trial, metrics_dict: "dict[str, object]") -> None:
    trial_value = getattr(trial, "value", None)
    if trial_value is None:
        trial_value = metrics_dict.get("z_macro_mean_std")
    value_str = f"{float(trial_value):.6f}" if trial_value is not None else "pending"
    print(
        f"[Trial {trial.number}] value={value_str} "
        f"| params={trial.params} | metrics={metrics_dict}"
    )


def check_convergence(study, patience: int = 10) -> bool:
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(trials) < patience:
        return False
    best_in_last = (
        max(t.value for t in trials[-patience:])
        if study.direction == optuna.study.StudyDirection.MAXIMIZE
        else min(t.value for t in trials[-patience:])
    )
    best_overall = study.best_value
    return abs(best_in_last - best_overall) < 1e-6


def get_data_loaders(batch_size: int, val_split: float = 0.2, seed: int = 42):
    torch.manual_seed(seed)
    from src.data.mafaulda_dataset import get_dataloaders
    result = get_dataloaders(batch_size=batch_size, val_split=val_split)
    if result is None or result[0] is None:
        return None, None
    train_loader, val_loader, _ = result
    return train_loader, val_loader


def evaluate_z_macro_health(ts_jepa, val_loader, device) -> float:
    ts_jepa.eval()
    z_list = []
    with torch.no_grad():
        for batch in val_loader:
            raw = batch[0].to(device)
            z = ts_jepa.get_z_macro(raw)
            z_list.append(z.cpu())
    z_all = torch.cat(z_list, dim=0)
    per_dim_std = z_all.std(dim=0)
    return float(per_dim_std.mean())
