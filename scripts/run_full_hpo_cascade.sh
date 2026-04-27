#!/usr/bin/env bash
set -e

REPO_ROOT="/media/sf_GenAI/GenAI-IES-IRAI-2026"
VENV_PATH="/tmp/genai-venv/bin/activate"

# Full-run studies go to a NEW DB to avoid Optuna's CategoricalDistribution
# value-space conflict with the existing CPU-adapted studies in optuna_studies.db.
export FULL_DB="results/optuna_studies_full.db"

source "$VENV_PATH"
export PYTHONPATH="$REPO_ROOT"
cd "$REPO_ROOT"

mkdir -p logs

timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="logs/hpo_cascade_${timestamp}.log"

exec > >(tee -a "$log_file") 2>&1

echo "=== [$(date)] Starting full HPO cascade. Log: $log_file ==="
echo "=== Using separate DB for full runs: $FULL_DB (preserves CPU-adapted studies in optuna_studies.db) ==="

# Wrap the cascade in a function so that set -e properly exits the script
# if any step fails (avoiding bash's set-e + &&-chain top-level quirk).
run_cascade() {
    echo "=== [$(date)] Step 1/6: TS-JEPA HPO (30 trials, epochs=[50,100,150])... ==="
    python3 scripts/optuna_hpo_tsjepa.py --n_trials 30 --storage "$FULL_DB"

    echo "=== [$(date)] Step 2/6: Decoder1 HPO (20 trials, epochs=[50,100,150])... ==="
    python3 scripts/optuna_hpo_decoder1.py --n_trials 20 --storage "$FULL_DB"

    echo "=== [$(date)] Step 3/6: Decoder2 HPO (20 trials, epochs=[50,100,150])... ==="
    python3 scripts/optuna_hpo_decoder2.py --n_trials 20 --storage "$FULL_DB"

    echo "=== [$(date)] Step 4/6: LDM retrain on new z_macro distribution... ==="
    python3 train_ldm_task12.py

    echo "=== [$(date)] Syncing LDM checkpoint (ldm.pth -> ldm_hpo_retrained.pth)... ==="
    python3 - <<'PY'
from pathlib import Path
import shutil
src = Path("results/ldm.pth")
dst = Path("results/ldm_hpo_retrained.pth")
if not src.exists():
    raise FileNotFoundError(f"Missing LDM checkpoint: {src}")
shutil.copy2(src, dst)
print(f"Copied {src} -> {dst}")
PY

    echo "=== [$(date)] Step 5/6: SDEdit HPO (20 trials, num_inference_steps=[100,250,500,1000])... ==="
    python3 scripts/optuna_hpo_sdedit.py --n_trials 20 --storage "$FULL_DB"

    echo "=== [$(date)] Step 6/6: Final SDEdit evaluation... ==="
    python3 scripts/run_full_sdedit_evaluation.py
}

run_cascade

echo ""
echo "=== [$(date)] Final evaluation report ==="
python3 - <<'PY'
from pathlib import Path
report_path = Path("results/sdedit_evaluation_report.txt")
if report_path.exists():
    print(report_path.read_text(encoding="utf-8"))
else:
    print(f"Missing report: {report_path}")
PY

echo "=== [$(date)] Best HPO params from full-run DB ($FULL_DB) ==="
python3 - <<PY
import optuna, os

db = os.environ.get("FULL_DB", "results/optuna_studies_full.db")
storage = f"sqlite:///{os.path.abspath(db)}"
study_names = ["tsjepa-hpo", "decoder1-hpo", "decoder2-hpo", "sdedit-hpo"]

for name in study_names:
    print(f"[{name}]")
    try:
        study = optuna.load_study(study_name=name, storage=storage)
        print(f"  total_trials: {len(study.trials)}")
        print(f"  best_trial:   {study.best_trial.number}")
        print(f"  best_value:   {study.best_value}")
        print(f"  best_params:  {study.best_params}")
    except Exception as exc:
        print(f"  ERROR: {exc}")
PY

echo ""
echo "=== CASCADE COMPLETE. Check results/ for updated checkpoints and logs/$log_file for full output. ==="
