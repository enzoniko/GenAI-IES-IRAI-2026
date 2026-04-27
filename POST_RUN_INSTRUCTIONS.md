# Post-Run Instructions

## 1. Confirm the cascade finished successfully

```bash
tail -50 logs/hpo_cascade_*.log
cat results/sdedit_evaluation_report.txt
ls -la results/*.pth
```

Success indicators:
- the cascade log ends with `CASCADE COMPLETE`
- `results/sdedit_evaluation_report.txt` exists and contains TSTR + MMD metrics
- refreshed checkpoints exist, especially:
  - `results/ts_jepa_hpo_best.pth`
  - `results/decoder1_hpo_best.pth`
  - `results/decoder2_hpo_best.pth`
  - `results/ldm_hpo_retrained.pth`
  - `results/ldm.pth`

## 2. Inspect best HPO params from the Optuna DB

Activate the environment first:

```bash
source /tmp/genai-venv/bin/activate
export PYTHONPATH=/media/sf_GenAI/GenAI-IES-IRAI-2026
cd /media/sf_GenAI/GenAI-IES-IRAI-2026
```

Query all studies:

```bash
python3 - <<'PY'
import optuna

storage = "sqlite:///results/optuna_studies.db"
for name in ["tsjepa-hpo", "decoder1-hpo", "decoder2-hpo", "sdedit-hpo"]:
    study = optuna.load_study(study_name=name, storage=storage)
    print(f"\n{name}")
    print(f"  best_trial: {study.best_trial.number}")
    print(f"  best_value: {study.best_value}")
    print(f"  best_params: {study.best_params}")
PY
```

If the scripts did **not** auto-update `src/configs.py`, manually copy these values into:
- `JEPA_CONFIG`
  - `patch_size`
  - `ema_decay`
- `SDEDIT_GUIDANCE_SETTINGS`
  - `guidance_scale`
  - `strength`
  - `num_inference_steps`

Also confirm:
- `ORACLE_MODE = "raw"` remains in place for the current evaluation path
- downstream evaluation/HPO scripts read `results/ldm_hpo_retrained.pth`; the cascade runner syncs this file from the freshly retrained `results/ldm.pth`

Reference file:

```bash
python3 - <<'PY'
from pathlib import Path
print(Path('src/configs.py').read_text(encoding='utf-8'))
PY
```

Notes:
- the HPO scripts save best params into checkpoints and the Optuna DB
- `create_or_load_study(..., load_if_exists=True)` means reruns **resume** existing studies
- `study.optimize(..., n_trials=N)` adds **N more trials**, not total-trial cap behavior

## 3. Recompile the paper-ready results summary

Primary sources to read:
- `results/sdedit_evaluation_report.txt`
- `results/optuna_studies.db`
- checkpoint `best_params` payloads if needed

Useful checkpoint inspection command:

```bash
python3 - <<'PY'
import torch

for path in [
    'results/ts_jepa_hpo_best.pth',
    'results/decoder1_hpo_best.pth',
    'results/decoder2_hpo_best.pth',
]:
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    print(f"\n{path}")
    print(ckpt.get('best_params'))
PY
```

Update `.sisyphus/evidence/results-summary.md` with:
- TS-JEPA best HPO params
- Decoder1 best HPO params
- Decoder2 best HPO params
- SDEdit best HPO params
- final TSTR/oracle ratio
- `mmd_global`
- `mmd_class_1`
- `mmd_class_2`
- `mmd_class_3`
- any refreshed `z_macro` health metric

Template fields to replace:

```md
### TS-JEPA HPO Results
- lr: <value>
- batch_size: <value>
- epochs: <value>
- patch_size: <value>
- ema_decay: <value>
- z_macro per-dim std: <value>

### SDEdit HPO Results
- guidance_scale: <value>
- strength: <value>
- num_inference_steps: <value>
- Best MMD: <value>

### Full Evaluation Metrics
- oracle_accuracy: <value>
- tstr_accuracy: <value>
- TSTR/Oracle ratio: <value>
- mmd_global: <value>
- mmd_class_1: <value>
- mmd_class_2: <value>
- mmd_class_3: <value>
```

## 4. Run the synthetic pipeline with the new params

```bash
source /tmp/genai-venv/bin/activate && PYTHONPATH=/media/sf_GenAI/GenAI-IES-IRAI-2026 python3 main_synthetic.py
```

## 5. Re-run only the final evaluation

```bash
source /tmp/genai-venv/bin/activate && PYTHONPATH=/media/sf_GenAI/GenAI-IES-IRAI-2026 python3 scripts/run_full_sdedit_evaluation.py
```

## 6. Expected runtime

On CPU this cascade will likely take many hours.

Rule-of-thumb expectation:
- TS-JEPA HPO: long multi-hour run
- Decoder1 HPO: multi-hour run
- Decoder2 HPO: multi-hour run
- LDM retrain: additional significant time
- SDEdit HPO with 100/250/500/1000 steps: potentially the slowest stage
- final evaluation: additional substantial time

Expect an overnight-to-day-long run on CPU-only hardware.

## 7. Evidence and outputs likely updated by the run

Likely overwritten or refreshed:
- `results/sdedit_evaluation_report.txt`
- `results/umap_sdedit_final_evaluation.png`
- `results/ts_jepa_hpo_best.pth`
- `results/decoder1_hpo_best.pth`
- `results/decoder2_hpo_best.pth`
- `results/ldm.pth`
- `results/ldm_hpo_retrained.pth`
- `logs/hpo_cascade_*.log`
- `.sisyphus/evidence/task-14-generation-count.txt`
- `.sisyphus/evidence/task-14-tstr-mmd-metrics.txt`
- `.sisyphus/evidence/task-14-final-umap.txt`
- `.sisyphus/evidence/task-7-*.txt`
- `.sisyphus/evidence/task-11-*.txt`
- `.sisyphus/evidence/task-12-*.txt`
- `.sisyphus/evidence/task-13-*.txt`
- `results/optuna_studies.db`

If you maintain paper evidence manually, also refresh any task-specific summary files that quote old CPU-adapted numbers.
