# IRAI 2026 Pipeline Review, Fix & Complete Results Generation

## TL;DR

> **Quick Summary**: Review and fix the physics-guided counterfactual fault synthesis pipeline (PINN training issues, integration bugs), then generate complete paper-ready results: component validation, UMAP visualizations, physics-guided SDEdit generation, baseline comparisons, and transfer classification — for both synthetic and MaFaulDa (16Hz, 3 fault classes).
> 
> **Deliverables**:
> - Fixed, validated pipeline (Phase 0→1→2) running on real MaFaulDa data
> - PINN with clustering residual features (old weights / fine-tuned / from-scratch comparison)
> - Complete experimental results: component validation, UMAP, SDEdit, baselines, transfer classification
> - 3-4 paper figures (twin plots: synthetic + MaFaulDa) + 1-2 results tables
> - Updated LaTeX paper with filled-in Results and Discussion sections
> 
> **Estimated Effort**: Large (5-7 days)
> **Parallel Execution**: YES - 6 waves
> **Critical Path**: T1 → T5 → T9 → T15 → T16 → T17 → T18 (Data → PINN → SDEdit → Transfer → Figures → Paper)

---

## Context

### Original Request
Review Gustavo's codebase changes connecting real MaFaulDa data to the IRAI pipeline, debug PINN training (high loss, poor residual clustering), fix integration issues, and generate complete paper results with baselines for the conference submission and GenAI Hackathon.

### Interview Summary
**Key Discussions**:
- **Timeline**: ~1 week soft deadline. Must triage ruthlessly but need complete results (competitive for hackathon)
- **Fault scope**: 3 fault classes + healthy at 16Hz: Imbalance, Vertical Misalignment, Overhang Ball Bearing (aligning with previous paper)
- **PINN issue**: Loss kinda high, residuals not clustering well by fault class
- **Critical discovery**: `normalization.npz` EXISTS in `previous-work-pinn/export_relobralo/` with original Xmin, Xmax, ymin, ymax — old weights CAN be loaded properly
- **Old PINN**: ConfigurablePINN with [128,128] hidden layers, Tanh, float64 — need to verify Gustavo's architecture matches
- **Baselines**: Neither implemented. Paper proposes Vanilla DDPM + Label-conditioned diffusion. Also need to reason about best baseline choices
- **Synthetic**: Need fresh results with improved code; validate fairness/difficulty
- **Compute**: Consumer GPU (12-24GB VRAM)

**Research Findings**:
- Project has 3 phases: Phase 0 (PINN + data processing), Phase 1 (TS-JEPA + decoders), Phase 2 (SDEdit guided by PINN)
- Gustavo's changes (Apr 10-15): centralized configs, Strategy D differentiation, wavelet features, early stopping, 42 files changed, 2090 insertions
- MathFeatureExtractor outputs 2240-dim embeddings — potentially too high for small sample counts (curse of dimensionality may explain poor clustering)
- PINN uses float64, other models float32 — dtype crossing boundaries need explicit casts
- torchaudio.functional.lfilter used for IIR filtering — dependency risk
- TS-JEPA training is INDEPENDENT of PINN — major parallelism opportunity

### Metis Review
**Identified Gaps** (addressed in plan):
- **PINN success gate missing**: Added quantitative criteria (Silhouette > 0.3, kNN accuracy > 70%) to T10
- **normalization.npz compatibility unvalidated**: Made T2 the critical first validation — verify Strategy D output ranges vs old normalization bounds
- **No PINN fallback strategy**: Added explicit fallback in T9 (continuous residuals without clustering, simpler physics proxy, or reframe claim)
- **Missing smoke tests**: Each training task now has a fast sanity check step
- **2240-dim feature space concern**: T10 now includes dimensionality reduction investigation (PCA/feature selection before UMAP)
- **Severity level handling**: T1 processing explicitly addresses how severity subtypes are handled within the 3 chosen classes
- **UMAP non-determinism**: All UMAP calls now fix random_state=42 and report it
- **Time-boxing**: PINN debugging (T5+T9) gets 2-day hard limit with explicit fallback
- **Baseline fairness**: Same hyperparameter budget, training epochs, and data for all methods
- **Class imbalance**: T1 checks class distribution and applies stratification if needed
- **Multiple seeds**: Transfer classification (T16) runs 5 seeds with mean±std reporting

---

## Work Objectives

### Core Objective
Debug and fix the IRAI pipeline to produce physically meaningful PINN residual features that cluster by fault class, validate all Phase 1/2 components on real MaFaulDa data, run baselines and transfer classification, and generate complete paper-ready figures and tables.

### Concrete Deliverables
- Fixed `src/models/pinn.py` and `src/models/oracles.py` producing discriminative residual embeddings
- Trained models: PINN, TS-JEPA, Decoder 1, Decoder 2, LDM — all saved as checkpoints
- UMAP visualizations of PINN residual feature space showing fault clusters
- SDEdit-generated counterfactual traces for 3 fault classes
- Baseline comparison results (Vanilla DDPM + Label-conditioned diffusion)
- Transfer classification table (train-synthetic-test-real) with 5-seed statistics
- 3-4 composite paper figures (twin plots) + 1-2 results tables
- Updated `main_M2_6_pages.tex` with complete Results and Discussion sections

### Definition of Done
- [ ] `python main_phase0.py` completes without error, PINN residuals produce Silhouette score > 0.3 on 4-class UMAP
- [ ] `python main_phase1.py` produces TS-JEPA + Decoder 1 + Decoder 2 checkpoints with Decoder 1 envelope RMSE documented
- [ ] `python main_phase2.py` generates counterfactual traces that visually align with real fault signals
- [ ] Transfer classification TSTR accuracy > 50% (4-class, significantly above 25% chance)
- [ ] All paper figures saved in `assets/` and referenced in LaTeX
- [ ] Paper compiles with `pdflatex main_M2_6_pages.tex` without errors

### Must Have
- PINN residual features that visibly cluster by fault class on UMAP (quantitative: Silhouette > 0.3)
- Working end-to-end pipeline (data → PINN → TS-JEPA → decoders → LDM → SDEdit → counterfactuals)
- At least 2 baselines for comparison (Vanilla DDPM + Label-conditioned)
- Transfer classification results with statistical significance (5 seeds)
- Twin plots (synthetic left, MaFaulDa right) for key results
- Reproducibility: fixed random seeds, logged hyperparameters, saved configs

### Must NOT Have (Guardrails)
- **No architecture experimentation**: Use architectures as described in the paper. No "let's try Transformer instead of TS-JEPA"
- **No cosmetic refactoring**: Only modify code that directly blocks pipeline from producing results. No reorganizing imports, adding type hints, or renaming variables for style
- **No baseline proliferation**: Exactly 2 baselines (Vanilla DDPM + Label-conditioned). Additional baselines are future work
- **No multi-speed generalization**: 16Hz only. Variable speed is explicitly deferred in paper
- **No hyperparameter sweeps beyond budget**: PINN gets max 5 configs, other components max 3
- **No excessive PINN debugging**: 2-day time-box. If clustering fails, execute fallback strategy
- **No silent dtype coercion**: Every float64↔float32 boundary must be explicit and logged
- **No undocumented training runs**: Every run logs seed, config, git hash, final metrics

---

## Verification Strategy (MANDATORY)

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: YES (tests/ directory with test_pinn_predictions.py, test_oracle.py, test_math_extractor.py)
- **Automated tests**: Tests-after (run existing tests + add smoke tests for each component)
- **Framework**: pytest
- **Verification method**: Experiments ARE the verification — UMAP clustering, transfer classification accuracy, visual signal comparison

### QA Policy
Every task MUST include agent-executed QA scenarios. Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **Data processing**: Bash — verify tensor shapes, dtype, value ranges, class distribution
- **PINN training**: Bash — verify loss convergence, residual magnitudes, UMAP clustering metric
- **Component training**: Bash — verify checkpoint files, reconstruction quality metrics
- **SDEdit generation**: Bash — verify generated trace shapes, UMAP trajectory, signal comparison plots
- **Baselines + Classification**: Bash — verify accuracy tables, statistical significance

---

## Execution Strategy

### Parallel Execution Waves

> TS-JEPA training is INDEPENDENT of PINN — this enables significant parallelism.
> Critical path: Data → PINN Fix → SDEdit → Transfer Classification → Figures → LaTeX.
> JEPA chain runs in parallel: Data → TS-JEPA → Decoders → (merges at SDEdit).

```
Wave 1 (Start Immediately — foundation + validation):
├── Task 1: Environment setup + MaFaulDa download + data processing [quick]
├── Task 2: Validation shim — norm compat, arch match, pipeline health [quick]
├── Task 3: Code audit of critical pipeline files + fix blocking issues [deep]
└── Task 4: Paper deliverables specification (figures, tables, metrics) [writing]

Wave 2 (After Wave 1 — parallel PINN diagnostic + TS-JEPA + prep):
├── Task 5: PINN diagnostic — load old weights, evaluate on new data [deep]
├── Task 6: TS-JEPA encoder training on MaFaulDa (INDEPENDENT of PINN!) [unspecified-high]
├── Task 7: Implement baseline code (Vanilla DDPM + Label-conditioned) [unspecified-high]
└── Task 8: Synthetic system re-run + fairness/difficulty validation [unspecified-high]

Wave 3 (After Wave 2 — PINN fix + Decoder 1 + LDM in parallel):
├── Task 9: PINN training — fix, fine-tune, from-scratch (TIME-BOXED 2 days) [deep]
├── Task 10: PINN variant comparison + UMAP + selection (depends: T9) [unspecified-high]
├── Task 11: Decoder 1 training (depends: T6 TS-JEPA frozen) [unspecified-high]
└── Task 12: LDM training on z_macro (depends: T6 TS-JEPA frozen) [unspecified-high]

Wave 4 (After Wave 3 — Decoder 2 + validation + baselines):
├── Task 13: Decoder 2 CVAE training (depends: T6, T11) [unspecified-high]
├── Task 14: Phase 1 validation + twin plots (depends: T11, T13) [unspecified-high]
└── Task 15: Run baseline experiments (depends: T7, T12) [unspecified-high]

Wave 5 (After Wave 4 — generation + evaluation):
├── Task 16: Physics-guided SDEdit generation (depends: T9/10, T11, T12, T13) [deep]
├── Task 17: Transfer classification experiment (depends: T15, T16) [unspecified-high]
└── Task 18: Generate paper figures + tables + update LaTeX (depends: all) [writing]

Wave FINAL (After ALL tasks — 4 parallel reviews, then user okay):
├── Task F1: Plan compliance audit (oracle)
├── Task F2: Code quality review (unspecified-high)
├── Task F3: Real manual QA (unspecified-high)
└── Task F4: Scope fidelity check (deep)
-> Present results -> Get explicit user okay

Critical Path: T1 → T5 → T9 → T10 → T16 → T17 → T18 → F1-F4 → user okay
Parallel Speedup: ~40% faster than sequential (TS-JEPA + Decoders chain runs alongside PINN chain)
Max Concurrent: 4 (Waves 1 & 2)
```

### Dependency Matrix

| Task | Depends On | Blocks | Wave |
|------|-----------|--------|------|
| T1 | — | T2-T8 | 1 |
| T2 | T1 | T5, T9 | 1 |
| T3 | T1 | T9 | 1 |
| T4 | — | T18 | 1 |
| T5 | T1, T2 | T9, T10 | 2 |
| T6 | T1 | T11, T12, T13 | 2 |
| T7 | — | T15 | 2 |
| T8 | T1 | T18 | 2 |
| T9 | T3, T5 | T10, T16 | 3 |
| T10 | T9 | T16 | 3 |
| T11 | T6 | T13, T14 | 3 |
| T12 | T6 | T15, T16 | 3 |
| T13 | T6, T11 | T14, T16 | 4 |
| T14 | T11, T13 | T18 | 4 |
| T15 | T7, T12 | T17 | 4 |
| T16 | T9/10, T11, T12, T13 | T17 | 5 |
| T17 | T15, T16 | T18 | 5 |
| T18 | T4, T8, T10, T14, T17 | F1-F4 | 5 |

### Agent Dispatch Summary

- **Wave 1**: **4** — T1 → `quick`, T2 → `quick`, T3 → `deep`, T4 → `writing`
- **Wave 2**: **4** — T5 → `deep`, T6 → `unspecified-high`, T7 → `unspecified-high`, T8 → `unspecified-high`
- **Wave 3**: **4** — T9 → `deep`, T10 → `unspecified-high`, T11 → `unspecified-high`, T12 → `unspecified-high`
- **Wave 4**: **3** — T13 → `unspecified-high`, T14 → `unspecified-high`, T15 → `unspecified-high`
- **Wave 5**: **3** — T16 → `deep`, T17 → `unspecified-high`, T18 → `writing`
- **FINAL**: **4** — F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

> Implementation + Test = ONE Task. Never separate.
> EVERY task MUST have: Recommended Agent Profile + Parallelization info + QA Scenarios.
> **A task WITHOUT QA Scenarios is INCOMPLETE. No exceptions.**

### Wave 1 — Foundation & Validation

- [x] 1. Environment Setup + MaFaulDa Download + Data Processing

  **What to do**:
  - Install all Python dependencies from requirements (torch, torchaudio, scipy, scikit-learn, umap-learn, matplotlib, etc.). Verify GPU is available via `torch.cuda.is_available()`
  - **CRITICAL CONFIG FIX (must be done FIRST)**: The current `src/configs.py` has `TARGET_HZ = 17.0`, `NUM_CLASSES = 42`, and `rotation_hz = 30` — all wrong for our paper. Update `src/configs.py`:
    - Set `TARGET_HZ = 16.0`
    - Set `NUM_CLASSES = 4` (Normal=0, Imbalance=1, Vertical Misalignment=2, Overhang Ball Bearing=3)
    - Update `rotation_hz` to `16` in the synthetic config section
    - Update `DATA_DIR_PROCESSED` path to reflect 16Hz
    - Update the fault class label mapping — the current 42-class expanded mapping in `src/data/mafaulda_dataset.py` must be collapsed to the 4 target classes. Severity subtypes within each fault class should be merged (e.g., all imbalance severity levels → class 1)
  - Download MaFaulDa dataset using `scripts/download_dataset.sh -normal -imbalance -vertical -overhang -unzip` (only the 4 needed conditions — healthy + 3 fault classes). Target directory: `data/raw-mafaulda/`. **NOTE**: Script flags are `-vertical` NOT `-vertical-misalignment`
  - Run `python tools/process_mafaulda.py` (or equivalent processing entry point) to produce processed tensors. If no standalone script exists, run `python main_phase0.py` with PINN training disabled/skipped (processing only)
  - Verify class distribution: count samples per class, check for severe imbalance, log exact counts
  - Verify processed tensor shapes: expected 4-channel signals (underhang radial, underhang tangential, overhang radial, overhang tangential), time-aligned, at correct sample rate
  - Verify value ranges: accelerations should be physically reasonable (not all zeros, no NaN/Inf, no extreme outliers)
  - Save a summary of: number of samples per class, signal length, sample rate, feature dimensions

  **Must NOT do**:
  - Do NOT download fault conditions beyond the 4 needed (no horizontal misalignment, no underhang bearing)
  - Do NOT modify the data processing ALGORITHM in `clean_mafaulda_processor.py` (Strategy D logic, filters, integration) — only update CONFIGURATION values in `configs.py` and class label mapping in `mafaulda_dataset.py`
  - Do NOT change rotation speed from 16Hz

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Downloading data and running existing scripts — no complex logic needed
  - **Skills**: []
    - No specialized skills needed — this is shell commands + Python script execution
  - **Skills Evaluated but Omitted**:
    - `playwright`: No browser interaction needed — data download is via shell script

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T2, T3, T4)
  - **Parallel Group**: Wave 1 (with Tasks 2, 3, 4)
  - **Blocks**: T2, T3, T5, T6, T8 (all downstream tasks need processed data)
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References**:
  - `scripts/download_dataset.sh` — Download script. Supported flags: `-normal -imbalance -vertical -overhang -unzip`. NOTE: use `-vertical` NOT `-vertical-misalignment`
  - `src/data/clean_mafaulda_processor.py` — The data processing pipeline. "Strategy D" differentiation: acceleration → velocity → position via numerical integration with drift mitigation. Do NOT modify the algorithm — only config values
  - `src/configs.py` — Central config. MUST be updated: `TARGET_HZ = 17.0` → `16.0`, `NUM_CLASSES = 42` → `4`, `rotation_hz: 30` → `16`. Also check `DATA_DIR_PROCESSED` path
  - `src/data/mafaulda_dataset.py` — MaFaulDa DataLoader class. Contains the 42-class expanded label mapping that must be collapsed to 4 classes. Check `__init__` for label mapping dict

  **API/Type References**:
  - `src/data/mafaulda_dataset.py` — MaFaulDa DataLoader class. Check its `__init__` for expected directory structure and file format

  **External References**:
  - README.md — Dataset download instructions and directory structure expectations

  **WHY Each Reference Matters**:
  - `download_dataset.sh`: Provides the exact flags to download only needed conditions — avoid downloading 20GB of unnecessary data
  - `clean_mafaulda_processor.py`: This is the code that transforms raw CSVs into tensors — its output shapes/dtypes are what ALL downstream components expect
  - `configs.py`: Single source of truth for hyperparameters — must verify 16Hz and correct fault class mapping before any training

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Config updated to 16Hz / 4-class and data download completes
    Tool: Bash
    Preconditions: src/configs.py and src/data/mafaulda_dataset.py accessible
    Steps:
      1. Update src/configs.py: set TARGET_HZ=16.0, NUM_CLASSES=4, rotation_hz=16
      2. Update src/data/mafaulda_dataset.py: collapse 42-class mapping to 4 classes (Normal=0, Imbalance=1, VertMisalign=2, OverhangBB=3)
      3. Run: bash scripts/download_dataset.sh -normal -imbalance -vertical -overhang -unzip
      4. Verify directory exists: ls data/raw-mafaulda/
      5. Count subdirectories: should see normal/, imbalance/, vertical-misalignment/, overhang/ (4 directories)
      6. Verify at least one .csv file exists in each subdirectory
    Expected Result: Config updated. 4 condition directories present, each containing CSV files
    Failure Indicators: Config update breaks imports, missing directories, download errors
    Evidence: .sisyphus/evidence/task-1-config-update.txt, .sisyphus/evidence/task-1-data-download.txt

  Scenario: Data processing produces valid tensors for all 4 classes
    Tool: Bash (Python one-liner or short script)
    Preconditions: Raw data downloaded successfully
    Steps:
      1. Run data processing (main_phase0.py with PINN training skipped, or tools/process_mafaulda.py)
      2. Load processed data and print: tensor shapes, dtypes, value ranges (min/max/mean/std per channel), sample counts per class
      3. Assert: no NaN or Inf values in any tensor
      4. Assert: at least 10 samples per class (absolute minimum)
      5. Assert: signal values in physically reasonable range (not all zeros, max < 1e6)
    Expected Result: 4 classes present, all tensors finite, shapes consistent (N_samples × 4_channels × seq_length)
    Failure Indicators: NaN/Inf values, missing classes, zero-length tensors, only 1 class
    Evidence: .sisyphus/evidence/task-1-data-validation.txt
  ```

  **Evidence to Capture:**
  - [ ] task-1-config-update.txt — before/after diff of configs.py and mafaulda_dataset.py
  - [ ] task-1-data-download.txt — download log with directory listing
  - [ ] task-1-data-validation.txt — tensor shapes, dtypes, value ranges, class distribution

  **Commit**: YES
  - Message: `feat(data): configure 16Hz 4-class setup and download+process MaFaulDa`
  - Files: `src/configs.py`, `src/data/mafaulda_dataset.py`, `data/` (processed outputs)
  - Pre-commit: verify processed data loads without error

---

- [x] 2. Validation Shim — Normalization Compatibility, Architecture Match, Pipeline Health

  **What to do**:
  - **Normalization compatibility check**: Load `previous-work-pinn/export_relobralo/normalization.npz` (contains Xmin(10,), Xmax(10,), ymin(4,), ymax(4,)). Compare the feature order [vel_uh_rad, vel_uh_tan, vel_oh_rad, vel_oh_tan, pos_uh_rad, pos_uh_tan, pos_oh_rad, pos_oh_tan, omega, time] against what `src/data/clean_mafaulda_processor.py` Strategy D actually produces. Check if the value ranges of new processed data fall within [Xmin, Xmax] bounds. Log any mismatches
  - **Architecture match check**: Compare `previous-work-pinn/export_relobralo/model_config.json` metadata (hidden sizes, activation, input/output dims) with `src/models/pinn.py`. Check if hidden layer sizes match. Check `src/configs.py` for PINN hidden size config. If mismatch, document it clearly — T5 will need to handle it. **IMPORTANT**: Do NOT `import architecture.py` directly — it has a broken dependency on `direct_analysis.data_utils` which doesn't exist in this repo. Instead, READ `architecture.py` as text to extract the class structure, and compare with `src/models/pinn.py` by reading both files
  - **Attempt weight loading**: Try `torch.load('previous-work-pinn/export_relobralo/weights.pth')` and inspect state_dict keys. Instantiate `src/models/pinn.py` ConfigurablePINN with [128,128] hidden layers (based on model_config.json), then try `model.load_state_dict(state_dict)`. Log key mismatches if any. If keys don't match, print both sets of keys for comparison
  - **Pipeline smoke test**: Import each module: `src/models/pinn`, `src/models/ts_jepa`, `src/models/decoder1`, `src/models/decoder2_cvae`, `src/models/latent_diffusion`, `src/models/oracles`, `src/models/feature_extractors`. Instantiate each with default configs. Verify no import errors, no missing dependencies
  - **Run existing tests**: `pytest tests/ -v` — run test_pinn_predictions.py, test_oracle.py, test_math_extractor.py. Document pass/fail
  - **dtype boundary audit**: Grep codebase for `.float()`, `.double()`, `.to(torch.float32)`, `.to(torch.float64)` — map every dtype transition point. Document which modules expect float64 vs float32

  **Must NOT do**:
  - Do NOT fix any issues found — only DOCUMENT them. T3 and T5 will handle fixes
  - Do NOT modify any source files
  - Do NOT train anything

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Read-only validation and diagnostic — no code changes, just testing and documenting
  - **Skills**: []
  - **Skills Evaluated but Omitted**:
    - None applicable — pure diagnostic task

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T1, T3, T4 — but T2 needs T1's processed data for normalization range check)
  - **Parallel Group**: Wave 1 (starts after T1 completes, parallel with T3, T4)
  - **Blocks**: T5 (PINN diagnostic needs to know if weights load), T9 (PINN training needs architecture match info)
  - **Blocked By**: T1 (needs processed data for normalization range comparison)

  **References**:

  **Pattern References**:
  - `previous-work-pinn/export_relobralo/architecture.py` — READ AS TEXT ONLY (do NOT import — broken `direct_analysis.data_utils` dependency). Extract the ConfigurablePINN class structure by reading the file, note layer names and shapes for comparison
  - `previous-work-pinn/export_relobralo/model_config.json` — Metadata about the exported model (hidden sizes, activation, etc.). Use this as the primary source of truth for architecture params

  **API/Type References**:
  - `previous-work-pinn/export_relobralo/normalization.npz` — Load with `np.load()`. Contains arrays: `Xmin`, `Xmax` (shape 10), `ymin`, `ymax` (shape 4)
  - `previous-work-pinn/export_relobralo/weights.pth` — PyTorch state dict. Load with `torch.load()`, inspect `.keys()`
  - `src/models/pinn.py` — Current PINN implementation. Check `__init__` for layer definitions, compare hidden sizes with old architecture
  - `src/configs.py` — Check PINN-related config values (hidden sizes, activation, input/output dimensions)

  **Test References**:
  - `tests/test_pinn_predictions.py` — Existing PINN test. Run to check baseline behavior
  - `tests/test_oracle.py` — Oracle test (PriorWorkOracle wrapping PINN)
  - `tests/test_math_extractor.py` — MathFeatureExtractor test

  **WHY Each Reference Matters**:
  - `architecture.py` + `model_config.json`: The old PINN used [128,128] hidden — if current code defaults to [64,64], weights won't load. This is a showstopper for the fine-tuning strategy
  - `normalization.npz`: If Strategy D produces features in a different order or range than the old normalization, old PINN weights will produce garbage. This determines whether fine-tuning is viable
  - Existing tests: Quick signal that modules are functional before we invest time in training

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Normalization compatibility report generated
    Tool: Bash (Python script)
    Preconditions: T1 processed data available, normalization.npz exists
    Steps:
      1. Load normalization.npz: print Xmin, Xmax shapes and values
      2. Load processed MaFaulDa data from T1: compute min/max per feature dimension
      3. Compare: for each of the 10 input features, print (old_Xmin, old_Xmax) vs (new_min, new_max)
      4. Flag any feature where new data range exceeds old normalization bounds by >20%
    Expected Result: Clear comparison table showing which features are compatible and which diverge
    Failure Indicators: Cannot load normalization.npz, processed data has different number of features than 10
    Evidence: .sisyphus/evidence/task-2-norm-compat.txt

  Scenario: Architecture match check and weight loading attempt
    Tool: Bash (Python script)
    Preconditions: Both PINN implementations available
    Steps:
      1. Read architecture.py AS TEXT (do NOT import) — extract class names, layer definitions, hidden sizes
      2. Read model_config.json — extract hidden_layers, activation, input/output dims
      3. Import new PINN from src/models/pinn.py, instantiate with [128,128] hidden (matching old model)
      4. Print new model's layer names and shapes
      5. Load weights.pth via torch.load(), print state_dict keys
      6. Attempt model.load_state_dict(state_dict) — log SUCCESS or FAILURE with mismatched keys
    Expected Result: Clear YES/NO on weight compatibility with detailed key comparison
    Failure Indicators: Cannot instantiate new PINN with [128,128], state_dict key names completely different
    Evidence: .sisyphus/evidence/task-2-arch-match.txt

  Scenario: Existing test suite passes
    Tool: Bash
    Preconditions: Dependencies installed
    Steps:
      1. Run: pytest tests/ -v --tb=short
      2. Capture full output
    Expected Result: Tests pass OR failures are documented with reasons
    Failure Indicators: Import errors blocking test execution entirely
    Evidence: .sisyphus/evidence/task-2-test-results.txt
  ```

  **Evidence to Capture:**
  - [ ] task-2-norm-compat.txt — feature-by-feature normalization comparison table
  - [ ] task-2-arch-match.txt — architecture comparison + weight loading result
  - [ ] task-2-test-results.txt — pytest output
  - [ ] task-2-dtype-audit.txt — all dtype transition points in codebase

  **Commit**: NO (read-only diagnostic — no files changed)

---

- [x] 3. Code Audit of Critical Pipeline Files + Fix Blocking Issues

  **What to do**:
  - **Audit `src/configs.py`**: Verify all hyperparameters are consistent (PINN hidden sizes match architecture, sequence lengths align between data processing and model expectations, learning rates are reasonable). Check that fault class mapping includes exactly: Normal(0), Imbalance(1), Vertical Misalignment(2), Overhang Ball Bearing(3)
  - **Audit `src/data/clean_mafaulda_processor.py`**: Verify Strategy D differentiation is mathematically correct (acceleration → velocity → position via cumulative trapezoidal integration with detrending). Check Butterworth filter parameters. Check that severity subtypes within each fault class are handled correctly (merged or properly labeled). Verify output feature order matches what PINN expects
  - **Audit `src/models/pinn.py`**: Verify physics residual equations match the rotor-bearing equations of motion from the paper. Check float64 is used consistently throughout. Verify ReLoBRaLo loss integration. Check gradient computation (autograd usage)
  - **Audit `src/models/oracles.py`**: Verify PriorWorkOracle correctly wraps PINN + MathFeatureExtractor. Check VJP (Vector-Jacobian Product) computation for SDEdit guidance. Verify frozen parameters during inference
  - **Audit dtype boundaries**: Using T2's dtype audit, insert explicit `.to(torch.float64)` / `.to(torch.float32)` casts at every boundary. Add logging when dtype conversion happens
  - **Fix ALL blocking issues found**: Any bug that would prevent the pipeline from running end-to-end. Document each fix with before/after
  - **Verify PINN architecture config**: If T2 found [128,128] vs [64,64] mismatch, update `src/configs.py` to use [128,128] to match old weights (or add a config option for both)

  **Must NOT do**:
  - Do NOT refactor for style — only fix bugs that block execution
  - Do NOT add new features or abstractions
  - Do NOT change the physics equations or model architectures (beyond matching old weights)
  - Do NOT rename variables or reorganize imports

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Requires careful mathematical verification of physics equations, understanding of numerical integration, and dtype analysis across multiple files. High reasoning load
  - **Skills**: []
  - **Skills Evaluated but Omitted**:
    - None — this is pure code analysis and targeted fixes

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T4; partially with T2 — can start audit while T2 runs, but needs T2 results for dtype fixes)
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 4)
  - **Blocks**: T9 (PINN training needs clean code)
  - **Blocked By**: T1 (needs to see processed data shapes), T2 (needs diagnostic results for targeted fixes)

  **References**:

  **Pattern References**:
  - `previous-work-pinn/basicPINNv8.py` — The ORIGINAL PINN training script with verified physics equations. Compare residual equations line-by-line with `src/models/pinn.py`
  - `previous-work-pinn/LoadDatav3.py` — Original data loading. Compare Strategy D implementation with `clean_mafaulda_processor.py` to verify differentiation/integration is equivalent
  - `previous-work-pinn/relobralo_training.py` — Original ReLoBRaLo training loop. Compare loss balancing with `src/models/relobralo_loss.py`

  **API/Type References**:
  - `src/configs.py` — All hyperparameters. Check PINN_HIDDEN_SIZES, SEQ_LENGTH, SAMPLE_RATE, FAULT_CLASSES
  - `src/models/feature_extractors.py` — MathFeatureExtractor producing 2240-dim embeddings. Verify input/output shapes
  - `src/models/relobralo_loss.py` — ReLoBRaLo adaptive loss. Verify alpha=0.9, rho=0.1, temperature=2.0 defaults

  **External References**:
  - `main_M2_6_pages.tex` — The paper itself. Physics equations in Section II/III should match code exactly

  **WHY Each Reference Matters**:
  - `basicPINNv8.py`: This is the PROVEN implementation. Any deviation in Gustavo's rewrite could explain the "high loss" issue. Line-by-line comparison of the 4 residual equations + 2 mass constraints is essential
  - `LoadDatav3.py`: Data processing correctness directly affects PINN training. If Strategy D produces different features than the old loader, the physics equations receive wrong inputs
  - The paper LaTeX: The equations in the code MUST match the equations in the paper — any discrepancy is a critical bug

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: All critical files audited with issues documented
    Tool: Bash
    Preconditions: Source files accessible
    Steps:
      1. Create audit report documenting: files reviewed, issues found (with line numbers), fixes applied
      2. For each physics equation in pinn.py: compare with basicPINNv8.py and paper LaTeX — document match/mismatch
      3. For each dtype boundary: document the transition and whether explicit cast was added
      4. Run: pytest tests/ -v — all tests still pass after fixes
    Expected Result: Audit report with 0 remaining blocking issues. Tests pass
    Failure Indicators: Physics equation mismatch found but not fixable, tests fail after fixes
    Evidence: .sisyphus/evidence/task-3-audit-report.txt

  Scenario: Pipeline imports and instantiates without errors after fixes
    Tool: Bash (Python)
    Preconditions: Fixes applied
    Steps:
      1. Python script that imports ALL modules: pinn, ts_jepa, decoder1, decoder2_cvae, latent_diffusion, oracles, feature_extractors
      2. Instantiate each with configs from src/configs.py
      3. Create dummy input tensor, run forward pass through each model
      4. Print output shapes and dtypes for each
    Expected Result: All models instantiate and produce outputs with correct shapes and expected dtypes
    Failure Indicators: Import errors, shape mismatches, dtype errors on forward pass
    Evidence: .sisyphus/evidence/task-3-smoke-test.txt
  ```

  **Evidence to Capture:**
  - [ ] task-3-audit-report.txt — complete audit with issues found, fixes applied, equation comparison
  - [ ] task-3-smoke-test.txt — module import + forward pass verification

  **Commit**: YES
  - Message: `fix(pipeline): correct blocking issues found in code audit`
  - Files: modified source files (src/models/pinn.py, src/configs.py, etc.)
  - Pre-commit: `pytest tests/ -v`

---

- [x] 4. Paper Deliverables Specification

  **What to do**:
  - Read `main_M2_6_pages.tex` thoroughly. Identify which sections are complete (Introduction, Related Work, Methodology) and which are placeholders (Results, Discussion, Conclusion)
  - Define the EXACT figure list for the paper (budget: 3-4 composite figures in 6 pages):
    - **Figure A**: Component validation twin plot — PINN residual UMAP (left: synthetic, right: MaFaulDa). Shows physics features cluster by fault class
    - **Figure B**: Generation quality twin plot — SDEdit trajectory on UMAP manifold (left: synthetic, right: MaFaulDa). Shows physics-guided navigation from healthy to fault
    - **Figure C**: Signal comparison — generated counterfactual traces overlaid with real fault signals (time domain + frequency domain). Shows physical plausibility
    - **Figure D (optional if space)**: Baseline comparison UMAP — vanilla DDPM vs label-conditioned vs physics-guided. Visual ablation
  - Define the EXACT table list (budget: 1-2 tables):
    - **Table 1**: Transfer classification results — TSTR accuracy (train synthetic, test real) for: Ours (physics-guided), Baseline 1 (Vanilla DDPM), Baseline 2 (Label-conditioned), Real-only classifier. 5-seed mean ± std
    - **Table 2 (optional)**: Component metrics — PINN Silhouette score, Decoder 1 envelope RMSE, CVAE ELBO, LDM FID-like metric
  - Specify exact file paths for each figure asset (e.g., `assets/fig_component_validation.pdf`)
  - Document figure sizes: IEEE double-column format, figure width = 3.5in (single col) or 7.0in (double col)
  - Draft figure captions (1-2 sentences each) — these drive what the experiments must produce

  **Must NOT do**:
  - Do NOT write Results/Discussion prose — only specify what figures/tables are needed
  - Do NOT modify the LaTeX file yet — only read and plan
  - Do NOT design more than 4 figures or 2 tables

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: Reading academic paper, planning visual deliverables, drafting captions — writing-oriented task
  - **Skills**: []
  - **Skills Evaluated but Omitted**:
    - `playwright`: No browser needed
    - `frontend-ui-ux`: Not web UI design

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T1, T2, T3 — independent of data/code)
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 3)
  - **Blocks**: T18 (paper figure generation needs this specification)
  - **Blocked By**: None (can start immediately — only reads the existing LaTeX)

  **References**:

  **Pattern References**:
  - `main_M2_6_pages.tex` — The paper. Read Section structure, existing figures, placeholder sections
  - `assets/` — Existing figure assets. Check what's already there vs what's needed

  **External References**:
  - IEEE conference format: double-column, 6 pages max, figure guidelines

  **WHY Each Reference Matters**:
  - `main_M2_6_pages.tex`: Must understand current paper structure to know exactly which results sections need filling. The Methodology section defines what experiments are promised
  - `assets/`: Avoid recreating figures that might already exist from M2 milestone

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Complete deliverables spec document produced
    Tool: Bash
    Preconditions: LaTeX file readable
    Steps:
      1. Verify deliverables spec file exists at .sisyphus/evidence/task-4-paper-spec.md
      2. Check it contains: exact figure list (3-4), exact table list (1-2), file paths for each asset, figure sizes, draft captions
      3. Verify each figure has: description, data source, what it demonstrates, file path
      4. Verify each table has: columns defined, row labels defined, data source
    Expected Result: Spec document with 3-4 figures and 1-2 tables fully specified
    Failure Indicators: Missing figure descriptions, no file paths, no captions
    Evidence: .sisyphus/evidence/task-4-paper-spec.md
  ```

  **Evidence to Capture:**
  - [ ] task-4-paper-spec.md — complete deliverables specification

  **Commit**: NO (planning document only — no code changes)

---

### Wave 2 — Parallel PINN Diagnostic + TS-JEPA + Prep

- [x] 5. PINN Diagnostic — Load Old Weights, Evaluate on New MaFaulDa Data

  **What to do**:
  - **Load old PINN weights**: Using T2's architecture match results, load `previous-work-pinn/export_relobralo/weights.pth` into a ConfigurablePINN instance with [128,128] hidden layers (matching old architecture). Apply normalization from `normalization.npz` — scale new MaFaulDa processed data using old Xmin/Xmax bounds
  - **If T2 found architecture mismatch (keys don't match)**: Create a dependency-free shim — copy ONLY the ConfigurablePINN and ConfigurableMLP class definitions from `previous-work-pinn/export_relobralo/architecture.py` into a new temporary file (e.g., `src/models/pinn_old_compat.py`), stripping the broken `from direct_analysis.data_utils import ...` line. Then instantiate from that shim and load weights. Do NOT try to import architecture.py directly (it has a broken dependency)
  - **Evaluate on MaFaulDa data**: Forward pass through loaded PINN on healthy + 3 fault classes. Compute physics residuals (4 equations). Record residual magnitudes per class
  - **Extract features**: Pass residuals through `src/models/feature_extractors.py` MathFeatureExtractor. Get 2240-dim embeddings per sample
  - **Compute clustering metrics**: Run UMAP (random_state=42) on the 2240-dim features. Compute Silhouette score on 4-class labels. Compute kNN accuracy (k=5, 5-fold cross-validation) on the features
  - **Visualize**: Generate UMAP scatter plot colored by fault class. Save as `.sisyphus/evidence/task-5-pinn-old-weights-umap.png`
  - **Dimensionality check**: Also try PCA reducing 2240→50 dims before UMAP — does clustering improve? This tests Metis's hypothesis about curse of dimensionality
  - **Document results**: Clear comparison table: {metric: value} for old weights on new data. This becomes the baseline for T9 (training comparison)

  **Must NOT do**:
  - Do NOT modify the PINN weights or train anything — this is evaluation only
  - Do NOT modify the MathFeatureExtractor — evaluate as-is
  - Do NOT spend time optimizing UMAP parameters — use defaults + random_state=42

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Requires careful handling of normalization alignment, architecture compatibility, and quantitative evaluation. High reasoning load for diagnosing whether old weights transfer
  - **Skills**: []
  - **Skills Evaluated but Omitted**:
    - None applicable — pure computation + analysis

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T6, T7, T8)
  - **Parallel Group**: Wave 2 (with Tasks 6, 7, 8)
  - **Blocks**: T9 (PINN training strategy depends on how well old weights perform), T10 (variant comparison needs this as baseline)
  - **Blocked By**: T1 (processed data), T2 (architecture match + normalization compat results)

  **References**:

  **Pattern References**:
  - `previous-work-pinn/export_relobralo/architecture.py` — READ AS TEXT ONLY (broken `direct_analysis.data_utils` dependency). If state_dict keys don't match `src/models/pinn.py`, copy the ConfigurablePINN + ConfigurableMLP class definitions into a dependency-free shim file, stripping the broken import
  - `previous-work-pinn/export_relobralo/normalization.npz` — Scale inputs using the old normalization bounds. Check `previous-work-pinn/basicPINNv8.py` for the exact formula used (likely `x_norm = (x - Xmin) / (Xmax - Xmin)`)

  **API/Type References**:
  - `previous-work-pinn/export_relobralo/weights.pth` — State dict to load
  - `previous-work-pinn/export_relobralo/model_config.json` — Architecture metadata (hidden sizes, activation). Primary source for constructing matching architecture
  - `src/models/feature_extractors.py:MathFeatureExtractor` — 2240-dim feature extraction. Check `forward()` signature — what input shape does it expect?
  - `src/models/pinn.py` — Check `compute_residuals()` or equivalent method for getting physics residuals

  **External References**:
  - `previous-work-pinn/basicPINNv8.py` — Reference for how old code evaluated residuals and did normalization. Lines where `Xmin`, `Xmax` are applied

  **WHY Each Reference Matters**:
  - `architecture.py`: If current pinn.py has different hidden sizes, we MUST use the old architecture class to load old weights. This is the difference between "old weights load" and "old weights are garbage"
  - `normalization.npz`: Wrong normalization = garbage outputs even with correct weights. Must replicate EXACT same scaling as original training
  - `basicPINNv8.py`: Shows HOW normalization was applied during original training — (x - min)/(max - min) or (x - mean)/std or something else

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Old PINN weights loaded and evaluated on MaFaulDa
    Tool: Bash (Python script)
    Preconditions: T1 processed data available, T2 confirmed weight loading approach
    Steps:
      1. Load old PINN weights (using old architecture class if needed)
      2. Apply normalization from normalization.npz to MaFaulDa processed data
      3. Forward pass through PINN on all 4 classes
      4. Print residual magnitudes: mean ± std per class (4 residual equations × 4 classes = 16 values)
      5. Run MathFeatureExtractor on residuals → 2240-dim features
      6. Compute Silhouette score on 4-class labels
      7. Compute kNN accuracy (k=5, 5-fold CV)
      8. Generate UMAP plot (random_state=42) colored by class
    Expected Result: Silhouette score, kNN accuracy, and UMAP plot produced. Even if clustering is poor, the diagnostic numbers are recorded
    Failure Indicators: Weight loading fails entirely, NaN residuals, all-zero features
    Evidence: .sisyphus/evidence/task-5-pinn-old-weights-umap.png, .sisyphus/evidence/task-5-pinn-diagnostic.txt

  Scenario: Dimensionality reduction comparison
    Tool: Bash (Python script)
    Preconditions: 2240-dim features computed
    Steps:
      1. Apply PCA(n_components=50) to 2240-dim features
      2. Compute Silhouette score on PCA-reduced features
      3. Compute kNN accuracy on PCA-reduced features
      4. Generate UMAP from PCA-reduced features (random_state=42)
      5. Compare metrics: raw-2240 vs PCA-50 — which clusters better?
    Expected Result: Comparison table showing {metric: raw_value, pca_value}
    Failure Indicators: PCA fails (singular matrix), worse results in all metrics
    Evidence: .sisyphus/evidence/task-5-pca-comparison.txt
  ```

  **Evidence to Capture:**
  - [ ] task-5-pinn-old-weights-umap.png — UMAP scatter colored by fault class
  - [ ] task-5-pinn-diagnostic.txt — residual magnitudes, Silhouette, kNN accuracy
  - [ ] task-5-pca-comparison.txt — raw vs PCA-reduced clustering comparison

  **Commit**: NO (evaluation only — no code changes)

---

- [x] 6. TS-JEPA Encoder Training on MaFaulDa

  **What to do**:
  - **KEY INSIGHT**: TS-JEPA training is COMPLETELY INDEPENDENT of PINN. It learns time-series representations from the raw signal data, not from PINN features. This enables critical parallelism — start training immediately while PINN debugging happens in T5/T9
  - Load processed MaFaulDa data from T1. Prepare DataLoaders using `src/data/mafaulda_dataset.py`
  - Review TS-JEPA config in `src/configs.py`: check latent dimension (z_macro size), encoder architecture, masking strategy, learning rate, epochs
  - Train TS-JEPA using `src/models/ts_jepa.py` — this is the JEPA encoder that maps multi-channel time-series → z_macro latent vector
  - **Smoke test first**: Run 2-3 epochs, verify loss decreases, verify z_macro output shape is correct, verify no NaN/Inf in outputs
  - **Full training**: Train for configured number of epochs with early stopping
  - Save checkpoint to `results/ts_jepa.pth`
  - **Quick validation**: Encode all training data through trained TS-JEPA. Compute UMAP of z_macro space (random_state=42) colored by fault class. Even without PINN guidance, z_macro should show SOME structure (not necessarily clean clusters, but not pure noise either)

  **Must NOT do**:
  - Do NOT modify TS-JEPA architecture — use as configured
  - Do NOT tune hyperparameters beyond what's in configs.py (max 3 configs if first doesn't converge)
  - Do NOT wait for PINN results — train independently

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: ML training loop — needs monitoring, early stopping check, checkpoint management. More than quick but not architecture-level complexity
  - **Skills**: []
  - **Skills Evaluated but Omitted**:
    - None — standard training task

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T5, T7, T8 — COMPLETELY independent of PINN)
  - **Parallel Group**: Wave 2 (with Tasks 5, 7, 8)
  - **Blocks**: T11 (Decoder 1 needs frozen TS-JEPA), T12 (LDM needs z_macro), T13 (Decoder 2 needs TS-JEPA)
  - **Blocked By**: T1 (needs processed MaFaulDa data)

  **References**:

  **Pattern References**:
  - `src/pipelines/train_phase1.py` — The Phase 1 training orchestrator. Check if it handles TS-JEPA training with proper train/val split, early stopping, checkpoint saving. Use this if it orchestrates correctly; otherwise train directly
  - `main_phase1.py` — Entry point for Phase 1. Check what args it accepts and what it trains

  **API/Type References**:
  - `src/models/ts_jepa.py` — TS-JEPA model class. Check `forward()`, `encode()`, loss computation, z_macro output shape
  - `src/data/mafaulda_dataset.py` — DataLoader. Check batch format: what tensors does it yield? (signals, labels, omega?)
  - `src/configs.py` — TS-JEPA config: `JEPA_LATENT_DIM`, `JEPA_HIDDEN_DIM`, `JEPA_EPOCHS`, learning rate

  **WHY Each Reference Matters**:
  - `train_phase1.py`: May already handle TS-JEPA training correctly. If so, just run it. If not, need to know what's missing
  - `ts_jepa.py`: Must understand input format (channels × seq_length) and output format (z_macro dimension) for downstream tasks
  - `mafaulda_dataset.py`: Must match data loader output format to TS-JEPA input expectations

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: TS-JEPA trains and converges on MaFaulDa
    Tool: Bash (Python)
    Preconditions: T1 processed data available
    Steps:
      1. Run TS-JEPA training (via main_phase1.py or direct training script)
      2. Monitor: loss at epoch 1, loss at epoch N/2, loss at final epoch
      3. Verify loss decreased by at least 50% from epoch 1 to final
      4. Verify checkpoint saved to results/ts_jepa.pth
      5. Load checkpoint, encode 10 samples, print z_macro shape and value range
    Expected Result: Loss converges, z_macro has expected shape (batch_size, latent_dim), values are finite
    Failure Indicators: Loss increases or NaN, z_macro all zeros, checkpoint not saved
    Evidence: .sisyphus/evidence/task-6-tsjepa-training.txt

  Scenario: z_macro UMAP shows non-trivial structure
    Tool: Bash (Python)
    Preconditions: TS-JEPA trained
    Steps:
      1. Encode all training data through frozen TS-JEPA
      2. Run UMAP(random_state=42) on z_macro vectors
      3. Plot scatter colored by fault class, save as PNG
      4. Compute Silhouette score on z_macro space
    Expected Result: UMAP shows some structure (not random noise). Silhouette > 0.0 (any positive value is acceptable at this stage)
    Failure Indicators: UMAP is uniform random scatter, Silhouette < 0.0
    Evidence: .sisyphus/evidence/task-6-tsjepa-umap.png, .sisyphus/evidence/task-6-tsjepa-metrics.txt
  ```

  **Evidence to Capture:**
  - [ ] task-6-tsjepa-training.txt — training log with loss curve
  - [ ] task-6-tsjepa-umap.png — UMAP of z_macro colored by class
  - [ ] task-6-tsjepa-metrics.txt — Silhouette score, z_macro shape/range

  **Commit**: YES (groups with T11, T13 later)
  - Message: `feat(phase1): train TS-JEPA encoder on MaFaulDa 16Hz data`
  - Files: `results/ts_jepa.pth`
  - Pre-commit: verify checkpoint loads and produces finite outputs

---

- [x] 7. Implement Baseline Code (Vanilla DDPM + Label-Conditioned Diffusion)

  **What to do**:
  - **Vanilla DDPM baseline**: Implement a standard DDPM that generates time-series in the z_macro latent space WITHOUT any physics guidance (no oracle, no VJP). It should use the SAME LDM architecture from `src/models/latent_diffusion.py` but with guidance_scale=0 or the oracle disabled. The point: show that without physics guidance, generated samples don't land in correct fault regions
  - **Label-conditioned diffusion baseline**: Implement a class-conditional DDPM that takes a fault class label as conditioning input (one-hot or embedding). It uses class information to guide generation but still no physics. The point: show that even with class labels, physics-guided method produces more physically plausible samples
  - **Fairness constraint**: SAME architecture (LatentDiffusionMLP), SAME number of training epochs, SAME latent space (z_macro from TS-JEPA), SAME noise schedule. The ONLY difference is the guidance mechanism (none vs class-label vs physics-oracle)
  - **Implementation approach**: 
    - Vanilla DDPM: Reuse `src/models/latent_diffusion.py` as-is, just skip the oracle guidance during sampling
    - Label-conditioned: Modify `src/models/latent_diffusion.py` to accept class labels — add a class embedding layer that's concatenated to the diffusion timestep embedding. Create as `src/models/baselines.py` (new file)
  - Write training functions for both baselines
  - Do NOT train yet — just implement and verify forward pass works. T15 will run the actual experiments

  **Must NOT do**:
  - Do NOT implement more than 2 baselines
  - Do NOT use a different architecture for baselines — fairness requires same backbone
  - Do NOT tune baseline hyperparameters to intentionally make them worse
  - Do NOT train baselines in this task — only implement code

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Requires understanding the existing LDM code, creating a well-structured baseline module, and ensuring fairness. Non-trivial ML implementation but scoped
  - **Skills**: []
  - **Skills Evaluated but Omitted**:
    - None — standard PyTorch implementation task

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T5, T6, T8 — independent of training results)
  - **Parallel Group**: Wave 2 (with Tasks 5, 6, 8)
  - **Blocks**: T15 (baseline experiments need this code)
  - **Blocked By**: None (can start immediately — only reads existing code)

  **References**:

  **Pattern References**:
  - `src/models/latent_diffusion.py` — The EXISTING LDM implementation. Vanilla DDPM = this without guidance. Study: architecture, noise schedule, sampling loop, how oracle guidance is currently injected
  - `src/pipelines/run_sdedit_phase2.py` — How SDEdit currently uses oracle guidance. Understand the guidance mechanism so baselines correctly ablate it

  **API/Type References**:
  - `src/models/latent_diffusion.py:LatentDiffusionMLP` — Exact class to reuse/extend. Check `forward()`, `sample()`, timestep embedding, noise schedule
  - `src/models/oracles.py:PriorWorkOracle` — Understand what this provides (VJP gradients) so baseline correctly removes it

  **External References**:
  - `main_M2_6_pages.tex` — Paper Section that describes baselines. Must match paper's description of what baselines are

  **WHY Each Reference Matters**:
  - `latent_diffusion.py`: The Vanilla DDPM baseline IS this model with guidance disabled. Must understand exactly where guidance enters the sampling loop
  - `run_sdedit_phase2.py`: Shows the guidance injection point. Baseline removes this specific intervention
  - Paper LaTeX: Baselines must match what the paper claims they are — reviewer will check

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Baseline code instantiates and runs forward pass
    Tool: Bash (Python)
    Preconditions: Source code written
    Steps:
      1. Import VanillaDDPM and LabelConditionedDDPM from src/models/baselines.py
      2. Instantiate both with same config as main LDM
      3. Create dummy z_macro input (batch=4, dim=latent_dim)
      4. Run forward pass (training mode) through both — compute loss
      5. Run sampling (inference mode) through both — generate z_macro samples
      6. Print: output shapes, loss values (should be finite), generated sample ranges
    Expected Result: Both baselines produce finite outputs with correct shapes
    Failure Indicators: Import errors, NaN loss, shape mismatches
    Evidence: .sisyphus/evidence/task-7-baseline-smoke.txt

  Scenario: Fairness verification — architecture parity confirmed
    Tool: Bash (Python)
    Preconditions: Baseline code exists
    Steps:
      1. Count parameters: main LDM, Vanilla DDPM, Label-conditioned DDPM
      2. Verify Vanilla DDPM has SAME parameter count as main LDM (minus oracle)
      3. Verify Label-conditioned has parameter count = main LDM + class_embedding_params (small overhead OK)
      4. Verify same noise schedule (beta_start, beta_end, num_steps) used by all 3
    Expected Result: Parameter counts documented, noise schedules match
    Failure Indicators: >10% parameter count difference between methods (excluding guidance components)
    Evidence: .sisyphus/evidence/task-7-fairness-check.txt
  ```

  **Evidence to Capture:**
  - [ ] task-7-baseline-smoke.txt — forward pass verification for both baselines
  - [ ] task-7-fairness-check.txt — parameter count comparison and noise schedule match

  **Commit**: YES
  - Message: `feat(baselines): implement Vanilla DDPM and Label-conditioned diffusion baselines`
  - Files: `src/models/baselines.py`
  - Pre-commit: `python -c "from src.models.baselines import VanillaDDPM, LabelConditionedDDPM"`

---

- [x] 8. Synthetic System Re-run + Fairness/Difficulty Validation

  **What to do**:
  - **CRITICAL — Paper-code alignment**: The paper (Section IV-A, Eq. 4) describes a 1D damped harmonic oscillator: `m*x'' + c*x' + k*x = A*sin(ω₀*t)` with nominal params `m=1, c=0.5, k=4.0, A=1.0, ω₀=1.0`. Three fault classes: stiffness reduction (k→0.5k), damping increase (c→2c), forcing perturbation (A→2A). Observable = acceleration x''(t). The existing `src/data/synthetic_dataset.py` implements a DIFFERENT system (4-channel vibration, imbalance/outer-race faults). You MUST create a NEW synthetic generator `src/data/paper_synthetic_oscillator.py` that matches the paper's description exactly:
    - Implement the 1D oscillator ODE with RK45 integration (use `scipy.integrate.solve_ivp`)
    - Healthy class: nominal parameters
    - Fault class 0 (stiffness reduction): k → 0.5*k
    - Fault class 1 (damping increase): c → 2*c
    - Fault class 2 (forcing perturbation): A → 2*A
    - Add Gaussian sensor noise to acceleration observable
    - Output format must be compatible with the pipeline (return tensors matching the shape/interface expected by downstream components)
    - Include a `get_dataloaders()` function matching the interface in `synthetic_dataset.py`
  - **Synthetic mode routing**: The current codebase does NOT have a synthetic mode switch in `main_phase0.py`/`main_phase1.py`/`main_phase2.py`. You must create a lightweight routing mechanism:
    - Option A: Add a `--synthetic` flag to `main_phase0.py` that swaps `MaFaulDaDataset` for the paper oscillator dataset and adjusts configs accordingly (e.g., `NUM_CLASSES`, `SEQ_LENGTH`)
    - Option B: Create a separate `main_synthetic.py` entry point that imports the Phase 0/1/2 training functions but uses the paper oscillator dataset
    - Choose the simpler option. The oscillator system should produce data in a format compatible with downstream components
  - **Run FULL pipeline on synthetic oscillator**: Run Phase 0 → Phase 1 → Phase 2 on the paper-aligned oscillator data. This validates the pipeline works end-to-end AND produces the results described in Section IV-A of the paper
  - **Compare to M2 results**: The `results-synthetic/` directory may contain old synthetic results. Compare new results with old — did Gustavo's changes improve, degrade, or maintain synthetic performance?
  - **Fairness/difficulty validation** (internal only — not reported in paper):
    - Generate synthetic fault data using the pipeline
    - Train a simple classifier (logistic regression or small MLP) on REAL synthetic data → test on REAL synthetic data. Record accuracy = "oracle" upper bound
    - Train same classifier on GENERATED synthetic data → test on REAL synthetic data. Record accuracy = "TSTR" metric
    - If TSTR ≈ oracle → synthetic data might be "too easy" (pipeline just memorizing). Flag this
    - Compute FID-like distance between generated and real distributions in feature space
  - **Sanity check**: If the pipeline fails on synthetic data, it WILL fail on MaFaulDa. This is a critical gate check
  - Save all results to `results-synthetic/` (overwrite old)

  **Must NOT do**:
  - Do NOT debug synthetic-specific issues for more than 2 hours — real MaFaulDa is the priority
  - Do NOT modify the existing `src/data/synthetic_dataset.py` (it may be used elsewhere). Instead, create a NEW file `src/data/paper_synthetic_oscillator.py` for the paper's 1D oscillator
  - Synthetic results ARE reported in the paper (Section IV-A). The fairness/difficulty metrics (oracle accuracy, TSTR ratio) are internal validation only — do NOT put those specific metrics in the paper

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Full pipeline execution + quantitative analysis. Needs to monitor training, compute metrics, compare results
  - **Skills**: []
  - **Skills Evaluated but Omitted**:
    - None — standard ML pipeline execution

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T5, T6, T7)
  - **Parallel Group**: Wave 2 (with Tasks 5, 6, 7)
  - **Blocks**: T18 (paper may reference synthetic validation)
  - **Blocked By**: T1 (needs env setup), T3 (needs code fixes applied — but can start without if T3 is slow, using unfixed code as a baseline)

  **References**:

  **Pattern References**:
  - `main_phase0.py` — Phase 0 entry point. Check how to run on synthetic data (flag or auto-detect)
  - `main_phase1.py` — Phase 1 entry point
  - `main_phase2.py` — Phase 2 entry point
  - `results-synthetic/` — Old synthetic results for comparison

  **API/Type References**:
  - `src/data/synthetic_dataset.py` — Reference for the `get_dataloaders()` interface. Your new `paper_synthetic_oscillator.py` must provide the same interface
  - `main_M2_6_pages.tex:315-320` — The paper's oscillator equation (Eq. 4) and fault definitions. Your synthetic generator MUST match these EXACTLY: m=1, c=0.5, k=4.0, A=1.0, ω₀=1.0, three faults: k→0.5k, c→2c, A→2A
  - `src/configs.py` — Check if there's a DATASET_TYPE or DATA_SOURCE config to switch between synthetic and MaFaulDa

  **WHY Each Reference Matters**:
  - `main_phase*.py`: These are the orchestrators. Running them in sequence validates the full pipeline
  - `synthetic_dataset.py`: Interface reference — your new oscillator must match this output format
  - `main_M2_6_pages.tex:315-320`: The ground truth for what the synthetic system must implement. Any deviation = paper-code mismatch
  - `results-synthetic/`: Baseline for comparison — did code changes help or hurt?

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Paper-aligned oscillator created and matches Eq. 4
    Tool: Bash (Python)
    Preconditions: scipy installed
    Steps:
      1. Verify src/data/paper_synthetic_oscillator.py exists
      2. Import and instantiate the oscillator dataset
      3. Check it produces 3 fault classes + healthy: stiffness reduction (k→0.5k), damping increase (c→2c), forcing perturbation (A→2A)
      4. Verify nominal params match paper: m=1, c=0.5, k=4.0, A=1.0, ω₀=1.0
      5. Verify output interface compatible with get_dataloaders() pattern
      6. Verify observable is acceleration x''(t) with additive Gaussian noise
    Expected Result: New oscillator file exists, matches paper's Eq. 4 exactly, produces labeled tensors
    Failure Indicators: Wrong parameter values, wrong fault definitions, output shape incompatible with pipeline
    Evidence: .sisyphus/evidence/task-8-oscillator-check.txt

  Scenario: Synthetic mode routing created and full pipeline completes
    Tool: Bash
    Preconditions: Dependencies installed, code fixes from T3 applied (if available), paper_synthetic_oscillator.py created
    Steps:
      1. Create synthetic mode routing (--synthetic flag or main_synthetic.py)
      2. Run Phase 0 (synthetic): PINN training on oscillator data
      3. Run Phase 1 (synthetic): TS-JEPA + Decoders training
      4. Run Phase 2 (synthetic): LDM + SDEdit generation
      5. Verify each produces output files in results-synthetic/
      6. Check no errors or warnings about NaN/Inf
    Expected Result: All 3 phases complete on paper-aligned synthetic data. Checkpoint files and UMAP plots saved to results-synthetic/
    Failure Indicators: Any phase crashes, NaN in training, missing output files, oscillator data shape incompatible with pipeline
    Evidence: .sisyphus/evidence/task-8-synthetic-pipeline.txt

  Scenario: Difficulty/fairness validation — generated data is non-trivial
    Tool: Bash (Python script)
    Preconditions: Synthetic generation complete
    Steps:
      1. Train logistic regression on REAL synthetic data → test on REAL synthetic data → "oracle" accuracy
      2. Train logistic regression on GENERATED synthetic data → test on REAL synthetic data → "TSTR" accuracy
      3. Compute ratio: TSTR / oracle
      4. If ratio > 0.95 → FLAG as potentially too easy
      5. If ratio < 0.3 → FLAG as generated data is useless
      6. Reasonable range: 0.5-0.9
    Expected Result: TSTR accuracy documented, ratio computed, assessment: "fair" / "too easy" / "too hard"
    Failure Indicators: Oracle accuracy < 50% (synthetic data itself isn't separable), TSTR/oracle > 0.98
    Evidence: .sisyphus/evidence/task-8-fairness.txt
  ```

  **Evidence to Capture:**
  - [ ] task-8-oscillator-check.txt — verification that paper_synthetic_oscillator.py matches Eq. 4
  - [ ] task-8-synthetic-pipeline.txt — full pipeline execution log
  - [ ] task-8-fairness.txt — oracle accuracy, TSTR accuracy, ratio, assessment

  **Commit**: YES
  - Message: `feat(synthetic): add paper-aligned 1D oscillator generator and synthetic pipeline routing`
  - Files: `src/data/paper_synthetic_oscillator.py`, synthetic mode routing changes (--synthetic flag or main_synthetic.py)
  - Pre-commit: `python -c "from src.data.paper_synthetic_oscillator import get_dataloaders; print('OK')"`

---

### Wave 3 — PINN Fix + Decoder 1 + LDM (Parallel)

- [x] 9. PINN Training — Fix, Fine-Tune, From-Scratch (TIME-BOXED: 2 days)

  **What to do**:
  - Based on T5 diagnostic results and T3 code fixes, execute up to 3 PINN training strategies:
  - **Strategy A — Fine-tune old weights**: Load old PINN weights (from T5). Apply T3's fixes. Fine-tune on MaFaulDa data with LOW learning rate (1/10th of original). Use ReLoBRaLo loss balancing (alpha=0.9, rho=0.1, temperature=2.0). Train for N epochs (check configs.py). Monitor: physics residual loss per equation, total loss, gradient norms
  - **Strategy B — Train from scratch**: Initialize PINN with fresh random weights (same architecture: [128,128], Tanh, float64). Train on MaFaulDa from scratch using full learning rate. Same ReLoBRaLo config. Train same number of epochs as Strategy A for fair comparison
  - **Strategy C — Old weights as-is** (from T5): If T5 showed decent clustering, this is the "zero additional training" option. Just use old weights directly
  - **For each strategy**: Save checkpoint, record final loss (total + per-equation), compute residuals on all classes
  - **Time-box enforcement**: Set a hard deadline. If after 2 days no strategy achieves Silhouette > 0.3 on residual features, STOP and proceed to fallback:
    - **Fallback option 1**: Use continuous residual features WITHOUT expecting clean clusters — reframe the paper claim from "clusters" to "physics-informed representation" that improves transfer classification
    - **Fallback option 2**: Use a simpler physics proxy (e.g., frequency-domain energy ratios) alongside PINN residuals
    - **Fallback option 3**: Report negative result honestly — "PINN residuals on real data require further investigation" — and lean on the synthetic PoC + baselines for the paper
  - **MAX 5 hyperparameter configs total across all strategies** (from guardrails)

  **Must NOT do**:
  - Do NOT change PINN architecture (hidden sizes, activation, number of equations)
  - Do NOT exceed 5 total training configurations
  - Do NOT spend more than 2 days total on PINN training
  - Do NOT remove ReLoBRaLo loss balancing — it's part of the paper's contribution

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Most critical task in the plan. Requires understanding physics equations, ReLoBRaLo dynamics, normalization alignment, and deciding strategy based on T5 diagnostic. High judgment needed
  - **Skills**: []
  - **Skills Evaluated but Omitted**:
    - None — deep ML training + diagnosis

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T11, T12 — different models, different GPU workloads)
  - **Parallel Group**: Wave 3 (with Tasks 10, 11, 12)
  - **Blocks**: T10 (variant comparison), T16 (SDEdit needs best PINN)
  - **Blocked By**: T3 (code fixes), T5 (diagnostic results determine strategy)

  **References**:

  **Pattern References**:
  - `previous-work-pinn/relobralo_training.py` — The PROVEN training loop for PINN with ReLoBRaLo. Replicate this training procedure exactly, adapted for current codebase
  - `main_phase0.py` — Current Phase 0 entry point. Check if it handles PINN training correctly or if manual training script is needed
  - `previous-work-pinn/basicPINNv8.py` — Original PINN with complete training setup. Reference for: learning rate, weight initialization, gradient clipping, epoch count

  **API/Type References**:
  - `src/models/pinn.py:ConfigurablePINN` — The model to train. Check `forward()`, `compute_loss()`, gradient requirements
  - `src/models/relobralo_loss.py` — Adaptive loss balancing. Check initialization, update step, how it's called in training loop
  - `src/configs.py` — PINN training config: learning rate, epochs, hidden sizes, gradient clipping
  - `previous-work-pinn/export_relobralo/weights.pth` — Starting weights for Strategy A (fine-tuning)

  **WHY Each Reference Matters**:
  - `relobralo_training.py`: This produced working results before. Any deviation from this training procedure is a potential cause of "high loss" issue
  - `basicPINNv8.py`: Contains the EXACT optimizer setup, LR schedule, weight init that worked. Compare with current training loop

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: At least 2 PINN strategies trained and compared
    Tool: Bash (Python)
    Preconditions: T3 fixes applied, T5 diagnostic complete
    Steps:
      1. Train Strategy A (fine-tune): log loss at epoch 1, 10, 50, final
      2. Train Strategy B (from scratch): log loss at same epoch checkpoints
      3. For each: save checkpoint, compute residuals on all 4 classes
      4. Print comparison table: {strategy, final_total_loss, per_equation_losses[4], training_time}
    Expected Result: At least 2 strategies complete. Loss values documented. One strategy has lower loss
    Failure Indicators: Training diverges (NaN loss), both strategies have same final loss (no learning)
    Evidence: .sisyphus/evidence/task-9-pinn-training.txt

  Scenario: Time-box respected — fallback documented if needed
    Tool: Bash
    Preconditions: Training strategies attempted
    Steps:
      1. Check total time spent on PINN training
      2. If Silhouette > 0.3 achieved: document winning strategy
      3. If Silhouette <= 0.3 after all strategies: document fallback decision (option 1, 2, or 3)
      4. In either case: save the BEST checkpoint to results/pinn.pth
    Expected Result: Clear decision documented — either "Strategy X works" or "Fallback Y chosen with justification"
    Failure Indicators: No decision made, training still running after 2 days
    Evidence: .sisyphus/evidence/task-9-pinn-decision.txt
  ```

  **Evidence to Capture:**
  - [ ] task-9-pinn-training.txt — loss curves, per-equation losses, training time for each strategy
  - [ ] task-9-pinn-decision.txt — winning strategy or fallback decision with justification

  **Commit**: YES
  - Message: `fix(pinn): train and validate PINN on MaFaulDa — [strategy chosen]`
  - Files: `results/pinn.pth`, any training script modifications
  - Pre-commit: verify checkpoint loads

---

- [x] 10. PINN Variant Comparison + UMAP Generation + Best Model Selection

  **What to do**:
  - Load ALL PINN variants from T9 (fine-tuned, from-scratch) + T5 (old weights as-is)
  - For EACH variant:
    - Forward pass through PINN on all MaFaulDa data (4 classes)
    - Extract physics residuals (4 equations)
    - Pass through MathFeatureExtractor → 2240-dim embeddings
    - Apply PCA(n_components=50) if T5 showed it helps (otherwise use raw)
    - Compute Silhouette score on 4-class labels
    - Compute kNN accuracy (k=5, 5-fold CV)
    - Generate UMAP plot (random_state=42, n_neighbors=15, min_dist=0.1)
  - **Create comparison table**: {variant, Silhouette, kNN_accuracy, total_loss}
  - **Select best variant**: Highest Silhouette score wins. If all below 0.3 → use fallback from T9
  - **Generate FINAL UMAP plots** for the paper: 
    - Best variant UMAP with clean styling (colorblind-friendly palette, clear legend, axis labels)
    - Save to `assets/` with paper-ready resolution (300 DPI, PDF format)
  - **Critical decision point**: If best Silhouette < 0.3, document the fallback strategy and how it affects downstream tasks (T16 SDEdit guidance, T17 transfer classification)

  **Must NOT do**:
  - Do NOT train additional PINN variants — only evaluate what T9 produced
  - Do NOT modify UMAP hyperparameters to artificially improve clustering appearance
  - Do NOT cherry-pick samples for the UMAP plot

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Quantitative comparison across multiple models, metric computation, figure generation. Systematic but not architecturally complex
  - **Skills**: []
  - **Skills Evaluated but Omitted**:
    - `visual-engineering`: This is matplotlib/scientific plotting, not web UI
    - `frontend-ui-ux`: Not applicable

  **Parallelization**:
  - **Can Run In Parallel**: NO (sequential after T9 — needs all variants)
  - **Parallel Group**: Wave 3 (after T9 completes, parallel with T11, T12 which are independent)
  - **Blocks**: T16 (SDEdit needs best PINN variant)
  - **Blocked By**: T9 (needs all trained variants), T5 (needs old-weights baseline)

  **References**:

  **Pattern References**:
  - T5 evidence files — Reuse the evaluation pipeline built in T5. Just run it on T9's new checkpoints
  - `tools/visualize_processed_data.py` — May have reusable plotting functions. Check for UMAP styling code

  **API/Type References**:
  - `src/models/feature_extractors.py:MathFeatureExtractor` — Feature extraction pipeline
  - `sklearn.metrics.silhouette_score` — Clustering metric
  - `umap.UMAP` — Dimensionality reduction for visualization

  **WHY Each Reference Matters**:
  - T5 evidence: Contains the evaluation script already built. Reuse for efficiency
  - `MathFeatureExtractor`: The 2240-dim feature space where clustering is evaluated. Must use same extractor for fair comparison

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: All PINN variants evaluated and compared
    Tool: Bash (Python)
    Preconditions: T9 checkpoints + T5 baseline available
    Steps:
      1. Load each variant (old-weights, fine-tuned, from-scratch)
      2. Compute: Silhouette score, kNN accuracy for each
      3. Print comparison table: {variant, silhouette, knn_acc, total_loss}
      4. Identify best variant
    Expected Result: Comparison table with 3 rows. Best variant clearly identified
    Failure Indicators: Cannot load a checkpoint, all variants have identical metrics
    Evidence: .sisyphus/evidence/task-10-variant-comparison.txt

  Scenario: Paper-ready UMAP figure generated
    Tool: Bash (Python)
    Preconditions: Best variant selected
    Steps:
      1. Generate UMAP for best variant with paper styling
      2. Save as assets/fig_pinn_umap_mafaulda.pdf (300 DPI, vector)
      3. Verify file exists and is non-empty
      4. Verify 4 classes visible in plot with distinct colors
    Expected Result: PDF figure file with clean, publication-ready UMAP
    Failure Indicators: Empty plot, all points same color, file < 10KB
    Evidence: .sisyphus/evidence/task-10-best-umap.png (preview), assets/fig_pinn_umap_mafaulda.pdf (final)
  ```

  **Evidence to Capture:**
  - [ ] task-10-variant-comparison.txt — comparison table for all PINN variants
  - [ ] task-10-best-umap.png — preview of best UMAP for quick inspection
  - [ ] assets/fig_pinn_umap_mafaulda.pdf — paper-ready figure

  **Commit**: YES (groups with T9)
  - Message: `feat(pinn): select best PINN variant, generate paper-ready UMAP`
  - Files: `assets/fig_pinn_umap_mafaulda.pdf`, evaluation scripts
  - Pre-commit: verify PDF exists and is non-empty

---

- [x] 11. Decoder 1 (Deterministic Envelope) Training

  **What to do**:
  - Freeze TS-JEPA encoder from T6 (load `results/ts_jepa.pth`, set `requires_grad=False`)
  - Train Decoder 1 (`src/models/decoder1.py`) to reconstruct the low-frequency envelope from z_macro
  - Decoder 1 maps z_macro → raw signal reconstruction. It should capture the macroscopic physical behavior (envelope) — the "predictable" part of the signal
  - **Smoke test**: 2-3 epochs, verify loss decreases, verify output shape matches input signal shape
  - **Full training**: Train with early stopping on validation loss
  - Record: final reconstruction RMSE, training epochs, best validation loss
  - Save checkpoint to `results/decoder1.pth`
  - **Quick visual check**: Overlay Decoder 1 reconstruction on a real signal from each class. Save comparison plot

  **Must NOT do**:
  - Do NOT unfreeze TS-JEPA — it stays frozen
  - Do NOT modify Decoder 1 architecture
  - Do NOT spend more than 3 training configs

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Standard ML training with monitoring. Needs to handle frozen encoder, early stopping, checkpoint management
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T9, T10, T12 — Decoder 1 is independent of PINN)
  - **Parallel Group**: Wave 3 (with Tasks 9, 10, 12)
  - **Blocks**: T13 (Decoder 2 needs Decoder 1 for residual computation), T14 (validation)
  - **Blocked By**: T6 (needs frozen TS-JEPA checkpoint)

  **References**:

  **Pattern References**:
  - `src/pipelines/train_phase1.py` — Phase 1 orchestrator. Check if Decoder 1 training is already implemented here with early stopping
  - `main_phase1.py` — Entry point. May handle Decoder 1 training automatically after TS-JEPA

  **API/Type References**:
  - `src/models/decoder1.py` — Decoder 1 class. Check: `forward(z_macro) → signal`, loss function, expected input/output shapes
  - `src/models/ts_jepa.py` — Needs `encode()` method to produce z_macro from input signals
  - `src/configs.py` — Decoder 1 config: hidden sizes, learning rate, epochs

  **WHY Each Reference Matters**:
  - `train_phase1.py`: May already orchestrate Decoder 1 training correctly. If so, just invoke it
  - `decoder1.py`: Must understand what "envelope reconstruction" means in this architecture — is it a full waveform reconstruction or a smoothed envelope?

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Decoder 1 trains and produces reconstructions
    Tool: Bash (Python)
    Preconditions: TS-JEPA trained (T6)
    Steps:
      1. Load frozen TS-JEPA from results/ts_jepa.pth
      2. Train Decoder 1, log loss at epoch 1, N/2, final
      3. Verify loss decreased by at least 30%
      4. Save checkpoint to results/decoder1.pth
      5. Reconstruct 1 signal per class, overlay with original, save plot
      6. Compute RMSE per class
    Expected Result: Checkpoint saved. RMSE documented per class. Reconstruction visually resembles original envelope
    Failure Indicators: Loss doesn't decrease, RMSE > 2× signal std, reconstruction is flat/constant
    Evidence: .sisyphus/evidence/task-11-decoder1-training.txt, .sisyphus/evidence/task-11-reconstruction.png
  ```

  **Evidence to Capture:**
  - [ ] task-11-decoder1-training.txt — loss curve, final RMSE per class
  - [ ] task-11-reconstruction.png — overlay of reconstruction vs original for each class

  **Commit**: YES (groups with T6, T13)
  - Message: `feat(phase1): train Decoder 1 envelope reconstruction`
  - Files: `results/decoder1.pth`
  - Pre-commit: verify checkpoint loads and produces finite outputs

---

- [x] 12. LDM Training on z_macro Space

  **What to do**:
  - Freeze TS-JEPA encoder from T6 (load `results/ts_jepa.pth`, set `requires_grad=False`)
  - Encode all MaFaulDa training data through frozen TS-JEPA → z_macro vectors
  - Train the Latent Diffusion Model (`src/models/latent_diffusion.py:LatentDiffusionMLP`) to learn the distribution of z_macro vectors using DDPM
  - This is the generative model that will later be guided by the PINN oracle in SDEdit (T16)
  - **Smoke test**: 2-3 epochs, verify DDPM loss decreases, verify generated z_macro samples have correct shape and finite values
  - **Full training**: Train for configured epochs. Monitor: MSE loss on noise prediction, sample quality (generate z_macro samples, check they're in reasonable range)
  - Save checkpoint to `results/ldm.pth` (or whatever the project convention is)
  - **Quick validation**: Generate 100 unconditional z_macro samples. Project onto T6's UMAP space. Do they fall within the learned data manifold? Save plot

  **Must NOT do**:
  - Do NOT modify LDM architecture
  - Do NOT add guidance at this stage — this is unconditional LDM training. Guidance comes in T16 (SDEdit)
  - Do NOT spend more than 3 training configs

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Diffusion model training — needs careful monitoring of noise prediction quality, sample generation, checkpoint management
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T9, T10, T11 — LDM trains on z_macro, independent of PINN and decoders)
  - **Parallel Group**: Wave 3 (with Tasks 9, 10, 11)
  - **Blocks**: T15 (baseline experiments need LDM), T16 (SDEdit needs LDM)
  - **Blocked By**: T6 (needs frozen TS-JEPA to encode data into z_macro)

  **References**:

  **Pattern References**:
  - `main_phase2.py` — Phase 2 entry point. Check if LDM training is handled here
  - `src/pipelines/run_sdedit_phase2.py` — May contain LDM training + SDEdit in sequence. Check if LDM training can be run independently

  **API/Type References**:
  - `src/models/latent_diffusion.py:LatentDiffusionMLP` — The diffusion model. Check: `forward()` (noise prediction), `sample()` (generation), noise schedule params, z_macro input dimension
  - `src/configs.py` — LDM config: num_diffusion_steps, beta_schedule, learning rate, epochs

  **WHY Each Reference Matters**:
  - `run_sdedit_phase2.py`: LDM training may be embedded here. Need to extract or call the training portion only
  - `latent_diffusion.py`: Must understand the DDPM implementation — forward process (add noise), reverse process (denoise), loss (MSE on noise prediction)

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: LDM trains and generates plausible z_macro samples
    Tool: Bash (Python)
    Preconditions: TS-JEPA trained (T6)
    Steps:
      1. Encode training data through frozen TS-JEPA → z_macro dataset
      2. Train LDM on z_macro, log loss at epoch 1, N/2, final
      3. Verify loss decreased by at least 30%
      4. Save checkpoint
      5. Generate 100 unconditional z_macro samples
      6. Compute: mean and std of generated vs real z_macro — should be similar order of magnitude
      7. Project generated + real onto UMAP (random_state=42), save plot
    Expected Result: LDM checkpoint saved. Generated samples have similar distribution to real z_macro
    Failure Indicators: Loss doesn't decrease, generated samples are all identical, generated mean/std differs from real by >10×
    Evidence: .sisyphus/evidence/task-12-ldm-training.txt, .sisyphus/evidence/task-12-ldm-samples-umap.png
  ```

  **Evidence to Capture:**
  - [ ] task-12-ldm-training.txt — loss curve, generated vs real distribution stats
  - [ ] task-12-ldm-samples-umap.png — UMAP of generated + real z_macro

  **Commit**: YES (groups with T16)
  - Message: `feat(phase2): train LDM on z_macro latent space`
  - Files: `results/ldm.pth` (or equivalent checkpoint path)
  - Pre-commit: verify checkpoint loads and generates finite samples

---

### Wave 4 — Decoder 2 + Validation + Baselines

- [x] 13. Decoder 2 (CVAE Jitter) Training

  **What to do**:
  - Freeze both TS-JEPA (T6) and Decoder 1 (T11)
  - Compute Decoder 1 residuals: for each training sample, compute `residual = original_signal - decoder1_reconstruction`. This is the high-frequency "jitter" component
  - Train Decoder 2 CVAE (`src/models/decoder2_cvae.py`) to model the distribution of these residuals, conditioned on z_macro
  - Decoder 2 is a Conditional VAE: encoder maps (residual, z_macro) → latent z_jitter; decoder maps (z_jitter, z_macro) → reconstructed residual
  - **Smoke test**: 2-3 epochs, verify ELBO loss has both reconstruction and KL terms, both decrease, output shape matches residual shape
  - **Full training**: Train with early stopping on validation ELBO
  - Record: final ELBO, reconstruction loss, KL divergence, training epochs
  - Save checkpoint to `results/decoder2.pth`
  - **Quick validation**: Sample jitter from Decoder 2 for several z_macro values. Add to Decoder 1 output. Verify total reconstruction (envelope + jitter) looks more realistic than envelope alone

  **Must NOT do**:
  - Do NOT unfreeze TS-JEPA or Decoder 1
  - Do NOT modify CVAE architecture
  - Do NOT spend more than 3 training configs

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: VAE training requires monitoring both reconstruction and KL components. Needs Decoder 1 outputs as preprocessing step
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T14, T15 — once T11 is done)
  - **Parallel Group**: Wave 4 (with Tasks 14, 15)
  - **Blocks**: T14 (Phase 1 validation needs complete reconstruction), T16 (SDEdit needs full decoder chain)
  - **Blocked By**: T6 (TS-JEPA), T11 (Decoder 1 — needs its reconstructions to compute residuals)

  **References**:

  **Pattern References**:
  - `src/pipelines/train_phase1.py` — Check if Decoder 2 CVAE training is orchestrated here after Decoder 1
  - `main_phase1.py` — May handle all Phase 1 training in sequence

  **API/Type References**:
  - `src/models/decoder2_cvae.py` — CVAE class. Check: `forward(residual, z_macro)`, `encode()`, `decode()`, `sample()`, ELBO loss computation
  - `src/models/decoder1.py` — Need `forward(z_macro)` to get envelope, then `original - envelope = residual`

  **WHY Each Reference Matters**:
  - `decoder2_cvae.py`: Must understand the conditioning mechanism — how z_macro enters the CVAE encoder and decoder
  - `train_phase1.py`: May already handle the residual computation and Decoder 2 training pipeline

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Decoder 2 CVAE trains with decreasing ELBO
    Tool: Bash (Python)
    Preconditions: TS-JEPA (T6) and Decoder 1 (T11) trained
    Steps:
      1. Load frozen TS-JEPA + Decoder 1
      2. Compute residuals: original - decoder1_output for all training data
      3. Train Decoder 2 CVAE, log ELBO, recon_loss, KL at epoch 1, N/2, final
      4. Verify ELBO decreased
      5. Verify KL > 0 (posterior not collapsed to prior)
      6. Save checkpoint to results/decoder2.pth
    Expected Result: CVAE checkpoint saved. ELBO converged. KL term is positive (not posterior collapse)
    Failure Indicators: KL → 0 (posterior collapse), ELBO increases, recon_loss doesn't improve
    Evidence: .sisyphus/evidence/task-13-decoder2-training.txt

  Scenario: Full reconstruction (envelope + jitter) improves realism
    Tool: Bash (Python)
    Preconditions: Decoder 2 trained
    Steps:
      1. For 1 sample per class: compute envelope (Dec1), sample jitter (Dec2), combine
      2. Plot: original vs envelope-only vs envelope+jitter (3 subplots per class)
      3. Compute: RMSE(envelope-only, original) vs RMSE(envelope+jitter, original) — jitter should improve
    Expected Result: envelope+jitter reconstruction is closer to original than envelope alone
    Failure Indicators: Adding jitter makes reconstruction worse (higher RMSE)
    Evidence: .sisyphus/evidence/task-13-reconstruction-comparison.png
  ```

  **Evidence to Capture:**
  - [ ] task-13-decoder2-training.txt — ELBO, recon_loss, KL at key epochs
  - [ ] task-13-reconstruction-comparison.png — envelope-only vs envelope+jitter vs original

  **Commit**: YES (groups with T6, T11)
  - Message: `feat(phase1): train Decoder 2 CVAE for jitter modeling`
  - Files: `results/decoder2.pth`
  - Pre-commit: verify checkpoint loads

---

- [x] 14. Phase 1 Complete Validation + Twin Plots

  **What to do**:
  - **Full Phase 1 pipeline validation**: Load all trained Phase 1 components (TS-JEPA, Decoder 1, Decoder 2). Run the complete encoding-decoding pipeline: signal → TS-JEPA → z_macro → Decoder 1 → envelope → Decoder 2 → jitter → full reconstruction
  - **Quantitative metrics per class**: 
    - Reconstruction RMSE (full: envelope + jitter)
    - Decoder 1 envelope RMSE alone
    - CVAE ELBO on held-out validation set
    - z_macro UMAP Silhouette score (from T6 results)
  - **Twin plot generation (synthetic + MaFaulDa)**: 
    - If T8 synthetic results are available: create side-by-side comparison plots
    - Left panel: synthetic system reconstruction quality
    - Right panel: MaFaulDa reconstruction quality
    - Same axes, same scale, same styling for fair comparison
  - **Signal domain validation**:
    - Time domain: overlay reconstruction on original for 1 sample per class
    - Frequency domain: compare PSD (Power Spectral Density) of reconstruction vs original — fault-characteristic frequencies should be preserved
  - Save all figures for T18 (paper)

  **Must NOT do**:
  - Do NOT retrain any component
  - Do NOT generate more than 4 comparison plots (space-constrained paper)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Multi-component evaluation, metric computation, publication-quality figure generation
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T13 once T11 done, with T15)
  - **Parallel Group**: Wave 4 (with Tasks 13, 15)
  - **Blocks**: T18 (paper needs these figures and metrics)
  - **Blocked By**: T11 (Decoder 1), T13 (Decoder 2)

  **References**:

  **Pattern References**:
  - `tools/plot_imbalance_evaluation.py` — May have reusable plotting/evaluation code
  - `tools/visualize_processed_data.py` — Plotting utilities
  - T8 evidence files — Synthetic results for twin plot comparison

  **API/Type References**:
  - All Phase 1 model checkpoints: `results/ts_jepa.pth`, `results/decoder1.pth`, `results/decoder2.pth`

  **WHY Each Reference Matters**:
  - `plot_imbalance_evaluation.py`: 449 lines of evaluation tooling — may already implement the metrics and plots we need
  - T8 synthetic results: Twin plots require both synthetic AND MaFaulDa results side by side

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Full Phase 1 reconstruction pipeline verified
    Tool: Bash (Python)
    Preconditions: All Phase 1 models trained
    Steps:
      1. Load TS-JEPA, Decoder 1, Decoder 2
      2. Run: signal → z_macro → envelope → jitter → full_reconstruction
      3. Compute per-class: envelope RMSE, full RMSE, relative improvement
      4. Print metrics table
      5. Generate time-domain overlay plot (4 classes × 3 lines: original, envelope, full)
    Expected Result: Full reconstruction RMSE < envelope-only RMSE for all classes. Metrics documented
    Failure Indicators: Full reconstruction worse than envelope alone, NaN in reconstruction
    Evidence: .sisyphus/evidence/task-14-phase1-metrics.txt, .sisyphus/evidence/task-14-reconstruction-overlay.png

  Scenario: Twin plots (synthetic + MaFaulDa) generated
    Tool: Bash (Python)
    Preconditions: T8 synthetic results + Phase 1 MaFaulDa results both available
    Steps:
      1. Load synthetic Phase 1 results from results-synthetic/
      2. Load MaFaulDa Phase 1 results from results/
      3. Create twin plot: left=synthetic, right=MaFaulDa, same metrics
      4. Save as assets/fig_phase1_validation.pdf (300 DPI, vector, IEEE width)
    Expected Result: Twin plot figure ready for paper, both panels showing reconstruction quality
    Failure Indicators: Synthetic results missing (T8 didn't complete), vastly different scales between panels
    Evidence: .sisyphus/evidence/task-14-twin-plot-preview.png, assets/fig_phase1_validation.pdf
  ```

  **Evidence to Capture:**
  - [ ] task-14-phase1-metrics.txt — per-class metrics table
  - [ ] task-14-reconstruction-overlay.png — time-domain reconstruction overlay
  - [ ] task-14-twin-plot-preview.png — preview of twin plot
  - [ ] assets/fig_phase1_validation.pdf — paper-ready twin plot figure

  **Commit**: NO (evaluation only — figures saved separately)

---

- [x] 15. Run Baseline Experiments

  **What to do**:
  - Load baseline implementations from T7 (Vanilla DDPM + Label-conditioned DDPM)
  - Load trained LDM from T12 (for architecture/config parity reference)
  - **Train both baselines**: Using SAME z_macro data (from frozen TS-JEPA), SAME number of epochs, SAME batch size, SAME noise schedule as the main LDM
    - Vanilla DDPM: Train LatentDiffusionMLP without guidance → generates z_macro samples unconditionally
    - Label-conditioned DDPM: Train with class label conditioning → generates z_macro samples conditioned on target fault class
  - **Generate samples from each baseline**: For each fault class, generate N samples (same N as main method will generate in T16)
  - **Decode baseline samples**: Pass generated z_macro through frozen Decoder 1 + Decoder 2 → reconstructed signals
  - **Evaluate baseline quality**:
    - Compute PINN residual features on decoded signals (using best PINN from T10)
    - UMAP of baseline-generated features overlaid on real data
    - FID-like metric: MMD (Maximum Mean Discrepancy) between generated and real feature distributions
    - per-class generation quality: do generated samples land in correct fault region on UMAP?
  - Save all baseline checkpoints and results

  **Must NOT do**:
  - Do NOT tune baseline hyperparameters to make them look worse
  - Do NOT use different training data or preprocessing for baselines
  - Do NOT add extra training tricks to baselines that the main method doesn't have

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Training 2 models + comprehensive evaluation. Fairness-critical — must be systematic
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T13, T14)
  - **Parallel Group**: Wave 4 (with Tasks 13, 14)
  - **Blocks**: T17 (transfer classification needs baseline-generated data)
  - **Blocked By**: T7 (baseline code), T12 (LDM config reference + z_macro data)

  **References**:

  **Pattern References**:
  - T12 evidence — LDM training procedure. Baselines must follow EXACT same procedure
  - `src/models/baselines.py` (from T7) — Baseline implementations

  **API/Type References**:
  - `src/models/baselines.py:VanillaDDPM` — No-guidance DDPM baseline
  - `src/models/baselines.py:LabelConditionedDDPM` — Class-conditional baseline
  - Best PINN from T10 — For evaluating generated sample quality via physics features

  **WHY Each Reference Matters**:
  - T12 training procedure: Fairness requires identical training protocol. Same epochs, LR schedule, batch size
  - Best PINN: Quality evaluation uses physics features — the PINN is the discriminator that judges whether generated data is physically plausible

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Both baselines trained with identical protocol to main method
    Tool: Bash (Python)
    Preconditions: T7 baseline code, T12 training reference
    Steps:
      1. Train Vanilla DDPM: log loss curve, epochs, batch_size, LR
      2. Train Label-conditioned DDPM: log same metrics
      3. Verify: same epochs as main LDM (from T12), same noise schedule, same optimizer
      4. Generate 100 samples per class from each baseline
      5. Decode through Decoder 1 + 2
      6. Evaluate with PINN features: compute MMD between generated and real per class
    Expected Result: Both baselines trained. Training protocols match main LDM. Per-class generation quality documented
    Failure Indicators: Training protocol differs (more/fewer epochs), baseline loss doesn't converge
    Evidence: .sisyphus/evidence/task-15-baseline-training.txt

  Scenario: Baseline UMAP shows physics-guided method is superior
    Tool: Bash (Python)
    Preconditions: Baseline samples generated
    Steps:
      1. Extract PINN features from: real data, Vanilla DDPM samples, Label-conditioned samples
      2. Plot UMAP with 3 groups overlaid (use different markers: real=dots, vanilla=triangles, label-cond=squares)
      3. Compute per-class MMD for each method vs real data
      4. Print comparison: {method, class, MMD_score}
    Expected Result: Baseline samples less well-clustered than physics-guided (to be generated in T16). MMD documented
    Failure Indicators: Baselines cluster perfectly (our method has no advantage)
    Evidence: .sisyphus/evidence/task-15-baseline-umap.png, .sisyphus/evidence/task-15-baseline-metrics.txt
  ```

  **Evidence to Capture:**
  - [ ] task-15-baseline-training.txt — training logs, protocol parity verification
  - [ ] task-15-baseline-umap.png — UMAP overlay of baseline samples + real data
  - [ ] task-15-baseline-metrics.txt — per-class MMD scores

  **Commit**: YES
  - Message: `feat(baselines): train and evaluate Vanilla DDPM + Label-conditioned baselines`
  - Files: baseline checkpoints, `src/models/baselines.py` if modified
  - Pre-commit: verify both baselines generate finite samples

---

### Wave 5 — Generation + Evaluation + Paper

- [x] 16. Physics-Guided SDEdit Counterfactual Generation

  **What to do**:
  - This is the CORE CONTRIBUTION of the paper — generating counterfactual fault signals guided by PINN physics
  - **Load all components**: Best PINN (from T10), frozen TS-JEPA (T6), trained LDM (T12), Decoder 1 (T11), Decoder 2 (T13), PriorWorkOracle (wrapping PINN + MathFeatureExtractor)
  - **SDEdit procedure** (from `src/pipelines/run_sdedit_phase2.py`):
    1. Start with healthy z_macro (encode healthy signals through TS-JEPA)
    2. Add noise to z_macro (forward diffusion to timestep t_start)
    3. Run reverse diffusion with PINN oracle guidance: at each denoising step, compute VJP gradient from oracle, steer z_macro toward target fault class physics
    4. Decode final z_macro through Decoder 1 + Decoder 2 → counterfactual signal
  - **Generate for all 3 fault classes**: For each of {Imbalance, Vertical Misalignment, Overhang Ball Bearing}, generate N counterfactual samples starting from healthy data. N should match or exceed the smallest real fault class size
  - **Evaluate generated counterfactuals**:
    - Extract PINN features from generated signals
    - UMAP: overlay generated on real data — generated samples should land near real fault clusters
    - Time domain: compare generated traces with real fault signals visually
    - Frequency domain: verify fault-characteristic frequencies are present in generated signals
    - MMD between generated and real fault distributions (compare with T15 baseline MMD)
  - **SDEdit trajectory visualization**: For 1 representative generation per class, capture z_t at every denoising step. Project trajectory onto UMAP. This shows how physics guidance "bends" the diffusion path toward the target fault region
  - **If PINN fallback was triggered** (T9/T10): Adapt SDEdit guidance accordingly. If using "continuous residual features" approach, oracle guidance may be weaker but should still provide directional bias

  **Must NOT do**:
  - Do NOT modify the SDEdit algorithm beyond what T3 fixed
  - Do NOT cherry-pick generated samples for visualization
  - Do NOT tune SDEdit hyperparameters (t_start, guidance_scale) beyond 3 configs
  - Do NOT generate for fault classes outside the 3 specified

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Most architecturally complex task — SDEdit with physics guidance requires understanding the full pipeline interaction (PINN → VJP → diffusion → decoders). High reasoning load
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: Partially (with T17 after first batch of generated samples is ready)
  - **Parallel Group**: Wave 5 (with Tasks 17, 18)
  - **Blocks**: T17 (transfer classification needs generated data), T18 (paper needs figures)
  - **Blocked By**: T9/T10 (best PINN), T11 (Decoder 1), T12 (LDM), T13 (Decoder 2)

  **References**:

  **Pattern References**:
  - `src/pipelines/run_sdedit_phase2.py` — THE SDEdit pipeline. This is the code that orchestrates generation. Study it thoroughly: noise schedule, guidance injection point, denoising loop, trajectory capture
  - `main_phase2.py` — Phase 2 entry point. Check how it invokes SDEdit and what arguments it passes

  **API/Type References**:
  - `src/models/oracles.py:PriorWorkOracle` — Oracle wrapper. Check: `compute_guidance()` or equivalent — how VJP is computed and returned, what the guidance_scale parameter does
  - `src/models/latent_diffusion.py:LatentDiffusionMLP` — Reverse diffusion sampling loop. Check: `sample()` method, how external guidance gradients are injected
  - `src/models/feature_extractors.py:MathFeatureExtractor` — For evaluating generated samples

  **External References**:
  - `main_M2_6_pages.tex` Section on SDEdit — Paper describes the generation procedure. Generated results must match what the paper claims

  **WHY Each Reference Matters**:
  - `run_sdedit_phase2.py`: This IS the generation code. Any bugs here directly affect the paper's main result
  - `PriorWorkOracle`: The physics guidance mechanism. If VJP computation is wrong, generated samples won't be physics-guided
  - Paper LaTeX: Must generate exactly what the paper claims — SDEdit trajectories, physics-guided counterfactuals for 3 specific fault classes

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: SDEdit generates counterfactual samples for all 3 fault classes
    Tool: Bash (Python)
    Preconditions: All Phase 1 + 2 components loaded
    Steps:
      1. Run SDEdit generation for each fault class (Imbalance, Vert. Misalign., Overhang BB)
      2. Generate N samples per class (N = min real class size)
      3. Verify: generated signal shapes match real signal shapes (channels, seq_length)
      4. Verify: no NaN/Inf in generated signals
      5. Extract PINN features from generated signals
      6. Compute Silhouette score of generated features against real class centroids
      7. Compute per-class MMD (generated vs real)
    Expected Result: 3 × N generated samples. MMD lower than baselines (T15). Silhouette > 0 on combined (real + generated)
    Failure Indicators: NaN in generated signals, all generated samples identical, MMD worse than baselines
    Evidence: .sisyphus/evidence/task-16-sdedit-generation.txt

  Scenario: SDEdit trajectory shows physics guidance effect on UMAP
    Tool: Bash (Python)
    Preconditions: Trajectory captured during generation
    Steps:
      1. For 1 generation per class: plot z_t trajectory on UMAP (T6 UMAP projection)
      2. Color trajectory points by denoising step (start=light, end=dark)
      3. Mark: starting point (healthy cluster), ending point (target fault cluster)
      4. Save trajectory plot with real data clusters in background
    Expected Result: Trajectory visibly moves from healthy cluster toward target fault cluster
    Failure Indicators: Trajectory is random walk (no directionality), ends far from target cluster
    Evidence: .sisyphus/evidence/task-16-sdedit-trajectory.png

  Scenario: Generated signals are physically plausible in time + frequency domain
    Tool: Bash (Python)
    Preconditions: Generated signals available
    Steps:
      1. For 1 sample per class: plot time-domain overlay (generated vs real)
      2. Compute PSD for generated and real, plot frequency domain comparison
      3. Check: fault-characteristic frequency peaks present in generated signals?
    Expected Result: Generated signals visually similar to real faults, characteristic frequencies preserved
    Failure Indicators: Generated signals are noise, no characteristic frequencies, completely different waveform shape
    Evidence: .sisyphus/evidence/task-16-signal-comparison.png
  ```

  **Evidence to Capture:**
  - [ ] task-16-sdedit-generation.txt — sample counts, MMD scores, Silhouette
  - [ ] task-16-sdedit-trajectory.png — UMAP trajectory visualization
  - [ ] task-16-signal-comparison.png — time + frequency domain comparison

  **Commit**: YES
  - Message: `feat(phase2): generate physics-guided counterfactual signals via SDEdit`
  - Files: generated data outputs, any evaluation scripts
  - Pre-commit: verify generated samples exist and are finite

---

- [x] 17. Transfer Classification Experiment (TSTR)

  **What to do**:
  - **THE KEY QUANTITATIVE RESULT for the paper**: Train-Synthetic-Test-Real (TSTR) classification
  - **Setup classifier**: Simple architecture — logistic regression OR small MLP (2 layers, 64 hidden). Must be same classifier for ALL methods. Use PINN residual features (2240-dim or PCA-reduced from T10) as input, NOT raw signals
  - **Experiment design** (for EACH method):
    - **Ours (Physics-guided SDEdit)**: Train classifier on T16 generated data → test on real MaFaulDa test set
    - **Baseline 1 (Vanilla DDPM)**: Train on T15 Vanilla DDPM generated data → test on same real test set
    - **Baseline 2 (Label-conditioned)**: Train on T15 Label-conditioned generated data → test on same real test set
    - **Upper bound (Real-only)**: Train on real MaFaulDa train set → test on real test set
  - **Statistical rigor**: Run EACH experiment with 5 different random seeds. Report mean ± std accuracy
  - **Train/test split**: Use STRATIFIED split (same class proportions). Test set must be the EXACT same across all methods (same seed for split)
  - **Metrics per experiment**: 4-class accuracy, per-class F1 score, confusion matrix
  - **Significance test**: Paired t-test or Wilcoxon between our method and each baseline (5 seeds). Report p-values
  - **Create results table**: Main result table for the paper

  **Must NOT do**:
  - Do NOT use a complex classifier (no ResNets, no Transformers) — simple classifier isolates data quality
  - Do NOT use different classifiers for different methods
  - Do NOT cherry-pick seeds
  - Do NOT use different test sets for different methods

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Systematic experiment design with statistical rigor. Multiple training runs, careful seed management, significance testing
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: Partially (with T18 after table is ready)
  - **Parallel Group**: Wave 5 (with Tasks 16, 18)
  - **Blocks**: T18 (paper needs results table)
  - **Blocked By**: T15 (baseline generated data), T16 (physics-guided generated data)

  **References**:

  **Pattern References**:
  - T8 evidence — Synthetic fairness validation used similar TSTR setup. Reuse classifier code
  - `tools/plot_imbalance_evaluation.py` — May have classification evaluation code

  **API/Type References**:
  - `sklearn.linear_model.LogisticRegression` or `sklearn.neural_network.MLPClassifier` — Simple classifiers
  - `sklearn.model_selection.StratifiedKFold` — For stratified split
  - `sklearn.metrics.accuracy_score, f1_score, confusion_matrix` — Evaluation metrics
  - `scipy.stats.wilcoxon` — Statistical significance test

  **External References**:
  - `main_M2_6_pages.tex` — Paper's experimental design section. Results must match the claimed evaluation protocol

  **WHY Each Reference Matters**:
  - Same classifier for all methods: fairness. Any classifier difference confounds the comparison
  - Statistical significance: 5 seeds + p-values are expected by IEEE reviewers
  - Paper LaTeX: Results table structure must match what the paper's Results section promises

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: TSTR experiment completes for all 4 methods × 5 seeds
    Tool: Bash (Python)
    Preconditions: All generated data available (T15, T16), real MaFaulDa test set defined
    Steps:
      1. Define stratified train/test split of real data (seed=42 for split)
      2. For each method (ours, vanilla, label-cond, real-only):
         For each seed (0-4):
           Train classifier, evaluate on test set, record accuracy + per-class F1
      3. Compute per-method: mean ± std accuracy across 5 seeds
      4. Run Wilcoxon test: ours vs vanilla, ours vs label-cond
      5. Print results table
    Expected Result: 4 methods × 5 seeds = 20 training runs. Our method accuracy > baselines. p-value < 0.05 preferred
    Failure Indicators: Our method worse than baselines, accuracy < 25% (random chance for 4 classes)
    Evidence: .sisyphus/evidence/task-17-tstr-results.txt

  Scenario: Results table is paper-ready
    Tool: Bash (Python)
    Preconditions: All experiments complete
    Steps:
      1. Format LaTeX table: columns = Method | Accuracy (mean±std) | F1 (mean±std) | p-value
      2. Rows: Physics-guided (Ours), Vanilla DDPM, Label-conditioned, Real-only (upper bound)
      3. Bold best result
      4. Save LaTeX snippet to .sisyphus/evidence/task-17-results-table.tex
    Expected Result: Copy-pasteable LaTeX table
    Failure Indicators: Missing methods, no std reported, no significance values
    Evidence: .sisyphus/evidence/task-17-results-table.tex, .sisyphus/evidence/task-17-confusion-matrices.png
  ```

  **Evidence to Capture:**
  - [ ] task-17-tstr-results.txt — full results: 4 methods × 5 seeds × per-class metrics
  - [ ] task-17-results-table.tex — LaTeX table snippet
  - [ ] task-17-confusion-matrices.png — confusion matrix for each method (best seed)

  **Commit**: YES
  - Message: `feat(eval): transfer classification TSTR experiment with 5 seeds`
  - Files: experiment scripts, results
  - Pre-commit: verify results table has all 4 methods

---

- [x] 18. Generate Paper Figures + Tables + Update LaTeX

  **What to do**:
  - **Collect all evidence files** from T4 (spec), T8 (synthetic), T10 (PINN UMAP), T14 (Phase 1 validation), T15 (baselines), T16 (SDEdit), T17 (TSTR)
  - **Create missing paper infrastructure**:
    - Create `assets/` directory if it doesn't exist
    - Create `references.bib` — extract all `\cite{...}` keys from `main_M2_6_pages.tex` and populate with proper BibTeX entries. The paper cites works on PINNs, diffusion models, SDEdit, TS-JEPA, ReLoBRaLo, MaFaulDa, etc. Use official publication data for each entry
    - Create or generate `method_diagram.png` — the pipeline architecture figure referenced at line 294 of the LaTeX. This should show the Phase 0→1→2 flow (PINN → TS-JEPA → Decoders → LDM → SDEdit). Can be a clean block diagram
    - Update existing `\includegraphics` paths in the LaTeX: the current references to `assets/synth_evaluation_imbalance_pt1.png`, `assets/synth_evaluation_imbalance_pt2.jpg`, and `assets/synth_umap_sdedit_trajectory_imbalance.png` must be updated to point to the new figure file paths generated in this task
  - **Generate final figures** per T4 specification (expected ~3-4 composite figures):
    - **Figure 1 — Component Validation**: Twin plot (synthetic left, MaFaulDa right). PINN UMAP (T10) + z_macro UMAP (T6). Shows physics features cluster by fault class
    - **Figure 2 — SDEdit Generation**: SDEdit trajectory on UMAP (T16) + generated vs real signal comparison (time domain). Shows physics-guided navigation from healthy to fault
    - **Figure 3 — Baseline Comparison**: UMAP overlay of all methods (T15 + T16). Visual ablation showing physics guidance advantage
    - **Figure 4 (if space)**: Frequency domain validation or per-class reconstruction quality
  - **All figures must be**: 
    - PDF/vector format, 300 DPI minimum
    - IEEE double-column width: 3.5in (single col) or 7.0in (double col)
    - Colorblind-friendly palette (use viridis/colorblind-safe discrete palette)
    - Clear legends, axis labels, readable font size (≥8pt in final print)
  - **Generate final table** (T17 results → LaTeX table)
  - **Update `main_M2_6_pages.tex`**:
    - Fill in Results section: reference figures, report key numbers, compare methods
    - Fill in Discussion section: interpret results, acknowledge limitations, propose future work
    - Update Conclusion: summarize findings
    - Ensure all `\includegraphics` point to correct `assets/` paths
    - Verify paper compiles: `pdflatex main_M2_6_pages.tex`
  - **Space management**: Total paper is 6 pages. Introduction + Related Work + Methodology are done. Results + Discussion + Conclusion must fit remaining space (~2-2.5 pages). Prioritize figures over text

  **Must NOT do**:
  - Do NOT rewrite Introduction, Related Work, or Methodology sections
  - Do NOT exceed 4 figures + 2 tables
  - Do NOT fabricate or misrepresent results
  - Do NOT change experimental numbers from what was actually measured

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: Academic paper writing + figure compilation. Requires strong technical writing, LaTeX skills, and understanding of IEEE format constraints
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO (needs ALL upstream results)
  - **Parallel Group**: Wave 5 (last task before verification)
  - **Blocks**: F1-F4 (final verification)
  - **Blocked By**: T4 (figure spec), T8 (synthetic), T10 (PINN UMAP), T14 (Phase 1 validation), T17 (TSTR results)

  **References**:

  **Pattern References**:
  - `main_M2_6_pages.tex` — The paper itself. Read existing sections for writing style, notation conventions, and cross-reference patterns. **NOTE**: Currently references `method_diagram.png` (line 294), `assets/synth_evaluation_imbalance_pt1.png` (line 394), `assets/synth_evaluation_imbalance_pt2.jpg` (line 401), `assets/synth_umap_sdedit_trajectory_imbalance.png` (line 407) — NONE of these files exist yet. Also uses `\bibliography{references}` (line 427) but `references.bib` does not exist
  - `references.bib` — DOES NOT EXIST. Must be created from scratch. Extract all `\cite{key}` from the LaTeX and populate with proper BibTeX entries
  - `assets/` — Directory DOES NOT EXIST. Must be created. All figure PDFs will go here

  **API/Type References**:
  - All evidence files from T4-T17 in `.sisyphus/evidence/`
  - All figure assets in `assets/`

  **External References**:
  - IEEE conference formatting guidelines
  - `main_M2_6_pages.tex` preamble — Check loaded packages, figure macros, table macros

  **WHY Each Reference Matters**:
  - `main_M2_6_pages.tex`: Must match existing writing style, notation (e.g., $z_{\text{macro}}$ not $z_{macro}$), and cross-reference scheme
  - Evidence files: These contain the ACTUAL numbers to report. Never fabricate — copy from evidence

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Paper infrastructure created (references.bib, method_diagram, assets/)
    Tool: Bash
    Preconditions: LaTeX file readable, citation keys extractable
    Steps:
      1. Create assets/ directory: mkdir assets
      2. Extract all \cite{...} keys from main_M2_6_pages.tex
      3. Create references.bib with proper BibTeX entries for each citation key
      4. Create method_diagram.png (pipeline block diagram: Phase 0→1→2)
      5. Verify: references.bib is valid BibTeX (bibtex main_M2_6_pages parses without errors)
    Expected Result: references.bib created with all cited entries, method_diagram.png created, assets/ directory exists
    Failure Indicators: Missing citation entries, invalid BibTeX syntax, bibtex errors
    Evidence: .sisyphus/evidence/task-18-paper-infra.txt

  Scenario: Paper compiles with all figures and tables
    Tool: Bash
    Preconditions: All figure PDFs in assets/, LaTeX updated, references.bib created
    Steps:
      1. Run: pdflatex main_M2_6_pages.tex (twice for cross-refs)
      2. Run: bibtex main_M2_6_pages
      3. Run: pdflatex main_M2_6_pages.tex (twice more for bibliography)
      4. Verify: no errors, no undefined references, no missing figures
      5. Check: output PDF exists, page count ≤ 6
      6. Verify: all figure paths resolve (no "Figure ??" in text)
    Expected Result: Paper compiles cleanly. 6 pages. All figures visible. Bibliography renders
    Failure Indicators: Compilation errors, missing figures, missing bib entries, page overflow (> 6 pages)
    Evidence: .sisyphus/evidence/task-18-latex-compile.txt

  Scenario: Results section contains all required numbers
    Tool: Bash (grep)
    Preconditions: LaTeX updated
    Steps:
      1. Grep main_M2_6_pages.tex for: "Silhouette", "accuracy", "TSTR", "MMD", "RMSE"
      2. Verify each metric appears with a concrete number (not placeholder)
      3. Check: TSTR results match task-17-tstr-results.txt exactly
      4. Check: PINN Silhouette matches task-10-variant-comparison.txt exactly
    Expected Result: All key metrics present in paper with correct values matching evidence
    Failure Indicators: Placeholder values ("XX%"), numbers not matching evidence files
    Evidence: .sisyphus/evidence/task-18-results-verification.txt

  Scenario: Paper fits within 6-page limit
    Tool: Bash
    Preconditions: Paper compiled
    Steps:
      1. Check PDF page count
      2. If > 6 pages: identify which content to trim (figures smaller, text more concise)
      3. If < 5 pages: identify if more detail should be added
    Expected Result: Exactly 6 pages (or 5.5-6.0 range)
    Failure Indicators: > 6 pages (rejected by conference), < 4 pages (looks incomplete)
    Evidence: .sisyphus/evidence/task-18-page-count.txt
  ```

  **Evidence to Capture:**
  - [ ] task-18-paper-infra.txt — references.bib creation log, method_diagram confirmation
  - [ ] task-18-latex-compile.txt — compilation log (pdflatex + bibtex)
  - [ ] task-18-results-verification.txt — number cross-check against evidence files
  - [ ] task-18-page-count.txt — final page count

  **Commit**: YES
  - Message: `docs(paper): add results, figures, references.bib, and complete LaTeX`
  - Files: `main_M2_6_pages.tex`, `references.bib`, `method_diagram.png`, `assets/*.pdf`
  - Pre-commit: `pdflatex main_M2_6_pages.tex` compiles without errors

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.

- [x] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists (run commands, check outputs). For each "Must NOT Have": search codebase for forbidden patterns — reject with file:line if found. Check evidence files exist in `.sisyphus/evidence/`. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

  ```
  Scenario: All Must Have deliverables verified
    Tool: Bash (grep, python, file checks)
    Preconditions: All T1-T18 marked complete
    Steps:
      1. Read .sisyphus/plans/irai-pipeline-results.md — extract all "Must Have" items
      2. For each Must Have: run the verification command or check file existence
         - PINN checkpoint: verify results/pinn.pth exists and loads without error
         - UMAP plots: verify results/umap_*.png files exist
         - Transfer classification: verify .sisyphus/evidence/task-17-*.txt files contain accuracy numbers
         - LaTeX: run pdflatex main_M2_6_pages.tex and verify 0 errors
         - Baselines: verify .sisyphus/evidence/task-15-*.txt files exist
      3. For each "Must NOT Have": grep codebase for forbidden patterns (architecture changes, extra speeds, etc.)
      4. Count evidence files in .sisyphus/evidence/ — verify at least one per task
    Expected Result: All Must Have items present, all Must NOT Have items absent, all evidence files exist
    Failure Indicators: Any Must Have missing, any forbidden pattern found, evidence directory sparse
    Evidence: .sisyphus/evidence/F1-compliance-audit.txt
  ```

- [x] F2. **Code Quality Review** — `unspecified-high`
  Run `pytest tests/` + check all training scripts complete. Review changed files for: `as any`/`@ts-ignore` (N/A for Python), empty excepts, print statements in production paths, commented-out code, unused imports. Check for dtype consistency (float64/float32 boundaries logged). Verify all checkpoints save correctly.
  Output: `Tests [N pass/N fail] | Files [N clean/N issues] | VERDICT`

  ```
  Scenario: Tests pass and code quality checks clean
    Tool: Bash
    Preconditions: All implementation tasks complete
    Steps:
      1. Run: pytest tests/ -v — capture output, count pass/fail
      2. Run: python -c "import py_compile; py_compile.compile('src/models/pinn.py', doraise=True)" for each src/ file — verify no syntax errors
      3. Grep all src/ files for: bare "except:" (empty catch), "# TODO" or "# HACK" left behind, "print(" in non-tool files
      4. Grep for dtype red flags: "float32" in pinn.py (should be float64), ".float()" calls near PINN boundaries
      5. For each checkpoint in results/*.pth: torch.load() and verify keys match expected model state_dict
    Expected Result: All tests pass, no syntax errors, no quality red flags, all checkpoints loadable
    Failure Indicators: Test failures, bare excepts, dtype crossing without explicit cast, corrupt checkpoints
    Evidence: .sisyphus/evidence/F2-code-quality.txt
  ```

- [x] F3. **Full Pipeline QA** — `unspecified-high`
  Start from clean state. Run full pipeline: `main_phase0.py` → `main_phase1.py` → `main_phase2.py`. Verify each produces expected outputs. Check UMAP plots visually (clusters visible?). Check generated signals visually (plausible?). Check transfer classification numbers (above chance?). Save evidence to `.sisyphus/evidence/final-qa/`.
  Output: `Scenarios [N/N pass] | Integration [N/N] | VERDICT`

  ```
  Scenario: Full pipeline end-to-end execution succeeds
    Tool: Bash
    Preconditions: MaFaulDa data downloaded and processed, all checkpoints from T1-T18 available
    Steps:
      1. Run: python main_phase0.py — verify completes without error, results/pinn.pth updated
      2. Run: python main_phase1.py — verify completes, results/ts_jepa.pth + decoder*.pth updated
      3. Run: python main_phase2.py — verify completes, UMAP + generated samples saved
      4. Check results/ directory: count .pth files (expect >= 5), count .png files (expect >= 3)
      5. Check transfer classification output: accuracy > 25% (above 4-class chance)
      6. Check UMAP plots: use look_at tool to verify clusters are visually distinguishable
    Expected Result: All 3 phases complete, checkpoints saved, UMAP shows clusters, classification above chance
    Failure Indicators: Any phase crashes, NaN in logs, UMAP shows single blob, accuracy < 25%
    Evidence: .sisyphus/evidence/final-qa/F3-pipeline-run.txt, .sisyphus/evidence/final-qa/F3-umap-check.png

  Scenario: Cross-task integration — synthetic + MaFaulDa results consistent
    Tool: Bash
    Preconditions: Both synthetic and MaFaulDa pipeline runs complete
    Steps:
      1. Verify results-synthetic/ contains checkpoint files and UMAPs
      2. Verify results/ contains MaFaulDa checkpoint files and UMAPs
      3. Check that twin plot figures in assets/ contain both panels (synthetic left, MaFaulDa right)
      4. Verify LaTeX compiles with all figure references resolved: pdflatex main_M2_6_pages.tex | grep "Warning.*undefined"
    Expected Result: Both result sets present, twin plots have both panels, LaTeX compiles cleanly
    Failure Indicators: Missing result set, single-panel figures, undefined LaTeX references
    Evidence: .sisyphus/evidence/final-qa/F3-integration.txt
  ```

- [x] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", verify actual changes match spec. Check 16Hz only used (no other speeds). Check exactly 3 fault classes + healthy. Check no architecture experimentation. Check no cosmetic refactoring. Check reproducibility artifacts (seeds, configs, git hash logged). Flag unaccounted changes.
  Output: `Tasks [N/N compliant] | Guardrails [N/N respected] | VERDICT`

  ```
  Scenario: All guardrails respected — no scope creep
    Tool: Bash (grep, git diff)
    Preconditions: All implementation tasks complete
    Steps:
      1. Grep src/configs.py for TARGET_HZ — must be exactly 16.0, no other values
      2. Grep src/configs.py for NUM_CLASSES — must be exactly 4
      3. Grep src/models/pinn.py for hidden layer sizes — must match [128,128], no changes from plan
      4. Run: git diff main --stat — list all changed files. For each: verify it was mentioned in at least one task
      5. Check for cosmetic-only changes: files with only whitespace/comment/import-order diffs
      6. Grep for random seeds: verify random_state=42 or equivalent in UMAP calls and training scripts
      7. Check results/ for unexpected files not mentioned in any task
    Expected Result: All guardrails respected, no unaccounted files, reproducibility seeds present
    Failure Indicators: Wrong TARGET_HZ, architecture changes, unaccounted file changes, missing seeds
    Evidence: .sisyphus/evidence/F4-scope-fidelity.txt
  ```

---

## Commit Strategy

- After T1 (data processing): `feat(data): configure 16Hz 4-class setup and download+process MaFaulDa`
- After T3 (code fixes): `fix(pipeline): correct critical issues found in code audit`
- After T8 (synthetic): `feat(synthetic): add paper-aligned 1D oscillator generator and synthetic pipeline routing`
- After T9+T10 (PINN): `fix(pinn): resolve PINN training issues, validate residual clustering`
- After T11+T13 (decoders): `feat(phase1): train validated Decoder 1 + Decoder 2 CVAE`
- After T16+T17 (generation+classification): `feat(phase2): complete SDEdit generation and transfer classification`
- After T18 (paper): `docs(paper): add results, figures, references.bib, and complete LaTeX`

---

## Success Criteria

### Verification Commands
```bash
pytest tests/ -v                           # All existing tests pass
python main_phase0.py                      # PINN training completes, loss < threshold
python main_phase1.py                      # Phase 1 components trained and saved
python main_phase2.py                      # SDEdit generates counterfactuals
pdflatex main_M2_6_pages.tex              # Paper compiles without errors
```

### Final Checklist
- [ ] PINN residual UMAP shows visible fault clusters (Silhouette > 0.3)
- [ ] All Phase 1 components validated (envelope RMSE, CVAE ELBO documented)
- [ ] SDEdit trajectory navigates from healthy to target fault on UMAP
- [ ] Transfer classification TSTR accuracy > 50% (4-class)
- [ ] Baselines show lower performance than physics-guided method
- [ ] Twin plots (synthetic + MaFaulDa) for all key results
- [ ] LaTeX compiles with all figures and tables
- [ ] All training runs logged with seeds, configs, metrics
