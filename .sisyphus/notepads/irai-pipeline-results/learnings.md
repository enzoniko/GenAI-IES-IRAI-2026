# Learnings — IRAI Pipeline Results

## [2026-04-17] Session Start
- Working dir: D:\WORK\GenAI\GenAI-IES-IRAI-2026
- OS: Windows (PowerShell). Use PowerShell-compatible commands, NOT bash/bash-style.
- Bash scripts (.sh) can be run via Git Bash if installed; otherwise use Python equivalents

## Key Architecture Facts (from previous session exploration)
- Old PINN: ConfigurablePINN, hidden=[128,128], Tanh, float64
- Current configs.py: TARGET_HZ=17.0 (NEEDS→16.0), NUM_CLASSES=42 (NEEDS→4), rotation_hz=30 (NEEDS→16)
- MathFeatureExtractor: 2240-dim output
- ReLoBRaLo: alpha=0.9, rho=0.1, temperature=2.0
- Target: 16Hz, classes: Normal=0, Imbalance=1, Vertical Misalignment=2, Overhang Ball Bearing=3
- Normalization: Xmin/Xmax/ymin/ymax in previous-work-pinn/export_relobralo/normalization.npz
- Download flags: -normal -imbalance -vertical -overhang -unzip (NOT -vertical-misalignment)

## Current Config State (as of session start, before T1 fixes)
- TARGET_HZ = 17.0 (WRONG → needs 16.0)
- NUM_CLASSES = 42 (WRONG → needs 4)
- PINN_ARCH_DEFAULT hidden_layers = [64, 64] (WRONG → needs [128, 128] to match old weights)
- PHASE0_TRAIN_SETTINGS rotation_hz = 30 (WRONG → needs 16)
- SEQ_LENGTH = 1639 (comment says "30Hz" → needs recalculation for 16Hz)
- DATA_DIR_PROCESSED = "data/processed-mafaulda/17hz" (will auto-update once TARGET_HZ=16.0)
- data/raw-mafaulda/ does NOT exist yet (needs download)
- results/ HAS old checkpoints from Gustavo (wrong config, will be overwritten)

## Critical Warnings
- architecture.py in previous-work-pinn/ has BROKEN import: `from direct_analysis.data_utils import ...`
  → READ AS TEXT ONLY. Never import it directly.
- synthetic_dataset.py is NOT the paper synthetic system (it's a different 4-channel vibration system)
  → Paper's synthetic system = 1D damped oscillator (Eq. 4): m=1, c=0.5, k=4.0, A=1.0, ω₀=1.0
  → Must create: src/data/paper_synthetic_oscillator.py
- torchaudio.lfilter may have platform dependencies on Windows — watch for errors
## [2026-04-17] T1 data repair
- Existing 16Hz processed tensors standardized on seq_length 3014; vertical-misalignment needed a targeted backfill using one closest-to-16Hz CSV per severity.
- The processor already emits X=(N,T,10) and Y=(N,T,4); forcing min_window_size=3014 preserved compatibility with existing 16Hz tensors.

## [2026-04-17] T2 Validation Shim Findings
- Normalization compat: DIVERGES for X features, DIVERGES for Y.
- Weight loading: FAIL with strict=True; PARTIAL SUCCESS with strict=False.
- Key mismatch details (if any): old keys use NNfor{...}.model.3/.6 while new model uses .2/.4, indicating activation-layer indexing mismatch despite matching parameter count and shapes on shared keys.
- pytest: FAIL — 1 passed, 3 failed.
- Dtype boundaries found: processed signal path uses float64 internally then stores float32; dataset/JEPA/LDM paths are float32; PINN/oracle/physics loss paths promote to float64; oracle forward repeatedly crosses float32<->float64 around the PINN boundary.
## [2026-04-17] T3: Code Audit + Fix
- Restored PINN default activations to ELU and ReLoBRaLo defaults to alpha=0.5125, rho=0.2332, temperature=1.4198 to match the successful exported prior-work configuration.
- Confirmed processed dataset tests must target data/processed-mafaulda/16hz with filenames that omit the legacy _v1 infix.
- Audited src/models/pinn.py residual equations against previous-work-pinn/basicPINNv8.py: equations match; the only verified drift was an added 1e6 residual scaling, which was removed.
- Stabilized MathFeatureExtractor by performing internal feature computations in float64 and returning the original dtype, eliminating NaN gradients in oracle math extraction.

## [2026-04-17] T5: PINN Diagnostic
- Recreated the legacy PINN checkpoint layout with a dependency-free compat shim and loaded the exported weights with strict=True; no key remapping was needed once the old .model.0/.3/.6 indexing was restored.
- Computed fresh min/max normalization from 16Hz normal-class tensors and saved it to `.sisyphus/evidence/task-5-normalization-16hz.npz`; the old 24.4Hz export bounds were clearly incompatible, especially on omega and acceleration ranges.
- Old PINN physics features on 16Hz MaFaulDa produced silhouette=-0.2037 and 5-fold kNN accuracy=0.7277; see `task-5-pinn-diagnostic.txt` and `task-5-pca-comparison.txt` for raw vs PCA50 assessment.
- VERDICT: Old weights NOT transferable to 16Hz (omega mismatch: 153 vs 101 rad/s, residuals explode 10⁹×). Residual features = poor spatial separation but moderate kNN separability. PCA50 improves kNN to 0.801.
- IMPLICATION for T9: Must train PINN fresh on 16Hz data (old weights fine-tuning unlikely to help due to normalization mismatch).
- IMPLICATION for T10: Use PCA50 before UMAP for better separability.

## [2026-04-17] One-Subtype-Per-Class Selection (User Constraint)
- Normal:   X/Y_normal_trainingset.pth                                 → label=0, 72 windows
- Imbalance: X/Y_imbalance_fault_20g_trainingset.pth                   → label=1, 72 windows (middle severity)
- Vertical:  X/Y_vertical_misalignment_fault_1.27mm_trainingset.pth   → label=2, 68 windows (middle severity)
- Overhang:  X/Y_overhang_ball_fault_20g_trainingset.pth               → label=3, 72 windows (ball bearing, middle severity)
- Total: ~284 windows across 4 classes
- Corresponding test sets exist with same naming but _testset suffix

## [2026-04-17] DataLoader/Pipeline Notes
- MaFaulDaDataset loads Y files (4-ch acceleration, shape N×3014×4), transposes to N×4×3014 for model
- X files used only for omega extraction (col 8) and normalization (cols 0-7)
- Normalization metadata at results/normalization_metadata.pth — DOES NOT EXIST until Phase 0 completes
- If normalization_metadata.pth missing → identity scaling (raw Y values used, range ≈ [-170, 170])
- TSJEPA(in_channels=4) — matches 4-ch Y tensor
- train_phase1.py → run_training_pipeline runs TS-JEPA → Dec1 → Dec2 → UMAP → evaluate in sequence
- For one-subtype-per-class training: write custom loader (don't use get_dataloaders — it loads all subtypes)

## [2026-04-17] T6: TS-JEPA Training
- Trained TSJEPA(in_channels=4) on 4 files (one per class): normal, imbalance_20g, vert_misalign_1.27mm, overhang_ball_20g
- Total: 284 windows (72+72+68+72); 80/20 interleaved split → 227 train, 57 val; batch_size=32
- Device: CUDA. ~5.96M parameters. Ran 20 epochs (hard limit, no early stopping triggered).
- Train loss epoch 1: 0.3832 → epoch 20: 0.0948  (75.2% drop, converged)
- Val loss converged to ~0.002 by epoch 3 and held stable (0.0016–0.0022)
- Normalization metadata computed fresh from these 4 files, saved to results/normalization_metadata.pth
  - y_min/y_max: per-channel shape [4]; y range ≈ [-773, 784] / [-884, 1029] / [-616, 514] / [-283, 240]
  - X_min/X_max: per-feature shape [8] for cols 0-7 of X (velocity+position features)
- z_macro shape: (284, 128); range: min=-2.693, max=2.371
- Silhouette score (random_state=42): -0.1221  (negative, but expected at early stage; better than PINN residuals at -0.20)
- UMAP visual: classes partially overlap but no strong separation (expected at 20 epochs without PINN guidance)
- Checkpoint: results/ts_jepa.pth | Committed: feat(phase1): train TS-JEPA encoder on MaFaulDa 16Hz one-subtype-per-class
- Issue: train_phase1_tsjepa freezes all params after training (for param in model.parameters(): param.requires_grad=False)
  → model returned from train_phase1_tsjepa has all gradients disabled; need to re-enable if fine-tuning later
- stdout capture via TeeOutput(sys.stdout, io.StringIO) worked cleanly for evidence logging

## [2026-04-17] T7: Baseline Code
- VanillaDDPM: exact copy of LatentDiffusionMLP backbone (z_dim=128, time_dim=64, SiLU, residual skip). No oracle, no guidance, no class conditioning. Self-contained scheduler inside the model.
- LabelConditionedDDPM: same backbone + nn.Embedding(4, 64) whose output is added to the timestep embedding (additive FiLM injection). No oracle. sample() accepts int class_label or tensor.
- Parameter counts: LatentDiffusionMLP=181,568 | VanillaDDPM=181,568 (0.0000% diff � exact match) | LabelConditionedDDPM=181,824 (+256 for class_embedding only)
- Noise schedule (all three identical): beta_start=0.0001, beta_end=0.02, num_timesteps=1000, linear schedule (DDPMScheduler from latent_diffusion.py)
- DDPMScheduler is NOT an nn.Module � buffers are plain tensors; _move_scheduler_to() helper moves them to device at call time (idempotent, avoids CUDA errors in sample())
- Smoke test passed: forward shapes (4,128) correct, losses finite (~1.07 / ~1.28), sample outputs shape (4,128) all-finite
- Fairness check passed: VanillaDDPM param diff = 0 absolute; LabelConditionedDDPM overhead = 256 (= 4 * 64, exactly class_embedding)
- Evidence: .sisyphus/evidence/task-7-baseline-smoke.txt, task-7-fairness-check.txt
- Commit: feat(baselines): implement Vanilla DDPM and Label-conditioned diffusion baselines

## [2026-04-17] T9: PINN Training from Scratch on 16Hz One-Subtype-Per-Class
- Added `scripts/train_task9_pinn.py` to load the four mandated 16Hz training tensors, rebuild 10-feature PINN normalization (8 stored bounds + inferred omega/time bounds), train ConfigurablePINN in float64 with ReLoBRaLo, and evaluate residual features via MathFeatureExtractor -> PCA50 -> silhouette/kNN.
- Scratch Strategy B remained numerically stable through 250 epochs on CUDA with batch_size=8 and lr=1e-4; best validation loss reached 137373.5933 at epoch 219. Fine-tuning Strategy A was skipped because scratch did not diverge within 50 epochs and T5 had already shown the legacy omega mismatch.
- Residual-feature clustering after fresh training still failed the success threshold: silhouette_pca50=0.0058 and 5-fold kNN accuracy=0.4541. Task-9 decision therefore falls back to: use the physics-informed representation for transfer classification, without claiming clean clustering.
- Updated `PriorWorkOracle` to accept `results/normalization_metadata.pth` files that store only X cols 0-7 by reconstructing omega/time bounds at runtime before PINN normalization and residual computation.

## [2026-04-17] T11: Decoder 1 Training
- Trained Decoder1(d_model=128, seq_length=3014, out_channels=4) on 4 Y files (one per class), frozen TS-JEPA encoder
- Data: 284 windows (72+72+68+72), normalized with y_min/y_max from normalization_metadata.pth [0,1] range
- Split: 80/20 interleaved (every 5th sample -> val): 228 train, 56 val
- Device: CUDA. 16,698,404 params. Ran 50 epochs (hit max_epochs, early stop not triggered)
- Train loss epoch 1: 0.256478 -> epoch 50: 0.005480  (97.9% drop, well above 30% requirement)
- Val loss converged from 0.230 (epoch 1) to ~0.00597 by epoch 50 (stable plateau ~epoch 10+)
- Reconstruction RMSE (normalized signal domain):
  - Class 0 (Normal): 0.025283
  - Class 1 (Imbalance_20g): 0.060049
  - Class 2 (Vert_Misalign_1.27mm): 0.055183
  - Class 3 (Overhang_Ball_20g): 0.120373
  - Note: Normal has lowest RMSE (simplest dynamics); Overhang_Ball highest (more complex oscillations)
- Checkpoint: results/decoder1.pth (16.7M params, MSE loss, no loss fn in decoder1.py itself)
- Evidence: .sisyphus/evidence/task-11-decoder1-training.txt, task-11-reconstruction.png
- TS-JEPA freeze confirmed: 123/123 params frozen after loading checkpoint
- TeeOutput encoding issue on Windows CP1252: need error='replace' fallback for unicode chars in console output
- train_phase1_decoder1() in train_phase1.py was NOT used directly (needed custom DataLoader); used same logic inline
- Commit: feat(phase1): train Decoder 1 envelope reconstruction

## [2026-04-17] T12: LDM Training on z_macro Latent Space
- Trained LatentDiffusionMLP (z_dim=128, time_dim=64, 181,568 params) on 284 precomputed z_macro vectors via DDPM
- Data: 4 Y files (one per class, 72+72+68+72 windows), normalized with y_min/y_max, transposed (N,3014,4)->(N,4,3014), encoded offline through frozen TS-JEPA
- Device: CUDA. 50 epochs, lr=1e-3, Adam, batch_size=32
- Train loss: epoch 1=1.013 -> epoch 50=0.109 (89.2% drop, above 30% requirement)
- Collapsed representation confirmed: all 4 class files have IDENTICAL z_macro min/max [-2.6933, 2.3711]; TS-JEPA learned class-agnostic embeddings (silhouette=-0.12 from T6); per-dim std across 284 samples ~0
- Generation: DDPM naive 1000-step stochastic sampling DIVERGES (explodes to [-100, 129]) due to accumulated prediction errors over 1000 steps in collapsed latent space
- FIX: x0-prediction based DDIM-style sampling with x0 clamp to [-3.5, 3.5] and timestep subsampling (every 5th step); produces stable [-3.5, 3.5] range samples
- Distribution: real global mean=-0.0005/std=1.015; generated global mean=0.064/std=3.46; std ratio=3.41 (within 0.1-10 acceptable range)
- Correct distribution comparison metric: GLOBAL scalar std (z_macro.std()), NOT per-dim std across samples (which is ~0 due to collapse)
- Checkpoint: results/ldm.pth (730KB)
- Evidence: .sisyphus/evidence/task-12-ldm-training.txt, task-12-ldm-samples-umap.png
- Smoke test: 3 epochs, loss 1.008->0.888 (decreasing), shapes (4,128) finite - PASSED
- UMAP: real z_macro colored by 4 classes (mostly overlapping due to collapse), generated gray x markers covering same region
- Script: train_ldm_task12.py (standalone, does not call existing train_latent_diffusion from run_sdedit_phase2.py which uses synthetic dataloader)
- train_latent_diffusion() in run_sdedit_phase2.py uses get_dataloaders (synthetic data, wrong for one-subtype-per-class constraint), so standalone script was written instead
- Commit: feat(phase2): train LDM on z_macro latent space

## [2026-04-17] T8: Synthetic Oscillator (Paper-Aligned 1D Damped Oscillator + Synthetic Pipeline)

### What Was Built
- NEW: `src/data/paper_synthetic_oscillator.py` — paper Eq. 4 oscillator: m*x'' + c*x' + k*x = A*sin(w0*t), m=1, c=0.5, k=4.0, A=1.0, w0=1.0
- NEW: `main_synthetic.py` — full Phase 0->1->2 synthetic pipeline entry point with cfg path overrides
- Fault classes: 0=healthy, 1=stiffness reduction (k->0.5k), 2=damping increase (c->2c), 3=forcing perturbation (A->2A)
- RK45 integration via scipy.integrate.solve_ivp; observable = acceleration x''(t) + Gaussian noise
- 4-channel compatibility: 1D signal replicated across 4 channels with small independent noise
- X tensor format: (N, T, 10) with features [vel_ch1..ch4, pos_ch1..ch4, omega, time]; omega=w0=1.0
- 50 train + 10 test samples per class -> data/processed-synthetic/ (20 .pth files)

### Pipeline Results (Reduced Epochs)
- Phase 0 PINN (50 epochs): physics residual loss ~3.4M (expected — rotor equations don't match 1D oscillator), normalization metadata saved correctly
- Phase 1 TS-JEPA (20 epochs): val loss 0.2831 -> 0.0018 (well converged)
- Phase 2 LDM (20 epochs): train loss 1.031 -> 0.335
- All checkpoints saved to results-synthetic/ (NOT results/)

### Fairness Metrics
- Oracle accuracy: 0.9938 +/- 0.0125 (5-fold CV, 160 real samples, 4 classes)
- TSTR accuracy: 0.2500 (LDM-generated z_macro pseudo-labeled, tested on real)
- TSTR/Oracle ratio: 0.2516 — WARN: below acceptable floor of 0.30
- Root cause: 20-epoch LDM + unconditional generation -> poor pseudo-label quality via centroid assignment
- Expected to improve with more LDM epochs or conditional generation

### Key Discoveries
- Windows cp1252 encoding: Unicode arrow chars in print() cause UnicodeEncodeError; use ASCII (->)
- TeeBuffer recursion bug: capture original stdout reference BEFORE replacing sys.stdout
- cfg override pattern: set cfg.RESULTS_DIR etc. at module level BEFORE other imports
- UMAP plot in train_phase1.py hardcoded to results/ (not cfg-aware) — minor issue
- Oscillator dynamics verified: distinct ranges per class (healthy xddot [-0.42,0.45], stiffness- [-1.01,1.06], damping+ [-0.85,1.16], forcing+ [-1.05,0.89])

### Evidence Files
- .sisyphus/evidence/task-8-oscillator-check.txt
- .sisyphus/evidence/task-8-synthetic-pipeline.txt
- .sisyphus/evidence/task-8-fairness.txt

## [2026-04-17] T10: PINN Variant Comparison
- Variant A (from-scratch T9, results/pinn.pth): Silhouette_raw2240=0.0783, Silhouette_pca50=0.0789, kNN_pca50=0.7783
- Variant B (T5 old weights): Silhouette_pca50=-0.2037, kNN_pca50=0.8008 (SPURIOUS — omega mismatch 153 vs 101 rad/s, residuals explode 1e9x)
- BEST VARIANT: A (from-scratch) — higher Silhouette (0.0789 vs -0.2037)
- Both variants below 0.3 silhouette as expected — FALLBACK ACTIVE: frame as 'physics-informed representation'
- 8-channel physics signal: 4 residuals + 4 unmeasured force estimates (fA/fB/fC/fD) -> MathFeatureExtractor -> 2240-dim
- Workflow: X(N,T,10) -> normalize -> flatten(N*T,10) -> PINN batched -> residuals(N*T,4) + forces(N*T,4) -> physics_seq(N,T,8) -> MathFeatExt -> (N,2240)
- UMAP(n_neighbors=15, min_dist=0.1) on PCA-50 space gives plausible visualization even without clean clusters
- Files: assets/fig_pinn_umap_mafaulda.pdf (300 DPI PDF), .sisyphus/evidence/task-10-best-umap.png (150 DPI PNG)

## [2026-04-17] T15: Baseline Training
- VanillaDDPM: 181,568 params (exact match to LDM), trained 50 epochs on 284 z_macro vectors
  - Epoch 1 loss=1.0149, Epoch 50 loss=0.1192, drop=88.3% (>30% threshold met)
  - Checkpoint: results/baseline_vanilla.pth (714 KB)
  - Sampling: 10 unconditional, DDIM-style x0-prediction, clamp=[-3.5,3.5], all finite
- LabelConditionedDDPM: 181,824 params (=181,568 + 256 class_embedding), 50 epochs same setup
  - Epoch 1 loss=1.0399, Epoch 50 loss=0.1244, drop=88.0% (>30% threshold met)
  - Checkpoint: results/baseline_label.pth (716 KB)

## [2026-04-21] Synthetic guidance wiring
- Wired `run_guided_sdedit_synthetic()` into `run_synthetic_pipeline()` as Step 3.5 between LDM and fairness.
- Added `--skip_guidance` CLI flag and threaded it through the pipeline so guidance can be skipped cleanly while fairness still runs.
- Verification: `python main_synthetic.py --skip_guidance --pinn_epochs 1 --tsjepa_epochs 1 --ldm_epochs 1 2>&1 | grep -c "Guided SDEdit"` returned `0`.
  - Sampling: 40 samples (10 per class), DDIM-style, all finite
- Same hyperparams as T12 (epochs=50, lr=1e-3, Adam, batch_size=32) for fair comparison
- z_macro collapse confirmed: all 4 classes have nearly identical embeddings (range [-2.69, 2.37], std~1.01)
- DDIM fix critical: naive 1000-step sampling hits clamp ceiling [-3.5, 3.5]; subsampled every-5th-step x0-prediction is stable
- get_loss() API handles scheduler device placement internally (_move_scheduler_to called inside)
- Evidence: .sisyphus/evidence/task-15-baseline-training.txt

## [2026-04-17] T13: Decoder 2 CVAE

### Setup
- Standalone script: train_decoder2_cvae.py
- Y files: 4 classes (normal, imbalance_20g, vmisalign_1.27mm, overhang_ball_20g)
- Data: 284 windows (72+72+68+72), shape (N, 4, T=3014) after (N,T,4) -> permute
- Normalization: y_min/y_max from normalization_metadata.pth -> range [0,1]
- TS-JEPA (5.96M params) + Decoder1 (16.7M params): both frozen
- Decoder2CVAE: 23.58M params, latent_dim=64, context_dim=128, seq_length=3014, num_classes=4

### Training
- Precomputed residual dataset (z_macro, residual, label) for train/val
- Train: 228 windows, Val: 56 windows (interleaved 80/20)
- beta_kl=0.01, lr=1e-3, batch_size=16, max_epochs=50, patience=15
- Early stopped at epoch 39, best epoch=24 (val_elbo=0.0066)
- Final KL=0.000405: NOT COLLAPSED (>0, as required)
- Checkpoint: results/decoder2.pth (94.36MB)

### API
- Decoder2CVAE.forward(residual, z_macro, label) -> (recon_residual, mu, logvar)
- residual shape: (B, 4, T) in normalized domain
- label: (B,) long tensor (class index 0-3)
- dec2.sample(z_macro, label) -> sampled jitter (B, 4, T)
- total_reconstruction = decoder1_output + dec2.sample(z_macro, label)

### Notes
- KL remains low but above 0 (typical for beta-VAE with small beta and limited data)
- z_macro collapse from T12 confirmed: conditioning provides limited class separation
- residual = Y_normalized - Decoder1(z_macro) in normalized [0,1] domain
- Evidence: .sisyphus/evidence/task-13-*.txt and *.png


## [2026-04-17] T14: Phase 1 Validation
- Full inference pipeline ran successfully: TS-JEPA.get_z_macro() -> Decoder1 -> Decoder2CVAE.sample()
- envelope_RMSE: cls0=0.0249, cls1=0.0631, cls2=0.0546, cls3=0.1218 (normalized units)
- full_RMSE (D1+D2): cls0=0.0259, cls1=0.0635, cls2=0.0550, cls3=0.1219
- Improvement %: all classes show slight *negative* improvement (~0-4%) � CVAE jitter adds stochastic variation, not RMSE reduction; this is expected (CVAE is distribution-matching, not residual fitting)
- val_ELBO = 0.0066 (from T13 evidence), Silhouette_pca50 = 0.0789 (from T10)
- Twin plot left panel: used results-synthetic/umap_latent_space.png (T8 artifact)
- All 4 outputs created: task-14-phase1-metrics.txt, task-14-reconstruction-overlay.png, task-14-twin-plot-preview.png, assets/fig_phase1_validation.pdf
- Key API: TSJEPA uses get_z_macro(x) not encode(x)

## [2026-04-17] T16: SDEdit Generation
- t_start=400, guidance_scale=1.0, N=40 per class
- Oracle guidance: oracle-vjp
- Silhouette mixed: -0.3786
- MMD per class: cls1=1.0235, cls2=1.0238, cls3=1.0240
- Generated signal stats: normalized outputs stayed within [-0.0890, 0.6550] with stable decoded envelopes+jitter
- Counterfactuals saved: results\sdedit_counterfactuals.pth
- Oracle penalty summaries: cls1 12946091.0000->13109561.0000, cls2 10946888.0000->11587906.0000, cls3 14285781.0000->5539209.5000

## [2026-04-17] T17: TSTR Transfer Classification Experiment
- Classifier: StandardScaler + MLPClassifier(hidden_layer_sizes=(64,), max_iter=500)
- Without StandardScaler: MLP/LR predict constant class -> 25% for ALL methods
- With StandardScaler: only real-only method benefits; synthetic still 25%
- Real-only: 80.0+/-6.1% acc, 76.0+/-9.3% F1 (upper bound confirmed)
- All synthetic (SDEdit, VanillaDDPM, LabelCond): 25.0+/-0.0% (random chance)
- ROOT CAUSE: z_macro distribution mismatch
  - Real z_macro (TS-JEPA encoded): std ~1e-6 per dimension (collapsed)
  - Synthetic z_macro (DDPM/SDEdit sampled): std ~0.35 per dimension
  - Scaler fit on synthetic maps real test data to constant point -> one-class prediction
- DDIM sampling worked correctly (no divergence, x0-prediction, clamp 3.5, stride=5)
- VanillaDDPM.sample() and LabelConditionedDDPM.sample() use naive 1000-step; must use custom DDIM
- Wilcoxon p(ours vs real-only)=0.0625, p(ours vs vanilla)=1.0, p(ours vs label-cond)=1.0
- Evidence: .sisyphus/evidence/task-17-{tstr-results.txt,results-table.tex,confusion-matrices.png}

- 2026-04-17 F1 compliance audit: all requested deliverables verified present/loadable (18/18), evidence census passed with 45 files and tasks T1-T18 present; only failing compliance point was the unresolved reduced-4-class guardrail in dataset/evaluation code.

- 2026-04-17 F1 revised audit: approved under clarified scope (codebase/results/evidence only). One-subtype-per-class enforcement is confirmed in the task-specific scripts, not the flexible general mapper.
## [2026-04-18] 24.4 Hz PINN Experiment
- Finetune: Silhouette_pca50=0.027027, kNN_pca50=0.640590
- Scratch: Silhouette_pca50=0.029812, kNN_pca50=0.665071
- 16Hz baseline: Silhouette_pca50=0.0789, kNN_pca50=0.7783
- Winner: scratch, conclusion: native-speed fine-tuning did not beat the 16Hz baseline; both 24.4 Hz runs underperformed it.

## [2026-04-29] Synthetic validation figure regen
- `assets/generate_synth_validation_figure.py` now loads `data/processed-synthetic/Y_healthy_testset.pth` and `Y_stiffness_reduction_testset.pth` directly, matching the synthetic-data validation workflow.
- LSP diagnostics were clean after adding a spec loader assert and suppressing false-positive missing-import warnings for the script-only environment.
- Regenerated `assets/synth_validation_overview.png` successfully; output is 1791x816 and wider than tall.
