# Decisions — IRAI Pipeline Results

## [2026-04-17] Session Start

### PINN Strategy
- Try 3 strategies: (A) fine-tune old weights, (B) train from scratch, (C) old weights as-is
- Success criteria: Silhouette > 0.3 on residual UMAP, kNN accuracy > 70%
- 2-day hard time-box. Fallback if criteria not met: reframe claim to "physics-informed representation"

### Baselines
- Exactly 2: Vanilla DDPM + Label-conditioned DDPM
- Same architecture (LatentDiffusionMLP), same epochs, same latent space
- Only difference: guidance mechanism

### Synthetic System
- Must implement paper's Eq. 4 (1D oscillator) in src/data/paper_synthetic_oscillator.py
- Fairness metrics (oracle/TSTR ratio) are internal validation only, not reported in paper
- Synthetic results ARE reported in paper (Section IV-A)

### Figure Budget
- Max 4 figures, max 2 tables (IEEE 6-page double-column)
- Twin plots: synthetic (left) + MaFaulDa (right)

### Seeds and Reproducibility
- UMAP: random_state=42
- Transfer classification: 5 seeds (0,1,2,3,4), report mean±std

### [2026-04-17] Deliverables Specification (T4)
- **Figure A (PINN Residual UMAP)**: `assets/fig_component_validation.pdf`. Demonstrate clustering of physics features.
- **Figure B (SDEdit Trajectories)**: `assets/fig_sdedit_trajectory.pdf`. Demonstrate physics-guided denoising path.
- **Figure C (Signal Comparison)**: `assets/fig_signal_comparison.pdf`. Demonstrate physical plausibility via real vs. synth signal overlays.
- **Table 1 (TSTR Accuracy)**: Train synthetic, test real. 5-seed mean ± std for all baselines vs. ours.
- **Table 2 (Component Metrics)**: Silhouette Score, RMSE, ELBO for all Phase 1/2 components.
- **Bib Fix**: `references.bib` must be created manually using the consolidated bib block at the bottom of the LaTeX file.
- **Assets Dir**: Directory `assets/` must be initialized for all figure outputs.
## [2026-04-17] T1 data repair
- Updated the hardcoded metadata label_strategy in clean_mafaulda_processor.py to mafaulda_reduced_4 so future processor metadata aligns with the reduced 4-class dataset.

- 2026-04-17 F1 audit scope decision: exclude LaTeX completeness/compilation and bibliography from compliance counts; assess deliverables using checkpoints, figures, evidence files, and actual artifact-generation scripts.
