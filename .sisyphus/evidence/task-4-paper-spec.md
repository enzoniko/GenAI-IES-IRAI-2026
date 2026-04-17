# Task 4: Paper Deliverables Specification

This document specifies the figures, tables, and missing assets required for the IEEE conference paper "Towards Physics-Guided Counterfactual Fault Synthesis...".

## 1. Figure Specification (Budget: 3-4 Figures)

The paper follows a "twin plot" motif: Synthetic results on the left, MaFaulDa results on the right.

### Figure A: Component Validation (PINN Residual Clustering)
- **Reference**: `\ref{fig:component_validation}` (to be added)
- **Path**: `assets/fig_component_validation.pdf`
- **Description**: UMAP projection of the 2240-dim physics-informed feature space derived from PINN residuals.
- **Layout**: 2 subplots (Left: Synthetic Oscillator, Right: MaFaulDa 16Hz).
- **Goal**: Demonstrate that physics-based residuals naturally cluster by fault class (Healthy, Imbalance, Misalignment, etc.) without the PINN having seen fault data during training.
- **Data Source**: T10 (PINN variant selection/comparison).
- **Draft Caption**: Figure A. Physics-informed embedding space. UMAP projections of PINN-derived residual features for the synthetic system (left) and MaFaulDa (right) show that different fault types form distinct, well-separated clusters based on their specific violations of healthy-system dynamics.

### Figure B: Generation Quality (SDEdit Trajectories)
- **Reference**: `\ref{fig:sdedit_trajectory}` (replaces `\ref{fig:sdedit_synth}`)
- **Path**: `assets/fig_sdedit_trajectory.pdf`
- **Description**: UMAP manifold showing the iterative denoising path from a healthy starting point to a target fault cluster.
- **Layout**: 2 subplots (Left: Synthetic, Right: MaFaulDa).
- **Goal**: Visualize the physics-guided "bending" of the latent diffusion path towards a target fault manifold.
- **Data Source**: T16 (SDEdit Generation).
- **Draft Caption**: Figure B. Physics-guided SDEdit trajectories. The iterative denoising process successfully steers healthy latent states toward the target fault clusters (stars) in the physics-informed embedding space for both experimental systems.

### Figure C: Signal Comparison (Counterfactual Traces)
- **Reference**: `\ref{fig:signal_comparison}` (replaces `\ref{fig:synth_evaluation_imbalance_pt1/2}`)
- **Path**: `assets/fig_signal_comparison.pdf`
- **Description**: Time-domain and frequency-domain (FFT) comparison of real vs. synthesized counterfactual traces.
- **Layout**: Composite grid. Top row: Synthetic (Real vs Synth). Bottom row: MaFaulDa (Real vs Synth).
- **Goal**: Visual verification of physical plausibility and frequency alignment.
- **Data Source**: T16 (SDEdit Generation).
- **Draft Caption**: Figure C. Comparison of real and synthesized counterfactual traces. The generated signals (red) accurately capture the characteristic time-domain envelopes and frequency signatures of real fault signals (blue) while maintaining physical consistency.

### Figure D (Optional): Baseline Comparison
- **Reference**: `\ref{fig:baseline_comparison}`
- **Path**: `assets/fig_baseline_comparison.pdf`
- **Description**: UMAP showing where samples from Vanilla DDPM, Label-conditioned, and Physics-guided diffusion land relative to real fault clusters.
- **Draft Caption**: Figure D. Baseline comparison. Only the physics-guided approach consistently produces samples that reside within the physical manifolds defined by real-world fault observations.

---

## 2. Table Specification (Budget: 1-2 Tables)

### Table 1: Transfer Classification Results (Main)
- **Columns**: Method (Real-only, Vanilla DDPM, Label-conditioned, Ours/Physics-guided), Class-level Accuracy (Normal, Imbalance, Misalign, Bearing), Overall Mean ± Std.
- **Data Source**: T17 (Transfer Classification).
- **Paradigm**: TSTR (Train on Synthetic, Test on Real). 5 seeds required.
- **Goal**: Quantitative proof that synthetic data captures transferable, discriminative fault structure.

### Table 2: Component Performance Metrics
- **Columns**: Component (PINN, TS-JEPA, Decoder 1, CVAE, LDM), Metric (Silhouette Score, Reconstruction RMSE, ELBO, FID/MMD).
- **Data Source**: T14 (Phase 1 Validation).

---

## 3. LaTeX Status & Missing Assets

### Incomplete/Placeholder Sections
- **Section V (Results)**: Currently has placeholders and some preliminary synthetic plots that need replacement with twin plots.
- **Section VI (Discussion)**: Needs expansion based on quantitative TSTR results.
- **Section VII (Conclusion)**: Summary of findings.

### Missing/Needed Files
- `references.bib`: Does not exist. Must be created in T18 with consolidated citations from the LaTeX footer.
- `assets/`: Directory needs to be created.
- `method_diagram.png`: Referenced in Section III, currently missing.
- `assets/fig_component_validation.pdf`
- `assets/fig_sdedit_trajectory.pdf`
- `assets/fig_signal_comparison.pdf`

## 4. Experiment Mapping

| Figure/Table | Experiment Source | Key Metric |
| :--- | :--- | :--- |
| Fig A | T10 (PINN Compare) | Silhouette Score > 0.3 |
| Fig B | T16 (SDEdit Gen) | Trajectory Convergence |
| Fig C | T16 (SDEdit Gen) | Visual Plausibility / FFT Match |
| Tab 1 | T17 (Transfer Clf) | TSTR Accuracy > 50% |
| Tab 2 | T14 (Component Val)| RMSE / ELBO |
