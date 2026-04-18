# Issues — IRAI Pipeline Results

## [2026-04-17] Session Start

### Known Pre-existing Issues (from code exploration)
- T1: configs.py has wrong TARGET_HZ (17.0→16.0), NUM_CLASSES (42→4), rotation_hz (30→16)
- T1: mafaulda_dataset.py has 42-class mapping that must be collapsed to 4 classes
- T2/T5: architecture.py has broken import (direct_analysis.data_utils doesn't exist)
- T8: synthetic_dataset.py is wrong system — need new paper_synthetic_oscillator.py
- Missing files: references.bib, method_diagram.png, assets/ directory (created in T18)
- Possible: torchaudio.lfilter platform issue on Windows

### Status
- None of the above have been fixed yet — T1, T3 will address them
## [2026-04-17] T1 data repair
- Verification found 24 vertical files present as required, but overhang still totals 28 files rather than the plan's older expectation of 32 because only overhang_ball and overhang_cage tensors exist in the current processed set.

## [2026-04-17] T2 Issues Found
- Normalization bounds diverge sharply from processed 16Hz data: 7/10 X features diverge, omega old bound is effectively fixed at 153.4 rad/s while new data spans about 96.5-104.2 rad/s, and all 4 Y channels exceed old ranges by large margins.
- Weight loading is not strict-compatible: legacy checkpoint keys target sequential indices .3/.6 while current PINN state dict expects .2/.4 because the old exported architecture uses ELU activations inserted between linear layers.
- model_config.json contradicts inherited assumptions: exported metadata says activation=elu, not tanh.
- pytest failures: oracle math extractor test now requires explicit omega; oracle target buffer test still expects class index 5 though NUM_CLASSES=4; processed dataset test still looks for legacy path data/processed-mafaulda/v1/ instead of 16hz.
- Dtype seam remains significant: float32 data/latent pipeline interfaces with float64 PINN/oracle physics, with repeated conversions inside oracle forward.
## [2026-04-17] T3: Code Audit + Fix
- Verification exposed a blocking gradient issue in tests/test_math_extractor that was not limited to the missing omega argument; float32 feature extraction overflowed on large physics features and produced NaN gradients.
- PriorWorkOracle still reports PINN checkpoint shape mismatch and falls back to initialized weights during smoke tests because legacy exported architecture indexing differs; this remains expected until dedicated checkpoint remapping work.

## [2026-04-17] T5 Issues Found
- Exported metadata does not preserve the trained dropout probability, but the checkpoint key layout proves Dropout modules existed. The compat shim therefore uses a positive placeholder dropout rate only to recreate legacy sequential indices; because evaluation runs in model.eval(), dropout remains inactive.

- 2026-04-17 F1 compliance audit: REJECT. Guardrail breach remains in reduced 4-class scope: src/data/mafaulda_dataset.py:63-72 includes extra subtypes, and src/pipelines/train_phase1.py:189-190 still references legacy 42-class labels.
