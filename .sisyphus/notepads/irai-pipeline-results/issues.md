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
