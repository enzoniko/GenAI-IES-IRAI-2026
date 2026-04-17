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
