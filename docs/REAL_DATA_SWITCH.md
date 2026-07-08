# Switching to real MaFaulDa (`dataset=mafaulda`)

The synthetic phase was built so this switch is config-only. What the
real-data phase must provide, and what changes automatically:

## The flag

```bash
python run.py train-all data.name=mafaulda data.root=data/processed-mafaulda/16hz
```

Every module downstream of `src/data/registry.py` consumes only the
`DatasetBundle` contract (`src/data/contract.py`). `tests/test_real_data_switch.py`
verifies the contract resolves end-to-end via the stub bundle.

## What the real-data phase must provide

1. **Processed tensors** in the legacy format (`Y_<category>_trainingset.pth`
   (N, T, 4) + `X_<category>_trainingset.pth` with omega at `[:, 0, 8]`),
   produced by `src/data/clean_mafaulda_processor.py` /
   `scripts/download_dataset.sh`. `src/data/mafaulda_real.py::RealBundle`
   loads them; extend `LABEL_MAPPING` there to change the class set.
2. **A PINN for the real rig.** `RotorPINN` residuals come from
   `src/physics/rotor_model.py`; the real rotor will NOT satisfy them exactly
   — this is the known hard problem (silhouette 0.006-0.079 in the withdrawn
   paper). The gate in `run.py verify-geometry` (silhouette >= 0.3) applies
   unchanged; do not proceed past a failed gate.
3. **Kinematics**: `derive_kinematics(..., omega=...)` uses synchronous
   offset removal and needs windows >= 1 revolution. Real accelerometers add
   DC drift — if residual floors are high, add a high-pass stage analogous to
   the legacy Strategy D (keep it IN the shared function so train/inference
   stay consistent).

## What degrades gracefully

- `meta["has_ground_truth"] = False` makes Block A1/A3 and the E3
  distance-to-true-cluster metric skip/reduce automatically (`jitter_phys is
  None`; Block A logs `skipped`).
- `clean == raw`: Decoder1 trains on raw (envelope supervision unavailable).
- Geometry (C1), guidance (B), correlation (C3), controls (D), benchmarking
  (F) run unchanged.

## Honesty guardrails (unchanged)

- `oracle.mode=raw` (the legacy bypass) still works for diagnostics but tags
  every run manifest `bypass=true`; `run.py report` excludes those rows from
  headline tables.
