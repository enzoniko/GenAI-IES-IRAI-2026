# Geometric Compatibility in Density-Guided Fault Synthesis — `mafaulda_synthetic`

Codebase for the journal extension of the withdrawn IRAI paper. It tests one
pre-registered hypothesis (**H**): *in density-guided generative fault
synthesis, fidelity is governed by the geometric compatibility between a
fault's cluster shape in the physics-informed embedding space and the
distributional form assumed by the guidance density estimator — to first
order invariantly across generative backbones.*

The synthetic-first strategy: every experiment of the research proposal runs
on `mafaulda_synthetic`, a simulated rotor benchmark whose governing ODE is
**the same equations the PINN's physics loss uses** (single source of truth:
[src/physics/rotor_model.py](src/physics/rotor_model.py)), and whose fault
classes form a **geometry dial** — cluster non-Gaussianity 𝒢_y is a designed,
controlled independent variable instead of an observational one. Real
MaFaulDa becomes a config flag later ([docs/REAL_DATA_SWITCH.md](docs/REAL_DATA_SWITCH.md)).

## Quickstart

```bash
pip install -r requirements.txt
python -m pytest                      # 32 tests, ~10 min (generates tiny datasets)
python run.py smoke-all               # entire pipeline + all blocks, tiny sizes
```

Everything goes through `run.py` with dot-overrides:

```bash
python run.py <command> [--config file.yaml] [--smoke] [key.path=value ...]
```

| Command | What it does |
|---|---|
| `generate-data` | build the synthetic tensors (deterministic, cached by config hash) |
| `train-pinn` → `train-oracle` → `verify-geometry` | physics stage + the two scientific gates |
| `train-jepa` / `train-dec1` / `train-dec2` / `train-ldm` | representation + generation stack |
| `block-a` … `block-f` | proposal experiment blocks (see below) |
| `full-run` | all of the above in order, stage progress + ETA in `results/<exp>/PROGRESS.md`, resumes past finished stages |
| `report` | aggregate all run manifests into `report/REPORT.md` (bypass-tagged rows excluded from headline tables) |

Config schema: [src/configs/schema.py](src/configs/schema.py). Every run
writes `run.json` (resolved config + git SHA + seed + outputs) into its own
directory under `results/<experiment>/` — paper numbers come from manifests,
never by hand.

## Layout

```
src/physics/      rotor ODE + residuals (simulator AND PINN share these) + fault samplers
src/data/         DatasetBundle contract, mafaulda_synthetic generator, real/stub adapters
src/models/       TS-JEPA v2 (anti-collapse), decoders (+beta-TCVAE), LDM+DDIM, RotorPINN, MINE
src/oracle/       PhysicsEmbedding (PINN->Phi->PCA, differentiable) + guidance densities
                  (Ledoit-Wolf Gaussian / BIC-GMM / RealNVP flow)
src/backbones/    ours (corrected guided SDEdit) / signal-space CVAE / FaultDiffusion-style
src/training/     one trainer per component; artifacts.json chains stages
src/evaluation/   TSTR (3 probes), per-class MMD, G_y (Henze-Zirkler/Mardia/Epps-Pulley),
                  C3 correlation with pre-registered |rho| >= 0.4 threshold
src/experiments/  engine + blocks A-F drivers
docs/             extension plan, real-data switch, extension report (.tex), withdrawn paper
legacy/           quarantined pre-extension code (reference only, never imported)
```

## Experiment blocks (research proposal mapping)

- **A** — disentanglement controls: MINE I(z; jitter) JEPA-vs-AE (A1), SIGReg
  encoder arm (A2), envelope/jitter spectral leakage (A3)
- **B** — core: guidance estimator {Gaussian, GMM, Flow} × backbone × class ×
  speed × seed factorial (B1/B2) + few-shot sweep n ∈ {5..100} with held-out
  log-likelihood (B3)
- **C** — geometry: cluster indices + 𝒢_y per (class, speed) (C1),
  Δ-silhouette transfer (C2), **the primary geometry↔fidelity correlation
  (C3)** with backbone-invariance grading
- **D** — controls: guidance-interval and t0 ablations (D1/D2), gradient-norm
  diagnostic (D3), multi-speed (D4)
- **E** — extensions: β-TCVAE (E1), latent modality / class-cond LDM (E2),
  open-set interpolation with distance-to-true-held-out-cluster (E3),
  cross-machine variant-B zero-shot (E4)
- **F** — benchmarking (F1), probe sensitivity (F2), causal guidance ablation
  incl. wrong-target control (F3)

## Validation status

**Passing now** (reduced scale `experiment=gate_v2`; see
[docs/extension_report.tex](docs/paper/extension_report.tex) for full detail):

- Physics: simulator satisfies its own healthy equations to <1e-6 relative;
  closed-form frequency response matched <1%; synchronous kinematics recovery
  residual <0.05% of forcing at all speeds.
- **PINN gate PASSED**: physics-embedding per-speed silhouette **0.436**
  (withdrawn paper on real data: 0.006–0.079), kNN **97.4%**.
- **Geometry dial PASSED**: 𝒢_y spans healthy 1.28 → bimodal 17.5 (8× spread
  across fault classes, designed ordering checks all pass).
- TS-JEPA v2: no latent collapse (z_std 0.71, effective rank ~19; the v1
  collapse — per-dim std ≈ 1e-6 — is what killed the original TSTR).
- Guided SDEdit: small-t0 round-trip at reconstruction floor; guidance
  penalty descends monotonically; steering toward imbalance grows the 1X
  line 77k → 271k (causal direction confirmed).
- 32 unit tests; `smoke-all` covers every stage and block.

**Waiting on the full run**: the C3 correlations + invariance grade (the
hypothesis verdict), B3 few-shot curves, D1/D2 frontiers, E outcomes, and the
F1 benchmark table. Launch:

```powershell
python run.py full-run experiment=full_v1 data.version=full1 `
  sdedit.guidance_interval=5 eval.n_gen_per_class=100 "eval.seeds=[0,1,2]" `
  pinn.epochs=150 > logs_full_v1.log 2>&1
```

Follow progress: `Get-Content logs_full_v1.log -Tail 20 -Wait` or open
`results/full_v1/PROGRESS.md`. The run is resumable — rerunning the same
command skips completed stages.

## Honesty guardrails

`oracle.mode=raw` (the bypass that silently produced the withdrawn paper's
headline numbers) still exists for diagnostics, but every artifact it touches
is tagged `bypass=true` and `run.py report` refuses to mix those rows into
headline tables.
