# Implementation Plan — Geometric-Compatibility Extension on `mafaulda_synthetic`

**Audience:** an autonomous AI coding agent implementing this end-to-end.
**Scope:** rebuild/fix the codebase and implement *every* experiment block (A–F) of the research
proposal ("Geometric Compatibility as the Determinant of Fidelity in Density-Guided Generative
Fault Synthesis for CPS") **using synthetic data only** (`mafaulda_synthetic`). All interfaces must
be built so that switching to real MaFaulDa later is a single config flag (`dataset=mafaulda`),
not a code change.
**Non-goals (deferred):** anything requiring real MaFaulDa signals — do not download, process, or
train on real data in this phase. Real-data hooks are specified, but only as interfaces.

---

## 0. Why this plan looks the way it does (read first)

### 0.1 What actually failed last time (verified against code + run logs)

The withdrawn paper's failure chain, reconstructed from `.sisyphus/notepads/irai-pipeline-results/*`
and the code itself:

1. **TS-JEPA latent collapse on real data.** `src/models/ts_jepa.py` has *no anti-collapse
   mechanism* (no variance/covariance regularization; z_macro is a mean-pool of the context
   encoder). On MaFaulDa the encoded latents collapsed to per-dimension std ≈ 1e-6 across samples
   (T12/T17 logs). Generated latents (std ≈ 0.35) then lived in a different distribution than real
   latents, so every TSTR probe collapsed to 25% (random chance, T17).
2. **The PINN never fit the real data.** Residual-feature clustering reached silhouette
   0.006–0.079 (T9/T10), far below the 0.3 success gate. Root cause: the 3-mass rotor equations +
   normalization were calibrated for a different operating point (ω ≈ 153 rad/s vs 101 rad/s in
   the 16 Hz subset), and no amount of retraining fixed the geometry.
3. **The headline results silently bypassed the physics.** `src/configs.py` ships
   `ORACLE_MODE = "raw"`: the "physics-informed embedding" used for the reported MaFaulDa numbers
   is `MathFeatureExtractor` applied to the **raw signal**, no PINN at all. This is one of the
   reporting inaccuracies that motivated withdrawal — the new codebase must make this impossible
   to do silently.
4. **The SDEdit implementation does not implement the paper.**
   - `src/pipelines/run_sdedit_phase2.py:109`: `t_start = int(num_inference_steps * strength)`.
     With the HPO-selected config (`num_inference_steps=10, strength=0.1062`) this gives
     `t_start = 1` — the "guided diffusion" performed exactly **one** reverse step. The reported
     HPO optimum is a degenerate configuration that effectively turned guidance off.
   - `run_sdedit_phase2.py:144`: the guidance penalty is a plain **MSE to the class-mean
     embedding**, not the Mahalanobis penalty with Ledoit–Wolf covariance the paper describes.
     `PriorWorkOracle.target_distributions` stores only a mean vector per class — no covariance
     anywhere in the repo.
   - No gradient clipping (τ), no guidance interval N, no per-step gradient-norm logging.
5. **Paper/code dimension mismatch.** Paper says F = 1456; code produces 2240 (PINN mode, 8
   channels) or 1120 (raw mode, 4 channels). Nothing in the repo produces 1456.
6. **The TSTR metric was circular and measured in the wrong space.** `compute_tstr`
   (`src/evaluation/metrics.py`) pseudo-labels generated samples with a NearestCentroid fitted on
   *real* labels, and T17 evaluated classifiers on z_macro embeddings rather than on signals.
7. **Structural rot.** Mutable module-level config (`src/configs.py`) that entry scripts
   monkey-patch *before importing anything else* (see `main_simple_synthetic.py:8–20`); ~10
   one-off root scripts each re-implementing loaders and training loops; hardcoded `results/`
   paths inside library code (`run_sdedit_phase2.py:190,262`); binary checkpoints and
   `__pycache__` committed to git; three different "synthetic datasets"
   (`synthetic_dataset.py`, `paper_synthetic_oscillator.py`, `SimpleSHOPINN` inline in
   `main_simple_synthetic.py`) none of which has a governing ODE the PINN's physics loss actually
   matches; float32/float64 seams crossed repeatedly inside the oracle.

### 0.2 The three design decisions that everything else follows from

**Decision 1 — The synthetic benchmark simulates the exact ODE system the PINN encodes.**
Previous synthetic attempts failed *by construction*: the 4-channel harmonic benchmark and the 1D
oscillator have no relation to the rotor equations in `src/models/pinn.py:397–454`, so the PINN's
physics residual was meaningless noise on them (T8 logs: "physics residual loss ~3.4M — expected,
rotor equations don't match 1D oscillator"). `mafaulda_synthetic` inverts this: we numerically
integrate a 2-plane, 2-mass coupled rotor ODE (masses M2/M3 with damping D2/D3, stiffnesses
K1/K2, rotor mass M1 with eccentricity E1 producing unbalance forcing M1·ω²·E1·[cos ωt, sin ωt])
whose **healthy** regime satisfies the PINN residual equations with fA..fD ≈ 0. The equations are
defined **once**, in a shared `physics.py` module consumed by *both* the simulator and the PINN
loss, so agreement is guaranteed by construction, not by hope. Consequences:
- The PINN trains to near-zero physics residual on healthy data — the failure mode that killed
  the real-data paper is eliminated in the controlled setting.
- The data format (4 acceleration channels x2¨, y2¨, x3¨, y3¨; ω side channel; X tensors with
  velocity/position features) is **identical** to the processed-MaFaulDa format, so the
  real-data switch later is genuinely a flag.
- Faults are physically meaningful parameter/forcing perturbations, so "deviation from healthy
  physics" — the quantity the whole method is built on — has exact ground truth.

**Decision 2 — Fault classes are a *geometry dial*.** The proposal's hypothesis H says: cluster
non-Gaussianity 𝒢_y in the physics-informed embedding space determines synthesis fidelity. On
real data 𝒢_y is observational — you get whatever geometry MaFaulDa gives you, confounded with
everything else. On synthetic data we can **manipulate the independent variable directly**: we
design fault classes whose embedding-cluster geometry spans unimodal-Gaussian → bimodal →
heavy-tailed → curved-continuum, by controlling how fault parameters are sampled per class. This
turns C3 from a correlation study into something much closer to a causal experiment, which is the
single strongest scientific reason to do synthetic-first.

**Decision 3 — Rewrite the pipeline layer, keep the validated leaf modules.** The leaf math is
mostly sound and audited (`MathFeatureExtractor`, `ConfigurablePINN` residuals, `relobralo_loss`,
CVAE/decoder architectures, DDPM scheduler math). What is rotten is the orchestration layer:
global config, entry-script sprawl, silent mode switches, no experiment tracking. We keep the
leaves (with targeted fixes), and rebuild everything above them under `src/` with a typed config
system, a dataset registry, and one entry point. Old root scripts move to `legacy/` untouched
(reference only) and are deleted from the import path.

### 0.3 Other decisions, with justification

| Decision | Choice | Why |
|---|---|---|
| Window geometry | keep `fs=50 kHz`, `T=3014`, 4 channels | Byte-compatible with processed MaFaulDa tensors → real-data switch is config-only. Simulation cost is negligible (linear ODE, closed-form or RK45 over 0.06 s). |
| Operating speeds | ω ∈ {12, 16, 20} Hz (config list) | D4 needs a speed axis; 3 speeds × classes gives enough (class × speed) cells for C3 correlations without exploding compute. 16 Hz stays the "home" speed to mirror the prior paper. |
| Embedding for density fitting | PCA-whiten the Φ-features to d=32 (fit on train split, healthy+fault), densities fitted in that space | GMM/normalizing-flow density estimation in 2240-D with ≤100 samples/class is statistically meaningless. PCA is linear ⇒ the guidance penalty stays differentiable end-to-end. d=32 is a config knob. |
| 𝒢_y (non-Gaussianity) statistics | Henze–Zirkler + Mardia skewness/kurtosis + Epps–Pulley energy statistic, all computed in the PCA space; report all three | Proposal's threat #4 requires more than one defensible measure. HZ and Mardia are standard multivariate normality tests; Epps–Pulley/energy is the SIGReg statistic, tying A2 and C1 together. |
| Guidance penalty family | `GuidanceDensity` interface: Gaussian (Mahalanobis, Ledoit–Wolf), GMM (k selected by BIC ≤ 4), MAF normalizing flow | Exactly the B1/B2 ladder in the proposal. All three expose `fit(V_y)`, `log_prob(v)`, `held_out_ll()`, and a differentiable `penalty(v)`. |
| Backbones | (1) ours (JEPA+LDM+SDEdit), (2) conditional VAE in signal space, (3) FaultDiffusion-style few-shot conditional DDPM | Proposal Block F requires ≥2 non-ours backbones. FaultDiffusion has no public code — implement a faithful *-style* variant (frozen backbone trained on healthy + per-class difference adapter) and say so honestly in reporting. |
| TSTR protocol | Classifiers consume a **fixed, backbone-agnostic feature map of raw signal windows** (the `MathFeatureExtractor` on raw 4-ch windows). Generated labels come from the generation condition (never pseudo-labeled). Oracle = same probe trained on real train split. Probes: logistic / 1-NN / RBF-SVM (F2). 5 seeds, mean ± CI | Fixes failure #6. Evaluating in signal-feature space measures the *signals*, not the latent space that produced them, and makes cross-backbone comparison fair. |
| Anti-collapse | VICReg-style variance+covariance regularization on z_macro **always on**; SIGReg as a config flag (off by default) | Collapse is a proven failure (0.1). A2 requires a with/without-SIGReg arm, so SIGReg must be a flag, and the baseline arm still needs *some* collapse protection or C3 is confounded by a broken encoder. |
| Config system | frozen dataclasses + YAML, composed per experiment; every run writes `run.json` (resolved config + git SHA + seed + metric outputs) into its own run directory | Kills the monkey-patched global config. Run manifests are what Phase-5 aggregation and the paper tables read — no more hand-copied numbers (reporting-accuracy guardrail). |
| PINN bypass | `oracle.mode` stays available as `raw` for *diagnostics* but every artifact it touches is tagged `bypass=true` in the manifest, and the aggregator refuses to mix bypass and non-bypass rows in one table | Makes failure #3 structurally impossible to repeat. |

---

## 1. Target architecture

```
src/
├── configs/                    # typed config schema + YAML experiment files
│   ├── schema.py               # frozen dataclasses: DataCfg, PinnCfg, JepaCfg, ...
│   └── experiments/*.yaml      # one file per experiment block / ablation arm
├── physics/
│   ├── rotor_model.py          # THE single source of truth: symbolic healthy ODE,
│   │                           #   residual functions, parameter dataclass
│   └── faults.py               # fault parameterizations + per-class samplers (geometry dial)
├── data/
│   ├── contract.py             # Sample/Batch dataclasses, DatasetBundle protocol
│   ├── registry.py             # get_dataset(cfg) -> DatasetBundle ("mafaulda_synthetic" | "mafaulda")
│   ├── mafaulda_synthetic.py   # simulator → windows (+ envelope/jitter ground truth)
│   └── mafaulda_real.py        # thin port of existing loader; NOT exercised this phase
├── models/                     # kept leaves, fixed: pinn.py, feature_extractors.py,
│   │                           #   relobralo_loss.py, decoder1.py, decoder2_cvae.py (+β-TCVAE),
│   │                           #   ts_jepa.py (v2, anti-collapse), latent_diffusion.py (+DDIM, +class-cond)
│   └── mine.py                 # MINE mutual-information estimator (A1)
├── oracle/
│   ├── embedding.py            # PhysicsEmbedding: PINN → Φ → (optional) PCA projection
│   └── guidance.py             # GuidanceDensity: Gaussian | GMM | Flow (B1/B2)
├── backbones/
│   ├── base.py                 # SynthesisBackbone protocol: fit(bundle), synthesize(...)
│   ├── ours_sdedit.py          # JEPA + Dec1/Dec2 + LDM + guided SDEdit (v2, correct)
│   ├── cvae_backbone.py        # conditional VAE in signal space
│   └── faultdiff_backbone.py   # FaultDiffusion-style few-shot conditional DDPM
├── evaluation/
│   ├── fidelity.py             # TSTR (3 probes), per-class MMD, ΔSilhouette transfer score
│   ├── geometry.py             # Silhouette/CH/DB + 𝒢_y (HZ, Mardia, Epps–Pulley)
│   ├── disentanglement.py      # I(z;ε) via MINE, spectral coherence (A1/A3)
│   └── correlate.py            # C3: geometry↔fidelity correlation w/ bootstrap CIs
├── training/                   # one trainer per component; shared EarlyStopping/loops
└── experiments/
    ├── runner.py               # run(experiment_yaml) → run_dir with run.json
    └── blocks/exp_{a..f}.py    # Block A–F drivers
run.py                          # single CLI entry point
tests/                          # unit + smoke tests (see §4, task 0.3)
legacy/                         # old root scripts, moved verbatim, never imported
```

**The dataset contract** (everything downstream depends only on this):

```python
@dataclass
class Sample:
    raw: Tensor          # (4, T) float32 — what sensors would measure
    clean: Tensor        # (4, T) envelope ground truth (== raw for real data)
    jitter: Tensor|None  # (4, T) raw - clean (None for real data)
    label: int           # 0 = healthy, 1..C = fault classes
    omega: float         # rad/s
    speed_hz: float
    fault_params: dict|None   # ground-truth fault parameters (None for real data)

class DatasetBundle(Protocol):
    train/val/test loaders (windowed, seeded splits)
    meta: channels, T, fs, class names, speeds, normalization bounds
    fewshot_fault_sets(n_per_class, seed) -> features-source subsets   # B3
```

`mafaulda_real.py` implements the same protocol with `jitter=None, fault_params=None`. Every
module must handle those `None`s (experiments A1/A3 that need ground truth declare
`requires_ground_truth=True` and are automatically skipped on real data).

---

## 2. The `mafaulda_synthetic` benchmark specification

### 2.1 Healthy physics (single source of truth: `src/physics/rotor_model.py`)

Two lumped bearing masses (underhang M2, overhang M3), each moving in x and y, coupled through
shaft stiffness, driven by rotor (M1) unbalance:

```
M2·ẍ2 + D2·ẋ2 + K1·x2 + Kc·(x2 − x3) = M1·ω²·E1·cos(ωt + φ0)
M2·ÿ2 + D2·ẏ2 + K1·y2 + Kc·(y2 − y3) = M1·ω²·E1·sin(ωt + φ0) − M2·g
M3·ẍ3 + D3·ẋ3 + K2·x3 + Kc·(x3 − x2) = 0
M3·ÿ3 + D3·ẏ3 + K2·y3 + Kc·(y3 − y2) = −M3·g
```

Default parameters follow `PINN_ARCH_DEFAULT` in the current configs (M1=50, M2=M3=3.5,
D=3000, K1=3.46e6, K2=3.81e6, E1=5e-6) with `Kc` chosen so coupled natural frequencies sit in
[80, 400] Hz — verify and tune during task 1.1 so signals are neither quasi-static nor aliased.

Implementation requirements:
- `rotor_model.py` exposes `simulate(params, omega, t) -> (pos, vel, acc)` **and**
  `residuals(pos, vel, acc, params, omega, t) -> (r1..r4)`. A unit test integrates the healthy
  system and asserts `‖r_i‖ ≈ 0` (< 1e-6 relative) — this test *is* the guarantee that PINN
  physics matches the simulator.
- The PINN's `compute_residuals` must be refactored to call these same functions (adapter around
  the existing `ConfigurablePINN`, keeping its trainable-parameter machinery). The mass-constraint
  residuals become config-optional (already supported via `enable_mass_constraints`).
- Linear system ⇒ solve with `scipy.integrate.solve_ivp` (RK45, `max_step ≤ dt/2`) or the exact
  matrix-exponential solution; either is fine, but validate energy/steady-state amplitude against
  the closed-form frequency response in the unit test.
- Observables: accelerations `(ẍ2, ÿ2, ẍ3, ÿ3)` → the 4 channels. X tensors `(N, T, 10)` =
  [4 velocities, 4 positions, ω, t] exactly like the MaFaulDa processed format.

### 2.2 Measurement model

`raw = clean + jitter`, where `clean` is the ODE acceleration and
`jitter = broadband Gaussian noise (σ_n) + high-frequency resonance hum (a_h·sin(2π·f_h·t + φ_c), f_h ≈ 5 kHz, per-channel phase)`.
σ_n and a_h are config knobs (defaults: σ_n = 5% RMS of healthy clean signal, a_h = 10%).
Both components are stored separately per window (ground truth for A1/A3).

### 2.3 Fault classes — the geometry dial

Six classes + healthy. Each class defines (i) a physical mechanism and (ii) a **sampling law over
its fault parameters** — the sampling law is what shapes the embedding-cluster geometry:

| # | Name | Mechanism | Parameter sampling law | Designed geometry |
|---|---|---|---|---|
| 0 | healthy | — | nominal params, small i.i.d. jitter on (D, K) ±1% | tight unimodal |
| 1 | imbalance_uni | E1 → κ·E1 | κ ~ N(6, 0.3²) truncated > 1 | **unimodal, near-Gaussian** (low 𝒢) |
| 2 | imbalance_bi | E1 → κ·E1, phase φ0 shift | (κ, φ0) ~ ½N((4, 0), Σ₁) + ½N((9, π/2), Σ₂) | **bimodal** (mixture, high 𝒢) |
| 3 | bpfo_impulsive | additive impact train: exp-decay impulses at f_bpfo = n_b/2·f_r·(1−d/D·cosθ) exciting a 2–3 kHz resonance | amplitude ~ LogNormal(μ_a, 0.5) | **heavy-tailed / impulsive** (high 𝒢, kurtotic) |
| 4 | misalign_cont | stiffness anisotropy ΔK on K1 (x vs y) | severity s ~ U(0.05, 0.5), ΔK = s·K1 | **elongated continuum** (moderate 𝒢, non-elliptical) |
| 5 | looseness_skew | intermittent stiffness dropouts (K1 → 0.6·K1 during random fraction of each rotation) | dropout duty ~ Beta(2, 8) | **skewed** unimodal (moderate 𝒢) |
| 6 | combo_ring | imbalance with φ0 ~ U(0, 2π), fixed κ | phase uniform on circle | **ring/curved manifold** (high 𝒢, zero-mean trap for Gaussian estimator) |

Notes for the implementer:
- Classes 1 vs 2 differ **only** in the sampling law — this is the cleanest matched pair for H
  (same mechanism, different geometry). Preserve that property; do not "improve" class 2's realism.
- Class 3 is additive forcing (impacts enter the ODE as an external force on M3), not a signal
  overlay — the physics residual must genuinely deviate.
- Class 6 exists to break the single-Gaussian estimator maximally (its class mean is near the
  healthy manifold). If any class must be cut for compute, cut 5 before 6.
- Per class × speed: 600 train / 200 val / 200 test windows (config). Windows are independent
  draws (fresh fault-parameter sample + fresh noise), not slices of one long run — this avoids
  the temporal-leakage problem the real pipeline had to solve with interleaved splits.
- Everything is generated by `run.py generate-data dataset=mafaulda_synthetic seed=...`, cached
  under `data/processed-mafaulda-synthetic/<version>/`, with a `metadata.json` capturing the full
  generator config hash. Regeneration is deterministic given (config, seed).

### 2.4 Designed-geometry verification (gate for everything downstream)

After task 1.3 (PINN trained), compute Φ-embeddings of the *real* (simulated) fault windows,
PCA-32, then 𝒢_y per class. **Gate:** the rank ordering of 𝒢_y must match the design intent
(classes 2, 3, 6 ≫ classes 1, 5 ≳ 4… exact order flexible, but 1 must be lowest of the fault
classes and 2/6 must be clearly non-Gaussian by HZ test p < 0.01). If the dial does not produce a
spread of 𝒢_y, iterate on §2.3 sampling laws **before** building anything in Blocks B–F, because
H is untestable without variance in the independent variable.

---

## 3. Component-level fixes (what to keep, change, delete)

| Module | Verdict | Actions |
|---|---|---|
| `src/models/pinn.py` | keep, refactor | Extract residual math to `physics/rotor_model.py`; PINN calls it. Delete plotting/training code embedded in the file (trainer moves to `src/training/`). Keep ReLoBRaLo. Add `from_config` + strict checkpoint round-trip test. |
| `src/models/pinn_old_compat.py` | delete (→ `legacy/`) | Real-data-only shim for dead checkpoints. |
| `src/models/feature_extractors.py` | keep | Audited, differentiable, float64-internally. Parameterize `num_freq_bins`/wavelet levels via config; document output dim formula; add gradient-flow unit test (exists partially in `tests/test_math_extractor.py` — port it). |
| `src/models/ts_jepa.py` | rewrite (v2) | Add VICReg-style variance+covariance loss on pooled z_macro (weights in config); optional SIGReg term (A2 flag); collapse monitors logged every epoch (per-dim std across batch, effective rank). Keep tokenizer/EMA scheme. |
| `src/models/decoder1.py`, `decoder2_cvae.py` | keep, small fixes | Make `seq_length`/`num_classes` constructor-required (no cfg import inside models — models must be config-free). Add β-TCVAE total-correlation option to Decoder2 (E1). |
| `src/models/latent_diffusion.py` | keep + extend | Add a proper `DDIMSampler` (subsampled schedule, x0-prediction, optional clamp) as library code — currently DDIM exists only ad hoc inside task scripts. Add optional class-conditioning (FiLM on time embedding — reuse T7's `LabelConditionedDDPM` design) for E2. |
| `src/models/oracles.py` | rewrite → `src/oracle/embedding.py` | Split: (a) `PhysicsEmbedding` = frozen PINN → Φ → PCA projection, differentiable, explicit `mode ∈ {pinn, raw}` with `raw` tagged as diagnostic; (b) all target-distribution logic moves to `oracle/guidance.py` (`GuidanceDensity`). Kill the buffer-of-means design. Kill the omega `BatchTuple` hack — omega is a first-class field of `Batch`. |
| `src/pipelines/run_sdedit_phase2.py` | rewrite → `backbones/ours_sdedit.py` | Correct SDEdit (§3.1 below). Plotting moves to `evaluation/plots.py`; no hardcoded paths. |
| `src/evaluation/metrics.py` | rewrite → `evaluation/fidelity.py` | TSTR per §0.3 (no pseudo-labels, signal-space features, 3 probes, multi-seed). Keep the MMD median-heuristic implementation (it's fine); add per-class MMD driver. |
| `src/configs.py` | delete | Replaced by `src/configs/schema.py` + YAML. A grep-gate test asserts no module under `src/` imports a global mutable config. |
| root `main_*.py`, `train_*_task*.py`, `scripts/task*` | move to `legacy/` | Keep for reference; excluded from packaging and linting. `scripts/download_dataset.sh` stays (real-data phase). |
| `.gitignore` | fix | Add `*.pth`, `__pycache__/`, `results*/`, `data/processed-*/`. `git rm --cached` the committed binaries. |
| `tests/*` | port + green | Fix the four existing test files against the new layout; all tests must pass before Phase 1 is considered done. |

### 3.1 Correct guided SDEdit (normative spec for `backbones/ours_sdedit.py`)

```
inputs: x_healthy (B,4,T), target class y, guidance density q_y, config:
        t0_frac ∈ (0,1]           # fraction of the TRAINING schedule (default 0.5 → t0=500)
        n_infer                    # inference steps, DDIM-subsampled from [0, t0] (default 50)
        guidance_scale η, guidance_interval N, clip τ
1. z_h = jepa.get_z_macro(x_healthy)
2. t0 = int(t0_frac * scheduler.num_train_timesteps)          # NOT n_infer * strength
3. z_t = add_noise(z_h, ε, t0)
4. for t in ddim_schedule(t0 → 0, n_infer):
     ε̂ = ldm(z_t, t)
     if step_index % N == 0:                                   # guidance interval
         with grad: v = pca(Φ(pinn(dec1(z_t))))                # differentiable chain
                    L = -q_y.log_prob(v)                       # density penalty (B1/B2 swappable)
         g = ∇_{z_t} L;  g = clip_norm(g, τ)
         log ‖g‖₂, L per step                                  # D3 requires this
     z_{t-1} = ddim_step(ε̂, t, z_t) − η·g
5. x̂ = dec1(z_0) + dec2.sample(z_0, y)
returns x̂ plus a StepTrace (t, L, ‖g‖, z_t snapshots) for D3/plots
```

Sanity gate (task 3.2): with η=0 the output must reconstruct a plausible healthy signal
(round-trip MSE within 2× of Decoder1's validation MSE), and with η>0 on class 1 the 1X harmonic
amplitude must increase monotonically over the trajectory. These two checks would have caught the
`t_start=1` bug immediately.

---

## 4. Step-by-step task list

Conventions: tasks are `P<phase>.<n>`; each lists **Deliverables**, **Depends**, and
**Acceptance** (machine-checkable where possible). Run everything through
`python run.py <command>`; every training/eval command accepts `--smoke` (tiny sizes, <5 min
total) — implement `--smoke` from day one, it is the regression harness for the whole plan.
Recommended global defaults: `seed=0` canonical, 5 seeds {0..4} for all reported metrics; device
auto (CUDA if available); float32 everywhere except PINN internals (float64, as now).

### Phase 0 — Repo hygiene & skeleton (no science yet)

- **P0.1 Quarantine legacy.** Create `legacy/`, move root-level `main_*.py`, `train_*_task*.py`,
  one-off `scripts/task*/t10/eval_raw_bypass` scripts there verbatim. Fix `.gitignore`,
  `git rm --cached` binaries/pycache. Tag the pre-refactor state (`git tag pre-extension`).
  *Acceptance:* `git status` clean of binaries; `pytest` still collectable; repo imports nothing
  from `legacy/`.
- **P0.2 Config system + run manifests.** `src/configs/schema.py` (frozen dataclasses:
  `RunCfg{data, pinn, jepa, decoders, ldm, sdedit, guidance, eval, seed, device, smoke}`), YAML
  loader with overrides (`run.py train-pinn data.speeds=[16] --smoke`), run-directory manager
  (`results/<experiment>/<timestamp>_<seed>/run.json` = resolved config + git SHA + outputs).
  Seeding utility (torch/numpy/random + deterministic flags).
  *Acceptance:* unit tests for load/override/freeze; grep-gate: no `import src.configs` anywhere.
- **P0.3 Test harness.** Port the 4 existing test files to the new layout; add `tests/smoke/`
  that runs the full pipeline `--smoke` (later tasks extend it). CI-style command:
  `python -m pytest -x`.
  *Acceptance:* all tests green on Windows (mind cp1252: ASCII-only prints).

### Phase 1 — Physics, data, PINN (the foundation the last paper never had)

- **P1.1 `physics/rotor_model.py`.** Healthy ODE + residual functions + closed-form frequency
  response; simulator (`solve_ivp` at 50 kHz, transient discarded).
  *Depends:* P0.2. *Acceptance:* residuals-on-simulation test < 1e-6 relative; steady-state
  amplitude matches frequency response < 1%; natural frequencies inside [80, 400] Hz.
- **P1.2 `physics/faults.py` + `data/mafaulda_synthetic.py`.** Seven classes per §2.3;
  measurement model per §2.2; writes processed tensors + `metadata.json` in the MaFaulDa
  processed format; implements `DatasetBundle` (windowing, per-speed splits,
  `fewshot_fault_sets`). CLI: `run.py generate-data`.
  *Depends:* P1.1. *Acceptance:* determinism test (same seed ⇒ identical tensors); spectral
  checks — class 1 boosts 1X line, class 3 shows f_bpfo impulse train + resonance band, class 4
  shows 2X and x/y asymmetry; plots saved for human review.
- **P1.3 PINN training (`run.py train-pinn`).** ConfigurablePINN + ReLoBRaLo on **healthy
  synthetic only**, all speeds. Normalization bounds computed from healthy train split, saved in
  run dir (not a magic global path).
  *Depends:* P1.2. *Acceptance (hard gate):* healthy val physics residual ≪ fault residuals
  (report ratio ≥ 10×); Φ-feature silhouette across the 7 classes (PCA-32) **≥ 0.3** — the gate
  the real data failed; if not met, debug PINN/physics before proceeding (this is the point of
  synthetic-first).
- **P1.4 `oracle/embedding.py`.** PhysicsEmbedding (frozen PINN → Φ → PCA-32 fitted on train
  Φ-features), differentiable; `mode=raw` diagnostic path with manifest tagging.
  *Depends:* P1.3. *Acceptance:* gradcheck-style test — finite, non-zero ∇ w.r.t. input window;
  bypass-tagging test.
- **P1.5 Geometry-dial verification (gate, §2.4).** `run.py verify-geometry` computes
  Silhouette/CH/DB + 𝒢_y (3 statistics) per class × speed → `geometry.json` + UMAP figure.
  *Depends:* P1.4. *Acceptance:* 𝒢_y ordering matches design (see §2.4); results persisted (these
  same numbers are C1's deliverable later).

### Phase 2 — Representation stack

- **P2.1 TS-JEPA v2 (`run.py train-jepa`).** Anti-collapse per §3; flag `jepa.sigreg ∈ {on,off}`
  (train both arms; store both checkpoints — A2/C3 need them).
  *Depends:* P1.2. *Acceptance:* per-dim std of z_macro across val set ∈ [0.1, 1.5]; effective
  rank ≥ 32; linear probe (logistic on z_macro) ≥ 70% on 7 classes — z_macro must carry class
  signal for SDEdit to have anything to steer.
- **P2.2 Decoder1 (`run.py train-dec1`).** Trained on `clean` targets (we *have* the envelope
  ground truth — do not train on raw, that was a real-data compromise).
  *Acceptance:* val MSE vs clean ≤ 1% of signal power; spectral check: < 5% of output energy
  above 2.5 kHz (envelope must not contain jitter band).
- **P2.3 Decoder2 CVAE (`run.py train-dec2`).** Residual = raw − Decoder1(z); β-TCVAE penalty
  behind `dec2.beta_tc` flag (E1 arm).
  *Acceptance:* val ELBO reported; KL > 0 (non-collapsed); PSD of sampled jitter within 3 dB of
  true jitter PSD in the 4–6 kHz band.
- **P2.4 Disentanglement tooling.** `models/mine.py` (MINE estimator) + `evaluation/
  disentanglement.py`: `I(z_macro; ε)` (A1: JEPA vs plain-AE baseline encoder — implement a
  small conv AE for the comparison), cross-spectral coherence between Decoder1 output and CVAE
  residual (A3).
  *Depends:* P2.1–2.3. *Acceptance:* MINE sanity test on correlated Gaussians (known MI within
  20%); tools run on smoke data.

### Phase 3 — Generation, guidance, backbones

- **P3.1 LDM + DDIM (`run.py train-ldm`).** Train on all-class latents; class-conditional variant
  behind `ldm.class_cond` (E2).
  *Acceptance:* unconditional samples' global mean/std within [0.5, 2]× of real latent stats;
  no clamp saturation with DDIM sampler.
- **P3.2 SDEdit v2.** Per §3.1 normative spec, in `backbones/ours_sdedit.py`.
  *Depends:* P3.1, P1.4. *Acceptance:* the two sanity gates in §3.1; StepTrace persisted.
- **P3.3 `oracle/guidance.py`.** `GaussianGuidance` (Ledoit–Wolf), `GMMGuidance` (full-cov, k by
  BIC ≤ 4), `FlowGuidance` (MAF, 5 layers, trained with early stopping on held-out LL). Common
  API: `fit(V_y, seed)`, `log_prob`, `held_out_ll`, `penalty` (differentiable).
  *Acceptance:* on designed toy data (2-component mixture) GMM/Flow held-out LL beats Gaussian;
  all penalties gradcheck; fitting with n=5 samples does not crash (shrinkage/regularization
  floors).
- **P3.4 Backbones.** `SynthesisBackbone` protocol; wrap ours; implement `cvae_backbone.py`
  (signal-space conditional VAE, ~Decoder2-scale) and `faultdiff_backbone.py` (frozen healthy
  DDPM on z or signal patches + per-class low-rank difference adapter trained on the few-shot
  fault set). **Crucial:** both external backbones must accept the same `GuidanceDensity` objects
  — for diffusion-style backbones as gradient guidance; for the CVAE via density-weighted
  rejection/importance re-sampling of its outputs (document this as the CVAE's guidance
  mechanism — cross-backbone comparability of Block B requires every backbone to consume q_y
  somehow).
  *Acceptance:* each backbone passes a common contract test: `fit(smoke_bundle)`,
  `synthesize(healthy, y, n=8)` returns finite (n,4,T) windows in ≤ config runtime.
- **P3.5 Evaluation suite.** `evaluation/fidelity.py` (TSTR ×3 probes ×5 seeds, per-class MMD on
  raw windows via Φ_raw features, ΔSilhouette transfer score per C2), `evaluation/geometry.py`
  (already partly done in P1.5), `evaluation/correlate.py` (Pearson/Spearman/Kendall + BCa
  bootstrap CIs over (class × speed) cells; pre-registered threshold |ρ| ≥ 0.4 encoded as a
  constant with a comment linking to the proposal).
  *Acceptance:* metric unit tests on synthetic toys (e.g., TSTR=oracle when generated==real;
  MMD→0 for identical distributions).

### Phase 4 — Experiment blocks (each = one driver in `src/experiments/blocks/`, one YAML, one `results/<block>/` tree)

Every driver iterates the full factorial declared in its YAML, one run dir per cell, and writes a
tidy `cells.parquet` (long format: backbone, estimator, class, speed, seed, metric, value).

- **P4.A Block A.** A1: `I(z;ε)` JEPA vs AE baseline (P2.4). A2: full C3 pipeline on the
  SIGReg-on encoder arm (uses P2.1's second checkpoint). A3: coherence quantification + report.
  *Acceptance:* A1/A3 numbers in manifest; A2 produces a parallel `cells.parquet` for C3.
- **P4.B Block B (core).** Factorial: estimator {Gaussian, GMM, Flow} × backbone {ours, CVAE,
  FaultDiff} × class {1..6} × speed {12,16,20} × seed {0..4}; B3: few-shot n ∈ {5,10,20,50,100}
  with held-out LL logged per fit (variance-vs-bias confound control).
  Budget guard: with defaults this is ≈ 3·3·6·3·5 = 810 synthesis cells (+5 few-shot levels on
  the home speed only — scope B3 to ω=16 Hz to keep ≤ 1350 total); n_gen per cell = 200 windows.
  If compute-bound, drop to 3 seeds *and say so in the manifest*, never silently.
- **P4.C Block C.** C1 = P1.5 outputs (recompute per speed). C2: ΔSilhouette transfer score
  (train metric-learning probe on synthetic-only, measure real-cluster silhouette gain vs
  real-trained probe). C3 (**primary**): correlate 𝒢_y with per-class fidelity (MMD primary,
  TSTR-recall secondary) across cells, per backbone and per estimator; produce the
  invariance-grading (a/b/c outcomes per proposal §Hypothesis).
  *Acceptance:* `c3_report.json` with ρ, CIs, per-backbone sign/magnitude table + the decision
  per the pre-registered falsification rule.
- **P4.D Block D.** D1: guidance interval N ∈ {1, 5, 10, 25, 50} (adjusted to n_infer=50; the
  proposal's {10..500} assumed 1000 steps — record the mapping in the YAML comment); cost–fidelity
  frontier. D2: t0_frac ∈ {0.2, 0.35, 0.5, 0.65, 0.8}. D3: aggregate StepTraces — correlate mean
  ‖g_t‖ and its variance with per-class MMD and 𝒢_y. D4: multi-speed — train at 16 Hz, synthesize
  /evaluate at 12/20 Hz; speed-conditioned vs speed-agnostic PINN normalization arms.
  *Acceptance:* four figures + `cells.parquet` rows; D3 correlation reported with CI.
- **P4.E Block E.** E1: β-TCVAE arm → rerun fidelity on ours-backbone, compare (uses P2.3 flag).
  E2: latent modality check (BIC of GMM on z_macro; if multimodal, class-conditional LDM arm
  from P3.1) → does capacity close the high-𝒢 gap? (H predicts no.) E3: open-set — hold out
  class 4 entirely; build q_target by interpolating class centroids/densities (Gaussian: convex
  combo of moments; Flow: latent-space interpolation); synthesize, then measure discriminability
  vs known classes and distance to the *true* held-out cluster (we have it — synthetic
  superpower). E4: "cross-machine" transfer — generate `mafaulda_synthetic_B` (different M/K/D
  values, +20–40% detuning; same fault taxonomy) and evaluate zero-shot TSTR of the
  A-trained pipeline on B (PINN and encoder frozen).
  *Acceptance:* each sub-experiment has its own manifest + figure; E3 explicitly reports
  open-space risk (fraction of synthesized samples closer to healthy than to any fault cluster).
- **P4.F Block F.** F1: multi-seed benchmark table of the 3 backbones (TSTR ×3 probes, per-class
  MMD, CIs) — this doubles as the B/C3 cross-backbone data. F2: probe-sensitivity report (do
  conclusions flip across linear/1NN/SVM? McNemar or bootstrap test). F3: causal ablation —
  ours with η=0 (no guidance) and ours with guidance-on-random-target (steer toward wrong class)
  vs full system; per-class deltas.
  *Acceptance:* benchmark table renders from `cells.parquet` (no hand-typed numbers); F3 shows
  guidance effect with CI.

### Phase 5 — Aggregation & paper artifacts

- **P5.1 Aggregator.** `run.py report`: reads every `cells.parquet` + `run.json`, refuses
  bypass-tagged rows in headline tables, emits `report/` with: master results table (F1), C3
  correlation figure (𝒢_y vs MMD, per backbone), D1/D2 frontier plots, geometry-dial UMAP grid,
  B3 few-shot curves with held-out-LL insets, E3 open-set figure. All figures regenerable from
  one command.
- **P5.2 Hypothesis decision memo.** Auto-generated `report/H_decision.md`: applies the
  pre-registered rule (|ρ| ≥ 0.4, sign consistency, estimator-flexibility gain ∝ 𝒢_y, capacity
  interventions fail to close gap) and states outcome (a)/(b)/(c). This is the seed of the
  paper's results section and keeps reporting honest.
- **P5.3 Real-data readiness checklist.** Verify `dataset=mafaulda` config resolves end-to-end
  against the `DatasetBundle` contract using a *stub* bundle (shapes only, no real data);
  document in `docs/REAL_DATA_SWITCH.md` exactly what the real-data phase must provide (processed
  tensors, PINN retraining at native speed, which experiments auto-skip without ground truth).

---

## 5. Risks & fallbacks

| Risk | Mitigation |
|---|---|
| PINN silhouette gate (P1.3) fails even on matched physics | Debug ladder: (1) check normalization bounds; (2) train per-speed PINNs; (3) reduce Φ to residual channels only (drop unmeasured-force channels). Do not proceed to Phase 2 with a failed gate — that is how the last paper died. |
| Geometry dial produces uniform 𝒢_y (P1.5) | Increase inter-mode separation of class 2, widen class 4 severity range, raise class 3 LogNormal σ. The dial parameters exist precisely for this iteration. |
| Flow guidance unstable with n≤20 samples (B3) | Regularization floor: fall back to GMM(k=1)=Gaussian at n=5; log the fallback in the manifest (it is itself a B3 finding, bias vs variance). |
| FaultDiffusion-style backbone underperforms embarrassingly | Report as-is with the "-style reimplementation" caveat; it is a test vehicle for H (its role is carrying Block B/C3, not winning F1). |
| Compute (CPU-only host) | All defaults sized for ≈ hours-per-block on CPU (small MLP LDM, T=3014, ≤200 gen/cell); `--smoke` for iteration; blocks are embarrassingly parallel over cells (cell-level resume: skip cells whose run.json exists). |
| Windows quirks | ASCII-only console output; `torchaudio.lfilter` used only in the real-data integration path — synthetic bundle must not import it (the old code needed lfilter only for accelerometer drift mitigation, which simulated data doesn't have). |

## 6. Definition of done

1. `python -m pytest` green; `python run.py smoke-all` runs generate-data → PINN → JEPA →
   decoders → LDM → SDEdit → all six block drivers on smoke sizes in < 30 min.
2. All Phase-1 gates passed (residual match, silhouette ≥ 0.3, geometry-dial ordering).
3. `results/**/cells.parquet` covers the full declared factorials; `run.py report` regenerates
   every figure/table from manifests alone.
4. `report/H_decision.md` states the hypothesis outcome under the pre-registered rule.
5. `docs/REAL_DATA_SWITCH.md` documents the flag-flip path, and the stub-bundle contract test
   passes for `dataset=mafaulda`.
