# Phase 2 & 3: Physics-Guided Counterfactual Fault Synthesis Pipeline

This repository contains a modular PyTorch codebase for training the Phase 2 component of a Cyber-Physical System's counterfactual fault synthesis methodology. 

The methodology leverages a Multivariate Time-Series Joint-Embedding Predictive Architecture (TS-JEPA) alongside deterministic and stochastic decoders (CVAE) to disentangle deterministic low-frequency macro-dynamics from high-frequency sensor noise across multi-channel sensors (4 channels).

## Methodology
The pipeline extracts physical features and synthesizes fault superpositions using natively multivariate architectures:
1. **TS-JEPA for Macro-State Encoding:** A 1D-CNN tokenizer fuses multi-channel spatial data concurrently into sequences, followed by a Transformer encoder to map the un-masked sequence into a global low-dimensional structural macro-latent vector (`z_macro`).
2. **Deterministic Decoder (Decoder 1):** Trained to map the structural `z_macro` to the raw ground-truth trace. Because TS-JEPA's self-supervised process functions as a natural mechanical low-pass filter, Decoder 1 successfully converges solely capturing the un-jittered low-frequency macroscopic physical behavior.
3. **Stochastic Decoder (Decoder 2 CVAE):** Learns the probabilistic distribution of the high-frequency residual left explicitly after subtracting Decoder 1's reconstruction from the original signal.
4. **Superposition:** Generates synthetic multivariate vibrations by combining Decoder 1 deterministic reconstructions and Decoder 2 generative noises.

## Directory Structure
```text
.
├── main_phase2.py          # Phase 2 training entry point
├── main_phase3.py          # Phase 3 SDEdit & LDM entry point
├── README.md
├── results/                # Trained checkpoints and visualization artifacts
│   ├── *.pth               # Model weights (TS-JEPA, Decoders)
│   └── *.png               # UMAP and SDEdit trajectory plots
└── src/
    ├── data/
    │   ├── __init__.py
    │   └── synthetic_dataset.py
    ├── models/
    │   ├── __init__.py
    │   ├── decoder1.py
    │   ├── decoder2_cvae.py
    │   ├── latent_diffusion.py
    │   ├── oracles.py
    │   └── ts_jepa.py
    └── pipelines/
        ├── __init__.py
        ├── run_sdedit_phase3.py
        └── train_phase2.py
```

## Running the Pipeline

### Phase 2: TS-JEPA & CVAE Training
Execute the Phase 2 orchestrator script to handle training the TS-JEPA encoder, the deterministic Decoder 1, and the stochastic CVAE (Decoder 2).

```bash
python main_phase2.py --epochs 100
```

### Phase 3: Physics-Guided SDEdit Architecture
We have implemented the translation mechanisms that morph healthy `z_macro` states into specific physical faults using **Stochastic Differential Editing (SDEdit)**.

**Core Components for Phase 3**:
1. **Mock Oracle (`src/models/oracles.py`)**: A static neural structure acting as our target physics constraint module. It enforces Vector-Jacobian Product (VJP) backpropagation gradients.
2. **Latent Diffusion Model (`src/models/latent_diffusion.py`)**: An efficient DDPM structure learning the intrinsic distribution of the `z_macro` sequence.
3. **The SDEdit Pipeline (`src/pipelines/run_sdedit_phase3.py`)**: Orchestrates iterative sampling via Forward Noise Injection and Guided Reverse Denoising.

#### Running Phase 3:
Execute the Phase 3 endpoint to load (or train) Phase 2 checkpoints and initiate LDM compilation.

```bash
python main_phase3.py --ldm_epochs 50
```

#### Analytical UMAP Progression
The `run_sdedit_phase3` iteration loop captures the dimensional state of $z_t$ during every execution step. We project this trajectory back over our baseline TS-JEPA UMAP manifold:
- `results/umap_sdedit_trajectory_class_1.png`
- `results/umap_sdedit_trajectory_class_2.png`

This visually validates how physics gradients explicitly bend the transformation path out of pure randomness into defined targeted clusters!
