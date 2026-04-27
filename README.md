# Physics-Guided Counterfactual Fault Synthesis Pipeline

This repository contains a modular PyTorch codebase for training the Phase 2 component of a Cyber-Physical System's counterfactual fault synthesis methodology.

The methodology leverages a Physics-Informed Neural Network (PINN) as an Oracle constraint, a Multivariate Time-Series Joint-Embedding Predictive Architecture (TS-JEPA), and deterministic/stochastic decoders (CVAE) to disentangle low-frequency macro-dynamics from high-frequency sensor noise across 4-channel spatial/orbital coordinates.

## Methodology
The pipeline operates in three distinct phases:
1. **Phase 0 (Physical Foundation):** Pre-processes raw MaFaulDa CSVs into differentiable physical states (Acceleration, Velocity, Position). Trains a PINN Oracle via a dynamically balanced ReLoBRaLo loss on healthy signals to extract an 8D physics feature space.
2. **Phase 1 (Representation Learning):** 
    *   **TS-JEPA:** Maps multi-channel sequences into a global low-dimensional structural macro-latent vector (`z_macro`).
    *   **Decoder 1 (Deterministic):** Maps `z_macro` to the raw ground-truth trace, converging to capture un-jittered low-frequency macroscopic physical behavior.
    *   **Decoder 2 (CVAE):** Learns the probabilistic distribution of the high-frequency residual left after subtracting Decoder 1's reconstruction.
3. **Phase 2 (Counterfactual Synthesis via SDEdit):** Orchestrates iterative sampling via Forward Noise Injection and Guided Reverse Denoising on the Latent Diffusion Model (LDM). The pre-trained Phase 0 PINN acts as an Oracle, computing Vector-Jacobian Products (VJPs) to explicitly steer the generative diffusion path toward targeted physical fault distributions, tracking dynamic rotational speeds (`omega`) on the fly.

## Directory Structure
```text
.
├── main_phase0.py          # Phase 0 physical preprocessing & PINN training
├── main_phase1.py          # Phase 1 training entry point
├── main_phase2.py          # Phase 2 SDEdit & LDM entry point
├── README.md
├── results/                # Trained checkpoints and visualization artifacts
│   ├── *.pth               # Model weights (TS-JEPA, Decoders)
│   └── *.png               # UMAP and SDEdit trajectory plots
└── src/
    ├── data/
    │   ├── __init__.py
    │   ├── clean_mafaulda_processor.py  # SciPy Signal processing & integration drift mitigation
    │   ├── mafaulda_dataset.py          # Custom MaFaulDa DataLoader with dynamic omega side-channels
    │   └── synthetic_dataset.py
    ├── models/
    │   ├── __init__.py
    │   ├── decoder1.py
    │   ├── decoder2_cvae.py
    │   ├── latent_diffusion.py
    │   ├── oracles.py                   # PriorWorkOracle wrapping the frozen PINN for diffusions
    │   ├── pinn.py                      # Core Physics-Informed Neural Network
    │   ├── relobralo_loss.py            # Dynamic loss balancing
    │   └── ts_jepa.py
    └── pipelines/
        ├── __init__.py
        ├── run_sdedit_phase2.py
        └── train_phase1.py
```

## Running the Pipeline

### Phase 1: TS-JEPA & CVAE Training
Execute the Phase 1 orchestrator script to handle training the TS-JEPA encoder, the deterministic Decoder 1, and the stochastic CVAE (Decoder 2).

```bash
python main_phase1.py --epochs 100
```

### Phase 2: Physics-Guided SDEdit Architecture
We have implemented the translation mechanisms that morph healthy `z_macro` states into specific physical faults using **Stochastic Differential Editing (SDEdit)**.

**Core Components for Phase 2**:
1. **Mock Oracle (`src/models/oracles.py`)**: A static neural structure acting as our target physics constraint module. It enforces Vector-Jacobian Product (VJP) backpropagation gradients.
2. **Latent Diffusion Model (`src/models/latent_diffusion.py`)**: An efficient DDPM structure learning the intrinsic distribution of the `z_macro` sequence.
3. **The SDEdit Pipeline (`src/pipelines/run_sdedit_phase2.py`)**: Orchestrates iterative sampling via Forward Noise Injection and Guided Reverse Denoising.

#### Running Phase 2:
Execute the Phase 2 endpoint to load (or train) Phase 1 checkpoints and initiate LDM compilation.

```bash
python main_phase2.py --ldm_epochs 50
```

#### Analytical UMAP Progression
The `run_sdedit_phase2` iteration loop captures the dimensional state of $z_t$ during every execution step. We project this trajectory back over our baseline TS-JEPA UMAP manifold:
- `results/umap_sdedit_trajectory_class_1.png`
- `results/umap_sdedit_trajectory_class_2.png`

This visually validates how physics gradients explicitly bend the transformation path out of pure randomness into defined targeted clusters!

## Dataset

The project uses the [MaFaulDa (Machinery Fault Database)](https://www02.smt.ufrj.br/~offshore/mfs/page_01.html). 

| Condition | Download Link |
| :--- | :--- |
| **Normal (no fault)** | [normal.zip](https://www02.smt.ufrj.br/~offshore/mfs/database/mafaulda/normal.zip) |
| **Horizontal Misalignment** | [horizontal-misalignment.zip](https://www02.smt.ufrj.br/~offshore/mfs/database/mafaulda/horizontal-misalignment.zip) |
| **Vertical Misalignment** | [vertical-misalignment.zip](https://www02.smt.ufrj.br/~offshore/mfs/database/mafaulda/vertical-misalignment.zip) |
| **Imbalance** | [imbalance.zip](https://www02.smt.ufrj.br/~offshore/mfs/database/mafaulda/imbalance.zip) |
| **Underhang Bearing** | [underhang.zip](https://www02.smt.ufrj.br/~offshore/mfs/database/mafaulda/underhang.zip) |
| **Overhang Bearing** | [overhang.zip](https://www02.smt.ufrj.br/~offshore/mfs/database/mafaulda/overhang.zip) |

### Downloading the Dataset

We provide a utility script to automate downloading and optionally extracting the datasets into the required `data/raw-mafaulda` directory structure.

**Usage Examples:**
```bash
# Download all dataset zips
./scripts/download_dataset.sh -all

# Download only specific subsets
./scripts/download_dataset.sh -normal -imbalance

# Download all zip files and automatically extract their contents
./scripts/download_dataset.sh -all -unzip
```

Installing torch just for CPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
