"""
Centralized Configuration for Physics-Guided Counterfactual Fault Synthesis.
Grouped by Pipeline Phase and Model Component.
"""

import os

# ==============================================================================
# GLOBAL SETTINGS & PATHS
# ==============================================================================
DATASET_VERSION = "v1"
DATA_DIR_RAW = "data/raw-mafaulda/"
DATA_DIR_PROCESSED = f"data/processed-mafaulda/{DATASET_VERSION}"
RESULTS_DIR = "results"

NORM_METADATA_PATH = os.path.join(RESULTS_DIR, "normalization_metadata.pth")
PINN_MODEL_PATH = os.path.join(RESULTS_DIR, "pinn.pth")
JEPA_MODEL_PATH = os.path.join(RESULTS_DIR, "ts_jepa.pth")
DEC1_MODEL_PATH = os.path.join(RESULTS_DIR, "decoder1.pth")
DEC2_MODEL_PATH = os.path.join(RESULTS_DIR, "decoder2.pth")
LDM_MODEL_PATH = os.path.join(RESULTS_DIR, "ldm.pth")

# ==============================================================================
# PHASE 0: PHYSICAL FOUNDATION (PINN)
# ==============================================================================
PHASE0_TRAIN_SETTINGS = {
    'epochs': 100000,
    'batch_size': 128,
    'rotation_hz': 30,
    'early_stop_patience': 15,
    'early_stop_min_delta': 1e-4,
    'learning_rate': 1e-4,
}

PINN_ARCH_DEFAULT = {
    'unmeasured_net_config': {
        'hidden_layers': [64, 64],
        'activation': 'tanh',
        'dropout_rate': 0.0,
        'init_method': 'xavier_normal'
    },
    'acceleration_net_config': {
        'hidden_layers': [64, 64],
        'activation': 'tanh',
        'dropout_rate': 0.0,
        'init_method': 'xavier_normal'
    },
    'param_init_config': {
        'method': 'fixed',
        'values': {
            'M1': 50.0, 'M2': 3.5, 'M3': 3.5,
            'D1': 3000.0, 'D2': 3000.0, 'D3': 3000.0,
            'K1': 3.4635e6, 'K2': 3.8127e6, 'E1': 5.0e-6
        }
    },
    'enable_mass_constraints': True
}

# ==============================================================================
# PHASE 1: MODULAR TRAINING (JEPA & DECODERS)
# ==============================================================================
PHASE1_TRAIN_SETTINGS = {
    'max_epochs': 100,
    'batch_size': 32,
    'learning_rate': 1e-3,
    'beta_kl': 0.01,
    'early_stop_patience': 5,
}

JEPA_CONFIG = {
    'd_model': 128,
    'patch_size': 50,
    'nhead': 4,
    'num_layers': 4,
    'ema_decay': 0.99,
}

# ==============================================================================
# PHASE 2: GUIDED SYNTHESIS (LDM & SDEDIT)
# ==============================================================================
PHASE2_TRAIN_SETTINGS = {
    'ldm_epochs': 50,
    'learning_rate': 1e-3,
}

SDEDIT_GUIDANCE_SETTINGS = {
    'num_inference_steps': 1000,
    'guidance_scale': 0.5,
    'strength': 0.5,
}

# ==============================================================================
# PHYSICS CONSTANTS
# ==============================================================================
GRAVITY = 9.81
SAMPLING_RATE = 50000  # Hz
