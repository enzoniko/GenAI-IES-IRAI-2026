"""
Centralized Configuration for Physics-Guided Counterfactual Fault Synthesis.
Grouped by Pipeline Phase and Model Component.
"""

import os

# ==============================================================================
# GLOBAL SETTINGS & PATHS
# ==============================================================================
# Dataset organization
DATASET_VERSION = "v1"
TARGET_HZ = 16.0
DATA_DIR_RAW = "data/raw-mafaulda/"
# Processed data will be in categorical subfolders named after the frequency (e.g. data/processed-mafaulda/16hz/)
DATA_DIR_PROCESSED = f"data/processed-mafaulda/{int(TARGET_HZ)}hz"
RESULTS_DIR = "results"

# Number of classes in the reduced MaFaulDa dataset
NUM_CLASSES = 4
SEQ_LENGTH = 3014 # Standardized for 16Hz biological windows
# Oracle mode for PriorWorkOracle: "pinn" uses the full physics path, "raw" bypasses it.
ORACLE_MODE = "pinn"

# How much windows will be used from the dataset
WINDOW_PCT = 0.90


# Signal Processing Settings
SIGNAL_PROCESSING_STRATEGY = 'previous_strategy'  # Options: 'fft', 'previous_strategy'
# TO VERIFY THE PREVIOUS CUTOFF AND HOW IT WAS CALCULATED
CUTOFF_HZ = 2.0  # Hz, for drift mitigation filtering

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
        'rotation_hz': 16,
    'training_windows': 1000,
    'test_windows': 20,
    'early_stop_patience': 20,
    'early_stop_min_delta': 1e-4,
    'learning_rate': 1e-4,
}

PINN_ARCH_DEFAULT = {
    'unmeasured_net_config': {
        'hidden_layers': [128, 128],
        'activation': 'elu',
        'dropout_rate': 0.0,
        'init_method': 'xavier_normal'
    },
    'acceleration_net_config': {
        'hidden_layers': [128, 128],
        'activation': 'elu',
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

RELOBRALO_DEFAULT = {
    'alpha': 0.5125,
    'rho': 0.2332,
    'temperature': 1.4198,
}

# ── Point-Wise PINN Training (Prior Paper Exact Hyperparameters) ──────────────
PINN_POINTWISE_SETTINGS = {
    'epochs': 20000,
    'batch_size': 256,
    'learning_rate': 0.0039199623708041885,
    'max_samples': 1000,
    'early_stop_patience': 100,
    'early_stop_min_delta': 1e-7,
    'lr_scheduler_patience': 50,
    'lr_scheduler_factor': 0.5,
}

PINN_POINTWISE_ARCH = {
    'unmeasured_net_config': {
        'hidden_layers': [128, 128],
        'activation': 'elu',
        'dropout_rate': 0.24900923127192412,
        'init_method': 'xavier_uniform'
    },
    'acceleration_net_config': {
        'hidden_layers': [128, 128],
        'activation': 'elu',
        'dropout_rate': 0.24900923127192412,
        'init_method': 'xavier_uniform'
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
    'early_stop_patience': 25,
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
