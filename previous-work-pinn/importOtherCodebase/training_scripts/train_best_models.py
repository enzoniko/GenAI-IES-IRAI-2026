#!/usr/bin/env python3
"""
Best Models Training Script

This script trains the best configurations for all 8 PINN methods using the hyperparameters
from best_trial_parameters.json:

PINN Methods:
- relobralo, constant_weight, brdr, pecann (originally supported)
- adaptive_lbpin, alpinn, dwpinn, gradnorm (Bayesian optimization winners)

The script trains all methods on the same data samples and saves the trained models
along with their scaling parameters for easy recovery in other scripts.

Usage:
    python train_best_models.py [--data-path Data] [--max-samples 10000] [--output-dir results/best_models]
                               [--synthetic] [--simulation-id SIM_ID] [--methods relobralo brdr ...]
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import time
import random

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.basicPINNv8 import ConfigurablePINN, adaptive_custom_loss, get_default_pinn_config
PINN_VERSION = "dimensional"
print("Using original dimensional PINN")

# Import training modules
from training_scripts.relobralo_training import ReLoBRaLoLoss, add_method_arguments as add_relobralo_args
from training_scripts.constant_weight_pinn_training import ConstantWeightLoss, add_method_arguments as add_constant_weight_args
from training_scripts.brdr_training import BRDRLoss, add_method_arguments as add_brdr_args
from training_scripts.pecann_training import PECANNLoss, add_method_arguments as add_pecann_args
from training_scripts.adaptive_lbpin_training import GaussianLikelihoodLoss, add_method_arguments as add_adaptive_lbpin_args
from training_scripts.alpinn_training import ALPINNLoss, add_method_arguments as add_alpinn_args
from training_scripts.dwpinn_training import DWPINNLoss, add_method_arguments as add_dwpinn_args
from training_scripts.gradnorm_training import GradNormLoss, add_method_arguments as add_gradnorm_args

from common_utils import (
    MultiLossEarlyStopping, ReduceLROnPlateau,
    get_pinn_config, get_param_init_config, setup_device, create_output_directory,
    plot_training_history, add_common_arguments, parse_common_arguments, print_training_config,
    collect_raw_residuals, collect_model_parameters,
    initialize_comprehensive_history, update_comprehensive_history,
    unified_data_preparation, ensure_double_precision, verify_data_structure, plot_trajectory_predictions
)


def load_best_parameters(json_path: str) -> dict:
    """Load best trial parameters from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def get_method_config(method_name: str, best_params: dict, data_config: dict) -> dict:
    """Extract configuration for a specific method from best parameters."""
    if method_name not in best_params:
        raise ValueError(f"Method {method_name} not found in best parameters")

    trial = best_params[method_name]

    # Base configuration from common arguments
    config = data_config.copy()

    # Override with method-specific parameters
    method_params = trial['method_params']
    arch_params = trial['architecture_params']
    train_params = trial['training_params']

    # Update architecture parameters
    config.update({
        'hidden_layers': arch_params['hidden_layers'],
        'activation': arch_params['activation'],
        'dropout_rate': arch_params['dropout_rate'],
        'init_method': arch_params['init_method'],
        'learning_rate': train_params['learning_rate'],
        'batch_size': train_params['batch_size'],
        'early_patience': train_params['early_patience'],
        'lr_patience': train_params['lr_patience']
    })

    return config, method_params


def create_loss_method(method_name: str, method_params: dict, enable_mass_constraints: bool = True):
    """Create the appropriate loss method based on method name."""
    if method_name == 'relobralo':
        return ReLoBRaLoLoss(enable_mass_constraints=enable_mass_constraints, **method_params)
    elif method_name == 'constant_weight':
        # For constant weight, we need to handle the weights specially
        weights = {}
        for key, value in method_params.items():
            if key.startswith('w_'):
                # Convert w_data -> data, w_res1 -> phys_res1, etc.
                component_key = key[2:]  # Remove 'w_' prefix
                if component_key == 'data':
                    weights['data'] = value
                elif component_key.startswith('res'):
                    weights[f'phys_res{component_key[3:]}'] = value  # w_res1 -> phys_res1
                elif component_key.startswith('mass'):
                    weights[f'phys_mass{component_key[4:]}'] = value  # w_mass1 -> phys_mass1
        return ConstantWeightLoss(enable_mass_constraints=enable_mass_constraints, weights=weights)
    elif method_name == 'brdr':
        return BRDRLoss(enable_mass_constraints=enable_mass_constraints, **method_params)
    elif method_name == 'pecann':
        return PECANNLoss(enable_mass_constraints=enable_mass_constraints, **method_params)
    elif method_name == 'adaptive_lbpin':
        return GaussianLikelihoodLoss(enable_mass_constraints=enable_mass_constraints, **method_params)
    elif method_name == 'alpinn':
        return ALPINNLoss(enable_mass_constraints=enable_mass_constraints, **method_params)
    elif method_name == 'dwpinn':
        return DWPINNLoss(enable_mass_constraints=enable_mass_constraints, **method_params)
    elif method_name == 'gradnorm':
        return GradNormLoss(enable_mass_constraints=enable_mass_constraints, **method_params)
    else:
        raise ValueError(f"Unknown method: {method_name}")


def train_single_method(method_name: str, config: dict, method_params: dict, data: dict,
                       device: torch.device, output_dir: str):
    """Train a single method and save results."""
    print(f"\n{'='*60}")
    print(f"Training {method_name.upper()}")
    print(f"{'='*60}")

    # Create PINN configuration
    pinn_config = get_pinn_config(
        config['hidden_layers'], config['activation'],
        config['dropout_rate'], config['init_method']
    )
    param_init_config = get_param_init_config(synthetic=config.get('synthetic', False))

    # Initialize model
    if PINN_VERSION == "v2":
        model = ConfigurablePINN(
            unmeasured_net_config=pinn_config,
            acceleration_net_config=pinn_config,
            rotor_net_config=pinn_config,
            param_init_config=param_init_config,
            enable_mass_constraints=not config.get('synthetic', False)
        ).to(device)
    elif PINN_VERSION == "v1":
        model = ConfigurablePINN(
            unmeasured_net_config=pinn_config,
            acceleration_net_config=pinn_config,
            param_init_config=param_init_config,
            enable_mass_constraints=not config.get('synthetic', False)
        ).to(device)
    else:
        model = ConfigurablePINN(pinn_config, pinn_config, param_init_config,
                                enable_mass_constraints=not config.get('synthetic', False)).to(device)

    # Create loss method
    loss_method = create_loss_method(method_name, method_params,
                                    enable_mass_constraints=not config.get('synthetic', False))

    # Setup optimizer and schedulers
    opt = optim.Adam(model.parameters(), lr=config['learning_rate'])
    lr_sched = ReduceLROnPlateau(opt, patience=config['lr_patience'])
    early = MultiLossEarlyStopping(config['early_patience'], config.get('min_delta', 1e-7),
                                  ['data_val', 'val_total'])

    # Initialize history
    hist = initialize_comprehensive_history(synthetic=config.get('synthetic', False))

    # Add method-specific tracking
    if method_name == 'relobralo':
        hist.update({
            'weight_data': [], 'weight_phys_res1': [], 'weight_phys_res2': [],
            'weight_phys_res3': [], 'weight_phys_res4': [], 'weight_phys_mass1': [],
            'weight_phys_mass2': []
        })
    elif method_name == 'constant_weight':
        hist.update({
            'weight_data': [], 'weight_phys_res1': [], 'weight_phys_res2': [],
            'weight_phys_res3': [], 'weight_phys_res4': [], 'weight_phys_mass1': [],
            'weight_phys_mass2': []
        })
    elif method_name == 'brdr':
        hist.update({
            'weight_data': [], 'weight_phys_res1': [], 'weight_phys_res2': [],
            'weight_phys_res3': [], 'weight_phys_res4': [], 'weight_phys_mass1': [],
            'weight_phys_mass2': [], 'scale_factor_s': []
        })
    elif method_name == 'pecann':
        hist.update({
            'weight_data': [], 'weight_phys_res1': [], 'weight_phys_res2': [],
            'weight_phys_res3': [], 'weight_phys_res4': [], 'weight_phys_mass1': [],
            'weight_phys_mass2': [], 'mu': [], 'lambda_data_mean': [], 'lambda_physics_mean': [],
            'constraint_violation': []
        })
    elif method_name == 'adaptive_lbpin':
        hist.update({
            'sigma_data': [], 'sigma_phys': [],
            'sigma_phys_res1': [], 'sigma_phys_res2': [], 'sigma_phys_res3': [],
            'sigma_phys_res4': []
        })
        if not config.get('synthetic', False):
            hist.update({
                'sigma_phys_mass1': [], 'sigma_phys_mass2': []
            })
    elif method_name == 'alpinn':
        hist.update({
            'lambda_data': [], 'lambda_phys_res1': [], 'lambda_phys_res2': [],
            'lambda_phys_res3': [], 'lambda_phys_res4': []
        })
        if not config.get('synthetic', False):
            hist.update({
                'lambda_phys_mass1': [], 'lambda_phys_mass2': []
            })
    elif method_name == 'dwpinn':
        hist.update({
            'weight_data': [], 'weight_phys_res1': [], 'weight_phys_res2': [],
            'weight_phys_res3': [], 'weight_phys_res4': [], 'weight_phys_mass1': [],
            'weight_phys_mass2': []
        })
    elif method_name == 'gradnorm':
        hist.update({
            'weight_data': [], 'weight_phys_res1': [], 'weight_phys_res2': [],
            'weight_phys_res3': [], 'weight_phys_res4': [], 'weight_phys_mass1': [],
            'weight_phys_mass2': [], 'training_phase': []
        })

    # Training loop
    print(f"Starting {method_name} training...")
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in tqdm(range(config['epochs']), desc=f"Training {method_name}"):
        # Training step
        model.train()
        tr_total = tr_data = 0.0
        num_phys_components = 4 if config.get('synthetic', False) else 6
        tr_phys_components = [0.0] * num_phys_components
        tr_raw_residuals = []

        for xb, yb in data['train']:
            xb, yb = xb.to(device), yb.to(device)
            xb = xb.double()
            yb = yb.double()

            y_pred = model(xb)
            loss_components = loss_method(model, xb, yb, data['Xmax'].to(device),
                                        data['Xmin'].to(device), data['ymax'].to(device),
                                        data['ymin'].to(device))
            total = loss_method.step(loss_components, opt, model, xb, yb)

            raw_residuals = collect_raw_residuals(model, xb, yb, y_pred, data, device)
            tr_raw_residuals.append(raw_residuals)

            tr_total += total.item()
            tr_data += loss_components[0].item()
            for i in range(num_phys_components):
                tr_phys_components[i] += loss_components[i+1].item()

        # Average training metrics
        n_train = len(data['train'])
        epoch_metrics = {
            'train_total': tr_total / n_train,
            'data_train': tr_data / n_train,
            'phys_train': sum(tr_phys_components) / (num_phys_components * n_train)
        }

        phys_keys = ['phys_res1', 'phys_res2', 'phys_res3', 'phys_res4']
        if not config.get('synthetic', False):
            phys_keys.extend(['phys_mass1', 'phys_mass2'])
        for i, key in enumerate(phys_keys):
            epoch_metrics[f'{key}_train'] = tr_phys_components[i] / n_train

        # Average raw residuals
        avg_raw_residuals = {'data_residuals': {}, 'physical_residuals': {}}
        for key in ['x2_ddot_mae', 'y2_ddot_mae', 'x3_ddot_mae', 'y3_ddot_mae', 'total_mae']:
            values = [res['data_residuals'][key] for res in tr_raw_residuals]
            avg_raw_residuals['data_residuals'][key] = np.mean(values)

        phys_residual_keys = ['res1_mean', 'res2_mean', 'res3_mean', 'res4_mean']
        if not config.get('synthetic', False):
            phys_residual_keys.extend(['res_mass1_mean', 'res_mass2_mean'])
        for key in phys_residual_keys:
            values = [res['physical_residuals'][key] for res in tr_raw_residuals]
            avg_raw_residuals['physical_residuals'][key] = np.mean(values)

        phys_residual_std_keys = ['res1_std', 'res2_std', 'res3_std', 'res4_std']
        if not config.get('synthetic', False):
            phys_residual_std_keys.extend(['res_mass1_std', 'res_mass2_std'])
        for key in phys_residual_std_keys:
            values = [res['physical_residuals'][key] for res in tr_raw_residuals]
            avg_raw_residuals['physical_residuals'][key] = np.mean(values)

        epoch_metrics['raw_residuals'] = avg_raw_residuals
        update_comprehensive_history(hist, epoch_metrics, model, is_training=True)

        # Store method-specific weights (only for methods that actually use adaptive weights)
        if hasattr(loss_method, 'current_weights') and method_name not in ['adaptive_lbpin', 'alpinn']:
            for key, value in loss_method.current_weights.items():
                hist[f'weight_{key}'].append(value)

        # Store additional method-specific parameters
        if method_name == 'pecann':
            hist['mu'].append(loss_method.mu)
            hist['lambda_data_mean'].append(loss_method.mu)
            hist['lambda_physics_mean'].append(loss_method.eta)
            hist['constraint_violation'].append(loss_method.eta)
        elif method_name == 'brdr':
            avg_weight = np.mean(list(loss_method.current_weights.values()))
            hist['scale_factor_s'].append(avg_weight)
        elif method_name == 'adaptive_lbpin':
            # Store sigma values using the current_sigmas method
            sigma_data, sigma_phys_components = loss_method.current_sigmas()
            hist['sigma_data'].append(sigma_data.item())
            hist['sigma_phys'].append(sum(c.item() for c in sigma_phys_components) / len(sigma_phys_components))
            # Store individual sigma components
            sigma_names = ['sigma_phys_res1', 'sigma_phys_res2', 'sigma_phys_res3', 'sigma_phys_res4']
            if len(sigma_phys_components) > 4:  # Has mass constraints
                sigma_names.extend(['sigma_phys_mass1', 'sigma_phys_mass2'])
            for i, name in enumerate(sigma_names):
                if i < len(sigma_phys_components):
                    hist[name].append(sigma_phys_components[i].item())
        elif method_name == 'alpinn':
            # Store lambda values
            lambda_components = [loss_method.lambda_data, loss_method.lambda_phys_res1,
                               loss_method.lambda_phys_res2, loss_method.lambda_phys_res3,
                               loss_method.lambda_phys_res4]
            if hasattr(loss_method, 'lambda_phys_mass1'):
                lambda_components.extend([loss_method.lambda_phys_mass1, loss_method.lambda_phys_mass2])
            lambda_names = ['lambda_data', 'lambda_phys_res1', 'lambda_phys_res2',
                          'lambda_phys_res3', 'lambda_phys_res4']
            if hasattr(loss_method, 'lambda_phys_mass1'):
                lambda_names.extend(['lambda_phys_mass1', 'lambda_phys_mass2'])
            for name, value in zip(lambda_names, lambda_components):
                hist[name].append(value.item())
        elif method_name == 'gradnorm':
            hist['training_phase'].append(getattr(loss_method, 'current_phase', 'unknown'))

        # Validation step
        model.eval()
        val_total = val_data = 0.0
        val_phys_components = [0.0] * num_phys_components
        val_raw_residuals = []

        with torch.no_grad():
            for xb, yb in data['val']:
                xb, yb = xb.to(device), yb.to(device)
                xb = xb.double()
                yb = yb.double()

                y_pred = model(xb)
                loss_components = loss_method(model, xb, yb, data['Xmax'].to(device),
                                            data['Xmin'].to(device), data['ymax'].to(device),
                                            data['ymin'].to(device))

                # Calculate validation loss
                if method_name == 'constant_weight':
                    data_loss_val = loss_components[0]
                    phys_losses_val = torch.stack(loss_components[1:])
                    weighted_data = method_params['w_data'] * data_loss_val
                    physics_weights = []
                    for i, key in enumerate(phys_keys):
                        weight_key = f'w_{key[5:]}' if key.startswith('phys_res') else f'w_mass{key[9:]}' if key.startswith('phys_mass') else key
                        physics_weights.append(method_params.get(weight_key, 1.0))
                    physics_weights = torch.tensor(physics_weights, dtype=phys_losses_val.dtype, device=phys_losses_val.device)
                    sum_w = torch.clamp(physics_weights.sum(), min=1e-12)
                    weighted_phys_mean = (phys_losses_val * physics_weights).sum() / sum_w
                    weighted_total = weighted_data + weighted_phys_mean
                elif method_name in ['relobralo', 'brdr']:
                    if method_name == 'relobralo':
                        # Use running weights for ReLoBRaLo validation
                        if hasattr(loss_method, 'running_weights') and loss_method.running_weights is not None:
                            weights = loss_method.running_weights
                        else:
                            weights = torch.ones(len(loss_components), dtype=loss_components[0].dtype, device=loss_components[0].device)
                    else:  # brdr
                        # For BRDR validation, use the current weights from the loss method
                        # These should be scalar values, not point-wise weights
                        if hasattr(loss_method, 'current_weights') and loss_method.current_weights:
                            weights = [loss_method.current_weights.get(key, 1.0) for key in loss_method.loss_keys]
                        else:
                            weights = [1.0] * len(loss_components)
                        weights = torch.tensor(weights, dtype=loss_components[0].dtype, device=loss_components[0].device)
                    weighted_total = torch.sum(torch.stack([weights[i] * loss_components[i] for i in range(len(loss_components))]))
                elif method_name == 'pecann':
                    try:
                        # Reconstruct ALM loss for validation
                        constraint_residuals = loss_method._get_constraint_residuals(model, xb, yb)
                        objective_loss = loss_components[0]
                        penalty_term = 0.0
                        for key in loss_method.loss_keys:
                            residual = constraint_residuals[key]
                            # Ensure residual is properly shaped for mean computation
                            if residual.dim() > 0:
                                penalty_term += torch.mean(residual**2)
                            else:
                                penalty_term += residual**2
                        penalty_term = (loss_method.mu / 2) * penalty_term
                        multiplier_term = 0.0
                        for key in loss_method.loss_keys:
                            residual = constraint_residuals[key]
                            # Ensure residual is properly shaped for mean computation
                            if residual.dim() > 0:
                                multiplier_term += torch.mean(loss_method.lambdas[key] * residual)
                            else:
                                multiplier_term += loss_method.lambdas[key] * residual
                        weighted_total = objective_loss + penalty_term + multiplier_term
                    except Exception as e:
                        print(f"Warning: PECANN validation reconstruction failed: {e}")
                        # Fallback to simple weighted sum
                        weights = torch.ones(len(loss_components), dtype=loss_components[0].dtype, device=loss_components[0].device)
                        weighted_total = torch.sum(torch.stack([weights[i] * loss_components[i] for i in range(len(loss_components))]))
                else:
                    # For other methods, use their current weights if available
                    if hasattr(loss_method, 'current_weights'):
                        weights = [loss_method.current_weights.get(key, 1.0) for key in loss_method.loss_keys]
                        weights = torch.tensor(weights, dtype=loss_components[0].dtype, device=loss_components[0].device)
                        weighted_total = torch.sum(weights * torch.stack(loss_components))
                    else:
                        # Fallback to simple sum
                        weighted_total = torch.sum(torch.stack(loss_components))

                val_total += weighted_total.item()
                val_data += loss_components[0].item()
                for i in range(num_phys_components):
                    val_phys_components[i] += loss_components[i+1].item()

                raw_residuals = collect_raw_residuals(model, xb, yb, y_pred, data, device)
                val_raw_residuals.append(raw_residuals)

        # Average validation metrics
        n_val = len(data['val'])
        val_epoch_metrics = {
            'val_total': val_total / n_val,
            'data_val': val_data / n_val,
            'phys_val': sum(val_phys_components) / (num_phys_components * n_val)
        }

        for i, key in enumerate(phys_keys):
            val_epoch_metrics[f'{key}_val'] = val_phys_components[i] / n_val

        val_avg_raw_residuals = {'data_residuals': {}, 'physical_residuals': {}}
        for key in ['x2_ddot_mae', 'y2_ddot_mae', 'x3_ddot_mae', 'y3_ddot_mae', 'total_mae']:
            values = [res['data_residuals'][key] for res in val_raw_residuals]
            val_avg_raw_residuals['data_residuals'][key] = np.mean(values)

        for key in phys_residual_keys:
            values = [res['physical_residuals'][key] for res in val_raw_residuals]
            val_avg_raw_residuals['physical_residuals'][key] = np.mean(values)

        for key in phys_residual_std_keys:
            values = [res['physical_residuals'][key] for res in val_raw_residuals]
            val_avg_raw_residuals['physical_residuals'][key] = np.mean(values)

        val_epoch_metrics['raw_residuals'] = val_avg_raw_residuals
        update_comprehensive_history(hist, val_epoch_metrics, model, is_training=False)

        # Save best model
        current_val_loss = hist['val_total'][-1]
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_model_state = model.state_dict().copy()

        # Learning rate scheduling and early stopping
        lr_sched.step(current_val_loss)
        early({'data_val': hist['data_val'][-1], 'val_total': hist['val_total'][-1]})
        if early.early_stop:
            print(f"Early stopping at epoch {epoch}")
            break

    # Test set evaluation
    print(f"\n{'='*50}")
    print(f"TEST SET EVALUATION FOR {method_name.upper()}")
    print(f"{'='*50}")

    if 'test' in data:
        model.eval()
        test_total = test_data = 0.0
        test_phys_components = [0.0] * num_phys_components
        test_raw_residuals = []

        with torch.no_grad():
            for xb, yb in data['test']:
                xb, yb = xb.to(device), yb.to(device)
                xb = xb.double()
                yb = yb.double()

                y_pred = model(xb)
                loss_components = loss_method(model, xb, yb, data['Xmax'].to(device),
                                           data['Xmin'].to(device), data['ymax'].to(device),
                                           data['ymin'].to(device))

                # Calculate total loss based on loss method type
                try:
                    method_name = type(loss_method).__name__
                    if hasattr(loss_method, 'running_weights') and loss_method.running_weights is not None:
                        # ReLoBRaLo and properly initialized BRDR have running_weights
                        if method_name == 'BRDRLoss':
                            # For BRDR, use the current weights from the loss method
                            if hasattr(loss_method, 'current_weights'):
                                weights = [loss_method.current_weights.get(key, 1.0) for key in loss_method.loss_keys]
                                weights_tensor = torch.tensor(weights, dtype=loss_components[0].dtype, device=loss_components[0].device)
                                weighted_total = torch.sum(weights_tensor * torch.stack(loss_components))
                            else:
                                weighted_total = torch.sum(torch.stack(loss_components))
                        else:
                            # ReLoBRaLo
                            weighted_total = torch.sum(torch.stack([loss_method.running_weights[i] * loss_components[i] for i in range(len(loss_components))]))
                    elif hasattr(loss_method, 'current_weights'):
                        # ConstantWeightLoss and PECANNLoss have current_weights
                        weights = [loss_method.current_weights.get(key, 1.0) for key in loss_method.loss_keys]
                        weights_tensor = torch.tensor(weights, dtype=loss_components[0].dtype, device=loss_components[0].device)
                        weighted_total = torch.sum(weights_tensor * torch.stack(loss_components))
                    else:
                        # Fallback: simple sum
                        weighted_total = torch.sum(torch.stack(loss_components))
                except (AttributeError, TypeError) as e:
                    # If anything goes wrong with weights, fall back to simple sum
                    method_name = type(loss_method).__name__
                    print(f"Warning: Weight calculation failed for {method_name} ({e}), using simple sum")
                    weighted_total = torch.sum(torch.stack(loss_components))

                test_total += weighted_total.item()
                test_data += loss_components[0].item()
                for i in range(num_phys_components):
                    test_phys_components[i] += loss_components[i+1].item()

                raw_residuals = collect_raw_residuals(model, xb, yb, y_pred, data, device)
                test_raw_residuals.append(raw_residuals)

        # Average test metrics
        n_test = len(data['test'])
        test_metrics = {
            'test_total': test_total / n_test,
            'data_test': test_data / n_test,
            'phys_test': sum(test_phys_components) / (num_phys_components * n_test)
        }

        # Define phys_keys for test evaluation (same as in training)
        phys_keys = ['phys_res1', 'phys_res2', 'phys_res3', 'phys_res4']
        if not config.get('synthetic', False):  # Assuming synthetic is stored in config
            phys_keys.extend(['phys_mass1', 'phys_mass2'])

        for i, key in enumerate(phys_keys):
            test_metrics[f'{key}_test'] = test_phys_components[i] / n_test

        # Average test raw residuals
        test_avg_raw_residuals = {'data_residuals': {}, 'physical_residuals': {}}
        for key in ['x2_ddot_mae', 'y2_ddot_mae', 'x3_ddot_mae', 'y3_ddot_mae', 'total_mae']:
            values = [res['data_residuals'][key] for res in test_raw_residuals]
            test_avg_raw_residuals['data_residuals'][key] = np.mean(values)

        # Define phys_residual_keys for test evaluation
        phys_residual_keys = ['res1_mean', 'res2_mean', 'res3_mean', 'res4_mean']
        if not config.get('synthetic', False):
            phys_residual_keys.extend(['res_mass1_mean', 'res_mass2_mean'])

        for key in phys_residual_keys:
            values = [res['physical_residuals'][key] for res in test_raw_residuals]
            test_avg_raw_residuals['physical_residuals'][key] = np.mean(values)

        # Define phys_residual_std_keys for test evaluation
        phys_residual_std_keys = ['res1_std', 'res2_std', 'res3_std', 'res4_std']
        if not config.get('synthetic', False):
            phys_residual_std_keys.extend(['res_mass1_std', 'res_mass2_std'])

        for key in phys_residual_std_keys:
            values = [res['physical_residuals'][key] for res in test_raw_residuals]
            test_avg_raw_residuals['physical_residuals'][key] = np.mean(values)

        print("Test Loss Components:")
        print(f"Test Total: {test_metrics['test_total']:.6f}")
        print(f"Data Test: {test_metrics['data_test']:.6f}")
        print(f"Physics Test: {test_metrics['phys_test']:.6f}")
        for key in phys_keys:
            if key in test_metrics:
                print(f"{key}_test: {test_metrics[key + '_test']:.6f}")

        # Check for potential issues
        if test_metrics['test_total'] > 1000 or np.isnan(test_metrics['test_total']):
            print("WARNING: Test total loss is unusually high or NaN!")
        if test_metrics['data_test'] > 1000 or np.isnan(test_metrics['data_test']):
            print("WARNING: Test data loss is unusually high or NaN!")

        print("\nTest Data Residuals (MAE):")
        for key, value in test_avg_raw_residuals['data_residuals'].items():
            if value > 1000 or np.isnan(value):
                print(f"{key}: {value:.6f} ⚠️")
            else:
                print(f"{key}: {value:.6f}")

        print("\nTest Physics Residuals (Mean):")
        for key in phys_residual_keys:
            if key in test_avg_raw_residuals['physical_residuals']:
                value = test_avg_raw_residuals['physical_residuals'][key]
                if abs(value) > 1000 or np.isnan(value):
                    print(f"{key}: {value:.6f} ⚠️")
                else:
                    print(f"{key}: {value:.6f}")

        print("\nTest Physics Residuals (Std):")
        for key in phys_residual_std_keys:
            if key in test_avg_raw_residuals['physical_residuals']:
                value = test_avg_raw_residuals['physical_residuals'][key]
                if abs(value) > 1000 or np.isnan(value):
                    print(f"{key}: {value:.6f} ⚠️")
                else:
                    print(f"{key}: {value:.6f}")

        # Update history with test metrics
        update_comprehensive_history(hist, test_metrics, model, is_training=False)
        update_comprehensive_history(hist, {'raw_residuals': test_avg_raw_residuals}, model, is_training=False)

    else:
        print("No test set available for evaluation")

    # Save results
    method_dir = os.path.join(output_dir, method_name)
    os.makedirs(method_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(method_dir, f"{method_name}_best_model.pth")
    torch.save(best_model_state, model_path)

    # Save training history
    np.savez(os.path.join(method_dir, f"{method_name}_history.npz"), **hist)

    # Save scaling parameters
    scaling_params = {
        'Xmin': data['Xmin'].cpu().numpy(),
        'Xmax': data['Xmax'].cpu().numpy(),
        'ymin': data['ymin'].cpu().numpy(),
        'ymax': data['ymax'].cpu().numpy()
    }
    np.savez(os.path.join(method_dir, f"{method_name}_scaling.npz"), **scaling_params)

    # Save final parameters
    model.load_state_dict(best_model_state)
    final_params = collect_model_parameters(model)
    np.savez(os.path.join(method_dir, f"{method_name}_parameters.npz"), **final_params)

    # Save configuration
    config_to_save = {
        'method_name': method_name,
        'method_params': method_params,
        'architecture_params': {
            'hidden_layers': config['hidden_layers'],
            'activation': config['activation'],
            'dropout_rate': config['dropout_rate'],
            'init_method': config['init_method']
        },
        'training_params': {
            'learning_rate': config['learning_rate'],
            'batch_size': config['batch_size'],
            'early_patience': config['early_patience'],
            'lr_patience': config['lr_patience'],
            'epochs': config['epochs']
        },
        'data_params': {
            'max_samples': config.get('max_samples'),
            'synthetic': config.get('synthetic', False),
            'simulation_id': config.get('simulation_id')
        },
        'best_val_loss': best_val_loss,
        'final_params': final_params
    }

    with open(os.path.join(method_dir, f"{method_name}_config.json"), 'w') as f:
        json.dump(config_to_save, f, indent=2, default=str)

    print(f"{method_name.upper()} training complete!")
    print(f"  - Best validation loss: {best_val_loss:.6f}")
    print(f"  - Model saved to: {model_path}")
    print(f"  - Results saved to: {method_dir}")

    return best_val_loss


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Best Models')
    parser = add_common_arguments(parser)

    # Data arguments
    parser.add_argument('--synthetic', action='store_true',
                       help='Use synthetic data instead of experimental data')
    parser.add_argument('--simulation-id', type=int, default=None,
                       help='Simulation ID to use (required if --synthetic is used)')
    parser.add_argument('--data-path', type=str, default='Data',
                       help='Path to data directory (default: Data)')

    # Output arguments
    parser.add_argument('--methods', nargs='+',
                       choices=['relobralo', 'constant_weight', 'brdr', 'pecann', 'adaptive_lbpin', 'alpinn', 'dwpinn', 'gradnorm'],
                       default=['relobralo', 'constant_weight', 'brdr', 'pecann', 'adaptive_lbpin', 'alpinn', 'dwpinn', 'gradnorm'],
                       help='Methods to train (default: all 8 PINN methods)')

    args = parser.parse_args()

    # Set default output directory if not provided
    if args.output_dir is None:
        args.output_dir = 'results/best_models'

    # Load best parameters
    best_params_path = "training_scripts/best_trial_parameters.json"
    if not os.path.exists(best_params_path):
        raise FileNotFoundError(f"Best parameters file not found: {best_params_path}")

    best_params = load_best_parameters(best_params_path)
    print(f"Loaded best parameters for {len(best_params)} methods")

    # Setup device
    device = setup_device(args.device)

    # Data configuration
    data_config = {
        'max_samples': args.max_samples,
        'synthetic': args.synthetic,
        'simulation_id': args.simulation_id,
        'data_path': args.data_path,
        'batch_size': 256,  # Will be overridden by method-specific batch size
        'epochs': 10000,    # Will be overridden by method-specific epochs
        'device': args.device
    }

    # Load and prepare data once (all methods use same data)
    print("Loading and preparing data...")
    data = unified_data_preparation(
        data_path=args.data_path,
        max_samples=args.max_samples,
        synthetic=args.synthetic,
        simulation_id=args.simulation_id,
        batch_size=256  # Temporary batch size, will be overridden
    )
    data = ensure_double_precision(data)

    # Verify data structure
    sample_batch = next(iter(data['train']))
    X_sample, y_sample = sample_batch
    verify_data_structure(X_sample, y_sample, synthetic=args.synthetic)

    # Split validation set in half to create test set
    print("Splitting validation set for test evaluation...")
    val_dataset = data['val'].dataset
    val_size = len(val_dataset)
    test_size = val_size // 2
    val_size = val_size - test_size

    from torch.utils.data import random_split, DataLoader
    val_dataset_split, test_dataset = random_split(
        val_dataset,
        [val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )

    # Update data loaders
    data['val'] = DataLoader(val_dataset_split, batch_size=256, shuffle=False)
    data['test'] = DataLoader(test_dataset, batch_size=256, shuffle=False)

    print(f"Data loaded successfully:")
    print(f"  - Training batches: {len(data['train'])}")
    print(f"  - Validation batches: {len(data['val'])}")
    print(f"  - Test batches: {len(data['test'])}")
    print(f"  - Input features: {X_sample.size(-1)}")
    print(f"  - Output features: {y_sample.size(-1)}")

    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Train each method
    results = {}
    methods_to_train = args.methods

    for method_name in methods_to_train:
        if method_name not in best_params:
            print(f"Warning: Method {method_name} not found in best parameters, skipping...")
            continue

        try:
            # Get method configuration
            config, method_params = get_method_config(method_name, best_params, data_config)

            # Create new data loaders with method-specific batch size and drop_last=True
            # This ensures all batches have exactly the same size, preventing tensor resizing issues
            # Set random seed to ensure reproducible data splits across methods
            torch.manual_seed(42)
            np.random.seed(42)

            data_method = unified_data_preparation(
                data_path=args.data_path,
                max_samples=args.max_samples,
                synthetic=args.synthetic,
                simulation_id=args.simulation_id,
                batch_size=config['batch_size']  # Use method-specific batch size
            )
            data_method = ensure_double_precision(data_method)

            # Split validation set in half to create test set for this method
            val_dataset_method = data_method['val'].dataset
            val_size_method = len(val_dataset_method)
            test_size_method = val_size_method // 2
            val_size_method = val_size_method - test_size_method

            val_dataset_split_method, test_dataset_method = random_split(
                val_dataset_method,
                [val_size_method, test_size_method],
                generator=torch.Generator().manual_seed(42)
            )

            # Update data loaders for this method (using original batch size first)
            data_method['val'] = DataLoader(val_dataset_split_method, batch_size=config['batch_size'], shuffle=False)
            data_method['test'] = DataLoader(test_dataset_method, batch_size=config['batch_size'], shuffle=False)

            # Now apply batch size adjustments for consistency
            train_size = len(data_method['train'].dataset)
            val_size = len(data_method['val'].dataset)
            test_size = len(data_method['test'].dataset)

            # Apply method-specific batch size constraints for stable training
            if method_name == 'brdr':
                # BRDR needs sufficient samples per batch for stable statistics
                min_batch_size = max(16, min(config['batch_size'], val_size // 2, train_size // 2, test_size // 2))
                effective_batch_size = min(config['batch_size'], val_size, train_size, test_size, min_batch_size)
            elif method_name in ['relobralo', 'pecann']:
                # These methods benefit from larger batches for stable gradient computation
                min_batch_size = max(8, min(config['batch_size'], val_size // 4, train_size // 4, test_size // 4))
                effective_batch_size = min(config['batch_size'], val_size, train_size, test_size, min_batch_size)
            else:
                # For constant_weight and other methods, ensure at least reasonable batch size
                min_batch_size = max(4, min(config['batch_size'], val_size // 8, train_size // 8, test_size // 8))
                effective_batch_size = min(config['batch_size'], val_size, train_size, test_size, min_batch_size)

            print(f"Adjusting batch size from {config['batch_size']} to {effective_batch_size} for compatibility")
            print(f"Train dataset size: {train_size}, Val dataset size: {val_size}, Test dataset size: {test_size}")

            # Ensure we have at least 1 validation batch
            if val_size < effective_batch_size:
                effective_batch_size = val_size
                print(f"Further reducing batch size to {effective_batch_size} to ensure validation batch")

            data_method['train'] = DataLoader(
                data_method['train'].dataset,
                batch_size=effective_batch_size,
                shuffle=True,
                drop_last=True  # This prevents smaller last batches
            )
            data_method['val'] = DataLoader(
                data_method['val'].dataset,
                batch_size=effective_batch_size,
                shuffle=False,
                drop_last=True  # This prevents smaller last batches
            )
            data_method['test'] = DataLoader(
                data_method['test'].dataset,
                batch_size=effective_batch_size,
                shuffle=False,
                drop_last=True  # This prevents smaller last batches
            )

            print(f"Final batch counts - Train: {len(data_method['train'])}, Val: {len(data_method['val'])}, Test: {len(data_method['test'])}")

            # Additional safety check for all methods
            if len(data_method['train']) == 0 or len(data_method['val']) == 0 or len(data_method['test']) == 0:
                raise ValueError(f"{method_name.upper()} method requires at least one training, validation, and test batch. "
                        f"Train size: {train_size}, Val size: {val_size}, Test size: {test_size}, Batch size: {effective_batch_size}")

            # Verify batch size consistency across all batches for all methods
            print(f"Verifying batch size consistency for {method_name.upper()}...")
            train_batch_sizes = []
            val_batch_sizes = []
            test_batch_sizes = []

            # Check training batches
            for xb, yb in data_method['train']:
                train_batch_sizes.append(xb.shape[0])
                break  # Just check the first batch since drop_last=True ensures all are same size

            # Check validation batches
            for xb, yb in data_method['val']:
                val_batch_sizes.append(xb.shape[0])
                break  # Just check the first batch since drop_last=True ensures all are same size

            # Check test batches
            for xb, yb in data_method['test']:
                test_batch_sizes.append(xb.shape[0])
                break  # Just check the first batch since drop_last=True ensures all are same size

            print(f"Training batch size: {train_batch_sizes[0] if train_batch_sizes else 0}")
            print(f"Validation batch size: {val_batch_sizes[0] if val_batch_sizes else 0}")
            print(f"Test batch size: {test_batch_sizes[0] if test_batch_sizes else 0}")

            # Check for batch size consistency
            batch_sizes = [size for size in [train_batch_sizes[0] if train_batch_sizes else None,
                                           val_batch_sizes[0] if val_batch_sizes else None,
                                           test_batch_sizes[0] if test_batch_sizes else None]
                          if size is not None]

            if len(set(batch_sizes)) > 1:
                print(f"Warning: Batch size mismatch detected for {method_name.upper()}! Sizes: {batch_sizes}")
                # Use the smallest batch size for consistency
                min_batch_size = min(batch_sizes)
                print(f"Adjusting all loaders to use batch size {min_batch_size}")

                data_method['train'] = DataLoader(
                    data_method['train'].dataset,
                    batch_size=min_batch_size,
                    shuffle=True,
                    drop_last=True
                )
                data_method['val'] = DataLoader(
                    data_method['val'].dataset,
                    batch_size=min_batch_size,
                    shuffle=False,
                    drop_last=True
                )
                data_method['test'] = DataLoader(
                    data_method['test'].dataset,
                    batch_size=min_batch_size,
                    shuffle=False,
                    drop_last=True
                )

            # Print configuration
            print_training_config(config, method_name.upper(), method_params)

            # Train the method
            best_val_loss = train_single_method(
                method_name, config, method_params, data_method, device, output_dir
            )

            results[method_name] = {
                'best_val_loss': best_val_loss,
                'config': config,
                'method_params': method_params
            }

        except Exception as e:
            print(f"Error training {method_name}: {e}")
            continue

    # Save summary
    summary = {
        'training_summary': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'methods_trained': list(results.keys()),
            'data_config': {
                'max_samples': args.max_samples,
                'synthetic': args.synthetic,
                'simulation_id': args.simulation_id
            },
            'results': results
        }
    }

    with open(os.path.join(output_dir, 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Print final summary
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE - SUMMARY")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir}")
    print(f"Methods trained: {list(results.keys())}")
    print("\nBest validation losses:")
    for method, result in results.items():
        print(".6f")
    print(f"\nAll models and scaling parameters saved for easy recovery!")
    print(f"Use the load_best_model() function from common_utils to load trained models.")


if __name__ == '__main__':
    main()
