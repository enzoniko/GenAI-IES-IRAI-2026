#!/usr/bin/env python3
"""
Best Models Validation Script

This script validates the best trained models for relobralo, constant_weight, brdr, and pecann
methods by testing them on healthy data samples to ensure they can replicate their training performance.

The script loads the best models from the optimization results and tests them using the same
normalization parameters calculated from the training data.

Usage:
    python validate_best_models.py [--data-path Data] [--max-samples 10000] [--test-split 0.3]
                               [--synthetic] [--simulation-id SIM_ID]
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm
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

from common_utils import (
    get_pinn_config, get_param_init_config, setup_device, create_output_directory,
    collect_raw_residuals, unified_data_preparation, ensure_double_precision, verify_data_structure
)


def load_best_parameters(json_path: str) -> dict:
    """Load best trial parameters from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def get_method_config(method_name: str, best_params: dict) -> tuple:
    """Extract configuration for a specific method from best parameters."""
    if method_name not in best_params:
        raise ValueError(f"Method {method_name} not found in best parameters")

    trial = best_params[method_name]

    # Get method-specific parameters
    method_params = trial['method_params']
    arch_params = trial['architecture_params']

    # Build configuration
    config = {
        'hidden_layers': arch_params['hidden_layers'],
        'activation': arch_params['activation'],
        'dropout_rate': arch_params['dropout_rate'],
        'init_method': arch_params['init_method'],
        'method_params': method_params
    }

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
    else:
        raise ValueError(f"Unknown method: {method_name}")


def find_best_model_path(method_name: str, trial_identifier: str) -> str:
    """Find the path to the best model for a given method and trial."""
    # Based on the JSON structure, all methods use starting_point_2
    model_path = f"bayesian_results/{method_name}/starting_point_2/results/{trial_identifier}/best_model.pth"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return model_path


def load_model_from_checkpoint(model_path: str, method_config: dict, device: torch.device):
    """Load the exact ConfigurablePINN model with the correct architecture."""
    try:
        # Create the exact ConfigurablePINN model with the correct architecture
        from models.basicPINNv8 import ConfigurablePINN

        # Create PINN configuration from method parameters
        pinn_config = get_pinn_config(
            method_config['hidden_layers'],
            method_config['activation'],
            method_config['dropout_rate'],
            method_config['init_method']
        )
        param_init_config = get_param_init_config(synthetic=False)  # Assume real data

        # Create the exact ConfigurablePINN model
        model = ConfigurablePINN(pinn_config, pinn_config, param_init_config,
                                enable_mass_constraints=True)  # Real data has mass constraints

        # Load the saved state dict
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)

        # Ensure model is in double precision and on correct device
        model = model.double().to(device)

        print(f"Successfully loaded ConfigurablePINN model with architecture: {method_config['hidden_layers']}")
        return model

    except Exception as e:
        print(f"Error loading ConfigurablePINN model: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_model(method_name: str, model: nn.Module, loss_method, data: dict, device: torch.device, test_samples: int = None, synthetic: bool = False, args=None):
    """Test a single model and return performance metrics."""
    print(f"\n{'='*50}")
    print(f"TESTING {method_name.upper()}")
    print(f"{'='*50}")

    model.eval()

    # Use test data if available, otherwise use validation data
    test_data = data.get('test', data['val'])
    if test_samples:
        # Randomly sample from test data
        test_indices = random.sample(range(len(test_data.dataset)), min(test_samples, len(test_data.dataset)))
        test_subset = torch.utils.data.Subset(test_data.dataset, test_indices)
        test_loader = torch.utils.data.DataLoader(test_subset, batch_size=test_data.batch_size, shuffle=False)
    else:
        test_loader = test_data

    print(f"Testing on {len(test_loader)} batches")

    # Initialize accumulators
    total_data_loss = 0.0
    total_phys_loss = 0.0
    total_samples = 0

    # For raw residuals
    all_raw_residuals = []

    with torch.no_grad():
        for xb, yb in tqdm(test_loader, desc=f"Testing {method_name}"):
            xb, yb = xb.to(device), yb.to(device)
            xb = xb.double()
            yb = yb.double()

            # Get model predictions
            y_pred = model(xb)

            # Use the proper adaptive_custom_loss function from basicPINNv8.py
            # This will compute actual physics residuals using the model's compute_residuals method
            from models.basicPINNv8 import adaptive_custom_loss

            loss_components = adaptive_custom_loss(
                model, xb, yb,
                data['Xmax'].to(device),
                data['Xmin'].to(device),
                data['ymax'].to(device),
                data['ymin'].to(device)
            )

            # Now we can use collect_raw_residuals since we have the proper ConfigurablePINN model
            # with compute_residuals method
            try:
                raw_residuals = collect_raw_residuals(model, xb, yb, y_pred, data, device)
                all_raw_residuals.append(raw_residuals)
            except Exception as e:
                print(f"Warning: Could not collect residuals: {e}")
                # Fallback to dummy structure
                if len(all_raw_residuals) == 0:
                    all_raw_residuals.append({
                        'data_residuals': {
                            'x2_ddot_mae': 0.0, 'y2_ddot_mae': 0.0, 'x3_ddot_mae': 0.0, 'y3_ddot_mae': 0.0, 'total_mae': 0.0
                        },
                        'physical_residuals': {
                            'res1_mean': 0.0, 'res2_mean': 0.0, 'res3_mean': 0.0, 'res4_mean': 0.0,
                            'res_mass1_mean': 0.0, 'res_mass2_mean': 0.0,
                            'res1_std': 0.0, 'res2_std': 0.0, 'res3_std': 0.0, 'res4_std': 0.0,
                            'res_mass1_std': 0.0, 'res_mass2_std': 0.0
                        }
                    })
                else:
                    all_raw_residuals.append(all_raw_residuals[0])

            # Accumulate losses
            total_data_loss += loss_components[0].item() * xb.size(0)
            # Sum all physics losses (indices 1 to end)
            phys_loss_sum = sum(loss_components[i].item() for i in range(1, len(loss_components)))
            total_phys_loss += phys_loss_sum * xb.size(0)
            total_samples += xb.size(0)

    # Calculate averages
    avg_data_loss = total_data_loss / total_samples
    avg_phys_loss = total_phys_loss / total_samples

    print("\nLoss Components:")
    print(f"  Data Loss: {avg_data_loss:.6f}")
    print(f"  Physics Loss: {avg_phys_loss:.6f}")

    # Calculate raw residual averages
    avg_raw_residuals = {
        'data_residuals': {},
        'physical_residuals': {}
    }

    # Average data residuals (MAE)
    data_keys = ['x2_ddot_mae', 'y2_ddot_mae', 'x3_ddot_mae', 'y3_ddot_mae', 'total_mae']
    for key in data_keys:
        values = [res['data_residuals'][key] for res in all_raw_residuals]
        avg_raw_residuals['data_residuals'][key] = np.mean(values)

    # Average physical residuals (mean absolute values)
    phys_keys = ['res1_mean', 'res2_mean', 'res3_mean', 'res4_mean']
    if not synthetic:  # Add mass constraints if using real data
        phys_keys.extend(['res_mass1_mean', 'res_mass2_mean'])

    for key in phys_keys:
        values = [res['physical_residuals'][key] for res in all_raw_residuals]
        avg_raw_residuals['physical_residuals'][key] = np.mean(values)

    print("\nAcceleration MAE (Data Residuals):")
    for key in ['x2_ddot_mae', 'y2_ddot_mae', 'x3_ddot_mae', 'y3_ddot_mae', 'total_mae']:
        value = avg_raw_residuals['data_residuals'][key]
        key_name = key.replace('x2_ddot_mae', 'x2_ddot').replace('y2_ddot_mae', 'y2_ddot').replace('x3_ddot_mae', 'x3_ddot').replace('y3_ddot_mae', 'y3_ddot')
        if abs(value) > 1000 or np.isnan(value):
            print(f"  {key_name}: {value:.6f} (WARNING: High value)")
        else:
            print(f"  {key_name}: {value:.6f}")

    print("\nPhysical Residuals (Mean Absolute Values):")
    for key in phys_keys:
        value = avg_raw_residuals['physical_residuals'][key]
        key_name = key.replace('_mean', '').replace('res', 'residual_')
        if np.isnan(value):
            print(f"  {key_name}: N/A (Not available - requires full PINN model)")
        elif abs(value) > 1000:
            print(f"  {key_name}: {value:.6f} (WARNING: High value)")
        else:
            print(f"  {key_name}: {value:.6f}")

    # Check for potential issues
    if avg_data_loss > 1000 or np.isnan(avg_data_loss):
        print("WARNING: Average data loss is unusually high or NaN!")
    if avg_phys_loss > 1000 or np.isnan(avg_phys_loss):
        print("WARNING: Average physics loss is unusually high or NaN!")

    return {
        'data_loss': avg_data_loss,
        'phys_loss': avg_phys_loss,
        'raw_residuals': avg_raw_residuals
    }


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description='Validate Best Models')
    parser.add_argument('--data-path', type=str, default='Data',
                       help='Path to data directory (default: Data)')
    parser.add_argument('--max-samples', type=int, default=10000,
                       help='Maximum number of samples to use (default: 10000)')
    parser.add_argument('--test-samples', type=int, default=None,
                       help='Number of test samples to use (default: use all available)')
    parser.add_argument('--synthetic', action='store_true',
                       help='Use synthetic data instead of experimental data')
    parser.add_argument('--simulation-id', type=int, default=None,
                       help='Simulation ID to use (required if --synthetic is used)')

    args = parser.parse_args()

    # Methods to validate
    methods_to_validate = ['relobralo', 'constant_weight', 'brdr', 'pecann']

    print(f"{'='*80}")
    print("BEST MODELS VALIDATION")
    print(f"{'='*80}")
    print(f"Methods to validate: {methods_to_validate}")
    print(f"Data path: {args.data_path}")
    print(f"Max samples: {args.max_samples}")
    print(f"Test samples: {args.test_samples or 'all available'}")
    print(f"Synthetic data: {args.synthetic}")

    # Load best parameters
    best_params_path = "training_scripts/best_trial_parameters.json"
    if not os.path.exists(best_params_path):
        raise FileNotFoundError(f"Best parameters file not found: {best_params_path}")

    best_params = load_best_parameters(best_params_path)
    print(f"Loaded best parameters for {len(best_params)} methods")

    # Setup device
    device = setup_device('cpu')  # Use CPU for validation
    print(f"Using device: {device}")

    # Load and prepare data (same as training)
    print("\nLoading and preparing data...")
    data = unified_data_preparation(
        data_path=args.data_path,
        max_samples=args.max_samples,
        synthetic=args.synthetic,
        simulation_id=args.simulation_id,
        batch_size=256  # Standard batch size
    )
    data = ensure_double_precision(data)

    # Verify data structure
    sample_batch = next(iter(data['train']))
    X_sample, y_sample = sample_batch
    verify_data_structure(X_sample, y_sample, synthetic=args.synthetic)

    print(f"Data loaded successfully:")
    print(f"  - Training batches: {len(data['train'])}")
    print(f"  - Validation batches: {len(data['val'])}")
    print(f"  - Input features: {X_sample.size(-1)}")
    print(f"  - Output features: {y_sample.size(-1)}")

    # Split validation set to create test set
    print("\nSplitting validation set for testing...")
    val_dataset = data['val'].dataset
    val_size = len(val_dataset)
    test_size = min(1000, val_size // 3)  # Use up to 1000 test samples
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

    print(f"Test set created with {len(data['test'])} batches")

    # Validate each method
    validation_results = {}

    for method_name in methods_to_validate:
        if method_name not in best_params:
            print(f"Warning: Method {method_name} not found in best parameters, skipping...")
            continue

        try:
            print(f"\n{'='*80}")
            print(f"VALIDATING {method_name.upper()}")
            print(f"{'='*80}")

            # Get method configuration
            config, method_params = get_method_config(method_name, best_params)
            trial_identifier = best_params[method_name]['trial_identifier']

            print(f"Trial identifier: {trial_identifier}")
            print(f"Architecture: {config['hidden_layers']} layers, {config['activation']}")
            print(f"Method parameters: {method_params}")

            # Find and load model
            model_path = find_best_model_path(method_name, trial_identifier)
            print(f"Loading model from: {model_path}")

            # Load model directly from checkpoint with correct architecture
            model = load_model_from_checkpoint(model_path, config, device)
            print("Model loaded successfully")

            # Create loss method
            loss_method = create_loss_method(method_name, method_params,
                                           enable_mass_constraints=not args.synthetic)

            # Test the model
            results = test_model(method_name, model, loss_method, data, device, args.test_samples, args.synthetic, args)

            validation_results[method_name] = {
                'trial_identifier': trial_identifier,
                'config': config,
                'method_params': method_params,
                'results': results
            }

        except Exception as e:
            print(f"Error validating {method_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print summary
    print(f"\n{'='*80}")
    print("VALIDATION SUMMARY")
    print(f"{'='*80}")

    if validation_results:
        print("Method Performance Summary:")
        print("-" * 50)

        for method_name, result in validation_results.items():
            print(f"\n{method_name.upper()}:")
            print(f"  Trial: {result['trial_identifier']}")
            print(f"  Data Loss: {result['results']['data_loss']:.6f}")
            print(f"  Physics Loss: {result['results']['phys_loss']:.6f}")

            # Show key acceleration MAEs
            data_res = result['results']['raw_residuals']['data_residuals']
            print(f"  Total MAE: {data_res['total_mae']:.6f}")
            print(f"  x2_ddot MAE: {data_res['x2_ddot_mae']:.6f}")
            print(f"  y2_ddot MAE: {data_res['y2_ddot_mae']:.6f}")

            # Show key physical residuals
            phys_res = result['results']['raw_residuals']['physical_residuals']
            print(f"  Physics Res1 (mean): {phys_res['res1_mean']:.6f}")
            print(f"  Physics Res2 (mean): {phys_res['res2_mean']:.6f}")
            print(f"  Physics Res3 (mean): {phys_res['res3_mean']:.6f}")
            print(f"  Physics Res4 (mean): {phys_res['res4_mean']:.6f}")
            if not args.synthetic:  # Only show mass residuals for real data
                print(f"  Mass Constraint 1 (mean): {phys_res['res_mass1_mean']:.6f}")
                print(f"  Mass Constraint 2 (mean): {phys_res['res_mass2_mean']:.6f}")

    else:
        print("No methods were successfully validated!")

    print(f"\nValidation complete!")
    print(f"Successfully validated {len(validation_results)} out of {len(methods_to_validate)} methods")


if __name__ == '__main__':
    main()
