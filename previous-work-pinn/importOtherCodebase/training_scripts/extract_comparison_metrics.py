#!/usr/bin/env python3
"""
Extract Comparable Metrics from PINN Training Results

This script extracts standardized metrics from all training scripts to enable
fair comparison between different adaptive weighting methods.

Metrics extracted:
1. Raw data loss (MAE/RMSE)
2. Raw physics residuals (absolute values)
3. Learned physical parameters (M1, M2, M3, D1, D2, D3, K1, K2, E1)
4. Method-specific weighting parameters
5. Final validation performance
6. Raw data residuals (MAE) for each acceleration component
7. Raw physical residuals (mean and std) for each residual component
8. Model parameters evolution (final values)

Usage:
    python extract_comparison_metrics.py [--results-dir results] [--output-file comparison_metrics.csv]
"""

import os
import sys
import numpy as np
import pandas as pd
import argparse
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.basicPINNv7 import ConfigurablePINN, get_default_pinn_config


def load_training_history(file_path: str) -> Dict[str, np.ndarray]:
    """Load training history from .npz file."""
    try:
        data = np.load(file_path)
        return {key: data[key] for key in data.files}
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}


def load_final_parameters(file_path: str) -> Dict[str, float]:
    """Load final parameters from .npz file."""
    try:
        data = np.load(file_path)
        return {key: float(data[key]) for key in data.files}
    except Exception as e:
        print(f"Error loading parameters from {file_path}: {e}")
        return {}


def extract_final_metrics(history: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Extract final epoch metrics from training history."""
    metrics = {}
    
    # Get final values for all available metrics
    for key, values in history.items():
        if len(values) > 0:
            metrics[f"final_{key}"] = float(values[-1])
    
    return metrics


def extract_raw_residuals(history: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Extract raw residual values from training history."""
    residuals = {}
    
    # Raw data residuals (MAE) for each acceleration component
    data_residual_keys = [
        'raw_x2_ddot_mae_train', 'raw_y2_ddot_mae_train', 
        'raw_x3_ddot_mae_train', 'raw_y3_ddot_mae_train',
        'raw_total_mae_train',
        'raw_x2_ddot_mae_val', 'raw_y2_ddot_mae_val', 
        'raw_x3_ddot_mae_val', 'raw_y3_ddot_mae_val',
        'raw_total_mae_val'
    ]
    
    for key in data_residual_keys:
        if key in history and len(history[key]) > 0:
            residuals[f"final_{key}"] = float(history[key][-1])
    
    # Raw physical residuals (mean and std)
    phys_residual_keys = [
        'raw_res1_mean_train', 'raw_res2_mean_train', 'raw_res3_mean_train',
        'raw_res4_mean_train', 'raw_res_mass1_mean_train', 'raw_res_mass2_mean_train',
        'raw_res1_mean_val', 'raw_res2_mean_val', 'raw_res3_mean_val',
        'raw_res4_mean_val', 'raw_res_mass1_mean_val', 'raw_res_mass2_mean_val',
        'raw_res1_std_train', 'raw_res2_std_train', 'raw_res3_std_train',
        'raw_res4_std_train', 'raw_res_mass1_std_train', 'raw_res_mass2_std_train',
        'raw_res1_std_val', 'raw_res2_std_val', 'raw_res3_std_val',
        'raw_res4_std_val', 'raw_res_mass1_std_val', 'raw_res_mass2_std_val'
    ]
    
    for key in phys_residual_keys:
        if key in history and len(history[key]) > 0:
            residuals[f"final_{key}"] = float(history[key][-1])
    
    return residuals


def extract_model_parameters(history: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Extract final model parameters from training history."""
    params = {}
    
    # Model parameters tracked over time
    param_keys = ['param_M1', 'param_M2', 'param_M3', 'param_D1', 'param_D2', 
                 'param_D3', 'param_K1', 'param_K2', 'param_E1']
    
    for key in param_keys:
        if key in history and len(history[key]) > 0:
            # Extract final parameter value
            param_name = key.replace('param_', '')
            params[f"final_{param_name}"] = float(history[key][-1])
    
    return params


def extract_weighting_parameters(history: Dict[str, np.ndarray], method: str) -> Dict[str, float]:
    """Extract method-specific weighting parameters."""
    weights = {}
    
    # Common weight keys
    weight_keys = ['weight_data', 'weight_phys_res1', 'weight_phys_res2', 
                  'weight_phys_res3', 'weight_phys_res4', 'weight_phys_mass1', 'weight_phys_mass2']
    
    # Extract final weights
    for key in weight_keys:
        if key in history and len(history[key]) > 0:
            weights[f"final_{key}"] = float(history[key][-1])
    
    # Method-specific parameters
    if method == 'adaptive_lbpin':
        sigma_keys = ['sigma_data', 'sigma_phys', 'sigma_phys_res1', 'sigma_phys_res2',
                     'sigma_phys_res3', 'sigma_phys_res4', 'sigma_phys_mass1', 'sigma_phys_mass2']
        for key in sigma_keys:
            if key in history and len(history[key]) > 0:
                weights[f"final_{key}"] = float(history[key][-1])
    
    elif method == 'alpinn':
        # AL-PINN uses lambda_data, lambda_phys_res1, etc.
        lambda_keys = [k for k in history.keys() if k.startswith('lambda_')]
        for key in lambda_keys:
            if len(history[key]) > 0:
                weights[f"final_{key}"] = float(history[key][-1])
    
    elif method == 'brdr':
        if 'scale_factor_s' in history and len(history['scale_factor_s']) > 0:
            weights['final_scale_factor_s'] = float(history['scale_factor_s'][-1])
    
    elif method == 'relobralo':
        relobralo_keys = ['alpha', 'rho', 'temperature']
        for key in relobralo_keys:
            if key in history and len(history[key]) > 0:
                weights[f"final_{key}"] = float(history[key][-1])
    
    elif method == 'pecann':
        pecann_keys = ['mu', 'lambda_data_mean', 'lambda_physics_mean', 'constraint_violation']
        for key in pecann_keys:
            if key in history and len(history[key]) > 0:
                weights[f"final_{key}"] = float(history[key][-1])
    
    elif method == 'gradnorm':
        if 'alpha' in history and len(history['alpha']) > 0:
            weights['final_alpha'] = float(history['alpha'][-1])
        if 'weight_lr' in history and len(history['weight_lr']) > 0:
            weights['final_weight_lr'] = float(history['weight_lr'][-1])
    
    elif method == 'dwpinn':
        if 'weight_lr' in history and len(history['weight_lr']) > 0:
            weights['final_weight_lr'] = float(history['weight_lr'][-1])
    
    return weights


def process_method_results(results_dir: str, method: str) -> Dict[str, float]:
    """Process results for a specific method."""
    results_path = Path(results_dir)
    
    # Handle different directory structures and file naming patterns
    history_files = []
    parameter_files = []
    
    # Try different file naming patterns for history files
    possible_history_patterns = [
        f"{method}_history.npz",
        f"{method}_pinn_history.npz",
        f"{method}_results/{method}_history.npz",
        f"{method}_results/{method}_pinn_history.npz"
    ]
    
    for pattern in possible_history_patterns:
        if "/" in pattern:
            # Subdirectory pattern
            method_dir = results_path / pattern.split("/")[0]
            if method_dir.exists():
                files = list(method_dir.glob(pattern.split("/")[1]))
                history_files.extend(files)
        else:
            # Direct file pattern
            files = list(results_path.glob(pattern))
            history_files.extend(files)
    
    # Look for parameter files
    parameter_patterns = [
        f"{method}_parameters.npz",
        f"{method}_pinn_parameters.npz",
        f"{method}_results/{method}_parameters.npz",
        f"{method}_results/{method}_pinn_parameters.npz"
    ]
    
    for pattern in parameter_patterns:
        if "/" in pattern:
            # Subdirectory pattern
            method_dir = results_path / pattern.split("/")[0]
            if method_dir.exists():
                files = list(method_dir.glob(pattern.split("/")[1]))
                parameter_files.extend(files)
        else:
            # Direct file pattern
            files = list(results_path.glob(pattern))
            parameter_files.extend(files)
    
    if not history_files:
        print(f"Warning: No history file found for {method}")
        print(f"  Looked for patterns: {possible_history_patterns}")
        return {}
    
    history_file = history_files[0]
    history = load_training_history(str(history_file))
    
    if not history:
        return {}
    
    # Extract metrics
    metrics: Dict[str, Any] = {}
    metrics['method'] = method
    metrics['history_file'] = str(history_file)
    
    # Final metrics
    final_metrics = extract_final_metrics(history)
    metrics.update(final_metrics)
    
    # Raw residuals
    raw_residuals = extract_raw_residuals(history)
    metrics.update(raw_residuals)
    
    # Model parameters from history
    model_params = extract_model_parameters(history)
    metrics.update(model_params)
    
    # Weighting parameters
    weights = extract_weighting_parameters(history, method)
    metrics.update(weights)
    
    # Final parameters from separate file (if available)
    if parameter_files:
        param_file = parameter_files[0]
        final_params = load_final_parameters(str(param_file))
        # Add prefix to distinguish from history parameters
        for key, value in final_params.items():
            metrics[f"final_param_{key}"] = value
        metrics['parameter_file'] = str(param_file)
    
    return metrics


def main():
    """Main function to extract and compare metrics from all methods."""
    parser = argparse.ArgumentParser(description='Extract Comparable Metrics from PINN Training Results')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory containing training results')
    parser.add_argument('--output-file', type=str, default='comparison_metrics.csv',
                       help='Output CSV file for comparison metrics')
    args = parser.parse_args()
    
    # List of methods to process
    methods = [
        'adaptive_lbpin',
        'alpinn', 
        'constant_weight_pinn',
        'brdr',
        'relobralo',
        'pecann',
        'gradnorm',
        'dwpinn'
    ]
    
    print("Extracting comparable metrics from training results...")
    print(f"Results directory: {args.results_dir}")
    
    all_metrics = []
    
    for method in methods:
        print(f"\nProcessing {method}...")
        metrics = process_method_results(args.results_dir, method)
        
        if metrics:
            all_metrics.append(metrics)
            print(f"  ✓ Extracted {len(metrics)} metrics")
        else:
            print(f"  ✗ No results found")
    
    if not all_metrics:
        print("\nNo results found for any method!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_metrics)
    
    # Reorder columns for better readability
    column_order = ['method', 'history_file']
    
    # Add parameter file if available
    if 'parameter_file' in df.columns:
        column_order.append('parameter_file')
    
    # Add final loss metrics
    loss_columns = [col for col in df.columns if col.startswith('final_') and 'loss' in col.lower()]
    column_order.extend(sorted(loss_columns))
    
    # Add raw data residual metrics (MAE)
    data_residual_columns = [col for col in df.columns if col.startswith('final_raw_') and 'mae' in col.lower()]
    column_order.extend(sorted(data_residual_columns))
    
    # Add raw physical residual metrics
    phys_residual_columns = [col for col in df.columns if col.startswith('final_raw_res')]
    column_order.extend(sorted(phys_residual_columns))
    
    # Add weighting parameters
    weight_columns = [col for col in df.columns if col.startswith('final_weight_') or 
                     (col.startswith('final_') and 'weight' not in col and 'param' not in col and 'raw' not in col)]
    column_order.extend(sorted(weight_columns))
    
    # Add model parameters from history
    history_param_columns = [col for col in df.columns if col.startswith('final_') and 'param_' in col and 'final_param_' not in col]
    column_order.extend(sorted(history_param_columns))
    
    # Add final parameters from separate file
    final_param_columns = [col for col in df.columns if col.startswith('final_param_')]
    column_order.extend(sorted(final_param_columns))
    
    # Add remaining columns
    remaining_columns = [col for col in df.columns if col not in column_order]
    column_order.extend(sorted(remaining_columns))
    
    # Reorder DataFrame
    df = df[column_order]
    
    # Save results
    output_path = Path(args.output_file)
    df.to_csv(output_path, index=False)
    
    print(f"\nResults saved to {output_path}")
    print(f"Total methods processed: {len(df)}")
    print(f"Total metrics extracted: {len(df.columns)}")
    
    # Print summary
    print(f"\nSummary of extracted metrics:")
    print(f"  - Loss metrics: {len(loss_columns)}")
    print(f"  - Data residual metrics (MAE): {len(data_residual_columns)}")
    print(f"  - Physical residual metrics: {len(phys_residual_columns)}")
    print(f"  - Weighting parameters: {len(weight_columns)}")
    print(f"  - Model parameters (history): {len(history_param_columns)}")
    print(f"  - Final parameters (separate file): {len(final_param_columns)}")
    
    # Show available methods
    print(f"\nAvailable methods: {list(df['method'])}")
    
    # Show sample of key metrics
    print(f"\nSample of key metrics (final validation losses):")
    key_metrics = ['method', 'final_data_val', 'final_phys_val', 'final_val_total']
    available_metrics = [col for col in key_metrics if col in df.columns]
    if available_metrics:
        print(df[available_metrics].to_string(index=False))
    
    # Show sample of raw residuals
    print(f"\nSample of raw data residuals (MAE):")
    mae_metrics = ['method', 'final_raw_total_mae_val', 'final_raw_x2_ddot_mae_val', 
                   'final_raw_y2_ddot_mae_val', 'final_raw_x3_ddot_mae_val', 'final_raw_y3_ddot_mae_val']
    available_mae_metrics = [col for col in mae_metrics if col in df.columns]
    if available_mae_metrics:
        print(df[available_mae_metrics].to_string(index=False))


if __name__ == '__main__':
    main() 