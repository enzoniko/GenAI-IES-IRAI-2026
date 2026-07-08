"""
Common utilities for PINN training scripts.

This module contains shared functions and classes used across all training scripts
to reduce code duplication and ensure consistency.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import os
import time
import argparse
import json
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from matplotlib import cm

# PINN version configuration
PINN_VERSION = "dimensional"


class MultiLossEarlyStopping:
    """Stop when *all* monitored losses fail to improve for `patience` epochs."""
    
    def __init__(self, patience: int, min_delta: float, loss_names: List[str]):
        self.patience = patience
        self.min_delta = min_delta
        self.loss_names = loss_names
        self.best: Dict[str, Optional[float]] = {k: None for k in loss_names}
        self.count = {k: 0 for k in loss_names}
        self.early_stop = False
    
    def __call__(self, loss_dict: Dict[str, float]):
        stalled = 0
        for k in self.loss_names:
            val = loss_dict[k]
            best = self.best[k]
            if best is None or val < best - self.min_delta:
                self.best[k] = val
                self.count[k] = 0
            else:
                self.count[k] += 1
            if self.count[k] >= self.patience: 
                stalled += 1
        if stalled == len(self.loss_names):
            self.early_stop = True
            print("\nEarly stopping triggered.")


class ReduceLROnPlateau:
    """Learning rate scheduler that reduces LR when validation loss plateaus."""
    
    def __init__(self, opt: optim.Optimizer, factor: float = 0.5, patience: int = 10, min_lr: float = 1e-7):
        self.opt = opt
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best = None
        self.count = 0
    
    def step(self, loss: float):
        if self.best is None or loss < self.best:
            self.best = loss
            self.count = 0
        else:
            self.count += 1
            if self.count >= self.patience:
                for g in self.opt.param_groups:
                    new_lr = max(g['lr'] * self.factor, self.min_lr)
                    if g['lr'] > new_lr:
                        g['lr'] = new_lr
                        print(f"\nLR -> {new_lr:.2e}")
                self.count = 0


def collect_raw_residuals(model, x_batch, y_batch, y_pred, data_dict, device):
    """
    Collect raw data residuals (MAE) and physical residuals for detailed analysis.
    
    Args:
        model: The PINN model
        x_batch: Input batch
        y_batch: Target batch
        y_pred: Model predictions
        data_dict: Data dictionary with normalization parameters
        device: Device to use
        
    Returns:
        Dictionary containing raw residuals
    """
    with torch.no_grad():
        # Denormalize predictions and targets for accurate MAE calculation
        y_pred_denorm = y_pred * (data_dict['ymax'].to(device) - data_dict['ymin'].to(device)) + data_dict['ymin'].to(device)
        y_batch_denorm = y_batch * (data_dict['ymax'].to(device) - data_dict['ymin'].to(device)) + data_dict['ymin'].to(device)
        
        # Calculate raw data residuals (MAE) for each acceleration component
        data_residuals = torch.abs(y_pred_denorm - y_batch_denorm)  # Shape: (batch_size, 4)
        
        # Split into individual acceleration components
        x2_ddot_residual = data_residuals[:, 0]  # x2 acceleration residual
        y2_ddot_residual = data_residuals[:, 1]  # y2 acceleration residual
        x3_ddot_residual = data_residuals[:, 2]  # x3 acceleration residual
        y3_ddot_residual = data_residuals[:, 3]  # y3 acceleration residual
        
        # Calculate physical residuals using the model's compute_residuals method
        # Pass normalization parameters for proper denormalization inside the model
        results = model.compute_residuals(x_batch, y_pred, data_dict['Xmax'].to(device), data_dict['Xmin'].to(device), data_dict['ymax'].to(device), data_dict['ymin'].to(device))
        
        # Handle both old tuple format and new dictionary format for compatibility
        if isinstance(results, dict):
            # New dimensionless PINN format
            physical_residuals = results['residuals']
            res1, res2, res3, res4, res_mass1, res_mass2 = physical_residuals
        else:
            # Old dimensional PINN format (backwards compatibility)
            res1, res2, res3, res4, res_mass1, res_mass2 = results
        
        # Always return all keys for compatibility with training scripts
        # The model.compute_residuals already handles conditional mass constraints correctly
        return {
            'data_residuals': {
                'x2_ddot_mae': x2_ddot_residual.mean().item(),
                'y2_ddot_mae': y2_ddot_residual.mean().item(),
                'x3_ddot_mae': x3_ddot_residual.mean().item(),
                'y3_ddot_mae': y3_ddot_residual.mean().item(),
                'total_mae': data_residuals.mean().item()
            },
            'physical_residuals': {
                'res1_mean': res1.mean().item(),
                'res2_mean': res2.mean().item(),
                'res3_mean': res3.mean().item(),
                'res4_mean': res4.mean().item(),
                'res_mass1_mean': res_mass1.mean().item(),
                'res_mass2_mean': res_mass2.mean().item(),
                'res1_std': res1.std().item() if res1.numel() > 1 else 0.0,
                'res2_std': res2.std().item() if res2.numel() > 1 else 0.0,
                'res3_std': res3.std().item() if res3.numel() > 1 else 0.0,
                'res4_std': res4.std().item() if res4.numel() > 1 else 0.0,
                'res_mass1_std': res_mass1.std().item() if res_mass1.numel() > 1 else 0.0,
                'res_mass2_std': res_mass2.std().item() if res_mass2.numel() > 1 else 0.0
            }
        }


def collect_model_parameters(model):
    """
    Collect current model parameters for tracking.
    
    Args:
        model: The PINN model
        
    Returns:
        Dictionary containing current parameter values
    """
    params = {
        'M1': float(model.M1.detach().cpu()),
        'M2': float(model.M2.detach().cpu()),
        'M3': float(model.M3.detach().cpu()),
        'D1': float(model.D1.detach().cpu()),
        'D2': float(model.D2.detach().cpu()),
        'D3': float(model.D3.detach().cpu()),
        'K1': float(model.K1.detach().cpu()),
        'K2': float(model.K2.detach().cpu()),
        'E1': float(model.E1.detach().cpu())
    }
    
    # Add clearance parameter if available (for dimensionless PINN)
    if hasattr(model, 'c'):
        params['c'] = float(model.c.detach().cpu())
    
    return params


def initialize_comprehensive_history(synthetic: bool = False):
    """
    Initialize comprehensive history dictionary with all tracking keys.
    
    Args:
        synthetic: Whether using synthetic data (affects mass constraints)
        
    Returns:
        Dictionary with initialized history lists
    """
    hist = {
        # Standard loss metrics
        'train_total': [], 'val_total': [],
        'data_train': [], 'data_val': [],
        'phys_train': [], 'phys_val': [],
        'phys_res1_train': [], 'phys_res2_train': [], 'phys_res3_train': [],
        'phys_res4_train': [],
        'phys_res1_val': [], 'phys_res2_val': [], 'phys_res3_val': [],
        'phys_res4_val': [],
        
        # Raw data residuals (MAE)
        'raw_x2_ddot_mae_train': [], 'raw_y2_ddot_mae_train': [], 
        'raw_x3_ddot_mae_train': [], 'raw_y3_ddot_mae_train': [],
        'raw_total_mae_train': [],
        'raw_x2_ddot_mae_val': [], 'raw_y2_ddot_mae_val': [], 
        'raw_x3_ddot_mae_val': [], 'raw_y3_ddot_mae_val': [],
        'raw_total_mae_val': [],
        
        # Raw physical residuals (mean)
        'raw_res1_mean_train': [], 'raw_res2_mean_train': [], 
        'raw_res3_mean_train': [], 'raw_res4_mean_train': [],
        'raw_res1_mean_val': [], 'raw_res2_mean_val': [], 
        'raw_res3_mean_val': [], 'raw_res4_mean_val': [],
        
        # Raw physical residuals (std)
        'raw_res1_std_train': [], 'raw_res2_std_train': [], 
        'raw_res3_std_train': [], 'raw_res4_std_train': [],
        'raw_res1_std_val': [], 'raw_res2_std_val': [], 
        'raw_res3_std_val': [], 'raw_res4_std_val': [],
        
        # Model parameters
        'param_M1': [], 'param_M2': [], 'param_M3': [],
        'param_D1': [], 'param_D2': [], 'param_D3': [],
        'param_K1': [], 'param_K2': [], 'param_E1': []
    }
    
    # Conditionally add mass constraint keys for real data
    if not synthetic:
        hist.update({
            'phys_mass1_train': [], 'phys_mass2_train': [],
            'phys_mass1_val': [], 'phys_mass2_val': [],
            'raw_res_mass1_mean_train': [], 'raw_res_mass2_mean_train': [],
            'raw_res_mass1_mean_val': [], 'raw_res_mass2_mean_val': [],
            'raw_res_mass1_std_train': [], 'raw_res_mass2_std_train': [],
            'raw_res_mass1_std_val': [], 'raw_res_mass2_std_val': []
        })
    
    # Add training phase tracking
    hist['training_phase'] = []
    
    return hist


def update_comprehensive_history(hist, epoch_metrics, model, is_training=True):
    """
    Update comprehensive history with epoch metrics and model parameters.
    
    Args:
        hist: History dictionary
        epoch_metrics: Dictionary with epoch-specific metrics
        model: The PINN model
        is_training: Whether this is training or validation data
    """
    suffix = '_train' if is_training else '_val'
    
    # Update standard loss metrics
    for key, value in epoch_metrics.items():
        if key in hist:
            hist[key].append(value)
    
    # Update raw data residuals
    if 'raw_residuals' in epoch_metrics:
        raw_data = epoch_metrics['raw_residuals']['data_residuals']
        for key, value in raw_data.items():
            hist_key = f"raw_{key}{suffix}"
            if hist_key in hist:
                hist[hist_key].append(value)
    
    # Update raw physical residuals
    if 'raw_residuals' in epoch_metrics:
        raw_phys = epoch_metrics['raw_residuals']['physical_residuals']
        for key, value in raw_phys.items():
            hist_key = f"raw_{key}{suffix}"
            if hist_key in hist:
                hist[hist_key].append(value)
    
    # Update model parameters
    params = collect_model_parameters(model)
    for param_name, param_value in params.items():
        hist_key = f"param_{param_name}"
        if hist_key in hist:
            hist[hist_key].append(param_value)


def prepare_data(X: torch.Tensor, y: torch.Tensor, batch: int, max_samples: Optional[int] = None) -> Dict:
    """
    Normalize and split data; optionally down-sample to `max_samples`.
    
    Args:
        X: Input tensor
        y: Target tensor
        batch: Batch size
        max_samples: Maximum number of samples to use
        
    Returns:
        Dictionary containing data loaders and normalization parameters
    """
    if max_samples is not None and X.size(0) > max_samples:
        idx = torch.randperm(X.size(0))[:max_samples]
        X, y = X[idx], y[idx]
    
    X_min, X_max = X.min(0, keepdim=True)[0], X.max(0, keepdim=True)[0]
    y_min, y_max = y.min(0, keepdim=True)[0], y.max(0, keepdim=True)[0]

    # Normalize inputs: features 0-7 normalized by data-driven min/max, features 8-9 (omega, t)
    # normalized using fixed dataset-wide ranges: omega [73.3, 377] rad/s, t [0, 5] s
    X_norm = X.clone()
    # Data-driven normalization for first 8 columns
    X_norm[:, 0:8] = (X[:, 0:8] - X_min[:, 0:8]) / (X_max[:, 0:8] - X_min[:, 0:8] + 1e-12)

    # Fixed normalization for omega and t
    omega_min_fixed, omega_max_fixed = 73.3, 377.0
    t_min_fixed, t_max_fixed = 0.0, 5.0
    X_norm[:, 8] = (X[:, 8] - omega_min_fixed) / (omega_max_fixed - omega_min_fixed + 1e-12)
    X_norm[:, 9] = (X[:, 9] - t_min_fixed) / (t_max_fixed - t_min_fixed + 1e-12)

    # Override X_min/X_max for omega,t so denormalization in the model is consistent
    X_min = X_min.clone()
    X_max = X_max.clone()
    X_min[:, 8] = omega_min_fixed
    X_max[:, 8] = omega_max_fixed
    X_min[:, 9] = t_min_fixed
    X_max[:, 9] = t_max_fixed

    X = X_norm
    y = (y - y_min) / (y_max - y_min + 1e-12)

    ds = TensorDataset(X, y)
    n = len(ds)
    val_len = int(0.2 * n)
    test_len = int(0.2 * n)
    train_len = n - val_len - test_len
    train_ds, val_ds, _ = random_split(ds, [train_len, val_len, test_len])

    return {
        'train': DataLoader(train_ds, batch_size=batch, shuffle=True),
        'val': DataLoader(val_ds, batch_size=batch, shuffle=False),
        'Xmin': X_min, 'Xmax': X_max, 'ymin': y_min, 'ymax': y_max
    }


def load_and_prepare_data(data_path: str = "Data", max_samples: Optional[int] = None, synthetic: bool = False, simulation_id: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
    """
    Load and prepare data for training.
    
    Args:
        data_path: Path to data directory
        max_samples: Maximum number of samples to use
        synthetic: If True, load synthetic data from .npz files
        simulation_id: Specific simulation ID to load (required if synthetic=True)
        
    Returns:
        Tuple of (X, y, metadata) tensors. metadata is None for experimental data.
    """
    if synthetic:
        if simulation_id is None:
            raise ValueError("simulation_id must be provided when synthetic=True")
        
        # Load synthetic data from .npz file
        import json
        from pathlib import Path
        
        sim_dir = Path(data_path) / f"simulation_{simulation_id:06d}"
        time_series_file = sim_dir / "time_series.npz"
        metadata_file = sim_dir / "metadata.json"
        
        if not time_series_file.exists():
            raise FileNotFoundError(f"Time series file not found: {time_series_file}")
        
        # Load time series data
        time_series_data = np.load(time_series_file, allow_pickle=True)
        
        # Extract data
        time = time_series_data['time']  # Shape: (5000,)
        positions = time_series_data['positions'].item()
        velocities = time_series_data['velocities'].item()
        accelerations = time_series_data['accelerations'].item()
        
        # Load metadata for parameters
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Use PINN-compatible parameters if available, otherwise fall back to original
        if 'pinn_parameters' in metadata:
            parameters = metadata['pinn_parameters']
            print(f"Using PINN-compatible parameters for simulation {simulation_id}")
        else:
            parameters = metadata['parameters']
            print(f"PINN-compatible parameters not found, using original parameters for simulation {simulation_id}")
            print("Note: Parameter mapping may not be optimal for PINN training")
        
        # Create input features for PINN (10 features as expected by basicPINNv7)
        # Format: [x2_dot, y2_dot, x3_dot, y3_dot, x2, y2, x3, y3, omega, t]
        X = np.column_stack([
            velocities['X2_dot'],  # x2_dot
            velocities['Y2_dot'],  # y2_dot
            velocities['X3_dot'],  # x3_dot
            velocities['Y3_dot'],  # y3_dot
            positions['X2'],       # x2
            positions['Y2'],       # y2
            positions['X3'],       # x3
            positions['Y3'],       # y3
            np.full_like(time, parameters['Omega']),  # omega (constant)
            time                   # t
        ])
        
        # Create output targets (4 accelerations as expected by basicPINNv7)
        # Format: [x2_ddot, y2_ddot, x3_ddot, y3_ddot]
        # Always use exact accelerations from simulation (never finite differences)
        if len(accelerations['X2_ddot']) == len(time):
            # Accelerations are already time series - use exact physics-based accelerations
            y = np.column_stack([
                accelerations['X2_ddot'],  # x2_ddot
                accelerations['Y2_ddot'],  # y2_ddot
                accelerations['X3_ddot'],  # x3_ddot
                accelerations['Y3_ddot']   # y3_ddot
            ])
        else:
            # This should never happen with properly generated synthetic data
            # If it does, it indicates a data generation issue
            raise ValueError(
                f"Accelerations are not time series for simulation {simulation_id}. "
                f"Expected length {len(time)}, got {len(accelerations['X2_ddot'])}. "
                f"This indicates the synthetic data was not generated correctly. "
                f"Please regenerate the synthetic dataset."
            )
        
        # Convert to tensors (keep in physical units; normalization happens downstream)
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        
        # Create metadata dictionary with ground truth parameters only
        metadata_dict = {
            'ground_truth_params': parameters,
            'simulation_id': simulation_id
        }
        
        if max_samples is not None and X.size(0) > max_samples:
            idx = torch.randperm(X.size(0))[:max_samples]
            X, y = X[idx], y[idx]
        
        return X, y, metadata_dict
    
    else:
        # Experimental data loading
        X = torch.load(f'{data_path}/X_normal_v3.pth')[:, :249998, :]
        y = torch.load(f'{data_path}/Y_normal_v3.pth')[:, :249998, :]
        
        # Check if data already has time feature (V3 data has 10 features including time)
        if X.size(-1) == 10:
            # V3 data already includes time feature, no need to add another
            print("Using V3 data with built-in time feature")
        else:
            # Original data format - add time feature
            print("Using original data format - adding time feature")
            t = torch.arange(0, 2e-5 * X.size(1), 2e-5).view(1, -1, 1).expand_as(X[:, :, :1])
            X = torch.cat((X, t), dim=2)
        
        # Flatten first two dims
        X = X.flatten(0, 1)
        y = y.flatten(0, 1)
        
        if max_samples is not None and X.size(0) > max_samples:
            idx = torch.randperm(X.size(0))[:max_samples]
            X, y = X[idx], y[idx]
        
        return X, y, None


def prepare_synthetic_data(X: torch.Tensor, y: torch.Tensor, metadata: Dict, batch_size: int, max_samples: Optional[int] = None) -> Dict:
    """
    Prepare synthetic data for training without double normalization.
    
    Args:
        X: Input tensor (already normalized)
        y: Target tensor (already normalized)
        metadata: Metadata dictionary containing normalization parameters
        batch_size: Batch size for data loaders
        max_samples: Maximum number of samples to use
        
    Returns:
        Dictionary containing data loaders and normalization parameters
    """
    from torch.utils.data import DataLoader, TensorDataset, random_split
    
    # Apply max_samples if needed
    if max_samples is not None and X.size(0) > max_samples:
        idx = torch.randperm(X.size(0))[:max_samples]
        X, y = X[idx], y[idx]
    
    ds = TensorDataset(X, y)
    n = len(ds)
    val_len = int(0.2 * n)
    test_len = int(0.2 * n)
    train_len = n - val_len - test_len
    train_ds, val_ds, _ = random_split(ds, [train_len, val_len, test_len])

    return {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        # Use the correct normalization parameters from the metadata
        'Xmin': metadata['normalization_params']['X_min'],
        'Xmax': metadata['normalization_params']['X_max'],
        'ymin': metadata['normalization_params']['y_min'],
        'ymax': metadata['normalization_params']['y_max']
    }


def get_pinn_config(hidden_layers: List[int], activation: str, dropout_rate: float, init_method: str) -> Dict:
    """
    Create PINN configuration dictionary.
    
    Args:
        hidden_layers: List of hidden layer sizes
        activation: Activation function name
        dropout_rate: Dropout rate
        init_method: Weight initialization method
        
    Returns:
        Dictionary with PINN configuration
    """
    return {
        'hidden_layers': hidden_layers,
        'activation': activation,
        'dropout_rate': dropout_rate,
        'init_method': init_method
    }


def get_param_init_config(synthetic: bool = False) -> Dict:
    """Get default parameter initialization configuration."""
    if synthetic:
        # Configuration optimized for synthetic data with PINN-compatible parameters
        # These values are based on the parameter mapping from synthetic data
        return {
            'method': 'fixed',
            'values': {
                'M1': 15.0, 'M2': 1.0, 'M3': 1.0,  # Masses from synthetic data
                'D1': 100.0, 'D2': 100.0, 'D3': 700.0,  # Damping: Ds1, Ds2, Db
                'K1': 1.2e6 + 5.0e6, 'K2': 1.2e6 + 5.0e6,  # Ks1 + Kb, Ks2 + Kb
                'E1': 5.0e-5 / 15.0  # mu_eps / M1 (eccentricity)
            }
        }
    else:
        # Configuration for real data (original values)
        return {
            'method': 'fixed',
            'values': {
                'M1': 10.0, 'M2': 10.0, 'M3': 11.0,
                'D1': 1.0, 'D2': 1.0, 'D3': 1.0,
                'K1': 1000.0, 'K2': 1000.0, 'E1': 0.1
            }
        }


def setup_device(device_arg: Optional[str] = None) -> torch.device:
    """
    Setup device for training.
    
    Args:
        device_arg: Device specification (cuda/cpu)
        
    Returns:
        torch.device object
    """
    if device_arg:
        device = torch.device(device_arg)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}  |  CUDA available: {torch.cuda.is_available()}")
    return device


def create_output_directory(method_name: str, output_dir: Optional[str] = None) -> Tuple[str, str]:
    """
    Create output directory for results.
    
    Args:
        method_name: Name of the training method
        output_dir: Custom output directory (optional)
        
    Returns:
        Tuple of (output_dir, model_path)
    """
    if output_dir is None:
        output_dir = f"results/{method_name}_{int(time.time())}"
    
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "best_model.pth")
    
    return output_dir, model_path


def plot_training_history(hist: Dict, output_dir: str, method_name: str, 
                         include_weights: bool = True, include_method_params: bool = True):
    """
    Plot training history with method-specific visualizations.
    
    Args:
        hist: Training history dictionary
        output_dir: Output directory
        method_name: Name of the training method
        include_weights: Whether to include weight evolution plots
        include_method_params: Whether to include method-specific parameter plots
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Check if we have training phase information for pre-training visualization
    has_training_phases = 'training_phase' in hist and len(hist['training_phase']) > 0
    
    # Determine number of subplots
    num_plots = 3  # Total, Data vs Physics, Individual Physics
    if include_weights:
        num_plots += 1
    if include_method_params:
        num_plots += 1
    
    fig, axes = plt.subplots(num_plots, 1, figsize=(15, 5 * num_plots), sharex=True)
    if num_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # Find phase transition point if available
    phase_transition_epoch = None
    if has_training_phases:
        for i, phase in enumerate(hist['training_phase']):
            if i > 0 and hist['training_phase'][i-1] == 'pretrain' and phase == 'full':
                phase_transition_epoch = i
                break
    
    def add_phase_transition_line(ax, epoch, label='Pre-train → Full training'):
        """Add a vertical line to mark phase transition."""
        if epoch is not None:
            ax.axvline(x=epoch, color='red', linestyle='--', alpha=0.7, linewidth=2, label=label)
    
    # Plot total losses
    axes[plot_idx].plot(hist['train_total'], label='Train Total', color='b')
    axes[plot_idx].plot(hist['val_total'], label='Val Total', color='r')
    add_phase_transition_line(axes[plot_idx], phase_transition_epoch)
    axes[plot_idx].set_title('Total Loss Evolution', fontsize=16)
    axes[plot_idx].set_ylabel('Loss', fontsize=12)
    axes[plot_idx].legend()
    axes[plot_idx].set_yscale('log')
    plot_idx += 1
    
    # Plot data vs physics losses
    axes[plot_idx].plot(hist['data_train'], label='Data (Train)', color='b')
    axes[plot_idx].plot(hist['data_val'], label='Data (Val)', color='r')
    axes[plot_idx].plot(hist['phys_train'], label='Physics (Train)', color='g')
    axes[plot_idx].plot(hist['phys_val'], label='Physics (Val)', color='orange')
    add_phase_transition_line(axes[plot_idx], phase_transition_epoch, '')  # Empty label to avoid duplicate
    axes[plot_idx].set_title('Data vs Physics Loss Evolution', fontsize=16)
    axes[plot_idx].set_ylabel('Loss', fontsize=12)
    axes[plot_idx].legend()
    axes[plot_idx].set_yscale('log')
    plot_idx += 1
    
    # Plot individual physics loss components
    phys_keys = ['phys_res1', 'phys_res2', 'phys_res3', 'phys_res4', 'phys_mass1', 'phys_mass2']
    cmap = plt.colormaps.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, len(phys_keys)))
    for i, key in enumerate(phys_keys):
        if f'{key}_train' in hist:
            axes[plot_idx].plot(hist[f'{key}_train'], label=key, color=colors[i])
    add_phase_transition_line(axes[plot_idx], phase_transition_epoch, '')  # Empty label to avoid duplicate
    axes[plot_idx].set_title('Individual Physics Losses (Training)', fontsize=16)
    axes[plot_idx].set_ylabel('Loss', fontsize=12)
    axes[plot_idx].legend()
    axes[plot_idx].set_yscale('log')
    plot_idx += 1
    
    # Plot weights if available
    if include_weights:
        weight_keys = [k for k in hist.keys() if k.startswith('weight_')]
        if weight_keys:
            for i, key in enumerate(weight_keys):
                axes[plot_idx].plot(hist[key], label=key.replace('weight_', ''), 
                                  color=colors[i % len(colors)])
            add_phase_transition_line(axes[plot_idx], phase_transition_epoch, '')  # Empty label to avoid duplicate
            axes[plot_idx].set_title(f'{method_name} Weights Evolution', fontsize=16)
            axes[plot_idx].set_ylabel('Weight Value', fontsize=12)
            axes[plot_idx].legend()
            plot_idx += 1
    
    # Plot method-specific parameters if available
    if include_method_params:
        method_param_keys = [k for k in hist.keys() if k not in ['train_total', 'val_total', 'data_train', 
                                                                'data_val', 'phys_train', 'phys_val'] and 
                           not k.startswith('weight_') and not k.startswith('phys_')]
        if method_param_keys:
            for i, key in enumerate(method_param_keys):
                axes[plot_idx].plot(hist[key], label=key, color=colors[i % len(colors)])
            axes[plot_idx].set_title(f'{method_name} Parameters Evolution', fontsize=16)
            axes[plot_idx].set_ylabel('Parameter Value', fontsize=12)
            axes[plot_idx].legend()
            plot_idx += 1
    
    # Set x-label for the last subplot
    axes[-1].set_xlabel('Epoch', fontsize=12)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{method_name.lower()}_evolution.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_trajectory_predictions(model, data_dict, output_dir, method_name, 
                               sample_indices=[0, 1, 2], device='cpu'):
    """
    Plot trajectory predictions vs true values and phase space visualizations.
    
    This function now properly:
    1. Uses the full dataset instead of just validation batches
    2. Integrates predicted accelerations to get velocities and positions
    3. Creates proper trajectory plots with continuous lines
    4. Handles both 2D and 3D data appropriately
    
    Args:
        model: Trained PINN model
        data_dict: Data dictionary with normalization parameters and full dataset
        output_dir: Output directory for saving plots
        method_name: Name of the training method
        sample_indices: List of sample indices to plot
        device: Device to use for computations
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.integrate import cumtrapz
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    print(f"[DEBUG] Starting trajectory prediction plotting for {method_name}")
    
    # Load the full dataset from the original files, not just validation batches
    try:
        # Try to access full data if available in data_dict
        if 'X_full' in data_dict and 'y_full' in data_dict:
            X_full = data_dict['X_full'].to(device)
            y_full = data_dict['y_full'].to(device)
            print(f"[DEBUG] Using stored full dataset: X_full.shape = {X_full.shape}, y_full.shape = {y_full.shape}")
        else:
            # Reconstruct full dataset from data loaders
            print(f"[DEBUG] Reconstructing full dataset from data loaders")
            X_batches = []
            y_batches = []
            
            # Combine training and validation data
            for batch_x, batch_y in data_dict['train']:
                X_batches.append(batch_x)
                y_batches.append(batch_y)
            for batch_x, batch_y in data_dict['val']:
                X_batches.append(batch_x)
                y_batches.append(batch_y)
            
            X_full = torch.cat(X_batches, dim=0).to(device)
            y_full = torch.cat(y_batches, dim=0).to(device)
            print(f"[DEBUG] Reconstructed dataset: X_full.shape = {X_full.shape}, y_full.shape = {y_full.shape}")
    
    except Exception as e:
        print(f"[ERROR] Failed to load full dataset: {e}")
        print(f"[DEBUG] Falling back to validation batch")
        # Fallback to validation batch
        val_loader = data_dict['val']
        sample_batch = next(iter(val_loader))
        X_full, y_full = sample_batch
        X_full = X_full.to(device)
        y_full = y_full.to(device)
        print(f"[DEBUG] Fallback dataset: X_full.shape = {X_full.shape}, y_full.shape = {y_full.shape}")
    
    # Get model predictions on full dataset
    model.eval()
    print(f"[DEBUG] Getting model predictions...")
    with torch.no_grad():
        # Process in batches to avoid memory issues
        batch_size = 256
        y_pred_batches = []
        for i in range(0, X_full.shape[0], batch_size):
            batch_x = X_full[i:i+batch_size]
            # Ensure double precision for model compatibility
            batch_x = batch_x.double()
            batch_pred = model(batch_x)
            y_pred_batches.append(batch_pred)
        y_pred = torch.cat(y_pred_batches, dim=0)
    
    print(f"[DEBUG] Model predictions shape: {y_pred.shape}")
    
    # Denormalize data for plotting
    print(f"[DEBUG] Denormalizing data...")
    X_range = data_dict['Xmax'].to(device) - data_dict['Xmin'].to(device)
    y_range = data_dict['ymax'].to(device) - data_dict['ymin'].to(device)
    
    X_denorm = X_full * X_range + data_dict['Xmin'].to(device)
    y_denorm = y_full * y_range + data_dict['ymin'].to(device)
    y_pred_denorm = y_pred * y_range + data_dict['ymin'].to(device)
    
    # Convert to numpy for plotting
    X_np = X_denorm.cpu().numpy()
    y_np = y_denorm.cpu().numpy()
    y_pred_np = y_pred_denorm.cpu().numpy()
    
    print(f"[DEBUG] Denormalized data shapes:")
    print(f"[DEBUG]   X_np.shape = {X_np.shape}")
    print(f"[DEBUG]   y_np.shape = {y_np.shape}")
    print(f"[DEBUG]   y_pred_np.shape = {y_pred_np.shape}")
    
    # Check data dimensionality
    is_3d = len(X_np.shape) == 3
    print(f"[DEBUG] Data is 3D: {is_3d}")
    
    # Feature names for clarity
    input_names = ['x2_dot', 'y2_dot', 'x3_dot', 'y3_dot', 'x2', 'y2', 'x3', 'y3', 'omega', 't']
    output_names = ['x2_ddot', 'y2_ddot', 'x3_ddot', 'y3_ddot']
    
    # For 2D data, we need to create a time-ordered trajectory by sorting by time
    if not is_3d and X_np.shape[1] >= 10:  # Ensure we have time feature
        print(f"[DEBUG] Processing 2D data for trajectory integration")
        
        # Sort data by time (last feature) to create proper trajectory
        time_col = X_np[:, -1]  # Time is last feature
        time_sort_idx = np.argsort(time_col)
        
        # Sort all data by time
        X_np_sorted = X_np[time_sort_idx]
        y_np_sorted = y_np[time_sort_idx]
        y_pred_np_sorted = y_pred_np[time_sort_idx]
        
        print(f"[DEBUG] Data sorted by time, range: {time_col.min():.4f} to {time_col.max():.4f}")
        
        # Extract time and time step
        time_points = X_np_sorted[:, -1]  # Last column is time
        dt = np.diff(time_points)
        dt = np.append(dt, dt[-1])  # Append last dt to match array size
        
        print(f"[DEBUG] Time step statistics: mean={np.mean(dt):.6f}, std={np.std(dt):.6f}")
        
        # Integrate accelerations to get trajectories
        print(f"[DEBUG] Integrating predicted accelerations to get trajectories...")
        
        # Extract current velocities and positions from input features
        true_vel_x2 = X_np_sorted[:, 0]  # x2_dot
        true_vel_y2 = X_np_sorted[:, 1]  # y2_dot
        true_vel_x3 = X_np_sorted[:, 2]  # x3_dot
        true_vel_y3 = X_np_sorted[:, 3]  # y3_dot
        true_pos_x2 = X_np_sorted[:, 4]  # x2
        true_pos_y2 = X_np_sorted[:, 5]  # y2
        true_pos_x3 = X_np_sorted[:, 6]  # x3
        true_pos_y3 = X_np_sorted[:, 7]  # y3
        
        # Predicted accelerations
        pred_acc_x2 = y_pred_np_sorted[:, 0]  # x2_ddot
        pred_acc_y2 = y_pred_np_sorted[:, 1]  # y2_ddot
        pred_acc_x3 = y_pred_np_sorted[:, 2]  # x3_ddot
        pred_acc_y3 = y_pred_np_sorted[:, 3]  # y3_ddot
        
        # True accelerations
        true_acc_x2 = y_np_sorted[:, 0]
        true_acc_y2 = y_np_sorted[:, 1]
        true_acc_x3 = y_np_sorted[:, 2]
        true_acc_y3 = y_np_sorted[:, 3]
        
        # Integrate predicted accelerations to get predicted velocities
        pred_vel_x2 = cumtrapz(pred_acc_x2, time_points, initial=true_vel_x2[0])
        pred_vel_y2 = cumtrapz(pred_acc_y2, time_points, initial=true_vel_y2[0])
        pred_vel_x3 = cumtrapz(pred_acc_x3, time_points, initial=true_vel_x3[0])
        pred_vel_y3 = cumtrapz(pred_acc_y3, time_points, initial=true_vel_y3[0])
        
        # Integrate predicted velocities to get predicted positions
        pred_pos_x2 = cumtrapz(pred_vel_x2, time_points, initial=true_pos_x2[0])
        pred_pos_y2 = cumtrapz(pred_vel_y2, time_points, initial=true_pos_y2[0])
        pred_pos_x3 = cumtrapz(pred_vel_x3, time_points, initial=true_pos_x3[0])
        pred_pos_y3 = cumtrapz(pred_vel_y3, time_points, initial=true_pos_y3[0])
        
        print(f"[DEBUG] Integration completed. Trajectory length: {len(time_points)}")
        
        # Create trajectory comparison plots
        print(f"[DEBUG] Creating trajectory comparison plots...")
        
        # Plot 1: Acceleration Comparison
        fig1, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig1.suptitle(f'{method_name} - Acceleration Predictions vs True Values', fontsize=16)
        
        acc_data = [
            (true_acc_x2, pred_acc_x2, 'x2_ddot', 'X2 Acceleration'),
            (true_acc_y2, pred_acc_y2, 'y2_ddot', 'Y2 Acceleration'),
            (true_acc_x3, pred_acc_x3, 'x3_ddot', 'X3 Acceleration'),
            (true_acc_y3, pred_acc_y3, 'y3_ddot', 'Y3 Acceleration')
        ]
        
        for i, (true_data, pred_data, var_name, title) in enumerate(acc_data):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            ax.plot(time_points, true_data, label=f'True {var_name}', linewidth=2, alpha=0.8)
            ax.plot(time_points, pred_data, label=f'Pred {var_name}', linestyle='--', linewidth=2, alpha=0.8)
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Acceleration (m/s²)')
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{method_name.lower()}_acceleration_predictions.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Velocity Comparison  
        fig2, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig2.suptitle(f'{method_name} - Velocity Predictions vs True Values', fontsize=16)
        
        vel_data = [
            (true_vel_x2, pred_vel_x2, 'x2_dot', 'X2 Velocity'),
            (true_vel_y2, pred_vel_y2, 'y2_dot', 'Y2 Velocity'),
            (true_vel_x3, pred_vel_x3, 'x3_dot', 'X3 Velocity'),
            (true_vel_y3, pred_vel_y3, 'y3_dot', 'Y3 Velocity')
        ]
        
        for i, (true_data, pred_data, var_name, title) in enumerate(vel_data):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            ax.plot(time_points, true_data, label=f'True {var_name}', linewidth=2, alpha=0.8)
            ax.plot(time_points, pred_data, label=f'Pred {var_name}', linestyle='--', linewidth=2, alpha=0.8)
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Velocity (m/s)')
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{method_name.lower()}_velocity_predictions.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 3: Position Comparison
        fig3, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig3.suptitle(f'{method_name} - Position Predictions vs True Values', fontsize=16)
        
        pos_data = [
            (true_pos_x2, pred_pos_x2, 'x2', 'X2 Position'),
            (true_pos_y2, pred_pos_y2, 'y2', 'Y2 Position'),
            (true_pos_x3, pred_pos_x3, 'x3', 'X3 Position'),
            (true_pos_y3, pred_pos_y3, 'y3', 'Y3 Position')
        ]
        
        for i, (true_data, pred_data, var_name, title) in enumerate(pos_data):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            ax.plot(time_points, true_data, label=f'True {var_name}', linewidth=2, alpha=0.8)
            ax.plot(time_points, pred_data, label=f'Pred {var_name}', linestyle='--', linewidth=2, alpha=0.8)
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Position (m)')
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{method_name.lower()}_position_predictions.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
    else:
        print(f"[DEBUG] Handling 3D data or insufficient features for integration")
        # Fallback for 3D data or insufficient features
        fig1, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig1.suptitle(f'{method_name} - Direct Acceleration Comparison', fontsize=16)
        
        # Plot direct acceleration comparison for first few samples
        max_samples = min(4, X_np.shape[0])
        for i in range(max_samples):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            if is_3d and X_np.shape[1] > 1:
                # For 3D data, plot time series
                time_data = X_np[i, :, -1] if X_np.shape[2] >= 10 else np.arange(X_np.shape[1])
                for j, name in enumerate(output_names[:min(4, y_np.shape[2] if len(y_np.shape) > 2 else y_np.shape[1])]):
                    true_vals = y_np[i, :, j] if len(y_np.shape) > 2 else [y_np[i, j]]
                    pred_vals = y_pred_np[i, :, j] if len(y_pred_np.shape) > 2 else [y_pred_np[i, j]]
                    ax.plot(time_data, true_vals, label=f'True {name}', linewidth=2, alpha=0.8)
                    ax.plot(time_data, pred_vals, label=f'Pred {name}', linestyle='--', linewidth=2, alpha=0.8)
            else:
                # For 2D data, create bar comparison
                x_pos = np.arange(len(output_names))
                width = 0.35
                true_vals = y_np[i, :len(output_names)]
                pred_vals = y_pred_np[i, :len(output_names)]
                
                ax.bar(x_pos - width/2, true_vals, width, label='True', alpha=0.8)
                ax.bar(x_pos + width/2, pred_vals, width, label='Predicted', alpha=0.8)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(output_names)
            
            ax.set_xlabel('Time (s)' if is_3d else 'Component')
            ax.set_ylabel('Acceleration (m/s²)')
            ax.set_title(f'Sample {i}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(max_samples, 4):
            row, col = i // 2, i % 2
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{method_name.lower()}_acceleration_predictions.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot: Phase Space Analysis using integrated trajectories
    print(f"[DEBUG] Creating phase space analysis...")
    
    # Use integrated trajectory data if available (for 2D case with integration)
    has_integration_data = (not is_3d and X_np.shape[1] >= 10 and 
                           'pred_pos_x2' in locals() and 'pred_pos_y2' in locals() and
                           'pred_pos_x3' in locals() and 'pred_pos_y3' in locals())
    
    print(f"[DEBUG] Integration data available: {has_integration_data}")
    print(f"[DEBUG] - is_3d: {is_3d}")
    print(f"[DEBUG] - X_np.shape[1] >= 10: {X_np.shape[1] >= 10}")
    print(f"[DEBUG] - pred_pos_x2 in locals: {'pred_pos_x2' in locals()}")
    
    if has_integration_data:
        # Use integrated trajectory data
        fig2, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig2.suptitle(f'{method_name} - Phase Space Analysis (Integrated Trajectories)', fontsize=16)
        
        phase_data = [
            (true_pos_x2, true_vel_x2, pred_pos_x2, pred_vel_x2, 'Underhang X (Radial)'),
            (true_pos_y2, true_vel_y2, pred_pos_y2, pred_vel_y2, 'Underhang Y (Tangential)'),
            (true_pos_x3, true_vel_x3, pred_pos_x3, pred_vel_x3, 'Overhang X (Radial)'),
            (true_pos_y3, true_vel_y3, pred_pos_y3, pred_vel_y3, 'Overhang Y (Tangential)')
        ]
        
        for i, (true_pos, true_vel, pred_pos, pred_vel, title) in enumerate(phase_data):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            # Plot true trajectory
            ax.plot(true_pos, true_vel, 'b-', label='True Trajectory', linewidth=2, alpha=0.7)
            ax.plot(pred_pos, pred_vel, 'r--', label='Predicted Trajectory', linewidth=2, alpha=0.7)
            
            # Mark start and end points
            ax.plot(true_pos[0], true_vel[0], 'go', markersize=8, label='Start (True)')
            ax.plot(true_pos[-1], true_vel[-1], 'bs', markersize=8, label='End (True)')
            ax.plot(pred_pos[0], pred_vel[0], 'ro', markersize=6, label='Start (Pred)')
            ax.plot(pred_pos[-1], pred_vel[-1], 'rs', markersize=6, label='End (Pred)')
            
            ax.set_xlabel('Position (m)')
            ax.set_ylabel('Velocity (m/s)')
            ax.set_title(f'{title} Phase Space')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{method_name.lower()}_phase_space.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
    else:
        # Fallback for other data types
        fig2, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig2.suptitle(f'{method_name} - Phase Space Analysis', fontsize=16)
        
        # Bearing components to analyze
        bearings = [
            ('Underhang Radial', 4, 0),    # x2 position vs x2_dot velocity
            ('Underhang Tangential', 5, 1), # y2 position vs y2_dot velocity  
            ('Overhang Radial', 6, 2),      # x3 position vs x3_dot velocity
            ('Overhang Tangential', 7, 3)   # y3 position vs y3_dot velocity
        ]
        
        for i, (bearing_name, pos_idx, vel_idx) in enumerate(bearings):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            if is_3d and X_np.shape[1] > 1:
                # 3D data: [samples, time_steps, features]
                sample_idx = 0
                if sample_idx < X_np.shape[0] and pos_idx < X_np.shape[2] and vel_idx < X_np.shape[2]:
                    positions = X_np[sample_idx, :, pos_idx]
                    velocities = X_np[sample_idx, :, vel_idx]
                    
                    # Create phase space plot
                    ax.plot(positions, velocities, 'b-', linewidth=1, alpha=0.7, label='True Trajectory')
                    ax.plot(positions[0], velocities[0], 'go', markersize=8, label='Start')
                    ax.plot(positions[-1], velocities[-1], 'ro', markersize=8, label='End')
                    
                    ax.set_xlabel('Position (m)')
                    ax.set_ylabel('Velocity (m/s)')
                    ax.set_title(f'{bearing_name} Phase Space')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
            else:
                # 2D data: plot all points
                if pos_idx < X_np.shape[1] and vel_idx < X_np.shape[1]:
                    positions = X_np[:, pos_idx]
                    velocities = X_np[:, vel_idx]
                    
                    # Create scatter plot colored by time/order
                    scatter = ax.scatter(positions, velocities, c=np.arange(len(positions)), 
                                       cmap='viridis', alpha=0.6, s=20)
                    plt.colorbar(scatter, ax=ax, label='Time Index')
                    
                    ax.set_xlabel('Position (m)')
                    ax.set_ylabel('Velocity (m/s)')
                    ax.set_title(f'{bearing_name} Phase Space')
                    ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{method_name.lower()}_phase_space.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot: Orbit Plots using integrated trajectories
    print(f"[DEBUG] Creating orbit plots...")
    
    if has_integration_data:
        # Use integrated trajectory data for proper orbits
        fig3, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig3.suptitle(f'{method_name} - Bearing Orbits (Integrated Trajectories)', fontsize=16)
        
        orbit_data = [
            (true_pos_x2, true_pos_y2, pred_pos_x2, pred_pos_y2, 'Underhang Bearing (X2-Y2)'),
            (true_pos_x3, true_pos_y3, pred_pos_x3, pred_pos_y3, 'Overhang Bearing (X3-Y3)'),
            (true_pos_x2, true_pos_x3, pred_pos_x2, pred_pos_x3, 'System X-Positions (X2-X3)'),
            (true_pos_y2, true_pos_y3, pred_pos_y2, pred_pos_y3, 'System Y-Positions (Y2-Y3)')
        ]
        
        for i, (true_x, true_y, pred_x, pred_y, title) in enumerate(orbit_data):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            # Plot true and predicted orbits
            ax.plot(true_x, true_y, 'b-', label='True Orbit', linewidth=2, alpha=0.7)
            ax.plot(pred_x, pred_y, 'r--', label='Predicted Orbit', linewidth=2, alpha=0.7)
            
            # Mark start and end points
            ax.plot(true_x[0], true_y[0], 'go', markersize=8, label='Start (True)')
            ax.plot(true_x[-1], true_y[-1], 'bs', markersize=8, label='End (True)')
            ax.plot(pred_x[0], pred_y[0], 'ro', markersize=6, label='Start (Pred)')
            ax.plot(pred_x[-1], pred_y[-1], 'rs', markersize=6, label='End (Pred)')
            
            ax.set_xlabel('X Position (m)')
            ax.set_ylabel('Y Position (m)')
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{method_name.lower()}_orbits.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot: 3D Trajectory using integrated data
        print(f"[DEBUG] Creating 3D trajectory plot...")
        fig4 = plt.figure(figsize=(15, 10))
        ax = fig4.add_subplot(111, projection='3d')
        
        # Plot true 3D trajectory
        ax.plot(true_pos_x2, true_pos_y2, true_vel_x2, 'b-', 
               label='True Trajectory', linewidth=2, alpha=0.8)
        ax.plot(pred_pos_x2, pred_pos_y2, pred_vel_x2, 'r--', 
               label='Predicted Trajectory', linewidth=2, alpha=0.8)
        
        # Mark start and end points
        ax.scatter(true_pos_x2[0], true_pos_y2[0], true_vel_x2[0], 
                  c='green', s=100, label='Start (True)')
        ax.scatter(true_pos_x2[-1], true_pos_y2[-1], true_vel_x2[-1], 
                  c='blue', s=100, label='End (True)')
        ax.scatter(pred_pos_x2[0], pred_pos_y2[0], pred_vel_x2[0], 
                  c='orange', s=80, label='Start (Pred)')
        ax.scatter(pred_pos_x2[-1], pred_pos_y2[-1], pred_vel_x2[-1], 
                  c='red', s=80, label='End (Pred)')
        
        ax.set_xlabel('X2 Position (m)')
        ax.set_ylabel('Y2 Position (m)')
        ax.set_zlabel('X2 Velocity (m/s)')
        ax.set_title(f'{method_name} - 3D Trajectory (Underhang Bearing)')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{method_name.lower()}_3d_trajectory.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
    else:
        # Fallback for other data types
        print(f"[DEBUG] Using fallback orbit and 3D plots for non-integrated data")
        
        # Orbit plots fallback
        fig3, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig3.suptitle(f'{method_name} - Bearing Orbits', fontsize=16)
        
        orbits = [
            ('Underhang Bearing', 4, 5),  # x2 vs y2
            ('Overhang Bearing', 6, 7),   # x3 vs y3
            ('System X-Positions', 4, 6), # x2 vs x3
            ('System Y-Positions', 5, 7)  # y2 vs y3
        ]
        
        for i, (orbit_name, x_idx, y_idx) in enumerate(orbits):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            if is_3d and X_np.shape[1] > 1:
                # 3D data: plot time series as orbit
                sample_idx = 0
                if sample_idx < X_np.shape[0] and x_idx < X_np.shape[2] and y_idx < X_np.shape[2]:
                    x_pos = X_np[sample_idx, :, x_idx]
                    y_pos = X_np[sample_idx, :, y_idx]
                    
                    ax.plot(x_pos, y_pos, 'b-', linewidth=2, alpha=0.7, label='True Orbit')
                    ax.plot(x_pos[0], y_pos[0], 'go', markersize=8, label='Start')
                    ax.plot(x_pos[-1], y_pos[-1], 'ro', markersize=8, label='End')
                    
                    ax.set_xlabel('X Position (m)')
                    ax.set_ylabel('Y Position (m)')
                    ax.set_title(f'{orbit_name} Orbit')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    ax.axis('equal')
            else:
                # 2D data: plot all points as trajectory
                if x_idx < X_np.shape[1] and y_idx < X_np.shape[1]:
                    x_pos = X_np[:, x_idx]
                    y_pos = X_np[:, y_idx]
                    
                    # Sort by time if available
                    if X_np.shape[1] >= 10:  # Has time column
                        time_col = X_np[:, -1]
                        sort_idx = np.argsort(time_col)
                        x_pos = x_pos[sort_idx]
                        y_pos = y_pos[sort_idx]
                    
                    ax.plot(x_pos, y_pos, 'b-', linewidth=1, alpha=0.7, label='Trajectory')
                    ax.plot(x_pos[0], y_pos[0], 'go', markersize=8, label='Start')
                    ax.plot(x_pos[-1], y_pos[-1], 'ro', markersize=8, label='End')
                    
                    ax.set_xlabel('X Position (m)')
                    ax.set_ylabel('Y Position (m)')
                    ax.set_title(f'{orbit_name} Orbit')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    ax.axis('equal')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{method_name.lower()}_orbits.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3D Trajectory fallback
        fig4 = plt.figure(figsize=(15, 10))
        ax = fig4.add_subplot(111, projection='3d')
        
        if is_3d and X_np.shape[1] > 1:
            # 3D data: plot actual trajectory
            sample_idx = 0
            if sample_idx < X_np.shape[0] and X_np.shape[2] >= 8:
                x_pos = X_np[sample_idx, :, 4]  # x2
                y_pos = X_np[sample_idx, :, 5]  # y2  
                z_pos = X_np[sample_idx, :, 0]  # x2_dot
                
                ax.plot(x_pos, y_pos, z_pos, 'b-', linewidth=2, alpha=0.8, label='True Trajectory')
                ax.scatter(x_pos[0], y_pos[0], z_pos[0], c='green', s=100, label='Start')
                ax.scatter(x_pos[-1], y_pos[-1], z_pos[-1], c='red', s=100, label='End')
                
                ax.set_xlabel('X Position (m)')
                ax.set_ylabel('Y Position (m)')
                ax.set_zlabel('X Velocity (m/s)')
                ax.set_title(f'{method_name} - 3D Trajectory (Underhang Bearing)')
                ax.legend()
        else:
            # 2D data: create 3D trajectory from available data
            if X_np.shape[1] >= 8:
                # Sort by time
                if X_np.shape[1] >= 10:
                    time_col = X_np[:, -1]
                    sort_idx = np.argsort(time_col)
                    X_sorted = X_np[sort_idx]
                else:
                    X_sorted = X_np
                
                x_pos = X_sorted[:, 4]  # x2
                y_pos = X_sorted[:, 5]  # y2  
                z_pos = X_sorted[:, 0]  # x2_dot
                
                ax.plot(x_pos, y_pos, z_pos, 'b-', linewidth=2, alpha=0.8, label='True Trajectory')
                ax.scatter(x_pos[0], y_pos[0], z_pos[0], c='green', s=100, label='Start')
                ax.scatter(x_pos[-1], y_pos[-1], z_pos[-1], c='red', s=100, label='End')
                
                ax.set_xlabel('X Position (m)')
                ax.set_ylabel('Y Position (m)')
                ax.set_zlabel('X Velocity (m/s)')
                ax.set_title(f'{method_name} - 3D Trajectory (Underhang Bearing)')
                ax.legend()
            else:
                ax.text(0, 0, 0, 'Insufficient data for 3D plot', 
                       ha='center', va='center', fontsize=12)
                ax.set_title(f'{method_name} - 3D Trajectory (Insufficient Data)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{method_name.lower()}_3d_trajectory.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"[DEBUG] Trajectory plots saved to {output_dir}")
    print(f"Generated plots:")
    if has_integration_data:
        print(f"  - {method_name.lower()}_acceleration_predictions.png")
        print(f"  - {method_name.lower()}_velocity_predictions.png")
        print(f"  - {method_name.lower()}_position_predictions.png")
    else:
        print(f"  - {method_name.lower()}_acceleration_predictions.png")
    print(f"  - {method_name.lower()}_phase_space.png") 
    print(f"  - {method_name.lower()}_orbits.png")
    print(f"  - {method_name.lower()}_3d_trajectory.png")


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add common command-line arguments to an argument parser.
    
    Args:
        parser: Argument parser to add arguments to
        
    Returns:
        Updated argument parser
    """
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=2000, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--max-samples', type=int, default=1000, help='Maximum number of samples to use')
    
    # Early stopping and scheduling
    parser.add_argument('--early-patience', type=int, default=200, help='Early stopping patience')
    parser.add_argument('--lr-patience', type=int, default=50, help='Learning rate scheduler patience')
    parser.add_argument('--min-delta', type=float, default=1e-7, help='Minimum improvement for early stopping')
    
    # PINN architecture
    parser.add_argument('--hidden-layers', type=int, nargs='+', default=[128, 128, 64], 
                       help='Hidden layer sizes')
    parser.add_argument('--activation', type=str, default='tanh', 
                       choices=['tanh', 'relu', 'leaky_relu', 'elu', 'selu', 'gelu', 'sigmoid', 'swish'], 
                       help='Activation function')
    parser.add_argument('--dropout-rate', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--init-method', type=str, default='xavier_normal', 
                       choices=['xavier_normal', 'xavier_uniform', 'kaiming_normal', 'kaiming_uniform'],
                       help='Weight initialization method')
    
    # Output and device
    parser.add_argument('--output-dir', type=str, help='Output directory (auto-generated if not specified)')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], help='Device to use')
    
    return parser


def parse_common_arguments(args: argparse.Namespace) -> Dict:
    """
    Parse common arguments and return configuration dictionary.
    
    Args:
        args: Parsed arguments
        
    Returns:
        Configuration dictionary
    """
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'max_samples': args.max_samples,
        'early_patience': args.early_patience,
        'lr_patience': args.lr_patience,
        'min_delta': args.min_delta,
        'hidden_layers': args.hidden_layers,
        'activation': args.activation,
        'dropout_rate': args.dropout_rate,
        'init_method': args.init_method,
        'output_dir': args.output_dir,
        'device': args.device
    }
    
    return config


def print_training_config(config: Dict, method_name: str, method_params: Optional[Dict] = None):
    """
    Print training configuration.
    
    Args:
        config: Training configuration
        method_name: Name of the training method
        method_params: Method-specific parameters
    """
    print(f"\n{'='*60}")
    print(f"Starting {method_name} training with:")
    print(f"{'='*60}")
    
    print("Training Hyperparameters:")
    print(f"  - Epochs: {config['epochs']}")
    print(f"  - Batch size: {config['batch_size']}")
    print(f"  - Learning rate: {config['learning_rate']}")
    print(f"  - Max samples: {config['max_samples']}")
    print(f"  - Early stopping patience: {config['early_patience']}")
    print(f"  - LR scheduler patience: {config['lr_patience']}")
    
    print("\nPINN Architecture:")
    print(f"  - Hidden layers: {config['hidden_layers']}")
    print(f"  - Activation: {config['activation']}")
    print(f"  - Dropout rate: {config['dropout_rate']}")
    print(f"  - Init method: {config['init_method']}")
    
    if method_params:
        print(f"\n{method_name} Parameters:")
        for key, value in method_params.items():
            print(f"  - {key}: {value}")
    
    print(f"{'='*60}\n") 


def unified_data_preparation(data_path: str = "Data", max_samples: Optional[int] = None,
                           synthetic: bool = False, simulation_id: Optional[int] = None,
                           batch_size: int = 256) -> Dict:
    """
    Unified data preparation function that handles both experimental and synthetic data.
    
    This function consolidates the three separate data preparation functions into one
    that ensures consistent data structure matching basicPINNv8.py expectations.
    
    Expected data structure for basicPINNv8.py:
    - Input: 10 features [x2_dot, y2_dot, x3_dot, y3_dot, x2, y2, x3, y3, omega, t]
    - Output: 4 accelerations [x2_ddot, y2_ddot, x3_ddot, y3_ddot]
    
    Parameters:
    - data_path: Path to data directory
    - max_samples: Maximum number of samples to use
    - synthetic: If True, load synthetic data from .npz files
    - simulation_id: Specific simulation ID to load (required if synthetic=True)
    - batch_size: Batch size for data loaders
    
    Returns:
    - Dictionary containing data loaders and normalization parameters
    """
    if synthetic:
        if simulation_id is None:
            raise ValueError("simulation_id must be provided when synthetic=True")
        
        # Load synthetic data
        X, y, metadata = load_and_prepare_data(
            data_path=data_path,
            max_samples=max_samples,
            synthetic=True,
            simulation_id=simulation_id
        )
        
        # Synthetic data is already in the correct format
        # X: [x2_dot, y2_dot, x3_dot, y3_dot, x2, y2, x3, y3, omega, t] (10 features)
        # y: [x2_ddot, y2_ddot, x3_ddot, y3_ddot] (4 accelerations)
        
        # Apply max_samples if needed
        if max_samples is not None and X.size(0) > max_samples:
            idx = torch.randperm(X.size(0))[:max_samples]
            X, y = X[idx], y[idx]
        
        # Normalize inputs using dataset-derived ranges for all features (including omega, t)
        X_min = X.min(0, keepdim=True)[0]
        X_max = X.max(0, keepdim=True)[0]
        y_min = y.min(0, keepdim=True)[0]
        y_max = y.max(0, keepdim=True)[0]

        X = (X - X_min) / (X_max - X_min + 1e-12)
        y = (y - y_min) / (y_max - y_min + 1e-12)

        # Create data loaders
        ds = TensorDataset(X, y)
        n = len(ds)
        val_len = int(0.2 * n)
        train_len = n - val_len
        train_ds, val_ds = random_split(ds, [train_len, val_len])

        return {
            'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False),
            'Xmin': X_min,
            'Xmax': X_max,
            'ymin': y_min,
            'ymax': y_max,
            'metadata': metadata  # Include metadata for PINN-compatible parameters
        }
    
    else:
        # Load experimental data
        X, y, _ = load_and_prepare_data(
            data_path=data_path,
            max_samples=max_samples,
            synthetic=False
        )
        
        # Experimental data needs to be processed to match expected format
        # The data from LoadData.py has structure:
        # X: [velocities(4), displacements(4), omega(1)] + time = 10 features
        # y: [accelerations(4)] = 4 features
        
        # Verify data structure
        if X.size(-1) != 10:
            raise ValueError(f"Expected 10 input features, got {X.size(-1)}")
        if y.size(-1) != 4:
            raise ValueError(f"Expected 4 output features, got {y.size(-1)}")
        
        # Apply max_samples if needed
        if max_samples is not None and X.size(0) > max_samples:
            idx = torch.randperm(X.size(0))[:max_samples]
            X, y = X[idx], y[idx]
        
        # Normalize data to [0,1] range
        X_min, X_max = X.min(0, keepdim=True)[0], X.max(0, keepdim=True)[0]
        y_min, y_max = y.min(0, keepdim=True)[0], y.max(0, keepdim=True)[0]
        X = (X - X_min) / (X_max - X_min + 1e-12)
        y = (y - y_min) / (y_max - y_min + 1e-12)

        # Create data loaders
        ds = TensorDataset(X, y)
        n = len(ds)
        val_len = int(0.2 * n)
        train_len = n - val_len
        train_ds, val_ds = random_split(ds, [train_len, val_len])

        return {
            'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False),
            'Xmin': X_min, 'Xmax': X_max, 'ymin': y_min, 'ymax': y_max
        }


def verify_data_structure(X: torch.Tensor, y: torch.Tensor, synthetic: bool = False) -> bool:
    """
    Verify that data structure matches basicPINNv8.py expectations.
    
    Parameters:
    - X: Input tensor
    - y: Target tensor
    - synthetic: Whether using synthetic data
    
    Returns:
    - True if structure is correct, raises ValueError otherwise
    """
    # Check input dimensions
    if X.size(-1) != 10:
        raise ValueError(f"Expected 10 input features, got {X.size(-1)}")
    
    # Check output dimensions
    if y.size(-1) != 4:
        raise ValueError(f"Expected 4 output features, got {y.size(-1)}")
    
    # Check that X and y have same batch size
    if X.size(0) != y.size(0):
        raise ValueError(f"Batch sizes don't match: X={X.size(0)}, y={y.size(0)}")
    
    # For experimental data, verify the feature order
    if not synthetic:
        # The expected order from LoadData.py:
        # X: [x2_dot, y2_dot, x3_dot, y3_dot, x2, y2, x3, y3, omega, t]
        # y: [x2_ddot, y2_ddot, x3_ddot, y3_ddot]
        
        # Check that time feature is present (last feature)
        if X.size(-1) < 10:
            raise ValueError("Experimental data should have 10 features including time")
    
    print(f"Data structure verified: X={X.shape}, y={y.shape}")
    return True

def ensure_double_precision(data_dict: Dict) -> Dict:
    """
    Ensure all tensors in the data dictionary are double precision.
    
    Parameters:
    - data_dict: Dictionary containing training data with keys like 'train', 'val', 'Xmax', etc.
    
    Returns:
    - Updated dictionary with all tensors converted to double precision
    """
    # Convert normalization parameters to double precision
    for key in ['Xmax', 'Xmin', 'ymax', 'ymin']:
        if key in data_dict and data_dict[key] is not None:
            data_dict[key] = data_dict[key].double()
    
    # Convert data loaders - this is trickier since we can't modify them in place
    # The double precision conversion will happen in the training loops
    
    return data_dict


def load_best_model(method_name: str, models_dir: str = "results/best_models",
                   device: Optional[str] = None) -> Tuple[nn.Module, Dict, Dict]:
    """
    Load a trained best model along with its scaling parameters and configuration.

    Args:
        method_name: Name of the method ('relobralo', 'constant_weight', 'brdr', 'pecann')
        models_dir: Directory containing the trained models
        device: Device to load the model on (if None, uses CUDA if available, else CPU)

    Returns:
        Tuple of (model, scaling_params, config)
        - model: Loaded PyTorch model
        - scaling_params: Dictionary with Xmin, Xmax, ymin, ymax
        - config: Dictionary with method configuration

    Example:
        model, scaling, config = load_best_model('relobralo')
        # Use model for predictions
        # Use scaling for data normalization/denormalization
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    # Map simple model names to directory names
    model_name_mapping = {
        'relobralo': 'ReLoBRaLoLoss',
        'constant_weight': 'ConstantWeightLoss',
        'brdr': 'BRDRLoss',
        'pecann': 'PECANNLoss',
        'adaptive_lbpin': 'GaussianLikelihoodLoss',  # This one uses the class name
        'alpinn': 'ALPINNLoss',
        'dwpinn': 'DWPINNLoss',
        'gradnorm': 'GradNormLoss'
    }

    # Use mapped name if available, otherwise use original name
    dir_name = model_name_mapping.get(method_name, method_name)
    method_dir = os.path.join(models_dir, dir_name)

    if not os.path.exists(method_dir):
        raise FileNotFoundError(f"Model directory not found: {method_dir}")

    # Load configuration
    config_path = os.path.join(method_dir, f"{dir_name}_config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Load scaling parameters
    scaling_path = os.path.join(method_dir, f"{dir_name}_scaling.npz")
    scaling_data = np.load(scaling_path)
    scaling_params = {
        'Xmin': torch.tensor(scaling_data['Xmin'], dtype=torch.float64, device=device),
        'Xmax': torch.tensor(scaling_data['Xmax'], dtype=torch.float64, device=device),
        'ymin': torch.tensor(scaling_data['ymin'], dtype=torch.float64, device=device),
        'ymax': torch.tensor(scaling_data['ymax'], dtype=torch.float64, device=device)
    }

    # Create model architecture
    from models.basicPINNv8 import ConfigurablePINN, get_default_pinn_config

    arch_params = config['architecture_params']
    pinn_config = get_pinn_config(
        arch_params['hidden_layers'],
        arch_params['activation'],
        arch_params['dropout_rate'],
        arch_params['init_method']
    )

    # Determine if synthetic data was used
    synthetic = config['data_params'].get('synthetic', False)

    # Create parameter initialization config
    param_init_config = get_param_init_config(synthetic=synthetic)

    # Create model
    if PINN_VERSION == "v2":
        model = ConfigurablePINN(
            unmeasured_net_config=pinn_config,
            acceleration_net_config=pinn_config,
            rotor_net_config=pinn_config,
            param_init_config=param_init_config,
            enable_mass_constraints=not synthetic
        ).to(device)
    elif PINN_VERSION == "v1":
        model = ConfigurablePINN(
            unmeasured_net_config=pinn_config,
            acceleration_net_config=pinn_config,
            param_init_config=param_init_config,
            enable_mass_constraints=not synthetic
        ).to(device)
    else:
        model = ConfigurablePINN(pinn_config, pinn_config, param_init_config,
                                enable_mass_constraints=not synthetic).to(device)

    # Load trained weights
    model_path = os.path.join(method_dir, f"{dir_name}_best_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    return model, scaling_params, config


def get_available_best_models(models_dir: str = "results/best_models") -> List[str]:
    """
    Get list of available trained best models.

    Args:
        models_dir: Directory containing the trained models

    Returns:
        List of available method names
    """
    if not os.path.exists(models_dir):
        return []

    available_methods = []
    for item in os.listdir(models_dir):
        method_dir = os.path.join(models_dir, item)
        if os.path.isdir(method_dir):
            config_path = os.path.join(method_dir, f"{item}_config.json")
            if os.path.exists(config_path):
                available_methods.append(item)

    return sorted(available_methods)