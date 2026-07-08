#!/usr/bin/env python3
"""
ReLoBRaLo PINN Training Script

Implements the Relative Loss Balancing with Random Lookback (ReLoBRaLo) method.
This method uses adaptive weighting with random lookback to balance data and physics losses.

Usage:
    python relobralo_training.py [--epochs 2000] [--batch-size 256] [--lr 1e-4] [--max-samples 1000]
                                [--alpha 0.9] [--rho 0.1] [--temperature 2.0]
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
import argparse
from tqdm import tqdm
import random

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.basicPINNv8 import ConfigurablePINN, adaptive_custom_loss, get_default_pinn_config
PINN_VERSION = "dimensional"
print("Using original dimensional PINN")
from common_utils import (
    MultiLossEarlyStopping, ReduceLROnPlateau, prepare_data, load_and_prepare_data,
    get_pinn_config, get_param_init_config, setup_device, create_output_directory,
    plot_training_history, add_common_arguments, parse_common_arguments, print_training_config,
    prepare_synthetic_data, collect_raw_residuals, collect_model_parameters, 
    initialize_comprehensive_history, update_comprehensive_history,
    unified_data_preparation, ensure_double_precision, verify_data_structure, plot_trajectory_predictions
)


class ReLoBRaLoLoss(nn.Module):
    """
    ReLoBRaLo (Relative Loss Balancing with Random Lookback) loss balancing method.
    
    This method implements a relative loss balancing approach with random lookback
    to prevent overfitting to specific loss components.
    """
    
    def __init__(self, alpha: float = 1.0, rho: float = 0.1, temperature: float = 1.0, enable_mass_constraints: bool = True):
        super().__init__()
        self.alpha = alpha  # Relative loss balancing parameter
        self.rho = rho  # Random lookback parameter
        self.temperature = temperature  # Temperature for softmax
        
        # Initialize running statistics
        self.running_losses = None
        self.running_weights = None
        
        # State variables
        self.loss_history = []
        self._initialized = False
        
        # Loss component keys - conditionally include mass constraints
        if enable_mass_constraints:
            self.loss_keys = ['data', 'phys_res1', 'phys_res2', 'phys_res3', 'phys_res4', 'phys_mass1', 'phys_mass2']
        else:
            self.loss_keys = ['data', 'phys_res1', 'phys_res2', 'phys_res3', 'phys_res4']
        
        # Store current weights for logging
        self.current_weights = {key: 1.0 for key in self.loss_keys}
    
    def forward(self, model, X_batch, y_batch, X_max, X_min, y_max, y_min):
        """Calculate all loss components using adaptive_custom_loss."""
        loss_components = adaptive_custom_loss(model, X_batch, y_batch, X_max, X_min, y_max, y_min)
        return loss_components
    
    def _reset_statistics(self):
        """Reset running statistics to safe values."""
        if self.running_losses is not None:
            device = self.running_losses.device
            self.running_losses = torch.ones_like(self.running_losses) * 1e4
            self.running_weights = torch.ones_like(self.running_weights)
            print("Warning: Statistics reset due to corruption")
    
    def _initialize_statistics(self, loss_components):
        """Initialize running statistics if not already done."""
        if self.running_losses is None:
            device = loss_components[0].device
            # CRITICAL FIX: Initialize with safe finite values
            self.running_losses = torch.ones(len(loss_components), device=device) * 1e4
            self.running_weights = torch.ones(len(loss_components), device=device)
            
            # CRITICAL FIX: Ensure initial values are finite
            if not torch.isfinite(self.running_losses).all():
                self.running_losses = torch.ones(len(loss_components), device=device) * 1e4
            if not torch.isfinite(self.running_weights).all():
                self.running_weights = torch.ones(len(loss_components), device=device)
    
    def _update_weights(self, loss_components):
        """Update weights using ReLoBRaLo algorithm."""
        self._initialize_statistics(loss_components)
        
        # Convert to tensor for batch processing and validate
        current_losses = torch.stack([loss.detach() for loss in loss_components])
        
        # CRITICAL FIX: Check for invalid loss components
        if not torch.isfinite(current_losses).all():
            print(f"Warning: Invalid loss components detected: {current_losses}")
            # Replace invalid values with finite defaults
            current_losses = torch.nan_to_num(current_losses, nan=1e4, posinf=1e4, neginf=1e4)
        
        # Update running losses with exponential moving average
        self.running_losses = self.alpha * self.running_losses + (1 - self.alpha) * current_losses
        
        # CRITICAL FIX: Ensure running losses are finite
        if not torch.isfinite(self.running_losses).all():
            print(f"Warning: Invalid running losses detected: {self.running_losses}")
            # Reset running losses to safe values
            self._reset_statistics()
        
        # Random lookback: with probability rho, use a random previous state
        if random.random() < self.rho:
            # Use random lookback - perturb the running losses
            noise = torch.randn_like(self.running_losses) * 0.1
            lookback_losses = self.running_losses + noise
        else:
            lookback_losses = self.running_losses
        
        # CRITICAL FIX: Ensure lookback losses are finite
        if not torch.isfinite(lookback_losses).all():
            print(f"Warning: Invalid lookback losses detected: {lookback_losses}")
            lookback_losses = torch.ones_like(lookback_losses) * 1e4
        
        # Compute relative loss ratios with protection
        mean_loss = torch.mean(lookback_losses)
        if not torch.isfinite(mean_loss):
            print(f"Warning: Invalid mean loss: {mean_loss}")
            mean_loss = torch.tensor(1e4, device=lookback_losses.device)
        
        relative_ratios = lookback_losses / (mean_loss + 1e-8)
        
        # CRITICAL FIX: Ensure relative ratios are finite
        if not torch.isfinite(relative_ratios).all():
            print(f"Warning: Invalid relative ratios detected: {relative_ratios}")
            relative_ratios = torch.ones_like(relative_ratios)
        
        # Apply temperature scaling and softmax with protection
        logits = -relative_ratios / self.temperature
        
        # CRITICAL FIX: Ensure logits are finite before softmax
        if not torch.isfinite(logits).all():
            print(f"Warning: Invalid logits detected: {logits}")
            logits = torch.zeros_like(logits)
        
        weights = torch.softmax(logits, dim=0)
        
        # CRITICAL FIX: Ensure weights are finite and valid
        if not torch.isfinite(weights).all():
            print(f"Warning: Invalid weights detected: {weights}")
            weights = torch.ones_like(weights) / len(weights)
        
        # Normalize weights to sum to number of components
        weights = weights / torch.sum(weights) * len(weights)
        
        # CRITICAL FIX: Final validation of weights
        if not torch.isfinite(weights).all():
            print(f"Warning: Invalid normalized weights detected: {weights}")
            weights = torch.ones_like(weights)
        
        # Update running weights with exponential moving average
        self.running_weights = self.alpha * self.running_weights + (1 - self.alpha) * weights
        
        # CRITICAL FIX: Ensure running weights are finite
        if not torch.isfinite(self.running_weights).all():
            print(f"Warning: Invalid running weights detected: {self.running_weights}")
            self._reset_statistics()
        
        return self.running_weights
    
    def step(self, loss_components, optimizer, model, X_batch, y_batch):
        """
        Performs optimization step with ReLoBRaLo weight adjustment.
        
        The algorithm:
        1. Update running loss statistics
        2. Apply random lookback with probability rho
        3. Compute relative loss ratios
        4. Apply temperature-scaled softmax to get weights
        5. Optimize model parameters
        """
        # CRITICAL FIX: Validate loss components before processing
        for i, loss in enumerate(loss_components):
            if not torch.isfinite(loss):
                print(f"Warning: Invalid loss component {i}: {loss}")
                # Replace with a safe default
                loss_components[i] = torch.tensor(1e4, device=loss.device, requires_grad=True)
        
        # 1. Update weights using ReLoBRaLo algorithm
        weights = self._update_weights(loss_components)
        
        # 2. Apply weights to loss components
        weighted_losses = []
        for i, loss in enumerate(loss_components):
            weighted_loss = weights[i] * loss
            weighted_losses.append(weighted_loss)
        
        # 3. Compute total loss
        total_loss = torch.stack(weighted_losses).sum()
        
        # CRITICAL FIX: Check if total loss is valid
        if not torch.isfinite(total_loss):
            print(f"Warning: Invalid total loss: {total_loss}")
            # Skip this optimization step
            return total_loss
        
        # 4. Optimize model parameters with gradient clipping
        optimizer.zero_grad()
        total_loss.backward()
        
        # CRITICAL FIX: Add gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 5. Update current weights for logging
        # CRITICAL FIX: Handle dynamic loss components - weights array might be longer than loss_keys
        for i, key in enumerate(self.loss_keys):
            if i < len(weights):
                self.current_weights[key] = weights[i].item()
            else:
                # For synthetic data, mass constraint weights are not used
                break
        
        return total_loss


def add_method_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add method-specific command-line arguments."""
    parser.add_argument('--alpha', type=float, default=0.9,
                       help='EMA factor for statistics (default: 0.9)')
    parser.add_argument('--rho', type=float, default=0.1,
                       help='Random lookback probability (default: 0.1)')
    parser.add_argument('--temperature', type=float, default=2.0,
                       help='Temperature for softmax (default: 2.0)')
    
    # Synthetic data arguments
    parser.add_argument('--synthetic', action='store_true',
                       help='Use synthetic data instead of experimental data')
    parser.add_argument('--simulation-id', type=int, default=None,
                       help='Simulation ID to use (required if --synthetic is used)')
    parser.add_argument('--data-path', type=str, default='Data',
                       help='Path to data directory (default: Data)')
    
    return parser


def main():
    """Main training function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='ReLoBRaLo PINN Training')
    parser = add_common_arguments(parser)
    parser = add_method_arguments(parser)
    args = parser.parse_args()
    
    # Parse configuration
    config = parse_common_arguments(args)
    method_params = {
        'alpha': args.alpha,
        'rho': args.rho,
        'temperature': args.temperature
    }
    
    # Setup device and output directory
    device = setup_device(config['device'])
    output_dir, model_path = create_output_directory('relobralo_pinn', config['output_dir'])
    
    # Print configuration
    print_training_config(config, 'ReLoBRaLo PINN', method_params)
    
    # Load and prepare data
    print("Loading and preparing data...")
    
    # Use unified data preparation function
    data = unified_data_preparation(
        data_path=args.data_path,
        max_samples=config['max_samples'],
        synthetic=args.synthetic,
        simulation_id=args.simulation_id,
        batch_size=config['batch_size']
    )
    
    # Ensure all data tensors are double precision
    data = ensure_double_precision(data)
    
    # Verify data structure matches basicPINNv8.py expectations
    # Get a sample batch to verify structure
    sample_batch = next(iter(data['train']))
    X_sample, y_sample = sample_batch
    verify_data_structure(X_sample, y_sample, synthetic=args.synthetic)
    
    print(f"Data loaded successfully:")
    print(f"  - Training batches: {len(data['train'])}")
    print(f"  - Validation batches: {len(data['val'])}")
    print(f"  - Input features: {X_sample.size(-1)}")
    print(f"  - Output features: {y_sample.size(-1)}")
    
    # Create PINN configuration
    pinn_config = get_pinn_config(
        config['hidden_layers'], config['activation'], 
        config['dropout_rate'], config['init_method']
    )
    param_init_config = get_param_init_config(synthetic=args.synthetic)
    
    # Initialize model and loss function
    print("Initializing model and loss function...")
    
    # Create model with version-specific constructor
    if PINN_VERSION == "v2":
        # v2 requires 3 network configs (unmeasured, acceleration, rotor)
        model = ConfigurablePINN(
            unmeasured_net_config=pinn_config,
            acceleration_net_config=pinn_config, 
            rotor_net_config=pinn_config,  # NEW for v2
            param_init_config=param_init_config,
            enable_mass_constraints=not args.synthetic
        ).to(device)
    elif PINN_VERSION == "v1":
        # v1 requires 2 network configs (unmeasured, acceleration) 
        model = ConfigurablePINN(
            unmeasured_net_config=pinn_config,
            acceleration_net_config=pinn_config,
            param_init_config=param_init_config,
            enable_mass_constraints=not args.synthetic
        ).to(device)
    else:
        # Original dimensional PINN
        model = ConfigurablePINN(pinn_config, pinn_config, param_init_config, enable_mass_constraints=not args.synthetic).to(device)
    loss_method = ReLoBRaLoLoss(enable_mass_constraints=not args.synthetic, **method_params)
    opt = optim.Adam(model.parameters(), lr=config['learning_rate'])
    lr_sched = ReduceLROnPlateau(opt, patience=config['lr_patience'])
    
    # Initialize early stopping and comprehensive history containers
    early = MultiLossEarlyStopping(config['early_patience'], config['min_delta'], 
                                  ['data_val', 'val_total'])
    hist = initialize_comprehensive_history(synthetic=args.synthetic)
    
    # Add method-specific tracking
    hist.update({
        'weight_data': [], 'weight_phys_res1': [], 'weight_phys_res2': [],
        'weight_phys_res3': [], 'weight_phys_res4': [], 'weight_phys_mass1': [],
        'weight_phys_mass2': []
    })
    
    # Training loop
    print("Starting training...")
    for epoch in tqdm(range(config['epochs']), desc="Training"):
        # Training step
        model.train()
        tr_total = tr_data = 0.0
        # Conditionally set the number of physics components to track
        num_phys_components = 4 if args.synthetic else 6
        tr_phys_components = [0.0] * num_phys_components  # Use conditional number of physics components
        tr_raw_residuals = []
        
        for xb, yb in data['train']:
            xb, yb = xb.to(device), yb.to(device)
            
            # Ensure double precision
            xb = xb.double()
            yb = yb.double()
            
            # Get model predictions with protection
            try:
                y_pred = model(xb)
                
                # CRITICAL FIX: Check if model output is valid
                if not torch.isfinite(y_pred).all():
                    print(f"Warning: Invalid model output detected at epoch {epoch}")
                    continue
                
            except Exception as e:
                print(f"Error in model forward pass: {e}")
                continue
            
            # Get loss components and perform method-specific step
            try:
                loss_components = loss_method(model, xb, yb, data['Xmax'].to(device), 
                                            data['Xmin'].to(device), data['ymax'].to(device), 
                                            data['ymin'].to(device))
                
                # CRITICAL FIX: Check if loss components are valid
                if any(not torch.isfinite(loss) for loss in loss_components):
                    print(f"Warning: Invalid loss components detected at epoch {epoch}")
                    continue
                
                total = loss_method.step(loss_components, opt, model, xb, yb)
                
                # CRITICAL FIX: Check if total loss is valid
                if not torch.isfinite(total):
                    print(f"Warning: Invalid total loss at epoch {epoch}: {total}")
                    continue
                
            except Exception as e:
                print(f"Error in loss computation: {e}")
                continue
            
            # Collect raw residuals for this batch
            try:
                raw_residuals = collect_raw_residuals(model, xb, yb, y_pred, data, device)
                tr_raw_residuals.append(raw_residuals)
            except Exception as e:
                print(f"Error in residual collection: {e}")
                continue
            
            # Accumulate losses
            tr_total += total.item()
            tr_data += loss_components[0].item()
            for i in range(num_phys_components):
                tr_phys_components[i] += loss_components[i+1].item()
        
        # Average over batches
        n_train = len(data['train'])
        if n_train == 0:
            print(f"Warning: No valid training batches at epoch {epoch}")
            continue
            
        epoch_metrics = {
            'train_total': tr_total / n_train,
            'data_train': tr_data / n_train,
            'phys_train': sum(tr_phys_components) / (num_phys_components * n_train)
        }
        
        # Individual physics components
        phys_keys = ['phys_res1', 'phys_res2', 'phys_res3', 'phys_res4']
        if not args.synthetic:
            phys_keys.extend(['phys_mass1', 'phys_mass2'])
        for i, key in enumerate(phys_keys):
            epoch_metrics[f'{key}_train'] = tr_phys_components[i] / n_train
        
        # Average raw residuals across batches
        avg_raw_residuals = {
            'data_residuals': {},
            'physical_residuals': {}
        }
        
        # Average data residuals
        for key in ['x2_ddot_mae', 'y2_ddot_mae', 'x3_ddot_mae', 'y3_ddot_mae', 'total_mae']:
            values = [res['data_residuals'][key] for res in tr_raw_residuals]
            avg_raw_residuals['data_residuals'][key] = np.mean(values)
        
        # Average physical residuals
        phys_residual_keys = ['res1_mean', 'res2_mean', 'res3_mean', 'res4_mean']
        if not args.synthetic:
            phys_residual_keys.extend(['res_mass1_mean', 'res_mass2_mean'])
        
        for key in phys_residual_keys:
            values = [res['physical_residuals'][key] for res in tr_raw_residuals]
            avg_raw_residuals['physical_residuals'][key] = np.mean(values)
        
        # Add std keys for physics residuals
        phys_residual_std_keys = ['res1_std', 'res2_std', 'res3_std', 'res4_std']
        if not args.synthetic:
            phys_residual_std_keys.extend(['res_mass1_std', 'res_mass2_std'])
        
        for key in phys_residual_std_keys:
            values = [res['physical_residuals'][key] for res in tr_raw_residuals]
            avg_raw_residuals['physical_residuals'][key] = np.mean(values)
        
        epoch_metrics['raw_residuals'] = avg_raw_residuals
        
        # Update comprehensive history for training
        update_comprehensive_history(hist, epoch_metrics, model, is_training=True)
        
        # Store current weights
        for key, value in loss_method.current_weights.items():
            hist[f'weight_{key}'].append(value)
        
        # Validation step
        model.eval()
        val_total = val_data = 0.0
        val_phys_components = [0.0] * num_phys_components # Use the same conditional number
        val_raw_residuals = []
        
        with torch.no_grad():
            for xb, yb in data['val']:
                xb, yb = xb.to(device), yb.to(device)
                
                # Ensure double precision
                xb = xb.double()
                yb = yb.double()
                
                # Get model predictions with protection
                try:
                    y_pred = model(xb)
                    
                    # CRITICAL FIX: Check if model output is valid
                    if not torch.isfinite(y_pred).all():
                        print(f"Warning: Invalid model output in validation at epoch {epoch}")
                        continue
                    
                except Exception as e:
                    print(f"Error in model forward pass during validation: {e}")
                    continue
                
                try:
                    loss_components = loss_method(model, xb, yb, data['Xmax'].to(device),
                                               data['Xmin'].to(device), data['ymax'].to(device),
                                               data['ymin'].to(device))
                    
                    # CRITICAL FIX: Check if loss components are valid
                    if any(not torch.isfinite(loss) for loss in loss_components):
                        print(f"Warning: Invalid loss components in validation at epoch {epoch}")
                        continue
                    
                    # Calculate total loss with current weights (no weight update in validation)
                    weighted_total = torch.sum(torch.stack([loss_method.running_weights[i] * loss_components[i] for i in range(len(loss_components))]))
                    
                    # CRITICAL FIX: Check if weighted total is valid
                    if not torch.isfinite(weighted_total):
                        print(f"Warning: Invalid weighted total in validation at epoch {epoch}: {weighted_total}")
                        continue
                    
                except Exception as e:
                    print(f"Error in loss computation during validation: {e}")
                    continue
                
                val_total += weighted_total.item()
                val_data += loss_components[0].item()
                for i in range(num_phys_components):
                    val_phys_components[i] += loss_components[i+1].item()
                
                # Collect raw residuals for validation
                try:
                    raw_residuals = collect_raw_residuals(model, xb, yb, y_pred, data, device)
                    val_raw_residuals.append(raw_residuals)
                except Exception as e:
                    print(f"Error in residual collection during validation: {e}")
                    continue
        
        # Average over validation batches
        n_val = len(data['val'])
        if n_val == 0:
            print(f"Warning: No valid validation batches at epoch {epoch}")
            continue
            
        val_epoch_metrics = {
            'val_total': val_total / n_val,
            'data_val': val_data / n_val,
            'phys_val': sum(val_phys_components) / (num_phys_components * n_val)
        }
        
        # Individual validation physics components
        for i, key in enumerate(phys_keys):
            val_epoch_metrics[f'{key}_val'] = val_phys_components[i] / n_val # Use the same conditional phys_keys list
        
        # Average validation raw residuals
        val_avg_raw_residuals = {
            'data_residuals': {},
            'physical_residuals': {}
        }
        
        # Average validation data residuals
        for key in ['x2_ddot_mae', 'y2_ddot_mae', 'x3_ddot_mae', 'y3_ddot_mae', 'total_mae']:
            values = [res['data_residuals'][key] for res in val_raw_residuals]
            val_avg_raw_residuals['data_residuals'][key] = np.mean(values)
        
        # Average validation physical residuals
        phys_residual_keys = ['res1_mean', 'res2_mean', 'res3_mean', 'res4_mean']
        if not args.synthetic:
            phys_residual_keys.extend(['res_mass1_mean', 'res_mass2_mean'])
        
        for key in phys_residual_keys:
            values = [res['physical_residuals'][key] for res in val_raw_residuals]
            val_avg_raw_residuals['physical_residuals'][key] = np.mean(values)
        
        # Add std keys for physics residuals
        phys_residual_std_keys = ['res1_std', 'res2_std', 'res3_std', 'res4_std']
        if not args.synthetic:
            phys_residual_std_keys.extend(['res_mass1_std', 'res_mass2_std'])
        
        for key in phys_residual_std_keys:
            values = [res['physical_residuals'][key] for res in val_raw_residuals]
            val_avg_raw_residuals['physical_residuals'][key] = np.mean(values)
        
        val_epoch_metrics['raw_residuals'] = val_avg_raw_residuals
        
        # Update comprehensive history for validation
        update_comprehensive_history(hist, val_epoch_metrics, model, is_training=False)
        
        # Learning rate scheduling and early stopping
        current_val_loss = hist['val_total'][-1]
        
        # CRITICAL FIX: Check if validation loss is valid before using it
        if not np.isfinite(current_val_loss):
            print(f"Warning: Invalid validation loss at epoch {epoch}: {current_val_loss}")
            # Skip this epoch for scheduling/stopping decisions
            continue
        
        lr_sched.step(current_val_loss)
        
        if current_val_loss < min(hist['val_total']):
            torch.save(model.state_dict(), model_path)
        
        early({'data_val': hist['data_val'][-1], 'val_total': hist['val_total'][-1]})
        if early.early_stop:
            break
    
    # Save results and create plots
    print("Saving results and creating plots...")
    np.savez(os.path.join(output_dir, 'relobralo_history.npz'), **hist)
    plot_training_history(hist, output_dir, 'ReLoBRaLo PINN', 
                         include_weights=True, include_method_params=True)
    
    # Create trajectory and phase space plots
    print("Creating trajectory and phase space visualizations...")
    plot_trajectory_predictions(model, data, output_dir, 'ReLoBRaLo PINN', device=device)
    
    # Always save the final model for parameter extraction
    torch.save(model.state_dict(), model_path)
    
    # Save final parameters for comparison
    final_params = collect_model_parameters(model)
    np.savez(os.path.join(output_dir, 'relobralo_parameters.npz'), **final_params)
    
    print(f'Training complete. Model saved to {model_path}')
    print(f'Results saved to {output_dir}')
    print(f'Final parameters: {final_params}')
    print(f'Final weights: {loss_method.current_weights}')


if __name__ == '__main__':
    main() 