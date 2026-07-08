#!/usr/bin/env python3
"""
BRDR PINN Training Script

Implements the Balanced Residual Distribution Regularization (BRDR) method.
This method uses adaptive weighting based on residual distribution statistics
to balance data and physics losses.

Usage:
    python brdr_training.py [--epochs 2000] [--batch-size 256] [--lr 1e-4] [--max-samples 1000]
                           [--beta-c 0.999] [--beta-w 0.999] [--epsilon 1e-8]
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
import argparse
from tqdm import tqdm
from typing import Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.basicPINNv8 import ConfigurablePINN, adaptive_custom_loss, get_default_pinn_config
PINN_VERSION = "dimensional"
print("Using original dimensional PINN")
from common_utils import (
    MultiLossEarlyStopping, ReduceLROnPlateau, 
    get_pinn_config, get_param_init_config, setup_device, create_output_directory,
    plot_training_history, add_common_arguments, parse_common_arguments, print_training_config,
    collect_raw_residuals, collect_model_parameters, 
    initialize_comprehensive_history, update_comprehensive_history,
    unified_data_preparation, ensure_double_precision, verify_data_structure, plot_trajectory_predictions
)


class BRDRLoss(nn.Module):
    """
    BRDR (Balanced Relative Dynamic Reweighting) loss balancing method.
    
    This method implements a balanced relative dynamic reweighting approach
    that adaptively adjusts loss weights based on relative loss ratios.
    """
    
    def __init__(self, beta_c: float = 0.99, beta_w: float = 0.99, epsilon: float = 1e-8, enable_mass_constraints: bool = True):
        super().__init__()
        self.beta_c = beta_c  # Exponential moving average factor for constraints
        self.beta_w = beta_w  # Exponential moving average factor for weights
        self.epsilon = epsilon  # Small constant to prevent division by zero
        
        # Initialize state variables
        self.point_weights = {}
        self.ema_residuals_sq_sq = {}
        self._initialized = False
        
        # Adaptive scale factor
        self.s = 1.0
        
        # State variables
        self.running_constraints = None
        self.running_weights = None
        self._initialized = False
        
        # Store normalization parameters for dimensionless PINN compatibility
        self.X_max = None
        self.X_min = None
        self.y_max = None
        self.y_min = None
        
        # Loss component keys - conditionally include mass constraints
        if enable_mass_constraints:
            self.loss_keys = ['data', 'phys_res1', 'phys_res2', 'phys_res3', 'phys_res4', 'phys_mass1', 'phys_mass2']
        else:
            self.loss_keys = ['data', 'phys_res1', 'phys_res2', 'phys_res3', 'phys_res4']
        
        # Store current weights for logging
        self.current_weights = {key: 1.0 for key in self.loss_keys}
        
    def forward(self, model, X_batch, y_batch, X_max, X_min, y_max, y_min):
        """Calculate all loss components using adaptive_custom_loss."""
        # Store normalization parameters for later use in _get_raw_residuals
        self.X_max = X_max
        self.X_min = X_min
        self.y_max = y_max
        self.y_min = y_min
        
        loss_components = adaptive_custom_loss(model, X_batch, y_batch, X_max, X_min, y_max, y_min)
        return loss_components
    
    def _get_raw_residuals(self, model, X_batch, y_batch):
        """Get raw residuals for point-wise weighting."""
        # Get model predictions
        y_pred = model(X_batch)
        
        # Get raw residuals from model - need to pass normalization parameters for dimensionless PINN
        try:
            # Try new format with normalization parameters
            residuals_result = model.compute_residuals(X_batch, y_pred, 
                                                     self.X_max, self.X_min, 
                                                     self.y_max, self.y_min)
            
            # Handle new dimensionless PINN format (dictionary) vs old format (tuple)
            if isinstance(residuals_result, dict):
                residuals = residuals_result['residuals']
            else:
                residuals = residuals_result
        except (TypeError, AttributeError):
            # Fallback to old format without normalization parameters
            residuals_result = model.compute_residuals(X_batch, y_pred)
            if isinstance(residuals_result, dict):
                residuals = residuals_result['residuals']
            else:
                residuals = residuals_result
            
        residual1, residual2, residual3, residual4, residualMass1, residualMass2 = residuals
        
        # Handle NaNs and clip extreme values
        residual1 = torch.nan_to_num(residual1, nan=0.0, posinf=1e10, neginf=-1e10)
        residual2 = torch.nan_to_num(residual2, nan=0.0, posinf=1e10, neginf=-1e10)
        residual3 = torch.nan_to_num(residual3, nan=0.0, posinf=1e10, neginf=-1e10)
        residual4 = torch.nan_to_num(residual4, nan=0.0, posinf=1e10, neginf=-1e10)
        residualMass1 = torch.nan_to_num(residualMass1, nan=0.0, posinf=1e10, neginf=-1e10)
        residualMass2 = torch.nan_to_num(residualMass2, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Clip residuals
        residual1 = torch.clamp(residual1, -1e8, 1e8)
        residual2 = torch.clamp(residual2, -1e8, 1e8)
        residual3 = torch.clamp(residual3, -1e8, 1e8)
        residual4 = torch.clamp(residual4, -1e8, 1e8)
        residualMass1 = torch.clamp(residualMass1, -1e8, 1e8)
        residualMass2 = torch.clamp(residualMass2, -1e8, 1e8)
        
        # CRITICAL FIX: Ensure all residuals are properly shaped as batched tensors
        # Mass constraint residuals might be scalars, so we need to expand them to match batch size
        batch_size = X_batch.shape[0]
        
        # Ensure all residuals have the same shape as the batch
        if residual1.dim() == 0:
            residual1 = residual1.expand(batch_size)
        if residual2.dim() == 0:
            residual2 = residual2.expand(batch_size)
        if residual3.dim() == 0:
            residual3 = residual3.expand(batch_size)
        if residual4.dim() == 0:
            residual4 = residual4.expand(batch_size)
        if residualMass1.dim() == 0:
            residualMass1 = residualMass1.expand(batch_size)
        if residualMass2.dim() == 0:
            residualMass2 = residualMass2.expand(batch_size)
        
        return {
            'data_res': y_pred - y_batch,  # Data residual
            'phys_res1_res': residual1,
            'phys_res2_res': residual2,
            'phys_res3_res': residual3,
            'phys_res4_res': residual4,
            'phys_mass1_res': residualMass1,
            'phys_mass2_res': residualMass2
        }
    
    def step(self, loss_components, optimizer, model, X_batch, y_batch):
        """
        Performs optimization step with BRDR point-wise weight adjustment.
        
        The algorithm:
        1. Get raw residuals for all components
        2. Update point-wise statistics
        3. Calculate inverse residual decay rates
        4. Apply global normalization
        5. Update point-wise weights
        6. Calculate weighted losses
        7. Apply adaptive scale factor
        """
        # 1. Get raw residuals for all components
        residuals = self._get_raw_residuals(model, X_batch, y_batch)
        
        # 2. Initialize state variables on first step
        if not self._initialized:
            for key in self.loss_keys:
                current_residual = residuals[f'{key}_res']
                self.point_weights[key] = torch.ones_like(current_residual, device=current_residual.device)
                self.ema_residuals_sq_sq[key] = torch.zeros_like(current_residual, device=current_residual.device)
            self._initialized = True
        
        # 3. Check if batch size changed and resize tensors if needed
        batch_size = X_batch.shape[0]
        for key in self.loss_keys:
            current_residual = residuals[f'{key}_res']
            expected_size = current_residual.shape

            # Check if point_weights need to be resized
            if self.point_weights[key].shape != expected_size:
                print(f"Warning: Resizing point_weights for {key} from {self.point_weights[key].shape} to {expected_size}")
                # Ensure the tensor is properly sized and on the correct device
                self.point_weights[key] = torch.ones_like(current_residual, device=current_residual.device)
                self.ema_residuals_sq_sq[key] = torch.zeros_like(current_residual, device=current_residual.device)

            # Additional safety check: ensure tensors are not empty and have correct shape
            if self.point_weights[key].numel() == 0 or self.point_weights[key].shape != expected_size:
                print(f"Critical: Fixing empty or malformed tensor for {key}")
                self.point_weights[key] = torch.ones(expected_size, device=current_residual.device)
                self.ema_residuals_sq_sq[key] = torch.zeros(expected_size, device=current_residual.device)
        
        # 4. Store all irdr values for global normalization
        all_irdr_values = []
        
        # 5. Process each loss component
        for key in self.loss_keys:
            current_residual = residuals[f'{key}_res']
            
            # Safety check: ensure residual has proper shape
            if current_residual.dim() == 0:
                # If it's a scalar, expand it to match batch size
                batch_size = X_batch.shape[0]
                current_residual = current_residual.expand(batch_size)
            
            with torch.no_grad():
                current_residual_sq = current_residual.detach()**2
                
                # Update EMA of R^4 for this component
                self.ema_residuals_sq_sq[key].copy_(
                    self.beta_c * self.ema_residuals_sq_sq[key] +
                    (1 - self.beta_c) * (current_residual_sq**2)
                )
                
                # Calculate inverse residual decay rate (irdr) for this component
                irdr = current_residual_sq / (torch.sqrt(self.ema_residuals_sq_sq[key]) + self.epsilon)
                
                # Safety check: ensure irdr has proper shape before appending
                if irdr.dim() == 0:
                    batch_size = X_batch.shape[0]
                    irdr = irdr.expand(batch_size)
                
                all_irdr_values.append(irdr)
        
        # 6. Global normalization: calculate mean across ALL points from ALL components
        with torch.no_grad():
            # Safety check: ensure all tensors are properly shaped before concatenation
            valid_irdr_values = []
            for irdr in all_irdr_values:
                if irdr.dim() > 0 and irdr.numel() > 0:
                    # Flatten the tensor to 1D to ensure compatibility
                    irdr_flat = irdr.flatten()
                    valid_irdr_values.append(irdr_flat)
                else:
                    # Skip zero-dimensional or empty tensors
                    print(f"Warning: Skipping zero-dimensional tensor in BRDR")
                    continue
            
            if not valid_irdr_values:
                # If no valid tensors, use a default approach
                print("Warning: No valid tensors for global normalization, using fallback")
                global_mean_irdr = torch.tensor(1.0, device=X_batch.device)
            else:
                try:
                    # Ensure all tensors have the same shape for concatenation
                    target_shape = valid_irdr_values[0].shape
                    compatible_irdr_values = []

                    for irdr in valid_irdr_values:
                        if irdr.shape == target_shape and irdr.numel() > 0:
                            compatible_irdr_values.append(irdr)
                        else:
                            #print(f"Warning: Skipping incompatible tensor with shape {irdr.shape}")
                            pass

                    if not compatible_irdr_values:
                        print("Warning: No compatible tensors for global normalization, using fallback")
                        global_mean_irdr = torch.tensor(1.0, device=X_batch.device)
                    else:
                        global_irdr_tensor = torch.cat(compatible_irdr_values)
                        global_mean_irdr = torch.mean(global_irdr_tensor)

                except Exception as e:
                    print(f"Error in global tensor operations: {e}")
                    print(f"valid_irdr_values shapes: {[v.shape for v in valid_irdr_values]}")
                    print(f"valid_irdr_values types: {[type(v) for v in valid_irdr_values]}")
                    # Use fallback instead of raising
                    print("Using fallback global mean")
                    global_mean_irdr = torch.tensor(1.0, device=X_batch.device)
            
            # Update weights using global normalization
            for i, key in enumerate(self.loss_keys):
                if i < len(all_irdr_values):
                    component_irdr = all_irdr_values[i]
                    
                    # Safety check: ensure component_irdr has proper shape
                    if component_irdr.dim() == 0:
                        batch_size = X_batch.shape[0]
                        component_irdr = component_irdr.expand(batch_size)
                    
                    # Flatten the component_irdr to match the flattened global tensor
                    component_irdr_flat = component_irdr.flatten()
                    
                    # Normalize using GLOBAL mean (key insight from paper)
                    # Convert tensor to scalar properly
                    try:
                        if global_mean_irdr.dim() == 0:
                            # 0-dimensional tensor - use .item()
                            global_mean_scalar = global_mean_irdr.item()
                        else:
                            # Multi-dimensional tensor - take mean first, then .item()
                            global_mean_scalar = global_mean_irdr.mean().item()
                    except (AttributeError, RuntimeError):
                        # Fallback if tensor operations fail
                        global_mean_scalar = 1.0

                    try:
                        w_ref_flat = component_irdr_flat / (global_mean_scalar + self.epsilon)
                        
                        # Reshape w_ref back to match the original point_weights shape
                        w_ref = w_ref_flat.view_as(self.point_weights[key])
                        
                        # Update point-wise weights with exponential moving average
                        self.point_weights[key].copy_(
                            self.beta_w * self.point_weights[key] + (1 - self.beta_w) * w_ref
                        )
                    except Exception as e:
                        print(f"Error in weight update for {key}: {e}")
                        print(f"component_irdr_flat shape: {component_irdr_flat.shape}")
                        print(f"global_mean_scalar: {global_mean_scalar}, type: {type(global_mean_scalar)}")
                        print(f"self.point_weights[{key}] shape: {self.point_weights[key].shape}")
                        raise
                else:
                    # If we don't have enough irdr values, use default weight
                    print(f"Warning: No irdr value for {key}, using default weight")
                    self.point_weights[key].copy_(
                        self.beta_w * self.point_weights[key] + (1 - self.beta_w) * torch.ones_like(self.point_weights[key])
                    )
        
        # 7. Calculate weighted losses for each component
        final_losses = {}
        for key in self.loss_keys:
            current_residual = residuals[f'{key}_res']
            final_losses[key] = torch.mean(self.point_weights[key] * (current_residual**2))
        
        # 8. Calculate total loss
        total_loss = torch.stack(list(final_losses.values())).sum()
        
        # 9. Implement adaptive scale factor 's'
        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        
        # Calculate squared L2 norm of gradients
        grad_norm_sq = sum(
            p.grad.detach().pow(2).sum()
            for p in model.parameters()
            if p.grad is not None
        )
        
        # Update adaptive scale factor
        s_old = self.s
        s_max = (2 * total_loss.detach()) / (grad_norm_sq + self.epsilon)
        learning_rate = optimizer.param_groups[0]['lr']
        beta_s = 1.0 - learning_rate
        self.s = beta_s * s_old + (1 - beta_s) * s_max
        
        # Apply scale factor correction to gradients
        with torch.no_grad():
            scale_factor = self.s / s_old if s_old != 0 else 1.0
            for param in model.parameters():
                if param.grad is not None:
                    param.grad *= scale_factor
        
        # 10. Apply optimizer step with corrected gradients
        optimizer.step()
        
        # 11. Update current weights for logging (use mean of point weights)
        for key in self.loss_keys:
            self.current_weights[key] = torch.mean(self.point_weights[key]).item()
        
        return total_loss


def add_method_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add method-specific command-line arguments."""
    parser.add_argument('--beta-c', type=float, default=0.999,
                       help='EMA factor for statistics (default: 0.999)')
    parser.add_argument('--beta-w', type=float, default=0.999,
                       help='EMA factor for weights (default: 0.999)')
    parser.add_argument('--epsilon', type=float, default=1e-8,
                       help='Numerical stability constant (default: 1e-8)')

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
    parser = argparse.ArgumentParser(description='BRDR PINN Training')
    parser = add_common_arguments(parser)
    parser = add_method_arguments(parser)
    args = parser.parse_args()
    
    # Parse configuration
    config = parse_common_arguments(args)
    method_params = {
        'beta_c': args.beta_c,
        'beta_w': args.beta_w,
        'epsilon': args.epsilon
    }
    
    # Setup device and output directory
    device = setup_device(config['device'])
    output_dir, model_path = create_output_directory('brdr_pinn', config['output_dir'])
    
    # Print configuration
    print_training_config(config, 'BRDR PINN', method_params)
    
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
    loss_method = BRDRLoss(enable_mass_constraints=not args.synthetic, **method_params)
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
        'weight_phys_mass2': [], 'scale_factor_s': []
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
            
            # Get model predictions
            y_pred = model(xb)
            
            # Get loss components and perform method-specific step
            loss_components = loss_method(model, xb, yb, data['Xmax'].to(device), 
                                        data['Xmin'].to(device), data['ymax'].to(device), 
                                        data['ymin'].to(device))
            total = loss_method.step(loss_components, opt, model, xb, yb)
            
            # Collect raw residuals for this batch
            raw_residuals = collect_raw_residuals(model, xb, yb, y_pred, data, device)
            tr_raw_residuals.append(raw_residuals)
            
            # Accumulate losses with proper error handling
            try:
                tr_total += total.item()
                tr_data += loss_components[0].item()
                for i in range(num_phys_components):
                    tr_phys_components[i] += loss_components[i+1].item()
            except (RuntimeError, IndexError) as e:
                print(f"Error in loss accumulation: {e}")
                print(f"total shape: {total.shape}, type: {type(total)}")
                print(f"loss_components types and shapes: {[(type(lc), lc.shape if hasattr(lc, 'shape') else 'no shape') for lc in loss_components]}")
                raise
        
        # Average over batches
        n_train = len(data['train'])
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
        
        # Store scale factor (average weight)
        avg_weight = np.mean(list(loss_method.current_weights.values()))
        hist['scale_factor_s'].append(avg_weight)
        
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
                
                # Get model predictions
                y_pred = model(xb)
                
                loss_components = loss_method(model, xb, yb, data['Xmax'].to(device),
                                           data['Xmin'].to(device), data['ymax'].to(device),
                                           data['ymin'].to(device))
                
                # Calculate total loss with current weights (no weight update in validation)
                # For validation, we need to compute the weighted losses using current point weights
                val_weighted_losses = []
                for i, key in enumerate(loss_method.loss_keys):
                    # Get the mean weight for this component
                    mean_weight = torch.mean(loss_method.point_weights[key]).item()
                    val_weighted_losses.append(mean_weight * loss_components[i])
                weighted_total = torch.sum(torch.stack(val_weighted_losses))
                val_total += weighted_total.item()
                val_data += loss_components[0].item()
                for i in range(num_phys_components):
                    val_phys_components[i] += loss_components[i+1].item()
                
                # Collect raw residuals for validation
                raw_residuals = collect_raw_residuals(model, xb, yb, y_pred, data, device)
                val_raw_residuals.append(raw_residuals)
        
        # Average over validation batches
        n_val = len(data['val'])
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
        lr_sched.step(current_val_loss)
        
        if current_val_loss < min(hist['val_total']):
            torch.save(model.state_dict(), model_path)
        
        early({'data_val': hist['data_val'][-1], 'val_total': hist['val_total'][-1]})
        if early.early_stop:
            break
    
    # Save results and create plots
    print("Saving results and creating plots...")
    np.savez(os.path.join(output_dir, 'brdr_history.npz'), **hist)
    plot_training_history(hist, output_dir, 'BRDR PINN', 
                         include_weights=True, include_method_params=True)
    
    # Create trajectory and phase space plots
    print("Creating trajectory and phase space visualizations...")
    plot_trajectory_predictions(model, data, output_dir, 'BRDR PINN', device=device)
    
    # Always save the final model for parameter extraction
    torch.save(model.state_dict(), model_path)
    
    
    # Save final parameters for comparison
    final_params = collect_model_parameters(model)
    np.savez(os.path.join(output_dir, 'brdr_parameters.npz'), **final_params)
    
    print(f'Training complete. Model saved to {model_path}')
    print(f'Results saved to {output_dir}')
    print(f'Final parameters: {final_params}')
    print(f'Final weights: {loss_method.current_weights}')


if __name__ == '__main__':
    main() 