#!/usr/bin/env python3
"""
GradNorm PINN Training Script

Implements the GradNorm algorithm for adaptive loss balancing in PINN training.
This method dynamically adjusts loss weights based on gradient norms to ensure
balanced training across all loss components.

Usage:
    python gradnorm_training.py [--epochs 2000] [--batch-size 256] [--lr 1e-4] [--max-samples 1000]
                               [--alpha 1.5] [--weight-lr 0.025]
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
import argparse
import csv
import time
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.basicPINNv8 import ConfigurablePINN, adaptive_custom_loss, get_default_pinn_config
PINN_VERSION = "dimensional"
print("Using original dimensional PINN")
from training_scripts.common_utils import (
    MultiLossEarlyStopping, ReduceLROnPlateau, prepare_data, load_and_prepare_data,
    get_pinn_config, get_param_init_config, setup_device, create_output_directory,
    plot_training_history, add_common_arguments, parse_common_arguments, print_training_config,
    prepare_synthetic_data, collect_raw_residuals, collect_model_parameters, 
    initialize_comprehensive_history, update_comprehensive_history,
    unified_data_preparation, verify_data_structure, plot_trajectory_predictions
)


class DebugLogger:
    """
    CSV logger for debugging information during PINN training.
    
    Captures loss components, gradient norms, weights, and parameters
    for later analysis and visualization.
    """
    
    def __init__(self, output_dir: str, enabled: bool = True):
        self.enabled = enabled
        if not self.enabled:
            return
            
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # CSV file paths
        self.losses_file = os.path.join(output_dir, 'debug_losses.csv')
        self.gradients_file = os.path.join(output_dir, 'debug_gradients.csv')
        self.weights_file = os.path.join(output_dir, 'debug_weights.csv')
        self.parameters_file = os.path.join(output_dir, 'debug_parameters.csv')
        self.magnitudes_file = os.path.join(output_dir, 'debug_magnitudes.csv')
        
        # Initialize CSV files with headers
        self._init_csv_files()
        
    def _init_csv_files(self):
        """Initialize CSV files with appropriate headers."""
        
        # Losses CSV
        with open(self.losses_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'batch', 'phase', 'timestamp',
                'data_loss', 'phys_res1_loss', 'phys_res2_loss', 'phys_res3_loss', 'phys_res4_loss',
                'phys_mass1_loss', 'phys_mass2_loss', 'total_weighted_loss'
            ])
        
        # Gradients CSV
        with open(self.gradients_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'batch', 'phase', 'timestamp', 'parameter_name', 'grad_norm'
            ])
        
        # Weights CSV  
        with open(self.weights_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'batch', 'phase', 'timestamp',
                'weight_data', 'weight_phys_res1', 'weight_phys_res2', 'weight_phys_res3', 'weight_phys_res4',
                'weight_phys_mass1', 'weight_phys_mass2',
                'avg_grad_norm', 'target_grad_norm_data', 'target_grad_norm_phys_res1', 'target_grad_norm_phys_res2',
                'target_grad_norm_phys_res3', 'target_grad_norm_phys_res4'
            ])
        
        # Parameters CSV
        with open(self.parameters_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'batch', 'phase', 'timestamp',
                'M1', 'M2', 'M3', 'D1', 'D2', 'D3', 'K1', 'K2', 'E1'
            ])
        
        # Magnitudes CSV
        with open(self.magnitudes_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'phase', 'timestamp', 'metric_type', 'component', 'mean', 'std', 'min', 'max', 'scale_ratio', 'learned_c'
            ])
    
    def log_losses(self, epoch: int, batch: int, phase: str, loss_components: list, total_loss: float):
        """Log loss components to CSV."""
        if not self.enabled:
            return
            
        timestamp = time.time()
        
        # Pad loss_components to ensure we have 7 components (for compatibility)
        losses = [loss.item() if hasattr(loss, 'item') else loss for loss in loss_components]
        while len(losses) < 7:
            losses.append(0.0)
        
        with open(self.losses_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, batch, phase, timestamp,
                losses[0], losses[1], losses[2], losses[3], losses[4], losses[5], losses[6],
                total_loss
            ])
    
    def log_gradients(self, epoch: int, batch: int, phase: str, model):
        """Log gradient norms for all model parameters."""
        if not self.enabled:
            return
            
        timestamp = time.time()
        
        with open(self.gradients_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = torch.norm(param.grad).item()
                    writer.writerow([epoch, batch, phase, timestamp, name, grad_norm])
                else:
                    writer.writerow([epoch, batch, phase, timestamp, name, 0.0])
    
    def log_weights(self, epoch: int, batch: int, phase: str, weights_dict: dict, 
                   grad_norms: list = None, relative_inverse_rates: list = None):
        """Log GradNorm weights and related information."""
        if not self.enabled:
            return
            
        timestamp = time.time()
        
        # Extract weights (pad to 7 components)
        weight_keys = ['data', 'phys_res1', 'phys_res2', 'phys_res3', 'phys_res4', 'phys_mass1', 'phys_mass2']
        weights = [weights_dict.get(key, 0.0) for key in weight_keys]
        
        # Calculate additional metrics if available
        avg_grad_norm = np.mean(grad_norms) if grad_norms else 0.0
        target_grad_norms = []
        if grad_norms and relative_inverse_rates:
            for i, rate in enumerate(relative_inverse_rates[:5]):  # Only first 5 (main physics losses)
                target_grad_norms.append(avg_grad_norm * rate)
        while len(target_grad_norms) < 5:
            target_grad_norms.append(0.0)
        
        with open(self.weights_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, batch, phase, timestamp,
                weights[0], weights[1], weights[2], weights[3], weights[4], weights[5], weights[6],
                avg_grad_norm, target_grad_norms[0], target_grad_norms[1], target_grad_norms[2],
                target_grad_norms[3], target_grad_norms[4]
            ])
    
    def log_parameters(self, epoch: int, batch: int, phase: str, model):
        """Log physical parameter values."""
        if not self.enabled:
            return
            
        timestamp = time.time()
        
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
        
        with open(self.parameters_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, batch, phase, timestamp,
                params['M1'], params['M2'], params['M3'], params['D1'], params['D2'],
                params['D3'], params['K1'], params['K2'], params['E1']
            ])
    
    def log_magnitudes(self, epoch: int, phase: str, model, X_batch, y_batch, X_max, X_min, y_max, y_min):
        """Log magnitude analysis data."""
        if not self.enabled:
            return
            
        timestamp = time.time()
        
        with torch.no_grad():
            # Get model predictions
            y_pred = model(X_batch)
            
            # Ensure normalization tensors are on same device as input tensors
            device = X_batch.device
            X_max = X_max.to(device)
            X_min = X_min.to(device)
            y_max = y_max.to(device)
            y_min = y_min.to(device)
            
            # Denormalize inputs for analysis
            X_range = X_max - X_min + 1e-12
            X_denorm = X_batch * X_range + X_min
            
            # Denormalize outputs for analysis
            y_range = y_max - y_min + 1e-12
            y_denorm = y_batch * y_range + y_min
            y_pred_denorm = y_pred * y_range + y_min
            
            # Compute raw residuals and get full results dictionary  
            results_dict = model.compute_residuals(X_batch, y_pred, X_max, X_min, y_max, y_min)
            
            # Handle both old tuple format and new dictionary format for compatibility
            if isinstance(results_dict, dict):
                # New dimensionless PINN format
                residuals = results_dict['residuals']
                res1, res2, res3, res4, res_mass1, res_mass2 = residuals
                dimless_kinematics = results_dict.get('dimensionless_kinematics', {})
                dimless_params = results_dict.get('dimensionless_params', {})
                learned_c = model.c.item() if hasattr(model, 'c') else 0.0
            else:
                # Old dimensional PINN format (backwards compatibility)
                res1, res2, res3, res4, res_mass1, res_mass2 = results_dict
                dimless_kinematics = {}
                dimless_params = {}
                learned_c = 0.0
            
            # Feature names for inputs
            feature_names = ['x2_dot', 'y2_dot', 'x3_dot', 'y3_dot', 'x2', 'y2', 'x3', 'y3', 'omega', 't']
            output_names = ['x2_ddot', 'y2_ddot', 'x3_ddot', 'y3_ddot']
            residual_names = ['res1', 'res2', 'res3', 'res4', 'res_mass1', 'res_mass2']
            
            # Data loss for scale ratio calculations
            data_loss = torch.sqrt(torch.mean((y_pred - y_batch)**2) + 1e-12).item()
            
            with open(self.magnitudes_file, 'a', newline='') as f:
                writer = csv.writer(f)
                
                # Log normalized input statistics
                X_norm = X_batch.cpu().numpy()
                for i, name in enumerate(feature_names):
                    if i < X_norm.shape[1]:
                        values = X_norm[:, i]
                        writer.writerow([
                            epoch, phase, timestamp, 'input_normalized', name,
                            np.mean(values), np.std(values), np.min(values), np.max(values), 1.0, learned_c
                        ])
                
                # Log denormalized input statistics
                X_denorm_np = X_denorm.cpu().numpy()
                for i, name in enumerate(feature_names):
                    if i < X_denorm_np.shape[1]:
                        values = X_denorm_np[:, i]
                        writer.writerow([
                            epoch, phase, timestamp, 'input_denormalized', name,
                            np.mean(values), np.std(values), np.min(values), np.max(values), 1.0, learned_c
                        ])
                
                # Log normalized output statistics
                y_norm = y_batch.cpu().numpy()
                for i, name in enumerate(output_names):
                    if i < y_norm.shape[1]:
                        values = y_norm[:, i]
                        writer.writerow([
                            epoch, phase, timestamp, 'output_normalized', name,
                            np.mean(values), np.std(values), np.min(values), np.max(values), 1.0, learned_c
                        ])
                
                # Log denormalized output statistics
                y_denorm_np = y_denorm.cpu().numpy()
                for i, name in enumerate(output_names):
                    if i < y_denorm_np.shape[1]:
                        values = y_denorm_np[:, i]
                        writer.writerow([
                            epoch, phase, timestamp, 'output_denormalized', name,
                            np.mean(values), np.std(values), np.min(values), np.max(values), 1.0, learned_c
                        ])
                
                # Log prediction statistics
                y_pred_denorm_np = y_pred_denorm.cpu().numpy()
                for i, name in enumerate(output_names):
                    if i < y_pred_denorm_np.shape[1]:
                        values = y_pred_denorm_np[:, i]
                        writer.writerow([
                            epoch, phase, timestamp, 'prediction_denormalized', name,
                            np.mean(values), np.std(values), np.min(values), np.max(values), 1.0, learned_c
                        ])
                
                # Log raw residual statistics
                raw_residuals = [res1.cpu().numpy(), res2.cpu().numpy(), res3.cpu().numpy(), 
                               res4.cpu().numpy(), res_mass1.cpu().numpy(), res_mass2.cpu().numpy()]
                
                for i, (name, res_array) in enumerate(zip(residual_names, raw_residuals)):
                    values = res_array.flatten()
                    res_rmse = np.sqrt(np.mean(values**2))
                    scale_ratio = res_rmse / (data_loss + 1e-12)
                    writer.writerow([
                        epoch, phase, timestamp, 'residual_raw', name,
                        np.mean(values), np.std(values), np.min(values), np.max(values), scale_ratio, learned_c
                    ])
                
                # Log physical parameter magnitudes
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
                
                for name, value in params.items():
                    writer.writerow([
                        epoch, phase, timestamp, 'parameter', name,
                        value, 0.0, value, value, 1.0, learned_c
                    ])
                
                # Log dimensionless kinematic statistics (for dimensionless PINN)
                for name, tensor in dimless_kinematics.items():
                    if tensor.numel() > 0:  # Check if tensor is not empty
                        values = tensor.cpu().numpy().flatten()
                        writer.writerow([
                            epoch, phase, timestamp, 'dimless_kinematic', name,
                            np.mean(values), np.std(values), np.min(values), np.max(values), 1.0, learned_c
                        ])
                
                # Log dimensionless parameter statistics (for dimensionless PINN)
                for name, tensor in dimless_params.items():
                    if tensor.numel() > 0:  # Check if tensor is not empty
                        values = tensor.cpu().numpy().flatten()
                        writer.writerow([
                            epoch, phase, timestamp, 'dimless_parameter', name,
                            np.mean(values), np.std(values), np.min(values), np.max(values), 1.0, learned_c
                        ])
    



class PretrainEarlyStopping:
    """Early stopping specifically for pre-training phase based on data loss."""
    
    def __init__(self, patience: int = 50, min_delta: float = 1e-7):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.count = 0
        self.should_stop = False
    
    def __call__(self, loss: float):
        if self.best_loss is None or loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.count = 0
        else:
            self.count += 1
            
        if self.count >= self.patience:
            self.should_stop = True
            print(f"\nPre-training early stopping triggered after {self.count} epochs without improvement")
    
    def reset(self):
        """Reset the early stopping state."""
        self.best_loss = None
        self.count = 0
        self.should_stop = False


class PhysicsOnlyEarlyStopping:
    """Early stopping specifically for physics-only training phase based on physics residuals."""
    
    def __init__(self, patience: int = 50, min_delta: float = 1e-7):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.count = 0
        self.should_stop = False
    
    def __call__(self, loss: float):
        if self.best_loss is None or loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.count = 0
        else:
            self.count += 1
            
        if self.count >= self.patience:
            self.should_stop = True
            print(f"\nPhysics-only early stopping triggered after {self.count} epochs without improvement")
    
    def reset(self):
        """Reset the early stopping state."""
        self.best_loss = None
        self.count = 0
        self.should_stop = False


class GradNormLoss(nn.Module):
    """
    GradNorm loss balancing method.
    
    This method implements gradient normalization to balance loss components
    by adjusting weights based on gradient magnitudes.
    """
    
    def __init__(self, alpha: float = 1.5, weight_lr: float = 0.01, enable_mass_constraints: bool = True, debug: bool = False, debug_logger=None):
        super().__init__()
        self.alpha = alpha  # GradNorm parameter
        self.weight_lr = weight_lr  # Learning rate for weights
        self.debug = debug  # Debug flag for verbose output
        self.debug_logger = debug_logger  # CSV logger for debugging
        
        # State variables
        self.initial_losses = None
        self.running_weights = None
        self._initialized = False
        
        # Pre-training state
        self.pretrain_mode = False
        self.pretrain_initial_loss = None
        
        # Physics-only training state
        self.physics_only_mode = False
        self.physics_only_initial_loss = None
        
        # Store normalization parameters
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
        self.last_lambdas = {key: 1.0 for key in self.loss_keys}
        self.clamp_warnings = set()  # Track which weights have been clamped
        
    def set_normalization_params(self, X_max, X_min, y_max, y_min):
        """Set normalization parameters for denormalization."""
        self.X_max = X_max
        self.X_min = X_min
        self.y_max = y_max
        self.y_min = y_min
    
    def enable_pretrain_mode(self):
        """Enable pre-training mode (data loss only)."""
        self.pretrain_mode = True
        self.pretrain_initial_loss = None
        self.physics_only_mode = False  # Ensure mutual exclusivity
        print("Pre-training mode enabled - using data loss only")
    
    def disable_pretrain_mode(self):
        """Disable pre-training mode and initialize GradNorm for full training."""
        self.pretrain_mode = False
        # Reset GradNorm state for full training
        self.initial_losses = None
        self.running_weights = None
        self._initialized = False
        print("Pre-training mode disabled - switching to full GradNorm training")
    
    def enable_physics_only_mode(self):
        """Enable physics-only mode (physics residuals only, no data loss)."""
        self.physics_only_mode = True
        self.physics_only_initial_loss = None
        self.pretrain_mode = False  # Ensure mutual exclusivity
        print("Physics-only mode enabled - using physics residuals only")
    
    def disable_physics_only_mode(self):
        """Disable physics-only mode and initialize GradNorm for full training."""
        self.physics_only_mode = False
        # Reset GradNorm state for full training
        self.initial_losses = None
        self.running_weights = None
        self._initialized = False
        print("Physics-only mode disabled - switching to full GradNorm training")
        
    def forward(self, model, X_batch, y_batch, X_max=None, X_min=None, y_max=None, y_min=None):
        """Calculate all loss components using adaptive_custom_loss."""
        # Use provided parameters or stored ones
        X_max = X_max if X_max is not None else self.X_max
        X_min = X_min if X_min is not None else self.X_min
        y_max = y_max if y_max is not None else self.y_max
        y_min = y_min if y_min is not None else self.y_min
        
        loss_components = adaptive_custom_loss(model, X_batch, y_batch, X_max, X_min, y_max, y_min, debug=self.debug)
        return loss_components
    
    def _get_initial_losses(self, losses):
        """Get initial loss values for relative inverse training rate calculation."""
        if self.initial_losses is None:
            self.initial_losses = [loss.item() for loss in losses]
        return self.initial_losses
    
    def _get_last_shared_layer(self, model):
        """Get the last shared layer for gradient norm calculation."""
        # For PINN, we'll use the last layer of the acceleration network
        return model.NNforAccelerations.model[-1]
    
    def step(self, loss_components, optimizer, model, X_batch, y_batch, epoch=0, batch=0):
        """
        Performs optimization step with GradNorm weight adjustment.
        
        In pre-training mode, only uses data loss for optimization.
        In full training mode, applies the full GradNorm algorithm:
        1. Calculate relative inverse training rates
        2. Calculate average inverse training rate
        3. Calculate relative inverse training rates
        4. Update weights based on gradient norms
        5. Normalize weights to maintain their sum
        6. Optimize model parameters
        """
        # Ensure double precision
        X_batch = X_batch.double()
        y_batch = y_batch.double()
        
        # Handle pre-training mode
        if self.pretrain_mode:
            return self._pretrain_step(loss_components, optimizer, model, X_batch, y_batch, epoch, batch)
        
        # Handle physics-only mode
        if self.physics_only_mode:
            return self._physics_only_step(loss_components, optimizer, model, X_batch, y_batch, epoch, batch)
        
        # 1. Get initial losses if not set
        initial_losses = self._get_initial_losses(loss_components)
        
        # 2. Calculate relative inverse training rates L(t)/L(0)
        current_losses = [loss.item() for loss in loss_components]
        relative_losses = [current_losses[i] / (initial_losses[i] + 1e-12) for i in range(len(current_losses))]
        
        # 3. Calculate average inverse training rate
        avg_relative_loss = np.mean(relative_losses)
        
        # 4. Calculate relative inverse training rates
        relative_inverse_rates = [(avg_relative_loss / (rel_loss + 1e-12)) ** self.alpha for rel_loss in relative_losses]
        
        # 5. Initialize weights if not done
        if self.running_weights is None:
            self.running_weights = torch.ones(len(loss_components), device=loss_components[0].device, dtype=torch.float64, requires_grad=False)
        
        # 6. Calculate gradient norms for each loss component
        grad_norms = []
        for i, loss in enumerate(loss_components):
            # Zero gradients
            optimizer.zero_grad()
            
            # Check if loss requires gradients
            if loss.requires_grad and loss.grad_fn is not None:
                # Backward pass for this loss component
                loss.backward(retain_graph=True)
                
                # Get gradient norm from last shared layer
                last_layer = self._get_last_shared_layer(model)
                if last_layer.weight.grad is not None:
                    grad_norm = torch.norm(last_layer.weight.grad).item()
                else:
                    grad_norm = 1e-12  # Small value if no gradient
            else:
                # If loss doesn't require gradients, use a small default value
                grad_norm = 1e-12
            
            grad_norms.append(grad_norm)
        
        # Debug information now captured in CSV logging system
        
        # 7. Calculate target gradient norm
        avg_grad_norm = np.mean(grad_norms)
        
        # 8. Update weights
        for i in range(len(self.running_weights)):
            target_grad_norm = avg_grad_norm * relative_inverse_rates[i]
            current_grad_norm = grad_norms[i]
            
            if current_grad_norm > 1e-12:
                weight_update = (target_grad_norm / current_grad_norm - 1.0) * self.weight_lr
                self.running_weights[i] = self.running_weights[i] * (1.0 + weight_update)
            else:
                # If gradient is too small, don't update weight
                pass
        
        # 9. Normalize weights to maintain their sum
        weight_sum = self.running_weights.sum().item()
        if weight_sum > 0:
            self.running_weights = self.running_weights / weight_sum * len(self.running_weights)
        
        # 10. Clamp weights to prevent extreme values
        for i in range(len(self.running_weights)):
            if self.running_weights[i] < 0.1 and i not in self.clamp_warnings:
                # Use a safe way to get the key name
                key_name = self.loss_keys[i] if i < len(self.loss_keys) else f"component_{i}"
                print(f"Warning: Weight {i} ({key_name}) clamped to 0.1")
                self.clamp_warnings.add(i)
            self.running_weights[i] = torch.clamp(self.running_weights[i], 0.1, 10.0)
        
        if self.debug:
            print(f"  Updated weights: {[f'{w.item():.6e}' for w in self.running_weights]}")
        
        # 11. Recompute loss components to avoid graph conflicts
        # This is the key fix - we recompute the loss components fresh
        fresh_loss_components = adaptive_custom_loss(model, X_batch, y_batch, 
                                                   self.X_max, self.X_min, 
                                                   self.y_max, self.y_min, debug=self.debug)
        
        # 12. Calculate weighted total loss for model optimization
        weighted_losses = []
        if self.running_weights is not None:
            # Only use the number of weights that correspond to loss keys
            num_weights_to_use = min(len(self.running_weights), len(self.loss_keys))
            for i in range(num_weights_to_use):
                weighted_losses.append(self.running_weights[i] * fresh_loss_components[i])
            total_loss = torch.stack(weighted_losses).sum()
        else:
            total_loss = torch.stack([fresh_loss_components[i] for i in range(len(self.loss_keys))]).sum()
        
        # Total loss information now captured in CSV logging
        
        # 13. Update logging weights first
        if self.running_weights is not None:
            self.last_lambdas = {
                key: self.running_weights[i].item()
                for i, key in enumerate(self.loss_keys)
                if i < len(self.running_weights)
            }
        else:
            self.last_lambdas = {key: 1.0 for key in self.loss_keys}
        
        # 14. CSV Logging
        if self.debug_logger:
            phase = "pretrain" if self.pretrain_mode else "full"
            self.debug_logger.log_losses(epoch, batch, phase, fresh_loss_components, total_loss.item())
            self.debug_logger.log_weights(epoch, batch, phase, self.last_lambdas, grad_norms, relative_inverse_rates)
            self.debug_logger.log_parameters(epoch, batch, phase, model)
        
        # 15. Optimize model parameters
        optimizer.zero_grad()
        total_loss.backward()
        
        # Log gradients after backward pass
        if self.debug_logger:
            phase = "pretrain" if self.pretrain_mode else "full"
            self.debug_logger.log_gradients(epoch, batch, phase, model)
        
        optimizer.step()
        
        return total_loss
    
    def _pretrain_step(self, loss_components, optimizer, model, X_batch, y_batch, epoch=0, batch=0):
        """
        Performs pre-training step using only data loss.
        
        Args:
            loss_components: Tuple of loss components from adaptive_custom_loss
            optimizer: Model optimizer
            model: PINN model
            X_batch: Input batch
            y_batch: Target batch
            epoch: Current epoch
            batch: Current batch
            
        Returns:
            Data loss tensor
        """
        # Get only the data loss (first component)
        data_loss = loss_components[0]
        
        # Track initial data loss for pre-training
        if self.pretrain_initial_loss is None:
            self.pretrain_initial_loss = data_loss.item()
            print(f"Pre-training initial data loss: {self.pretrain_initial_loss:.6e}")
        
        # Debug information now captured in CSV logging system
        
        # Set weights for logging (data=1, physics=0)
        self.last_lambdas = {key: 1.0 if key == 'data' else 0.0 for key in self.loss_keys}
        
        # CSV Logging for pre-training
        if self.debug_logger:
            self.debug_logger.log_losses(epoch, batch, "pretrain", loss_components, data_loss.item())
            self.debug_logger.log_weights(epoch, batch, "pretrain", self.last_lambdas)
            self.debug_logger.log_parameters(epoch, batch, "pretrain", model)
        
        # Optimize using only data loss
        optimizer.zero_grad()
        data_loss.backward()
        
        # Log gradients after backward pass
        if self.debug_logger:
            self.debug_logger.log_gradients(epoch, batch, "pretrain", model)
        
        optimizer.step()
        
        return data_loss
    
    def _physics_only_step(self, loss_components, optimizer, model, X_batch, y_batch, epoch=0, batch=0):
        """
        Performs physics-only training step using only physics residuals.
        
        This allows us to see if the physics formulation can converge independently
        of data fitting, providing insights into the physics constraint solvability.
        
        Args:
            loss_components: Tuple of loss components from adaptive_custom_loss
            optimizer: Model optimizer
            model: PINN model
            X_batch: Input batch
            y_batch: Target batch
            epoch: Current epoch
            batch: Current batch
            
        Returns:
            Physics loss tensor (sum of physics residuals)
        """
        # Get physics residuals only (skip data loss at index 0)
        physics_components = loss_components[1:]  # Skip data loss
        
        # Calculate total physics loss
        physics_loss = torch.stack(physics_components).sum()
        
        # Track initial physics loss
        if self.physics_only_initial_loss is None:
            self.physics_only_initial_loss = physics_loss.item()
            print(f"Physics-only initial loss: {self.physics_only_initial_loss:.6e}")
        
        # Set weights for logging (physics=1, data=0)
        self.last_lambdas = {key: 0.0 if key == 'data' else 1.0 for key in self.loss_keys}
        
        # CSV Logging for physics-only training
        if self.debug_logger:
            self.debug_logger.log_losses(epoch, batch, "physics_only", loss_components, physics_loss.item())
            self.debug_logger.log_weights(epoch, batch, "physics_only", self.last_lambdas)
            self.debug_logger.log_parameters(epoch, batch, "physics_only", model)
        
        # Optimize using only physics loss
        optimizer.zero_grad()
        physics_loss.backward()
        
        # Log gradients after backward pass
        if self.debug_logger:
            self.debug_logger.log_gradients(epoch, batch, "physics_only", model)
        
        optimizer.step()
        
        return physics_loss


def ensure_double_precision(data_dict):
    """Convert all tensors in data dictionary to double precision."""
    for key, value in data_dict.items():
        if isinstance(value, torch.Tensor):
            data_dict[key] = value.double()
        elif isinstance(value, dict):
            ensure_double_precision(value)
    return data_dict


def add_method_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add method-specific command-line arguments."""
    parser.add_argument('--alpha', type=float, default=1.5,
                       help='Balancing factor for relative inverse training rates (default: 1.5)')
    parser.add_argument('--weight-lr', type=float, default=0.025,
                       help='Learning rate for weight optimization (default: 0.025)')
    
    # Pre-training arguments (data-only)
    parser.add_argument('--pretrain-epochs', type=int, default=0,
                       help='Number of epochs to pre-train on data loss only (default: 500)')
    parser.add_argument('--pretrain-lr', type=float, default=None,
                       help='Learning rate for pre-training phase (default: same as main training)')
    parser.add_argument('--pretrain-patience', type=int, default=50,
                       help='Early stopping patience for pre-training phase (default: 50)')
    parser.add_argument('--pretrain-lr-patience', type=int, default=20,
                       help='Learning rate scheduler patience for pre-training (default: 20)')
    
    # Physics-only training arguments
    parser.add_argument('--physics-only-epochs', type=int, default=0,
                       help='Number of epochs to train on physics residuals only (default: 0 = no physics-only training)')
    parser.add_argument('--physics-only-lr', type=float, default=None,
                       help='Learning rate for physics-only phase (default: same as main training)')
    parser.add_argument('--physics-only-patience', type=int, default=50,
                       help='Early stopping patience for physics-only phase (default: 50)')
    parser.add_argument('--physics-only-lr-patience', type=int, default=20,
                       help='Learning rate scheduler patience for physics-only training (default: 20)')
    
    # Transition arguments
    parser.add_argument('--reset-optimizer', action='store_true', default=True,
                       help='Reset optimizer state when transitioning between training phases')
    parser.add_argument('--auto-transition', action='store_true', default=True,
                       help='Automatically transition to next phase when current phase plateaus')
    
    # Debug arguments
    parser.add_argument('--debug', action='store_true',
                       help='Enable verbose debug output during training')
    parser.add_argument('--csv-logging', action='store_true', default=True,
                       help='Enable CSV logging for detailed debugging and analysis')
    
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
    parser = argparse.ArgumentParser(description='GradNorm PINN Training')
    parser = add_common_arguments(parser)
    parser = add_method_arguments(parser)
    args = parser.parse_args()
    
    # Parse configuration
    config = parse_common_arguments(args)
    
    # Separate GradNorm-specific parameters from training configuration
    gradnorm_params = {
        'alpha': args.alpha,
        'weight_lr': args.weight_lr
    }
    
    # Training configuration parameters
    training_config = {
        'pretrain_epochs': args.pretrain_epochs,
        'pretrain_lr': args.pretrain_lr if args.pretrain_lr is not None else config['learning_rate'],
        'pretrain_patience': args.pretrain_patience,
        'pretrain_lr_patience': args.pretrain_lr_patience,
        'physics_only_epochs': args.physics_only_epochs,
        'physics_only_lr': args.physics_only_lr if args.physics_only_lr is not None else config['learning_rate'],
        'physics_only_patience': args.physics_only_patience,
        'physics_only_lr_patience': args.physics_only_lr_patience,
        'reset_optimizer': args.reset_optimizer,
        'auto_transition': args.auto_transition,
        'debug': args.debug,
        'csv_logging': args.csv_logging
    }
    
    # For display purposes, combine them
    method_params = {**gradnorm_params, **training_config}
    
    # Setup device and output directory
    device = setup_device(config['device'])
    output_dir, model_path = create_output_directory('gradnorm_pinn', config['output_dir'])
    
    # Initialize debug logger
    debug_logger = DebugLogger(output_dir, enabled=training_config['csv_logging'])
    if training_config['csv_logging']:
        print(f"CSV logging enabled. Debug data will be saved to: {output_dir}")
        print("Use 'python debug_analysis.py <results_directory>' to analyze results")
    
    # Print configuration
    print_training_config(config, 'GradNorm PINN', method_params)
    
    # Print multi-phase training configuration if enabled
    total_special_epochs = training_config['physics_only_epochs'] + training_config['pretrain_epochs']
    if total_special_epochs > 0:
        print(f"Multi-Phase Training Configuration:")
        
        if training_config['physics_only_epochs'] > 0:
            print(f"  Phase 1 - Physics-Only Training:")
            print(f"    - Physics-only epochs: {training_config['physics_only_epochs']}")
            print(f"    - Physics-only learning rate: {training_config['physics_only_lr']}")
        
        if training_config['pretrain_epochs'] > 0:
            phase_num = 2 if training_config['physics_only_epochs'] > 0 else 1
            print(f"  Phase {phase_num} - Data-Only Pre-training:")
            print(f"    - Pre-training epochs: {training_config['pretrain_epochs']}")
            print(f"    - Pre-training learning rate: {training_config['pretrain_lr']}")
        
        # Determine final phase number
        has_both_phases = (training_config['physics_only_epochs'] > 0 and 
                           training_config['pretrain_epochs'] > 0)
        final_phase_num = 3 if has_both_phases else (2 if total_special_epochs > 0 else 1)
        print(f"  Phase {final_phase_num} - Full Training:")
        print(f"    - Full training epochs: {config['epochs'] - total_special_epochs}")
        print(f"    - Reset optimizer between phases: {training_config['reset_optimizer']}")
        print(f"    - Auto-transition enabled: {training_config['auto_transition']}")
        print(f"{'='*60}\n")
    
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
    loss_method = GradNormLoss(enable_mass_constraints=not args.synthetic, debug=training_config['debug'], debug_logger=debug_logger, **gradnorm_params)
    
    # Set normalization parameters in the loss method
    loss_method.set_normalization_params(
        data['Xmax'].to(device),
        data['Xmin'].to(device),
        data['ymax'].to(device),
        data['ymin'].to(device)
    )
    
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
        'weight_phys_mass2': [], 'training_phase': []  # Track pre-training vs full training
    })
    
    # Set up multi-phase training
    physics_only_epochs = training_config['physics_only_epochs']
    pretrain_epochs = training_config['pretrain_epochs']
    total_epochs = config['epochs']
    
    # Initialize phase-specific early stopping and LR schedulers
    physics_only_early = None
    physics_only_lr_sched = None
    pretrain_early = None
    pretrain_lr_sched = None
    
    if physics_only_epochs > 0:
        physics_only_early = PhysicsOnlyEarlyStopping(
            patience=training_config['physics_only_patience'],
            min_delta=config['min_delta']
        )
        physics_only_lr_sched = ReduceLROnPlateau(opt, patience=training_config['physics_only_lr_patience'])
    
    if pretrain_epochs > 0:
        pretrain_early = PretrainEarlyStopping(
            patience=training_config['pretrain_patience'],
            min_delta=config['min_delta']
        )
        pretrain_lr_sched = ReduceLROnPlateau(opt, patience=training_config['pretrain_lr_patience'])
    
    # Enable initial training mode
    if physics_only_epochs > 0:
        loss_method.enable_physics_only_mode()
        print(f"Starting with physics-only training for up to {physics_only_epochs} epochs")
    elif pretrain_epochs > 0:
        loss_method.enable_pretrain_mode()
        print(f"Starting with data-only pre-training for up to {pretrain_epochs} epochs")
    else:
        print("Starting with full GradNorm training")
    
    if training_config['auto_transition']:
        print("Auto-transition enabled: will switch phases when current phase plateaus")
    
    # Training loop
    print("Starting training...")
    physics_only_transition_epoch = None  # Track when we transitioned from physics-only
    pretrain_transition_epoch = None  # Track when we transitioned from pre-training
    
    for epoch in tqdm(range(total_epochs), desc="Training"):
        # Determine current phase based on epoch and active mode
        if loss_method.physics_only_mode and epoch < physics_only_epochs:
            current_phase = "physics_only"
        elif loss_method.pretrain_mode and epoch < physics_only_epochs + pretrain_epochs:
            current_phase = "pretrain"
        else:
            current_phase = "full"
        
        # Check for early transition from physics-only training
        if (current_phase == "physics_only" and physics_only_early is not None and 
            physics_only_early.should_stop and training_config['auto_transition']):
            print(f"\n{'='*60}")
            physics_only_transition_epoch = epoch
            loss_method.disable_physics_only_mode()
            
            # Determine next phase
            if pretrain_epochs > 0:
                print(f"AUTO-TRANSITION: Switching from physics-only to data-only pre-training at epoch {epoch}")
                current_phase = "pretrain"
                loss_method.enable_pretrain_mode()
            else:
                print(f"AUTO-TRANSITION: Switching from physics-only to full training at epoch {epoch}")
            current_phase = "full"
            print(f"{'='*60}")
            
            # Optionally reset optimizer
            if training_config['reset_optimizer']:
                print("Resetting optimizer state...")
                opt = optim.Adam(model.parameters(), lr=config['learning_rate'])
                if current_phase == "pretrain":
                    pretrain_lr_sched = ReduceLROnPlateau(opt, patience=training_config['pretrain_lr_patience'])
                else:
                    lr_sched = ReduceLROnPlateau(opt, patience=config['lr_patience'])
            
            # Reset early stopping for next phase
            if current_phase == "pretrain":
                pretrain_early.reset()
            else:
                early = MultiLossEarlyStopping(config['early_patience'], config['min_delta'], 
                                              ['data_val', 'val_total'])
        
        # Check for early transition from pre-training
        elif (current_phase == "pretrain" and pretrain_early is not None and 
            pretrain_early.should_stop and training_config['auto_transition']):
            
            # Optionally reset optimizer
            if training_config['reset_optimizer']:
                print("Resetting optimizer state...")
                opt = optim.Adam(model.parameters(), lr=config['learning_rate'])
                lr_sched = ReduceLROnPlateau(opt, patience=config['lr_patience'])
            
            # Reset early stopping to give full training a fresh start
            early = MultiLossEarlyStopping(config['early_patience'], config['min_delta'], 
                                          ['data_val', 'val_total'])
        
        # Standard physics-only to next phase transition
        elif epoch == physics_only_epochs and physics_only_epochs > 0 and loss_method.physics_only_mode:
            print(f"\n{'='*60}")
            physics_only_transition_epoch = epoch
            loss_method.disable_physics_only_mode()
            
            # Determine next phase
            if pretrain_epochs > 0:
                print(f"PHASE TRANSITION: Switching from physics-only to data-only pre-training at epoch {epoch}")
                current_phase = "pretrain"
                loss_method.enable_pretrain_mode()
            else:
                print(f"PHASE TRANSITION: Switching from physics-only to full training at epoch {epoch}")
            current_phase = "full"
            print(f"{'='*60}")
            
            # Optionally reset optimizer
            if training_config['reset_optimizer']:
                print("Resetting optimizer state...")
                opt = optim.Adam(model.parameters(), lr=config['learning_rate'])
                if current_phase == "pretrain":
                    pretrain_lr_sched = ReduceLROnPlateau(opt, patience=training_config['pretrain_lr_patience'])
                else:
                    lr_sched = ReduceLROnPlateau(opt, patience=config['lr_patience'])
            
            # Reset early stopping for next phase
            if current_phase == "pretrain":
                pretrain_early.reset()
            else:
                early = MultiLossEarlyStopping(config['early_patience'], config['min_delta'], 
                                              ['data_val', 'val_total'])
        
        # Standard pre-training to full training transition
        elif epoch == physics_only_epochs + pretrain_epochs and pretrain_epochs > 0 and loss_method.pretrain_mode:
            print(f"\n{'='*60}")
            print(f"PHASE TRANSITION: Switching from pre-training to full training at epoch {epoch}")
            print(f"{'='*60}")
            pretrain_transition_epoch = epoch
            
            # Disable pre-training mode
            loss_method.disable_pretrain_mode()
            current_phase = "full"
            
            # Optionally reset optimizer
            if training_config['reset_optimizer']:
                print("Resetting optimizer state...")
                opt = optim.Adam(model.parameters(), lr=config['learning_rate'])
                lr_sched = ReduceLROnPlateau(opt, patience=config['lr_patience'])
            
            # Reset early stopping to give full training a fresh start
            early = MultiLossEarlyStopping(config['early_patience'], config['min_delta'], 
                                          ['data_val', 'val_total'])
        
        # Adjust learning rate for current phase
        if current_phase == "physics_only" and training_config['physics_only_lr'] != config['learning_rate']:
            # Set physics-only learning rate
            for param_group in opt.param_groups:
                param_group['lr'] = training_config['physics_only_lr']
        elif current_phase == "pretrain" and training_config['pretrain_lr'] != config['learning_rate']:
            # Set pre-training learning rate
            for param_group in opt.param_groups:
                param_group['lr'] = training_config['pretrain_lr']
        elif current_phase == "full" and (pretrain_transition_epoch == epoch or physics_only_transition_epoch == epoch):
            # Reset to main learning rate when switching to full training
            for param_group in opt.param_groups:
                param_group['lr'] = config['learning_rate']
        
        # Training step
        model.train()
        tr_total = tr_data = 0.0
        # Conditionally set the number of physics components to track
        num_phys_components = 4 if args.synthetic else 6
        tr_phys_components = [0.0] * num_phys_components  # Use conditional number of physics components
        tr_raw_residuals = []
        
        for batch_idx, (xb, yb) in enumerate(data['train']):
            xb, yb = xb.to(device), yb.to(device)
            
            # Ensure double precision
            xb = xb.double()
            yb = yb.double()
            
            # Get model predictions
            y_pred = model(xb)
            
            # Get loss components and perform method-specific step
            loss_components = loss_method(model, xb, yb)
            total = loss_method.step(loss_components, opt, model, xb, yb, epoch, batch_idx)
            
            # Log magnitude analysis for first batch of each epoch (to avoid excessive data)
            if batch_idx == 0 and training_config['csv_logging']:
                current_phase = "pretrain" if loss_method.pretrain_mode else "full"
                debug_logger.log_magnitudes(epoch, current_phase, model, xb, yb, 
                                           data['Xmax'], data['Xmin'], data['ymax'], data['ymin'])
            
            # Collect raw residuals for this batch
            raw_residuals = collect_raw_residuals(model, xb, yb, y_pred, data, device)
            tr_raw_residuals.append(raw_residuals)
            
            # Accumulate losses
            tr_total += total.item()
            tr_data += loss_components[0].item()
            for i in range(num_phys_components):
                tr_phys_components[i] += loss_components[i+1].item()
        
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
        
        # Store current weights and training phase
        for key, value in loss_method.last_lambdas.items():
            hist[f'weight_{key}'].append(value)
        hist['training_phase'].append(current_phase)
        
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
                
                loss_components = loss_method(model, xb, yb)
                
                # Calculate total loss with current weights (no weight update in validation)
                if loss_method.running_weights is not None:
                    weighted_total = torch.sum(torch.stack([loss_method.running_weights[i] * loss_components[i] for i in range(len(loss_components))]))
                else:
                    weighted_total = torch.sum(torch.stack(loss_components))
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
        current_data_val_loss = hist['data_val'][-1]
        current_physics_val_loss = hist['phys_val'][-1]  # Physics validation loss
        
        # Handle phase-specific monitoring
        if current_phase == "physics_only":
            # Use physics validation loss for physics-only training
            if physics_only_early is not None:
                physics_only_early(current_physics_val_loss)
            if physics_only_lr_sched is not None:
                physics_only_lr_sched.step(current_physics_val_loss)
        elif current_phase == "pretrain":
            # Use data validation loss for pre-training
            if pretrain_early is not None:
                pretrain_early(current_data_val_loss)
            if pretrain_lr_sched is not None:
                pretrain_lr_sched.step(current_data_val_loss)
        else:
            # Use total validation loss for full training
            lr_sched.step(current_val_loss)
            early({'data_val': current_data_val_loss, 'val_total': current_val_loss})
            if early.early_stop:
                break
        
        # Save best model based on validation loss
        if current_val_loss < min(hist['val_total']):
            torch.save(model.state_dict(), model_path)
    
    # Save results and create plots
    print("Saving results and creating plots...")
    np.savez(os.path.join(output_dir, 'gradnorm_history.npz'), **hist)
    plot_training_history(hist, output_dir, 'GradNorm PINN', 
                         include_weights=True, include_method_params=True)
    
    # Create trajectory and phase space plots
    print("Creating trajectory and phase space visualizations...")
    plot_trajectory_predictions(model, data, output_dir, 'GradNorm PINN', device=device)
    
    # Always save the final model for parameter extraction
    torch.save(model.state_dict(), model_path)
    
    # Save final parameters for comparison
    final_params = collect_model_parameters(model)
    np.savez(os.path.join(output_dir, 'gradnorm_parameters.npz'), **final_params)
    
    print(f'Training complete. Model saved to {model_path}')
    print(f'Results saved to {output_dir}')
    print(f'Final parameters: {final_params}')
    print(f'Final weights: {loss_method.last_lambdas}')


if __name__ == '__main__':
    main() 