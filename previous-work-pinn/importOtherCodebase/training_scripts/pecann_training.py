#!/usr/bin/env python3
"""
PECANN PINN Training Script

Implements the Physics-Enhanced Constrained Artificial Neural Network (PECANN) method.
This method uses adaptive weighting with constraint violation tracking to balance data and physics losses.

Usage:
    python pecann_training.py [--epochs 2000] [--batch-size 256] [--lr 1e-4] [--max-samples 1000]
                             [--mu-initial 1.0] [--mu-max 1e4] [--epsilon 1e-8]
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
import argparse
from tqdm import tqdm

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


class PECANNLoss(nn.Module):
    """
    Physics-Enhanced Constrained Artificial Neural Network (PECANN) loss balancing method.
    
    This method implements the correct Augmented Lagrangian Method with conditional updates.
    """
    
    def __init__(self, mu_initial: float = 1.0, mu_max: float = 1e4, epsilon: float = 1e-8, enable_mass_constraints: bool = True):
        super().__init__()
        self.mu = mu_initial  # Initial penalty parameter
        self.mu_max = mu_max  # Maximum penalty parameter
        self.epsilon = epsilon  # Tolerance for constraint violation
        self.enable_mass_constraints = enable_mass_constraints  # Whether to include mass constraints
        
        # State variables
        self.lambdas = {}
        self.eta = 0.0  # Stores the previous constraint violation measure
        self._initialized = False
        
        # Loss component keys - conditionally include mass constraints
        if enable_mass_constraints:
            self.loss_keys = ['data', 'phys_res1', 'phys_res2', 'phys_res3', 'phys_res4', 'phys_mass1', 'phys_mass2']
        else:
            self.loss_keys = ['data', 'phys_res1', 'phys_res2', 'phys_res3', 'phys_res4']
        
        # Store current weights for logging
        self.current_weights = {key: 1.0 for key in self.loss_keys}
        
        # Store normalization parameters for dimensionless PINN compatibility
        self.X_max = None
        self.X_min = None
        self.y_max = None
        self.y_min = None
        
    def forward(self, model, X_batch, y_batch, X_max, X_min, y_max, y_min):
        """Calculate all loss components using adaptive_custom_loss."""
        # Store normalization parameters for later use in _get_constraint_residuals
        self.X_max = X_max
        self.X_min = X_min
        self.y_max = y_max
        self.y_min = y_min
        
        loss_components = adaptive_custom_loss(model, X_batch, y_batch, X_max, X_min, y_max, y_min)
        return loss_components
    
    def _get_constraint_residuals(self, model, X_batch, y_batch):
        """Get constraint residuals for ALM."""
        # Get model predictions
        y_pred = model(X_batch)
        
        # Data constraint residual (prediction vs target)
        c_data = y_pred - y_batch
        
        # Physics constraint residuals (from model) - need to pass normalization parameters for dimensionless PINN
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
        
        # Build constraint dictionary conditionally
        constraint_dict = {
            'data': c_data,
            'phys_res1': residual1,
            'phys_res2': residual2,
            'phys_res3': residual3,
            'phys_res4': residual4,
        }
        
        # Conditionally include mass constraints
        if self.enable_mass_constraints:
            constraint_dict['phys_mass1'] = residualMass1
            constraint_dict['phys_mass2'] = residualMass2
        
        return constraint_dict
    
    def step(self, loss_components, optimizer, model, X_batch, y_batch):
        """
        Performs optimization step with PECANN ALM weight adjustment.
        
        The algorithm:
        1. Get constraint residuals
        2. Initialize multipliers on first step
        3. Construct Augmented Lagrangian Loss
        4. Update network parameters
        5. Apply conditional update for multipliers and penalty
        """
        # 1. Get constraint residuals
        constraint_residuals = self._get_constraint_residuals(model, X_batch, y_batch)
        
        # 2. Initialize multipliers on first step
        if not self._initialized:
            for key in self.loss_keys:
                current_residual = constraint_residuals[key]
                self.lambdas[key] = torch.zeros_like(current_residual, requires_grad=False)
            self._initialized = True
        
        # 3. Check if batch size changed and resize tensors if needed
        batch_size = X_batch.shape[0]
        for key in self.loss_keys:
            current_residual = constraint_residuals[key]
            expected_size = current_residual.shape
            
            # Check if lambdas need to be resized
            if self.lambdas[key].shape != expected_size:
                print(f"Warning: Resizing lambdas for {key} from {self.lambdas[key].shape} to {expected_size}")
                self.lambdas[key] = torch.zeros_like(current_residual, requires_grad=False)
        
        # 4. Construct the Augmented Lagrangian Loss
        # L(θ,λ,μ) = J(θ) + <λ, C(θ)> + (μ/2) * ||C(θ)||^2
        
        # Objective loss (data loss)
        objective_loss = loss_components[0]  # Data loss
        
        # Penalty term: (μ/2) * ||C(θ)||^2
        penalty_term = 0.0
        for key in self.loss_keys:
            residual = constraint_residuals[key]
            penalty_term += torch.mean(residual**2)
        penalty_term = (self.mu / 2) * penalty_term
        
        # Multiplier term: <λ, C(θ)>
        multiplier_term = 0.0
        for key in self.loss_keys:
            residual = constraint_residuals[key]
            multiplier_term += torch.mean(self.lambdas[key] * residual)
        
        # Total ALM loss
        total_loss = objective_loss + penalty_term + multiplier_term
        
        # 5. Update network parameters
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # 6. Conditional update for multipliers (λ) and penalty (μ)
        with torch.no_grad():
            # Calculate current total constraint violation
            current_violation = 0.0
            for key in self.loss_keys:
                residual = constraint_residuals[key]
                current_violation += torch.mean(residual**2)
            current_violation = torch.sqrt(current_violation)

            # Algorithm 1 from PECANN paper: conditional update
            if (current_violation >= 0.25 * self.eta) and (current_violation > self.epsilon):
                # Update penalty parameter mu
                self.mu = min(2 * self.mu, self.mu_max)
                # Update Lagrange multipliers
                for key in self.loss_keys:
                    residual = constraint_residuals[key]
                    # Ensure lambda tensor exists and has correct size
                    if key not in self.lambdas or self.lambdas[key].shape != residual.shape:
                        self.lambdas[key] = torch.zeros_like(residual, requires_grad=False)
                    self.lambdas[key] += self.mu * residual

            # Record the current penalty loss for the next iteration's check
            try:
                self.eta = current_violation.item()
            except (AttributeError, RuntimeError):
                self.eta = 0.0
        
        # 7. Update current weights for logging (use penalty parameter as weight indicator)
        for key in self.loss_keys:
            self.current_weights[key] = self.mu
        
        return total_loss


def add_method_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add method-specific command-line arguments."""
    parser.add_argument('--mu-initial', type=float, default=1.0,
                       help='Initial penalty parameter (default: 1.0)')
    parser.add_argument('--mu-max', type=float, default=1e4,
                       help='Maximum penalty parameter (default: 1e4)')
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
    parser = argparse.ArgumentParser(description='PECANN PINN Training')
    parser = add_common_arguments(parser)
    parser = add_method_arguments(parser)
    args = parser.parse_args()
    
    # Parse configuration
    config = parse_common_arguments(args)
    method_params = {
        'mu_initial': args.mu_initial,
        'mu_max': args.mu_max,
        'epsilon': args.epsilon
    }
    
    # Setup device and output directory
    device = setup_device(config['device'])
    output_dir, model_path = create_output_directory('pecann_pinn', config['output_dir'])
    
    # Print configuration
    print_training_config(config, 'PECANN PINN', method_params)
    
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
    loss_method = PECANNLoss(enable_mass_constraints=not args.synthetic, **method_params)
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
        'weight_phys_mass2': [], 'mu': [], 'lambda_data_mean': [], 'lambda_physics_mean': [],
        'constraint_violation': []
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
            # PECANN doesn't return constraint_violation anymore, it's handled internally
            
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
        
        # Store current weights and method parameters
        for key, value in loss_method.current_weights.items():
            hist[f'weight_{key}'].append(value)
        
        hist['mu'].append(loss_method.mu)
        # PECANN uses mu and eta instead of lambda averages
        hist['lambda_data_mean'].append(loss_method.mu)
        hist['lambda_physics_mean'].append(loss_method.eta)
        # PECANN constraint violation is handled internally, use eta as indicator
        hist['constraint_violation'].append(loss_method.eta)
        
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
                # For PECANN, we need to reconstruct the ALM loss for validation
                constraint_residuals = loss_method._get_constraint_residuals(model, xb, yb)
                
                # Check if lambdas need to be resized for validation batch
                for key in loss_method.loss_keys:
                    current_residual = constraint_residuals[key]
                    expected_size = current_residual.shape

                    # Check if lambdas need to be resized
                    if loss_method.lambdas[key].shape != expected_size:
                        print(f"Warning: Resizing lambdas for {key} in validation from {loss_method.lambdas[key].shape} to {expected_size}")
                        loss_method.lambdas[key] = torch.zeros_like(current_residual, requires_grad=False)
                        # Ensure the tensor is on the same device
                        loss_method.lambdas[key] = loss_method.lambdas[key].to(current_residual.device)
                
                # Objective loss
                objective_loss = loss_components[0]

                # Penalty term
                penalty_term = 0.0
                for key in loss_method.loss_keys:
                    residual = constraint_residuals[key]
                    penalty_term += torch.mean(residual**2)
                penalty_term = (loss_method.mu / 2) * penalty_term

                # Multiplier term
                multiplier_term = 0.0
                for i, key in enumerate(loss_method.loss_keys):
                    residual = constraint_residuals[key]
                    lambda_tensor = loss_method.lambdas[key]

                    # Ensure tensors are on the same device
                    if lambda_tensor.device != residual.device:
                        lambda_tensor = lambda_tensor.to(residual.device)

                    # Ensure tensor sizes match for multiplication
                    if lambda_tensor.shape != residual.shape:
                        print(f"Warning: Size mismatch for {key}: lambda={lambda_tensor.shape}, residual={residual.shape}")
                        # Resize lambda to match residual
                        lambda_tensor = torch.zeros_like(residual, requires_grad=False).to(residual.device)
                        loss_method.lambdas[key] = lambda_tensor

                    multiplier_term += torch.mean(lambda_tensor * residual)
                
                weighted_total = objective_loss + penalty_term + multiplier_term
                
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
    np.savez(os.path.join(output_dir, 'pecann_history.npz'), **hist)
    plot_training_history(hist, output_dir, 'PECANN PINN', 
                         include_weights=True, include_method_params=True)
    
    # Create trajectory and phase space plots
    print("Creating trajectory and phase space visualizations...")
    plot_trajectory_predictions(model, data, output_dir, 'PECANN PINN', device=device)
    
    # Always save the final model for parameter extraction
    torch.save(model.state_dict(), model_path)
    
    # Save final parameters for comparison
    final_params = collect_model_parameters(model)
    np.savez(os.path.join(output_dir, 'pecann_parameters.npz'), **final_params)
    
    print(f'Training complete. Model saved to {model_path}')
    print(f'Results saved to {output_dir}')
    print(f'Final parameters: {final_params}')
    print(f'Final weights: {loss_method.current_weights}')
    print(f'Final mu: {loss_method.mu}')


if __name__ == '__main__':
    main() 