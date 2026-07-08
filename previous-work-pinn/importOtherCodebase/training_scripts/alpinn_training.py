#!/usr/bin/env python3
"""
AL-PINN Training Script

Implements the Adaptive Learning Physics-Informed Neural Network (AL-PINN) method.
This method uses learnable Lagrangian multipliers to adaptively balance data and physics losses.

Usage:
    python alpinn_training.py [--epochs 2000] [--batch-size 256] [--lr 1e-4] [--max-samples 1000]
                             [--beta 1.0] [--lambda-lr 1e-4]
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
    MultiLossEarlyStopping, ReduceLROnPlateau, prepare_data, load_and_prepare_data,
    get_pinn_config, get_param_init_config, setup_device, create_output_directory,
    plot_training_history, add_common_arguments, parse_common_arguments, print_training_config,
    prepare_synthetic_data, collect_raw_residuals, collect_model_parameters, 
    initialize_comprehensive_history, update_comprehensive_history,
    unified_data_preparation, ensure_double_precision, verify_data_structure, plot_trajectory_predictions
)


class ALPINNLoss(nn.Module):
    """
    ALPINN (Adaptive Loss Physics-Informed Neural Network) loss balancing method.
    
    This method implements adaptive loss balancing using Lagrangian multipliers
    to ensure all physics constraints are satisfied during training.
    """
    
    def __init__(self, beta: float = 1.0, lambda_lr: float = 1e-4, enable_mass_constraints: bool = True):
        super().__init__()
        self.beta = beta  # Balancing parameter
        self.lambda_lr = lambda_lr  # Learning rate for Lagrangian multipliers
        self.enable_mass_constraints = enable_mass_constraints
        
        # Initialize Lagrangian multipliers for each loss component
        self.lambda_data = nn.Parameter(torch.tensor(1.0))
        self.lambda_phys_res1 = nn.Parameter(torch.tensor(1.0))
        self.lambda_phys_res2 = nn.Parameter(torch.tensor(1.0))
        self.lambda_phys_res3 = nn.Parameter(torch.tensor(1.0))
        self.lambda_phys_res4 = nn.Parameter(torch.tensor(1.0))
        
        # Conditionally initialize mass constraint multipliers
        if enable_mass_constraints:
            self.lambda_phys_mass1 = nn.Parameter(torch.tensor(1.0))
            self.lambda_phys_mass2 = nn.Parameter(torch.tensor(1.0))
        
        # Store current lambda values for logging
        self.current_lambdas = {}
        
        # State variables
        self.lambdas = {}
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
    
    def step(self, loss_components, optimizer, model, X_batch, y_batch):
        """
        Performs optimization step with ALPINN weight adjustment.
        
        The algorithm:
        1. Extract data and physics losses
        2. Calculate weighted losses with Lagrangian multipliers
        3. Add regularization terms
        4. Optimize both model and Lagrangian parameters
        """
        # Extract losses
        data_loss = loss_components[0]
        phys_losses = loss_components[1:]  # All physics losses
        
        # Calculate weighted loss with Lagrangian multipliers
        weighted_data_loss = self.lambda_data * data_loss
        
        # Base physics losses (always present)
        weighted_phys_losses = [
            self.lambda_phys_res1 * phys_losses[0],
            self.lambda_phys_res2 * phys_losses[1],
            self.lambda_phys_res3 * phys_losses[2],
            self.lambda_phys_res4 * phys_losses[3],
        ]
        
        # Conditionally add mass constraint losses
        if self.enable_mass_constraints and len(phys_losses) > 4:
            weighted_phys_losses.extend([
                self.lambda_phys_mass1 * phys_losses[4],
                self.lambda_phys_mass2 * phys_losses[5]
            ])
        
        # Regularization term to encourage λ ≈ 1
        lambda_reg_terms = [
            (self.lambda_data - 1.0)**2,
            (self.lambda_phys_res1 - 1.0)**2,
            (self.lambda_phys_res2 - 1.0)**2,
            (self.lambda_phys_res3 - 1.0)**2,
            (self.lambda_phys_res4 - 1.0)**2,
        ]
        
        # Conditionally add mass constraint regularization
        if self.enable_mass_constraints:
            lambda_reg_terms.extend([
                (self.lambda_phys_mass1 - 1.0)**2,
                (self.lambda_phys_mass2 - 1.0)**2
            ])
        
        lambda_reg = self.beta * sum(lambda_reg_terms)
        
        # Total loss
        total_loss = weighted_data_loss + sum(weighted_phys_losses) + lambda_reg
        
        # Optimize both model parameters and Lagrangian multipliers
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Update current lambda values for logging
        self.current_lambdas = {
            'data': self.lambda_data.item(),
            'phys_res1': self.lambda_phys_res1.item(),
            'phys_res2': self.lambda_phys_res2.item(),
            'phys_res3': self.lambda_phys_res3.item(),
            'phys_res4': self.lambda_phys_res4.item(),
        }
        
        # Conditionally add mass constraint lambdas
        if self.enable_mass_constraints:
            self.current_lambdas.update({
                'phys_mass1': self.lambda_phys_mass1.item(),
                'phys_mass2': self.lambda_phys_mass2.item(),
            })
        
        return total_loss


def add_method_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add method-specific command-line arguments."""
    parser.add_argument('--beta', type=float, default=1.0,
                       help='Regularization strength for Lagrangian multipliers (default: 1.0)')
    parser.add_argument('--lambda-lr', type=float, default=1e-4,
                       help='Learning rate for Lagrangian multipliers (default: 1e-4)')

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
    parser = argparse.ArgumentParser(description='AL-PINN Training')
    parser = add_common_arguments(parser)
    parser = add_method_arguments(parser)
    args = parser.parse_args()
    
    # Parse configuration
    config = parse_common_arguments(args)
    method_params = {
        'beta': args.beta,
        'lambda_lr': args.lambda_lr
    }
    
    # Setup device and output directory
    device = setup_device(config['device'])
    output_dir, model_path = create_output_directory('alpinn', config['output_dir'])
    
    # Print configuration
    print_training_config(config, 'AL-PINN', method_params)
    
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
    loss_method = ALPINNLoss(enable_mass_constraints=not args.synthetic, **method_params)
    
    # Optimizer includes both model and loss function parameters
    opt = optim.Adam(list(model.parameters()) + list(loss_method.parameters()), 
                    lr=config['learning_rate'])
    lr_sched = ReduceLROnPlateau(opt, patience=config['lr_patience'])
    
    # Initialize early stopping and comprehensive history containers
    early = MultiLossEarlyStopping(config['early_patience'], config['min_delta'], 
                                  ['data_val', 'val_total'])
    hist = initialize_comprehensive_history(synthetic=args.synthetic)
    
    # Add method-specific tracking
    hist.update({
        'lambda_data': [], 'lambda_phys_res1': [], 'lambda_phys_res2': [],
        'lambda_phys_res3': [], 'lambda_phys_res4': []
    })
    
    # Conditionally add mass constraint lambda keys
    if not args.synthetic:
        hist.update({
            'lambda_phys_mass1': [], 'lambda_phys_mass2': []
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
        
        # Log current lambda values
        for key, value in loss_method.current_lambdas.items():
            hist_key = f'lambda_{key}'
            if hist_key in hist:
                hist[hist_key].append(value)
        
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
                
                # Calculate total loss with current lambdas (no lambda update in validation)
                weighted_data_loss = loss_method.lambda_data * loss_components[0]
                weighted_phys_losses = [
                    loss_method.lambda_phys_res1 * loss_components[1],
                    loss_method.lambda_phys_res2 * loss_components[2],
                    loss_method.lambda_phys_res3 * loss_components[3],
                    loss_method.lambda_phys_res4 * loss_components[4],
                ]
                
                # Conditionally add mass constraint losses
                if loss_method.enable_mass_constraints and len(loss_components) > 5:
                    weighted_phys_losses.extend([
                        loss_method.lambda_phys_mass1 * loss_components[5],
                        loss_method.lambda_phys_mass2 * loss_components[6]
                    ])
                
                # Regularization terms
                lambda_reg_terms = [
                    (loss_method.lambda_data - 1.0)**2 + 
                    (loss_method.lambda_phys_res1 - 1.0)**2 + 
                    (loss_method.lambda_phys_res2 - 1.0)**2 + 
                    (loss_method.lambda_phys_res3 - 1.0)**2 + 
                    (loss_method.lambda_phys_res4 - 1.0)**2
                ]
                
                # Conditionally add mass constraint regularization
                if loss_method.enable_mass_constraints:
                    lambda_reg_terms.extend([
                        (loss_method.lambda_phys_mass1 - 1.0)**2 + 
                        (loss_method.lambda_phys_mass2 - 1.0)**2
                    ])
                
                lambda_reg = method_params['beta'] * sum(lambda_reg_terms)
                
                total_val = weighted_data_loss + sum(weighted_phys_losses) + lambda_reg
                
                val_total += total_val.item()
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
    np.savez(os.path.join(output_dir, 'alpinn_history.npz'), **hist)
    plot_training_history(hist, output_dir, 'AL-PINN', 
                         include_weights=False, include_method_params=True)
    
    # Create trajectory and phase space plots
    print("Creating trajectory and phase space visualizations...")
    plot_trajectory_predictions(model, data, output_dir, 'AL-PINN', device=device)
    
    # Always save the final model for parameter extraction
    torch.save(model.state_dict(), model_path)
    
    # Save final parameters for comparison
    final_params = collect_model_parameters(model)
    np.savez(os.path.join(output_dir, 'alpinn_parameters.npz'), **final_params)
    
    print(f'Training complete. Model saved to {model_path}')
    print(f'Results saved to {output_dir}')
    print(f'Final parameters: {final_params}')
    print(f'Final lambdas: {loss_method.current_lambdas}')


if __name__ == '__main__':
    main() 