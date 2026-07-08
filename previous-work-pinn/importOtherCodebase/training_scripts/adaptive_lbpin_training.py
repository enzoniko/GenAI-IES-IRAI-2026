#!/usr/bin/env python3
"""
Adaptive LBPIN Training Script

Implements the Gaussian Likelihood Loss method with learnable log-sigmas for data & physics losses.
This method automatically adapts uncertainty estimates during training.

Usage:
    python adaptive_lbpin_training.py [--epochs 2000] [--batch-size 256] [--lr 1e-4] [--max-samples 1000]
                                     [--sigma-init 0.6931]
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


class GaussianLikelihoodLoss(nn.Module):
    """
    Gaussian likelihood loss with adaptive weighting for PINN training.
    
    This loss function implements adaptive weighting using Gaussian likelihood
    estimation to balance data and physics losses during training.
    """
    
    def __init__(self, sigma_init: float = 1.0, enable_mass_constraints: bool = True):
        super().__init__()
        self.sigma_init = sigma_init
        self.enable_mass_constraints = enable_mass_constraints

        # Loss component keys - conditionally include mass constraints
        if enable_mass_constraints:
            self.loss_keys = ['data', 'phys_res1', 'phys_res2', 'phys_res3', 'phys_res4', 'phys_mass1', 'phys_mass2']
        else:
            self.loss_keys = ['data', 'phys_res1', 'phys_res2', 'phys_res3', 'phys_res4']

        # Initialize log standard deviations for data and physics components
        self.log_eps_data = nn.Parameter(torch.tensor(0.0))

        # Initialize physics components - conditionally include mass constraints
        if enable_mass_constraints:
            self.log_eps_phys_components = nn.ParameterList([
                nn.Parameter(torch.tensor(0.0)) for _ in range(6)  # 4 residuals + 2 mass constraints
            ])
        else:
            self.log_eps_phys_components = nn.ParameterList([
                nn.Parameter(torch.tensor(0.0)) for _ in range(4)  # 4 residuals only
            ])

        # Store current sigma values for logging
        self._current_sigmas_dict = {}

        # For compatibility with validation code, provide current_weights
        self.current_weights = {key: 1.0 for key in self.loss_keys}

    def forward(self, model, X_batch, y_batch, X_max, X_min, y_max, y_min):
        """Calculate all loss components using adaptive_custom_loss."""
        loss_components = adaptive_custom_loss(model, X_batch, y_batch, X_max, X_min, y_max, y_min)
        return loss_components
    
    def step(self, loss_components, optimizer, model, X_batch, y_batch):
        """
        Performs optimization step with adaptive weighting.
        
        The algorithm:
        1. Extract data and physics losses
        2. Calculate weighted losses using Gaussian likelihood
        3. Update log standard deviations based on loss magnitudes
        4. Optimize both model parameters and log standard deviations
        """
        # Extract losses
        data_loss = loss_components[0]
        phys_losses = loss_components[1:]  # All physics losses
        
        # Calculate epsilons (standard deviations)
        eps2_d = torch.exp(self.log_eps_data * 2)
        
        # Calculate weighted data loss
        data_term = 0.5 / eps2_d * data_loss
        
        # Physics loss terms (individual for each component)
        phys_terms = []
        for i, phys_loss in enumerate(phys_losses):
            # Ensure we don't access beyond available components
            if i < len(self.log_eps_phys_components):
                eps2_p_i = torch.exp(self.log_eps_phys_components[i] * 2)
                phys_terms.append(0.5 / eps2_p_i * phys_loss)
            else:
                # For synthetic data, skip mass constraint components
                continue
        
        # Regularization terms
        data_reg = self.log_eps_data
        phys_reg = sum(self.log_eps_phys_components)
        
        # Total loss
        total_loss = data_term + sum(phys_terms) + data_reg + phys_reg
        
        # Optimize both model parameters and log standard deviations
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Update current sigma values for logging
        self._current_sigmas_dict = {
            'data': torch.exp(self.log_eps_data).item(),
            'phys_components': [torch.exp(log_eps).item() for log_eps in self.log_eps_phys_components]
        }
        
        return total_loss
    
    def current_sigmas(self):
        """Return current sigma values for logging."""
        sigma_data = torch.exp(self.log_eps_data)
        sigma_phys_components = [torch.exp(log_eps) for log_eps in self.log_eps_phys_components]
        return sigma_data, sigma_phys_components


def add_method_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add method-specific command-line arguments."""
    parser.add_argument('--sigma-init', type=float, default=0.6931, 
                       help='Initial sigma values (log scale, default: 0.6931 ~= log(2))')

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
    parser = argparse.ArgumentParser(description='Adaptive LBPIN Training')
    parser = add_common_arguments(parser)
    parser = add_method_arguments(parser)
    args = parser.parse_args()
    
    # Parse configuration
    config = parse_common_arguments(args)
    method_params = {'sigma_init': args.sigma_init}
    
    # Setup device and output directory
    device = setup_device(config['device'])
    output_dir, model_path = create_output_directory('adaptive_lbpin', config['output_dir'])
    
    # Print configuration
    print_training_config(config, 'Adaptive LBPIN', method_params)
    
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
    loss_method = GaussianLikelihoodLoss(enable_mass_constraints=not args.synthetic, **method_params)
    
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
        'sigma_data': [], 'sigma_phys': [],
        'sigma_phys_res1': [], 'sigma_phys_res2': [], 'sigma_phys_res3': [],
        'sigma_phys_res4': []
    })
    
    # Conditionally add mass constraint sigma keys
    if not args.synthetic:
        hist.update({
            'sigma_phys_mass1': [], 'sigma_phys_mass2': []
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
                
                # Calculate total loss (no method-specific updates in validation)
                data_loss = loss_components[0]
                phys_losses = loss_components[1:]
                
                # Data loss term
                eps2_d = torch.exp(loss_method.log_eps_data * 2)
                data_term = 0.5 / eps2_d * data_loss
                
                # Physics loss terms (individual for each component)
                phys_terms = []
                for i, phys_loss in enumerate(phys_losses):
                    # Ensure we don't access beyond available components
                    if i < len(loss_method.log_eps_phys_components):
                        eps2_p_i = torch.exp(loss_method.log_eps_phys_components[i] * 2)
                        phys_terms.append(0.5 / eps2_p_i * phys_loss)
                    else:
                        # For synthetic data, skip mass constraint components
                        continue
                
                # Regularization terms
                data_reg = loss_method.log_eps_data
                phys_reg = sum(loss_method.log_eps_phys_components)
                
                # Total loss
                total_val = data_term + sum(phys_terms) + data_reg + phys_reg
                
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
        
        # Log current sigma values
        sigma_data, sigma_phys_components = loss_method.current_sigmas()
        hist['sigma_data'].append(sigma_data.item())
        hist['sigma_phys'].append(sum(sigma_phys_components).item() / len(sigma_phys_components))  # Average for backward compatibility
        
        # Log individual sigma values - conditionally handle mass constraints
        sigma_names = ['sigma_phys_res1', 'sigma_phys_res2', 'sigma_phys_res3', 'sigma_phys_res4']
        if not args.synthetic:
            sigma_names.extend(['sigma_phys_mass1', 'sigma_phys_mass2'])
        
        for i, name in enumerate(sigma_names):
            if i < len(sigma_phys_components):
                hist[name].append(sigma_phys_components[i].item())
        
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
    np.savez(os.path.join(output_dir, 'adaptive_lbpin_history.npz'), **hist)
    plot_training_history(hist, output_dir, 'Adaptive LBPIN', 
                         include_weights=False, include_method_params=True)
    
    # Create trajectory and phase space plots
    print("Creating trajectory and phase space visualizations...")
    plot_trajectory_predictions(model, data, output_dir, 'Adaptive LBPIN', device=device)
    
    # Always save the final model for parameter extraction
    torch.save(model.state_dict(), model_path)
    
    # Save final parameters for comparison
    final_params = collect_model_parameters(model)
    np.savez(os.path.join(output_dir, 'adaptive_lbpin_parameters.npz'), **final_params)
    
    print(f'Training complete. Model saved to {model_path}')
    print(f'Results saved to {output_dir}')
    print(f'Final parameters: {final_params}')
    print(f'Final sigmas - Data: {sigma_data:.4f}, Physics components: {[f"{s:.4f}" for s in sigma_phys_components]}')


if __name__ == '__main__':
    main() 