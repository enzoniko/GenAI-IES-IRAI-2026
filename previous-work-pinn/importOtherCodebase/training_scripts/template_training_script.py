#!/usr/bin/env python3
"""
Template Training Script

This is a template for creating refactored training scripts with command-line
argument support and common utilities. Copy this file and modify it for each
specific loss balancing method.
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

from models.basicPINNv8 import ConfigurablePINN, adaptive_custom_loss
from common_utils import (
    MultiLossEarlyStopping, ReduceLROnPlateau, prepare_data, load_and_prepare_data,
    get_pinn_config, get_param_init_config, setup_device, create_output_directory,
    plot_training_history, add_common_arguments, parse_common_arguments, print_training_config,
    prepare_synthetic_data, collect_raw_residuals, collect_model_parameters, 
    initialize_comprehensive_history, update_comprehensive_history
)


class TemplateLossMethod(nn.Module):
    """
    Template loss method class.
    
    Replace this with the specific loss balancing method implementation.
    """
    
    def __init__(self, param1: float = 1.0, param2: float = 0.1):
        super().__init__()
        self.param1 = param1
        self.param2 = param2
        # Add method-specific initialization here
        
    def forward(self, model, X_batch, y_batch, X_max, X_min, y_max, y_min):
        """Calculate all loss components using adaptive_custom_loss."""
        loss_components = adaptive_custom_loss(model, X_batch, y_batch, X_max, X_min, y_max, y_min)
        return loss_components
    
    def step(self, loss_components, optimizer, model, X_batch, y_batch):
        """
        Performs optimization step with the specific method.
        
        Replace this with the method-specific optimization logic.
        """
        # Method-specific optimization logic here
        total_loss = torch.stack(loss_components).sum()
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return total_loss


def add_method_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add method-specific command-line arguments."""
    parser.add_argument('--param1', type=float, default=1.0, 
                       help='Method parameter 1 (default: 1.0)')
    parser.add_argument('--param2', type=float, default=0.1, 
                       help='Method parameter 2 (default: 0.1)')
    
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
    parser = argparse.ArgumentParser(description='Template Training Script')
    parser = add_common_arguments(parser)
    parser = add_method_arguments(parser)
    args = parser.parse_args()
    
    # Parse configuration
    config = parse_common_arguments(args)
    method_params = {
        'param1': args.param1,
        'param2': args.param2
    }
    
    # Setup device and output directory
    device = setup_device(config['device'])
    output_dir, model_path = create_output_directory('template_method', config['output_dir'])
    
    # Print configuration
    print_training_config(config, 'Template Method', method_params)
    
    # Load and prepare data
    print("Loading and preparing data...")
    
    # Check if using synthetic data
    if args.synthetic:
        if args.simulation_id is None:
            raise ValueError("--simulation-id must be provided when using --synthetic")
        print(f"Loading synthetic data from simulation {args.simulation_id}")
        X, y, metadata = load_and_prepare_data(
            data_path=args.data_path,
            max_samples=config['max_samples'],
            synthetic=True,
            simulation_id=args.simulation_id
        )
        
        # FIX: Use helper function to avoid double normalization
        data = prepare_synthetic_data(X, y, metadata, config['batch_size'], config['max_samples'])
    else:
        print("Loading experimental data")
        X, y, metadata = load_and_prepare_data(
            data_path=args.data_path,
            max_samples=config['max_samples'],
            synthetic=False
        )
        # The original path remains unchanged for experimental data
        data = prepare_data(X, y, config['batch_size'], config['max_samples'])
    
    # Create PINN configuration
    pinn_config = get_pinn_config(
        config['hidden_layers'], config['activation'], 
        config['dropout_rate'], config['init_method']
    )
    param_init_config = get_param_init_config(synthetic=args.synthetic)
    
    # Initialize model and loss function
    print("Initializing model and loss function...")
    model = ConfigurablePINN(pinn_config, pinn_config, param_init_config, enable_mass_constraints=not args.synthetic).to(device)
    loss_method = TemplateLossMethod(**method_params).to(device)
    opt = optim.Adam(model.parameters(), lr=config['learning_rate'])
    lr_sched = ReduceLROnPlateau(opt, patience=config['lr_patience'])
    
    # Initialize early stopping and history containers
    early = MultiLossEarlyStopping(config['early_patience'], config['min_delta'], 
                                  ['data_val', 'val_total'])
    
    # Determine number of physics components based on synthetic flag
    num_phys_components = 4 if args.synthetic else 6
    phys_keys = ['phys_res1', 'phys_res2', 'phys_res3', 'phys_res4']
    if not args.synthetic:
        phys_keys.extend(['phys_mass1', 'phys_mass2'])
    
    hist = {
        'train_total': [], 'val_total': [],
        'data_train': [], 'data_val': [],
        'phys_train': [], 'phys_val': [],
        'phys_res1_train': [], 'phys_res2_train': [], 'phys_res3_train': [],
        'phys_res4_train': [],
        'phys_res1_val': [], 'phys_res2_val': [], 'phys_res3_val': [],
        'phys_res4_val': []
    }
    
    # Conditionally add mass constraint keys
    if not args.synthetic:
        hist.update({
            'phys_mass1_train': [], 'phys_mass2_train': [],
            'phys_mass1_val': [], 'phys_mass2_val': []
        })
    
    # Training loop
    print("Starting training...")
    for epoch in tqdm(range(config['epochs']), desc="Training"):
        # Training step
        model.train()
        tr_total = tr_data = 0.0
        tr_phys_components = [0.0] * num_phys_components
        
        for xb, yb in data['train']:
            xb, yb = xb.to(device), yb.to(device)
            
            # Get loss components and perform method-specific step
            loss_components = loss_method(xb, yb, data['Xmax'].to(device), 
                                        data['Xmin'].to(device), data['ymax'].to(device), 
                                        data['ymin'].to(device))
            total = loss_method.step(loss_components, opt, model, xb, yb)
            
            # Accumulate losses
            tr_total += total.item()
            tr_data += loss_components[0].item()
            for i in range(num_phys_components):
                tr_phys_components[i] += loss_components[i+1].item()
        
        # Average over batches
        n_train = len(data['train'])
        hist['train_total'].append(tr_total / n_train)
        hist['data_train'].append(tr_data / n_train)
        hist['phys_train'].append(sum(tr_phys_components) / (num_phys_components * n_train))
        
        # Individual physics components
        for i, key in enumerate(phys_keys):
            hist[f'{key}_train'].append(tr_phys_components[i] / n_train)
        
        # Validation step
        model.eval()
        val_total = val_data = 0.0
        val_phys_components = [0.0] * num_phys_components
        
        with torch.no_grad():
            for xb, yb in data['val']:
                xb, yb = xb.to(device), yb.to(device)
                loss_components = loss_method(xb, yb, data['Xmax'].to(device),
                                           data['Xmin'].to(device), data['ymax'].to(device),
                                           data['ymin'].to(device))
                
                # Calculate total loss (no method-specific updates in validation)
                total_val = torch.stack(loss_components).sum()
                val_total += total_val.item()
                val_data += loss_components[0].item()
                for i in range(num_phys_components):
                    val_phys_components[i] += loss_components[i+1].item()
        
        # Average over validation batches
        n_val = len(data['val'])
        hist['val_total'].append(val_total / n_val)
        hist['data_val'].append(val_data / n_val)
        hist['phys_val'].append(sum(val_phys_components) / (num_phys_components * n_val))
        
        # Individual validation physics components
        for i, key in enumerate(phys_keys):
            hist[f'{key}_val'].append(val_phys_components[i] / n_val)
        
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
    np.savez(os.path.join(output_dir, 'template_method_history.npz'), **hist)
    plot_training_history(hist, output_dir, 'Template Method', 
                         include_weights=False, include_method_params=False)
    
    print(f'Training complete. Model saved to {model_path}')
    print(f'Results saved to {output_dir}')


if __name__ == '__main__':
    main() 