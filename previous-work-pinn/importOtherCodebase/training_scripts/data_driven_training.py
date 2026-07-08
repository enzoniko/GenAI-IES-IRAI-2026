#!/usr/bin/env python3
"""
Data-Driven Model Training Script with Optional Regularization

This script trains a pure data-driven MLP model using the same architecture parameters
as the best PINN models but without physics constraints. Supports optional regularization
for fair baseline comparisons.

Architecture Parameters (from best PINN):
- Hidden layers: [128, 128]
- Activation: elu
- Dropout rate: 0.24900923127192412
- Init method: xavier_uniform

Training Parameters:
- Epochs: 20000
- Batch size: 256
- Learning rate: 0.0039199623708041885
- Early stopping patience: 100
- LR scheduler patience: 50

Regularization Options:
- None: Pure data-driven (MSE only)
- L2: L2 weight regularization
- L1: L1 weight regularization
- Tikhonov: Tikhonov regularization (provides similar benefits to Jacobian regularization)

Usage:
    python data_driven_training.py [--data-path Data] [--max-samples 1000] [--output-dir results/data_driven]
                                   [--regularization l2] [--lambda-reg 1e-4]
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.basicPINNv8 import ConfigurableMLP
from training_scripts.common_utils import (
    MultiLossEarlyStopping, ReduceLROnPlateau,
    get_pinn_config, setup_device, create_output_directory,
    plot_training_history, add_common_arguments, parse_common_arguments, print_training_config,
    unified_data_preparation, ensure_double_precision, verify_data_structure, plot_trajectory_predictions
)


class DataDrivenLoss(nn.Module):
    """
    Data-driven loss with optional regularization for baseline comparisons.
    """

    def __init__(self, regularization=None, lambda_reg=1e-4, regularization_type='l2'):
        """
        Args:
            regularization: Type of regularization ('l2', 'l1', 'tikhonov', 'jacobian', None)
            lambda_reg: Regularization strength
            regularization_type: Specific type for certain regularizations
        """
        super().__init__()
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.regularization_type = regularization_type

    def _compute_regularization(self, model, X_norm):
        """Compute regularization term."""
        if self.regularization is None:
            return 0.0

        reg_loss = 0.0

        if self.regularization == 'l2':
            # L2 regularization on weights
            for param in model.parameters():
                reg_loss += torch.sum(param ** 2)

        elif self.regularization == 'l1':
            # L1 regularization on weights
            for param in model.parameters():
                reg_loss += torch.sum(torch.abs(param))

        elif self.regularization == 'tikhonov':
            # Tikhonov regularization (ridge regression style)
            for param in model.parameters():
                reg_loss += torch.sum(param ** 2)

        # Note: Jacobian regularization removed due to gradient computation issues
        # Tikhonov regularization provides similar benefits for regularization

        return self.lambda_reg * reg_loss

    def forward(self, model, X_batch, y_batch, X_max, X_min, y_max, y_min):
        """Calculate loss with optional regularization."""
        # Normalize inputs
        X_norm = (X_batch - X_min) / (X_max - X_min + 1e-12)

        # Get model predictions
        y_pred_norm = model(X_norm)

        # Denormalize predictions
        y_pred = y_pred_norm * (y_max - y_min) + y_min

        # Calculate data loss
        mse_loss = nn.MSELoss()(y_pred, y_batch)

        # Add regularization if specified
        reg_loss = self._compute_regularization(model, X_norm)

        total_loss = mse_loss + reg_loss

        return total_loss

    def step(self, loss, optimizer, model, X_batch, y_batch):
        """Perform optimization step."""
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss


def add_method_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add method-specific command-line arguments."""
    # Synthetic data arguments
    parser.add_argument('--synthetic', action='store_true',
                       help='Use synthetic data instead of experimental data')
    parser.add_argument('--simulation-id', type=int, default=None,
                       help='Simulation ID to use (required if --synthetic is used)')
    parser.add_argument('--data-path', type=str, default='Data',
                       help='Path to data directory (default: Data)')

    # Regularization arguments
    parser.add_argument('--regularization', type=str, choices=['l2', 'l1', 'tikhonov', None],
                       default=None, help='Type of regularization to apply')
    parser.add_argument('--lambda-reg', type=float, default=1e-4,
                       help='Regularization strength (default: 1e-4)')
    parser.add_argument('--regularization-type', type=str, default='l2',
                       help='Specific regularization type (for future extensions)')

    return parser


def main():
    """Main training function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Data-Driven Model Training')
    parser = add_common_arguments(parser)
    parser = add_method_arguments(parser)
    args = parser.parse_args()

    # Set fixed configuration based on best PINN parameters
    config = {
        'epochs': 20000,
        'batch_size': 256,
        'learning_rate': 0.0039199623708041885,
        'early_patience': 100,
        'lr_patience': 50,
        'hidden_layers': [128, 128],
        'activation': 'elu',
        'dropout_rate': 0.24900923127192412,
        'init_method': 'xavier_uniform',
        'max_samples': getattr(args, 'max_samples', 1000),
        'device': getattr(args, 'device', 'cpu'),
        'output_dir': getattr(args, 'output_dir', 'results/data_driven'),
        'regularization': getattr(args, 'regularization', None),
        'lambda_reg': getattr(args, 'lambda_reg', 1e-4),
        'regularization_type': getattr(args, 'regularization_type', 'l2')
    }

    method_params = {
        'regularization': config['regularization'],
        'lambda_reg': config['lambda_reg'],
        'regularization_type': config['regularization_type']
    }

    # Setup device and output directory
    device = setup_device(config['device'])
    output_dir, model_path = create_output_directory('data_driven', config['output_dir'])

    # Print configuration
    print_training_config(config, 'Data-Driven Model', method_params)

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

    # Verify data structure
    sample_batch = next(iter(data['train']))
    X_sample, y_sample = sample_batch
    verify_data_structure(X_sample, y_sample, synthetic=args.synthetic)

    print(f"Data loaded successfully:")
    print(f"  - Training batches: {len(data['train'])}")
    print(f"  - Validation batches: {len(data['val'])}")
    print(f"  - Input features: {X_sample.size(-1)}")
    print(f"  - Output features: {y_sample.size(-1)}")

    # Create model architecture
    print("Initializing data-driven model...")
    model = ConfigurableMLP(
        input_dim=X_sample.size(-1),  # 10 features
        hidden_layers=config['hidden_layers'],
        output_dim=y_sample.size(-1),  # 4 outputs
        activation=config['activation'],
        dropout_rate=config['dropout_rate'],
        init_method=config['init_method']
    ).to(device).double()

    # Create loss function with regularization if specified
    loss_method = DataDrivenLoss(
        regularization=args.regularization,
        lambda_reg=args.lambda_reg,
        regularization_type=args.regularization_type
    )

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    lr_scheduler = ReduceLROnPlateau(optimizer, patience=config['lr_patience'])

    # Initialize early stopping
    early_stopping = MultiLossEarlyStopping(config['early_patience'], 1e-6, ['val_loss'])

    # Add data-driven specific tracking
    train_losses = []
    val_losses = []

    # Training loop
    print("Starting training...")
    for epoch in tqdm(range(config['epochs']), desc="Training"):
        # Training step
        model.train()
        train_loss = 0.0

        for xb, yb in data['train']:
            xb, yb = xb.to(device), yb.to(device)

            # Ensure double precision for model compatibility
            xb = xb.double()
            yb = yb.double()

            # Calculate loss and perform step
            loss = loss_method(model, xb, yb, data['Xmax'].to(device),
                             data['Xmin'].to(device), data['ymax'].to(device),
                             data['ymin'].to(device))
            total_loss = loss_method.step(loss, optimizer, model, xb, yb)

            train_loss += total_loss.item()

        # Average training loss
        avg_train_loss = train_loss / len(data['train'])
        train_losses.append(avg_train_loss)

        # Validation step
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for xb, yb in data['val']:
                xb, yb = xb.to(device), yb.to(device)

                # Ensure double precision for model compatibility
                xb = xb.double()
                yb = yb.double()

                # Calculate validation loss
                loss = loss_method(model, xb, yb, data['Xmax'].to(device),
                                 data['Xmin'].to(device), data['ymax'].to(device),
                                 data['ymin'].to(device))

                val_loss += loss.item()

        # Average validation loss
        avg_val_loss = val_loss / len(data['val'])
        val_losses.append(avg_val_loss)

        # Learning rate scheduling and early stopping
        lr_scheduler.step(avg_val_loss)

        # Save best model (skip on first epoch when val_losses is empty)
        if len(val_losses) > 1 and avg_val_loss < min(val_losses[:-1]):
            torch.save(model.state_dict(), model_path)

        early_stopping({'val_loss': avg_val_loss})
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    # Save results and create plots
    print("Saving results and creating plots...")

    # Create history dictionary for saving
    history_to_save = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'epochs': list(range(len(train_losses)))
    }
    np.savez(os.path.join(output_dir, 'data_driven_history.npz'), **history_to_save)

    # Create a simplified history dictionary for plotting
    plot_history = {
        'train_total': train_losses,
        'val_total': val_losses,
        # Add dummy keys expected by plot_training_history (data-driven has no physics)
        'data_train': train_losses,
        'data_val': val_losses,
        'phys_train': [0.0] * len(train_losses),  # No physics loss for data-driven
        'phys_val': [0.0] * len(val_losses),
        # Add individual physics components (all zero for data-driven)
        'phys_res1_train': [0.0] * len(train_losses),
        'phys_res2_train': [0.0] * len(train_losses),
        'phys_res3_train': [0.0] * len(train_losses),
        'phys_res4_train': [0.0] * len(train_losses),
        'phys_mass1_train': [0.0] * len(train_losses),
        'phys_mass2_train': [0.0] * len(train_losses),
        'phys_res1_val': [0.0] * len(val_losses),
        'phys_res2_val': [0.0] * len(val_losses),
        'phys_res3_val': [0.0] * len(val_losses),
        'phys_res4_val': [0.0] * len(val_losses),
        'phys_mass1_val': [0.0] * len(val_losses),
        'phys_mass2_val': [0.0] * len(val_losses)
    }
    plot_training_history(plot_history, output_dir, 'Data-Driven Model',
                         include_weights=False, include_method_params=False)

    # Create trajectory predictions plot
    print("Creating trajectory visualizations...")

    # Ensure data is in double precision for compatibility with model
    data_for_plotting = data.copy()
    data_for_plotting['Xmax'] = data['Xmax'].double()
    data_for_plotting['Xmin'] = data['Xmin'].double()
    data_for_plotting['ymax'] = data['ymax'].double()
    data_for_plotting['ymin'] = data['ymin'].double()

    plot_trajectory_predictions(model, data_for_plotting, output_dir, 'Data-Driven Model', device=device)

    # Always save the final model for parameter extraction
    torch.save(model.state_dict(), model_path)

    # Save scaling parameters for use in preprocessing
    scaling_params_path = os.path.join(output_dir, 'scaling_params.npz')
    np.savez(scaling_params_path,
             Xmin=data['Xmin'].cpu().numpy(),
             Xmax=data['Xmax'].cpu().numpy(),
             ymin=data['ymin'].cpu().numpy(),
             ymax=data['ymax'].cpu().numpy())
    print(f'Scaling parameters saved to {scaling_params_path}')

    # Save final model information (data-driven models don't have physical parameters)
    model_info = {
        'input_dim': model.input_dim,
        'output_dim': model.output_dim,
        'hidden_layers': model.hidden_layers,
        'activation': model.activation,
        'dropout_rate': model.dropout_rate,
        'init_method': model.init_method,
        'total_params': sum(p.numel() for p in model.parameters())
    }
    np.savez(os.path.join(output_dir, 'data_driven_parameters.npz'), **model_info)

    print(f'Training complete. Model saved to {model_path}')
    print(f'Results saved to {output_dir}')
    print(f'Model info: {model_info}')


if __name__ == '__main__':
    main()
