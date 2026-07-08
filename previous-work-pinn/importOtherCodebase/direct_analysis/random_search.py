#!/usr/bin/env python3
"""
Random Search Script for Direct PINN Optimization

This script runs a random search to find the best hyperparameters
for the Direct PINN models.
"""

import os
import sys
import torch
import numpy as np
import argparse
import time

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from direct_analysis.cross_validation import random_search, train_best_model
from Models.basicPINNv7 import get_default_pinn_config

def main():
    """
    Main function to run the random search.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Random Search for Direct PINN Optimization")
    
    parser.add_argument("--output-dir", default=None, type=str,
                       help="Directory to save results (default: results/direct_pinn_rs_<timestamp>)")
    
    parser.add_argument("--n-trials", default=10, type=int,
                       help="Number of random trials to run (default: 10)")
    
    parser.add_argument("--cv-folds", default=3, type=int,
                       help="Number of cross-validation folds (default: 3)")
    
    parser.add_argument("--max-samples", default=10000, type=int,
                       help="Maximum number of samples to use (default: 10000)")
    
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use for training (default: cuda if available, else cpu)")
    
    parser.add_argument("--epochs", default=10, type=int,
                       help="Maximum number of epochs per trial (default: 500)")
    
    parser.add_argument("--skip-final-training", action="store_true",
                       help="Skip training the best model on the full dataset")
    
    args = parser.parse_args()
    
    # Create output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = int(time.time())
        output_dir = f'results/direct_pinn_rs_{timestamp}'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Print configuration
    print("Random Search Configuration:")
    print(f"  Number of trials: {args.n_trials}")
    print(f"  Cross-validation folds: {args.cv_folds}")
    print(f"  Maximum samples: {args.max_samples}")
    print(f"  Device: {args.device}")
    print(f"  Output directory: {output_dir}")
    
    # Define default model config space
    model_config_space = {
        'unmeasured_net_config': {
            'hidden_layers': [
                [32, 32], [64, 64], [128, 128], 
                [32, 32, 32], [64, 64, 64]
            ],
            'activation': ['tanh', 'relu', 'leaky_relu'],
            'dropout_rate': [0.0, 0.1, 0.2],
            'init_method': ['xavier_normal', 'kaiming_normal']
        },
        'acceleration_net_config': {
            'hidden_layers': [
                [32, 32], [64, 64], [128, 128], 
                [32, 32, 32], [64, 64, 64]
            ],
            'activation': ['tanh', 'relu', 'leaky_relu'],
            'dropout_rate': [0.0, 0.1, 0.2],
            'init_method': ['xavier_normal', 'kaiming_normal']
        },
        'param_init_config': {
            'method': ['fixed', 'uniform'],
            'values': [
                # Fixed values configuration
                {
                    'M1': 10.0, 'M2': 10.0, 'M3': 11.0,
                    'D1': 10.0, 'D2': 10.0, 'D3': 10.0,
                    'K1': 10.0, 'K2': 10.0, 'E1': 10.0
                },
                # Uniform ranges configuration (min, max tuples)
                {
                    'M1': (5.0, 15.0), 'M2': (5.0, 15.0), 'M3': (6.0, 16.0),
                    'D1': (5.0, 15.0), 'D2': (5.0, 15.0), 'D3': (5.0, 15.0),
                    'K1': (5.0, 15.0), 'K2': (5.0, 15.0), 'E1': (5.0, 15.0)
                }
            ]
        }
    }
    
    # Define default training config space
    training_config_space = {
        'batch_size': [64, 100, 128, 256],
        'epochs': [args.epochs],  # Use the provided value
        'alpha': [0.8, 0.9, 0.95],
        'learning_rate': [1e-2, 1e-3, 1e-4],
        'update_frequency': [50, 100, 200],
        'early_stopping_patience': [50, 100, 200],
        'lr_patience': [25, 50, 100]
    }
    
    # Load data
    print("\nLoading data...")
    X = torch.load('Data/X_normal.pth')[:, :249998, :]
    y = torch.load('Data/Y_normal.pth')[:, :249998, :]
    
    print(f"Original data shapes: X={X.shape}, y={y.shape}")
    
    # Add time feature to X
    t = torch.arange(0, 2e-5 * X.size(1), 2e-5).view(1, -1, 1).expand(X.size(0), -1, -1)[:, :249998, :]
    X = torch.cat((X, t), dim=2)
    
    print(f"After adding time feature: X={X.shape}, y={y.shape}")
    
    # Reshape the tensors to 2D
    X = X.reshape(-1, X.size(-1))
    y = y.reshape(-1, y.size(-1))
    
    print(f"After reshaping: X={X.shape}, y={y.shape}")
    
    # Run random search
    print("\nStarting random search...")
    search_results = random_search(
        X=X,
        y=y,
        n_trials=args.n_trials,
        model_config_space=model_config_space,
        training_config_space=training_config_space,
        cv_folds=args.cv_folds,
        output_dir=output_dir,
        device=args.device,
        max_samples=args.max_samples
    )
    
    # Train best model on full dataset if requested
    if not args.skip_final_training:
        print("\nTraining best model on full dataset...")
        best_model_dir = os.path.join(output_dir, "best_model")
        
        model, history, data_dict = train_best_model(
            X=X,
            y=y,
            best_model_config=search_results['best_model_config'],
            best_training_config=search_results['best_training_config'],
            output_dir=best_model_dir,
            device=args.device
        )
        
        # Save the final trained model
        print(f"\nBest model training complete. Model saved to {best_model_dir}")
    
    print("\nRandom search complete!")

if __name__ == "__main__":
    main() 