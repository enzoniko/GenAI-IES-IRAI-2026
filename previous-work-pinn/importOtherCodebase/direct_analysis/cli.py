"""
Command-line Interface for Direct PINN Analysis

This module provides a command-line interface for running direct PINN
analysis and visualization functions.
"""

import sys
import time
import os
import torch
import numpy as np
import argparse

# add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from direct_analysis.data_utils import prepare_data, prepare_dataset_for_residuals, normalize_data
from direct_analysis.training import train_model, train_model_adaptive
from direct_analysis.residuals import compute_residuals
from direct_analysis.adaptive_loss import adaptive_custom_loss
from direct_analysis.cross_validation import k_fold_cross_validation, random_search, train_best_model
# Add these missing imports:
from direct_analysis.visualization.boxplots import plot_residuals_by_variable_and_frequency
from direct_analysis.visualization.metrics_plots import plot_metrics_for_publication
from direct_analysis.visualization.data_utils import create_plots_from_saved_data, run_visualization_only

from Models.basicPINNv3 import PINN, custom_loss
from Models.basicPINNv7 import ConfigurablePINN, get_default_pinn_config
from Data.LoadData import data_paths, get_omegas

def main(output_dir=None, args=None):
    """
    Main function to run the direct PINN analysis pipeline
    """
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create timestamp and setup output directory
    timestamp = int(time.time())
    if output_dir is None:
        output_dir = f'results/direct_pinn_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    model_save_path = os.path.join(models_dir, "direct_pinn_model_final.pth")
    
    # Training configuration parameters
    max_samples = 10000       # Maximum number of samples to use for training
    batch_size = 100         # Batch size for training
    epochs = 10 # int(1e6)         # Maximum number of epochs
    
    print("Loading and preparing data...")
    # Load normal data first
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
    
    # Prepare data for training
    data_dict = prepare_data(X, y, batch_size=batch_size, normalize=True, 
                            test_size=0.1, val_size=0.2, max_samples=max_samples)
    
    # Save normalization parameters
    torch.save(data_dict['X_min'], os.path.join(models_dir, 'X_min.pth'))
    torch.save(data_dict['X_max'], os.path.join(models_dir, 'X_max.pth'))
    torch.save(data_dict['y_min'], os.path.join(models_dir, 'y_min.pth'))
    torch.save(data_dict['y_max'], os.path.join(models_dir, 'y_max.pth'))
    
    # Use the configurable model or the standard model based on arguments
    if args and args.use_configurable_model:
        print("\nInitializing configurable PINN model...")
        model_config = get_default_pinn_config()
        model = ConfigurablePINN(
            unmeasured_net_config=model_config['unmeasured_net_config'],
            acceleration_net_config=model_config['acceleration_net_config'],
            param_init_config=model_config['param_init_config']
        ).to(device)
    else:
        print("\nInitializing standard PINN model...")
        model = PINN().to(device)
    
    # Train the model with either standard or improved optimizer
    if args and args.improved_optimizer:
        print("\nUsing improved adaptive optimizer...")
        print("Training will show Lagrange multipliers (λd=data, λp=physics, λr=regularization) in progress bar")
        model, history = train_model_adaptive(
            model=model,
            data_dict=data_dict,
            device=device,
            custom_loss_fn=adaptive_custom_loss,
            epochs=epochs,
            alpha=0.9,                # Weight smoothing factor
            learning_rate=1e-4,       # Learning rate for manual parameter update (reduced from 1e-3)
            update_frequency=100,       # Update frequency for weights
            model_save_path=model_save_path,
            early_stopping_patience=2000,
            lr_patience=1000
        )
        
        # Save training history with adaptive optimizer fields
        np.savez(os.path.join(models_dir, 'training_history.npz'), 
                train_loss=np.array(history['train_loss']),
                val_loss=np.array(history['val_loss']),
                data_weight=np.array(history['data_weight']),
                phys_weight=np.array(history['phys_weight']),
                reg_weight=np.array(history['reg_weight']),
                data_loss=np.array(history['data_loss']),
                phys_loss=np.array(history['phys_loss']),
                reg_loss=np.array(history['reg_loss']))
    else:
        print("\nUsing standard optimizer...")
        model, history = train_model(
            model=model,
            data_dict=data_dict,
            device=device,
            custom_loss_fn=custom_loss,
            epochs=epochs,
            early_stopping_patience=2000,
            lr_patience=1000,
            initial_alpha=1000.0,
            initial_beta=1.0,
            model_save_path=model_save_path
        )
        
        # Save training history with standard optimizer fields
        np.savez(os.path.join(models_dir, 'training_history.npz'), 
                train_loss=np.array(history['train_loss']),
                val_loss=np.array(history['val_loss']),
                alpha=np.array(history['alpha']),
                beta=np.array(history['beta']))
    
    # Compute residuals for all datasets
    print("\nComputing residuals for all datasets...")
    all_residuals = {}
    
    # Prepare data loaders for all datasets
    residual_data_loaders = {}
    
    for key in list(data_paths.keys()):
        print(f"\nProcessing {key} data...")
        X_temp = torch.load(f'Data/X_{key}.pth')[:, :249998, :]
        y_temp = torch.load(f'Data/Y_{key}.pth')[:, :249998, :]
        
        # Use all available data
        X_temp = X_temp.clone()  # Clone to avoid modifying the original data
        y_temp = y_temp.clone()
        
        # Add time feature to X
        t_temp = torch.arange(0, 2e-5 * X_temp.size(1), 2e-5).view(1, -1, 1).expand(X_temp.size(0), -1, -1)[:, :249998, :]
        X_temp = torch.cat((X_temp, t_temp), dim=2)
        
        # Keep the truncation as needed for the direct_pinn approach
        X_temp = X_temp[:, :249998, :]
        y_temp = y_temp[:, :249998, :]
        
        # Reshape
        X_temp = X_temp.reshape(-1, X_temp.size(-1))
        y_temp = y_temp.reshape(-1, y_temp.size(-1))
        
        # Create data loader
        loader = prepare_dataset_for_residuals(
            X_temp, y_temp, device, batch_size=batch_size,
            X_min=data_dict['X_min'], X_max=data_dict['X_max'],
            y_min=data_dict['y_min'], y_max=data_dict['y_max']
        )
        
        residual_data_loaders[key] = loader
    
    # Compute residuals using standardized approach
    print("\nComputing residuals for all datasets...")
    residuals = compute_residuals(
        model=model,
        data_dict=residual_data_loaders,
        device=device,
        include_physical_residuals=True,
        compute_dtw=not args.skip_dtw
    )
    
    # Standardize and compress residuals
    print("\nStandardizing and compressing residuals...")

    # Save residuals with compression enabled
    print("\nSaving residuals incrementally...")
    residuals_filename = os.path.join(models_dir, "direct_pinn_residuals.pth")
    torch.save(residuals, residuals_filename)
    print(f"Saved residuals to {residuals_filename}")

    # Print the optimized model parameters again for reference
    print("\nFinal optimized model parameters:")
    for name, param in model.named_parameters():
        if not name.startswith('layers'):  # Only print physical parameters, not network weights
            print(f"{name}: {param.item():.6f}")

    # Test loading
    try:
        loaded_residuals = torch.load(residuals_filename)
        print(f"Successfully loaded residuals with {len(loaded_residuals)} data types")
    except Exception as e:
        print(f"Error testing loading of residuals: {e}")
    
    # Generate plots from the residuals
    print("\nGenerating plots...")
    plot_residuals_by_variable_and_frequency(residuals, output_dir)
    plot_metrics_for_publication(residuals, output_dir)
    
    # Create README file with instructions
    readme_path = os.path.join(output_dir, "README.md")
    data_dir = os.path.join(output_dir, "data")
    
    readme_content = f"""# Direct PINN Analysis Results

This directory contains results from the direct PINN analysis pipeline.

## Directory Structure:
- {os.path.join(output_dir, 'models')}/: Contains trained model file and residuals
- {os.path.join(output_dir, 'data')}/: Contains the raw data for all plots in NPZ format
- {os.path.join(output_dir, 'plots')}/: Contains boxplot visualizations for each data type
- {os.path.join(output_dir, 'comparisons')}/: Contains metrics comparison plots

## Trained Model:
The Direct PINN model is trained to predict the system dynamics directly from the physics-informed neural network approach.

## How to regenerate plots from saved data:
```python
from direct_analysis.visualization.data_utils import create_plots_from_saved_data

# Regenerate plots from saved data
create_plots_from_saved_data(data_dir='{data_dir}')

# To use a custom output prefix:
create_plots_from_saved_data(data_dir='{data_dir}', output_prefix='custom')
```

You can also use the CLI:
```bash
python -m direct_analysis.cli recreate {data_dir}
```

## How to visualize saved residuals:
```bash
python -m direct_analysis.cli visualize {residuals_path}
```

The saved data includes all metrics, statistics, and visualization parameters used in the original plots.
You can modify the output formats and visualizations without recomputing the model training and residuals.
"""

    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"Created README file with instructions at {readme_path}")
    print(f"\nDirect PINN analysis completed successfully!")
    
    return output_dir

def run_cross_validation(args):
    """
    Run k-fold cross-validation for the Direct PINN model.
    
    Parameters:
    - args: Command-line arguments
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    timestamp = int(time.time())
    output_dir = args.output_dir if args.output_dir else f'results/direct_pinn_cv_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    X = torch.load('Data/X_normal.pth')[:, :249998, :]
    y = torch.load('Data/Y_normal.pth')[:, :249998, :]
    
    # Add time feature to X
    t = torch.arange(0, 2e-5 * X.size(1), 2e-5).view(1, -1, 1).expand(X.size(0), -1, -1)[:, :249998, :]
    X = torch.cat((X, t), dim=2)
    
    # Reshape the tensors to 2D
    X = X.reshape(-1, X.size(-1))
    y = y.reshape(-1, y.size(-1))
    
    # Get default model configuration
    model_config = get_default_pinn_config()
    
    # Training configuration
    training_config = {
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'alpha': 0.9,
        'learning_rate': args.learning_rate,
        'update_frequency': 100,
        'early_stopping_patience': 200,
        'lr_patience': 100
    }
    
    # Run cross-validation
    cv_results = k_fold_cross_validation(
        X=X,
        y=y,
        k=args.k_folds,
        model_config=model_config,
        training_config=training_config,
        output_dir=output_dir,
        device=device,
        max_samples=args.max_samples
    )
    
    print("\nCross-validation complete.")
    print(f"Results saved to {output_dir}")
    
    return output_dir

def run_random_search(args):
    """
    Run random search for hyperparameter optimization.
    
    Parameters:
    - args: Command-line arguments
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    timestamp = int(time.time())
    output_dir = args.output_dir if args.output_dir else f'results/direct_pinn_rs_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    X = torch.load('Data/X_normal.pth')[:, :120000, :] # Take only the first 120000 samples 
    y = torch.load('Data/Y_normal.pth')[:, :120000, :]
    
    # Add time feature to X
    t = torch.arange(0, 2e-5 * X.size(1), 2e-5).view(1, -1, 1).expand(X.size(0), -1, -1)[:, :120000, :]
    X = torch.cat((X, t), dim=2)
    
    # Reshape the tensors to 2D
    X = X.reshape(-1, X.size(-1))
    y = y.reshape(-1, y.size(-1))
    
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
                    'M1': (1.0, 15.0), 'M2': (1.0, 15.0), 'M3': (1.0, 16.0),
                    'D1': (1.0, 15.0), 'D2': (1.0, 15.0), 'D3': (1.0, 15.0),
                    'K1': (1.0, 15.0), 'K2': (1.0, 15.0), 'E1': (1.0, 15.0)
                }
            ]
        }
    }
    
    # Define default training config space
    training_config_space = {
        'batch_size': [64, 100, 128, 256],
        'epochs': [args.epochs if hasattr(args, 'epochs') else 10],
        'alpha': [0.8, 0.9, 0.95],
        'learning_rate': [1e-2, 1e-3, 1e-4],
        'update_frequency': [50, 100, 200],
        'early_stopping_patience': [20],
        'lr_patience': [10]
    }
    
    # Run random search
    search_results = random_search(
        X=X,
        y=y,
        n_trials=args.n_trials,
        model_config_space=model_config_space,
        training_config_space=training_config_space,
        cv_folds=args.k_folds,
        output_dir=output_dir,
        device=device,
        max_samples=args.max_samples
    )
    
    # Train the best model if requested
    if not args.skip_final_training:
        print("\nTraining best model on full dataset...")
        best_model_dir = os.path.join(output_dir, "best_model")
        
        model, history, data_dict = train_best_model(
            X=X,
            y=y,
            best_model_config=search_results['best_model_config'],
            best_training_config=search_results['best_training_config'],
            output_dir=best_model_dir,
            device=device
        )
        
        # Compute residuals
        print("\nComputing residuals with the best model...")
        residual_data_loaders = {}
        
        for key in list(data_paths.keys()):
            print(f"\nProcessing {key} data...")
            X_temp = torch.load(f'Data/X_{key}.pth')[:, :249998, :]
            y_temp = torch.load(f'Data/Y_{key}.pth')[:, :249998, :]
            
            # Add time feature to X
            t_temp = torch.arange(0, 2e-5 * X_temp.size(1), 2e-5).view(1, -1, 1).expand(X_temp.size(0), -1, -1)[:, :249998, :]
            X_temp = torch.cat((X_temp, t_temp), dim=2)
            
            # Reshape
            X_temp = X_temp.reshape(-1, X_temp.size(-1))
            y_temp = y_temp.reshape(-1, y_temp.size(-1))
            
            # Create data loader
            loader = prepare_dataset_for_residuals(
                X_temp, y_temp, device, batch_size=100,
                X_min=data_dict['X_min'], X_max=data_dict['X_max'],
                y_min=data_dict['y_min'], y_max=data_dict['y_max']
            )
            
            residual_data_loaders[key] = loader
        
        # Compute residuals
        residuals = compute_residuals(
            model=model,
            data_dict=residual_data_loaders,
            device=device,
            include_physical_residuals=True,
            compute_dtw=not args.skip_dtw
        )
        
        # Save residuals
        residuals_filename = os.path.join(best_model_dir, "best_model_residuals.pth")
        torch.save(residuals, residuals_filename)
        
        # Generate plots
        plot_residuals_by_variable_and_frequency(residuals, best_model_dir)
        plot_metrics_for_publication(residuals, best_model_dir)
    
    print("\nRandom search complete.")
    print(f"Results saved to {output_dir}")
    
    return output_dir

def run_cli():
    """
    Command-line interface entry point for the direct PINN analysis.
    """
    parser = argparse.ArgumentParser(description="Direct PINN Analysis CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Full analysis command
    parser_run = subparsers.add_parser("run", help="Run full training and analysis")
    parser_run.add_argument("--output-dir", default=None, help="Directory to save all outputs")
    parser_run.add_argument("--skip-dtw", action="store_true", default=True,
                        help="Skip computation of DTW distances to save time")
    parser_run.add_argument("--improved-optimizer", action="store_true", default=True,
                        help="Use the improved adaptive optimizer for training")
    parser_run.add_argument("--use-configurable-model", action="store_true", default=False,
                        help="Use the configurable PINN model (basicPINNv7) instead of the standard model")
    
    # Cross-validation command
    parser_cv = subparsers.add_parser("cross-validation", help="Run k-fold cross-validation")
    parser_cv.add_argument("--k-folds", type=int, default=5, help="Number of folds (default: 5)")
    parser_cv.add_argument("--output-dir", default=None, help="Directory to save outputs")
    parser_cv.add_argument("--max-samples", type=int, default=10000, help="Maximum number of samples (default: 10000)")
    parser_cv.add_argument("--epochs", type=int, default=1000, help="Maximum number of epochs (default: 1000)")
    parser_cv.add_argument("--batch-size", type=int, default=100, help="Batch size (default: 100)")
    parser_cv.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    
    # Random search command
    parser_rs = subparsers.add_parser("random-search", help="Run random search for hyperparameter optimization")
    parser_rs.add_argument("--n-trials", type=int, default=10, help="Number of random trials (default: 10)")
    parser_rs.add_argument("--k-folds", type=int, default=3, help="Number of CV folds per trial (default: 3)")
    parser_rs.add_argument("--output-dir", default=None, help="Directory to save outputs")
    parser_rs.add_argument("--max-samples", type=int, default=10000, help="Maximum number of samples (default: 10000)")
    parser_rs.add_argument("--skip-dtw", action="store_true", default=True, help="Skip DTW distance computation")
    parser_rs.add_argument("--skip-final-training", action="store_true", default=False, help="Skip final training of best model")
    # Visualization-only command
    parser_visualize = subparsers.add_parser("visualize", help="Run only visualization for saved residuals")
    parser_visualize.add_argument("residuals_path", help="Path to the saved residuals file (.pth)")
    parser_visualize.add_argument("--output-dir", help="Directory to save visualization outputs")
    
    # Recreate plots command
    parser_recreate = subparsers.add_parser("recreate", help="Recreate plots from saved data")
    parser_recreate.add_argument("data_dir", help="Directory containing saved data files (.npz)")
    parser_recreate.add_argument("--output-dir", help="Directory to save recreated plots")
    
    #args = parser.parse_args()

    #if args.command == "random-search" or args.command is None:
    # Create a hardcoded args for random search
    args = argparse.Namespace()
    args.output_dir = "results/random_search"
    args.n_trials = 10
    args.k_folds = 3
    args.max_samples = 10000
    args.skip_dtw = True
    args.skip_final_training = False

    run_random_search(args)

    """ if args.command == "run" or args.command is None:
        if hasattr(args, 'output_dir') and args.output_dir:
            output_dir = args.output_dir
            os.makedirs(output_dir, exist_ok=True)
        else:
            timestamp = int(time.time())
            output_dir = f'results/direct_pinn_{timestamp}'
            os.makedirs(output_dir, exist_ok=True)
        
        main(output_dir, args)
    
    elif args.command == "cross-validation":
        run_cross_validation(args)
    
    elif args.command == "random-search":
        
    
    elif args.command == "visualize":
        output_dir = args.output_dir if args.output_dir else f'results/direct_pinn_vis_{int(time.time())}'
        os.makedirs(output_dir, exist_ok=True)
        run_visualization_only(args.residuals_path, output_dir)
    
    elif args.command == "recreate":
        output_dir = args.output_dir if args.output_dir else f'results/direct_pinn_recreated_{int(time.time())}'
        os.makedirs(output_dir, exist_ok=True)
        create_plots_from_saved_data(args.data_dir, output_dir)
    
    else:
        parser.print_help() """

if __name__ == "__main__":
    run_cli() 