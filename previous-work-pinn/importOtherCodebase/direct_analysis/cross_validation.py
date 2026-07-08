"""
Cross-Validation and Hyperparameter Search Module for Direct PINN Training

This module provides functions for performing k-fold cross-validation
and hyperparameter search for Direct PINN models.
"""

import torch
import numpy as np
import os
from tqdm import tqdm
import random
import time
from torch.utils.data import DataLoader, TensorDataset, random_split, SubsetRandomSampler
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

# Import from the same package
from direct_analysis.data_utils import normalize_data, prepare_data
from direct_analysis.training import train_model_adaptive
from direct_analysis.adaptive_loss import adaptive_custom_loss

# Import from Models package
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from Models.basicPINNv7 import ConfigurablePINN, get_default_pinn_config


def k_fold_cross_validation(X, y, k=5, model_config=None, training_config=None, 
                           output_dir=None, device='cpu', max_samples=None):
    """
    Perform k-fold cross-validation for training Direct PINN models.
    
    Parameters:
    - X: Input tensor
    - y: Target tensor
    - k: Number of folds (default: 5)
    - model_config: Configuration for the ConfigurablePINN model
    - training_config: Configuration for the training process
    - output_dir: Directory to save the outputs
    - device: Device to use for training ('cpu' or 'cuda')
    - max_samples: Maximum number of samples to use (for faster execution)
    
    Returns:
    - List of trained models
    - Cross-validation scores
    - Dictionary with validation metrics
    """
    # Process configs
    if model_config is None:
        model_config = get_default_pinn_config()
    
    if training_config is None:
        training_config = {
            'batch_size': 100,
            'epochs': 1000,
            'alpha': 0.9,  # Weight smoothing factor
            'learning_rate': 1e-4,
            'update_frequency': 100
        }
    
    if output_dir is None:
        timestamp = int(time.time())
        output_dir = f'results/direct_pinn_cv_{timestamp}'
    
    os.makedirs(output_dir, exist_ok=True)
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Ensure we have 2D tensors (samples x features)
    if X.dim() > 2:
        X = X.reshape(-1, X.size(-1))
    if y.dim() > 2:
        y = y.reshape(-1, y.size(-1))
    
    # Take a subset of samples if specified
    if max_samples is not None and max_samples < X.size(0):
        indices = torch.randperm(X.size(0))[:max_samples]
        X = X[indices]
        y = y[indices]
    
    # Create dataset
    dataset = TensorDataset(X, y)
    
    # Initialize k-fold cross-validation
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_indices = list(kf.split(range(len(dataset))))
    
    # Keep track of models and scores
    cv_models = []
    cv_scores = []
    training_histories = []
    
    # Cross-validation loop
    for fold, (train_idx, val_idx) in enumerate(fold_indices):
        print(f"\n=== Fold {fold+1}/{k} ===")
        
        # Create data samplers
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        
        # Create data loaders
        train_loader = DataLoader(dataset, batch_size=training_config['batch_size'], 
                                 sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=training_config['batch_size'], 
                               sampler=val_sampler)
        
        # Compute normalization parameters from training data
        X_train = torch.stack([dataset[i][0] for i in train_idx])
        y_train = torch.stack([dataset[i][1] for i in train_idx])
        
        X_norm, X_min, X_max = normalize_data(X_train, method='min_max')
        y_norm, y_min, y_max = normalize_data(y_train, method='min_max')
        
        # Move normalization parameters to device
        X_min = X_min.to(device)
        X_max = X_max.to(device)
        y_min = y_min.to(device)
        y_max = y_max.to(device)
        
        # Prepare normalized dataset
        train_norm = TensorDataset(X_norm, y_norm)
        train_loader_norm = DataLoader(train_norm, batch_size=training_config['batch_size'], 
                                      shuffle=True)
        
        # Extract and normalize validation data
        X_val = torch.stack([dataset[i][0] for i in val_idx])
        y_val = torch.stack([dataset[i][1] for i in val_idx])
        X_val_norm = normalize_data(X_val, min_values=X_min.cpu(), max_values=X_max.cpu(), 
                                   method='min_max')
        y_val_norm = normalize_data(y_val, min_values=y_min.cpu(), max_values=y_max.cpu(), 
                                   method='min_max')
        val_norm = TensorDataset(X_val_norm, y_val_norm)
        val_loader_norm = DataLoader(val_norm, batch_size=training_config['batch_size'], 
                                    shuffle=False)
        
        # Initialize model
        model = ConfigurablePINN(
            unmeasured_net_config=model_config['unmeasured_net_config'],
            acceleration_net_config=model_config['acceleration_net_config'],
            param_init_config=model_config['param_init_config']
        ).to(device)
        
        # Create data dictionary for trainer
        data_dict = {
            'train_loader': train_loader_norm,
            'val_loader': val_loader_norm,
            'X_min': X_min,
            'X_max': X_max,
            'y_min': y_min,
            'y_max': y_max,
            'normalized': True
        }
        
        # Train the model
        model_save_path = os.path.join(models_dir, f"fold_{fold+1}_model.pth")
        model, history = train_model_adaptive(
            model=model,
            data_dict=data_dict,
            device=device,
            custom_loss_fn=adaptive_custom_loss,
            epochs=training_config['epochs'],
            alpha=training_config['alpha'],
            learning_rate=training_config['learning_rate'],
            update_frequency=training_config['update_frequency'],
            model_save_path=model_save_path,
            early_stopping_patience=training_config.get('early_stopping_patience', 100),
            lr_patience=training_config.get('lr_patience', 50)
        )
        
        # Save the training history
        history_path = os.path.join(models_dir, f"fold_{fold+1}_history.npz")
        np.savez(history_path, 
                train_loss=np.array(history['train_loss']),
                val_loss=np.array(history['val_loss']),
                data_weight=np.array(history['data_weight']),
                phys_weight=np.array(history['phys_weight']),
                reg_weight=np.array(history['reg_weight']),
                data_loss=np.array(history['data_loss']),
                phys_loss=np.array(history['phys_loss']),
                reg_loss=np.array(history['reg_loss']))
        
        # Keep track of the model, score, and history
        cv_models.append(model)
        cv_scores.append(min(history['val_loss']))  # Use best validation loss as score
        training_histories.append(history)
        
        # Save normalization parameters
        torch.save(X_min, os.path.join(models_dir, f'fold_{fold+1}_X_min.pth'))
        torch.save(X_max, os.path.join(models_dir, f'fold_{fold+1}_X_max.pth'))
        torch.save(y_min, os.path.join(models_dir, f'fold_{fold+1}_y_min.pth'))
        torch.save(y_max, os.path.join(models_dir, f'fold_{fold+1}_y_max.pth'))
    
    # Compute cross-validation stats
    cv_scores = np.array(cv_scores)
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    
    print(f"\nCross-validation complete.")
    print(f"Mean score: {cv_mean:.6f} ± {cv_std:.6f}")
    
    # Create summary plot
    plt.figure(figsize=(15, 10))
    
    # Plot validation loss across folds
    plt.subplot(2, 2, 1)
    for fold, history in enumerate(training_histories):
        plt.plot(history['val_loss'], label=f'Fold {fold+1}')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss Across Folds')
    plt.legend()
    plt.grid(True)
    
    # Plot data vs physics weight across folds
    plt.subplot(2, 2, 2)
    for fold, history in enumerate(training_histories):
        plt.plot(history['data_weight'], label=f'Fold {fold+1} (Data)')
        plt.plot(history['phys_weight'], ':', label=f'Fold {fold+1} (Physics)')
    plt.xlabel('Epoch')
    plt.ylabel('Weight')
    plt.title('Adaptive Weights Across Folds')
    plt.legend()
    plt.grid(True)
    
    # Plot model parameters across folds
    plt.subplot(2, 2, 3)
    param_names = ['M1', 'M2', 'M3', 'D1', 'D2', 'D3', 'K1', 'K2', 'E1']
    param_values = []
    
    for model in cv_models:
        model_params = []
        for param_name in param_names:
            param_value = getattr(model, param_name).item()
            model_params.append(param_value)
        param_values.append(model_params)
    
    param_values = np.array(param_values)
    param_means = np.mean(param_values, axis=0)
    param_stds = np.std(param_values, axis=0)
    
    plt.bar(param_names, param_means, yerr=param_stds, capsize=5)
    plt.ylabel('Parameter Value')
    plt.title('Model Parameters Across Folds')
    plt.grid(True)
    
    # Plot scores per fold
    plt.subplot(2, 2, 4)
    plt.bar(range(1, k+1), cv_scores)
    plt.axhline(y=cv_mean, color='r', linestyle='-', label=f'Mean: {cv_mean:.4f}')
    plt.axhline(y=cv_mean+cv_std, color='r', linestyle=':', label=f'±Std: {cv_std:.4f}')
    plt.axhline(y=cv_mean-cv_std, color='r', linestyle=':')
    plt.xlabel('Fold')
    plt.ylabel('Best Validation Loss')
    plt.title('Cross-Validation Scores')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cross_validation_summary.png'))
    
    # Return cross-validation results
    cv_results = {
        'models': cv_models,
        'scores': cv_scores,
        'mean_score': cv_mean,
        'std_score': cv_std,
        'histories': training_histories,
        'param_means': param_means,
        'param_stds': param_stds,
        'param_names': param_names
    }
    
    return cv_results


def random_search(X, y, n_trials=10, model_config_space=None, training_config_space=None, 
                 cv_folds=3, output_dir=None, device='cpu', max_samples=None):
    """
    Perform random search over hyperparameters with cross-validation.
    
    Parameters:
    - X: Input tensor
    - y: Target tensor
    - n_trials: Number of random trials
    - model_config_space: Dictionary with hyperparameter ranges for model
    - training_config_space: Dictionary with hyperparameter ranges for training
    - cv_folds: Number of cross-validation folds
    - output_dir: Directory to save the outputs
    - device: Device to use for training ('cpu' or 'cuda')
    - max_samples: Maximum number of samples to use (for faster execution)
    
    Returns:
    - Best model configuration
    - Best training configuration
    - Summary of all trials
    """
    # Setup default config spaces if not provided
    if model_config_space is None:
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
                    {
                        'M1': 10.0, 'M2': 10.0, 'M3': 11.0,
                        'D1': 10.0, 'D2': 10.0, 'D3': 10.0,
                        'K1': 10.0, 'K2': 10.0, 'E1': 10.0
                    },
                    {
                        'M1': (5.0, 15.0), 'M2': (5.0, 15.0), 'M3': (6.0, 16.0),
                        'D1': (5.0, 15.0), 'D2': (5.0, 15.0), 'D3': (5.0, 15.0),
                        'K1': (5.0, 15.0), 'K2': (5.0, 15.0), 'E1': (5.0, 15.0)
                    }
                ]
            }
        }
    
    if training_config_space is None:
        training_config_space = {
            'batch_size': [64, 100, 128, 256],
            'epochs': [500, 1000, 1500],
            'alpha': [0.8, 0.9, 0.95],
            'learning_rate': [1e-2, 1e-3, 1e-4],
            'update_frequency': [50, 100, 200],
            'early_stopping_patience': [50, 100, 200],
            'lr_patience': [25, 50, 100]
        }
    
    if output_dir is None:
        timestamp = int(time.time())
        output_dir = f'results/direct_pinn_rs_{timestamp}'
    
    os.makedirs(output_dir, exist_ok=True)
    trial_dir = os.path.join(output_dir, "trials")
    os.makedirs(trial_dir, exist_ok=True)
    
    # Keep track of all trials
    trial_results = []
    best_score = float('inf')
    best_model_config = None
    best_training_config = None
    best_trial_idx = -1
    
    for trial in range(n_trials):
        print(f"\n=== Trial {trial+1}/{n_trials} ===")
        
        # Sample random configurations
        # Get corresponding values based on the selected method
        method = random.choice(model_config_space['param_init_config']['method'])
        values_idx = 0 if method == 'fixed' else 1
        
        model_config = {
            'unmeasured_net_config': {
                'hidden_layers': random.choice(model_config_space['unmeasured_net_config']['hidden_layers']),
                'activation': random.choice(model_config_space['unmeasured_net_config']['activation']),
                'dropout_rate': random.choice(model_config_space['unmeasured_net_config']['dropout_rate']),
                'init_method': random.choice(model_config_space['unmeasured_net_config']['init_method'])
            },
            'acceleration_net_config': {
                'hidden_layers': random.choice(model_config_space['acceleration_net_config']['hidden_layers']),
                'activation': random.choice(model_config_space['acceleration_net_config']['activation']),
                'dropout_rate': random.choice(model_config_space['acceleration_net_config']['dropout_rate']),
                'init_method': random.choice(model_config_space['acceleration_net_config']['init_method'])
            },
            'param_init_config': {
                'method': method,
                'values': model_config_space['param_init_config']['values'][values_idx]
            }
        }
        
        training_config = {
            'batch_size': random.choice(training_config_space['batch_size']),
            'epochs': random.choice(training_config_space['epochs']),
            'alpha': random.choice(training_config_space['alpha']),
            'learning_rate': random.choice(training_config_space['learning_rate']),
            'update_frequency': random.choice(training_config_space['update_frequency']),
            'early_stopping_patience': random.choice(training_config_space['early_stopping_patience']),
            'lr_patience': random.choice(training_config_space['lr_patience'])
        }
        
        # Create trial directory
        trial_output_dir = os.path.join(trial_dir, f"trial_{trial+1}")
        os.makedirs(trial_output_dir, exist_ok=True)
        
        # Log configurations
        with open(os.path.join(trial_output_dir, 'config.txt'), 'w') as f:
            f.write("=== Model Configuration ===\n")
            f.write(f"Unmeasured Network: {model_config['unmeasured_net_config']}\n")
            f.write(f"Acceleration Network: {model_config['acceleration_net_config']}\n")
            f.write(f"Parameter Initialization: {model_config['param_init_config']}\n\n")
            f.write("=== Training Configuration ===\n")
            for key, value in training_config.items():
                f.write(f"{key}: {value}\n")
        
        # Perform cross-validation with the sampled configuration
        cv_results = k_fold_cross_validation(
            X=X, 
            y=y, 
            k=cv_folds,
            model_config=model_config,
            training_config=training_config,
            output_dir=trial_output_dir,
            device=device,
            max_samples=max_samples
        )
        
        # Extract results
        mean_score = cv_results['mean_score']
        std_score = cv_results['std_score']
        
        # Track the best configuration
        if mean_score < best_score:
            best_score = mean_score
            best_model_config = model_config
            best_training_config = training_config
            best_trial_idx = trial
        
        # Save trial results
        trial_result = {
            'trial': trial+1,
            'model_config': model_config,
            'training_config': training_config,
            'mean_score': mean_score,
            'std_score': std_score,
            'param_means': cv_results['param_means'],
            'param_stds': cv_results['param_stds']
        }
        trial_results.append(trial_result)
        
        # Save results to file
        np.savez(os.path.join(output_dir, 'trial_results.npz'),
                trials=[t['trial'] for t in trial_results],
                mean_scores=[t['mean_score'] for t in trial_results],
                std_scores=[t['std_score'] for t in trial_results])
        
        # Plot current results
        plt.figure(figsize=(10, 6))
        plt.errorbar(
            [t['trial'] for t in trial_results],
            [t['mean_score'] for t in trial_results],
            yerr=[t['std_score'] for t in trial_results],
            marker='o', linestyle='-', capsize=5
        )
        plt.axhline(y=best_score, color='r', linestyle='--', 
                   label=f'Best score: {best_score:.6f} (Trial {best_trial_idx+1})')
        plt.xlabel('Trial')
        plt.ylabel('Cross-Validation Score (Mean ± Std)')
        plt.title('Random Search Progress')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'random_search_progress.png'))
        plt.close()
    
    # Final summary
    print("\n=== Random Search Complete ===")
    print(f"Best score: {best_score:.6f} (Trial {best_trial_idx+1})")
    print("\nBest model configuration:")
    print(f"Unmeasured Network: {best_model_config['unmeasured_net_config']}")
    print(f"Acceleration Network: {best_model_config['acceleration_net_config']}")
    print(f"Parameter Initialization: {best_model_config['param_init_config']}\n")
    print("Best training configuration:")
    for key, value in best_training_config.items():
        print(f"{key}: {value}")
    
    # Create detailed summary file
    with open(os.path.join(output_dir, 'search_summary.txt'), 'w') as f:
        f.write("=== Random Search Summary ===\n\n")
        f.write(f"Number of trials: {n_trials}\n")
        f.write(f"Cross-validation folds: {cv_folds}\n")
        f.write(f"Best score: {best_score:.6f} (Trial {best_trial_idx+1})\n\n")
        
        f.write("Best model configuration:\n")
        f.write(f"Unmeasured Network: {best_model_config['unmeasured_net_config']}\n")
        f.write(f"Acceleration Network: {best_model_config['acceleration_net_config']}\n")
        f.write(f"Parameter Initialization: {best_model_config['param_init_config']}\n\n")
        
        f.write("Best training configuration:\n")
        for key, value in best_training_config.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\n=== All Trials ===\n\n")
        for i, result in enumerate(trial_results):
            f.write(f"Trial {i+1}:\n")
            f.write(f"  Score: {result['mean_score']:.6f} ± {result['std_score']:.6f}\n")
            f.write(f"  Unmeasured Network: {result['model_config']['unmeasured_net_config']}\n")
            f.write(f"  Acceleration Network: {result['model_config']['acceleration_net_config']}\n")
            f.write(f"  Parameter Initialization: {result['model_config']['param_init_config']}\n")
            f.write(f"  Training Config: {result['training_config']}\n\n")
    
    # Save best configurations
    np.savez(os.path.join(output_dir, 'best_config.npz'),
            best_trial=best_trial_idx+1,
            best_score=best_score,
            best_model_config=best_model_config,
            best_training_config=best_training_config)
    
    # Return search results
    search_results = {
        'best_model_config': best_model_config,
        'best_training_config': best_training_config,
        'best_score': best_score,
        'best_trial_idx': best_trial_idx,
        'all_trials': trial_results
    }
    
    return search_results


def train_best_model(X, y, best_model_config, best_training_config, output_dir=None, 
                    device='cpu'):
    """
    Train the best model from the random search on the full dataset.
    
    Parameters:
    - X: Input tensor
    - y: Target tensor
    - best_model_config: Best model configuration from random search
    - best_training_config: Best training configuration from random search
    - output_dir: Directory to save the outputs
    - device: Device to use for training ('cpu' or 'cuda')
    
    Returns:
    - Trained model
    - Training history
    """
    # Setup output directory
    if output_dir is None:
        timestamp = int(time.time())
        output_dir = f'results/direct_pinn_best_{timestamp}'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure we have 2D tensors (samples x features)
    if X.dim() > 2:
        X = X.reshape(-1, X.size(-1))
    if y.dim() > 2:
        y = y.reshape(-1, y.size(-1))

    # Split data into train, validation, and test
    train_size = 0.7
    val_size = 0.15
    test_size = 0.15
    
    dataset = TensorDataset(X, y)
    train_samples = int(train_size * len(dataset))
    val_samples = int(val_size * len(dataset))
    test_samples = len(dataset) - train_samples - val_samples
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_samples, val_samples, test_samples]
    )
    
    # Compute normalization parameters from training data
    X_train = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
    y_train = torch.stack([train_dataset[i][1] for i in range(len(train_dataset))])
    
    X_norm, X_min, X_max = normalize_data(X_train, method='min_max')
    y_norm, y_min, y_max = normalize_data(y_train, method='min_max')
    
    # Move normalization parameters to device
    X_min = X_min.to(device)
    X_max = X_max.to(device)
    y_min = y_min.to(device)
    y_max = y_max.to(device)
    
    # Create normalized datasets
    train_norm = TensorDataset(X_norm, y_norm)
    
    # Extract and normalize validation data
    X_val = torch.stack([val_dataset[i][0] for i in range(len(val_dataset))])
    y_val = torch.stack([val_dataset[i][1] for i in range(len(val_dataset))])
    X_val_norm = normalize_data(X_val, min_values=X_min.cpu(), max_values=X_max.cpu(), method='min_max')
    y_val_norm = normalize_data(y_val, min_values=y_min.cpu(), max_values=y_max.cpu(), method='min_max')
    val_norm = TensorDataset(X_val_norm, y_val_norm)
    
    # Extract and normalize test data
    X_test = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
    y_test = torch.stack([test_dataset[i][1] for i in range(len(test_dataset))])
    X_test_norm = normalize_data(X_test, min_values=X_min.cpu(), max_values=X_max.cpu(), method='min_max')
    y_test_norm = normalize_data(y_test, min_values=y_min.cpu(), max_values=y_max.cpu(), method='min_max')
    test_norm = TensorDataset(X_test_norm, y_test_norm)
    
    # Create data loaders
    train_loader = DataLoader(train_norm, batch_size=best_training_config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_norm, batch_size=best_training_config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_norm, batch_size=best_training_config['batch_size'], shuffle=False)
    
    # Initialize model with best configuration
    model = ConfigurablePINN(
        unmeasured_net_config=best_model_config['unmeasured_net_config'],
        acceleration_net_config=best_model_config['acceleration_net_config'],
        param_init_config=best_model_config['param_init_config']
    ).to(device)
    
    # Create data dictionary for trainer
    data_dict = {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'X_min': X_min,
        'X_max': X_max,
        'y_min': y_min,
        'y_max': y_max,
        'normalized': True
    }
    
    # Train the model
    model_save_path = os.path.join(output_dir, "best_model.pth")
    model, history = train_model_adaptive(
        model=model,
        data_dict=data_dict,
        device=device,
        custom_loss_fn=adaptive_custom_loss,
        epochs=best_training_config['epochs'],
        alpha=best_training_config['alpha'],
        learning_rate=best_training_config['learning_rate'],
        update_frequency=best_training_config['update_frequency'],
        model_save_path=model_save_path,
        early_stopping_patience=best_training_config.get('early_stopping_patience', 100),
        lr_patience=best_training_config.get('lr_patience', 50)
    )
    
    # Save training history
    np.savez(os.path.join(output_dir, 'training_history.npz'), 
            train_loss=np.array(history['train_loss']),
            val_loss=np.array(history['val_loss']),
            data_weight=np.array(history['data_weight']),
            phys_weight=np.array(history['phys_weight']),
            reg_weight=np.array(history['reg_weight']),
            data_loss=np.array(history['data_loss']),
            phys_loss=np.array(history['phys_loss']),
            reg_loss=np.array(history['reg_loss']))
    
    # Save normalization parameters
    torch.save(X_min, os.path.join(output_dir, 'X_min.pth'))
    torch.save(X_max, os.path.join(output_dir, 'X_max.pth'))
    torch.save(y_min, os.path.join(output_dir, 'y_min.pth'))
    torch.save(y_max, os.path.join(output_dir, 'y_max.pth'))
    
    # Save configuration
    with open(os.path.join(output_dir, 'final_config.txt'), 'w') as f:
        f.write("=== Model Configuration ===\n")
        f.write(f"Unmeasured Network: {best_model_config['unmeasured_net_config']}\n")
        f.write(f"Acceleration Network: {best_model_config['acceleration_net_config']}\n")
        f.write(f"Parameter Initialization: {best_model_config['param_init_config']}\n\n")
        f.write("=== Training Configuration ===\n")
        for key, value in best_training_config.items():
            f.write(f"{key}: {value}\n")
    
    # Plot training history
    plt.figure(figsize=(15, 10))
    
    # Plot loss curves
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot adaptive weights
    plt.subplot(2, 2, 2)
    plt.plot(history['data_weight'], label='Data Weight')
    plt.plot(history['phys_weight'], label='Physics Weight')
    plt.plot(history['reg_weight'], label='Regularization Weight')
    plt.xlabel('Epoch')
    plt.ylabel('Weight')
    plt.title('Adaptive Weights')
    plt.legend()
    plt.grid(True)
    
    # Plot loss components
    plt.subplot(2, 2, 3)
    plt.plot(history['data_loss'], label='Data Loss')
    plt.plot(history['phys_loss'], label='Physics Loss')
    plt.plot(history['reg_loss'], label='Regularization Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Component')
    plt.title('Loss Components')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    # Get model parameters
    param_names = ['M1', 'M2', 'M3', 'D1', 'D2', 'D3', 'K1', 'K2', 'E1']
    param_values = []
    
    for param_name in param_names:
        param_value = getattr(model, param_name).item()
        param_values.append(param_value)
    
    # Plot final parameters
    plt.subplot(2, 2, 4)
    plt.bar(param_names, param_values)
    plt.ylabel('Parameter Value')
    plt.title('Final Model Parameters')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_summary.png'))
    plt.close()
    
    print("\nBest model training complete.")
    print(f"Model and training artifacts saved to {output_dir}")
    
    return model, history, data_dict 