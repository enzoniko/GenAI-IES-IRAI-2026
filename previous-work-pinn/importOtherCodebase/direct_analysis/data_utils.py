"""
Data Utilities for Direct PINN Analysis

This module contains functions for loading and preprocessing data
for direct PINN models.
"""

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, random_split

def normalize_data(data, min_values=None, max_values=None, method='min_max', mean_values=None, std_values=None):
    """
    Normalize data using the specified scaling method.
    
    Parameters:
    - data: Tensor to normalize
    - min_values: Optional tensor of minimum values for each feature (for min-max scaling)
    - max_values: Optional tensor of maximum values for each feature (for min-max scaling)
    - method: Scaling method to use ('min_max' or 'standard')
    - mean_values: Optional tensor of mean values for each feature (for standard scaling)
    - std_values: Optional tensor of standard deviation values for each feature (for standard scaling)
    
    Returns:
    - Normalized tensor
    - min_values/mean_values (if not provided)
    - max_values/std_values (if not provided)
    """
    if method == 'min_max':
        if min_values is None or max_values is None:
            min_values = torch.min(data, dim=0)[0]
            max_values = torch.max(data, dim=0)[0]
            return (data - min_values) / (max_values - min_values), min_values, max_values
        else:
            return (data - min_values) / (max_values - min_values)
    elif method == 'standard':
        if mean_values is None or std_values is None:
            mean_values = torch.mean(data, dim=0)
            std_values = torch.std(data, dim=0)
            return (data - mean_values) / std_values, mean_values, std_values
        else:
            return (data - mean_values) / std_values
    else:
        raise ValueError(f"Unknown scaling method: {method}. Use 'min_max' or 'standard'.")

def prepare_data(X, y, batch_size=1000, normalize=True, test_size=0.1, val_size=0.1, max_samples=None, scaling_method='min_max'):
    """
    Prepare data for direct PINN training.
    
    Parameters:
    - X: Input tensor
    - y: Target tensor
    - batch_size: Batch size for data loaders
    - normalize: Whether to normalize the data
    - test_size: Proportion of data to use for testing
    - val_size: Proportion of data to use for validation
    - max_samples: Maximum number of samples to use (for faster training)
    - scaling_method: Method to use for scaling ('min_max' or 'standard')
    
    Returns:
    - Dictionary containing data loaders, normalization parameters, etc.
    """
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
    
    # Split data into train, validation, and test sets
    dataset = TensorDataset(X, y)
    train_size = 1.0 - test_size - val_size
    train_samples = int(train_size * len(dataset))
    val_samples = int(val_size * len(dataset))
    test_samples = len(dataset) - train_samples - val_samples
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_samples, val_samples, test_samples]
    )
    
    # Extract training data for normalization
    X_train = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
    y_train = torch.stack([train_dataset[i][1] for i in range(len(train_dataset))])
    
    # Normalize data if requested
    if normalize:
        # Calculate normalization parameters from training data
        if scaling_method == 'min_max':
            X_norm, X_min, X_max = normalize_data(X_train, method=scaling_method)
            y_norm, y_min, y_max = normalize_data(y_train, method=scaling_method)
            
            # Create normalized datasets
            train_norm = TensorDataset(X_norm, y_norm)
            
            # Extract and normalize validation data
            X_val = torch.stack([val_dataset[i][0] for i in range(len(val_dataset))])
            y_val = torch.stack([val_dataset[i][1] for i in range(len(val_dataset))])
            X_val_norm = normalize_data(X_val, min_values=X_min, max_values=X_max, method=scaling_method)
            y_val_norm = normalize_data(y_val, min_values=y_min, max_values=y_max, method=scaling_method)
            val_norm = TensorDataset(X_val_norm, y_val_norm)
            
            # Extract and normalize test data
            X_test = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
            y_test = torch.stack([test_dataset[i][1] for i in range(len(test_dataset))])
            X_test_norm = normalize_data(X_test, min_values=X_min, max_values=X_max, method=scaling_method)
            y_test_norm = normalize_data(y_test, min_values=y_min, max_values=y_max, method=scaling_method)
            test_norm = TensorDataset(X_test_norm, y_test_norm)
            
            # Create data loaders
            train_loader = DataLoader(train_norm, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_norm, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_norm, batch_size=batch_size, shuffle=False)
            
            return {
                'train_loader': train_loader,
                'val_loader': val_loader,
                'test_loader': test_loader,
                'X_min': X_min,
                'X_max': X_max,
                'y_min': y_min,
                'y_max': y_max,
                'normalized': True,
                'scaling_method': scaling_method
            }
        elif scaling_method == 'standard':
            X_norm, X_mean, X_std = normalize_data(X_train, method=scaling_method)
            y_norm, y_mean, y_std = normalize_data(y_train, method=scaling_method)
            
            # Create normalized datasets
            train_norm = TensorDataset(X_norm, y_norm)
            
            # Extract and normalize validation data
            X_val = torch.stack([val_dataset[i][0] for i in range(len(val_dataset))])
            y_val = torch.stack([val_dataset[i][1] for i in range(len(val_dataset))])
            X_val_norm = normalize_data(X_val, mean_values=X_mean, std_values=X_std, method=scaling_method)
            y_val_norm = normalize_data(y_val, mean_values=y_mean, std_values=y_std, method=scaling_method)
            val_norm = TensorDataset(X_val_norm, y_val_norm)
            
            # Extract and normalize test data
            X_test = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
            y_test = torch.stack([test_dataset[i][1] for i in range(len(test_dataset))])
            X_test_norm = normalize_data(X_test, mean_values=X_mean, std_values=X_std, method=scaling_method)
            y_test_norm = normalize_data(y_test, mean_values=y_mean, std_values=y_std, method=scaling_method)
            test_norm = TensorDataset(X_test_norm, y_test_norm)
            
            # Create data loaders
            train_loader = DataLoader(train_norm, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_norm, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_norm, batch_size=batch_size, shuffle=False)
            
            return {
                'train_loader': train_loader,
                'val_loader': val_loader,
                'test_loader': test_loader,
                'X_mean': X_mean,
                'X_std': X_std,
                'y_mean': y_mean,
                'y_std': y_std,
                'normalized': True,
                'scaling_method': scaling_method
            }
    else:
        # Create non-normalized data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'normalized': False,
            'scaling_method': None
        }

def prepare_dataset_for_residuals(X, y, device, batch_size=1024, X_min=None, X_max=None, y_min=None, y_max=None):
    """
    Prepare dataset for residual computation.
    
    Parameters:
    - X: Input tensor
    - y: Target tensor
    - device: Device to use for computation
    - batch_size: Batch size for data loaders
    - X_min, X_max, y_min, y_max: Normalization parameters
    
    Returns:
    - Data loader for the dataset
    """
    # Ensure we have 2D tensors (samples x features)
    if X.dim() > 2:
        X = X.reshape(-1, X.size(-1))
    if y.dim() > 2:
        y = y.reshape(-1, y.size(-1))
    
    # Normalize data if normalization parameters are provided
    if X_min is not None and X_max is not None:
        X = normalize_data(X, X_min.cpu(), X_max.cpu())
    
    if y_min is not None and y_max is not None:
        y = normalize_data(y, y_min.cpu(), y_max.cpu())
    
    # Create dataset and data loader
    dataset = TensorDataset(X, y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    return data_loader 