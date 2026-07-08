"""
Adaptive Loss Functions for Direct PINN Analysis

This module provides custom loss functions that are compatible with the
adaptive weight training approach.
"""

import torch
import torch.nn as nn

def adaptive_custom_loss(model, x, y_true, max_x_values=None, min_x_values=None, max_y_values=None, min_y_values=None):
    """
    Custom loss function that returns separate loss components for adaptive weighting.
    
    This function is based on the adaptive approach from basicPINNv5.py with individual
    Lagrangian multipliers for each residual component.
    
    Parameters:
    - model: The PINN model being trained
    - x: Input data (normalized)
    - y_true: Target outputs (normalized)
    - max_x_values: Maximum values of input features (for normalization reference)
    - min_x_values: Minimum values of input features (for normalization reference)
    - max_y_values: Maximum values of output features (for normalization reference)
    - min_y_values: Minimum values of output features (for normalization reference)
    
    Returns:
    - Tuple of individual loss components for adaptive weighting
    """
    try:
        # Get model predictions - expecting normalized inputs
        y_pred = model(x)
        
        # Compute the data loss (using RMSE)
        data_loss = torch.sqrt(nn.MSELoss()(y_pred, y_true) + 1e-8)
        
        # Compute physics-based residuals directly using normalized values
        residual1, residual2, residual3, residual4, residualMass1, residualMass2 = model.compute_residuals(x, y_pred)
        
        # Handle NaNs and clip extreme values
        residual1 = torch.nan_to_num(residual1, nan=0.0, posinf=1e10, neginf=-1e10)
        residual2 = torch.nan_to_num(residual2, nan=0.0, posinf=1e10, neginf=-1e10)
        residual3 = torch.nan_to_num(residual3, nan=0.0, posinf=1e10, neginf=-1e10)
        residual4 = torch.nan_to_num(residual4, nan=0.0, posinf=1e10, neginf=-1e10)
        residualMass1 = torch.nan_to_num(residualMass1, nan=0.0, posinf=1e10, neginf=-1e10)
        residualMass2 = torch.nan_to_num(residualMass2, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Clip residuals to avoid extreme values
        residual1 = torch.clamp(residual1, -1e8, 1e8)
        residual2 = torch.clamp(residual2, -1e8, 1e8)
        residual3 = torch.clamp(residual3, -1e8, 1e8)
        residual4 = torch.clamp(residual4, -1e8, 1e8)
        residualMass1 = torch.clamp(residualMass1, -1e8, 1e8)
        residualMass2 = torch.clamp(residualMass2, -1e8, 1e8)
        
        # Compute individual RMSE losses for each residual
        res1_loss = torch.sqrt(torch.mean(residual1**2) + 1e-8)
        res2_loss = torch.sqrt(torch.mean(residual2**2) + 1e-8)
        res3_loss = torch.sqrt(torch.mean(residual3**2) + 1e-8)
        res4_loss = torch.sqrt(torch.mean(residual4**2) + 1e-8)
        resMass1_loss = torch.sqrt(torch.mean(residualMass1**2) + 1e-8)
        resMass2_loss = torch.sqrt(torch.mean(residualMass2**2) + 1e-8)
        
        # Check for NaN or Inf values
        for loss_name, loss_val in [
            ('data_loss', data_loss), 
            ('res1_loss', res1_loss), 
            ('res2_loss', res2_loss),
            ('res3_loss', res3_loss), 
            ('res4_loss', res4_loss),
            ('resMass1_loss', resMass1_loss),
            ('resMass2_loss', resMass2_loss)
        ]:
            if torch.isnan(loss_val) or torch.isinf(loss_val):
                print(f"Warning: {loss_name} is {loss_val}")
                # Replace with a high but finite value
                if loss_name == 'data_loss':
                    data_loss = torch.tensor(1.0, device=x.device)
                elif loss_name == 'res1_loss':
                    res1_loss = torch.tensor(1.0, device=x.device)
                elif loss_name == 'res2_loss':
                    res2_loss = torch.tensor(1.0, device=x.device)
                elif loss_name == 'res3_loss':
                    res3_loss = torch.tensor(1.0, device=x.device)
                elif loss_name == 'res4_loss':
                    res4_loss = torch.tensor(1.0, device=x.device)
                elif loss_name == 'resMass1_loss':
                    resMass1_loss = torch.tensor(1.0, device=x.device)
                elif loss_name == 'resMass2_loss':
                    resMass2_loss = torch.tensor(1.0, device=x.device)
                    
        return data_loss, res1_loss, res2_loss, res3_loss, res4_loss, resMass1_loss, resMass2_loss
    
    except Exception as e:
        # Fallback values in case of error
        print(f"Error in adaptive_custom_loss: {e}")
        device = x.device
        return (
            torch.tensor(1.0, device=device),
            torch.tensor(1.0, device=device),
            torch.tensor(1.0, device=device),
            torch.tensor(1.0, device=device),
            torch.tensor(1.0, device=device),
            torch.tensor(1.0, device=device),
            torch.tensor(1.0, device=device)
        ) 