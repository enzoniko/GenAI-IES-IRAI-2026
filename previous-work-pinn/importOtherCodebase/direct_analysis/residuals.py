"""
Residuals Computation Module for Direct PINN Analysis

This module contains functions for computing residuals from
direct PINN models in the standardized format.
"""

import torch
import numpy as np
from tqdm import tqdm
from dtw import dtw
import sys
import os

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utils.residuals_utils import get_rotation_segments, compute_dtw_distances, standardize_residuals, print_residuals_structure

def compute_residuals(model, data_dict, device, include_physical_residuals=True, compute_dtw=False):
    """
    Compute residuals for direct PINN models in legacy format
    
    Parameters:
    - model: Trained PINN model
    - data_dict: Dictionary containing data loaders
    - device: Device to use for computation
    - include_physical_residuals: Whether to include physical residuals (r1, r2, r3, r4)
    - compute_dtw: Parameter kept for compatibility but not used
    
    Returns:
    - Dictionary of legacy format residuals {data_type: tensor, ...}
    """
    model.eval()
    legacy_residuals = {}
    
    # Process each data loader in the data_dict
    for data_type, data_loader in tqdm(list(data_dict.items()), desc="Computing residuals"):
        print(f"Processing {data_type} data")
        
        # Collect all residuals for this data type
        all_data_residuals = []
        all_phys_r1, all_phys_r2, all_phys_r3, all_phys_r4 = [], [], [], []
        
        for X_batch, y_batch in tqdm(data_loader, desc=f"Processing batches", leave=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            with torch.no_grad():
                # Get model predictions
                y_pred = model(X_batch)
                
                # Compute prediction errors (data residuals)
                batch_residuals = y_pred - y_batch
                
                # Compute physical residuals if requested
                if include_physical_residuals:
                    r1, r2, r3, r4, _, _ = model.compute_residuals(X_batch, y_pred)
                    
                    all_phys_r1.append(r1.cpu())
                    all_phys_r2.append(r2.cpu())
                    all_phys_r3.append(r3.cpu())
                    all_phys_r4.append(r4.cpu())
                
                # Store data residuals
                all_data_residuals.append(batch_residuals.cpu())
        
        # Combine all batches
        combined_data_residuals = torch.cat(all_data_residuals, dim=0)
        
        # Process physical residuals if available
        if include_physical_residuals:
            # Combine physical residuals
            phys_r1 = torch.cat(all_phys_r1, dim=0)
            phys_r2 = torch.cat(all_phys_r2, dim=0)
            phys_r3 = torch.cat(all_phys_r3, dim=0)
            phys_r4 = torch.cat(all_phys_r4, dim=0)
            
            # Ensure all residuals are properly shaped
            if phys_r1.dim() > 1 and phys_r1.size(1) == 1:
                phys_r1 = phys_r1.squeeze(1)
            if phys_r2.dim() > 1 and phys_r2.size(1) == 1:
                phys_r2 = phys_r2.squeeze(1)
            if phys_r3.dim() > 1 and phys_r3.size(1) == 1:
                phys_r3 = phys_r3.squeeze(1)
            if phys_r4.dim() > 1 and phys_r4.size(1) == 1:
                phys_r4 = phys_r4.squeeze(1)
            
            # Stack physical residuals into one tensor [samples, 4]
            physical_residuals = torch.stack([phys_r1, phys_r2, phys_r3, phys_r4], dim=1)
            
            # Check if there's an extra dimension and remove it if needed
            if physical_residuals.dim() > 2 and physical_residuals.size(2) == 1:
                physical_residuals = physical_residuals.squeeze(-1)
            
            # Combine data and physical residuals for legacy format
            combined_residuals = torch.cat([combined_data_residuals, physical_residuals], dim=1)
        else:
            combined_residuals = combined_data_residuals
        
        # Store in legacy format
        legacy_residuals[data_type] = combined_residuals
    
    return legacy_residuals 