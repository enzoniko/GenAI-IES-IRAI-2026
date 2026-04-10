import torch
import torch.nn as nn
import random
from .pinn import ConfigurablePINN # Note: adaptive_custom_loss moved here

class ReLoBRaLoLoss(nn.Module):
    """
    ReLoBRaLo (Relative Loss Balancing with Random Lookback) loss balancing method.
    Extracted pure logic from the previous work.
    """
    
    def __init__(self, alpha: float = 1.0, rho: float = 0.1, temperature: float = 1.0, enable_mass_constraints: bool = True):
        super().__init__()
        self.alpha = alpha  
        self.rho = rho      
        self.temperature = temperature  
        
        self.running_losses = None
        self.running_weights = None
        
        self.loss_history = []
        self._initialized = False
        
        if enable_mass_constraints:
            self.loss_keys = ['data', 'phys_res1', 'phys_res2', 'phys_res3', 'phys_res4', 'phys_mass1', 'phys_mass2']
        else:
            self.loss_keys = ['data', 'phys_res1', 'phys_res2', 'phys_res3', 'phys_res4']
        
        self.current_weights = {key: 1.0 for key in self.loss_keys}
    
    def forward(self, model, X_batch, y_batch, X_max, X_min, y_max, y_min):
        loss_components = adaptive_custom_loss(model, X_batch, y_batch, X_max, X_min, y_max, y_min)
        return list(loss_components)
    
    def _reset_statistics(self):
        if self.running_losses is not None:
            self.running_losses = torch.ones_like(self.running_losses) * 1e4
            self.running_weights = torch.ones_like(self.running_weights)
    
    def _initialize_statistics(self, loss_components):
        if self.running_losses is None:
            device = loss_components[0].device
            self.running_losses = torch.ones(len(loss_components), device=device) * 1e4
            self.running_weights = torch.ones(len(loss_components), device=device)
            
            if not torch.isfinite(self.running_losses).all():
                self.running_losses = torch.ones(len(loss_components), device=device) * 1e4
            if not torch.isfinite(self.running_weights).all():
                self.running_weights = torch.ones(len(loss_components), device=device)
    
    def _update_weights(self, loss_components):
        self._initialize_statistics(loss_components)
        current_losses = torch.stack([loss.detach() for loss in loss_components])
        
        if not torch.isfinite(current_losses).all():
            current_losses = torch.nan_to_num(current_losses, nan=1e4, posinf=1e4, neginf=1e4)
        
        self.running_losses = self.alpha * self.running_losses + (1 - self.alpha) * current_losses
        
        if not torch.isfinite(self.running_losses).all():
            self._reset_statistics()
        
        if random.random() < self.rho:
            noise = torch.randn_like(self.running_losses) * 0.1
            lookback_losses = self.running_losses + noise
        else:
            lookback_losses = self.running_losses
        
        if not torch.isfinite(lookback_losses).all():
            lookback_losses = torch.ones_like(lookback_losses) * 1e4
        
        mean_loss = torch.mean(lookback_losses)
        if not torch.isfinite(mean_loss):
            mean_loss = torch.tensor(1e4, device=lookback_losses.device)
        
        relative_ratios = lookback_losses / (mean_loss + 1e-8)
        
        if not torch.isfinite(relative_ratios).all():
            relative_ratios = torch.ones_like(relative_ratios)
        
        logits = -relative_ratios / self.temperature
        
        if not torch.isfinite(logits).all():
            logits = torch.zeros_like(logits)
        
        weights = torch.softmax(logits, dim=0)
        
        if not torch.isfinite(weights).all():
            weights = torch.ones_like(weights) / len(weights)
        
        weights = weights / torch.sum(weights) * len(weights)
        
        if not torch.isfinite(weights).all():
            weights = torch.ones_like(weights)
        
        self.running_weights = self.alpha * self.running_weights + (1 - self.alpha) * weights
        
        if not torch.isfinite(self.running_weights).all():
            self._reset_statistics()
        
        return self.running_weights
    
    def step(self, loss_components, optimizer, model, X_batch, y_batch):
        for i, loss in enumerate(loss_components):
            if not torch.isfinite(loss):
                loss_components[i] = torch.tensor(1e4, device=loss.device, requires_grad=True)
        
        weights = self._update_weights(loss_components)
        
        weighted_losses = []
        for i, loss in enumerate(loss_components):
            weighted_loss = weights[i] * loss
            weighted_losses.append(weighted_loss)
        
        total_loss = torch.stack(weighted_losses).sum()
        
        if not torch.isfinite(total_loss):
            return total_loss
        
        # Zero gradients from previous step
        optimizer.zero_grad()

        # Calculates how to change weights (backpropagation)
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        for i, key in enumerate(self.loss_keys):
            if i < len(weights):
                self.current_weights[key] = weights[i].item()
            else:
                break
        
        return total_loss

def adaptive_custom_loss(model, x, y_true, X_max=None, X_min=None, y_max=None, y_min=None, debug=False):
    """
    Custom loss function that returns individual loss components for adaptive weighting.
    Computes physics residuals using denormalized values for proper physics scaling.
    
    Parameters:
    - model: The PINN model being trained
    - x: Input data tensor (already normalized)
    - y_true: Target output tensor (already normalized)
    - X_max, X_min: Input normalization parameters (used for denormalization in physics)
    - y_max, y_min: Output normalization parameters (used for denormalization in physics)
    - debug: Whether to print debug information
    
    Returns:
    - Tuple of individual loss components
    """
    try:
        # Ensure double precision
        x = x.double()
        y_true = y_true.double()
        
        # Get model predictions (model expects normalized inputs)
        y_pred = model(x)
        
        # Compute the data loss using RMSE on normalized values
        data_loss = torch.sqrt(torch.mean((y_pred - y_true)**2) + 1e-12)
        
        # Compute physics-based residuals using denormalized values for proper physics
        # Pass normalization parameters for denormalization in compute_residuals
        residuals = model.compute_residuals(x, y_pred, X_max, X_min, y_max, y_min)
        
        # Extract and handle individual residuals
        residual1, residual2, residual3, residual4, residualMass1, residualMass2 = residuals
        
        # Handle NaNs only (no clipping)
        residual1 = torch.nan_to_num(residual1, nan=0.0)
        residual2 = torch.nan_to_num(residual2, nan=0.0)
        residual3 = torch.nan_to_num(residual3, nan=0.0)
        residual4 = torch.nan_to_num(residual4, nan=0.0)
        residualMass1 = torch.nan_to_num(residualMass1, nan=0.0)
        residualMass2 = torch.nan_to_num(residualMass2, nan=0.0)
        
        # Compute individual RMSE losses for each residual
        res1_loss = torch.sqrt(torch.mean(residual1**2) + 1e-12)
        res2_loss = torch.sqrt(torch.mean(residual2**2) + 1e-12)
        res3_loss = torch.sqrt(torch.mean(residual3**2) + 1e-12)
        res4_loss = torch.sqrt(torch.mean(residual4**2) + 1e-12)
        
        # For synthetic data (enable_mass_constraints=False), mass residuals are zeros
        # We should not include them in the loss computation to avoid constant losses
        if model.enable_mass_constraints:
            # For real data: compute actual mass constraint losses
            resMass1_loss = torch.sqrt(torch.mean(residualMass1**2) + 1e-12)
            resMass2_loss = torch.sqrt(torch.mean(residualMass2**2) + 1e-12)
        else:
            # For synthetic data: use zeros to maintain compatibility with training scripts
            # but these won't affect the actual loss computation
            resMass1_loss = torch.tensor(0.0, device=x.device, dtype=torch.float64, requires_grad=True)
            resMass2_loss = torch.tensor(0.0, device=x.device, dtype=torch.float64, requires_grad=True)
        
        # Debug functionality has been moved to CSV logging system
        
        return data_loss, res1_loss, res2_loss, res3_loss, res4_loss, resMass1_loss, resMass2_loss
    
    except Exception as e:
        # Fallback values in case of error
        device = x.device
        print(f"Error in adaptive_custom_loss: {e}")
        return (
            torch.tensor(1.0, device=device, dtype=torch.float64, requires_grad=True),
            torch.tensor(1.0, device=device, dtype=torch.float64, requires_grad=True),
            torch.tensor(1.0, device=device, dtype=torch.float64, requires_grad=True),
            torch.tensor(1.0, device=device, dtype=torch.float64, requires_grad=True),
            torch.tensor(1.0, device=device, dtype=torch.float64, requires_grad=True),
            torch.tensor(1.0, device=device, dtype=torch.float64, requires_grad=True),
            torch.tensor(1.0, device=device, dtype=torch.float64, requires_grad=True)
        )
