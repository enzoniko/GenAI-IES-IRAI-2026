import torch
import torch.nn as nn
import random
from .pinn import adaptive_custom_loss

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
        
        optimizer.zero_grad()
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        for i, key in enumerate(self.loss_keys):
            if i < len(weights):
                self.current_weights[key] = weights[i].item()
            else:
                break
        
        return total_loss
