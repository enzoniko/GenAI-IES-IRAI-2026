"""ReLoBRaLo (Relative Loss Balancing with Random Lookback) — pure logic,
no config/module coupling. Ported from the validated prior-work code."""
import random

import torch
import torch.nn as nn


class ReLoBRaLoLoss(nn.Module):
    def __init__(self, num_terms: int, alpha: float = 0.5125, rho: float = 0.2332,
                 temperature: float = 1.4198):
        super().__init__()
        self.num_terms = num_terms
        self.alpha, self.rho, self.temperature = alpha, rho, temperature
        self.running_losses: torch.Tensor | None = None
        self.running_weights: torch.Tensor | None = None

    def weights(self, loss_components: list[torch.Tensor]) -> torch.Tensor:
        device = loss_components[0].device
        if self.running_losses is None:
            self.running_losses = torch.ones(self.num_terms, device=device) * 1e4
            self.running_weights = torch.ones(self.num_terms, device=device)
        current = torch.stack([l.detach() for l in loss_components])
        current = torch.nan_to_num(current, nan=1e4, posinf=1e4, neginf=1e4)
        self.running_losses = self.alpha * self.running_losses + (1 - self.alpha) * current
        lookback = self.running_losses
        if random.random() < self.rho:
            lookback = lookback + torch.randn_like(lookback) * 0.1
        ratios = lookback / (lookback.mean() + 1e-8)
        w = torch.softmax(-ratios / self.temperature, dim=0)
        w = w / w.sum() * self.num_terms
        self.running_weights = self.alpha * self.running_weights + (1 - self.alpha) * w
        return self.running_weights

    def combine(self, loss_components: list[torch.Tensor]) -> torch.Tensor:
        w = self.weights(loss_components)
        return torch.stack([w[i] * l for i, l in enumerate(loss_components)]).sum()
