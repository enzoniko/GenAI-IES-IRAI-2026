"""MINE mutual-information estimator (Belghazi et al. 2018) for A1."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class _TNet(nn.Module):
    def __init__(self, dx, dy, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dx + dy, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=1)).squeeze(-1)


def estimate_mi(X: torch.Tensor, Y: torch.Tensor, epochs: int = 300,
                batch_size: int = 256, lr: float = 5e-4, hidden: int = 128,
                seed: int = 0, device: str | torch.device = "cpu") -> float:
    """Donsker-Varadhan MI lower bound in nats, with EMA-corrected gradient.
    X (N, dx), Y (N, dy); rows are paired samples."""
    torch.manual_seed(seed)
    dev = torch.device(device)
    Xs = ((X - X.mean(0)) / (X.std(0) + 1e-8)).float().to(dev)
    Ys = ((Y - Y.mean(0)) / (Y.std(0) + 1e-8)).float().to(dev)
    N = Xs.shape[0]
    net = _TNet(Xs.shape[1], Ys.shape[1], hidden).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    ema = None
    estimates = []
    for ep in range(epochs):
        idx = torch.randperm(N, device=dev)[: min(batch_size, N)]
        x, y = Xs[idx], Ys[idx]
        y_shuf = y[torch.randperm(y.shape[0], device=dev)]
        t_joint = net(x, y).mean()
        t_marg = net(x, y_shuf)
        # EMA-corrected DV to reduce gradient bias
        exp_marg = torch.exp(t_marg).mean()
        ema = exp_marg.detach() if ema is None else 0.99 * ema + 0.01 * exp_marg.detach()
        loss = -(t_joint - exp_marg / ema.clamp(min=1e-8) * 1.0)  # biased-corrected surrogate
        mi = (t_joint - torch.log(exp_marg.clamp(min=1e-8))).item()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if ep >= epochs - 50:
            estimates.append(mi)
    return float(np.mean(estimates))
