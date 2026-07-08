"""Latent DDPM denoiser + scheduler + proper DDIM sampler (library code —
DDIM previously existed only ad hoc inside legacy task scripts).
Optional class conditioning (FiLM on the time embedding) for experiment E2."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=time.device) * -emb)
        emb = time[:, None].float() * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class LatentDiffusionMLP(nn.Module):
    def __init__(self, z_dim=128, time_dim=64, hidden=256, num_classes: int | None = None):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 2), nn.GELU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        self.class_embed = nn.Embedding(num_classes, time_dim) if num_classes else None
        self.fc1 = nn.Linear(z_dim, hidden)
        self.fc_time1 = nn.Linear(time_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc_time2 = nn.Linear(time_dim, hidden)
        self.fc3 = nn.Linear(hidden, z_dim)
        self.act = nn.SiLU()

    def forward(self, x, time, label: torch.Tensor | None = None):
        t = self.time_mlp(time)
        if self.class_embed is not None and label is not None:
            t = t + self.class_embed(label)
        h = self.act(self.fc1(x) + self.fc_time1(t))
        h2 = self.act(self.fc2(h) + self.fc_time2(t)) + h
        return self.fc3(h2)


class DDPMScheduler:
    def __init__(self, num_train_timesteps=1000, beta_start=1e-4, beta_end=0.02,
                 device: torch.device | str = "cpu"):
        self.num_train_timesteps = num_train_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def to(self, device):
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                setattr(self, k, v.to(device))
        return self

    def add_noise(self, x0, noise, timesteps):
        a = self.sqrt_alphas_cumprod[timesteps].view(-1, 1)
        b = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1)
        return a * x0 + b * noise


class DDIMSampler:
    """Deterministic DDIM (eta=0) over a subsampled schedule, x0-prediction
    form with optional x0 clamping for stability in collapsed/low-data
    latent spaces."""

    def __init__(self, scheduler: DDPMScheduler, x0_clamp: float | None = 4.0):
        self.s = scheduler
        self.x0_clamp = x0_clamp

    def schedule(self, t_start: int, n_steps: int) -> list[int]:
        """Descending timesteps from t_start-1 to 0, n_steps values."""
        n = min(n_steps, t_start)
        ts = torch.linspace(t_start - 1, 0, n).round().long()
        return torch.unique(ts, sorted=True).flip(0).tolist()

    def predict_x0(self, z_t, eps_pred, t: int):
        x0 = (z_t - self.s.sqrt_one_minus_alphas_cumprod[t] * eps_pred) \
             / self.s.sqrt_alphas_cumprod[t]
        if self.x0_clamp is not None:
            x0 = x0.clamp(-self.x0_clamp, self.x0_clamp)
        return x0

    def step(self, z_t, eps_pred, t: int, t_prev: int):
        x0 = self.predict_x0(z_t, eps_pred, t)
        if t_prev < 0:
            return x0
        return self.s.sqrt_alphas_cumprod[t_prev] * x0 \
            + self.s.sqrt_one_minus_alphas_cumprod[t_prev] * eps_pred

    @torch.no_grad()
    def sample(self, model, n: int, z_dim: int, device, n_steps: int = 50,
               label: torch.Tensor | None = None):
        z = torch.randn(n, z_dim, device=device)
        ts = self.schedule(self.s.num_train_timesteps, n_steps)
        for i, t in enumerate(ts):
            t_prev = ts[i + 1] if i + 1 < len(ts) else -1
            eps = model(z, torch.full((n,), t, device=device, dtype=torch.long), label)
            z = self.step(z, eps, t, t_prev)
        return z
