"""
Ablation baselines for the physics-guided diffusion paper.

VanillaDDPM
    Same MLP backbone and noise schedule as LatentDiffusionMLP.
    No oracle guidance, no class conditioning.
    Parameter count intentionally matches the main LDM (excluding oracle params).

LabelConditionedDDPM
    Same backbone + a lightweight class embedding (nn.Embedding(num_classes, time_dim))
    whose output is added to the timestep embedding (FiLM-style injection).
    No oracle guidance.
    Parameter count ≈ main LDM + class_embedding params (small overhead).

Both expose:
    forward()  — noise prediction (training)
    get_loss() — DDPM MSE loss (training helper)
    sample()   — standard DDPM reverse denoising (inference, no external guidance)

Noise schedule constants mirror the main DDPMScheduler defaults exactly:
    beta_start=0.0001, beta_end=0.02, num_timesteps=1000, linear schedule.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.latent_diffusion import DDPMScheduler, SinusoidalPositionEmbeddings

# ---------------------------------------------------------------------------
# Shared noise-schedule constants — must be identical to main LDM
# ---------------------------------------------------------------------------
_BETA_START: float = 0.0001
_BETA_END: float = 0.02
_NUM_TIMESTEPS: int = 1000


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _move_scheduler_to(scheduler: DDPMScheduler, device: torch.device) -> None:
    """Move all DDPMScheduler tensor buffers to *device* (in-place, idempotent)."""
    for attr in (
        "betas",
        "alphas",
        "alphas_cumprod",
        "alphas_cumprod_prev",
        "sqrt_alphas_cumprod",
        "sqrt_one_minus_alphas_cumprod",
        "sqrt_recip_alphas",
        "posterior_variance",
    ):
        buf = getattr(scheduler, attr, None)
        if buf is not None:
            setattr(scheduler, attr, buf.to(device))
    scheduler.device = str(device)


# ---------------------------------------------------------------------------
# VanillaDDPM
# ---------------------------------------------------------------------------

class VanillaDDPM(nn.Module):
    """
    Ablation baseline: pure DDPM on z_macro latents.

    Architecture is an exact copy of LatentDiffusionMLP (same MLP layers,
    same sinusoidal time embedding, same noise schedule) so that parameter
    counts are comparable.  The oracle / VJP guidance present in the main
    SDEdit pipeline is deliberately absent.
    """

    def __init__(
        self,
        z_dim: int = 128,
        time_dim: int = 64,
        beta_start: float = _BETA_START,
        beta_end: float = _BETA_END,
        num_timesteps: int = _NUM_TIMESTEPS,
    ) -> None:
        super().__init__()

        self.z_dim = z_dim
        self.time_dim = time_dim
        self.num_timesteps = num_timesteps

        # Noise schedule — identical hyperparameters to the main LDM
        self.scheduler = DDPMScheduler(
            num_train_timesteps=num_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
        )

        # Timestep MLP — mirrors LatentDiffusionMLP exactly
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.GELU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        # MLP backbone — mirrors LatentDiffusionMLP exactly
        self.fc1 = nn.Linear(z_dim, 256)
        self.fc_time1 = nn.Linear(time_dim, 256)

        self.fc2 = nn.Linear(256, 256)
        self.fc_time2 = nn.Linear(time_dim, 256)

        self.fc3 = nn.Linear(256, z_dim)

        self.act = nn.SiLU()

    # ------------------------------------------------------------------
    def forward(self, z_macro: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict the noise residual for the given noisy latent and timestep.

        Args:
            z_macro: (B, z_dim)  noisy latent vectors
            t:       (B,)        integer diffusion timesteps

        Returns:
            predicted_noise: (B, z_dim)
        """
        te = self.time_mlp(t)               # (B, time_dim)

        h = self.fc1(z_macro) + self.fc_time1(te)
        h = self.act(h)

        h2 = self.fc2(h) + self.fc_time2(te)
        h2 = self.act(h2) + h               # residual skip

        return self.fc3(h2)

    # ------------------------------------------------------------------
    def get_loss(self, z_macro: torch.Tensor) -> torch.Tensor:
        """Compute DDPM MSE loss (noise prediction) over a batch of clean latents.

        Args:
            z_macro: (B, z_dim)  clean latent vectors

        Returns:
            loss: scalar tensor
        """
        device = z_macro.device
        B = z_macro.shape[0]

        _move_scheduler_to(self.scheduler, device)

        noise = torch.randn_like(z_macro)
        t = torch.randint(0, self.num_timesteps, (B,), device=device).long()

        noisy = self.scheduler.add_noise(z_macro, noise, t)
        noise_pred = self.forward(noisy, t)

        return F.mse_loss(noise_pred, noise)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        device: str,
        z_macro_init: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate z_macro samples via standard DDPM reverse denoising.

        No oracle guidance is applied — this is the pure diffusion baseline.

        Args:
            n_samples:     number of samples to generate
            device:        torch device string (e.g. 'cpu', 'cuda')
            z_macro_init:  optional (n_samples, z_dim) starting latent;
                           if None, starts from pure Gaussian noise

        Returns:
            z: (n_samples, z_dim) generated latent vectors
        """
        dev = torch.device(device)
        _move_scheduler_to(self.scheduler, dev)

        z = (
            z_macro_init.to(dev)
            if z_macro_init is not None
            else torch.randn(n_samples, self.z_dim, device=dev)
        )

        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((n_samples,), t, device=dev, dtype=torch.long)
            noise_pred = self.forward(z, t_batch)
            z = self.scheduler.step(noise_pred, t, z)

        return z


# ---------------------------------------------------------------------------
# LabelConditionedDDPM
# ---------------------------------------------------------------------------

class LabelConditionedDDPM(nn.Module):
    """
    Ablation baseline: DDPM + fault-class label conditioning, no physics oracle.

    The class index is embedded via nn.Embedding(num_classes, time_dim) and
    **added** to the sinusoidal timestep embedding before the residual MLP blocks.
    This is a minimal FiLM-style injection that adds only
    ``num_classes × time_dim`` parameters over VanillaDDPM.

    Supported classes (NUM_CLASSES=4):
        0 — Normal
        1 — Imbalance
        2 — VerticalMisalignment
        3 — OverhangBB
    """

    def __init__(
        self,
        z_dim: int = 128,
        time_dim: int = 64,
        num_classes: int = 4,
        beta_start: float = _BETA_START,
        beta_end: float = _BETA_END,
        num_timesteps: int = _NUM_TIMESTEPS,
    ) -> None:
        super().__init__()

        self.z_dim = z_dim
        self.time_dim = time_dim
        self.num_classes = num_classes
        self.num_timesteps = num_timesteps

        # Noise schedule — identical hyperparameters to the main LDM
        self.scheduler = DDPMScheduler(
            num_train_timesteps=num_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
        )

        # Timestep MLP — mirrors LatentDiffusionMLP exactly
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.GELU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        # Class conditioning embedding: num_classes × time_dim = 4 × 64 = 256 params
        self.class_embedding = nn.Embedding(num_classes, time_dim)

        # MLP backbone — mirrors LatentDiffusionMLP exactly
        self.fc1 = nn.Linear(z_dim, 256)
        self.fc_time1 = nn.Linear(time_dim, 256)

        self.fc2 = nn.Linear(256, 256)
        self.fc_time2 = nn.Linear(time_dim, 256)

        self.fc3 = nn.Linear(256, z_dim)

        self.act = nn.SiLU()

    # ------------------------------------------------------------------
    def forward(
        self,
        z_macro: torch.Tensor,
        t: torch.Tensor,
        class_label: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the noise residual conditioned on fault class.

        Args:
            z_macro:     (B, z_dim)  noisy latent vectors
            t:           (B,)        integer diffusion timesteps
            class_label: (B,)        integer class indices in [0, num_classes)

        Returns:
            predicted_noise: (B, z_dim)
        """
        te = self.time_mlp(t)                       # (B, time_dim)
        ce = self.class_embedding(class_label)      # (B, time_dim)
        te = te + ce                                # additive FiLM injection

        h = self.fc1(z_macro) + self.fc_time1(te)
        h = self.act(h)

        h2 = self.fc2(h) + self.fc_time2(te)
        h2 = self.act(h2) + h                       # residual skip

        return self.fc3(h2)

    # ------------------------------------------------------------------
    def get_loss(
        self,
        z_macro: torch.Tensor,
        class_label: torch.Tensor,
    ) -> torch.Tensor:
        """Compute DDPM MSE loss conditioned on class labels.

        Args:
            z_macro:     (B, z_dim)  clean latent vectors
            class_label: (B,)        integer class indices

        Returns:
            loss: scalar tensor
        """
        device = z_macro.device
        B = z_macro.shape[0]

        _move_scheduler_to(self.scheduler, device)

        noise = torch.randn_like(z_macro)
        t = torch.randint(0, self.num_timesteps, (B,), device=device).long()

        noisy = self.scheduler.add_noise(z_macro, noise, t)
        noise_pred = self.forward(noisy, t, class_label)

        return F.mse_loss(noise_pred, noise)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        class_label: Union[int, torch.Tensor],
        device: str,
        z_macro_init: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate z_macro samples conditioned on a fault class.

        No oracle guidance is applied — this is the label-conditioned baseline.

        Args:
            n_samples:     number of samples to generate
            class_label:   int or (n_samples,) tensor of class indices
            device:        torch device string (e.g. 'cpu', 'cuda')
            z_macro_init:  optional (n_samples, z_dim) starting latent;
                           if None, starts from pure Gaussian noise

        Returns:
            z: (n_samples, z_dim) generated latent vectors
        """
        dev = torch.device(device)
        _move_scheduler_to(self.scheduler, dev)

        if isinstance(class_label, int):
            label_tensor = torch.full(
                (n_samples,), class_label, device=dev, dtype=torch.long
            )
        else:
            label_tensor = class_label.to(dev)

        z = (
            z_macro_init.to(dev)
            if z_macro_init is not None
            else torch.randn(n_samples, self.z_dim, device=dev)
        )

        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((n_samples,), t, device=dev, dtype=torch.long)
            noise_pred = self.forward(z, t_batch, label_tensor)
            z = self.scheduler.step(noise_pred, t, z)

        return z
