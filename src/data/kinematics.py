"""Differentiable acceleration -> velocity -> position derivation.

Used identically at PINN training time and inside the guided-synthesis loop,
so the PINN never sees a train/inference mismatch (the failure mode the real
pipeline hit with its Strategy-D filtering).

The integration constants (v0, x0) are unobservable from acceleration alone.
For rotating machinery they are recovered SYNCHRONOUSLY: the steady-state
response is periodic in the rotation, so the true velocity/position average
to zero over any integer number of revolutions. With the measured rotation
speed (tachometer) we remove the mean computed over the largest integer
number of revolutions that fits the window — exact for periodic content,
no torchaudio, fully differentiable. Windows must cover >= 1 revolution
(DataCfg.T defaults to one revolution at the slowest configured speed).
"""
from __future__ import annotations

import torch


def _cumtrapz(y: torch.Tensor, dt: float) -> torch.Tensor:
    mid = 0.5 * (y[..., 1:] + y[..., :-1])
    integ = torch.cumsum(mid * dt, dim=-1)
    zero = torch.zeros_like(y[..., :1])
    return torch.cat([zero, integ], dim=-1)


def _sync_mean(y: torch.Tensor, span: torch.Tensor) -> torch.Tensor:
    """Per-window mean over the first `span` samples. y: (B, ..., T), span: (B,) long."""
    T = y.shape[-1]
    idx = torch.arange(T, device=y.device).view(*([1] * (y.dim() - 1)), T)
    span_b = span.view(-1, *([1] * (y.dim() - 1))).to(y.device)
    mask = (idx < span_b).to(y.dtype)
    return (y * mask).sum(dim=-1, keepdim=True) / span_b.to(y.dtype)


def _detrend(y: torch.Tensor) -> torch.Tensor:
    n = y.shape[-1]
    x = torch.linspace(-1.0, 1.0, n, device=y.device, dtype=y.dtype)
    xm = x - x.mean()
    ym = y - y.mean(dim=-1, keepdim=True)
    slope = (ym * xm).sum(dim=-1, keepdim=True) / (xm * xm).sum()
    return ym - slope * xm


def derive_kinematics(
    acc: torch.Tensor, fs: float, omega: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """acc: (B, C, T) physical acceleration. omega: (B,) rad/s (measured).

    Returns (vel, pos), same shape. With `omega`, offsets are removed
    synchronously (exact for rotation-periodic content); without it, falls
    back to linear detrending (legacy behaviour, partial-window error).
    """
    dt = 1.0 / fs
    T = acc.shape[-1]
    if omega is None:
        a = _detrend(acc)
        vel = _detrend(_cumtrapz(a, dt))
        pos = _detrend(_cumtrapz(vel, dt))
        return vel, pos
    samples_per_rev = 2 * torch.pi * fs / omega.to(acc.dtype)          # (B,)
    n_rev = torch.clamp((T / samples_per_rev).floor(), min=1.0)
    span = torch.clamp((n_rev * samples_per_rev).round().long(), max=T)
    a = acc - _sync_mean(acc, span)
    vel = _cumtrapz(a, dt)
    vel = vel - _sync_mean(vel, span)
    pos = _cumtrapz(vel, dt)
    pos = pos - _sync_mean(pos, span)
    return vel, pos
