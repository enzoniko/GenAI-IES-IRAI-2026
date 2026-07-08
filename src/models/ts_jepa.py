"""TS-JEPA v2 — masked latent prediction + explicit anti-collapse.

v1 collapsed on real data (per-dim std ~1e-6, T12/T17 logs) because nothing
prevented the mean-pooled z_macro from degenerating. v2 adds VICReg-style
variance/covariance regularization on z_macro (always on) and an optional
SIGReg (Epps-Pulley) term for the A2 experimental arm.
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe)


class Tokenizer(nn.Module):
    def __init__(self, in_channels, patch_size, d_model):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.conv(x).transpose(1, 2)


def epps_pulley(z: torch.Tensor, n_proj: int = 16, seed: int = 0) -> torch.Tensor:
    """Sketched Epps-Pulley statistic of z (B, d) against N(0, 1) on random
    1-D projections; differentiable. Small = more isotropic-Gaussian."""
    B, d = z.shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    u = torch.randn(d, n_proj, generator=g).to(z.device, z.dtype)
    u = u / u.norm(dim=0, keepdim=True)
    s = z @ u                                                    # (B, k)
    t = torch.linspace(-4.0, 4.0, 33, device=z.device, dtype=z.dtype)
    w = torch.exp(-0.5 * t**2)
    w = w / w.sum()
    ts = t.view(-1, 1, 1) * s.view(1, B, n_proj)                 # (T, B, k)
    phi_re = torch.cos(ts).mean(dim=1)                           # (T, k)
    phi_im = torch.sin(ts).mean(dim=1)
    phi0 = torch.exp(-0.5 * t**2).view(-1, 1)
    err = (phi_re - phi0) ** 2 + phi_im**2
    return (w.view(-1, 1) * err).sum(dim=0).mean()


class TSJEPA(nn.Module):
    def __init__(self, in_channels=4, patch_size=25, d_model=128, nhead=4,
                 num_layers=4, dim_feedforward=512, ema_decay=0.99,
                 var_coef=1.0, cov_coef=0.05, sigreg_coef=0.0):
        super().__init__()
        self.tokenizer = Tokenizer(in_channels, patch_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                         dim_feedforward=dim_feedforward, batch_first=True)
        self.context_encoder = nn.TransformerEncoder(enc, num_layers=num_layers)
        pred = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                          dim_feedforward=dim_feedforward, batch_first=True)
        self.predictor = nn.TransformerEncoder(pred, num_layers=2)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.ema_decay = ema_decay
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.var_coef, self.cov_coef, self.sigreg_coef = var_coef, cov_coef, sigreg_coef

    @torch.no_grad()
    def update_ema(self):
        for cp, tp in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            tp.data = self.ema_decay * tp.data + (1.0 - self.ema_decay) * cp.data

    def get_z_macro(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)
        tokens = tokens + self.pos_enc.pe[: tokens.size(1), :].unsqueeze(0)
        return self.context_encoder(tokens).mean(dim=1)

    def _predictive_terms(self, x):
        tokens = self.tokenizer(x)
        B, N, D = tokens.shape
        num_mask = N // 2
        perm = torch.randperm(N, device=x.device)
        mask_idx, keep_idx = perm[:num_mask], perm[num_mask:]
        with torch.no_grad():
            tgt = self.target_encoder(tokens + self.pos_enc.pe[:N, :].unsqueeze(0))
            tgt_unobs = tgt[:, mask_idx, :]
        obs = tokens[:, keep_idx, :] + self.pos_enc.pe[keep_idx, :].unsqueeze(0)
        ctx = self.context_encoder(obs)
        pred_in = self.mask_token.expand(B, num_mask, -1) + self.pos_enc.pe[mask_idx, :].unsqueeze(0)
        predicted = self.predictor(torch.cat([ctx, pred_in], dim=1))[:, -num_mask:, :]
        return predicted, tgt_unobs

    def loss(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        pred, tgt = self._predictive_terms(x)
        pred_loss = F.mse_loss(pred, tgt)
        z = self.get_z_macro(x)                                   # (B, d)
        zc = z - z.mean(dim=0, keepdim=True)
        std = torch.sqrt(zc.var(dim=0) + 1e-6)
        var_loss = F.relu(1.0 - std).pow(2).mean()                # hinge at std=1
        B, d = z.shape
        cov = (zc.T @ zc) / max(B - 1, 1)
        off = cov - torch.diag(torch.diag(cov))
        cov_loss = off.pow(2).sum() / d
        total = pred_loss + self.var_coef * var_loss + self.cov_coef * cov_loss
        out = {"pred": pred_loss, "var": var_loss, "cov": cov_loss}
        if self.sigreg_coef > 0:
            sig = epps_pulley(zc / (std + 1e-6))
            total = total + self.sigreg_coef * sig
            out["sigreg"] = sig
        out["total"] = total
        out["z_std_mean"] = std.mean().detach()
        return out

    def collapse_metrics(self, z: torch.Tensor) -> dict[str, float]:
        zc = z - z.mean(dim=0, keepdim=True)
        std = zc.std(dim=0)
        cov = (zc.T @ zc) / max(z.shape[0] - 1, 1)
        ev = torch.linalg.eigvalsh(cov).clamp(min=0)
        p = ev / ev.sum().clamp(min=1e-12)
        eff_rank = torch.exp(-(p * torch.log(p + 1e-12)).sum()).item()
        return {"z_std_mean": std.mean().item(), "z_std_min": std.min().item(),
                "effective_rank": eff_rank}
