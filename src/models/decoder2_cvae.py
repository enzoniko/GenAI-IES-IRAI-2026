"""Conditional VAE for jitter synthesis, config-free, with optional
beta-TCVAE total-correlation penalty (experiment E1)."""
import math

import torch
import torch.nn as nn


class EncoderCVAE(nn.Module):
    def __init__(self, in_channels=4, context_dim=128, latent_dim=64,
                 num_classes=7, label_embed_dim=16):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.AdaptiveAvgPool1d(100),
        )
        self.label_embed = nn.Embedding(num_classes, label_embed_dim)
        self.fc_joint = nn.Sequential(
            nn.Linear(128 * 100 + context_dim + label_embed_dim, 512), nn.GELU(),
            nn.Linear(512, 256), nn.GELU(),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, residual, z_macro, label):
        h = self.cnn(residual).flatten(1)
        h = self.fc_joint(torch.cat([h, z_macro, self.label_embed(label)], dim=1))
        return self.fc_mu(h), self.fc_logvar(h)


class DecoderCVAE(nn.Module):
    def __init__(self, latent_dim=64, context_dim=128, seq_length=4167,
                 out_channels=4, num_classes=7, label_embed_dim=16):
        super().__init__()
        self.init_length, self.init_channels = 125, 256
        self.label_embed = nn.Embedding(num_classes, label_embed_dim)
        self.fc_proj = nn.Sequential(
            nn.Linear(latent_dim + context_dim + label_embed_dim, 512), nn.GELU(),
            nn.Linear(512, self.init_channels * self.init_length),
        )
        self.cnn = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(256, 128, kernel_size=5, padding=2), nn.BatchNorm1d(128), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(128, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(64, 32, kernel_size=5, padding=2), nn.BatchNorm1d(32), nn.GELU(),
            nn.Upsample(size=seq_length, mode="linear", align_corners=False),
            nn.Conv1d(32, out_channels, kernel_size=5, padding=2),
        )

    def forward(self, z, z_macro, label):
        x = self.fc_proj(torch.cat([z, z_macro, self.label_embed(label)], dim=1))
        return self.cnn(x.view(-1, self.init_channels, self.init_length))


class Decoder2CVAE(nn.Module):
    def __init__(self, seq_length=4167, in_channels=4, context_dim=128,
                 latent_dim=64, num_classes=7, label_embed_dim=16,
                 beta_kl=0.01, beta_tc=0.0):
        super().__init__()
        self.encoder = EncoderCVAE(in_channels, context_dim, latent_dim,
                                   num_classes, label_embed_dim)
        self.decoder = DecoderCVAE(latent_dim, context_dim, seq_length,
                                   in_channels, num_classes, label_embed_dim)
        self.latent_dim = latent_dim
        self.beta_kl, self.beta_tc = beta_kl, beta_tc

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def forward(self, residual, z_macro, label):
        mu, logvar = self.encoder(residual, z_macro, label)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z, z_macro, label), mu, logvar, z

    def loss(self, residual, z_macro, label) -> dict[str, torch.Tensor]:
        recon, mu, logvar, z = self.forward(residual, z_macro, label)
        rec = torch.mean((recon - residual) ** 2)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total = rec + self.beta_kl * kl
        out = {"recon": rec, "kl": kl}
        if self.beta_tc > 0:
            tc = self._total_correlation(z, mu, logvar)
            total = total + self.beta_tc * tc
            out["tc"] = tc
        out["total"] = total
        return out

    @staticmethod
    def _total_correlation(z, mu, logvar):
        """Minibatch-weighted-sampling TC estimate (beta-TCVAE, Chen et al. 2018)."""
        B, D = z.shape
        lv = logvar.unsqueeze(0)
        m = mu.unsqueeze(0)
        zz = z.unsqueeze(1)
        log_qzx = -0.5 * ((zz - m) ** 2 / lv.exp() + lv + math.log(2 * math.pi))  # (B,B,D)
        log_qz = torch.logsumexp(log_qzx.sum(dim=2), dim=1) - math.log(B)
        log_qz_prod = (torch.logsumexp(log_qzx, dim=1) - math.log(B)).sum(dim=1)
        return (log_qz - log_qz_prod).mean()

    def sample(self, z_macro, label):
        z = torch.randn(z_macro.size(0), self.latent_dim, device=z_macro.device)
        return self.decoder(z, z_macro, label)
