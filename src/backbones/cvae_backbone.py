"""External backbone #1: signal-space conditional VAE (independent of our
encoder/decoders — a genuinely different generative mechanism for Block F).

Guidance mechanism: density-weighted resampling — generate `oversample * n`
candidates, keep the top-n by guidance log-probability of their physics
embedding. This is how a non-diffusion backbone consumes the same
GuidanceDensity objects (documented design decision)."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from src.configs import RunCfg
from src.training import get_artifact, load_oracle, register_artifact
from src.training.common import EarlyStopping


class _SignalCVAE(nn.Module):
    def __init__(self, T: int, n_classes: int, latent_dim=64, label_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc = nn.Sequential(
            nn.Conv1d(4, 32, 7, stride=4, padding=3), nn.GELU(),
            nn.Conv1d(32, 64, 7, stride=4, padding=3), nn.GELU(),
            nn.Conv1d(64, 128, 7, stride=4, padding=3), nn.GELU(),
            nn.AdaptiveAvgPool1d(32),
        )
        self.label_embed = nn.Embedding(n_classes, label_dim)
        self.fc_mu = nn.Linear(128 * 32 + label_dim, latent_dim)
        self.fc_lv = nn.Linear(128 * 32 + label_dim, latent_dim)
        self.dec_fc = nn.Linear(latent_dim + label_dim, 128 * 64)
        self.dec = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="nearest"),
            nn.Conv1d(128, 64, 7, padding=3), nn.GELU(),
            nn.Upsample(scale_factor=4, mode="nearest"),
            nn.Conv1d(64, 32, 7, padding=3), nn.GELU(),
            nn.Upsample(size=T, mode="linear", align_corners=False),
            nn.Conv1d(32, 4, 7, padding=3),
        )

    def decode(self, z, label):
        h = self.dec_fc(torch.cat([z, self.label_embed(label)], dim=1))
        return self.dec(h.view(-1, 128, 64))

    def forward(self, x, label):
        h = self.enc(x).flatten(1)
        h = torch.cat([h, self.label_embed(label)], dim=1)
        mu, lv = self.fc_mu(h), self.fc_lv(h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        return self.decode(z, label), mu, lv


class CVAEBackbone:
    name = "cvae"

    def __init__(self, beta_kl=0.01, oversample=4, epochs=60, lr=1e-3, batch=32):
        self.beta_kl, self.oversample = beta_kl, oversample
        self.epochs, self.lr, self.batch = epochs, lr, batch
        self.ready = False

    def prepare(self, cfg: RunCfg, bundle, device) -> None:
        self.cfg, self.bundle, self.device = cfg, bundle, device
        n_cls = len(bundle.meta["class_names"])
        self.model = _SignalCVAE(bundle.meta["T"], n_cls).to(device)
        self.oracle = load_oracle(cfg, bundle, device=device)
        try:
            p = get_artifact(cfg, "cvae_backbone")
            self.model.load_state_dict(torch.load(p, map_location=device, weights_only=True))
        except (KeyError, FileNotFoundError):
            self._train()
        self.model.eval()
        self.ready = True

    def _train(self):
        epochs = 5 if self.cfg.smoke else self.epochs
        tr = self.bundle.loader("train", self.batch, shuffle=True)
        va = self.bundle.loader("val", self.batch)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        es = EarlyStopping(patience=10)
        best = None
        for ep in range(epochs):
            self.model.train()
            for b in tr:
                x, lab = b.raw.to(self.device), b.label.to(self.device)
                rec, mu, lv = self.model(x, lab)
                loss = torch.mean((rec - x) ** 2) - self.beta_kl * 0.5 * torch.mean(
                    1 + lv - mu.pow(2) - lv.exp())
                opt.zero_grad()
                loss.backward()
                opt.step()
            self.model.eval()
            vl = 0.0
            with torch.no_grad():
                for b in va:
                    x, lab = b.raw.to(self.device), b.label.to(self.device)
                    rec, mu, lv = self.model(x, lab)
                    vl += torch.mean((rec - x) ** 2).item()
            vl /= max(len(va), 1)
            if es(vl):
                best = {k: v.clone() for k, v in self.model.state_dict().items()}
            if es.stop:
                break
        if best is not None:
            self.model.load_state_dict(best)
        path = Path(self.cfg.results_root) / self.cfg.experiment / "cvae_backbone.pth"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        register_artifact(self.cfg, "cvae_backbone", path)

    @torch.no_grad()
    def synthesize(self, target_class: int, n: int, guidance=None,
                   source: dict | None = None, sdedit_overrides=None) -> dict:
        assert self.ready
        m = n * (self.oversample if guidance is not None else 1)
        z = torch.randn(m, self.model.latent_dim, device=self.device)
        lab = torch.full((m,), target_class, device=self.device, dtype=torch.long)
        x = self.model.decode(z, lab)
        a = self.bundle.arrays("val", classes=[0])
        om = a["omega"][torch.randint(0, a["omega"].shape[0], (m,))].to(self.device)
        if guidance is not None:
            v = self.oracle.embed(x, om)
            lp = guidance.log_prob(v)
            top = torch.topk(lp, k=n).indices
            x, om = x[top], om[top]
        return {"x": x[:n], "omega": om[:n], "trace": None}
