"""External backbone #2: FaultDiffusion-STYLE few-shot latent diffusion.

Faithful reimplementation of the mechanism described in FaultDiffusion
(arXiv:2511.15174) — no public code exists: a diffusion backbone trained on
NORMAL data only is frozen; a small per-class "positive-negative difference"
adapter, trained on a few-shot fault set, shifts the noise prediction toward
the fault distribution.

Scope note (documented confound): the latent space and the decoders are
shared with ours (signal-space diffusion at T~4200 is out of CPU budget).
The generative MECHANISM — frozen-normal backbone + few-shot adapter,
no physics guidance during its own training — is what differs."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from src.configs import RunCfg
from src.models import DDIMSampler, LatentDiffusionMLP
from src.training import (
    get_artifact, load_decoder1, load_decoder2, load_jepa, load_oracle,
    make_scheduler, register_artifact,
)


class _Adapter(nn.Module):
    """Low-rank residual on the noise prediction, conditioned on t."""

    def __init__(self, z_dim=128, rank=16, time_dim=64):
        super().__init__()
        from src.models.latent_diffusion import SinusoidalPositionEmbeddings
        self.time = nn.Sequential(SinusoidalPositionEmbeddings(time_dim),
                                  nn.Linear(time_dim, rank))
        self.down = nn.Linear(z_dim, rank)
        self.up = nn.Linear(rank, z_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, z, t):
        return self.up(torch.tanh(self.down(z) + self.time(t)))


class FaultDiffBackbone:
    name = "faultdiff"

    def __init__(self, fewshot_n=50, adapter_epochs=400, lr=1e-3):
        self.fewshot_n, self.adapter_epochs, self.lr = fewshot_n, adapter_epochs, lr
        self.adapters: dict[int, _Adapter] = {}
        self.ready = False

    def prepare(self, cfg: RunCfg, bundle, device) -> None:
        self.cfg, self.bundle, self.device = cfg, bundle, device
        n_cls = len(bundle.meta["class_names"])
        self.jepa = load_jepa(cfg, device=device)
        self.dec1 = load_decoder1(cfg, bundle.meta["T"], device=device)
        self.dec2 = load_decoder2(cfg, bundle.meta["T"], n_cls, device=device)
        self.oracle = load_oracle(cfg, bundle, device=device)
        self.scheduler = make_scheduler(cfg, device)
        self.ddim = DDIMSampler(self.scheduler)
        self.backbone = LatentDiffusionMLP(z_dim=cfg.jepa.d_model,
                                           time_dim=cfg.ldm.time_dim,
                                           hidden=cfg.ldm.hidden).to(device)
        try:
            p = get_artifact(cfg, "fd_backbone")
            ckpt = torch.load(p, map_location=device, weights_only=True)
            self.backbone.load_state_dict(ckpt["state_dict"])
            self.z_mean = ckpt["z_mean"].to(device)
            self.z_std = ckpt["z_std"].to(device)
        except (KeyError, FileNotFoundError):
            self._train_backbone()
            self.z_mean = self.z_mean.to(device)
            self.z_std = self.z_std.to(device)
        for p_ in self.backbone.parameters():
            p_.requires_grad = False
        self.backbone.eval()
        self.ready = True

    def _encode(self, loader):
        zs = []
        with torch.no_grad():
            for b in loader:
                zs.append(self.jepa.get_z_macro(b.raw.to(self.device)))
        return torch.cat(zs)

    def _train_backbone(self):
        """Frozen-normal stage: DDPM on HEALTHY latents only (standardized)."""
        epochs = 5 if self.cfg.smoke else self.cfg.ldm.epochs
        Z = self._encode(self.bundle.loader("train", 64, shuffle=False, classes=[0]))
        self.z_mean, self.z_std = Z.mean(0), Z.std(0) + 1e-6
        Z = (Z - self.z_mean) / self.z_std
        opt = torch.optim.Adam(self.backbone.parameters(), lr=self.cfg.ldm.lr)
        for ep in range(epochs):
            perm = torch.randperm(Z.shape[0], device=self.device)
            for s0 in range(0, Z.shape[0], 64):
                z = Z[perm[s0:s0 + 64]]
                noise = torch.randn_like(z)
                ts = torch.randint(0, self.scheduler.num_train_timesteps,
                                   (z.shape[0],), device=self.device)
                pred = self.backbone(self.scheduler.add_noise(z, noise, ts), ts)
                loss = torch.mean((pred - noise) ** 2)
                opt.zero_grad()
                loss.backward()
                opt.step()
        path = Path(self.cfg.results_root) / self.cfg.experiment / "fd_backbone.pth"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.backbone.state_dict(),
                    "z_mean": self.z_mean.cpu(), "z_std": self.z_std.cpu()}, path)
        register_artifact(self.cfg, "fd_backbone", path)

    def fit_adapter(self, target_class: int, fewshot: dict | None = None) -> _Adapter:
        """Few-shot difference adapter on n fault windows (default: bundle few-shot set)."""
        if target_class in self.adapters and fewshot is None:
            return self.adapters[target_class]
        if fewshot is None:
            fewshot = self.bundle.fewshot_fault_sets(self.fewshot_n, seed=self.cfg.seed)
        fs = fewshot[target_class]
        x = self.bundle.normalize(fs["raw_phys"]).to(self.device)
        with torch.no_grad():
            Z = self.jepa.get_z_macro(x)
        Z = (Z - self.z_mean) / self.z_std
        adapter = _Adapter(z_dim=self.cfg.jepa.d_model,
                           time_dim=self.cfg.ldm.time_dim).to(self.device)
        opt = torch.optim.Adam(adapter.parameters(), lr=self.lr)
        epochs = 50 if self.cfg.smoke else self.adapter_epochs
        for ep in range(epochs):
            noise = torch.randn_like(Z)
            ts = torch.randint(0, self.scheduler.num_train_timesteps,
                               (Z.shape[0],), device=self.device)
            noisy = self.scheduler.add_noise(Z, noise, ts)
            with torch.no_grad():
                base = self.backbone(noisy, ts)
            pred = base + adapter(noisy, ts)
            loss = torch.mean((pred - noise) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.adapters[target_class] = adapter
        return adapter

    def synthesize(self, target_class: int, n: int, guidance=None,
                   source: dict | None = None, sdedit_overrides=None) -> dict:
        assert self.ready
        adapter = self.fit_adapter(target_class)
        z = torch.randn(n, self.cfg.jepa.d_model, device=self.device)
        ts = self.ddim.schedule(self.scheduler.num_train_timesteps, self.cfg.sdedit.n_infer)
        sc = self.cfg.sdedit
        a = self.bundle.arrays("val", classes=[0])
        om = a["omega"][torch.randint(0, a["omega"].shape[0], (n,))].to(self.device)
        for i, t in enumerate(ts):
            t_prev = ts[i + 1] if i + 1 < len(ts) else -1
            tt = torch.full((n,), t, device=self.device, dtype=torch.long)
            with torch.no_grad():
                eps = self.backbone(z, tt) + adapter(z, tt)
            g = torch.zeros_like(z)
            if guidance is not None and i % max(1, sc.guidance_interval) == 0:
                z_req = z.detach().requires_grad_(True)
                v = self.oracle.embed(self.dec1(z_req * self.z_std + self.z_mean), om)
                pen = guidance.penalty(v)
                g = torch.autograd.grad(pen, z_req)[0]
                gn = g.norm()
                if gn > sc.clip_tau:
                    g = g * (sc.clip_tau / gn)
            with torch.no_grad():
                z = self.ddim.step(z, eps, t, t_prev) - sc.guidance_scale * g
        with torch.no_grad():
            lab = torch.full((n,), target_class, device=self.device, dtype=torch.long)
            z_out = z * self.z_std + self.z_mean
            x = self.dec1(z_out) + self.dec2.sample(z_out, lab)
        return {"x": x.detach(), "omega": om, "trace": None}
