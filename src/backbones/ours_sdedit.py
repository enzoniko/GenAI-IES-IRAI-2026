"""Ours: JEPA latent + physics-guided SDEdit (v2 — the CORRECT implementation).

Fixes vs the withdrawn paper's code:
  * t0 lives on the TRAINING schedule: t0 = t0_frac * num_train_timesteps
    (the old code computed `int(num_inference_steps * strength)` = 1 step).
  * density penalty = -log q_y(v) through a GuidanceDensity (the old code
    used MSE to a class-mean embedding while the paper claimed Mahalanobis).
  * guidance interval N, gradient clipping tau, per-step ||g|| logging (D3).
"""
from __future__ import annotations

import dataclasses

import torch

from src.configs import RunCfg, SdeditCfg
from src.models import DDIMSampler
from src.training import (
    load_decoder1, load_decoder2, load_jepa, load_ldm, load_oracle, make_scheduler,
)
from .base import StepTrace


class OursSDEditBackbone:
    name = "ours"

    def __init__(self):
        self.ready = False

    def prepare(self, cfg: RunCfg, bundle, device) -> None:
        self.cfg, self.bundle, self.device = cfg, bundle, device
        n_cls = len(bundle.meta["class_names"])
        self.jepa = load_jepa(cfg, device=device)
        self.dec1 = load_decoder1(cfg, bundle.meta["T"], device=device)
        self.dec2 = load_decoder2(cfg, bundle.meta["T"], n_cls, device=device)
        self.ldm = load_ldm(cfg, n_cls, device=device)
        self.scheduler = make_scheduler(cfg, device)
        self.oracle = load_oracle(cfg, bundle, device=device)
        self.ddim = DDIMSampler(self.scheduler)
        self.ready = True

    def _source_healthy(self, n: int, source: dict | None):
        if source is None:
            a = self.bundle.arrays("val", classes=[0])
            idx = torch.randperm(a["raw"].shape[0])[:n]
            source = {"raw": a["raw"][idx], "omega": a["omega"][idx]}
        x = source["raw"][:n].to(self.device)
        om = source["omega"][:n].to(self.device)
        if x.shape[0] < n:  # tile if fewer healthy sources than requested
            reps = -(-n // x.shape[0])
            x, om = x.repeat(reps, 1, 1)[:n], om.repeat(reps)[:n]
        return x, om

    def synthesize(self, target_class: int, n: int, guidance=None,
                   source: dict | None = None,
                   sdedit_overrides: dict | None = None) -> dict:
        assert self.ready
        sc: SdeditCfg = self.cfg.sdedit
        if sdedit_overrides:
            sc = dataclasses.replace(sc, **sdedit_overrides)
        x_h, omega = self._source_healthy(n, source)
        with torch.no_grad():
            z = self.jepa.get_z_macro(x_h)
        # diffusion runs in the LDM's standardized latent space
        zm, zs = self.ldm.z_mean, self.ldm.z_std
        z = (z - zm) / zs
        t0 = max(1, int(sc.t0_frac * self.scheduler.num_train_timesteps))
        noise = torch.randn_like(z)
        z_t = self.scheduler.add_noise(
            z, noise, torch.full((z.shape[0],), t0 - 1, device=self.device, dtype=torch.long))
        ts = self.ddim.schedule(t0, sc.n_infer)
        trace = StepTrace() if sc.record_trace else None
        for i, t in enumerate(ts):
            t_prev = ts[i + 1] if i + 1 < len(ts) else -1
            with torch.no_grad():
                eps = self.ldm(z_t, torch.full((z.shape[0],), t, device=self.device,
                                               dtype=torch.long))
            g = torch.zeros_like(z_t)
            if guidance is not None and i % max(1, sc.guidance_interval) == 0:
                z_req = z_t.detach().requires_grad_(True)
                x_tmp = self.dec1(z_req * zs + zm)
                v = self.oracle.embed(x_tmp, omega)
                pen = guidance.penalty(v)
                g = torch.autograd.grad(pen, z_req)[0]
                gn = g.norm()
                if gn > sc.clip_tau:
                    g = g * (sc.clip_tau / gn)
                if trace is not None:
                    trace.t.append(t)
                    trace.penalty.append(float(pen.detach()))
                    trace.grad_norm.append(float(gn.detach()))
                    trace.z_snapshots.append(z_t.detach().cpu())
            with torch.no_grad():
                z_t = self.ddim.step(z_t, eps, t, t_prev) - sc.guidance_scale * g
        with torch.no_grad():
            lab = torch.full((z_t.shape[0],), target_class, device=self.device,
                             dtype=torch.long)
            z_out = z_t * zs + zm
            x_out = self.dec1(z_out) + self.dec2.sample(z_out, lab)
        return {"x": x_out.detach(), "omega": omega, "trace": trace,
                "z_final": z_out.detach()}
