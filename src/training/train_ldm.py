"""P3.1: latent DDPM on z_macro (all classes); optional class conditioning (E2)."""
from __future__ import annotations

import torch

from src.configs import RunCfg, RunDir, resolve_device, seed_everything
from src.data import get_dataset
from src.models import DDIMSampler, DDPMScheduler, LatentDiffusionMLP
from .common import EarlyStopping, get_artifact, register_artifact
from .train_phase1 import load_jepa


def make_scheduler(cfg: RunCfg, device="cpu") -> DDPMScheduler:
    return DDPMScheduler(cfg.ldm.num_train_timesteps, cfg.ldm.beta_start,
                         cfg.ldm.beta_end, device=device)


def train_ldm(cfg: RunCfg, bundle=None, run: RunDir | None = None,
              artifact_name: str = "ldm") -> RunDir:
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    bundle = bundle or get_dataset(cfg.data)
    run = run or RunDir(cfg, artifact_name)
    lc = cfg.ldm
    jepa = load_jepa(cfg, device=device)
    n_cls = len(bundle.meta["class_names"])
    model = LatentDiffusionMLP(z_dim=cfg.jepa.d_model, time_dim=lc.time_dim,
                               hidden=lc.hidden,
                               num_classes=n_cls if lc.class_cond else None).to(device)
    sch = make_scheduler(cfg, device)

    def encode(split):
        zs, labs = [], []
        with torch.no_grad():
            for b in bundle.loader(split, lc.batch_size):
                zs.append(jepa.get_z_macro(b.raw.to(device)))
                labs.append(b.label.to(device))
        return torch.cat(zs), torch.cat(labs)

    Ztr, Ltr = encode("train")
    Zva, Lva = encode("val")
    # standardize latents: eps-prediction diffusion assumes ~N(0,1)-scale data;
    # raw z_macro (per-dim std ~0.7, nonzero means) biased the denoiser and made
    # DDIM trajectories drift off-manifold (measured: eta=0 SDEdit round-trip
    # MSE 65.7 vs 0.26 direct reconstruction before this fix)
    z_mean, z_std = Ztr.mean(0), Ztr.std(0) + 1e-6
    Ztr = (Ztr - z_mean) / z_std
    Zva = (Zva - z_mean) / z_std
    opt = torch.optim.Adam(model.parameters(), lr=lc.lr)
    es = EarlyStopping(patience=lc.early_stop_patience)
    best_state = None
    n = Ztr.shape[0]
    for epoch in range(lc.epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        tr_tot, nb = 0.0, 0
        for s0 in range(0, n, lc.batch_size):
            idx = perm[s0:s0 + lc.batch_size]
            z, lab = Ztr[idx], Ltr[idx]
            noise = torch.randn_like(z)
            ts = torch.randint(0, sch.num_train_timesteps, (z.shape[0],), device=device)
            noisy = sch.add_noise(z, noise, ts)
            pred = model(noisy, ts, lab if lc.class_cond else None)
            loss = torch.mean((pred - noise) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_tot += loss.item()
            nb += 1
        model.eval()
        with torch.no_grad():
            noise = torch.randn_like(Zva)
            ts = torch.randint(0, sch.num_train_timesteps, (Zva.shape[0],), device=device,
                               generator=torch.Generator(device=device).manual_seed(epoch))
            pred = model(sch.add_noise(Zva, noise, ts), ts,
                         Lva if lc.class_cond else None)
            va_loss = torch.mean((pred - noise) ** 2).item()
        if es(va_loss):
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or es.stop:
            print(f"[ldm] epoch {epoch}: train {tr_tot / max(nb,1):.4f} val {va_loss:.4f}")
        if es.stop:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    # sample-statistics acceptance: DDIM samples must live at the real latent scale
    model.eval()
    ddim = DDIMSampler(sch)
    lab = torch.zeros(64, dtype=torch.long, device=device) if lc.class_cond else None
    zgen = ddim.sample(model, 64, cfg.jepa.d_model, device, n_steps=50, label=lab)
    ratio = (zgen.std() / Ztr.std()).item()
    ckpt = run.file(f"{artifact_name}.pth")
    torch.save({"state_dict": model.state_dict(),
                "z_mean": z_mean.cpu(), "z_std": z_std.cpu()}, ckpt)
    register_artifact(cfg, artifact_name, ckpt)
    run.log(val_loss=es.best, sample_std_ratio=ratio, ckpt=str(ckpt))
    return run


def load_ldm(cfg: RunCfg, n_classes: int, device="cpu",
             artifact_name: str = "ldm") -> LatentDiffusionMLP:
    lc = cfg.ldm
    m = LatentDiffusionMLP(z_dim=cfg.jepa.d_model, time_dim=lc.time_dim, hidden=lc.hidden,
                           num_classes=n_classes if lc.class_cond else None)
    ckpt = torch.load(get_artifact(cfg, artifact_name), map_location="cpu",
                      weights_only=True)
    m.load_state_dict(ckpt["state_dict"])
    # latent standardization stats (see train_ldm) — consumers must run the
    # diffusion in normalized space and denormalize before decoding
    m.z_mean = ckpt["z_mean"].to(device)
    m.z_std = ckpt["z_std"].to(device)
    m.to(device).eval()
    for p in m.parameters():
        p.requires_grad = False
    return m
