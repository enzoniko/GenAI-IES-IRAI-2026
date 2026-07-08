"""Phase-1 trainers: TS-JEPA v2, Decoder1 (envelope), Decoder2 (CVAE jitter)."""
from __future__ import annotations

import torch

from src.configs import RunCfg, RunDir, resolve_device, seed_everything
from src.data import get_dataset
from src.models import Decoder1, Decoder2CVAE, TSJEPA
from .common import EarlyStopping, get_artifact, register_artifact


def _make_jepa(cfg: RunCfg, sigreg: bool | None = None) -> TSJEPA:
    j = cfg.jepa
    coef = j.sigreg_coef if sigreg is None else (0.1 if sigreg else 0.0)
    return TSJEPA(in_channels=4, patch_size=j.patch_size, d_model=j.d_model,
                  nhead=j.nhead, num_layers=j.num_layers,
                  dim_feedforward=j.dim_feedforward, ema_decay=j.ema_decay,
                  var_coef=j.var_coef, cov_coef=j.cov_coef, sigreg_coef=coef)


def load_jepa(cfg: RunCfg, name: str = "jepa", device="cpu") -> TSJEPA:
    m = _make_jepa(cfg)
    m.load_state_dict(torch.load(get_artifact(cfg, name), map_location="cpu",
                                 weights_only=True))
    m.to(device).eval()
    for p in m.parameters():
        p.requires_grad = False
    return m


def train_jepa(cfg: RunCfg, bundle=None, run: RunDir | None = None,
               artifact_name: str = "jepa") -> RunDir:
    """P2.1. Set cfg.jepa.sigreg_coef > 0 for the A2 arm (artifact 'jepa_sigreg')."""
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    bundle = bundle or get_dataset(cfg.data)
    run = run or RunDir(cfg, artifact_name)
    j = cfg.jepa
    model = _make_jepa(cfg).to(device)
    tr = bundle.loader("train", j.batch_size, shuffle=True)
    va = bundle.loader("val", j.batch_size)
    opt = torch.optim.Adam(model.parameters(), lr=j.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
    es = EarlyStopping(patience=j.early_stop_patience)
    best_state = None
    for epoch in range(j.epochs):
        model.train()
        tr_tot, nb = 0.0, 0
        for b in tr:
            out = model.loss(b.raw.to(device))
            opt.zero_grad()
            out["total"].backward()
            opt.step()
            model.update_ema()
            tr_tot += out["total"].item()
            nb += 1
        model.eval()
        va_tot, zs = 0.0, []
        with torch.no_grad():
            for b in va:
                out = model.loss(b.raw.to(device))
                va_tot += out["total"].item()
                zs.append(model.get_z_macro(b.raw.to(device)))
        va_loss = va_tot / max(len(va), 1)
        cm = model.collapse_metrics(torch.cat(zs))
        sched.step(va_loss)
        if es(va_loss):
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"[jepa] epoch {epoch}: train {tr_tot / max(nb,1):.4f} val {va_loss:.4f} "
              f"z_std {cm['z_std_mean']:.3f} rank {cm['effective_rank']:.1f}")
        if es.stop:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        zs = torch.cat([model.get_z_macro(b.raw.to(device)) for b in va])
    cm = model.collapse_metrics(zs)
    ckpt = run.file(f"{artifact_name}.pth")
    torch.save(model.state_dict(), ckpt)
    register_artifact(cfg, artifact_name, ckpt)
    run.log(val_loss=es.best, collapse=cm, ckpt=str(ckpt))
    return run


def train_decoder1(cfg: RunCfg, bundle=None, run: RunDir | None = None) -> RunDir:
    """P2.2: envelope decoder trained on CLEAN targets (ground truth available)."""
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    bundle = bundle or get_dataset(cfg.data)
    run = run or RunDir(cfg, "dec1")
    jepa = load_jepa(cfg, device=device)
    model = Decoder1(d_model=cfg.jepa.d_model, seq_length=bundle.meta["T"]).to(device)
    tr = bundle.loader("train", cfg.dec1.batch_size, shuffle=True)
    va = bundle.loader("val", cfg.dec1.batch_size)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.dec1.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
    es = EarlyStopping(patience=cfg.dec1.early_stop_patience)
    best_state = None
    for epoch in range(cfg.dec1.epochs):
        model.train()
        tr_tot, nb = 0.0, 0
        for b in tr:
            with torch.no_grad():
                z = jepa.get_z_macro(b.raw.to(device))
            loss = torch.mean((model(z) - b.clean.to(device)) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_tot += loss.item()
            nb += 1
        model.eval()
        va_tot = 0.0
        with torch.no_grad():
            for b in va:
                z = jepa.get_z_macro(b.raw.to(device))
                va_tot += torch.mean((model(z) - b.clean.to(device)) ** 2).item()
        va_loss = va_tot / max(len(va), 1)
        sched.step(va_loss)
        if es(va_loss):
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0 or es.stop:
            print(f"[dec1] epoch {epoch}: train {tr_tot / max(nb,1):.5f} val {va_loss:.5f}")
        if es.stop:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    # spectral acceptance: envelope output must not contain the jitter band
    model.eval()
    with torch.no_grad():
        b = next(iter(va))
        y = model(jepa.get_z_macro(b.raw.to(device)))
        spec = torch.fft.rfft(y.double(), dim=-1).abs() ** 2
        freqs = torch.fft.rfftfreq(y.shape[-1], 1 / bundle.meta["fs"]).to(spec.device)
        hf_frac = (spec[..., freqs > 4000].sum() / spec.sum()).item()
    ckpt = run.file("dec1.pth")
    torch.save(model.state_dict(), ckpt)
    register_artifact(cfg, "dec1", ckpt)
    run.log(val_mse=es.best, hf_energy_frac=hf_frac, ckpt=str(ckpt))
    return run


def load_decoder1(cfg: RunCfg, T: int, device="cpu") -> Decoder1:
    m = Decoder1(d_model=cfg.jepa.d_model, seq_length=T)
    m.load_state_dict(torch.load(get_artifact(cfg, "dec1"), map_location="cpu",
                                 weights_only=True))
    m.to(device).eval()
    for p in m.parameters():
        p.requires_grad = False
    return m


def train_decoder2(cfg: RunCfg, bundle=None, run: RunDir | None = None) -> RunDir:
    """P2.3: CVAE on the residual raw - Decoder1(z). beta_tc>0 = E1 arm."""
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    bundle = bundle or get_dataset(cfg.data)
    run = run or RunDir(cfg, "dec2")
    jepa = load_jepa(cfg, device=device)
    dec1 = load_decoder1(cfg, bundle.meta["T"], device=device)
    n_cls = len(bundle.meta["class_names"])
    d2 = cfg.dec2
    model = Decoder2CVAE(seq_length=bundle.meta["T"], num_classes=n_cls,
                         context_dim=cfg.jepa.d_model, latent_dim=d2.latent_dim,
                         label_embed_dim=d2.label_embed_dim,
                         beta_kl=d2.beta_kl, beta_tc=d2.beta_tc).to(device)
    tr = bundle.loader("train", d2.batch_size, shuffle=True)
    va = bundle.loader("val", d2.batch_size)
    opt = torch.optim.Adam(model.parameters(), lr=d2.lr)
    es = EarlyStopping(patience=d2.early_stop_patience)
    best_state = None

    def residual_of(b):
        with torch.no_grad():
            z = jepa.get_z_macro(b.raw.to(device))
            res = b.raw.to(device) - dec1(z)
        return z, res

    for epoch in range(d2.epochs):
        model.train()
        tr_tot, nb = 0.0, 0
        for b in tr:
            z, res = residual_of(b)
            out = model.loss(res, z, b.label.to(device))
            opt.zero_grad()
            out["total"].backward()
            opt.step()
            tr_tot += out["total"].item()
            nb += 1
        model.eval()
        va_tot, kl_sum = 0.0, 0.0
        with torch.no_grad():
            for b in va:
                z, res = residual_of(b)
                out = model.loss(res, z, b.label.to(device))
                va_tot += out["total"].item()
                kl_sum += out["kl"].item()
        va_loss = va_tot / max(len(va), 1)
        if es(va_loss):
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0 or es.stop:
            print(f"[dec2] epoch {epoch}: train {tr_tot / max(nb,1):.5f} "
                  f"val {va_loss:.5f} kl {kl_sum / max(len(va),1):.5f}")
        if es.stop:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    ckpt = run.file("dec2.pth")
    torch.save(model.state_dict(), ckpt)
    register_artifact(cfg, "dec2", ckpt)
    run.log(val_elbo=es.best, final_kl=kl_sum / max(len(va), 1), ckpt=str(ckpt))
    return run


def load_decoder2(cfg: RunCfg, T: int, n_classes: int, device="cpu") -> Decoder2CVAE:
    d2 = cfg.dec2
    m = Decoder2CVAE(seq_length=T, num_classes=n_classes, context_dim=cfg.jepa.d_model,
                     latent_dim=d2.latent_dim, label_embed_dim=d2.label_embed_dim,
                     beta_kl=d2.beta_kl, beta_tc=d2.beta_tc)
    m.load_state_dict(torch.load(get_artifact(cfg, "dec2"), map_location="cpu",
                                 weights_only=True))
    m.to(device).eval()
    for p in m.parameters():
        p.requires_grad = False
    return m
