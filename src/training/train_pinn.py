"""P1.3: train the RotorPINN on HEALTHY synthetic data only (all speeds).

Pointwise regression + physics residuals balanced by ReLoBRaLo. Inputs are
derived from the raw acceleration via the SAME synchronous-kinematics
pipeline used at guidance time (train/inference consistency)."""
from __future__ import annotations

import torch

from src.configs import RunCfg, RunDir, resolve_device, seed_everything
from src.data import derive_kinematics, get_dataset
from src.models import Normalization, ReLoBRaLoLoss, RotorPINN
from src.physics import RotorParams
from .common import EarlyStopping, register_artifact


def build_pointwise(bundle, split: str, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Healthy windows -> pointwise (X (N*T,10), Y (N*T,4)), physical units."""
    a = bundle.arrays(split, classes=[0])
    acc = a["raw_phys"].double()                          # (N, 4, T)
    omega = a["omega"].double()
    fs = bundle.meta["fs"]
    vel, pos = derive_kinematics(acc, fs=fs, omega=omega)
    N, C, T = acc.shape
    t = (torch.arange(T, dtype=torch.float64) / fs).expand(N, T)
    om = omega.view(N, 1).expand(N, T)
    X = torch.cat([
        vel.permute(0, 2, 1).reshape(N * T, 4),
        pos.permute(0, 2, 1).reshape(N * T, 4),
        om.reshape(N * T, 1), t.reshape(N * T, 1),
    ], dim=1)
    Y = acc.permute(0, 2, 1).reshape(N * T, 4)
    return X, Y


def train_pinn(cfg: RunCfg, bundle=None, run: RunDir | None = None) -> RunDir:
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    bundle = bundle or get_dataset(cfg.data)
    run = run or RunDir(cfg, "pinn")
    pc = cfg.pinn

    Xtr, Ytr = build_pointwise(bundle, "train", device)
    Xva, Yva = build_pointwise(bundle, "val", device)
    norm = Normalization.fit(Xtr, Ytr)
    params = RotorParams.variant(cfg.data.variant if cfg.data.name == "mafaulda_synthetic" else "A")
    model = RotorPINN(norm, params=params, hidden_layers=pc.hidden_layers,
                      activation=pc.activation, dropout=pc.dropout).to(device)
    # force scale: residuals are in Newtons, O(F0); let the net output that scale
    F0 = params.M1 * (Xtr[:, 8].mean().item()) ** 2 * params.E1
    model.force_scale = F0

    Xtr_n = norm.norm_x(Xtr)
    Ytr_n = norm.norm_y(Ytr)
    Xva_n = norm.norm_x(Xva).to(device)
    Yva_n = norm.norm_y(Yva).to(device)

    relo = ReLoBRaLoLoss(num_terms=6, alpha=pc.relobralo_alpha,
                         rho=pc.relobralo_rho, temperature=pc.relobralo_temperature)
    opt = torch.optim.Adam(model.parameters(), lr=pc.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=8)
    es = EarlyStopping(patience=pc.early_stop_patience)
    r_scale = F0**2  # normalize residual MSE to O(1)

    def terms(xb, yb):
        acc_n, forces = model(xb)
        data = torch.mean((acc_n - yb) ** 2)
        res = model.compute_residuals(xb, acc_n, forces)
        rl = [torch.mean(res[:, i] ** 2) / r_scale for i in range(4)]
        fpen = pc.lambda_force * torch.mean(forces**2) / r_scale
        return [data, *rl, fpen]

    n = Xtr_n.shape[0]
    best_state = None
    for epoch in range(pc.epochs):
        model.train()
        perm = torch.randperm(n)
        tr_tot = 0.0
        nb = 0
        for s0 in range(0, n, pc.batch_size):
            idx = perm[s0:s0 + pc.batch_size]
            xb, yb = Xtr_n[idx].to(device), Ytr_n[idx].to(device)
            loss = relo.combine(terms(xb, yb))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_tot += loss.item()
            nb += 1
        model.eval()
        with torch.no_grad():
            va_terms = terms(Xva_n, Yva_n)
            va = sum(t.item() for t in va_terms)
        sched.step(va)
        if es(va):
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or es.stop:
            print(f"[pinn] epoch {epoch}: train {tr_tot / max(nb,1):.5f} "
                  f"val {va:.5f} (data {va_terms[0].item():.5f})")
        if es.stop:
            break
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        va_terms = terms(Xva_n, Yva_n)
    ckpt = run.file("pinn.pth")
    model.save(ckpt, extra={"val_terms": [t.item() for t in va_terms]})
    register_artifact(cfg, "pinn", ckpt)
    run.log(val_data_mse=va_terms[0].item(),
            val_residual_mse=[t.item() for t in va_terms[1:5]],
            val_force_pen=va_terms[5].item(), ckpt=str(ckpt))
    return run
