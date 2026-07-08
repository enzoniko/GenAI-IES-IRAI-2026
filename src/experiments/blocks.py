"""Experiment blocks A–F. Each driver consumes a RunCfg (which selects the
factorial via cfg.eval.seeds / data.speeds_hz / guidance kinds passed as
arguments), writes cells + block JSON into its RunDir, and returns the RunDir.
"""
from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.configs import RunCfg, RunDir, resolve_device, seed_everything
from src.data import get_dataset
from src.evaluation import (
    correlate_geometry_fidelity, delta_silhouette_transfer, geometry_table,
    invariance_grade, mi_z_jitter, signal_features, spectral_leakage, tstr,
)
from src.training import (
    load_decoder1, load_decoder2, load_jepa, train_decoder2, train_jepa,
)
from .engine import Context, Progress, generate_and_score, save_cells

ESTIMATORS = ("gaussian", "gmm", "flow")
BACKBONE_NAMES = ("ours", "cvae", "faultdiff")


# ===========================================================================
# Block A — embedding & disentanglement controls
# ===========================================================================
class _ConvAE(nn.Module):
    """Plain autoencoder baseline for A1 (no JEPA objective)."""

    def __init__(self, T, d=128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(4, 32, 7, stride=4, padding=3), nn.GELU(),
            nn.Conv1d(32, 64, 7, stride=4, padding=3), nn.GELU(),
            nn.Conv1d(64, 128, 7, stride=4, padding=3), nn.GELU(),
            nn.AdaptiveAvgPool1d(8), nn.Flatten(), nn.Linear(128 * 8, d),
        )
        self.dec_fc = nn.Linear(d, 128 * 32)
        self.dec = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="nearest"), nn.Conv1d(128, 64, 7, padding=3),
            nn.GELU(),
            nn.Upsample(scale_factor=4, mode="nearest"), nn.Conv1d(64, 32, 7, padding=3),
            nn.GELU(),
            nn.Upsample(size=T, mode="linear", align_corners=False),
            nn.Conv1d(32, 4, 7, padding=3),
        )

    def forward(self, x):
        z = self.enc(x)
        return self.dec(self.dec_fc(z).view(-1, 128, 32)), z


def run_block_a(cfg: RunCfg) -> RunDir:
    run = RunDir(cfg, "block_a")
    device = resolve_device(cfg.device)
    bundle = get_dataset(cfg.data)
    if not bundle.meta["has_ground_truth"]:
        run.log(skipped="requires ground truth (jitter) — not available on real data")
        return run
    seed_everything(cfg.seed)
    out: dict = {}

    # A1: I(z ; jitter) for JEPA vs plain AE
    jepa = load_jepa(cfg, device=device)
    mi_epochs = 60 if cfg.smoke else 300
    out["a1_mi_jepa"] = mi_z_jitter(jepa.get_z_macro, bundle, device, mi_epochs=mi_epochs)
    ae = _ConvAE(bundle.meta["T"], d=cfg.jepa.d_model).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    epochs = 2 if cfg.smoke else 20
    tr = bundle.loader("train", cfg.jepa.batch_size, shuffle=True)
    for _ in range(epochs):
        for b in tr:
            x = b.raw.to(device)
            rec, _ = ae(x)
            loss = torch.mean((rec - x) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
    ae.eval()
    out["a1_mi_ae"] = mi_z_jitter(lambda x: ae(x)[1], bundle, device, mi_epochs=mi_epochs)

    # A2: SIGReg arm — train a SIGReg-regularized encoder, record collapse +
    # embedding stats (full C3 rerun on this arm = separate experiment config)
    sig_cfg = dataclasses.replace(cfg, jepa=dataclasses.replace(cfg.jepa, sigreg_coef=0.1))
    sig_run = train_jepa(sig_cfg, bundle=bundle, artifact_name="jepa_sigreg")
    out["a2_sigreg_collapse"] = sig_run.manifest["outputs"]["collapse"]
    with torch.no_grad():
        zs = torch.cat([load_jepa(cfg, "jepa_sigreg", device).get_z_macro(b.raw.to(device))
                        for b in bundle.loader("val", 32)])
    from src.models import epps_pulley
    out["a2_sigreg_epps_pulley_z"] = float(epps_pulley(
        (zs - zs.mean(0)) / (zs.std(0) + 1e-6)))

    # A3: spectral leakage between decoded envelope and CVAE residual
    dec1 = load_decoder1(cfg, bundle.meta["T"], device=device)
    a = bundle.arrays("val")
    x = a["raw"][:16].to(device)
    with torch.no_grad():
        env = dec1(jepa.get_z_macro(x))
        resid = x - env
    out["a3_leakage"] = spectral_leakage(env, resid, bundle.meta["fs"])

    run.log(**out)
    (run.file("block_a.json")).write_text(json.dumps(out, indent=2, default=float))
    return run


# ===========================================================================
# Block B — guidance density estimation (core, cross-backbone)
# ===========================================================================
def run_block_b(cfg: RunCfg, backbones=BACKBONE_NAMES, estimators=ESTIMATORS,
                fewshot_sweep=(5, 10, 20, 50, 100)) -> RunDir:
    run = RunDir(cfg, "block_b")
    ctx = Context(cfg)
    rows = []
    speeds = ctx.speeds
    seeds = list(cfg.eval.seeds)
<<<<<<< HEAD
    b3_seeds = seeds[: min(3, len(seeds))]
    total = (len(backbones) * len(estimators) * len(speeds) * len(seeds)
             + len(estimators) * len(fewshot_sweep) * len(b3_seeds))
    prog = Progress(total, "block_b")
=======
>>>>>>> 402ec57b6d9bb0588135a73b98da6f5e885b7903
    # B1/B2: estimator x backbone x class x speed x seed
    for bb in backbones:
        for est in estimators:
            for sp in speeds:
                for seed in seeds:
                    r, _ = generate_and_score(ctx, bb, est, sp, seed)
                    rows.extend(r)
                    save_cells(rows, run.file("cells.csv"))
<<<<<<< HEAD
                    prog.tick(f"{bb}/{est}/sp{sp:g}/s{seed}")
=======
>>>>>>> 402ec57b6d9bb0588135a73b98da6f5e885b7903
    # B3: few-shot sensitivity (home speed, primary backbone, all estimators)
    home = 16.0 if 16.0 in speeds else speeds[0]
    for est in estimators:
        for n_fs in fewshot_sweep:
<<<<<<< HEAD
            for seed in b3_seeds:
=======
            for seed in seeds[: min(3, len(seeds))]:
>>>>>>> 402ec57b6d9bb0588135a73b98da6f5e885b7903
                r, _ = generate_and_score(ctx, backbones[0], est, home, seed,
                                          n_fewshot=n_fs)
                rows.extend(r)
                save_cells(rows, run.file("cells.csv"))
<<<<<<< HEAD
                prog.tick(f"B3 {est}/n{n_fs}/s{seed}")
=======
>>>>>>> 402ec57b6d9bb0588135a73b98da6f5e885b7903
    df = save_cells(rows, run.file("cells.csv"))
    run.log(n_rows=len(df), cells=str(run.file("cells.csv")))
    return run


# ===========================================================================
# Block C — geometry characterization + the PRIMARY C3 test
# ===========================================================================
def run_block_c(cfg: RunCfg, cells_csv: str | None = None) -> RunDir:
    run = RunDir(cfg, "block_c")
    ctx = Context(cfg)
    # C1: geometry table in the PCA embedding space (test split = unbiased)
    geo = geometry_table(ctx.emb["test"], ctx.feats["test"]["label"].numpy(),
                         ctx.feats["test"]["speed_hz"].numpy(), seed=cfg.seed)
    geo_df = pd.DataFrame(geo)
    geo_df.to_csv(run.file("geometry.csv"), index=False)

    out = {"c1_geometry": geo}
    # C2 + C3 need generated cells: reuse Block B output if provided,
    # otherwise generate a minimal default arm.
    if cells_csv is None:
        rows = []
        home = 16.0 if 16.0 in ctx.speeds else ctx.speeds[0]
        for seed in cfg.eval.seeds[: min(2, len(cfg.eval.seeds))]:
            r, extras = generate_and_score(ctx, "ours", "gaussian", home, seed)
            rows.extend(r)
        cells = pd.DataFrame(rows)
        # C2 on the last generation
        real_test = ctx.real_eval("test", home)
        out["c2_delta_silhouette"] = delta_silhouette_transfer(
            extras["gen_feats"], extras["gen_labels"],
            real_test["feats"], real_test["labels"])
    else:
        cells = pd.read_csv(cells_csv)
    # C3: join G_y onto cells and correlate
    merged = cells.merge(geo_df[["class", "speed", "henze_zirkler", "epps_pulley",
                                 "mardia_skew", "mardia_kurt_excess"]],
                         on=["class", "speed"], how="inner")
    merged.to_csv(run.file("cells_with_geometry.csv"), index=False)
    corr_frames = {}
    for gng_col in ("henze_zirkler", "epps_pulley"):
        for fid_col in ("mmd", "tstr_recall_logistic"):
            key = f"{gng_col}|{fid_col}"
            cdf = correlate_geometry_fidelity(merged, gng_col=gng_col,
                                              fidelity_col=fid_col)
            corr_frames[key] = cdf
            cdf.to_csv(run.file(f"c3_corr_{gng_col}_{fid_col}.csv"), index=False)
    primary = corr_frames["henze_zirkler|mmd"]
    grade = invariance_grade(primary) if "backbone" in primary.columns and len(primary) \
        else {"grade": "insufficient"}
    out["c3_primary"] = primary.to_dict(orient="records")
    out["c3_invariance"] = grade
    run.log(**{k: v for k, v in out.items() if k != "c1_geometry"},
            geometry_csv=str(run.file("geometry.csv")))
    (run.file("block_c.json")).write_text(json.dumps(out, indent=2, default=float))
    return run


# ===========================================================================
# Block D — guidance-strength and sampling controls
# ===========================================================================
def run_block_d(cfg: RunCfg, intervals=(1, 2, 5, 10, 25),
                t0_fracs=(0.2, 0.35, 0.5, 0.65, 0.8)) -> RunDir:
    run = RunDir(cfg, "block_d")
    ctx = Context(cfg)
    home = 16.0 if 16.0 in ctx.speeds else ctx.speeds[0]
    seeds = cfg.eval.seeds[: min(2, len(cfg.eval.seeds))]
    rows, trace_stats = [], []
<<<<<<< HEAD
    prog = Progress((len(intervals) + len(t0_fracs) + len(ctx.speeds)) * len(seeds),
                    "block_d")
=======
>>>>>>> 402ec57b6d9bb0588135a73b98da6f5e885b7903
    for N in intervals:                                   # D1
        for seed in seeds:
            r, ex = generate_and_score(ctx, "ours", "gaussian", home, seed,
                                       sdedit_overrides={"guidance_interval": N})
            rows.extend(r)
            for c, tr in ex["traces"].items():            # D3 source data
                trace_stats.append({"class": c, "guidance_interval": N,
                                    "t0_frac": cfg.sdedit.t0_frac, "seed": seed,
                                    "grad_norm_mean": float(np.mean(tr.grad_norm)),
                                    "grad_norm_std": float(np.std(tr.grad_norm)),
                                    "penalty_start": tr.penalty[0],
                                    "penalty_end": tr.penalty[-1]})
<<<<<<< HEAD
            prog.tick(f"D1 N={N}/s{seed}")
=======
>>>>>>> 402ec57b6d9bb0588135a73b98da6f5e885b7903
    for t0 in t0_fracs:                                   # D2
        for seed in seeds:
            r, ex = generate_and_score(ctx, "ours", "gaussian", home, seed,
                                       sdedit_overrides={"t0_frac": t0})
            rows.extend(r)
<<<<<<< HEAD
            prog.tick(f"D2 t0={t0}/s{seed}")
=======
>>>>>>> 402ec57b6d9bb0588135a73b98da6f5e885b7903
    df = save_cells(rows, run.file("cells.csv"))
    tdf = pd.DataFrame(trace_stats)
    tdf.to_csv(run.file("d3_traces.csv"), index=False)
    # D3: correlate grad-norm with per-class MMD
    d3 = {}
    if len(tdf) and len(df):
        m = tdf.groupby("class")["grad_norm_mean"].mean()
        f = df[df["estimator"] == "gaussian"].groupby("class")["mmd"].mean()
        joined = pd.concat([m, f], axis=1).dropna()
        if len(joined) >= 3:
            from scipy import stats
            rho = stats.spearmanr(joined["grad_norm_mean"], joined["mmd"])
            d3 = {"spearman_rho": float(rho.statistic), "p": float(rho.pvalue)}
    # D4: multi-speed transfer (uses artifacts trained on cfg.data as-is;
    # per-speed-trained arms are separate experiment configs)
    d4_rows = []
    for sp in ctx.speeds:
        for seed in seeds:
            r, _ = generate_and_score(ctx, "ours", "gaussian", sp, seed)
            d4_rows.extend(r)
<<<<<<< HEAD
            prog.tick(f"D4 sp{sp:g}/s{seed}")
=======
>>>>>>> 402ec57b6d9bb0588135a73b98da6f5e885b7903
    save_cells(d4_rows, run.file("d4_cells.csv"))
    run.log(d3_gradnorm_vs_mmd=d3, n_rows=len(df), n_d4_rows=len(d4_rows))
    return run


# ===========================================================================
# Block E — capacity interventions and extensions
# ===========================================================================
def run_block_e(cfg: RunCfg, holdout_class: int = 4) -> RunDir:
    run = RunDir(cfg, "block_e")
    ctx = Context(cfg)
    device = ctx.device
    home = 16.0 if 16.0 in ctx.speeds else ctx.speeds[0]
    seed = cfg.eval.seeds[0]
    out: dict = {}

    # E1: beta-TCVAE arm — retrain Decoder2 with TC penalty, compare fidelity
    tc_cfg = dataclasses.replace(cfg, dec2=dataclasses.replace(cfg.dec2, beta_tc=1.0),
                                 experiment=cfg.experiment + "_e1tc")
    import shutil
    from pathlib import Path
    src_art = Path(cfg.results_root) / cfg.experiment / "artifacts.json"
    dst_dir = Path(cfg.results_root) / tc_cfg.experiment
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_art, dst_dir / "artifacts.json")
    train_decoder2(tc_cfg, bundle=ctx.bundle)
    ctx_tc = Context(tc_cfg)
    r_tc, _ = generate_and_score(ctx_tc, "ours", "gaussian", home, seed)
    r_base, _ = generate_and_score(ctx, "ours", "gaussian", home, seed)
    out["e1_mmd_base"] = float(np.nanmean([r["mmd"] for r in r_base]))
    out["e1_mmd_tcvae"] = float(np.nanmean([r["mmd"] for r in r_tc]))

    # E2: latent modality check (BIC over k) on z_macro
    jepa = load_jepa(cfg, device=device)
    with torch.no_grad():
        Z = torch.cat([jepa.get_z_macro(b.raw.to(device)).cpu()
                       for b in ctx.bundle.loader("train", 64)]).numpy()
    from sklearn.mixture import GaussianMixture
    bics = {k: float(GaussianMixture(k, covariance_type="diag", random_state=0)
                     .fit(Z).bic(Z)) for k in (1, 2, 4, 8)}
    out["e2_latent_bic"] = bics
    out["e2_multimodal"] = min(bics, key=bics.get) > 1

    # E3: open-set — build the target density by interpolating the two nearest
    # OTHER fault clusters, synthesize, measure distance to the TRUE held-out
    # cluster (synthetic superpower) + open-space risk
    emb_tr, lab_tr = ctx.emb["train"], ctx.feats["train"]["label"].numpy()
    cents = {c: emb_tr[lab_tr == c].mean(axis=0) for c in ctx.fault_classes}
    others = [c for c in ctx.fault_classes if c != holdout_class]
    d = {c: np.linalg.norm(cents[c] - cents[holdout_class]) for c in others}
    n1, n2 = sorted(d, key=d.get)[:2]
    from src.oracle import GaussianGuidance
    V1 = torch.from_numpy(emb_tr[lab_tr == n1])
    V2 = torch.from_numpy(emb_tr[lab_tr == n2])
    interp = GaussianGuidance().fit(torch.cat([V1, V2]))       # moment interpolation
    bb = ctx.backbone("ours")
    src = ctx.healthy_source(home, cfg.eval.n_gen_per_class, seed)
    gen = bb.synthesize(holdout_class, cfg.eval.n_gen_per_class, guidance=interp,
                        source=src)
    with torch.no_grad():
        Vg = ctx.oracle.embed(gen["x"], gen["omega"]).cpu().numpy()
    true_c = cents[holdout_class]
    d_true = np.linalg.norm(Vg - true_c, axis=1).mean()
    d_others = {c: float(np.linalg.norm(Vg - cents[c], axis=1).mean()) for c in others}
    healthy_cent = emb_tr[lab_tr == 0].mean(axis=0)
    d_healthy = np.linalg.norm(Vg - healthy_cent, axis=1)
    nearest_fault = np.min(np.stack([np.linalg.norm(Vg - cents[c], axis=1)
                                     for c in ctx.fault_classes]), axis=0)
    out["e3"] = {"holdout": holdout_class, "anchors": [int(n1), int(n2)],
                 "dist_to_true_holdout": float(d_true),
                 "dist_to_other_faults": d_others,
                 "open_space_risk": float((d_healthy < nearest_fault).mean())}

    # E4: cross-machine zero-shot — variant-B data through A-trained pipeline
    b_cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, variant="B"))
    bundle_b = get_dataset(b_cfg.data)
    a_te = bundle_b.arrays("test")
    m = a_te["label"] > 0
    feats_b = signal_features(a_te["raw"][m], device=device)
    labels_b = a_te["label"][m].numpy()
    r_base_all = ctx.real_eval("train", None)
    gen_rows, extras = generate_and_score(ctx, "ours", "gaussian", home, seed)
    ts = tstr(extras["gen_feats"], extras["gen_labels"],
              r_base_all["feats"], r_base_all["labels"], feats_b, labels_b,
              probes=("logistic",), seed=seed)
    out["e4_zero_shot_tstr_on_B"] = ts["logistic"]

    run.log(**out)
    (run.file("block_e.json")).write_text(json.dumps(out, indent=2, default=float))
    return run


# ===========================================================================
# Block F — benchmarking, probe sensitivity, causal ablation
# ===========================================================================
def run_block_f(cfg: RunCfg, backbones=BACKBONE_NAMES) -> RunDir:
    run = RunDir(cfg, "block_f")
    ctx = Context(cfg)
    home = 16.0 if 16.0 in ctx.speeds else ctx.speeds[0]
    rows = []
<<<<<<< HEAD
    f3_seeds = cfg.eval.seeds[: min(3, len(cfg.eval.seeds))]
    prog = Progress(len(backbones) * len(cfg.eval.seeds) + 2 * len(f3_seeds), "block_f")
=======
>>>>>>> 402ec57b6d9bb0588135a73b98da6f5e885b7903
    # F1: multi-seed benchmark (gaussian guidance for every backbone)
    for bb in backbones:
        for seed in cfg.eval.seeds:
            r, _ = generate_and_score(ctx, bb, "gaussian", home, seed)
            rows.extend(r)
            save_cells(rows, run.file("cells.csv"))
<<<<<<< HEAD
            prog.tick(f"F1 {bb}/s{seed}")
    # F3: causal ablation on ours — no guidance / wrong-target guidance
    for seed in f3_seeds:
        r, _ = generate_and_score(ctx, "ours", None, home, seed)
        rows.extend(r)
        prog.tick(f"F3 unguided/s{seed}")
        r, _ = generate_and_score(ctx, "ours", "gaussian", home, seed,
                                  wrong_target=True)
        rows.extend(r)
        prog.tick(f"F3 wrong-target/s{seed}")
=======
    # F3: causal ablation on ours — no guidance / wrong-target guidance
    for seed in cfg.eval.seeds[: min(3, len(cfg.eval.seeds))]:
        r, _ = generate_and_score(ctx, "ours", None, home, seed)
        rows.extend(r)
        r, _ = generate_and_score(ctx, "ours", "gaussian", home, seed,
                                  wrong_target=True)
        rows.extend(r)
>>>>>>> 402ec57b6d9bb0588135a73b98da6f5e885b7903
    df = save_cells(rows, run.file("cells.csv"))
    # F2: probe sensitivity — do conclusions flip across probes?
    f1 = df[(df["estimator"] == "gaussian") & (~df.get("wrong_target", False))]
    probe_means = {p: f1.groupby("backbone")[f"ratio_{p}"].mean().to_dict()
                   for p in cfg.eval.probes}
    rankings = {p: sorted(v, key=v.get, reverse=True) for p, v in probe_means.items()}
    f2_consistent = len({tuple(r) for r in rankings.values()}) == 1
    # F3 summary
    ours = df[df["backbone"] == "ours"]
    f3 = {
        "guided_mmd": float(ours[(ours["estimator"] == "gaussian")
                                 & (~ours["wrong_target"])]["mmd"].mean()),
        "unguided_mmd": float(ours[ours["estimator"] == "none"]["mmd"].mean()),
        "wrong_target_mmd": float(ours[ours["wrong_target"] == True]["mmd"].mean()),
        "guided_recall": float(ours[(ours["estimator"] == "gaussian")
                                    & (~ours["wrong_target"])]["tstr_recall_logistic"].mean()),
        "unguided_recall": float(ours[ours["estimator"] == "none"]["tstr_recall_logistic"].mean()),
    }
    run.log(f1_probe_means=probe_means, f2_rankings=rankings,
            f2_probe_consistent=f2_consistent, f3_causal=f3, n_rows=len(df))
    return run
