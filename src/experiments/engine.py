"""Shared experiment engine: context loading, density fitting, and the
generate-and-score loop every block driver reuses.

Cell conventions (long format, one row per generated class):
  backbone, estimator, class, speed, seed, n_fewshot,
  mmd, tstr_recall_<probe>, tstr_acc_<probe>, oracle_acc_<probe>, ratio_<probe>,
  held_out_ll, plus any sdedit override columns.
Bypass runs (oracle.mode == 'raw') are tagged bypass=True on every row.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.backbones import make_backbone
from src.configs import RunCfg, resolve_device, seed_everything
from src.data import get_dataset
from src.evaluation import per_class_mmd, signal_features, tstr
from src.oracle import make_guidance
from src.training import load_features, load_oracle


class Context:
    """Loads everything the blocks need once: bundle, oracle, cached features,
    PCA-space class clusters, evaluation feature maps of real data."""

    def __init__(self, cfg: RunCfg):
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.bundle = get_dataset(cfg.data)
        self.oracle = load_oracle(cfg, self.bundle, device=self.device)
        self.feats = {s: load_features(cfg, s) for s in ("train", "val", "test")}
        # PCA-space embeddings per split
        self.emb = {s: self.oracle.project(f["phi"].to(self.device)).cpu().numpy()
                    for s, f in self.feats.items()}
        self.n_classes = len(self.bundle.meta["class_names"])
        self.fault_classes = list(range(1, self.n_classes))
        self.speeds = list(self.bundle.meta["speeds_hz"])
        self._real_eval_cache: dict = {}
        self._backbones: dict = {}

    # -- real-data evaluation features (fixed map), cached per speed ---------
    def real_eval(self, split: str, speed: float | None):
        key = (split, speed)
        if key not in self._real_eval_cache:
            speeds = None if speed is None else [speed]
            a = self.bundle.arrays(split, speeds=speeds)
            m = a["label"] > 0
            self._real_eval_cache[key] = {
                "feats": signal_features(a["raw"][m], device=self.device),
                "labels": a["label"][m].numpy(),
            }
        return self._real_eval_cache[key]

    def backbone(self, name: str):
        if name not in self._backbones:
            bb = make_backbone(name)
            bb.prepare(self.cfg, self.bundle, self.device)
            self._backbones[name] = bb
        return self._backbones[name]

    # -- guidance densities ---------------------------------------------------
    def fit_density(self, kind: str, target_class: int, n_fewshot: int, seed: int):
        """Fit q_y on n few-shot REAL fault windows; held-out LL on val features."""
        fs = self.bundle.fewshot_fault_sets(n_fewshot, seed=seed)[target_class]
        x = self.bundle.normalize(fs["raw_phys"]).to(self.device)
        om = fs["omega"].to(self.device)
        with torch.no_grad():
            V = self.oracle.embed(x, om).cpu()
        est = make_guidance(kind, self.cfg.guidance)
        est.fit(V, seed=seed)
        # held-out: val-split features of this class (excluding overlap is
        # negligible: few-shot draws come from the same val pool — documented)
        mask = self.feats["val"]["label"].numpy() == target_class
        V_held = torch.from_numpy(self.emb["val"][mask])
        ll = est.held_out_ll(V_held)
        return est, ll

    # -- healthy sources per speed ---------------------------------------------
    def healthy_source(self, speed: float | None, n: int, seed: int):
        a = self.bundle.arrays("val", classes=[0],
                               speeds=None if speed is None else [speed])
        g = np.random.default_rng(seed)
        idx = torch.from_numpy(g.choice(a["raw"].shape[0], size=min(n, a["raw"].shape[0]),
                                        replace=False))
        return {"raw": a["raw"][idx], "omega": a["omega"][idx]}


def generate_and_score(
    ctx: Context,
    backbone_name: str,
    estimator: str | None,          # None = unguided (F3)
    speed: float,
    seed: int,
    n_fewshot: int = 50,
    n_gen: int | None = None,
    sdedit_overrides: dict | None = None,
    classes: list[int] | None = None,
    wrong_target: bool = False,      # F3 control: steer toward a random wrong class
) -> tuple[list[dict], dict]:
    """One factorial cell-group: synthesize every fault class at `speed`,
    compute per-class MMD + pooled TSTR. Returns (rows, extras)."""
    cfg = ctx.cfg
    seed_everything(seed)
    n_gen = n_gen or cfg.eval.n_gen_per_class
    classes = classes or ctx.fault_classes
    bb = ctx.backbone(backbone_name)
    gen_feats, gen_labels = [], []
    rows, traces = [], {}
    real_test = ctx.real_eval("test", speed)
    real_train = ctx.real_eval("train", speed)
    rng = np.random.default_rng(seed)
    for c in classes:
        guid, ll = (None, None)
        if estimator is not None:
            target = c
            if wrong_target:
                others = [k for k in classes if k != c]
                target = int(rng.choice(others))
            guid, ll = ctx.fit_density(estimator, target, n_fewshot, seed)
        src = ctx.healthy_source(speed, n_gen, seed + 1000 * c)
        out = bb.synthesize(c, n_gen, guidance=guid, source=src,
                            sdedit_overrides=sdedit_overrides)
        f = signal_features(out["x"].cpu(), device=ctx.device)
        gen_feats.append(f)
        gen_labels.append(np.full(f.shape[0], c))
        if out.get("trace") is not None:
            traces[c] = out["trace"]
        rows.append({
            "backbone": backbone_name, "estimator": estimator or "none",
            "class": c, "speed": speed, "seed": seed, "n_fewshot": n_fewshot,
            "held_out_ll": ll, "bypass": cfg.oracle.mode == "raw",
            "wrong_target": wrong_target,
            **(sdedit_overrides or {}),
        })
    G = np.vstack(gen_feats)
    L = np.concatenate(gen_labels)
    mmds = per_class_mmd(G, L, real_test["feats"], real_test["labels"],
                         max_samples=cfg.eval.mmd_max_samples, seed=seed)
    ts = tstr(G, L, real_train["feats"], real_train["labels"],
              real_test["feats"], real_test["labels"],
              probes=cfg.eval.probes, seed=seed)
    for row in rows:
        c = row["class"]
        row["mmd"] = mmds.get(c, np.nan)
        for probe, res in ts.items():
            row[f"tstr_recall_{probe}"] = res["per_class_recall"].get(c, np.nan)
            row[f"tstr_acc_{probe}"] = res["tstr_acc"]
            row[f"oracle_acc_{probe}"] = res["oracle_acc"]
            row[f"ratio_{probe}"] = res["ratio"]
    return rows, {"traces": traces, "gen_feats": G, "gen_labels": L}


def save_cells(rows: list[dict], path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    path = str(path)
    if path.endswith(".parquet"):
        try:
            df.to_parquet(path)
        except Exception:
            path = path.replace(".parquet", ".csv")
            df.to_csv(path, index=False)
    else:
        df.to_csv(path, index=False)
    return df
