"""Single CLI entry point.

Usage:
  python run.py <command> [--config path.yaml] [--smoke] [key.path=value ...]

Commands:
  generate-data     build the mafaulda_synthetic tensors
  train-pinn        P1.3 (healthy-only physics training)
  train-oracle      P1.4 (embedding + PCA + cached features; needs train-pinn)
  verify-geometry   P1.5 gate (geometry dial ordering)
  train-jepa        P2.1 (anti-collapse encoder)
  train-dec1        P2.2   train-dec2  P2.3   train-ldm  P3.1
  train-all         everything above in order
  block-a .. block-f   experiment blocks
  full-run          everything in order with stage progress + ETA
                    (results/<exp>/PROGRESS.md); resumes past finished stages
  smoke-all         full pipeline + all blocks at smoke sizes
  report            aggregate manifests into report/ (refuses bypass rows)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from src.configs import load_config, replace


def _parse(argv):
    cmd = argv[1] if len(argv) > 1 else "help"
    yaml_path, overrides, smoke = None, [], False
    i = 2
    while i < len(argv):
        a = argv[i]
        if a == "--config":
            yaml_path = argv[i + 1]
            i += 2
        elif a == "--smoke":
            smoke = True
            i += 1
        elif "=" in a:
            overrides.append(a)
            i += 1
        else:
            raise SystemExit(f"Unrecognized argument: {a}")
    cfg = load_config(yaml_path, overrides)
    if smoke and not cfg.smoke:
        from src.configs import apply_smoke
        cfg = apply_smoke(replace(cfg, "smoke", True))
    return cmd, cfg


def verify_geometry(cfg):
    """P1.5 gate: the geometry dial must produce the DESIGNED G_y ordering."""
    import numpy as np
    from src.configs import RunDir
    from src.evaluation import geometry_table
    from src.experiments import Context

    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier

    ctx = Context(cfg)
    run = RunDir(cfg, "verify_geometry")
    labels = ctx.feats["test"]["label"].numpy()
    speeds = ctx.feats["test"]["speed_hz"].numpy()
    rows = geometry_table(ctx.emb["test"], labels, speeds, seed=cfg.seed)
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(run.file("geometry.csv"), index=False)
    hz = df.groupby("class")["henze_zirkler"].mean()
    names = ctx.bundle.meta["class_names"]
    by_name = {names[int(c)]: float(v) for c, v in hz.items()}
    # class-informativeness of the embedding (per speed, then averaged)
    knns = [float(cross_val_score(KNeighborsClassifier(5),
                                  ctx.emb["test"][speeds == sp], labels[speeds == sp],
                                  cv=5).mean()) for sp in np.unique(speeds)]
    knn = float(np.mean(knns))
    sil = float(df["global_silhouette"].mean())     # per-speed silhouettes, averaged
    checks = {
        # P1.3 gate: physics embedding must separate classes
        "silhouette_per_speed_ge_0.3": sil >= 0.3,
        "knn_per_speed_ge_0.9": knn >= 0.9,
    }
    if "imbalance_bi" in by_name and "healthy" in by_name:
        faults = {k: v for k, v in by_name.items() if k != "healthy"}
        # P1.5 gate: the geometry dial must produce its DESIGNED ordering
        checks["healthy_most_gaussian"] = by_name["healthy"] <= min(faults.values())
        checks["bimodal_least_gaussian"] = by_name["imbalance_bi"] >= max(faults.values())
        checks["matched_pair_bi_gt_uni"] = (
            by_name["imbalance_bi"] > by_name.get("imbalance_uni", np.inf))
        checks["gng_spread_ge_3x"] = max(faults.values()) >= 3 * min(faults.values())
    gate = all(checks.values()) if checks else False
    run.log(gng_by_class=by_name, checks=checks, gate_passed=gate,
            mean_silhouette=sil, knn_accuracy=knn)
    print(json.dumps({"gng_by_class": by_name, "checks": checks,
                      "gate_passed": gate, "mean_silhouette": sil,
                      "knn_accuracy": knn}, indent=2))
    return run


def train_all(cfg):
    from src.data import get_dataset
    from src.training import (
        train_decoder1, train_decoder2, train_jepa, train_ldm, train_oracle,
        train_pinn,
    )
    bundle = get_dataset(cfg.data)
    train_pinn(cfg, bundle)
    train_oracle(cfg, bundle)
    train_jepa(cfg, bundle)
    train_decoder1(cfg, bundle)
    train_decoder2(cfg, bundle)
    train_ldm(cfg, bundle)


<<<<<<< HEAD
def full_run(cfg, resume: bool = True):
    """Everything, in order, with stage-level progress + ETA in
    results/<experiment>/PROGRESS.md. `resume=True` skips training stages
    whose artifacts already exist and blocks that already produced output."""
    import time

    from src.data import get_dataset
    from src.training import get_artifact, train_decoder1, train_decoder2, \
        train_jepa, train_ldm, train_oracle, train_pinn
    import src.experiments as ex

    prog_path = Path(cfg.results_root) / cfg.experiment / "PROGRESS.md"
    prog_path.parent.mkdir(parents=True, exist_ok=True)
    state: list[dict] = []

    def flush(eta_note=""):
        lines = [f"# full-run progress — experiment `{cfg.experiment}`",
                 f"updated {time.strftime('%Y-%m-%d %H:%M:%S')}  {eta_note}", ""]
        for s in state:
            lines.append(f"- [{'x' if s['done'] else ' '}] {s['name']}"
                         + (f" — {s['mins']:.1f} min" if s["done"] else
                            (" — RUNNING" if s.get("running") else "")))
        prog_path.write_text("\n".join(lines), encoding="utf-8")

    def has_artifact(name):
        try:
            return get_artifact(cfg, name).exists()
        except (KeyError, FileNotFoundError):
            return False

    def has_block(letter):
        """A block counts as done if one of its run dirs logged outputs."""
        base = Path(cfg.results_root) / cfg.experiment
        if not base.exists():
            return False
        for p in base.iterdir():
            if p.is_dir() and p.name.startswith(f"block_{letter}_") \
                    and (p / "run.json").exists():
                if json.loads((p / "run.json").read_text()).get("outputs"):
                    return True
        return False

    bundle_holder = {}

    def bundle():
        if "b" not in bundle_holder:
            bundle_holder["b"] = get_dataset(cfg.data)
        return bundle_holder["b"]

    def block_c_with_b_cells(_cfg):
        base = Path(cfg.results_root) / cfg.experiment
        bdirs = sorted(p for p in base.iterdir()
                       if p.is_dir() and p.name.startswith("block_b_")
                       and (p / "cells.csv").exists())
        cells = str(bdirs[-1] / "cells.csv") if bdirs else None
        return ex.run_block_c(_cfg, cells_csv=cells)

    stages = [
        ("generate-data", lambda c: get_dataset(c.data), lambda: False),
        ("train-pinn", lambda c: train_pinn(c, bundle()), lambda: has_artifact("pinn")),
        ("train-oracle", lambda c: train_oracle(c, bundle()),
         lambda: has_artifact("oracle")),
        ("verify-geometry", verify_geometry, lambda: False),
        ("train-jepa", lambda c: train_jepa(c, bundle()), lambda: has_artifact("jepa")),
        ("train-dec1", lambda c: train_decoder1(c, bundle()),
         lambda: has_artifact("dec1")),
        ("train-dec2", lambda c: train_decoder2(c, bundle()),
         lambda: has_artifact("dec2")),
        ("train-ldm", lambda c: train_ldm(c, bundle()), lambda: has_artifact("ldm")),
        ("block-a", ex.run_block_a, lambda: has_block("a")),
        ("block-b", ex.run_block_b, lambda: has_block("b")),
        ("block-c", block_c_with_b_cells, lambda: has_block("c")),
        ("block-d", ex.run_block_d, lambda: has_block("d")),
        ("block-e", ex.run_block_e, lambda: has_block("e")),
        ("block-f", ex.run_block_f, lambda: has_block("f")),
        ("report", report, lambda: False),
    ]
    state.extend({"name": n, "done": False, "mins": 0.0} for n, _, _ in stages)
    flush()
    t_start = time.time()
    for i, (name, fn, skip) in enumerate(stages):
        if resume and skip():
            print(f"===== [{i+1}/{len(stages)}] {name}: SKIP (already done)", flush=True)
            state[i]["done"] = True
            flush()
            continue
        print(f"===== [{i+1}/{len(stages)}] {name}: START "
              f"{time.strftime('%H:%M:%S')} =====", flush=True)
        state[i]["running"] = True
        flush()
        t0 = time.time()
        fn(cfg)
        state[i].update(done=True, running=False, mins=(time.time() - t0) / 60)
        flush(f"(total elapsed {((time.time() - t_start) / 3600):.1f} h)")
        print(f"===== {name}: DONE in {state[i]['mins']:.1f} min =====", flush=True)
    print(f"FULL-RUN COMPLETE - see report/REPORT.md and {prog_path}", flush=True)


=======
>>>>>>> 402ec57b6d9bb0588135a73b98da6f5e885b7903
def report(cfg):
    """P5.1 aggregator: read manifests + cells, refuse bypass in headlines."""
    import pandas as pd
    root = Path(cfg.results_root)
    out_dir = Path("report")
    out_dir.mkdir(exist_ok=True)
    manifests, cells = [], []
    for rj in root.rglob("run.json"):
        m = json.loads(rj.read_text())
        m["_dir"] = str(rj.parent)
        manifests.append(m)
        cp = rj.parent / "cells.csv"
        if cp.exists():
            df = pd.read_csv(cp)
            df["_run"] = m["name"]
            df["_bypass"] = m.get("bypass", False)
            cells.append(df)
    lines = ["# Aggregated results\n"]
    lines.append(f"{len(manifests)} runs found under {root}/\n")
    if cells:
        allc = pd.concat(cells, ignore_index=True)
        headline = allc[~allc["_bypass"]]
        n_bypass = int(allc["_bypass"].sum())
        if n_bypass:
            lines.append(f"**{n_bypass} bypass-tagged rows EXCLUDED from headline tables.**\n")
        if len(headline):
            summary = headline.groupby(["backbone", "estimator"]).agg(
                mmd=("mmd", "mean"),
                recall=("tstr_recall_logistic", "mean"),
                ratio=("ratio_logistic", "mean"), n=("mmd", "size"))
            lines.append("## Fidelity by backbone x estimator\n")
            lines.append(summary.to_markdown())
            headline.to_csv(out_dir / "all_cells.csv", index=False)
    for m in manifests:
        keep = {k: v for k, v in m.get("outputs", {}).items()
                if not isinstance(v, (list, dict)) or k in ("c3_invariance", "f3_causal")}
        if keep:
            lines.append(f"\n## {m['name']} ({m['_dir']})\n")
            lines.append("```json\n" + json.dumps(keep, indent=2, default=str) + "\n```")
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote report/REPORT.md ({len(manifests)} runs)")


def main():
    cmd, cfg = _parse(sys.argv)
    if cmd == "generate-data":
        from src.data.mafaulda_synthetic import generate
        generate(cfg.data)
    elif cmd == "train-pinn":
        from src.training import train_pinn
        train_pinn(cfg)
    elif cmd == "train-oracle":
        from src.training import train_oracle
        train_oracle(cfg)
    elif cmd == "verify-geometry":
        verify_geometry(cfg)
    elif cmd == "train-jepa":
        from src.training import train_jepa
        train_jepa(cfg)
    elif cmd == "train-dec1":
        from src.training import train_decoder1
        train_decoder1(cfg)
    elif cmd == "train-dec2":
        from src.training import train_decoder2
        train_decoder2(cfg)
    elif cmd == "train-ldm":
        from src.training import train_ldm
        train_ldm(cfg)
    elif cmd == "train-all":
        train_all(cfg)
    elif cmd.startswith("block-"):
        import src.experiments as ex
        getattr(ex, f"run_block_{cmd.split('-')[1]}")(cfg)
    elif cmd == "smoke-all":
        from src.configs import apply_smoke
        cfg = apply_smoke(replace(cfg, "smoke", True))
        train_all(cfg)
        verify_geometry(cfg)
        import src.experiments as ex
        for blk in "abcdef":
            print(f"=== block {blk} ===")
            getattr(ex, f"run_block_{blk}")(cfg)
        print("SMOKE-ALL COMPLETE")
<<<<<<< HEAD
    elif cmd == "full-run":
        full_run(cfg)
=======
>>>>>>> 402ec57b6d9bb0588135a73b98da6f5e885b7903
    elif cmd == "report":
        report(cfg)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
