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

    ctx = Context(cfg)
    run = RunDir(cfg, "verify_geometry")
    rows = geometry_table(ctx.emb["test"], ctx.feats["test"]["label"].numpy(),
                          ctx.feats["test"]["speed_hz"].numpy(), seed=cfg.seed)
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(run.file("geometry.csv"), index=False)
    hz = df.groupby("class")["henze_zirkler"].mean()
    names = ctx.bundle.meta["class_names"]
    by_name = {names[int(c)]: float(v) for c, v in hz.items()}
    checks = {}
    if "imbalance_uni" in by_name:
        fault_named = {k: v for k, v in by_name.items() if k != "healthy"}
        low_group = ("imbalance_uni", "looseness_skew", "misalign_cont")
        high_group = [k for k in ("imbalance_bi", "bpfo_impulsive", "combo_ring")
                      if k in fault_named]
        checks["uni_is_least_gaussian_of_low_group"] = by_name["imbalance_uni"] <= min(
            by_name.get(k, np.inf) for k in low_group)
        if high_group:
            checks["high_geometry_classes_above_uni"] = all(
                by_name[k] > by_name["imbalance_uni"] for k in high_group)
    gate = all(checks.values()) if checks else False
    sil = float(df["global_silhouette"].mean())
    run.log(gng_by_class=by_name, checks=checks, gate_passed=gate,
            mean_silhouette=sil)
    print(json.dumps({"gng_by_class": by_name, "checks": checks,
                      "gate_passed": gate, "mean_silhouette": sil}, indent=2))
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
    elif cmd == "report":
        report(cfg)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
