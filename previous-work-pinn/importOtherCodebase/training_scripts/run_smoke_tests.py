#!/usr/bin/env python3
"""
Run smoke tests for all training scripts (synthetic and real data).

Usage examples:
  python training_scripts/run_smoke_tests.py --epochs 100 --batch-size 64 --learning-rate 1e-3 --max-samples 500 --device cpu --synthetic-data-path training_scripts/analytical_analysis/data_v2 --synthetic-sim-id 0

  # Skip real data tests
  python training_scripts/run_smoke_tests.py --skip-real

Notes:
  - Real data tests require Data/v3/X_normal_v3.pth and Data/v3/Y_normal_v3.pth in project root.
  - Synthetic path should contain simulation_000000, etc.
  - Uses dimensional PINN model (basicPINNv8.py) with 9 parameters: M1, M2, M3, D1, D2, D3, K1, K2, E1
"""

import argparse
import os
import sys
import time
import subprocess
from pathlib import Path


METHOD_SCRIPTS = [
    'adaptive_lbpin_training.py',
    'alpinn_training.py',
    'brdr_training.py',
    'constant_weight_pinn_training.py',
    'dwpinn_training.py',
    'gradnorm_training.py',
    'pecann_training.py',
    'relobralo_training.py',
]


def file_exists(path: Path) -> bool:
    try:
        return path.exists()
    except Exception:
        return False


def build_common_args(args: argparse.Namespace):
    """Build common arguments including architecture parameters."""
    return [
        '--epochs', str(args.epochs),
        '--batch-size', str(args.batch_size),
        '--learning-rate', str(args.learning_rate),
        '--max-samples', str(args.max_samples),
        '--device', args.device,
        # Add required architecture parameters
        '--hidden-layers', '64', '64',  # Simple 2-layer network for smoke test
        '--activation', 'tanh',  # Default activation
        '--dropout-rate', '0.0',  # No dropout for smoke test
        '--init-method', 'xavier_normal',  # Default initialization
        # Add training parameters
        '--early-patience', '50',  # Short patience for smoke test
        '--lr-patience', '25',  # Short LR patience for smoke test
        '--min-delta', '1e-6',  # Reasonable min delta
    ]


def run_cmd(cmd: list[str], cwd: Path) -> tuple[int, float, str, str]:
    start = time.time()
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    dur = time.time() - start
    return proc.returncode, dur, proc.stdout or '', proc.stderr or ''


def load_learned_params(result_dir: Path, script: str) -> dict:
    """Load learned parameters from training results.
    
    Note: The dimensional PINN model (basicPINNv8.py) has 9 parameters:
    M1, M2, M3, D1, D2, D3, K1, K2, E1
    """
    method = script.replace('_training.py', '')
    npz = result_dir / f"{method}_parameters.npz"
    params = {}
    if npz.exists():
        try:
            import numpy as np
            data = np.load(npz, allow_pickle=True)
            # Only look for the 9 parameters that exist in the dimensional model
            for k in ['M1','M2','M3','D1','D2','D3','K1','K2','E1']:
                if k in data:
                    params[k] = float(data[k])
        except Exception:
            pass
    return params


def load_final_losses(result_dir: Path, script: str) -> dict:
    """Load final-epoch losses from the saved history .npz if present.

    Returns keys among: train_total, val_total, data_train, data_val, phys_train, phys_val.
    """
    method = script.replace('_training.py', '')
    npz = result_dir / f"{method}_history.npz"
    losses = {}
    if npz.exists():
        try:
            import numpy as np
            data = np.load(npz, allow_pickle=True)
            def last_or_none(key: str):
                if key in data:
                    arr = data[key]
                    try:
                        # support object arrays or lists
                        return float(arr[-1])
                    except Exception:
                        return None
                return None
            for key in ['train_total','val_total','data_train','data_val','phys_train','phys_val']:
                v = last_or_none(key)
                if v is not None:
                    losses[key] = v
        except Exception:
            pass
    return losses


def load_ground_truth(synthetic_dir: Path, sim_id: int) -> dict:
    """Load ground truth parameters from synthetic data metadata.
    
    Note: The dimensional PINN model expects 9 parameters, but synthetic data
    may have different parameter names or additional parameters.
    """
    import json
    meta = synthetic_dir / f"simulation_{sim_id:06d}" / 'metadata.json'
    if not meta.exists():
        return {}
    try:
        with open(meta, 'r') as f:
            md = json.load(f)
        
        # Prefer PINN-compatible parameters for better comparison
        if 'pinn_parameters' in md:
            print(f"Using PINN-compatible parameters for simulation {sim_id}")
            return md['pinn_parameters']
        elif 'parameters' in md:
            print(f"PINN-compatible parameters not found, using original parameters for simulation {sim_id}")
            return md['parameters']
        else:
            return md.get('ground_truth_params', {})
    except Exception:
        return {}


def compute_param_errors(learned: dict, truth: dict) -> dict:
    """Compute parameter errors between learned and ground truth values.
    
    Only compares parameters that exist in both dictionaries.
    """
    import math
    errors = {}
    # Only compare the 9 parameters that exist in the dimensional model
    for k in ['M1','M2','M3','D1','D2','D3','K1','K2','E1']:
        if k in learned and k in truth and isinstance(truth[k], (int,float)):
            try:
                errors[k] = abs(float(learned[k]) - float(truth[k]))
            except Exception:
                errors[k] = math.nan
    return errors


def main():
    parser = argparse.ArgumentParser(description='Run smoke tests for all training scripts')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--max-samples', type=int, default=500)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--synthetic-data-path', type=str, default='training_scripts/analytical_analysis/data_v2')
    parser.add_argument('--synthetic-sim-id', type=int, default=0)
    parser.add_argument('--skip-real', action='store_true')
    parser.add_argument('--output-root', type=str, default='results/smoke_all')
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    out_root = project_root / args.output_root
    out_root.mkdir(parents=True, exist_ok=True)

    # Detect real data files
    real_X = project_root / 'Data' / 'v3' / 'X_normal_v3.pth'
    real_Y = project_root / 'Data' / 'v3' / 'Y_normal_v3.pth'
    real_available = (not args.skip_real) and file_exists(real_X) and file_exists(real_Y)

    # Detect synthetic dataset availability
    syn_dir = project_root / args.synthetic_data_path
    syn_ok = file_exists(syn_dir / f'simulation_{args.synthetic_sim_id:06d}' / 'time_series.npz') and \
             file_exists(syn_dir / f'simulation_{args.synthetic_sim_id:06d}' / 'metadata.json')

    print(f"Project root: {project_root}")
    print(f"Synthetic dataset: {syn_dir} (sim {args.synthetic_sim_id}) -> {'FOUND' if syn_ok else 'MISSING'}")
    print(f"Real dataset: Data/v3/X_normal_v3.pth & Data/v3/Y_normal_v3.pth -> {'FOUND' if real_available else 'MISSING'}")
    print(f"Using dimensional PINN model (basicPINNv8.py) with 9 parameters: M1, M2, M3, D1, D2, D3, K1, K2, E1")
    print(f"Note: Synthetic data now includes PINN-compatible parameter mapping for optimal training compatibility")

    if not syn_ok:
        print("ERROR: Synthetic dataset not found. Aborting.")
        sys.exit(2)

    common = build_common_args(args)
    results = []

    # Synthetic runs
    for script in METHOD_SCRIPTS:
        script_path = project_root / 'training_scripts' / script
        method_name = script.replace('_training.py', '')
        out_dir_path = out_root / f'{method_name}_synthetic'
        out_dir = str(out_dir_path)
        cmd = [
            sys.executable, str(script_path),
            *common,
            '--output-dir', out_dir,
            '--synthetic',
            '--simulation-id', str(args.synthetic_sim_id),
            '--data-path', str(syn_dir),
        ]
        print(f"\n[RUN] Synthetic: {script} -> {out_dir}")
        code, dur, out, err = run_cmd(cmd, project_root)
        learned = load_learned_params(Path(out_dir), script)
        losses = load_final_losses(Path(out_dir), script)
        truth = load_ground_truth(syn_dir, args.synthetic_sim_id)
        errors = compute_param_errors(learned, truth)
        results.append({
            'script': script,
            'mode': 'synthetic',
            'returncode': code,
            'duration_sec': dur,
            'learned_params': learned,
            'final_losses': losses,
            'ground_truth': truth,
            'abs_errors': errors,
            'stdout': out,
            'stderr': err,
        })

    # Real runs (optional)
    if real_available:
        for script in METHOD_SCRIPTS:
            script_path = project_root / 'training_scripts' / script
            method_name = script.replace('_training.py', '')
            out_dir_path = out_root / f'{method_name}_real'
            out_dir = str(out_dir_path)
            cmd = [
                sys.executable, str(script_path),
                *common,
                '--output-dir', out_dir,
                '--data-path', 'Data',
            ]
            print(f"\n[RUN] Real: {script} -> {out_dir}")
            code, dur, out, err = run_cmd(cmd, project_root)
            learned = load_learned_params(Path(out_dir), script)
            losses = load_final_losses(Path(out_dir), script)
            results.append({
                'script': script,
                'mode': 'real',
                'returncode': code,
                'duration_sec': dur,
                'learned_params': learned,
                'final_losses': losses,
                'ground_truth': {},
                'abs_errors': {},
                'stdout': out,
                'stderr': err,
            })
    else:
        print("\n[SKIP] Real data tests skipped (files not found or --skip-real provided).")

    # Summary
    print("\n===== SMOKE TEST SUMMARY =====")
    failures = 0
    for r in results:
        status = 'OK' if r['returncode'] == 0 else 'FAIL'
        if r['returncode'] != 0:
            failures += 1
        print(f"{r['mode']:<10} {r['script']:<35} -> {status} in {r['duration_sec']:.1f}s")
        # Parameters and errors
        if r.get('learned_params'):
            lp = r['learned_params']
            err = r.get('abs_errors', {})
            gt = r.get('ground_truth', {})
            def fmt(v):
                try:
                    return f"{float(v):.4g}"
                except Exception:
                    return str(v)
            keys = sorted(lp.keys())
            if keys:
                print("  learned:", ', '.join([f"{k}={fmt(lp[k])}" for k in keys]))
            if gt:
                keys_gt = sorted({k for k in gt.keys() if isinstance(gt[k], (int,float))})
                if keys_gt:
                    print("  ground :", ', '.join([f"{k}={fmt(gt[k])}" for k in keys_gt]))
            if err:
                keys_e = sorted(err.keys())
                if keys_e:
                    print("  |diff| :", ', '.join([f"{k}={fmt(err[k])}" for k in keys_e]))

        # Losses
        losses = r.get('final_losses', {}) or {}
        if losses:
            def fmt_loss(x):
                try:
                    return f"{float(x):.3e}"
                except Exception:
                    return str(x)
            # Show train totals and components
            parts = []
            for key in ['train_total','data_train','phys_train','val_total','data_val','phys_val']:
                if key in losses:
                    parts.append(f"{key}={fmt_loss(losses[key])}")
            if parts:
                print("  losses :", ', '.join(parts))

        # Save error excerpts to show at the end if failed
        if r['returncode'] != 0:
            # Print a short immediate hint
            if r.get('stderr'):
                snippet = r['stderr'][-400:]
                print("  stderr :", snippet.replace('\n', '\n           '))

    if failures:
        print(f"\nFailures: {failures}")
        print("\n===== ERROR DETAILS =====")
        for r in results:
            if r['returncode'] != 0:
                print(f"{r['mode']:<10} {r['script']:<35} -> EXIT {r['returncode']}")
                if r.get('stderr'):
                    print("stderr:\n" + r['stderr'])
                if r.get('stdout'):
                    print("stdout (tail):\n" + r['stdout'][-1000:])
        sys.exit(1)
    else:
        print("\nAll smoke tests passed.")


if __name__ == '__main__':
    main()


