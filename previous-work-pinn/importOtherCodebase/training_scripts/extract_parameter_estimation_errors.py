#!/usr/bin/env python3
"""
Extract Average Absolute Parameter Estimation Errors for Synthetic Data

This script computes, for each method and each parameter, the average absolute error between the estimated and ground truth parameter values across all simulations. It also averages other metrics (losses, residuals) per simulation.

Outputs:
- parameter_estimation_errors.csv: Table of average absolute errors and metrics per method
- parameter_estimation_errors.png: Bar plot of average errors per parameter/method

Usage:
    python extract_parameter_estimation_errors.py --results-dir synthetic_bayesian_results --data-dir training_scripts/analytical_analysis/data --output-file parameter_estimation_errors.csv
"""
import os
import sys
import numpy as np
import pandas as pd
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt

PARAMS = ['M1', 'M2', 'M3', 'D1', 'D2', 'D3', 'K1', 'K2', 'E1']
METHODS = [
    'adaptive_lbpin', 'alpinn', 'constant_weight', 'brdr',
    'relobralo', 'pecann', 'gradnorm', 'dwpinn'
]


def load_ground_truth_params(metadata_path):
    with open(metadata_path, 'r') as f:
        meta = json.load(f)
    return meta['parameters']

def load_estimated_params(param_file):
    arr = np.load(param_file)
    return {k: float(arr[k]) for k in arr.files if k in PARAMS}

def find_param_file(sim_dir, method):
    # Try common locations
    candidates = list(Path(sim_dir).glob(f'{method}_parameters.npz'))
    if not candidates:
        # Try subdirs
        for sub in Path(sim_dir).rglob(f'{method}_parameters.npz'):
            candidates.append(sub)
    return candidates[0] if candidates else None

def find_history_file(sim_dir, method):
    candidates = list(Path(sim_dir).glob(f'{method}_history.npz'))
    if not candidates:
        for sub in Path(sim_dir).rglob(f'{method}_history.npz'):
            candidates.append(sub)
    return candidates[0] if candidates else None

def main():
    parser = argparse.ArgumentParser(description='Extract parameter estimation errors for synthetic data')
    parser.add_argument('--results-dir', type=str, required=True, help='Directory with optimization results')
    parser.add_argument('--data-dir', type=str, required=True, help='Directory with synthetic data (ground truth)')
    parser.add_argument('--output-file', type=str, default='parameter_estimation_errors.csv', help='CSV output file')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    data_dir = Path(args.data_dir)
    output_file = Path(args.output_file)

    # Find all simulation directories
    sim_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith('simulation_')])
    if not sim_dirs:
        print(f"No simulation directories found in {results_dir}")
        sys.exit(1)

    # For each method, collect errors and metrics
    summary = {m: {p: [] for p in PARAMS} for m in METHODS}
    metrics = {m: [] for m in METHODS}
    n_sims = 0
    for sim_dir in sim_dirs:
        sim_id = int(sim_dir.name.split('_')[-1])
        # Get ground truth
        gt_meta = data_dir / f'simulation_{sim_id:06d}' / 'metadata.json'
        if not gt_meta.exists():
            continue
        gt_params = load_ground_truth_params(gt_meta)
        n_sims += 1
        for method in METHODS:
            # Find estimated params
            method_dir = sim_dir / method / 'process_00' / 'results' / 'trial_000'
            param_file = find_param_file(method_dir, method)
            if not param_file or not param_file.exists():
                continue
            est_params = load_estimated_params(param_file)
            # Compute abs error for each param
            for p in PARAMS:
                if p in est_params and p in gt_params:
                    err = abs(est_params[p] - gt_params[p])
                    summary[method][p].append(err)
            # Also collect losses/metrics if available
            hist_file = find_history_file(method_dir, method)
            if hist_file and hist_file.exists():
                arr = np.load(hist_file)
                # Use final value for each metric
                row = {f'final_{k}': float(arr[k][-1]) for k in arr.files if arr[k].ndim == 1 and len(arr[k]) > 0}
                row['simulation_id'] = sim_id
                metrics[method].append(row)

    # Aggregate: average error per param per method
    rows = []
    for method in METHODS:
        row = {'method': method}
        for p in PARAMS:
            vals = summary[method][p]
            row[f'avg_abs_error_{p}'] = np.mean(vals) if vals else np.nan
        # Also average other metrics
        if metrics[method]:
            df = pd.DataFrame(metrics[method])
            for col in df.columns:
                if col != 'simulation_id':
                    row[f'avg_{col}'] = df[col].mean()
        rows.append(row)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_file, index=False)
    print(f"Saved parameter estimation errors to {output_file}")

    # Plot: bar plot of avg abs error per param per method
    fig, ax = plt.subplots(figsize=(12, 6))
    for p in PARAMS:
        vals = [r[f'avg_abs_error_{p}'] for r in rows]
        ax.bar([f'{m}' for m in METHODS], vals, label=p, alpha=0.7)
    ax.set_ylabel('Average Absolute Error')
    ax.set_title('Average Absolute Parameter Estimation Error per Method')
    ax.legend(PARAMS, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig_path = output_file.with_suffix('.png')
    plt.savefig(fig_path, dpi=200)
    print(f"Saved bar plot to {fig_path}")

if __name__ == '__main__':
    main() 