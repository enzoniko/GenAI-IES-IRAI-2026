import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

from src.data.clean_mafaulda_processor import CleanMaFaulDaProcessor
import src.configs as cfg

def calculate_stats(data):
    """Compute basic statistics for comparison."""
    return {
        'max': np.max(np.abs(data)),
        'mean': np.mean(data),
        'std': np.std(data),
        'drift': data[-1] - data[0]
    }

def main():
    # Setup paths
    raw_dir = Path("/home/gust/machine-learn/GenAI-IES-IRAI-2026/data/raw-mafaulda")
    processed_dir = Path("/home/gust/machine-learn/GenAI-IES-IRAI-2026/data/processed-mafaulda/v1")
    
    # Try to find a sample file
    categories = ['normal', 'imbalance/10g', 'overhang/ball_fault/0g']
    sample_file = None
    for cat in categories:
        dir_path = raw_dir / cat
        if dir_path.exists():
            csvs = list(dir_path.glob("*.csv"))
            if csvs:
                sample_file = csvs[0]
                break
                
    if not sample_file:
        print(f"Error: No CSV files found in {raw_dir}")
        return
    
    print(f"Comparing strategies on: {sample_file}")
    
    # Load raw data
    print("Loading CSV...")
    df = pd.read_csv(sample_file, header=None)
    acc_raw = df.iloc[:, 2].values.astype(np.float32)
    acc_raw = acc_raw / 0.0102 
    print(f"Loaded {len(acc_raw)} samples.")
    
    # Process with FFT strategy
    print("Processing with FFT strategy...")
    proc_fft = CleanMaFaulDaProcessor(raw_dir, processed_dir, strategy='fft')
    a_fft, v_fft, p_fft = proc_fft.process_signal(acc_raw)
    print("FFT strategy complete.")
    
    # Process with Previous Strategy (Strategy D)
    print("Processing with Previous Strategy (Strategy D)...")
    proc_prev = CleanMaFaulDaProcessor(raw_dir, processed_dir, strategy='previous_strategy')
    a_prev, v_prev, p_prev = proc_prev.process_signal(acc_raw)
    print("Previous Strategy complete.")
    
    # ── Statistics Display ──────────────────────────────────────────────────
    stats_a_fft = calculate_stats(a_fft)
    stats_v_fft = calculate_stats(v_fft)
    stats_p_fft = calculate_stats(p_fft)
    
    stats_a_prev = calculate_stats(a_prev)
    stats_v_prev = calculate_stats(v_prev)
    stats_p_prev = calculate_stats(p_prev)
    
    print("\n--- STATISTICS COMPARISON ---")
    data = []
    for sig in ['Acceleration', 'Velocity', 'Position']:
        row = [sig]
        if sig == 'Acceleration': sf, sp = stats_a_fft, stats_a_prev
        elif sig == 'Velocity': sf, sp = stats_v_fft, stats_v_prev
        else: sf, sp = stats_p_fft, stats_p_prev
        
        print(f"\n{sig}:")
        print(f"  FFT Strategy      | Max: {sf['max']:.4e} | Drift: {sf['drift']:.4e} | Std: {sf['std']:.4e}")
        print(f"  Previous (D)      | Max: {sp['max']:.4e} | Drift: {sp['drift']:.4e} | Std: {sp['std']:.4e}")

    # ── Visualization (First 0.2 seconds for detail) ─────────────────────────
    # We plot the full signal and a zoomed window
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    t = np.arange(len(acc_raw)) / cfg.SAMPLING_RATE
    
    # Plot Acceleration
    axes[0].plot(t, a_fft, label='FFT (Pure Freq)', alpha=0.7)
    axes[0].plot(t, a_prev, label='Previous (Strategy D)', alpha=0.7, linestyle='--')
    axes[0].set_title("Acceleration Comparison")
    axes[0].set_ylabel("m/s^2")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot Velocity
    axes[1].plot(t, v_fft, label='FFT', alpha=0.7)
    axes[1].plot(t, v_prev, label='Previous', alpha=0.7, linestyle='--')
    axes[1].set_title("Velocity Comparison")
    axes[1].set_ylabel("m/s")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot Position
    axes[2].plot(t, p_fft, label='FFT', alpha=0.7)
    axes[2].plot(t, p_prev, label='Previous', alpha=0.7, linestyle='--')
    axes[2].set_title("Position Comparison (Displacement)")
    axes[2].set_ylabel("meters")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = "/home/gust/machine-learn/GenAI-IES-IRAI-2026/results/strategy_comparison.png"
    plt.savefig(plot_path)
    print(f"\nComparison plot saved to: {plot_path}")
    # Note: plt.show() might not work in headless environments, but png is saved.

if __name__ == "__main__":
    main()
