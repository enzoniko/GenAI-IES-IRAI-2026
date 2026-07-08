#!/usr/bin/env python3
"""
PINN Paper-Ready Spectrograms Generator

This script generates publication-ready synchrosqueezed wavelet spectrograms from PINN model residuals.
Creates 4 figures (one per PINN model) with optimized layout for paper inclusion.

Features:
- 4 figures: one for each PINN model (relobralo, constant_weight, brdr, pecann)
- Each figure: 8×3 subplot grid (8 residuals × 3 data classes)
- Uses only rotation speed closest to 16 Hz
- Space-optimized layout with minimal spacing
- Thin green separators between subplots
- Single shared colorbar per figure
- Uses Bayesian optimization models for consistency

Usage:
    python pinn_paper_spectrograms.py --output-dir paper_spectrograms
    python pinn_paper_spectrograms.py --model relobralo --output-dir custom_dir
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from scipy.stats import kurtosis

# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training_scripts.pinn_preprocessing import (
    get_rotation_speed_separated_features,
    load_pinn_data
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_condition_mapping() -> Dict[str, str]:
    """
    Map data file names to condition names used in the analysis.
    Uses the same 3 conditions as pinn_spectrogram_generator.py.

    Returns:
        Dictionary mapping condition keys to data file names (without .pth extension)
    """
    return {
        'normal': 'X_normal_v3',
        'overhang_ball_fault_35g': 'X_overhang_ball_fault_35g_v3',
        'vertical_misalignment_fault_1.90mm': 'X_vertical_misalignment_fault_1.90mm_v3',
    }


def get_condition_display_names() -> Dict[str, str]:
    """
    Get display names for conditions (abbreviated for space).

    Returns:
        Dictionary mapping condition keys to display names
    """
    return {
        'normal': 'Normal',
        'overhang_ball_fault_35g': 'OH Ball',
        'vertical_misalignment_fault_1.90mm': 'Vert. Mis.',
    }


def extract_rotation_speeds_from_pinn_data(data_path: str) -> np.ndarray:
    """
    Extract rotation speeds (omega) from processed PINN data.

    Args:
        data_path: Path to the X .pth file

    Returns:
        Array of rotation speeds in Hz, one per rotation speed condition
    """
    # Load the X data
    X_data, _ = load_pinn_data(data_path)

    # Extract omega values (index 8) and take the first value from each rotation speed
    # X_data shape: [n_rotation_speeds, n_timestamps_per_speed, 10]
    omega_rad_per_sec = X_data[:, 0, 8].numpy()  # Take first timestamp for each rotation speed

    # Convert from rad/s to Hz
    omega_hz = omega_rad_per_sec / (2 * np.pi)

    logger.debug(f"Extracted {len(omega_hz)} rotation speeds: {omega_hz}")

    return omega_hz


def find_closest_16hz_speed_index(rotation_speeds_hz: np.ndarray) -> int:
    """
    Find the index of the rotation speed closest to 16 Hz.

    Args:
        rotation_speeds_hz: Array of rotation speeds in Hz

    Returns:
        Index of the speed closest to 16 Hz
    """
    target_speed = 16.0
    differences = np.abs(rotation_speeds_hz - target_speed)
    closest_idx = np.argmin(differences)
    
    logger.info(f"Target: {target_speed} Hz, Found: {rotation_speeds_hz[closest_idx]:.2f} Hz (index {closest_idx})")
    
    return closest_idx


def truncate_to_first_0_05_seconds(data_path: str, max_samples: int = 2500) -> str:
    """
    Truncate data files to first 0.05 seconds (2500 samples at 50kHz).

    Args:
        data_path: Path to the original .pth file
        max_samples: Maximum number of samples to keep (default: 2500 for 0.05s at 50kHz)

    Returns:
        Path to the truncated data file
    """
    # Create truncated data paths
    base_name = os.path.basename(data_path)
    dir_name = os.path.dirname(data_path)

    if 'X_' in base_name:
        x_truncated_path = os.path.join(dir_name, base_name.replace('.pth', f'_truncated_{max_samples}.pth'))
        y_base_name = base_name.replace('X_', 'Y_')
        y_truncated_path = os.path.join(dir_name, y_base_name.replace('.pth', f'_truncated_{max_samples}.pth'))
    else:
        # Fallback for files that don't follow X_/Y_ naming
        x_truncated_path = os.path.join(dir_name, base_name.replace('.pth', f'_truncated_{max_samples}.pth'))
        y_truncated_path = x_truncated_path.replace('X_', 'Y_')

    if os.path.exists(x_truncated_path) and os.path.exists(y_truncated_path):
        logger.info(f"Using existing truncated files: {x_truncated_path}")
        return x_truncated_path

    logger.info(f"Truncating {data_path} to first {max_samples} samples")

    # Load original data
    X_full, y_full = load_pinn_data(data_path)

    # Truncate along the time dimension (axis 1)
    X_truncated = X_full[:, :max_samples, :]
    y_truncated = y_full[:, :max_samples, :]

    logger.info(f"Original shapes: X={X_full.shape}, y={y_full.shape}")
    logger.info(f"Truncated shapes: X={X_truncated.shape}, y={y_truncated.shape}")

    # Save truncated data
    torch.save(X_truncated, x_truncated_path)
    torch.save(y_truncated, y_truncated_path)

    return x_truncated_path


def extract_16hz_residual_data(model_name: str, condition_key: str, data_file: str,
                              data_dir: str = "Data/v3", chunk_size_mb: int = 50,
                              max_samples: int = 2500, use_data_driven: bool = False) -> Optional[np.ndarray]:
    """
    Extract residuals for the rotation speed closest to 16 Hz.

    Args:
        model_name: Name of the PINN model (ignored if use_data_driven=True)
        condition_key: Condition identifier
        data_file: Data file name without extension
        data_dir: Directory containing the data files
        chunk_size_mb: Chunk size for memory-efficient processing
        max_samples: Maximum number of samples to use
        use_data_driven: If True, use data-driven model instead of PINN

    Returns:
        Array of shape [n_timestamps, 8] containing residuals for 16Hz speed, or None if failed
    """
    data_path = os.path.join(data_dir, f"{data_file}.pth")

    if not os.path.exists(data_path):
        logger.warning(f"Data file not found: {data_path}")
        return None

    logger.info(f"Processing condition: {condition_key} with model: {model_name}")

    try:
        # Truncate data to first 0.05 seconds for faster processing
        truncated_data_path = truncate_to_first_0_05_seconds(data_path, max_samples)

        # Extract residuals using rotation speed separated features
        if use_data_driven:
            # Use data-driven preprocessing
            from training_scripts.data_driven_preprocessing import get_rotation_speed_separated_features as get_data_driven_features
            pinn_residuals = get_data_driven_features(
                data_path=truncated_data_path,
                chunk_size_mb=chunk_size_mb
            )
            logger.info(f"Extracted data-driven residuals with shape: {pinn_residuals.shape}")
        else:
            # Use PINN preprocessing
            pinn_residuals = get_rotation_speed_separated_features(
                data_path=truncated_data_path,
                model_name=model_name,
                chunk_size_mb=chunk_size_mb,
                use_bayesian_models=True
            )
            logger.info(f"Extracted PINN residuals with shape: {pinn_residuals.shape}")

        # Extract actual rotation speeds from the data
        rotation_speeds_hz = extract_rotation_speeds_from_pinn_data(truncated_data_path)

        # Find the rotation speed closest to 16 Hz
        closest_16hz_idx = find_closest_16hz_speed_index(rotation_speeds_hz)

        # Extract data for this specific rotation speed
        residuals_16hz = pinn_residuals[closest_16hz_idx]  # Shape: [n_timestamps, 8]

        logger.info(f"Extracted residuals for 16Hz speed with shape: {residuals_16hz.shape}")

        return residuals_16hz

    except Exception as e:
        logger.error(f"Failed to process condition {condition_key}: {e}")
        return None


def calculate_global_colorbar_limits_paper(all_residual_data: Dict[str, np.ndarray],
                                          sampling_rate: int = 50000) -> Tuple[float, float]:
    """
    Calculate global colorbar limits across all conditions for consistent scaling.

    Args:
        all_residual_data: Dictionary mapping condition keys to residual arrays
        sampling_rate: Sampling rate in Hz

    Returns:
        Tuple of (global_vmin, global_vmax) for colorbar limits
    """
    import ssqueezepy as ssq

    logger.info("Calculating global colorbar limits for paper spectrograms...")

    all_values = []
    valid_spectrograms_found = False

    # Process each condition and residual to get global limits
    for condition_key, residual_data in all_residual_data.items():
        logger.debug(f"Processing condition {condition_key} for global limits")

        if residual_data is not None and residual_data.shape[1] >= 8:
            # Sample from first 2 residuals for efficiency
            for res_idx in range(2):
                data_seg = residual_data[:, res_idx]

                if len(data_seg) >= 100:  # Ensure segment is long enough
                    try:
                        Tx, _, _, _ = ssq.ssq_cwt(data_seg, fs=sampling_rate, wavelet='morlet')
                        Tx_abs = np.abs(Tx)
                        Tx_abs[Tx_abs == 0] = np.finfo(float).eps

                        # Normalize the spectrogram
                        Tx_abs = (Tx_abs - np.min(Tx_abs)) / (np.max(Tx_abs) - np.min(Tx_abs) + 1e-12)

                        all_values.append(Tx_abs.ravel())
                        valid_spectrograms_found = True
                    except Exception as e:
                        logger.warning(f"Failed to compute spectrogram for {condition_key}, residual {res_idx}: {e}")
                        continue

    if valid_spectrograms_found and all_values:
        all_values = np.concatenate(all_values)
        # Use percentiles for robust limits
        global_vmin = np.percentile(all_values, 5)
        global_vmax = np.percentile(all_values, 95)

        # Ensure valid range
        if global_vmax <= global_vmin:
            global_vmax = global_vmin + 1e-10

        logger.info(f"Global colorbar limits: vmin={global_vmin:.6f}, vmax={global_vmax:.6f}")
        return global_vmin, global_vmax
    else:
        logger.warning("No valid spectrograms found for global colorbar limits, using defaults")
        return 0.0, 1.0


def calculate_spectrogram_metrics(Tx_abs: np.ndarray, ssq_freqs: np.ndarray,
                                time_axis: np.ndarray) -> Dict[str, float]:
    """
    Calculate various metrics from a spectrogram.

    Args:
        Tx_abs: Absolute values of the spectrogram coefficients
        ssq_freqs: Frequency bins
        time_axis: Time axis

    Returns:
        Dictionary containing various spectrogram metrics
    """
    metrics = {}

    # Basic energy metrics
    metrics['peak_energy'] = float(np.max(Tx_abs))
    metrics['rms_energy'] = float(np.sqrt(np.mean(Tx_abs**2)))
    metrics['total_energy'] = float(np.sum(Tx_abs))

    # Spectral centroid (center of mass of the spectrum)
    # Average frequency weighted by energy
    freq_weights = np.sum(Tx_abs, axis=1)  # Sum across time for each frequency
    if np.sum(freq_weights) > 0:
        metrics['spectral_centroid'] = float(np.sum(ssq_freqs * freq_weights) / np.sum(freq_weights))
    else:
        metrics['spectral_centroid'] = 0.0

    # Spectral bandwidth (spread of the spectrum)
    if np.sum(freq_weights) > 0:
        centroid = metrics['spectral_centroid']
        variance = np.sum(freq_weights * (ssq_freqs - centroid)**2) / np.sum(freq_weights)
        metrics['spectral_bandwidth'] = float(np.sqrt(variance))
    else:
        metrics['spectral_bandwidth'] = 0.0

    # Peak frequency (frequency with maximum energy)
    peak_freq_idx = np.argmax(np.sum(Tx_abs, axis=1))
    metrics['peak_frequency'] = float(ssq_freqs[peak_freq_idx])

    # Spectral kurtosis (measure of "tailedness")
    # Flatten the spectrogram and compute kurtosis
    flat_spectrogram = Tx_abs.flatten()
    if len(flat_spectrogram) > 0 and np.std(flat_spectrogram) > 0:
        metrics['spectral_kurtosis'] = float(kurtosis(flat_spectrogram))
    else:
        metrics['spectral_kurtosis'] = 0.0

    # Spectral flatness (measure of how flat/noisy the spectrum is)
    # Geometric mean / arithmetic mean
    if np.all(Tx_abs > 0):
        geometric_mean = np.exp(np.mean(np.log(Tx_abs[Tx_abs > 0])))
        arithmetic_mean = np.mean(Tx_abs)
        if arithmetic_mean > 0:
            metrics['spectral_flatness'] = float(geometric_mean / arithmetic_mean)
        else:
            metrics['spectral_flatness'] = 0.0
    else:
        metrics['spectral_flatness'] = 0.0

    # Spectral entropy (measure of spectral disorder)
    # Normalize the spectrogram
    normalized_spectrogram = Tx_abs / np.sum(Tx_abs) if np.sum(Tx_abs) > 0 else Tx_abs
    normalized_spectrogram = normalized_spectrogram[normalized_spectrogram > 0]
    if len(normalized_spectrogram) > 0:
        metrics['spectral_entropy'] = float(-np.sum(normalized_spectrogram * np.log2(normalized_spectrogram)))
    else:
        metrics['spectral_entropy'] = 0.0

    return metrics


def print_spectrogram_metrics_paper(model_name: str, all_metrics: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    """
    Print spectrogram metrics in a formatted way for paper spectrograms.

    Args:
        model_name: Name of the PINN model
        all_metrics: Nested dictionary with metrics [residual][condition][metric_name]
    """
    print(f"\n{'='*80}")
    print(f"SPECTROGRAM METRICS - {model_name.upper()} Model (16 Hz)")
    print(f"{'='*80}")

    residual_names = ['data_res1', 'data_res2', 'data_res3', 'data_res4',
                     'phys_res1', 'phys_res2', 'phys_res3', 'phys_res4']

    condition_names = ['normal', 'overhang_ball_fault_35g', 'vertical_misalignment_fault_1.90mm']
    condition_display = {
        'normal': 'Normal',
        'overhang_ball_fault_35g': 'OH Ball',
        'vertical_misalignment_fault_1.90mm': 'Vert. Mis.'
    }

    for residual in residual_names:
        if residual in all_metrics:
            print(f"\n{residual.upper()}:")
            print("-" * 40)

            # Print header
            header = f"{'Condition':<15} {'Peak Energy':<12} {'RMS Energy':<12} {'Total Energy':<13} {'Centroid':<10} {'Bandwidth':<11} {'Peak Freq':<11} {'Kurtosis':<10} {'Flatness':<10} {'Entropy':<10}"
            print(header)
            print("-" * len(header))

            for condition in condition_names:
                if condition in all_metrics[residual]:
                    metrics = all_metrics[residual][condition]
                    cond_display = condition_display.get(condition, condition)
                    line = (f"{cond_display:<15} "
                           f"{metrics.get('peak_energy', 0):<12.4f} "
                           f"{metrics.get('rms_energy', 0):<12.4f} "
                           f"{metrics.get('total_energy', 0):<13.2e} "
                           f"{metrics.get('spectral_centroid', 0):<10.1f} "
                           f"{metrics.get('spectral_bandwidth', 0):<11.1f} "
                           f"{metrics.get('peak_frequency', 0):<11.1f} "
                           f"{metrics.get('spectral_kurtosis', 0):<10.2f} "
                           f"{metrics.get('spectral_flatness', 0):<10.4f} "
                           f"{metrics.get('spectral_entropy', 0):<10.2f}")
                    print(line)

    print(f"\n{'='*80}\n")


def generate_paper_spectrogram_figure(model_name: str, all_residual_data: Dict[str, np.ndarray],
                                     output_path: str, sampling_rate: int = 50000,
                                     global_vmin: float = None, global_vmax: float = None) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Generate a paper-ready spectrogram figure for a single PINN or data-driven model.

    Args:
        model_name: Name of the PINN model or 'data_driven' for data-driven model
        all_residual_data: Dictionary mapping condition keys to residual arrays
        output_path: Path to save the figure
        sampling_rate: Sampling rate in Hz
        global_vmin: Global minimum value for colorbar
        global_vmax: Global maximum value for colorbar

    Returns:
        Dictionary containing spectrogram metrics for each residual and condition
    """
    import ssqueezepy as ssq

    # Define residual and condition order
    residual_names = ['data_res1', 'data_res2', 'data_res3', 'data_res4',
                     'phys_res1', 'phys_res2', 'phys_res3', 'phys_res4']

    condition_order = ['normal', 'overhang_ball_fault_35g', 'vertical_misalignment_fault_1.90mm']
    condition_display_names = get_condition_display_names()

    # Initialize metrics dictionary
    all_metrics = {residual: {} for residual in residual_names}

    # Use provided global colorbar limits if available
    if global_vmin is None or global_vmax is None:
        # Fallback: calculate limits for this model only
        global_vmin, global_vmax = calculate_global_colorbar_limits_paper(all_residual_data, sampling_rate)

    # Create figure with optimized layout for paper
    fig = plt.figure(figsize=(10, 20))  # Tall figure for 8 rows
    
    # Create GridSpec with minimal spacing (no colorbar needed)
    gs = GridSpec(8, 3, figure=fig, 
                  hspace=0.08,  # Slightly more vertical spacing for labels
                  wspace=0.05,  # Slightly more horizontal spacing for labels
                  left=0.10, right=0.98, top=0.95, bottom=0.08)  # More space for labels

    # Create subplots
    axes = []
    for row in range(8):
        row_axes = []
        for col in range(3):
            ax = fig.add_subplot(gs[row, col])
            row_axes.append(ax)
        axes.append(row_axes)

    # Plot spectrograms
    for res_idx, residual_name in enumerate(residual_names):
        for cond_idx, condition_key in enumerate(condition_order):
            ax = axes[res_idx][cond_idx]
            
            if condition_key in all_residual_data and all_residual_data[condition_key] is not None:
                residual_data = all_residual_data[condition_key]
                
                if residual_data.shape[1] > res_idx:
                    data_seg = residual_data[:, res_idx]
                    
                    if len(data_seg) >= 100:
                        try:
                            # Compute synchrosqueezed transform
                            Tx, _, ssq_freqs, _ = ssq.ssq_cwt(data_seg, fs=sampling_rate, wavelet='morlet')
                            Tx_abs = np.abs(Tx)
                            Tx_abs[Tx_abs == 0] = np.finfo(float).eps

                            # Calculate metrics BEFORE normalization for more meaningful results
                            time_axis = np.linspace(0, len(data_seg) / sampling_rate, len(data_seg))
                            metrics = calculate_spectrogram_metrics(Tx_abs, ssq_freqs, time_axis)
                            all_metrics[residual_name][condition_key] = metrics

                            # Normalize the spectrogram
                            Tx_abs = (Tx_abs - np.min(Tx_abs)) / (np.max(Tx_abs) - np.min(Tx_abs) + 1e-12)

                            im = ax.imshow(Tx_abs, extent=[time_axis[0], time_axis[-1],
                                                         ssq_freqs[-1], ssq_freqs[0]],
                                         aspect='auto', cmap='magma', vmin=global_vmin, vmax=global_vmax)

                            ax.set_ylim(ssq_freqs[-1], ssq_freqs[0])

                        except Exception as e:
                            ax.text(0.5, 0.5, f'Error', ha='center', va='center',
                                   transform=ax.transAxes, fontsize=20, color='red')
                    else:
                        ax.text(0.5, 0.5, 'Insufficient\ndata', ha='center', va='center',
                               transform=ax.transAxes, fontsize=20)
                else:
                    ax.text(0.5, 0.5, 'No residual', ha='center', va='center',
                           transform=ax.transAxes, fontsize=20)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=20)

            # Configure axes based on position in grid
            
            # Y-axis labels: Only on leftmost column (first column)
            if cond_idx == 0:
                # Clean residual name (remove underscores and capitalize)
                clean_residual_name = residual_name.replace('_', ' ').replace('res', 'Res')

                # Only top row gets frequency unit
                if res_idx == 0:
                    ax.set_ylabel(f'{clean_residual_name} [kHz]', fontsize=20)
                else:
                    ax.set_ylabel(clean_residual_name, fontsize=20)

                # Convert y-axis ticks to kHz and remove .0
                # Reduce number of ticks to avoid overlapping
                yticks = ax.get_yticks()
                if len(yticks) > 3:
                    step = max(1, len(yticks) // 3)  # Show at most 3 ticks
                    yticks = yticks[::step]
                    ax.set_yticks(yticks)
                ax.set_yticklabels([f'{tick/1000:.1f}'.rstrip('0').rstrip('.') for tick in yticks], fontsize=18)
            else:
                ax.set_yticklabels([])
                ax.set_ylabel('')

            # X-axis labels: Only on bottom row (last row)
            if res_idx == len(residual_names) - 1:
                ax.set_xlabel('Time [×10⁻² s]', fontsize=20)
                # Convert x-axis ticks to 1e-2 seconds and remove .0
                # Reduce number of ticks to avoid overlapping
                xticks = ax.get_xticks()
                # Keep only every other tick or fewer ticks to prevent overlap
                if len(xticks) > 3:
                    step = max(1, len(xticks) // 3)  # Show at most 3 ticks
                    xticks = xticks[::step]
                    ax.set_xticks(xticks)
                ax.set_xticklabels([f'{tick*100:.1f}'.rstrip('0').rstrip('.') for tick in xticks], fontsize=18)
            else:
                ax.set_xticklabels([])
                ax.set_xlabel('')

            # Column titles: Only on top row
            if res_idx == 0:
                condition_display = condition_display_names.get(condition_key, condition_key)
                ax.set_title(condition_display, fontsize=20, pad=10)

            # Add thin green separators using plot instead of axvline/axhline to avoid transform issues
            # Vertical separators between columns (except after last column)
            if cond_idx < len(condition_order) - 1:
                # Draw line at right edge of current subplot
                ax.plot([1, 1], [0, 1], color='green', linewidth=0.5, transform=ax.transAxes, clip_on=False)
            
            # Horizontal separators between rows (except after last row)
            if res_idx < len(residual_names) - 1:
                # Draw line at bottom edge of current subplot
                ax.plot([0, 1], [0, 0], color='green', linewidth=0.5, transform=ax.transAxes, clip_on=False)

    # No colorbar needed - consistent scale across all figures for comparison

    # Set main title
    if model_name == 'data_driven':
        title_text = 'Data-Driven Model Residuals Spectrograms (16 Hz)'
    else:
        title_text = f'PINN Residuals Spectrograms - {model_name.upper()} Model (16 Hz)'
    fig.suptitle(title_text, fontsize=22, y=0.98, fontweight='bold')

    # Save with high DPI for paper quality
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    logger.info(f"Paper-ready spectrogram saved to: {output_path}")

    return all_metrics


def generate_all_paper_spectrograms(models: List[str] = None, output_dir: str = "paper_spectrograms",
                                   data_dir: str = "Data/v3", chunk_size_mb: int = 50,
                                   max_samples: int = 2500, use_data_driven: bool = False) -> None:
    """
    Generate paper-ready spectrograms for all specified models.

    Args:
        models: List of model names to process (default: all available models)
        output_dir: Output directory for the figures
        data_dir: Directory containing the data files
        chunk_size_mb: Chunk size for memory-efficient processing
        max_samples: Maximum number of samples to use
        use_data_driven: If True, use data-driven model instead of PINN
    """
    if models is None:
        if use_data_driven:
            models = ['data_driven']
        else:
            models = ['relobralo', 'constant_weight', 'brdr', 'pecann']

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    condition_mapping = get_condition_mapping()

    # First pass: calculate global colorbar limits across ALL models and conditions
    logger.info("Calculating global colorbar limits across all models...")
    all_models_residual_data = {}

    for model_name in models:
        logger.info(f"Extracting data for global limits calculation: {model_name}")
        model_residual_data = {}

        for condition_key, data_file in condition_mapping.items():
            residual_data = extract_16hz_residual_data(
                model_name=model_name,
                condition_key=condition_key,
                data_file=data_file,
                data_dir=data_dir,
                chunk_size_mb=chunk_size_mb,
                max_samples=max_samples,
                use_data_driven=use_data_driven
            )

            if residual_data is not None:
                model_residual_data[f"{model_name}_{condition_key}"] = residual_data

        all_models_residual_data.update(model_residual_data)

    # Calculate global limits across ALL models and conditions
    global_vmin, global_vmax = calculate_global_colorbar_limits_paper(all_models_residual_data)
    logger.info(f"Global colorbar limits for all figures: vmin={global_vmin:.6f}, vmax={global_vmax:.6f}")

    # Second pass: generate figures with consistent colorbar
    for model_name in models:
        logger.info(f"Generating paper spectrograms for model: {model_name}")

        # Extract 16Hz residual data for all conditions
        all_residual_data = {}

        for condition_key, data_file in condition_mapping.items():
            residual_data = extract_16hz_residual_data(
                model_name=model_name,
                condition_key=condition_key,
                data_file=data_file,
                data_dir=data_dir,
                chunk_size_mb=chunk_size_mb,
                max_samples=max_samples,
                use_data_driven=use_data_driven
            )
            
            all_residual_data[condition_key] = residual_data

        # Check if we have any valid data
        valid_conditions = [k for k, v in all_residual_data.items() if v is not None]
        if not valid_conditions:
            logger.error(f"No valid data found for model {model_name}")
            continue

        logger.info(f"Valid conditions for {model_name}: {valid_conditions}")

        # Handle data-driven case differently (only generate one figure)
        if use_data_driven and model_name == 'data_driven':
            # Generate the paper figure
            output_path = os.path.join(output_dir, "paper_spectrogram_data_driven.png")

            try:
                logger.info(f"Generating figure for data-driven model at {output_path}")
                model_metrics = generate_paper_spectrogram_figure(
                    model_name='data_driven',
                    all_residual_data=all_residual_data,
                    output_path=output_path,
                    global_vmin=global_vmin,
                    global_vmax=global_vmax
                )
                logger.info(f"Successfully generated figure for data-driven model")

                # Print metrics for this model
                print_spectrogram_metrics_paper('data_driven', model_metrics)
                break  # Only process once for data-driven
            except Exception as e:
                logger.error(f"Failed to generate paper spectrogram for data-driven model: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue
        else:
            # Generate the paper figure for PINN models
            output_path = os.path.join(output_dir, f"paper_spectrogram_{model_name}.png")

            try:
                logger.info(f"Generating figure for {model_name} at {output_path}")
                model_metrics = generate_paper_spectrogram_figure(
                    model_name=model_name,
                    all_residual_data=all_residual_data,
                    output_path=output_path,
                    global_vmin=global_vmin,
                    global_vmax=global_vmax
                )
                logger.info(f"Successfully generated figure for {model_name}")

                # Print metrics for this model
                print_spectrogram_metrics_paper(model_name, model_metrics)
            except Exception as e:
                logger.error(f"Failed to generate paper spectrogram for {model_name}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue

    logger.info("Paper-ready spectrogram generation completed!")


def main():
    """Main function for paper-ready PINN spectrogram generation."""
    parser = argparse.ArgumentParser(description="Generate paper-ready spectrograms from PINN model residuals")

    parser.add_argument("--model", type=str, help="PINN model name (relobralo, constant_weight, brdr, pecann)")
    parser.add_argument("--all-models", action="store_true", help="Process all available PINN models", default=True)
    parser.add_argument("--output-dir", type=str, default="paper_spectrograms", help="Output directory for spectrograms")
    parser.add_argument("--data-dir", type=str, default="Data/v3", help="Directory containing data files")
    parser.add_argument("--chunk-size-mb", type=int, default=50, help="Chunk size for memory-efficient processing")
    parser.add_argument("--max-samples", type=int, default=2500, help="Maximum samples to use (2500 for 0.05s at 50kHz)")
    parser.add_argument("--data-driven", action="store_true", help="Use data-driven model instead of PINN")

    args = parser.parse_args()

    # Define available models
    available_models = ['relobralo', 'constant_weight', 'brdr', 'pecann']

    # Determine which models to process
    # Data-driven flag takes precedence over everything
    if args.data_driven:
        models_to_process = ['data_driven']
        use_data_driven = True
        logger.info("Data-driven flag detected - processing only data-driven model")
    elif args.model:
        # Specific PINN model requested
        if args.model not in available_models:
            logger.error(f"Unknown model: {args.model}. Available models: {available_models}")
            return
        models_to_process = [args.model]
        use_data_driven = False
        logger.info(f"Specific model requested: {args.model}")
    else:
        # Default to all PINN models
        models_to_process = available_models
        use_data_driven = False
        logger.info("No specific model or data-driven flag - processing all PINN models")

    logger.info(f"Models to process: {models_to_process}")
    logger.info(f"Use data-driven: {use_data_driven}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Using data from rotation speed closest to 16 Hz")
    logger.info(f"Using first {args.max_samples} samples (≈{args.max_samples/50000:.3f}s at 50kHz)")

    # Generate paper spectrograms
    generate_all_paper_spectrograms(
        models=models_to_process,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        chunk_size_mb=args.chunk_size_mb,
        max_samples=args.max_samples,
        use_data_driven=use_data_driven
    )


if __name__ == "__main__":
    main()
