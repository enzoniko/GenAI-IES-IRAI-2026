#!/usr/bin/env python3
"""
PINN Residuals Spectrogram Generator (Optimized)

This script generates synchrosqueezed wavelet spectrograms from PINN model residuals
using an optimized approach for faster processing and focused analysis.

Uses Bayesian optimization models for consistency with original optimization results.

Optimized Features:
- Processes only 3 specific conditions: normal, overhang_ball_fault_35g, vertical_misalignment_fault_1.90mm
- Uses first 0.05 seconds of data (2500 samples at 50kHz) for faster processing
- Creates single plot per condition with residuals as rows and rotation speeds as columns
- Memory-efficient processing with smaller chunk sizes
- **Shared colorbar across all plots** for consistent scaling and comparison
- Custom spectrogram layout optimized for PINN residual analysis

Usage:
    python pinn_spectrogram_generator.py --model relobralo
    python pinn_spectrogram_generator.py --all-models --output-dir optimized_pinn_spectrograms
"""

import os
import argparse
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from scipy.stats import kurtosis

# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training_scripts.pinn_preprocessing import (
    get_rotation_speed_separated_features,
    preprocess_pinn_data,
    load_pinn_data
)

from common.visualization.wavelet_spectrograms import generate_spectrograms

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_condition_mapping() -> Dict[str, str]:
    """
    Map data file names to condition names used in the analysis.
    Limited to the 3 specific conditions requested for faster processing.

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
    Get display names for conditions.

    Returns:
        Dictionary mapping condition keys to display names
    """
    return {
        'normal': 'Normal',
        'overhang_ball_fault_35g': 'Overhang Ball (35g)',
        'vertical_misalignment_fault_1.90mm': 'Vertical Misalignment (1.90mm)',
    }


def extract_rotation_speeds_from_pinn_data(data_path: str) -> np.ndarray:
    """
    Extract rotation speeds (omega) from processed PINN data.

    From LoadDatav3.py, we know that:
    - X tensor has 10 features: 4 velocities, 4 positions, 1 omega, 1 time
    - Omega is stored as the 9th feature (index 8) in rad/s
    - We convert to Hz for display: Hz = rad/s / (2π)

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


def convert_pinn_residuals_to_spectrogram_format(pinn_residuals: np.ndarray,
                                                rotation_speeds_hz: np.ndarray,
                                                seq_length: int = 100,
                                                max_columns: int = 10) -> Dict[str, List[np.ndarray]]:
    """
    Convert PINN residuals from [n_rotation_speeds, n_timestamps_per_speed, 8] format
    to the spectrogram-expected format.

    The spectrogram functions expect a dictionary where each key represents a "condition"
    and the value is either a tensor or list of segments. Since PINN residuals are already
    separated by rotation speed, we'll treat each rotation speed as a separate condition.

    Args:
        pinn_residuals: PINN residuals in shape [n_rotation_speeds, n_timestamps_per_speed, 8]
        rotation_speeds_hz: Array of rotation speeds in Hz, one per rotation speed condition
        seq_length: Length of each segment for spectrogram processing
        max_columns: Maximum number of rotation speeds to include (to avoid too many columns)

    Returns:
        Dictionary with rotation speed keys containing lists of segments
    """
    n_rotation_speeds, n_timestamps_per_speed, n_features = pinn_residuals.shape

    # Limit the number of columns to avoid overcrowded plots
    n_speeds_to_use = min(n_rotation_speeds, max_columns)

    # Sort by rotation speed and select the most representative ones
    speed_indices = np.argsort(rotation_speeds_hz)[:n_speeds_to_use]

    spectrogram_data = {}

    for speed_idx in speed_indices:
        # Extract data for this rotation speed: [n_timestamps_per_speed, 8]
        speed_data = pinn_residuals[speed_idx]
        speed_hz = rotation_speeds_hz[speed_idx]

        # Create segments for this rotation speed
        segments = []
        for start_idx in range(0, n_timestamps_per_speed - seq_length + 1, seq_length):
            end_idx = start_idx + seq_length
            segment = speed_data[start_idx:end_idx]  # Shape: [seq_length, 8]
            segments.append(segment)

        # Use actual rotation speed in Hz as condition name
        condition_name = f"{speed_hz:.1f}Hz"
        spectrogram_data[condition_name] = segments

        logger.debug(f"Created {len(segments)} segments for rotation speed {speed_hz:.1f} Hz")

    logger.info(f"Using {len(spectrogram_data)} rotation speeds out of {n_rotation_speeds} available")

    return spectrogram_data


def save_pinn_residuals_as_torch_dict(pinn_residuals_dict: Dict[str, np.ndarray],
                                     output_path: str) -> None:
    """
    Save PINN residuals in the format expected by the spectrogram functions.

    Args:
        pinn_residuals_dict: Dictionary mapping conditions to PINN residual arrays
        output_path: Path to save the torch dictionary
    """
    # Convert to torch tensors and save
    torch_dict = {}
    for condition, residuals_array in pinn_residuals_dict.items():
        torch_dict[condition] = torch.from_numpy(residuals_array).float()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(torch_dict, output_path)
    logger.info(f"Saved PINN residuals dictionary to {output_path}")


def process_single_condition(model_name: str, condition_key: str, data_file: str,
                           data_dir: str = "Data/v3", chunk_size_mb: int = 50,
                           max_samples: int = 2500) -> Optional[Dict[str, np.ndarray]]:
    """
    Process a single condition to extract PINN residuals.

    Args:
        model_name: Name of the PINN model
        condition_key: Condition identifier (e.g., 'normal', 'imbalance')
        data_file: Data file name without extension
        data_dir: Directory containing the data files
        chunk_size_mb: Chunk size for memory-efficient processing
        max_samples: Maximum number of samples to use (default: 2500 for 0.05s at 50kHz)

    Returns:
        Dictionary mapping residual names to arrays, or None if processing failed
    """
    data_path = os.path.join(data_dir, f"{data_file}.pth")

    if not os.path.exists(data_path):
        logger.warning(f"Data file not found: {data_path}")
        return None

    logger.info(f"Processing condition: {condition_key} with model: {model_name}")

    try:
        # Truncate data to first 0.05 seconds for faster processing
        truncated_data_path = truncate_to_first_0_05_seconds(data_path, max_samples)

        # Extract PINN residuals using rotation speed separated features
        pinn_residuals = get_rotation_speed_separated_features(
            data_path=truncated_data_path,
            model_name=model_name,
            chunk_size_mb=chunk_size_mb,
            use_bayesian_models=True
        )

        logger.info(f"Extracted PINN residuals with shape: {pinn_residuals.shape}")

        # Extract actual rotation speeds from the data
        rotation_speeds_hz = extract_rotation_speeds_from_pinn_data(truncated_data_path)

        # Convert to dictionary format for spectrograms
        spectrogram_format = convert_pinn_residuals_to_spectrogram_format(
            pinn_residuals, rotation_speeds_hz, max_columns=8  # Limit to 8 columns for readability
        )

        return spectrogram_format

    except Exception as e:
        logger.error(f"Failed to process condition {condition_key}: {e}")
        return None


def process_all_conditions_for_model(model_name: str, conditions: List[str] = None,
                                   data_dir: str = "Data/v3", chunk_size_mb: int = 50,
                                   max_samples: int = 2500) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Process all conditions for a given PINN model.

    Args:
        model_name: Name of the PINN model
        conditions: List of condition keys to process (default: all available)
        data_dir: Directory containing the data files
        chunk_size_mb: Chunk size for memory-efficient processing
        max_samples: Maximum number of samples to use (default: 2500 for 0.05s at 50kHz)

    Returns:
        Dictionary mapping condition keys to their spectrogram-formatted residuals
    """
    condition_mapping = get_condition_mapping()

    if conditions is None:
        conditions = list(condition_mapping.keys())

    all_condition_data = {}

    for condition_key in conditions:
        if condition_key in condition_mapping:
            data_file = condition_mapping[condition_key]

            condition_data = process_single_condition(
                model_name=model_name,
                condition_key=condition_key,
                data_file=data_file,
                data_dir=data_dir,
                chunk_size_mb=chunk_size_mb,
                max_samples=max_samples
            )

            if condition_data is not None:
                all_condition_data[condition_key] = condition_data
                logger.info(f"Successfully processed condition: {condition_key}")
            else:
                logger.warning(f"Skipped condition: {condition_key}")
        else:
            logger.warning(f"Unknown condition: {condition_key}")

    return all_condition_data


def calculate_global_colorbar_limits(all_condition_data: Dict[str, Dict[str, np.ndarray]],
                                     sampling_rate: int = 50000, normalize_residuals: bool = False,
                                     normalize_spectogram: bool = True) -> Tuple[float, float]:
    """
    Calculate global colorbar limits across all conditions for consistent scaling.

    Args:
        all_condition_data: Dictionary mapping condition keys to their spectrogram data
        sampling_rate: Sampling rate in Hz
        normalize_residuals: Whether to normalize residuals
        normalize_spectogram: Whether to normalize spectrograms

    Returns:
        Tuple of (global_vmin, global_vmax) for colorbar limits
    """
    import ssqueezepy as ssq

    logger.info("Calculating global colorbar limits across all conditions...")

    all_values = []
    valid_segments_found = False

    residual_names = ['data_res1', 'data_res2', 'data_res3', 'data_res4',
                     'phys_res1', 'phys_res2', 'phys_res3', 'phys_res4']

    # Process each condition
    for condition_key, condition_data in all_condition_data.items():
        logger.debug(f"Processing condition {condition_key} for global limits")

        # Sort rotation speeds numerically
        def extract_speed_value(speed_key):
            try:
                return float(speed_key.replace('Hz', ''))
            except:
                return 0

        rotation_speeds = sorted(condition_data.keys(), key=extract_speed_value)

        # Sample from each condition (use first 2 rotation speeds for efficiency)
        for speed_key in rotation_speeds[:2]:
            if speed_key in condition_data:
                segments = condition_data[speed_key]
                if len(segments) > 0:
                    # Use first 2 segments per speed for efficiency
                    for segment in segments[:2]:
                        if segment.shape[1] >= 8:  # Ensure we have all 8 residuals
                            for res_idx in range(8):  # Process all 8 residuals
                                data_seg = segment[:, res_idx]

                                if len(data_seg) >= 100:  # Ensure segment is long enough
                                    if normalize_residuals:
                                        data_mean = np.mean(data_seg)
                                        data_std = np.std(data_seg)
                                        if data_std > 0:
                                            data_seg = (data_seg - data_mean) / data_std
                                        else:
                                            data_seg = data_seg - data_mean

                                    try:
                                        Tx, _, _, _ = ssq.ssq_cwt(data_seg, fs=sampling_rate, wavelet='morlet')
                                        Tx_abs = np.abs(Tx)
                                        Tx_abs[Tx_abs == 0] = np.finfo(float).eps

                                        if normalize_spectogram:
                                            Tx_abs = (Tx_abs - np.min(Tx_abs)) / (np.max(Tx_abs) - np.min(Tx_abs) + 1e-12)

                                        all_values.append(Tx_abs.ravel())
                                        valid_segments_found = True
                                    except Exception as e:
                                        logger.warning(f"Failed to compute spectrogram for segment: {e}")
                                        continue

    if valid_segments_found and all_values:
        all_values = np.concatenate(all_values)
        # Use percentiles for robust limits
        if len(all_values) > 1000:
            global_vmin = np.percentile(all_values, 5)
            global_vmax = np.percentile(all_values, 95)
        else:
            global_vmin = np.min(all_values)
            global_vmax = np.max(all_values)

        # Ensure valid range
        if global_vmax <= global_vmin:
            global_vmax = global_vmin + 1e-10

        logger.info(".6f")
        return global_vmin, global_vmax
    else:
        logger.warning("No valid segments found for global colorbar limits, using defaults")
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


def print_spectrogram_metrics(condition_name: str, model_name: str,
                            all_metrics: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    """
    Print spectrogram metrics in a formatted way.

    Args:
        condition_name: Name of the condition
        model_name: Name of the model
        all_metrics: Nested dictionary with metrics [residual][speed][metric_name]
    """
    print(f"\n{'='*80}")
    print(f"SPECTROGRAM METRICS - {condition_name} ({model_name})")
    print(f"{'='*80}")

    residual_names = ['data_res1', 'data_res2', 'data_res3', 'data_res4',
                     'phys_res1', 'phys_res2', 'phys_res3', 'phys_res4']

    for residual in residual_names:
        if residual in all_metrics:
            print(f"\n{residual.upper()}:")
            print("-" * 40)

            # Get all rotation speeds for this residual
            speeds = list(all_metrics[residual].keys())

            # Print header
            header = f"{'Rotation Speed':<15} {'Peak Energy':<12} {'RMS Energy':<12} {'Total Energy':<13} {'Centroid':<10} {'Bandwidth':<11} {'Peak Freq':<11} {'Kurtosis':<10} {'Flatness':<10} {'Entropy':<10}"
            print(header)
            print("-" * len(header))

            for speed in sorted(speeds, key=lambda x: float(x.replace('Hz', '')) if x.replace('Hz', '').replace('.', '').isdigit() else 0):
                if speed in all_metrics[residual]:
                    metrics = all_metrics[residual][speed]
                    line = (f"{speed:<15} "
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


def plot_custom_spectrogram_layout(condition_data: Dict[str, np.ndarray], condition_name: str,
                                  model_name: str, output_path: str, sampling_rate: int = 50000,
                                  normalize_residuals: bool = False, normalize_spectogram: bool = True,
                                  global_vmin: float = None, global_vmax: float = None) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Create a custom spectrogram plot with residuals as rows and rotation speeds as columns.

    Args:
        condition_data: Dictionary mapping rotation speed keys to residual segments
        condition_name: Name of the condition for the plot title
        model_name: Name of the PINN model
        output_path: Path to save the plot
        sampling_rate: Sampling rate in Hz (default: 50000)
        normalize_residuals: Whether to normalize residuals
        normalize_spectogram: Whether to normalize spectrograms
        global_vmin: Global minimum value for colorbar (shared across all plots)
        global_vmax: Global maximum value for colorbar (shared across all plots)

    Returns:
        Dictionary containing spectrogram metrics for each residual and rotation speed
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import ssqueezepy as ssq

    # Get residual names and rotation speeds
    residual_names = ['data_res1', 'data_res2', 'data_res3', 'data_res4',
                     'phys_res1', 'phys_res2', 'phys_res3', 'phys_res4']

    # Sort rotation speeds numerically (extract numbers from strings like "12.3Hz")
    def extract_speed_value(speed_key):
        try:
            return float(speed_key.replace('Hz', ''))
        except:
            return 0

    rotation_speeds = sorted(condition_data.keys(), key=extract_speed_value)

    n_residuals = len(residual_names)
    n_speeds = len(rotation_speeds)

    # Create figure with custom layout
    fig = plt.figure(figsize=(4 * n_speeds, 3 * n_residuals))
    gs = GridSpec(n_residuals, n_speeds, figure=fig, hspace=0.3, wspace=0.3)

    # Initialize metrics dictionary
    all_metrics = {residual: {} for residual in residual_names}

    # Use provided global colorbar limits or defaults
    if global_vmin is None or global_vmax is None:
        global_vmin, global_vmax = 0, 1
        logger.warning("No global colorbar limits provided, using defaults")

    # Plot each residual x rotation speed combination
    for res_idx, residual_name in enumerate(residual_names):
        for speed_idx, speed_key in enumerate(rotation_speeds):
            ax = fig.add_subplot(gs[res_idx, speed_idx])

            if speed_key in condition_data:
                segments = condition_data[speed_key]

                # Concatenate all segments for this rotation speed
                data_concat = []
                for segment in segments:
                    if segment.shape[1] > res_idx:
                        data_seg = segment[:, res_idx]
                        data_concat.append(data_seg)

                if data_concat:
                    data_concat = np.concatenate(data_concat)

                    if len(data_concat) >= 100:  # Ensure we have enough data for spectrogram
                        if normalize_residuals:
                            data_mean = np.mean(data_concat)
                            data_std = np.std(data_concat)
                            if data_std > 0:
                                data_concat = (data_concat - data_mean) / data_std
                            else:
                                data_concat = data_concat - data_mean

                        try:
                            # Compute synchrosqueezed transform
                            Tx, _, ssq_freqs, _ = ssq.ssq_cwt(data_concat, fs=sampling_rate, wavelet='morlet')
                            Tx_abs = np.abs(Tx)
                            Tx_abs[Tx_abs == 0] = np.finfo(float).eps

                            # Calculate metrics BEFORE normalization for more meaningful results
                            time_axis = np.linspace(0, len(data_concat) / sampling_rate, len(data_concat))
                            metrics = calculate_spectrogram_metrics(Tx_abs, ssq_freqs, time_axis)
                            all_metrics[residual_name][speed_key] = metrics

                            if normalize_spectogram:
                                min_Tx_abs = np.min(Tx_abs)
                                max_Tx_abs = np.max(Tx_abs)
                                if max_Tx_abs > min_Tx_abs:
                                    Tx_abs = (Tx_abs - min_Tx_abs) / (max_Tx_abs - min_Tx_abs)

                            im = ax.imshow(Tx_abs, extent=[time_axis[0], time_axis[-1],
                                                         ssq_freqs[-1], ssq_freqs[0]],
                                         aspect='auto', cmap='magma', vmin=global_vmin, vmax=global_vmax)

                            # Set labels with larger font size
                            if speed_idx == 0:
                                ax.set_ylabel(f'{residual_name}\nFrequency [Hz]', fontsize=14)
                            if res_idx == 0:
                                ax.set_title(f'{speed_key}', fontsize=14)
                            if res_idx == n_residuals - 1:
                                ax.set_xlabel('Time [sec]', fontsize=14)

                            # Set tick label font sizes
                            ax.tick_params(axis='both', which='major', labelsize=12)
                            ax.tick_params(axis='both', which='minor', labelsize=12)

                            ax.set_ylim(ssq_freqs[-1], ssq_freqs[0])

                        except Exception as e:
                            ax.text(0.5, 0.5, f'Error:\n{str(e)[:20]}...', ha='center', va='center',
                                   transform=ax.transAxes, fontsize=12)
                    else:
                        ax.text(0.5, 0.5, 'Insufficient\ndata', ha='center', va='center', transform=ax.transAxes, fontsize=12)
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=12)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=12)

    # Always add colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.05, 0.02, 0.9])

    # Create a proper colorbar even if limits are default
    if global_vmin < global_vmax:
        mappable = plt.cm.ScalarMappable(cmap='magma')
        mappable.set_clim(global_vmin, global_vmax)
        cbar = fig.colorbar(mappable, cax=cbar_ax)
        cbar.set_label('|Coefficient|', fontsize=14)
        cbar.ax.tick_params(labelsize=12)
    else:
        # Fallback colorbar with default range
        mappable = plt.cm.ScalarMappable(cmap='magma')
        mappable.set_clim(0, 1)
        cbar = fig.colorbar(mappable, cax=cbar_ax)
        cbar.set_label('|Coefficient| (default range)', fontsize=14)
        cbar.ax.tick_params(labelsize=12)

    # Set main title
    fig.suptitle(f'PINN Residuals Spectrograms - {condition_name} ({model_name})',
                 y=0.98, fontsize=16, fontweight='bold')

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return all_metrics


def generate_pinn_spectrograms(model_name: str, output_dir: str = None,
                             conditions: List[str] = None, normalize_residuals: bool = False,
                             normalize_spectogram: bool = True, chunk_size_mb: int = 50,
                             max_samples: int = 2500) -> None:
    """
    Generate synchrosqueezed wavelet spectrograms for PINN model residuals.

    Args:
        model_name: Name of the PINN model ('relobralo', 'constant_weight', 'brdr', 'pecann')
        output_dir: Output directory for spectrograms (auto-generated if None)
        conditions: List of conditions to process (default: all)
        normalize_residuals: Whether to normalize residuals before processing
        normalize_spectogram: Whether to normalize spectrograms
        chunk_size_mb: Chunk size for memory-efficient PINN processing
        max_samples: Maximum number of samples to use (default: 2500 for 0.05s at 50kHz)
    """
    if output_dir is None:
        output_dir = f"pinn_spectrograms_{model_name}"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Generating PINN spectrograms for model: {model_name}")
    logger.info(f"Output directory: {output_dir}")

    # Process all conditions for this model
    all_condition_data = process_all_conditions_for_model(
        model_name=model_name,
        conditions=conditions,
        chunk_size_mb=chunk_size_mb,
        max_samples=max_samples
    )

    if not all_condition_data:
        logger.error("No condition data was successfully processed")
        return

    # Get condition display names
    condition_display_names = get_condition_display_names()

    # Calculate global colorbar limits across all conditions
    global_vmin, global_vmax = calculate_global_colorbar_limits(
        all_condition_data=all_condition_data,
        normalize_residuals=normalize_residuals,
        normalize_spectogram=normalize_spectogram
    )

    # Generate custom spectrograms for each condition
    for condition_key, condition_data in all_condition_data.items():
        logger.info(f"Generating custom spectrogram for condition: {condition_key}")

        # Create output path for the custom plot
        condition_display_name = condition_display_names.get(condition_key, condition_key)
        output_path = os.path.join(output_dir, f"pinn_spectrogram_{condition_key}_{model_name}.png")

        # Generate the custom spectrogram with residuals as rows and rotation speeds as columns
        try:
            condition_metrics = plot_custom_spectrogram_layout(
                condition_data=condition_data,
                condition_name=condition_display_name,
                model_name=model_name,
                output_path=output_path,
                normalize_residuals=normalize_residuals,
                normalize_spectogram=normalize_spectogram,
                global_vmin=global_vmin,
                global_vmax=global_vmax
            )
            logger.info(f"Custom spectrogram for {condition_key} saved to: {output_path}")

            # Print metrics for this condition
            print_spectrogram_metrics(condition_display_name, model_name, condition_metrics)

        except Exception as e:
            logger.error(f"Failed to generate custom spectrogram for {condition_key}: {e}")

    logger.info(f"PINN spectrogram generation completed for model: {model_name}")


def main():
    """Main function for PINN spectrogram generation."""
    parser = argparse.ArgumentParser(description="Generate optimized spectrograms from PINN model residuals")

    parser.add_argument("--model", type=str, help="PINN model name (relobralo, constant_weight, brdr, pecann)")
    parser.add_argument("--all-models", action="store_true", help="Process all available PINN models")
    parser.add_argument("--conditions", nargs="*", type=str, help="Specific conditions to process (default: optimized set)")
    parser.add_argument("--output-dir", type=str, help="Base output directory for spectrograms")
    parser.add_argument("--data-dir", type=str, default="Data/v3", help="Directory containing data files")
    parser.add_argument("--normalize-residuals", action="store_true", help="Normalize residuals before processing")
    parser.add_argument("--normalize-spectogram", action="store_true", default=True, help="Normalize spectrogram")
    parser.add_argument("--chunk-size-mb", type=int, default=50, help="Chunk size for memory-efficient processing (default: 50MB)")
    parser.add_argument("--max-samples", type=int, default=2500, help="Maximum samples to use (default: 2500 for 0.05s at 50kHz)")

    args = parser.parse_args()

    # Define available models
    available_models = ['relobralo', 'constant_weight', 'brdr', 'pecann']

    # Determine which models to process
    if args.all_models:
        models_to_process = available_models
    elif args.model:
        if args.model not in available_models:
            logger.error(f"Unknown model: {args.model}. Available models: {available_models}")
            return
        models_to_process = [args.model]
    else:
        logger.error("Please specify either --model or --all-models")
        return

    # Set default conditions if none specified (optimized set)
    if args.conditions is None:
        args.conditions = list(get_condition_mapping().keys())

    # Get condition display names for logging
    condition_display_names = get_condition_display_names()
    logger.info(f"Processing conditions: {[condition_display_names.get(c, c) for c in args.conditions]}")
    logger.info(f"Using first {args.max_samples} samples (≈{args.max_samples/50000:.3f}s at 50kHz)")

    # Process each model
    for model_name in models_to_process:
        try:
            # Set model-specific output directory
            model_output_dir = args.output_dir
            if model_output_dir is None:
                model_output_dir = f"optimized_pinn_spectrograms_{model_name}"
            elif len(models_to_process) > 1:
                model_output_dir = os.path.join(args.output_dir, model_name)

            # Generate spectrograms for this model
            generate_pinn_spectrograms(
                model_name=model_name,
                output_dir=model_output_dir,
                conditions=args.conditions,
                normalize_residuals=args.normalize_residuals,
                normalize_spectogram=args.normalize_spectogram,
                chunk_size_mb=args.chunk_size_mb,
                max_samples=args.max_samples
            )

        except Exception as e:
            logger.error(f"Failed to process model {model_name}: {e}")
            continue

    logger.info("All optimized PINN spectrogram generation tasks completed!")


if __name__ == "__main__":
    main()
