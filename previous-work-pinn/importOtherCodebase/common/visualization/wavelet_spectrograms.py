"""
Synchrosqueezed Wavelet Spectrogram Visualization Module

This module provides functions for generating synchrosqueezed wavelet
spectrograms from residual data, allowing visualization of time-frequency
characteristics across different rotation speeds and conditions.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import ssqueezepy as ssq  # Synchrosqueezed CWT package
import torch

def process_residuals(residuals_dict, seq_length=100):
    """
    Convert residual tensors (or already segmented lists) into sequences.
    If a given key already holds a list of segments, pass it through.
    Otherwise, split the tensor into fixed-length chunks.
    """
    processed = {}
    for key, value in tqdm(residuals_dict.items(), desc="Processing residuals"):
        if isinstance(value, list):
            processed[key] = value
        else:
            data = value.cpu().numpy() if hasattr(value, 'cpu') else value
            n = data.shape[0]
            n_seqs = n // seq_length
            sequences = np.split(data[:n_seqs * seq_length], n_seqs) if n_seqs > 0 else []
            processed[key] = sequences
    return processed

def adjust_segments_by_rotation(residuals_dict, data_paths, get_omegas):
    """
    Adjust segmentation based on rotation speed.
    Each key in the dictionary is assumed to correspond to a condition.
    The segmentation is modified such that each segment corresponds to one rotation.
    
    Parameters:
    - residuals_dict: Dictionary of residuals by condition
    - data_paths: Dictionary mapping conditions to file paths
    - get_omegas: Function to extract omega values from file paths
    
    Returns:
    - Modified residuals dictionary with segments adjusted by rotation
    """
    sampling_rate = 50000  # 50 kHz

    for key in residuals_dict:
        print(f"Original length for {key}: {len(residuals_dict[key])}")
        file_path = data_paths[key]
        omegas = get_omegas(file_path)
        omegas = omegas / (2 * np.pi)  # Convert from rad/s to Hz
        num_rotations = 1
        datapoints_per_rotation = sampling_rate / omegas
        datapoints_needed = np.ceil(num_rotations * datapoints_per_rotation).to(torch.int32)
        n_blocks = len(residuals_dict[key]) // 250000
        segments = []
        for i in range(n_blocks):
            seg_length = datapoints_needed[i] if i < len(datapoints_needed) else 200
            start_idx = i * 250000
            end_idx = start_idx + seg_length
            if end_idx <= (i + 1) * 250000:
                segments.append(residuals_dict[key][start_idx:end_idx])
        residuals_dict[key] = segments
        print(f"Segmented into {len(segments)} samples for {key}")
    return residuals_dict

def compute_rotation_speed(segment, sampling_rate=50000):
    """
    Compute rotation speed in Hz assuming each segment represents one full rotation.
    """
    seg_len = len(segment)
    return sampling_rate / seg_len if seg_len > 0 else np.nan

def group_segments_by_rotation_speed(segments, sampling_rate=50000, decimals=1):
    """
    Group segments by their computed rotation speed.
    Speeds are rounded to the specified number of decimals.
    Returns a dictionary with keys as the rounded speeds.
    """
    groups = {}
    for seg in segments:
        speed = compute_rotation_speed(seg, sampling_rate)
        key = round(speed, decimals)
        groups.setdefault(key, []).append(seg)
    return groups

def compute_normal_limits(normal_segments, sampling_rate=50000, n_features=8, normalize_residuals=False, normalize_spectogram=False):
    """
    Compute the colormap limits (vmin, vmax) and a representative frequency axis
    using only the 'normal' samples.
    """
    all_values = []
    global_freqs = None
    groups = group_segments_by_rotation_speed(normal_segments, sampling_rate)
    for speed in tqdm(groups):
        seg_group = groups[speed]
        for row in range(n_features):
            data_concat = []
            for seg in seg_group:
                # For multi-feature segments, pick the corresponding column.
                if seg.ndim == 2 and seg.shape[1] >= n_features:
                    data_seg = seg[:, row]
                else:
                    if row == 0:
                        data_seg = seg
                    else:
                        continue
                data_concat.append(data_seg)
            if len(data_concat) == 0:
                continue
            data_concat = np.concatenate(data_concat)

            if normalize_residuals:
                min_val = np.min(data_concat)
                max_val = np.max(data_concat)
                if max_val > min_val:  # Avoid division by zero if all values are the same
                    data_concat = (data_concat - min_val) / (max_val - min_val)

            # Compute the synchrosqueezed transform.
            Tx, _, ssq_freqs, _ = ssq.ssq_cwt(data_concat, fs=sampling_rate, wavelet='morlet')
            Tx_abs = np.abs(Tx)
            # Avoid taking log of zero entries if later applying a log-scale.
            Tx_abs[Tx_abs == 0] = np.finfo(float).eps

            if normalize_spectogram:
                min_Tx_abs = np.min(Tx_abs)
                max_Tx_abs = np.max(Tx_abs)
                if max_Tx_abs > min_Tx_abs:
                    Tx_abs = (Tx_abs - min_Tx_abs) / (max_Tx_abs - min_Tx_abs)

            all_values.append(Tx_abs.ravel())
            # Capture frequency axis once.
            if global_freqs is None:
                global_freqs = ssq_freqs
    if len(all_values) > 0:
        all_values = np.concatenate(all_values)
        global_vmin = np.percentile(all_values, 5)
        global_vmax = np.percentile(all_values, 95)
    else:
        global_vmin, global_vmax = 0, 1
    return global_vmin, global_vmax, global_freqs

def plot_synchrosqueezed_wavelet_by_feature(segments, condition, output_path_prefix,
                                            sampling_rate=50000,
                                            global_vmin=None, global_vmax=None,
                                            global_freqs=None,
                                            normalize_residuals=False,
                                            normalize_spectogram=False):
    """
    Plot synchrosqueezed wavelet spectrograms for each rotation-speed group and each feature.
    If global_vmin, global_vmax, and global_freqs are provided, use these to set a common colormap scale
    and frequency axis for all figures.
    """
    groups = group_segments_by_rotation_speed(segments, sampling_rate)
    unique_speeds = sorted(groups.keys())
    n_speeds = len(unique_speeds)
    n_features = 8

    fig, axs = plt.subplots(n_features, n_speeds,
                            figsize=(5 * n_speeds, 3 * n_features),
                            sharex='col', sharey='row')
    if n_speeds == 1:
        axs = axs.reshape(n_features, 1)

    mappable = None
    for col, speed in enumerate(unique_speeds):
        seg_group = groups[speed]
        for row in range(n_features):
            ax = axs[row, col]
            data_concat = []
            boundaries = []  # boundaries in seconds
            cum_time = 0.0
            for seg in seg_group:
                if seg.ndim == 2 and seg.shape[1] >= n_features:
                    data_seg = seg[:, row]
                else:
                    if row == 0:
                        data_seg = seg
                    else:
                        continue
                data_concat.append(data_seg)
                seg_duration = len(data_seg) / sampling_rate
                cum_time += seg_duration
                boundaries.append(cum_time)
            if len(data_concat) == 0:
                continue
            data_concat = np.concatenate(data_concat)

            if normalize_residuals:
                min_val = np.min(data_concat)
                max_val = np.max(data_concat)
                if max_val > min_val:  # Avoid division by zero if all values are the same
                    data_concat = (data_concat - min_val) / (max_val - min_val)

            # Compute the synchrosqueezed transform for this concatenated data.
            Tx, _, ssq_freqs, _ = ssq.ssq_cwt(data_concat, fs=sampling_rate, wavelet='morlet')
            Tx_abs = np.abs(Tx)
            Tx_abs[Tx_abs == 0] = np.finfo(float).eps

            if normalize_spectogram:
                min_Tx_abs = np.min(Tx_abs)
                max_Tx_abs = np.max(Tx_abs)
                if max_Tx_abs > min_Tx_abs:
                    Tx_abs = (Tx_abs - min_Tx_abs) / (max_Tx_abs - min_Tx_abs)

            # Use the provided global parameters if available.
            if global_vmin is None or global_vmax is None or global_freqs is None:
                vmin = np.percentile(Tx_abs, 5)
                vmax = np.percentile(Tx_abs, 95)
                freq_axis = ssq_freqs
            else:
                vmin = global_vmin
                vmax = global_vmax
                freq_axis = global_freqs

            time_axis = np.linspace(0, len(data_concat) / sampling_rate, len(data_concat))

            im = ax.imshow(Tx_abs, extent=[time_axis[0], time_axis[-1],
                                                    freq_axis[-1], freq_axis[0]],
                           aspect='auto', cmap='magma', vmin=vmin, vmax=vmax)
            mappable = im if mappable is None else mappable

            for boundary in boundaries[:-1]:
                ax.axvline(x=boundary, color='w', linestyle='--', linewidth=1)
            if col == 0:
                ax.set_ylabel(f"Feature {row+1}\nFrequency [Hz]")
            if row == n_features - 1:
                ax.set_xlabel("Time [sec]")
            if row == 0:
                ax.set_title(f"Speed: {speed:.1f} Hz")
            # Set the y-limits to use the shared frequency axis.
            ax.set_ylim(freq_axis[-1], freq_axis[0])

    if mappable is not None:
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.87, 0.05, 0.02, 0.9])
        fig.colorbar(mappable, cax=cbar_ax, label='|Coefficient|')

    fig.suptitle(f"Synchrosqueezed Wavelet Spectrogram by Feature - {condition}", y=1.02)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(f"{output_path_prefix}_synchrosqueezed_wavelet_by_feature.png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_spectrograms(residuals_path, output_dir=None, normalize_residuals=False, normalize_spectogram=True):
    """
    Generate synchrosqueezed wavelet spectrograms from residuals.
    
    Parameters:
    - residuals_path: Path to the residuals file (.pth)
    - output_dir: Directory to save the spectrograms (default: generated from residuals path)
    - normalize_residuals: Whether to normalize residuals before processing
    - normalize_spectogram: Whether to normalize the spectrogram
    """
    # Create output directory if not provided
    if output_dir is None:
        base_name = os.path.splitext(os.path.basename(residuals_path))[0]
        output_dir = f"spectrograms_{base_name}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the residuals
    print(f"Loading residuals from {residuals_path}")
    residuals_dict = torch.load(residuals_path)
    
    # Process residuals
    processed = process_residuals(residuals_dict)
    
    # Compute limits from normal data
    if "normal" in processed:
        global_vmin, global_vmax, global_freqs = compute_normal_limits(
            processed["normal"], 
            normalize_residuals=normalize_residuals, 
            normalize_spectogram=normalize_spectogram
        )
    else:
        # Fallback: use the first available condition
        first_condition = next(iter(processed))
        global_vmin, global_vmax, global_freqs = compute_normal_limits(
            processed[first_condition], 
            normalize_residuals=normalize_residuals, 
            normalize_spectogram=normalize_spectogram
        )
    
    print(f"Using global vmin, vmax: {global_vmin}, {global_vmax}")
    
    # Generate spectrograms for each condition
    for condition, segments in tqdm(processed.items(), desc="Processing conditions"):
        print(f"\nProcessing condition: {condition} with {len(segments)} segments")
        
        # Create condition-specific output directory
        condition_dir = os.path.join(output_dir, condition)
        os.makedirs(condition_dir, exist_ok=True)
        
        output_prefix = os.path.join(condition_dir, condition)
        
        # Generate the spectrograms
        plot_synchrosqueezed_wavelet_by_feature(
            segments, 
            condition, 
            output_prefix,
            global_vmin=global_vmin,
            global_vmax=global_vmax,
            global_freqs=global_freqs,
            normalize_residuals=normalize_residuals,
            normalize_spectogram=normalize_spectogram
        )
    
    print(f"Spectrograms saved to {output_dir}")
    return output_dir 