"""
Preprocessing functions for residual data.

This module handles preprocessing of residuals from different model types
(direct PINN, hybrid RNN-PINN, and data-driven models) to convert them into 
a standardized format for anomaly detection.
"""

import os
import torch
import numpy as np
from tqdm import tqdm
import glob

def extract_data(sample):
    """
    Given a sample, extract its raw data.
    
    Parameters:
    - sample: Input sample (can be a dict or raw data)
    
    Returns:
    - Raw data
    """
    if isinstance(sample, dict) and 'data' in sample:
        return sample['data']
    return sample

def process_residuals_direct_pinn(residuals_dict, seg_length=None):
    """
    Process residuals from direct PINN format.
    
    Parameters:
    - residuals_dict: Dictionary of residuals by condition
    - seg_length: Optional segment length (if None, determined automatically)
    
    Returns:
    - Dictionary of processed residuals by condition
    """
    processed = {}
    
    # Process each condition
    for key, data in tqdm(residuals_dict.items(), desc="Processing direct PINN residuals"):
        # Skip if already processed
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            processed[key] = data
            continue
            
        # Handle different input formats
        if isinstance(data, list):
            processed[key] = data
            continue
            
        # Convert tensor to numpy
        np_data = data.cpu().numpy() if hasattr(data, 'cpu') else np.asarray(data)
        
        # Determine segment length if not provided
        if seg_length is None:
            # Default segmentation: split into chunks of roughly 200-1000 points
            total_len = len(np_data)
            if total_len <= 1000:
                seg_length = total_len
            else:
                seg_length = max(200, min(1000, total_len // 10))
        
        # Split into segments
        n_segments = len(np_data) // seg_length
        segments = []
        
        for i in range(n_segments):
            start_idx = i * seg_length
            end_idx = (i + 1) * seg_length
            segment = np_data[start_idx:end_idx]
            
            # Skip empty segments
            if len(segment) == 0:
                continue
                
            # Store as dictionary if not already
            if isinstance(segment, dict):
                segments.append(segment)
            else:
                segments.append({'data': segment})
        
        processed[key] = segments
        
    return processed

def process_residuals_hybrid_rnn(residuals_dict):
    """
    Process residuals from hybrid RNN-PINN format.
    
    In hybrid RNN format, each key in residuals_dict corresponds to a data type (normal, fault1, etc.)
    and maps to another dictionary where keys are sequence indices and values are dictionaries with
    'data', 'rot_speed', and 'dtw_distances' keys.
    
    Parameters:
    - residuals_dict: Dictionary of residuals by condition
    
    Returns:
    - Dictionary of processed residuals by condition
    """
    processed = {}
    
    # Process each condition
    for data_type, sequences in tqdm(residuals_dict.items(), desc="Processing hybrid RNN residuals"):
        processed_sequences = []
        
        # Process each sequence
        for seq_idx, seq_data in sequences.items():
            # Extract data and metadata
            data = seq_data['data'].numpy() if hasattr(seq_data['data'], 'numpy') else seq_data['data']
            rot_speed = seq_data.get('rot_speed', None)
            
            # Create processed sequence
            processed_seq = {'data': data}
            if rot_speed is not None:
                processed_seq['rot_speed'] = rot_speed
                
            # Add DTW distances if available    
            if 'dtw_distances' in seq_data:
                processed_seq['dtw_distances'] = seq_data['dtw_distances']
                
            processed_sequences.append(processed_seq)
            
        processed[data_type] = processed_sequences
        
    return processed

def process_residuals_data_driven(residuals_dict):
    """
    Process residuals from data-driven models.
    
    Data-driven residual format can vary, but typically follows a similar
    structure to either direct PINN or hybrid RNN-PINN formats.
    
    Parameters:
    - residuals_dict: Dictionary of residuals by condition
    
    Returns:
    - Dictionary of processed residuals by condition
    """
    processed = {}
    
    # Check the structure to determine the appropriate processing method
    first_key = next(iter(residuals_dict))
    first_value = residuals_dict[first_key]
    
    # If the value is a dict with sequence indices as keys and data dicts as values
    # (similar to hybrid RNN format)
    if isinstance(first_value, dict) and all(isinstance(v, dict) and 'data' in v for v in first_value.values()):
        return process_residuals_hybrid_rnn(residuals_dict)
    
    # If the value is a list of dicts with 'data' key (already processed segments)
    elif isinstance(first_value, list) and len(first_value) > 0 and isinstance(first_value[0], dict):
        return residuals_dict
    
    # Otherwise, treat it as a tensor or list that needs to be segmented (like direct PINN)
    else:
        return process_residuals_direct_pinn(residuals_dict)

def preprocess_residuals(residuals_path, source_type="direct"):
    """
    Load and preprocess residuals based on source type.
    
    Parameters:
    - residuals_path: Path to the residuals file (.pth)
    - source_type: "direct", "hybrid", or "data_driven"
    
    Returns:
    - Processed residuals dictionary
    """
    print(f"Loading residuals from {residuals_path}...")
    
    # Load residuals
    try:
        residuals_dict = torch.load(residuals_path)
    except Exception as e:
        raise ValueError(f"Error loading residuals from {residuals_path}: {e}")
    
    # Process based on source type
    if source_type == "direct":
        # Process direct PINN residuals
        try:
            # Try to import rotation speed information
            from Data.LoadData import data_paths, get_omegas
            
            # Add rotation speed information from file metadata
            for key in residuals_dict:
                print(f"Original length for {key}: {len(residuals_dict[key])}")
                file_path = data_paths[key]
                
                # Compute rotation speeds (Hz) from file metadata
                omegas = get_omegas(file_path) / (2 * np.pi)
                
                # Calculate segment length based on rotation speed
                num_rotations = 1
                sampling_rate = 50000  # 50kHz
                datapoints_per_rotation = sampling_rate / omegas
                datapoints_needed = np.ceil(num_rotations * datapoints_per_rotation).to(torch.int)
                
                # Split into segments with rotation speed metadata
                n_blocks = len(residuals_dict[key]) // 250000
                segments = []
                
                for i in range(n_blocks):
                    # Get segment length for this block
                    seg_length = int(datapoints_needed[i]) if i < len(datapoints_needed) else 200
                    
                    # Determine number of segments to extract per block
                    if key == 'normal':
                        max_segments = min(5, 250000 // seg_length)
                    else:
                        max_segments = min(10, 250000 // seg_length)
                    
                    # Extract segments
                    for j in range(max_segments):
                        start_idx = i * 250000 + j * seg_length
                        end_idx = start_idx + seg_length
                        
                        if end_idx <= (i + 1) * 250000:
                            segment = residuals_dict[key][start_idx:end_idx]
                            if len(segment) > 0:
                                segments.append({'data': segment, 'rot_speed': float(omegas[i])})
                
                residuals_dict[key] = segments
                print(f"Segmented into {len(segments)} samples for {key}")
            
            return process_residuals_direct_pinn(residuals_dict)
        
        except ImportError:
            # Fall back to basic segmentation if rotation speed info is unavailable
            print("Warning: Unable to import Data.LoadData. Using basic segmentation without rotation speed metadata.")
            return process_residuals_direct_pinn(residuals_dict)
    
    elif source_type == "hybrid":
        # Hybrid RNN residuals are already in the right format with speed information
        return process_residuals_hybrid_rnn(residuals_dict)
    
    elif source_type == "data_driven":
        # Process data-driven model residuals
        return process_residuals_data_driven(residuals_dict)
    
    else:
        raise ValueError(f"Invalid source_type: {source_type}. Must be 'direct', 'hybrid', or 'data_driven'.")

def find_latest_residuals(base_path="."):
    """
    Find the latest residuals files for all model types.
    
    Parameters:
    - base_path: Base directory to search
    
    Returns:
    - Dictionary with model types as keys and residual paths as values
    """
    residuals_files = {}
    
    # Find hybrid model residuals
    hybrid_patterns = [
        "hybrid_rnn_residuals_*.pth", 
        "hybrid_results/*residuals*.pth", 
        "hybrid_analysis_results/*residuals*.pth",
        "results/hybrid_residuals.pth"
    ]
    
    hybrid_files = []
    for pattern in hybrid_patterns:
        hybrid_files.extend(glob.glob(os.path.join(base_path, pattern)))
    
    if hybrid_files:
        # Sort by modification time (most recent first)
        latest_hybrid = max(hybrid_files, key=os.path.getmtime)
        residuals_files["hybrid"] = latest_hybrid
    
    # Find direct PINN residuals
    direct_patterns = [
        "direct_pinn_residuals_*.pth",
        "direct_results/*residuals*.pth",
        "direct_analysis_results/*residuals*.pth"
    ]
    
    direct_files = []
    for pattern in direct_patterns:
        direct_files.extend(glob.glob(os.path.join(base_path, pattern)))
    
    if direct_files:
        latest_direct = max(direct_files, key=os.path.getmtime)
        residuals_files["direct"] = latest_direct
    
    # Find data-driven model residuals
    data_driven_patterns = [
        "data_driven_*_models/*_residuals.pth"
    ]
    
    # Dictionary to store latest residuals for each data-driven model type
    data_driven_by_type = {}
    
    for pattern in data_driven_patterns:
        for file_path in glob.glob(os.path.join(base_path, pattern)):
            # Extract model type (e.g., 'lstm', 'simple_rnn')
            file_name = os.path.basename(file_path)
            model_type = file_name.split('_residuals.pth')[0]
            
            # Update if this file is newer than the current latest for this model type
            if model_type not in data_driven_by_type or \
               os.path.getmtime(file_path) > os.path.getmtime(data_driven_by_type[model_type]):
                data_driven_by_type[model_type] = file_path
    
    # Add all data-driven models to the result
    for model_type, file_path in data_driven_by_type.items():
        residuals_files[f"data_driven_{model_type}"] = file_path
    
    return residuals_files 