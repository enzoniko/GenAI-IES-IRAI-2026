"""
Common utilities for standardized residuals format.

This module provides common functions to help standardize the residuals
format across all analysis modules.
"""

import torch
import numpy as np
from tqdm import tqdm
from fastdtw import fastdtw
import os
import h5py
import datetime

def get_rotation_segments(data, omegas, sampling_rate=50000):
    """
    Split data into complete rotation segments based on rotation frequencies.
    
    Parameters:
    - data: Tensor of shape [num_blocks, timestamps, variables]
    - omegas: Tensor of rotation speeds in rad/s
    - sampling_rate: Sampling rate in Hz
    
    Returns:
    - Dictionary mapping rotation frequencies to lists of rotation segments
    """
    segments_by_freq = {}
    
    for block_idx, block_data in enumerate(data):
        if block_idx >= len(omegas):
            continue
            
        # Convert angular velocity to frequency in Hz
        freq_hz = float(omegas[block_idx]) / (2 * np.pi)
        freq_key = round(freq_hz, 2)  # Round to 2 decimal places for consistent keys
        
        # Calculate points per complete rotation
        points_per_rotation = int(sampling_rate / freq_hz)
        
        # Calculate how many complete rotations we can extract
        num_complete_rotations = len(block_data) // points_per_rotation
        
        # Initialize the list for this frequency if it doesn't exist
        if freq_key not in segments_by_freq:
            segments_by_freq[freq_key] = []
        
        # Extract each complete rotation
        for i in range(num_complete_rotations):
            start_idx = i * points_per_rotation
            end_idx = (i + 1) * points_per_rotation
            
            # Extract segment for this complete rotation
            segment = block_data[start_idx:end_idx]
            
            # Add to the appropriate frequency bucket
            segments_by_freq[freq_key].append(segment)
    
    return segments_by_freq

def compute_dtw_distances(actual, predicted):
    """
    Compute DTW distances between actual and predicted values.
    
    Parameters:
    - actual: Tensor of shape [timestamps, variables]
    - predicted: Tensor of shape [timestamps, variables]
    
    Returns:
    - List of DTW distances for each variable
    """
    dtw_distances = []
    
    for var_idx in range(actual.shape[1]):
        actual_var = actual[:, var_idx].cpu().numpy()
        predicted_var = predicted[:, var_idx].cpu().numpy()
        
        try:
            # Compute fast DTW distance for this variable
            dtw_dist, _ = fastdtw(actual_var, predicted_var)
            dtw_distances.append(dtw_dist)
        except Exception as e:
            print(f"Error computing DTW for variable {var_idx}: {e}")
            dtw_distances.append(np.nan)
    
    return dtw_distances

def standardize_residuals(residuals_dict, to_numpy=False):
    """
    Standardize residuals dictionary to the common format.
    
    Parameters:
    - residuals_dict: Dictionary of residuals
    - to_numpy: Whether to convert tensors to NumPy arrays for more efficient storage
      (Default: False - keep as PyTorch tensors)
    
    Returns:
    - Standardized residuals dictionary
    """
    standardized_dict = {
        "data_residuals": {}
    }
    
    # Process data residuals
    for data_type, freq_dict in residuals_dict.items():
        standardized_dict["data_residuals"][data_type] = {}
        
        for freq, residuals_list in freq_dict.items():
            # Combine all residuals for this frequency into a single tensor
            residuals_tensors = [r for r, _ in residuals_list]
            dtw_tensors = [torch.tensor(d).unsqueeze(0) for _, d in residuals_list]
            
            combined_residuals = torch.stack(residuals_tensors, dim=0)
            combined_dtw = torch.cat(dtw_tensors, dim=0)
            
            # Convert to numpy if requested (to save space when saving)
            if to_numpy:
                combined_residuals = combined_residuals.cpu().numpy()
                combined_dtw = combined_dtw.cpu().numpy()
            
            standardized_dict["data_residuals"][data_type][freq] = {
                "residuals": combined_residuals,
                "DTW": combined_dtw
            }
    
    # Process physical residuals if available
    if "physical_residuals" in residuals_dict:
        standardized_dict["physical_residuals"] = {}
        
        for data_type, freq_dict in residuals_dict["physical_residuals"].items():
            standardized_dict["physical_residuals"][data_type] = {}
            
            for freq, residuals_list in freq_dict.items():
                # Combine all residuals for this frequency into a single tensor
                residuals_tensors = [r for r, _ in residuals_list]
                dtw_tensors = [torch.tensor(d).unsqueeze(0) for _, d in residuals_list]
                
                combined_residuals = torch.stack(residuals_tensors, dim=0)
                combined_dtw = torch.cat(dtw_tensors, dim=0)
                
                # Check for and remove extra dimension in physical residuals
                if combined_residuals.dim() == 4 and combined_residuals.size(3) == 1:
                    combined_residuals = combined_residuals.squeeze(3)
                
                # Convert to numpy if requested (to save space when saving)
                if to_numpy:
                    combined_residuals = combined_residuals.cpu().numpy()
                    combined_dtw = combined_dtw.cpu().numpy()
                
                standardized_dict["physical_residuals"][data_type][freq] = {
                    "residuals": combined_residuals,
                    "DTW": combined_dtw
                }
    
    return standardized_dict

def print_residuals_structure(residuals_dict):
    """
    Print the structure of the residuals dictionary without the actual data.
    
    Parameters:
    - residuals_dict: The standardized residuals dictionary
    """
    print("\n=== RESIDUALS STRUCTURE ===")
    
    for top_key in residuals_dict.keys():
        print(f"└─ {top_key}")
        
        # Print data types
        for data_type in residuals_dict[top_key].keys():
            print(f"   └─ {data_type}")
            
            # Print frequencies
            for freq in residuals_dict[top_key][data_type].keys():
                print(f"      └─ {freq} Hz")
                
                # Print contents (without the actual data)
                freq_data = residuals_dict[top_key][data_type][freq]
                
                # Print shape of residuals tensor
                if 'residuals' in freq_data:
                    shape = freq_data['residuals'].shape
                    print(f"         └─ residuals: Tensor{shape}")
                
                # Print shape of DTW tensor
                if 'DTW' in freq_data:
                    dtw_shape = freq_data['DTW'].shape
                    print(f"         └─ DTW: Tensor{dtw_shape}")
    
    print("\nFormat is: {data/physical}_residuals → data_type → frequency → {residuals, DTW}")
    print("=== END OF STRUCTURE ===\n") 

def save_residuals_incremental(residuals_dict, filename):
    """
    Save residuals incrementally to an HDF5 file to avoid memory issues.
    
    Parameters:
    - residuals_dict: Dictionary of residuals in the standardized format
    - filename: Path to save the HDF5 file
    """
    print(f"Saving residuals incrementally to {filename}...")
    
    # Create or overwrite the file
    with h5py.File(filename, 'w') as f:
        # Save structure for data residuals
        if "data_residuals" in residuals_dict:
            data_group = f.create_group("data_residuals")
            
            # Process each data type sequentially to avoid memory issues
            for data_type in list(residuals_dict["data_residuals"].keys()):
                print(f"  Saving data type: {data_type}")
                data_type_group = data_group.create_group(data_type)
                
                # Process each frequency
                for freq in residuals_dict["data_residuals"][data_type].keys():
                    freq_group = data_type_group.create_group(str(freq))
                    
                    # Get the residuals and DTW values
                    residuals = residuals_dict["data_residuals"][data_type][freq]["residuals"]
                    dtw = residuals_dict["data_residuals"][data_type][freq]["DTW"]
                    
                    # Check if residuals is torch tensor or numpy array
                    if isinstance(residuals, torch.Tensor):
                        residuals_np = residuals.cpu().numpy()
                    else:
                        residuals_np = residuals  # Already a NumPy array
                    
                    # Check if DTW is torch tensor or numpy array
                    if isinstance(dtw, torch.Tensor):
                        dtw_np = dtw.cpu().numpy()
                    else:
                        dtw_np = dtw  # Already a NumPy array
                    
                    # Save as datasets
                    freq_group.create_dataset("residuals", data=residuals_np)
                    freq_group.create_dataset("DTW", data=dtw_np)
        
        # Save structure for physical residuals if present
        if "physical_residuals" in residuals_dict:
            phys_group = f.create_group("physical_residuals")
            
            # Process each data type sequentially
            for data_type in list(residuals_dict["physical_residuals"].keys()):
                print(f"  Saving physical data type: {data_type}")
                data_type_group = phys_group.create_group(data_type)
                
                # Process each frequency
                for freq in residuals_dict["physical_residuals"][data_type].keys():
                    freq_group = data_type_group.create_group(str(freq))
                    
                    # Get the residuals and DTW values
                    residuals = residuals_dict["physical_residuals"][data_type][freq]["residuals"]
                    dtw = residuals_dict["physical_residuals"][data_type][freq]["DTW"]
                    
                    # Check if residuals is torch tensor or numpy array
                    if isinstance(residuals, torch.Tensor):
                        residuals_np = residuals.cpu().numpy()
                    else:
                        residuals_np = residuals  # Already a NumPy array
                    
                    # Check if DTW is torch tensor or numpy array
                    if isinstance(dtw, torch.Tensor):
                        dtw_np = dtw.cpu().numpy()
                    else:
                        dtw_np = dtw  # Already a NumPy array
                    
                    # Save as datasets
                    freq_group.create_dataset("residuals", data=residuals_np)
                    freq_group.create_dataset("DTW", data=dtw_np)
    
    print(f"Residuals saved successfully to {filename}")

def load_residuals_h5(filename):
    """
    Load residuals from an HDF5 file, converting back to PyTorch tensors.
    
    Parameters:
    - filename: Path to the HDF5 file
    
    Returns:
    - Dictionary of residuals with PyTorch tensors
    """
    print(f"Loading residuals from {filename}...")
    
    result = {}
    
    try:
        with h5py.File(filename, 'r') as f:
            # Load data residuals
            if "data_residuals" in f:
                result["data_residuals"] = {}
                
                for data_type in f["data_residuals"]:
                    result["data_residuals"][data_type] = {}
                    
                    for freq in f["data_residuals"][data_type]:
                        result["data_residuals"][data_type][float(freq)] = {
                            "residuals": torch.from_numpy(f["data_residuals"][data_type][freq]["residuals"][:]),
                            "DTW": torch.from_numpy(f["data_residuals"][data_type][freq]["DTW"][:])
                        }
            
            # Load physical residuals if present
            if "physical_residuals" in f:
                result["physical_residuals"] = {}
                
                for data_type in f["physical_residuals"]:
                    result["physical_residuals"][data_type] = {}
                    
                    for freq in f["physical_residuals"][data_type]:
                        result["physical_residuals"][data_type][float(freq)] = {
                            "residuals": torch.from_numpy(f["physical_residuals"][data_type][freq]["residuals"][:]),
                            "DTW": torch.from_numpy(f["physical_residuals"][data_type][freq]["DTW"][:])
                        }
        
        print(f"Successfully loaded residuals with {len(result['data_residuals'])} data types")
        return result
    
    except Exception as e:
        print(f"Error loading residuals from {filename}: {e}")
        return None

def load_residuals(path, combine=False):
    """
    Universal loader for residuals that handles both HDF5 and PyTorch formats.
    
    Parameters:
    - path: Path to the residuals file
    - combine: Ignored parameter (kept for backward compatibility)
    
    Returns:
    - Loaded residuals dictionary
    """
    if path.endswith('.h5'):
        return load_residuals_h5(path)
    else:
        try:
            return torch.load(path)
        except Exception as e:
            print(f"Error loading residuals from {path}: {e}")
            return None

def convert_residuals_to_numpy(residuals_dict):
    """
    Convert all torch tensors in the residuals dictionary to NumPy arrays for better compression.
    
    Parameters:
    - residuals_dict: Dictionary of residuals in the standardized format
    
    Returns:
    - Dictionary with same structure but tensors converted to NumPy arrays
    """
    numpy_dict = {}
    
    for residual_type, data_types in residuals_dict.items():
        numpy_dict[residual_type] = {}
        
        for data_type, frequencies in data_types.items():
            numpy_dict[residual_type][data_type] = {}
            
            for freq, freq_data in frequencies.items():
                numpy_dict[residual_type][data_type][freq] = {}
                
                # Convert residuals tensor to NumPy
                if isinstance(freq_data["residuals"], torch.Tensor):
                    numpy_dict[residual_type][data_type][freq]["residuals"] = freq_data["residuals"].cpu().numpy()
                else:
                    numpy_dict[residual_type][data_type][freq]["residuals"] = freq_data["residuals"]
                
                # Convert DTW values to NumPy if it's a tensor
                if isinstance(freq_data["DTW"], torch.Tensor):
                    numpy_dict[residual_type][data_type][freq]["DTW"] = freq_data["DTW"].cpu().numpy()
                else:
                    numpy_dict[residual_type][data_type][freq]["DTW"] = freq_data["DTW"]
    
    return numpy_dict

def convert_numpy_to_tensors(residuals_dict):
    """
    Convert all NumPy arrays in the residuals dictionary back to torch tensors.
    
    Parameters:
    - residuals_dict: Dictionary of residuals with NumPy arrays
    
    Returns:
    - Dictionary with same structure but NumPy arrays converted to tensors
    """
    tensor_dict = {}
    
    for residual_type, data_types in residuals_dict.items():
        tensor_dict[residual_type] = {}
        
        for data_type, frequencies in data_types.items():
            tensor_dict[residual_type][data_type] = {}
            
            for freq, freq_data in frequencies.items():
                tensor_dict[residual_type][data_type][freq] = {}
                
                # Convert residuals NumPy array to tensor
                if isinstance(freq_data["residuals"], np.ndarray):
                    tensor_dict[residual_type][data_type][freq]["residuals"] = torch.from_numpy(freq_data["residuals"])
                else:
                    tensor_dict[residual_type][data_type][freq]["residuals"] = freq_data["residuals"]
                
                # Convert DTW values to tensor if it's a NumPy array
                if isinstance(freq_data["DTW"], np.ndarray):
                    tensor_dict[residual_type][data_type][freq]["DTW"] = torch.from_numpy(freq_data["DTW"])
                else:
                    tensor_dict[residual_type][data_type][freq]["DTW"] = freq_data["DTW"]
    
    return tensor_dict

def save_raw_residuals(residuals_dict, filename, compression=4):
    """
    Save raw residuals data in a compact format without reorganizing by rotations.
    
    Parameters:
    - residuals_dict: Dictionary containing raw residuals
    - filename: Output HDF5 file path
    - compression: Compression level (0-9, higher = more compression but slower)
    """
    with h5py.File(filename, 'w') as h5file:
        # Create metadata group
        meta_group = h5file.create_group('metadata')
        meta_group.attrs['format_version'] = 2.0
        meta_group.attrs['creation_date'] = datetime.datetime.now().isoformat()
        
        # Save data residuals in raw format
        if 'data_residuals' in residuals_dict:
            data_group = h5file.create_group('data_residuals_raw')
            
            # Store metadata about frequencies
            freq_group = h5file.create_group('data_frequencies')
            
            for data_type, data in residuals_dict['data_residuals'].items():
                # Create group for this data type
                type_group = data_group.create_group(data_type)
                type_freq_group = freq_group.create_group(data_type)
                
                # Store raw tensors for each frequency without separation into rotations
                for freq, freq_data in data.items():
                    # Store frequency as attribute
                    freq_str = str(freq)
                    
                    # Store frequency metadata
                    type_freq_group.attrs[freq_str] = float(freq)
                    
                    # Get the raw residuals tensor
                    if isinstance(freq_data, dict) and 'residuals' in freq_data:
                        # If already in standardized format, use the raw tensor
                        residuals_tensor = freq_data['residuals']
                    else:
                        # Assuming it's the raw tensor
                        residuals_tensor = freq_data
                    
                    # Convert to numpy if needed
                    if isinstance(residuals_tensor, torch.Tensor):
                        residuals_tensor = residuals_tensor.cpu().numpy()
                    
                    # Store the raw tensor with compression
                    type_group.create_dataset(
                        freq_str, 
                        data=residuals_tensor,
                        compression="gzip", 
                        compression_opts=compression
                    )
        
        # Save physical residuals in raw format if available
        if 'physical_residuals' in residuals_dict and residuals_dict['physical_residuals']:
            phys_group = h5file.create_group('physical_residuals_raw')
            
            # Store metadata about frequencies
            freq_group = h5file.create_group('physical_frequencies') if 'data_frequencies' not in h5file else h5file['data_frequencies']
            
            for data_type, data in residuals_dict['physical_residuals'].items():
                # Create group for this data type
                type_group = phys_group.create_group(data_type)
                if data_type not in freq_group:
                    type_freq_group = freq_group.create_group(data_type)
                else:
                    type_freq_group = freq_group[data_type]
                
                # Store raw tensors for each frequency without separation into rotations
                for freq, freq_data in data.items():
                    # Store frequency as attribute
                    freq_str = str(freq)
                    
                    # Store frequency metadata
                    if freq_str not in type_freq_group.attrs:
                        type_freq_group.attrs[freq_str] = float(freq)
                    
                    # Get the raw residuals tensor
                    if isinstance(freq_data, dict) and 'residuals' in freq_data:
                        # If already in standardized format, use the raw tensor
                        residuals_tensor = freq_data['residuals']
                    else:
                        # Assuming it's the raw tensor
                        residuals_tensor = freq_data
                    
                    # Convert to numpy if needed
                    if isinstance(residuals_tensor, torch.Tensor):
                        residuals_tensor = residuals_tensor.cpu().numpy()
                    
                    # Store the raw tensor with compression
                    type_group.create_dataset(
                        freq_str, 
                        data=residuals_tensor,
                        compression="gzip", 
                        compression_opts=compression
                    )

def load_raw_residuals(filename):
    """
    Load raw residuals data from HDF5 file.
    
    Parameters:
    - filename: Path to HDF5 file
    
    Returns:
    - Dictionary containing raw residuals
    """
    residuals_dict = {
        'data_residuals': {},
        'physical_residuals': {}
    }
    
    with h5py.File(filename, 'r') as h5file:
        # Check format version
        if 'metadata' in h5file and 'format_version' in h5file['metadata'].attrs:
            version = h5file['metadata'].attrs['format_version']
            if version != 2.0:
                print(f"Warning: Loading file with version {version}, expected 2.0")
        
        # Load data residuals
        if 'data_residuals_raw' in h5file:
            for data_type in h5file['data_residuals_raw']:
                residuals_dict['data_residuals'][data_type] = {}
                
                for freq in h5file['data_residuals_raw'][data_type]:
                    # Load data
                    data = h5file['data_residuals_raw'][data_type][freq][()]
                    
                    # Convert to tensor if needed
                    if isinstance(data, np.ndarray):
                        data = torch.from_numpy(data).float()
                    
                    # Store with frequency as key
                    residuals_dict['data_residuals'][data_type][float(freq)] = data
        
        # Load physical residuals
        if 'physical_residuals_raw' in h5file:
            for data_type in h5file['physical_residuals_raw']:
                residuals_dict['physical_residuals'][data_type] = {}
                
                for freq in h5file['physical_residuals_raw'][data_type]:
                    # Load data
                    data = h5file['physical_residuals_raw'][data_type][freq][()]
                    
                    # Convert to tensor if needed
                    if isinstance(data, np.ndarray):
                        data = torch.from_numpy(data).float()
                    
                    # Store with frequency as key
                    residuals_dict['physical_residuals'][data_type][float(freq)] = data
    
    return residuals_dict

def standardize_raw_residuals(raw_residuals, sampling_rate=50000, compute_dtw=False):
    """
    Convert raw residuals to standardized format with rotations.
    
    Parameters:
    - raw_residuals: Dictionary with raw residuals
    - sampling_rate: Sampling rate in Hz
    - compute_dtw: Whether to compute DTW distances
    
    Returns:
    - Standardized residuals dictionary
    """
    standardized = {
        'data_residuals': {},
        'physical_residuals': {}
    }
    
    # Process data residuals
    for data_type, freqs in raw_residuals['data_residuals'].items():
        standardized['data_residuals'][data_type] = {}
        
        for freq, raw_tensor in freqs.items():
            # Calculate points per rotation
            points_per_rotation = int(sampling_rate / float(freq))
            
            # Reshape data into rotations
            if len(raw_tensor.shape) == 2:  # [samples, variables]
                # Calculate total rotations
                total_points = raw_tensor.shape[0]
                num_rotations = min(100, total_points // points_per_rotation)
                
                # Skip if not enough data
                if num_rotations == 0:
                    continue
                
                # Extract rotations
                rotations = []
                for i in range(num_rotations):
                    start_idx = i * points_per_rotation
                    end_idx = (i + 1) * points_per_rotation
                    rotation = raw_tensor[start_idx:end_idx]
                    rotations.append(rotation)
                
                # Stack rotations
                if rotations:
                    rotations_tensor = torch.stack(rotations)
                    
                    # Compute DTW if requested
                    if compute_dtw:
                        dtw_values = torch.zeros(rotations_tensor.shape[0], rotations_tensor.shape[2])
                        for i in range(rotations_tensor.shape[0]):
                            zero_tensor = torch.zeros_like(rotations_tensor[i])
                            dtw_values[i] = compute_dtw_distances(zero_tensor, rotations_tensor[i])
                    else:
                        dtw_values = torch.zeros(rotations_tensor.shape[0], rotations_tensor.shape[2])
                    
                    # Store in standardized format
                    standardized['data_residuals'][data_type][freq] = {
                        'residuals': rotations_tensor,
                        'DTW': dtw_values
                    }
            
            elif len(raw_tensor.shape) == 3:  # [rotations, points, variables]
                # Already in rotation format, just add DTW
                if compute_dtw:
                    dtw_values = torch.zeros(raw_tensor.shape[0], raw_tensor.shape[2])
                    for i in range(raw_tensor.shape[0]):
                        zero_tensor = torch.zeros_like(raw_tensor[i])
                        dtw_values[i] = compute_dtw_distances(zero_tensor, raw_tensor[i])
                else:
                    dtw_values = torch.zeros(raw_tensor.shape[0], raw_tensor.shape[2])
                
                standardized['data_residuals'][data_type][freq] = {
                    'residuals': raw_tensor,
                    'DTW': dtw_values
                }
    
    # Process physical residuals if available
    if 'physical_residuals' in raw_residuals and raw_residuals['physical_residuals']:
        for data_type, freqs in raw_residuals['physical_residuals'].items():
            standardized['physical_residuals'][data_type] = {}
            
            for freq, raw_tensor in freqs.items():
                # Calculate points per rotation
                points_per_rotation = int(sampling_rate / float(freq))
                
                # Reshape data into rotations
                if len(raw_tensor.shape) == 2:  # [samples, variables]
                    # Calculate total rotations
                    total_points = raw_tensor.shape[0]
                    num_rotations = total_points // points_per_rotation
                    
                    # Skip if not enough data
                    if num_rotations == 0:
                        continue
                    
                    # Extract rotations
                    rotations = []
                    for i in range(num_rotations):
                        start_idx = i * points_per_rotation
                        end_idx = (i + 1) * points_per_rotation
                        rotation = raw_tensor[start_idx:end_idx]
                        rotations.append(rotation)
                    
                    # Stack rotations
                    if rotations:
                        rotations_tensor = torch.stack(rotations)
                        
                        # Compute DTW if requested
                        if compute_dtw:
                            dtw_values = torch.zeros(rotations_tensor.shape[0], rotations_tensor.shape[2])
                            for i in range(rotations_tensor.shape[0]):
                                zero_tensor = torch.zeros_like(rotations_tensor[i])
                                dtw_values[i] = compute_dtw_distances(zero_tensor, rotations_tensor[i])
                        else:
                            dtw_values = torch.zeros(rotations_tensor.shape[0], rotations_tensor.shape[2])
                        
                        # Store in standardized format
                        standardized['physical_residuals'][data_type][freq] = {
                            'residuals': rotations_tensor,
                            'DTW': dtw_values
                        }
                
                elif len(raw_tensor.shape) == 3:  # [rotations, points, variables]
                    # Already in rotation format, just add DTW
                    if compute_dtw:
                        dtw_values = torch.zeros(raw_tensor.shape[0], raw_tensor.shape[2])
                        for i in range(raw_tensor.shape[0]):
                            zero_tensor = torch.zeros_like(raw_tensor[i])
                            dtw_values[i] = compute_dtw_distances(zero_tensor, raw_tensor[i])
                    else:
                        dtw_values = torch.zeros(raw_tensor.shape[0], raw_tensor.shape[2])
                    
                    standardized['physical_residuals'][data_type][freq] = {
                        'residuals': raw_tensor,
                        'DTW': dtw_values
                    }
    
    return standardized

def load_and_standardize_residuals(filename, sampling_rate=50000, compute_dtw=False):
    """
    Load raw residuals and convert to standardized format in one step.
    
    Parameters:
    - filename: Path to HDF5 file
    - sampling_rate: Sampling rate in Hz
    - compute_dtw: Whether to compute DTW distances
    
    Returns:
    - Standardized residuals dictionary
    """
    raw_residuals = load_raw_residuals(filename)
    return standardize_raw_residuals(raw_residuals, sampling_rate, compute_dtw)

def load_legacy_residuals(filename):
    """
    Load residuals from legacy format (residuals_dict.pth)
    
    Parameters:
    - filename: Path to the legacy residuals file
    
    Returns:
    - Dictionary of legacy format residuals {data_type: tensor, ...}
    """
    if os.path.exists(filename):
        return torch.load(filename)
    else:
        raise FileNotFoundError(f"Residuals file not found: {filename}")

def standardize_legacy_residuals(legacy_residuals, sampling_rate=50000, compute_dtw=False, 
                                 freq_by_data_type=None):
    """
    Convert legacy residuals to standardized format with rotation segmentation
    
    Parameters:
    - legacy_residuals: Dictionary {data_type: tensor, ...} from legacy format
    - sampling_rate: Sampling rate in Hz
    - compute_dtw: Whether to compute DTW distances
    - freq_by_data_type: Dictionary mapping data types to frequencies
      Format: {data_type: [freq1, freq2, ...], ...}
    
    Returns:
    - Standardized residuals dictionary
    """
    # If frequencies aren't provided, try to get them from the data
    if freq_by_data_type is None:
        try:
            from Data.LoadData import data_paths, get_omegas
            freq_by_data_type = {}
            for data_type in legacy_residuals.keys():
                if data_type in data_paths:
                    omegas = get_omegas(data_paths[data_type])
                    freqs_hz = [float(omega) / (2 * np.pi) for omega in omegas]
                    freqs_hz = [round(freq, 2) for freq in freqs_hz]
                    freq_by_data_type[data_type] = freqs_hz
                else:
                    # Fallback for unknown data types
                    freq_by_data_type[data_type] = [20.0 if data_type == 'normal' else 30.0]
        except ImportError:
            print("Warning: Could not import data_paths. Using default frequencies.")
            freq_by_data_type = {data_type: [20.0 if data_type == 'normal' else 30.0] 
                                for data_type in legacy_residuals.keys()}
    
    # Define the standardized format dictionary
    standardized = {
        "data_residuals": {},
        "physical_residuals": {}
    }
    
    # Process each data type
    for data_type, residuals_tensor in tqdm(legacy_residuals.items(), 
                                           desc="Converting legacy residuals"):
        # Initialize dictionaries for this data type
        standardized["data_residuals"][data_type] = {}
        standardized["physical_residuals"][data_type] = {}
        
        # The legacy data has 4 output variables (data residuals) followed by physical residuals
        # Separate them
        data_residuals = residuals_tensor[:, :4]  # First 4 columns
        physical_residuals = residuals_tensor[:, 4:]  # Remaining columns
        
        # Get frequencies for this data type
        freqs = freq_by_data_type.get(data_type, [20.0 if data_type == 'normal' else 30.0])
        
        # For each frequency, organize into rotations
        for freq in freqs:
            # Calculate points per rotation
            points_per_rotation = int(sampling_rate / freq)
            
            # Calculate how many complete rotations we can extract
            total_points = data_residuals.shape[0]
            num_rotations = min(100, total_points // points_per_rotation)
            
            if num_rotations == 0:
                print(f"  Warning: Not enough data for even one rotation at {freq} Hz")
                continue
            
            # Process data residuals
            data_rotations = []
            for r in range(num_rotations):
                start_idx = r * points_per_rotation
                end_idx = (r + 1) * points_per_rotation
                
                if end_idx <= total_points:
                    rotation_data = data_residuals[start_idx:end_idx]
                    data_rotations.append(rotation_data)
            
            if not data_rotations:
                continue
                
            # Stack into tensor [num_rotations, points_per_rotation, variables]
            data_rotations_tensor = torch.stack(data_rotations)
            
            # Compute DTW if requested
            if compute_dtw:
                dtw_distances = torch.zeros(len(data_rotations), data_rotations[0].shape[1])
                for i, rotation in enumerate(data_rotations):
                    zero_tensor = torch.zeros_like(rotation)
                    dtw_distances[i] = compute_dtw_distances(zero_tensor, rotation)
            else:
                # Placeholder DTW
                dtw_distances = torch.zeros(len(data_rotations), data_rotations[0].shape[1])
            
            # Add to standardized format
            standardized["data_residuals"][data_type][freq] = {
                "residuals": data_rotations_tensor,
                "DTW": dtw_distances
            }
            
            # Process physical residuals
            phys_rotations = []
            for r in range(num_rotations):
                start_idx = r * points_per_rotation
                end_idx = (r + 1) * points_per_rotation
                
                if end_idx <= total_points:
                    rotation_phys = physical_residuals[start_idx:end_idx]
                    phys_rotations.append(rotation_phys)
            
            if phys_rotations:
                # Stack into tensor
                phys_rotations_tensor = torch.stack(phys_rotations)
                
                # Compute DTW if requested
                if compute_dtw:
                    phys_dtw_distances = torch.zeros(len(phys_rotations), phys_rotations[0].shape[1])
                    for i, rotation in enumerate(phys_rotations):
                        zero_tensor = torch.zeros_like(rotation)
                        phys_dtw_distances[i] = compute_dtw_distances(zero_tensor, rotation)
                else:
                    # Placeholder DTW
                    phys_dtw_distances = torch.zeros(len(phys_rotations), phys_rotations[0].shape[1])
                
                # Add to standardized format
                standardized["physical_residuals"][data_type][freq] = {
                    "residuals": phys_rotations_tensor,
                    "DTW": phys_dtw_distances
                }
    
    return standardized

def load_and_standardize_legacy(filename, sampling_rate=50000, compute_dtw=False, 
                               freq_by_data_type=None):
    """
    Load legacy residuals and convert to standardized format in one step
    
    Parameters:
    - filename: Path to legacy residuals file
    - sampling_rate: Sampling rate in Hz
    - compute_dtw: Whether to compute DTW distances
    - freq_by_data_type: Dictionary mapping data types to frequencies
    
    Returns:
    - Standardized residuals dictionary
    """
    legacy_residuals = load_legacy_residuals(filename)
    return standardize_legacy_residuals(
        legacy_residuals, sampling_rate, compute_dtw, freq_by_data_type
    ) 