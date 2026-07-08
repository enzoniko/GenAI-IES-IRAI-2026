"""
Data processing for residuals and feature extraction
"""
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional, Union
from tqdm import tqdm
import logging

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class LabeledSample:
    """
    A sample with data and hierarchical labels
    
    Attributes:
        data: The residual data
        data_type: Type of data (e.g., normal, inner, outer)
        rot_speed: Rotation speed of the sample
        label: Combined label (data_type + "_" + str(rot_speed))
    """
    data: np.ndarray
    data_type: str
    rot_speed: float
    
    @property
    def group(self) -> str:
        """Get the high-level group for this sample"""
        if self.data_type == 'normal':
            return 'normal'
        elif 'horizontal_misalignment_fault' in self.data_type:
            return 'horizontal_misalignment_fault'
        elif 'vertical_misalignment_fault' in self.data_type:
            return 'vertical_misalignment_fault'
        elif 'imbalance_fault' in self.data_type:
            return 'imbalance_fault'
        elif 'overhang_ball_fault' in self.data_type:
            return 'overhang_ball_fault'
        elif 'overhang_cage_fault' in self.data_type:
            return 'overhang_cage_fault'
        elif 'overhang_outer_race_fault' in self.data_type:
            return 'overhang_outer_race_fault'
        elif 'underhang_ball_fault' in self.data_type:
            return 'underhang_ball_fault'
        elif 'underhang_cage_fault' in self.data_type:
            return 'underhang_cage_fault'
        elif 'underhang_outer_race_fault' in self.data_type:
            return 'underhang_outer_race_fault'
        else:
            return 'unknown'
    
    @property
    def level1_label(self) -> str:
        """Get the level 1 label (group)"""
        return self.group
    
    @property
    def level2_label(self) -> str:
        """Get the level 2 label (data_type)"""
        return self.data_type
    
    @property
    def level3_label(self) -> str:
        """Get the level 3 label (data_type_speed)"""
        return f"{self.data_type}_{self.rot_speed:.2f}"
    
    @property
    def label(self) -> str:
        """Get the combined label for this sample (same as level3_label)"""
        return self.level3_label


class ResidualProcessor:
    """Process residuals with either simple FFT representations or advanced features"""
    
    def __init__(self, fft_length=128, num_channels=4, advanced_features=False,
                 sampling_rate=50000, include_tsfresh=False, wavelet='db4', wavelet_level=3,
                 max_sequence_length=None, num_workers=None, batch_size=10):
        """
        Process residuals to fixed-length feature representations
        
        Args:
            fft_length: Length of FFT output to keep (will take first N frequencies)
            num_channels: Number of channels in the residual data
            advanced_features: Whether to use advanced feature extraction from v1
            sampling_rate: Sampling rate in Hz (for advanced features)
            include_tsfresh: Whether to include tsfresh features (for advanced features)
            wavelet: Wavelet type to use (for advanced features)
            wavelet_level: Wavelet decomposition level (for advanced features)
            max_sequence_length: Maximum sequence length for padding (auto-detect if None)
            num_workers: Number of worker processes for parallel feature extraction
            batch_size: Batch size for parallel processing
        """
        self.fft_length = fft_length
        self.num_channels = num_channels
        self.advanced_features = advanced_features
        self.sampling_rate = sampling_rate
        self.include_tsfresh = include_tsfresh
        self.wavelet = wavelet
        self.level = wavelet_level
        self.max_sequence_length = max_sequence_length
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.scaler = StandardScaler()
        self.fitted = False
        
        # Load advanced feature extraction if needed
        if advanced_features:
            from .parallel_features import parallel_extract_features
            self.parallel_extract_features = parallel_extract_features
    
    def calculate_max_sequence_length(self, residuals_dict):
        """
        Calculate the maximum sequence length based on the slowest rotation speed
        
        Args:
            residuals_dict: Dictionary of residuals (either original or processed)
            
        Returns:
            Maximum sequence length calculated from the slowest rotation speed
        """
        # Try to use direct get_omegas to find the rotation speeds
        try:
            from Data.LoadData import data_paths, get_omegas
            
            # Collect all rotation speeds
            all_rotation_speeds = []
            
            print("Calculating maximum sequence length based on rotation speeds...")
            
            # Get rotation speeds for each data type using get_omegas
            for data_type in residuals_dict.keys():
                if data_type in data_paths:
                    file_path = data_paths[data_type]
                    
                    # Get rotation speeds in Hz from omegas
                    omegas = get_omegas(file_path)
                    rot_speeds = [float(omega) / (2 * np.pi) for omega in omegas]
                    
                    # Add to collection of speeds
                    all_rotation_speeds.extend(rot_speeds)
            
            # Find the slowest rotation speed
            if all_rotation_speeds:
                slowest_rotation = min(all_rotation_speeds)
                print(f"Slowest rotation speed: {slowest_rotation} Hz")
                
                # Calculate points per rotation for the slowest speed
                points_per_rotation = int(np.ceil(self.sampling_rate / slowest_rotation))
                
                print(f"Maximum sequence length: {points_per_rotation} points")
                return points_per_rotation
            else:
                print("No rotation speeds found, using default length of 5000 points")
                return 5000
                
        except (ImportError, KeyError) as e:
            print(f"Error accessing rotation speeds: {e}")
            print("Using default length of 5000 points")
            return 5000
    
    def extract_data(self, sample: Union[Dict, np.ndarray, torch.Tensor]) -> np.ndarray:
        """Extract data from sample dictionary if needed"""
        if isinstance(sample, dict) and 'data' in sample:
            data = sample['data']
        elif isinstance(sample, LabeledSample):
            data = sample.data
        else:
            data = sample
            
        # Convert to numpy if needed
        if hasattr(data, 'cpu'):
            data = data.cpu().numpy()
        return np.asarray(data)
    
    def process_sample(self, sample: Union[Dict, np.ndarray, torch.Tensor]) -> np.ndarray:
        """Process a single sample to feature representation"""
        data = self.extract_data(sample)
        
        # Ensure we have 2D data (time, channels)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        if not self.advanced_features:
            # Simple v0 FFT-based processing
            # Apply FFT to each channel
            fft_features = []
            for i in range(min(data.shape[1], self.num_channels)):
                # Compute FFT magnitude
                fft_vals = np.abs(np.fft.rfft(data[:, i]))
                # Take first N components
                if len(fft_vals) >= self.fft_length:
                    fft_features.append(fft_vals[:self.fft_length])
                else:
                    # Zero-pad if too short
                    padding = np.zeros(self.fft_length - len(fft_vals))
                    fft_features.append(np.concatenate([fft_vals, padding]))
            
            # If we have fewer channels than expected, pad with zeros
            while len(fft_features) < self.num_channels:
                fft_features.append(np.zeros(self.fft_length))
            
            # Stack channels
            return np.stack(fft_features, axis=0)
        else:
            # Advanced feature extraction from v1
            # Import the feature extractors here to avoid circular imports
            from .feature_extractors import extract_features_combined, extract_v0_features
            
            # Extract features using the combined extractor (uses max_sequence_length)
            features = extract_features_combined(
                data,
                sample_rate=self.sampling_rate,
                wavelet=self.wavelet,
                level=self.level,
                include_tsfresh=self.include_tsfresh,
                max_length=self.max_sequence_length
            )
            
            return features
    
    def fit_transform(self, samples: List[Union[Dict, np.ndarray]]) -> np.ndarray:
        """Process a list of samples and fit the scaler"""
        if not self.advanced_features:
            # Simple v0 processing (process each sample individually)
            processed = [self.process_sample(sample) for sample in tqdm(samples, desc="Processing samples")]
            # Reshape for scaler: (n_samples, n_channels, fft_length) -> (n_samples, n_channels*fft_length)
            flat_data = np.array([p.flatten() for p in processed])
            
            # Fit and transform
            normalized = self.scaler.fit_transform(flat_data)
            self.fitted = True
            
            # Reshape back to original format
            return normalized.reshape(-1, self.num_channels, self.fft_length)
        else:
            # Advanced processing with parallel features
            print("Using advanced feature extraction from v1 with multiprocessing")
            # Extract data from samples
            sample_data = [self.extract_data(sample) for sample in samples]
            
            # Calculate max sequence length if not provided
            if self.max_sequence_length is None:
                # Get the maximum length from the data
                self.max_sequence_length = max([data.shape[0] for data in sample_data])
                print(f"Auto-detected max sequence length from data: {self.max_sequence_length}")
            
            # Process all samples at once using parallel processing
            processed = self.parallel_extract_features(
                sample_data,
                sampling_rate=self.sampling_rate,
                wavelet=self.wavelet,
                level=self.level,
                include_tsfresh=self.include_tsfresh,
                max_workers=self.num_workers,
                batch_size=self.batch_size,
                max_length=self.max_sequence_length,
                use_v0_processing=False
            )
            
            # Reshape for scaler
            original_shape = processed.shape
            flat_data = processed.reshape(processed.shape[0], -1)
            
            # Fit and transform
            normalized = self.scaler.fit_transform(flat_data)
            self.fitted = True
            
            # Reshape back to original format
            return normalized.reshape(original_shape)
    
    def transform(self, samples: List[Union[Dict, np.ndarray]]) -> np.ndarray:
        """Process a list of samples using pre-fitted scaler"""
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit_transform first.")
        
        if not self.advanced_features:
            # Simple v0 processing
            processed = [self.process_sample(sample) for sample in tqdm(samples, desc="Processing samples")]
            flat_data = np.array([p.flatten() for p in processed])
            normalized = self.scaler.transform(flat_data)
            return normalized.reshape(-1, self.num_channels, self.fft_length)
        else:
            # Advanced processing with parallel features
            print("Using advanced feature extraction from v1 with multiprocessing")
            # Extract data from samples
            sample_data = [self.extract_data(sample) for sample in samples]
            
            # Process all samples at once using parallel processing
            processed = self.parallel_extract_features(
                sample_data,
                sampling_rate=self.sampling_rate,
                wavelet=self.wavelet,
                level=self.level,
                include_tsfresh=self.include_tsfresh,
                max_workers=self.num_workers,
                batch_size=self.batch_size,
                max_length=self.max_sequence_length,
                use_v0_processing=False
            )
            
            # Reshape for scaler
            original_shape = processed.shape
            flat_data = processed.reshape(processed.shape[0], -1)
            
            # Apply transformation
            normalized = self.scaler.transform(flat_data)
            
            # Reshape back to original format
            return normalized.reshape(original_shape)


def organize_residuals_by_label(residuals_dict: Dict[str, List[Dict]]) -> Dict[str, List[LabeledSample]]:
    """
    Organize residuals by combined labels (data_type_rot_speed)
    
    Args:
        residuals_dict: Dictionary mapping data types to lists of samples
        
    Returns:
        Dictionary mapping combined labels to lists of LabeledSample objects
    """
    labeled_samples = {}
    
    for data_type, samples in residuals_dict.items():
        for sample in samples:
            # Extract rotation speed
            rot_speed = sample.get('rot_speed', 0.0)
            
            # Create labeled sample
            labeled_sample = LabeledSample(
                data=sample['data'] if 'data' in sample else sample,
                data_type=data_type,
                rot_speed=rot_speed
            )
            
            # Add to dict by level3_label (full label)
            if labeled_sample.level3_label not in labeled_samples:
                labeled_samples[labeled_sample.level3_label] = []
            labeled_samples[labeled_sample.level3_label].append(labeled_sample)
    
    return labeled_samples


def preprocess_residuals(residuals_dict: Dict[str, Any], source: str, include_physical: bool = False) -> Dict[str, List[LabeledSample]]:
    """
    Load and preprocess residuals based on the source
    
    Args:
        residuals_dict: Dictionary of residuals
        source: Source type ('direct' or 'hybrid')
        include_physical: Whether to include physical residuals
    
    Returns:
        Organized dictionary mapping labels to samples
    """
    if source == "direct":

            
        from Data.LoadData import data_paths, get_omegas
        
        # Track data residuals for potential merging with physical residuals
        data_residuals = {}
        physical_residuals = {}
        
        # Check if we have both data and physical residuals
        has_data_residuals = "data_residuals" in residuals_dict
        has_physical_residuals = "physical_residuals" in residuals_dict and include_physical
        
        if has_data_residuals and has_physical_residuals:
            # Process both data and physical residuals together
            print("Processing both data and physical residuals")
            processed_dict = {}
            
            # Process data residuals
            for data_type, data in residuals_dict["data_residuals"].items():
                file_path = data_paths[data_type]
                # Compute rotation speeds (e.g., in Hz) from file metadata
                omegas = get_omegas(file_path) / (2 * np.pi)
                num_rotations = 1
                sampling_rate = 50000  # 50kHz
                datapoints_per_rotation = sampling_rate / omegas
                datapoints_needed = np.ceil(num_rotations * datapoints_per_rotation).to(torch.int32)
                n_blocks = len(data) // 250000
                
                # Store data segments for merging with physical data
                data_segments = {}
                for i in range(n_blocks):
                    seg_length = int(datapoints_needed[i]) if i < len(datapoints_needed) else 200
                    if data_type == 'normal':
                        max_segments = min(2, 250000 // seg_length)
                    else:
                        max_segments = min(2, 250000 // seg_length)
                    
                    for j in range(max_segments):
                        start_idx = i * 250000 + j * seg_length
                        end_idx = start_idx + seg_length
                        if end_idx <= (i + 1) * 250000:
                            segment = data[start_idx:end_idx]
                            if len(segment) > 0:
                                # Create a key for this segment
                                key = (data_type, float(omegas[i]), i, j)
                                data_segments[key] = segment
                
                # Store for potential merging
                data_residuals[data_type] = {
                    'segments': data_segments,
                    'omegas': omegas
                }
            
            # Process physical residuals
            if "physical_residuals" in residuals_dict:
                for data_type, data in residuals_dict["physical_residuals"].items():
                    file_path = data_paths[data_type]
                    # Compute rotation speeds (e.g., in Hz) from file metadata
                    omegas = get_omegas(file_path) / (2 * np.pi)
                    num_rotations = 1
                    sampling_rate = 50000  # 50kHz
                    datapoints_per_rotation = sampling_rate / omegas
                    datapoints_needed = np.ceil(num_rotations * datapoints_per_rotation).to(torch.int32)
                    n_blocks = len(data) // 250000
                    
                    # Store physical segments for merging with data
                    physical_segments = {}
                    for i in range(n_blocks):
                        seg_length = int(datapoints_needed[i]) if i < len(datapoints_needed) else 200
                        if data_type == 'normal':
                            max_segments = min(2, 250000 // seg_length)
                        else:
                            max_segments = min(2, 250000 // seg_length)
                        
                        for j in range(max_segments):
                            start_idx = i * 250000 + j * seg_length
                            end_idx = start_idx + seg_length
                            if end_idx <= (i + 1) * 250000:
                                segment = data[start_idx:end_idx]
                                if len(segment) > 0:
                                    # Create a key for this segment
                                    key = (data_type, float(omegas[i]), i, j)
                                    physical_segments[key] = segment
                    
                    # Store for potential merging
                    physical_residuals[data_type] = {
                        'segments': physical_segments,
                        'omegas': omegas
                    }
            
            # Merge data and physical residuals
            for data_type in data_residuals:
                if data_type not in processed_dict:
                    processed_dict[data_type] = []
                
                # Get segments from both data and physical residuals
                data_segments = data_residuals[data_type]['segments']
                
                # Check if we have physical residuals for this data type
                if data_type in physical_residuals:
                    physical_segments = physical_residuals[data_type]['segments']
                    
                    # Merge segments with the same key
                    for key in data_segments:
                        if key in physical_segments:
                            # Combine data and physical residuals
                            data_segment = data_segments[key]
                            physical_segment = physical_segments[key]
                            
                            # Ensure same number of points
                            min_points = min(len(data_segment), len(physical_segment))
                            data_segment = data_segment[:min_points]
                            physical_segment = physical_segment[:min_points]
                            
                            # Concatenate along the channel dimension
                            data_channels = data_segment.shape[1] if data_segment.ndim > 1 else 1
                            phys_channels = physical_segment.shape[1] if physical_segment.ndim > 1 else 1
                            
                            # Reshape if needed
                            if data_segment.ndim == 1:
                                data_segment = data_segment.reshape(-1, 1)
                            if physical_segment.ndim == 1:
                                physical_segment = physical_segment.reshape(-1, 1)
                            
                            # Combine the segments
                            combined_segment = torch.cat([data_segment, physical_segment], dim=1)
                            
                            # Add to processed dict
                            data_type, rot_speed, _, _ = key
                            processed_dict[data_type].append({
                                'data': combined_segment,
                                'rot_speed': rot_speed
                            })
                        else:
                            # Only data residuals available for this key
                            data_type, rot_speed, _, _ = key
                            processed_dict[data_type].append({
                                'data': data_segments[key],
                                'rot_speed': rot_speed
                            })
                else:
                    # Only data residuals available for this data type
                    for key, segment in data_segments.items():
                        data_type, rot_speed, _, _ = key
                        processed_dict[data_type].append({
                            'data': segment,
                            'rot_speed': rot_speed
                        })
            
            # Convert processed_dict to the original format expected by organize_residuals_by_label
            return organize_residuals_by_label(processed_dict)
        
        else:
            for key in residuals_dict:
                if key == "data_residuals" or key == "physical_residuals":
                    continue
                
                print(f"Original length for {key}: {len(residuals_dict[key])}")
                file_path = data_paths[key]
                # Compute rotation speeds (e.g., in Hz) from file metadata
                omegas = get_omegas(file_path) / (2 * np.pi)
                num_rotations = 1
                sampling_rate = 50000  # 50kHz
                datapoints_per_rotation = sampling_rate / omegas
                datapoints_needed = np.ceil(num_rotations * datapoints_per_rotation).to(torch.int32)
                n_blocks = min(len(residuals_dict[key])// 250000, 48)
                print("NBLOCKS", n_blocks)
                segments = []
                for i in range(n_blocks):
                    seg_length = int(datapoints_needed[i]) if i < len(datapoints_needed) else 200
                    # For "normal" samples, take up to min(30, floor(250000 / seg_length)) segments; for others, only 10 segments.
                    if key == 'normal':
                        max_segments = min(100, 250000 // seg_length)
                    else:
                        max_segments = min(100, 250000 // seg_length)
                    for j in range(max_segments):
                        start_idx = i * 250000 + j * seg_length
                        end_idx = start_idx + seg_length
                        if end_idx <= (i + 1) * 250000:
                            if include_physical:
                                segment = residuals_dict[key][start_idx:end_idx][:, :8]
                            else:
                                segment = residuals_dict[key][start_idx:end_idx][:, :4]
                            if len(segment) > 0:
                                segments.append({'data': segment, 'rot_speed': float(omegas[i])})
                residuals_dict[key] = segments
                print(f"Segmented into {len(segments)} samples for {key}")
    
    elif source == "hybrid":
        # Hybrid RNN residuals are already segmented with rotation speed information
        processed = {}
        # Check if we have both data and physical residuals
        has_data_residuals = "data_residuals" in residuals_dict
        has_physical_residuals = "physical_residuals" in residuals_dict and include_physical
        
        if has_data_residuals and has_physical_residuals:
            # Process both data and physical residuals together
            print("Processing both data and physical residuals for hybrid model")
            
            # Process data residuals
            data_samples = {}
            for data_type, sequences in residuals_dict["data_residuals"].items():
                for seq_idx, seq_data in sequences.items():
                    # Extract the data and rotation speed from the sequence
                    data = seq_data['data'].numpy() if hasattr(seq_data['data'], 'numpy') else seq_data['data']
                    rot_speed = seq_data.get('rot_speed', 0.0)
                    
                    # Store for potential merging with physical data
                    key = (data_type, rot_speed, seq_idx)
                    data_samples[key] = data
            
            # Process physical residuals
            physical_samples = {}
            for data_type, sequences in residuals_dict["physical_residuals"].items():
                for seq_idx, seq_data in sequences.items():
                    # Extract the data and rotation speed from the sequence
                    data = seq_data['data'].numpy() if hasattr(seq_data['data'], 'numpy') else seq_data['data']
                    rot_speed = seq_data.get('rot_speed', 0.0)
                    
                    # Store for potential merging with data residuals
                    key = (data_type, rot_speed, seq_idx)
                    physical_samples[key] = data
            
            # Merge data and physical residuals
            for key in data_samples:
                data_type, rot_speed, seq_idx = key
                data_residual = data_samples[key]
                
                if key in physical_samples:
                    # Both data and physical residuals exist for this key
                    physical_residual = physical_samples[key]
                    
                    # Ensure same number of points
                    min_points = min(len(data_residual), len(physical_residual))
                    data_residual = data_residual[:min_points]
                    physical_residual = physical_residual[:min_points]
                    
                    # Concatenate along the channel/variable dimension
                    if data_residual.ndim == 1:
                        data_residual = data_residual.reshape(-1, 1)
                    if physical_residual.ndim == 1:
                        physical_residual = physical_residual.reshape(-1, 1)
                    
                    combined_data = np.concatenate([data_residual, physical_residual], axis=1)
                    
                    # Create labeled sample with combined data
                    labeled_sample = LabeledSample(
                        data=combined_data,
                        data_type=data_type,
                        rot_speed=rot_speed
                    )
                else:
                    # Only data residuals available
                    labeled_sample = LabeledSample(
                        data=data_residual,
                        data_type=data_type,
                        rot_speed=rot_speed
                    )
                
                # Add to processed dictionary
                if labeled_sample.level3_label not in processed:
                    processed[labeled_sample.level3_label] = []
                processed[labeled_sample.level3_label].append(labeled_sample)
            
            return processed
        else:
            # Original processing without physical residuals
            for data_type, sequences in residuals_dict.items():
                if data_type == "data_residuals" or data_type == "physical_residuals":
                    continue
                
                processed_sequences = []
                for seq_idx, seq_data in sequences.items():
                    # Extract the data and rotation speed from the sequence
                    data = seq_data['data'].numpy() if hasattr(seq_data['data'], 'numpy') else seq_data['data']
                    rot_speed = seq_data.get('rot_speed', 0.0)
                    
                    # Create labeled sample
                    labeled_sample = LabeledSample(
                        data=data,
                        data_type=data_type,
                        rot_speed=rot_speed
                    )
                    
                    # Add to dict by level3_label (full label)
                    if labeled_sample.level3_label not in processed:
                        processed[labeled_sample.level3_label] = []
                    processed[labeled_sample.level3_label].append(labeled_sample)
        
            return processed
    
    # Organize by label
    return organize_residuals_by_label(residuals_dict) 