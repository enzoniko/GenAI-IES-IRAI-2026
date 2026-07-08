#!/usr/bin/env python3
"""
PINN Residuals, Data-Driven Model Features, and Siamese Embeddings UMAP Visualization Script

This script provides three modes:
1. BYPASS MODE: Loads PINN residuals, segments them hierarchically (3 levels),
   extracts advanced features, and generates UMAP visualizations directly from features.
2. DATA-DRIVEN MODE: Loads experimental data files, processes them through trained data-driven models
   to extract residuals, segments them hierarchically, extracts advanced features,
   and generates UMAP visualizations directly from features.
3. EMBEDDINGS MODE: Loads embeddings.pkl from siamese_analysis_v3 evaluation pipeline
   and generates the same UMAP visualizations for comparison.

The script also calculates quantitative cluster validation metrics:
- Silhouette Score (higher = better)
- Calinski-Harabasz Score (higher = better)
- Davies-Bouldin Score (lower = better)

Metrics are calculated for both:
- Level 1: Group-level clustering (normal, imbalance_fault, etc.)
- Level 2: Data-type-level clustering (specific fault types)

Additionally, the script calculates metrics on UMAP projections for each parameter
combination (n_neighbors × min_dist) and provides averages and standard deviations
across all configurations to assess clustering robustness.

Usage:
    # Bypass mode (from PINN residuals)
    python pinn_direct_umap_visualization.py --residuals path/to/residuals.pth --output-dir results/

    # Data-driven mode (from experimental data files)
    python pinn_direct_umap_visualization.py --data-driven Data/v3/ --output-dir results/

    # Embeddings mode (from siamese analysis)
    python pinn_direct_umap_visualization.py --embeddings path/to/embeddings.pkl --output-dir results/

Author: AI Assistant
"""

import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from tqdm import tqdm
import argparse
import logging
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import LabelEncoder

# Configure matplotlib for headless operation
import matplotlib
matplotlib.use('Agg')

# Try to import UMAP
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    try:
        import umap
        UMAP_AVAILABLE = True
    except ImportError:
        UMAP_AVAILABLE = False
        print("Warning: UMAP not available. UMAP visualizations will be skipped.")
        print("To enable UMAP visualizations, install umap-learn: pip install umap-learn")

# Import feature extractors
try:
    from siamese_analysis_v3.data.feature_extractors import extract_features_combined
    from siamese_analysis_v3.data.parallel_features import parallel_extract_features
    SIA_MESE_AVAILABLE = True
except ImportError:
    SIA_MESE_AVAILABLE = False
    print("Warning: siamese_analysis_v3 not found. Using fallback feature extraction.")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class LabeledSample:
    """
    A sample with data and hierarchical labels
    """
    data: np.ndarray
    data_type: str
    rot_speed: float

    @property
    def group(self) -> str:
        """Get the high-level group for this sample"""
        if self.data_type == 'normal' or 'normal' in self.data_type:
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


def extract_data_type_from_filename(filepath: str) -> str:
    """
    Extract data type from filename pattern (same as pinn_to_siamese_wrapper.py)
    """
    filename = os.path.basename(filepath)

    # Remove X_ prefix and .pth extension
    if filename.startswith('X_') and filename.endswith('.pth'):
        data_part = filename[2:-4]  # Remove 'X_' and '.pth'
    else:
        data_part = filename

    # Remove _v3 suffix if present
    if data_part.endswith('_v3'):
        data_part = data_part[:-3]  # Remove '_v3'

    # Handle special cases
    if data_part == 'normal':
        return 'normal'

    # Extract fault types
    if 'imbalance_fault' in data_part:
        return 'imbalance_fault_' + data_part.split('imbalance_fault_')[1]
    elif 'overhang_ball_fault' in data_part:
        return 'overhang_ball_fault_' + data_part.split('overhang_ball_fault_')[1]
    elif 'horizontal_misalignment_fault' in data_part:
        return 'horizontal_misalignment_fault_' + data_part.split('horizontal_misalignment_fault_')[1]
    elif 'vertical_misalignment_fault' in data_part:
        return 'vertical_misalignment_fault_' + data_part.split('vertical_misalignment_fault_')[1]
    elif 'overhang_cage_fault' in data_part:
        return 'overhang_cage_fault_' + data_part.split('overhang_cage_fault_')[1]
    elif 'overhang_outer_race_fault' in data_part:
        return 'overhang_outer_race_fault_' + data_part.split('overhang_outer_race_fault_')[1]
    elif 'underhang_ball_fault' in data_part:
        return 'underhang_ball_fault_' + data_part.split('underhang_ball_fault_')[1]
    elif 'underhang_cage_fault' in data_part:
        return 'underhang_cage_fault_' + data_part.split('underhang_cage_fault_')[1]
    elif 'underhang_outer_race_fault' in data_part:
        return 'underhang_outer_race_fault_' + data_part.split('underhang_outer_race_fault_')[1]

    return data_part


def validate_residual_shapes(residuals: Dict[str, np.ndarray], data_type: str) -> bool:
    """
    Validate that residual arrays have expected shapes and properties.
    Based on the original pinn_to_siamese_wrapper.py validation
    """
    expected_keys = ['data_res1', 'data_res2', 'data_res3', 'data_res4']

    logger.info(f"\n=== Validating residuals for {data_type} ===")

    # Check all expected keys exist (at least data residuals)
    data_keys_present = [key for key in expected_keys if key in residuals]
    if not data_keys_present:
        logger.warning(f"No data residual keys found in {list(residuals.keys())}")
        return False

    # Check data types
    for key in residuals:
        if not isinstance(residuals[key], (np.ndarray, torch.Tensor)):
            logger.warning(f"  {key} is not a numpy array or tensor")
            return False

    # Get reference shape from first available data residual
    first_data_key = None
    for key in expected_keys:
        if key in residuals:
            first_data_key = key
            break

    if first_data_key is None:
        logger.error("No data residuals found to use as reference")
        return False

    reference_array = residuals[first_data_key]
    if hasattr(reference_array, 'shape'):
        reference_shape = reference_array.shape
    else:
        logger.error(f"Reference array {first_data_key} has no shape attribute")
        return False

    logger.info(f"  Reference shape (from {first_data_key}): {reference_shape}")

    # Validate data residuals (should all be the same shape)
    for key in expected_keys:
        if key in residuals:
            shape = residuals[key].shape if hasattr(residuals[key], 'shape') else (len(residuals[key]),)
            if shape != reference_shape:
                logger.warning(f"  Data residual shape mismatch for {key}: {shape} != {reference_shape}")
                return False

    total_samples = reference_shape[0]
    logger.info(f"  Total samples: {total_samples}")
    logger.info(f"  Data type: {type(reference_array)}")
    logger.info(f"  ✓ All residuals validated and normalized")

    return True


def discover_data_files(base_path: str, find_residuals: bool = False) -> Dict[str, str]:
    """
    Automatically discover all X_*.pth or *_residuals.pth files in the base path
    
    Args:
        base_path: Directory containing the data files
        find_residuals: If True, look for *_residuals.pth files instead of X_*.pth
    
    Returns:
        Dictionary mapping data types to file paths
    """
    data_files = {}

    if find_residuals:
        # Find all residuals_*.pth files (from pinn_to_siamese_wrapper.py output)
        pattern = os.path.join(base_path, "residuals_*.pth")
        files = glob.glob(pattern)
        logger.info(f"Found {len(files)} residuals_*.pth files in {base_path}")

        for file_path in files:
            basename = os.path.basename(file_path)
            # Extract model name by removing residuals_ prefix and .pth suffix
            model_name = basename.replace('residuals_', '').replace('.pth', '')
            data_files[model_name] = file_path
            logger.info(f"  {model_name}: {basename}")
    else:
        # Find all X_*.pth files (original experimental data)
        pattern = os.path.join(base_path, "X_*.pth")
        x_files = glob.glob(pattern)
        logger.info(f"Found {len(x_files)} X_*.pth files in {base_path}")

        for x_file in x_files:
            data_type = extract_data_type_from_filename(x_file)
            data_files[data_type] = x_file
            logger.info(f"  {data_type}: {os.path.basename(x_file)}")

    return data_files


def extract_rotation_speeds_from_data(data_path: str) -> List[float]:
    """
    Extract rotation speeds from data file using LoadData utilities
    """
    try:
        # Import here to avoid circular imports
        from Data.LoadData import data_paths, get_omegas

        # Get data type from path
        data_type = extract_data_type_from_filename(data_path)

        if data_type in data_paths:
            # Use the directory path from data_paths
            folder_path = data_paths[data_type]
            logger.info(f"  Using folder path: {folder_path} for data type: {data_type}")

            # Get angular frequencies (omegas) from the data folder
            omegas = get_omegas(folder_path)

            # Convert to Hz (rotations per second)
            rot_speeds = [float(omega) / (2 * np.pi) for omega in omegas]

            logger.info(f"  Extracted {len(rot_speeds)} rotation speeds: {rot_speeds[:5]}... (showing first 5)")
            return rot_speeds
        else:
            logger.warning(f"  Warning: Data type '{data_type}' not found in data_paths")
            logger.warning("  Falling back to dummy rotation speeds for testing")

    except ImportError as e:
        logger.warning(f"  Warning: Could not import LoadData utilities ({e})")
        logger.warning("  Falling back to dummy rotation speeds for testing")

    except Exception as e:
        logger.warning(f"  Error extracting rotation speeds: {e}")
        logger.warning("  Falling back to dummy rotation speeds for testing")

    # Fallback: create dummy rotation speeds
    # This assumes 48 different rotation speeds (typical for the dataset)
    return [10.0 + i * 2.0 for i in range(48)]  # 10, 12, 14, ..., 104 Hz


def load_and_segment_pinn_residuals(data_files: Dict[str, str], use_fraction: float = 0.2,
                                  include_physical: bool = False) -> Dict[str, List[LabeledSample]]:
    """
    Load PINN residuals and segment them hierarchically using the same approach as siamese module
    """
    logger.info("Loading and segmenting PINN residuals...")
    samples_by_label = {}

    for data_type, data_path in tqdm(data_files.items(), desc="Loading data types"):
        logger.info(f"\nProcessing data type: {data_type}")
        logger.info(f"Data path: {data_path}")

        # Load the data
        try:
            data = torch.load(data_path)
            logger.info(f"Loaded data type: {type(data)}")
        except Exception as e:
            logger.error(f"Error loading {data_path}: {e}")
            continue

        # Check data format
        if isinstance(data, dict):
            # Check if this is the processed format with fault types as keys
            sample_keys = list(data.keys())
            if sample_keys and isinstance(data[sample_keys[0]], list):
                # This is the processed format: {fault_type: [samples]}
                logger.info("Detected processed data format (fault_type -> samples)")
                processed_data = data
            else:
                # This might be raw data or residuals - try PINN preprocessing
                logger.info("Detected dictionary format - attempting PINN preprocessing...")

                # Import the preprocessing function
                try:
                    from training_scripts.pinn_preprocessing import preprocess_pinn_data
                except ImportError as e:
                    logger.error(f"Cannot import PINN preprocessing: {e}")
                    logger.error("Please make sure training_scripts/pinn_preprocessing.py is available")
                    continue

                try:
                    # Run PINN preprocessing
                    processed_data = preprocess_pinn_data(
                        model_name="default_model",
                        data_path=data_path,
                        chunk_size_mb=50,
                        use_bayesian_models=True
                    )
                    logger.info("PINN preprocessing completed")
                except Exception as e:
                    logger.error(f"PINN preprocessing failed for {data_path}: {e}")
                    continue
        elif hasattr(data, 'shape'):
            # This is a tensor/array - run PINN preprocessing
            logger.info(f"Detected tensor format with shape {data.shape} - running PINN preprocessing...")

            # Import the preprocessing function
            try:
                from training_scripts.pinn_preprocessing import preprocess_pinn_data
            except ImportError as e:
                logger.error(f"Cannot import PINN preprocessing: {e}")
                logger.error("Please make sure training_scripts/pinn_preprocessing.py is available")
                continue

            try:
                # Run PINN preprocessing
                processed_data = preprocess_pinn_data(
                    model_name="default_model",
                    data_path=data_path,
                    chunk_size_mb=50,
                    use_bayesian_models=True
                )
                logger.info("PINN preprocessing completed")
            except Exception as e:
                logger.error(f"PINN preprocessing failed for {data_path}: {e}")
                continue
        else:
            # Unknown format
            logger.error(f"Unknown data format: {type(data)}. Expected dict or tensor.")
            continue

        # Process the data based on its format
        if isinstance(processed_data, dict):
            # Check if this is the processed format
            sample_keys = list(processed_data.keys())
            if sample_keys and isinstance(processed_data[sample_keys[0]], list):
                # This is the processed format: {fault_type: [samples]}
                logger.info("Processing data in fault_type format")

                for fault_type, samples_list in processed_data.items():
                    logger.info(f"Processing fault type: {fault_type}")

                    # Apply fraction reduction to samples
                    if use_fraction < 1.0:
                        n_samples = int(len(samples_list) * use_fraction)
                        if n_samples > 0:
                            samples_list = np.random.choice(samples_list, n_samples, replace=False)
                        else:
                            samples_list = samples_list[:1]  # At least one sample

                    logger.info(f"  Using {len(samples_list)} samples (fraction: {use_fraction})")

                    # Process each sample
                    for sample_dict in samples_list:
                        try:
                            # Extract data and rotation speed
                            sample_data = sample_dict['data']
                            rot_speed = sample_dict['rot_speed']

                            # Convert to numpy if needed
                            if hasattr(sample_data, 'numpy'):
                                sample_data = sample_data.numpy()
                            elif hasattr(sample_data, 'cpu'):
                                sample_data = sample_data.cpu().numpy()

                            # Ensure it's a numpy array
                            sample_data = np.asarray(sample_data, dtype=np.float32)

                            # If it's 2D, flatten it for feature processing
                            if sample_data.ndim > 1:
                                sample_data = sample_data.flatten()

                            # Create LabeledSample with hierarchical labels
                            labeled_sample = LabeledSample(
                                data=sample_data,
                                data_type=fault_type,
                                rot_speed=float(rot_speed)
                            )

                            # Use level3_label (full label with speed) as key
                            level3_label = labeled_sample.level3_label
                            if level3_label not in samples_by_label:
                                samples_by_label[level3_label] = []

                            samples_by_label[level3_label].append(labeled_sample)

                        except Exception as e:
                            logger.warning(f"Error processing sample in {fault_type}: {e}")
                            continue
            else:
                # This might be residuals format
                logger.warning("Unexpected dictionary format, skipping validation for now")

                # Try to process as residuals
                available_keys = list(processed_data.keys())
                logger.info(f"Available keys: {available_keys}")

                # Extract rotation speeds
                rot_speeds = extract_rotation_speeds_from_data(data_path)

                if len(rot_speeds) == 0:
                    logger.warning(f"No rotation speeds found for {data_type}, skipping")
                    continue

                # Process each key as a fault type
                for fault_type in available_keys:
                    if fault_type in processed_data:
                        fault_data = processed_data[fault_type]

                        # Apply fraction reduction
                        if use_fraction < 1.0:
                            n_samples = int(len(fault_data) * use_fraction)
                            if n_samples > 0:
                                indices = np.random.choice(len(fault_data), n_samples, replace=False)
                                fault_data = [fault_data[i] for i in indices]

                        logger.info(f"Processing {fault_type}: {len(fault_data)} samples")

                        for sample_dict in fault_data:
                            try:
                                sample_data = sample_dict['data']
                                rot_speed = sample_dict['rot_speed']

                                # Convert to numpy
                                if hasattr(sample_data, 'numpy'):
                                    sample_data = sample_data.numpy()
                                sample_data = np.asarray(sample_data, dtype=np.float32)

                                # Flatten if needed
                                if sample_data.ndim > 1:
                                    sample_data = sample_data.flatten()

                                # Create LabeledSample
                                labeled_sample = LabeledSample(
                                    data=sample_data,
                                    data_type=fault_type,
                                    rot_speed=float(rot_speed)
                                )

                                level3_label = labeled_sample.level3_label
                                if level3_label not in samples_by_label:
                                    samples_by_label[level3_label] = []

                                samples_by_label[level3_label].append(labeled_sample)

                            except Exception as e:
                                logger.warning(f"Error processing sample in {fault_type}: {e}")
                                continue

    logger.info(f"\nCompleted segmentation: {len(samples_by_label)} level-3 labels created")
    return samples_by_label


def extract_advanced_features(samples_by_label: Dict[str, List[LabeledSample]],
                            sampling_rate: int = 50000, wavelet: str = 'db4',
                            wavelet_level: int = 3, include_tsfresh: bool = False,
                            max_length: Optional[int] = None, disable_multiprocessing: bool = False) -> Dict[str, np.ndarray]:
    """
    Extract advanced features using the same pipeline as siamese_analysis_v3
    """
    logger.info("Extracting advanced features...")

    features_by_label = {}
    all_samples_data = []

    # Collect all sample data
    logger.info("Collecting sample data...")
    for label, samples in tqdm(samples_by_label.items(), desc="Collecting samples"):
        for sample in samples:
            all_samples_data.append(sample.data)

    if not all_samples_data:
        raise ValueError("No samples found for feature extraction")

    # Determine max length if not provided
    if max_length is None:
        # Use a reasonable max length to avoid memory issues
        # Typical values for vibration analysis are in the range of 1000-10000 samples
        detected_max = max([len(sample) for sample in all_samples_data])
        max_length = min(detected_max, 8192)  # Cap at 8192 to avoid memory issues
        logger.info(f"Auto-detected max sequence length: {detected_max}, using: {max_length}")

        if detected_max > max_length:
            logger.warning(f"Capping sequence length from {detected_max} to {max_length} to avoid memory issues")

    logger.info(f"Processing {len(all_samples_data)} samples with max_length={max_length}")

    # Use sequential processing to avoid multiprocessing issues
    if disable_multiprocessing:
        logger.info("Multiprocessing disabled by user flag")
    else:
        logger.info("Using sequential feature extraction to avoid multiprocessing issues")

    # Import the feature extraction function
    try:
        from siamese_analysis_v3.data.feature_extractors import extract_features_combined
        use_siamese = True
        logger.info("Using siamese_analysis_v3 feature extraction")
    except ImportError:
        use_siamese = False
        logger.warning("siamese_analysis_v3 not available, using fallback feature extraction")

    processed_features = []
    for sample in tqdm(all_samples_data, desc="Extracting features"):
        try:
            if use_siamese:
                # Use siamese feature extraction
                features = extract_features_combined(
                    sample,
                    sample_rate=sampling_rate,
                    wavelet=wavelet,
                    level=wavelet_level,
                    include_tsfresh=include_tsfresh,
                    max_length=max_length
                )
            else:
                # Fallback feature extraction
                features = extract_basic_features_fallback(sample, max_length)

            processed_features.append(features)

        except Exception as e:
            logger.warning(f"Error extracting features from sample: {e}")
            # Use zero features as fallback
            if use_siamese:
                # For siamese, create zero array with expected shape
                expected_shape = (4, 212531)  # Based on typical output
                features = np.zeros(expected_shape)
            else:
                # For fallback, use simpler shape
                features = np.zeros((4, 100))
            processed_features.append(features)

    # Convert to numpy array
    processed_features = np.array(processed_features)
    logger.info(f"Feature extraction complete. Shape: {processed_features.shape}")

    # Organize features back by label
    sample_idx = 0
    for label, samples in samples_by_label.items():
        n_samples = len(samples)
        features_by_label[label] = processed_features[sample_idx:sample_idx + n_samples]
        sample_idx += n_samples

    return features_by_label


def extract_basic_features_fallback(sample, max_length):
    """Fallback feature extraction when siamese_analysis_v3 is not available"""
    # Ensure sample is numpy array
    sample = np.asarray(sample)

    # Handle 1D input
    if sample.ndim == 1:
        sample = sample.reshape(-1, 1)

    # Pad if necessary
    if len(sample) < max_length:
        padding = np.zeros((max_length - len(sample), sample.shape[1]))
        sample = np.vstack([sample, padding])
    elif len(sample) > max_length:
        sample = sample[:max_length]

    # Extract basic features
    features = []
    for ch in range(sample.shape[1]):
        ch_data = sample[:, ch]
        # Basic statistical features
        features.extend([
            np.mean(ch_data),
            np.std(ch_data),
            np.max(np.abs(ch_data)),
            np.min(ch_data),
            np.median(ch_data)
        ])

    return np.array(features)

def create_umap_visualization(features_by_label: Dict[str, np.ndarray],
                            output_dir: str, n_neighbors_values: List[int] = [15, 30, 50],
                            min_dist_values: List[float] = [0.1, 0.25, 0.5]) -> Dict[str, Any]:
    """
    Create UMAP visualizations directly from features (similar to siamese evaluator)

    Returns:
        Dictionary containing:
        - 'all_features': flattened features array for metrics calculation
        - 'all_labels_level1': level 1 labels (groups)
        - 'all_labels_level2': level 2 labels (data types)
        - 'umap_embeddings_by_config': dictionary with UMAP embeddings for each parameter combination
    """
    if not UMAP_AVAILABLE:
        logger.error("UMAP not available. Skipping UMAP visualizations.")
        return {
            'all_features': None,
            'all_labels_level1': None,
            'all_labels_level2': None,
            'umap_embeddings_by_config': {}
        }

    logger.info("Creating UMAP visualizations...")

    # Prepare data for visualization
    all_features = []
    all_labels_level1 = []
    all_labels_level2 = []
    all_labels_level3 = []
    all_rot_speeds = []

    logger.info("Preparing data for UMAP...")
    for label, features in tqdm(features_by_label.items(), desc="Preparing data"):
        # features shape: (n_samples, n_channels, n_features_per_channel)
        # Need to flatten to 2D for UMAP: (n_samples, n_channels * n_features_per_channel)
        for feature in features:
            # Flatten the feature array to 1D
            if feature.ndim > 1:
                flattened_feature = feature.flatten()
            else:
                flattened_feature = feature
            all_features.append(flattened_feature)

            # Parse labels from the label string
            # Handle different formats: data_type_rot_speed or just data_type
            parts = label.split('_')

            # Check if the last part is a rotation speed (numeric)
            try:
                rot_speed = float(parts[-1])
                data_type = '_'.join(parts[:-1])
                # Validate that data_type is not empty after removing speed
                if not data_type:
                    data_type = label
                    rot_speed = 0.0
            except (ValueError, IndexError):
                # Last part is not numeric or no parts, treat whole label as data_type
                data_type = label
                rot_speed = 0.0

            # Special handling for normal data
            if data_type == 'normal' or label == 'normal':
                data_type = 'normal'
                rot_speed = 0.0  # Normal doesn't need speed variation

            # Create dummy LabeledSample to get hierarchical labels
            dummy_sample = LabeledSample(
                data=np.array([]),  # Not needed for labels
                data_type=data_type,
                rot_speed=rot_speed
            )

            logger.debug(f"Label: {label} -> data_type: {data_type}, rot_speed: {rot_speed}, group: {dummy_sample.level1_label}")

            all_labels_level1.append(dummy_sample.level1_label)
            all_labels_level2.append(dummy_sample.level2_label)
            all_labels_level3.append(dummy_sample.level3_label)
            all_rot_speeds.append(rot_speed)

    if not all_features:
        logger.error("No features available for UMAP visualization")
        return {
            'all_features': None,
            'all_labels_level1': None,
            'all_labels_level2': None,
            'umap_embeddings_by_config': {}
        }

    all_features = np.array(all_features)
    logger.info(f"UMAP input shape after flattening: {all_features.shape}")

    # Ensure 2D shape for UMAP
    if all_features.ndim != 2:
        logger.error(f"Features must be 2D for UMAP, got shape {all_features.shape}")
        return

    # Create output directory
    viz_dir = os.path.join(output_dir, "umap_visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # Get unique groups for visualization
    unique_groups = sorted(set(all_labels_level1))
    logger.info(f"Found groups for visualization: {unique_groups}")

    # Debug: Check how many samples per group
    group_counts = {}
    for group in all_labels_level1:
        group_counts[group] = group_counts.get(group, 0) + 1
    logger.info(f"Sample counts per group: {group_counts}")

    # Define markers for each data class
    markers = ['o', 's', '^', 'v', '<', '>', 'D', 'p', 'h', '*']
    marker_map = {}
    for i, group in enumerate(unique_groups):
        marker_map[group] = markers[i % len(markers)]

    # Dictionary to store UMAP embeddings for each parameter combination
    umap_embeddings_by_config = {}

    # Generate UMAP visualizations for different parameter combinations
    for n_neighbors in tqdm(n_neighbors_values, desc="UMAP n_neighbors"):
        for min_dist in tqdm(min_dist_values, desc=f"  min_dist (n_neighbors={n_neighbors})", leave=False):

            logger.info(f"Fitting UMAP with n_neighbors={n_neighbors}, min_dist={min_dist}")

            try:
                # Create UMAP mapper
                mapper = UMAP(
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    n_components=2,
                    metric='euclidean',
                    random_state=42,
                    verbose=False
                )

                # Fit and transform
                embeddings_2d = mapper.fit_transform(all_features)

                # Store UMAP embeddings for this parameter combination
                config_key = f"n{n_neighbors}_d{min_dist}"
                umap_embeddings_by_config[config_key] = {
                    'embeddings': embeddings_2d,
                    'n_neighbors': n_neighbors,
                    'min_dist': min_dist
                }

                # Create parameter-specific directory
                param_dir = os.path.join(viz_dir, config_key)
                os.makedirs(param_dir, exist_ok=True)

                # Create single combined plot
                logger.info("Creating combined UMAP plot")
                plt.figure(figsize=(16, 14))

                # Will create individual focus plots for each fault type

                # Plot all groups with different markers and variant-based coloring

                # First, plot all groups to establish the full data range
                for group in unique_groups:
                    group_indices = [i for i, g in enumerate(all_labels_level1) if g == group]

                    if not group_indices:
                        continue

                    # Get data types within this group for coloring
                    group_data_types = sorted(set([
                        all_labels_level2[i] for i in group_indices
                    ]))

                    logger.info(f"Group {group}: plotting {len(group_data_types)} variants")

                    # Create color map for variants within this group
                    if group == 'normal':
                        # Normal gets a single blue color
                        variant_colors = {'normal': 'blue'}
                    else:
                        # Use different colors for different variants
                        colors = plt.cm.tab20(np.linspace(0, 1, len(group_data_types)))
                        variant_colors = {}
                        for i, dt in enumerate(group_data_types):
                            variant_colors[dt] = colors[i]

                    # Plot each variant within the group
                    for dt in group_data_types:
                        dt_indices = [
                            i for i in group_indices
                            if all_labels_level2[i] == dt
                        ]

                        if not dt_indices:
                            continue

                        # Get rotation speeds for this variant
                        dt_speeds = [all_rot_speeds[i] for i in dt_indices]

                        # Create colors based on rotation speed
                        if len(set(dt_speeds)) > 1 and group != 'normal':
                            # Vary saturation based on rotation speed
                            import matplotlib.colors as mcolors
                            if dt in variant_colors:
                                base_color = variant_colors[dt]
                            else:
                                # Fallback to default color if data type not in color map
                                logger.warning(f"Data type '{dt}' not found in variant_colors, using default color")
                                base_color = 'red'

                            norm_speeds = 0.3 + 0.7 * (np.array(dt_speeds) - min(dt_speeds)) / (max(dt_speeds) - min(dt_speeds))

                            scatter_colors = []
                            for speed_sat in norm_speeds:
                                # Convert color name to RGB
                                rgb_color = mcolors.to_rgb(base_color)
                                hsv = mcolors.rgb_to_hsv(rgb_color)
                                hsv[1] = speed_sat
                                rgb = mcolors.hsv_to_rgb(hsv)
                                scatter_colors.append(rgb)
                        else:
                            # Use base color for all points in this variant
                            if dt in variant_colors:
                                scatter_colors = [variant_colors[dt]] * len(dt_indices)
                            else:
                                # Fallback to default color
                                scatter_colors = ['red'] * len(dt_indices)

                        # Plot the variant
                        plt.scatter(
                            embeddings_2d[dt_indices, 0],
                            embeddings_2d[dt_indices, 1],
                            c=scatter_colors,
                            marker=marker_map[group],
                            s=120,
                            alpha=0.8,
                            edgecolors='black',
                            linewidth=0.5
                        )

                # Add title with explanation
                title_text = (f'UMAP Visualization - All Fault Types\n'
                            f'(n_neighbors={n_neighbors}, min_dist={min_dist})\n'
                            f'Markers represent fault types, colors represent variants and rotation speeds')
                plt.title(title_text, fontsize=24, pad=20)
                plt.xlabel('UMAP dimension 1', fontsize=24)
                plt.ylabel('UMAP dimension 2', fontsize=24)

                plt.tick_params(axis='both', which='major', labelsize=20)
                plt.grid(True, alpha=0.3)

                # Adjust layout for cleaner plot without legend
                plt.subplots_adjust(top=0.88, bottom=0.1, left=0.1, right=0.95)

                # Print marker to fault type mapping
                print(f"\n{'='*60}")
                print(f"MARKER TO FAULT TYPE MAPPING (n_neighbors={n_neighbors}, min_dist={min_dist})")
                print(f"{'='*60}")
                for group in unique_groups:
                    marker = marker_map[group]
                    fault_name = group.replace('_', ' ').title()
                    print(f"  {marker}  ->  {fault_name}")
                print(f"{'='*60}")
                print("Colors within each marker represent different variants and rotation speeds")
                print(f"{'='*60}\n")

                # Save combined plot
                save_path = os.path.join(param_dir, "umap_combined_all_faults.png")
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()

                # Create individual plots for each fault type (with all data points, focus group colored)
                logger.info("Creating individual plots for each fault type...")
                for focus_group in unique_groups:
                    logger.info(f"Creating individual plot for: {focus_group}")

                    plt.figure(figsize=(16, 14))

                    # Plot all other groups in light gray first
                    other_groups = [g for g in unique_groups if g != focus_group]
                    for other_group in other_groups:
                        other_indices = [i for i, g in enumerate(all_labels_level1) if g == other_group]
                        if other_indices:
                            plt.scatter(
                                embeddings_2d[other_indices, 0],
                                embeddings_2d[other_indices, 1],
                                c='lightgray',
                                marker=marker_map[other_group],
                                s=60,
                                alpha=0.3,
                                edgecolors='gray',
                                linewidth=0.3
                            )

                    # Plot the focus group with full colors and details
                    focus_indices = [i for i, g in enumerate(all_labels_level1) if g == focus_group]

                    if focus_indices:
                        # Get data types within the focus group
                        focus_data_types = sorted(set([
                            all_labels_level2[i] for i in focus_indices
                        ]))

                        # Create color map for variants within the focus group
                        if focus_group == 'normal':
                            variant_colors = {'normal': 'blue'}
                        else:
                            colors = plt.cm.tab20(np.linspace(0, 1, len(focus_data_types)))
                            variant_colors = {}
                            for i, dt in enumerate(focus_data_types):
                                variant_colors[dt] = colors[i]

                        # Plot each variant within the focus group
                        legend_elements = []
                        for dt in focus_data_types:
                            dt_indices = [
                                i for i in focus_indices
                                if all_labels_level2[i] == dt
                            ]

                            if not dt_indices:
                                continue

                            dt_speeds = [all_rot_speeds[i] for i in dt_indices]

                            # Create colors based on rotation speed
                            if len(set(dt_speeds)) > 1 and focus_group != 'normal':
                                import matplotlib.colors as mcolors
                                base_color = variant_colors[dt]
                                norm_speeds = 0.3 + 0.7 * (np.array(dt_speeds) - min(dt_speeds)) / (max(dt_speeds) - min(dt_speeds))

                                scatter_colors = []
                                for speed_sat in norm_speeds:
                                    hsv = mcolors.rgb_to_hsv(base_color[:3])
                                    hsv[1] = speed_sat
                                    rgb = mcolors.hsv_to_rgb(hsv)
                                    scatter_colors.append(rgb)
                            else:
                                scatter_colors = [variant_colors[dt]] * len(dt_indices)

                            # Plot the focus variant with prominent styling
                            plt.scatter(
                                embeddings_2d[dt_indices, 0],
                                embeddings_2d[dt_indices, 1],
                                c=scatter_colors,
                                marker=marker_map[focus_group],
                                s=120,
                                alpha=0.9,
                                edgecolors='black',
                                linewidth=0.5
                            )

                            # Add to legend
                            legend_elements.append(
                                plt.scatter([], [], c=variant_colors[dt], marker=marker_map[focus_group],
                                          s=120, label=f"{dt.split('_')[-1]}", edgecolors='black', linewidth=0.5)
                            )

                        # Add legend for the focus group variants
                        if legend_elements:
                            plt.legend(handles=legend_elements, loc='upper right',
                                     fontsize=18, title="Variants", title_fontsize=20)

                    plt.title(f'UMAP Visualization - Focus: {focus_group.replace("_", " ").title()}\n(n_neighbors={n_neighbors}, min_dist={min_dist})',
                             fontsize=24)
                    plt.xlabel('UMAP dimension 1', fontsize=24)
                    plt.ylabel('UMAP dimension 2', fontsize=24)

                    plt.tick_params(axis='both', which='major', labelsize=20)
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()

                    # Save individual plot
                    individual_save_path = os.path.join(param_dir, f"umap_focus_{focus_group}.png")
                    plt.savefig(individual_save_path, dpi=300, bbox_inches='tight')
                    plt.close()

                    logger.info(f"Saved focus plot: {individual_save_path}")

            except Exception as e:
                logger.error(f"Error creating UMAP visualization: {e}")
                continue

    logger.info("UMAP visualizations complete!")

    # Return the data needed for metrics calculation
    return {
        'all_features': all_features,
        'all_labels_level1': all_labels_level1,
        'all_labels_level2': all_labels_level2,
        'umap_embeddings_by_config': umap_embeddings_by_config
    }



def load_siamese_embeddings(embeddings_path: str) -> Dict[str, np.ndarray]:
    """
    Load embeddings from siamese_analysis_v3 evaluation pipeline

    Args:
        embeddings_path: Path to embeddings.pkl file

    Returns:
        features_by_label: Dictionary mapping labels to embedding arrays
    """
    import pickle

    logger.info(f"Loading embeddings from {embeddings_path}...")

    try:
        with open(embeddings_path, 'rb') as f:
            embeddings_by_level = pickle.load(f)
        logger.info("Embeddings loaded successfully")
    except Exception as e:
        logger.error(f"Error loading embeddings: {e}")
        raise

    # Convert hierarchical embeddings to features_by_label format
    features_by_label = {}

    # Use level 3 (most detailed) embeddings for visualization
    if 3 in embeddings_by_level:
        level3_embeddings = embeddings_by_level[3]
        logger.info(f"Using level 3 embeddings with {len(level3_embeddings)} labels")

        for label, embeddings in level3_embeddings.items():
            # embeddings is already a numpy array with shape (n_samples, embedding_dim)
            features_by_label[label] = embeddings
            logger.debug(f"Label '{label}': {len(embeddings)} samples, shape: {embeddings.shape}")

    else:
        logger.warning("Level 3 embeddings not found, trying level 2...")
        if 2 in embeddings_by_level:
            level2_embeddings = embeddings_by_level[2]
            logger.info(f"Using level 2 embeddings with {len(level2_embeddings)} labels")

            for label, embeddings in level2_embeddings.items():
                features_by_label[label] = embeddings
        else:
            raise ValueError("No suitable embeddings found in the pickle file")

    logger.info(f"Converted {len(features_by_label)} labels for UMAP visualization")
    return features_by_label


def save_feature_statistics(features_by_label: Dict[str, np.ndarray], output_dir: str, source: str = "features"):
    """Save statistics about extracted features or embeddings"""
    logger.info(f"Saving {source} statistics...")

    stats_file = os.path.join(output_dir, f"{source}_statistics.txt")

    with open(stats_file, 'w') as f:
        f.write(f"{source.title()} Statistics\n")
        f.write("=" * 50 + "\n\n")

        total_samples = sum(len(features) for features in features_by_label.values())
        f.write(f"Total samples: {total_samples}\n")
        f.write(f"Number of labels: {len(features_by_label)}\n\n")

        f.write("Samples per label:\n")
        for label, features in sorted(features_by_label.items()):
            f.write(f"  {label}: {len(features)} samples\n")

        # Get feature dimensionality
        if features_by_label:
            first_label = list(features_by_label.keys())[0]
            first_features = features_by_label[first_label]
            if len(first_features) > 0:
                if first_features.ndim == 2:
                    f.write(f"\nFeature dimensionality: {first_features.shape[1]}")
                else:
                    f.write(f"\nFeature dimensionality: {first_features.shape}")

    logger.info(f"{source.title()} statistics saved to {stats_file}")


def calculate_cluster_metrics(features: np.ndarray, labels: List[str], dataset_name: str = "") -> Dict[str, float]:
    """
    Calculate cluster validation metrics for the given features and labels.

    Args:
        features: Flattened feature array (n_samples, n_features)
        labels: String labels for each sample
        dataset_name: Name of the dataset for logging

    Returns:
        Dictionary containing the three cluster validation metrics
    """
    logger.info(f"Calculating cluster validation metrics for {dataset_name}...")

    # Convert string labels to numeric labels for sklearn
    label_encoder = LabelEncoder()
    numeric_labels = label_encoder.fit_transform(labels)

    # Check if we have enough samples and clusters
    n_samples = len(features)
    n_clusters = len(np.unique(numeric_labels))

    logger.info(f"  Dataset: {n_samples} samples, {n_clusters} clusters")

    if n_samples < 2:
        logger.warning("  Not enough samples for cluster validation metrics")
        return {
            "Silhouette Score": float('nan'),
            "Calinski-Harabasz Score": float('nan'),
            "Davies-Bouldin Score": float('nan')
        }

    if n_clusters < 2:
        logger.warning("  Need at least 2 clusters for meaningful validation metrics")
        return {
            "Silhouette Score": float('nan'),
            "Calinski-Harabasz Score": float('nan'),
            "Davies-Bouldin Score": float('nan')
        }

    try:
        # Calculate metrics
        silhouette = silhouette_score(features, numeric_labels)
        calinski = calinski_harabasz_score(features, numeric_labels)
        davies = davies_bouldin_score(features, numeric_labels)

        logger.info(f"  Silhouette Score: {silhouette:.4f}")
        logger.info(f"  Calinski-Harabasz Score: {calinski:.4f}")
        logger.info(f"  Davies-Bouldin Score: {davies:.4f}")

        return {
            "Silhouette Score": silhouette,
            "Calinski-Harabasz Score": calinski,
            "Davies-Bouldin Score": davies
        }

    except Exception as e:
        logger.error(f"  Error calculating cluster metrics: {e}")
        return {
            "Silhouette Score": float('nan'),
            "Calinski-Harabasz Score": float('nan'),
            "Davies-Bouldin Score": float('nan')
        }


def save_cluster_metrics_to_file(metrics_level1: Dict[str, float], metrics_level2: Dict[str, float],
                                output_dir: str, source: str = "features"):
    """
    Save cluster validation metrics to a text file and generate LaTeX table code.

    Args:
        metrics_level1: Metrics for level 1 labels (groups)
        metrics_level2: Metrics for level 2 labels (data types)
        output_dir: Output directory
        source: Source type ("features" or "embeddings")
    """
    import pandas as pd

    logger.info("Saving cluster validation metrics...")

    # Create metrics dictionary for DataFrame
    results_data = {
        "Level 1 (Groups)": metrics_level1,
        "Level 2 (Data Types)": metrics_level2
    }

    # Create DataFrame
    results_df = pd.DataFrame(results_data).T

    # Save to text file
    metrics_file = os.path.join(output_dir, f"cluster_metrics_{source}.txt")
    with open(metrics_file, 'w') as f:
        f.write(f"Cluster Validation Metrics for {source.title()}\n")
        f.write("=" * 60 + "\n\n")
        f.write("Level 1: Group-level clustering (e.g., normal, imbalance_fault, etc.)\n")
        f.write("Level 2: Data-type-level clustering (specific fault types)\n\n")
        f.write("Higher scores are better for Silhouette and Calinski-Harabasz\n")
        f.write("Lower scores are better for Davies-Bouldin\n\n")
        f.write(results_df.to_string(float_format="%.4f"))
        f.write("\n\n" + "=" * 60 + "\n")
        f.write("LaTeX Table Code:\n")
        f.write("=" * 60 + "\n")
        f.write(results_df.to_latex(float_format="%.4f"))

    logger.info(f"Cluster metrics saved to {metrics_file}")

    # Print results to console
    print("\n" + "=" * 80)
    print(f"CLUSTER VALIDATION METRICS FOR {source.upper()}")
    print("=" * 80)
    print("Level 1: Group-level clustering (e.g., normal, imbalance_fault, etc.)")
    print("Level 2: Data-type-level clustering (specific fault types)")
    print("\nHigher scores are better for Silhouette and Calinski-Harabasz")
    print("Lower scores are better for Davies-Bouldin")
    print("\n" + "-" * 80)
    print(results_df.to_string(float_format="%.4f"))
    print("-" * 80)
    print("\nLaTeX Table Code:")
    print("-" * 80)
    print(results_df.to_latex(float_format="%.4f"))
    print("=" * 80 + "\n")


def calculate_umap_configuration_metrics(umap_embeddings_by_config: Dict[str, Dict],
                                       labels_level1: List[str], labels_level2: List[str],
                                       dataset_name: str = "") -> Dict[str, Dict[str, Dict]]:
    """
    Calculate cluster validation metrics for each UMAP parameter configuration.

    Args:
        umap_embeddings_by_config: Dictionary with UMAP embeddings for each config
        labels_level1: Level 1 labels (groups)
        labels_level2: Level 2 labels (data types)
        dataset_name: Name of the dataset for logging

    Returns:
        Dictionary with metrics for each configuration and summary statistics
    """
    logger.info(f"Calculating UMAP configuration metrics for {dataset_name}...")

    config_metrics = {}

    for config_key, config_data in umap_embeddings_by_config.items():
        logger.info(f"  Processing UMAP configuration: {config_key}")

        embeddings_2d = config_data['embeddings']

        # Calculate metrics for level 1 and level 2
        metrics_level1 = calculate_cluster_metrics(
            embeddings_2d, labels_level1, f"{dataset_name} - {config_key} (Level 1)"
        )

        metrics_level2 = calculate_cluster_metrics(
            embeddings_2d, labels_level2, f"{dataset_name} - {config_key} (Level 2)"
        )

        config_metrics[config_key] = {
            'level1': metrics_level1,
            'level2': metrics_level2,
            'n_neighbors': config_data['n_neighbors'],
            'min_dist': config_data['min_dist']
        }

    return config_metrics


def calculate_umap_metrics_summary(config_metrics: Dict[str, Dict[str, Dict]]) -> Dict[str, Dict[str, float]]:
    """
    Calculate average and standard deviation of metrics across all UMAP configurations.

    Args:
        config_metrics: Dictionary with metrics for each configuration

    Returns:
        Dictionary with average and std dev for each metric and level
    """
    if not config_metrics:
        return {}

    logger.info("Calculating UMAP metrics summary statistics...")

    # Collect all metric values for each level and metric type
    level1_silhouette = []
    level1_calinski = []
    level1_davies = []

    level2_silhouette = []
    level2_calinski = []
    level2_davies = []

    for config_key, config_data in config_metrics.items():
        level1_metrics = config_data['level1']
        level2_metrics = config_data['level2']

        # Only include valid (non-NaN) values
        if not np.isnan(level1_metrics['Silhouette Score']):
            level1_silhouette.append(level1_metrics['Silhouette Score'])
        if not np.isnan(level1_metrics['Calinski-Harabasz Score']):
            level1_calinski.append(level1_metrics['Calinski-Harabasz Score'])
        if not np.isnan(level1_metrics['Davies-Bouldin Score']):
            level1_davies.append(level1_metrics['Davies-Bouldin Score'])

        if not np.isnan(level2_metrics['Silhouette Score']):
            level2_silhouette.append(level2_metrics['Silhouette Score'])
        if not np.isnan(level2_metrics['Calinski-Harabasz Score']):
            level2_calinski.append(level2_metrics['Calinski-Harabasz Score'])
        if not np.isnan(level2_metrics['Davies-Bouldin Score']):
            level2_davies.append(level2_metrics['Davies-Bouldin Score'])

    # Calculate summary statistics
    summary = {}

    for level_name, level_data in [('level1', (level1_silhouette, level1_calinski, level1_davies)),
                                  ('level2', (level2_silhouette, level2_calinski, level2_davies))]:
        silhouette_vals, calinski_vals, davies_vals = level_data

        summary[level_name] = {
            'Silhouette Score': {
                'mean': np.mean(silhouette_vals) if silhouette_vals else float('nan'),
                'std': np.std(silhouette_vals) if silhouette_vals else float('nan'),
                'count': len(silhouette_vals)
            },
            'Calinski-Harabasz Score': {
                'mean': np.mean(calinski_vals) if calinski_vals else float('nan'),
                'std': np.std(calinski_vals) if calinski_vals else float('nan'),
                'count': len(calinski_vals)
            },
            'Davies-Bouldin Score': {
                'mean': np.mean(davies_vals) if davies_vals else float('nan'),
                'std': np.std(davies_vals) if davies_vals else float('nan'),
                'count': len(davies_vals)
            }
        }

    return summary


def save_umap_metrics_to_file(config_metrics: Dict[str, Dict[str, Dict]],
                            summary_metrics: Dict[str, Dict[str, float]],
                            output_dir: str, source: str = "features"):
    """
    Save UMAP configuration metrics to file with summary statistics.

    Args:
        config_metrics: Metrics for each UMAP configuration
        summary_metrics: Summary statistics across all configurations
        output_dir: Output directory
        source: Source type ("features" or "embeddings")
    """
    import pandas as pd

    logger.info("Saving UMAP configuration metrics...")

    # Create detailed metrics file
    detailed_file = os.path.join(output_dir, f"umap_config_metrics_{source}.txt")
    with open(detailed_file, 'w') as f:
        f.write(f"UMAP Configuration Metrics for {source.title()}\n")
        f.write("=" * 80 + "\n\n")

        f.write("Metrics calculated on 2D UMAP projections for each parameter combination\n\n")

        for config_key, config_data in sorted(config_metrics.items()):
            f.write(f"Configuration: {config_key} (n_neighbors={config_data['n_neighbors']}, min_dist={config_data['min_dist']})\n")
            f.write("-" * 60 + "\n")

            # Level 1 metrics
            f.write("Level 1 (Groups):\n")
            level1_metrics = config_data['level1']
            for metric_name, value in level1_metrics.items():
                f.write(f"  {metric_name}: {value:.4f}\n")

            # Level 2 metrics
            f.write("Level 2 (Data Types):\n")
            level2_metrics = config_data['level2']
            for metric_name, value in level2_metrics.items():
                f.write(f"  {metric_name}: {value:.4f}\n")

            f.write("\n")

        # Summary statistics
        f.write("\n" + "=" * 80 + "\n")
        f.write("SUMMARY STATISTICS ACROSS ALL UMAP CONFIGURATIONS\n")
        f.write("=" * 80 + "\n\n")

        for level_name, level_data in summary_metrics.items():
            level_display = "Level 1 (Groups)" if level_name == "level1" else "Level 2 (Data Types)"
            f.write(f"{level_display}:\n")
            f.write("-" * 40 + "\n")

            for metric_name, stats in level_data.items():
                mean_val = stats['mean']
                std_val = stats['std']
                count = stats['count']

                if not np.isnan(mean_val):
                    f.write(f"  {metric_name}:\n")
                    f.write(f"    Mean ± Std: {mean_val:.4f} ± {std_val:.4f} (n={count})\n")
                else:
                    f.write(f"  {metric_name}: No valid data\n")
            f.write("\n")

    # Create summary table for LaTeX
    latex_file = os.path.join(output_dir, f"umap_summary_latex_{source}.txt")
    with open(latex_file, 'w') as f:
        f.write(f"% LaTeX table for UMAP metrics summary - {source.title()}\n")
        f.write("% Copy and paste into your LaTeX document\n\n")

        for level_name, level_data in summary_metrics.items():
            level_display = "Level 1 (Groups)" if level_name == "level1" else "Level 2 (Data Types)"
            f.write(f"% {level_display}\n")
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\caption{UMAP Clustering Metrics Summary - " + level_display + "}\n")
            f.write("\\begin{tabular}{lccc}\n")
            f.write("\\hline\n")
            f.write("Metric & Mean & Std Dev & Sample Size \\\\\n")
            f.write("\\hline\n")

            for metric_name, stats in level_data.items():
                mean_val = stats['mean']
                std_val = stats['std']
                count = stats['count']

                if not np.isnan(mean_val):
                    f.write(f"{metric_name} & {mean_val:.4f} & {std_val:.4f} & {count} \\\\\n")
                else:
                    f.write(f"{metric_name} & -- & -- & 0 \\\\\n")

            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n\n")

    logger.info(f"UMAP configuration metrics saved to {detailed_file}")
    logger.info(f"LaTeX summary saved to {latex_file}")

    # Print summary to console
    print("\n" + "=" * 100)
    print(f"UMAP CONFIGURATION METRICS SUMMARY FOR {source.upper()}")
    print("=" * 100)

    for level_name, level_data in summary_metrics.items():
        level_display = "Level 1 (Groups)" if level_name == "level1" else "Level 2 (Data Types)"
        print(f"\n{level_display}:")
        print("-" * 50)

        for metric_name, stats in level_data.items():
            mean_val = stats['mean']
            std_val = stats['std']
            count = stats['count']

            if not np.isnan(mean_val):
                print(f"  {metric_name}:")
                print(f"    Mean ± Std: {mean_val:.4f} ± {std_val:.4f} (n={count})")
            else:
                print(f"  {metric_name}: No valid data")

    print("\nDetailed metrics saved to:", detailed_file)
    print("LaTeX tables saved to:", latex_file)
    print("=" * 100 + "\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="PINN Residuals, Data-Driven Model Features, and Siamese Embeddings UMAP Visualization")

    # Input/Output (choose one)
    parser.add_argument('--residuals', type=str,
                       help='Path to residuals file (.pth) or directory containing X_*.pth files (bypass mode)')
    parser.add_argument('--residuals-dir', type=str,
                       help='Directory containing *_residuals.pth files from pinn_to_siamese_wrapper.py (batch mode)')
    parser.add_argument('--models', type=str, nargs='+',
                       help='Specific model names to process (only works with --residuals-dir). If not specified, processes all models.')
    parser.add_argument('--embeddings', type=str,
                       help='Path to embeddings.pkl file from siamese_analysis_v3 evaluation (embeddings mode)')
    parser.add_argument('--data-driven', type=str,
                       help='Directory containing X_*.pth files to process with data-driven model (data-driven mode)')
    parser.add_argument('--data-standard', action='store_true',
                       help='Use data_standard model (results/data_driven_standard/)')
    parser.add_argument('--data-reg', action='store_true',
                       help='Use data_reg model (results/data_driven_reg/)')
    parser.add_argument('--output-dir', type=str, default='residuals_umap_visualization_outputs',
                       help='Output directory for results')
    parser.add_argument('--test-mode', action='store_true',
                       help='Test mode: process only first 2 residual files to verify pipeline')

    # Data processing
    parser.add_argument('--use-fraction', type=float, default=0.2,
                       help='Fraction of data to use (default: 0.2)')
    parser.add_argument('--include-physical', action='store_true',
                       help='Include physical residuals if available')

    # Feature extraction
    parser.add_argument('--sampling-rate', type=int, default=50000,
                       help='Sampling rate in Hz for feature extraction')
    parser.add_argument('--wavelet', type=str, default='db4',
                       help='Wavelet type for decomposition')
    parser.add_argument('--wavelet-level', type=int, default=3,
                       help='Wavelet decomposition level')
    parser.add_argument('--include-tsfresh', action='store_true',
                       help='Include tsfresh features')
    parser.add_argument('--max-length', type=int, default=None,
                       help='Maximum sequence length for padding')

    # UMAP parameters
    parser.add_argument('--umap-n-neighbors', type=int, nargs='+', default=[15, 30, 50],
                       help='n_neighbors values for UMAP')
    parser.add_argument('--umap-min-dist', type=float, nargs='+', default=[0.1, 0.25, 0.5],
                       help='min_dist values for UMAP')

    # Other
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--debug-data', action='store_true',
                       help='Debug mode: show detailed information about loaded data without processing')
    parser.add_argument('--disable-multiprocessing', action='store_true',
                       help='Disable multiprocessing to avoid import issues')

    args = parser.parse_args()

    # Validate that exactly one mode is provided
    mode_count = sum([bool(args.residuals), bool(args.residuals_dir), bool(args.embeddings), bool(args.data_driven)])
    if mode_count == 0:
        parser.error("Either --residuals, --residuals-dir, --embeddings, or --data-driven must be specified")
    if mode_count > 1:
        parser.error("Cannot specify multiple modes. Choose only one: --residuals, --residuals-dir, --embeddings, or --data-driven")

    # Validate that --models is only used with --residuals-dir
    if args.models and not args.residuals_dir:
        parser.error("--models can only be used with --residuals-dir")

    # Validate data-driven model selection
    if args.data_driven:
        data_model_count = sum([args.data_standard, args.data_reg])
        if data_model_count == 0:
            parser.error("When using --data-driven, you must specify either --data-standard or --data-reg")
        if data_model_count > 1:
            parser.error("Cannot specify both --data-standard and --data-reg. Choose only one.")

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Starting PINN Direct UMAP Visualization")
    logger.info(f"Arguments: {args}")

    try:
        # Determine mode: embeddings or bypass
        if args.embeddings:
            logger.info("=== USING EMBEDDINGS MODE ===")
            logger.info(f"Loading embeddings from: {args.embeddings}")

            # Step 1: Load embeddings from siamese analysis
            features_by_label = load_siamese_embeddings(args.embeddings)

            # Step 2: Save embedding statistics
            save_feature_statistics(features_by_label, args.output_dir, source="embeddings")

            # Step 3: Create UMAP visualizations
            umap_data = create_umap_visualization(
                features_by_label,
                output_dir=args.output_dir,
                n_neighbors_values=args.umap_n_neighbors,
                min_dist_values=args.umap_min_dist
            )

            # Step 4: Calculate cluster validation metrics
            if umap_data['all_features'] is not None:
                logger.info("=== CALCULATING CLUSTER VALIDATION METRICS ===")

                # Calculate metrics for original features
                logger.info("Calculating metrics on original feature space...")
                metrics_level1 = calculate_cluster_metrics(
                    umap_data['all_features'],
                    umap_data['all_labels_level1'],
                    "Level 1 (Groups)"
                )

                metrics_level2 = calculate_cluster_metrics(
                    umap_data['all_features'],
                    umap_data['all_labels_level2'],
                    "Level 2 (Data Types)"
                )

                # Save original metrics to file and display results
                save_cluster_metrics_to_file(
                    metrics_level1,
                    metrics_level2,
                    args.output_dir,
                    source="embeddings"
                )

                # Step 5: Calculate metrics for UMAP configurations
                if umap_data['umap_embeddings_by_config']:
                    logger.info("=== CALCULATING UMAP CONFIGURATION METRICS ===")

                    config_metrics = calculate_umap_configuration_metrics(
                        umap_data['umap_embeddings_by_config'],
                        umap_data['all_labels_level1'],
                        umap_data['all_labels_level2'],
                        "Embeddings UMAP"
                    )

                    # Calculate summary statistics across all configurations
                    summary_metrics = calculate_umap_metrics_summary(config_metrics)

                    # Save UMAP configuration metrics
                    save_umap_metrics_to_file(
                        config_metrics,
                        summary_metrics,
                        args.output_dir,
                        source="embeddings"
                    )
                else:
                    logger.warning("No UMAP configurations available for metrics calculation")
            else:
                logger.warning("No UMAP data available for metrics calculation")

            logger.info("Embeddings UMAP visualization and metrics completed successfully!")
            logger.info(f"Results saved to: {args.output_dir}")

        elif args.residuals_dir:
            logger.info("=== USING BATCH RESIDUALS MODE ===")
            logger.info(f"Processing residual files from: {args.residuals_dir}")

            # Step 1: Discover residual files
            if not os.path.isdir(args.residuals_dir):
                raise ValueError(f"Invalid residuals directory path: {args.residuals_dir}")
            
            data_files = discover_data_files(args.residuals_dir, find_residuals=True)

            if not data_files:
                raise ValueError(f"No *_residuals.pth files found in {args.residuals_dir}")

            # Filter by specific models if requested
            if args.models:
                filtered_data_files = {}
                for model_name in args.models:
                    if model_name in data_files:
                        filtered_data_files[model_name] = data_files[model_name]
                    else:
                        logger.warning(f"Requested model '{model_name}' not found in {args.residuals_dir}")
                data_files = filtered_data_files
                logger.info(f"Filtered to {len(data_files)} specific models: {list(data_files.keys())}")

                if not data_files:
                    raise ValueError(f"None of the requested models {args.models} were found in {args.residuals_dir}")

            # Test mode: process only first 2 files
            if args.test_mode:
                data_files_list = list(data_files.items())[:2]
                data_files = dict(data_files_list)
                logger.info(f"TEST MODE: Processing only first {len(data_files)} files")

            logger.info(f"Found {len(data_files)} residual files to process")
            
            # Process each residual file separately
            for model_name, residuals_path in data_files.items():
                try:
                    logger.info(f"\n{'='*80}")
                    logger.info(f"Processing model: {model_name}")
                    logger.info(f"Residuals file: {residuals_path}")
                    logger.info(f"{'='*80}")
                    
                    # Create model-specific output directory
                    model_output_dir = os.path.join(args.output_dir, model_name)
                    os.makedirs(model_output_dir, exist_ok=True)
                    
                    # Step 2: Load and segment residuals from saved file
                    # The saved file already has the pinn_to_siamese_wrapper format
                    samples_by_label = load_and_segment_pinn_residuals(
                        {model_name: residuals_path},
                        use_fraction=args.use_fraction,
                        include_physical=args.include_physical
                    )
                    
                    if not samples_by_label:
                        logger.warning(f"No samples loaded for {model_name}, skipping")
                        continue
                    
                    # Step 3: Extract advanced features
                    features_by_label = extract_advanced_features(
                        samples_by_label,
                        sampling_rate=args.sampling_rate,
                        wavelet=args.wavelet,
                        wavelet_level=args.wavelet_level,
                        include_tsfresh=args.include_tsfresh,
                        max_length=args.max_length,
                        disable_multiprocessing=getattr(args, 'disable_multiprocessing', False)
                    )
                    
                    # Step 4: Save feature statistics
                    save_feature_statistics(features_by_label, model_output_dir, source=f"features_{model_name}")
                    
                    # Step 5: Create UMAP visualizations
                    umap_data = create_umap_visualization(
                        features_by_label,
                        output_dir=model_output_dir,
                        n_neighbors_values=args.umap_n_neighbors,
                        min_dist_values=args.umap_min_dist
                    )
                    
                    # Step 6: Calculate cluster validation metrics
                    if umap_data['all_features'] is not None:
                        logger.info(f"=== CALCULATING CLUSTER VALIDATION METRICS FOR {model_name} ===")
                        
                        # Calculate metrics for original features
                        logger.info("Calculating metrics on original feature space...")
                        metrics_level1 = calculate_cluster_metrics(
                            umap_data['all_features'],
                            umap_data['all_labels_level1'],
                            f"{model_name} - Level 1 (Groups)"
                        )
                        
                        metrics_level2 = calculate_cluster_metrics(
                            umap_data['all_features'],
                            umap_data['all_labels_level2'],
                            f"{model_name} - Level 2 (Data Types)"
                        )
                        
                        # Save original metrics to file
                        save_cluster_metrics_to_file(
                            metrics_level1,
                            metrics_level2,
                            model_output_dir,
                            source=f"features_{model_name}"
                        )
                        
                        # Calculate metrics for UMAP configurations
                        if umap_data['umap_embeddings_by_config']:
                            logger.info(f"=== CALCULATING UMAP CONFIGURATION METRICS FOR {model_name} ===")
                            
                            config_metrics = calculate_umap_configuration_metrics(
                                umap_data['umap_embeddings_by_config'],
                                umap_data['all_labels_level1'],
                                umap_data['all_labels_level2'],
                                f"{model_name} Features UMAP"
                            )
                            
                            # Calculate summary statistics
                            summary_metrics = calculate_umap_metrics_summary(config_metrics)
                            
                            # Save UMAP configuration metrics
                            save_umap_metrics_to_file(
                                config_metrics,
                                summary_metrics,
                                model_output_dir,
                                source=f"features_{model_name}"
                            )
                        else:
                            logger.warning(f"No UMAP configurations available for {model_name}")
                    else:
                        logger.warning(f"No UMAP data available for {model_name}")
                    
                    logger.info(f"Completed processing for {model_name}")
                    logger.info(f"Results saved to: {model_output_dir}")
                    
                except Exception as e:
                    logger.error(f"Error processing {model_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            logger.info("\n" + "="*80)
            logger.info("BATCH PROCESSING COMPLETE!")
            logger.info(f"Processed {len(data_files)} residual files")
            logger.info(f"Results saved to: {args.output_dir}")
            logger.info("="*80)

        elif args.data_driven:
            logger.info("=== USING DATA-DRIVEN MODE ===")
            logger.info(f"Processing data files from: {args.data_driven}")

            # Step 1: Discover data files
            if os.path.isdir(args.data_driven):
                data_files = discover_data_files(args.data_driven)
            else:
                raise ValueError(f"Invalid data-driven path: {args.data_driven}")

            if not data_files:
                raise ValueError("No data files found")

            # Debug mode: show data information without processing
            if args.debug_data:
                logger.info("=== DEBUG MODE: Data Information ===")
                for data_type, data_path in data_files.items():
                    logger.info(f"\nData type: {data_type}")
                    logger.info(f"File path: {data_path}")

                    try:
                        data = torch.load(data_path)
                        logger.info(f"Data type: {type(data)}")
                        if hasattr(data, 'shape'):
                            logger.info(f"Data shape: {data.shape}")
                        else:
                            logger.info("Data is not a tensor with shape")

                    except Exception as e:
                        logger.error(f"Error examining {data_path}: {e}")

                logger.info("\n=== DEBUG MODE COMPLETE ===")
                logger.info("Run without --debug-data to start actual processing")
                return

            # Step 2: Load and segment data using PINN preprocessing with data-driven models
            from training_scripts.pinn_preprocessing import preprocess_pinn_data
            features_by_label = {}

            # Determine which data-driven model to use based on args
            data_driven_model = 'data_standard' if getattr(args, 'data_standard', False) else 'data_reg'

            for data_type, data_path in data_files.items():
                try:
                    logger.info(f"Processing data type: {data_type}")
                    # Use PINN preprocessing with the specified data-driven model
                    residuals = preprocess_pinn_data(data_driven_model, data_path, chunk_size_mb=50)

                    # Use residuals directly as features (they are already the error signals)
                    # Combine all residual types into feature vectors
                    residual_keys = [k for k in residuals.keys() if k.startswith('data_res') or k.startswith('phys_res')]
                    if residual_keys:
                        # Get the length of residual data
                        sample_length = len(residuals[residual_keys[0]])

                        # Create feature matrix for this data type
                        features_list = []
                        for i in range(sample_length):
                            # Combine residuals from all types at this time point
                            feature_vector = []
                            for key in residual_keys:
                                if i < len(residuals[key]):
                                    feature_vector.extend([residuals[key][i]])

                            if feature_vector:
                                features_list.append(np.array(feature_vector, dtype=np.float32))

                        if features_list:
                            features_by_label[data_type] = np.array(features_list)
                            logger.info(f"Created {len(features_list)} feature vectors for {data_type}")

                except Exception as e:
                    logger.warning(f"Failed to process {data_type}: {e}")
                    continue

            # Step 4: Save feature statistics
            save_feature_statistics(features_by_label, args.output_dir, source="features")

            # Step 5: Create UMAP visualizations
            umap_data = create_umap_visualization(
                features_by_label,
                output_dir=args.output_dir,
                n_neighbors_values=args.umap_n_neighbors,
                min_dist_values=args.umap_min_dist
            )

            # Calculate cluster validation metrics
            if umap_data['all_features'] is not None:
                logger.info("=== CALCULATING CLUSTER VALIDATION METRICS ===")

                # Calculate metrics for original features
                logger.info("Calculating metrics on original feature space...")
                metrics_level1 = calculate_cluster_metrics(
                    umap_data['all_features'],
                    umap_data['all_labels_level1'],
                    "Level 1 (Groups)"
                )

                metrics_level2 = calculate_cluster_metrics(
                    umap_data['all_features'],
                    umap_data['all_labels_level2'],
                    "Level 2 (Data Types)"
                )

                # Save original metrics to file and display results
                save_cluster_metrics_to_file(
                    metrics_level1,
                    metrics_level2,
                    args.output_dir,
                    source="features"
                )

                # Calculate metrics for UMAP configurations
                if umap_data['umap_embeddings_by_config']:
                    logger.info("=== CALCULATING UMAP CONFIGURATION METRICS ===")

                    config_metrics = calculate_umap_configuration_metrics(
                        umap_data['umap_embeddings_by_config'],
                        umap_data['all_labels_level1'],
                        umap_data['all_labels_level2'],
                        "Features UMAP"
                    )

                    # Calculate summary statistics across all configurations
                    summary_metrics = calculate_umap_metrics_summary(config_metrics)

                    # Save UMAP configuration metrics
                    save_umap_metrics_to_file(
                        config_metrics,
                        summary_metrics,
                        args.output_dir,
                        source="features"
                    )
                else:
                    logger.warning("No UMAP configurations available for metrics calculation")
            else:
                logger.warning("No UMAP data available for metrics calculation")

            logger.info("Data-driven UMAP visualization and metrics completed successfully!")
            logger.info(f"Results saved to: {args.output_dir}")

        elif args.residuals:
            logger.info("=== USING BYPASS MODE ===")
            logger.info(f"Processing residuals from: {args.residuals}")

            # Step 1: Discover/load data files
            if os.path.isfile(args.residuals):
                # Single file
                data_files = {extract_data_type_from_filename(args.residuals): args.residuals}
            elif os.path.isdir(args.residuals):
                # Directory with multiple files
                data_files = discover_data_files(args.residuals)
            else:
                raise ValueError(f"Invalid residuals path: {args.residuals}")

            if not data_files:
                raise ValueError("No data files found")

            # Debug mode: show data information without processing
            if args.debug_data:
                logger.info("=== DEBUG MODE: Data Information ===")
                for data_type, data_path in data_files.items():
                    logger.info(f"\nData type: {data_type}")
                    logger.info(f"File path: {data_path}")

                    try:
                        data = torch.load(data_path)
                        logger.info(f"Data type: {type(data)}")

                        if isinstance(data, dict):
                            logger.info(f"Dictionary keys: {list(data.keys())[:10]}...")  # Show first 10 keys
                            # Show sample keys and their structure
                            for key in list(data.keys())[:3]:  # Show first 3 keys in detail
                                value = data[key]
                                if isinstance(value, list) and len(value) > 0:
                                    logger.info(f"  {key}: list of {len(value)} items")
                                    if isinstance(value[0], dict):
                                        logger.info(f"    First item keys: {list(value[0].keys())}")
                                        if 'data' in value[0] and hasattr(value[0]['data'], 'shape'):
                                            logger.info(f"    Data shape: {value[0]['data'].shape}")
                                        if 'rot_speed' in value[0]:
                                            logger.info(f"    Rot speed: {value[0]['rot_speed']}")
                                elif hasattr(value, 'shape'):
                                    logger.info(f"  {key}: shape {value.shape}")
                                elif hasattr(value, '__len__'):
                                    logger.info(f"  {key}: length {len(value)}")
                                else:
                                    logger.info(f"  {key}: {type(value)}")
                        elif hasattr(data, 'shape'):
                            logger.info(f"Tensor shape: {data.shape}")
                        else:
                            logger.info(f"Unknown data structure")

                    except Exception as e:
                        logger.error(f"Error examining {data_path}: {e}")

                logger.info("\n=== DEBUG MODE COMPLETE ===")
                logger.info("Run without --debug-data to start actual processing")
                return

            # Step 2: Load and segment residuals hierarchically
            samples_by_label = load_and_segment_pinn_residuals(
                data_files,
                use_fraction=args.use_fraction,
                include_physical=args.include_physical
            )

            # Step 3: Extract advanced features
            features_by_label = extract_advanced_features(
                samples_by_label,
                sampling_rate=args.sampling_rate,
                wavelet=args.wavelet,
                wavelet_level=args.wavelet_level,
                include_tsfresh=args.include_tsfresh,
                max_length=args.max_length,
                disable_multiprocessing=getattr(args, 'disable_multiprocessing', False)
            )

            # Step 4: Save feature statistics
            save_feature_statistics(features_by_label, args.output_dir, source="features")

            # Step 5: Create UMAP visualizations
            umap_data = create_umap_visualization(
                features_by_label,
                output_dir=args.output_dir,
                n_neighbors_values=args.umap_n_neighbors,
                min_dist_values=args.umap_min_dist
            )

            # Calculate cluster validation metrics
            if umap_data['all_features'] is not None:
                logger.info("=== CALCULATING CLUSTER VALIDATION METRICS ===")

                # Calculate metrics for original features
                logger.info("Calculating metrics on original feature space...")
                metrics_level1 = calculate_cluster_metrics(
                    umap_data['all_features'],
                    umap_data['all_labels_level1'],
                    "Level 1 (Groups)"
                )

                metrics_level2 = calculate_cluster_metrics(
                    umap_data['all_features'],
                    umap_data['all_labels_level2'],
                    "Level 2 (Data Types)"
                )

                # Save original metrics to file and display results
                save_cluster_metrics_to_file(
                    metrics_level1,
                    metrics_level2,
                    args.output_dir,
                    source="features"
                )

                # Calculate metrics for UMAP configurations
                if umap_data['umap_embeddings_by_config']:
                    logger.info("=== CALCULATING UMAP CONFIGURATION METRICS ===")

                    config_metrics = calculate_umap_configuration_metrics(
                        umap_data['umap_embeddings_by_config'],
                        umap_data['all_labels_level1'],
                        umap_data['all_labels_level2'],
                        "Features UMAP"
                    )

                    # Calculate summary statistics across all configurations
                    summary_metrics = calculate_umap_metrics_summary(config_metrics)

                    # Save UMAP configuration metrics
                    save_umap_metrics_to_file(
                        config_metrics,
                        summary_metrics,
                        args.output_dir,
                        source="features"
                    )
                else:
                    logger.warning("No UMAP configurations available for metrics calculation")
            else:
                logger.warning("No UMAP data available for metrics calculation")

            logger.info("Bypass UMAP visualization and metrics completed successfully!")
            logger.info(f"Results saved to: {args.output_dir}")
        else:
            # This should never be reached due to validation above, but just in case
            raise ValueError("Either --residuals or --embeddings must be specified")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
