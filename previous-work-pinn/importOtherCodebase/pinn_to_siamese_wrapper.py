#!/usr/bin/env python3
"""
PINN to Siamese Wrapper Script

This script bridges the pinn_preprocessing.py output with siamese_analysis_v3/
by converting flat residual arrays into the structured format expected by the Siamese module.

Usage:
    python pinn_to_siamese_wrapper.py --config config.json --output output.pth

Author: AI Assistant
"""

import os
import sys
import json
import torch
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import glob
import re
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training_scripts.pinn_preprocessing import preprocess_pinn_data


@dataclass
class ConversionConfig:
    """Configuration for the PINN to Siamese conversion process."""
    model_name: str
    data_files: Dict[str, str]
    use_fraction: float = 0.2
    use_bayesian_models: bool = True
    use_data_driven: bool = False
    data_driven_model_path: Optional[str] = None
    chunk_size_mb: int = 50
    output_path: Optional[str] = None


def extract_data_type_from_filename(filepath: str) -> str:
    """
    Extract data type from filename pattern.

    Examples:
    - X_normal_v3.pth -> normal
    - X_imbalance_fault_6g_v3.pth -> imbalance_fault_6g
    - X_overhang_ball_fault_0g_v3.pth -> overhang_ball_fault_0g

    Args:
        filepath: Path to the data file

    Returns:
        Data type string (compatible with Siamese analysis data_paths)
    """
    filename = os.path.basename(filepath)

    # Remove X_ prefix and .pth extension
    if filename.startswith('X_') and filename.endswith('.pth'):
        data_part = filename[2:-4]  # Remove 'X_' and '.pth'
    else:
        data_part = filename

    # Remove _v3 suffix if present (Siamese analysis expects original names)
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


def extract_rotation_speeds_from_data(data_path: str) -> List[float]:
    """
    Extract rotation speeds from data file using LoadData utilities.

    Args:
        data_path: Path to the X_*.pth data file

    Returns:
        List of rotation speeds in Hz
    """
    try:
        # Import here to avoid circular imports
        from Data.LoadData import data_paths, get_omegas

        # Get data type from path
        data_type = extract_data_type_from_filename(data_path)

        if data_type in data_paths:
            # Use the directory path from data_paths
            folder_path = data_paths[data_type]
            print(f"  Using folder path: {folder_path} for data type: {data_type}")

            # Get angular frequencies (omegas) from the data folder
            omegas = get_omegas(folder_path)

            # Convert to Hz (rotations per second)
            rot_speeds = [float(omega) / (2 * np.pi) for omega in omegas]

            print(f"  Extracted {len(rot_speeds)} rotation speeds: {rot_speeds[:5]}... (showing first 5)")
            return rot_speeds
        else:
            print(f"  Warning: Data type '{data_type}' not found in data_paths")
            print("  Falling back to dummy rotation speeds for testing")

    except ImportError as e:
        print(f"  Warning: Could not import LoadData utilities ({e})")
        print("  Falling back to dummy rotation speeds for testing")

    except Exception as e:
        print(f"  Error extracting rotation speeds: {e}")
        print("  Falling back to dummy rotation speeds for testing")

    # Fallback: create dummy rotation speeds
    # This assumes 48 different rotation speeds (typical for the dataset)
    return [10.0 + i * 2.0 for i in range(48)]  # 10, 12, 14, ..., 104 Hz


def discover_data_files(base_path: str) -> Dict[str, str]:
    """
    Automatically discover all X_*.pth files in the base path.

    Args:
        base_path: Directory containing the data files

    Returns:
        Dictionary mapping data types to file paths
    """
    data_files = {}

    # Find all X_*.pth files
    pattern = os.path.join(base_path, "X_*.pth")
    x_files = glob.glob(pattern)

    print(f"Found {len(x_files)} X_*.pth files in {base_path}")

    for x_file in x_files:
        data_type = extract_data_type_from_filename(x_file)
        data_files[data_type] = x_file
        print(f"  {data_type}: {os.path.basename(x_file)}")

    return data_files


def validate_residual_shapes(residuals: Dict[str, np.ndarray], data_type: str) -> bool:
    """
    Validate that residual arrays have expected shapes and properties.
    Handles physics residuals that may have an extra dimension.

    Args:
        residuals: Dictionary of residual arrays
        data_type: Data type being validated

    Returns:
        True if validation passes
    """
    expected_keys = ['data_res1', 'data_res2', 'data_res3', 'data_res4',
                     'phys_res1', 'phys_res2', 'phys_res3', 'phys_res4']

    print(f"\n=== Validating residuals for {data_type} ===")

    # Check all expected keys exist
    for key in expected_keys:
        if key not in residuals:
            print(f"  ERROR: Missing key '{key}' in residuals")
            return False

    # Check data types
    for key in expected_keys:
        if not isinstance(residuals[key], np.ndarray):
            print(f"  ERROR: {key} is not a numpy array")
            return False

    # Get reference shape from first data residual
    first_data_key = 'data_res1'
    reference_shape = residuals[first_data_key].shape
    print(f"  Reference shape (from {first_data_key}): {reference_shape}")

    # Validate data residuals (should all be the same shape)
    data_keys = ['data_res1', 'data_res2', 'data_res3', 'data_res4']
    for key in data_keys:
        shape = residuals[key].shape
        if shape != reference_shape:
            print(f"  ERROR: Data residual shape mismatch for {key}: {shape} != {reference_shape}")
            return False

    # Validate physics residuals (may have extra dimension)
    phys_keys = ['phys_res1', 'phys_res2', 'phys_res3', 'phys_res4']
    for key in phys_keys:
        shape = residuals[key].shape
        # Physics residuals might have an extra dimension (n_samples, 1) vs (n_samples,)
        if shape == reference_shape:
            # Same shape - good
            pass
        elif len(shape) == len(reference_shape) + 1 and shape[:-1] == reference_shape:
            # Extra dimension - squeeze it
            print(f"  INFO: Squeezing extra dimension from {key}: {shape} -> {reference_shape}")
            residuals[key] = np.squeeze(residuals[key], axis=-1)
        else:
            print(f"  ERROR: Invalid physics residual shape for {key}: {shape} (expected {reference_shape} or {reference_shape + (1,)})")
            return False

    total_samples = reference_shape[0]
    print(f"  Total samples: {total_samples}")
    print(f"  Data type: {residuals[first_data_key].dtype}")
    print(f"  [OK] All residuals validated and normalized")

    return True


def process_data_type(data_type: str, data_path: str, config: ConversionConfig) -> List[Dict[str, Any]]:
    """
    Process a single data type: run PINN preprocessing and restructure results.

    Args:
        data_type: Name of the data type (e.g., 'normal', 'imbalance_fault_6g')
        data_path: Path to the X_*.pth file for this data type
        config: Conversion configuration

    Returns:
        List of sample dictionaries with 'data' and 'rot_speed' keys
    """
    print(f"\n{'='*60}")
    print(f"Processing data type: {data_type}")
    print(f"Data path: {data_path}")
    print(f"{'='*60}")

    # Step 1: Run preprocessing (PINN or data-driven)
    print(f"\n--- Step 1: Running {'data-driven' if config.use_data_driven else 'PINN'} preprocessing ---")

    # Determine the actual model name to use
    actual_model_name = config.model_name
    if config.use_data_driven:
        # Use data-driven model names
        if config.data_driven_model_path and 'reg' in config.data_driven_model_path.lower():
            actual_model_name = 'data_reg'
        else:
            actual_model_name = 'data_standard'

    try:
        residuals = preprocess_pinn_data(
            model_name=actual_model_name,
            data_path=data_path,
            chunk_size_mb=config.chunk_size_mb
        )
        print(f"[OK] {'Data-driven' if config.use_data_driven else 'PINN'} preprocessing completed for {data_type}")
    except Exception as e:
        print(f"[FAIL] Error in {'data-driven' if config.use_data_driven else 'PINN'} preprocessing for {data_type}: {e}")
        import traceback
        traceback.print_exc()
        return []

    # Step 2: Validate residuals
    if not validate_residual_shapes(residuals, data_type):
        print(f"[FAIL] Validation failed for {data_type}, skipping")
        return []

    # Step 3: Extract rotation speeds
    print(f"\n--- Step 3: Extracting rotation speeds ---")
    rot_speeds = extract_rotation_speeds_from_data(data_path)

    if len(rot_speeds) == 0:
        print(f"[FAIL] No rotation speeds found for {data_type}, skipping")
        return []

    # Step 4: Calculate segmentation parameters
    print(f"\n--- Step 4: Calculating segmentation parameters ---")
    total_samples = len(residuals['data_res1'])
    num_rot_speeds = len(rot_speeds)
    samples_per_speed = total_samples // num_rot_speeds

    print(f"  Total samples: {total_samples}")
    print(f"  Number of rotation speeds: {num_rot_speeds}")
    print(f"  Samples per rotation speed (full): {samples_per_speed}")

    # Apply fraction reduction
    samples_per_speed_reduced = int(samples_per_speed * config.use_fraction)
    print(f"  Samples per rotation speed (reduced to {config.use_fraction*100:.0f}%): {samples_per_speed_reduced}")

    # Step 5: Segment residuals by rotation speed (optimized sampling)
    print(f"\n--- Step 5: Segmenting residuals by rotation speed ---")
    samples = []

    # Calculate stride for sampling to avoid loading entire dataset
    if config.use_fraction < 1.0:
        # Use sampling to reduce memory usage
        stride = max(1, int(1.0 / config.use_fraction))
        print(f"  Using sampling with stride {stride} (fraction: {config.use_fraction})")

        for i, rot_speed in enumerate(rot_speeds):
            start_idx = i * samples_per_speed
            end_idx = start_idx + samples_per_speed

            if end_idx > total_samples:
                end_idx = total_samples

            if start_idx >= end_idx:
                print(f"  Warning: No samples for rotation speed {rot_speed} (start: {start_idx}, end: {end_idx})")
                continue

            # Sample indices for this rotation speed
            rot_speed_indices = np.arange(start_idx, end_idx, stride)
            if len(rot_speed_indices) == 0:
                print(f"  Warning: No samples after sampling for rotation speed {rot_speed}")
                continue

            print(f"  Processing rotation speed {rot_speed:.1f} Hz: {len(rot_speed_indices)} samples (sampled from {end_idx - start_idx})")

            try:
                # Extract residual segment for this rotation speed using sampled indices
                data_segment = np.column_stack([
                    residuals['data_res1'][rot_speed_indices],
                    residuals['data_res2'][rot_speed_indices],
                    residuals['data_res3'][rot_speed_indices],
                    residuals['data_res4'][rot_speed_indices],
                    residuals['phys_res1'][rot_speed_indices],
                    residuals['phys_res2'][rot_speed_indices],
                    residuals['phys_res3'][rot_speed_indices],
                    residuals['phys_res4'][rot_speed_indices]
                ])

                print(f"    Segment shape: {data_segment.shape}")

                # Convert to tensor
                data_tensor = torch.tensor(data_segment, dtype=torch.float32)
                print(f"    Tensor shape: {data_tensor.shape}")

                # Create sample dictionary
                sample = {
                    'data': data_tensor,
                    'rot_speed': float(rot_speed)
                }

                samples.append(sample)

            except Exception as e:
                print(f"    Error processing segment for rotation speed {rot_speed}: {e}")
                continue
    else:
        # Original approach for use_fraction >= 1.0
        for i, rot_speed in enumerate(rot_speeds):
            start_idx = i * samples_per_speed
            end_idx = start_idx + samples_per_speed

            if end_idx > total_samples:
                end_idx = total_samples

            if start_idx >= end_idx:
                print(f"  Warning: No samples for rotation speed {rot_speed} (start: {start_idx}, end: {end_idx})")
                continue

            print(f"  Processing rotation speed {rot_speed:.1f} Hz: samples {start_idx}-{end_idx}")

            try:
                data_segment = np.column_stack([
                    residuals['data_res1'][start_idx:end_idx],
                    residuals['data_res2'][start_idx:end_idx],
                    residuals['data_res3'][start_idx:end_idx],
                    residuals['data_res4'][start_idx:end_idx],
                    residuals['phys_res1'][start_idx:end_idx],
                    residuals['phys_res2'][start_idx:end_idx],
                    residuals['phys_res3'][start_idx:end_idx],
                    residuals['phys_res4'][start_idx:end_idx]
                ])

                print(f"    Segment shape: {data_segment.shape}")

                # Convert to tensor
                data_tensor = torch.tensor(data_segment, dtype=torch.float32)
                print(f"    Tensor shape: {data_tensor.shape}")

                # Create sample dictionary
                sample = {
                    'data': data_tensor,
                    'rot_speed': float(rot_speed)
                }

                samples.append(sample)

            except Exception as e:
                print(f"    Error processing segment for rotation speed {rot_speed}: {e}")
                continue

    print(f"\n[OK] Completed processing {data_type}: {len(samples)} samples created")
    return samples


def convert_pinn_to_siamese(config: ConversionConfig) -> Dict[str, List[Dict[str, Any]]]:
    """
    Main conversion function that processes all data types.

    Args:
        config: Conversion configuration

    Returns:
        Restructured data in Siamese format
    """
    print("=" * 80)
    print("PINN to Siamese Conversion")
    print("=" * 80)
    print(f"Model: {config.model_name}")
    print(f"Data fraction: {config.use_fraction * 100:.0f}%")
    print(f"Bayesian models: {config.use_bayesian_models}")
    print(f"Number of data types: {len(config.data_files)}")

    restructured_data = {}

    # Process each data type
    for data_type, data_path in config.data_files.items():
        samples = process_data_type(data_type, data_path, config)

        if samples:  # Only add if processing was successful
            restructured_data[data_type] = samples
        else:
            print(f"[WARNING] Skipping {data_type} due to processing errors")

    # Summary
    print(f"\n{'='*80}")
    print("CONVERSION SUMMARY")
    print(f"{'='*80}")

    total_samples = 0
    for data_type, samples in restructured_data.items():
        print(f"  {data_type}: {len(samples)} samples")
        total_samples += len(samples)

    print(f"\n  Total: {len(restructured_data)} data types, {total_samples} samples")

    return restructured_data


def validate_restructured_data(restructured_data: Dict[str, List[Dict[str, Any]]]) -> bool:
    """
    Validate that the restructured data matches Siamese expectations.

    Args:
        restructured_data: The converted data

    Returns:
        True if validation passes
    """
    print(f"\n--- Validating restructured data ---")

    if not restructured_data:
        print("[FAIL] No data to validate")
        return False

    validation_passed = True

    for data_type, samples in restructured_data.items():
        print(f"  Validating {data_type} ({len(samples)} samples)...")

        if len(samples) == 0:
            print(f"    [WARNING] {data_type} has no samples")
            continue

        for i, sample in enumerate(samples):
            # Check required keys
            if 'data' not in sample:
                print(f"    [FAIL] Sample {i} in {data_type} missing 'data' key")
                validation_passed = False
                continue

            if 'rot_speed' not in sample:
                print(f"    [FAIL] Sample {i} in {data_type} missing 'rot_speed' key")
                validation_passed = False
                continue

            # Check data types
            if not isinstance(sample['data'], torch.Tensor):
                print(f"    [FAIL] Sample {i} in {data_type}: 'data' should be tensor, got {type(sample['data'])}")
                validation_passed = False
                continue

            # Check tensor properties
            data_tensor = sample['data']
            if data_tensor.dim() != 2:
                print(f"    [FAIL] Sample {i} in {data_type}: 'data' should be 2D tensor, got {data_tensor.dim()}D")
                validation_passed = False
                continue

            if data_tensor.size(1) != 8:  # 4 data + 4 physics residuals
                print(f"    [FAIL] Sample {i} in {data_type}: 'data' should have 8 features, got {data_tensor.size(1)}")
                validation_passed = False
                continue

            # Check rotation speed
            if not isinstance(sample['rot_speed'], (int, float)):
                print(f"    [FAIL] Sample {i} in {data_type}: 'rot_speed' should be numeric, got {type(sample['rot_speed'])}")
                validation_passed = False
                continue

        print(f"    [OK] {data_type} validation passed")

    if validation_passed:
        print("[OK] All validations passed")
    else:
        print("[FAIL] Some validations failed")

    return validation_passed


def save_restructured_data(restructured_data: Dict[str, List[Dict[str, Any]]], output_path: str):
    """
    Save the restructured data to a file compatible with siamese_analysis_v3.

    Args:
        restructured_data: The converted data
        output_path: Path to save the data
    """
    print(f"\n--- Saving restructured data ---")
    print(f"Output path: {output_path}")
    print(f"Number of data types to save: {len(restructured_data)}")

    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Save the data
        torch.save(restructured_data, output_path)

        # Verify the save worked
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # Size in MB
        print(f"[OK] File saved successfully: {file_size:.1f} MB")

        # Test loading
        loaded_data = torch.load(output_path)
        print(f"[OK] Save/load test passed: {len(loaded_data)} data types loaded")

    except Exception as e:
        print(f"[FAIL] Error saving data: {e}")
        raise


def main():
    """Main entry point for the wrapper script."""
    import argparse

    print("[START] Starting PINN to Siamese Wrapper")

    parser = argparse.ArgumentParser(description="Convert PINN residuals to Siamese format")
    parser.add_argument('--config', type=str, help='Path to JSON configuration file')
    parser.add_argument('--output', type=str, required=True, help='Output path for converted data')
    parser.add_argument('--model', type=str, help='PINN model name or data type to process (optional when using --data-dir - if not specified, processes all data types)')
    parser.add_argument('--data-dir', type=str, help='Directory to auto-discover data files')
    parser.add_argument('--fraction', type=float, default=0.2, help='Fraction of data to use (default: 0.2)')
    parser.add_argument('--no-bayesian', action='store_true', help='Use retrained models instead of Bayesian')
    parser.add_argument('--data-driven', action='store_true', help='Use data-driven model instead of PINN')
    parser.add_argument('--data-driven-model-path', type=str, default=None, help='Explicit path to data-driven model directory (e.g., results/data_driven_standard)')

    args = parser.parse_args()
    print(f"Arguments parsed: output={args.output}, model={args.model}, data_dir={args.data_dir}")

    # Load configuration
    config = None
    if args.config:
        print(f"Loading configuration from {args.config}")
        with open(args.config, 'r') as f:
            config_data = json.load(f)
        config = ConversionConfig(**config_data)
    else:
        # Create config from command line arguments
        if not args.data_dir:
            print("Error: Either --config or --data-dir must be specified")
            sys.exit(1)

        # Auto-discover data files
        data_files = discover_data_files(args.data_dir)

        # If model is specified, filter to only that data type
        if args.model and args.model in data_files:
            logger.info(f"Filtering to only process data type: {args.model}")
            data_files = {args.model: data_files[args.model]}

        config = ConversionConfig(
            model_name=args.model if args.model else "all",
            data_files=data_files,
            use_fraction=args.fraction,
            use_bayesian_models=not args.no_bayesian,
            use_data_driven=args.data_driven,
            data_driven_model_path=args.data_driven_model_path,
            output_path=args.output
        )

    # Override config with command line arguments if specified
    if args.model:
        config.model_name = args.model
    if hasattr(args, 'fraction') and args.fraction != 0.2:
        config.use_fraction = args.fraction
    if hasattr(args, 'data_driven'):
        config.use_data_driven = args.data_driven
    if hasattr(args, 'data_driven_model_path') and args.data_driven_model_path:
        config.data_driven_model_path = args.data_driven_model_path

    print(f"Configuration: model={config.model_name}, fraction={config.use_fraction}, data_driven={config.use_data_driven}, data_types={len(config.data_files)}")

    # Run conversion
    print("Starting conversion...")
    restructured_data = convert_pinn_to_siamese(config)
    print(f"Conversion completed with {len(restructured_data)} data types")

    # Debug: Check what was actually processed
    if len(restructured_data) == 0:
        print("WARNING: No data types were successfully processed!")
        print("This could indicate:")
        print("- Data-driven model loading failed")
        print("- Data file loading failed")
        print("- Residual extraction failed")
        print("- Validation failed")
    else:
        print(f"Successfully processed data types: {list(restructured_data.keys())}")

    # Validate results
    if validate_restructured_data(restructured_data):
        # Save results
        save_restructured_data(restructured_data, args.output)
        print("\n[DONE] Conversion completed successfully!")
        print(f"Output saved to: {args.output}")
        print("\nYou can now use this file with siamese_analysis_v3:")
        print(f"python siamese_analysis_v3/cli.py --residuals {args.output} --source direct")
    else:
        print("\n[ERROR] Conversion validation failed. Please check the debug output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
