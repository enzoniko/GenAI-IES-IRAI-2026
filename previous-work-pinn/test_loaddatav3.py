"""
Comprehensive test script for LoadDatav3.py implementation.

This script validates the complete 3-phase pipeline and compares results
with previous versions to ensure the improvements are working correctly.
"""

import numpy as np
import matplotlib.pyplot as plt
from LoadDatav3 import MaFaulDaProcessorV3
import os
from pathlib import Path
import torch
import time

def test_phase_1_data_preparation():
    """Test Phase 1: Foundational Data Preparation."""
    print("Testing Phase 1: Foundational Data Preparation")
    print("=" * 50)
    
    processor = MaFaulDaProcessorV3()
    
    # Load all normal data for testing
    all_data = processor._load_all_normal_data()
    
    # Check data loading
    assert len(all_data) > 0, "No data files loaded"
    print(f"Successfully loaded {len(all_data)} files")
    
    # Check voltage to acceleration conversion on first file
    if all_data:
        first_file = all_data[0]
        print(f"Testing conversion on file: {first_file['filename']}")
        
        for channel in ['acc_underhang_rad', 'acc_overhang_rad']:
            if channel in first_file:
                acceleration = first_file[channel]
                print(f" {channel}:")
                print(f"  Range: {np.min(acceleration):.4f} to {np.max(acceleration):.4f} m/s²")
                print(f"  Mean: {np.mean(acceleration):.4f} m/s²")
                print(f"  Std: {np.std(acceleration):.4f} m/s²")
                
                # Check if values are in reasonable range
                if 0.1 <= np.max(np.abs(acceleration)) <= 10.0:
                    print(f"   Magnitude in expected range")
                else:
                    print(f"   Magnitude outside expected range")
    
    print("Phase 1 test completed!")

def test_phase_2_strategy_evaluation():
    """Test Phase 2: Strategy Evaluation."""
    print("\nTesting Phase 2: Strategy Evaluation")
    print("=" * 50)
    
    processor = MaFaulDaProcessorV3()
    
    # Load test data
    all_data = processor._load_all_normal_data()
    assert len(all_data) > 0, "No data files loaded"
    
    # Test individual strategies on first file
    test_acceleration = all_data[0]['acc_overhang_rad']
    
    print("Testing individual strategies on first file...")
    
    # Strategy A
    result_a = processor.strategy_a_filtering_only(test_acceleration, 2.0)
    print(f"Strategy A (cutoff=2.0 Hz):")
    print(f"  Drift score: {result_a.drift_score:.6f}")
    print(f"  Stability score: {result_a.stability_score:.6f}")
    print(f"  Physical score: {result_a.physical_score:.6f}")
    print(f"  Total score: {result_a.total_score:.6f}")
    
    # Strategy B
    result_b = processor.strategy_b_detrend_filtering(test_acceleration, 2.0)
    print(f"Strategy B (cutoff=2.0 Hz):")
    print(f"  Drift score: {result_b.drift_score:.6f}")
    print(f"  Stability score: {result_b.stability_score:.6f}")
    print(f"  Physical score: {result_b.physical_score:.6f}")
    print(f"  Total score: {result_b.total_score:.6f}")
    
    # Strategy C
    result_c = processor.strategy_c_baseline_correction(test_acceleration, 2.0)
    print(f"Strategy C (cutoff=2.0 Hz):")
    print(f"  Drift score: {result_c.drift_score:.6f}")
    print(f"  Stability score: {result_c.stability_score:.6f}")
    print(f"  Physical score: {result_c.physical_score:.6f}")
    print(f"  Total score: {result_c.total_score:.6f}")
    
    # Run full evaluation (this will take longer now)
    print("\nRunning full strategy evaluation across all files...")
    evaluation_results = processor.phase_2_comparative_evaluation()
    
    print(f"\nEvaluation Results:")
    for strategy, results in evaluation_results.items():
        print(f"  {strategy}: Combined score = {results['combined_score']:.4f}")
    
    print("Phase 2 test completed!")

def test_phase_3_dataset_generation():
    """Test Phase 3: Dataset Generation."""
    print("\nTesting Phase 3: Dataset Generation")
    print("=" * 50)
    
    processor = MaFaulDaProcessorV3()
    
    # First run strategy evaluation
    evaluation_results = processor.phase_2_comparative_evaluation()
    
    # Test single file processing
    print("Testing single file processing...")
    try:
        # Use the first normal file for testing
        normal_dir = Path("Data/normal")
        csv_files = list(normal_dir.glob("*.csv"))
        if csv_files:
            test_file = str(csv_files[0])
            single_result = processor._process_single_file_with_optimal_strategy(test_file)
            print(f" Single file processing successful")
            print(f"  Keys: {list(single_result.keys())}")
            
            # Check data structure
            expected_keys = ['time', 'tachometer', 'acc_underhang_rad', 'acc_underhang_tan',
                            'acc_overhang_rad', 'acc_overhang_tan', 'vel_underhang_rad',
                            'vel_underhang_tan', 'vel_overhang_rad', 'vel_overhang_tan',
                            'pos_underhang_rad', 'pos_underhang_tan', 'pos_overhang_rad',
                            'pos_overhang_tan']
            
            for key in expected_keys:
                if key in single_result:
                    print(f"   {key}: {single_result[key].shape}")
                else:
                    print(f"   Missing: {key}")
        else:
            print(" No normal files found for testing")
            return
        
    except Exception as e:
        print(f" Single file processing failed: {e}")
        return
    
    # Test tensor conversion
    print("\nTesting tensor conversion...")
    try:
        X_tensor, Y_tensor = processor._convert_to_pinn_tensors([single_result])
        print(f" Tensor conversion successful")
        print(f"  X shape: {X_tensor.shape}")
        print(f"  Y shape: {Y_tensor.shape}")
        
        # Check feature structure
        print(f"  X features: {X_tensor.shape[-1]} (should be 10)")
        print(f"  Y features: {Y_tensor.shape[-1]} (should be 4)")
        
        if X_tensor.shape[-1] == 10 and Y_tensor.shape[-1] == 4:
            print("   Feature structure correct")
        else:
            print("   Feature structure incorrect")
            
    except Exception as e:
        print(f" Tensor conversion failed: {e}")
    
    print("Phase 3 test completed!")

def compare_with_previous_versions():
    """Compare results with previous versions."""
    print("\nComparing with Previous Versions")
    print("=" * 50)
    
    # Load previous versions if available
    versions = []
    
    try:
        X_v2 = torch.load("Data/X_normal_v2.pth")
        Y_v2 = torch.load("Data/Y_normal_v2.pth")
        versions.append(("v2", X_v2, Y_v2))
        print(" Loaded v2 data")
    except FileNotFoundError:
        print(" v2 data not found")
    
    try:
        X_orig = torch.load("Data/X_normal.pth")
        Y_orig = torch.load("Data/Y_normal.pth")
        versions.append(("original", X_orig, Y_orig))
        print(" Loaded original data")
    except FileNotFoundError:
        print(" Original data not found")
    
    if not versions:
        print("No previous versions available for comparison")
        return
    
    # Process with v3
    processor = MaFaulDaProcessorV3()
    evaluation_results = processor.phase_2_comparative_evaluation()
    
    # Test single file with v3
    normal_dir = Path("Data/normal")
    csv_files = list(normal_dir.glob("*.csv"))
    if csv_files:
        test_file = str(csv_files[0])
        single_result = processor._process_single_file_with_optimal_strategy(test_file)
        X_v3, Y_v3 = processor._convert_to_pinn_tensors([single_result])
    else:
        print(" No normal files found for comparison")
        return
    
    print(f"\nData Structure Comparison:")
    print(f"  v3: X={X_v3.shape}, Y={Y_v3.shape}")
    
    for name, X, Y in versions:
        print(f"  {name}: X={X.shape}, Y={Y.shape}")
    
    # Compare magnitudes
    print(f"\nMagnitude Comparison (first sample):")
    print(f"  v3 acceleration range: {Y_v3[0].min():.6f} to {Y_v3[0].max():.6f}")
    print(f"  v3 position range: {X_v3[0, :, 4:8].min():.6f} to {X_v3[0, :, 4:8].max():.6f}")
    
    for name, X, Y in versions:
        print(f"  {name} acceleration range: {Y[0].min():.6f} to {Y[0].max():.6f}")
        print(f"  {name} position range: {X[0, :, :4].min():.6f} to {X[0, :, :4].max():.6f}")

def test_validation_metrics():
    """Test validation metrics and quality assessment."""
    print("\nTesting Validation Metrics")
    print("=" * 50)
    
    processor = MaFaulDaProcessorV3()
    
    # Load test data
    all_data = processor._load_all_normal_data()
    if not all_data:
        print(" No data files loaded for validation testing")
        return
    
    # Test on both clean and pathological channels
    test_channels = {
        'clean': 'acc_overhang_rad',
        'pathological': 'acc_underhang_rad'
    }
    
    # Use first file for testing
    first_file = all_data[0]
    
    for channel_type, channel_name in test_channels.items():
        if channel_name in first_file:
            print(f"\nTesting {channel_type} channel ({channel_name}):")
            
            acceleration = first_file[channel_name]
            
            # Test all strategies
            strategies = {
                'A': processor.strategy_a_filtering_only,
                'B': processor.strategy_b_detrend_filtering,
                'C': processor.strategy_c_baseline_correction
            }
            
            for strategy_name, strategy_func in strategies.items():
                try:
                    result = strategy_func(acceleration, 2.0)
                    print(f"  {strategy_name}: drift={result.drift_score:.6f}, "
                          f"stability={result.stability_score:.6f}, "
                          f"physical={result.physical_score:.6f}")
                except Exception as e:
                    print(f"  {strategy_name}: Error - {e}")

def run_complete_pipeline_test():
    """Run the complete pipeline test."""
    print("\nRunning Complete Pipeline Test")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        processor = MaFaulDaProcessorV3()
        processor.run_complete_pipeline()
        
        # Check if output files were created
        output_files = ["Data/X_normal_v3.pth", "Data/Y_normal_v3.pth"]
        for file_path in output_files:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                print(f" {file_path} created ({file_size:.1f} MB)")
            else:
                print(f" {file_path} not created")
        
        elapsed_time = time.time() - start_time
        print(f"\n Complete pipeline test finished in {elapsed_time:.1f} seconds")
        
    except Exception as e:
        print(f" Complete pipeline test failed: {e}")

def main():
    """Run all tests."""
    print("LoadDatav3 Comprehensive Test Suite")
    print("=" * 60)
    
    # Check if normal directory exists
    if not Path("Data/normal").exists():
        print("Error: Data/normal directory not found!")
        print("Please ensure the normal operation CSV files are in Data/normal/")
        return
    
    # Run all tests
    test_phase_1_data_preparation()
    test_phase_2_strategy_evaluation()
    test_phase_3_dataset_generation()
    compare_with_previous_versions()
    test_validation_metrics()
    run_complete_pipeline_test()
    
    print("\n All tests completed!")

if __name__ == "__main__":
    main() 