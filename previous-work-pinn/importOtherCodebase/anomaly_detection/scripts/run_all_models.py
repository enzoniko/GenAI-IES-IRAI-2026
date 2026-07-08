#!/usr/bin/env python3
"""
Run anomaly detection on the latest residuals from all models.

This script finds the most recent residuals from hybrid, direct, and data-driven models,
runs anomaly detection algorithms on each, and saves the results.
"""

import os
import argparse
import time
import numpy as np
from tqdm import tqdm

from anomaly_detection.preprocessing import preprocess_residuals, find_latest_residuals
from anomaly_detection.methods import EVTDetector, IsolationForestDetector
from anomaly_detection.evaluation import run_grid_search, evaluate_detector
from anomaly_detection.utils import save_results, plot_confusion_matrix, plot_roc_curves

def main():
    """Main function for running anomaly detection on all models."""
    parser = argparse.ArgumentParser(
        description='Run anomaly detection on the latest residuals from all models'
    )
    
    # Optional arguments
    parser.add_argument('--output-dir', type=str, default='anomaly_detection_results',
                       help='Directory to save results and plots')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of worker processes for grid search')
    parser.add_argument('--sample-rate', type=float, default=50000,
                       help='Sample rate in Hz for FFT features')
    parser.add_argument('--methods', type=str, nargs='+', 
                      choices=['evt', 'isolation_forest', 'all'], 
                      default=['all'],
                      help='Methods to evaluate')
    parser.add_argument('--skip-plots', action='store_true',
                       help='Skip generating plots')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find the latest residuals for all model types
    residuals_files = find_latest_residuals()
    
    # Define method classes to use
    methods_to_run = []
    if 'all' in args.methods or 'evt' in args.methods:
        methods_to_run.append(('EVT', EVTDetector))
    if 'all' in args.methods or 'isolation_forest' in args.methods:
        methods_to_run.append(('Isolation Forest', IsolationForestDetector))
    
    # Parameter grids for each method
    param_grids = {
        'EVT': {
            'agg_func': [np.max, np.mean],
            'threshold_u_quantile': [0.95],
            'quantile': [0.99],
            'use_extended_features': [False, True],
            'include_tsfresh': [False],
            'apply_pca': [False],
            'pca_n_components': [None],
            'speed_tolerance': [None]
        },
        'Isolation Forest': {
            'contamination': [0.05],
            'use_extended_features': [False, True],
            'include_tsfresh': [False],
            'apply_pca': [False],
            'pca_n_components': [None],
            'speed_tolerance': [None]
        }
    }
    
    # Process each model type
    all_results = {}
    
    for model_type, residuals_path in residuals_files.items():
        print(f"\n===== Processing {model_type} model =====")
        print(f"Using residuals from: {residuals_path}")
        
        # Determine source type based on model type
        if model_type == "hybrid":
            source_type = "hybrid"
        elif model_type == "direct":
            source_type = "direct"
        elif model_type.startswith("data_driven_"):
            source_type = "data_driven"
        else:
            print(f"Unknown model type: {model_type}. Skipping.")
            continue
        
        # Preprocess residuals
        try:
            processed = preprocess_residuals(residuals_path, source_type)
        except Exception as e:
            print(f"Error preprocessing residuals for {model_type}: {e}")
            continue
        
        # For the normal class, shuffle and split
        normal_samples = processed.get('normal', [])
        if not normal_samples:
            print(f"No 'normal' samples found for {model_type}. Skipping.")
            continue
            
        np.random.shuffle(normal_samples)
        split_idx = int(0.8 * len(normal_samples))
        normal_train, normal_test = normal_samples[:split_idx], normal_samples[split_idx:]
        
        # Prepare test samples dictionary
        test_samples = {'normal': normal_test}
        for key, sequences in processed.items():
            if key != 'normal':
                test_samples[key] = sequences
        
        # Print statistics
        print("\nResiduals statistics:")
        for key, sequences in processed.items():
            print(f"  {key}: {len(sequences)} sequences")
        
        print(f"\nTraining on {len(normal_train)} normal samples, testing on {len(normal_test)} normal samples")
        
        # Run grid search for each method
        model_results = {}
        best_detectors = {}
        
        start_time = time.time()
        
        for method_name, method_class in methods_to_run:
            print(f"\n=== Grid searching {method_name} for {model_type} ===")
            
            best_params, best_metrics, best_detector = run_grid_search(
                method_class, 
                param_grids[method_name], 
                normal_train, 
                test_samples, 
                num_workers=args.workers, 
                sample_rate=args.sample_rate
            )
            
            model_results[method_name] = {'params': best_params, 'metrics': best_metrics}
            best_detectors[method_name] = best_detector
            
            print(f"\n=== Best Parameters for {method_name} on {model_type} ===")
            print(best_params)
            print(f"\n=== Best Metrics for {method_name} on {model_type} ===")
            print(best_metrics)
        
        # Calculate total time
        total_time = time.time() - start_time
        print(f"\nTotal execution time for {model_type}: {total_time:.2f} seconds")
        
        # Save results for this model
        model_output_dir = os.path.join(args.output_dir, model_type)
        os.makedirs(model_output_dir, exist_ok=True)
        
        results_path = os.path.join(model_output_dir, "results.json")
        save_results(model_results, results_path)
        print(f"Results for {model_type} saved to {results_path}")
        
        # Generate plots if requested
        if not args.skip_plots and best_detectors:
            # Generate ROC curves
            roc_path = os.path.join(model_output_dir, 'roc_curves.png')
            plot_roc_curves(best_detectors, test_samples, roc_path)
            print(f"ROC curves for {model_type} saved to {roc_path}")
            
            # Generate confusion matrices for each method
            for method_name, detector in best_detectors.items():
                y_true = []
                y_pred = []
                
                for label, samples in test_samples.items():
                    true_label = 0 if label == 'normal' else 1
                    preds = detector.detect(samples)
                    
                    y_true.extend([true_label] * len(preds))
                    y_pred.extend(preds)
                
                cm_path = os.path.join(
                    model_output_dir, 
                    f'{method_name.lower().replace(" ", "_")}_confusion_matrix.png'
                )
                plot_confusion_matrix(y_true, y_pred, labels=['Normal', 'Anomaly'], output_path=cm_path)
                print(f"Confusion matrix for {method_name} on {model_type} saved to {cm_path}")
        
        # Store results for this model
        all_results[model_type] = model_results
    
    # Generate comparison of results across all models
    print("\n===== Comparison of Results Across All Models =====")
    
    # Create a summary report
    summary_path = os.path.join(args.output_dir, "summary.txt")
    
    with open(summary_path, 'w') as f:
        f.write("# Anomaly Detection Results Summary\n\n")
        
        header = f"{'Model Type':<20} | {'Method':<20} | {'F1 Score':<10} | {'Accuracy':<10} | {'AUC':<10}\n"
        f.write(header)
        f.write("-" * len(header) + "\n")
        
        print(header, end='')
        print("-" * len(header))
        
        for model_type, results in all_results.items():
            for method_name, method_results in results.items():
                metrics = method_results.get('metrics', {})
                line = f"{model_type:<20} | {method_name:<20} | {metrics.get('f1', 0):.4f}      | {metrics.get('accuracy', 0):.4f}      | {metrics.get('auc', 0):.4f}\n"
                f.write(line)
                print(line, end='')
    
    print(f"\nSummary report saved to {summary_path}")
    print(f"\nAll results and plots saved in {args.output_dir}")

if __name__ == "__main__":
    main() 