"""
Command-line interface for anomaly detection.

This module provides a command-line interface for running anomaly detection
on residual data from physics-informed and data-driven models.
"""

import os
import argparse
import numpy as np
import time
from itertools import product

from anomaly_detection.methods import EVTDetector, IsolationForestDetector
from anomaly_detection.preprocessing import preprocess_residuals
from anomaly_detection.evaluation import run_grid_search, evaluate_detector
from anomaly_detection.utils import save_results, plot_confusion_matrix, plot_roc_curves

def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description='Anomaly Detection on Model Residuals')
    
    # Required arguments
    parser.add_argument('--residuals', type=str, required=True, 
                        help='Path to the residuals file (.pth)')
    
    # Optional arguments
    parser.add_argument('--source', type=str, choices=['direct', 'hybrid', 'data_driven'], default='direct',
                        help='Source type of residuals: direct PINN, hybrid RNN-PINN, or data-driven models')
    parser.add_argument('--output', type=str, default='anomaly_detection_results.json',
                        help='Output file for the results')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of worker processes for grid search')
    parser.add_argument('--sample-rate', type=float, default=50000,
                        help='Sample rate in Hz for FFT features')
    parser.add_argument('--methods', type=str, nargs='+', 
                        choices=['evt', 'isolation_forest', 'all'], 
                        default=['evt'],
                        help='Methods to evaluate')
    parser.add_argument('--plot-dir', type=str, default='anomaly_detection_plots',
                        help='Directory for saving plots')
    parser.add_argument('--skip-plots', action='store_true',
                        help='Skip generating plots')
    
    args = parser.parse_args()
    
    # Preprocess residuals
    processed = preprocess_residuals(args.residuals, args.source)
    
    # For the normal class, shuffle and split
    normal_samples = processed.get('normal', [])
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
    
    # Define methods to evaluate
    methods_to_run = []
    if 'all' in args.methods or 'evt' in args.methods:
        methods_to_run.append(('EVT', EVTDetector))
    if 'all' in args.methods or 'isolation_forest' in args.methods:
        methods_to_run.append(('Isolation Forest', IsolationForestDetector))
    
    # Parameter grids for each method
    param_grids = {
        'EVT': {
            'agg_func': [np.max, np.mean, np.median, np.std],
            'threshold_u_quantile': [0.9, 0.95],
            'quantile': [0.99, 0.995],
            'use_extended_features': [False, True],
            'include_tsfresh': [False],
            'apply_pca': [False, True],
            'pca_n_components': [1, 5, 10, None],
            'speed_tolerance': [0.1, 0.5, None]
        },
        'Isolation Forest': {
            'contamination': [0.01, 0.05],
            'use_extended_features': [False, True],
            'include_tsfresh': [False],
            'apply_pca': [False, True],
            'pca_n_components': [1, 5, 10, None],
            'speed_tolerance': [0.1, 0.5, None]
        }
    }

    param_grids = {
        'EVT': {
            'agg_func': [np.max, np.mean, np.median, np.std],
            'threshold_u_quantile': [0.9, 0.95],
            'quantile': [0.99, 0.995],
            'use_extended_features': [False, True],
            'include_tsfresh': [False],
            'apply_pca': [False, True],
            'pca_n_components': [5],
            'speed_tolerance': [0.1, 0.5, None]
        },
    }
    
    # Run grid search for each method
    results = {}
    best_detectors = {}
    
    start_time = time.time()
    
    for method_name, method_class in methods_to_run:
        print(f"\n=== Grid searching {method_name} ===")
        
        best_params, best_metrics, best_detector = run_grid_search(
            method_class, 
            param_grids[method_name], 
            normal_train, 
            test_samples, 
            num_workers=args.workers, 
            sample_rate=args.sample_rate
        )
        
        results[method_name] = {'params': best_params, 'metrics': best_metrics}
        best_detectors[method_name] = best_detector
        
        print(f"\n=== Best Parameters for {method_name} ===")
        print(best_params)
        print(f"\n=== Best Metrics for {method_name} ===")
        print(best_metrics)
    
    total_time = time.time() - start_time
    
    # Print final results
    print("\n=== Final Results ===")
    header = f"{'Method':<20} | {'Accuracy':<8} | {'Precision':<8} | {'Recall':<8} | {'F1':<8} | {'AUC':<8}"
    print(header)
    print("-" * len(header))
    
    for method, result in results.items():
        m = result.get('metrics', {})
        print(f"{method:<20} | {m.get('accuracy', 0):.4f}   | {m.get('precision', 0):.4f}   | {m.get('recall', 0):.4f}   | {m.get('f1', 0):.4f}   | {m.get('auc', 0):.4f}")
        print(f"Best Parameters: {result.get('params', {})}")
        print("-" * len(header))
    
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    
    # Save results to file
    save_results(results, args.output)
    print(f"Results saved to {args.output}")
    
    # Generate plots if requested
    if not args.skip_plots:
        # Create output directory for plots
        if not os.path.exists(args.plot_dir):
            os.makedirs(args.plot_dir)
        
        # Generate performance comparison plots
        if len(best_detectors) > 0:
            # Generate ROC curves
            roc_path = os.path.join(args.plot_dir, 'roc_curves.png')
            plot_roc_curves(best_detectors, test_samples, roc_path)
            print(f"ROC curves saved to {roc_path}")
            
            # Generate confusion matrices for each method
            for method_name, detector in best_detectors.items():
                y_true = []
                y_pred = []
                
                for label, samples in test_samples.items():
                    true_label = 0 if label == 'normal' else 1
                    preds = detector.detect(samples)
                    
                    y_true.extend([true_label] * len(preds))
                    y_pred.extend(preds)
                
                cm_path = os.path.join(args.plot_dir, f'{method_name.lower().replace(" ", "_")}_confusion_matrix.png')
                plot_confusion_matrix(y_true, y_pred, labels=['Normal', 'Anomaly'], output_path=cm_path)
                print(f"Confusion matrix for {method_name} saved to {cm_path}")

if __name__ == "__main__":
    main() 