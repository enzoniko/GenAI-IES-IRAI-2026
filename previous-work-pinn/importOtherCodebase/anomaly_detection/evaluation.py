"""
Evaluation utilities for anomaly detection methods.

This module provides functions for evaluating anomaly detection methods
using standard metrics and running grid searches to find optimal parameters.
"""

import numpy as np
from tqdm import tqdm
import concurrent.futures
from itertools import product
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from anomaly_detection.feature_extraction import extract_features_extended

def evaluate_detector(detector, test_samples, threshold=None):
    """
    Evaluate a detector on test samples.
    
    Parameters:
    - detector: Fitted detector instance
    - test_samples: Dictionary of test samples by condition
    - threshold: Optional threshold override
    
    Returns:
    - Dictionary of evaluation metrics
    """
    y_true, y_score, y_pred = [], [], []
    
    # Process each condition
    for label, samples in test_samples.items():
        # True label: 0 for normal, 1 for anomaly
        true_label = 0 if label == 'normal' else 1
        
        # Get scores and predictions
        scores = detector.score(samples)
        y_true.extend([true_label] * len(scores))
        y_score.extend(scores)
        
        # Get binary predictions
        if threshold is not None:
            preds = scores > threshold if hasattr(detector, '_model') else scores > threshold
        else:
            preds = detector.detect(samples)
            
        y_pred.extend(preds)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_score)
    }
    
    return metrics

def evaluate_grid_worker(method_class, params, normal_train, test_samples, sample_rate=1.0):
    """
    Worker function to evaluate a method with specific parameters.
    
    Parameters:
    - method_class: Class of the detector to evaluate
    - params: Dictionary of parameters to try
    - normal_train: Normal samples for training
    - test_samples: Dictionary of test samples by condition
    - sample_rate: Sampling rate in Hz
    
    Returns:
    - Tuple of (parameters, metrics, f1_score)
    """
    try:
        # Extract feature extraction parameters
        use_extended = params.pop('use_extended_features', False)
        include_tsfresh = params.pop('include_tsfresh', False) 
        apply_pca = params.pop('apply_pca', False)
        pca_n_components = params.pop('pca_n_components', None)
        speed_tolerance = params.pop('speed_tolerance', None)
        
        # Build feature extractor
        if use_extended:
            feature_extractor_local = lambda seq, sample_rate=sample_rate: extract_features_extended(
                seq, sample_rate=sample_rate, include_tsfresh=include_tsfresh)
        else:
            feature_extractor_local = None
        
        # Compute target rotation speed if available
        if len(normal_train) > 0 and isinstance(normal_train[0], dict) and 'rot_speed' in normal_train[0]:
            speeds = [sample['rot_speed'] for sample in normal_train if isinstance(sample, dict) and 'rot_speed' in sample]
            computed_target_speed = np.median(speeds) if speeds else None
        else:
            computed_target_speed = None
            
        # Initialize and fit detector
        detector = method_class(
            feature_extractor=feature_extractor_local,
            apply_pca=apply_pca,
            pca_n_components=pca_n_components,
            **params
        )
        
        detector.fit(
            normal_train, 
            sample_rate=sample_rate,
            target_speed=computed_target_speed,
            speed_tolerance=speed_tolerance
        )
        
        # Evaluate detector
        metrics = evaluate_detector(detector, test_samples)
        
        # Restore original parameters
        params.update({
            'use_extended_features': use_extended,
            'include_tsfresh': include_tsfresh,
            'apply_pca': apply_pca,
            'pca_n_components': pca_n_components,
            'speed_tolerance': speed_tolerance
        })
        
        return (params, metrics, metrics['f1'])
    
    except Exception as e:
        print(f"Error evaluating {method_class.__name__} with parameters {params}: {e}")
        return (params, None, -np.inf)

def run_grid_search(method_class, param_grid, normal_train, test_samples, 
                  num_workers=4, sample_rate=1.0):
    """
    Run grid search to find optimal parameters for a detector.
    
    Parameters:
    - method_class: Class of the detector to evaluate
    - param_grid: Dictionary with parameter names as keys and lists of values to try
    - normal_train: Normal samples for training
    - test_samples: Dictionary of test samples by condition
    - num_workers: Number of parallel workers (default: 4)
    - sample_rate: Sampling rate in Hz (default: 1.0)
    
    Returns:
    - Tuple of (best_params, best_metrics, best_detector)
    """
    # Generate all parameter combinations
    keys = list(param_grid.keys())
    params_list = [dict(zip(keys, v)) for v in product(*param_grid.values())]
    
    # Initialize tracking variables
    best_f1 = -np.inf
    best_params = None
    best_metrics = None
    
    # Run grid search with parallel execution
    num_workers = min(len(params_list), num_workers)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(evaluate_grid_worker, method_class, params.copy(), 
                           normal_train, test_samples, sample_rate)
            for params in params_list
        ]
        
        for future in tqdm(concurrent.futures.as_completed(futures), 
                          total=len(futures), 
                          desc=f"Grid search for {method_class.__name__}"):
            params_out, metrics, f1 = future.result()
            
            if f1 > best_f1:
                best_f1 = f1
                best_params = params_out
                best_metrics = metrics
    
    # Create and fit the best detector
    if best_params is not None:
        # Extract feature extraction parameters
        use_extended = best_params.pop('use_extended_features', False)
        include_tsfresh = best_params.pop('include_tsfresh', False) 
        apply_pca = best_params.pop('apply_pca', False)
        pca_n_components = best_params.pop('pca_n_components', None)
        speed_tolerance = best_params.pop('speed_tolerance', None)
        
        # Build feature extractor
        if use_extended:
            feature_extractor_local = lambda seq, sample_rate=sample_rate: extract_features_extended(
                seq, sample_rate=sample_rate, include_tsfresh=include_tsfresh)
        else:
            feature_extractor_local = None
            
        # Compute target rotation speed if available
        if len(normal_train) > 0 and isinstance(normal_train[0], dict) and 'rot_speed' in normal_train[0]:
            speeds = [sample['rot_speed'] for sample in normal_train if isinstance(sample, dict) and 'rot_speed' in sample]
            computed_target_speed = np.median(speeds) if speeds else None
        else:
            computed_target_speed = None
            
        # Create and fit best detector
        best_detector = method_class(
            feature_extractor=feature_extractor_local,
            apply_pca=apply_pca,
            pca_n_components=pca_n_components,
            **best_params
        )
        
        best_detector.fit(
            normal_train, 
            sample_rate=sample_rate,
            target_speed=computed_target_speed,
            speed_tolerance=speed_tolerance
        )
        
        # Restore original parameters
        best_params.update({
            'use_extended_features': use_extended,
            'include_tsfresh': include_tsfresh,
            'apply_pca': apply_pca,
            'pca_n_components': pca_n_components,
            'speed_tolerance': speed_tolerance
        })
    else:
        best_detector = None
    
    return best_params, best_metrics, best_detector 