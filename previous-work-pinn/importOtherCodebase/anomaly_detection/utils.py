"""
Utility functions for anomaly detection.

This module provides various helper functions used across the anomaly detection module.
"""

import os
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def ensure_dir(directory):
    """
    Ensure that a directory exists, creating it if necessary.
    
    Parameters:
    - directory: Path to the directory
    """
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def save_results(results, output_path):
    """
    Save results dictionary to a file.
    
    Parameters:
    - results: Dictionary of results
    - output_path: Path to save the results
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    ensure_dir(output_dir)
    
    # Prepare results for serialization (convert numpy types to Python types)
    serializable_results = {}
    
    for method, result in results.items():
        serializable_results[method] = {
            'params': {k: str(v) if isinstance(v, (np.ndarray, np.generic)) else v 
                     for k, v in result.get('params', {}).items()},
            'metrics': {k: float(v) if isinstance(v, (np.ndarray, np.generic)) else v 
                      for k, v in result.get('metrics', {}).items()}
        }
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)

def plot_confusion_matrix(y_true, y_pred, labels=None, output_path=None):
    """
    Plot confusion matrix.
    
    Parameters:
    - y_true: True labels
    - y_pred: Predicted labels
    - labels: List of label names
    - output_path: Path to save the plot (if None, the plot is shown)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title('Confusion Matrix')
    
    if output_path:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        ensure_dir(output_dir)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_roc_curves(detectors, test_samples, output_path=None):
    """
    Plot ROC curves for multiple detectors.
    
    Parameters:
    - detectors: Dictionary of detector instances by name
    - test_samples: Dictionary of test samples by condition
    - output_path: Path to save the plot (if None, the plot is shown)
    """
    from sklearn.metrics import roc_curve, auc
    
    plt.figure(figsize=(10, 8))
    
    # Process each detector
    for name, detector in detectors.items():
        y_true = []
        y_score = []
        
        # Process each condition
        for label, samples in test_samples.items():
            # True label: 0 for normal, 1 for anomaly
            true_label = 0 if label == 'normal' else 1
            scores = detector.score(samples)
            
            y_true.extend([true_label] * len(scores))
            y_score.extend(scores)
        
        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')
    
    # Plot formatting
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    
    if output_path:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        ensure_dir(output_dir)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show() 