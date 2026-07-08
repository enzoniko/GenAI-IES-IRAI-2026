"""
Anomaly Detection Module

This module provides tools for detecting anomalies in residual data
from physics-informed and data-driven models.

Main Features:
- Feature extraction from time series data
- Multiple anomaly detection methods (EVT, Isolation Forest)
- Preprocessing for different residual types
- Evaluation metrics and reporting
- Grid search for optimal parameters
"""

from anomaly_detection.methods import EVTDetector, IsolationForestDetector
from anomaly_detection.preprocessing import preprocess_residuals
from anomaly_detection.evaluation import evaluate_detector, run_grid_search
from anomaly_detection.cli import main as cli_main

__all__ = [
    'EVTDetector',
    'IsolationForestDetector',
    'preprocess_residuals',
    'evaluate_detector',
    'run_grid_search',
    'cli_main'
] 