"""
Anomaly detection methods.

This package contains various anomaly detection methods, including:
- Extreme Value Theory (EVT)
- Isolation Forest
"""

from anomaly_detection.methods.evt import EVTDetector
from anomaly_detection.methods.isolation_forest import IsolationForestDetector

__all__ = ['EVTDetector', 'IsolationForestDetector'] 