"""
Base classes and interfaces for anomaly detection methods.
"""

from abc import ABC, abstractmethod
import numpy as np

class AnomalyDetector(ABC):
    """
    Abstract base class for anomaly detection methods.
    
    All anomaly detection methods should inherit from this class
    and implement the required methods.
    """
    
    @abstractmethod
    def fit(self, normal_samples, **kwargs):
        """
        Fit the detector on normal samples.
        
        Parameters:
        - normal_samples: Samples representing normal behavior
        - **kwargs: Additional arguments specific to the method
        """
        pass
    
    @abstractmethod
    def score(self, samples, **kwargs):
        """
        Generate anomaly scores for samples.
        
        Parameters:
        - samples: Samples to score
        - **kwargs: Additional arguments specific to the method
        
        Returns:
        - Array of anomaly scores
        """
        pass
    
    @abstractmethod
    def detect(self, samples, threshold=None, **kwargs):
        """
        Detect anomalies in samples.
        
        Parameters:
        - samples: Samples to evaluate
        - threshold: Threshold for anomaly detection
        - **kwargs: Additional arguments specific to the method
        
        Returns:
        - Boolean array indicating whether each sample is anomalous
        """
        pass
    
    @property
    @abstractmethod
    def threshold(self):
        """Return the current anomaly threshold."""
        pass

def extract_data(sample):
    """
    Given a sample, extract its raw data.
    
    If the sample is a dictionary with a 'data' key, return sample['data'].
    Otherwise, return the sample as-is.
    
    Parameters:
    - sample: Input sample (can be a dict or raw data)
    
    Returns:
    - Raw data
    """
    if isinstance(sample, dict) and 'data' in sample:
        return sample['data']
    return sample

def filter_by_rotation_speed(samples, target_speed=None, tolerance=None):
    """
    Filter samples by rotation speed.
    
    Parameters:
    - samples: List of samples
    - target_speed: Target rotation speed
    - tolerance: Allowable deviation from target speed
    
    Returns:
    - Filtered list of samples
    """
    if target_speed is None or tolerance is None:
        return samples
    
    filtered = []
    for sample in samples:
        if isinstance(sample, dict) and 'rot_speed' in sample:
            if abs(sample['rot_speed'] - target_speed) <= tolerance:
                filtered.append(sample)
        else:
            # If no rotation speed info, include by default
            filtered.append(sample)
    return filtered 