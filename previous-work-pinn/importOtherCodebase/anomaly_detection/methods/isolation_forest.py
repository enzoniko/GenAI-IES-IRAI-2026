"""
Isolation Forest based anomaly detection.
"""

import numpy as np
from sklearn.ensemble import IsolationForest as SklearnIsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from anomaly_detection.methods.base import AnomalyDetector, extract_data, filter_by_rotation_speed
from anomaly_detection.feature_extraction import extract_basic_features

class IsolationForestDetector(AnomalyDetector):
    """
    Anomaly detector based on Isolation Forest.
    
    This method isolates anomalies by building random decision trees and
    identifying samples that have short average path lengths.
    """
    
    def __init__(self, contamination=0.05, feature_extractor=None, apply_pca=False, 
                pca_n_components=None, random_state=42):
        """
        Initialize the Isolation Forest detector.
        
        Parameters:
        - contamination: Expected ratio of anomalies in the data (default: 0.05)
        - feature_extractor: Function to extract features (default: None)
        - apply_pca: Whether to apply PCA to features (default: False)
        - pca_n_components: Number of PCA components (default: None)
        - random_state: Random seed for reproducibility (default: 42)
        """
        self.contamination = contamination
        self.feature_extractor = feature_extractor
        self.apply_pca = apply_pca
        self.pca_n_components = pca_n_components
        self.random_state = random_state
        
        self._model = None
        self._pca_model = None
        self._scaler_model = None
        self._threshold = None
    
    def fit(self, normal_samples, sample_rate=1.0, target_speed=None, speed_tolerance=None):
        """
        Fit the Isolation Forest detector on normal samples.
        
        Parameters:
        - normal_samples: Samples representing normal behavior
        - sample_rate: Sampling rate for feature extraction
        - target_speed: Target rotation speed for filtering
        - speed_tolerance: Tolerance for rotation speed filtering
        
        Returns:
        - self
        """
        # Filter samples by rotation speed if specified
        filtered_samples = filter_by_rotation_speed(normal_samples, target_speed, speed_tolerance)
        
        # Extract features
        if self.feature_extractor:
            normal_data = np.array([
                self.feature_extractor(extract_data(seq), sample_rate=sample_rate) 
                for seq in filtered_samples
            ])
        else:
            normal_data = np.array([extract_basic_features(extract_data(seq)) for seq in filtered_samples])
        
        # Apply PCA if requested
        if self.apply_pca and normal_data.shape[1] > 1:
            self._scaler_model = StandardScaler()
            normal_data_scaled = self._scaler_model.fit_transform(normal_data)
            self._pca_model = PCA(n_components=self.pca_n_components)
            normal_data_pca = self._pca_model.fit_transform(normal_data_scaled)
            normal_data = normal_data_pca
        elif self.apply_pca and normal_data.shape[1] <= 1 and normal_data.shape[1] > 0:
            self._scaler_model = StandardScaler()
            normal_data_scaled = self._scaler_model.fit_transform(normal_data)
            normal_data = normal_data_scaled
        
        # Fit the isolation forest model
        self._model = SklearnIsolationForest(contamination=self.contamination, 
                                           random_state=self.random_state)
        self._model.fit(normal_data)
        
        # Compute threshold
        train_scores = self._model.decision_function(normal_data)
        self._threshold = np.quantile(train_scores, self.contamination)
        
        return self
    
    def score(self, samples, sample_rate=1.0):
        """
        Generate anomaly scores for samples.
        
        Parameters:
        - samples: Samples to score
        - sample_rate: Sampling rate for feature extraction
        
        Returns:
        - Array of anomaly scores
        """
        if self._model is None:
            raise ValueError("The detector has not been fitted yet. Call fit() first.")
        
        # Extract features
        if self.feature_extractor:
            test_data = np.array([
                self.feature_extractor(extract_data(seq), sample_rate=sample_rate) 
                for seq in samples
            ])
        else:
            test_data = np.array([extract_basic_features(extract_data(seq)) for seq in samples])
        
        # Apply transformations
        if self.apply_pca and self._pca_model is not None and test_data.shape[1] > 1:
            test_data_scaled = self._scaler_model.transform(test_data)
            test_data = self._pca_model.transform(test_data_scaled)
        elif self.apply_pca and self._scaler_model is not None and test_data.shape[1] <= 1 and test_data.shape[1] > 0:
            test_data_scaled = self._scaler_model.transform(test_data)
            test_data = test_data_scaled
        
        # Return decision function scores
        # Note: Lower scores in isolation forest indicate anomalies, so we negate them
        return -self._model.decision_function(test_data)
    
    def detect(self, samples, threshold=None, sample_rate=1.0):
        """
        Detect anomalies in samples.
        
        Parameters:
        - samples: Samples to evaluate
        - threshold: Override the computed threshold if provided
        - sample_rate: Sampling rate for feature extraction
        
        Returns:
        - Boolean array indicating whether each sample is anomalous
        """
        scores = self.score(samples, sample_rate)
        threshold = threshold if threshold is not None else -self.threshold
        return scores > threshold
    
    @property
    def threshold(self):
        """Return the current anomaly threshold."""
        return self._threshold 