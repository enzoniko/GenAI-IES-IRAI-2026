"""
Extreme Value Theory (EVT) based anomaly detection.
"""

import numpy as np
from scipy.stats import genpareto
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from anomaly_detection.methods.base import AnomalyDetector, extract_data, filter_by_rotation_speed

class EVTDetector(AnomalyDetector):
    """
    Anomaly detector based on Extreme Value Theory.
    
    This method models the tail of a distribution using EVT and
    sets an anomaly threshold based on this model.
    """
    
    def __init__(self, agg_func=np.max, threshold_u_quantile=0.95, quantile=0.999, 
                normalize=True, feature_extractor=None, apply_pca=False,
                pca_n_components=None):
        """
        Initialize the EVT detector.
        
        Parameters:
        - agg_func: Aggregation function for raw data (default: np.max)
        - threshold_u_quantile: Quantile for threshold_u (default: 0.95)
        - quantile: Quantile for the final threshold (default: 0.999)
        - normalize: Whether to normalize by sequence length (default: True)
        - feature_extractor: Function to extract features (default: None)
        - apply_pca: Whether to apply PCA to features (default: False)
        - pca_n_components: Number of PCA components (default: None)
        """
        self.agg_func = agg_func
        self.threshold_u_quantile = threshold_u_quantile
        self.quantile = quantile
        self.normalize = normalize
        self.feature_extractor = feature_extractor
        self.apply_pca = apply_pca
        self.pca_n_components = pca_n_components
        
        self._threshold = None
        self._pca_model = None
        self._scaler_model = None
        self._genpareto_params = None
        self._threshold_u = None
    
    def fit(self, normal_samples, sample_rate=1.0, target_speed=None, speed_tolerance=None):
        """
        Fit the EVT detector on normal samples.
        
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
        
        # Initialize PCA and scaling if needed
        if self.feature_extractor and self.apply_pca:
            normal_feats = np.array([
                self.feature_extractor(extract_data(seq), sample_rate=sample_rate) 
                for seq in filtered_samples
            ])
            self._scaler_model = StandardScaler()
            normal_feats_scaled = self._scaler_model.fit_transform(normal_feats)
            self._pca_model = PCA(n_components=self.pca_n_components)
            normal_feats_pca = self._pca_model.fit_transform(normal_feats_scaled)
            aggregated = [np.linalg.norm(f) for f in normal_feats_pca]
        else:
            aggregated = [self._aggregate(seq, sample_rate) for seq in filtered_samples]
        
        # Compute the threshold_u
        self._threshold_u = np.quantile(aggregated, self.threshold_u_quantile)
        
        # Compute the GPD parameters using excesses
        excess = [x - self._threshold_u for x in aggregated if x > self._threshold_u]
        if len(excess) == 0:
            self._threshold = self._threshold_u
            self._genpareto_params = None
        else:
            self._genpareto_params = genpareto.fit(excess, floc=0)
            self._threshold = self._threshold_u + genpareto.ppf(self.quantile, *self._genpareto_params)
        
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
        if self.feature_extractor and self.apply_pca:
            feats = np.array([
                self.feature_extractor(extract_data(seq), sample_rate=sample_rate) 
                for seq in samples
            ])
            feats_scaled = self._scaler_model.transform(feats)
            feats_pca = self._pca_model.transform(feats_scaled) if self._pca_model else feats_scaled
            scores = [np.linalg.norm(f) for f in feats_pca]
        else:
            scores = [self._aggregate(seq, sample_rate) for seq in samples]
        
        return np.array(scores)
    
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
        threshold = threshold if threshold is not None else self.threshold
        return scores > threshold
    
    def _aggregate(self, seq, sample_rate=1.0):
        """
        Aggregate sequence values into a single score.
        
        Parameters:
        - seq: Input sequence
        - sample_rate: Sampling rate for feature extraction
        
        Returns:
        - Aggregated score
        """
        # Extract data if needed
        seq = extract_data(seq)
        
        if self.feature_extractor:
            feat = self.feature_extractor(seq, sample_rate=sample_rate)
            if self._scaler_model:
                feat = self._scaler_model.transform(feat.reshape(1, -1))[0]
            if self._pca_model:
                feat = self._pca_model.transform(feat.reshape(1, -1))[0]
            return np.linalg.norm(feat)
        else:
            # Process raw sequence data
            seq = seq.cpu().numpy() if hasattr(seq, 'cpu') else np.asarray(seq)
            val = self.agg_func(np.abs(seq))
            return val / np.sqrt(len(seq)) if self.normalize else val
    
    @property
    def threshold(self):
        """Return the current anomaly threshold."""
        return self._threshold 