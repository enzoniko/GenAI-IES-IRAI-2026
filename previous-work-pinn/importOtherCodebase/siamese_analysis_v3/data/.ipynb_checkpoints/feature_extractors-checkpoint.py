"""
Feature extraction methods for residual analysis
Based on the EVT anomaly detection approach
"""

import numpy as np
import torch
from scipy.stats import kurtosis, skew
import pywt
import pandas as pd
import warnings

# Suppress warnings from tsfresh if imported
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from tsfresh.feature_extraction import extract_features
    from tsfresh.utilities.dataframe_functions import impute
    TSFRESH_AVAILABLE = True
except ImportError:
    TSFRESH_AVAILABLE = False
    print("Warning: tsfresh not available. Some features will be disabled.")

###########################################
# Basic Statistical Features
###########################################

def _extract_basic_features(seq):
    """Extract basic statistical features from a 1D sequence"""
    # Assumes seq is 1D
    mean_val = np.mean(seq)
    median_val = np.median(seq)
    std_val = np.std(seq)
    max_val = np.max(seq)
    min_val = np.min(seq)
    kurt_val = kurtosis(seq)
    skew_val = skew(seq)
    energy_val = np.sum(seq ** 2)
    
    # Calculate entropy from histogram
    hist, _ = np.histogram(seq, bins='auto', density=True)
    hist = hist + 1e-8  # Avoid log(0)
    entropy_val = -np.sum(hist * np.log(hist))
    
    peak_val = np.max(np.abs(seq))
    return np.array([mean_val, median_val, std_val, max_val, min_val,
                     kurt_val, skew_val, energy_val, entropy_val, peak_val])

def extract_basic_features(seq):
    """Extract basic statistical features from sequence (handles multi-channel)"""
    seq = np.asarray(seq)
    if seq.ndim == 1:
        return _extract_basic_features(seq)
    else:
        feats = []
        # Process each channel (assumes axis=0 is time, axis=1 are channels)
        for i in range(seq.shape[1]):
            feats.append(_extract_basic_features(seq[:, i]))
        return np.concatenate(feats)

###########################################
# FFT-based Features
###########################################

def _extract_fft_features(seq, sample_rate=1.0):
    """Extract frequency domain features from a 1D sequence"""
    # Assumes seq is 1D
    fft_vals = np.fft.rfft(seq)
    freqs = np.fft.rfftfreq(len(seq), d=1.0/sample_rate)
    magnitude = np.abs(fft_vals)
    
    # Handle edge case of all zeros
    if np.sum(magnitude) == 0:
        spectral_centroid = 0
        spectral_bandwidth = 0
    else:
        spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
        spectral_bandwidth = np.sqrt(np.sum(magnitude * (freqs - spectral_centroid)**2) / np.sum(magnitude))
    
    total_energy = np.sum(magnitude**2)
    cumulative_energy = np.cumsum(magnitude**2)
    roll_off_threshold = 0.85 * total_energy
    roll_off_idx = np.where(cumulative_energy >= roll_off_threshold)[0]
    roll_off = freqs[roll_off_idx[0]] if len(roll_off_idx) > 0 else 0
    
    eps = 1e-8
    geometric_mean = np.exp(np.mean(np.log(magnitude + eps)))
    arithmetic_mean = np.mean(magnitude + eps)
    spectral_flatness = geometric_mean / arithmetic_mean
    
    dominant_frequency = freqs[np.argmax(magnitude)]
    return np.array([spectral_centroid, spectral_bandwidth, roll_off, 
                     spectral_flatness, dominant_frequency])

def extract_fft_features(seq, sample_rate=1.0):
    """Extract frequency domain features (handles multi-channel)"""
    seq = np.asarray(seq)
    if seq.ndim == 1:
        return _extract_fft_features(seq, sample_rate)
    else:
        feats = []
        for i in range(seq.shape[1]):
            feats.append(_extract_fft_features(seq[:, i], sample_rate))
        return np.concatenate(feats)

###########################################
# Wavelet Features
###########################################

def _extract_wavelet_features(seq, wavelet='db4', level=3):
    """Extract wavelet transform features from a 1D sequence"""
    # Assumes seq is 1D
    try:
        coeffs = pywt.wavedec(seq, wavelet, level=level)
        features = []
        for coeff in coeffs:
            mean_val = np.mean(coeff)
            std_val = np.std(coeff)
            energy_val = np.sum(np.square(coeff))
            
            # Handle histogram calculation with error protection
            try:
                # First check if the coefficient has any variation
                if np.std(coeff) < 1e-10:
                    # Almost constant data, use a simple histogram
                    entropy_val = 0
                else:
                    # Use a fixed number of bins instead of 'auto'
                    hist, _ = np.histogram(coeff, bins=10, density=True)
                    hist = hist + 1e-8  # Avoid log(0)
                    entropy_val = -np.sum(hist * np.log(hist))
            except Exception as e:
                # Fallback in case of histogram error
                print(f"Histogram error: {e}, using entropy=0")
                entropy_val = 0
                
            features.extend([mean_val, std_val, energy_val, entropy_val])
        return np.array(features)
    except Exception as e:
        # Fallback in case of general wavelet error
        print(f"Wavelet error: {e}, using zeros")
        # Return a zero array of expected length: 4 features per level
        expected_length = 4 * (level + 1)  # +1 for the approximation coefficients
        return np.zeros(expected_length)

def extract_wavelet_features(seq, wavelet='db4', level=3):
    """Extract wavelet transform features (handles multi-channel)"""
    seq = np.asarray(seq)
    if seq.ndim == 1:
        return _extract_wavelet_features(seq, wavelet, level)
    else:
        feats = []
        for i in range(seq.shape[1]):
            feats.append(_extract_wavelet_features(seq[:, i], wavelet, level))
        return np.concatenate(feats)

###########################################
# TSFresh Features (if available)
###########################################

def _extract_tsfresh_features(seq):
    """Extract tsfresh features from a 1D sequence"""
    if not TSFRESH_AVAILABLE:
        return np.array([])
    
    # Assumes seq is 1D
    df = pd.DataFrame({'id': 0, 'time': np.arange(len(seq)), 'value': seq})
    features_df = extract_features(df, column_id='id', column_sort='time')
    impute(features_df)
    features_df = features_df.fillna(0)
    return features_df.values.flatten()

def extract_tsfresh_features(seq):
    """Extract tsfresh features (handles multi-channel)"""
    if not TSFRESH_AVAILABLE:
        return np.array([])
    
    seq = np.asarray(seq)
    if seq.ndim == 1:
        return _extract_tsfresh_features(seq)
    else:
        feats = []
        for i in range(seq.shape[1]):
            feats.append(_extract_tsfresh_features(seq[:, i]))
        return np.concatenate(feats)

###########################################
# Combined Feature Extraction
###########################################

def extract_features_combined(sample, sample_rate=50000, wavelet='db4', level=3, include_tsfresh=False, max_length=None):
    """
    Extract combined features while preserving channel structure
    
    Args:
        sample: Input data with shape [time_steps, channels]
        sample_rate: Sampling rate in Hz
        wavelet: Wavelet type
        level: Wavelet decomposition level
        include_tsfresh: Whether to include tsfresh features
        max_length: Maximum length to pad sequences to before FFT
        
    Returns:
        Features with shape [num_channels, features_per_channel]
    """
    # Make sure sample is numpy array
    sample = np.asarray(sample)
    
    # Handle scalar or 1D input
    if sample.ndim == 0 or (sample.ndim == 1 and len(sample) == 1):
        sample = np.array([[sample]])
    elif sample.ndim == 1:
        sample = sample.reshape(-1, 1)  # Assume 1D is a single channel
    
    # Get number of channels
    num_channels = sample.shape[1] if sample.ndim > 1 else 1
    
    # If max_length not provided, use the actual length
    if max_length is None:
        max_length = sample.shape[0]
        
    # Extract features by channel
    features_by_channel = []
    
    for ch in range(num_channels):
        # Get channel data
        channel_data = sample[:, ch] if sample.ndim > 1 else sample
        
        # 1. Extract statistical features
        mean = np.mean(channel_data)
        std = np.std(channel_data)
        rms = np.sqrt(np.mean(np.square(channel_data)))
        peak = np.max(np.abs(channel_data))
        skewness = np.mean(((channel_data - mean) / std)**3) if std > 0 else 0
        kurtosis = np.mean(((channel_data - mean) / std)**4) if std > 0 else 0
        crest_factor = peak / rms if rms > 0 else 0
        
        # Additional statistical features
        p2p = np.max(channel_data) - np.min(channel_data)
        median = np.median(channel_data)
        
        # 2. Compute FFT features - always pad to max_length first
        # Pad to max_length to ensure consistent FFT length
        if len(channel_data) < max_length:
            padded_data = np.pad(channel_data, (0, max_length - len(channel_data)), 'constant')
        else:
            padded_data = channel_data[:max_length]
        
        #fft_magnitude = np.abs(np.fft.rfft(padded_data))
        fft_magnitude = np.array([])
        fft_features = _extract_fft_features(padded_data, sample_rate)

        fft_features = np.concatenate([fft_magnitude, fft_features])
        
        # 3. Wavelet features
        wavelet_features = _extract_wavelet_features(channel_data, wavelet, level)
        
    
        
        # Combine all features for this channel
        basic_feats = np.array([mean, std, rms, peak, skewness, kurtosis, crest_factor, p2p, median])
        channel_feats = np.concatenate([basic_feats, fft_features, wavelet_features])
        features_by_channel.append(channel_feats)
    
    # Return stacked features by channel
    return np.array(features_by_channel)

def extract_v0_features(sample, sample_rate=50000, fft_length=128, max_length=None):
    """
    Extract features using the simplified v0 approach (only FFT magnitudes)
    
    Args:
        sample: Input data with shape [time_steps, channels]
        sample_rate: Sampling rate in Hz
        fft_length: Number of FFT magnitude components to extract
        max_length: Maximum length to pad sequences to before FFT
        
    Returns:
        Features with shape [num_channels, fft_length]
    """
    # Make sure sample is numpy array
    sample = np.asarray(sample)
    
    # Handle scalar or 1D input
    if sample.ndim == 0 or (sample.ndim == 1 and len(sample) == 1):
        sample = np.array([[sample]])
    elif sample.ndim == 1:
        sample = sample.reshape(-1, 1)  # Assume 1D is a single channel
    
    # Get number of channels
    num_channels = sample.shape[1] if sample.ndim > 1 else 1
    
    # If max_length not provided, use the actual length
    if max_length is None:
        max_length = sample.shape[0]
    
    # Extract FFT features by channel
    features_by_channel = []
    
    for ch in range(num_channels):
        # Get channel data
        channel_data = sample[:, ch] if sample.ndim > 1 else sample
        
        # Pad to max_length to ensure consistent FFT length
        if len(channel_data) < max_length:
            padded_data = np.pad(channel_data, (0, max_length - len(channel_data)), 'constant')
        else:
            padded_data = channel_data[:max_length]
        
        # FFT magnitude
        fft_vals = np.abs(np.fft.rfft(padded_data))
        
        # Take first N frequencies (where N = fft_length)
        if len(fft_vals) >= fft_length:
            fft_features = fft_vals[:fft_length]
        else:
            # If fewer components available, pad with zeros
            fft_features = np.pad(fft_vals, (0, fft_length - len(fft_vals)), 'constant')
        
        features_by_channel.append(fft_features)
    
    # Return stacked features by channel
    return np.array(features_by_channel) 