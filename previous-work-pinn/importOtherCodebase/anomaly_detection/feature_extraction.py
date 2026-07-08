"""
Feature extraction techniques for time series anomaly detection.

This module provides various methods for extracting features from time series data,
including basic statistical features, FFT-based features, wavelet features,
and tsfresh features.
"""

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
import pywt
#from tsfresh.feature_extraction import extract_features
#from tsfresh.utilities.dataframe_functions import impute

def extract_basic_features(seq):
    """
    Extract basic statistical features from a time series.
    
    Parameters:
    - seq: Time series data, can be 1D or 2D (multiple channels)
    
    Returns:
    - Array of features
    """
    seq = np.asarray(seq)
    if seq.ndim == 1:
        return _extract_basic_features(seq)
    else:
        feats = []
        # Process each channel (assumes axis=0 is time, axis=1 are channels)
        for i in range(seq.shape[1]):
            feats.append(_extract_basic_features(seq[:, i]))
        return np.concatenate(feats)

def _extract_basic_features(seq):
    """Internal function for extracting features from a 1D sequence."""
    # Assumes seq is 1D.
    mean_val = np.mean(seq)
    median_val = np.median(seq)
    std_val = np.std(seq)
    max_val = np.max(seq)
    min_val = np.min(seq)
    kurt_val = kurtosis(seq)
    skew_val = skew(seq)
    energy_val = np.sum(seq ** 2)
    hist, _ = np.histogram(seq, bins='auto', density=True)
    hist = hist + 1e-8  # Avoid log(0)
    entropy_val = -np.sum(hist * np.log(hist))
    peak_val = np.max(np.abs(seq))
    return np.array([mean_val, median_val, std_val, max_val, min_val,
                     kurt_val, skew_val, energy_val, entropy_val, peak_val])

def extract_fft_features(seq, sample_rate=1.0):
    """
    Extract frequency domain features using FFT.
    
    Parameters:
    - seq: Time series data, can be 1D or 2D (multiple channels)
    - sample_rate: Sampling rate in Hz
    
    Returns:
    - Array of FFT-based features
    """
    seq = np.asarray(seq)
    if seq.ndim == 1:
        return _extract_fft_features(seq, sample_rate)
    else:
        feats = []
        for i in range(seq.shape[1]):
            feats.append(_extract_fft_features(seq[:, i], sample_rate))
        return np.concatenate(feats)

def _extract_fft_features(seq, sample_rate=1.0):
    """Internal function for extracting FFT features from a 1D sequence."""
    # Assumes seq is 1D.
    fft_vals = np.fft.rfft(seq)
    freqs = np.fft.rfftfreq(len(seq), d=1.0/sample_rate)
    magnitude = np.abs(fft_vals)
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
    return np.array([spectral_centroid, spectral_bandwidth, roll_off, spectral_flatness, dominant_frequency])


def extract_wavelet_features(seq, wavelet='db4', level=3):
    """
    Extract wavelet-based features.
    
    Parameters:
    - seq: Time series data, can be 1D or 2D (multiple channels)
    - wavelet: Wavelet type (default: 'db4')
    - level: Decomposition level (default: 3)
    
    Returns:
    - Array of wavelet-based features
    """
    seq = np.asarray(seq)
    if seq.ndim == 1:
        return _extract_wavelet_features(seq, wavelet, level)
    else:
        feats = []
        for i in range(seq.shape[1]):
            feats.append(_extract_wavelet_features(seq[:, i], wavelet, level))
        return np.concatenate(feats)

def _extract_wavelet_features(seq, wavelet='db4', level=3):
    """Internal function for extracting wavelet features from a 1D sequence."""
    # Assumes seq is 1D.
    coeffs = pywt.wavedec(seq, wavelet, level=level)
    features = []
    for coeff in coeffs:
        mean_val = np.mean(coeff)
        std_val = np.std(coeff)
        energy_val = np.sum(np.square(coeff))
        hist, _ = np.histogram(coeff, bins='auto', density=True)
        hist = hist + 1e-8
        entropy_val = -np.sum(hist * np.log(hist))
        features.extend([mean_val, std_val, energy_val, entropy_val])
    return np.array(features)


def extract_features_extended(seq, sample_rate=1.0, wavelet='db4', level=3, include_tsfresh=False):
    """
    Extract a comprehensive set of features by combining multiple feature types.
    
    Parameters:
    - seq: Time series data, can be 1D or 2D (multiple channels)
    - sample_rate: Sampling rate in Hz
    - wavelet: Wavelet type for wavelet features
    - level: Decomposition level for wavelet features
    - include_tsfresh: Whether to include tsfresh features (can be slow)
    
    Returns:
    - Array of combined features
    """
    basic = extract_basic_features(seq)
    fft_feat = extract_fft_features(seq, sample_rate)
    wavelet_feat = extract_wavelet_features(seq, wavelet, level)
    #if include_tsfresh:
    #    tsf_feat = extract_tsfresh_features(seq)
    #    return np.concatenate([basic, fft_feat, wavelet_feat, tsf_feat])
    #else:
    return np.concatenate([basic, fft_feat, wavelet_feat]) 