"""
feature_extractors.py
---------------------
Houses the differentiable, purely mathematical feature extractor used by the
PriorWorkOracle to convert PINN physics outputs into an embedding vector.

The legacy Conv1D and FFT neural-network extractors have been intentionally
removed. The Oracle now relies exclusively on MathFeatureExtractor, which
computes deterministic statistical and spectral features from the 8-channel
physics signal (4 PINN residuals + 4 unmeasured force parameters).
"""

import torch
import torch.nn as nn


class MathFeatureExtractor(nn.Module):
    """
    A 100% mathematical, differentiable feature engineering extractor.
    This class replaces neural network layers (like Convs or MLPs) with standard statistical
    and DSP (Digital Signal Processing) equations.
    
    The output is the pure mathematically calculated features, meaning NO learnable parameters
    are involved, fully satisfying analytical traceability constraints.
    """
    def __init__(self, in_channels=8):
        super().__init__()
        self.in_channels = in_channels
        
        # Define the exact number of features we extract mathematically.
        # Time-domain: mean, std, rms, peak, crest, skewness, kurtosis, shape_factor, impulse_factor
        self.num_stat_features = 9
        
        # Frequency-domain: Adaptive pooling enforces 256 frequency bins regardless of input signal length
        self.num_freq_bins = 256
        
        # The true embedding dimension is naturally dictated by the number of math features
        # For 8 channels, this is: 8 * (9 + 256) = 2120 dimensions
        self.output_dim = self.in_channels * (self.num_stat_features + self.num_freq_bins)
        
        # Adaptive pooling to ensure we get exactly 256 frequency features even if 
        # the rotation speed (and thus sequence length) varies.
        self.adaptive_pool = nn.AdaptiveAvgPool1d(self.num_freq_bins)

    def forward(self, x):
        """
        Extracts features using standard, deterministic equations.
        Input: (Batch, Sequence Length, Channels)
        """
        B, L, C = x.shape
        
        # Epsilon is added to denominators and square roots to prevent 
        # division-by-zero or zero-gradient singularities (exploding gradients!)
        eps = 1e-8
        
        # =======================================================
        # 1. TIME DOMAIN STATISTICAL MOMENTS & EQUATIONS
        # =======================================================
        
        # Average value of the signal
        mean = x.mean(dim=1)  # Shape: (B, C)
        
        # Variance and Standard Deviation
        # unbiased=False prevents NaN gradients if sequence is too small
        var = x.var(dim=1, unbiased=False) + eps 
        std = torch.sqrt(var) # Shape: (B, C)
        
        # Root Mean Square (RMS) - Energy of the signal
        rms = torch.sqrt(torch.mean(x**2, dim=1) + eps) # Shape: (B, C)
        
        # Absolute Peak value
        peak = torch.max(torch.abs(x), dim=1)[0] # Shape: (B, C)
        
        # Crest Factor: Peak value divided by the RMS
        # Indicates how extreme the peaks are compared to the average energy
        crest = peak / rms # Shape: (B, C)
        
        # Skewness: Asymmetry of the signal distribution
        # E[(x - mu)^3] / sigma^3
        diff = x - mean.unsqueeze(1)
        skewness = torch.mean(diff**3, dim=1) / (std**3) # Shape: (B, C)
        
        # Kurtosis: "Tailedness" or heavily clustered peaks
        # E[(x - mu)^4] / sigma^4
        kurtosis = torch.mean(diff**4, dim=1) / (var**2) # Shape: (B, C)
        
        # Shape Factor: RMS divided by the mean absolute value
        mean_abs = torch.mean(torch.abs(x), dim=1) + eps
        shape_factor = rms / mean_abs # Shape: (B, C)
        
        # Impulse Factor: Peak divided by the mean absolute value
        impulse_factor = peak / mean_abs # Shape: (B, C)
        
        # Stack all 9 time-domain features for each channel
        # Shape becomes (Batch, Channels, 9) -> then flattened to (Batch, Channels * 9)
        time_features = torch.stack([
            mean, std, rms, peak, crest, skewness, kurtosis, shape_factor, impulse_factor
        ], dim=-1)
        time_features = time_features.view(B, -1)
        
        # =======================================================
        # 2. FREQUENCY DOMAIN FAST FOURIER TRANSFORM (FFT)
        # =======================================================
        
        # Differentiable Real-FFT across the sequence dimension (dim=1)
        fft_complex = torch.fft.rfft(x, dim=1)
        
        # Extract magnitude and normalize by sequence length
        magnitudes = torch.abs(fft_complex) / L
        
        # Prepare for pooling: Conv/Pool layers expect (Batch, Channels, Sequence)
        magnitudes = magnitudes.transpose(1, 2)
        
        # Pool to exactly 256 frequency bins (handles variable length windows)
        pooled_freqs = self.adaptive_pool(magnitudes) # Shape: (B, C, 256)
        
        # Flatten to (Batch, Channels * 256)
        freq_features = pooled_freqs.reshape(B, -1)
        
        # =======================================================
        # 3. COMBINE AND RETURN
        # =======================================================
        
        # Concatenate into the final native mathematical embedding
        combined_features = torch.cat([time_features, freq_features], dim=1) # Shape: (B, C * (9 + 256))
        
        return combined_features

