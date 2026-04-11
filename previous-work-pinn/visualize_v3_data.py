"""
Specialized visualization script for LoadDatav3 processed data.

This script focuses on assessing the quality of the v3 processed data
and comparing it with previous versions to identify any issues.

Enhancements:
- Plot all rotation speeds side-by-side and concatenated to see evolution across speed.
- Remove distracting arrow overlays in phase plots; cleaner visuals by default.
- Detect speed segments via `omega` (and/or per-sample speeds) and color changes at transitions (~5s per speed).
- Cross-compare trajectories between v3 loader outputs and legacy loader outputs (from `LoadData.py`).
- Provide a comprehensive routine that generates both intra-loader (speeds within each way) and inter-loader comparisons.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
import os
from pathlib import Path
import pandas as pd
from typing import List, Optional, Tuple, Dict
import argparse

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class V3DataVisualizer:
    """Specialized visualizer for v3 processed data quality assessment."""
    
    def __init__(self, data_dir: str = "Data"):
        """Initialize the visualizer."""
        self.data_dir = Path(data_dir)
        
        # Feature names for v3 data
        self.x_feature_names = [
            'vel_underhang_rad', 'vel_underhang_tan', 'vel_overhang_rad', 'vel_overhang_tan',
            'pos_underhang_rad', 'pos_underhang_tan', 'pos_overhang_rad', 'pos_overhang_tan',
            'omega', 'time'
        ]
        
        self.y_feature_names = [
            'acc_underhang_rad', 'acc_underhang_tan', 'acc_overhang_rad', 'acc_overhang_tan'
        ]
    
    def load_v3_data(self, category: str = "normal") -> tuple:
        """Load v3 processed data for a given category (normal or faulty)."""
        # Prefer Data/v3/ as primary location, fallback to Data/
        primary_dir = self.data_dir / "v3"
        candidates = [primary_dir, self.data_dir]
        for base in candidates:
            try:
                X_path = base / f"X_{category}_v3.pth"
                Y_path = base / f"Y_{category}_v3.pth"
                X_v3 = torch.load(X_path)
                Y_v3 = torch.load(Y_path)
                print(f"✓ Loaded v3 data [{category}] from '{base}': X={X_v3.shape}, Y={Y_v3.shape}")
                return X_v3, Y_v3
            except FileNotFoundError:
                continue
        print(f"❌ v3 data not found for category '{category}' in {primary_dir} or {self.data_dir}!")
        return None, None

    def load_legacy_normal(self) -> tuple:
        """Load legacy normal data produced by `LoadData.py` (displ/vel + omega)."""
        try:
            X = torch.load(self.data_dir / "X_normal.pth")
            Y = torch.load(self.data_dir / "Y_normal.pth")
            print(f"✓ Loaded legacy normal data: X={X.shape}, Y={Y.shape}")
            return X, Y
        except FileNotFoundError:
            print("⚠ legacy normal data not found (X_normal.pth / Y_normal.pth)")
            return None, None
    
    def load_comparison_data(self) -> dict:
        """Load data from all available versions for comparison."""
        data = {}
        
        # Try to load v2 data
        try:
            X_v2 = torch.load(self.data_dir / "X_normal_v2.pth")
            Y_v2 = torch.load(self.data_dir / "Y_normal_v2.pth")
            data['v2'] = (X_v2, Y_v2)
            print(f"✓ Loaded v2 data: X={X_v2.shape}, Y={Y_v2.shape}")
        except FileNotFoundError:
            print("⚠ v2 data not found")
        
        # Try to load original data
        try:
            X_orig = torch.load(self.data_dir / "X_normal.pth")
            Y_orig = torch.load(self.data_dir / "Y_normal.pth")
            data['original'] = (X_orig, Y_orig)
            print(f"✓ Loaded original data: X={X_orig.shape}, Y={Y_orig.shape}")
        except FileNotFoundError:
            print("⚠ original data not found")
        
        return data
    
    def analyze_magnitudes(self, X_v3, Y_v3, comparison_data=None):
        """Analyze magnitude ranges and identify issues."""
        print("\n" + "="*60)
        print("MAGNITUDE ANALYSIS")
        print("="*60)
        
        # Expected ranges
        expected_ranges = {
            'acceleration': (0.1, 10.0),      # m/s²
            'velocity': (0.005, 0.05),        # m/s (5-50 mm/s)
            'position': (1e-5, 5e-4)          # m (10-500 μm)
        }
        
        # Analyze v3 data
        print("\n📊 V3 Data Magnitude Analysis:")
        print("-" * 40)
        
        # Acceleration analysis (Y data)
        acc_ranges = []
        for i in range(Y_v3.shape[-1]):
            min_val = Y_v3[:, :, i].min().item()
            max_val = Y_v3[:, :, i].max().item()
            acc_ranges.append((min_val, max_val))
            print(f"  {self.y_feature_names[i]}: {min_val:.6f} to {max_val:.6f} m/s²")
        
        # Velocity analysis (X data, features 0-3)
        vel_ranges = []
        for i in range(4):
            min_val = X_v3[:, :, i].min().item()
            max_val = X_v3[:, :, i].max().item()
            vel_ranges.append((min_val, max_val))
            print(f"  {self.x_feature_names[i]}: {min_val:.6f} to {max_val:.6f} m/s")
        
        # Position analysis (X data, features 4-7)
        pos_ranges = []
        for i in range(4, 8):
            min_val = X_v3[:, :, i].min().item()
            max_val = X_v3[:, :, i].max().item()
            pos_ranges.append((min_val, max_val))
            print(f"  {self.x_feature_names[i]}: {min_val:.6f} to {max_val:.6f} m")
        
        # Omega analysis (X data, feature 8)
        omega_min = X_v3[:, :, 8].min().item()
        omega_max = X_v3[:, :, 8].max().item()
        print(f"  {self.x_feature_names[8]}: {omega_min:.6f} to {omega_max:.6f} rad/s")
        
        # Check against expected ranges
        print(f"\n🔍 Range Validation:")
        print("-" * 40)
        
        # Check accelerations
        acc_in_range = 0
        for i, (min_val, max_val) in enumerate(acc_ranges):
            if expected_ranges['acceleration'][0] <= max_val <= expected_ranges['acceleration'][1]:
                acc_in_range += 1
                status = "✅"
            else:
                status = "❌"
            print(f"  {status} {self.y_feature_names[i]}: {min_val:.3f} to {max_val:.3f} m/s²")
        
        # Check velocities
        vel_in_range = 0
        for i, (min_val, max_val) in enumerate(vel_ranges):
            if expected_ranges['velocity'][0] <= max_val <= expected_ranges['velocity'][1]:
                vel_in_range += 1
                status = "✅"
            else:
                status = "❌"
            print(f"  {status} {self.x_feature_names[i]}: {min_val:.6f} to {max_val:.6f} m/s")
        
        # Check positions
        pos_in_range = 0
        for i, (min_val, max_val) in enumerate(pos_ranges):
            if expected_ranges['position'][0] <= max_val <= expected_ranges['position'][1]:
                pos_in_range += 1
                status = "✅"
            else:
                status = "❌"
            print(f"  {status} {self.x_feature_names[i+4]}: {min_val:.6f} to {max_val:.6f} m")
        
        # Summary
        print(f"\n📈 Summary:")
        print(f"  Accelerations in range: {acc_in_range}/4 ({acc_in_range/4*100:.1f}%)")
        print(f"  Velocities in range: {vel_in_range}/4 ({vel_in_range/4*100:.1f}%)")
        print(f"  Positions in range: {pos_in_range}/4 ({pos_in_range/4*100:.1f}%)")
        
        # Compare with other versions if available
        if comparison_data:
            print(f"\n📊 Comparison with Other Versions:")
            print("-" * 40)
            
            for version, (X, Y) in comparison_data.items():
                print(f"\n{version.upper()} Version:")
                acc_min = Y.min().item()
                acc_max = Y.max().item()
                pos_min = X[:, :, :4].min().item()  # First 4 features are positions
                pos_max = X[:, :, :4].max().item()
                print(f"  Acceleration range: {acc_min:.6f} to {acc_max:.6f} m/s²")
                print(f"  Position range: {pos_min:.6f} to {pos_max:.6f} m")
    
    def plot_magnitude_comparison(self, X_v3, Y_v3, comparison_data=None):
        """Plot magnitude comparisons between versions."""
        print(f"\n📊 Generating magnitude comparison plots...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Acceleration comparison
        ax1 = axes[0, 0]
        versions = ['v3']
        acc_ranges = [Y_v3.min().item(), Y_v3.max().item()]
        
        if comparison_data:
            for version, (X, Y) in comparison_data.items():
                versions.append(version)
                acc_ranges.extend([Y.min().item(), Y.max().item()])
        
        # Plot acceleration ranges
        for i, version in enumerate(versions):
            if i == 0:
                min_val, max_val = acc_ranges[0], acc_ranges[1]
            else:
                min_val, max_val = acc_ranges[2*i], acc_ranges[2*i+1]
            
            ax1.barh(version, max_val - min_val, left=min_val, alpha=0.7, label=version)
        
        ax1.axvline(x=0.1, color='red', linestyle='--', alpha=0.7, label='Expected min')
        ax1.axvline(x=10.0, color='red', linestyle='--', alpha=0.7, label='Expected max')
        ax1.set_xlabel('Acceleration (m/s²)')
        ax1.set_title('Acceleration Magnitude Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Position comparison
        ax2 = axes[0, 1]
        pos_ranges = [X_v3[:, :, 4:8].min().item(), X_v3[:, :, 4:8].max().item()]
        
        if comparison_data:
            for version, (X, Y) in comparison_data.items():
                pos_ranges.extend([X[:, :, :4].min().item(), X[:, :, :4].max().item()])
        
        # Plot position ranges
        for i, version in enumerate(versions):
            if i == 0:
                min_val, max_val = pos_ranges[0], pos_ranges[1]
            else:
                min_val, max_val = pos_ranges[2*i], pos_ranges[2*i+1]
            
            ax2.barh(version, max_val - min_val, left=min_val, alpha=0.7, label=version)
        
        ax2.axvline(x=1e-5, color='red', linestyle='--', alpha=0.7, label='Expected min')
        ax2.axvline(x=5e-4, color='red', linestyle='--', alpha=0.7, label='Expected max')
        ax2.set_xlabel('Position (m)')
        ax2.set_title('Position Magnitude Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Time series comparison (first sample)
        ax3 = axes[1, 0]
        sample_idx = 0
        time_steps = 1000  # Plot first 1000 time steps
        
        # Plot v3 acceleration
        ax3.plot(Y_v3[sample_idx, :time_steps, 0].numpy(), 'b-', label='v3', linewidth=1)
        
        if comparison_data and 'original' in comparison_data:
            X_orig, Y_orig = comparison_data['original']
            ax3.plot(Y_orig[sample_idx, :time_steps, 0].numpy(), 'r-', label='original', linewidth=1)
        
        ax3.set_xlabel('Time Steps')
        ax3.set_ylabel('Acceleration (m/s²)')
        ax3.set_title('Acceleration Time Series Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Histogram comparison
        ax4 = axes[1, 1]
        
        # Plot v3 acceleration histogram
        ax4.hist(Y_v3[sample_idx, :, 0].numpy().flatten(), bins=50, alpha=0.7, 
                label='v3', density=True)
        
        if comparison_data and 'original' in comparison_data:
            X_orig, Y_orig = comparison_data['original']
            ax4.hist(Y_orig[sample_idx, :, 0].numpy().flatten(), bins=50, alpha=0.7, 
                    label='original', density=True)
        
        ax4.set_xlabel('Acceleration (m/s²)')
        ax4.set_ylabel('Density')
        ax4.set_title('Acceleration Distribution Comparison')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle('V3 Data Magnitude & Distribution Comparison', y=1.02)
        plt.show()
    
    def plot_feature_analysis(self, X_v3, Y_v3):
        """Detailed analysis of each feature."""
        print(f"\n🔍 Generating feature analysis plots...")
        
        # Create comprehensive feature analysis
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        
        # Sample to analyze
        sample_idx = 0
        time_steps = 5000  # Analyze first 5000 time steps
        
        # Plot accelerations (Y features)
        for i in range(4):
            ax = axes[0, i]
            data = Y_v3[sample_idx, :time_steps, i].numpy()
            ax.plot(data, linewidth=1)
            ax.set_title(f'{self.y_feature_names[i]}')
            ax.set_ylabel('Acceleration (m/s²)')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_val = np.mean(data)
            std_val = np.std(data)
            ax.text(0.02, 0.98, f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Plot velocities (X features 0-3)
        for i in range(4):
            ax = axes[1, i]
            data = X_v3[sample_idx, :time_steps, i].numpy()
            ax.plot(data, linewidth=1)
            ax.set_title(f'{self.x_feature_names[i]}')
            ax.set_ylabel('Velocity (m/s)')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_val = np.mean(data)
            std_val = np.std(data)
            ax.text(0.02, 0.98, f'Mean: {mean_val:.6f}\nStd: {std_val:.6f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Plot positions (X features 4-7)
        for i in range(4):
            ax = axes[2, i]
            data = X_v3[sample_idx, :time_steps, i+4].numpy()
            ax.plot(data, linewidth=1)
            ax.set_title(f'{self.x_feature_names[i+4]}')
            ax.set_ylabel('Position (m)')
            ax.set_xlabel('Time Steps')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_val = np.mean(data)
            std_val = np.std(data)
            ax.text(0.02, 0.98, f'Mean: {mean_val:.6f}\nStd: {std_val:.6f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        plt.suptitle('V3 Feature Analysis (Sample 0)', y=1.02)
        plt.show()
    
    def plot_drift_analysis(self, X_v3, Y_v3):
        """Analyze drift in the processed signals."""
        print(f"\n📈 Generating drift analysis plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Sample to analyze
        sample_idx = 0
        
        # Position drift analysis
        ax1 = axes[0, 0]
        positions = X_v3[sample_idx, :, 4:8].numpy()  # All position features
        time_axis = np.arange(positions.shape[0])
        
        for i in range(4):
            ax1.plot(time_axis, positions[:, i], label=self.x_feature_names[i+4], linewidth=1)
        
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Position (m)')
        ax1.set_title('Position Drift over Time — v3 normal')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Velocity drift analysis
        ax2 = axes[0, 1]
        velocities = X_v3[sample_idx, :, :4].numpy()  # All velocity features
        
        for i in range(4):
            ax2.plot(time_axis, velocities[:, i], label=self.x_feature_names[i], linewidth=1)
        
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Velocity (m/s)')
        ax2.set_title('Velocity Drift over Time — v3 normal')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Acceleration drift analysis
        ax3 = axes[1, 0]
        accelerations = Y_v3[sample_idx, :, :].numpy()  # All acceleration features
        
        for i in range(4):
            ax3.plot(time_axis, accelerations[:, i], label=self.y_feature_names[i], linewidth=1)
        
        ax3.set_xlabel('Time Steps')
        ax3.set_ylabel('Acceleration (m/s²)')
        ax3.set_title('Acceleration Drift over Time — v3 normal')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Cumulative drift analysis
        ax4 = axes[1, 1]
        
        # Calculate cumulative drift for each position feature
        drift_metrics = []
        for i in range(4):
            pos_data = positions[:, i]
            drift = np.abs(pos_data[-1] - pos_data[0]) / (np.max(np.abs(pos_data)) + 1e-10)
            drift_metrics.append(drift)
            ax4.bar(self.x_feature_names[i+4], drift, alpha=0.7)
        
        ax4.set_ylabel('Drift Score')
        ax4.set_title('Position Drift Metrics — v3 normal')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # Add drift score text
        avg_drift = np.mean(drift_metrics)
        ax4.text(0.5, 0.9, f'Average Drift: {avg_drift:.6f}', 
                transform=ax4.transAxes, ha='center',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        plt.tight_layout()
        plt.suptitle('Drift Analysis — v3 normal', y=1.02)
        plt.show()
    
    def plot_phase_space_analysis(self, X_v3, Y_v3):
        """Analyze phase space behavior of rotor-bearing system."""
        print(f"\n🔄 Generating phase space analysis plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Sample to analyze
        sample_idx = 0
        time_steps = 10000  # Analyze first 10000 time steps
        
        # Phase space plots for each bearing
        bearings = ['underhang', 'overhang']
        directions = ['rad', 'tan']
        
        for i, bearing in enumerate(bearings):
            for j, direction in enumerate(directions):
                ax = axes[i, j]
                
                # Get position and velocity data
                pos_idx = 4 + i * 2 + j  # Position features: 4-7
                vel_idx = i * 2 + j      # Velocity features: 0-3
                
                positions = X_v3[sample_idx, :time_steps, pos_idx].numpy()
                velocities = X_v3[sample_idx, :time_steps, vel_idx].numpy()
                
                # Create phase space plot with time-colored points and colorbar
                sc = ax.scatter(positions, velocities, c=np.arange(len(positions)),
                                cmap='viridis', alpha=0.6, s=1)
                ax.set_xlabel('Position (m)')
                ax.set_ylabel('Velocity (m/s)')
                ax.set_title(f'Phase Space: {bearing.capitalize()} {direction.upper()} (sample {sample_idx}, first {time_steps} pts)')
                ax.grid(True, alpha=0.3)
                cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Time index')
                
        # Arrows removed for cleaner visualization
        
        plt.tight_layout()
        plt.suptitle('Phase Space Analysis (v3 normal)', y=1.02)
        plt.show()
    
    def plot_frequency_spectrum_analysis(self, X_v3, Y_v3):
        """Analyze frequency spectrum of rotor-bearing vibrations."""
        print(f"\n📊 Generating frequency spectrum analysis plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Sample to analyze
        sample_idx = 0
        time_steps = 50000  # Use more data for better frequency resolution
        
        # Calculate sampling frequency
        fs = 50000  # 50 kHz sampling rate
        
        bearings = ['underhang', 'overhang']
        directions = ['rad', 'tan']
        
        for i, bearing in enumerate(bearings):
            for j, direction in enumerate(directions):
                ax = axes[i, j]
                
                # Get acceleration data
                acc_idx = i * 2 + j  # Acceleration features: 0-3
                acceleration = Y_v3[sample_idx, :time_steps, acc_idx].numpy()
                
                # Calculate FFT
                fft_result = np.fft.fft(acceleration)
                freqs = np.fft.fftfreq(len(acceleration), 1/fs)
                
                # Plot positive frequencies only
                positive_freqs = freqs[:len(freqs)//2]
                positive_fft = np.abs(fft_result[:len(freqs)//2])
                
                # Convert to dB scale
                fft_db = 20 * np.log10(positive_fft + 1e-10)
                
                ax.semilogx(positive_freqs, fft_db)
                ax.set_xlabel('Frequency (Hz)')
                ax.set_ylabel('Magnitude (dB)')
                ax.set_title(f'{bearing.capitalize()} {direction.upper()} Frequency Spectrum')
                ax.grid(True, alpha=0.3)
                
                # Add rotation frequency markers
                rotation_freq = 12.3  # Hz (737 rpm / 60)
                ax.axvline(x=rotation_freq, color='red', linestyle='--', alpha=0.7, label='1x RPM')
                ax.axvline(x=rotation_freq*2, color='orange', linestyle='--', alpha=0.7, label='2x RPM')
                ax.axvline(x=rotation_freq*3, color='yellow', linestyle='--', alpha=0.7, label='3x RPM')
                ax.legend()
        
        plt.tight_layout()
        plt.suptitle('Frequency Spectrum Analysis', y=1.02)
        plt.show()
    
    def plot_rotor_dynamics_analysis(self, X_v3, Y_v3):
        """Analyze rotor dynamics and orbital motion."""
        print(f"\n⚙️ Generating rotor dynamics analysis plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Sample to analyze
        sample_idx = 0
        time_steps = 10000
        
        # 1. Rotor orbit plot (position vs position)
        ax1 = axes[0, 0]
        underhang_rad_pos = X_v3[sample_idx, :time_steps, 4].numpy()  # pos_underhang_rad
        underhang_tan_pos = X_v3[sample_idx, :time_steps, 5].numpy()  # pos_underhang_tan
        overhang_rad_pos = X_v3[sample_idx, :time_steps, 6].numpy()   # pos_overhang_rad
        overhang_tan_pos = X_v3[sample_idx, :time_steps, 7].numpy()   # pos_overhang_tan
        
        # Scale positions for better visualization (convert to μm)
        scale_factor = 1e6  # Convert to μm
        
        ax1.scatter(underhang_rad_pos * scale_factor, underhang_tan_pos * scale_factor, 
                   c=np.arange(len(underhang_rad_pos)), cmap='viridis', alpha=0.6, s=1, label='Underhang')
        ax1.scatter(overhang_rad_pos * scale_factor, overhang_tan_pos * scale_factor, 
                   c=np.arange(len(overhang_rad_pos)), cmap='plasma', alpha=0.6, s=1, label='Overhang')
        ax1.set_xlabel('Radial Position (μm)')
        ax1.set_ylabel('Tangential Position (μm)')
        ax1.set_title('Rotor Orbit (pos_y vs pos_x) — v3 normal')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # 2. Rotor centerline plot
        ax2 = axes[0, 1]
        time_axis = np.arange(time_steps) / 50000  # Convert to seconds
        
        ax2.plot(time_axis, underhang_rad_pos * scale_factor, label='Underhang Radial', alpha=0.7)
        ax2.plot(time_axis, overhang_rad_pos * scale_factor, label='Overhang Radial', alpha=0.7)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Radial Position (μm)')
        ax2.set_title('Rotor Centerline Motion — v3 normal')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Rotor speed vs vibration amplitude
        ax3 = axes[1, 0]
        omega_values = X_v3[sample_idx, :time_steps, 8].numpy()  # Angular velocity
        vibration_amplitude = np.sqrt(underhang_rad_pos**2 + underhang_tan_pos**2) * scale_factor
        
        ax3.scatter(omega_values, vibration_amplitude, c=np.arange(len(omega_values)), 
                   cmap='viridis', alpha=0.6, s=1)
        ax3.set_xlabel('Angular Velocity (rad/s)')
        ax3.set_ylabel('Vibration Amplitude (μm)')
        ax3.set_title('Angular Speed vs Vibration Amplitude — v3 normal')
        ax3.grid(True, alpha=0.3)
        
        # 4. Rotor whirl analysis
        ax4 = axes[1, 1]
        # Calculate whirl frequency from phase difference
        phase_diff = np.angle(underhang_rad_pos + 1j * underhang_tan_pos)
        phase_diff = np.unwrap(phase_diff)
        whirl_freq = np.gradient(phase_diff) * 50000 / (2 * np.pi)  # Convert to Hz
        
        ax4.plot(time_axis, whirl_freq, alpha=0.7)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Whirl Frequency (Hz)')
        ax4.set_title('Rotor Whirl Frequency — v3 normal')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle('Rotor Dynamics Analysis — v3 normal', y=1.02)
        plt.show()
    
    def plot_bearing_health_indicators(self, X_v3, Y_v3):
        """Analyze bearing health indicators and vibration patterns."""
        print(f"\n🏥 Generating bearing health indicator plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Sample to analyze
        sample_idx = 0
        time_steps = 20000
        
        # 1. Kurtosis analysis (indicator of bearing health)
        ax1 = axes[0, 0]
        bearings = ['Underhang', 'Overhang']
        directions = ['Radial', 'Tangential']
        
        kurtosis_values = []
        labels = []
        
        for i, bearing in enumerate(bearings):
            for j, direction in enumerate(directions):
                acc_idx = i * 2 + j
                acceleration = Y_v3[sample_idx, :time_steps, acc_idx].numpy()
                
                # Calculate kurtosis
                mean = np.mean(acceleration)
                std = np.std(acceleration)
                kurtosis = np.mean(((acceleration - mean) / std) ** 4)
                kurtosis_values.append(kurtosis)
                labels.append(f'{bearing} {direction}')
        
        bars = ax1.bar(labels, kurtosis_values, alpha=0.7)
        ax1.set_ylabel('Kurtosis')
        ax1.set_title('Bearing Vibration Kurtosis — v3 normal (health indicator)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Color bars based on kurtosis values (higher = more impulsive = potential issues)
        for bar, kurt in zip(bars, kurtosis_values):
            if kurt > 5:
                bar.set_color('red')
            elif kurt > 3:
                bar.set_color('orange')
            else:
                bar.set_color('green')
        
        # 2. RMS vibration levels
        ax2 = axes[0, 1]
        rms_values = []
        
        for i, bearing in enumerate(bearings):
            for j, direction in enumerate(directions):
                acc_idx = i * 2 + j
                acceleration = Y_v3[sample_idx, :time_steps, acc_idx].numpy()
                rms = np.sqrt(np.mean(acceleration**2))
                rms_values.append(rms)
        
        bars = ax2.bar(labels, rms_values, alpha=0.7)
        ax2.set_ylabel('RMS Acceleration (m/s²)')
        ax2.set_title('RMS Acceleration — v3 normal')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 3. Peak-to-peak vibration
        ax3 = axes[1, 0]
        p2p_values = []
        
        for i, bearing in enumerate(bearings):
            for j, direction in enumerate(directions):
                acc_idx = i * 2 + j
                acceleration = Y_v3[sample_idx, :time_steps, acc_idx].numpy()
                p2p = np.max(acceleration) - np.min(acceleration)
                p2p_values.append(p2p)
        
        bars = ax3.bar(labels, p2p_values, alpha=0.7)
        ax3.set_ylabel('Peak-to-Peak Acceleration (m/s²)')
        ax3.set_title('Peak-to-Peak Acceleration — v3 normal')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 4. Crest factor (peak/RMS ratio)
        ax4 = axes[1, 1]
        crest_factors = []
        
        for i, bearing in enumerate(bearings):
            for j, direction in enumerate(directions):
                acc_idx = i * 2 + j
                acceleration = Y_v3[sample_idx, :time_steps, acc_idx].numpy()
                peak = np.max(np.abs(acceleration))
                rms = np.sqrt(np.mean(acceleration**2))
                crest_factor = peak / rms if rms > 0 else 0
                crest_factors.append(crest_factor)
        
        bars = ax4.bar(labels, crest_factors, alpha=0.7)
        ax4.set_ylabel('Crest Factor')
        ax4.set_title('Crest Factor (peak/RMS) — v3 normal')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle('Bearing Health Indicators — v3 normal', y=1.02)
        plt.show()

    # ---- New utilities for speed segmentation and multi-speed plotting ----

    def _infer_sampling_rate_from_time(self, X_v3_sample: np.ndarray) -> Optional[float]:
        """Infer sampling rate from the 'time' feature if present (index 9)."""
        if X_v3_sample.shape[1] >= 10:
            time_vals = X_v3_sample[:, 9]
            if np.all(np.isfinite(time_vals)) and np.ptp(time_vals) > 0:
                dt = np.median(np.diff(time_vals))
                if dt > 0:
                    return 1.0 / dt
        return None

    def _segment_indices_by_omega(self, omega: np.ndarray, min_segment_seconds: float,
                                  fs_fallback: float = 50000.0) -> List[Tuple[int, int]]:
        """Segment contiguous time ranges where omega is approximately constant.

        Uses a run-length style segmentation with a tolerance.
        """
        # Smooth and quantize to stabilize segmentation
        window = max(1, int(0.01 * len(omega)))  # 1% length smoothing window
        if window > 1:
            kernel = np.ones(window) / window
            omega_smooth = np.convolve(omega, kernel, mode='same')
        else:
            omega_smooth = omega

        # Quantize omega to nearest 1 rad/s to detect plateaus
        omega_q = np.round(omega_smooth, 0)

        # Infer fs from optional time column if available later; fallback provided
        # We'll fill fs in caller if time feature is present

        # Initial segmentation by change in quantized omega
        change_points = np.where(np.abs(np.diff(omega_q)) > 0)[0] + 1
        boundaries = np.r_[0, change_points, len(omega_q)]
        segments = [(int(boundaries[i]), int(boundaries[i+1])) for i in range(len(boundaries)-1)]

        # Enforce minimum segment duration by merging short segments with neighbors
        segments_merged: List[Tuple[int, int]] = []
        min_len = int(min_segment_seconds * fs_fallback)
        for start, end in segments:
            if not segments_merged:
                segments_merged.append((start, end))
                continue
            if (end - start) < min_len:
                prev_start, prev_end = segments_merged[-1]
                segments_merged[-1] = (prev_start, end)
            else:
                segments_merged.append((start, end))

        return segments_merged

    def _collect_speed_segments_v3(self, X_v3: torch.Tensor) -> Tuple[List[np.ndarray], List[float]]:
        """Collect per-speed segments across all samples for v3 data.

        Returns:
            segments: list of index arrays (time indices) per concatenated segment (sample-wise)
            omega_values: representative omega per segment
        """
        segments: List[np.ndarray] = []
        omega_values: List[float] = []
        num_samples = X_v3.shape[0]

        for s in range(num_samples):
            x_np = X_v3[s].numpy()
            fs_est = self._infer_sampling_rate_from_time(x_np) or 50000.0
            omega_series = x_np[:, 8]
            seg_indices = self._segment_indices_by_omega(omega_series, min_segment_seconds=4.0, fs_fallback=fs_est)
            for start, end in seg_indices:
                segments.append(np.arange(start, end, dtype=int))
                omega_values.append(float(np.median(omega_series[start:end])))

        return segments, omega_values

    def _downsample(self, arr: np.ndarray, max_points: int = 20000) -> np.ndarray:
        if arr.shape[0] <= max_points:
            return arr
        stride = max(1, arr.shape[0] // max_points)
        return arr[::stride]

    def plot_all_speeds_side_by_side(self, X_v3: torch.Tensor, Y_v3: torch.Tensor,
                                      sample_idx: Optional[int] = None,
                                      time_steps: Optional[int] = None) -> None:
        """Plot selected signals for all speeds side by side.

        If sample_idx is None, iterate all samples. Otherwise, only that sample.
        """
        samples_to_show = [sample_idx] if sample_idx is not None else list(range(X_v3.shape[0]))

        # For each sample, use median omega as label
        labels = []
        for s in samples_to_show:
            omega_med = float(torch.median(X_v3[s, :, 8]).item())
            labels.append(omega_med)

        num_cols = 4
        num_rows = int(np.ceil(len(samples_to_show) / num_cols))
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(4*num_cols, 3*num_rows), squeeze=False)

        for idx, s in enumerate(samples_to_show):
            r, c = divmod(idx, num_cols)
            ax = axes[r, c]
            t_slice = slice(0, time_steps) if time_steps else slice(None)
            # Plot one example channel per type: acc_u_rad (Y0), vel_u_rad (X0), pos_u_rad (X4)
            ax.plot(self._downsample(Y_v3[s, t_slice, 0].numpy()), label='acc_underhang_rad', alpha=0.8)
            ax.plot(self._downsample(X_v3[s, t_slice, 0].numpy()), label='vel_underhang_rad', alpha=0.8)
            ax.plot(self._downsample(X_v3[s, t_slice, 4].numpy()), label='pos_underhang_rad', alpha=0.8)
            ax.set_title(f'Underhang acc/vel/pos — sample {s} (ω≈{labels[idx]:.1f} rad/s)')
            ax.grid(True, alpha=0.3)
            if r == 0 and c == 0:
                ax.legend(fontsize=8)

        # Hide any unused axes
        for j in range(idx+1, num_rows*num_cols):
            r, c = divmod(j, num_cols)
            axes[r, c].axis('off')

        plt.tight_layout()
        plt.suptitle('All speeds side-by-side — v3 normal (per-sample median ω labeled)', y=1.02)
        plt.show()

    def plot_phase_space_all_speeds(self, X_v3: torch.Tensor,
                                     time_steps: Optional[int] = None,
                                     overlay: bool = False) -> None:
        """Phase-space plots per bearing/direction across all speeds.

        If overlay=True, overlay all speeds in each subplot; otherwise side-by-side grid.
        """
        bearings = ['underhang', 'overhang']
        directions = ['rad', 'tan']

        if overlay:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            for i, bearing in enumerate(bearings):
                for j, direction in enumerate(directions):
                    ax = axes[i, j]
                    for s in range(X_v3.shape[0]):
                        t_slice = slice(0, time_steps) if time_steps else slice(None)
                        pos_idx = 4 + i * 2 + j
                        vel_idx = i * 2 + j
                        pos = self._downsample(X_v3[s, t_slice, pos_idx].numpy())
                        vel = self._downsample(X_v3[s, t_slice, vel_idx].numpy())
                        omega_med = float(torch.median(X_v3[s, :, 8]).item())
                        ax.plot(pos, vel, alpha=0.6, label=f'sample {s} (ω≈{omega_med:.0f})')
                    ax.set_xlabel('Position (m)')
                    ax.set_ylabel('Velocity (m/s)')
                    ax.set_title(f'Phase Space (overlay): {bearing.capitalize()} {direction.upper()} — v3 normal')
                    ax.grid(True, alpha=0.3)
            # Single shared legend
            handles, labels = [], []
            for i in range(2):
                for j in range(2):
                    h, l = axes[i, j].get_legend_handles_labels()
                    handles += h
                    labels += l
            if handles:
                fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=8)
            plt.tight_layout()
            plt.suptitle('Phase Space across all speeds — v3 normal (overlay by sample)', y=1.02)
            plt.show()
            return

        num_cols = 4
        num_rows = int(np.ceil(X_v3.shape[0] / num_cols))
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(4*num_cols, 3*num_rows), squeeze=False)
        for s in range(X_v3.shape[0]):
            r, c = divmod(s, num_cols)
            ax = axes[r, c]
            t_slice = slice(0, time_steps) if time_steps else slice(None)
            pos = self._downsample(X_v3[s, t_slice, 4].numpy())
            vel = self._downsample(X_v3[s, t_slice, 0].numpy())
            omega_med = float(torch.median(X_v3[s, :, 8]).item())
            ax.scatter(pos, vel, s=1, alpha=0.6)
            ax.set_title(f'Underhang Radial Phase — sample {s} (ω≈{omega_med:.0f})')
            ax.grid(True, alpha=0.3)
            if r == num_rows - 1:
                ax.set_xlabel('pos_underhang_rad (m)')
            if c == 0:
                ax.set_ylabel('vel_underhang_rad (m/s)')
        for j in range(s+1, num_rows*num_cols):
            r, c = divmod(j, num_cols)
            axes[r, c].axis('off')
        plt.tight_layout()
        plt.suptitle('Phase space per speed — v3 normal (underhang radial)', y=1.02)
        plt.show()

    def plot_concatenated_phase_with_speed_colors(self, X_v3: torch.Tensor,
                                                   sample: int = 0,
                                                   time_steps: Optional[int] = None) -> None:
        """Concatenate segments for one sample and color by speed transitions (~5s).

        Uses omega-based segmentation and changes color for each segment.
        """
        x_np = X_v3[sample].numpy()
        t_slice = slice(0, time_steps) if time_steps else slice(None)
        x_np = x_np[t_slice]
        fs_est = self._infer_sampling_rate_from_time(x_np) or 50000.0
        segments = self._segment_indices_by_omega(x_np[:, 8], min_segment_seconds=4.0, fs_fallback=fs_est)

        bearings = ['underhang', 'overhang']
        directions = ['rad', 'tan']
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        cmap = plt.get_cmap('tab20')

        for i, bearing in enumerate(bearings):
            for j, direction in enumerate(directions):
                ax = axes[i, j]
                pos_idx = 4 + i * 2 + j
                vel_idx = i * 2 + j
                for seg_id, (start, end) in enumerate(segments):
                    pos = x_np[start:end, pos_idx]
                    vel = x_np[start:end, vel_idx]
                    color = cmap(seg_id % 20)
                    ax.plot(pos, vel, color=color, alpha=0.8)
                ax.set_xlabel('Position (m)')
                ax.set_ylabel('Velocity (m/s)')
                ax.set_title(f'{bearing.capitalize()} {direction.upper()} (segments colored by speed)')
                ax.grid(True, alpha=0.3)

        # Segment color legend (index only)
        handles = [plt.Line2D([0], [0], color=cmap(i % 20), lw=2) for i in range(len(segments))]
        labels = [f'segment {i+1}' for i in range(len(segments))]
        axes[0, 0].legend(handles, labels, fontsize=8, ncol=4, loc='upper right')

        plt.tight_layout()
        plt.suptitle(f'Concatenated phase by speed segments — v3 sample {sample}', y=1.02)
        plt.show()

    def compare_v3_vs_legacy_phase_trajectories(self, X_v3: torch.Tensor, X_legacy: torch.Tensor,
                                                 sample_v3: int = 0, sample_legacy: int = 0,
                                                 time_steps: Optional[int] = None) -> None:
        """Side-by-side comparison of phase trajectories for v3 vs legacy loader outputs.

        Shows underhang radial pos-vel and overhang radial pos-vel.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # v3
        v3_slice = slice(0, time_steps) if time_steps else slice(None)
        u_pos_v3 = self._downsample(X_v3[sample_v3, v3_slice, 4].numpy())
        u_vel_v3 = self._downsample(X_v3[sample_v3, v3_slice, 0].numpy())
        o_pos_v3 = self._downsample(X_v3[sample_v3, v3_slice, 6].numpy())
        o_vel_v3 = self._downsample(X_v3[sample_v3, v3_slice, 2].numpy())

        # legacy (displacements at 0-3, velocities at 4-7)
        leg_slice = slice(0, time_steps) if time_steps else slice(None)
        u_pos_leg = self._downsample(X_legacy[sample_legacy, leg_slice, 0].numpy())
        u_vel_leg = self._downsample(X_legacy[sample_legacy, leg_slice, 4].numpy())
        o_pos_leg = self._downsample(X_legacy[sample_legacy, leg_slice, 2].numpy())
        o_vel_leg = self._downsample(X_legacy[sample_legacy, leg_slice, 6].numpy())

        axes[0, 0].plot(u_pos_v3, u_vel_v3, label='v3', alpha=0.8)
        axes[0, 0].plot(u_pos_leg, u_vel_leg, label='legacy', alpha=0.8)
        axes[0, 0].set_title('Underhang Radial (pos vs vel) — v3 vs legacy')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()

        axes[0, 1].plot(o_pos_v3, o_vel_v3, label='v3', alpha=0.8)
        axes[0, 1].plot(o_pos_leg, o_vel_leg, label='legacy', alpha=0.8)
        axes[0, 1].set_title('Overhang Radial (pos vs vel) — v3 vs legacy')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()

        # Also compare tangential if available (v3 pos 5/7, vel 1/3; legacy pos 1/3, vel 5/7)
        u_pos_t_v3 = self._downsample(X_v3[sample_v3, v3_slice, 5].numpy())
        u_vel_t_v3 = self._downsample(X_v3[sample_v3, v3_slice, 1].numpy())
        o_pos_t_v3 = self._downsample(X_v3[sample_v3, v3_slice, 7].numpy())
        o_vel_t_v3 = self._downsample(X_v3[sample_v3, v3_slice, 3].numpy())

        u_pos_t_leg = self._downsample(X_legacy[sample_legacy, leg_slice, 1].numpy())
        u_vel_t_leg = self._downsample(X_legacy[sample_legacy, leg_slice, 5].numpy())
        o_pos_t_leg = self._downsample(X_legacy[sample_legacy, leg_slice, 3].numpy())
        o_vel_t_leg = self._downsample(X_legacy[sample_legacy, leg_slice, 7].numpy())

        axes[1, 0].plot(u_pos_t_v3, u_vel_t_v3, label='v3', alpha=0.8)
        axes[1, 0].plot(u_pos_t_leg, u_vel_t_leg, label='legacy', alpha=0.8)
        axes[1, 0].set_title('Underhang Tangential (pos vs vel) — v3 vs legacy')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()

        axes[1, 1].plot(o_pos_t_v3, o_vel_t_v3, label='v3', alpha=0.8)
        axes[1, 1].plot(o_pos_t_leg, o_vel_t_leg, label='legacy', alpha=0.8)
        axes[1, 1].set_title('Overhang Tangential (pos vs vel) — v3 vs legacy')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()

        plt.tight_layout()
        plt.suptitle('Phase trajectories — v3 normal vs legacy normal', y=1.02)
        plt.show()

    def comprehensive_speed_and_loader_comparison(self) -> None:
        """Run comprehensive comparisons across speeds and across loaders.

        - v3: side-by-side speeds, overlay phase, concatenated segments with color changes
        - legacy: load and compare phase trajectories with v3
        """
        X_v3, Y_v3 = self.load_v3_data("normal")
        if X_v3 is None:
            return

        # Intra-v3 comparisons across speeds
        self.plot_all_speeds_side_by_side(X_v3, Y_v3, sample_idx=None, time_steps=20000)
        self.plot_phase_space_all_speeds(X_v3, time_steps=20000, overlay=True)
        # Concatenated segments for sample 0 (adjust if needed)
        self.plot_concatenated_phase_with_speed_colors(X_v3, sample=0, time_steps=100000)

        # Inter-loader comparison
        X_leg, Y_leg = self.load_legacy_normal()
        if X_leg is not None:
            self.compare_v3_vs_legacy_phase_trajectories(X_v3, X_leg, sample_v3=0, sample_legacy=0, time_steps=20000)

        # Compare normal vs a faulty category at similar speed if available
        faulty_candidates = [
            'overhang_ball_fault_20g', 'underhang_ball_fault_20g',
            'overhang_outer_race_fault_20g', 'underhang_outer_race_fault_20g',
            'horizontal_misalignment_fault_1.0mm', 'vertical_misalignment_fault_1.27mm',
            'imbalance_fault_20g'
        ]
        for cat in faulty_candidates:
            X_fault, Y_fault = self.load_v3_data(cat)
            if X_fault is not None:
                try:
                    self.plot_normal_vs_faulty_same_speed_phase(X_v3, X_fault, normal_label='normal', faulty_label=cat)
                except Exception:
                    pass
                break
    
    def comprehensive_quality_assessment(self):
        """Perform comprehensive quality assessment of v3 data."""
        print("V3 Data Quality Assessment")
        print("=" * 60)
        
        # Load v3 data
        X_v3, Y_v3 = self.load_v3_data("normal")
        if X_v3 is None:
            return
        
        # Load comparison data
        comparison_data = self.load_comparison_data()
        
        # Perform analyses
        self.analyze_magnitudes(X_v3, Y_v3, comparison_data)
        self.plot_magnitude_comparison(X_v3, Y_v3, comparison_data)
        self.plot_feature_analysis(X_v3, Y_v3)
        self.plot_drift_analysis(X_v3, Y_v3)
        
        # Enhanced rotor-bearing specific visualizations
        self.plot_phase_space_analysis(X_v3, Y_v3)
        self.plot_frequency_spectrum_analysis(X_v3, Y_v3)
        self.plot_rotor_dynamics_analysis(X_v3, Y_v3)
        self.plot_bearing_health_indicators(X_v3, Y_v3)

        # Additional comprehensive comparisons
        self.comprehensive_speed_and_loader_comparison()
        
        print(f"\n✅ Quality assessment complete!")
        print(f"Check the plots above to evaluate the v3 data quality.")

    def plot_normal_vs_faulty_same_speed_phase(self, X_norm: torch.Tensor, X_fault: torch.Tensor,
                                               normal_label: str = 'normal', faulty_label: str = 'faulty',
                                               time_steps: Optional[int] = 50000) -> None:
        """Compare phase plots of normal vs faulty at similar rotation speeds.

        Matches samples by nearest median ω and overlays their phase (pos vs vel) for
        underhang/overhang radial and tangential components.
        """
        # Compute median ω per sample
        omega_norm = [float(torch.median(X_norm[i, :, 8]).item()) for i in range(X_norm.shape[0])]
        omega_fault = [float(torch.median(X_fault[i, :, 8]).item()) for i in range(X_fault.shape[0])]

        # For each normal sample, find closest faulty sample by ω
        pairs = []
        for i, w in enumerate(omega_norm):
            j = int(np.argmin([abs(w - wf) for wf in omega_fault]))
            pairs.append((i, j))

        # Limit to first 4 pairs for readability
        pairs = pairs[:4]

        bearings = ['underhang', 'overhang']
        directions = ['rad', 'tan']
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        for i, bearing in enumerate(bearings):
            for j, direction in enumerate(directions):
                ax = axes[i, j]
                pos_idx = 4 + i * 2 + j
                vel_idx = i * 2 + j
                for (idx_n, idx_f) in pairs:
                    t_slice = slice(0, time_steps) if time_steps else slice(None)
                    pos_n = self._downsample(X_norm[idx_n, t_slice, pos_idx].numpy(), 40000)
                    vel_n = self._downsample(X_norm[idx_n, t_slice, vel_idx].numpy(), 40000)
                    pos_f = self._downsample(X_fault[idx_f, t_slice, pos_idx].numpy(), 40000)
                    vel_f = self._downsample(X_fault[idx_f, t_slice, vel_idx].numpy(), 40000)
                    w_n = omega_norm[idx_n]
                    w_f = omega_fault[idx_f]
                    ax.plot(pos_n, vel_n, alpha=0.7, label=f'{normal_label} (ω≈{w_n:.0f})')
                    ax.plot(pos_f, vel_f, alpha=0.7, label=f'{faulty_label} (ω≈{w_f:.0f})')
                ax.set_xlabel('Position (m)')
                ax.set_ylabel('Velocity (m/s)')
                ax.set_title(f'Phase: {bearing.capitalize()} {direction.upper()} — normal vs {faulty_label}')
                ax.grid(True, alpha=0.3)
                # Unique legend entries
                handles, labels = ax.get_legend_handles_labels()
                uniq = dict(zip(labels, handles))
                ax.legend(uniq.values(), uniq.keys(), fontsize=8)

        plt.tight_layout()
        plt.suptitle(f'Phase comparison at similar speeds — {normal_label} vs {faulty_label} (v3)', y=1.02)
        plt.show()

def main():
    """Run the v3 data quality assessment and optional normal vs faulty comparison."""
    parser = argparse.ArgumentParser(description="Visualize v3 processed data and optionally compare normal vs a faulty category.")
    parser.add_argument(
        "--faulty-category", "--faulty", dest="faulty_category", type=str, default=None,
        help="Faulty category label (e.g., 'imbalance_fault_20g', 'overhang_outer_race_fault_20g', etc.)."
    )
    parser.add_argument(
        "--data-dir", dest="data_dir", type=str, default="Data",
        help="Base data directory containing v3 tensors (default: Data)."
    )
    parser.add_argument(
        "--compare-steps", dest="compare_steps", type=int, default=50000,
        help="Number of time steps to use for normal vs faulty comparison plots (default: 50000)."
    )
    args = parser.parse_args()

    visualizer = V3DataVisualizer(data_dir=args.data_dir)
    # Always run the general quality assessment on normal
    visualizer.comprehensive_quality_assessment()

    # If a faulty category was provided, generate normal vs faulty comparison plots
    if args.faulty_category:
        print(f"\nRunning normal vs faulty comparison for category: {args.faulty_category}")
        X_norm, Y_norm = visualizer.load_v3_data("normal")
        X_fault, Y_fault = visualizer.load_v3_data(args.faulty_category)
        if X_norm is None or X_fault is None:
            print("⚠ Unable to load both normal and faulty v3 datasets; skipping comparison plots.")
        else:
            visualizer.plot_normal_vs_faulty_same_speed_phase(
                X_norm, X_fault,
                normal_label="normal",
                faulty_label=args.faulty_category,
                time_steps=args.compare_steps,
            )

if __name__ == "__main__":
    main() 