"""
Complete implementation of the MaFaulDa signal processing plan.

This script implements the comprehensive 3-phase approach:
Phase 1: Foundational Data Preparation and Universal Scaling
Phase 2: Comparative Evaluation of Robust Drift Mitigation Strategies  
Phase 3: Full Dataset Generation for PINN

Notes specific to this dataset:
- There is no tachometer signal used. The rotational speed (omega) is encoded in
  the CSV filename as frequency in Hz (e.g., Data/normal/12.288.csv).
- We parse that filename, convert to rad/s (omega = Hz * 2π), and store a constant
  omega per file as an input feature.

The goal is to establish a single, robust signal processing pipeline that works
effectively on both clean and pathological accelerometer channels.
"""

import pandas as pd
import numpy as np
import torch
import os
from scipy.signal import butter, filtfilt, savgol_filter
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Callable
import warnings
from dataclasses import dataclass
import time
try:
    # Reuse category paths from legacy loader for consistent folder mapping
    from Data.LoadData import data_paths as LEGACY_DATA_PATHS
except Exception:
    # Robust fallback mirroring Data/LoadData.py so all categories are available
    LEGACY_DATA_PATHS = {
        'normal': 'Data/normal/',
        'overhang_ball_fault_0g': 'Data/overhang/ball_fault/0g/',
        'overhang_ball_fault_6g': 'Data/overhang/ball_fault/6g/',
        'overhang_ball_fault_20g': 'Data/overhang/ball_fault/20g/',
        'overhang_ball_fault_35g': 'Data/overhang/ball_fault/35g/',
        'overhang_cage_fault_0g': 'Data/overhang/cage_fault/0g/',
        'overhang_cage_fault_6g': 'Data/overhang/cage_fault/6g/',
        'overhang_cage_fault_20g': 'Data/overhang/cage_fault/20g/',
        'overhang_cage_fault_35g': 'Data/overhang/cage_fault/35g/',
        'overhang_outer_race_fault_0g': 'Data/overhang/outer_race/0g/',
        'overhang_outer_race_fault_6g': 'Data/overhang/outer_race/6g/',
        'overhang_outer_race_fault_20g': 'Data/overhang/outer_race/20g/',
        'overhang_outer_race_fault_35g': 'Data/overhang/outer_race/35g/',
        'underhang_ball_fault_0g': 'Data/underhang/ball_fault/0g/',
        'underhang_ball_fault_6g': 'Data/underhang/ball_fault/6g/',
        'underhang_ball_fault_20g': 'Data/underhang/ball_fault/20g/',
        'underhang_ball_fault_35g': 'Data/underhang/ball_fault/35g/',
        'underhang_cage_fault_0g': 'Data/underhang/cage_fault/0g/',
        'underhang_cage_fault_6g': 'Data/underhang/cage_fault/6g/',
        'underhang_cage_fault_20g': 'Data/underhang/cage_fault/20g/',
        'underhang_cage_fault_35g': 'Data/underhang/cage_fault/35g/',
        'underhang_outer_race_fault_0g': 'Data/underhang/outer_race/0g/',
        'underhang_outer_race_fault_6g': 'Data/underhang/outer_race/6g/',
        'underhang_outer_race_fault_20g': 'Data/underhang/outer_race/20g/',
        'underhang_outer_race_fault_35g': 'Data/underhang/outer_race/35g/',
        'horizontal_misalignment_fault_0.5mm': 'Data/horizontal-misalignment/0.5mm/',
        'horizontal_misalignment_fault_1.0mm': 'Data/horizontal-misalignment/1.0mm/',
        'horizontal_misalignment_fault_1.5mm': 'Data/horizontal-misalignment/1.5mm/',
        'horizontal_misalignment_fault_2.0mm': 'Data/horizontal-misalignment/2.0mm/',
        'vertical_misalignment_fault_0.51mm': 'Data/vertical-misalignment/0.51mm/',
        'vertical_misalignment_fault_0.63mm': 'Data/vertical-misalignment/0.63mm/',
        'vertical_misalignment_fault_1.27mm': 'Data/vertical-misalignment/1.27mm/',
        'vertical_misalignment_fault_1.40mm': 'Data/vertical-misalignment/1.40mm/',
        'vertical_misalignment_fault_1.78mm': 'Data/vertical-misalignment/1.78mm/',
        'vertical_misalignment_fault_1.90mm': 'Data/vertical-misalignment/1.90mm/',
        'imbalance_fault_6g': 'Data/imbalance/6g/',
        'imbalance_fault_10g': 'Data/imbalance/10g/',
        'imbalance_fault_15g': 'Data/imbalance/15g/',
        'imbalance_fault_20g': 'Data/imbalance/20g/',
        'imbalance_fault_25g': 'Data/imbalance/25g/',
        'imbalance_fault_30g': 'Data/imbalance/30g/',
        'imbalance_fault_35g': 'Data/imbalance/35g/',
    }

# Constants from the MaFaulDa documentation
SAMPLING_RATE = 50000  # 50 kHz
SENSITIVITY = 0.0102   # V per m/s^2 (100 mV/g = 10.2 mV per m/s^2)
dt = 1.0 / SAMPLING_RATE  # Time step
GRAVITY = 9.80665  # m/s^2

# Column names for clarity
COLUMN_NAMES = [
    'tachometer',
    'acc_underhang_ax', 'acc_underhang_rad', 'acc_underhang_tan',
    'acc_overhang_ax', 'acc_overhang_rad', 'acc_overhang_tan',
    'microphone'
]

# Expected magnitude ranges for validation (adjusted based on actual data characteristics)
EXPECTED_RANGES = {
    'acceleration': (0.1, 5.0),       # m/s² (adjusted based on actual data)
    'velocity': (0.001, 0.03),        # m/s (1-30 mm/s, adjusted)
    'position': (1e-6, 1e-3)          # m (1 μm to 1 mm, adjusted)
}

@dataclass
class ProcessingResult:
    """Container for processing results and metrics."""
    acceleration: np.ndarray
    velocity: np.ndarray
    position: np.ndarray
    drift_score: float
    stability_score: float
    physical_score: float
    total_score: float
    parameters: Dict

class MaFaulDaProcessorV3:
    """Advanced processor implementing the complete 3-phase plan."""
    
    def __init__(self, default_cutoff_hz: float = 2.0, forced_strategy: Optional[str] = None,
                 assume_csv_is_acceleration: bool = False,
                 evaluation_scope: str = 'normal'):
        """
        Initialize the processor for comprehensive strategy evaluation.
        """
        # Normalize and validate forced strategy if provided
        if forced_strategy is not None:
            forced_strategy = forced_strategy.upper()
            if forced_strategy not in {'A', 'B', 'C', 'D'}:
                raise ValueError("forced_strategy must be one of {'A','B','C','D'} or None")

        self.forced_strategy: Optional[str] = forced_strategy
        self.optimal_strategy = forced_strategy or 'B'  # Will be updated by Phase 2 when enabled
        self.optimal_parameters = {
            'clean_channel': {'params': {'strategy': self.optimal_strategy, 'cutoff_freq': default_cutoff_hz}},
            'pathological_channel': {'params': {'strategy': self.optimal_strategy, 'cutoff_freq': default_cutoff_hz}}
        }
        self.default_cutoff_hz = default_cutoff_hz
        self.assume_csv_is_acceleration: bool = assume_csv_is_acceleration
        self.cutoff_grid_hz = np.array([0.5, 1.0, 2.0, 3.0, 5.0], dtype=float)
        self.evaluation_max_samples: int = 250_000  # Cap samples per file during evaluation
        # Evaluation scope: 'normal' (default) or 'all' (use every category in LEGACY_DATA_PATHS)
        if evaluation_scope not in {'normal', 'all'}:
            raise ValueError("evaluation_scope must be 'normal' or 'all'")
        self.evaluation_scope: str = evaluation_scope
        
        if self.forced_strategy:
            print(f"Initialized processor (Forced Strategy {self.forced_strategy}, cutoff={default_cutoff_hz:.2f} Hz)")
        else:
            print(f"Initialized processor (Strategy B default, cutoff={default_cutoff_hz:.2f} Hz)")

        print(f"Input interpretation: {'ACCELERATION (no conversion)' if self.assume_csv_is_acceleration else 'VOLTAGE -> acceleration via sensitivity'}")
        print(f"Evaluation scope: {self.evaluation_scope}")
    
    def _load_data_from_directory(self, target_dir: Path, category_name: Optional[str] = None) -> List[Dict]:
        """Load all CSV files from a directory and convert to acceleration arrays.

        Returns a list of dictionaries (one per file) holding accelerations for all channels,
        time, filename and optional category.
        """
        if not target_dir.exists():
            print(f"Error: {target_dir} directory not found!")
            return []
        csv_files = sorted(target_dir.glob("*.csv"))
        print(f"Loading {len(csv_files)} files from {target_dir}...")

        all_data: List[Dict] = []
        for i, csv_file in enumerate(csv_files):
            if i % 10 == 0:
                print(f"  Loading file {i+1}/{len(csv_files)}: {csv_file.name}")
            try:
                df_raw = pd.read_csv(csv_file, header=None, names=COLUMN_NAMES)

                file_data: Dict[str, np.ndarray] = {}
                for col in [
                    'acc_underhang_ax', 'acc_underhang_rad', 'acc_underhang_tan',
                    'acc_overhang_ax', 'acc_overhang_rad', 'acc_overhang_tan',
                ]:
                    if self.assume_csv_is_acceleration:
                        acceleration = df_raw[col].values
                    else:
                        voltage_signal = df_raw[col].values
                        acceleration = self._convert_voltage_to_acceleration(voltage_signal)
                    file_data[col] = acceleration

                file_data['time'] = np.arange(len(df_raw)) * dt
                file_data['filename'] = csv_file.name
                if category_name is not None:
                    file_data['category'] = category_name

                all_data.append(file_data)
            except Exception as e:
                print(f"Error loading {csv_file.name}: {e}")
                continue

        return all_data

    def _load_all_normal_data(self) -> List[Dict]:
        """Load all normal data files for comprehensive evaluation (backwards compatible)."""
        normal_dir = Path("Data/normal")
        all_data = self._load_data_from_directory(normal_dir, category_name='normal')
        print(f"Successfully loaded {len(all_data)} normal files")
        return all_data

    def _load_all_categories_data(self) -> List[Dict]:
        """Load all data files across every category defined in LEGACY_DATA_PATHS."""
        print("Loading data across all categories for strategy evaluation...")
        aggregated: List[Dict] = []
        # Ensure deterministic order: normal first, then alphabetically
        categories = list(LEGACY_DATA_PATHS.items())
        categories.sort(key=lambda kv: (kv[0] != 'normal', kv[0]))
        for category_name, category_dir in categories:
            dir_path = Path(category_dir)
            category_data = self._load_data_from_directory(dir_path, category_name=category_name)
            aggregated.extend(category_data)
            print(f"  -> {category_name}: {len(category_data)} files")
        print(f"Successfully loaded {len(aggregated)} files across {len(categories)} categories")
        return aggregated
    
    def _convert_voltage_to_acceleration(self, voltage_signal: np.ndarray) -> np.ndarray:
        """Convert voltage signal to acceleration in m/s²."""
        # Convert voltage directly to m/s²
        # SENSITIVITY is already in V per m/s², so no need for gravity conversion
        acceleration_ms2 = voltage_signal / SENSITIVITY
        return acceleration_ms2
    
    def _design_filter(self, cutoff_freq: float, filter_order: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """Design high-pass Butterworth filter."""
        return butter(filter_order, cutoff_freq, btype='high', analog=False, fs=SAMPLING_RATE)
    
    def _apply_filter(self, signal: np.ndarray, cutoff_freq: float, filter_order: int = 4) -> np.ndarray:
        """Apply high-pass filter to signal."""
        b, a = self._design_filter(cutoff_freq, filter_order)
        return filtfilt(b, a, signal)
    
    def _integrate_acceleration_to_velocity(self, acceleration: np.ndarray) -> np.ndarray:
        """Integrate acceleration to velocity using cumulative trapezoid."""
        return cumulative_trapezoid(acceleration, dx=dt, initial=0)
    
    def _integrate_velocity_to_position(self, velocity: np.ndarray) -> np.ndarray:
        """Integrate velocity to position using cumulative trapezoid."""
        return cumulative_trapezoid(velocity, dx=dt, initial=0)
    
    def _linear_detrend(self, signal: np.ndarray) -> np.ndarray:
        """Apply linear detrending to signal."""
        x = np.arange(len(signal))
        coeffs = np.polyfit(x, signal, 1)
        trend = np.polyval(coeffs, x)
        return signal - trend
    
    def _fit_drift_curve(self, velocity: np.ndarray, window_length: int = 1001) -> np.ndarray:
        """Fit a smooth curve to model velocity drift."""
        # Use Savitzky-Golay filter to smooth the velocity signal
        if len(velocity) < window_length:
            window_length = len(velocity) // 2
            if window_length % 2 == 0:
                window_length += 1
        
        return savgol_filter(velocity, window_length, 3)
    
    def _velocity_fit_differentiation(self, acceleration: np.ndarray) -> np.ndarray:
        """Apply velocity-fit differentiation for complex drift correction."""
        # Step 1: Integrate to get drifting velocity
        raw_velocity = self._integrate_acceleration_to_velocity(acceleration)
        
        # Step 2: Fit smooth curve to model drift
        drift_model = self._fit_drift_curve(raw_velocity)
        
        # Step 3: Differentiate to get acceleration trend
        acceleration_trend = np.gradient(drift_model, dt)
        
        # Step 4: Subtract trend from original
        return acceleration - acceleration_trend
    
    def strategy_a_filtering_only(self, acceleration: np.ndarray, cutoff_freq: float) -> ProcessingResult:
        """Strategy A: Optimized filtering only."""
        # Enforce zero-mean to reduce integration drift
        acceleration = acceleration - np.mean(acceleration)

        # Apply high-pass filter (pre-integration conditioning)
        filtered_acc = self._apply_filter(acceleration, cutoff_freq)

        # Integrate to velocity (do not re-filter the integrated signal)
        velocity = self._integrate_acceleration_to_velocity(filtered_acc)

        # Integrate velocity to position
        position = self._integrate_velocity_to_position(velocity)
        
        # Calculate scores
        scores = self._calculate_scores(position, velocity, filtered_acc)
        
        return ProcessingResult(
            acceleration=filtered_acc,
            velocity=velocity,
            position=position,
            **scores,
            parameters={'strategy': 'A', 'cutoff_freq': cutoff_freq}
        )
    
    def strategy_b_detrend_filtering(self, acceleration: np.ndarray, cutoff_freq: float) -> ProcessingResult:
        """Strategy B: Detrending followed by filtering."""
        # Step 0: Enforce zero-mean
        acceleration = acceleration - np.mean(acceleration)

        # Step 1: Linear detrend
        detrended_acc = self._linear_detrend(acceleration)

        # Step 2: Apply filter
        filtered_acc = self._apply_filter(detrended_acc, cutoff_freq)

        # Step 3: Integrate to velocity (do not re-filter)
        velocity = self._integrate_acceleration_to_velocity(filtered_acc)

        # Step 4: Integrate to position
        position = self._integrate_velocity_to_position(velocity)
        
        # Calculate scores
        scores = self._calculate_scores(position, velocity, filtered_acc)
        
        return ProcessingResult(
            acceleration=filtered_acc,
            velocity=velocity,
            position=position,
            **scores,
            parameters={'strategy': 'B', 'cutoff_freq': cutoff_freq}
        )

    def strategy_d_legacy_b(self, acceleration: np.ndarray, cutoff_freq: float) -> ProcessingResult:
        """Strategy D: Legacy Strategy B (detrend + HPF on acc, HPF on vel, then integrate).

        This reproduces the previous B implementation for comparison purposes.
        Note: This approach can distort the physical a-v-x relationship.
        """
        # Step 0: Enforce zero-mean to mirror current preconditioning
        acceleration = acceleration - np.mean(acceleration)

        # Step 1: Linear detrend
        detrended_acc = self._linear_detrend(acceleration)

        # Step 2: High-pass filter acceleration
        filtered_acc = self._apply_filter(detrended_acc, cutoff_freq)

        # Step 3: Integrate to velocity
        velocity = self._integrate_acceleration_to_velocity(filtered_acc)

        # Step 4: High-pass filter velocity (legacy behavior)
        filtered_vel = self._apply_filter(velocity, cutoff_freq)

        # Step 5: Integrate filtered velocity to position
        position = self._integrate_velocity_to_position(filtered_vel)

        # Calculate scores (use filtered_vel consistent with legacy path)
        scores = self._calculate_scores(position, filtered_vel, filtered_acc)

        return ProcessingResult(
            acceleration=filtered_acc,
            velocity=filtered_vel,
            position=position,
            **scores,
            parameters={'strategy': 'D', 'cutoff_freq': cutoff_freq}
        )
    
    def strategy_c_baseline_correction(self, acceleration: np.ndarray, cutoff_freq: float) -> ProcessingResult:
        """Strategy C: Advanced baseline correction (velocity-fit differentiation)."""
        # Step 0: Enforce zero-mean
        acceleration = acceleration - np.mean(acceleration)

        # Step 1: Apply baseline correction
        corrected_acc = self._velocity_fit_differentiation(acceleration)

        # Step 2: Apply filter
        filtered_acc = self._apply_filter(corrected_acc, cutoff_freq)

        # Step 3: Integrate to velocity (do not re-filter)
        velocity = self._integrate_acceleration_to_velocity(filtered_acc)

        # Step 4: Integrate to position
        position = self._integrate_velocity_to_position(velocity)
        
        # Calculate scores
        scores = self._calculate_scores(position, velocity, filtered_acc)
        
        return ProcessingResult(
            acceleration=filtered_acc,
            velocity=velocity,
            position=position,
            **scores,
            parameters={'strategy': 'C', 'cutoff_freq': cutoff_freq}
        )
    
    def _calculate_scores(self, position: np.ndarray, velocity: np.ndarray, acceleration: np.ndarray) -> Dict:
        """Calculate evaluation scores for processing quality."""
        # Drift score: measure of position drift
        drift_score = np.abs(position[-1] - position[0]) / (np.max(np.abs(position)) + 1e-10)
        
        # Stability score: measure of signal stability
        position_std = np.std(position)
        velocity_std = np.std(velocity)
        stability_score = 1.0 / (1.0 + position_std + velocity_std)
        
        # Physical score: check if values are in expected ranges
        acc_peak = np.max(np.abs(acceleration))
        vel_peak = np.max(np.abs(velocity))
        pos_peak = np.max(np.abs(position))
        
        acc_in_range = EXPECTED_RANGES['acceleration'][0] <= acc_peak <= EXPECTED_RANGES['acceleration'][1]
        vel_in_range = EXPECTED_RANGES['velocity'][0] <= vel_peak <= EXPECTED_RANGES['velocity'][1]
        pos_in_range = EXPECTED_RANGES['position'][0] <= pos_peak <= EXPECTED_RANGES['position'][1]
        
        physical_score = (acc_in_range + vel_in_range + pos_in_range) / 3.0
        
        # Total score (lower is better)
        total_score = drift_score + (1.0 - stability_score) + (1.0 - physical_score)
        
        return {
            'drift_score': drift_score,
            'stability_score': stability_score,
            'physical_score': physical_score,
            'total_score': total_score
        }
    
    def _evaluate_strategy_on_channel(self, strategy_func: Callable, channel: str, 
                                     cutoff_range: np.ndarray, all_data: List[Dict]) -> Tuple[float, Dict]:
        """Evaluate a strategy on a specific channel across all files with parameter optimization."""
        best_score = float('inf')
        best_params = None
        
        # Test each cutoff frequency
        for cutoff in cutoff_range:
            total_score = 0.0
            valid_files = 0
            
            # Evaluate on all files
            for file_data in all_data:
                try:
                    acceleration = file_data[channel]
                    result = strategy_func(acceleration, cutoff)
                    total_score += result.total_score
                    valid_files += 1
                except Exception as e:
                    # Skip files that fail
                    continue
            
            # Calculate average score across all files
            if valid_files > 0:
                avg_score = total_score / valid_files
                if avg_score < best_score:
                    best_score = avg_score
                    best_params = {'strategy': result.parameters['strategy'], 
                                 'cutoff_freq': cutoff, 
                                 'total_score': avg_score,
                                 'files_evaluated': valid_files}
        
        return best_score, best_params
    
    def _evaluate_strategy_on_channels(self, strategy_key: str, strategy_func: Callable,
                                       channels: List[str], cutoff_range: np.ndarray,
                                       all_data: List[Dict]) -> Tuple[float, Dict]:
        """Evaluate a strategy on a set of channels across all files with parameter optimization.

        Returns the best average score and the corresponding parameters.
        """
        best_score = float('inf')
        best_params: Optional[Dict] = None

        for cutoff in cutoff_range:
            total_score = 0.0
            valid_count = 0

            for file_data in all_data:
                for ch in channels:
                    try:
                        acc = file_data[ch]
                        # Limit samples for evaluation speed/memory
                        if self.evaluation_max_samples and len(acc) > self.evaluation_max_samples:
                            acc = acc[: self.evaluation_max_samples]
                        result = strategy_func(acc, float(cutoff))
                        total_score += result.total_score
                        valid_count += 1
                    except Exception:
                        continue

            if valid_count > 0:
                avg_score = total_score / valid_count
                if avg_score < best_score:
                    best_score = avg_score
                    best_params = {
                        'strategy': strategy_key,
                        'cutoff_freq': float(cutoff),
                        'avg_score': float(avg_score),
                        'files_evaluated': len(all_data),
                        'channels_evaluated': channels,
                    }

        return best_score, (best_params or {})

    def phase_2_comparative_evaluation(self) -> Dict:
        """Phase 2: Evaluate strategies A/B/C on healthy data and select best per group.

        Groups:
          - clean_channel: all channels except 'acc_underhang_rad'
          - pathological_channel: only 'acc_underhang_rad'
        """
        if self.forced_strategy is not None:
            print(f"Phase 2: Skipped (forced Strategy {self.forced_strategy})")
            return {
                self.forced_strategy: {
                    'clean_channel': {'params': {'strategy': self.forced_strategy, 'cutoff_freq': self.default_cutoff_hz}},
                    'pathological_channel': {'params': {'strategy': self.forced_strategy, 'cutoff_freq': self.default_cutoff_hz}},
                    'combined_score': None,
                }
            }
        # Decide evaluation scope
        if self.evaluation_scope == 'all':
            print("Phase 2: Comparative evaluation on ALL categories (not only normal)")
            all_data = self._load_all_categories_data()
        else:
            print("Phase 2: Comparative evaluation on healthy (normal) data")
            all_data = self._load_all_normal_data()
        if not all_data:
            print("  No normal data found; falling back to Strategy B with default cutoff.")
            return {
                'B': {
                    'clean_channel': {'params': {'strategy': 'B', 'cutoff_freq': self.default_cutoff_hz}},
                    'pathological_channel': {'params': {'strategy': 'B', 'cutoff_freq': self.default_cutoff_hz}},
                    'combined_score': None,
                }
            }

        clean_channels = [
            'acc_underhang_tan', 'acc_overhang_rad', 'acc_overhang_tan', 'acc_overhang_ax', 'acc_underhang_ax'
        ]
        pathological_channels = ['acc_underhang_rad']

        strategies = {
            'A': self.strategy_a_filtering_only,
            'B': self.strategy_b_detrend_filtering,
            'C': self.strategy_c_baseline_correction,
            'D': self.strategy_d_legacy_b,
        }

        results: Dict[str, Dict] = {}
        best_clean: Tuple[str, Dict] = ('B', {'cutoff_freq': self.default_cutoff_hz, 'avg_score': float('inf')})
        best_patho: Tuple[str, Dict] = ('B', {'cutoff_freq': self.default_cutoff_hz, 'avg_score': float('inf')})

        print(f"  Evaluating cutoffs (Hz): {self.cutoff_grid_hz.tolist()}")
        for key, fn in strategies.items():
            clean_score, clean_params = self._evaluate_strategy_on_channels(key, fn, clean_channels, self.cutoff_grid_hz, all_data)
            patho_score, patho_params = self._evaluate_strategy_on_channels(key, fn, pathological_channels, self.cutoff_grid_hz, all_data)
            results[key] = {
                'clean_channel': {'params': clean_params},
                'pathological_channel': {'params': patho_params},
                'combined_score': float(clean_score + patho_score),
            }
            print(f"    Strategy {key}: clean avg={clean_score:.4f}, patho avg={patho_score:.4f}")

            if clean_score < best_clean[1].get('avg_score', float('inf')):
                best_clean = (key, clean_params)
            if patho_score < best_patho[1].get('avg_score', float('inf')):
                best_patho = (key, patho_params)

        # Store optimal selections
        self.optimal_parameters = {
            'clean_channel': {'params': best_clean[1]},
            'pathological_channel': {'params': best_patho[1]},
        }
        self.optimal_strategy = best_clean[0] if best_clean[0] == best_patho[0] else 'mixed'

        print("  Selected:")
        print(f"    clean_channel -> Strategy {best_clean[0]} @ cutoff={best_clean[1]['cutoff_freq']} Hz (avg={best_clean[1]['avg_score']:.4f})")
        print(f"    pathological_channel -> Strategy {best_patho[0]} @ cutoff={best_patho[1]['cutoff_freq']} Hz (avg={best_patho[1]['avg_score']:.4f})")

        return results
    
    def _extract_omega_from_filename(self, file_path: str) -> float:
        """Extract rotation speed from CSV filename as Hz and convert to rad/s.

        Example: '.../12.288.csv' -> 12.288 Hz -> 12.288 * 2π rad/s
        """
        try:
            hz_value = float(Path(file_path).stem)
        except Exception:
            # Fallback to 0.0 if filename parsing fails
            hz_value = 0.0
        return hz_value * 2 * np.pi
    
    def _process_single_file_with_optimal_strategy(self, file_path: str) -> Dict[str, np.ndarray]:
        """Process a single file using the optimal strategy."""
        # Load raw data
        df_raw = pd.read_csv(file_path, header=None, names=COLUMN_NAMES)
        
        # Convert voltage to acceleration for all channels
        processed_data = {
            'time': np.arange(len(df_raw)) * dt,
            'omega': self._extract_omega_from_filename(file_path)
        }
        
        # Get optimal parameters (selected in Phase 2) or forced
        if self.forced_strategy is not None:
            clean_params = {'strategy': self.forced_strategy, 'cutoff_freq': self.default_cutoff_hz}
            patho_params = {'strategy': self.forced_strategy, 'cutoff_freq': self.default_cutoff_hz}
        else:
            clean_params = self.optimal_parameters.get('clean_channel', {}).get('params', {'strategy': 'B', 'cutoff_freq': self.default_cutoff_hz})
            patho_params = self.optimal_parameters.get('pathological_channel', {}).get('params', {'strategy': 'B', 'cutoff_freq': self.default_cutoff_hz})
        
        # Process each channel
        for col in ['acc_underhang_ax', 'acc_underhang_rad', 'acc_underhang_tan',
                   'acc_overhang_ax', 'acc_overhang_rad', 'acc_overhang_tan']:
            # Optionally interpret data directly as acceleration
            if self.assume_csv_is_acceleration:
                acceleration = df_raw[col].values
            else:
                voltage_signal = df_raw[col].values
                acceleration = self._convert_voltage_to_acceleration(voltage_signal)
            
            # Choose parameters based on channel quality
            if 'underhang_rad' in col:  # Pathological channel
                params = patho_params
            else:  # Clean channels
                params = clean_params
            
            cutoff_freq = params['cutoff_freq']
            
            # Apply selected strategy
            strat_key = params.get('strategy', 'B')
            if strat_key == 'A':
                result = self.strategy_a_filtering_only(acceleration, cutoff_freq)
            elif strat_key == 'C':
                result = self.strategy_c_baseline_correction(acceleration, cutoff_freq)
            elif strat_key == 'D':
                result = self.strategy_d_legacy_b(acceleration, cutoff_freq)
            else:
                result = self.strategy_b_detrend_filtering(acceleration, cutoff_freq)
            
            # Extract bearing and direction
            parts = col.split('_')
            bearing = parts[1]  # underhang or overhang
            direction = parts[2]  # rad or tan
            
            # Store results
            processed_data[f'acc_{bearing}_{direction}'] = result.acceleration
            processed_data[f'vel_{bearing}_{direction}'] = result.velocity
            processed_data[f'pos_{bearing}_{direction}'] = result.position
        
        return processed_data
    
    def _process_dir_to_results(self, dir_path: Path) -> List[Dict[str, np.ndarray]]:
        """Process all CSV files in a directory to processed results dicts."""
        csv_files = sorted([f for f in Path(dir_path).glob("*.csv")])
        print(f"  Found {len(csv_files)} CSV files in {dir_path}")
        all_results: List[Dict[str, np.ndarray]] = []
        for i, csv_file in enumerate(csv_files):
            print(f"    Processing file {i+1}/{len(csv_files)}: {csv_file.name}")
            try:
                results = self._process_single_file_with_optimal_strategy(str(csv_file))
                all_results.append(results)
            except Exception as e:
                print(f"    Error processing {csv_file.name}: {e}")
                continue
        return all_results

    def _save_results_as_tensors(self, all_results: List[Dict[str, np.ndarray]], output_dir: str, out_name: str) -> None:
        if not all_results:
            print(f"  No results to save for {out_name}")
            return
        X_data, Y_data = self._convert_to_pinn_tensors(all_results)
        output_path = Path(output_dir) / "v3"
        output_path.mkdir(exist_ok=True)
        x_path = output_path / f"X_{out_name}_v3.pth"
        y_path = output_path / f"Y_{out_name}_v3.pth"
        torch.save(X_data, x_path)
        torch.save(Y_data, y_path)
        print(f"  Saved: {x_path}, {y_path}  | shapes: X={X_data.shape}, Y={Y_data.shape}")

    def phase_3_generate_for_category(self, category_name: str, category_dir: str, output_dir: str = "Data") -> None:
        """Generate v3 tensors for a specific category (normal or faulty)."""
        if self.optimal_strategy is None:
            raise ValueError("Must run Phase 2 evaluation first!")
        dir_path = Path(category_dir)
        if not dir_path.exists():
            print(f"  Skip {category_name}: directory not found -> {dir_path}")
            return
        print(f"Processing category: {category_name}  ({dir_path})")
        all_results = self._process_dir_to_results(dir_path)
        self._save_results_as_tensors(all_results, output_dir, category_name)

    def phase_3_full_dataset_generation(self, data_dir: str = "Data", output_dir: str = "Data") -> None:
        """Phase 3: Generate full dataset for PINN training (all categories)."""
        print("\nPhase 3: Full Dataset Generation for PINN (all categories)")
        print("=" * 50)

        if self.optimal_strategy is None:
            print("Error: Must run Phase 2 evaluation first!")
            return

        # Iterate over legacy category paths to ensure parity with v2
        categories = list(LEGACY_DATA_PATHS.items())
        # Prioritize normal first
        categories.sort(key=lambda kv: (kv[0] != 'normal', kv[0]))
        for name, path in categories:
            self.phase_3_generate_for_category(name, path, output_dir)
    
    def _convert_to_pinn_tensors(self, all_results: List[Dict[str, np.ndarray]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert processed results to PINN-compatible tensors."""
        X_features = []
        Y_features = []

        if not all_results:
            return torch.empty(0), torch.empty(0)

        # Ensure consistent length across files by truncating to the shortest
        min_len = min(len(res['time']) for res in all_results)

        for results in all_results:
            # Y tensor (Targets): 4 accelerations
            y_sample = np.column_stack([
                results['acc_underhang_rad'][:min_len],
                results['acc_underhang_tan'][:min_len],
                results['acc_overhang_rad'][:min_len],
                results['acc_overhang_tan'][:min_len]
            ])
            Y_features.append(y_sample)

            # X tensor (Inputs): 10 features (matching training script expectations)
            # Velocities (4 features)
            velocities = np.column_stack([
                results['vel_underhang_rad'][:min_len],
                results['vel_underhang_tan'][:min_len],
                results['vel_overhang_rad'][:min_len],
                results['vel_overhang_tan'][:min_len]
            ])

            # Positions (4 features)
            positions = np.column_stack([
                results['pos_underhang_rad'][:min_len],
                results['pos_underhang_tan'][:min_len],
                results['pos_overhang_rad'][:min_len],
                results['pos_overhang_tan'][:min_len]
            ])

            # Omega (1 feature) - constant from filename
            omega_array = np.full((min_len, 1), results['omega'])

            # Time (1 feature) - normalized time array
            time_array = results['time'][:min_len].reshape(-1, 1)

            # Combine all features (velocities + positions + omega + time = 10 features)
            x_sample = np.column_stack([velocities, positions, omega_array, time_array])
            X_features.append(x_sample)

        # Convert to tensors
        X_tensor = torch.tensor(np.stack(X_features, axis=0), dtype=torch.float32)
        Y_tensor = torch.tensor(np.stack(Y_features, axis=0), dtype=torch.float32)

        return X_tensor, Y_tensor
    
    def run_complete_pipeline(self, data_dir: str = "Data", output_dir: str = "Data") -> None:
        """Run the complete 3-phase pipeline."""
        print("MaFaulDa Complete Processing Pipeline v3")
        print("=" * 50)
        
        # Phase 1: Data preparation (comprehensive loading)
        print("Phase 1: Comprehensive Data Preparation")
        print("  Loading all normal files for strategy evaluation...")
        
        # Phase 2: Strategy evaluation across all files
        evaluation_results = self.phase_2_comparative_evaluation()
        
        # Phase 3: Full dataset generation
        self.phase_3_full_dataset_generation(data_dir, output_dir)
        
        print("\n Complete pipeline finished!")
        if self.forced_strategy:
            print(f"Strategy used (forced): {self.forced_strategy}")
        else:
            print(f"Optimal strategy: {self.optimal_strategy}")
        print("Files created:")
        print("  - X_<category>_v3.pth (10 features: velocities, positions, omega, time)")
        print("  - Y_<category>_v3.pth (4 features: accelerations)")

def main():
    """Run the complete processing pipeline."""
    processor = MaFaulDaProcessorV3(
        forced_strategy='D',
        assume_csv_is_acceleration=True,
        evaluation_scope='all',  # evaluate using all categories
    )
    processor.run_complete_pipeline()

if __name__ == "__main__":
    main() 