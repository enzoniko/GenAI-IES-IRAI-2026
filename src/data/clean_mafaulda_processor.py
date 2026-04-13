import pandas as pd
import numpy as np
import torch
import sys
import os
import json
from pathlib import Path
from scipy.fft import rfft, irfft, rfftfreq
from scipy.signal import butter, filtfilt
from scipy.integrate import cumulative_trapezoid

import src.configs as cfg

class CleanMaFaulDaProcessor:
    """
    A clean and straightforward processor for the MaFaulDa dataset.
    Focused solely on extracting the necessary features for the PINN, integrating acceleration
    to obtain velocity and position, while mitigating integration drift (originating from Strategy D).
    """
    def __init__(self, raw_data_dir: str, processed_data_dir: str, 
                 cutoff_hz: float = None, strategy: str = None):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.cutoff_hz = cutoff_hz or cfg.CUTOFF_HZ
        self.strategy = strategy or cfg.SIGNAL_PROCESSING_STRATEGY
        self.fs = cfg.SAMPLING_RATE
        self.dt = 1.0 / self.fs
        
        self.target_columns = [
            'acc_underhang_rad', 'acc_underhang_tan',
            'acc_overhang_rad', 'acc_overhang_tan'
        ]
        self.column_names = [
            'tachometer',
            'acc_underhang_ax', 'acc_underhang_rad', 'acc_underhang_tan',
            'acc_overhang_ax', 'acc_overhang_rad', 'acc_overhang_tan',
            'microphone'
        ]
        
    def _linear_detrend(self, data: np.ndarray) -> np.ndarray:
        x = np.arange(len(data))
        coeffs = np.polyfit(x, data, 1)
        return data - np.polyval(coeffs, x)

    def _apply_filter(self, signal: np.ndarray, cutoff: float, order: int = 4) -> np.ndarray:
        """Apply zero-phase Butterworth high-pass filter (SciPy filtfilt)."""
        b, a = butter(order, cutoff, btype='high', fs=self.fs)
        return filtfilt(b, a, signal)

    def _extract_omega(self, file_path: Path) -> float:
        try:
            hz_value = float(file_path.stem)
        except ValueError:
            hz_value = 0.0
        return hz_value * 2 * np.pi

    # Now using FFT
    def process_signal(self, acc: np.ndarray) -> tuple:
        """Dispatcher for signal processing strategies."""
        if self.strategy == 'previous_strategy':
            return self._process_signal_previous(acc)
        else:
            return self._process_signal_fft(acc)

    def _process_signal_fft(self, acc: np.ndarray) -> tuple:
        """Original FFT-based integration (Pure Frequency Domain)."""
        # 1. Zero-mean and linear trend removal (detrend)
        acc = acc - np.mean(acc)
        acc = self._linear_detrend(acc)
        
        # 2. Convert to Frequency Domain via Real FFT
        n = len(acc)
        acc_fft = rfft(acc)
        freqs = rfftfreq(n, d=self.dt)
        
        # Compute angular frequency (omega)
        omega = 2 * np.pi * freqs
        omega[0] = 1.0  # Avoid division by zero at DC (0 Hz)
        
        # 3. Frequency domain integration
        # Velocity = Acceleration / (j * omega)
        vel_fft = acc_fft / (1j * omega)
        # Position = Acceleration / (-omega^2)
        pos_fft = acc_fft / (-(omega ** 2))
        
        # Strictly eliminate drift by setting the DC (0 Hz) and sub-cutoff bins to 0.0
        low_freq_mask = freqs < self.cutoff_hz
        acc_fft[low_freq_mask] = 0.0
        vel_fft[low_freq_mask] = 0.0
        pos_fft[low_freq_mask] = 0.0
        
        # 4. Convert back to Time Domain via Inverse Real FFT
        acc_filt = irfft(acc_fft, n=n)
        vel_filt = irfft(vel_fft, n=n)
        pos = irfft(pos_fft, n=n)
        
        return acc_filt, vel_filt, pos

    def _process_signal_previous(self, acc_np: np.ndarray) -> tuple:
        """
        Differentiable reproduction of Strategy D (Previous Strategy).
        
        --- DIFFERENTIABILITY EXPLANATION ---
        This method is now completely differentiable because:
        1. It replaces NumPy operations with PyTorch counterparts (torch.cumsum, linalg.lstsq).
        2. It implements the IIR filter (Butterworth) using a forward-backward pass 
           (filtfilt) in PyTorch, allowing Autograd to track the gradients through 
           the recursive filtering operation.
        3. This enables the entire preprocessing pipeline to be part of a 
           larger differentiable chain (e.g., if we want to learn filter 
           cutoffs or use this logic inside a PINN loss).

        --- STRATEGY D INTENTION REPRODUCTION ---
        Strategy D's core innovation was the 'Double-Filter' approach:
        1. ACCEL FILTER: First high-pass to remove low-frequency gravity bias.
        2. VELOCITY FILTER: Second high-pass after integration. This kills 
           the integration constants and the 'random walk' drift that naturally 
           emerges when integrating even slightly noisy acceleration.
        -------------------------------------
        """
        # Move to Torch tensor to enable differentiability
        # USE DOUBLE PRECISION (float64) for internal processing
        # 4th order IIR filters with extremely low cutoffs (2Hz @ 50kHz) 
        # are numerically unstable in 32-bit float precision.
        acc = torch.from_numpy(acc_np).double()
        
        # STEP 1: CONDITIONING (Zero-mean + Detrend)
        # Why: Remove stationary offsets and linear thermal drift.
        acc = acc - acc.mean()
        acc = self._torch_detrend(acc)
        
        # STEP 2: ACCELERATION FILTERING
        # Why: Removes raw sensor tilt and low-frequency mechanical vibration 
        # below the cutoff.
        b_coef, a_coef = self._get_filter_coefs()
        acc_filt = self._torch_filtfilt(acc, b_coef, a_coef)
        
        # STEP 3: INTEGRATION -> VELOCITY
        # Why: Convert Accel to Velocity using trapezoidal rule.
        vel = self._torch_integrate(acc_filt)
        
        # STEP 4: VELOCITY FILTERING (The critical Strategy D 'Second Filter')
        # why: Integration generates a cumulative error curve. This high-pass 
        # filter 're-centers' the velocity signal around zero, preventing 
        # the position from exploding in step 5.
        vel_filt = self._torch_filtfilt(vel, b_coef, a_coef)
        
        # STEP 5: INTEGRATION -> POSITION
        # why: Final displacement extraction.
        pos = self._torch_integrate(vel_filt)
        
        # Final cast back to float32 for model training/storage compatibility
        return (acc_filt.float().detach().numpy(), 
                vel_filt.float().detach().numpy(), 
                pos.float().detach().numpy())

    def _torch_detrend(self, y: torch.Tensor) -> torch.Tensor:
        """Differentiable linear detrender using Least Squares."""
        n = y.shape[-1]
        # x coordinates normalized to [0,1]
        x = torch.linspace(0, 1, n, device=y.device, dtype=y.dtype)
        # Linear model matrix: [x, 1]
        A = torch.stack([x, torch.ones_like(x)], dim=-1) 
        # Least squares solve: (batch, n) inputs handled via unsqueeze
        res = torch.linalg.lstsq(A, y.unsqueeze(-1))
        slope, intercept = res.solution[..., 0, :], res.solution[..., 1, :]
        trend = (slope * x + intercept).squeeze(-1)
        return y - trend

    def _torch_integrate(self, y: torch.Tensor) -> torch.Tensor:
        """Differentiable trapezoidal integration."""
        # Standard trapezoidal rule: area[i] = (y[i] + y[i-1])/2 * dt
        y_mid = 0.5 * (y[..., 1:] + y[..., :-1])
        integral = torch.cumsum(y_mid * self.dt, dim=-1)
        # Pad leading zero to match initial sample time t=0
        z = torch.zeros((*y.shape[:-1], 1), device=y.device, dtype=y.dtype)
        return torch.cat([z, integral], dim=-1)

    def _torch_filtfilt(self, x: torch.Tensor, b: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Differentiable zero-phase IIR filtering (forward-backward)."""
        import torchaudio.functional as F
        # Forward pass through IIR recursive filter
        y = F.lfilter(x, a, b, clamp=False)
        # Flip, filter again, and flip back to ensure zero phase delay 
        # (matching scipy.signal.filtfilt exactly)
        y = torch.flip(y, dims=[-1])
        y = F.lfilter(y, a, b, clamp=False)
        y = torch.flip(y, dims=[-1])
        return y

    def _get_filter_coefs(self):
        """Get Butterworth coefficients as Torch tensors in float64."""
        from scipy.signal import butter
        b, a = butter(4, self.cutoff_hz, btype='high', fs=self.fs)
        return torch.from_numpy(b).double(), torch.from_numpy(a).double()

    def find_best_matches(self, target_hz: float) -> dict:
        """For each leaf directory, find the CSV with frequency closest to target_hz."""
        categories = self.discover_categories()
        best_matches = {}
        
        for cat_path in categories:
            csv_files = list(cat_path.glob("*.csv"))
            if not csv_files:
                continue
            # Find closest freq
            best_csv = min(csv_files, key=lambda f: abs(float(f.stem) - target_hz))
            best_matches[cat_path] = best_csv
            
        return best_matches

    def calculate_global_min_window(self, matches: dict) -> int:
        """Determine the minimum window size across all selected best-match files."""
        min_size = float('inf')
        for csv_path in matches.values():
            hz_value = float(csv_path.stem)
            window_size = int(np.round(self.fs / hz_value))
            if window_size < min_size:
                min_size = window_size
        return int(min_size)

    def discover_categories(self) -> list:
        """Find all leaf directories containing .csv files."""
        categories = []
        for path in self.raw_data_dir.rglob("*"):
            if path.is_dir() and any(path.glob("*.csv")):
                categories.append(path)
        return sorted(categories)

    def get_category_label(self, category_path: Path) -> str:
        """Construct a standardized label name matching legacy naming conventions."""
        rel_path = category_path.relative_to(self.raw_data_dir)
        parts = list(rel_path.parts)
        
        if parts == ['normal']:
            return 'normal'
            
        # Join with underscore, replace hyphens
        name = "_".join(parts).replace('-', '_')
        
        # Legacy compatibility fixes (inserting '_fault' where missing)
        if 'misalignment' in name and 'fault' not in name:
            # horizontal_misalignment_0.5mm -> horizontal_misalignment_fault_0.5mm
            parts_list = name.split('_')
            name = "_".join(parts_list[:-1]) + "_fault_" + parts_list[-1]
        elif 'imbalance' in name and 'fault' not in name:
            # imbalance_6g -> imbalance_fault_6g
            parts_list = name.split('_')
            name = "_".join(parts_list[:-1]) + "_fault_" + parts_list[-1]
        elif 'outer_race' in name and 'fault' not in name:
            # overhang_outer_race_0g -> overhang_outer_race_fault_0g
            name = name.replace('outer_race', 'outer_race_fault')
            
        return name

    def run(self, target_hz: float = cfg.TARGET_HZ, 
            rotation: int = None, 
            training_windows: int = None, 
            test_windows: int = None):
        """
        Orchestrate Selective Frequency Processing.
        """
        print(f"Starting Selective Frequency Processing (Target: {target_hz} Hz)...")
        
        # 1. Find best matches
        matches = self.find_best_matches(target_hz)
        print(f"  Found {len(matches)} categories with files matching target Hz.")
        
        # 2. Calculate global minimum window size to avoid padding
        global_min_window = self.calculate_global_min_window(matches)
        print(f"  Global standardized window size (SEQ_LENGTH): {global_min_window}")
        
        # Update output directory to use target frequency
        old_processed_dir = self.processed_data_dir
        self.processed_data_dir = old_processed_dir.parent / f"{int(target_hz)}hz"
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Processed tensors will be saved to: {self.processed_data_dir}")

        # 3. Process each match
        for cat_path, csv_path in matches.items():
            label = self.get_category_label(cat_path)
            print(f"Processing category: {label} (File: {csv_path.name})")
            self.process_category(cat_path, label, rotation, training_windows, test_windows, 
                                  specific_csv=csv_path, min_window_size=global_min_window)
        
        # Save metadata info to the processed folder for the training pipeline to discover
        metadata = {
            "target_hz": target_hz,
            "seq_length": int(global_min_window),
            "num_channels_y": 4,
            "num_features_x": 10,
            "label_strategy": "mafaulda_expanded_42"
        }
        meta_path = self.processed_data_dir / "metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"  [Processor] Metadata saved to {meta_path}")
        
        print("\nAll categories processed successfully!")

    def process_category(self, target_dir: Path, label: str, rotation: int = None,
                         training_windows: int = None, test_windows: int = None,
                         specific_csv: Path = None, min_window_size: int = None):
        """Original run logic extracted to process a single directory."""
        if specific_csv:
            csv_files = [specific_csv]
        else:
            csv_files = sorted(target_dir.glob("*.csv"))
            if rotation:
                csv_files = [f for f in csv_files if int(float(f.stem)) == rotation]
        
        if not csv_files:
            print(f"  [Error] No CSV files found in {target_dir}")
            return
        
        # ── Rotation Frequency Filtering ─────────────────────────────────────
        # If we didn't already select a specific CSV via the matcher, apply
        # legacy integer-based filtering.
        if specific_csv is None and rotation is not None:
            target_rotation = int(rotation)
            matched_files = []
            for f in csv_files:
                try:
                    freq = float(f.stem)
                    if int(freq) == target_rotation:
                        matched_files.append((f, freq))
                except ValueError:
                    continue
                    
            if len(matched_files) == 0:
                print(f"  [Error] No CSV file found with integer frequency {target_rotation} in {target_dir}")
                sys.exit(1)
            elif len(matched_files) == 1:
                selected_file = matched_files[0][0]
                print(f"  Found exactly one match for {target_rotation} Hz: {selected_file.name}")
            else:
                matched_files.sort(key=lambda x: abs(x[1] - target_rotation))
                selected_file = matched_files[0][0]
                print(f"  Found multiple matches for {target_rotation} Hz. Selected the closest: {selected_file.name}")
                
            csv_files = [selected_file]

        X_list, Y_list = [], []

        for i, csv_file in enumerate(csv_files):
            if i % 10 == 0:
                print(f"    Processing file {i+1}/{len(csv_files)}...")
            
            df = pd.read_csv(csv_file, header=None, names=self.column_names)
            omega = self._extract_omega(csv_file)
            time_array = np.arange(len(df)) * self.dt
            
            acc_list, vel_list, pos_list = [], [], []
            
            # We only process the 4 channels of interest for the final Dataset
            for col in self.target_columns:
                # Convert raw voltage to actual physical acceleration (m/s^2)
                acc_raw = df[col].values / 0.0102
                acc_filt, vel_filt, pos = self.process_signal(acc_raw)
                
                acc_list.append(acc_filt)
                vel_list.append(vel_filt)
                pos_list.append(pos)
            
            # Y tensor: 4 acceleration features (PINN Target)
            Y = np.column_stack(acc_list)
            
            # X tensor: 4 velocities + 4 positions + omega + time = 10 features (PINN Input)
            velocities = np.column_stack(vel_list)
            positions = np.column_stack(pos_list)
            omega_array = np.full((len(time_array), 1), omega)
            time_col = time_array.reshape(-1, 1)
            
            # Stack all 10 columns: [Vel1, Vel2, Vel3, Vel4, Pos1, Pos2, Pos3, Pos4, Omega, Time]
            X = np.column_stack([velocities, positions, omega_array, time_col])
            
            X_list.append(X)
            Y_list.append(Y)

        # ── Windowing ────────────────────────────────────────────────────────
        # Extract the specific rotation frequency from the selected file
        hz_value = float(csv_files[0].stem)
        
        # Calculate the integer window size for 1 full rotation
        window_size = int(np.round(self.fs / hz_value))
        print(f"  [Processor] Calculated Window Size: {window_size} samples/rotation (at {hz_value:.2f} Hz)")

        X_3D_list, Y_3D_list = [], []

        for X, Y in zip(X_list, Y_list):
            # Find the largest multiple of window_size that fits in this array
            max_valid_length = (X.shape[0] // window_size) * window_size
            
            # Slice off the remainder at the end of the file
            X_sliced = X[:max_valid_length, :]
            Y_sliced = Y[:max_valid_length, :]
            
            # Reshape from 2D into 3D: (Num_Windows, Window_Size, Features)
            X_chunked = X_sliced.reshape(-1, window_size, X.shape[1])
            Y_chunked = Y_sliced.reshape(-1, window_size, Y.shape[1])
            
            # Apply truncation to global min window size if provided
            # This is critical to ensure uniform dimensions across ALL categories
            if min_window_size:
                X_chunked = X_chunked[:, :min_window_size, :]
                Y_chunked = Y_chunked[:, :min_window_size, :]
            
            X_3D_list.append(X_chunked)
            Y_3D_list.append(Y_chunked)

        # Concatenate all files along the window dimension
        X_all = np.concatenate(X_3D_list, axis=0)  # (N_total, min_window_size, 10)
        Y_all = np.concatenate(Y_3D_list, axis=0)  # (N_total, min_window_size, 4)
        N_total = X_all.shape[0]
        print(f"  [Processor] Extracted {N_total} windows (Standardized length: {X_all.shape[1]})")

        # ── Training / Test Split ─────────────────────────────────────────────
        # The test windows come from the CHRONOLOGICAL END of the recording.
        # This guarantees temporal disjointness — no test window will be a
        # direct neighbour of any training window.
        if test_windows is None:
            # Default: 3% of total windows, but at least 1
            test_windows = max(1, int(round(N_total * 0.03)))
        
        # Safety Check: If test_windows is too large for the current file, 
        # fall back to a 20% split to preserve training data integrity.
        if test_windows >= N_total * 0.5:
            print(f"  WARNING: Requested test_windows ({test_windows}) is too large for total data ({N_total}).")
            test_windows = max(1, int(round(N_total * 0.2)))
            print(f"  [Processor] Auto-adjusted test_windows to {test_windows} (20% of total).")
        
        if test_windows >= N_total:
             raise ValueError(f"CRITICAL: test_windows ({test_windows}) >= N_total ({N_total}).")

        # Split indices chronologically
        N_train_pool = N_total - test_windows   # windows available for training + val

        # Cap the training pool if training_windows is explicitly given
        if training_windows is not None:
            if training_windows > N_train_pool:
                print(f"  WARNING: training_windows ({training_windows}) > available "
                      f"pool ({N_train_pool}). Using all available.")
                training_windows = N_train_pool
            # Take the first training_windows windows from the training pool
            X_train_tensor = torch.tensor(X_all[:training_windows], dtype=torch.float32)
            Y_train_tensor = torch.tensor(Y_all[:training_windows], dtype=torch.float32)
        else:
            # Default: 75% of total, but cap at the available training pool
            default_train = min(int(round(N_total * cfg.WINDOW_PCT)), N_train_pool)
            X_train_tensor = torch.tensor(X_all[:default_train], dtype=torch.float32)
            Y_train_tensor = torch.tensor(Y_all[:default_train], dtype=torch.float32)

        # Test set: last test_windows windows (always from the chronological end)
        X_test_tensor = torch.tensor(X_all[N_total - test_windows:], dtype=torch.float32)
        Y_test_tensor = torch.tensor(Y_all[N_total - test_windows:], dtype=torch.float32)

        # Ensure output directory exists
        out_path = self.processed_data_dir
        out_path.mkdir(parents=True, exist_ok=True)

        # ── Save Tensors ─────────────────────────────────────────────────────
        # Save tensors
        torch.save(X_train_tensor, out_path / f"X_{label}_trainingset.pth")
        torch.save(Y_train_tensor, out_path / f"Y_{label}_trainingset.pth")
        
        torch.save(X_test_tensor, out_path / f"X_{label}_testset.pth")
        torch.save(Y_test_tensor, out_path / f"Y_{label}_testset.pth")
        
        print(f"    Final Training Shape: {X_train_tensor.shape} | {Y_train_tensor.shape}")


