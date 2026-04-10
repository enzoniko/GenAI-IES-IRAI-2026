import pandas as pd
import numpy as np
import torch
import sys
from pathlib import Path
from scipy.fft import rfft, irfft, rfftfreq

import src.constants as c

class CleanMaFaulDaProcessor:
    """
    A clean and straightforward processor for the MaFaulDa dataset.
    Focused solely on extracting the necessary features for the PINN, integrating acceleration
    to obtain velocity and position, while mitigating integration drift (originating from Strategy D).
    """
    def __init__(self, raw_data_dir: str, processed_data_dir: str, cutoff_hz: float = 2.0):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.cutoff_hz = cutoff_hz
        self.fs = 50000
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

    def _extract_omega(self, file_path: Path) -> float:
        try:
            hz_value = float(file_path.stem)
        except ValueError:
            hz_value = 0.0
        return hz_value * 2 * np.pi

    # Now using FFT
    def process_signal(self, acc: np.ndarray) -> tuple:
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

    def run(self, category: str = 'normal', rotation: int = None,
            training_windows: int = None, test_windows: int = None):
        """
        Process all CSV files for a given fault category and save the resulting tensors.

        Parameters
        ----------
        category : str
            Fault label to process ('normal', 'imbalance', 'overhang').
        rotation : int | None
            If given, restrict processing to the one CSV file whose filename
            (Hz value) matches this integer.
        training_windows : int | None
            How many rotation windows to keep in the training set.
            If None, defaults to 15% of all available windows (rounded).
        test_windows : int | None
            How many rotation windows to hold out as a temporally-disjoint test set.
            These come from the CHRONOLOGICAL END of the recording so they are
            guaranteed not to appear as neighbours of any training window.
            If None, defaults to 3% of all available windows (rounded), minimum 1.
        """
        target_dir = self.raw_data_dir / category
        print(f"  Loading CSVs from directory: {target_dir}")
        
        csv_files = sorted(target_dir.glob("*.csv"))
        if not csv_files:
            print(f"  [Error] No CSV files found in {target_dir}")
            return
        
        # ── Rotation Frequency Filtering ─────────────────────────────────────
        if rotation is not None:
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
            
            X = np.column_stack([velocities, positions, omega_array, time_col])
            
            X_list.append(X)
            Y_list.append(Y)

        # ── Windowing ────────────────────────────────────────────────────────
        # Extract the specific rotation frequency from the selected file
        hz_value = float(csv_files[0].stem)
        
        # Calculate the integer window size for 1 full rotation
        window_size = int(np.round(self.fs / hz_value))
        print(f"  Calculated Window Size: {window_size} points per rotation at {hz_value} Hz")

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
            
            X_3D_list.append(X_chunked)
            Y_3D_list.append(Y_chunked)

        # Concatenate all files along the window dimension
        X_all = np.concatenate(X_3D_list, axis=0)  # (N_total, window_size, 10)
        Y_all = np.concatenate(Y_3D_list, axis=0)  # (N_total, window_size, 4)
        N_total = X_all.shape[0]

        # ── Training / Test Split ─────────────────────────────────────────────
        # The test windows come from the CHRONOLOGICAL END of the recording.
        # This guarantees temporal disjointness — no test window will be a
        # direct neighbour of any training window.
        if test_windows is None:
            # Default: 3% of total windows, but at least 1
            test_windows = max(1, int(round(N_total * 0.03)))
        
        if test_windows >= N_total:
            raise ValueError(
                f"test_windows ({test_windows}) >= total windows ({N_total}). "
                "Reduce test_windows or process more data."
            )

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
            # Default: 15% of total, but cap at the available training pool
            default_train = min(int(round(N_total * 0.15)), N_train_pool)
            X_train_tensor = torch.tensor(X_all[:default_train], dtype=torch.float32)
            Y_train_tensor = torch.tensor(Y_all[:default_train], dtype=torch.float32)

        # Test set: last test_windows windows (always from the chronological end)
        X_test_tensor = torch.tensor(X_all[N_total - test_windows:], dtype=torch.float32)
        Y_test_tensor = torch.tensor(Y_all[N_total - test_windows:], dtype=torch.float32)

        # ── Save Tensors ─────────────────────────────────────────────────────
        out_path = self.processed_data_dir / c.DATASET_CURRENT_VERSION
        out_path.mkdir(parents=True, exist_ok=True)

        # Training set tensors (used by PINN + TS-JEPA + Decoders)
        torch.save(X_train_tensor, out_path / f"X_{category}_{c.DATASET_CURRENT_VERSION}_trainingset.pth")
        torch.save(Y_train_tensor, out_path / f"Y_{category}_{c.DATASET_CURRENT_VERSION}_trainingset.pth")

        # Test set tensors (held out for final evaluation — never touched during training)
        torch.save(X_test_tensor, out_path / f"X_{category}_{c.DATASET_CURRENT_VERSION}_testset.pth")
        torch.save(Y_test_tensor, out_path / f"Y_{category}_{c.DATASET_CURRENT_VERSION}_testset.pth")

        print(f"  Finished processing '{category}':")
        print(f"    Training set — X: {X_train_tensor.shape} | Y: {Y_train_tensor.shape}")
        print(f"    Test set     — X: {X_test_tensor.shape} | Y: {Y_test_tensor.shape}")
        print(f"    Saved to: {out_path}")


