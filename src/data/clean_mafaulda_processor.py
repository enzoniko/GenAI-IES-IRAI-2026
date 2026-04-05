import pandas as pd
import numpy as np
import torch
from pathlib import Path
from scipy.signal import butter, filtfilt
from scipy.integrate import cumulative_trapezoid

class CleanMaFaulDaProcessor:
    """
    A clean and straightforward processor for the MaFaulDa dataset.
    Focused solely on extracting the necessary features for the PINN, integrating acceleration
    to obtain velocity and position, while mitigating integration drift (originating from Strategy D).
    """
    def __init__(self, data_dir: str, output_dir: str, cutoff_hz: float = 2.0):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
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
        
    def _butter_highpass_filter(self, data: np.ndarray) -> np.ndarray:
        b, a = butter(4, self.cutoff_hz, btype='high', fs=self.fs)
        return filtfilt(b, a, data)
        
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

    def process_signal(self, acc: np.ndarray) -> tuple:
        # 1. Zero-mean and linear trend removal (detrend)
        acc = acc - np.mean(acc)
        acc = self._linear_detrend(acc)
        
        # 2. High-pass filter on acceleration (drift mitigation at the raw level)
        acc_filt = self._butter_highpass_filter(acc)
        
        # 3. Integration to obtain velocity
        vel = cumulative_trapezoid(acc_filt, dx=self.dt, initial=0)
        
        # 4. High-pass filter on velocity (corrects accumulated drift from the 1st integration)
        vel_filt = self._butter_highpass_filter(vel)
        
        # 5. Integration to obtain position
        pos = cumulative_trapezoid(vel_filt, dx=self.dt, initial=0)
        
        return acc_filt, vel_filt, pos

    def run(self, category: str = 'normal'):
        target_dir = self.data_dir / category
        print(f"  Loading CSVs from directory: {target_dir}")
        
        csv_files = sorted(target_dir.glob("*.csv"))
        if not csv_files:
            print(f"  [Error] No CSV files found in {target_dir}")
            return

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
                # We treat the input directly as acceleration
                acc_raw = df[col].values
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

        # Standardizing the size by truncating all vectors to the shortest sequence found
        min_len = min(x.shape[0] for x in X_list)
        X_tensor = torch.tensor(np.stack([x[:min_len] for x in X_list]), dtype=torch.float32)
        Y_tensor = torch.tensor(np.stack([y[:min_len] for y in Y_list]), dtype=torch.float32)

        out_path = self.output_dir / "v3"
        out_path.mkdir(parents=True, exist_ok=True)
        
        x_save_path = out_path / f"X_{category}_v3.pth"
        y_save_path = out_path / f"Y_{category}_v3.pth"
        
        torch.save(X_tensor, x_save_path)
        torch.save(Y_tensor, y_save_path)
        print(f"  Finished! Tensors saved to {out_path}")
        print(f"  X shape: {X_tensor.shape} | Y shape: {Y_tensor.shape}")
