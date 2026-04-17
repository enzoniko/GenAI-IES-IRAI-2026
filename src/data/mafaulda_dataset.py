import torch
from torch.utils.data import Dataset, DataLoader, Subset
import os
import glob
import json
import pandas as pd
import src.configs as cfg

class BatchTuple(tuple):
    """
    A custom tuple to pass the dynamic `omega` without breaking the 
    standard dataloader signature (which expects exactly 3 items).
    """
    def __new__(cls, r, c, l, o):
        return super(BatchTuple, cls).__new__(cls, (r, c, l))
    def __init__(self, r, c, l, o):
        self.omega = o

def mafaulda_collate_fn(batch):
    raw = torch.stack([item[0] for item in batch])
    clean = torch.stack([item[1] for item in batch])
    label = torch.stack([item[2] for item in batch])
    omega = torch.tensor([item[3] for item in batch], dtype=torch.float32)
    return BatchTuple(raw, clean, label, omega)

class MaFaulDaDataset(Dataset):
    """
    Data loader for the pre-processed physical MaFaulDa dataset.
    This replaces the raw CSV loading with the cleaned, detrended, 
    and perfectly rotation-windowed Y tensors.
    
    Contract:
    - __getitem__ must return (raw_trace, clean_trace, label, omega).
    - For real data without a "clean" version, raw_trace is returned twice.
    - Output Tensors:
        - trace: (4, window_length) float tensor
        - label: long tensor
    """
    def __init__(self, root_dir, num_samples: int = 1500, seq_length: int = None, num_channels: int = 4):
        # Determine sequence length: 
        # 1. Look for metadata.json in the processed folder (Auto-discovery)
        # 2. Fallback to passed argument
        # 3. Fallback to centralized config
        self.metadata = None
        meta_path = os.path.join(root_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                self.metadata = json.load(f)
                
        self.seq_length = (self.metadata.get('seq_length') if self.metadata else None) or seq_length or cfg.SEQ_LENGTH
        self.num_samples = num_samples
        self.num_channels = num_channels
        
        if self.metadata:
            print(f"  [Dataset] Auto-discovered metadata: {self.metadata['target_hz']} Hz | Sequence: {self.seq_length}")
        
        traces_list = []
        labels_list = []
        omegas_list = []
        
        # MaFaulDa folder structure mapping to the reduced 4-class scheme.
        # Non-target conditions are intentionally excluded.
        label_mapping = {
            'normal': 0,
            'imbalance_fault_6g': 1, 'imbalance_fault_10g': 1, 'imbalance_fault_15g': 1, 'imbalance_fault_20g': 1,
            'imbalance_fault_25g': 1, 'imbalance_fault_30g': 1, 'imbalance_fault_35g': 1,
            'vertical_misalignment_fault_0.51mm': 2, 'vertical_misalignment_fault_0.63mm': 2, 'vertical_misalignment_fault_1.27mm': 2,
            'vertical_misalignment_fault_1.40mm': 2, 'vertical_misalignment_fault_1.78mm': 2, 'vertical_misalignment_fault_1.90mm': 2,
            'overhang_ball_fault_0g': 3, 'overhang_ball_fault_6g': 3, 'overhang_ball_fault_20g': 3, 'overhang_ball_fault_35g': 3,
            'overhang_cage_fault_0g': 3, 'overhang_cage_fault_6g': 3, 'overhang_cage_fault_20g': 3, 'overhang_cage_fault_35g': 3,
            'overhang_outer_race_fault_0g': 3, 'overhang_outer_race_fault_6g': 3, 'overhang_outer_race_fault_20g': 3, 'overhang_outer_race_fault_35g': 3
        }
        
        print(f"Loading pre-processed datasets from {root_dir}...")
        
        # Load normalization metadata for consistent Min-Max scaling
        norm_path = cfg.NORM_METADATA_PATH
        if not os.path.exists(norm_path):
            print(f"  WARNING: Normalization metadata not found at {norm_path}.")
            print("  Reverting to identity scaling (raw values). Run Phase 0 to generate metadata!")
            metadata = None
        else:
            metadata = torch.load(norm_path, map_location='cpu', weights_only=True)
            y_min, y_max = metadata['y_min'], metadata['y_max']
            X_min, X_max = metadata['X_min'], metadata['X_max']
            print(f"  Successfully loaded normalization bounds from {norm_path}")
        
        # Scan directories and build the memory tensors
        for category, label_idx in label_mapping.items():
            y_files = glob.glob(os.path.join(root_dir, f"Y_{category}_trainingset.pth"))
            x_files = glob.glob(os.path.join(root_dir, f"X_{category}_trainingset.pth"))
            
            if not y_files:
                print(f"  WARNING: Tensors for '{category}' not found in {root_dir}")
                continue
            if y_files and x_files:
                # Load the first matching version
                y_tensor = torch.load(y_files[0], map_location='cpu', weights_only=True)
                x_tensor = torch.load(x_files[0], map_location='cpu', weights_only=True)
                
                # Standardize data type to float32 for model compatibility
                y_tensor = y_tensor.transpose(1, 2).float()
                
                # Extract the dynamic rotational speed (omega) from index 8
                omega_vals = x_tensor[:, 0, 8].float()
                
                # Apply Unified Min-Max Normalization
                if metadata is not None:
                    # Normalize Target Y (Acceleration)
                    y_min_f = y_min.view(1, 4, 1).float()
                    y_max_f = y_max.view(1, 4, 1).float()
                    y_tensor = (y_tensor - y_min_f) / (y_max_f - y_min_f + 1e-12)
                    
                    # Normalize Input X (Velocity and Position)
                    # Column indices: 0-3 (Vel), 4-7 (Pos)
                    for col in range(8):
                        x_tensor[:, :, col] = (x_tensor[:, :, col] - X_min[col]) / (X_max[col] - X_min[col] + 1e-12)
                
                # Ensure x_tensor is float32
                x_tensor = x_tensor.float()
                
                traces_list.append(y_tensor)
                labels_list.append(torch.full((y_tensor.shape[0],), label_idx, dtype=torch.long))
                omegas_list.append(omega_vals)
                
                print(f"  Loaded {y_tensor.shape[0]} windows for {category} class.")
            else:
                print(f"  WARNING: Tensors for '{category}' not found in {root_dir}")
                
        if len(traces_list) > 0:
            # Assuming uniform processing frequencies, the window_length will naturally match across classes
            self.all_traces = torch.cat(traces_list, dim=0)
            self.all_labels = torch.cat(labels_list, dim=0)
            self.all_omegas = torch.cat(omegas_list, dim=0)
            print(f"Dataset successfully built with {self.all_traces.shape[0]} total samples.")
        else:
            print("CRITICAL WARNING: No pre-processed .pth files were found. Dataset length is 0.")

    def __len__(self):
        # We cap the length to `num_samples` if provided to allow quick debugging
        length = self.all_traces.shape[0] if hasattr(self, 'all_traces') else 0
        return min(length, self.num_samples) if length > 0 else 0

    def __getitem__(self, idx):
        # O(1) lookup since everything is loaded in memory
        trace = self.all_traces[idx]
        label = self.all_labels[idx]
        omega = self.all_omegas[idx].item()
        
        # Fulfill the contract: return (raw, clean, label, omega)
        # Since there's no "clean" version computationally generated, we return the trace twice.
        return trace, trace, label, omega

# Absorb **kwargs to safely ignore synthetic-specific arguments like `seq_length` or `noise_std`
def get_dataloaders(batch_size=32, num_samples=1500, val_split=0.2, data_dir=cfg.DATA_DIR_PROCESSED, **kwargs):
    dataset = MaFaulDaDataset(root_dir=data_dir, num_samples=num_samples, seq_length=kwargs.get('seq_length', cfg.SEQ_LENGTH))

    if len(dataset) == 0:
        return None, None

    N = len(dataset)

    # Sequential interleaved split: every k-th window goes to validation.
    # This avoids placing time-adjacent full-rotation windows in both sets,
    # which would inflate validation metrics (the random_split problem).
    k = max(2, round(1.0 / val_split))          # e.g. val_split=0.2 → k=5
    val_indices   = list(range(0, N, k))         # 0, 5, 10, 15 ...
    train_indices = [i for i in range(N) if i % k != 0]

    train_dataset = Subset(dataset, train_indices)
    val_dataset   = Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  collate_fn=mafaulda_collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, collate_fn=mafaulda_collate_fn)

    return train_loader, val_loader, dataset.seq_length
