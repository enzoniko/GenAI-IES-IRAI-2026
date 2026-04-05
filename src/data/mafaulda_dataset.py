import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
import glob
import pandas as pd
import random

class BatchTuple(tuple):
    """
    A custom tuple to pass the dynamic `omega` without breaking the 
    standard diffusion dataloader signature (which expects exactly 3 items).
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
    Data loader for the real MaFaulDa dataset.
    This class adheres to the contract required by the training pipelines.
    
    Contract:
    - __getitem__ must return (raw_trace, clean_trace, label).
    - For real data without a "clean" version, raw_trace is returned twice.
    - Output Tensors:
        - trace: (4, 5000) float tensor
        - label: long tensor
    """
    def __init__(self, root_dir, num_samples: int = 1500, seq_length: int = 5000, num_channels: int = 4):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.num_channels = num_channels
        
        self.file_paths = []
        self.labels = []
        
        # MaFaulDa folder structure mapping to integer labels
        # 0: Normal/Healthy, 1: Imbalance, 2: Overhang Bearing (Example mapping)
        label_mapping = {
            'normal': 0,
            'imbalance': 1,
            'overhang': 2
        }
        
        # Scan directories and build the map of files
        for folder_name, label_idx in label_mapping.items():
            folder_path = os.path.join(root_dir, folder_name)
            if os.path.exists(folder_path):
                # Find all csv files in this directory and its subdirectories
                csv_files = glob.glob(os.path.join(folder_path, '**/*.csv'), recursive=True)
                for file in csv_files:
                    self.file_paths.append(file)
                    self.labels.append(label_idx)
            else:
                print(f"WARNING: Folder {folder_name} not found in {root_dir}. Check your directory names!")
                
        if len(self.file_paths) == 0:
            print("CRITICAL WARNING: No CSV files were found. Dataset length is 0.")

    def __len__(self):
        # Return the actual number of files we found, or a capped amount for quick testing
        return min(len(self.file_paths), self.num_samples) if self.file_paths else 0

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # 1. Read the CSV file. MaFaulDa has no headers, so we set header=None
        df = pd.read_csv(file_path, header=None)
        
        # 1.5 Extract dynamic rotational speed (omega) from the filename
        try:
            hz_value = float(os.path.basename(file_path).replace('.csv', ''))
        except ValueError:
            hz_value = 20.0
        omega = hz_value * 2 * 3.14159

        # 2. MaFaulDa has 8 columns. We must match the prior work PINN Oracle channels.
        # 0: tachometer, 1: under_ax, 2: under_rad, 3: under_tan
        # 4: over_ax,    5: over_rad,   6: over_tan,  7: microphone
        # We select the Radial and Tangential axes for BOTH bearings to capture 2D orbital motion.
        selected_columns = df.iloc[:, [2, 3, 5, 6]].values
        
        # 3. Convert to PyTorch tensor and transpose from (Length, Channels) to (Channels, Length)
        trace = torch.tensor(selected_columns, dtype=torch.float32).T
        
        # 3.5 Physical Scaling (Replaces Per-Sample Normalization)
        # The MaFaulDa sensors have a sensitivity of 100 mV/g = 0.0102 V / (m/s^2)
        
        # 1. Convert raw voltage to actual physical acceleration (m/s^2)
        trace = trace / 0.0102
        
        # 2. Global soft-scaling to keep values roughly in [-1, 1] for stable neural network training.
        # This preserves the relative physical magnitude (severe faults remain larger than healthy ones).
        trace = trace / 20.0
        
        # 4. Extract sequence length (with Random Cropping for Data Augmentation)
        signal_length = trace.shape[1]
        if signal_length > self.seq_length:
            # Random crop instead of deterministic crop
            start_idx = random.randint(0, signal_length - self.seq_length)
            trace = trace[:, start_idx : start_idx + self.seq_length]
        elif signal_length < self.seq_length:
            # If the file is too short (rare in MaFaulDa), pad it with zeros
            pad_size = self.seq_length - signal_length
            trace = torch.nn.functional.pad(trace, (0, pad_size))
        
        # Fulfill the contract: return (raw, clean, label)
        # Since there's no "clean" version, we return the trace twice.
        return trace.float(), trace.float(), torch.tensor(label, dtype=torch.long), omega

# Absorb **kwargs to safely ignore synthetic-specific arguments like `seq_length` or `noise_std`
def get_dataloaders(batch_size=32, num_samples=1500, val_split=0.2, data_dir='mafaulda-data', **kwargs):
    # Pointing to the real MaFaulDa dataset directory based on your handoff
    dataset = MaFaulDaDataset(root_dir=data_dir, num_samples=num_samples)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=mafaulda_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=mafaulda_collate_fn)
    
    return train_loader, val_loader