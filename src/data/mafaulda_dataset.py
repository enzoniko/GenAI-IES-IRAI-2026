import torch
from torch.utils.data import Dataset, DataLoader, Subset
import os
import glob
import pandas as pd
import src.constants as c

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
    def __init__(self, root_dir, num_samples: int = 1500, seq_length: int = 5000, num_channels: int = 4):
        # seq_length is absorbed as **kwargs natively handles it from generative architectures, 
        # but we rely on the internal physical window_length of the pre-processed data.
        self.num_samples = num_samples
        self.num_channels = num_channels
        
        traces_list = []
        labels_list = []
        omegas_list = []
        
        # MaFaulDa folder structure mapping to integer labels
        # 0: Normal/Healthy, 1: Imbalance, 2: Overhang Bearing
        label_mapping = {
            'normal': 0,
            'imbalance': 1,
            'overhang': 2
        }
        
        print(f"Loading pre-processed datasets from {root_dir}...")
        
        # Scan directories and build the memory tensors
        for category, label_idx in label_mapping.items():
            # Search for the trainingset tensors produced by CleanMaFaulDaProcessor.
            # The _trainingset suffix marks tensors intended for training + validation;
            # the _testset tensors are held out and never loaded here.
            y_files = glob.glob(os.path.join(root_dir, f"Y_{category}_*_trainingset.pth"))
            x_files = glob.glob(os.path.join(root_dir, f"X_{category}_*_trainingset.pth"))
            
            if y_files and x_files:
                # Load the first matching version
                y_tensor = torch.load(y_files[0], map_location='cpu', weights_only=True)
                x_tensor = torch.load(x_files[0], map_location='cpu', weights_only=True)
                
                # y_tensor comes in as (Num_Windows, Window_Size, Channels)
                # Conv1D architectures expect (Num_Windows, Channels, Window_Size)
                y_tensor = y_tensor.transpose(1, 2).float()
                
                # Extract the dynamic rotational speed (omega) from the X tensor's index 8
                # It is constant across the window, so we just take the first element's omega
                omega_vals = x_tensor[:, 0, 8].float()
                
                # Physical Soft-Scaling (to roughly [-1, 1] for stable NN convergence)
                # Note: clean_mafaulda_processor already converted voltage to m/s^2!
                y_tensor = y_tensor / c.PHYSICAL_SOFT_SCALE
                
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
def get_dataloaders(batch_size=32, num_samples=1500, val_split=0.2, data_dir='data/processed-mafaulda/v1', **kwargs):
    dataset = MaFaulDaDataset(root_dir=data_dir, num_samples=num_samples, seq_length=kwargs.get('seq_length', 5000))

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

    return train_loader, val_loader