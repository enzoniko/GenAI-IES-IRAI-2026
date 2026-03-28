import torch
from torch.utils.data import Dataset, DataLoader, random_split
import math

class VibrationDataset(Dataset):
    """
    Generates synthetic multivariate (4-channel) vibration traces on the fly.
    Classes:
    0: Healthy (Low-freq deterministic)
    1: Imbalance Fault (Healthy + Strong 1X harmonic)
    2: Outer-Race Fault (Healthy + Periodic BPFO impacts)
    
    Noise is added separately to all classes.
    Shape: (Batch, Channels=4, Length=5000)
    """
    def __init__(self, num_samples: int = 1500, seq_length: int = 5000, sample_rate: int = 50000, num_channels: int = 4, noise_std: float = 0.05):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.noise_std = noise_std
        
        # Time vector
        self.t = torch.linspace(0, (seq_length - 1) / sample_rate, seq_length)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        fault_label = idx % 3
        
        raw_traces = []
        clean_traces = []
        
        for c in range(self.num_channels):
            # Phase shifts and amplitude variations for each channel
            phase_shift = c * (math.pi / 8.0)
            amp_scale = 1.0 - (c * 0.1)
            
            # Healthy Envelope
            baseline = amp_scale * (torch.sin(2 * math.pi * 50 * self.t + phase_shift) + 
                                    0.5 * torch.sin(2 * math.pi * 100 * self.t + phase_shift))
            
            fault_signal = torch.zeros(self.seq_length)
            
            if fault_label == 1:
                # Imbalance
                fault_signal = 1.2 * amp_scale * torch.sin(2 * math.pi * 50 * self.t + math.pi/4 + phase_shift)
                
            elif fault_label == 2:
                # Outer-race fault (BPFO): 200 Hz
                bpfo_freq = 200.0
                t_mod = self.t % (1.0 / bpfo_freq)
                decay = 1000.0
                resonance_freq = 2000.0 + (c * 100) # Slightly different resonant freq per channel
                fault_signal = amp_scale * torch.exp(-decay * t_mod) * torch.sin(2 * math.pi * resonance_freq * self.t)
            
            # Jitter
            jitter = 0.1 * amp_scale * torch.sin(2 * math.pi * 5000 * self.t + phase_shift) + \
                     torch.randn(self.seq_length) * self.noise_std
                     
            clean_traces.append(baseline + fault_signal)
            raw_traces.append(baseline + fault_signal + jitter)
            
        raw_trace_tensor = torch.stack(raw_traces, dim=0) # [4, 5000]
        clean_trace_tensor = torch.stack(clean_traces, dim=0) # [4, 5000]
        
        return raw_trace_tensor.float(), clean_trace_tensor.float(), torch.tensor(fault_label, dtype=torch.long)

def get_dataloaders(batch_size=32, num_samples=1500, seq_length=5000, sample_rate=50000, noise_std=0.05, val_split=0.2):
    dataset = VibrationDataset(num_samples, seq_length, sample_rate, num_channels=4, noise_std=noise_std)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
