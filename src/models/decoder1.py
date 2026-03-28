import torch
import torch.nn as nn

class Decoder1(nn.Module):
    def __init__(self, d_model=128, seq_length=5000, out_channels=4):
        super().__init__()
        self.seq_length = seq_length
        self.out_channels = out_channels
        
        # Initial projection to a small sequence (125 length)
        self.init_length = 125
        self.init_channels = 256
        self.fc_proj = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.GELU(),
            nn.Linear(512, self.init_channels * self.init_length)
        )
        
        # CNN Upsampling blocks to scale from 125 -> 5000
        self.decoder_cnn = nn.Sequential(
            # Input: (Batch, 256, 125)
            nn.Upsample(scale_factor=2, mode='nearest'), # -> 250
            nn.Conv1d(256, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            
            nn.Upsample(scale_factor=2, mode='nearest'), # -> 500
            nn.Conv1d(128, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            
            nn.Upsample(scale_factor=2, mode='nearest'), # -> 1000
            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
            
            nn.Upsample(scale_factor=5, mode='nearest'), # -> 5000
            nn.Conv1d(32, out_channels, kernel_size=5, padding=2)
        )

    def forward(self, z_macro):
        # z_macro: (Batch, d_model)
        x = self.fc_proj(z_macro)
        x = x.view(-1, self.init_channels, self.init_length)
        out = self.decoder_cnn(x)
        return out
