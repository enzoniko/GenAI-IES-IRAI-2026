"""Deterministic envelope decoder z_macro -> (B, C, T). Config-free."""
import torch.nn as nn


class Decoder1(nn.Module):
    def __init__(self, d_model=128, seq_length=4167, out_channels=4):
        super().__init__()
        self.seq_length = seq_length
        self.init_length, self.init_channels = 125, 256
        self.fc_proj = nn.Sequential(
            nn.Linear(d_model, 512), nn.GELU(),
            nn.Linear(512, self.init_channels * self.init_length),
        )
        self.decoder_cnn = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(256, 128, kernel_size=5, padding=2), nn.BatchNorm1d(128), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(128, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(64, 32, kernel_size=5, padding=2), nn.BatchNorm1d(32), nn.GELU(),
            nn.Upsample(size=seq_length, mode="linear", align_corners=False),
            nn.Conv1d(32, out_channels, kernel_size=5, padding=2),
        )

    def forward(self, z_macro):
        x = self.fc_proj(z_macro)
        x = x.view(-1, self.init_channels, self.init_length)
        return self.decoder_cnn(x)
