import torch
import torch.nn as nn

class PriorWorkOracle(nn.Module):
    def __init__(self, in_channels=4, seq_len=5000, embed_dim=64):
        super().__init__()
        
        # A simple CNN mapping (B, 4, 5000) -> (B, 64)
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=15, stride=5, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=10, stride=5, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), # -> (B, 32, 1)
            nn.Flatten(),            # -> (B, 32)
            nn.Linear(32, embed_dim),
            nn.Tanh()                # Embeddings bounded between -1 and 1
        )
        
        # Initialize randomly and explicitly detach from gradient graph
        for param in self.parameters():
            param.requires_grad = False
            
        # Target fault clusters mimicking a well-structured prior Latent Space
        # 1: Imbalance, 2: Outer-Race
        self.register_buffer('dist_class_1', torch.zeros(embed_dim))
        self.register_buffer('dist_class_2', torch.zeros(embed_dim))

    def forward(self, x):
        """
        Input: (Batch, 4, 5000) physical raw trace
        Output: (Batch, 64) embedding
        """
        return self.encoder(x)
        
    def set_target_distribution(self, target_class, empirical_embedding):
        """Allows main.py to set the actual reachable distribution"""
        if target_class == 1:
            self.dist_class_1.copy_(empirical_embedding.detach())
        elif target_class == 2:
            self.dist_class_2.copy_(empirical_embedding.detach())

    
    def get_target_distribution(self, target_class):
        if target_class == 1:
            return self.dist_class_1
        elif target_class == 2:
            return self.dist_class_2
        else:
            raise ValueError(f"Oracle mock only holds distributions for faults 1 and 2, got {target_class}")
