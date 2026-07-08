"""
Siamese neural network implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseNetwork(nn.Module):
    """Siamese network with CNN encoder"""
    
    def __init__(self, in_channels=4, feature_size=128, embedding_size=16):
        """
        Siamese network with CNN encoder
        
        Args:
            in_channels: Number of input channels
            feature_size: Size of input features
            embedding_size: Size of embedding vector
        """
        super(SiameseNetwork, self).__init__()
        
        self.encoder = nn.Sequential(
            # First conv block
            nn.Conv1d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Second conv block
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Third conv block
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        
        # Calculate size after convolutions and pooling (feature_size / 8)
        conv_output_size = feature_size // 8 * 16
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_size)
        )
    
    def forward_one(self, x):
        """Encode one input"""
        x = self.encoder(x)
        x = self.fc(x)
        # L2 normalize embeddings
        x = F.normalize(x, p=2, dim=1)
        return x
    
    def forward(self, x1, x2=None):
        """
        Forward pass
        
        Args:
            x1: First input
            x2: Optional second input
        
        Returns:
            If x2 is provided, returns embeddings for both inputs
            Otherwise, returns embedding for x1
        """
        embedding1 = self.forward_one(x1)
        
        if x2 is not None:
            embedding2 = self.forward_one(x2)
            return embedding1, embedding2
        
        return embedding1


class TripletLoss(nn.Module):
    """Triplet loss with hard margin"""
    
    def __init__(self, margin=1.0):
        """
        Triplet loss with hard margin
        
        Args:
            margin: Margin for triplet loss
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        """
        Compute triplet loss
        
        Args:
            anchor, positive, negative: Embeddings
        
        Returns:
            loss: Mean triplet loss
        """
        # Compute distances
        pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
        neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
        
        # Compute triplet loss
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        
        # Return mean loss
        return torch.mean(loss) 