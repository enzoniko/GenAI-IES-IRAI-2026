import torch
import torch.nn as nn

class EncoderCVAE(nn.Module):
    def __init__(self, seq_length=5000, in_channels=4, context_dim=128, latent_dim=64, num_classes=3, label_embed_dim=16):
        super().__init__()
        
        # CNN downsampling the residual
        self.cnn = nn.Sequential(
            # Input: (Batch, 4, 5000)
            nn.Conv1d(in_channels, 32, kernel_size=5, stride=5, padding=2), # -> 1000
            nn.BatchNorm1d(32),
            nn.GELU(),
            
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2), # -> 500
            nn.BatchNorm1d(64),
            nn.GELU(),
            
            nn.Conv1d(64, 128, kernel_size=5, stride=5, padding=2), # -> 100
            nn.BatchNorm1d(128),
            nn.GELU(),
        )
        
        self.flat_dim = 128 * 100
        self.label_embed = nn.Embedding(num_classes, label_embed_dim)
        
        self.fc_joint = nn.Sequential(
            nn.Linear(self.flat_dim + context_dim + label_embed_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU()
        )
        
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
    def forward(self, residual, z_macro, label):
        # residual: (Batch, 4, 5000)
        cnn_features = self.cnn(residual)
        cnn_features = cnn_features.view(cnn_features.size(0), -1)
        
        l_emb = self.label_embed(label)
        
        x = torch.cat([cnn_features, z_macro, l_emb], dim=1)
        h = self.fc_joint(x)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class DecoderCVAE(nn.Module):
    def __init__(self, latent_dim=64, context_dim=128, seq_length=5000, out_channels=4, num_classes=3, label_embed_dim=16):
        super().__init__()
        
        self.init_length = 125
        self.init_channels = 256
        self.label_embed = nn.Embedding(num_classes, label_embed_dim)
        
        # Project concat(z, z_macro, label_embed) back to initial sequence state
        self.fc_proj = nn.Sequential(
            nn.Linear(latent_dim + context_dim + label_embed_dim, 512),
            nn.GELU(),
            nn.Linear(512, self.init_channels * self.init_length)
        )
        
        # Upsampling CNN
        self.cnn = nn.Sequential(
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
        
    def forward(self, z, z_macro, label):
        l_emb = self.label_embed(label)
        x = torch.cat([z, z_macro, l_emb], dim=1)
        x = self.fc_proj(x)
        x = x.view(-1, self.init_channels, self.init_length)
        out = self.cnn(x)
        return out

class Decoder2CVAE(nn.Module):
    def __init__(self, seq_length=5000, in_channels=4, context_dim=128, latent_dim=64, num_classes=3, label_embed_dim=16):
        super().__init__()
        self.encoder = EncoderCVAE(seq_length, in_channels, context_dim, latent_dim, num_classes, label_embed_dim)
        self.decoder = DecoderCVAE(latent_dim, context_dim, seq_length, in_channels, num_classes, label_embed_dim)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, residual, z_macro, label):
        mu, logvar = self.encoder(residual, z_macro, label)
        z = self.reparameterize(mu, logvar)
        recon_residual = self.decoder(z, z_macro, label)
        return recon_residual, mu, logvar

    def sample(self, z_macro, label):
        z = torch.randn(z_macro.size(0), self.encoder.fc_mu.out_features, device=z_macro.device)
        return self.decoder(z, z_macro, label)
