import torch
import torch.nn as nn
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class LatentDiffusionMLP(nn.Module):
    def __init__(self, z_dim=128, time_dim=64):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.GELU(),
            nn.Linear(time_dim * 2, time_dim)
        )
        
        # A simple series of ResNet-like MLP blocks
        self.fc1 = nn.Linear(z_dim, 256)
        self.fc_time1 = nn.Linear(time_dim, 256)
        
        self.fc2 = nn.Linear(256, 256)
        self.fc_time2 = nn.Linear(time_dim, 256)
        
        self.fc3 = nn.Linear(256, z_dim)
        
        self.act = nn.SiLU()

    def forward(self, x, time):
        # x is (Batch, 128)
        # time is (Batch,)
        
        t = self.time_mlp(time)
        
        h = self.fc1(x) + self.fc_time1(t)
        h = self.act(h)
        
        # Residual-like jump
        h2 = self.fc2(h) + self.fc_time2(t)
        h2 = self.act(h2) + h
        
        out = self.fc3(h2)
        return out

class DDPMScheduler:
    def __init__(self, num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02, device="cpu"):
        self.num_train_timesteps = num_train_timesteps
        self.device = device
        
        # Linear schedule
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(device), self.alphas_cumprod[:-1]])
        
        # Calculations for forward and reverse process
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def add_noise(self, original_samples, noise, timesteps):
        """Forward diffusion process"""
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].view(-1, 1)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1)
        
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def step(self, model_output, timestep, sample):
        """Reverse diffusion discrete step"""
        t = timestep
        
        # 1. compute alphas
        alpha_t = self.alphas[t]
        alpha_t_cumprod = self.alphas_cumprod[t]
        
        # 2. compute predicted original sample from predicted noise (also called x_0)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        
        # 3. compute derivative term (the mean)
        pred_mean = (sample - model_output * (1 - alpha_t) / sqrt_one_minus_alpha_cumprod_t) / torch.sqrt(alpha_t)
        
        # 4. output
        if t > 0:
            noise = torch.randn_like(sample)
            variance = self.posterior_variance[t]
            pred_sample = pred_mean + torch.sqrt(variance) * noise
        else:
            pred_sample = pred_mean
            
        return pred_sample
