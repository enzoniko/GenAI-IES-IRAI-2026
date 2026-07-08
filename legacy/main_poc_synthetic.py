import os
import math
import traceback
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import src.configs as cfg

cfg.RESULTS_DIR = "results-poc-synthetic"
os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
cfg.PINN_MODEL_PATH = "results-poc-synthetic/pinn.pth"
cfg.JEPA_MODEL_PATH = "results-poc-synthetic/ts_jepa.pth"
cfg.DEC1_MODEL_PATH = "results-poc-synthetic/decoder1.pth"
cfg.DEC2_MODEL_PATH = "results-poc-synthetic/decoder2.pth"
cfg.LDM_MODEL_PATH = "results-poc-synthetic/ldm.pth"
cfg.NUM_CLASSES = 3
cfg.SEQ_LENGTH = 5000
cfg.SAMPLING_RATE = 50000

from src.data.synthetic_dataset import get_dataloaders
from src.models.decoder1 import Decoder1
from src.models.decoder2_cvae import Decoder2CVAE
from src.models.latent_diffusion import DDPMScheduler, LatentDiffusionMLP
from src.models.ts_jepa import TSJEPA
from src.pipelines.train_phase1 import (
    extract_residuals_and_train_decoder2,
    train_phase1_decoder1,
    train_phase1_tsjepa,
)
from src.pipelines.run_sdedit_phase2 import train_latent_diffusion


# ==========================================
# 1. Trainable PINN & Oracle for PoC Data
# ==========================================
class TrainableSyntheticPINN(nn.Module):
    """
    Learns the parameters of the healthy PoC equation:
    H_c(t) = alpha_c * [sin(2pi*50*t + phi_c) + 0.5*sin(2pi*100*t + phi_c)]
    """
    def __init__(self, seq_length=5000):
        super().__init__()
        self.seq_length = seq_length
        # MLP to predict 4 alphas and 4 phis from the 4-channel signal
        self.mlp = nn.Sequential(
            nn.Linear(4 * seq_length, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 8) # 4 alphas, 4 phis
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        params = self.mlp(x_flat)
        alphas = params[:, :4]
        phis = params[:, 4:]
        return alphas, phis


class SyntheticOracle(nn.Module):
    """
    Wraps the PINN to output a physical embedding space for SGEdit.
    Embedding = [alphas, phis, residuals] (dim=12)
    """
    def __init__(self, pinn_model, seq_length=5000, sample_rate=50000):
        super().__init__()
        self.pinn = pinn_model
        self.seq_length = seq_length
        self.sample_rate = sample_rate
        # Register time grid as buffer so it moves to correct device
        t = torch.linspace(0, (seq_length - 1) / sample_rate, seq_length)
        self.register_buffer('t', t)
        
        self.embed_dim = 12 # 4 alphas + 4 phis + 4 residuals
        self.register_buffer('target_distributions', torch.zeros(cfg.NUM_CLASSES, self.embed_dim))
        
    def forward(self, x, omega=None):
        alphas, phis = self.pinn(x)
        
        # Calculate residuals
        t_view = self.t.view(1, 1, -1)
        alphas_view = alphas.unsqueeze(-1)
        phis_view = phis.unsqueeze(-1)
        
        # Predicted Healthy Envelope
        h_pred = alphas_view * (
            torch.sin(2 * math.pi * 50 * t_view + phis_view) + 
            0.5 * torch.sin(2 * math.pi * 100 * t_view + phis_view)
        )
        
        residuals = torch.mean((x - h_pred)**2, dim=2) # Shape: (batch, 4)
        
        # Physics-informed embedding
        embedding = torch.cat([alphas, phis, residuals], dim=1) # Shape: (batch, 12)
        return embedding
        
    def set_target_distribution(self, class_idx, embedding):
        self.target_distributions[class_idx].copy_(embedding)

    def get_target_distribution(self, class_idx):
        return self.target_distributions[class_idx]


def train_phase0_pinn(train_loader, device, epochs=30):
    print("\n=== Phase 0: Training PINN on Healthy PoC Equations ===")
    pinn = TrainableSyntheticPINN(seq_length=cfg.SEQ_LENGTH).to(device)
    optimizer = optim.Adam(pinn.parameters(), lr=1e-3)
    
    t_grid = torch.linspace(0, (cfg.SEQ_LENGTH - 1) / cfg.SAMPLING_RATE, cfg.SEQ_LENGTH).to(device)
    t_view = t_grid.view(1, 1, -1)
    
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        pinn.train()
        total_loss = 0
        batches = 0
        for raw, _, labels in train_loader:
            mask = labels == 0 # Train ONLY on healthy
            if mask.sum() == 0: continue
            healthy_raw = raw[mask].to(device)
            
            optimizer.zero_grad()
            alphas, phis = pinn(healthy_raw)
            alphas_view = alphas.unsqueeze(-1)
            phis_view = phis.unsqueeze(-1)
            
            h_pred = alphas_view * (
                torch.sin(2 * math.pi * 50 * t_view + phis_view) + 
                0.5 * torch.sin(2 * math.pi * 100 * t_view + phis_view)
            )
            
            loss = criterion(healthy_raw, h_pred)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1
            
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, PINN Residual Loss: {total_loss/max(1, batches):.6f}")
            
    torch.save(pinn.state_dict(), cfg.PINN_MODEL_PATH)
    return pinn


# ==========================================
# 2. Custom SGEdit Loop for Visualization
# ==========================================
def run_custom_sgedit(ts_jepa, decoder1, decoder2, ldm, oracle, scheduler, 
                      healthy_trace, target_class_idx, device, 
                      num_inference_steps=50, guidance_scale=0.5, strength=0.15):
    
    ts_jepa.eval()
    decoder1.eval()
    decoder2.eval()
    ldm.eval()
    oracle.eval()
    
    t_start = int(num_inference_steps * strength)
    
    with torch.no_grad():
        z_start = ts_jepa.get_z_macro(healthy_trace).float()
        orig_macro_recon = decoder1(z_start)
        
    noise = torch.randn_like(z_start)
    t_start_tensor = torch.tensor([t_start], device=device).long()
    z_t = scheduler.add_noise(z_start, noise, t_start_tensor)
    
    target_dist = oracle.get_target_distribution(target_class_idx).to(device)
    mse_penalty = nn.MSELoss()
    
    z_trajectory = [z_start.detach().cpu()]
    
    for t in reversed(range(0, t_start)):
        z_trajectory.append(z_t.detach().cpu())
        t_tensor = torch.tensor([t], device=device).long()
        with torch.no_grad():
            uncond_noise_pred = ldm(z_t, t_tensor)
            
        z_t = z_t.detach().requires_grad_(True)
        pred_trace = decoder1(z_t)
        pred_emb = oracle(pred_trace)
        
        penalty = mse_penalty(pred_emb.squeeze(0), target_dist)
        grad = torch.autograd.grad(penalty, z_t)[0]
        
        z_t.requires_grad_(False)
        z_t = scheduler.step(uncond_noise_pred, t, z_t)
        z_t = z_t - guidance_scale * grad

    label_tensor = torch.tensor([target_class_idx], dtype=torch.long, device=device)
    with torch.no_grad():
        final_macro_recon = decoder1(z_t)
        sampled_fault_jitter = decoder2.sample(z_t, label_tensor)
        final_counterfactual = final_macro_recon + sampled_fault_jitter
        
    z_trajectory.append(z_t.detach().cpu())
    
    return {
        'trajectory': torch.cat(z_trajectory, dim=0),
        'dec1_recon': final_macro_recon[0].cpu().numpy(),
        'cvae_jitter': sampled_fault_jitter[0].cpu().numpy(),
        'final_trace': final_counterfactual[0].cpu().numpy(),
        'orig_recon': orig_macro_recon[0].cpu().numpy()
    }


# ==========================================
# 3. Main Orchestration
# ==========================================
def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cpu")
    print(f"Device: {device}")

    # Data
    train_loader, val_loader = get_dataloaders(
        batch_size=32, num_samples=600, seq_length=cfg.SEQ_LENGTH, sample_rate=cfg.SAMPLING_RATE
    )
    
    # Phase 0
    pinn = train_phase0_pinn(train_loader, device, epochs=20)
    pinn.eval()
    
    # Oracle
    oracle = SyntheticOracle(pinn, seq_length=cfg.SEQ_LENGTH, sample_rate=cfg.SAMPLING_RATE).to(device)
    
    # Phase 1
    print("\n=== Phase 1: TS-JEPA + Decoders ===")
    ts_jepa = TSJEPA(in_channels=4).to(device)
    decoder1 = Decoder1(out_channels=4, seq_length=cfg.SEQ_LENGTH).to(device)
    decoder2 = Decoder2CVAE(in_channels=4, seq_length=cfg.SEQ_LENGTH).to(device)
    
    ts_jepa = train_phase1_tsjepa(ts_jepa, train_loader, val_loader, 15, device)
    decoder1 = train_phase1_decoder1(ts_jepa, decoder1, train_loader, val_loader, 15, device)
    decoder2 = extract_residuals_and_train_decoder2(ts_jepa, decoder1, decoder2, train_loader, val_loader, 15, device)
    
    # Phase 2 (LDM)
    print("\n=== Phase 2: Latent Diffusion Model ===")
    ldm = LatentDiffusionMLP(z_dim=int(cfg.JEPA_CONFIG['d_model']), time_dim=64).to(device)
    scheduler = DDPMScheduler(num_train_timesteps=100, device=device)
    for p in ts_jepa.parameters(): p.requires_grad = False
    ldm = train_latent_diffusion(ts_jepa, ldm, scheduler, train_loader, val_loader, device, epochs=15)
    
    # Oracle Calibration
    print("\n=== Calibrating Physics Oracle ===")
    oracle.eval()
    with torch.no_grad():
        for class_idx in range(cfg.NUM_CLASSES):
            class_embs = []
            for raw, _, labels in val_loader:
                mask = labels == class_idx
                if mask.sum() > 0:
                    class_embs.append(oracle(raw[mask].to(device)))
            if class_embs:
                avg_emb = torch.cat(class_embs, dim=0).mean(0)
                oracle.set_target_distribution(class_idx, avg_emb)
                
    # Get Healthy Trace
    healthy_raw = None
    for raw, _, labels in val_loader:
        mask = labels == 0
        if mask.any():
            healthy_raw = raw[mask][0:1].to(device)
            break

    # Run SGEdit for Fault 1 and Fault 2
    print("\n=== Generating Counterfactuals via SGEdit ===")
    results_f1 = run_custom_sgedit(ts_jepa, decoder1, decoder2, ldm, oracle, scheduler, healthy_raw, 1, device)
    results_f2 = run_custom_sgedit(ts_jepa, decoder1, decoder2, ldm, oracle, scheduler, healthy_raw, 2, device)

    # Calculate UMAP
    print("\n=== Plotting 5-Part Visualization ===")
    all_z, all_labels = [], []
    with torch.no_grad():
        for raw, _, labels in val_loader:
            all_z.append(ts_jepa.get_z_macro(raw.to(device)).cpu())
            all_labels.append(labels)
    all_z = torch.cat(all_z, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    traj_f1 = results_f1['trajectory'].numpy()
    
    import umap
    reducer = umap.UMAP(n_components=2, random_state=42)
    combined_data = np.concatenate([all_z, traj_f1], axis=0)
    combined_emb = reducer.fit_transform(combined_data)
    bg_emb = combined_emb[:len(all_z)]
    traj_emb = combined_emb[len(all_z):]

    # Plotting
    fig = plt.figure(figsize=(16, 18))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1.5], hspace=0.3, wspace=0.2)
    t_ax = np.linspace(0, cfg.SEQ_LENGTH / cfg.SAMPLING_RATE, cfg.SEQ_LENGTH)
    
    # Offset plotting function
    def plot_breakdown(ax, dec1, cvae, final, title):
        offset = 2.5
        # Plot only channel 0 for clarity
        ax.plot(t_ax, dec1[0] + offset, label='Decoder 1 Recon', color='blue')
        ax.plot(t_ax, cvae[0], label='CVAE Jitter', color='green', alpha=0.8)
        ax.plot(t_ax, final[0] - offset, label='Final Superposition', color='black')
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Acceleration (Offset)")
        ax.legend(loc='upper right', fontsize=8)

    # (a) & (b)
    ax_a = fig.add_subplot(gs[0, 0])
    plot_breakdown(ax_a, results_f1['dec1_recon'], results_f1['cvae_jitter'], results_f1['final_trace'], "(a) Fault Class 1 Synthesis")
    ax_b = fig.add_subplot(gs[0, 1])
    plot_breakdown(ax_b, results_f2['dec1_recon'], results_f2['cvae_jitter'], results_f2['final_trace'], "(b) Fault Class 2 Synthesis")
    
    # (c) & (d)
    h_trace = healthy_raw[0, 0].cpu().numpy()
    def plot_cf(ax, healthy, cf, title):
        ax.plot(t_ax, healthy, label='Original Healthy', color='blue', alpha=0.5)
        ax.plot(t_ax, cf[0], label='Guided Counterfactual', color='red', linestyle='dashed')
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Acceleration")
        ax.legend(loc='upper right', fontsize=8)

    ax_c = fig.add_subplot(gs[1, 0])
    plot_cf(ax_c, h_trace, results_f1['final_trace'], "(c) Counterfactual (Healthy -> Fault 1)")
    ax_d = fig.add_subplot(gs[1, 1])
    plot_cf(ax_d, h_trace, results_f2['final_trace'], "(d) Counterfactual (Healthy -> Fault 2)")
    
    # (e) UMAP
    ax_e = fig.add_subplot(gs[2, :])
    scatter = ax_e.scatter(bg_emb[:, 0], bg_emb[:, 1], c=all_labels, cmap='viridis', s=15, alpha=0.3)
    classes = ["Healthy", "Imbalance", "Outer-Race"]
    handles, _ = scatter.legend_elements(prop="colors")
    ax_e.legend(handles, classes, title="Fault Classes", loc="upper left")
    
    import matplotlib.cm as cm
    colors = cm.Reds(np.linspace(0.2, 1.0, len(traj_emb)))
    ax_e.scatter(traj_emb[1:, 0], traj_emb[1:, 1], c=colors[1:], s=40, edgecolors='none', label='SDEdit Guid. Path')
    ax_e.scatter(traj_emb[0, 0], traj_emb[0, 1], color='black', s=150, marker='X', label='Start (Healthy)')
    ax_e.scatter(traj_emb[-1, 0], traj_emb[-1, 1], color='darkred', s=200, marker='*', label='End (Class 1)')
    ax_e.set_title("(e) SDEdit Latent Trajectory in Physics-Informed Embedding Space")
    ax_e.set_xlabel("UMAP 1")
    ax_e.set_ylabel("UMAP 2")
    ax_e.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig("poc_synthetic_validation.png", dpi=300)
    plt.close()
    print("Done! Visualization saved to poc_synthetic_validation.png")

if __name__ == "__main__":
    main()
