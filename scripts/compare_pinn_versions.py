import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.configs as cfg
from src.models.pinn import ConfigurablePINN
from src.models.relobralo_loss import ReLoBRaLoLoss

# ── Legacy Model (No Non-Dimensionalization) ──────────────────────────────────
class LegacyPINN(ConfigurablePINN):
    def compute_residuals(self, x, pred, X_max=None, X_min=None, y_max=None, y_min=None):
        # Call super to get raw residuals (super has scaling at the end)
        # Actually, I'll copy the logic to avoid the division by 1e6
        
        x = x.double()
        pred = pred.double()
        
        M1 = torch.clamp(self.M1, min=0.1)    
        M2 = torch.clamp(self.M2, min=0.1)
        M3 = torch.clamp(self.M3, min=0.1)
        D1 = torch.clamp(self.D1, min=0.0)
        D2 = torch.clamp(self.D2, min=0.0)
        D3 = torch.clamp(self.D3, min=0.0)
        K1 = torch.clamp(self.K1, min=1.0)
        K2 = torch.clamp(self.K2, min=0.1)
        E1 = torch.clamp(self.E1, min=0.0)

        (x2_denorm, y2_denorm, x3_denorm, y3_denorm,
         x2_dot_denorm, y2_dot_denorm, x3_dot_denorm, y3_dot_denorm,
         x2_ddot_denorm, y2_ddot_denorm, x3_ddot_denorm, y3_ddot_denorm,
         omega_phys, t_phys) = self._denormalize_physical_variables(x, pred, X_max, X_min, y_max, y_min)
        
        K2_K1_ratio = K2 / K1
        
        # RAW Residuals (No Scaling)
        residual1 = K1*x2_denorm + K2*x3_denorm + M1*omega_phys**2*E1*torch.cos(omega_phys*t_phys) - self.fA
        residual2 = K1*y2_denorm + K2*y3_denorm - M1*self.g + M1*omega_phys**2*E1*torch.sin(omega_phys*t_phys) - self.fB
        residual3 = M3*x3_ddot_denorm + D3*x3_dot_denorm + K2*x3_denorm - K2_K1_ratio*M2*x2_ddot_denorm - K2_K1_ratio*D2*x2_dot_denorm - K2*x2_denorm - self.fC
        residual4 = M3*y3_ddot_denorm + D3*y3_dot_denorm + K2*y3_denorm - K2_K1_ratio*M2*y2_ddot_denorm - K2_K1_ratio*D2*y2_dot_denorm - K2*y2_denorm - K2_K1_ratio*M2*self.g + M3*self.g - self.fD

        if self.enable_mass_constraints:
            residualMass1 = M1 + M2 + M3 - 22.0
            residualMass2 = M2 - M3
            return residual1, residual2, residual3, residual4, residualMass1, residualMass2
        else:
            residualMass1 = torch.zeros_like(residual1)
            residualMass2 = torch.zeros_like(residual2)
            return residual1, residual2, residual3, residual4, residualMass1, residualMass2

# ── Training Benchmark ────────────────────────────────────────────────────────
def run_comparison(epochs=200):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Comparison Task: Legacy vs Current Math on {device}")
    
    # Load Real Normal Data (Subset)
    data_dir = cfg.DATA_DIR_PROCESSED
    X_raw = torch.load(os.path.join(data_dir, f"X_normal_{cfg.DATASET_VERSION}_trainingset.pth"), map_location='cpu')
    Y_raw = torch.load(os.path.join(data_dir, f"Y_normal_{cfg.DATASET_VERSION}_trainingset.pth"), map_location='cpu')
    
    # Select first batch
    X_raw = X_raw[:128].double()
    Y_raw = Y_raw[:128].double()
    
    X_max = X_raw.reshape(-1, X_raw.shape[-1]).max(dim=0)[0]
    X_min = X_raw.reshape(-1, X_raw.shape[-1]).min(dim=0)[0]
    y_max = Y_raw.reshape(-1, Y_raw.shape[-1]).max(dim=0)[0]
    y_min = Y_raw.reshape(-1, Y_raw.shape[-1]).min(dim=0)[0]
    
    X_norm = (X_raw - X_min) / (X_max - X_min + 1e-12)
    Y_norm = (Y_raw - y_min) / (y_max - y_min + 1e-12)
    
    # Flatten windows
    xb = X_norm.reshape(-1, X_norm.shape[-1]).to(device)
    yb = Y_norm.reshape(-1, Y_norm.shape[-1]).to(device)
    
    X_max, X_min = X_max.to(device), X_min.to(device)
    y_max, y_min = y_max.to(device), y_min.to(device)

    # Instantiate Models
    config = cfg.PINN_ARCH_DEFAULT
    model_legacy = LegacyPINN(**config).to(device)
    model_current = ConfigurablePINN(**config).to(device)
    
    # Important: Start both from the SAME weights for a fair test
    # We'll just train them independently as they start from the same init logic
    
    loss_legacy_method = ReLoBRaLoLoss(enable_mass_constraints=True)
    loss_current_method = ReLoBRaLoLoss(enable_mass_constraints=True)
    
    opt_legacy = optim.Adam(model_legacy.parameters(), lr=1e-4)
    opt_current = optim.Adam(model_current.parameters(), lr=1e-4)
    
    history = {
        'legacy': {'total': [], 'data': [], 'phys': [], 'm1': [], 'm2': [], 'm3': []},
        'current': {'total': [], 'data': [], 'phys': [], 'm1': [], 'm2': [], 'm3': []}
    }
    
    print("Training models...")
    for epoch in range(epochs):
        # Legacy Step
        model_legacy.train()
        l_comps = loss_legacy_method(model_legacy, xb, yb, X_max, X_min, y_max, y_min)
        l_total = loss_legacy_method.step(l_comps, opt_legacy, model_legacy, xb, yb)
        
        # Current Step
        model_current.train()
        c_comps = loss_current_method(model_current, xb, yb, X_max, X_min, y_max, y_min)
        c_total = loss_current_method.step(c_comps, opt_current, model_current, xb, yb)
        
        # Log metrics
        if isinstance(l_total, torch.Tensor):
            history['legacy']['total'].append(l_total.item())
            history['legacy']['data'].append(l_comps[0].item())
            history['legacy']['phys'].append(sum(c.item() for c in l_comps[1:5]))
            history['legacy']['m1'].append(model_legacy.M1.item())
            history['legacy']['m2'].append(model_legacy.M2.item())
            history['legacy']['m3'].append(model_legacy.M3.item())
            
        if isinstance(c_total, torch.Tensor):
            history['current']['total'].append(c_total.item())
            history['current']['data'].append(c_comps[0].item())
            history['current']['phys'].append(sum(c.item() for c in c_comps[1:5]))
            history['current']['m1'].append(model_current.M1.item())
            history['current']['m2'].append(model_current.M2.item())
            history['current']['m3'].append(model_current.M3.item())
            
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Legacy Phys: {history['legacy']['phys'][-1]:.2f} | Current Phys: {history['current']['phys'][-1]:.4f}")

    # ── Plotting ──────────────────────────────────────────────────────────────
    comp_results_dir = os.path.join(cfg.RESULTS_DIR, "comparison")
    os.makedirs(comp_results_dir, exist_ok=True)
    
    # 1. Physical Residuals Comparison (Log scale)
    plt.figure(figsize=(10, 6))
    plt.plot(history['legacy']['phys'], label='Legacy (No Scaling)')
    plt.plot(history['current']['phys'], label='Current (Scaled /1e6)')
    plt.yscale('log')
    plt.title("Physics Residual RMSE: Legacy vs Current")
    plt.xlabel("Epoch")
    plt.ylabel("Loss Magnitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("results/comparison/physics_loss_comparison.png")
    
    # 2. Mass Parameter Evolution (Target Total: 22kg)
    plt.figure(figsize=(10, 6))
    leg_total = np.array(history['legacy']['m1']) + np.array(history['legacy']['m2']) + np.array(history['legacy']['m3'])
    cur_total = np.array(history['current']['m1']) + np.array(history['current']['m2']) + np.array(history['current']['m3'])
    plt.plot(leg_total, label='Legacy Total Mass', linestyle='--')
    plt.plot(cur_total, label='Current Total Mass', linewidth=2)
    plt.axhline(y=22.0, color='r', linestyle=':', label='Target (22kg)')
    plt.title("Total Mass Convergence (M1 + M2 + M3)")
    plt.xlabel("Epoch")
    plt.ylabel("Mass (kg)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(comp_results_dir, "mass_convergence.png"))
    
    print(f"\nComparison complete. Results saved to {comp_results_dir}/")
    print(f"Final Legacy Mass: {leg_total[-1]:.2f} kg")
    print(f"Final Current Mass: {cur_total[-1]:.2f} kg")

if __name__ == "__main__":
    run_comparison(epochs=400)
