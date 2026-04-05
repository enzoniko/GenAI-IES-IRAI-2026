import torch
import torch.nn as nn
import sys
import os

from .pinn import ConfigurablePINN, get_default_pinn_config


class PriorWorkOracle(nn.Module):
    def __init__(self, in_channels=4, seq_len=5000, embed_dim=64):
        super().__init__()
        self.seq_len = seq_len
        self.dt = 1.0 / 50000.0  # MaFaulDa 50kHz sampling rate
        
        # 1. Instantiate the Prior Work PINN
        pinn_config = get_default_pinn_config()
        self.pinn = ConfigurablePINN(
            unmeasured_net_config=pinn_config['unmeasured_net_config'],
            acceleration_net_config=pinn_config['acceleration_net_config'],
            param_init_config=pinn_config['param_init_config'],
            enable_mass_constraints=True
        )
        
        # 2. Load the Weights (if available)
        # If the file doesn't exist, it safely continues using random weights for structural testing
        
        # Buffers for Phase 0 Min-Max Normalization bounds
        self.register_buffer('X_max', torch.ones(10))
        self.register_buffer('X_min', torch.zeros(10))
        self.register_buffer('y_max', torch.ones(4))
        self.register_buffer('y_min', torch.zeros(4))

        weight_path = os.path.join(os.path.dirname(__file__), "../../results/relobralo_model.pth")
        if os.path.exists(weight_path):
            try:
                ckpt = torch.load(weight_path, map_location='cpu', weights_only=True)
                if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                    self.pinn.load_state_dict(ckpt['model_state_dict'])
                    self.X_max.copy_(ckpt['X_max'])
                    self.X_min.copy_(ckpt['X_min'])
                    self.y_max.copy_(ckpt['y_max'])
                    self.y_min.copy_(ckpt['y_min'])
                else:
                    self.pinn.load_state_dict(ckpt)  # Fallback for old weights
                print("Successfully loaded prior PINN weights into the Oracle.")
            except RuntimeError:
                print("PINN weights found but shape mismatched. Using initialized weights.")
        else:
            print("PINN weights not found. Using randomly initialized physics for structural testing.")
            
        # 3. The Feature Extractor Projection
        # The PINN evaluates 4 residual equations + 4 unmeasured parameters = 8 raw physics features.
        # We project this 8D physics space into the LDM's requested embed_dim (e.g., 64).
        self.physics_to_latent = nn.Sequential(
            nn.Linear(8, 32),
            nn.GELU(),
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

    def set_dynamic_omega(self, omega: torch.Tensor):
        """Allows the diffusion loop to set the expected physical rotational speed."""
        self.dynamic_omega = omega

    def differentiable_integration(self, acc):
        """
        Replaces LoadDatav3.py SciPy math with Differentiable PyTorch operations.
        Allows gradients to flow backward from Velocity/Position into the generated Acceleration.
        """
        # Ensure zero-mean to prevent catastrophic integration drift
        acc = acc - acc.mean(dim=-1, keepdim=True)
        vel = torch.cumsum(acc, dim=-1) * self.dt
        vel = vel - vel.mean(dim=-1, keepdim=True)
        pos = torch.cumsum(vel, dim=-1) * self.dt
        return vel, pos

    def forward(self, x):
        """
        Input: (Batch, 4, 5000) physical raw trace
        Output: (Batch, 64) embedding
        """
        B, C, L = x.shape
        
        # 0. Restore physical scale from dataset soft-scaling (/ 20.0)
        physical_acc = x * 20.0
        
        # 1. Differentiable Kinematics
        vel, pos = self.differentiable_integration(physical_acc)
        
        # 2. Prepare 10-feature input for PINN (flattening time into the batch dimension)
        # The PINN expects inputs as (B*L, 10)
        vel_flat = vel.transpose(1, 2).reshape(B * L, 4)
        pos_flat = pos.transpose(1, 2).reshape(B * L, 4)
        
        # Omega (Speed) and Time grids
        if self.dynamic_omega is not None:
            omega_2d = self.dynamic_omega.view(-1, 1).expand(B, 1)
            omega = omega_2d.expand(B, L).reshape(B * L, 1).to(dtype=x.dtype)
        else:
            omega = torch.full((B * L, 1), 20.0 * 2 * 3.14159, device=x.device, dtype=x.dtype)
            
        time_steps = torch.arange(L, device=x.device, dtype=x.dtype) * self.dt
        time_grid = time_steps.unsqueeze(0).expand(B, L).reshape(B * L, 1)
        
        pinn_input = torch.cat([vel_flat, pos_flat, omega, time_grid], dim=-1)
        
        # 3. Min-Max Normalization (PINN expects [0,1] features)
        pinn_input_norm = (pinn_input - self.X_min) / (self.X_max - self.X_min + 1e-12)
        
        # 4. Extract Physics Features
        # ConfigurablePINN internally casts to double, so we temporarily cast our inputs
        pred_acc_norm = self.pinn(pinn_input_norm.double()).float() 
        unmeasured = torch.cat([self.pinn.fA, self.pinn.fB, self.pinn.fC, self.pinn.fD], dim=-1).float()
        
        # Compute residuals (pass bounds so PINN can denormalize internally for physical equations)
        res1, res2, res3, res4, _, _ = self.pinn.compute_residuals(
            pinn_input_norm.double(), 
            pred_acc_norm.double(),
            X_max=self.X_max, X_min=self.X_min,
            y_max=self.y_max, y_min=self.y_min
        )
        residuals = torch.cat([res1, res2, res3, res4], dim=-1).float()
        
        # 5. Form the final embedding
        physics_features = torch.cat([unmeasured, residuals], dim=-1) # Shape: (B*L, 8)
        physics_features = physics_features.view(B, L, 8).mean(dim=1) # Pool over time -> (B, 8)
        
        return self.physics_to_latent(physics_features)
        
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
