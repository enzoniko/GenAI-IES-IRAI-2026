import torch
import torch.nn as nn
import sys
import os
import src.constants as c

from .pinn import ConfigurablePINN, get_default_pinn_config
from .feature_extractors import MathFeatureExtractor


class PriorWorkOracle(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        # Sampling rate for MaFaulDa (50 kHz). Used to convert samples -> time for kinematics.
        self.dt = 1.0 / 50000.0
        # Note: seq_len is NOT stored — the data pipeline uses full-rotation windowing
        # (window_size = fs / rotation_hz), so L varies per batch. All ops here are length-agnostic.
        
        # 1. Instantiate the Physics-Informed Neural Network (PINN)
        pinn_config = get_default_pinn_config()
        self.pinn = ConfigurablePINN(
            unmeasured_net_config=pinn_config['unmeasured_net_config'],
            acceleration_net_config=pinn_config['acceleration_net_config'],
            param_init_config=pinn_config['param_init_config'],
            enable_mass_constraints=True
        )
        
        # 2. Load Pre-Trained PINN Weights (if available)
        # If the checkpoint file doesn't exist, the oracle still runs with random weights
        # (useful for structural/pipeline testing before Phase 0 is completed).
        
        # Normalization bounds — filled from the checkpoint saved at the end of Phase 0
        self.register_buffer('X_max', torch.ones(10))
        self.register_buffer('X_min', torch.zeros(10))
        self.register_buffer('y_max', torch.ones(4))
        self.register_buffer('y_min', torch.zeros(4))

        weight_path = os.path.join(os.path.dirname(__file__), "../../results/pinn.pth")
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
                    self.pinn.load_state_dict(ckpt)  # Fallback for old checkpoints without norm bounds
                print("Successfully loaded prior PINN weights into the Oracle.")
            except RuntimeError:
                print("PINN weights found but shape mismatched. Using initialized weights.")
        else:
            print("PINN weights not found. Using randomly initialized physics for structural testing.")
            
        # 3. The Purely Mathematical, Differentiable Feature Extractor
        # The PINN outputs 4 unmeasured force parameters + 4 physics residuals = 8 channels.
        # MathFeatureExtractor computes time-domain statistics and FFT magnitudes from these
        # 8 channels, producing a 2120-dim vector (8 * (9 time-stats + 256 FFT bins)).
        # No learnable weights — fully deterministic and differentiable.
        self.feature_extractor = MathFeatureExtractor(in_channels=8)
        self.embed_dim = self.feature_extractor.output_dim  # 2120

        # Freeze all parameters — the Oracle is a fixed physics evaluation tool,
        # not something that should be updated during Phase 2 training.
        for param in self.parameters():
            param.requires_grad = False
            
        # Empirical reference embeddings for each fault class, used to compute the
        # SDEdit guidance penalty (MSE between generated and target physics embeddings).
        # 1 = Imbalance, 2 = Outer-Race
        self.register_buffer('dist_class_1', torch.zeros(self.embed_dim))
        self.register_buffer('dist_class_2', torch.zeros(self.embed_dim))
        
        # Rotational speed set externally by the diffusion loop (rad/s)
        self.dynamic_omega = None

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

    # IS THIS METHOD USED TO EXTRACT THE FEATURES?
    def forward(self, x):
        """
        Extracts physics-informed features from one full-rotation window of accelerometer data.

        Input:  (Batch, 4, L) — 4-channel raw acceleration trace, where
                  L = window_size = fs / rotation_Hz  (varies per rotation speed!).
                  For example: L ≈ 2008 at 24.9 Hz, L = 1000 at 50 Hz.

        Output: (Batch, embed_dim)  — embedding vector whose dimension depends on extractor_type:
                  'conv1d' / 'fft' => fixed embed_dim (e.g. 64 or 1024)
                  'math'           => 2120  (8 channels × (9 time-stats + 256 FFT bins))
        """
        B, C, L = x.shape
        
        # 0. Restore physical scale from dataset soft-scaling
        physical_acc = x * c.PHYSICAL_SOFT_SCALE
        
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

        # Reshape to true sequence structure for deep extraction
        # Result Shape: (Batch, Sequence Length, Channels)
        physics_features = physics_features.view(B, L, 8)
        
        # Extract features differentiably across the time dimension!
        return self.feature_extractor(physics_features)
        
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
