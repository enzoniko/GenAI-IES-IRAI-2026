import torch
import torch.nn as nn
import sys
import os
import src.configs as cfg

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
        
        # 2. Load Normalization Metadata & Pre-Trained PINN Weights
        norm_path = os.path.join(os.path.dirname(__file__), "../../results/normalization_metadata.pth")
        if os.path.exists(norm_path):
            metadata = torch.load(norm_path, map_location='cpu', weights_only=True)
            self.register_buffer('X_max', metadata['X_max'])
            self.register_buffer('X_min', metadata['X_min'])
            self.register_buffer('y_max', metadata['y_max'])
            self.register_buffer('y_min', metadata['y_min'])
            print(f"Successfully loaded normalization metadata from {norm_path}")
        else:
            print("WARNING: Normalization metadata not found. Defaulting to identity.")
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
                else:
                    self.pinn.load_state_dict(ckpt)  # Fallback for old checkpoints without norm bounds
                print("Successfully loaded prior PINN weights into the Oracle.")
            except RuntimeError:
                print("PINN weights found but shape mismatched. Using initialized weights.")
        else:
            print("PINN weights not found. Using randomly initialized physics for structural testing.")
            
        # 3. The Purely Mathematical, Differentiable Feature Extractor
        # The PINN outputs 4 unmeasured force parameters + 4 physics residuals = 8 channels.
        # MathFeatureExtractor computes deterministic statistical, spectral, and wavelet features.
        # No learnable weights — fully deterministic and differentiable.
        self.feature_extractor = MathFeatureExtractor(in_channels=8)
        self.embed_dim = self.feature_extractor.output_dim  # 2240 (9 stats + 256 FFT + 15 Wavelet per channel)

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
    def forward(self, x, omega=None):
        """
        Extracts physics-informed features from one full-rotation window of accelerometer data.

        Input:  (Batch, 4, L) — 4-channel normalized acceleration trace in [0, 1].
                  L = window_size = fs / rotation_Hz.
                omega: (Batch, 1) or float — rotational speed in rad/s. 
                       If None, uses self.dynamic_omega.

        Output: (Batch, embed_dim) — 2240 features.
        """
        B, C, L = x.shape
        
        # 0. Restore physical scale via precise Min-Max denormalization
        # Input x is assumed to be normalized exactly как the dataset targets.
        y_min_exp = self.y_min.view(1, 4, 1)
        y_max_exp = self.y_max.view(1, 4, 1)
        physical_acc = x * (y_max_exp - y_min_exp + 1e-12) + y_min_exp
        
        # 1. Differentiable Kinematics
        vel, pos = self.differentiable_integration(physical_acc)
        
        # 2. Prepare 10-feature input for PINN (flattening time into the batch dimension)
        # The PINN expects inputs as (B*L, 10)
        vel_flat = vel.transpose(1, 2).reshape(B * L, 4)
        pos_flat = pos.transpose(1, 2).reshape(B * L, 4)
        
        # Omega (Speed) and Time grids
        effective_omega = omega if omega is not None else self.dynamic_omega
        if effective_omega is not None:
            if isinstance(effective_omega, (float, int)):
                effective_omega = torch.full((B, 1), effective_omega, device=x.device, dtype=x.dtype)
            omega_2d = effective_omega.view(-1, 1).expand(B, 1)
            omega_feed = omega_2d.expand(B, L).reshape(B * L, 1).to(dtype=x.dtype)
        else:
            raise ValueError("Oracle forward requires omega to be passed or set via set_dynamic_omega.")
            
        time_steps = torch.arange(L, device=x.device, dtype=x.dtype) * self.dt
        time_grid = time_steps.unsqueeze(0).expand(B, L).reshape(B * L, 1)
        
        pinn_input = torch.cat([vel_flat, pos_flat, omega_feed, time_grid], dim=-1)
        
        # 3. Extract Physics Features
        # The PINN expects inputs as (B*L, 10) in [0, 1] range.
        pinn_input_norm = (pinn_input - self.X_min) / (self.X_max - self.X_min + 1e-12)
        
        # Ensure double precision for PINN logic
        pinn_input_norm = pinn_input_norm.double()
        
        # Forward pass: obtain accelerations and implicit unmeasured parameters
        pred_acc_norm = self.pinn(pinn_input_norm).float() 
        unmeasured = torch.cat([self.pinn.fA, self.pinn.fB, self.pinn.fC, self.pinn.fD], dim=-1).float()
        
        # Compute residuals (pass bounds for internal denormalization)
        res1, res2, res3, res4, _, _ = self.pinn.compute_residuals(
            pinn_input_norm, 
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
