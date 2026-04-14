import torch
import torch.nn as nn
import sys
import os
import src.configs as cfg
import torchaudio.functional as F
from scipy.signal import butter

from .pinn import ConfigurablePINN
from .feature_extractors import MathFeatureExtractor


class PriorWorkOracle(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        # Sampling rate for MaFaulDa from centralized config. Used to convert samples -> time for kinematics.
        self.dt = 1.0 / cfg.SAMPLING_RATE
        # Note: seq_len is NOT stored — the data pipeline uses full-rotation windowing
        # (window_size = fs / rotation_hz), so L varies per batch. All ops here are length-agnostic.
        
        # 1. Instantiate the Physics-Informed Neural Network (PINN)
        pinn_config = cfg.PINN_ARCH_DEFAULT
        self.pinn = ConfigurablePINN(
            unmeasured_net_config=pinn_config['unmeasured_net_config'],
            acceleration_net_config=pinn_config['acceleration_net_config'],
            param_init_config=pinn_config['param_init_config'],
            enable_mass_constraints=True
        )
        
        # 2. Load Normalization Metadata & Pre-Trained PINN Weights
        norm_path = cfg.NORM_METADATA_PATH
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

        weight_path = cfg.PINN_MODEL_PATH
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
        # Replace deprecated class-specific buffers with a generic multi-class buffer
        self.register_buffer('target_distributions', torch.zeros(cfg.NUM_CLASSES, self.embed_dim))
        

    def _get_filter_coefs(self, device):
        """Get Butterworth coefficients as Torch tensors in float64."""
        # Note: We compute these on CPU once then move to device.
        # fs and cutoff are from configs.
        b, a = butter(4, cfg.CUTOFF_HZ, btype='high', fs=cfg.SAMPLING_RATE)
        return torch.from_numpy(b).to(device).double(), torch.from_numpy(a).to(device).double()

    def _torch_filtfilt(self, x: torch.Tensor, b: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Differentiable zero-phase IIR filtering (forward-backward)."""
        # Forward pass through IIR recursive filter
        y = F.lfilter(x, a, b, clamp=False)
        # Flip, filter again, and flip back to ensure zero phase delay 
        y = torch.flip(y, dims=[-1])
        y = F.lfilter(y, a, b, clamp=False)
        y = torch.flip(y, dims=[-1])
        return y

    def _torch_detrend(self, y: torch.Tensor) -> torch.Tensor:
        """Differentiable linear detrender using Least Squares."""
        n = y.shape[-1]
        # x coordinates normalized to [0,1]
        x = torch.linspace(0, 1, n, device=y.device, dtype=y.dtype)
        # Linear model matrix: [x, 1]
        A = torch.stack([x, torch.ones_like(x)], dim=-1) 
        # Expand A to match the leading batch dimensions of y
        A_expanded = A.expand(*y.shape[:-1], n, 2)
        # Least squares solve
        res = torch.linalg.lstsq(A_expanded, y.unsqueeze(-1))
        slope, intercept = res.solution[..., 0, :], res.solution[..., 1, :]
        trend = (slope * x + intercept).squeeze(-1)
        return y - trend

    def _torch_integrate(self, y: torch.Tensor) -> torch.Tensor:
        """Differentiable trapezoidal integration."""
        # Standard trapezoidal rule: area[i] = (y[i] + y[i-1])/2 * dt
        y_mid = 0.5 * (y[..., 1:] + y[..., :-1])
        integral = torch.cumsum(y_mid * self.dt, dim=-1)
        # Pad leading zero to match initial sample time t=0
        z = torch.zeros((*y.shape[:-1], 1), device=y.device, dtype=y.dtype)
        return torch.cat([z, integral], dim=-1)

    def differentiable_integration(self, acc):
        """
        Replaces simple cumsum with a robust Strategy D implementation (detrend + double filter).
        Allows gradients to flow backward from Velocity/Position into the generated Acceleration.
        """
        # USE DOUBLE PRECISION (float64) for internal processing of IIR filters
        orig_dtype = acc.dtype
        acc_db = acc.double()
        
        # 1. CONDITIONING (Zero-mean + Detrend)
        acc_db = acc_db - acc_db.mean(dim=-1, keepdim=True)
        acc_db = self._torch_detrend(acc_db)
        
        # 2. ACCELERATION FILTERING
        b_coef, a_coef = self._get_filter_coefs(acc.device)
        acc_filt = self._torch_filtfilt(acc_db, b_coef, a_coef)
        
        # 3. INTEGRATION -> VELOCITY
        vel = self._torch_integrate(acc_filt)
        
        # 4. VELOCITY FILTERING (The critical Strategy D 'Second Filter')
        # Integration generates a cumulative error curve; this high-pass 
        # filter 're-centers' velocity around zero.
        vel_filt = self._torch_filtfilt(vel, b_coef, a_coef)
        
        # 5. INTEGRATION -> POSITION
        pos = self._torch_integrate(vel_filt)
        
        # Cast back to original precision (float32) for model compatibility
        return vel_filt.to(orig_dtype), pos.to(orig_dtype)

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
        if omega is not None:
            if isinstance(omega, (float, int)):
                omega = torch.full((B, 1), omega, device=x.device, dtype=x.dtype)
            omega_2d = omega.view(-1, 1).expand(B, 1)
            omega_feed = omega_2d.expand(B, L).reshape(B * L, 1).to(dtype=x.dtype)
        else:
            raise ValueError("Reactive Physics Error: Oracle forward requires a measured 'omega' parameter. Hardcoded or forecasted physics are not allowed.")
            
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
        
    def set_target_distribution(self, class_idx, embedding):
        """Stores the target physical embedding for a specific fault class."""
        if 0 <= class_idx < self.target_distributions.shape[0]:
            self.target_distributions[class_idx].copy_(embedding)
        else:
            raise ValueError(f"Invalid class index {class_idx}. Maximum supported class index is {self.target_distributions.shape[0]-1}")

    def get_target_distribution(self, class_idx):
        """Retrieves the target physical embedding for a specific fault class."""
        if 0 <= class_idx < self.target_distributions.shape[0]:
            return self.target_distributions[class_idx]
        else:
            raise ValueError(f"Invalid class index {class_idx}. Maximum supported class index is {self.target_distributions.shape[0]-1}")
