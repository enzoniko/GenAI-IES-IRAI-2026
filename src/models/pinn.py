"""Physics-informed network for the mafaulda_synthetic rotor.

The physics residuals come from src/physics/rotor_model.py — the SAME
functions the simulator satisfies — so a healthy-trained PINN can reach
near-zero physics loss by construction (the property the real-data pipeline
never had).

Pointwise interface (mirrors the prior-work design):
  input  x_norm  (N, 10) standardized [vel x4, pos x4, omega, t]
  output acc_norm (N, 4) standardized accelerations
         forces   (N, 4) unmeasured corrective forces fA..fD [N] (physical)

Healthy training drives forces -> 0 (small-norm penalty) and residuals -> 0;
on faulted data the residuals/forces light up in the class-characteristic
pattern — the physics-informed embedding signal.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.physics import RotorParams, residuals as physics_residuals

ACT = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "gelu": nn.GELU, "selu": nn.SELU}


class ConfigurableMLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, activation="elu", dropout=0.0):
        super().__init__()
        layers, d = [], input_dim
        for h in hidden_layers:
            layers += [nn.Linear(d, h), ACT[activation]()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, output_dim))
        self.model = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.model(x)


class Normalization(nn.Module):
    """Standardization buffers for the 10 input features and 4 outputs."""

    def __init__(self, x_mean, x_std, y_mean, y_std):
        super().__init__()
        for name, val in [("x_mean", x_mean), ("x_std", x_std),
                          ("y_mean", y_mean), ("y_std", y_std)]:
            self.register_buffer(name, torch.as_tensor(val, dtype=torch.float64))

    @staticmethod
    def fit(X: torch.Tensor, Y: torch.Tensor) -> "Normalization":
        return Normalization(X.mean(0), X.std(0) + 1e-12, Y.mean(0), Y.std(0) + 1e-12)

    def norm_x(self, x):
        return (x - self.x_mean.to(x)) / self.x_std.to(x)

    def denorm_x(self, xn):
        return xn * self.x_std.to(xn) + self.x_mean.to(xn)

    def norm_y(self, y):
        return (y - self.y_mean.to(y)) / self.y_std.to(y)

    def denorm_y(self, yn):
        return yn * self.y_std.to(yn) + self.y_mean.to(yn)


class RotorPINN(nn.Module):
    def __init__(self, norm: Normalization, params: RotorParams | None = None,
                 hidden_layers=(128, 128), activation="elu", dropout=0.0,
                 force_scale: float = 1.0):
        super().__init__()
        self.norm = norm
        self.params = params or RotorParams()
        self.acc_net = ConfigurableMLP(10, list(hidden_layers), 4, activation, dropout)
        self.force_net = ConfigurableMLP(10, list(hidden_layers), 4, activation, dropout)
        self.force_scale = force_scale
        self.double()

    def forward(self, x_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x_norm (N, 10) -> (acc_norm (N,4), forces (N,4) in Newtons)."""
        return self.acc_net(x_norm), self.force_net(x_norm) * self.force_scale

    def compute_residuals(self, x_norm: torch.Tensor, acc_norm: torch.Tensor,
                          forces: torch.Tensor) -> torch.Tensor:
        """Healthy-physics residuals minus the corrective forces, (N, 4) [N]."""
        x = self.norm.denorm_x(x_norm)
        vel, pos, omega, t = x[:, 0:4], x[:, 4:8], x[:, 8], x[:, 9]
        acc = self.norm.denorm_y(acc_norm)
        r = physics_residuals(pos, vel, acc, omega, t, self.params)
        return r - forces

    def save(self, path, extra: dict | None = None):
        torch.save({"state_dict": self.state_dict(),
                    "norm": {k: getattr(self.norm, k) for k in
                             ("x_mean", "x_std", "y_mean", "y_std")},
                    "extra": extra or {}}, path)

    @staticmethod
    def load(path, params: RotorParams | None = None, **kw) -> "RotorPINN":
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        norm = Normalization(**{k: v for k, v in ckpt["norm"].items()})
        model = RotorPINN(norm, params=params, **kw)
        model.load_state_dict(ckpt["state_dict"])
        return model
