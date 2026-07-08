"""PhysicsEmbedding: frozen PINN -> differentiable feature extractor -> PCA.

The physics-informed embedding of a signal window. Fully differentiable so
VJP gradients can flow back to the generator's latent.

Channels fed to the feature extractor (mode='pinn', 12 channels):
  fA..fD      (4)  unmeasured corrective forces
  r1..r4      (4)  healthy-equation residuals evaluated on the MEASURED
                   (or generated) kinematics — the direct violation signal
  d1..d4      (4)  data residual: PINN-predicted minus measured acceleration

mode='raw' (DIAGNOSTIC ONLY — the legacy bypass that produced the withdrawn
paper's headline numbers) feeds the raw 4-channel signal directly. Every
artifact produced under mode='raw' is tagged bypass=true in run manifests.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.data.kinematics import derive_kinematics
from src.models import MathFeatureExtractor, RotorPINN


class PCAProjector(nn.Module):
    """Linear whitening PCA — differentiable, fitted once on train features."""

    def __init__(self, mean: torch.Tensor, components: torch.Tensor, scale: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("components", components)   # (F, d)
        self.register_buffer("scale", scale)              # (d,)

    @staticmethod
    def fit(V: torch.Tensor, dim: int) -> "PCAProjector":
        V = V.double()
        mean = V.mean(dim=0)
        Vc = V - mean
        # economy SVD; F may be large but N is moderate
        U, S, Vt = torch.linalg.svd(Vc, full_matrices=False)
        d = min(dim, S.shape[0])
        comps = Vt[:d].T.contiguous()                      # (F, d)
        scale = S[:d] / (max(V.shape[0] - 1, 1)) ** 0.5    # per-component std
        return PCAProjector(mean, comps, scale.clamp(min=1e-9))

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        return ((v.double() - self.mean) @ self.components) / self.scale


class PhysicsEmbedding(nn.Module):
    def __init__(self, pinn: RotorPINN | None, mode: str, fs: int,
                 signal_mean: torch.Tensor, signal_std: torch.Tensor,
                 fft_bins: int = 256, wavelet_levels: int = 4):
        super().__init__()
        if mode not in ("pinn", "raw"):
            raise ValueError(f"Unsupported oracle mode '{mode}'")
        if mode == "pinn" and pinn is None:
            raise ValueError("mode='pinn' requires a trained RotorPINN")
        self.mode = mode
        self.fs = fs
        self.pinn = pinn
        if pinn is not None:
            for p in self.pinn.parameters():
                p.requires_grad = False
        ch = 12 if mode == "pinn" else 4
        self.extractor = MathFeatureExtractor(in_channels=ch, fft_bins=fft_bins,
                                              wavelet_levels=wavelet_levels)
        self.register_buffer("sig_mean", signal_mean.view(1, 4, 1))
        self.register_buffer("sig_std", signal_std.view(1, 4, 1))
        self.pca: PCAProjector | None = None
        self.output_dim = self.extractor.output_dim

    # -- feature map ---------------------------------------------------------
    def forward(self, x_norm: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        """x_norm (B, 4, T) in model units; omega (B,) rad/s -> (B, F)."""
        x_phys = x_norm * self.sig_std.to(x_norm) + self.sig_mean.to(x_norm)
        if self.mode == "raw":
            return self.extractor(x_phys.permute(0, 2, 1))
        B, C, T = x_phys.shape
        acc = x_phys.double()
        vel, pos = derive_kinematics(acc, fs=self.fs, omega=omega)
        t = torch.arange(T, device=acc.device, dtype=torch.float64) / self.fs
        om = omega.double().view(B, 1).expand(B, T)
        X = torch.cat([
            vel.permute(0, 2, 1).reshape(B * T, 4),
            pos.permute(0, 2, 1).reshape(B * T, 4),
            om.reshape(B * T, 1), t.expand(B, T).reshape(B * T, 1),
        ], dim=1)
        x_in = self.pinn.norm.norm_x(X)
        acc_pred_n, forces = self.pinn(x_in)
        acc_meas = acc.permute(0, 2, 1).reshape(B * T, 4)
        acc_meas_n = self.pinn.norm.norm_y(acc_meas)
        phys_res = self.pinn.compute_residuals(x_in, acc_meas_n, forces)
        data_res = self.pinn.norm.denorm_y(acc_pred_n) - acc_meas
        feats = torch.cat([forces, phys_res, data_res], dim=1).view(B, T, 12)
        return self.extractor(feats)

    # -- PCA -----------------------------------------------------------------
    def fit_pca(self, V: torch.Tensor, dim: int) -> None:
        self.pca = PCAProjector.fit(V, dim)

    def project(self, v: torch.Tensor) -> torch.Tensor:
        assert self.pca is not None, "call fit_pca first"
        return self.pca(v)

    def embed(self, x_norm: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        """Full differentiable chain: signal -> Phi -> PCA space."""
        return self.project(self.forward(x_norm, omega))

    # -- persistence ---------------------------------------------------------
    def save(self, path):
        torch.save({
            "mode": self.mode, "fs": self.fs,
            "sig_mean": self.sig_mean, "sig_std": self.sig_std,
            "fft_bins": self.extractor.num_freq_bins,
            "wavelet_levels": self.extractor.num_wavelet_levels,
            "pca": None if self.pca is None else {
                "mean": self.pca.mean, "components": self.pca.components,
                "scale": self.pca.scale},
        }, path)

    @staticmethod
    def load(path, pinn: RotorPINN | None) -> "PhysicsEmbedding":
        d = torch.load(path, map_location="cpu", weights_only=True)
        emb = PhysicsEmbedding(pinn, d["mode"], d["fs"],
                               d["sig_mean"].flatten(), d["sig_std"].flatten(),
                               d["fft_bins"], d["wavelet_levels"])
        if d["pca"] is not None:
            emb.pca = PCAProjector(**d["pca"])
        return emb
