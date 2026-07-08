"""GuidanceDensity: the estimator family at the heart of hypothesis H
(Block B). Every estimator exposes the same API and a differentiable
penalty so any backbone can consume any density:

    fit(V)            V (n, d) torch, real fault features in PCA space
    log_prob(v)       (B,) differentiable
    penalty(v)        scalar = -mean log_prob (up to constants)
    held_out_ll(V)    mean log-likelihood on held-out samples (B3 confound)

Estimator bias vs variance in the few-shot regime is exactly what B3
measures — the Flow refuses to fit below `min_fit_samples` and records a
Gaussian fallback instead of silently producing garbage.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from sklearn.covariance import LedoitWolf
from sklearn.mixture import GaussianMixture


class GaussianGuidance(nn.Module):
    """Single Gaussian with Ledoit-Wolf shrinkage — the Mahalanobis penalty
    the withdrawn paper CLAIMED to use (the code used MSE-to-mean)."""

    kind = "gaussian"

    def __init__(self):
        super().__init__()
        self.fitted = False

    def fit(self, V: torch.Tensor, seed: int = 0):
        Vn = V.detach().cpu().double().numpy()
        mu = Vn.mean(axis=0)
        if Vn.shape[0] >= 2:
            cov = LedoitWolf().fit(Vn).covariance_
        else:
            cov = np.eye(Vn.shape[1])
        cov = cov + 1e-6 * np.eye(cov.shape[0])
        chol = np.linalg.cholesky(np.linalg.inv(cov))
        logdet = float(np.linalg.slogdet(cov)[1])
        self.register_buffer("mu", torch.from_numpy(mu))
        self.register_buffer("prec_chol", torch.from_numpy(chol))
        self.d = Vn.shape[1]
        self.log_norm = -0.5 * (self.d * math.log(2 * math.pi) + logdet)
        self.fitted = True
        return self

    def log_prob(self, v: torch.Tensor) -> torch.Tensor:
        diff = (v.double() - self.mu.to(v.device))
        m = (diff @ self.prec_chol.to(v.device)).pow(2).sum(dim=-1)
        return self.log_norm - 0.5 * m

    def penalty(self, v: torch.Tensor) -> torch.Tensor:
        return -self.log_prob(v).mean()

    def held_out_ll(self, V: torch.Tensor) -> float:
        with torch.no_grad():
            return self.log_prob(V).mean().item()


class GMMGuidance(nn.Module):
    kind = "gmm"

    def __init__(self, max_k: int = 4):
        super().__init__()
        self.max_k = max_k
        self.fitted = False
        self.chosen_k = None

    def fit(self, V: torch.Tensor, seed: int = 0):
        Vn = V.detach().cpu().double().numpy()
        n = Vn.shape[0]
        k_cap = max(1, min(self.max_k, n // 8))
        best, best_bic = None, np.inf
        for k in range(1, k_cap + 1):
            gm = GaussianMixture(n_components=k, covariance_type="full",
                                 reg_covar=1e-4, random_state=seed, n_init=2).fit(Vn)
            bic = gm.bic(Vn)
            if bic < best_bic:
                best, best_bic = gm, bic
        self.chosen_k = best.n_components
        self.register_buffer("weights", torch.from_numpy(best.weights_))
        self.register_buffer("means", torch.from_numpy(best.means_))
        precs = best.precisions_cholesky_                     # (k, d, d)
        self.register_buffer("prec_chol", torch.from_numpy(precs))
        d = Vn.shape[1]
        logdets = np.array([np.sum(np.log(np.diag(precs[i]))) for i in range(len(precs))])
        self.register_buffer("logdets", torch.from_numpy(logdets))
        self.d = d
        self.fitted = True
        return self

    def log_prob(self, v: torch.Tensor) -> torch.Tensor:
        v = v.double()
        dev = v.device
        diff = v.unsqueeze(1) - self.means.to(dev).unsqueeze(0)        # (B, k, d)
        y = torch.einsum("bkd,kde->bke", diff, self.prec_chol.to(dev))
        m = y.pow(2).sum(dim=-1)                                       # (B, k)
        log_comp = (-0.5 * (self.d * math.log(2 * math.pi) + m)
                    + self.logdets.to(dev).unsqueeze(0)
                    + torch.log(self.weights.to(dev)).unsqueeze(0))
        return torch.logsumexp(log_comp, dim=1)

    def penalty(self, v: torch.Tensor) -> torch.Tensor:
        return -self.log_prob(v).mean()

    def held_out_ll(self, V: torch.Tensor) -> float:
        with torch.no_grad():
            return self.log_prob(V).mean().item()


class _Coupling(nn.Module):
    def __init__(self, d, hidden, flip):
        super().__init__()
        self.d_a = d // 2 if not flip else d - d // 2
        self.d_b = d - self.d_a
        self.flip = flip
        self.net = nn.Sequential(
            nn.Linear(self.d_a, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * self.d_b),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def inverse(self, x):
        """x -> (u, logdet) for density evaluation."""
        xa, xb = (x[:, :self.d_a], x[:, self.d_a:]) if not self.flip else \
                 (x[:, self.d_b:], x[:, :self.d_b])
        st = self.net(xa)
        s, t = st.chunk(2, dim=1)
        s = torch.tanh(s) * 2.0
        ub = (xb - t) * torch.exp(-s)
        u = torch.cat([xa, ub], dim=1) if not self.flip else torch.cat([ub, xa], dim=1)
        return u, -s.sum(dim=1)


class FlowGuidance(nn.Module):
    """RealNVP density in the PCA space. Falls back to Gaussian below
    `min_fit_samples` (recorded in .fallback, surfaced by B3)."""

    kind = "flow"

    def __init__(self, layers: int = 5, hidden: int = 64, epochs: int = 300,
                 lr: float = 1e-3, min_fit_samples: int = 20):
        super().__init__()
        self.layers_n, self.hidden = layers, hidden
        self.epochs, self.lr = epochs, lr
        self.min_fit_samples = min_fit_samples
        self.fallback: GaussianGuidance | None = None
        self.fitted = False

    def _build(self, d):
        self.couplings = nn.ModuleList(
            [_Coupling(d, self.hidden, flip=(i % 2 == 1)) for i in range(self.layers_n)])
        self.register_buffer("mu", torch.zeros(d, dtype=torch.float64))
        self.register_buffer("sd", torch.ones(d, dtype=torch.float64))
        self.d = d

    def fit(self, V: torch.Tensor, seed: int = 0):
        torch.manual_seed(seed)
        V = V.detach().double()
        n, d = V.shape
        if n < self.min_fit_samples:
            self.fallback = GaussianGuidance().fit(V, seed)
            self.fitted = True
            return self
        self._build(d)
        self.double()
        self.mu.copy_(V.mean(0))
        self.sd.copy_(V.std(0) + 1e-6)
        n_val = max(2, int(0.15 * n))
        perm = torch.randperm(n)
        Vtr, Vva = V[perm[n_val:]], V[perm[:n_val]]
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
        best, best_state, patience = -np.inf, None, 0
        for ep in range(self.epochs):
            opt.zero_grad()
            loss = -self._log_prob_flow(Vtr).mean()
            loss.backward()
            opt.step()
            with torch.no_grad():
                vll = self._log_prob_flow(Vva).mean().item()
            if vll > best + 1e-4:
                best, patience = vll, 0
                best_state = {k: v.clone() for k, v in self.state_dict().items()}
            else:
                patience += 1
                if patience > 30:
                    break
        if best_state is not None:
            self.load_state_dict(best_state)
        self.fitted = True
        return self

    def _log_prob_flow(self, x):
        u = (x.double() - self.mu.to(x.device)) / self.sd.to(x.device)
        logdet = -torch.log(self.sd.to(x.device)).sum().expand(x.shape[0]).clone()
        for c in self.couplings:
            u, ld = c.inverse(u)
            logdet = logdet + ld
        base = -0.5 * (u.pow(2) + math.log(2 * math.pi)).sum(dim=1)
        return base + logdet

    def log_prob(self, v: torch.Tensor) -> torch.Tensor:
        if self.fallback is not None:
            return self.fallback.log_prob(v)
        return self._log_prob_flow(v)

    def penalty(self, v: torch.Tensor) -> torch.Tensor:
        return -self.log_prob(v).mean()

    def held_out_ll(self, V: torch.Tensor) -> float:
        with torch.no_grad():
            return self.log_prob(V).mean().item()


def make_guidance(kind: str, cfg) -> nn.Module:
    if kind == "gaussian":
        return GaussianGuidance()
    if kind == "gmm":
        return GMMGuidance(max_k=cfg.gmm_max_k)
    if kind == "flow":
        return FlowGuidance(layers=cfg.flow_layers, hidden=cfg.flow_hidden,
                            epochs=cfg.flow_epochs, lr=cfg.flow_lr)
    raise ValueError(f"Unknown guidance kind '{kind}'")
