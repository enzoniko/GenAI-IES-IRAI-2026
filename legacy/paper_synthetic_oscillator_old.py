"""
Paper-Aligned 1D Damped Oscillator Synthetic Generator
=======================================================
Implements the synthetic benchmark from the paper's Eq. 4 (Section IV-A):

    m*x''(t) + c*x'(t) + k*x(t) = A*sin(ω₀*t)

Healthy parameters:  m=1, c=0.5, k=4.0, A=1.0, ω₀=1.0 rad/s
Three fault classes (with known parameter deviations):
  0 – Healthy       m=1, c=0.5, k=4.0, A=1.0  (baseline)
  1 – Stiffness ↓   k→0.5k = 2.0              (crack analog)
  2 – Damping ↑     c→2c   = 1.0              (lubrication-degradation analog)
  3 – Forcing ↑     A→2A   = 2.0              (rotor-imbalance analog)

Observable = acceleration x''(t) + Gaussian sensor noise.

4-channel pipeline compatibility:
  Y tensor: (N, T, 4)  – four acceleration channels with small independent noise;
                         the first channel is the "true" acceleration, channels 2-4
                         are independent realisations with small perturbations.
  X tensor: (N, T, 10) – [vel_ch1..ch4, pos_ch1..ch4, omega, time]
                         where omega = ω₀ = 1.0 rad/s (constant rotation analog).

RK45 integration via scipy.integrate.solve_ivp.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from scipy.integrate import solve_ivp

import src.configs as cfg

# ── Physical parameters (paper Eq. 4) ────────────────────────────────────────
M        = 1.0    # kg
C_BASE   = 0.5    # N·s/m   (healthy damping)
K_BASE   = 4.0    # N/m     (healthy stiffness)
A_BASE   = 1.0    # N       (healthy forcing amplitude)
OMEGA_0  = 1.0    # rad/s   (driving frequency — constant rotation analog)

# ── Fault class parameter maps ────────────────────────────────────────────────
FAULT_PARAMS = {
    0: dict(m=M, c=C_BASE,    k=K_BASE,      A=A_BASE,    omega0=OMEGA_0),  # Healthy
    1: dict(m=M, c=C_BASE,    k=0.5*K_BASE,  A=A_BASE,    omega0=OMEGA_0),  # Stiffness ↓
    2: dict(m=M, c=2*C_BASE,  k=K_BASE,      A=A_BASE,    omega0=OMEGA_0),  # Damping ↑
    3: dict(m=M, c=C_BASE,    k=K_BASE,      A=2*A_BASE,  omega0=OMEGA_0),  # Forcing ↑
}

CLASS_NAMES = {
    0: "healthy",
    1: "stiffness_reduction",
    2: "damping_increase",
    3: "forcing_perturbation",
}

# ── Generation constants ──────────────────────────────────────────────────────
SEQ_LENGTH         = cfg.SEQ_LENGTH          # 3014
TARGET_HZ          = cfg.TARGET_HZ           # 16.0 Hz
NOISE_STD          = 0.01                    # sensor noise on observable (x'')
CHANNEL_NOISE_STD  = 0.005                   # inter-channel decorrelation noise
SAMPLES_PER_CLASS  = 60                      # 50 train + 10 test per class
N_CHANNELS         = 4


# ── ODE definition ────────────────────────────────────────────────────────────
def _ode_rhs(t, y, m, c, k, A, omega0):
    """RHS of 1D damped harmonic oscillator.

    dy/dt = [x', x''] where  x'' = (A*sin(omega0*t) - c*x' - k*x) / m
    """
    x, xdot = y
    xddot = (A * np.sin(omega0 * t) - c * xdot - k * x) / m
    return [xdot, xddot]


def _solve_window(fault_class: int, window_idx: int) -> tuple:
    """Numerically integrate the ODE for a single window using RK45.

    Returns
    -------
    x      : (T,) position
    xdot   : (T,) velocity
    xddot  : (T,) acceleration (computed from ODE, not finite differences)
    t_eval : (T,) time vector
    """
    p = FAULT_PARAMS[fault_class]

    dt       = 1.0 / TARGET_HZ
    t_span   = (0.0, SEQ_LENGTH * dt)
    t_eval   = np.linspace(0.0, SEQ_LENGTH * dt, SEQ_LENGTH)

    # Random initial conditions for sample diversity
    rng = np.random.RandomState(seed=window_idx * 4 + fault_class)
    x0      = rng.uniform(-0.5, 0.5)
    xdot0   = rng.uniform(-0.5, 0.5)

    sol = solve_ivp(
        _ode_rhs,
        t_span,
        [x0, xdot0],
        method='RK45',
        t_eval=t_eval,
        args=(p['m'], p['c'], p['k'], p['A'], p['omega0']),
        dense_output=False,
        rtol=1e-6,
        atol=1e-8,
    )

    x    = sol.y[0]
    xdot = sol.y[1]
    # Acceleration directly from the ODE (exact, no FD error)
    xddot = (p['A'] * np.sin(p['omega0'] * t_eval)
             - p['c'] * xdot - p['k'] * x) / p['m']

    return x, xdot, xddot, t_eval


# ── Dataset generation ────────────────────────────────────────────────────────
def generate_dataset(samples_per_class: int = SAMPLES_PER_CLASS):
    """Generate the full dataset for all 4 classes.

    Returns
    -------
    X      : float32 tensor (N, T, 10)  – [vel×4, pos×4, omega, time]
    Y      : float32 tensor (N, T, 4)   – acceleration (4 channels)
    labels : long   tensor (N,)
    """
    all_X, all_Y, all_labels = [], [], []

    for fault_class in range(4):
        for i in range(samples_per_class):
            x, xdot, xddot, t_eval = _solve_window(fault_class, window_idx=i)

            rng = np.random.RandomState(i * 137 + fault_class * 31)

            # ── Y: 4-channel acceleration with sensor noise ────────────────
            Y_window = np.zeros((SEQ_LENGTH, N_CHANNELS), dtype=np.float32)
            base_noise = rng.randn(SEQ_LENGTH) * NOISE_STD
            Y_window[:, 0] = (xddot + base_noise).astype(np.float32)
            for ch in range(1, N_CHANNELS):
                ch_noise = rng.randn(SEQ_LENGTH) * NOISE_STD
                decorr   = rng.randn(SEQ_LENGTH) * CHANNEL_NOISE_STD
                Y_window[:, ch] = (xddot + ch_noise + decorr).astype(np.float32)

            # ── X: [vel×4, pos×4, omega, time] ────────────────────────────
            X_window = np.zeros((SEQ_LENGTH, 10), dtype=np.float32)
            for ch in range(N_CHANNELS):
                ch_noise = rng.randn(SEQ_LENGTH) * CHANNEL_NOISE_STD
                X_window[:, ch]     = (xdot + ch_noise).astype(np.float32)       # vel
                X_window[:, ch + 4] = (x    + ch_noise * 0.5).astype(np.float32) # pos
            X_window[:, 8] = float(OMEGA_0)           # omega (constant)
            X_window[:, 9] = t_eval.astype(np.float32)  # time

            all_X.append(X_window)
            all_Y.append(Y_window)
            all_labels.append(fault_class)

    X      = torch.tensor(np.array(all_X),      dtype=torch.float32)
    Y      = torch.tensor(np.array(all_Y),      dtype=torch.float32)
    labels = torch.tensor(np.array(all_labels), dtype=torch.long)

    return X, Y, labels


def generate_and_save_tensors(output_dir: str,
                               train_per_class: int = 50,
                               test_per_class:  int = 10,
                               verbose: bool = True) -> None:
    """Generate the synthetic oscillator dataset and save .pth tensors.

    Saved files (Phase 0 compatible naming):
      output_dir/X_normal_trainingset.pth   (N_train, T, 10)
      output_dir/Y_normal_trainingset.pth   (N_train, T,  4)
      output_dir/X_normal_testset.pth       (N_test,  T, 10)
      output_dir/Y_normal_testset.pth       (N_test,  T,  4)

    And per-class files for Phase 1 (MaFaulDaDataset-compatible):
      output_dir/X_<class>_trainingset.pth  (N_train, T, 10)
      output_dir/Y_<class>_trainingset.pth  (N_train, T,  4) – stored as (N,T,4)
      output_dir/X_<class>_testset.pth
      output_dir/Y_<class>_testset.pth
    """
    os.makedirs(output_dir, exist_ok=True)
    total_per_class = train_per_class + test_per_class

    for fault_class in range(4):
        X_list, Y_list = [], []
        for i in range(total_per_class):
            x, xdot, xddot, t_eval = _solve_window(fault_class, window_idx=i)
            rng = np.random.RandomState(i * 137 + fault_class * 31)

            Y_window = np.zeros((SEQ_LENGTH, N_CHANNELS), dtype=np.float32)
            base_noise = rng.randn(SEQ_LENGTH) * NOISE_STD
            Y_window[:, 0] = (xddot + base_noise).astype(np.float32)
            for ch in range(1, N_CHANNELS):
                ch_noise = rng.randn(SEQ_LENGTH) * NOISE_STD
                decorr   = rng.randn(SEQ_LENGTH) * CHANNEL_NOISE_STD
                Y_window[:, ch] = (xddot + ch_noise + decorr).astype(np.float32)

            X_window = np.zeros((SEQ_LENGTH, 10), dtype=np.float32)
            for ch in range(N_CHANNELS):
                ch_noise = rng.randn(SEQ_LENGTH) * CHANNEL_NOISE_STD
                X_window[:, ch]     = (xdot + ch_noise).astype(np.float32)
                X_window[:, ch + 4] = (x    + ch_noise * 0.5).astype(np.float32)
            X_window[:, 8] = float(OMEGA_0)
            X_window[:, 9] = t_eval.astype(np.float32)

            X_list.append(X_window)
            Y_list.append(Y_window)

        X_all = torch.from_numpy(np.array(X_list))
        Y_all = torch.from_numpy(np.array(Y_list))

        X_train, X_test = X_all[:train_per_class], X_all[train_per_class:]
        Y_train, Y_test = Y_all[:train_per_class], Y_all[train_per_class:]

        class_name = CLASS_NAMES[fault_class]

        torch.save(X_train, os.path.join(output_dir, f"X_{class_name}_trainingset.pth"))
        torch.save(Y_train, os.path.join(output_dir, f"Y_{class_name}_trainingset.pth"))
        torch.save(X_test,  os.path.join(output_dir, f"X_{class_name}_testset.pth"))
        torch.save(Y_test,  os.path.join(output_dir, f"Y_{class_name}_testset.pth"))

        # Phase 0 expects "normal" naming for the healthy class
        if fault_class == 0:
            torch.save(X_train, os.path.join(output_dir, "X_normal_trainingset.pth"))
            torch.save(Y_train, os.path.join(output_dir, "Y_normal_trainingset.pth"))
            torch.save(X_test,  os.path.join(output_dir, "X_normal_testset.pth"))
            torch.save(Y_test,  os.path.join(output_dir, "Y_normal_testset.pth"))

        if verbose:
            print(f"  Class {fault_class} ({class_name}): {len(X_train)} train, {len(X_test)} test windows saved.")

    if verbose:
        print(f"All synthetic tensors saved to {output_dir}")


# ── PyTorch Dataset ───────────────────────────────────────────────────────────
class SyntheticOscillatorDataset(Dataset):
    """In-memory dataset of pre-generated oscillator windows.

    Conforms to the same contract as MaFaulDaDataset:
      __getitem__ returns (raw_trace, clean_trace, label, omega)
        – raw_trace : (4, T) float32
        – clean_trace: same as raw_trace (no separate clean version)
        – label     : long scalar
        – omega     : float = OMEGA_0
    """

    def __init__(self, data_dir: str,
                 split: str = 'train',
                 seq_length: int = SEQ_LENGTH):
        """
        Parameters
        ----------
        data_dir  : Directory containing *_trainingset.pth / *_testset.pth files.
        split     : 'train' or 'test'.
        seq_length: Expected sequence length (for validation only).
        """
        self.seq_length = seq_length
        suffix = "trainingset" if split == 'train' else "testset"

        traces_list  = []
        labels_list  = []
        omegas_list  = []

        for fault_class, class_name in CLASS_NAMES.items():
            y_path = os.path.join(data_dir, f"Y_{class_name}_{suffix}.pth")
            if not os.path.exists(y_path):
                continue
            y_tensor = torch.load(y_path, map_location='cpu', weights_only=True)
            # y_tensor stored as (N, T, 4) — transpose to (N, 4, T)
            y_tensor = y_tensor.transpose(1, 2).float()

            n = y_tensor.shape[0]
            traces_list.append(y_tensor)
            labels_list.append(torch.full((n,), fault_class, dtype=torch.long))
            omegas_list.append(torch.full((n,), OMEGA_0, dtype=torch.float32))

        if len(traces_list) == 0:
            raise RuntimeError(
                f"No oscillator tensors found in {data_dir} for split='{split}'."
            )

        self.all_traces = torch.cat(traces_list, dim=0)
        self.all_labels = torch.cat(labels_list, dim=0)
        self.all_omegas = torch.cat(omegas_list, dim=0)

    def __len__(self) -> int:
        return self.all_traces.shape[0]

    def __getitem__(self, idx):
        trace = self.all_traces[idx]
        label = self.all_labels[idx]
        omega = self.all_omegas[idx].item()
        return trace, trace, label, omega


# ── Collate function (matches MaFaulDa interface) ─────────────────────────────
class _BatchTuple(tuple):
    """Minimal clone of mafaulda_dataset.BatchTuple — avoids import cycle."""
    def __new__(cls, r, c, l, o):
        return super().__new__(cls, (r, c, l))
    def __init__(self, r, c, l, o):
        self.omega = o


def _osc_collate_fn(batch):
    raw   = torch.stack([b[0] for b in batch])
    clean = torch.stack([b[1] for b in batch])
    label = torch.stack([b[2] for b in batch])
    omega = torch.tensor([b[3] for b in batch], dtype=torch.float32)
    return _BatchTuple(raw, clean, label, omega)


def get_synthetic_dataloaders(data_dir:   str  = "data/processed-synthetic",
                               batch_size: int  = 32,
                               val_split:  float = 0.2):
    """Return (train_loader, val_loader, seq_length) for the oscillator dataset.

    Mimics the signature of mafaulda_dataset.get_dataloaders.
    """
    dataset = SyntheticOscillatorDataset(data_dir=data_dir, split='train')

    N = len(dataset)
    k = max(2, round(1.0 / val_split))  # every k-th sample goes to val
    val_indices   = list(range(0, N, k))
    train_indices = [i for i in range(N) if i % k != 0]

    train_subset = Subset(dataset, train_indices)
    val_subset   = Subset(dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size,
                              shuffle=True,  collate_fn=_osc_collate_fn)
    val_loader   = DataLoader(val_subset,   batch_size=batch_size,
                              shuffle=False, collate_fn=_osc_collate_fn)

    return train_loader, val_loader, dataset.seq_length
