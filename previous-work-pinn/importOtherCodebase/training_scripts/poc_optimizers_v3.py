# %%
# Cell 0: Import Necessary Libraries
# -----------------------------------
# This cell imports the fundamental libraries required for building and training
# Physics-Informed Neural Networks (PINNs) using PyTorch.

# PyTorch for building neural networks and automatic differentiation
import torch
import torch.nn as nn

# NumPy for numerical operations, especially for creating data arrays
import numpy as np

# Matplotlib for plotting the results, such as loss curves and solution comparisons
import matplotlib.pyplot as plt

# Set a default random seed for reproducibility of results
torch.manual_seed(42)
np.random.seed(42)

# Check if a GPU is available and set the device accordingly
# Using a GPU can significantly speed up the training of neural networks
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# %%
# Cell 1: Modular Functions for Analytical PDE Solutions
# -----------------------------------------------------
# This cell defines a set of classes to represent the analytical ("true")
# solutions for various PDEs. Each class now includes a `get_params` method
# that returns its specific physical parameters and suggested slider ranges,
# making them compatible with our interactive visualizer.

import torch
import numpy as np

class BasePDE:
    """
    Abstract base class for PDE definitions.
    """
    def solution(self, x, t):
        """
        Computes the analytical solution at given spatial (x) and temporal (t) points.
        """
        raise NotImplementedError("The 'solution' method must be implemented by the subclass.")

    def get_params(self):
        """
        Returns a dictionary of parameters and their suggested ranges for sliders.
        Format: {'param_name': (min, max, step)}
        """
        return {}

class HeatEquation1D(BasePDE):
    """
    Represents the 1D Heat (or Diffusion) Equation: u_t = alpha * u_xx
    """
    def __init__(self, alpha=0.1, k=1, L=1.0):
        self.alpha = alpha
        self.k = k
        self.L = L

    def solution(self, x, t):
        term_exp = torch.exp(-self.alpha * (self.k * np.pi / self.L)**2 * t)
        term_sin = torch.sin(self.k * np.pi * x / self.L)
        return term_exp * term_sin

    def get_params(self):
        return {
            'alpha': (0.01, 1.0, 0.01),
            'k': (1, 5, 1)
        }

class WaveEquation1D(BasePDE):
    """
    Represents the 1D Wave Equation: u_tt = c^2 * u_xx
    """
    def __init__(self, c=1.0, k=1, L=1.0):
        self.c = c
        self.k = k
        self.L = L

    def solution(self, x, t):
        term_cos = torch.cos(self.c * self.k * np.pi * t / self.L)
        term_sin = torch.sin(self.k * np.pi * x / self.L)
        return term_cos * term_sin

    def get_params(self):
        return {
            'c': (0.5, 5.0, 0.1),
            'k': (1, 5, 1)
        }


class ViscousBurgers1D(BasePDE):
    """
    Represents the 1D Viscous Burgers' Equation: u_t + u * u_x = nu * u_xx
    Using a simpler, more stable analytical solution suitable for PINNs.
    """
    def __init__(self, nu=0.1, A=1.0, k=1.0):
        self.nu = nu  # Viscosity parameter (increased for stability)
        self.A = A    # Amplitude parameter
        self.k = k    # Wave number parameter

    def solution(self, x, t):
        """
        Simple analytical solution: u(x,t) = A * sin(k*π*x) * exp(-nu*k²*π²*t)
        This is a decaying sine wave solution that satisfies Burgers' equation
        for small amplitudes where the nonlinear term u*u_x is negligible.
        """
        # Ensure tensors are on the same device and have proper shape
        pi = torch.tensor(np.pi, dtype=x.dtype, device=x.device)
        
        # Calculate the solution
        spatial_part = torch.sin(self.k * pi * x)
        temporal_part = torch.exp(-self.nu * (self.k * pi)**2 * t)
        
        return self.A * spatial_part * temporal_part

    def get_params(self):
        return {
            'nu': (0.01, 0.5, 0.01),    # Larger viscosity for stability
            'A': (0.1, 2.0, 0.1),       # Amplitude
            'k': (1, 3, 1)              # Wave number (integer)
        }

# --- Example Usage ---
# We can now instantiate a PDE and see its adjustable parameters.
heat_pde_instance = HeatEquation1D()
print("Heat Equation Parameters:", heat_pde_instance.get_params())

wave_pde_instance = WaveEquation1D()
print("Wave Equation Parameters:", wave_pde_instance.get_params())


# %%
# Cell 2: Interactive PDE Visualizer
# ----------------------------------
# This cell defines a reusable `PDEVisualizer` class for plotting PDE solutions.
# It is designed to be interactive, using sliders to change parameters like
# time and the PDE's physical constants. It can visualize both an analytical
# solution and a model's prediction, along with the error.

import numpy as np
import torch
import matplotlib.pyplot as plt
from ipywidgets import interactive, FloatSlider, IntSlider

class PDEVisualizer:
    """
    A class to create interactive visualizations for 1D PDE solutions.

    It can display:
    1. A plot comparing the true solution and a model's prediction at a specific time `t`.
    2. A 2D heatmap of the true solution over the full space-time domain.
    3. A plot of the pointwise error between the true and predicted solutions.
    """
    def __init__(self, pde_class, model=None, x_range=(0, 1), t_range=(0, 1)):
        """
        Initializes the visualizer.

        Args:
            pde_class (class): The class of the PDE to visualize (e.g., HeatEquation1D).
            model (torch.nn.Module, optional): The trained PINN model. Defaults to None.
            x_range (tuple): The spatial domain for plotting.
            t_range (tuple): The temporal domain for plotting.
        """
        self.pde_class = pde_class
        self.model = model
        self.x_range = x_range
        self.t_range = t_range

        # Create a grid for the heatmap plot
        self.x_grid = torch.linspace(x_range[0], x_range[1], 100)
        self.t_grid = torch.linspace(t_range[0], t_range[1], 100)
        self.T, self.X = torch.meshgrid(self.t_grid, self.x_grid, indexing='ij')

    def plot_solution(self, t, **pde_params):
        """
        The core plotting function. This is called by the interactive widget.
        """
        # 1. Instantiate the PDE with the current slider values
        pde_instance = self.pde_class(**pde_params)

        # 2. Setup the plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'{self.pde_class.__name__} at t = {t:.2f}', fontsize=16)

        # --- Plot 1: Solution at time `t` ---
        ax1 = axes[0]
        x_fine = torch.linspace(self.x_range[0], self.x_range[1], 200).view(-1, 1)
        t_slice = torch.full_like(x_fine, t)

        # Calculate true solution
        u_true = pde_instance.solution(x_fine, t_slice)
        ax1.plot(x_fine.numpy(), u_true.numpy(), 'b-', label='True Solution')

        # Calculate model prediction if a model is provided
        if self.model:
            # Prepare input for the model: (t, x) pairs
            model_input = torch.cat([t_slice, x_fine], dim=1)
            u_pred = self.model(model_input)
            ax1.plot(x_fine.numpy(), u_pred.detach().numpy(), 'r--', label='PINN Prediction')

        ax1.set_title('Solution Profile at Time t')
        ax1.set_xlabel('x')
        ax1.set_ylabel('u(x, t)')
        ax1.legend()
        ax1.grid(True)
        ax1.set_xlim(self.x_range)

        # --- Plot 2: Space-Time Heatmap ---
        ax2 = axes[1]
        u_heatmap = pde_instance.solution(self.X, self.T)
        c = ax2.pcolormesh(self.X, self.T, u_heatmap, cmap='viridis', shading='auto')
        fig.colorbar(c, ax=ax2)
        ax2.axhline(y=t, color='r', linestyle='--', label=f't = {t:.2f}') # Mark current time slice
        ax2.set_title('Full Space-Time Solution')
        ax2.set_xlabel('x')
        ax2.set_ylabel('t')
        ax2.legend()

        # --- Plot 3: Pointwise Error ---
        ax3 = axes[2]
        if self.model:
            error = torch.abs(u_true - u_pred.detach())
            ax3.plot(x_fine.numpy(), error.numpy(), 'g-')
            ax3.set_title('Pointwise Absolute Error')
            ax3.set_xlabel('x')
            ax3.set_ylabel('|True - PINN|')
            ax3.grid(True)
            ax3.set_xlim(self.x_range)
        else:
            ax3.text(0.5, 0.5, 'Error plot requires a PINN model', ha='center', va='center')
            ax3.set_title('Error Plot')
            ax3.set_xticks([])
            ax3.set_yticks([])

        plt.tight_layout(rect=(0, 0, 1, 0.96))
        plt.show()

    def create_interactive_plot(self):
        """
        Creates and displays the interactive widget.
        """
        # Create a slider for time
        sliders = {
            't': FloatSlider(min=self.t_range[0], max=self.t_range[1], step=0.01, value=0, description='Time (t)')
        }

        # Create sliders for the PDE's specific parameters
        pde_instance = self.pde_class() # Create a dummy instance to get params
        pde_params_info = pde_instance.get_params()

        for name, (p_min, p_max, p_step) in pde_params_info.items():
            if isinstance(p_step, int):
                sliders[name] = IntSlider(min=p_min, max=p_max, step=p_step, value=pde_instance.__dict__[name], description=name)
            else:
                sliders[name] = FloatSlider(min=p_min, max=p_max, step=p_step, value=pde_instance.__dict__[name], description=name)

        # Link sliders to the plotting function
        interactive_plot = interactive(self.plot_solution, **sliders)
        display(interactive_plot)


class EnhancedPDEVisualizer:
    """
    Enhanced PDE visualizer with improved heatmap functionality.
    Shows true solution, PINN prediction, and difference heatmaps side-by-side.
    """
    def __init__(self, pde_class, model=None, x_range=(0, 1), t_range=(0, 1)):
        self.pde_class = pde_class
        self.model = model
        self.x_range = x_range
        self.t_range = t_range

        # Create a grid for the heatmap plots
        self.x_grid = torch.linspace(x_range[0], x_range[1], 100)
        self.t_grid = torch.linspace(t_range[0], t_range[1], 100)
        self.T, self.X = torch.meshgrid(self.t_grid, self.x_grid, indexing='ij')

    def plot_solution(self, t, **pde_params):
        """Enhanced plotting function with multiple heatmaps."""
        # Instantiate the PDE with current parameters
        pde_instance = self.pde_class(**pde_params)

        if self.model:
            # Create a 2x3 subplot layout
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'{self.pde_class.__name__} - Enhanced Visualization at t = {t:.2f}', fontsize=16)
        else:
            # Create a 2x2 layout if no model is provided
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'{self.pde_class.__name__} - True Solution at t = {t:.2f}', fontsize=16)
            # Add dummy axes to maintain indexing
            flat_axes = axes.flatten()
            dummy_axes = np.array([None], dtype=object)
            axes = np.append(flat_axes, dummy_axes).reshape(2, 3)

        # --- Top Row: 1D slices at time t ---
        x_fine = torch.linspace(self.x_range[0], self.x_range[1], 200).view(-1, 1)
        t_slice = torch.full_like(x_fine, t)
        u_true = pde_instance.solution(x_fine, t_slice)

        # Plot 1: Solution profiles
        ax1 = axes[0, 0]
        ax1.plot(x_fine.numpy(), u_true.numpy(), 'b-', linewidth=2, label='True Solution')

        if self.model:
            model_input = torch.cat([t_slice, x_fine], dim=1)
            u_pred = self.model(model_input)
            ax1.plot(x_fine.numpy(), u_pred.detach().numpy(), 'r--', linewidth=2, label='PINN Prediction')

            # Plot 2: Pointwise error
            ax2 = axes[0, 1]
            error = torch.abs(u_true - u_pred.detach())
            ax2.plot(x_fine.numpy(), error.numpy(), 'g-', linewidth=2)
            ax2.set_title('Pointwise Absolute Error')
            ax2.set_xlabel('x')
            ax2.set_ylabel('|True - PINN|')
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(self.x_range)

            # Plot 3: Relative error
            ax3 = axes[0, 2]
            rel_error = torch.abs((u_true - u_pred.detach()) / (torch.abs(u_true) + 1e-10))
            ax3.plot(x_fine.numpy(), rel_error.numpy(), 'm-', linewidth=2)
            ax3.set_title('Pointwise Relative Error')
            ax3.set_xlabel('x')
            ax3.set_ylabel('|True - PINN| / |True|')
            ax3.grid(True, alpha=0.3)
            ax3.set_xlim(self.x_range)
        else:
            # If no model, show PDE parameter info
            ax2 = axes[0, 1]
            param_text = "PDE Parameters:\n"
            for name, value in pde_params.items():
                param_text += f"{name}: {value:.4f}\n"
            ax2.text(0.5, 0.5, param_text, ha='center', va='center',
                    transform=ax2.transAxes, fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            ax2.set_title('PDE Parameters')
            ax2.set_xticks([])
            ax2.set_yticks([])

        ax1.set_title('Solution Profile at Time t')
        ax1.set_xlabel('x')
        ax1.set_ylabel('u(x, t)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(self.x_range)

        # --- Bottom Row: 2D Heatmaps ---
        # Compute full space-time solutions
        u_true_full = pde_instance.solution(self.X, self.T)

        # Heatmap 1: True solution
        ax4 = axes[1, 0]
        im1 = ax4.pcolormesh(self.X, self.T, u_true_full, cmap='viridis', shading='auto')
        ax4.axhline(y=t, color='white', linestyle='--', linewidth=2, alpha=0.8)
        ax4.set_title('True Solution (Space-Time)')
        ax4.set_xlabel('x')
        ax4.set_ylabel('t')
        plt.colorbar(im1, ax=ax4, shrink=0.8)

        if self.model:
            # Compute PINN solution over full domain
            coords_full = torch.stack([self.T.flatten(), self.X.flatten()], dim=1)
            with torch.no_grad():
                u_pred_full = self.model(coords_full).reshape(self.T.shape)

            # Heatmap 2: PINN solution
            ax5 = axes[1, 1]
            im2 = ax5.pcolormesh(self.X, self.T, u_pred_full, cmap='viridis', shading='auto')
            ax5.axhline(y=t, color='white', linestyle='--', linewidth=2, alpha=0.8)
            ax5.set_title('PINN Solution (Space-Time)')
            ax5.set_xlabel('x')
            ax5.set_ylabel('t')
            plt.colorbar(im2, ax=ax5, shrink=0.8)

            # Heatmap 3: Difference
            ax6 = axes[1, 2]
            diff = torch.abs(u_true_full - u_pred_full)
            im3 = ax6.pcolormesh(self.X, self.T, diff, cmap='Reds', shading='auto')
            ax6.axhline(y=t, color='white', linestyle='--', linewidth=2, alpha=0.8)
            ax6.set_title('Absolute Difference |True - PINN|')
            ax6.set_xlabel('x')
            ax6.set_ylabel('t')
            plt.colorbar(im3, ax=ax6, shrink=0.8)

            # Add error statistics
            max_error = torch.max(diff).item()
            mean_error = torch.mean(diff).item()
            ax6.text(0.02, 0.98, f'Max Error: {max_error:.2e}\nMean Error: {mean_error:.2e}',
                    transform=ax6.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        else:
            # If no model, hide unused subplots
            if axes[1, 1] is not None:
                axes[1, 1].set_visible(False)
            if axes[1, 2] is not None:
                axes[1, 2].set_visible(False)

        plt.tight_layout(rect=(0, 0, 1, 0.96))
        plt.show()

    def create_interactive_plot(self, **default_pde_params):
        """Create interactive plot with enhanced visualization."""
        # Create time slider
        sliders = {
            't': FloatSlider(
                min=self.t_range[0], max=self.t_range[1], step=0.01,
                value=self.t_range[0], description='Time (t)'
            )
        }

        # Create PDE parameter sliders
        pde_instance = self.pde_class(**default_pde_params)
        pde_params_info = pde_instance.get_params()

        for name, (p_min, p_max, p_step) in pde_params_info.items():
            current_value = default_pde_params.get(name, getattr(pde_instance, name))
            if isinstance(p_step, int):
                sliders[name] = IntSlider(
                    min=p_min, max=p_max, step=p_step,
                    value=current_value, description=name
                )
            else:
                sliders[name] = FloatSlider(
                    min=p_min, max=p_max, step=p_step,
                    value=current_value, description=name
                )

        # Create and display interactive plot
        interactive_plot = interactive(self.plot_solution, **sliders)
        display(interactive_plot)


# --- Example Usage ---
# To see it in action, simply instantiate the visualizer and call the create method.
# For now, we pass `model=None`. Later, you can pass your trained PINN.

print("Creating interactive plot for the 1D Heat Equation...")
#heat_visualizer = PDEVisualizer(HeatEquation1D, model=None, x_range=(-1, 1), t_range=(0, 2))
#heat_visualizer.create_interactive_plot()

visualizer = PDEVisualizer(ViscousBurgers1D, model=None, x_range=(-1, 1), t_range=(0, 2))
visualizer.create_interactive_plot()



# %%
# Cell 3: PINN Dataset Generator
# ------------------------------
# This cell defines a class to generate all the necessary data points for training a PINN.
# A PINN's training data consists of three main types of points:
# 1. Initial/Boundary Condition Points: To enforce the problem's constraints.
# 2. Measurement Data Points: A sparse set of points with known "true" values (plus optional noise)
#    that simulate real-world sensor data.
# 3. Collocation Points: A large set of points inside the domain where we enforce the PDE itself.

import torch
import numpy as np
# We use Latin Hypercube Sampling for better coverage of the domain
from scipy.stats.qmc import LatinHypercube

class PINNDatasetGenerator:
    """
    Generates training data for a 1D PINN.
    """
    def __init__(self, pde_instance, x_range, t_range, device='cpu'):
        """
        Initializes the data generator.

        Args:
            pde_instance (BasePDE): An instance of a PDE class from Cell 1.
            x_range (tuple): The spatial domain (x_min, x_max).
            t_range (tuple): The temporal domain (t_min, t_max).
            device (str): The device to move the tensors to ('cpu' or 'cuda').
        """
        self.pde = pde_instance
        self.x_range = x_range
        self.t_range = t_range
        self.device = device

    def generate_data(self, n_initial, n_boundary, n_data, n_collocation, noise_level=0.0):
        """
        Generates and returns the full dataset.

        Args:
            n_initial (int): Number of initial condition points.
            n_boundary (int): Number of boundary condition points for each boundary.
            n_data (int): Number of measurement data points.
            n_collocation (int): Number of collocation points.
            noise_level (float): Standard deviation of Gaussian noise to add to measurement data.

        Returns:
            dict: A dictionary containing all the generated data as tensors.
        """
        # --- 1. Initial Condition (t=0) ---
        x_initial = self.x_range[0] + (self.x_range[1] - self.x_range[0]) * torch.rand(n_initial, 1)
        t_initial = torch.full_like(x_initial, self.t_range[0])
        coords_ic = torch.cat([t_initial, x_initial], dim=1)
        values_ic = self.pde.solution(x_initial, t_initial)

        # --- 2. Boundary Conditions (x=x_min and x=x_max) ---
        t_boundary = self.t_range[0] + (self.t_range[1] - self.t_range[0]) * torch.rand(n_boundary, 1)

        # Points at x = x_min
        x_boundary_min = torch.full_like(t_boundary, self.x_range[0])
        coords_bc_min = torch.cat([t_boundary, x_boundary_min], dim=1)
        values_bc_min = self.pde.solution(x_boundary_min, t_boundary)

        # Points at x = x_max
        x_boundary_max = torch.full_like(t_boundary, self.x_range[1])
        coords_bc_max = torch.cat([t_boundary, x_boundary_max], dim=1)
        values_bc_max = self.pde.solution(x_boundary_max, t_boundary)

        # Combine boundary points
        coords_bc = torch.cat([coords_bc_min, coords_bc_max], dim=0)
        values_bc = torch.cat([values_bc_min, values_bc_max], dim=0)

        # Use Latin Hypercube Sampling for better spatial coverage
        sampler = LatinHypercube(d=2) # 2 dimensions: t and x

        # --- 3. Measurement Data (sparse points inside the domain) ---
        sample_data = sampler.random(n=n_data)
        t_data_sampled = self.t_range[0] + (self.t_range[1] - self.t_range[0]) * torch.from_numpy(sample_data[:, 0:1]).float()
        x_data_sampled = self.x_range[0] + (self.x_range[1] - self.x_range[0]) * torch.from_numpy(sample_data[:, 1:2]).float()
        coords_data = torch.cat([t_data_sampled, x_data_sampled], dim=1)

        values_data = self.pde.solution(x_data_sampled, t_data_sampled)
        # Add optional noise
        if noise_level > 0:
            values_data += noise_level * torch.randn_like(values_data)

        # --- 4. Collocation Points (for enforcing the PDE) ---
        sample_collocation = sampler.random(n=n_collocation)
        t_collocation = self.t_range[0] + (self.t_range[1] - self.t_range[0]) * torch.from_numpy(sample_collocation[:, 0:1]).float()
        x_collocation = self.x_range[0] + (self.x_range[1] - self.x_range[0]) * torch.from_numpy(sample_collocation[:, 1:2]).float()
        coords_collocation = torch.cat([t_collocation, x_collocation], dim=1)
        # These points need to have gradients enabled for backpropagation
        # coords_collocation = coords_collocation.requires_grad_(True)

        # --- Assemble the dictionary ---
        dataset = {
            'initial': {'coords': coords_ic.to(self.device), 'values': values_ic.to(self.device)},
            'boundary': {'coords': coords_bc.to(self.device), 'values': values_bc.to(self.device)},
            'data': {'coords': coords_data.to(self.device), 'values': values_data.to(self.device)},
            'collocation': {'coords': coords_collocation.to(self.device)}
        }

        return dataset


# --- Example Usage ---
# Let's generate a dataset for the Heat Equation.

# 1. Instantiate the PDE
heat_pde_instance = HeatEquation1D(alpha=0.2, k=2)

# 2. Define domain and number of points
X_RANGE = (0, 1)
T_RANGE = (0, 1)
N_INITIAL = 100
N_BOUNDARY = 100
N_DATA = 2000 # Sparse sensor data
N_COLLOCATION = 10000 # Points for physics loss

# 3. Create the generator and generate data
data_generator = PINNDatasetGenerator(heat_pde_instance, X_RANGE, T_RANGE)
training_data = data_generator.generate_data(
    n_initial=N_INITIAL,
    n_boundary=N_BOUNDARY,
    n_data=N_DATA,
    n_collocation=N_COLLOCATION,
    noise_level=0.01 # Add 1% noise
)

# 4. Let's inspect the generated data
print("Dataset generated successfully!")
for key, value in training_data.items():
    print(f"\n--- {key.upper()} POINTS ---")
    for name, tensor in value.items():
        print(f"  '{name}' shape: {tensor.shape}")

# Optional: Visualize the generated points
plt.figure(figsize=(10, 6))
plt.scatter(training_data['initial']['coords'][:, 1].cpu().numpy(), training_data['initial']['coords'][:, 0].cpu().numpy(), s=10, label=f"Initial (N={N_INITIAL})")
plt.scatter(training_data['boundary']['coords'][:, 1].cpu().numpy(), training_data['boundary']['coords'][:, 0].cpu().numpy(), s=10, label=f"Boundary (N={N_BOUNDARY*2})")
plt.scatter(training_data['data']['coords'][:, 1].cpu().numpy(), training_data['data']['coords'][:, 0].cpu().numpy(), s=5, alpha=0.5, label=f"Measurement Data (N={N_DATA})")
plt.scatter(training_data['collocation']['coords'].detach()[:, 1].cpu().numpy(), training_data['collocation']['coords'].detach()[:, 0].cpu().numpy(), s=1, alpha=0.2, label=f"Collocation (N={N_COLLOCATION})")
plt.xlabel("Space (x)")
plt.ylabel("Time (t)")
plt.title("Distribution of Generated Training Points")
plt.legend()
plt.show()




# %%
# Cell 4: K-Fold and Test Set Generator
# -------------------------------------
# This cell defines a utility to split the generated dataset into a final test set
# and K-folds for cross-validation. This is crucial for robustly evaluating the
# performance of a machine learning model.
#
# We will split the 'data' (measurement) points, as these represent the sparse,
# real-world information we want our model to generalize from. The initial,
# boundary, and collocation points define the problem's physics and are typically
# not split.

import torch
from sklearn.model_selection import KFold, train_test_split

class PINNDataSplitter:
    """
    Handles splitting of the PINN dataset into a test set and K-folds for training/validation.
    """
    def __init__(self, dataset, n_splits=5, test_size=0.2, shuffle=True, random_state=42):
        """
        Initializes the data splitter.

        Args:
            dataset (dict): The full dataset dictionary from PINNDatasetGenerator.
            n_splits (int): The number of folds (K) for cross-validation.
            test_size (float): The proportion of measurement data to hold out for the test set.
            shuffle (bool): Whether to shuffle the data before splitting.
            random_state (int): Seed for reproducibility.
        """
        self.dataset = dataset
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

        # We primarily split the 'data' points.
        data_coords = dataset['data']['coords']
        data_values = dataset['data']['values']

        # --- 1. Create the initial hold-out test set from the measurement data ---
        train_val_indices, test_indices = train_test_split(
            range(data_coords.shape[0]),
            test_size=test_size,
            shuffle=shuffle,
            random_state=random_state
        )

        # The coordinates and values for the training+validation set
        self.train_val_coords = data_coords[train_val_indices]
        self.train_val_values = data_values[train_val_indices]

        # The coordinates and values for the final test set
        self.test_coords = data_coords[test_indices]
        self.test_values = data_values[test_indices]

        # --- 2. Setup the K-Fold splitter for the remaining data ---
        self.kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def get_test_set(self):
        """
        Returns the hold-out test set. This set should only be used for final evaluation.

        Returns:
            dict: A dataset dictionary containing the test data points plus all
                  the original IC, BC, and collocation points.
        """
        test_set = self.dataset.copy()
        test_set['data'] = {'coords': self.test_coords, 'values': self.test_values}
        return test_set

    def get_folds(self):
        """
        A generator that yields training and validation sets for each fold.

        Yields:
            tuple: A tuple containing two dataset dictionaries, (train_set, validation_set).
        """
        for train_indices, val_indices in self.kf.split(self.train_val_coords):
            # Get the training data for this fold
            fold_train_coords = self.train_val_coords[train_indices]
            fold_train_values = self.train_val_values[train_indices]

            # Get the validation data for this fold
            fold_val_coords = self.train_val_coords[val_indices]
            fold_val_values = self.train_val_values[val_indices]

            # --- Assemble the full training set for this fold ---
            # It includes its share of measurement data PLUS ALL other point types
            train_set = self.dataset.copy()
            train_set['data'] = {'coords': fold_train_coords, 'values': fold_train_values}

            # --- Assemble the full validation set for this fold ---
            # It includes its share of measurement data PLUS ALL other point types
            validation_set = self.dataset.copy()
            validation_set['data'] = {'coords': fold_val_coords, 'values': fold_val_values}

            yield train_set, validation_set


# --- Example Usage ---
# We use the 'training_data' dictionary generated in the previous cell.

# 1. Instantiate the splitter
# We'll use 5 folds for cross-validation and hold out 20% of data for the final test.
N_SPLITS = 5
TEST_SIZE = 0.2
data_splitter = PINNDataSplitter(training_data, n_splits=N_SPLITS, test_size=TEST_SIZE)

# 2. Get the final test set
final_test_set = data_splitter.get_test_set()
print(f"Created a hold-out test set with {final_test_set['data']['coords'].shape[0]} measurement points.")

# 3. Loop through the folds for training and validation
print(f"\nGenerating {N_SPLITS} folds for cross-validation...")
for i, (train_set, val_set) in enumerate(data_splitter.get_folds()):
    print(f"--- Fold {i+1}/{N_SPLITS} ---")
    print(f"  Training measurement points: {train_set['data']['coords'].shape[0]}")
    print(f"  Validation measurement points: {val_set['data']['coords'].shape[0]}")
    # In a real scenario, you would put your model training and validation logic here.
    # train_model(train_set)
    # evaluate_model(val_set)

# Let's check the number of points to be sure
original_data_size = training_data['data']['coords'].shape[0]
train_val_size = train_set['data']['coords'].shape[0] + val_set['data']['coords'].shape[0]
test_size = final_test_set['data']['coords'].shape[0]

print("\n--- Sanity Check ---")
print(f"Original measurement points: {original_data_size}")
print(f"Points used in one K-Fold loop (Train+Val): {train_val_size}")
print(f"Points in hold-out test set: {test_size}")
print(f"Total points accounted for: {train_val_size + test_size}")




# %%
# Cell 5: Modular PINN Architecture, Base Loss, and Trainer
# ---------------------------------------------------------
# This cell has been refactored to support complex optimization strategies
# by delegating the training step logic to the loss function itself.

import torch
import torch.nn as nn
from collections import defaultdict
import matplotlib.pyplot as plt

# --- 1. The Neural Network Architecture (Modified for GradNorm) ---
class PINN(nn.Module):
    """A Feed-Forward Neural Network with explicitly named shared layers."""
    def __init__(self, input_dim=2, output_dim=1, hidden_layers=4, hidden_units=20):
        super(PINN, self).__init__()

        shared_layers = []
        shared_layers.append(nn.Linear(input_dim, hidden_units))
        shared_layers.append(nn.Tanh())
        # The last hidden layer is now separate, so we loop one less time
        for _ in range(hidden_layers - 1):
            shared_layers.append(nn.Linear(hidden_units, hidden_units))
            shared_layers.append(nn.Tanh())

        # This is the main shared part of the network
        self.shared_net = nn.Sequential(*shared_layers)

        # This is the final layer whose weights W will be used by GradNorm
        self.last_shared_layer = nn.Linear(hidden_units, hidden_units)

        # This is the task-specific output layer
        self.output_layer = nn.Linear(hidden_units, output_dim)

    def forward(self, x):
        # Pass through the main shared network
        shared_output = self.shared_net(x)
        # Pass through the last shared layer and apply activation
        last_shared_output = torch.tanh(self.last_shared_layer(shared_output))
        # Final output from the head
        return self.output_layer(last_shared_output)

# --- 2. The Loss Function Structure (Refactored) ---
class BaseLoss(nn.Module):
    """
    Abstract base class for PINN loss functions.
    NOW INCLUDES a `step` method to handle the optimization logic.
    """
    def __init__(self, pde_instance):
        super().__init__()
        self.pde = pde_instance
        self.mse_loss = nn.MSELoss()
        # Define the standard loss component keys
        self.loss_keys = ['data', 'initial', 'boundary', 'pde']

    def calculate_residuals(self, model, coords):
        # Calculate PDE residuals using automatic differentiation
        u = model(coords)
        u_grads = torch.autograd.grad(u, coords, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_t = u_grads[:, 0:1]
        u_x = u_grads[:, 1:2]
        u_x_grads = torch.autograd.grad(u_x, coords, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_xx = u_x_grads[:, 1:2]
        if isinstance(self.pde, HeatEquation1D):
            residual = u_t - self.pde.alpha * u_xx
        elif isinstance(self.pde, ViscousBurgers1D):
            residual = u_t + u * u_x - self.pde.nu * u_xx
        elif isinstance(self.pde, WaveEquation1D):
            u_t_grads = torch.autograd.grad(u_t, coords, grad_outputs=torch.ones_like(u_t), create_graph=True)[0]
            u_tt = u_t_grads[:, 0:1]
            residual = u_tt - (self.pde.c**2) * u_xx
        else:
            raise NotImplementedError(f"Residual calculation for {type(self.pde).__name__} is not implemented.")
        return residual

    def forward(self, model, dataset):
        # CHANGE 1: This method now only calculates data-based losses.
        # The 'pde' loss will be calculated inside each subclass's step() method.
        losses = {
            'data': self.mse_loss(model(dataset['data']['coords']), dataset['data']['values']),
            'initial': self.mse_loss(model(dataset['initial']['coords']), dataset['initial']['values']),
            'boundary': self.mse_loss(model(dataset['boundary']['coords']), dataset['boundary']['values']),
        }
        return losses

    def forward_with_residuals(self, model, dataset):
        """
        Returns both scalar losses and raw residuals for BRDR-style algorithms.
        This method provides point-wise residuals needed for advanced loss balancing.
        """
        # Calculate predictions and residuals for each component
        data_pred = model(dataset['data']['coords'])
        initial_pred = model(dataset['initial']['coords'])
        boundary_pred = model(dataset['boundary']['coords'])

        # Calculate residuals (prediction - true_value)
        data_residual = (data_pred - dataset['data']['values']).squeeze()
        initial_residual = (initial_pred - dataset['initial']['values']).squeeze()
        boundary_residual = (boundary_pred - dataset['boundary']['values']).squeeze()

        # Calculate scalar losses
        losses = {
            'data': self.mse_loss(data_pred, dataset['data']['values']),
            'initial': self.mse_loss(initial_pred, dataset['initial']['values']),
            'boundary': self.mse_loss(boundary_pred, dataset['boundary']['values']),
        }

        # Return both losses and raw residuals
        residuals = {
            'data_res': data_residual,
            'initial_res': initial_residual,
            'boundary_res': boundary_residual
        }

        return losses, residuals

    def step(self, losses, optimizer, dataset, model):
        """
        Performs a standard optimization step.
        This method will be overridden by more complex loss functions.
        CHANGE 1: Now takes dataset and model parameters for PDE residual calculation.
        """
        # First, combine the losses to get a single value
        total_loss = self.combine(losses)

        # Standard optimization procedure
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        return total_loss

    def combine(self, losses):
        """
        Placeholder for combining losses. Must be implemented by subclasses
        that use the default `step` method.
        """
        raise NotImplementedError("The 'combine' method must be implemented by the subclass.")


# --- 3. The Trainer (Refactored) ---
class EarlyStopping:
    # (This class remains unchanged)
    def __init__(self, patience=50, min_delta=1e-6, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counters = {}
        self.best_losses = {}
        self.early_stop = False
    def __call__(self, losses):
        improvement_found = False
        for key, loss in losses.items():
            if key not in self.best_losses:
                self.best_losses[key] = loss
                self.counters[key] = 0
                improvement_found = True
            elif self.best_losses[key] - loss > self.min_delta:
                if self.verbose:
                    print(f"Improvement for {key}: {self.best_losses[key]:.6f} -> {loss:.6f}", end='\r')
                self.best_losses[key] = loss
                self.counters[key] = 0
                improvement_found = True
            else:
                self.counters[key] += 1
        if not improvement_found:
            if all(c >= self.patience for c in self.counters.values()):
                print("\nEarly stopping triggered: All monitored losses have stagnated.")
                self.early_stop = True
        else:
            for key in self.counters:
                if self.counters[key] > 0 and self.best_losses[key] - losses[key] <= self.min_delta:
                    pass
                else:
                    self.counters[key] = 0


from collections import defaultdict

class PINNTrainer:
    """
    Orchestrates the training of a PINN model.
    REFACTORED: The training loop is now generic and delegates the optimization
    step to the loss function's `step` method.
    Added: Optional validation-based checkpointing on the sum of component losses.
    """
    def __init__(self, model, loss_fn, optimizer, early_stopping=None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.early_stopping = early_stopping
        self.device = next(model.parameters()).device

    def _move_to_device(self, dataset):
        """Move dataset to device."""
        device_dataset = {}
        for key, value_dict in dataset.items():
            device_value_dict = {}
            for name, tensor in value_dict.items():
                device_value_dict[name] = tensor.to(self.device)
            device_dataset[key] = device_value_dict
        return device_dataset

    def train(self, train_dataset, epochs, val_dataset=None):
        history = defaultdict(list)
        train_ds = self._move_to_device(train_dataset)

        # --- setup checkpointing ---
        best_val_score = float('inf')
        best_state = None

        for epoch in range(epochs):
            self.model.train()

            # Ensure collocation coordinates are leaf tensors that require gradients
            # This is crucial inside the loop, especially for K-Fold CV,
            # to ensure the graph is correctly rebuilt for each fold's training.
            if 'collocation' in train_ds:
                train_ds['collocation']['coords'].requires_grad_(True)

            # 1) compute losses on training set
            losses = self.loss_fn(self.model, train_ds)

            # 2) CHANGE 2: Allow for complex weight updates that may require their own backward passes
            if hasattr(self.loss_fn, 'update_weights'):
                # Pass the model and dataset to the hook
                self.loss_fn.update_weights(self.model, train_ds, losses)

            # 3) delegate optimization to the loss function's step method
            # CHANGE 1: Pass train_ds and model for PDE residual calculation
            total_loss = self.loss_fn.step(losses, self.optimizer, train_ds, self.model)

            # 3) logging
            history['total_loss'].append(total_loss.item())
            for key, value in losses.items():
                history[f'loss_{key}'].append(value.item())
            if hasattr(self.loss_fn, 'last_lambdas'):
                for key, value in self.loss_fn.last_lambdas.items():
                    history[f'lambda_{key}'].append(value)
            if hasattr(self.loss_fn, 'log_eps_pde'):
                history['epsilon_pde'].append(torch.exp(self.loss_fn.log_eps_pde).item())
                history['epsilon_data'].append(torch.exp(self.loss_fn.log_eps_data).item())
                history['epsilon_ic'].append(torch.exp(self.loss_fn.log_eps_initial).item())
                history['epsilon_bc'].append(torch.exp(self.loss_fn.log_eps_boundary).item())

            if (epoch + 1) % 1000 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss.item():.4e}')

            # 4) early stopping (on training component losses)
            if self.early_stopping:
                comp = {k: v.item() for k, v in losses.items()}
                self.early_stopping(comp)
                if self.early_stopping.early_stop:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            # 5) validation & checkpointing
            if val_dataset is not None:
                self.model.eval()
                val_ds = self._move_to_device(val_dataset)
                # Enable gradients for collocation coordinates during validation
                # because PDE loss computation requires gradients
                if 'collocation' in val_ds:
                    val_ds['collocation']['coords'].requires_grad_(True)

                # Don't use no_grad context for validation since PDE loss needs gradients
                val_losses = self.loss_fn(self.model, val_ds)
                # sum *only* the data-based MSE terms (forward() no longer returns 'pde')
                data_loss_keys = ['data', 'initial', 'boundary']
                val_score = sum(val_losses[k].item() for k in data_loss_keys)
                if val_score < best_val_score:
                    best_val_score = val_score
                    # save both model and loss‐fn state if available
                    best_state = {
                        'model': self.model.state_dict(),
                        'loss_fn': (self.loss_fn.state_dict()
                                    if hasattr(self.loss_fn, 'state_dict')
                                    else None)
                    }

        # 6) restore best checkpoint
        if best_state is not None:
            self.model.load_state_dict(best_state['model'])
            if best_state['loss_fn'] is not None:
                self.loss_fn.load_state_dict(best_state['loss_fn'])

        return history


# %%
# Cell 6: Modular Interactive Training Dashboard Class (With K-Fold CV)
# ----------------------------------------------------
import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import torch
import inspect  # To inspect __init__ signatures

class PINNDashboard:
    """
    UI for running and visualizing PINN training experiments,
    with optional k-fold cross-validation.
    """
    def __init__(self, pde_map, loss_fn_map, use_kfold=False, n_splits=5, test_size=0.2):
        if not pde_map or not loss_fn_map:
            raise ValueError("pde_map and loss_fn_map cannot be empty.")
        self.pde_map = pde_map
        self.loss_fn_map = loss_fn_map
        self.use_kfold = use_kfold
        self.n_splits = n_splits
        self.test_size = test_size
        self._build_ui()

    def _build_ui(self):
        # PDE selector
        self.pde_dropdown = widgets.Dropdown(
            options=self.pde_map.keys(), description='Select PDE:', style={'description_width': 'initial'}
        )
        # Loss Balancer selector
        self.loss_fn_dropdown = widgets.Dropdown(
            options=self.loss_fn_map.keys(), description='Loss Balancer:', style={'description_width': 'initial'}
        )
        # K-fold toggle and splits
        self.kfold_checkbox = widgets.Checkbox(value=self.use_kfold, description='Use K-Fold CV')
        self.splits_slider = widgets.IntSlider(value=self.n_splits, min=2, max=10, step=1, description='Folds:')
        # Parameter containers
        self.pde_params_container = widgets.VBox()
        self.loss_params_container = widgets.VBox()
        # Training controls
        self.epochs_slider = widgets.IntSlider(value=2000, min=500, max=20000, step=500, description='Max Epochs:')
        self.lr_float = widgets.FloatLogSlider(value=1e-3, base=10, min=-5, max=-2, step=0.1, description='LR:')
        self.patience_slider = widgets.IntSlider(value=200, min=50, max=2000, step=50, description='Patience:')
        # Run button and output area
        self.run_button = widgets.Button(description="Run Training", button_style='success')
        self.output_area = widgets.Output()

        # Wire up callbacks
        self.pde_dropdown.observe(self._on_pde_change, names='value')
        self.loss_fn_dropdown.observe(self._on_loss_fn_change, names='value')
        self.run_button.on_click(self._on_run_button_clicked)

    def _on_pde_change(self, change):
        pde_class = self.pde_map[change['new']]
        pde_instance = pde_class()
        params = pde_instance.get_params()
        self.pde_params_container.children = [
            (widgets.IntSlider if isinstance(rng[0], int) else widgets.FloatSlider)(
                value=getattr(pde_instance, name), min=rng[0], max=rng[1], step=rng[2], description=f'{name}:'
            ) for name, rng in params.items()
        ]

    def _on_loss_fn_change(self, change):
        loss_cls = self.loss_fn_map[change['new']]
        if hasattr(loss_cls, 'get_params'):
            params = loss_cls.get_params()
            self.loss_params_container.children = [
                widgets.FloatSlider(value=val, min=min_, max=max_, step=step, description=f'{name}:')
                for name, (val, min_, max_, step) in params.items()
            ]
        else:
            self.loss_params_container.children = []

    def _on_run_button_clicked(self, b):
        with self.output_area:
            clear_output()
            # gather config
            pde_class = self.pde_map[self.pde_dropdown.value]
            pde_params = {w.description[:-1]: w.value for w in self.pde_params_container.children}
            loss_cls = self.loss_fn_map[self.loss_fn_dropdown.value]
            loss_params = {w.description[:-1]: w.value for w in self.loss_params_container.children}
            epochs = self.epochs_slider.value
            lr = self.lr_float.value
            patience = self.patience_slider.value
            use_kfold = self.kfold_checkbox.value
            n_splits = self.splits_slider.value

            print(f"PDE={self.pde_dropdown.value}, Loss={self.loss_fn_dropdown.value}, E={epochs}, LR={lr:.1e}, P={patience}, KFold={use_kfold}/{n_splits}")

            # device & PDE instance
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pde_inst = pde_class(**pde_params)
            X_RANGE, T_RANGE = (0,1), (0,1)

            # Dataset
            print("Generating data...")
            data = PINNDatasetGenerator(pde_inst, X_RANGE, T_RANGE, str(device)).generate_data(
                n_initial=100, n_boundary=100,
                n_data=2000, n_collocation=10000,
                noise_level=0.0
            )

            # Setup splitter if needed
            if use_kfold:
                splitter = PINNDataSplitter(data, n_splits=n_splits, test_size=self.test_size)
                folds = list(splitter.get_folds())
                fold_histories = []
                best_score = float('inf')
                best_model_state = None
                best_loss_fn_state = None

                for idx, (train_set, val_set) in enumerate(folds, 1):
                    print(f"Fold {idx}/{n_splits}")
                    model, loss_fn, history = self._train_one_fold(
                        train_set, val_set, pde_inst, loss_cls, loss_params,
                        epochs, lr, patience, device
                    )
                    fold_histories.append(history)

                    # Evaluate on validation set to find best fold
                    model.eval()
                    val_ds = self._move_to_device(val_set, device)
                    # Don't use no_grad context since PDE loss needs gradients
                    val_losses = loss_fn(model, val_ds)
                    # Use sum of component losses for model selection
                    val_score = sum(val_losses[k].item() for k in ['data', 'initial', 'boundary'])

                    if val_score < best_score:
                        best_score = val_score
                        best_model_state = model.state_dict().copy()
                        if hasattr(loss_fn, 'state_dict'):
                            best_loss_fn_state = loss_fn.state_dict().copy()

                # Create final model with best fold weights
                self.final_model = PINN().to(device)
                if best_model_state is not None:
                    self.final_model.load_state_dict(best_model_state)

                # Create final loss function for potential state restoration
                sig = inspect.signature(loss_cls.__init__)
                if 'model' in sig.parameters:
                    self.final_loss_fn = loss_cls(pde_inst, model=self.final_model, weights=loss_params)
                else:
                    self.final_loss_fn = loss_cls(pde_inst, weights=loss_params)

                if best_loss_fn_state is not None:
                    self.final_loss_fn.load_state_dict(best_loss_fn_state)

                # Average the histories for plotting
                history = self._average_fold_histories(fold_histories)
                print(f"Best fold validation score: {best_score:.6e}")
            else:
                # single training
                self.final_model, self.final_loss_fn, history = self._train_one_fold(
                    data, None, pde_inst, loss_cls, loss_params, epochs, lr, patience, device
                )

            # Store for visualization
            self.pde_class = pde_class
            self.pde_params = pde_params
            self.X_RANGE = X_RANGE
            self.T_RANGE = T_RANGE

            # Plot training history
            self._plot_history(history)

            # Create interactive visualization
            self._create_interactive_visualization()

    def _train_one_fold(self, train_ds, val_ds, pde_inst, loss_cls, loss_params, epochs, lr, patience, device):
        """Train a single fold and return the trained model, loss function, and history."""
        # Initialize model and loss function
        model = PINN().to(device)
        sig = inspect.signature(loss_cls.__init__)
        if 'model' in sig.parameters:
            loss_fn = loss_cls(pde_inst, model=model, weights=loss_params)
        else:
            loss_fn = loss_cls(pde_inst, weights=loss_params)

        # Setup optimizer and trainer
        # Filter out parameters that don't require gradients to avoid optimizer warnings
        model_params = [p for p in model.parameters() if p.requires_grad]
        loss_params = [p for p in loss_fn.parameters() if p.requires_grad]

        # Remove duplicates to prevent the duplicate parameter warning
        all_params = model_params + loss_params
        # Convert to set and back to list to remove duplicates based on object identity
        seen = set()
        unique_params = []
        for param in all_params:
            if id(param) not in seen:
                seen.add(id(param))
                unique_params.append(param)
        
        optimizer = torch.optim.Adam(unique_params, lr=lr)
        early_stopping = EarlyStopping(patience=patience)
        trainer = PINNTrainer(model, loss_fn, optimizer, early_stopping=early_stopping)

        # Train the model
        history = trainer.train(train_ds, epochs, val_dataset=val_ds)

        return model, loss_fn, history

    def _average_fold_histories(self, fold_histories):
        """Average the training histories across folds."""
        if not fold_histories:
            return {}

        # Get all keys from the first history
        all_keys = fold_histories[0].keys()
        averaged_history = {}

        for key in all_keys:
            # Stack all fold histories for this key
            values_per_fold = [hist[key] for hist in fold_histories if key in hist]
            if values_per_fold:
                # Take the minimum length to handle different stopping points
                min_length = min(len(values) for values in values_per_fold)
                truncated_values = [values[:min_length] for values in values_per_fold]
                # Average across folds
                averaged_history[key] = [
                    sum(fold_values[i] for fold_values in truncated_values) / len(truncated_values)
                    for i in range(min_length)
                ]

        return averaged_history

    def _move_to_device(self, dataset, device):
        """Move dataset to specified device and ensure gradients are properly set."""
        device_dataset = {}
        for key, value_dict in dataset.items():
            device_value_dict = {}
            for name, tensor in value_dict.items():
                device_value_dict[name] = tensor.to(device)
            device_dataset[key] = device_value_dict

        # Ensure collocation coordinates have gradients enabled
        if 'collocation' in device_dataset and 'coords' in device_dataset['collocation']:
            device_dataset['collocation']['coords'] = device_dataset['collocation']['coords'].detach().requires_grad_(True)

        return device_dataset

    def _plot_history(self, history):
        """Plot the training history."""
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(f'Training History for {self.loss_fn_dropdown.value}', fontsize=16)

        # Plot losses
        for key, value in history.items():
            if 'loss' in key:
                ax = axes[0]
                color = 'black' if key == 'total_loss' else None
                linewidth = 2 if key == 'total_loss' else 1
                linestyle = '-' if key == 'total_loss' else '--'
                alpha = 1.0 if key == 'total_loss' else 0.7
                ax.plot(value, label=key.replace('_', ' ').title(),
                       color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha)

        axes[0].set_ylabel('Loss')
        axes[0].set_yscale('log')
        axes[0].legend()
        axes[0].grid(True, which="both", ls="-", alpha=0.5)
        axes[0].set_title("Loss Components Over Epochs")

        # Plot adaptive parameters if they exist
        weight_keys_found = any('loss' not in key and 'total' not in key for key in history)
        if weight_keys_found:
            for key, value in history.items():
                if 'loss' not in key and 'total' not in key:
                    # Convert tensor values to numpy-compatible format
                    if isinstance(value, list) and len(value) > 0:
                        # Handle case where values might be tensors
                        plot_values = []
                        for v in value:
                            if hasattr(v, 'item'):  # if it's a tensor
                                plot_values.append(v.item())
                            elif hasattr(v, 'cpu'):  # if it's a CUDA tensor
                                plot_values.append(v.cpu().item())
                            else:
                                plot_values.append(float(v))
                        axes[1].plot(plot_values, label=key.replace('_', ' ').title())
                    else:
                        axes[1].plot(value, label=key.replace('_', ' ').title())
            axes[1].set_ylabel('Parameter Value')
            axes[1].set_xlabel('Epoch')
            axes[1].legend()
            axes[1].grid(True, which="both", ls="-", alpha=0.5)
            axes[1].set_title("Adaptive Parameter History")
        else:
            # Remove the second subplot if no adaptive parameters
            fig.set_figheight(6)
            axes[1].set_visible(False)

        plt.tight_layout(rect=(0, 0, 1, 0.96))
        plt.show()


    def _create_interactive_visualization(self):
        """Create the interactive visualization with improved heatmaps."""
        # Move model to CPU for visualization
        self.final_model.to('cpu')

        # Create the enhanced visualizer
        enhanced_visualizer = EnhancedPDEVisualizer(
            self.pde_class, self.final_model,
            x_range=self.X_RANGE, t_range=self.T_RANGE
        )
        enhanced_visualizer.create_interactive_plot(**self.pde_params)

    def display_dashboard(self):
        """Display the complete dashboard UI."""
        # Initialize parameter containers
        self._on_pde_change({'new': self.pde_dropdown.value})
        self._on_loss_fn_change({'new': self.loss_fn_dropdown.value})

        # Create the layout
        training_params_box = widgets.VBox([
            self.epochs_slider, self.lr_float, self.patience_slider
        ])

        controls = widgets.VBox([
            self.pde_dropdown, self.pde_params_container,
            widgets.HTML("<hr><b>Balancer Settings:</b>"),
            self.loss_fn_dropdown, self.loss_params_container,
            widgets.HTML("<hr><b>Cross-Validation:</b>"),
            self.kfold_checkbox, self.splits_slider,
            widgets.HTML("<hr><b>Training Settings:</b>"),
            training_params_box, self.run_button
        ], layout={'border': '1px solid #ccc', 'padding': '10px', 'width': '400px'})

        display(widgets.HBox([controls, self.output_area]))


# %%
# Cell 7: Simple Weighted Loss
# ----------------------------
# UPDATED: This class now implements the `step` method to be compatible
# with the modular PINNTrainer.

class SimpleWeightedLoss(BaseLoss):
    """
    A simple loss function that combines components using fixed weights.
    """
    def __init__(self, pde_instance, weights):
        super().__init__(pde_instance)
        # Filter the weights to only include relevant keys for this loss function
        self.weights = {k: v for k, v in weights.items() if k in ['data', 'initial', 'boundary', 'pde']}

    def step(self, losses, optimizer, dataset, model):
        """
        Combines losses with fixed weights and performs one optimization step.
        CHANGE 1: Now calculates PDE loss from raw residuals.
        """
        # 1. Calculate the PDE loss from raw residuals
        residual = self.calculate_residuals(model, dataset['collocation']['coords'])
        losses['pde'] = self.mse_loss(residual, torch.zeros_like(residual))

        # 2. Combine losses to get a single total_loss value
        total_loss = torch.tensor(0.0, device=next(model.parameters()).device, requires_grad=True)
        for key, value in losses.items():
            total_loss = total_loss + self.weights.get(key, 1.0) * value

        # 3. Perform the standard optimization procedure
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # 4. Return the total loss for logging
        return total_loss

    @staticmethod
    def get_params():
        """
        Defines the tunable weights for the UI.
        Format: {name: (default, min, max, step)}
        """
        return {
            'data': (1.0, 0.0, 100.0, 0.5),
            'initial': (1.0, 0.0, 100.0, 0.5),
            'boundary': (1.0, 0.0, 100.0, 0.5),
            'pde': (0.01, 0.0, 1.0, 0.01)
        }


# %%
# Cell 8: Baseline Experiment Execution
# --------------------------------------
# This cell uses the PINNDashboard class to create and display an interactive
# UI for running our baseline experiments. For now, we only provide the
# "SimpleWeightedLoss" as an option for the loss balancer.


# --- 1. Define the maps of available components for the dashboard ---

# Map of available Partial Differential Equations
pde_map = {
    '1D Heat Equation': HeatEquation1D,
    '1D Wave Equation': WaveEquation1D,
    '1D Viscous Burgers': ViscousBurgers1D
}

# Map of available Loss Functions (Optimizers)
# For now, this only contains our baseline method.
loss_fn_map = {
    "Simple Weighted Sum": SimpleWeightedLoss,
}


# --- 2. Instantiate and display the dashboard ---
dashboard = PINNDashboard(pde_map=pde_map, loss_fn_map=loss_fn_map, use_kfold=True, n_splits=3)
dashboard.display_dashboard()


# %% [markdown]
#  ### How Gaussian Likelihood Weighting (lbPINN) Works
# 
#  https://arxiv.org/pdf/2104.06217
# 
# 
# 
#  #### Intuitive Explanation
# 
# 
# 
#  Imagine the total loss function as a final exam score for your neural network, and each individual loss term (PDE, boundary, data) is a different subject on that exam. In a standard PINN, you manually decide the importance of each subject (e.g., "Physics is worth 50%, Boundary is worth 50%"). This is difficult to get right and might not be optimal throughout the entire learning process.
# 
# 
# 
#  The **lbPINN method** takes a different approach. Instead of fixed weights, it assigns a learnable "noise" parameter ($\epsilon$) to each subject (loss term).
# 
# 
# 
#  1.  **Noise as Uncertainty:** You can think of this noise parameter, $\epsilon$, as the model's estimate of its own uncertainty for that specific task. If a loss term is very large (the model is performing poorly on that subject), the model can "admit" it's very uncertain about it by increasing that term's noise parameter, $\epsilon$.
# 
# 
# 
#  2.  **Weighting by Confidence:** The actual weight applied to each loss term in the final calculation is inversely proportional to this uncertainty ($w \propto 1/\epsilon^2$). This means if the model is very uncertain ($\epsilon$ is high), it gives that loss term a *low weight*. This cleverly prevents a single, massive loss term from dominating the training process early on.
# 
# 
# 
#  3.  **The Catch (Regularization):** To prevent the model from simply becoming "uncertain" about everything (i.e., setting all $\epsilon$ values to infinity to ignore all losses), a penalty term, $log(\epsilon)$, is added to the final loss. This penalty term pushes the model to *reduce* its uncertainty (decrease $\epsilon$) over time.
# 
# 
# 
#  The result is a dynamic balancing act. The optimizer seeks to minimize the total loss. To do this, it must reduce both the weighted loss terms and the `log ε` penalty. This encourages the model to first reduce the actual error of a specific loss term, which then allows it to confidently decrease that term's noise parameter ($\epsilon$), which in turn *increases* its weight, forcing the model to pay even closer attention to it. This process happens automatically and simultaneously for all loss terms.
# 
# 
# 
#  #### The Underlying Method
# 
# 
# 
#  The paper formalizes this by establishing a Gaussian probabilistic model for each loss term. The likelihood of an observation is assumed to be a Gaussian distribution, with the mean being the neural network's output and the standard deviation being the learnable noise parameter, $\epsilon$.
# 
# 
# 
#  Maximizing the log-likelihood of this model is equivalent to minimizing its negative log-likelihood, which for a single loss term $L_i(\theta)$ becomes:
# 
# 
# 
#  $$\mathcal{L}_i(\theta, \epsilon_i) \propto \frac{1}{2\epsilon_i^2}L_i(\theta) + \log \epsilon_i$$
# 
# 
# 
#  The total loss function is the sum over all components (PDE, boundary, initial, data), as shown in the paper's Equation (8):
# 
# 
# 
#  $$\mathcal{L}(\epsilon, \theta, N) = \frac{1}{2\epsilon_f^2}L_{PDE} + \frac{1}{2\epsilon_b^2}L_{BC} + \frac{1}{2\epsilon_i^2}L_{IC} + \frac{1}{2\epsilon_d^2}L_{data} + \log(\epsilon_f \epsilon_b \epsilon_i \epsilon_d)$$
# 
# 
# 
#  The key is that both the network weights ($\theta$) and the noise parameters ($\epsilon$) are treated as trainable parameters and are optimized simultaneously using a gradient descent algorithm like Adam.
# 
# 
# 
# 

# %%
# Cell 9: Gaussian Likelihood Loss (lbPINN)
# -------------------------------------------
# UPDATED: This class now implements the `step` method to be compatible
# with the modular PINNTrainer.

class GaussianLikelihoodLoss(BaseLoss):
    """
    Combines loss components using learnable, uncertainty-based weights (lbPINN).
    The noise parameters (epsilon) are optimized along with the model weights.
    """
    def __init__(self, pde_instance, weights={'pde': 2.0, 'data': 2.0, 'initial': 2.0, 'boundary': 2.0}):
        super().__init__(pde_instance)
        # Create learnable parameters for the noise terms (epsilon)
        self.log_eps_pde = nn.Parameter(torch.tensor(float(weights.get('pde', 2.0))).log())
        self.log_eps_data = nn.Parameter(torch.tensor(float(weights.get('data', 2.0))).log())
        self.log_eps_initial = nn.Parameter(torch.tensor(float(weights.get('initial', 2.0))).log())
        self.log_eps_boundary = nn.Parameter(torch.tensor(float(weights.get('boundary', 2.0))).log())

    def step(self, losses, optimizer, dataset, model):
        """
        Combines losses with the lbPINN scheme and performs one optimization step.
        CHANGE 1: Now calculates PDE loss from raw residuals.
        """
        # 1. Calculate the PDE loss from raw residuals
        residual = self.calculate_residuals(model, dataset['collocation']['coords'])
        losses['pde'] = self.mse_loss(residual, torch.zeros_like(residual))

        # 2. Combine the losses to get a single total_loss value
        log_eps_pde = torch.clamp(self.log_eps_pde, min=-4.0)
        log_eps_data = torch.clamp(self.log_eps_data, min=-4.0)
        log_eps_initial = torch.clamp(self.log_eps_initial, min=-4.0)
        log_eps_boundary = torch.clamp(self.log_eps_boundary, min=-4.0)

        eps2_pde = torch.exp(log_eps_pde * 2)
        eps2_data = torch.exp(log_eps_data * 2)
        eps2_initial = torch.exp(log_eps_initial * 2)
        eps2_boundary = torch.exp(log_eps_boundary * 2)

        weighted_loss = (
            (0.5 / eps2_pde) * losses['pde'] +
            (0.5 / eps2_data) * losses['data'] +
            (0.5 / eps2_initial) * losses['initial'] +
            (0.5 / eps2_boundary) * losses['boundary']
        )
        regularization = log_eps_pde + log_eps_data + log_eps_initial + log_eps_boundary
        total_loss = weighted_loss + regularization

        # 3. Perform the standard optimization procedure
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # 4. Return the total loss for logging
        return total_loss

    @staticmethod
    def get_params():
        """
        Defines the tunable initial epsilon values for the UI.
        """
        return {
            'pde': (2.0, 0.1, 10.0, 0.1),
            'data': (2.0, 0.1, 10.0, 0.1),
            'initial': (2.0, 0.1, 10.0, 0.1),
            'boundary': (2.0, 0.1, 10.0, 0.1)
        }


# %%
# Cell 10: Full Experimental Dashboard with lbPINN
# ------------------------------------------------
# This cell re-instantiates the dashboard, now including the lbPINN
# method as a selectable option for comparative analysis.

# --- 1. Define the maps of available components for the dashboard ---

# Map of available Partial Differential Equations
pde_map = {
    '1D Heat Equation': HeatEquation1D,
    '1D Wave Equation': WaveEquation1D,
    '1D Viscous Burgers': ViscousBurgers1D
}

# Map of available Loss Functions (Optimizers)
# This map now includes both the baseline and the new method.
loss_fn_map = {
    "Simple Weighted Sum": SimpleWeightedLoss,
    "lbPINN (Gaussian Likelihood)": GaussianLikelihoodLoss
}


# --- 2. Instantiate and display the dashboard ---
# This creates the UI with all available options.
# The PINNDashboard class is defined in Cell 6.
# The Loss Function classes are defined in Cells 7 and 8.
# The PINNTrainer class (updated) is defined in Cell 5.

dashboard = PINNDashboard(pde_map=pde_map, loss_fn_map=loss_fn_map, use_kfold=True, n_splits=3)
dashboard.display_dashboard()


# %% [markdown]
#  # dwPINN: Dynamic Weight Strategy
# 
# 
# 
#  https://pmc.ncbi.nlm.nih.gov/articles/PMC9497516/#:~:text=The%20sequence%20of%20weights%20,The%20convergence%20of%20the%20weight
# 
# ### The Intuition: A Focused Tutor for Your Neural Network
# 
# Imagine training a neural network is like teaching a student multiple subjects at once: one subject is fitting the initial conditions, another is fitting the boundary conditions, and a third is making sure the underlying physics equation (the PDE) is satisfied.
# 
# A standard approach gives each subject a fixed importance (e.g., "the boundary is worth 30% of the final grade"). This is often suboptimal, as the student might struggle with one subject far more than the others.
# 
# The **dwPINN** method acts like an intelligent, focused tutor. At each step of the training, the tutor looks at the student's performance on all subjects and asks, "Where are you struggling the most right now?". It then tells the student to focus all their effort on that single, most difficult subject.
# 
# This is achieved through a competitive process called a **min-max game**:
# 
# 1.  **The "Max" Step (Tutor's Job):** The algorithm first finds the loss term with the highest error and dynamically increases its corresponding weight to *maximize* the total penalty.
# 2.  **The "Min" Step (Student's Job):** The neural network then performs a standard training step to *minimize* this new, heavily-weighted total loss.
# 
# This process repeats, forcing the network to sequentially address its biggest weaknesses, which leads to more balanced and efficient training.
# 
# ### The Method with Formulas
# 
# The dwPINN method formalizes this intuition with a simple but powerful set of equations.
# 
# #### 1. The Weighted Loss Function
# 
# First, a trainable weight ($w$) is assigned to each component of the total loss function ($\mathcal{J}$). The components are the Mean Squared Errors (MSE) for the initial data ($MSE_0$), boundary data ($MSE_b$), PDE residuals ($MSE_f$), and any other observation data ($MSE_u$).
# 
# $$\mathcal{J} = w_{u}MSE_{u} + w_{0}MSE_{0} + w_{b}MSE_{b} + w_{f}MSE_{f}$$
# 
# #### 2. The Min-Max Objective
# 
# The goal is to find the network parameters ($\tilde{\Theta}$) that minimize the loss, while simultaneously finding the weights ($w$) that maximize it. This creates the min-max objective:
# 
# $$\min_{\tilde{\Theta}} \max_{w} \mathcal{J}(\tilde{\Theta}, w)$$
# 
# 
# #### 3. The Update Rule
# 
# This objective is solved by alternating between gradient descent for the network parameters and gradient ascent for the weights[cite: 100].
# 
# * **Network Update (Minimization):** The network parameters $\tilde{\Theta}$ are updated using standard gradient descent to reduce the total loss $\mathcal{J}$.
# 
#     $$\tilde{\Theta}^{k+1} = \tilde{\Theta}^{k} - \eta_{k} \nabla_{\tilde{\Theta}}\mathcal{J}$$
# 
# 
# * **Weight Update (Maximization):** This is the core of the method. Each weight is updated using gradient ascent[cite: 100]. For example, the update for the initial condition weight, $w_0$, is:
# 
#     $$w_{0}^{k+1} = w_{0}^{k} + \eta_{w}^{k} \nabla_{w_{0}}\mathcal{J}$$
# 
# 
#     The paper shows that the gradient of $\mathcal{J}$ with respect to a weight $w_0$ is simply the corresponding loss term, $MSE_0$. This simplifies the update rule to a very intuitive formula:
# 
#     $$w_{0}^{k+1} = w_{0}^{k} + \eta_{w}^{k} \cdot MSE_{0}^{k}$$
# 
# This final equation clearly shows the method's logic: the weight $w_0$ is increased at each step by an amount proportional to its current error, $MSE_0$. A larger error results in a larger update, progressively penalizing the network for not fitting those points closely. The paper proves that this makes the weights **monotonically non-decreasing** and convergent during training.

# %%
# Cell 11: Dynamic Weight Loss (dwPINN) - Improved Implementation
# -----------------------------------------------------------------
# This class has been rewritten to exactly match the min-max algorithm and
# gradient ascent update rule described in the paper:
# "Dynamic Weight Strategy of Physics-Informed Neural Networks for the 2D Navier-Stokes Equations"
# by Li and Feng (2022).

class DWPINNLoss(BaseLoss):
    """
    Implements the Dynamic Weight (dwPINN) strategy based on the min-max
    optimization proposed by Li and Feng (2022).
    """
    def __init__(self, pde_instance, weights={'weight_lr': 0.01, 'use_conditional_update': True}):
        """
        Initializes the loss function according to the paper's description.

        Args:
            pde_instance: An instance of the PDE class.
            weights (dict): A dictionary of hyperparameters.
                - 'weight_lr' (float): The learning rate for the weight update (η_w in the paper).
                - 'use_conditional_update' (bool): If True, enables the stability improvement from
                  Algorithm 1, where weights are only updated if the corresponding loss has decreased.
        """
        super().__init__(pde_instance)
        self.loss_keys = ['data', 'initial', 'boundary', 'pde']
        self.weight_lr = weights.get('weight_lr', 0.01)
        self.use_conditional_update = weights.get('use_conditional_update', True)

        # Initialize weights to 1.0. The paper states they should be initialized to a non-negative value.
        # We use nn.Parameter with requires_grad=False to store state that is moved to the correct device by the trainer.
        initial_w = torch.ones(len(self.loss_keys), dtype=torch.float32)
        self.w = nn.Parameter(initial_w, requires_grad=False)

        # Store previous losses for the conditional update in Algorithm 1 
        self.prev_losses = None
        self.last_lambdas = {}

    def step(self, losses, optimizer, dataset, model):
        """
        Performs a full optimization step according to the dwPINN paper.
        This involves:
        1. A gradient descent step for the network parameters (minimization).
        2. A gradient ascent step for the loss weights (maximization).
        """
        # 1. Calculate the PDE loss from raw residuals
        residual = self.calculate_residuals(model, dataset['collocation']['coords'])
        losses['pde'] = self.mse_loss(residual, torch.zeros_like(residual))

        current_losses = torch.stack([losses[key] for key in self.loss_keys])

        # Ensure weights are on the same device as the losses
        if self.w.device != current_losses.device:
            self.w.data = self.w.data.to(current_losses.device)
            if self.prev_losses is not None:
                self.prev_losses = self.prev_losses.to(current_losses.device)

        # 2. Calculate the total weighted loss for the network parameter update
        total_loss = torch.sum(self.w * current_losses)

        # 3. Perform the gradient DESCENT step for the network parameters (the "min" part of min-max) 
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # 4. Perform the gradient ASCENT step for the weights (the "max" part of min-max) 
        with torch.no_grad():
            # The gradient of the objective function J w.r.t. a weight w_i is just the loss MSE_i.
            # So, the gradient ascent update is w_i = w_i + lr * MSE_i.
            grad_w = current_losses.detach()

            # Implement the conditional update from Algorithm 1 for stability 
            if self.use_conditional_update and self.prev_losses is not None:
                # Create a mask for which weights to update. Update only if loss has decreased.
                update_mask = (grad_w < self.prev_losses).float()
                self.w.data += self.weight_lr * grad_w * update_mask
            else:
                # Perform the standard update if it's the first step or the condition is disabled
                self.w.data += self.weight_lr * grad_w
            
            # The weights must be non-negative.
            self.w.data.clamp_(min=0)

        # Store current losses for the next step's conditional check
        self.prev_losses = current_losses.detach().clone()
        
        # Log the current weights for visualization
        self.last_lambdas = {key: self.w[i].item() for i, key in enumerate(self.loss_keys)}

        return total_loss

    @staticmethod
    def get_params():
        """Defines the tunable hyperparameters for the UI."""
        return {
            'weight_lr': (0.01, 0.0001, 0.1, 0.0001),
            # In a real UI, this would be a checkbox. We represent it as a slider 0.0 (False) to 1.0 (True).
            'use_conditional_update': (1.0, 0.0, 1.0, 1.0)
        }

# %%
# Cell 12: Full Experimental Dashboard with dwPINN
# -----------------------------------------------
# This cell integrates the newly implemented DWPINNLoss into the interactive
# PINNDashboard, allowing for direct comparison with the previous methods.

# Define the map of available loss functions, now including dwPINN
loss_fn_map = {
    "Simple Weighted Sum": SimpleWeightedLoss,
    "lbPINN (Gaussian Likelihood)": GaussianLikelihoodLoss,
    "dwPINN (Dynamic Weights)": DWPINNLoss  # Add the new method here
}

# Define the map of available PDEs
pde_map = {
    "1D Heat Equation": HeatEquation1D,
    "1D Wave Equation": WaveEquation1D,
    "1D Viscous Burgers": ViscousBurgers1D
}

# Instantiate and display the final dashboard
full_dashboard = PINNDashboard(pde_map, loss_fn_map, use_kfold=True, n_splits=3)
full_dashboard.display_dashboard()


# %% [markdown]
#  ### ReLoBRaLo: Explained
# 
# 
# 
#  https://arxiv.org/pdf/2110.09813
# 
# 
# 
#  ReLoBRaLo stands for **Re**lative **Lo**ss **B**alancing with **Ra**ndom **Lo**okback. At its core, it's a sophisticated method that tries to give more weight to the loss terms that are "falling behind" during training. It uses two key signals to decide which term is falling behind.
# 
# 
# 
#  #### Intuitive Explanation
# 
# 
# 
#  Imagine you are training a student (the PINN) on several subjects at once (the loss terms: PDE, boundary, initial, data).
# 
# 
# 
#  1.  **Are you keeping up? (The "ReLo" Part - Relative Loss)**
# 
#      * The method first looks at each subject's current score (`L_i`) and compares it to their score at the very beginning of the semester (`L_i(0)`). This ratio, `L_i / L_i(0)`, shows the *relative progress*.
# 
#      * If a subject's score hasn't improved much (the ratio is high), it means the student is struggling with it. ReLoBRaLo gives a higher weight to this struggling subject to make the student focus on it. To avoid reacting to noisy, one-off bad scores, it smooths this ratio using a moving average.
# 
# 
# 
#  2.  **Are you putting in enough effort? (The "Bra" Part - Gradient Balancing)**
# 
#      * Sometimes a student might be struggling not because the subject is hard, but because they aren't "sending enough brainpower" to it. In neural network terms, this is the gradient.
# 
#      * ReLoBRaLo checks the magnitude of the gradient each loss term sends to the network. If a loss term has a very small gradient, its "voice" is too quiet to influence the network's learning.
# 
#      * The method boosts the weight of terms with smaller relative gradients to ensure every subject has a voice in the training process.
# 
# 
# 
#  3.  **Let's not dwell on the past (The "RaLo" Part - Random Lookback)**
# 
#      * What if the student got a nearly perfect score on a subject in the first week and then got stuck? The "relative progress" ratio would always be tiny, so the algorithm would ignore that subject forever.
# 
#      * To prevent this, ReLoBRaLo has a "reset" button. Every so often (with a random probability `p`), it tells the student to forget their initial scores and use their *current* scores as the new baseline. This allows the system to dynamically re-evaluate which subjects are *currently* the most challenging.
# 
# 
# 
#  #### The Formulas
# 
# 
# 
#  Here are the key equations from the paper that formalize this intuition:
# 
# 
# 
#  1.  **Relative Loss Ratio:** At each step `t`, calculate the ratio of the current loss to the initial loss.
# 
#      > $\hat{L}_i(t) = L_i(\theta_t) / L_i(0)$
# 
# 
# 
#  2.  **Smoothed Ratio (Moving Average):** Smooth the ratio using a hyperparameter $\alpha$.
# 
#      > $\bar{L}_i(t) = (1-\alpha)\bar{L}_i(t-1) + \alpha \hat{L}_i(t)$
# 
# 
# 
#  3.  **Gradient Balancing Term:** Compute the norm of the gradient for each loss term, and normalize it by the gradient norm of the total weighted loss. This measures the relative "effort".
# 
#      > $\hat{g}_i(t) = \frac{||\nabla_\theta L_i(\theta_t)||_2}{||\nabla_\theta(\sum_j \lambda_j L_j(\theta_t))||_2 + \epsilon}$
# 
# 
# 
#  4.  **Weight Update Rule:** Combine the smoothed relative loss and the gradient term to update the weights ($\lambda_i$) using a softmax function, which makes the weights compete. `T` is a temperature parameter that controls the sharpness of the softmax.
# 
#      > $\lambda_i(t+1) = \lambda_i(t) \cdot \text{softmax}_i\left(\frac{\bar{L}_i(t)}{T \cdot \hat{g}_i(t)}\right)$
# 
# 
# 
#  5.  **Random Lookback:** With probability `p`, reset the baseline losses.
# 
#      > If `rand()` < `p`, then set $L_i(0) = L_i(\theta_t)$.
# 
# 
# 
# 

# %%
# Cell 13: ReLoBRaLo Loss Function Implementation (Improved)
# -------------------------------------------------------------
# This cell implements the Relative Loss Balancing with Random Lookback (ReLoBRaLo) method
# with two key improvements:
# 1. Direct Softmax assignment for \u03bb update (eqn (12) in the paper)
# 2. More aggressive lookback probability (default \u03c1=0.1) and adjustable \u03b1

import torch
import torch.nn as nn

class ReLoBRaLoLoss(BaseLoss):
    """
    Implements the Relative Loss Balancing with Random Lookback (ReLoBRaLo) method.
    Improvements:
      - Direct lambda assignment via softmax
      - Default rho increased to 0.1 for more frequent lookbacks
    """
    def __init__(self, pde_instance, model, weights={'alpha': 0.9, 'rho': 0.1, 'T': 2.0}):
        super().__init__(pde_instance)
        self.model = model
        self.loss_keys = ['data', 'initial', 'boundary', 'pde']
        num_losses = len(self.loss_keys)

        # Hyperparameters
        self.alpha = weights.get('alpha', 0.9)   # moving-average factor
        self.rho   = weights.get('rho', 0.1)     # lookback probability
        self.T     = weights.get('T', 2.0)       # softmax temperature
        self.epsilon = 1e-8

        # Resolve device from model parameters
        self.device = next(model.parameters()).device

        # State variables (on correct device)
        self.lambdas = nn.Parameter(
            torch.full((num_losses,), 1.0 / num_losses, device=self.device),
            requires_grad=False
        )
        self.initial_losses = nn.Parameter(
            torch.zeros(num_losses, device=self.device), requires_grad=False
        )
        self.moving_average_ratios = nn.Parameter(
            torch.ones(num_losses, device=self.device), requires_grad=False
        )

        self.first_step_done = False
        self.last_lambdas = {}

    @staticmethod
    def get_params():
        return {
            'alpha': (0.9, 0.0, 0.9999, 0.0001),
            'rho':   (0.1, 0.0, 0.5,    0.01),
            'T':     (2.0, 0.1, 10.0,   0.1)
        }

    def step(self, losses, optimizer, dataset, model):
        """
        CHANGE 1: Now calculates PDE loss from raw residuals and uses provided model.
        """
        # 1. Calculate the PDE loss from raw residuals
        residual = self.calculate_residuals(model, dataset['collocation']['coords'])
        losses['pde'] = self.mse_loss(residual, torch.zeros_like(residual))

        current_losses = torch.stack([losses[k] for k in self.loss_keys])

        # --- First step: initialize baselines and do a simple update ---
        if not self.first_step_done:
            with torch.no_grad():
                self.initial_losses.copy_(current_losses.detach())
            self.first_step_done = True

            total_loss = torch.sum(self.lambdas * current_losses)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            return total_loss

        # 2. Random lookback: occasionally reset baseline
        if torch.rand(1).item() < self.rho:
            with torch.no_grad():
                self.initial_losses.copy_(current_losses.detach())

        # 3. Update moving-average of relative ratios
        with torch.no_grad():
            ratios = current_losses / (self.initial_losses + self.epsilon)
            new_ma = (1 - self.alpha) * self.moving_average_ratios + self.alpha * ratios
            self.moving_average_ratios.copy_(new_ma)

        # 4. Compute gradient norms for each loss component
        grad_norms = []
        for key in self.loss_keys:
            grads = torch.autograd.grad(losses[key], model.parameters(),
                                        retain_graph=True, allow_unused=True)
            flat = torch.cat([g.flatten() for g in grads if g is not None])
            grad_norms.append(torch.norm(flat, p=2))
        grad_norms_tensor = torch.stack(grad_norms)

        # Compute total gradient norm for normalization
        total_for_grad = torch.sum(self.lambdas * current_losses)
        grads_total = torch.autograd.grad(total_for_grad, model.parameters(),
                                          retain_graph=True, allow_unused=True)
        flat_total = torch.cat([g.flatten() for g in grads_total if g is not None])
        norm_total = torch.norm(flat_total, p=2)

        g_hat = grad_norms_tensor / (norm_total + self.epsilon)

        # 5. Direct softmax assignment of lambdas (eqn (12))
        with torch.no_grad():
            logits = self.moving_average_ratios / (self.T * g_hat + self.epsilon)
            new_lambdas = torch.softmax(logits, dim=0)
            self.lambdas.copy_(new_lambdas)

        # 6. Final step with updated lambdas
        final_loss = torch.sum(self.lambdas * current_losses)
        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()

        # Logging for dashboard
        self.last_lambdas = {k: self.lambdas[i].item() for i, k in enumerate(self.loss_keys)}

        return final_loss


# %%
# Cell 14: Final Dashboard with All Methods including ReLoBRaLo
# -----------------------------------------------------------
# This cell integrates all implemented methods, including the advanced
# ReLoBRaLo, into the final interactive PINNDashboard for a full
# comparative analysis.

# Define the map of available loss functions, now including all methods
loss_fn_map = {
    "Simple Weighted Sum": SimpleWeightedLoss,
    "lbPINN (Gaussian Likelihood)": GaussianLikelihoodLoss,
    "dwPINN (Dynamic Weights)": DWPINNLoss,
    "ReLoBRaLo": ReLoBRaLoLoss  # Add the new ReLoBRaLo method
}

# Define the map of available PDEs
pde_map = {
    "1D Heat Equation": HeatEquation1D,
    "1D Wave Equation": WaveEquation1D,
    "1D Viscous Burgers": ViscousBurgers1D
}

# Instantiate and display the final dashboard
final_dashboard = PINNDashboard(pde_map, loss_fn_map, use_kfold=True, n_splits=5)
final_dashboard.display_dashboard()


# %% [markdown]
#  ### How BRDR Works: An Intuitive Explanation
# 
# 
# 
#  https://arxiv.org/abs/2407.01613
# 
# 
# 
#  The core idea behind BRDR is to solve a common problem in PINN training: the network often learns some parts of the physical domain very quickly while getting "stuck" on others. BRDR acts like a focused tutor that identifies exactly which training points are struggling and forces the model to pay more attention to them.
# 
# 
# 
#  1.  **Identifying Struggling Points:** The method doesn't just look at the overall loss (like PDE vs. Boundary). Instead, it monitors the learning progress of *every single collocation point* individually. It does this by calculating a metric called the **"inverse residual decay rate" (irdr)** for each point. A high `irdr` for a point means its residual is not decreasing; it's a "stubborn" point that the model is failing to learn.
# 
# 
# 
#  2. **Weighting by "Stubbornness":** The strategy is simple: give a higher weight to the points that are more stubborn (i.e., have a higher `irdr`). This prevents the model from ignoring difficult regions (like areas with sharp changes or high-frequency behavior) and focusing only on the "easy" parts of the solution.
# 
# 
# 
#  3. **Ensuring Stability:** To make the process stable, the weights are smoothed out over time using a moving average. Crucially, the collection of all point-wise weights is normalized so that their average is always 1. This keeps the weights bounded and prevents them from exploding, which is a problem some other adaptive methods can face.
# 
# 
# 
#  In essence, BRDR dynamically re-focuses the model's attention during training, forcing it to achieve a more uniform convergence across the entire problem domain, which leads to higher accuracy and faster training.
# 
# 
# 
#  #### The Formulas
# 
# 
# 
#  The BRDR algorithm is based on the following key calculations from the paper:
# 
# 
# 
#  1.  **Inverse Residual Decay Rate (irdr):** For each point, the `irdr` is the ratio of its current squared residual, $R^2(t)$, to the moving average of its past residuals. This is a measure of how much the current residual stands out from its recent history.
# 
# 
# 
#      $$irdr = \frac{R^2(t)}{\sqrt{\overline{R^4(t)} + \epsilon}}$$
# 
#      where $\overline{R^4(t)}$ is the exponential moving average (EMA) of the residual to the fourth power.
# 
# 
# 
#  2.  **EMA Update:** The moving average is updated at each step for each point:
# 
#      $$\overline{R^4(t)} = \beta_c \overline{R^4(t-1)} + (1 - \beta_c) R^4(t)$$
# 
#      where $\beta_c$ is a smoothing factor, typically close to 1.
# 
# 
# 
#  3.  **Weight Update:** The actual point-wise weights, $w_t$, are updated using their own EMA, driven by a reference weight, $w_t^{ref}$, that is proportional to the calculated `irdr`.
# 
#      $$w_t^{ref} = \frac{irdr_t}{\text{mean}(irdr_t)}$$
# 
#      $$w_t = \beta_w w_{t-1} + (1 - \beta_w) w_t^{ref}$$
# 
#      [cite_start]This ensures the weights adapt smoothly but purposefully toward the points with the slowest decay rates.
# 
# 

# %%
# Cell 15: BRDR Loss Function Implementation
# -------------------------------------------
# This cell implements the Balanced Residual Decay Rate (BRDR) method.
# It leverages the refactored framework to perform point-wise weighting
# on the PDE residuals.

import torch
import torch.nn as nn

class BRDRLoss(BaseLoss):
    """
    Complete implementation of the Balanced Residual Decay Rate (BRDR) method.
    Features:
    1. Point-wise weighting for ALL loss components (PDE, IC, BC, data)
    2. Adaptive scale factor 's' for dynamic learning rate adjustment
    3. Global normalization across all training points
    """
    def __init__(self, pde_instance, model, weights={'beta_c': 0.999, 'beta_w': 0.999}):
        super().__init__(pde_instance)
        self.model = model

        # Loss component keys
        self.loss_keys = ['data', 'initial', 'boundary', 'pde']

        # Hyperparameters for the moving averages
        self.beta_c = weights.get('beta_c', 0.999)  # For residual decay rate
        self.beta_w = weights.get('beta_w', 0.999)  # For the weights themselves
        self.epsilon = 1e-8

        # State variables for each loss component - will be initialized on first step
        self.point_weights = {}
        self.ema_residuals_sq_sq = {}
        self._initialized = False

        # Adaptive scale factor
        self.s = 1.0

        # For logging
        self.last_lambdas = {}

    def step(self, losses, optimizer, dataset, model):
        """
        Performs a full training step using the complete BRDR method with:
        1. Point-wise weighting for all components
        2. Adaptive scale factor
        """
        # 1. Get both scalar losses and raw residuals for all components
        scalar_losses, residuals = self.forward_with_residuals(model, dataset)

        # 2. Calculate PDE residual tensor
        pde_residual = self.calculate_residuals(model, dataset['collocation']['coords']).squeeze()
        residuals['pde_res'] = pde_residual
        scalar_losses['pde'] = self.mse_loss(pde_residual, torch.zeros_like(pde_residual))

        losses['pde'] = scalar_losses['pde']

        # 3. Store all irdr values for global normalization
        all_irdr_values = []

        # 4. Initialize state variables on first step
        if not self._initialized:
            for key in self.loss_keys:
                current_residual = residuals[f'{key}_res']
                self.point_weights[key] = torch.ones_like(current_residual, device=current_residual.device)
                self.ema_residuals_sq_sq[key] = torch.zeros_like(current_residual, device=current_residual.device)
            self._initialized = True

        # 5. Process each loss component
        for key in self.loss_keys:
            current_residual = residuals[f'{key}_res']

            with torch.no_grad():
                current_residual_sq = current_residual.detach()**2

                # Update EMA of R^4 for this component
                self.ema_residuals_sq_sq[key].copy_(
                    self.beta_c * self.ema_residuals_sq_sq[key] +
                    (1 - self.beta_c) * (current_residual_sq**2)
                )

                # Calculate inverse residual decay rate (irdr) for this component
                irdr = current_residual_sq / (torch.sqrt(self.ema_residuals_sq_sq[key]) + self.epsilon)
                all_irdr_values.append(irdr)

        # 5. Global normalization: calculate mean across ALL points from ALL components
        with torch.no_grad():
            global_irdr_tensor = torch.cat(all_irdr_values)
            global_mean_irdr = torch.mean(global_irdr_tensor)

            # Update weights using global normalization
            for i, key in enumerate(self.loss_keys):
                component_irdr = all_irdr_values[i]

                # Normalize using GLOBAL mean (key insight from paper)
                w_ref = component_irdr / (global_mean_irdr + self.epsilon)

                # Update point-wise weights with exponential moving average
                self.point_weights[key].copy_(
                    self.beta_w * self.point_weights[key] + (1 - self.beta_w) * w_ref
                )

        # 6. Calculate weighted losses for each component
        final_losses = {}
        for key in self.loss_keys:
            current_residual = residuals[f'{key}_res']
            final_losses[key] = torch.mean(self.point_weights[key] * (current_residual**2))

        # 7. Calculate total loss
        total_loss = torch.stack(list(final_losses.values())).sum()

        # 8. Implement adaptive scale factor 's' (Algorithm 1 from paper)
        # First, get gradients to calculate gradient norm
        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)

        # Calculate squared L2 norm of gradients
        grad_norm_sq = sum(
            p.grad.detach().pow(2).sum()
            for p in model.parameters()
            if p.grad is not None
        )

        # Update adaptive scale factor
        s_old = self.s
        s_max = (2 * total_loss.detach()) / (grad_norm_sq + self.epsilon)
        learning_rate = optimizer.param_groups[0]['lr']
        beta_s = 1.0 - learning_rate  # As suggested in the paper
        self.s = beta_s * s_old + (1 - beta_s) * s_max

        # Apply scale factor correction to gradients (Algorithm 1)
        with torch.no_grad():
            scale_factor = self.s / s_old if s_old != 0 else 1.0
            for param in model.parameters():
                if param.grad is not None:
                    param.grad *= scale_factor

        # 9. Apply optimizer step with corrected gradients
        optimizer.step()

        # 10. Logging
        self.last_lambdas = {
            f'mean_{key}_weight': torch.mean(self.point_weights[key]).item()
            for key in self.loss_keys
        }
        # Convert scale factor to float if it's a tensor
        if isinstance(self.s, torch.Tensor):
            self.last_lambdas['scale_factor_s'] = self.s.item()
        else:
            self.last_lambdas['scale_factor_s'] = self.s

        return total_loss

    @staticmethod
    def get_params():
        """Returns the tunable hyperparameters for the UI."""
        return {
            'beta_c': (0.999, 0.9, 0.9999, 0.0001),
            'beta_w': (0.999, 0.9, 0.9999, 0.0001),
        }


# %%
# Cell 16: Final Dashboard with All Methods including BRDR
# -----------------------------------------------------------
# This cell integrates all implemented methods, now including BRDR,
# into the final interactive PINNDashboard for a full
# comparative analysis.

# Define the map of available loss functions, now including all methods
loss_fn_map = {
    "Simple Weighted Sum": SimpleWeightedLoss,
    "lbPINN (Gaussian Likelihood)": GaussianLikelihoodLoss,
    "dwPINN (Dynamic Weights)": DWPINNLoss,
    "ReLoBRaLo": ReLoBRaLoLoss,
    "BRDR (Balanced Decay Rate)": BRDRLoss # Add the new BRDR method
}

# Define the map of available PDEs
pde_map = {
    "1D Heat Equation": HeatEquation1D,
    "1D Wave Equation": WaveEquation1D,
    "1D Viscous Burgers": ViscousBurgers1D
}

# Instantiate and display the final dashboard
final_dashboard = PINNDashboard(pde_map, loss_fn_map, use_kfold=True, n_splits=5)
final_dashboard.display_dashboard()


# %% [markdown]
# The GradNorm method aims to balance the training of multitask networks by dynamically adjusting the magnitudes of the backpropagated gradients for each task. It achieves this by modifying the multitask loss function, which is typically a weighted sum of individual task losses ($L = \sum_i w_i L_i$). GradNorm introduces an adaptive mechanism where the weights ($w_i$) can change at each training step ($w_i(t)$).
# 
# https://arxiv.org/abs/1711.02257#:~:text=gradient%20normalization%20,alpha
# 
# **GradNorm Algorithm Steps:**
# 
# 1.  **Initialize Weights:** The task weights ($w_i$) are initialized, typically to 1 for all tasks.
# 2.  **Calculate Initial Losses:** At the very beginning of training, the initial loss for each task ($L_i(0)$) is recorded. This serves as a baseline to determine the "training rate" of each task.
# 3.  **Calculate Loss Ratios:** At each training step *t*, the current loss for each task ($L_i(t)$) is divided by its initial loss ($L_i(0)$) to get a loss ratio ($\tilde{L}_i(t) = L_i(t) / L_i(0)$). A lower loss ratio indicates a faster training rate for that task.
# 4.  **Calculate Relative Inverse Training Rates:** The loss ratios are then normalized by the average loss ratio across all tasks to get the relative inverse training rate ($r_i(t) = \tilde{L}_i(t) / E_{task}[\tilde{L}_i(t)]$). If a task is training relatively quickly, its $r_i(t)$ will be lower, and vice versa.
# 5.  **Calculate Gradient Norms:** For each task *i*, the L2 norm of the gradient of the *weighted* single-task loss ($w_i(t)L_i(t)$) with respect to the weights of the last shared layer ($W$) of the network is computed ($G_W^{(i)}(t) = ||\nabla_W w_i(t)L_i(t)||_2$). The last shared layer is chosen to save on computation costs.
# 6.  **Calculate Average Gradient Norm:** The average gradient norm across all tasks is calculated ($\overline{G}_W(t) = E_{task}[G_W^{(i)}(t)]$). This acts as a common scale for the gradients.
# 7.  **Compute Target Gradient Norms:** A target gradient norm for each task is determined by scaling the average gradient norm by the relative inverse training rate raised to an asymmetry hyperparameter $\alpha$ ($G_W^{(i)}(t) \mapsto \overline{G}_W(t) \times [r_i(t)]^\alpha$). A higher $\alpha$ enforces stronger balancing of training rates.
# 8.  **Compute GradNorm Loss ($L_{grad}$):** A loss function, $L_{grad}$, is defined as the sum of the absolute differences between the actual gradient norms and their target gradient norms for each task ($L_{grad}(t; w_i(t)) = \sum_i |G_W^{(i)}(t) - \overline{G}_W(t) \times [r_i(t)]^\alpha|_1$). Importantly, the target gradient norm term ($\overline{G}_W(t) \times [r_i(t)]^\alpha$) is treated as a fixed constant during the differentiation of $L_{grad}$ with respect to $w_i(t)$ to prevent the weights from spuriously drifting to zero.
# 9.  **Update Task Weights ($w_i$):** The computed gradients $\nabla_{w_i}L_{grad}$ are used to update the task weights $w_i$. This step uses a separate optimizer for the weights.
# 10. **Renormalize Weights:** After updating, the weights $w_i(t)$ are renormalized such that their sum equals the number of tasks ($T$) ($\sum_i w_i(t) = T$). This decouples the gradient normalization from the global learning rate.
# 11. **Standard Model Update:** Finally, the standard total loss ($L(t) = \sum_i w_i(t)L_i(t)$) is used to compute gradients with respect to the model's parameters and update the model.
# 

# %%
# Cell 17: GradNorm Loss Function Implementation
# ----------------------------------------------
# This cell implements the GradNorm method, aligning perfectly with the paper.
# It uses the `update_weights` hook to perform its multi-pass gradient calculations.

import torch
import torch.nn as nn




class GradNormLoss(BaseLoss):
    """
    Improved GradNorm implementation with enhanced stability and configurability.
    This is a multi-pass method that requires the modifications made
    to the PINN and PINNTrainer classes to function correctly.
    """
    def __init__(self, pde_instance, model, weights={'alpha': 1.5, 'weight_lr': 0.025}):
        super().__init__(pde_instance)
        self.model = model
        self.alpha = weights.get('alpha', 1.5)
        self.weight_lr = weights.get('weight_lr', 0.025)  # Configurable learning rate
        
        # Validate alpha value for stability
        if self.alpha >= 3.0:
            print(f"Warning: alpha={self.alpha} may cause instability. Consider α < 3.0")

        # GradNorm requires a separate optimizer for the loss weights `w_i`
        # Create weights as a standalone tensor (not a parameter) to avoid duplicate parameter issues
        device = next(model.parameters()).device
        self.weights = torch.ones(len(self.loss_keys), device=device, requires_grad=True)
        self.weight_optimizer = torch.optim.Adam([self.weights], lr=self.weight_lr)

        self.initial_losses = None
        self.last_lambdas = {}
        self.clamp_warnings = set()  # Track which weights triggered warnings
    
    def parameters(self, recurse=True):
        """Override parameters() to exclude the weights parameter from the main optimizer."""
        # Since self.weights is not registered as a parameter (just a tensor attribute),
        # we can simply return the parent class parameters
        return super().parameters(recurse)

    def _get_initial_losses(self, losses):
        """
        Initialize losses with theoretical values where appropriate.
        For PDE tasks, use actual initial losses but can be extended for other loss types.
        """
        initial_losses = []
        for key in self.loss_keys:
            loss = losses[key]
            # For classification tasks (future extension)
            if hasattr(loss, 'num_classes'):
                # Use theoretical initial loss: log(num_classes)
                num_classes = getattr(loss, 'num_classes', 10)
                initial_losses.append(torch.log(torch.tensor(num_classes, dtype=torch.float, device=self.weights.device)))
            # For PDE residual - use a reasonable initial scale
            elif key == 'pde':
                # Use actual loss but ensure it's not too small
                initial_val = max(loss.item(), 0.1)
                initial_losses.append(initial_val)
            # Default: use actual initial loss
            else:
                initial_losses.append(loss.item())
        return torch.tensor(initial_losses, device=self.weights.device)

    def update_weights(self, model, dataset, losses):
        """
        The core of GradNorm with enhanced stability and weight management.
        This method is called by the trainer's hook and performs the multi-pass 
        optimization for the loss weights `w_i`.
        """
        # Calculate PDE loss since it's not in the initial losses dict
        pde_residual = self.calculate_residuals(model, dataset['collocation']['coords'])
        pde_loss = self.mse_loss(pde_residual, torch.zeros_like(pde_residual))
        losses['pde'] = pde_loss

        # 1. Initialize initial losses on the first step using theoretical values where appropriate
        if self.initial_losses is None:
            self.initial_losses = self._get_initial_losses(losses)

        # 2. Calculate Loss Ratios (inverse training rate)
        current_losses_tensor = torch.stack([losses[key] for key in self.loss_keys])
        loss_ratios = current_losses_tensor.detach() / self.initial_losses

        # 3. Calculate Relative Inverse Training Rates
        avg_loss_ratio = torch.mean(loss_ratios)
        relative_inverse_rates = loss_ratios / (avg_loss_ratio + 1e-8)  # Add stability term

        # 4. Calculate Gradient Norms for each task w.r.t the last shared layer
        grad_norms = []
        for i, key in enumerate(self.loss_keys):
            # All loss components are now available in the losses dict
            loss_component = losses[key]

            # Calculate the gradient of the weighted loss
            weighted_task_loss = self.weights[i] * loss_component

            grad_w = torch.autograd.grad(
                weighted_task_loss, model.last_shared_layer.parameters(),
                retain_graph=True, create_graph=True
            )[0]
            grad_norms.append(torch.norm(grad_w))

        grad_norms_tensor = torch.stack(grad_norms)

        # 5. Calculate the average gradient norm across all tasks
        avg_grad_norm = torch.mean(grad_norms_tensor.detach())

        # 6. Compute the GradNorm loss (L_grad)
        # Target is treated as a constant, so we detach it.
        target_grad_norms = avg_grad_norm * (relative_inverse_rates ** self.alpha)
        l_grad = torch.sum(torch.abs(grad_norms_tensor - target_grad_norms.detach()))

        # 7. Update the weights `w_i` by optimizing L_grad
        self.weight_optimizer.zero_grad()
        l_grad.backward()
        self.weight_optimizer.step()

        # 8. Enhanced weight renormalization and clamping with stability checks
        with torch.no_grad():
            # Check for NaN/inf values
            if torch.isnan(self.weights).any() or torch.isinf(self.weights).any():
                print("Warning: NaN/inf detected in weights. Resetting to uniform distribution.")
                self.weights.data = torch.ones_like(self.weights)
            
            # Renormalize to sum to T (number of tasks) with stability
            sum_weights = torch.sum(self.weights.data)
            if sum_weights < 1e-10:  # Prevent division by zero
                print("Warning: Weight sum too small. Resetting to uniform distribution.")
                self.weights.data = torch.ones_like(self.weights)
                sum_weights = len(self.loss_keys)
            
            self.weights.data = (self.weights.data / sum_weights) * len(self.loss_keys)
            
            # Apply improved clamping with warnings
            min_val, max_val = 1e-4, 10.0
            for i in range(len(self.weights)):
                if self.weights[i] < min_val and i not in self.clamp_warnings:
                    print(f"Warning: Weight for '{self.loss_keys[i]}' clamped to min {min_val}")
                    self.clamp_warnings.add(i)
                elif self.weights[i] > max_val and i not in self.clamp_warnings:
                    print(f"Warning: Weight for '{self.loss_keys[i]}' clamped to max {max_val}")
                    self.clamp_warnings.add(i)
                    
            self.weights.data = torch.clamp(self.weights.data, min=min_val, max=max_val)

    def step(self, losses, optimizer, dataset, model):
        """
        Performs the main optimization step for the model's parameters.
        For GradNorm, this is simple as the complex logic is in update_weights().
        """
        # Recompute all losses with a fresh computational graph to avoid
        # "backward through graph a second time" error
        fresh_losses = {}
        fresh_losses['data'] = self.mse_loss(model(dataset['data']['coords']), dataset['data']['values'])
        fresh_losses['initial'] = self.mse_loss(model(dataset['initial']['coords']), dataset['initial']['values'])
        fresh_losses['boundary'] = self.mse_loss(model(dataset['boundary']['coords']), dataset['boundary']['values'])

        # Calculate the PDE loss with fresh computational graph
        residual = self.calculate_residuals(model, dataset['collocation']['coords'])
        fresh_losses['pde'] = self.mse_loss(residual, torch.zeros_like(residual))

        # Update the original losses dict for logging purposes
        losses['pde'] = fresh_losses['pde']

        # Combine all losses using the weights `w_i` that were just updated by GradNorm
        current_losses_tensor = torch.stack([fresh_losses[key] for key in self.loss_keys])
        total_loss = torch.sum(self.weights.detach() * current_losses_tensor)

        # Standard optimization step for the model parameters
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Log the current weights for visualization
        self.last_lambdas = {f'weight_{key}': w.item() for key, w in zip(self.loss_keys, self.weights)}

        return total_loss

    @staticmethod
    def get_params():
        """Returns the tunable hyperparameters for the UI."""
        return {
            'alpha': (1.5, 0.0, 3.0, 0.1),
            'weight_lr': (0.025, 0.001, 0.1, 0.005)
        }


# %%
# Cell 18: Final Dashboard with All Methods including GradNorm
# -----------------------------------------------------------
# This cell integrates all implemented methods, including GradNorm,
# into the final interactive PINNDashboard for a full
# comparative analysis.

# Define the map of available loss functions, now including all methods
loss_fn_map = {
    "Simple Weighted Sum": SimpleWeightedLoss,
    "lbPINN (Gaussian Likelihood)": GaussianLikelihoodLoss,
    "dwPINN (Dynamic Weights)": DWPINNLoss,
    "ReLoBRaLo": ReLoBRaLoLoss,
    "BRDR (Balanced Decay Rate)": BRDRLoss,
    "GradNorm": GradNormLoss # Add the new GradNorm method
}

# Define the map of available PDEs
pde_map = {
    "1D Heat Equation": HeatEquation1D,
    "1D Wave Equation": WaveEquation1D,
    "1D Viscous Burgers": ViscousBurgers1D
}

# Instantiate and display the final dashboard
final_dashboard = PINNDashboard(pde_map, loss_fn_map, use_kfold=True, n_splits=5)
final_dashboard.display_dashboard()





# %% [markdown]
# ### AL-PINN (Son et al., 2022)
# 
# [cite_start]This method treats the training of a Physics-Informed Neural Network as a constrained optimization problem, which it then solves using a continuous gradient descent-ascent approach with the Augmented Lagrangian Method (ALM)[cite: 3].
# 
# #### Intuitive Explanation
# 
# Imagine you are training a dog (the neural network optimizer) to walk along a very specific path (the correct PDE solution). The initial and boundary conditions are like two leashes that keep the dog from straying at the start and edges of the path.
# 
# In the AL-PINN method, these leashes have two components:
# 1.  **A Spring (The Penalty Term `β`):** This is a fixed-stiffness spring. The more the dog pulls away from the correct starting/boundary point, the stronger the spring pulls it back. This provides a constant, predictable restoring force.
# 2.  **A Person Holding the Leash (The Multiplier `λ`):** This person continuously adjusts their pull. At every single step the dog takes, the person slightly increases or decreases their pull based on how far the dog is from where it should be.
# 
# The training is a **smooth, continuous process** where the dog's movement, the spring's tension, and the person's gentle, constant adjustments all happen simultaneously to guide the dog onto the correct path.
# 
# #### Formulas and Algorithm
# 
# [cite_start]The core idea is to minimize the PDE residual while treating the initial/boundary conditions as equality constraints[cite: 4, 76].
# 
# * **The Augmented Lagrangian Objective Function:**
#     [cite_start]The method transforms the constrained problem into a single objective function, `L_λ(θ)`, which the optimizer seeks to minimize with respect to the network parameters `θ` and maximize with respect to the multipliers `λ`[cite: 5, 78].
# 
#     $$\mathcal{L}_{\lambda}(\theta) = \underbrace{||Nu_{nn}(\theta) - f||_{L^2(\Omega)}^2}_{\text{PDE Residual Loss}} + \underbrace{\beta||Tu_{nn}(\theta) - g||_{L^2(\partial\Omega)}^2}_{\text{Penalty Term}} + \underbrace{\langle\lambda, Tu_{nn}(\theta) - g\rangle_{L^2(\partial\Omega)}}_{\text{Multiplier Term}}$$
#     [cite_start][cite: 78]
#     * [cite_start]`Nu - f` is the residual of the PDE itself[cite: 76]. The goal is to make this zero.
#     * [cite_start]`Tu - g` represents the error at the initial or boundary conditions (the constraint violation)[cite: 76].
#     * [cite_start]`β` is a fixed, positive penalty parameter that you choose before training[cite: 134]. It determines the "stiffness" of the penalty.
#     * [cite_start]`λ` is the Lagrange multiplier, which is a learnable parameter that adapts during training[cite: 5].
# 
# * **The Optimization Algorithm (Gradient Descent-Ascent):**
#     [cite_start]The network parameters `θ` and the multipliers `λ` are updated at every training step[cite: 78].
#     1.  [cite_start]**Network Update (Descent):** The network parameters `θ` are updated using gradient descent to minimize the total loss `L_λ(θ)`[cite: 78].
#         `θ ← θ - η_θ * ∇_θ L_λ(θ)`
#     2.  [cite_start]**Multiplier Update (Ascent):** The Lagrange multipliers `λ` are updated using gradient ascent to maximize the total loss `L_λ(θ)`[cite: 78]. This simplifies to adding the constraint violation scaled by a learning rate `η_λ`.
#         `λ ← λ + η_λ * (Tu_{nn}(\theta) - g)`
# 
# ---
# 
# ### PECANN (Basir & Senocak, 2022)
# 
# [cite_start]This method also uses the Augmented Lagrangian Method but employs a fundamentally different, **conditional** strategy for updating its parameters[cite: 421]. [cite_start]It is designed to strictly enforce high-fidelity data (like boundary conditions) as hard constraints[cite: 493].
# 
# #### Intuitive Explanation
# 
# Using the same dog-on-a-leash analogy, the PECANN method has a different training philosophy.
# 
# The person holding the leash (the optimizer) is more hands-off at first. They let the dog explore, but they watch it very carefully. An update only happens if a specific condition is met: **if the dog strays from the path AND it isn't making progress on its own toward getting back on track.**
# 
# When this condition is triggered:
# 1.  **The Person Yanks the Leash (The Multiplier `λ` Update):** Instead of a gentle tug, the person gives a strong, corrective yank to pull the dog back.
# 2.  **The Spring is Replaced (The Penalty `µ` Update):** The person immediately swaps the leash's spring for a much stiffer one (by increasing the penalty parameter `µ`).
# 
# This training is a **sparse, corrective process**. For many steps, nothing happens. Then, when the model performs poorly on the constraints, the algorithm applies a strong, aggressive correction to force it back in line.
# 
# #### Formulas and Algorithm
# 
# [cite_start]The goal is to minimize an objective (like PDE loss or low-fidelity data loss) subject to equality constraints from high-fidelity data (like ICs/BCs)[cite: 493].
# 
# * **The Augmented Lagrangian Objective Function:**
#     [cite_start]The formula is structurally similar to AL-PINN's but with a conventional `µ/2` notation for the penalty[cite: 481, 499].
# 
#     $$\mathcal{L}(\theta; \lambda, \mu) = \underbrace{\mathcal{J}(\theta)}_{\text{Objective Loss}} + \underbrace{\langle\lambda, C(\theta)\rangle}_{\text{Multiplier Term}} + \underbrace{\frac{\mu}{2} ||C(\theta)||^2}_{\text{Penalty Term}}$$
#     [cite_start][cite: 499]
#     * [cite_start]`J(θ)` is the main objective, which is the PDE residual loss (`J_F`) and optionally a loss for any noisy data (`J_M`)[cite: 493, 494].
#     * [cite_start]`C(θ)` is the violation of the initial or boundary conditions[cite: 478].
#     * [cite_start]`µ` is the penalty parameter, which is **not fixed** and increases during training[cite: 503].
#     * [cite_start]`λ` is the Lagrange multiplier[cite: 501].
# 
# * **The Optimization Algorithm (Conditional Updates):**
#     [cite_start]This is the defining feature of PECANN, detailed in `Algorithm 1` of the paper[cite: 503]. The network parameters `θ` are updated at every step, but the adaptive parameters `λ` and `µ` are not.
#     1.  **Network Update (Descent):** Standard gradient descent is used to update `θ` to minimize the total loss `L(θ; λ, µ)`.
#     2.  **Conditional Multiplier and Penalty Update:** The algorithm checks the constraint violation `C(θ)` at the end of the step. The updates to `λ` and `µ` only occur if the following conditions are met:
#         * [cite_start]The current constraint violation has not improved sufficiently compared to the previous step's violation[cite: 506].
#         * [cite_start]The current constraint violation is still larger than a small tolerance `ε`[cite: 507].
#     3.  **If the condition is met:**
#         * [cite_start]**Penalty Update:** The penalty `µ` is increased (typically doubled), up to a maximum value `µ_max`[cite: 503].
#             `µ ← min(2 * µ, µ_max)`
#         * [cite_start]**Multiplier Update:** The multipliers `λ` are updated using the *new, larger `µ`*, not a separate learning rate[cite: 502].
#             `λ ← λ + µ * C(θ)`

# %%
# %%
# Cell 19: AL-PINN and PECANN Implementations
# -------------------------------------------
# This cell contains the implementations for the two Augmented Lagrangian methods.

import torch
import torch.nn as nn

# --- AL-PINN Implementation (Son et al., 2022) ---
class ALPINNLoss(BaseLoss):
    """
    Implements the Augmented Lagrangian PINN (AL-PINN) method from Son et al.
    This method uses a continuous gradient descent-ascent update rule.
    """
    def __init__(self, pde_instance, model, weights={'beta': 1.0, 'lambda_lr': 1e-4}):
        super().__init__(pde_instance)
        self.model = model
        self.beta = weights.get('beta', 1.0)  # Penalty parameter (fixed)
        self.lambda_lr = weights.get('lambda_lr', 1e-4) # Learning rate for multipliers

        # State variables for Lagrange multipliers. They will be initialized on the first step.
        self.lambdas = {}
        self._initialized = False
        self.last_lambdas = {} # For logging

    @staticmethod
    def get_params():
        return {
            'beta': (1.0, 0.1, 100.0, 0.1),
            'lambda_lr': (1e-4, 1e-6, 1e-2, 1e-6)
        }

    def step(self, losses, optimizer, dataset, model):
        # 1. Get raw residuals for the constraints (Initial and Boundary conditions)
        # ALM operates on the raw constraint violations, not the MSE.
        _, residuals = self.forward_with_residuals(model, dataset)
        c_initial = residuals['initial_res']
        c_boundary = residuals['boundary_res']

        # The objective is the PDE residual loss
        pde_residual = self.calculate_residuals(model, dataset['collocation']['coords'])
        loss_pde = self.mse_loss(pde_residual, torch.zeros_like(pde_residual))
        losses['pde'] = loss_pde # For history tracking

        # 2. Initialize multipliers on the first step
        if not self._initialized:
            self.lambdas['initial'] = torch.zeros_like(c_initial, requires_grad=False)
            self.lambdas['boundary'] = torch.zeros_like(c_boundary, requires_grad=False)
            self._initialized = True

        # 3. Construct the Augmented Lagrangian Loss
        # L(θ,λ) = Loss_PDE + β * ||C(θ)||^2 + <λ, C(θ)>
        penalty_term = self.beta * (self.mse_loss(c_initial, torch.zeros_like(c_initial)) + \
                                    self.mse_loss(c_boundary, torch.zeros_like(c_boundary)))
        
        multiplier_term = torch.mean(self.lambdas['initial'] * c_initial) + \
                          torch.mean(self.lambdas['boundary'] * c_boundary)

        total_loss = loss_pde + penalty_term + multiplier_term

        # 4. Update network parameters (gradient descent)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # 5. Update Lagrange multipliers (gradient ascent)
        # λ ← λ + η_λ * ∇_λ(L)  where ∇_λ(L) is just C(θ)
        with torch.no_grad():
            self.lambdas['initial'] += self.lambda_lr * c_initial
            self.lambdas['boundary'] += self.lambda_lr * c_boundary

        # 6. Log the mean of the multipliers for visualization
        self.last_lambdas = {
            'lambda_initial_mean': torch.mean(self.lambdas['initial']).item(),
            'lambda_boundary_mean': torch.mean(self.lambdas['boundary']).item()
        }

        return total_loss

# --- PECANN Implementation (Basir & Senocak, 2022) ---
class PECANNLoss(BaseLoss):
    """
    Implements the Physics and Equality Constrained ANN (PECANN) method from Basir & Senocak.
    This method uses a conditional update for multipliers and the penalty parameter.
    """
    def __init__(self, pde_instance, model, weights={'mu_initial': 1.0, 'mu_max': 1e4, 'epsilon': 1e-8}):
        super().__init__(pde_instance)
        self.model = model
        
        # Hyperparameters
        self.mu = weights.get('mu_initial', 1.0)
        self.mu_max = weights.get('mu_max', 1e4)
        self.epsilon = weights.get('epsilon', 1e-8) # Tolerance for constraint violation

        # State variables
        self.lambdas = {}
        self.eta = 0.0  # Stores the previous constraint violation measure
        self._initialized = False
        self.last_lambdas = {} # For logging

    @staticmethod
    def get_params():
        return {
            'mu_initial': (1.0, 0.1, 100.0, 0.1),
            'mu_max': (1e4, 1e2, 1e6, 1e2),
            'epsilon': (1e-8, 1e-9, 1e-6, 1e-9)
        }

    def step(self, losses, optimizer, dataset, model):
        # 1. Get raw residuals for constraints and calculate objective loss
        _, residuals = self.forward_with_residuals(model, dataset)
        c_initial = residuals['initial_res']
        c_boundary = residuals['boundary_res']
        
        pde_residual = self.calculate_residuals(model, dataset['collocation']['coords'])
        loss_pde = self.mse_loss(pde_residual, torch.zeros_like(pde_residual))
        losses['pde'] = loss_pde

        # The objective includes PDE loss and any low-fidelity data loss
        # Here we only use PDE, but could add losses['data']
        objective_loss = loss_pde

        # 2. Initialize multipliers and state on the first step
        if not self._initialized:
            self.lambdas['initial'] = torch.zeros_like(c_initial, requires_grad=False)
            self.lambdas['boundary'] = torch.zeros_like(c_boundary, requires_grad=False)
            self._initialized = True

        # 3. Construct the Augmented Lagrangian Loss
        # L(θ,λ,μ) = J(θ) + <λ, C(θ)> + (μ/2) * ||C(θ)||^2
        penalty_term = (self.mu / 2) * (self.mse_loss(c_initial, torch.zeros_like(c_initial)) + \
                                        self.mse_loss(c_boundary, torch.zeros_like(c_boundary)))

        multiplier_term = torch.mean(self.lambdas['initial'] * c_initial) + \
                          torch.mean(self.lambdas['boundary'] * c_boundary)
        
        total_loss = objective_loss + penalty_term + multiplier_term

        # 4. Update network parameters
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # 5. Conditional update for multipliers (λ) and penalty (μ)
        with torch.no_grad():
            # Calculate current total constraint violation
            current_violation = torch.sqrt(self.mse_loss(c_initial, torch.zeros_like(c_initial)) + \
                                           self.mse_loss(c_boundary, torch.zeros_like(c_boundary)))

            # Algorithm 1 from PECANN paper
            if (current_violation >= 0.25 * self.eta) and (current_violation > self.epsilon):
                # Update penalty parameter mu
                self.mu = min(2 * self.mu, self.mu_max)
                # Update Lagrange multipliers
                self.lambdas['initial'] += self.mu * c_initial
                self.lambdas['boundary'] += self.mu * c_boundary

            # Record the current penalty loss for the next iteration's check
            self.eta = current_violation.item()

        # 6. Log parameters for visualization
        self.last_lambdas = {
            'mu': self.mu,
            'lambda_initial_mean': torch.mean(self.lambdas['initial']).item(),
            'lambda_boundary_mean': torch.mean(self.lambdas['boundary']).item()
        }

        return total_loss

# %%
# %%
# Cell 20: Final Dashboard with All Methods
# ------------------------------------------
# This cell integrates all implemented methods, including the Augmented
# Lagrangian methods, into the final interactive PINNDashboard for a full
# comparative analysis.

# Define the map of available loss functions, now including all methods
loss_fn_map = {
    "Simple Weighted Sum": SimpleWeightedLoss,
    "lbPINN (Gaussian Likelihood)": GaussianLikelihoodLoss,
    "dwPINN (Dynamic Weights)": DWPINNLoss,
    "ReLoBRaLo": ReLoBRaLoLoss,
    "BRDR (Balanced Decay Rate)": BRDRLoss,
    "GradNorm": GradNormLoss,
    "AL-PINN (Son et al.)": ALPINNLoss,
    "PECANN (Basir & Senocak)": PECANNLoss
}

# Define the map of available PDEs
pde_map = {
    "1D Heat Equation": HeatEquation1D,
    "1D Wave Equation": WaveEquation1D,
    "1D Viscous Burgers": ViscousBurgers1D
}

# Instantiate and display the final dashboard
final_dashboard = PINNDashboard(pde_map, loss_fn_map, use_kfold=True, n_splits=5)
final_dashboard.display_dashboard()

# %%
# %%
# Cell 20: Final Dashboard with All Methods
# ------------------------------------------
# This cell integrates all implemented methods, including the Augmented
# Lagrangian methods, into the final interactive PINNDashboard for a full
# comparative analysis.

# Define the map of available loss functions, now including all methods
loss_fn_map = {
    "Simple Weighted Sum": SimpleWeightedLoss,
    "lbPINN (Gaussian Likelihood)": GaussianLikelihoodLoss,
    "dwPINN (Dynamic Weights)": DWPINNLoss,
    "ReLoBRaLo": ReLoBRaLoLoss,
    "BRDR (Balanced Decay Rate)": BRDRLoss,
    "GradNorm": GradNormLoss,
    "AL-PINN (Son et al.)": ALPINNLoss,
    "PECANN (Basir & Senocak)": PECANNLoss
}

# Define the map of available PDEs
pde_map = {
    "1D Heat Equation": HeatEquation1D,
    "1D Wave Equation": WaveEquation1D,
    "1D Viscous Burgers": ViscousBurgers1D
}

# Instantiate and display the final dashboard
final_dashboard = PINNDashboard(pde_map, loss_fn_map, use_kfold=True, n_splits=5)
final_dashboard.display_dashboard()

# %%



