'''
Original Paper: Linking Physical Fidelity to Downstream Performance in Physics-Informed Fault Diagnosis

'''

import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

import src.configs as cfg

class ConfigurableMLP(nn.Module):
    """
    A configurable multi-layer perceptron neural network.
    
    Parameters:
    - input_dim: Input dimension
    - hidden_layers: List of integers defining the width of each hidden layer
    - output_dim: Output dimension
    - activation: Activation function (default: 'tanh')
    - dropout_rate: Dropout rate for regularization (default: 0.0)
    - init_method: Weight initialization method (default: 'xavier_normal')
    """
    def __init__(self, input_dim, hidden_layers, output_dim,
                 activation='tanh', dropout_rate=0.0, init_method='xavier_normal'):
        super(ConfigurableMLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.init_method = init_method
        
        layers = []
        
        # Input layer
        current_dim = input_dim
        
        # Hidden layers
        for h_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, h_dim))
            
            # Activation function
            if activation.lower() == 'tanh':
                layers.append(nn.Tanh())
            elif activation.lower() == 'relu':
                layers.append(nn.ReLU())
            elif activation.lower() == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.1))
            elif activation.lower() == 'elu':
                layers.append(nn.ELU())
            elif activation.lower() == 'selu':
                layers.append(nn.SELU())
            elif activation.lower() == 'gelu':
                layers.append(nn.GELU())
            else:
                raise ValueError(f"Unsupported activation function: {activation}")
            
            # Add dropout if specified
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            current_dim = h_dim
        
        # Output layer
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights(init_method)
        
    def _initialize_weights(self, method):
        """Initialize the weights using the specified method."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if method == 'xavier_normal':
                    nn.init.xavier_normal_(m.weight)
                elif method == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight)
                elif method == 'kaiming_normal':
                    nn.init.kaiming_normal_(m.weight, nonlinearity='tanh')
                elif method == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='tanh')
                elif method == 'orthogonal':
                    nn.init.orthogonal_(m.weight)
                elif method == 'normal':
                    nn.init.normal_(m.weight, mean=0, std=0.1)
                elif method == 'uniform':
                    nn.init.uniform_(m.weight, a=-0.1, b=0.1)
                
                # Initialize bias
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.model(x)

class ConfigurablePINN(nn.Module):
    """
    A configurable Physics-Informed Neural Network (PINN) model.
    
    Parameters:
    - unmeasured_net_config: Configuration for the unmeasured parameters network
        - hidden_layers: List of integers for hidden layer widths
        - activation: Activation function to use
        - dropout_rate: Dropout rate
        - init_method: Weight initialization method
    - acceleration_net_config: Configuration for the acceleration network
        - hidden_layers: List of integers for hidden layer widths
        - activation: Activation function to use
        - dropout_rate: Dropout rate
        - init_method: Weight initialization method
    - param_init_config: Configuration for physical parameter initialization
        - method: Initialization method ('fixed', 'uniform', 'normal')
        - values: Dictionary of initial values or distribution parameters
    - enable_mass_constraints: Whether to include mass constraints (default: True for real data compatibility)
    """
    def __init__(self, unmeasured_net_config=None, acceleration_net_config=None, param_init_config=None, enable_mass_constraints=True):
        super(ConfigurablePINN, self).__init__()
        
        # Store mass constraint flag
        self.enable_mass_constraints = enable_mass_constraints
        
        # Default configurations if not provided
        if unmeasured_net_config is None:
            unmeasured_net_config = cfg.PINN_ARCH_DEFAULT['unmeasured_net_config']
            
        if acceleration_net_config is None:
            acceleration_net_config = cfg.PINN_ARCH_DEFAULT['acceleration_net_config']
            
        if param_init_config is None:
            param_init_config = cfg.PINN_ARCH_DEFAULT['param_init_config']

        # Network for estimating unmeasured parameters (15 inputs, 4 outputs)
        self.NNforUnmeasured = ConfigurableMLP(
            input_dim=15,  # 10 features + 5 parameters
            hidden_layers=unmeasured_net_config['hidden_layers'],
            output_dim=4,  # 4 unmeasured parameters
            activation=unmeasured_net_config['activation'],
            dropout_rate=unmeasured_net_config['dropout_rate'],
            init_method=unmeasured_net_config['init_method']
        )
        
        # Network for estimating accelerations (10 inputs, 4 outputs)
        self.NNforAccelerations = ConfigurableMLP(
            input_dim=10,  # 10 features
            hidden_layers=acceleration_net_config['hidden_layers'],
            output_dim=4,  # 4 accelerations
            activation=acceleration_net_config['activation'],
            dropout_rate=acceleration_net_config['dropout_rate'],
            init_method=acceleration_net_config['init_method']
        )
        
        # Initialize learnable physical parameters based on configuration
        self._initialize_physical_parameters(param_init_config)
        
        # Register gravity constant as a buffer so it follows the model's device
        self.register_buffer('g', torch.tensor(cfg.GRAVITY, dtype=torch.float64))
        
        # Convert model to double precision
        self.double()
        
    def _initialize_physical_parameters(self, config):
        """
        Initialize physical parameters based on configuration.
        
        Parameters:
        - config: Dictionary with 'method' and 'values' keys
        """
        method = config['method']
        values = config['values']
        
        if method == 'fixed':
            # Initialize with fixed values
            self.M1 = nn.Parameter(torch.tensor(float(values.get('M1', 10.0)), dtype=torch.float64))
            self.M2 = nn.Parameter(torch.tensor(float(values.get('M2', 10.0)), dtype=torch.float64))
            self.M3 = nn.Parameter(torch.tensor(float(values.get('M3', 11.0)), dtype=torch.float64))
            self.D1 = nn.Parameter(torch.tensor(float(values.get('D1', 10.0)), dtype=torch.float64))
            self.D2 = nn.Parameter(torch.tensor(float(values.get('D2', 10.0)), dtype=torch.float64))
            self.D3 = nn.Parameter(torch.tensor(float(values.get('D3', 10.0)), dtype=torch.float64))
            self.K1 = nn.Parameter(torch.tensor(float(values.get('K1', 10.0)), dtype=torch.float64))
            self.K2 = nn.Parameter(torch.tensor(float(values.get('K2', 10.0)), dtype=torch.float64))
            self.E1 = nn.Parameter(torch.tensor(float(values.get('E1', 10.0)), dtype=torch.float64))
        
        elif method == 'uniform':
            # Initialize with uniform distribution
            M1_range = values.get('M1', (5.0, 15.0))
            M2_range = values.get('M2', (5.0, 15.0))
            M3_range = values.get('M3', (5.0, 15.0))
            D1_range = values.get('D1', (5.0, 15.0))
            D2_range = values.get('D2', (5.0, 15.0))
            D3_range = values.get('D3', (5.0, 15.0))
            K1_range = values.get('K1', (5.0, 15.0))
            K2_range = values.get('K2', (5.0, 15.0))
            E1_range = values.get('E1', (5.0, 15.0))
            
            self.M1 = nn.Parameter(torch.FloatTensor(1).uniform_(*M1_range).double())
            self.M2 = nn.Parameter(torch.FloatTensor(1).uniform_(*M2_range).double())
            self.M3 = nn.Parameter(torch.FloatTensor(1).uniform_(*M3_range).double())
            self.D1 = nn.Parameter(torch.FloatTensor(1).uniform_(*D1_range).double())
            self.D2 = nn.Parameter(torch.FloatTensor(1).uniform_(*D2_range).double())
            self.D3 = nn.Parameter(torch.FloatTensor(1).uniform_(*D3_range).double())
            self.K1 = nn.Parameter(torch.FloatTensor(1).uniform_(*K1_range).double())
            self.K2 = nn.Parameter(torch.FloatTensor(1).uniform_(*K2_range).double())
            self.E1 = nn.Parameter(torch.FloatTensor(1).uniform_(*E1_range).double())
        
        elif method == 'normal':
            # Initialize with normal distribution
            M1_params = values.get('M1', (10.0, 1.0))
            M2_params = values.get('M2', (10.0, 1.0))
            M3_params = values.get('M3', (11.0, 1.0))
            D1_params = values.get('D1', (10.0, 1.0))
            D2_params = values.get('D2', (10.0, 1.0))
            D3_params = values.get('D3', (10.0, 1.0))
            K1_params = values.get('K1', (10.0, 1.0))
            K2_params = values.get('K2', (10.0, 1.0))
            E1_params = values.get('E1', (10.0, 1.0))
            
            self.M1 = nn.Parameter(torch.FloatTensor(1).normal_(*M1_params).double())
            self.M2 = nn.Parameter(torch.FloatTensor(1).normal_(*M2_params).double())
            self.M3 = nn.Parameter(torch.FloatTensor(1).normal_(*M3_params).double())
            self.D1 = nn.Parameter(torch.FloatTensor(1).normal_(*D1_params).double())
            self.D2 = nn.Parameter(torch.FloatTensor(1).normal_(*D2_params).double())
            self.D3 = nn.Parameter(torch.FloatTensor(1).normal_(*D3_params).double())
            self.K1 = nn.Parameter(torch.FloatTensor(1).normal_(*K1_params).double())
            self.K2 = nn.Parameter(torch.FloatTensor(1).normal_(*K2_params).double())
            self.E1 = nn.Parameter(torch.FloatTensor(1).normal_(*E1_params).double())
        
        else:
            raise ValueError(f"Unsupported parameter initialization method: {method}")

    """
    Predicts the 4 accelerations and the 4 unmeasured physics parameters
    """
    def forward(self, x):
        # Ensure input is double precision
        x = x.double()
        
        # Enforce constraints on physical parameters
        M1 = torch.clamp(self.M1, min=0.1)    
        M2 = torch.clamp(self.M2, min=0.1)
        M3 = torch.clamp(self.M3, min=0.1)
        D1 = torch.clamp(self.D1, min=0.0)
        D2 = torch.clamp(self.D2, min=0.0)
        D3 = torch.clamp(self.D3, min=0.0)
        K1 = torch.clamp(self.K1, min=1.0)  # Ensure K1 is sufficiently away from zero
        K2 = torch.clamp(self.K2, min=0.1)
        E1 = torch.clamp(self.E1, min=0.0)

        # Assuming x has shape (batch_size, num_features)
        batch_size = x.size(0)

        # Stack parameters
        # Ensure parameters are properly shaped for concatenation
        # First reshape to [1, 1] if needed, then expand to batch size
        M1_reshaped = M1.view(1, 1)
        D1_reshaped = D1.view(1, 1)
        K1_reshaped = K1.view(1, 1)
        K2_reshaped = K2.view(1, 1)
        E1_reshaped = E1.view(1, 1)
        
        # Concatenate along dimension 1
        params = torch.cat([
            M1_reshaped,
            D1_reshaped,
            K1_reshaped,
            K2_reshaped,
            E1_reshaped
        ], dim=1)
        
        # Expand to match batch size
        params = params.expand(batch_size, -1)

        # Concatenate x with the expanded parameters
        input_for_unmeasured = torch.cat((x, params), dim=1)

        # Estimate the sum of the unknown unmeasured parameters
        fA, fB, fC, fD = torch.split(self.NNforUnmeasured(input_for_unmeasured), 1, dim=1)

        # Set these as attributes to be used in compute_residuals
        self.fA = fA
        self.fB = fB
        self.fC = fC
        self.fD = fD

        # Estimate the accelerations
        x2_ddot, y2_ddot, x3_ddot, y3_ddot = torch.split(self.NNforAccelerations(x), 1, dim=1)

        # Return the accelerations
        return torch.cat((x2_ddot, y2_ddot, x3_ddot, y3_ddot), dim=1)
    
    def _denormalize_physical_variables(self, x, pred, X_max=None, X_min=None, y_max=None, y_min=None):
        """
        Helper method to denormalize inputs and outputs for physical calculations.
        Returns denormalized positions, velocities, accelerations, omega, and time.
        """
        # Get predicted accelerations (normalized)
        x2_ddot, y2_ddot, x3_ddot, y3_ddot = torch.split(pred, 1, dim=1)

        # Get the input features (all normalized to [0,1])
        x2_dot, y2_dot, x3_dot, y3_dot, x2, y2, x3, y3, omega, t = torch.split(x, 1, dim=1)
        
        # Denormalize positions (features 4-7: x2, y2, x3, y3)
        if X_max is not None and X_min is not None:
            # Ensure normalization parameters are double precision
            X_max = X_max.double()
            X_min = X_min.double()
            y_max = y_max.double() if y_max is not None else None
            y_min = y_min.double() if y_min is not None else None
            
            # Handle 1D normalization parameters properly
            if X_min.dim() == 1:
                # Extract position normalization parameters (indices 4-7)
                pos_min = X_min[4:8].unsqueeze(0)  # x2, y2, x3, y3 - shape: (1, 4)
                pos_max = X_max[4:8].unsqueeze(0)  # shape: (1, 4)
            else:
                # Handle 2D case if needed
                pos_min = X_min[:, 4:8]  # x2, y2, x3, y3
                pos_max = X_max[:, 4:8]
            
            pos_range = pos_max - pos_min + 1e-12
            
            # Denormalize positions
            positions = torch.cat([x2, y2, x3, y3], dim=1)  # Shape: (batch_size, 4)
            positions_denorm = positions * pos_range + pos_min
            x2_denorm, y2_denorm, x3_denorm, y3_denorm = torch.split(positions_denorm, 1, dim=1)
            
            # Denormalize velocities (features 0-3: x2_dot, y2_dot, x3_dot, y3_dot)
            if X_min.dim() == 1:
                vel_min = X_min[0:4].unsqueeze(0)  # x2_dot, y2_dot, x3_dot, y3_dot - shape: (1, 4)
                vel_max = X_max[0:4].unsqueeze(0)  # shape: (1, 4)
            else:
                vel_min = X_min[:, 0:4]  # x2_dot, y2_dot, x3_dot, y3_dot
                vel_max = X_max[:, 0:4]
            
            vel_range = vel_max - vel_min + 1e-12
            
            velocities = torch.cat([x2_dot, y2_dot, x3_dot, y3_dot], dim=1)  # Shape: (batch_size, 4)
            velocities_denorm = velocities * vel_range + vel_min
            x2_dot_denorm, y2_dot_denorm, x3_dot_denorm, y3_dot_denorm = torch.split(velocities_denorm, 1, dim=1)
            
            # Denormalize accelerations (outputs)
            if y_max is not None and y_min is not None:
                if y_min.dim() == 1:
                    accel_range = (y_max - y_min + 1e-12).unsqueeze(0)  # shape: (1, 4)
                    y_min_expanded = y_min.unsqueeze(0)  # shape: (1, 4)
                else:
                    accel_range = y_max - y_min + 1e-12
                    y_min_expanded = y_min
                
                accelerations = torch.cat([x2_ddot, y2_ddot, x3_ddot, y3_ddot], dim=1)  # Shape: (batch_size, 4)
                accelerations_denorm = accelerations * accel_range + y_min_expanded
                x2_ddot_denorm, y2_ddot_denorm, x3_ddot_denorm, y3_ddot_denorm = torch.split(accelerations_denorm, 1, dim=1)
            else:
                # Fallback: use normalized values if denormalization params not available
                x2_ddot_denorm, y2_ddot_denorm, x3_ddot_denorm, y3_ddot_denorm = x2_ddot, y2_ddot, x3_ddot, y3_ddot

            # Denormalize omega and time using provided normalization parameters
            # Expected fixed ranges (dataset-wide): omega [73.3, 377] rad/s, t [0, 5] s
            if X_min.dim() == 1:
                omega_min = X_min[8].unsqueeze(0)
                omega_max = X_max[8].unsqueeze(0)
                t_min = X_min[9].unsqueeze(0)
                t_max = X_max[9].unsqueeze(0)
            else:
                omega_min = X_min[:, 8:9]
                omega_max = X_max[:, 8:9]
                t_min = X_min[:, 9:10]
                t_max = X_max[:, 9:10]

            omega_range = omega_max - omega_min + 1e-12
            t_range = t_max - t_min + 1e-12

            omega_phys = omega * omega_range + omega_min
            t_phys = t * t_range + t_min
        else:
            # Fallback: use normalized values if denormalization params not available
            x2_denorm, y2_denorm, x3_denorm, y3_denorm = x2, y2, x3, y3
            x2_dot_denorm, y2_dot_denorm, x3_dot_denorm, y3_dot_denorm = x2_dot, y2_dot, x3_dot, y3_dot
            x2_ddot_denorm, y2_ddot_denorm, x3_ddot_denorm, y3_ddot_denorm = x2_ddot, y2_ddot, x3_ddot, y3_ddot
            omega_phys, t_phys = omega, t

        return (x2_denorm, y2_denorm, x3_denorm, y3_denorm,
                x2_dot_denorm, y2_dot_denorm, x3_dot_denorm, y3_dot_denorm,
                x2_ddot_denorm, y2_ddot_denorm, x3_ddot_denorm, y3_ddot_denorm,
                omega_phys, t_phys)

    def compute_residuals(self, x, pred, X_max=None, X_min=None, y_max=None, y_min=None):
        """
        Compute the physics-based residuals for the system.
        
        Parameters:
        - x: Input tensor (batch_size, features) - NORMALIZED
        - pred: Predicted accelerations (batch_size, 4) - NORMALIZED
        - X_max, X_min: Input normalization parameters (used for denormalization)
        - y_max, y_min: Output normalization parameters (used for denormalization)
        
        Returns:
        - Tuple of residual tensors
        """
        # Ensure double precision
        x = x.double()
        pred = pred.double()
        
        # Use clamped parameters for residual calculations
        M1 = torch.clamp(self.M1, min=0.1)    
        M2 = torch.clamp(self.M2, min=0.1)
        M3 = torch.clamp(self.M3, min=0.1)
        D1 = torch.clamp(self.D1, min=0.0)
        D2 = torch.clamp(self.D2, min=0.0)
        D3 = torch.clamp(self.D3, min=0.0)
        K1 = torch.clamp(self.K1, min=1.0)  # Critical to prevent division by zero
        K2 = torch.clamp(self.K2, min=0.1)
        E1 = torch.clamp(self.E1, min=0.0)

        # Denormalize inputs and outputs for physics equations
        (x2_denorm, y2_denorm, x3_denorm, y3_denorm,
         x2_dot_denorm, y2_dot_denorm, x3_dot_denorm, y3_dot_denorm,
         x2_ddot_denorm, y2_ddot_denorm, x3_ddot_denorm, y3_ddot_denorm,
         omega_phys, t_phys) = self._denormalize_physical_variables(x, pred, X_max, X_min, y_max, y_min)
        
        # Safely compute K2/K1 ratio (K1 is guaranteed to be at least 1.0)
        K2_K1_ratio = K2 / K1
        
        # Calculate the residuals using DENORMALIZED values for proper physics
        residual1 = K1*x2_denorm + K2*x3_denorm + M1*omega_phys**2*E1*torch.cos(omega_phys*t_phys) - self.fA
        residual2 = K1*y2_denorm + K2*y3_denorm - M1*self.g + M1*omega_phys**2*E1*torch.sin(omega_phys*t_phys) - self.fB
        residual3 = M3*x3_ddot_denorm + D3*x3_dot_denorm + K2*x3_denorm - K2_K1_ratio*M2*x2_ddot_denorm - K2_K1_ratio*D2*x2_dot_denorm - K2*x2_denorm - self.fC
        residual4 = M3*y3_ddot_denorm + D3*y3_dot_denorm + K2*y3_denorm - K2_K1_ratio*M2*y2_ddot_denorm - K2_K1_ratio*D2*y2_dot_denorm - K2*y2_denorm - K2_K1_ratio*M2*self.g + M3*self.g - self.fD

        # --- NON-DIMENSIONALIZATION (PHYSICS SCALING) ---
        # The variables above like K1 generate magnitudes of O(10^6). When the network computes the raw loss (residual^2),
        # these errors are in the millions, causing massive Gradient Pathology (exploding gradients) and completely blinding 
        # the ReLoBRaLo balancer (which saturates due to numeric disproportion against the ~1.0 baseline data loss).
        # We perform mathematical "Non-Dimensionalization" by dividing the differential equations by the characteristic scale (1e6).
        # This rigorously maintains the Newtonian algebra (since 0 / 1e6 is still exactly 0) while pushing the residuals 
        # to a well-conditioned mathematical range (O(0.1) - O(1.0)) for the Neural Optimizer!
        force_scale = 1e6
        residual1 = residual1 / force_scale
        residual2 = residual2 / force_scale
        residual3 = residual3 / force_scale
        residual4 = residual4 / force_scale

        # Conditionally include mass constraints based on flag
        if self.enable_mass_constraints:
            # Extra residuals for mass constraints (only for real data)
            residualMass1 = M1 + M2 + M3 - 22.0  # The total mass of the system is 22 kg
            residualMass2 = M2 - M3  # The masses of the overhang and underhang are equal
            
            # Mass residuals debug prints controlled by adaptive_custom_loss
            
            return residual1, residual2, residual3, residual4, residualMass1, residualMass2
        else:
            # For synthetic data, return zeros for mass constraints to maintain compatibility
            # Use zeros_like with a tensor that requires gradients to ensure grad_fn exists
            residualMass1 = torch.zeros_like(residual1)
            residualMass2 = torch.zeros_like(residual2)
            return residual1, residual2, residual3, residual4, residualMass1, residualMass2




 