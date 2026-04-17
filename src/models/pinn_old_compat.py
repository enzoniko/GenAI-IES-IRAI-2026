"""Compatibility shim for legacy exported PINN checkpoints."""

# pyright: reportUnusedImport=false, reportUnknownMemberType=false, reportUnannotatedClassAttribute=false, reportImplicitOverride=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownArgumentType=false, reportPrivateImportUsage=false, reportUnknownVariableType=false, reportOperatorIssue=false, reportUnusedParameter=false, reportUnusedCallResult=false, reportAny=false, reportArgumentType=false, reportIndexIssue=false, reportConstantRedefinition=false, reportUninitializedInstanceVariable=false, reportUnusedVariable=false, reportUnnecessaryComparison=false

import torch
import torch.nn as nn


class ConfigurableMLP(nn.Module):
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
        current_dim = input_dim

        for h_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, h_dim))

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

            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            current_dim = h_dim

        layers.append(nn.Linear(current_dim, output_dim))

        self.model = nn.Sequential(*layers)
        self._initialize_weights(init_method)

    def _initialize_weights(self, method):
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

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.model(x)


class ConfigurablePINN(nn.Module):
    def __init__(self, unmeasured_net_config=None, acceleration_net_config=None, param_init_config=None, enable_mass_constraints=True):
        super(ConfigurablePINN, self).__init__()

        self.enable_mass_constraints = enable_mass_constraints

        if unmeasured_net_config is None:
            unmeasured_net_config = {
                'hidden_layers': [64, 64],
                'activation': 'tanh',
                'dropout_rate': 0.0,
                'init_method': 'xavier_normal'
            }

        if acceleration_net_config is None:
            acceleration_net_config = {
                'hidden_layers': [64, 64],
                'activation': 'tanh',
                'dropout_rate': 0.0,
                'init_method': 'xavier_normal'
            }

        if param_init_config is None:
            param_init_config = {
                'method': 'fixed',
                'values': {
                    'M1': 50.0, 'M2': 3.5, 'M3': 3.5,
                    'D1': 3000.0, 'D2': 3000.0, 'D3': 3000.0,
                    'K1': 3.4635e6, 'K2': 3.8127e6, 'E1': 5.0e-6
                }
            }

        self.NNforUnmeasured = ConfigurableMLP(
            input_dim=15,
            hidden_layers=unmeasured_net_config['hidden_layers'],
            output_dim=4,
            activation=unmeasured_net_config['activation'],
            dropout_rate=unmeasured_net_config['dropout_rate'],
            init_method=unmeasured_net_config['init_method']
        )

        self.NNforAccelerations = ConfigurableMLP(
            input_dim=10,
            hidden_layers=acceleration_net_config['hidden_layers'],
            output_dim=4,
            activation=acceleration_net_config['activation'],
            dropout_rate=acceleration_net_config['dropout_rate'],
            init_method=acceleration_net_config['init_method']
        )

        self._initialize_physical_parameters(param_init_config)
        self.register_buffer('g', torch.tensor(9.81, dtype=torch.float64))
        self.double()

    def _initialize_physical_parameters(self, config):
        method = config['method']
        values = config['values']

        if method == 'fixed':
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

    def forward(self, x):
        x = x.double()

        M1 = torch.clamp(self.M1, min=0.1)
        M2 = torch.clamp(self.M2, min=0.1)
        M3 = torch.clamp(self.M3, min=0.1)
        D1 = torch.clamp(self.D1, min=0.0)
        D2 = torch.clamp(self.D2, min=0.0)
        D3 = torch.clamp(self.D3, min=0.0)
        K1 = torch.clamp(self.K1, min=1.0)
        K2 = torch.clamp(self.K2, min=0.1)
        E1 = torch.clamp(self.E1, min=0.0)

        batch_size = x.size(0)

        M1_reshaped = M1.view(1, 1)
        D1_reshaped = D1.view(1, 1)
        K1_reshaped = K1.view(1, 1)
        K2_reshaped = K2.view(1, 1)
        E1_reshaped = E1.view(1, 1)

        params = torch.cat([
            M1_reshaped,
            D1_reshaped,
            K1_reshaped,
            K2_reshaped,
            E1_reshaped
        ], dim=1)
        params = params.expand(batch_size, -1)

        input_for_unmeasured = torch.cat((x, params), dim=1)
        fA, fB, fC, fD = torch.split(self.NNforUnmeasured(input_for_unmeasured), 1, dim=1)

        self.fA = fA
        self.fB = fB
        self.fC = fC
        self.fD = fD

        x2_ddot, y2_ddot, x3_ddot, y3_ddot = torch.split(self.NNforAccelerations(x), 1, dim=1)
        return torch.cat((x2_ddot, y2_ddot, x3_ddot, y3_ddot), dim=1)

    def compute_residuals(self, x, pred, X_max=None, X_min=None, y_max=None, y_min=None):
        x = x.double()
        pred = pred.double()

        M1 = torch.clamp(self.M1, min=0.1)
        M2 = torch.clamp(self.M2, min=0.1)
        M3 = torch.clamp(self.M3, min=0.1)
        D1 = torch.clamp(self.D1, min=0.0)
        D2 = torch.clamp(self.D2, min=0.0)
        D3 = torch.clamp(self.D3, min=0.0)
        K1 = torch.clamp(self.K1, min=1.0)
        K2 = torch.clamp(self.K2, min=0.1)
        E1 = torch.clamp(self.E1, min=0.0)

        x2_ddot, y2_ddot, x3_ddot, y3_ddot = torch.split(pred, 1, dim=1)
        x2_dot, y2_dot, x3_dot, y3_dot, x2, y2, x3, y3, omega, t = torch.split(x, 1, dim=1)

        if X_max is not None and X_min is not None:
            X_max = X_max.double()
            X_min = X_min.double()
            y_max = y_max.double() if y_max is not None else None
            y_min = y_min.double() if y_min is not None else None

            if X_min.dim() == 1:
                pos_min = X_min[4:8].unsqueeze(0)
                pos_max = X_max[4:8].unsqueeze(0)
            else:
                pos_min = X_min[:, 4:8]
                pos_max = X_max[:, 4:8]

            pos_range = pos_max - pos_min + 1e-12
            positions = torch.cat([x2, y2, x3, y3], dim=1)
            positions_denorm = positions * pos_range + pos_min
            x2_denorm, y2_denorm, x3_denorm, y3_denorm = torch.split(positions_denorm, 1, dim=1)

            if X_min.dim() == 1:
                vel_min = X_min[0:4].unsqueeze(0)
                vel_max = X_max[0:4].unsqueeze(0)
            else:
                vel_min = X_min[:, 0:4]
                vel_max = X_max[:, 0:4]

            vel_range = vel_max - vel_min + 1e-12
            velocities = torch.cat([x2_dot, y2_dot, x3_dot, y3_dot], dim=1)
            velocities_denorm = velocities * vel_range + vel_min
            x2_dot_denorm, y2_dot_denorm, x3_dot_denorm, y3_dot_denorm = torch.split(velocities_denorm, 1, dim=1)

            if y_max is not None and y_min is not None:
                if y_min.dim() == 1:
                    accel_range = (y_max - y_min + 1e-12).unsqueeze(0)
                    y_min_expanded = y_min.unsqueeze(0)
                else:
                    accel_range = y_max - y_min + 1e-12
                    y_min_expanded = y_min

                accelerations = torch.cat([x2_ddot, y2_ddot, x3_ddot, y3_ddot], dim=1)
                accelerations_denorm = accelerations * accel_range + y_min_expanded
                x2_ddot_denorm, y2_ddot_denorm, x3_ddot_denorm, y3_ddot_denorm = torch.split(accelerations_denorm, 1, dim=1)
            else:
                x2_ddot_denorm, y2_ddot_denorm, x3_ddot_denorm, y3_ddot_denorm = x2_ddot, y2_ddot, x3_ddot, y3_ddot

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
            x2_denorm, y2_denorm, x3_denorm, y3_denorm = x2, y2, x3, y3
            x2_dot_denorm, y2_dot_denorm, x3_dot_denorm, y3_dot_denorm = x2_dot, y2_dot, x3_dot, y3_dot
            x2_ddot_denorm, y2_ddot_denorm, x3_ddot_denorm, y3_ddot_denorm = x2_ddot, y2_ddot, x3_ddot, y3_ddot
            omega_phys, t_phys = omega, t

        K2_K1_ratio = K2 / K1

        residual1 = K1*x2_denorm + K2*x3_denorm + M1*omega_phys**2*E1*torch.cos(omega_phys*t_phys) - self.fA
        residual2 = K1*y2_denorm + K2*y3_denorm - M1*self.g + M1*omega_phys**2*E1*torch.sin(omega_phys*t_phys) - self.fB
        residual3 = M3*x3_ddot_denorm + D3*x3_dot_denorm + K2*x3_denorm - K2_K1_ratio*M2*x2_ddot_denorm - K2_K1_ratio*D2*x2_dot_denorm - K2*x2_denorm - self.fC
        residual4 = M3*y3_ddot_denorm + D3*y3_dot_denorm + K2*y3_denorm - K2_K1_ratio*M2*y2_ddot_denorm - K2_K1_ratio*D2*y2_dot_denorm - K2*y2_denorm - K2_K1_ratio*M2*self.g + M3*self.g - self.fD

        if self.enable_mass_constraints:
            residualMass1 = M1 + M2 + M3 - 22.0
            residualMass2 = M2 - M3
            return residual1, residual2, residual3, residual4, residualMass1, residualMass2
        else:
            residualMass1 = torch.zeros_like(residual1)
            residualMass2 = torch.zeros_like(residual2)
            return residual1, residual2, residual3, residual4, residualMass1, residualMass2
