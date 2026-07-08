# -*- coding: utf-8 -*-
"""
Rotating Machinery Simulation with Nonlinear Bearings

This script simulates a three-mass rotating system with nonlinear bearing supports.
It includes comprehensive parameter studies, visualization, and frequency analysis.

System Configuration:
- Mass 1: Central disk with unbalance
- Mass 2: Left bearing support 
- Mass 3: Right bearing support
- Nonlinear bearing stiffness and damping
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.fft import fft, fftfreq
import itertools
from typing import Dict, Tuple, List, Optional
import warnings

# Suppress potential warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

class RotatingMachinerySimulator:
    """
    A comprehensive simulator for rotating machinery with nonlinear bearings.
    """
    
    def __init__(self):
        self.simulation_revolutions = 100
        self.points_per_revolution = 128
        self.analysis_revolutions = 5  # For steady-state analysis
        
    def setup_parameters(self, omega_rad_s: float) -> Tuple[Dict, Dict]:
        """
        Defines physical system parameters and computes dimensionless groups.
        
        Args:
            omega_rad_s: Rotational speed in rad/s
            
        Returns:
            Tuple of (physical_parameters, dimensionless_parameters)
        """
        # Physical parameters (dimensional) in SI units
        p_dim = {
            'M1': 15.0,        # Central disk mass [kg]
            'M2': 1.0,         # Left bearing mass [kg]
            'M3': 1.0,         # Right bearing mass [kg]
            'Ks1': 1.2e6,      # Shaft stiffness 1 [N/m]
            'Ks2': 1.2e6,      # Shaft stiffness 2 [N/m]
            'Ds1': 100.0,      # Shaft damping 1 [Ns/m]
            'Ds2': 100.0,      # Shaft damping 2 [Ns/m]
            'Kb': 5.0e6,       # Linear bearing stiffness [N/m]
            'Db': 700.0,       # Bearing damping [Ns/m]
            'Kb_nl': 5.0e9,    # Nonlinear bearing stiffness [N/m³]
            'mu_eps': 5.0e-5,  # Mass unbalance × eccentricity [kg·m]
            'c': 1.0e-4,       # Characteristic length [m]
            'g': 9.81,         # Gravitational acceleration [m/s²]
            'Omega': omega_rad_s
        }
        
        # Calculate dimensionless parameters
        p_dimless = self._calculate_dimensionless_parameters(p_dim)
        
        return p_dim, p_dimless
    
    def _calculate_dimensionless_parameters(self, p_dim: Dict) -> Dict:
        """Calculate dimensionless parameter groups."""
        M1, c, Omega = p_dim['M1'], p_dim['c'], p_dim['Omega']
        
        return {
            # Mass ratios
            'mu2': p_dim['M2'] / M1,
            'mu3': p_dim['M3'] / M1,
            
            # Stiffness ratios
            'kappa_s1': p_dim['Ks1'] / (M1 * Omega**2),
            'kappa_s2': p_dim['Ks2'] / (M1 * Omega**2),
            'kappa_b': p_dim['Kb'] / (M1 * Omega**2),
            'kappa_b_nl': p_dim['Kb_nl'] * c**2 / (M1 * Omega**2),
            
            # Damping ratios
            'delta_s1': p_dim['Ds1'] / (M1 * Omega),
            'delta_s2': p_dim['Ds2'] / (M1 * Omega),
            'delta_b': p_dim['Db'] / (M1 * Omega),
            
            # Forcing parameters
            'U': p_dim['mu_eps'] / (M1 * c),
            'G1': p_dim['g'] / (c * Omega**2),
            'G2': p_dim['M2'] * p_dim['g'] / (M1 * c * Omega**2),
            'G3': p_dim['M3'] * p_dim['g'] / (M1 * c * Omega**2)
        }
    
    def model(self, tau: float, z: np.ndarray, p: Dict) -> List[float]:
        """
        Defines the system of 12 first-order dimensionless ODEs.
        
        Args:
            tau: Dimensionless time
            z: 12-element state vector [x1,y1,x2,y2,x3,y3,dx1,dy1,dx2,dy2,dx3,dy3]
            p: Dictionary of dimensionless parameters
            
        Returns:
            Derivative of the state vector
        """
        # Unpack state variables
        x1, y1, x2, y2, x3, y3, dx1, dy1, dx2, dy2, dx3, dy3 = z
        
        # Mass 1 (central disk) accelerations
        ddx1 = (p['U'] * np.cos(tau) -
                p['delta_s1'] * (dx1 - dx2) - p['kappa_s1'] * (x1 - x2) -
                p['delta_s2'] * (dx1 - dx3) - p['kappa_s2'] * (x1 - x3))
        
        ddy1 = (p['U'] * np.sin(tau) - p['G1'] -
                p['delta_s1'] * (dy1 - dy2) - p['kappa_s1'] * (y1 - y2) -
                p['delta_s2'] * (dy1 - dy3) - p['kappa_s2'] * (y1 - y3))
        
        # Bearing forces (nonlinear)
        force_bx2 = -p['delta_b'] * dx2 - p['kappa_b'] * x2 - p['kappa_b_nl'] * x2**3
        force_by2 = -p['delta_b'] * dy2 - p['kappa_b'] * y2 - p['kappa_b_nl'] * y2**3
        force_bx3 = -p['delta_b'] * dx3 - p['kappa_b'] * x3 - p['kappa_b_nl'] * x3**3
        force_by3 = -p['delta_b'] * dy3 - p['kappa_b'] * y3 - p['kappa_b_nl'] * y3**3
        
        # Mass 2 (left bearing) accelerations
        ddx2 = (p['delta_s1'] * (dx1 - dx2) + p['kappa_s1'] * (x1 - x2) + force_bx2) / p['mu2']
        ddy2 = (p['delta_s1'] * (dy1 - dy2) + p['kappa_s1'] * (y1 - y2) + force_by2 - p['G2']) / p['mu2']
        
        # Mass 3 (right bearing) accelerations
        ddx3 = (p['delta_s2'] * (dx1 - dx3) + p['kappa_s2'] * (x1 - x3) + force_bx3) / p['mu3']
        ddy3 = (p['delta_s2'] * (dy1 - dy3) + p['kappa_s2'] * (y1 - y3) + force_by3 - p['G3']) / p['mu3']
        
        return [dx1, dy1, dx2, dy2, dx3, dy3, ddx1, ddy1, ddx2, ddy2, ddx3, ddy3]
    
    def run_robust_simulation(self, model_func, t_span: List[float], y0: np.ndarray, 
                            t_eval: np.ndarray, args: Tuple) -> Optional[object]:
        """
        Attempts to solve the ODE system with multiple solvers for robustness.
        """
        solvers_to_try = ['RK45', 'LSODA', 'BDF', 'Radau']
        
        for solver in solvers_to_try:
            try:
                sol = solve_ivp(
                    fun=model_func,
                    t_span=t_span,
                    y0=y0,
                    method=solver,
                    t_eval=t_eval,
                    args=args,
                    dense_output=True,
                    rtol=1e-6,
                    atol=1e-9
                )
                if sol.success:
                    print(f"SUCCESS: Solver '{solver}' completed the integration.")
                    return sol
                else:
                    print(f"INFO: Solver '{solver}' finished but reported failure: {sol.message}")
            except Exception as e:
                print(f"ERROR: Solver '{solver}' raised an exception: {e}")
        
        print("FATAL: All attempted solvers failed.")
        return None
    
    def post_process(self, sol, p_dim: Dict) -> Optional[Dict]:
        """
        Converts dimensionless solution back to dimensional quantities.
        """
        if not sol or not sol.success:
            return None
        
        Omega = p_dim['Omega']
        c = p_dim['c']
        
        # Extract dimensionless results
        tau = sol.t
        z = sol.y
        
        # Convert to dimensional quantities
        time = tau / Omega
        
        # Displacements [m]
        X1, Y1, X2, Y2, X3, Y3 = z[0:6, :] * c
        
        # Velocities [m/s]
        Vx1, Vy1, Vx2, Vy2, Vx3, Vy3 = z[6:12, :] * c * Omega
        
        # Accelerations [m/s²] using numerical differentiation
        Ax1 = np.gradient(Vx1, time)
        Ay1 = np.gradient(Vy1, time)
        Ax2 = np.gradient(Vx2, time)
        Ay2 = np.gradient(Vy2, time)
        Ax3 = np.gradient(Vx3, time)
        Ay3 = np.gradient(Vy3, time)
        
        return {
            'time': time,
            'X1': X1, 'Y1': Y1, 'X2': X2, 'Y2': Y2, 'X3': X3, 'Y3': Y3,
            'Vx1': Vx1, 'Vy1': Vy1, 'Vx2': Vx2, 'Vy2': Vy2, 'Vx3': Vx3, 'Vy3': Vy3,
            'Ax1': Ax1, 'Ay1': Ay1, 'Ax2': Ax2, 'Ay2': Ay2, 'Ax3': Ax3, 'Ay3': Ay3
        }
    
    def run_single_simulation(self, omega_rad_s: float, custom_params: Dict = None) -> Optional[Dict]:
        """
        Run a single simulation with given parameters.
        
        Args:
            omega_rad_s: Rotational speed in rad/s
            custom_params: Dictionary of custom physical parameters to override defaults
            
        Returns:
            Dictionary containing simulation results
        """
        print(f"\n--- Running simulation for Omega: {omega_rad_s:.0f} rad/s ---")
        
        # Setup parameters
        physical_params, dimensionless_params = self.setup_parameters(omega_rad_s)
        
        # Override with custom parameters if provided
        if custom_params:
            for key, value in custom_params.items():
                if key in physical_params:
                    physical_params[key] = value
            # Recalculate dimensionless parameters
            dimensionless_params = self._calculate_dimensionless_parameters(physical_params)
        
        # Time span
        t_final = self.simulation_revolutions * (2 * np.pi / omega_rad_s)
        tau_final = omega_rad_s * t_final
        
        # Evaluation points
        num_points = self.simulation_revolutions * self.points_per_revolution
        tau_eval = np.linspace(0, tau_final, num_points)
        
        # Initial conditions (start from rest)
        z0 = np.zeros(12)
        
        # Run simulation
        solution = self.run_robust_simulation(
            model_func=self.model,
            t_span=[0, tau_final],
            y0=z0,
            t_eval=tau_eval,
            args=(dimensionless_params,)
        )
        
        # Post-process
        if solution:
            results = self.post_process(solution, physical_params)
            if results:
                return {
                    'omega_rad_s': omega_rad_s,
                    'physical_params': physical_params,
                    'results': results
                }
        
        return None
    
    def run_parameter_study(self, omega_list: List[float], parameter_ranges: Dict) -> List[Dict]:
        """
        Run multiple simulations across parameter ranges.
        
        Args:
            omega_list: List of rotational speeds to simulate
            parameter_ranges: Dictionary of parameter ranges to vary
            
        Returns:
            List of simulation results
        """
        simulation_results = []
        
        # Get parameter combinations
        param_keys = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        
        total_combinations = len(omega_list) * np.prod([len(v) for v in param_values])
        current_combination = 0
        
        print(f"\nStarting parameter study with {total_combinations} combinations...")
        
        for omega in omega_list:
            for param_combination_values in itertools.product(*param_values):
                current_combination += 1
                current_params = dict(zip(param_keys, param_combination_values))
                
                print(f"\nProgress: {current_combination}/{total_combinations}")
                print(f"Omega: {omega:.0f} rad/s, Params: {current_params}")
                
                result = self.run_single_simulation(omega, current_params)
                if result:
                    result['varied_params'] = current_params
                    simulation_results.append(result)
                    print("✓ Simulation completed successfully")
                else:
                    print("✗ Simulation failed")
        
        print(f"\nParameter study completed. {len(simulation_results)} successful simulations.")
        return simulation_results
    
    def visualize_single_simulation(self, result: Dict, show_forces: bool = True, 
                                  show_fft: bool = True) -> None:
        """
        Create comprehensive visualizations for a single simulation result.
        """
        if not result or 'results' not in result:
            print("No valid results to visualize.")
            return
        
        results = result['results']
        omega = result['omega_rad_s']
        
        # Get steady-state data for plotting
        plot_points = self.analysis_revolutions * self.points_per_revolution
        
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Main displacement and orbit plots
        fig1, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Central disk displacement
        axs[0, 0].plot(results['time'][-plot_points:], results['X1'][-plot_points:] * 1000, 
                      label='X1', linewidth=1.5)
        axs[0, 0].plot(results['time'][-plot_points:], results['Y1'][-plot_points:] * 1000, 
                      label='Y1', linewidth=1.5, alpha=0.8)
        axs[0, 0].set_title(f"Central Disk Displacement at {omega:.0f} rad/s")
        axs[0, 0].set_ylabel("Displacement (mm)")
        axs[0, 0].legend()
        axs[0, 0].grid(True)
        
        # Central disk orbit
        axs[0, 1].plot(results['X1'][-plot_points:] * 1000, results['Y1'][-plot_points:] * 1000,
                      linewidth=1.5, alpha=0.7)
        axs[0, 1].set_title("Central Disk Orbit")
        axs[0, 1].set_xlabel("X Displacement (mm)")
        axs[0, 1].set_ylabel("Y Displacement (mm)")
        axs[0, 1].axis('equal')
        axs[0, 1].grid(True)
        
        # Left bearing orbit
        axs[1, 0].plot(results['X2'][-plot_points:] * 1000, results['Y2'][-plot_points:] * 1000,
                      'orange', linewidth=1.5, alpha=0.7)
        axs[1, 0].set_title("Left Bearing Orbit")
        axs[1, 0].set_xlabel("X Displacement (mm)")
        axs[1, 0].set_ylabel("Y Displacement (mm)")
        axs[1, 0].axis('equal')
        axs[1, 0].grid(True)
        
        # Right bearing orbit
        axs[1, 1].plot(results['X3'][-plot_points:] * 1000, results['Y3'][-plot_points:] * 1000,
                      'green', linewidth=1.5, alpha=0.7)
        axs[1, 1].set_title("Right Bearing Orbit")
        axs[1, 1].set_xlabel("X Displacement (mm)")
        axs[1, 1].set_ylabel("Y Displacement (mm)")
        axs[1, 1].axis('equal')
        axs[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Bearing forces visualization
        if show_forces:
            self._plot_bearing_forces(results, omega, plot_points)
        
        # FFT analysis
        if show_fft:
            self._plot_fft_analysis(results, omega, plot_points)
    
    def _plot_bearing_forces(self, results: Dict, omega: float, plot_points: int) -> None:
        """Plot bearing forces."""
        # Calculate bearing forces (assuming these parameters from the original)
        Kb, Db, Kb_nl = 5.0e6, 700.0, 5.0e9
        
        Force_Bx2 = -Db * results['Vx2'][-plot_points:] - Kb * results['X2'][-plot_points:] - Kb_nl * results['X2'][-plot_points:]**3
        Force_By2 = -Db * results['Vy2'][-plot_points:] - Kb * results['Y2'][-plot_points:] - Kb_nl * results['Y2'][-plot_points:]**3
        Force_Bx3 = -Db * results['Vx3'][-plot_points:] - Kb * results['X3'][-plot_points:] - Kb_nl * results['X3'][-plot_points:]**3
        Force_By3 = -Db * results['Vy3'][-plot_points:] - Kb * results['Y3'][-plot_points:] - Kb_nl * results['Y3'][-plot_points:]**3
        
        fig2, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        time_steady = results['time'][-plot_points:]
        
        axs[0].plot(time_steady, Force_Bx2, label='Fx2 (Left Bearing)', linewidth=1.5)
        axs[0].plot(time_steady, Force_By2, label='Fy2 (Left Bearing)', linewidth=1.5, alpha=0.8)
        axs[0].set_title(f"Left Bearing Forces at {omega:.0f} rad/s")
        axs[0].set_ylabel("Force (N)")
        axs[0].legend()
        axs[0].grid(True)
        
        axs[1].plot(time_steady, Force_Bx3, label='Fx3 (Right Bearing)', linewidth=1.5)
        axs[1].plot(time_steady, Force_By3, label='Fy3 (Right Bearing)', linewidth=1.5, alpha=0.8)
        axs[1].set_title(f"Right Bearing Forces at {omega:.0f} rad/s")
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("Force (N)")
        axs[1].legend()
        axs[1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_fft_analysis(self, results: Dict, omega: float, plot_points: int) -> None:
        """Plot FFT analysis of accelerations."""
        time_steady = results['time'][-plot_points:]
        if len(time_steady) > 1:
            sampling_freq = 1.0 / (time_steady[1] - time_steady[0])
            
            accel_data = {
                'Ax1': results['Ax1'][-plot_points:],
                'Ay1': results['Ay1'][-plot_points:],
                'Ax2': results['Ax2'][-plot_points:],
                'Ay2': results['Ay2'][-plot_points:],
                'Ax3': results['Ax3'][-plot_points:],
                'Ay3': results['Ay3'][-plot_points:]
            }
            
            fig3, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
            
            mass_labels = ['Central Disk', 'Left Bearing', 'Right Bearing']
            accel_keys = [['Ax1', 'Ay1'], ['Ax2', 'Ay2'], ['Ax3', 'Ay3']]
            
            for i, (mass_label, keys) in enumerate(zip(mass_labels, accel_keys)):
                for key in keys:
                    data = accel_data[key]
                    windowed_data = data * np.hanning(len(data))
                    fft_values = fft(windowed_data)
                    fft_freqs = fftfreq(len(data), 1.0/sampling_freq)
                    
                    positive_freqs = fft_freqs[:len(data)//2]
                    magnitude = 2.0/len(data) * np.abs(fft_values[:len(data)//2])
                    
                    axs[i].plot(positive_freqs, magnitude, label=f'{key} FFT', linewidth=1.5, alpha=0.8)
                
                axs[i].set_title(f"{mass_label} Acceleration FFT at {omega:.0f} rad/s")
                axs[i].set_ylabel("Magnitude")
                axs[i].legend()
                axs[i].grid(True)
                axs[i].set_yscale('log')
                
                # Add rotational frequency line
                rotational_freq_Hz = omega / (2 * np.pi)
                axs[i].axvline(rotational_freq_Hz, color='r', linestyle='--', 
                              label=f'Rotational Freq ({rotational_freq_Hz:.1f} Hz)')
                axs[i].legend()
            
            axs[-1].set_xlabel("Frequency (Hz)")
            plt.tight_layout()
            plt.show()
    
    def analyze_parameter_study_results(self, simulation_results: List[Dict]) -> pd.DataFrame:
        """
        Analyze parameter study results and create summary metrics.
        """
        if not simulation_results:
            print("No simulation results to analyze.")
            return pd.DataFrame()
        
        analysis_data = []
        
        for sim_result in simulation_results:
            omega = sim_result['omega_rad_s']
            varied_params = sim_result.get('varied_params', {})
            results = sim_result['results']
            
            # Calculate steady-state metrics
            plot_points = self.analysis_revolutions * self.points_per_revolution
            
            # Extract steady-state data
            steady_state_data = {key: values[-plot_points:] for key, values in results.items()}
            
            # Calculate key metrics
            metrics = {
                'Omega_rad_s': omega,
                'Max_Abs_X1': np.max(np.abs(steady_state_data['X1'])),
                'Max_Abs_Y1': np.max(np.abs(steady_state_data['Y1'])),
                'Max_Abs_X2': np.max(np.abs(steady_state_data['X2'])),
                'Max_Abs_Y2': np.max(np.abs(steady_state_data['Y2'])),
                'Max_Abs_X3': np.max(np.abs(steady_state_data['X3'])),
                'Max_Abs_Y3': np.max(np.abs(steady_state_data['Y3'])),
                'Max_Abs_Ax1': np.max(np.abs(steady_state_data['Ax1'])),
                'Max_Abs_Ay1': np.max(np.abs(steady_state_data['Ay1'])),
                'RMS_X1': np.sqrt(np.mean(steady_state_data['X1']**2)),
                'RMS_Y1': np.sqrt(np.mean(steady_state_data['Y1']**2)),
                **varied_params  # Include varied parameters
            }
            
            analysis_data.append(metrics)
        
        return pd.DataFrame(analysis_data)
    
    def plot_parameter_trends(self, df_analysis: pd.DataFrame) -> None:
        """
        Plot trends from parameter study analysis.
        """
        if df_analysis.empty:
            print("No analysis data to plot.")
            return
        
        # Identify varied parameters (columns with more than one unique value)
        varied_param_cols = []
        for col in df_analysis.columns:
            if col not in ['Omega_rad_s'] and df_analysis[col].nunique() > 1:
                if col not in ['Max_Abs_X1', 'Max_Abs_Y1', 'Max_Abs_X2', 'Max_Abs_Y2', 
                              'Max_Abs_X3', 'Max_Abs_Y3', 'Max_Abs_Ax1', 'Max_Abs_Ay1', 
                              'RMS_X1', 'RMS_Y1']:
                    varied_param_cols.append(col)
        
        if not varied_param_cols:
            print("No varied parameters found for trend analysis.")
            return
        
        # Metrics to plot
        metrics_to_plot = ['Max_Abs_X1', 'Max_Abs_Y1', 'Max_Abs_Ax1', 'Max_Abs_Ay1']
        metric_labels = {
            'Max_Abs_X1': 'Max |X1| Displacement (m)',
            'Max_Abs_Y1': 'Max |Y1| Displacement (m)',
            'Max_Abs_Ax1': 'Max |Ax1| Acceleration (m/s²)',
            'Max_Abs_Ay1': 'Max |Ay1| Acceleration (m/s²)'
        }
        
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axs = plt.subplots(len(metrics_to_plot), 1, figsize=(12, 16), sharex=True)
        
        for i, metric in enumerate(metrics_to_plot):
            ax = axs[i]
            
            # Group by varied parameters and plot
            if len(varied_param_cols) == 1:
                # Single parameter variation
                param_col = varied_param_cols[0]
                for param_value in df_analysis[param_col].unique():
                    subset = df_analysis[df_analysis[param_col] == param_value].sort_values('Omega_rad_s')
                    ax.plot(subset['Omega_rad_s'], subset[metric], 
                           marker='o', label=f'{param_col} = {param_value:.1e}', linewidth=1.5)
            else:
                # Multiple parameter variations
                param_combinations = df_analysis[varied_param_cols].drop_duplicates()
                for _, combo in param_combinations.iterrows():
                    # Create filter for this combination
                    mask = pd.Series([True] * len(df_analysis))
                    label_parts = []
                    for param_col in varied_param_cols:
                        mask &= (df_analysis[param_col] == combo[param_col])
                        label_parts.append(f'{param_col}={combo[param_col]:.1e}')
                    
                    subset = df_analysis[mask].sort_values('Omega_rad_s')
                    ax.plot(subset['Omega_rad_s'], subset[metric], 
                           marker='o', label=', '.join(label_parts), linewidth=1.5)
            
            ax.set_title(metric_labels[metric])
            ax.set_ylabel(metric_labels[metric].split('(')[0].strip())
            ax.legend()
            ax.grid(True)
        
        axs[-1].set_xlabel("Rotational Speed, Ω (rad/s)")
        plt.tight_layout()
        plt.show()


def main():
    """
    Main execution function demonstrating the simulator capabilities.
    """
    # Initialize simulator
    simulator = RotatingMachinerySimulator()
    
    # Example 1: Single simulation
    print("=== SINGLE SIMULATION EXAMPLE ===")
    single_result = simulator.run_single_simulation(omega_rad_s=800.0)
    
    if single_result:
        simulator.visualize_single_simulation(single_result)
    
    # Example 2: Parameter study
    print("\n=== PARAMETER STUDY EXAMPLE ===")
    omega_list = [600.0, 800.0, 1000.0]
    parameter_ranges = {
        'mu_eps': [5.0e-5, 1.0e-4],  # Mass unbalance variation
        'Kb': [5.0e6, 7.0e6]         # Bearing stiffness variation
    }
    
    # Run parameter study
    param_study_results = simulator.run_parameter_study(omega_list, parameter_ranges)
    
    if param_study_results:
        # Analyze results
        df_analysis = simulator.analyze_parameter_study_results(param_study_results)
        print("\nParameter Study Analysis Results:")
        print(df_analysis.head())
        
        # Plot trends
        simulator.plot_parameter_trends(df_analysis)
        
        # Save results
        try:
            df_analysis.to_csv('parameter_study_results.csv', index=False)
            print("\nResults saved to 'parameter_study_results.csv'")
        except Exception as e:
            print(f"Error saving results: {e}")


class AdvancedAnalyzer:
    """
    Advanced analysis tools for rotating machinery simulation results.
    """
    
    def __init__(self):
        pass
    
    def waterfall_plot(self, simulation_results: List[Dict], mass_component: str = 'Ay1') -> None:
        """
        Create a waterfall plot showing frequency content across different speeds.
        
        Args:
            simulation_results: List of simulation results
            mass_component: Component to analyze ('Ax1', 'Ay1', 'Ax2', etc.)
        """
        if not simulation_results:
            print("No simulation results provided.")
            return
        
        # Extract FFT data for each simulation
        waterfall_data = []
        analysis_revolutions = 5
        points_per_revolution = 128
        plot_points = analysis_revolutions * points_per_revolution
        
        for sim_result in simulation_results:
            omega = sim_result['omega_rad_s']
            results = sim_result['results']
            
            if mass_component not in results:
                print(f"Component {mass_component} not found in results.")
                continue
            
            # Get steady-state data
            time_data = results['time'][-plot_points:]
            component_data = results[mass_component][-plot_points:]
            
            if len(time_data) > 1:
                # Calculate sampling frequency
                sampling_freq = 1.0 / (time_data[1] - time_data[0])
                
                # Apply window and compute FFT
                windowed_data = component_data * np.hanning(len(component_data))
                fft_values = fft(windowed_data)
                fft_freqs = fftfreq(len(component_data), 1.0/sampling_freq)
                
                # Take positive frequencies only
                positive_freqs = fft_freqs[:len(component_data)//2]
                magnitude = 2.0/len(component_data) * np.abs(fft_values[:len(component_data)//2])
                
                waterfall_data.append({
                    'omega': omega,
                    'frequencies': positive_freqs,
                    'magnitude': magnitude
                })
        
        if not waterfall_data:
            print("No valid FFT data for waterfall plot.")
            return
        
        # Sort by omega for consistent plotting
        waterfall_data.sort(key=lambda x: x['omega'])
        
        # Create 3D waterfall plot
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        for i, data in enumerate(waterfall_data):
            omega = data['omega']
            freqs = data['frequencies']
            mags = data['magnitude']
            
            # Limit frequency range for better visualization
            freq_limit = min(500, np.max(freqs))  # Hz
            freq_mask = freqs <= freq_limit
            
            # Plot as a line in 3D space
            ax.plot(freqs[freq_mask], [omega] * np.sum(freq_mask), 
                   np.log10(mags[freq_mask] + 1e-12), alpha=0.8, linewidth=1.5)
        
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Rotational Speed (rad/s)')
        ax.set_zlabel('Log₁₀ Magnitude')
        ax.set_title(f'Waterfall Plot - {mass_component} Acceleration')
        
        plt.tight_layout()
        plt.show()
    
    def campbell_diagram(self, simulation_results: List[Dict], mass_component: str = 'Ay1') -> None:
        """
        Create a Campbell diagram showing resonance frequencies vs rotational speed.
        
        Args:
            simulation_results: List of simulation results
            mass_component: Component to analyze
        """
        if not simulation_results:
            print("No simulation results provided.")
            return
        
        analysis_revolutions = 5
        points_per_revolution = 128
        plot_points = analysis_revolutions * points_per_revolution
        
        omega_values = []
        peak_frequencies = []
        peak_magnitudes = []
        
        for sim_result in simulation_results:
            omega = sim_result['omega_rad_s']
            results = sim_result['results']
            
            if mass_component not in results:
                continue
            
            # Get steady-state data
            time_data = results['time'][-plot_points:]
            component_data = results[mass_component][-plot_points:]
            
            if len(time_data) > 1:
                sampling_freq = 1.0 / (time_data[1] - time_data[0])
                
                # Compute FFT
                windowed_data = component_data * np.hanning(len(component_data))
                fft_values = fft(windowed_data)
                fft_freqs = fftfreq(len(component_data), 1.0/sampling_freq)
                
                positive_freqs = fft_freqs[:len(component_data)//2]
                magnitude = 2.0/len(component_data) * np.abs(fft_values[:len(component_data)//2])
                
                # Find peaks (frequencies with significant magnitude)
                # Use a threshold based on the maximum magnitude
                threshold = 0.1 * np.max(magnitude)
                peak_indices = np.where(magnitude > threshold)[0]
                
                # Store omega and corresponding peak frequencies
                for idx in peak_indices:
                    if positive_freqs[idx] > 1.0:  # Ignore very low frequencies
                        omega_values.append(omega)
                        peak_frequencies.append(positive_freqs[idx])
                        peak_magnitudes.append(magnitude[idx])
        
        if not omega_values:
            print("No peak frequencies found for Campbell diagram.")
            return
        
        # Create Campbell diagram
        plt.figure(figsize=(12, 8))
        
        # Scatter plot with magnitude as color
        scatter = plt.scatter(omega_values, peak_frequencies, 
                            c=np.log10(peak_magnitudes), 
                            s=30, alpha=0.7, cmap='viridis')
        
        # Add 1X line (synchronous frequency)
        omega_range = np.linspace(min(omega_values), max(omega_values), 100)
        sync_freq = omega_range / (2 * np.pi)
        plt.plot(omega_range, sync_freq, 'r--', linewidth=2, 
                label='1X (Synchronous)', alpha=0.8)
        
        # Add 2X line
        plt.plot(omega_range, 2 * sync_freq, 'g--', linewidth=2, 
                label='2X (Second Harmonic)', alpha=0.8)
        
        plt.xlabel('Rotational Speed (rad/s)')
        plt.ylabel('Frequency (Hz)')
        plt.title(f'Campbell Diagram - {mass_component} Acceleration')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.colorbar(scatter, label='Log₁₀ Magnitude')
        
        plt.tight_layout()
        plt.show()
    
    def orbit_analysis(self, simulation_results: List[Dict]) -> None:
        """
        Analyze and compare orbits across different operating conditions.
        """
        if not simulation_results:
            print("No simulation results provided.")
            return
        
        analysis_revolutions = 5
        points_per_revolution = 128
        plot_points = analysis_revolutions * points_per_revolution
        
        # Create subplot for each mass
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        mass_labels = ['Central Disk', 'Left Bearing', 'Right Bearing']
        mass_coords = [('X1', 'Y1'), ('X2', 'Y2'), ('X3', 'Y3')]
        
        for i, (mass_label, (x_coord, y_coord)) in enumerate(zip(mass_labels, mass_coords)):
            ax = axs[i]
            
            for sim_result in simulation_results:
                omega = sim_result['omega_rad_s']
                varied_params = sim_result.get('varied_params', {})
                results = sim_result['results']
                
                # Get steady-state orbit data
                x_data = results[x_coord][-plot_points:] * 1000  # Convert to mm
                y_data = results[y_coord][-plot_points:] * 1000
                
                # Create label
                if varied_params:
                    param_str = ', '.join([f'{k}={v:.1e}' for k, v in varied_params.items()])
                    label = f'Ω={omega:.0f}, {param_str}'
                else:
                    label = f'Ω={omega:.0f} rad/s'
                
                ax.plot(x_data, y_data, alpha=0.7, linewidth=1.5, label=label)
                
                # Mark starting point
                ax.plot(x_data[0], y_data[0], 'o', markersize=4, alpha=0.8)
            
            ax.set_title(f'{mass_label} Orbits')
            ax.set_xlabel('X Displacement (mm)')
            ax.set_ylabel('Y Displacement (mm)')
            ax.axis('equal')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    def statistical_summary(self, simulation_results: List[Dict]) -> pd.DataFrame:
        """
        Generate comprehensive statistical summary of simulation results.
        """
        if not simulation_results:
            print("No simulation results provided.")
            return pd.DataFrame()
        
        analysis_revolutions = 5
        points_per_revolution = 128
        plot_points = analysis_revolutions * points_per_revolution
        
        summary_data = []
        
        for sim_result in simulation_results:
            omega = sim_result['omega_rad_s']
            varied_params = sim_result.get('varied_params', {})
            results = sim_result['results']
            
            # Calculate comprehensive statistics for steady-state data
            stats = {'Omega_rad_s': omega}
            stats.update(varied_params)
            
            # For each displacement and acceleration component
            components = ['X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 
                         'Ax1', 'Ay1', 'Ax2', 'Ay2', 'Ax3', 'Ay3']
            
            for comp in components:
                if comp in results:
                    data = results[comp][-plot_points:]
                    
                    stats.update({
                        f'{comp}_max': np.max(data),
                        f'{comp}_min': np.min(data),
                        f'{comp}_mean': np.mean(data),
                        f'{comp}_std': np.std(data),
                        f'{comp}_rms': np.sqrt(np.mean(data**2)),
                        f'{comp}_max_abs': np.max(np.abs(data)),
                        f'{comp}_peak_to_peak': np.ptp(data)
                    })
            
            summary_data.append(stats)
        
        return pd.DataFrame(summary_data)


def example_advanced_analysis():
    """
    Demonstrate advanced analysis capabilities.
    """
    print("\n=== ADVANCED ANALYSIS EXAMPLE ===")
    
    # Initialize simulator and analyzer
    simulator = RotatingMachinerySimulator()
    analyzer = AdvancedAnalyzer()
    
    # Run a set of simulations for advanced analysis
    omega_list = [400.0, 600.0, 800.0, 1000.0, 1200.0]
    parameter_ranges = {
        'mu_eps': [5.0e-5, 8.0e-5],
        'Kb': [5.0e6, 7.0e6]
    }
    
    # Note: This will run 20 simulations (5 speeds × 2 unbalances × 2 stiffnesses)
    print("Running extended parameter study for advanced analysis...")
    advanced_results = simulator.run_parameter_study(omega_list, parameter_ranges)
    
    if advanced_results:
        print(f"Completed {len(advanced_results)} simulations for advanced analysis.")
        
        # Waterfall plot
        print("\nGenerating waterfall plot...")
        analyzer.waterfall_plot(advanced_results, 'Ay1')
        
        # Campbell diagram
        print("\nGenerating Campbell diagram...")
        analyzer.campbell_diagram(advanced_results, 'Ay1')
        
        # Orbit analysis
        print("\nGenerating orbit analysis...")
        analyzer.orbit_analysis(advanced_results)
        
        # Statistical summary
        print("\nGenerating statistical summary...")
        stats_df = analyzer.statistical_summary(advanced_results)
        print("\nStatistical Summary (first 5 rows):")
        print(stats_df.head())
        
        # Save comprehensive results
        try:
            stats_df.to_csv('advanced_analysis_results.csv', index=False)
            print("\nAdvanced analysis results saved to 'advanced_analysis_results.csv'")
        except Exception as e:
            print(f"Error saving advanced results: {e}")


if __name__ == "__main__":
    # Run main examples
    main()
    
    # Uncomment the line below to run advanced analysis
    example_advanced_analysis()