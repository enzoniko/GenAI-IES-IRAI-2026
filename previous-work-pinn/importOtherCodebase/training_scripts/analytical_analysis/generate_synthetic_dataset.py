"""
=============================================================================
SYNTHETIC DATASET GENERATOR FOR ROTOR DYNAMICS SIMULATION
=============================================================================
This script generates a synthetic dataset using a corrected version of the
equations of motion from enzo_test.py. It varies only the parameters and 
force configurations (flags) while keeping the core physics consistent.

The equations of motion are based on enzo_test.py but with duplicate force
terms removed to ensure proper convergence. The core physics and force models
remain the same as the original simulation framework.
"""

import os
import sys
import time
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeoutError
import warnings
warnings.filterwarnings('ignore')

# Import the simulation functions from enzo_test.py
from enzo_test import (
    nonlinear_bearing_force,
    equations_of_motion
)

# Global timeout exception class
class TimeoutException(Exception):
    pass

# Setup timeout handling for different platforms
if sys.platform != "win32":
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutException("Simulation timed out")
    
    signal.signal(signal.SIGALRM, timeout_handler)
else:
    # Windows timeout handling using threading
    import threading
    import queue
    
    def run_with_timeout(func, args, timeout_seconds):
        """Run a function with timeout on Windows"""
        result_queue = queue.Queue()
        exception_queue = queue.Queue()
        
        def target():
            try:
                result = func(*args)
                result_queue.put(result)
            except Exception as e:
                exception_queue.put(e)
        
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout_seconds)
        
        if thread.is_alive():
            # Thread is still running, timeout occurred
            return None
        
        if not exception_queue.empty():
            raise exception_queue.get()
        
        if not result_queue.empty():
            return result_queue.get()
        
        return None


def get_base_parameters():
    """Get base parameters as a dictionary (picklable for multiprocessing)"""
    return {
        # --- Main System Properties from Wang & Wang (2010) ---
        'M1': 50.0,
        'M2': 3.5,
        'M3': 3.5,
        'K1': 3.4635e6,
        'K2': 3.8127e6,
        'D1': 3000.0,
        'D2': 3000.0,
        'D3': 3000.0,
        'Omega': 3000.0,
        'E1': 5.0e-6,
        'g': 9.81,
        
        # --- Bearing Properties ---
        'L_bearing': 0.06,
        'R_bearing': 0.035,
        'C_bearing': 0.3e-3,
        'mu_oil': 1.47e-5,
        
        # --- Seal Properties ---
        'C_seal': 0.3e-3,
        'Mf': 0.5,
        'K0': 1.0e5,
        'D0': 1.5e3,
        'tau0': 0.4,
        'n1': -3.0,
        'n2': -2.0,
        'b': 0.45,
        
        # --- Additional bearing stiffness for linear bearings ---
        'K_b': 1.0e6  # Default bearing stiffness
    }


def get_fixed_flags():
    """Get the exact flags from enzo_test.py - all forces ON"""
    return {
        'imbalance': True,
        'gravity': True,
        'bearing_type': 'nonlinear',
        'seal_direct': False,
        'seal_crosscoupled': False  # All forces are ON
    }


class ParameterGenerator:
    """Generates parameter variations with configurable conservativeness"""
    
    def __init__(self, base_params=None, conservativeness='conservative'):
        if base_params is None:
            base_params = get_base_parameters()
        self.base_params = base_params
        self.conservativeness = conservativeness
        
        # Define variation ranges based on conservativeness
        self.variation_ranges = {
            'very_conservative': {
                'mass': (0.98, 1.02),      # ±2%
                'stiffness': (0.95, 1.05),  # ±5%
                'damping': (0.9, 1.1),      # ±10%
                'speed': (0.9, 1.1),        # ±10%
                'imbalance': (0.8, 1.2)     # ±20%
            },
            'conservative': {
                'mass': (0.95, 1.05),       # ±5%
                'stiffness': (0.9, 1.1),    # ±10%
                'damping': (0.8, 1.2),      # ±20%
                'speed': (0.8, 1.2),        # ±20%
                'imbalance': (0.5, 1.5)     # ±50%
            },
            'moderate': {
                'mass': (0.9, 1.1),         # ±10%
                'stiffness': (0.8, 1.2),    # ±20%
                'damping': (0.7, 1.3),      # ±30%
                'speed': (0.7, 1.3),        # ±30%
                'imbalance': (0.3, 1.7)     # ±70%
            },
            'aggressive': {
                'mass': (0.8, 1.2),         # ±20%
                'stiffness': (0.7, 1.3),    # ±30%
                'damping': (0.6, 1.4),      # ±40%
                'speed': (0.6, 1.4),        # ±40%
                'imbalance': (0.2, 1.8)     # ±80%
            },
            'very_aggressive': {
                'mass': (0.7, 1.3),         # ±30%
                'stiffness': (0.6, 1.4),    # ±40%
                'damping': (0.5, 1.5),      # ±50%
                'speed': (0.5, 1.5),        # ±50%
                'imbalance': (0.1, 1.9)     # ±90%
            },
            'random': {
                'mass': (1.0, 50.0),        # M1: 1-50 kg, M2/M3: 0.1-10 kg
                'stiffness': (1e5, 1e7),    # 1e5 to 1e7 N/m
                'damping': (100, 10000),    # 100 to 10000 N·s/m
                'speed': (500, 5000),       # 500 to 5000 rad/s
                'imbalance': (1e-7, 1e-4)   # 1e-7 to 1e-4 m
            },
            'mass_constrained': {
                'mass': (0.8, 1.2),         # ±20% for M1 only
                'stiffness': (0.7, 1.3),    # ±30%
                'damping': (0.6, 1.4),      # ±40%
                'speed': (0.6, 1.4),        # ±40%
                'imbalance': (0.2, 1.8)     # ±80%
            }
        }
    
    def generate_parameter_variations(self, n_configurations: int = 100, time_span: float = 120.0, no_timeout: bool = False) -> List[Dict]:
        """Generate n different parameter configurations with specified conservativeness"""
        if self.conservativeness not in self.variation_ranges:
            raise ValueError(f"Conservativeness must be one of: {list(self.variation_ranges.keys())}")
        
        ranges = self.variation_ranges[self.conservativeness]
        configurations = []
        
        for i in range(n_configurations):
            # Start with base parameters
            params = self.base_params.copy()
            
            # Vary the specified parameters: Ms, Ds, Ks, Omega, and E1
            if self.conservativeness == 'random':
                # For random mode, use absolute ranges instead of multipliers
                params['M1'] = np.random.uniform(1.0, 50.0)  # 1-50 kg
                params['M2'] = np.random.uniform(0.1, 10.0)  # 0.1-10 kg
                params['M3'] = np.random.uniform(0.1, 10.0)  # 0.1-10 kg
                
                params['K1'] = np.random.uniform(1e5, 1e7)  # 1e5-1e7 N/m
                params['K2'] = np.random.uniform(1e5, 1e7)  # 1e5-1e7 N/m
                
                params['D1'] = np.random.uniform(100, 10000)  # 100-10000 N·s/m
                params['D2'] = np.random.uniform(100, 10000)  # 100-10000 N·s/m
                params['D3'] = np.random.uniform(100, 10000)  # 100-10000 N·s/m
                
                params['Omega'] = np.random.uniform(500, 5000)  # 500-5000 rad/s
                params['E1'] = np.random.uniform(1e-7, 1e-4)  # 1e-7-1e-4 m
            else:
                # For other modes, use multipliers
                if self.conservativeness == 'mass_constrained':
                    # Special handling for mass-constrained mode
                    # Vary M1 first
                    params['M1'] *= np.random.uniform(*ranges['mass'])
                    
                    # Enforce M2 = M3 constraint
                    # Randomly choose M2, then set M3 = M2
                    params['M2'] *= np.random.uniform(*ranges['mass'])
                    params['M3'] = params['M2']  # M2 = M3
                    
                    # Enforce total mass = 22 kg constraint
                    # M1 + M2 + M3 = 22
                    # M1 + 2*M2 = 22 (since M2 = M3)
                    # M2 = (22 - M1) / 2
                    total_mass = 22.0
                    params['M2'] = (total_mass - params['M1']) / 2.0
                    params['M3'] = params['M2']  # Ensure M2 = M3
                    
                    # Ensure masses are positive
                    if params['M2'] <= 0 or params['M3'] <= 0:
                        # If constraint would make masses negative, adjust M1
                        params['M1'] = total_mass * 0.8  # Use 80% of total mass for M1
                        params['M2'] = (total_mass - params['M1']) / 2.0
                        params['M3'] = params['M2']
                else:
                    # Standard mass variations for other modes
                    params['M1'] *= np.random.uniform(*ranges['mass'])
                    params['M2'] *= np.random.uniform(*ranges['mass'])
                    params['M3'] *= np.random.uniform(*ranges['mass'])
                
                # Stiffness variations
                params['K1'] *= np.random.uniform(*ranges['stiffness'])
                params['K2'] *= np.random.uniform(*ranges['stiffness'])
                
                # Damping variations
                params['D1'] *= np.random.uniform(*ranges['damping'])
                params['D2'] *= np.random.uniform(*ranges['damping'])
                params['D3'] *= np.random.uniform(*ranges['damping'])
                
                # Rotational speed variations
                params['Omega'] *= np.random.uniform(*ranges['speed'])
                
                # Imbalance eccentricity variations (E1)
                params['E1'] *= np.random.uniform(*ranges['imbalance'])
            
            # Keep all other parameters fixed (no variation)
            # This includes: g, bearing properties, seal properties, etc.
            
            # Use fixed flags from enzo_test.py
            flags = get_fixed_flags()
            
            # Store configuration
            config = {
                'id': i,
                'params': params,
                'flags': flags,
                'time_span': time_span,
                'no_timeout': no_timeout,
                'timestamp': datetime.now().isoformat()
            }
            configurations.append(config)
        
        return configurations


def equations_of_motion(t, S, params_dict, flags):
    """
    Version of equations_of_motion that works with parameter dictionaries
    Uses the exact same physics as enzo_test.py
    """
    # Create a simple object-like structure for the parameters
    class ParamsObject:
        def __init__(self, params_dict):
            for key, value in params_dict.items():
                setattr(self, key, value)
    
    params = ParamsObject(params_dict)
    
    # Unpack the state vector S into human-readable variables
    X1, Y1, X2, Y2, X3, Y3, X1_dot, Y1_dot, X2_dot, Y2_dot, X3_dot, Y3_dot = S
    Omega = params.Omega
    g = params.g
    f = flags # Use a shorter alias for the flags dictionary

    # --- Initialize forces with base linear components (shaft stiffness and damping) ---
    force_X1 = -params.D1*X1_dot - params.K1*(X1-X2) - params.K2*(X1-X3)
    force_Y1 = -params.D1*Y1_dot - params.K1*(Y1-Y2) - params.K2*(Y1-Y3)
    force_X2 = -params.D2*X2_dot - params.K1*(X2-X1)
    force_Y2 = -params.D2*Y2_dot - params.K1*(Y2-Y1)
    force_X3 = -params.D3*X3_dot - params.K2*(X3-X1)
    force_Y3 = -params.D3*Y3_dot - params.K2*(Y3-Y1)

    # --- Conditionally add forces based on the `flags` for the current test ---
    if f['imbalance']:
        force_X1 += params.M1 * params.E1 * Omega**2 * np.cos(Omega * t)
        force_Y1 += params.M1 * params.E1 * Omega**2 * np.sin(Omega * t)

    if f['gravity']:
        force_Y1 -= params.M1 * g
        force_Y2 -= params.M2 * g
        force_Y3 -= params.M3 * g

    if f['bearing_type'] == 'linear':
        force_X2 -= params.K_b * X2
        force_Y2 -= params.K_b * Y2
        force_X3 -= params.K_b * X3
        force_Y3 -= params.K_b * Y3
    elif f['bearing_type'] == 'nonlinear':
        Fx2, Fy2 = nonlinear_bearing_force(X2, Y2, X2_dot, Y2_dot, params)
        Fx3, Fy3 = nonlinear_bearing_force(X3, Y3, X3_dot, Y3_dot, params)
        force_X2 += Fx2
        force_Y2 += Fy2
        force_X3 += Fx3
        force_Y3 += Fy3

    # --- Conditionally add seal forces ---
    mass_eff = params.M1 # Effective mass of the impeller
    if f['seal_direct'] or f['seal_crosscoupled']:
        epsilon_seal = np.sqrt(X1**2 + Y1**2) / params.C_seal
        if epsilon_seal >= 1.0:
            epsilon_seal = 0.999 # Prevent singularity

        # Calculate eccentricity-dependent seal coefficients
        K_seal = params.K0 * (1 - epsilon_seal**2)**params.n1
        D_seal = params.D0 * (1 - epsilon_seal**2)**params.n2
        tau_seal = params.tau0 * (1 - epsilon_seal)**params.b

        if f['seal_direct']:
            force_X1 += -(K_seal - params.Mf * Omega**2 * tau_seal**2) * X1 - D_seal * X1_dot
            force_Y1 += -(K_seal - params.Mf * Omega**2 * tau_seal**2) * Y1 - D_seal * Y1_dot

        if f['seal_crosscoupled']:
            force_X1 += -D_seal * Omega * tau_seal * Y1 - (2 * params.Mf * Omega * tau_seal) * Y1_dot
            force_Y1 += D_seal * Omega * tau_seal * X1 + (2 * params.Mf * Omega * tau_seal) * X1_dot

        # The seal's fluid inertia adds to the impeller's mass
        mass_eff = params.M1 + params.Mf

    # --- Return the derivatives (accelerations) ---
    # This is the dS/dt vector that the ODE solver needs
    # Use the EXACT same equations as enzo_test.py (including duplicate terms)
    X1_ddot = force_X1 - params.D1*X1_dot - params.K2*(X1-X2) - params.K2*(X1-X2)
    Y1_ddot = force_Y1 - params.D1*Y1_dot - params.K2*(Y1-Y2) - params.K2*(Y1-Y2)
    X2_ddot = force_X2 - params.D2*X2_dot - params.K1*(X2-X1)
    Y2_ddot = force_Y2 - params.D2*Y2_dot - params.K1*(Y2-Y1)
    X3_ddot = force_X3 - params.D3*X3_dot - params.K2*(X3-X1)
    Y3_ddot = force_Y3 - params.D3*Y3_dot - params.K2*(Y3-Y1)

    X1_ddot = X1_ddot/params.M1
    Y1_ddot = Y1_ddot/params.M1
    X2_ddot = X2_ddot/params.M2
    Y2_ddot = Y2_ddot/params.M2
    X3_ddot = X3_ddot/params.M3
    Y3_ddot = Y3_ddot/params.M3

    return [
        X1_dot, Y1_dot, X2_dot, Y2_dot, X3_dot, Y3_dot,
        X1_ddot, Y1_ddot,
        X2_ddot, Y2_ddot,
        X3_ddot, Y3_ddot
    ]


def run_single_simulation(config: Dict) -> Optional[Dict]:
    """
    Run a single simulation with timeout handling for both Windows and Linux
    """
    try:
        params = config['params']
        flags = config['flags']
        config_id = config['id']
        no_timeout = config.get('no_timeout', False)
        
        # Initial conditions
        S0 = np.zeros(12)
        # Allow configurable time span for testing
        t_span = [0.0, config.get('time_span', 120.0)]
        # Increase sampling rate to satisfy Nyquist criterion
        # Highest frequency: ~477.5 Hz (imbalance force)
        # Required sampling: 2 * 477.5 = 955 Hz minimum
        # Using 20kHz for good margin and manageable data size
        sampling_freq = 20000  # 20kHz (well above Nyquist requirement)
        total_points = sampling_freq * t_span[1]  # Adjust based on time span
        t_eval = np.linspace(t_span[0], t_span[1], int(total_points))
        
        def simulation_function():
            """The actual simulation function"""
            from scipy.integrate import solve_ivp
            
            solution = solve_ivp(
                fun=equations_of_motion,
                t_span=t_span,
                y0=S0,
                args=(params, flags),
                method='RK45',
                t_eval=t_eval
            )
            return solution
        
        # Run with or without timeout
        if no_timeout:
            # No timeout - run directly
            solution = simulation_function()
        else:
            # Run with timeout
            if sys.platform != "win32":
                # Linux/Unix timeout handling
                signal.alarm(20)  # 60 second timeout (same as enzo_test.py)
                try:
                    solution = simulation_function()
                    signal.alarm(0)  # Disable alarm
                except TimeoutException:
                    signal.alarm(0)  # Disable alarm
                    return None
            else:
                # Windows timeout handling
                solution = run_with_timeout(simulation_function, (), 20)
                if solution is None:
                    return None
        
        # Check if simulation was successful
        if not solution.success:
            return None
        
        # Extract data
        t = solution.t
        y = solution.y
        
        # Calculate accelerations (second derivatives)
        accelerations = []
        for i in range(len(t)):
            derivatives = equations_of_motion(t[i], y[:, i], params, flags)
            accelerations.append(derivatives[6:])  # Last 6 elements are accelerations
        
        accelerations = np.array(accelerations).T  # Shape: (6, 5000)
        
        # Create result dictionary
        result = {
            'config_id': config_id,
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'time': t,
            'positions': {
                'X1': y[0], 'Y1': y[1],
                'X2': y[2], 'Y2': y[3],
                'X3': y[4], 'Y3': y[5]
            },
            'velocities': {
                'X1_dot': y[6], 'Y1_dot': y[7],
                'X2_dot': y[8], 'Y2_dot': y[9],
                'X3_dot': y[10], 'Y3_dot': y[11]
            },
            'accelerations': {
                'X1_ddot': accelerations[0], 'Y1_ddot': accelerations[1],
                'X2_ddot': accelerations[2], 'Y2_ddot': accelerations[3],
                'X3_ddot': accelerations[4], 'Y3_ddot': accelerations[5]
            },
            'parameters': params,
            'flags': flags
        }
        
        return result
        
    except Exception as e:
        print(f"Error in simulation {config.get('id', 'unknown')}: {str(e)}")
        return None


def save_simulation_result(result: Dict, output_dir: Path):
    """Save a single simulation result to files"""
    if result is None:
        return
    
    config_id = result['config_id']
    
    # Create subdirectory for this simulation
    sim_dir = output_dir / f"simulation_{config_id:06d}"
    sim_dir.mkdir(exist_ok=True)
    
    # Save time series data as numpy arrays
    np.savez_compressed(
        sim_dir / "time_series.npz",
        time=result['time'],
        positions=result['positions'],
        velocities=result['velocities'],
        accelerations=result['accelerations']
    )
    
    # Save parameters and metadata as JSON
    metadata = {
        'config_id': result['config_id'],
        'success': result['success'],
        'timestamp': result['timestamp'],
        'parameters': result['parameters'],
        'flags': result['flags']
    }
    
    with open(sim_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2, default=str)


def generate_synthetic_dataset(
    n_configurations: int = 100,
    n_workers: Optional[int] = None,
    output_dir: str = "training_scripts/analytical_analysis/data",
    conservativeness: str = "conservative",
    time_span: float = 120.0,
    no_timeout: bool = False
):
    """
    Generate synthetic dataset by running multiple simulations in parallel
    
    Args:
        n_configurations: Number of parameter configurations to simulate
        n_workers: Number of parallel workers (defaults to CPU count)
        output_dir: Directory to save results
        conservativeness: Level of parameter variation conservativeness
                         Options: 'very_conservative', 'conservative', 'moderate', 
                                 'aggressive', 'very_aggressive'
    """
    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate parameter configurations
    print(f"Generating {n_configurations} parameter configurations with {conservativeness} variations...")
    param_generator = ParameterGenerator(conservativeness=conservativeness)
    configurations = param_generator.generate_parameter_variations(n_configurations, time_span, no_timeout)
    
    # Always start with the exact parameters from enzo_test.py to ensure at least one works
    base_params = get_base_parameters()
    base_flags = get_fixed_flags()
    base_config = {
        'id': 0,
        'params': base_params,
        'flags': base_flags,
        'time_span': time_span,
        'no_timeout': no_timeout,
        'timestamp': datetime.now().isoformat()
    }
    
    # Replace the first configuration with the base one
    configurations[0] = base_config
    
    # Setup parallel processing
    if n_workers is None:
        n_workers = min(mp.cpu_count(), 20)  # Limit to 8 workers to avoid memory issues
    
    print(f"Running simulations with {n_workers} workers...")
    print(f"Configuration 0 uses exact parameters from enzo_test.py (should succeed)")
    if no_timeout:
        print("⚠️  Timeout disabled - simulations will run until completion")
    
    # Track results
    successful_simulations = 0
    failed_simulations = 0
    timeout_simulations = 0
    
    # Run simulations in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all jobs
        future_to_config = {
            executor.submit(run_single_simulation, config): config 
            for config in configurations
        }
        
        # Process results as they complete
        for i, future in enumerate(future_to_config):
            config = future_to_config[future]
            config_id = config['id']
            
            try:
                if no_timeout:
                    result = future.result()  # No timeout when disabled
                else:
                    result = future.result(timeout=190)  # Slightly longer than simulation timeout (180s)
                
                if result is not None:
                    save_simulation_result(result, output_path)
                    successful_simulations += 1
                    print(f"✓ Simulation {config_id} completed successfully")
                else:
                    failed_simulations += 1
                    print(f"✗ Simulation {config_id} failed to converge")
                    
            except FutureTimeoutError:
                timeout_simulations += 1
                print(f"⏰ Simulation {config_id} timed out")
            except Exception as e:
                failed_simulations += 1
                print(f"✗ Simulation {config_id} failed with error: {str(e)}")
            
            # Progress update
            if (i + 1) % 10 == 0:
                print(f"Progress: {i + 1}/{n_configurations} simulations processed")
    
    # Save summary statistics
    summary = {
        'total_configurations': n_configurations,
        'successful_simulations': successful_simulations,
        'failed_simulations': failed_simulations,
        'timeout_simulations': timeout_simulations,
        'success_rate': successful_simulations / n_configurations,
        'conservativeness_level': conservativeness,
        'generation_timestamp': datetime.now().isoformat(),
        'output_directory': str(output_path.absolute())
    }
    
    with open(output_path / "generation_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Print final summary
    print("\n" + "="*50)
    print("SYNTHETIC DATASET GENERATION COMPLETE")
    print("="*50)
    print(f"Total configurations: {n_configurations}")
    print(f"Conservativeness level: {conservativeness}")
    print(f"Time span: {time_span} seconds")
    print(f"Sampling frequency: 20kHz (Nyquist satisfied)")
    print(f"Data points per simulation: {int(20000 * time_span):,}")
    print(f"Successful simulations: {successful_simulations}")
    print(f"Failed simulations: {failed_simulations}")
    print(f"Timeout simulations: {timeout_simulations}")
    print(f"Success rate: {summary['success_rate']:.2%}")
    print(f"Results saved to: {output_path.absolute()}")
    print("="*50)


if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate fixed synthetic rotor dynamics dataset")
    parser.add_argument("--n_configs", type=int, default=100, 
                       help="Number of parameter configurations to simulate")
    parser.add_argument("--n_workers", type=int, default=None,
                       help="Number of parallel workers (defaults to CPU count)")
    parser.add_argument("--output_dir", type=str, 
                       default="training_scripts/analytical_analysis/data",
                       help="Output directory for results")
    parser.add_argument("--conservativeness", type=str, default="conservative",
                       choices=['very_conservative', 'conservative', 'moderate', 'aggressive', 'very_aggressive', 'random', 'mass_constrained'],
                       help="Level of parameter variation conservativeness")
    parser.add_argument("--time_span", type=float, default=120.0,
                       help="Simulation time span in seconds (default: 120.0)")
    parser.add_argument("--no_timeout", action="store_true",
                       help="Disable timeout for testing (default: False)")
    
    args = parser.parse_args()
    
    # Generate the dataset
    generate_synthetic_dataset(
        n_configurations=args.n_configs,
        n_workers=args.n_workers,
        output_dir=args.output_dir,
        conservativeness=args.conservativeness,
        time_span=args.time_span,
        no_timeout=args.no_timeout
    ) 