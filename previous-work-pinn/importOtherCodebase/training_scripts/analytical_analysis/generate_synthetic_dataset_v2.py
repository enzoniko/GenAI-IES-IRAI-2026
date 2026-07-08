"""
=============================================================================
SYNTHETIC DATASET GENERATOR V2 FOR ROTOR DYNAMICS SIMULATION
=============================================================================
This script generates a synthetic dataset using the stable equations from 
baseForV2.py. It maintains the parameter variation and dataset generation 
capabilities of the original while using the proven stable dimensionless 
equations approach.

Key improvements:
- Uses dimensionless parameter approach for better numerical stability
- Multiple solver attempts for robustness
- Comprehensive parameter variations
- Enhanced visualization capabilities
- Better data organization and storage
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

# Import the stable simulator from baseForV2
from baseForV2 import RotatingMachinerySimulator

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
    """Get base parameters from baseForV2.py for rotor-bearing system"""
    return {
        # Physical parameters (dimensional) in SI units
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
        'Omega': 800.0     # Default rotational speed [rad/s]
    }


class ParameterGeneratorV2:
    """Generates parameter variations for the stable rotor-bearing system"""
    
    def __init__(self, base_params=None, conservativeness='conservative'):
        if base_params is None:
            base_params = get_base_parameters()
        self.base_params = base_params
        self.conservativeness = conservativeness
        
        # Define variation ranges based on conservativeness
        self.variation_ranges = {
            'very_conservative': {
                'mass': (0.98, 1.02),        # ±2%
                'stiffness': (0.95, 1.05),   # ±5%
                'damping': (0.9, 1.1),       # ±10%
                'speed': (0.9, 1.1),         # ±10%
                'unbalance': (0.8, 1.2),     # ±20%
                'bearing': (0.95, 1.05)      # ±5%
            },
            'conservative': {
                'mass': (0.95, 1.05),        # ±5%
                'stiffness': (0.9, 1.1),     # ±10%
                'damping': (0.8, 1.2),       # ±20%
                'speed': (0.8, 1.2),         # ±20%
                'unbalance': (0.5, 1.5),     # ±50%
                'bearing': (0.9, 1.1)        # ±10%
            },
            'moderate': {
                'mass': (0.9, 1.1),          # ±10%
                'stiffness': (0.8, 1.2),     # ±20%
                'damping': (0.7, 1.3),       # ±30%
                'speed': (0.7, 1.3),         # ±30%
                'unbalance': (0.3, 1.7),     # ±70%
                'bearing': (0.8, 1.2)        # ±20%
            },
            'aggressive': {
                'mass': (0.8, 1.2),          # ±20%
                'stiffness': (0.7, 1.3),     # ±30%
                'damping': (0.6, 1.4),       # ±40%
                'speed': (0.6, 1.4),         # ±40%
                'unbalance': (0.2, 1.8),     # ±80%
                'bearing': (0.7, 1.3)        # ±30%
            },
            'very_aggressive': {
                'mass': (0.7, 1.3),          # ±30%
                'stiffness': (0.6, 1.4),     # ±40%
                'damping': (0.5, 1.5),       # ±50%
                'speed': (0.5, 1.5),         # ±50%
                'unbalance': (0.1, 1.9),     # ±90%
                'bearing': (0.6, 1.4)        # ±40%
            },
            'random': {
                'mass': (1.0, 50.0),         # M1: 1-50 kg, M2/M3: 0.1-10 kg
                'stiffness': (1e5, 5e6),     # 1e5 to 5e6 N/m
                'damping': (50, 2000),       # 50 to 2000 N·s/m
                'speed': (300, 2000),        # 300 to 2000 rad/s
                'unbalance': (1e-6, 1e-3),   # 1e-6 to 1e-3 kg·m
                'bearing': (1e6, 2e7)        # 1e6 to 2e7 N/m
            },
            'mass_constrained': {
                'mass': (0.8, 1.2),          # ±20% for M1 only
                'stiffness': (0.7, 1.3),     # ±30%
                'damping': (0.6, 1.4),       # ±40%
                'speed': (0.6, 1.4),         # ±40%
                'unbalance': (0.2, 1.8),     # ±80%
                'bearing': (0.7, 1.3)        # ±30%
            }
        }
    
    def generate_parameter_variations(self, n_configurations: int = 100, 
                                    time_revolutions: int = 100) -> List[Dict]:
        """Generate n different parameter configurations"""
        if self.conservativeness not in self.variation_ranges:
            raise ValueError(f"Conservativeness must be one of: {list(self.variation_ranges.keys())}")
        
        ranges = self.variation_ranges[self.conservativeness]
        configurations = []
        
        for i in range(n_configurations):
            # Start with base parameters
            params = self.base_params.copy()
            
            if self.conservativeness == 'random':
                # For random mode, use absolute ranges
                params['M1'] = np.random.uniform(5.0, 50.0)      # 5-50 kg
                params['M2'] = np.random.uniform(0.5, 5.0)       # 0.5-5 kg
                params['M3'] = np.random.uniform(0.5, 5.0)       # 0.5-5 kg
                
                params['Ks1'] = np.random.uniform(5e5, 5e6)      # 0.5-5 MN/m
                params['Ks2'] = np.random.uniform(5e5, 5e6)      # 0.5-5 MN/m
                
                params['Ds1'] = np.random.uniform(50, 500)       # 50-500 N·s/m
                params['Ds2'] = np.random.uniform(50, 500)       # 50-500 N·s/m
                
                params['Kb'] = np.random.uniform(1e6, 2e7)       # 1-20 MN/m
                params['Db'] = np.random.uniform(200, 2000)      # 200-2000 N·s/m
                params['Kb_nl'] = np.random.uniform(1e8, 1e10)   # 0.1-10 GN/m³
                
                params['Omega'] = np.random.uniform(300, 2000)   # 300-2000 rad/s
                params['mu_eps'] = np.random.uniform(1e-6, 1e-3) # 1e-6 to 1e-3 kg·m
                
                # Keep characteristic length in reasonable range
                params['c'] = np.random.uniform(5e-5, 5e-4)      # 50-500 μm
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
                    
                    # Enforce total mass = 17 kg constraint (15 + 1 + 1 from base)
                    # M1 + M2 + M3 = 17
                    # M1 + 2*M2 = 17 (since M2 = M3)
                    # M2 = (17 - M1) / 2
                    total_mass = 17.0
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
                params['Ks1'] *= np.random.uniform(*ranges['stiffness'])
                params['Ks2'] *= np.random.uniform(*ranges['stiffness'])
                
                # Damping variations
                params['Ds1'] *= np.random.uniform(*ranges['damping'])
                params['Ds2'] *= np.random.uniform(*ranges['damping'])
                
                # Bearing parameters
                params['Kb'] *= np.random.uniform(*ranges['bearing'])
                params['Db'] *= np.random.uniform(*ranges['damping'])
                params['Kb_nl'] *= np.random.uniform(*ranges['bearing'])
                
                # Rotational speed variations
                params['Omega'] *= np.random.uniform(*ranges['speed'])
                
                # Unbalance variations
                params['mu_eps'] *= np.random.uniform(*ranges['unbalance'])
                
                # Small variations in characteristic length
                params['c'] *= np.random.uniform(0.5, 2.0)
            
            # Store configuration
            config = {
                'id': i,
                'params': params,
                'time_revolutions': time_revolutions,
                'timestamp': datetime.now().isoformat()
            }
            configurations.append(config)
        
        return configurations


def run_single_simulation_v2(config: Dict) -> Optional[Dict]:
    """
    Run a single simulation using the stable equations from baseForV2.py
    """
    try:
        params = config['params']
        config_id = config['id']
        time_revolutions = config.get('time_revolutions', 100)
        
        # Initialize simulator
        simulator = RotatingMachinerySimulator()
        simulator.simulation_revolutions = time_revolutions
        
        def simulation_function():
            """The actual simulation function using baseForV2"""
            omega_rad_s = params['Omega']
            
            # Ensure at least 1 second of data and 20kHz sampling
            min_time = 1.0  # At least 1 second
            target_sampling_freq = 20000  # 20kHz
            
            # Calculate time per revolution
            time_per_revolution = 2 * np.pi / omega_rad_s
            
            # Calculate minimum revolutions needed for 1 second
            min_revolutions = max(time_revolutions, int(np.ceil(min_time / time_per_revolution)))
            
            # Calculate points per revolution for 20kHz sampling
            points_per_revolution = int(np.ceil(target_sampling_freq * time_per_revolution))
            
            # Update simulator settings
            simulator.simulation_revolutions = min_revolutions
            simulator.points_per_revolution = max(128, points_per_revolution)  # At least 128 for stability
            
            # Extract custom parameters (all except Omega which is handled separately)
            custom_params = {k: v for k, v in params.items() if k != 'Omega'}
            
            # Run simulation using the stable implementation
            result = simulator.run_single_simulation(omega_rad_s, custom_params)
            return result
        
        # Run with timeout
        if sys.platform != "win32":
            # Linux/Unix timeout handling
            signal.alarm(180)  # 3 minute timeout
            try:
                result = simulation_function()
                signal.alarm(0)  # Disable alarm
            except TimeoutException:
                signal.alarm(0)  # Disable alarm
                return None
        else:
            # Windows timeout handling
            result = run_with_timeout(simulation_function, (), 180)
            if result is None:
                return None
        
        # Check if simulation was successful
        if result is None:
            return None
        
        # Reorganize data to match expected format
        time_data = result['results']['time']
        
        # Create comprehensive result dictionary
        final_result = {
            'config_id': config_id,
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'time': time_data,
            'positions': {
                'X1': result['results']['X1'],
                'Y1': result['results']['Y1'],
                'X2': result['results']['X2'],
                'Y2': result['results']['Y2'],
                'X3': result['results']['X3'],
                'Y3': result['results']['Y3']
            },
            'velocities': {
                'X1_dot': result['results']['Vx1'],
                'Y1_dot': result['results']['Vy1'],
                'X2_dot': result['results']['Vx2'],
                'Y2_dot': result['results']['Vy2'],
                'X3_dot': result['results']['Vx3'],
                'Y3_dot': result['results']['Vy3']
            },
            'accelerations': {
                'X1_ddot': result['results']['Ax1'],
                'Y1_ddot': result['results']['Ay1'],
                'X2_ddot': result['results']['Ax2'],
                'Y2_ddot': result['results']['Ay2'],
                'X3_ddot': result['results']['Ax3'],
                'Y3_ddot': result['results']['Ay3']
            },
            'parameters': result['physical_params'],
            'omega_rad_s': result['omega_rad_s']
        }
        
        return final_result
        
    except Exception as e:
        print(f"Error in simulation {config.get('id', 'unknown')}: {str(e)}")
        return None


def map_to_pinn_parameters(synthetic_params: Dict) -> Dict:
    """
    Map synthetic data parameters to PINN model parameters.
    
    The synthetic data uses a 3-mass rotor-bearing system with shaft stiffness/damping,
    while the PINN model uses a simplified 2-mass system with effective stiffness/damping.
    
    Parameter mapping rationale:
    - Masses: Direct mapping (M1, M2, M3)
    - Stiffnesses: Ks1 → K1, Ks2 → K2 (combine shaft and bearing effects)
    - Damping: Ds1 → D1, Ds2 → D2, Db → D3 (separate shaft and bearing damping)
    - Unbalance: mu_eps → E1 (eccentricity)
    
    Args:
        synthetic_params: Dictionary with synthetic data parameters
        
    Returns:
        Dictionary with PINN-compatible parameters
    """
    # Extract parameters
    M1 = synthetic_params.get('M1', 15.0)
    M2 = synthetic_params.get('M2', 1.0)
    M3 = synthetic_params.get('M3', 1.0)
    Ks1 = synthetic_params.get('Ks1', 1.2e6)
    Ks2 = synthetic_params.get('Ks2', 1.2e6)
    Ds1 = synthetic_params.get('Ds1', 100.0)
    Ds2 = synthetic_params.get('Ds2', 100.0)
    Kb = synthetic_params.get('Kb', 5.0e6)
    Db = synthetic_params.get('Db', 700.0)
    mu_eps = synthetic_params.get('mu_eps', 5.0e-5)
    
    # Map to PINN parameters
    pinn_params = {
        # Masses (direct mapping)
        'M1': M1,  # Central disk mass
        'M2': M2,  # Left bearing mass
        'M3': M3,  # Right bearing mass
        
        # Stiffnesses (combine shaft and bearing effects)
        'K1': Ks1 + Kb,  # Primary stiffness (shaft 1 + bearing)
        'K2': Ks2 + Kb,  # Secondary stiffness (shaft 2 + bearing)
        
        # Damping (separate shaft and bearing)
        'D1': Ds1,  # Primary damping (shaft 1)
        'D2': Ds2,  # Secondary damping (shaft 2)
        'D3': Db,   # Bearing damping
        
        # Eccentricity (unbalance converted to eccentricity)
        'E1': mu_eps / M1 if M1 > 0 else 0.0,  # Eccentricity = unbalance / mass
        
        # Additional parameters for reference (not used by PINN but useful for analysis)
        'Omega': synthetic_params.get('Omega', 800.0),  # Rotational speed
        'g': synthetic_params.get('g', 9.81),  # Gravity
    }
    
    return pinn_params


def save_simulation_result_v2(result: Dict, output_dir: Path):
    """Save a single simulation result to files with enhanced format"""
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
    
    # Get both original and PINN-mapped parameters
    original_params = result['parameters']
    pinn_params = map_to_pinn_parameters(original_params)
    
    # Save parameters and metadata as JSON
    metadata = {
        'config_id': result['config_id'],
        'success': result['success'],
        'timestamp': result['timestamp'],
        'parameters': original_params,  # Original synthetic data parameters
        'pinn_parameters': pinn_params,  # PINN-compatible parameters
        'omega_rad_s': result['omega_rad_s'],
        'simulation_info': {
            'total_time': float(result['time'][-1] - result['time'][0]),
            'time_points': len(result['time']),
            'sampling_rate': len(result['time']) / (result['time'][-1] - result['time'][0])
        },
        'parameter_mapping_info': {
            'description': 'Synthetic data parameters mapped to PINN model parameters',
            'mapping_rationale': {
                'M1, M2, M3': 'Direct mass mapping',
                'K1': 'Ks1 + Kb (shaft 1 + bearing stiffness)',
                'K2': 'Ks2 + Kb (shaft 2 + bearing stiffness)',
                'D1': 'Ds1 (shaft 1 damping)',
                'D2': 'Ds2 (shaft 2 damping)',
                'D3': 'Db (bearing damping)',
                'E1': 'mu_eps / M1 (eccentricity from unbalance)'
            }
        }
    }
    
    with open(sim_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2, default=str)


def generate_synthetic_dataset_v2(
    n_configurations: int = 100,
    n_workers: Optional[int] = None,
    output_dir: str = "training_scripts/analytical_analysis/data_v2",
    conservativeness: str = "conservative",
    time_revolutions: int = 100
):
    """
    Generate synthetic dataset using stable equations from baseForV2.py
    
    Args:
        n_configurations: Number of parameter configurations to simulate
        n_workers: Number of parallel workers (defaults to CPU count)
        output_dir: Directory to save results
        conservativeness: Level of parameter variation conservativeness
        time_revolutions: Number of revolutions to simulate
    """
    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate parameter configurations
    print(f"Generating {n_configurations} parameter configurations with {conservativeness} variations...")
    param_generator = ParameterGeneratorV2(conservativeness=conservativeness)
    configurations = param_generator.generate_parameter_variations(n_configurations, time_revolutions)
    
    # Always start with the exact parameters from baseForV2.py
    base_params = get_base_parameters()
    base_config = {
        'id': 0,
        'params': base_params,
        'time_revolutions': time_revolutions,
        'timestamp': datetime.now().isoformat()
    }
    
    # Replace the first configuration with the base one
    configurations[0] = base_config
    
    # Setup parallel processing
    if n_workers is None:
        n_workers = min(mp.cpu_count(), 8)  # Limit to avoid memory issues
    
    print(f"Running simulations with {n_workers} workers...")
    print(f"Configuration 0 uses exact parameters from baseForV2.py (should succeed)")
    print(f"Each simulation runs for {time_revolutions} revolutions")
    
    # Track results
    successful_simulations = 0
    failed_simulations = 0
    timeout_simulations = 0
    
    # Run simulations in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all jobs
        future_to_config = {
            executor.submit(run_single_simulation_v2, config): config 
            for config in configurations
        }
        
        # Process results as they complete
        for i, future in enumerate(future_to_config):
            config = future_to_config[future]
            config_id = config['id']
            
            try:
                result = future.result(timeout=200)  # Slightly longer than simulation timeout
                
                if result is not None:
                    save_simulation_result_v2(result, output_path)
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
        'time_revolutions': time_revolutions,
        'generation_timestamp': datetime.now().isoformat(),
        'output_directory': str(output_path.absolute()),
        'base_parameters': get_base_parameters()
    }
    
    with open(output_path / "generation_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Print final summary
    print("\n" + "="*60)
    print("SYNTHETIC DATASET GENERATION V2 COMPLETE")
    print("="*60)
    print(f"Total configurations: {n_configurations}")
    print(f"Conservativeness level: {conservativeness}")
    print(f"Simulation revolutions: {time_revolutions}")
    print(f"Using stable equations from baseForV2.py")
    print(f"Successful simulations: {successful_simulations}")
    print(f"Failed simulations: {failed_simulations}")
    print(f"Timeout simulations: {timeout_simulations}")
    print(f"Success rate: {summary['success_rate']:.2%}")
    print(f"Results saved to: {output_path.absolute()}")
    print("="*60)


if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic rotor dynamics dataset using stable equations")
    parser.add_argument("--n_configs", type=int, default=100, 
                       help="Number of parameter configurations to simulate")
    parser.add_argument("--n_workers", type=int, default=None,
                       help="Number of parallel workers (defaults to CPU count)")
    parser.add_argument("--output_dir", type=str, 
                       default="training_scripts/analytical_analysis/data_v2",
                       help="Output directory for results")
    parser.add_argument("--conservativeness", type=str, default="random",
                       choices=['very_conservative', 'conservative', 'moderate', 'aggressive', 'very_aggressive', 'random', 'mass_constrained'],
                       help="Level of parameter variation conservativeness")
    parser.add_argument("--time_revolutions", type=int, default=100,
                       help="Number of revolutions to simulate (default: 100)")
    
    args = parser.parse_args()
    
    # Generate the dataset
    generate_synthetic_dataset_v2(
        n_configurations=args.n_configs,
        n_workers=args.n_workers,
        output_dir=args.output_dir,
        conservativeness=args.conservativeness,
        time_revolutions=args.time_revolutions
    )
