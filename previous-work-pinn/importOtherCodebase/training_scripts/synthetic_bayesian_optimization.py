#!/usr/bin/env python3
"""
Synthetic Dataset Bayesian Hyperparameter Optimization

This script performs Bayesian optimization of PINN training methods using synthetic data
with known ground truth parameters. The PINN architecture is fixed, and only training
method parameters are optimized.

The evaluation includes:
1. Data fitting error (how well the PINN fits the synthetic data)
2. Physics residual error (how well the PINN satisfies physics constraints)  
3. Parameter estimation error (how accurately the PINN estimates true parameters)

For each simulation, the process:
1. Loads synthetic data with known ground truth parameters
2. For each training method, spawns 2 parallel processes with different random seeds
3. Each process runs 20 Bayesian optimization trials
4. Only varies training method parameters (not PINN architecture)
5. Evaluates parameter estimation accuracy against ground truth
6. Stores results in simulation-specific directory

Usage:
    python synthetic_bayesian_optimization.py [--n_trials 20] [--n_processes_per_method 2] [--test_mode]
"""

import os
import sys
import time
import json
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use dimensional PINN v8 configuration functions
from models.basicPINNv8 import get_default_pinn_config, get_synthetic_pinn_config
print("Using dimensional PINN v8 configuration")


class SyntheticBayesianHyperparameterOptimization:
    """
    Bayesian hyperparameter optimization for PINN training methods using synthetic data.
    
    This class optimizes only the training method parameters while keeping the PINN
    architecture fixed. Each trial trains on a single simulation from the synthetic dataset.
    
    For each simulation, runs 2 parallel processes per method, each with 20 trials.
    """
    
    def __init__(self, output_dir: str = "synthetic_bayesian_results", n_trials: int = 20, 
                 n_processes_per_method: int = 2, test_mode: bool = False, 
                 data_dir: str = "training_scripts/analytical_analysis/data", force_restart: bool = False):
        self.output_dir = Path(output_dir)
        self.n_trials = n_trials
        self.n_processes_per_method = n_processes_per_method
        self.test_mode = test_mode
        self.force_restart = force_restart
        
        # Make data_dir relative to project root
        project_root = Path(__file__).parent.parent
        self.data_dir = project_root / data_dir
        
        # Validate data directory
        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {self.data_dir}")
        
        logger.info(f"Using data directory: {self.data_dir}")
        
        # Fixed PINN architecture (same for all runs)
        self.fixed_pinn_config = get_synthetic_pinn_config()
        
        # Training methods to evaluate
        self.methods = {
            'pecann': 'pecann_training.py',
            'gradnorm': 'gradnorm_training.py',
            'relobralo': 'relobralo_training.py',
            'alpinn': 'alpinn_training.py',
            'adaptive_lbpin': 'adaptive_lbpin_training.py',
            'brdr': 'brdr_training.py',
            'dwpinn': 'dwpinn_training.py',
            'constant_weight': 'constant_weight_pinn_training.py'
        }
        
        # Get available simulation IDs
        self.simulation_ids = self._get_available_simulation_ids()
        
        if not self.simulation_ids:
            raise ValueError(f"No successful simulations found in {self.data_dir}")
        
        logger.info(f"Found {len(self.simulation_ids)} successful simulations")
        logger.info(f"Available simulation IDs: {self.simulation_ids[:10]}...")  # Show first 10
        
        # Validate a few simulations to ensure they have the expected structure
        self._validate_simulation_structure()
        
        # Create output structure
        self._create_output_structure()
        
        # Results storage
        self.all_results = []
        
    def _get_available_simulation_ids(self) -> List[int]:
        """Get list of available simulation IDs from the data directory."""
        simulation_ids = []
        
        if not self.data_dir.exists():
            logger.warning(f"Data directory {self.data_dir} does not exist")
            return simulation_ids
        
        # Look for simulation directories
        for item in self.data_dir.iterdir():
            if item.is_dir() and item.name.startswith("simulation_"):
                try:
                    sim_id = int(item.name.split("_")[1])
                    # Check if simulation has required files
                    time_series_file = item / "time_series.npz"
                    metadata_file = item / "metadata.json"
                    
                    if time_series_file.exists() and metadata_file.exists():
                        # Check if simulation was successful
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        if metadata.get('success', False):
                            simulation_ids.append(sim_id)
                except (ValueError, IndexError):
                    continue
        
        return sorted(simulation_ids)
    
    def _validate_simulation_structure(self):
        """Validate the structure of a few simulation directories to ensure they are valid."""
        logger.info("Validating simulation structure...")
        for sim_id in self.simulation_ids[:5]: # Validate first 5 simulations
            sim_dir = self.data_dir / f"simulation_{sim_id:06d}"
            if not sim_dir.exists():
                logger.warning(f"Simulation directory not found: {sim_dir}")
                continue
            
            time_series_file = sim_dir / "time_series.npz"
            metadata_file = sim_dir / "metadata.json"
            
            if not time_series_file.exists():
                logger.warning(f"Missing time_series.npz in {sim_dir}")
            if not metadata_file.exists():
                logger.warning(f"Missing metadata.json in {sim_dir}")
            
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                if 'success' not in metadata:
                    logger.warning(f"Metadata file missing 'success' key: {metadata_file}")
                if 'parameters' not in metadata and 'ground_truth_params' not in metadata:
                    logger.warning(f"Metadata file missing both 'parameters' and 'ground_truth_params' keys: {metadata_file}")
            except json.JSONDecodeError:
                logger.warning(f"Could not decode JSON from metadata file: {metadata_file}")
            except Exception as e:
                logger.warning(f"Error reading or parsing metadata file {metadata_file}: {e}")
        
        logger.info("Simulation structure validation complete.")
    
    def _create_output_structure(self):
        """Create the output directory structure."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for each simulation
        for sim_id in self.simulation_ids:
            sim_dir = self.output_dir / f"simulation_{sim_id:06d}"
            sim_dir.mkdir(exist_ok=True)
            
            # Create subdirectories for each method
            for method_name in self.methods.keys():
                method_dir = sim_dir / method_name
                method_dir.mkdir(exist_ok=True)
                
                # Create subdirectories for each process
                for process_id in range(self.n_processes_per_method):
                    process_dir = method_dir / f"process_{process_id:02d}"
                    process_dir.mkdir(exist_ok=True)
                    
                    # Create results and optuna_studies directories
                    (process_dir / "results").mkdir(exist_ok=True)
                    (process_dir / "optuna_studies").mkdir(exist_ok=True)
                    
                    # Clean up existing study database if in test mode or force restart
                    if self.test_mode or self.force_restart:
                        study_db = process_dir / "optuna_studies" / "study.db"
                        if study_db.exists():
                            study_db.unlink()
                            logger.info(f"Removed existing study database for {method_name} (sim {sim_id}, process {process_id})")
    
    def _suggest_training_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest training parameters (fixed architecture, only training params)."""
        if self.test_mode:
            # Test mode: 1000 epochs, 2 trials (increased for parameter convergence)
            return {
                'epochs': 1000,
                'batch_size': 32,
                'learning_rate': 1e-2,
                'lr_patience': 20,
                'early_patience': 50,
                'min_delta': 1e-6,
                'max_samples': 100  # Use 1000 samples in test mode
            }
        else:
            # Full mode: 2000 epochs, fixed parameters
            return {
                'epochs': 2000,
                'batch_size': 32,
                'learning_rate': 1e-3,
                'lr_patience': 20,
                'early_patience': 50,
                'min_delta': 1e-6,
                'max_samples': 1000  # Use all available data (5000 points per simulation)
            }
    
    def _suggest_method_params(self, trial: optuna.Trial, method_name: str) -> Dict[str, Any]:
        """Suggest method-specific parameters."""
        if method_name == 'pecann':
            return {
                'mu_initial': trial.suggest_float('mu_initial', 0.1, 10.0, log=True),
                'mu_max': trial.suggest_float('mu_max', 1e3, 1e5, log=True),
                'epsilon': trial.suggest_float('epsilon', 1e-9, 1e-7, log=True)
            }
        elif method_name == 'gradnorm':
            return {
                'alpha': trial.suggest_float('alpha', 0.5, 3.0),
                'weight_lr': trial.suggest_float('weight_lr', 0.01, 0.1)
            }
        elif method_name == 'relobralo':
            return {
                'alpha': trial.suggest_float('alpha', 0.1, 2.0),
                'rho': trial.suggest_float('rho', 0.01, 0.5),
                'temperature': trial.suggest_float('temperature', 0.5, 5.0)
            }
        elif method_name == 'alpinn':
            return {
                'beta': trial.suggest_float('beta', 0.1, 2.0),
                'lambda_lr': trial.suggest_float('lambda_lr', 1e-5, 1e-3, log=True)
            }
        elif method_name == 'adaptive_lbpin':
            return {
                'sigma_init': trial.suggest_float('sigma_init', 0.1, 2.0, log=True)
            }
        elif method_name == 'brdr':
            return {
                'beta_c': trial.suggest_float('beta_c', 0.9, 0.9999),
                'beta_w': trial.suggest_float('beta_w', 0.9, 0.9999),
                'epsilon': trial.suggest_float('epsilon', 1e-9, 1e-7, log=True)
            }
        elif method_name == 'dwpinn':
            return {
                'weight_lr': trial.suggest_float('weight_lr', 1e-4, 1e-2, log=True)
            }
        elif method_name == 'constant_weight':
            # Constant weight method doesn't have method-specific parameters
            # It uses fixed equal weights for all components
            return {}
        else:
            return {}
    
    def _build_command(self, method_name: str, training_params: Dict[str, Any], 
                      method_params: Dict[str, Any], output_dir: str, simulation_id: int) -> List[str]:
        """Build command to run training script."""
        script_path = self.methods[method_name]
        
        # Use relative path from project root
        cmd = [
            sys.executable, f'training_scripts/{script_path}',
            '--epochs', str(training_params['epochs']),
            '--batch-size', str(training_params['batch_size']),
            '--learning-rate', str(training_params['learning_rate']),
            '--lr-patience', str(training_params['lr_patience']),
            '--early-patience', str(training_params['early_patience']),
            '--min-delta', str(training_params['min_delta']),
            '--max-samples', str(training_params['max_samples']),
            '--output-dir', output_dir,
            '--device', 'cpu',  # Force CPU usage
            '--synthetic',
            '--simulation-id', str(simulation_id),
            '--data-path', str(self.data_dir)
        ]
        
        # Add fixed architecture parameters (same for all runs)
        cmd.extend(['--hidden-layers', '128', '128', '64'])  # Fixed architecture
        cmd.extend(['--activation', 'tanh'])
        cmd.extend(['--dropout-rate', '0.1'])
        cmd.extend(['--init-method', 'xavier_normal'])
        
        # Add method-specific parameters with proper name mapping
        method_param_mapping = {
            'pecann': {
                'mu_initial': 'mu-initial',
                'mu_max': 'mu-max', 
                'epsilon': 'epsilon'
            },
            'gradnorm': {
                'alpha': 'alpha',
                'weight_lr': 'weight-lr'
            },
            'relobralo': {
                'alpha': 'alpha',
                'rho': 'rho',
                'temperature': 'temperature'
            },
            'alpinn': {
                'beta': 'beta',
                'lambda_lr': 'lambda-lr'
            },
            'adaptive_lbpin': {
                'sigma_init': 'sigma-init'
            },
            'brdr': {
                'beta_c': 'beta-c',
                'beta_w': 'beta-w',
                'epsilon': 'epsilon'
            },
            'dwpinn': {
                'weight_lr': 'weight-lr'
            },
            'constant_weight': {}
        }
        
        # Add method-specific parameters using the mapping
        param_mapping = method_param_mapping.get(method_name, {})
        for key, value in method_params.items():
            if key in param_mapping:
                arg_name = param_mapping[key]
                cmd.extend([f'--{arg_name}', str(value)])
            else:
                # Fallback: convert underscores to hyphens
                arg_name = key.replace('_', '-')
                cmd.extend([f'--{arg_name}', str(value)])
        
        return cmd
    
    def _run_training(self, method_name: str, training_params: Dict[str, Any], 
                     method_params: Dict[str, Any], output_dir: str, simulation_id: int, 
                     trial: int) -> Dict[str, Any]:
        """Run a single training trial."""
        start_time = time.time()
        
        # Build command
        cmd = self._build_command(method_name, training_params, method_params, output_dir, simulation_id)
        
        # Create log file
        log_file = Path(output_dir) / f"trial_{trial:03d}.log"
        
        # Debug: print command being executed
        logger.debug(f"Executing command: {' '.join(cmd)}")
        logger.debug(f"Output directory: {output_dir}")
        logger.debug(f"Log file: {log_file}")
        
        try:
            # Run training from project root directory
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            logger.debug(f"Running from project root: {project_root}")
            
            with open(log_file, 'w') as f:
                result = subprocess.run(
                    cmd, 
                    stdout=f, 
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=project_root,  # Run from project root so Data/ directory is found
                    timeout=3600  # 1 hour timeout
                )
            
            duration = time.time() - start_time
            
            logger.debug(f"Training completed with return code: {result.returncode}")
            logger.debug(f"Training duration: {duration:.2f} seconds")
            
            if result.returncode == 0:
                # Extract metrics from output
                metrics = self._extract_metrics(output_dir, method_name, trial)
                metrics['duration'] = duration
                metrics['success'] = True
                metrics['log_file'] = str(log_file)
                
                # Extract parameter estimation error
                param_comparison = self._extract_parameter_error(output_dir, method_name, trial, simulation_id)
                metrics['param_comparison'] = param_comparison
                
                logger.debug(f"Successfully extracted metrics: {list(metrics.keys())}")
                return metrics
            else:
                # Try to read the log file to get more details
                try:
                    with open(log_file, 'r') as f:
                        log_content = f.read()
                        # Get last few lines for error context
                        last_lines = log_content.strip().split('\n')[-10:]
                        error_details = f'Training script returned code {result.returncode}. Last 10 lines:\n' + '\n'.join(last_lines)
                        logger.error(f"Training failed: {error_details}")
                except Exception as e:
                    error_details = f'Training script returned code {result.returncode}. Could not read log file: {e}'
                    logger.error(f"Training failed: {error_details}")
                
                return {
                    'success': False,
                    'error': error_details,
                    'duration': duration,
                    'log_file': str(log_file)
                }
                
        except subprocess.TimeoutExpired:
            logger.error(f"Training timed out after 3600 seconds for {method_name} (trial {trial})")
            return {
                'success': False,
                'error': 'Training timed out',
                'duration': time.time() - start_time,
                'log_file': str(log_file)
            }
        except Exception as e:
            logger.error(f"Unexpected error during training for {method_name} (trial {trial}): {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'duration': time.time() - start_time,
                'log_file': str(log_file)
            }
    
    def _extract_metrics(self, output_dir: str, method_name: str, trial: int) -> Dict[str, Any]:
        """Extract training metrics from output files."""
        output_path = Path(output_dir)
        history_file = output_path / f"{method_name}_history.npz"
        
        if not history_file.exists():
            logger.warning(f"History file not found: {history_file}")
            return {'error': 'History file not found'}
        
        try:
            history = np.load(history_file, allow_pickle=True)
            
            # Get final validation metrics
            metrics = {}
            
            # Check for NaN values in validation metrics
            if 'val_total' in history:
                val_total = history['val_total'][-1]
                if np.isnan(val_total) or np.isinf(val_total):
                    logger.warning(f"Invalid val_total for {method_name} (trial {trial}): {val_total}")
                    metrics['final_val_total'] = float('inf')
                else:
                    metrics['final_val_total'] = float(val_total)
            else:
                logger.warning(f"val_total not found in history for {method_name} (trial {trial})")
                metrics['final_val_total'] = float('inf')
            
            if 'data_val' in history:
                data_val = history['data_val'][-1]
                if np.isnan(data_val) or np.isinf(data_val):
                    logger.warning(f"Invalid data_val for {method_name} (trial {trial}): {data_val}")
                    metrics['final_data_val'] = float('inf')
                else:
                    metrics['final_data_val'] = float(data_val)
            else:
                logger.warning(f"data_val not found in history for {method_name} (trial {trial})")
                metrics['final_data_val'] = float('inf')
            
            if 'phys_val' in history:
                phys_val = history['phys_val'][-1]
                if np.isnan(phys_val) or np.isinf(phys_val):
                    logger.warning(f"Invalid phys_val for {method_name} (trial {trial}): {phys_val}")
                    metrics['final_phys_val'] = float('inf')
                else:
                    metrics['final_phys_val'] = float(phys_val)
            else:
                logger.warning(f"phys_val not found in history for {method_name} (trial {trial})")
                metrics['final_phys_val'] = float('inf')
            
            # Debug: print extracted metrics
            logger.debug(f"Extracted metrics for {method_name} (trial {trial}): {metrics}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to extract metrics for {method_name} (trial {trial}): {str(e)}")
            return {'error': f'Failed to extract metrics: {str(e)}'}
    
    def _extract_parameter_error(self, output_dir: str, method_name: str, trial: int, 
                                simulation_id: int) -> Dict[str, Any]:
        """Extract parameter estimation by comparing with ground truth."""
        output_path = Path(output_dir)
        
        # Try to load parameters from the .npz file first (more efficient)
        params_file = output_path / f"{method_name}_parameters.npz"
        
        if not params_file.exists():
            logger.warning(f"Parameters file not found: {params_file}")
            return {'error': 'Parameters file not found'}
        
        try:
            # Load learned parameters from .npz file
            params_data = np.load(params_file, allow_pickle=True)
            learned_params = {}
            
            # Extract parameters from the loaded data
            for param_name in ['M1', 'M2', 'M3', 'D1', 'D2', 'D3', 'K1', 'K2', 'E1']:
                if param_name in params_data:
                    learned_params[param_name] = float(params_data[param_name])
                else:
                    logger.warning(f"Parameter {param_name} not found in {params_file}")
                    learned_params[param_name] = float('nan')
            
            # Load ground truth parameters
            metadata_file = self.data_dir / f"simulation_{simulation_id:06d}" / "metadata.json"
            if not metadata_file.exists():
                logger.error(f"Metadata file not found: {metadata_file}")
                return {'error': 'Metadata file not found'}
            
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Check for parameters in the correct field (synthetic data uses 'parameters', not 'ground_truth_params')
            if 'parameters' in metadata:
                raw_params = metadata['parameters']
                
                # Extract physical parameters
                M1 = raw_params.get('M1', 15.0)
                M2 = raw_params.get('M2', 1.0)
                M3 = raw_params.get('M3', 1.0)
                Omega = raw_params.get('Omega', 800.0)
                c = raw_params.get('c', 1.0e-4)
                Ks1 = raw_params.get('Ks1', 1.2e6)
                Ks2 = raw_params.get('Ks2', 1.2e6)
                Ds1 = raw_params.get('Ds1', 100.0)
                Ds2 = raw_params.get('Ds2', 100.0)
                Db = raw_params.get('Db', 700.0)
                mu_eps = raw_params.get('mu_eps', 5.0e-5)
                
                # Map physical parameters to PINN parameters according to user specification:
                # PINN Parameter → JSON Parameter mapping
                ground_truth = {
                    # Mass parameters (direct mapping)
                    'M1': M1,                            # Mass of the central rotor/disk
                    'M2': M2,                            # Mass of the left bearing assembly
                    'M3': M3,                            # Mass of the right bearing assembly
                    
                    # Stiffness parameters (direct mapping)
                    'K1': Ks1,                          # Stiffness of shaft section between rotor and left bearing
                    'K2': Ks2,                          # Stiffness of shaft section between rotor and right bearing
                    
                    # Damping parameters
                    'D1': (Ds1 + Ds2) / 2,              # Overall shaft damping effects on rotor (average of Ds1/Ds2)
                    'D2': Db,                           # Damping of the bearings (Db used for both bearings)
                    'D3': Db,                           # Additional bearing damping (also maps to Db)
                    
                    # Mass unbalance parameter
                    'E1': mu_eps,                       # Mass unbalance × eccentricity
                    
                    # Characteristic length
                    'c': c,                             # Characteristic length (bearing clearance)
                }
            elif 'ground_truth_params' in metadata:
                ground_truth = metadata['ground_truth_params']
            else:
                logger.error(f"Neither 'parameters' nor 'ground_truth_params' found in metadata: {metadata_file}")
                return {'error': 'Parameters not found in metadata'}
            
            # Store parameter comparison data
            param_comparison = {
                'ground_truth': ground_truth,
                'learned_params': learned_params,
                'absolute_differences': {},
                'valid_params': 0,
                'total_params': 0
            }
            
            # Calculate absolute differences for each parameter
            # Now we can compare all parameters since we have ground truth physical values mapped correctly
            for param_name in ['M1', 'M2', 'M3', 'D1', 'D2', 'D3', 'K1', 'K2', 'E1', 'c']:
                if param_name in ground_truth and param_name in learned_params:
                    true_val = ground_truth[param_name]
                    learned_val = learned_params[param_name]
                    
                    # Check for invalid values
                    if (np.isnan(learned_val) or np.isinf(learned_val) or 
                        np.isnan(true_val) or np.isinf(true_val)):
                        logger.warning(f"Invalid value for {param_name}: learned={learned_val}, true={true_val}")
                        param_comparison['absolute_differences'][param_name] = float('inf')
                        param_comparison['total_params'] += 1
                        continue
                    
                    # Calculate absolute difference
                    abs_diff = abs(learned_val - true_val)
                    param_comparison['absolute_differences'][param_name] = abs_diff
                    param_comparison['total_params'] += 1
                    param_comparison['valid_params'] += 1
            
            # Log parameter comparison summary
            logger.info(f"Parameter estimation for {method_name} (trial {trial}): "
                      f"valid_params={param_comparison['valid_params']}/{param_comparison['total_params']}")
            
            return param_comparison
            
        except Exception as e:
            logger.error(f"Error extracting parameters for {method_name} (trial {trial}): {str(e)}")
            return {'error': str(e)}
    
    def _objective_function(self, method_name: str, simulation_id: int, process_id: int, trial: optuna.Trial) -> float:
        """Objective function for Optuna optimization."""
        # Suggest parameters
        training_params = self._suggest_training_params(trial)
        method_params = self._suggest_method_params(trial, method_name)
        
        # Create trial-specific output directory
        trial_output_dir = (self.output_dir / f"simulation_{simulation_id:06d}" / method_name / 
                           f"process_{process_id:02d}" / "results" / f"trial_{trial.number:03d}")
        trial_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Debug: print trial information
        logger.debug(f"Starting trial {trial.number} for {method_name} (sim {simulation_id}, process {process_id})")
        logger.debug(f"Training params: {training_params}")
        logger.debug(f"Method params: {method_params}")
        
        # Run training
        result = self._run_training(method_name, training_params, method_params, 
                                  str(trial_output_dir), simulation_id, trial.number)
        
        # Store result in trial user attributes
        trial.set_user_attr('result', result)
        
        if result['success'] and 'final_val_total' in result:
            # Check for invalid validation loss
            val_loss = result['final_val_total']
            if np.isnan(val_loss) or np.isinf(val_loss):
                logger.warning(f"Trial {trial.number} for {method_name} (sim {simulation_id}, process {process_id}) "
                             f"produced invalid validation loss: {val_loss}")
                # Return a high value for invalid losses
                return 1e6
            
            # Get parameter error for logging
            param_comparison = result.get('param_comparison', {})
            valid_params = param_comparison.get('valid_params', 'N/A')
            total_params = param_comparison.get('total_params', 'N/A')
            
            # Display parameter comparison for this trial
            if 'error' not in param_comparison:
                ground_truth = param_comparison.get('ground_truth', {})
                learned_params = param_comparison.get('learned_params', {})
                abs_diffs = param_comparison.get('absolute_differences', {})
                
                logger.info(f"Trial {trial.number} for {method_name} (sim {simulation_id}, process {process_id}) SUCCESS:")
                logger.info(f"  Validation Loss: {val_loss:.6f}")
                logger.info(f"  Valid Parameters: {valid_params}/{total_params}")
                logger.info(f"  Parameter Comparison:")
                logger.info(f"    {'Param':<6} {'Ground Truth':<12} {'Estimated':<12} {'|Diff|':<10}")
                logger.info(f"    {'-'*6} {'-'*12} {'-'*12} {'-'*10}")
                
                param_names = ['M1', 'M2', 'M3', 'D1', 'D2', 'D3', 'K1', 'K2', 'E1']
                for param_name in param_names:
                    true_val = ground_truth.get(param_name, 'N/A')
                    learned_val = learned_params.get(param_name, 'N/A')
                    abs_diff = abs_diffs.get(param_name, 'N/A')
                    
                    if isinstance(true_val, (int, float)) and isinstance(learned_val, (int, float)):
                        logger.info(f"    {param_name:<6} {true_val:<12.4f} {learned_val:<12.4f} {abs_diff:<10.4f}")
                    else:
                        logger.info(f"    {param_name:<6} {str(true_val):<12} {str(learned_val):<12} {str(abs_diff):<10}")
            else:
                logger.info(f"Trial {trial.number} for {method_name} (sim {simulation_id}, process {process_id}) SUCCESS: "
                           f"validation_loss={val_loss:.6f}, valid_params={valid_params}/{total_params}")
            
            # Return validation loss as objective (to minimize)
            return val_loss
        else:
            # Log detailed failure information
            failure_reason = "Unknown failure"
            if not result['success']:
                if 'error' in result:
                    failure_reason = f"Training error: {result['error']}"
                else:
                    failure_reason = "Training script returned non-zero exit code"
            elif 'final_val_total' not in result:
                failure_reason = "Missing final_val_total metric in results"
            
            logger.warning(f"Trial {trial.number} for {method_name} (sim {simulation_id}, process {process_id}) FAILED: {failure_reason}")
            
            # Log the command that was run for debugging
            cmd = self._build_command(method_name, training_params, method_params, str(trial_output_dir), simulation_id)
            logger.debug(f"Command that failed: {' '.join(cmd)}")
            
            # Log additional debugging information
            if 'log_file' in result:
                try:
                    with open(result['log_file'], 'r') as f:
                        log_content = f.read()
                        # Get last few lines for error context
                        last_lines = log_content.strip().split('\n')[-20:]  # Last 20 lines
                        logger.debug(f"Last 20 lines of log file:\n" + '\n'.join(last_lines))
                except Exception as e:
                    logger.debug(f"Could not read log file: {e}")
            
            # Return a high value for failed trials
            return 1e6
    
    def optimize_method_simulation_process(self, method_name: str, simulation_id: int, process_id: int) -> optuna.Study:
        """Run Bayesian optimization for a specific method, simulation, and process."""
        logger.info(f"Starting Bayesian optimization for {method_name} (simulation {simulation_id}, process {process_id})")
        
        # Create study
        study_name = f"{method_name}_simulation_{simulation_id}_process_{process_id}"
        study_dir = self.output_dir / f"simulation_{simulation_id:06d}" / method_name / f"process_{process_id:02d}" / "optuna_studies"
        
        # Create sampler with different seeds for different processes
        sampler = TPESampler(seed=simulation_id * 42 + process_id * 17)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        
        # Control whether to load existing studies
        # In test mode or when force_restart is True, don't load existing studies
        load_if_exists = not (self.test_mode or getattr(self, 'force_restart', False))
        
        study = optuna.create_study(
            study_name=study_name,
            storage=f"sqlite:///{study_dir}/study.db",
            sampler=sampler,
            pruner=pruner,
            direction="minimize",
            load_if_exists=load_if_exists
        )
        
        # Run optimization
        study.optimize(
            lambda trial: self._objective_function(method_name, simulation_id, process_id, trial),
            n_trials=self.n_trials,
            n_jobs=1  # Bayesian optimization must be sequential within each study
        )
        
        logger.info(f"Optimization completed for {method_name} (simulation {simulation_id}, process {process_id})")
        logger.info(f"Best value: {study.best_value}")
        logger.info(f"Best parameters: {study.best_params}")
        
        # Display best trial parameter comparison immediately
        best_trial = study.best_trial
        best_result = best_trial.user_attrs.get('result', {})
        if best_result.get('success', False):
            param_comparison = best_result.get('param_comparison', {})
            if 'error' not in param_comparison:
                ground_truth = param_comparison.get('ground_truth', {})
                learned_params = param_comparison.get('learned_params', {})
                abs_diffs = param_comparison.get('absolute_differences', {})
                
                logger.info(f"\n{'='*60}")
                logger.info(f"BEST TRIAL PARAMETERS - {method_name.upper()} (Sim {simulation_id}, Process {process_id})")
                logger.info(f"{'='*60}")
                logger.info(f"Validation Loss: {study.best_value:.6f}")
                logger.info(f"Valid Parameters: {param_comparison.get('valid_params', 0)}/{param_comparison.get('total_params', 0)}")
                logger.info(f"{'Parameter':<8} {'Ground Truth':<15} {'Estimated':<15} {'|Diff|':<12}")
                logger.info(f"{'-'*8} {'-'*15} {'-'*15} {'-'*12}")
                
                param_names = ['M1', 'M2', 'M3', 'D1', 'D2', 'D3', 'K1', 'K2', 'E1']
                for param_name in param_names:
                    true_val = ground_truth.get(param_name, 'N/A')
                    learned_val = learned_params.get(param_name, 'N/A')
                    abs_diff = abs_diffs.get(param_name, 'N/A')
                    
                    if isinstance(true_val, (int, float)) and isinstance(learned_val, (int, float)):
                        logger.info(f"{param_name:<8} {true_val:<15.6f} {learned_val:<15.6f} {abs_diff:<12.6f}")
                    else:
                        logger.info(f"{param_name:<8} {str(true_val):<15} {str(learned_val):<15} {str(abs_diff):<12}")
                logger.info(f"{'='*60}\n")
        
        return study
    
    def run_optimization(self):
        """Run Bayesian optimization for all methods, simulations, and processes."""
        logger.info(f"Starting comprehensive synthetic Bayesian optimization")
        logger.info(f"Methods: {list(self.methods.keys())}")
        logger.info(f"Simulations: {len(self.simulation_ids)}")
        logger.info(f"Processes per method: {self.n_processes_per_method}")
        logger.info(f"Trials per optimization: {self.n_trials}")
        logger.info(f"Output directory: {self.output_dir}")
        
        all_results = []
        studies = {}
        
        # Calculate maximum concurrent processes (methods * processes_per_method)
        max_concurrent_processes = len(self.methods) * self.n_processes_per_method
        logger.info(f"Maximum concurrent processes: {max_concurrent_processes}")
        
        # Process simulations in batches to limit memory usage
        # In test mode, only process the first simulation
        simulation_ids_to_process = [self.simulation_ids[0]] if self.test_mode else self.simulation_ids
        
        for simulation_id in simulation_ids_to_process:
            logger.info(f"Processing simulation {simulation_id}")
            
            # Create tasks for this simulation only
            simulation_tasks = []
            for method_name in self.methods.keys():
                for process_id in range(self.n_processes_per_method):
                    simulation_tasks.append((method_name, simulation_id, process_id))
            
            logger.info(f"Simulation {simulation_id}: {len(simulation_tasks)} tasks")
            
            # Run optimizations for this simulation with limited parallelism
            with ProcessPoolExecutor(max_workers=max_concurrent_processes) as executor:
                # Submit tasks for this simulation
                future_to_task = {
                    executor.submit(self.optimize_method_simulation_process, method_name, simulation_id, process_id): 
                    (method_name, simulation_id, process_id)
                    for method_name, simulation_id, process_id in simulation_tasks
                }
                
                # Collect results for this simulation
                for future in as_completed(future_to_task):
                    method_name, simulation_id, process_id = future_to_task[future]
                    try:
                        study = future.result()
                        studies[(method_name, simulation_id, process_id)] = study
                        
                        # Extract best trial results
                        best_trial = study.best_trial
                        result = {
                            'method': method_name,
                            'simulation_id': simulation_id,
                            'process_id': process_id,
                            'best_value': study.best_value,
                            'best_params': study.best_params,
                            'n_trials': len(study.trials),
                            'user_attrs': best_trial.user_attrs.get('result', {})
                        }
                        all_results.append(result)
                        
                        logger.info(f"Completed {method_name} (sim {simulation_id}, process {process_id}): "
                                  f"best_value={study.best_value:.6f}")
                        
                    except Exception as e:
                        logger.error(f"Error in {method_name} (sim {simulation_id}, process {process_id}): {str(e)}")
            
            logger.info(f"Completed simulation {simulation_id}")
        
        # Save results
        self._save_all_results(all_results)
        self._display_best_model_parameters(all_results, studies)
        self._generate_summary_report(all_results, studies)
        
        logger.info("Synthetic Bayesian optimization completed!")
        return all_results, studies
    
    def _save_all_results(self, results: List[Dict[str, Any]]):
        """Save all results to files."""
        # Save as JSON
        results_file = self.output_dir / "all_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save as CSV
        df = pd.DataFrame(results)
        csv_file = self.output_dir / "all_results.csv"
        df.to_csv(csv_file, index=False)
        
        logger.info(f"Results saved to {results_file} and {csv_file}")
    
    def _display_best_model_parameters(self, results: List[Dict[str, Any]], studies: Dict):
        """Display the best model parameters side by side with ground truth for each method and simulation."""
        logger.info("\n" + "=" * 80)
        logger.info("BEST MODEL PARAMETER COMPARISON")
        logger.info("=" * 80)
        
        # Group results by method and simulation
        method_sim_results = {}
        for result in results:
            if not result['user_attrs'].get('success', False):
                continue
                
            method = result['method']
            sim_id = result['simulation_id']
            key = (method, sim_id)
            
            if key not in method_sim_results:
                method_sim_results[key] = []
            method_sim_results[key].append(result)
        
        # For each method and simulation, find the best result and display parameters
        for (method, sim_id), method_results in method_sim_results.items():
            # Find the best result (lowest validation loss)
            best_result = min(method_results, key=lambda x: x['best_value'])
            
            # Get parameter comparison data
            param_comparison = best_result['user_attrs'].get('param_comparison', {})
            
            if 'error' in param_comparison:
                logger.info(f"\n{method.upper()} - Simulation {sim_id}: {param_comparison['error']}")
                continue
            
            ground_truth = param_comparison.get('ground_truth', {})
            learned_params = param_comparison.get('learned_params', {})
            abs_diffs = param_comparison.get('absolute_differences', {})
            
            logger.info(f"\n{method.upper()} - Simulation {sim_id} (Best Trial)")
            logger.info(f"Validation Loss: {best_result['best_value']:.6f}")
            logger.info("-" * 60)
            logger.info(f"{'Parameter':<8} {'Ground Truth':<15} {'Estimated':<15} {'|Diff|':<12}")
            logger.info("-" * 60)
            
            param_names = ['M1', 'M2', 'M3', 'D1', 'D2', 'D3', 'K1', 'K2', 'E1']
            for param_name in param_names:
                true_val = ground_truth.get(param_name, 'N/A')
                learned_val = learned_params.get(param_name, 'N/A')
                abs_diff = abs_diffs.get(param_name, 'N/A')
                
                if isinstance(true_val, (int, float)) and isinstance(learned_val, (int, float)):
                    logger.info(f"{param_name:<8} {true_val:<15.6f} {learned_val:<15.6f} {abs_diff:<12.6f}")
                else:
                    logger.info(f"{param_name:<8} {str(true_val):<15} {str(learned_val):<15} {str(abs_diff):<12}")
            
            # Save best model parameters to file for easy access
            best_params_file = self.output_dir / f"simulation_{sim_id:06d}" / f"{method}_best_parameters.json"
            best_params_file.parent.mkdir(parents=True, exist_ok=True)
            
            best_params_data = {
                'method': method,
                'simulation_id': sim_id,
                'validation_loss': best_result['best_value'],
                'best_trial_params': best_result['best_params'],
                'ground_truth': ground_truth,
                'learned_params': learned_params,
                'absolute_differences': abs_diffs,
                'valid_params': param_comparison.get('valid_params', 0),
                'total_params': param_comparison.get('total_params', 0)
            }
            
            with open(best_params_file, 'w') as f:
                json.dump(best_params_data, f, indent=2, default=str)
            
            logger.info(f"Best parameters saved to: {best_params_file}")
    
    def _generate_summary_report(self, results: List[Dict[str, Any]], studies: Dict):
        """Generate a comprehensive summary report."""
        if not results:
            logger.warning("No results to summarize")
            return
        
        # Create summary
        summary = {
            'total_optimizations': len(results),
            'methods_evaluated': list(set(r['method'] for r in results)),
            'simulations_evaluated': list(set(r['simulation_id'] for r in results)),
            'processes_per_method': self.n_processes_per_method,
            'trials_per_optimization': self.n_trials,
            'successful_optimizations': len([r for r in results if r['user_attrs'].get('success', False)]),
            'failed_optimizations': len([r for r in results if not r['user_attrs'].get('success', False)])
        }
        
        # Method-wise summary
        method_summary = {}
        for method in summary['methods_evaluated']:
            method_results = [r for r in results if r['method'] == method]
            successful_results = [r for r in method_results if r['user_attrs'].get('success', False)]
            
            method_summary[method] = {
                'total_optimizations': len(method_results),
                'successful_optimizations': len(successful_results),
                'success_rate': len(successful_results) / len(method_results) if method_results else 0,
                'avg_best_value': np.mean([r['best_value'] for r in successful_results]) if successful_results else float('inf')
            }
        
        summary['method_summary'] = method_summary
        
        # Save summary
        summary_file = self.output_dir / "optimization_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Print summary
        logger.info("=" * 60)
        logger.info("SYNTHETIC BAYESIAN OPTIMIZATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total optimizations: {summary['total_optimizations']}")
        logger.info(f"Successful optimizations: {summary['successful_optimizations']}")
        logger.info(f"Failed optimizations: {summary['failed_optimizations']}")
        logger.info(f"Success rate: {summary['successful_optimizations']/summary['total_optimizations']:.2%}")
        
        logger.info("\nMethod Performance Summary:")
        logger.info("-" * 40)
        for method, stats in method_summary.items():
            logger.info(f"{method:15s}: Success={stats['successful_optimizations']}/{stats['total_optimizations']} "
                       f"({stats['success_rate']:.1%}) | "
                       f"Avg Val Loss={stats['avg_best_value']:.6f}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Synthetic Dataset Bayesian Hyperparameter Optimization')
    parser.add_argument('--n_trials', type=int, default=20, help='Number of trials per optimization')
    parser.add_argument('--n_processes_per_method', type=int, default=1, help='Number of parallel processes per method')
    parser.add_argument('--test_mode', action='store_true', help='Run in test mode with fewer trials')
    parser.add_argument('--force_restart', action='store_true', help='Force restart: delete existing studies and start fresh')
    parser.add_argument('--output_dir', type=str, default='synthetic_bayesian_results', help='Output directory')
    parser.add_argument('--data_dir', type=str, default='training_scripts/analytical_analysis/data', help='Synthetic data directory')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.info("Debug logging enabled")
    
    if args.test_mode:
        args.n_trials = 2  # 2 trials in test mode
        logger.info("Running in test mode with 2 trials per optimization")
    
    # Create optimizer
    optimizer = SyntheticBayesianHyperparameterOptimization(
        output_dir=args.output_dir,
        n_trials=args.n_trials,
        n_processes_per_method=args.n_processes_per_method,
        test_mode=args.test_mode,
        data_dir=args.data_dir,
        force_restart=args.force_restart
    )
    
    # Run optimization
    results, studies = optimizer.run_optimization()
    
    logger.info("Synthetic Bayesian optimization completed successfully!")


if __name__ == '__main__':
    main() 