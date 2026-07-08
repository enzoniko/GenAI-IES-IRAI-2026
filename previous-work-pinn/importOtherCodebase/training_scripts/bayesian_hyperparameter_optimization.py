#!/usr/bin/env python3
"""
Unified Bayesian Hyperparameter Optimization for PINN Training Methods

This script performs comprehensive Bayesian optimization for all 8 PINN training methods,
optimizing both PINN architecture parameters and method-specific hyperparameters.

Methods supported:
1. adaptive_lbpin_training.py - Gaussian Likelihood Loss
2. gradnorm_training.py - GradNorm Loss Balancing
3. alpinn_training.py - Adaptive Loss PINN
4. brdr_training.py - Balanced Residual Distribution
5. relobralo_training.py - ReLoBRaLo Method
6. pecann_training.py - Physics-Informed Neural Networks
7. dwpinn_training.py - Dynamic Weight PINN
8. constant_weight_pinn_training.py - Constant Weight PINN

Optimization includes:
- PINN Architecture: hidden layers, activation, dropout, initialization
- Training Parameters: learning rate, batch size, epochs
- Method-specific parameters: alpha, beta, gamma, sigma_init, etc.

Optimization Metric: Final validation total loss (minimization)

Usage:
    python bayesian_hyperparameter_optimization.py [--n-trials 100] [--n-jobs 2] [--output-dir bayesian_results]
"""

import os
import sys
import subprocess
import argparse
import json
import time
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple
import logging

# Optuna for Bayesian optimization
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BayesianHyperparameterOptimization:
    """
    Comprehensive Bayesian hyperparameter optimization for PINN training methods.
    """
    
    def __init__(self, output_dir: str = "bayesian_results", n_trials: int = 100, n_jobs: int = 2, test_mode: bool = False, force_restart: bool = False):
        self.output_dir = Path(output_dir)
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.test_mode = test_mode
        self.force_restart = force_restart
        
        # Define method configurations with both architecture and method-specific parameters
        self.methods = {
            'adaptive_lbpin': {
                'script': 'adaptive_lbpin_training.py',
                'method_params': {
                    'sigma_init': (0.1, 2.0)  # log scale, corresponds to σ ≈ 0.1 to 7.4
                }
            },
            'gradnorm': {
                'script': 'gradnorm_training.py',
                'method_params': {
                    'alpha': (0.5, 3.0),
                    'weight_lr': (0.01, 0.1)
                }
            },
            'alpinn': {
                'script': 'alpinn_training.py',
                'method_params': {
                    'beta': (0.1, 2.0),
                    'lambda_lr': (1e-5, 1e-3)
                }
            },
            'brdr': {
                'script': 'brdr_training.py',
                'method_params': {
                    'beta_c': (0.9, 0.9999),
                    'beta_w': (0.9, 0.9999),
                    'epsilon': (1e-9, 1e-7)
                }
            },
            'relobralo': {
                'script': 'relobralo_training.py',
                'method_params': {
                    'alpha': (0.1, 2.0),
                    'rho': (0.01, 0.5),
                    'temperature': (0.5, 5.0)
                }
            },
            'pecann': {
                'script': 'pecann_training.py',
                'method_params': {
                    'mu_initial': (0.1, 10.0),
                    'mu_max': (1e3, 1e5),
                    'epsilon': (1e-9, 1e-7)
                }
            },
            'dwpinn': {
                'script': 'dwpinn_training.py',
                'method_params': {
                    'weight_lr': (1e-4, 1e-2)
                }
            },
            'constant_weight': {
                'script': 'constant_weight_pinn_training.py',
                'method_params': {
                    # Per-component loss weights (log scale to explore wide range)
                    'w_data': (1e-3, 1e3),
                    'w_res1': (1e-3, 1e3),
                    'w_res2': (1e-3, 1e3),
                    'w_res3': (1e-3, 1e3),
                    'w_res4': (1e-3, 1e3),
                    # Mass constraints present only for real data; still include here
                    'w_mass1': (1e-3, 1e3),
                    'w_mass2': (1e-3, 1e3)
                }
            }
        }
        
        # Define architecture and training parameter ranges
        self.architecture_params = {
            # Hidden layer configurations (different network sizes)
            'hidden_layers_config': [
                [64, 64],           # Small network
                [128, 128],         # Medium network
                [256, 256],         # Large network
                [128, 128, 64],     # Medium with 3 layers
                [256, 128, 64],     # Large to small
                [64, 128, 256],     # Small to large
                [128, 256, 128],    # Medium-large-medium
                [256, 256, 128, 64] # Very large network
            ],
            'activation': ['tanh', 'relu', 'leaky_relu', 'elu', 'selu', 'gelu'],
            'dropout_rate': (0.0, 0.3),
            'init_method': ['xavier_normal', 'xavier_uniform', 'kaiming_normal', 'kaiming_uniform']
        }
        
        self.training_params = {
            'batch_size': [128, 256, 512],
            'early_patience': [100, 200, 300],
            'lr_patience': [25, 50, 75]
        }
        
        # Fixed parameters for consistency (let schedulers handle learning rate and epochs)
        if self.test_mode:
            # Short epochs for testing
            epochs = 20
            max_samples = 100
        else:
            # Full training for overnight runs
            epochs = 20000
            max_samples = 1000
            
        self.fixed_params = {
            '--epochs': epochs,       # Many epochs, let early stopping handle it
            '--max-samples': max_samples,
            '--device': 'cpu',
            '--min-delta': 1e-7
        }
        
        # Create output directory structure
        self._create_output_structure()
        
    def _create_output_structure(self):
        """Create the output directory structure."""
        self.output_dir.mkdir(exist_ok=True)
        
        # Create directories for each method
        for method_name in self.methods.keys():
            method_dir = self.output_dir / method_name
            method_dir.mkdir(exist_ok=True)
            
            # Create subdirectories for different starting points
            for i in range(2):  # 2 different starting points
                start_point_dir = method_dir / f"starting_point_{i+1}"
                start_point_dir.mkdir(exist_ok=True)
                
                # Create directories for results
                (start_point_dir / "results").mkdir(exist_ok=True)
                (start_point_dir / "logs").mkdir(exist_ok=True)
                (start_point_dir / "plots").mkdir(exist_ok=True)
                (start_point_dir / "optuna_studies").mkdir(exist_ok=True)
        
        # Create global results directory
        (self.output_dir / "global_results").mkdir(exist_ok=True)
    
    def _suggest_architecture_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest PINN architecture parameters."""
        params = {}
        
        # Hidden layers (categorical choice)
        # Use static list to avoid dynamic value space issues
        hidden_layers_choices = list(range(len(self.architecture_params['hidden_layers_config'])))
        hidden_layers_idx = trial.suggest_categorical('hidden_layers_idx', hidden_layers_choices)
        params['hidden_layers'] = self.architecture_params['hidden_layers_config'][hidden_layers_idx]
        
        # Activation function
        # Use static list to avoid dynamic value space issues
        activation_choices = ['tanh', 'relu', 'leaky_relu', 'elu', 'selu', 'gelu']
        params['activation'] = trial.suggest_categorical('activation', activation_choices)
        
        # Dropout rate
        params['dropout_rate'] = trial.suggest_float('dropout_rate', 
                                                   *self.architecture_params['dropout_rate'])
        
        # Initialization method
        # Use static list to avoid dynamic value space issues
        init_method_choices = ['xavier_normal', 'xavier_uniform', 'kaiming_normal', 'kaiming_uniform']
        params['init_method'] = trial.suggest_categorical('init_method', init_method_choices)
        
        return params
    
    def _suggest_training_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest training hyperparameters."""
        params = {}
        
        # Learning rate - add to search space
        params['learning_rate'] = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        
        # Batch size
        # Use static list to avoid dynamic value space issues
        batch_size_choices = [128, 256, 512]
        params['batch_size'] = trial.suggest_categorical('batch_size', batch_size_choices)
        
        # Early stopping patience
        # Use static list to avoid dynamic value space issues
        early_patience_choices = [100, 200, 300]
        params['early_patience'] = trial.suggest_categorical('early_patience', early_patience_choices)
        
        # Learning rate scheduler patience
        # Use static list to avoid dynamic value space issues
        lr_patience_choices = [25, 50, 75]
        params['lr_patience'] = trial.suggest_categorical('lr_patience', lr_patience_choices)
        
        return params
    
    def _suggest_method_params(self, trial: optuna.Trial, method_name: str) -> Dict[str, Any]:
        """Suggest method-specific parameters."""
        method_config = self.methods[method_name]
        params = {}
        
        for param_name, (min_val, max_val) in method_config['method_params'].items():
            # Handle different parameter types
            if isinstance(min_val, bool) and isinstance(max_val, bool):
                params[param_name] = trial.suggest_categorical(param_name, [min_val, max_val])
            else:
                # Use log scale for weight parameters to cover several orders of magnitude
                use_log = param_name.startswith('w_') and min_val > 0
                params[param_name] = trial.suggest_float(param_name, min_val, max_val, log=use_log)
        
        return params
    
    def _build_command(self, method_name: str, architecture_params: Dict[str, Any], 
                      training_params: Dict[str, Any], method_params: Dict[str, Any], 
                      output_dir: str) -> List[str]:
        """Build the command to run a training script."""
        method_config = self.methods[method_name]
        script_path = method_config['script']
        
        # Base command - use relative path from project root
        cmd = ['python', f'training_scripts/{script_path}']
        
        # Add fixed parameters
        for param, value in self.fixed_params.items():
            cmd.extend([param, str(value)])
        
        # Add architecture parameters
        cmd.extend(['--hidden-layers'] + [str(x) for x in architecture_params['hidden_layers']])
        cmd.extend(['--activation', architecture_params['activation']])
        cmd.extend(['--dropout-rate', str(architecture_params['dropout_rate'])])
        cmd.extend(['--init-method', architecture_params['init_method']])
        
        # Add training parameters
        cmd.extend(['--batch-size', str(training_params['batch_size'])])
        cmd.extend(['--learning-rate', str(training_params['learning_rate'])])
        cmd.extend(['--early-patience', str(training_params['early_patience'])])
        cmd.extend(['--lr-patience', str(training_params['lr_patience'])])
        
        # Add method-specific parameters
        # Map parameter names to argument names (convert underscores to hyphens)
        for param_name, value in method_params.items():
            arg_name = param_name.replace('_', '-')  # Convert sigma_init to sigma-init
            if isinstance(value, bool):
                # Boolean parameters are flags - only add if True
                if value:
                    cmd.extend([f'--{arg_name}'])
            else:
                # Non-boolean parameters need both flag and value
                cmd.extend([f'--{arg_name}', str(value)])
        
        # Add output directory
        cmd.extend(['--output-dir', output_dir])
        
        return cmd
    
    def _run_training(self, method_name: str, architecture_params: Dict[str, Any], 
                     training_params: Dict[str, Any], method_params: Dict[str, Any], 
                     output_dir: str, trial: int) -> Dict[str, Any]:
        """Run a single training trial."""
        start_time = time.time()
        
        # Build command
        cmd = self._build_command(method_name, architecture_params, training_params, method_params, output_dir)
        
        # Create log file
        log_file = Path(output_dir) / "logs" / f"trial_{trial:03d}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info(f"Starting trial {trial} for {method_name}")
            logger.info(f"Architecture: {architecture_params}")
            logger.info(f"Training: {training_params}")
            logger.info(f"Method: {method_params}")
            
            # Run the training script from project root directory
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            with open(log_file, 'w') as f:
                result = subprocess.run(
                    cmd, 
                    stdout=f, 
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=project_root,  # Run from project root so Data/ directory is found
                    timeout=7200  # 2 hours timeout for longer training
                )
            
            # Check if training was successful
            success = result.returncode == 0
            
            # Log training result
            if success:
                logger.info(f"Training script for {method_name} trial {trial} completed successfully (return code: {result.returncode})")
                # Add small delay to ensure files are fully written
                time.sleep(0.5)
            else:
                logger.error(f"Training script for {method_name} trial {trial} failed with return code: {result.returncode}")
                # Try to read the log file to get more details
                try:
                    with open(log_file, 'r') as f:
                        log_content = f.read()
                        # Get last few lines for error context
                        last_lines = log_content.strip().split('\n')[-10:]
                        logger.error(f"Last 10 lines of training log:")
                        for line in last_lines:
                            logger.error(f"  {line}")
                except Exception as e:
                    logger.error(f"Could not read log file: {e}")
            
            # Extract metrics if successful
            metrics = {}
            if success:
                try:
                    metrics = self._extract_metrics(output_dir, method_name, trial)
                    logger.info(f"Successfully extracted metrics for {method_name} trial {trial}: {list(metrics.keys())}")
                except Exception as e:
                    logger.warning(f"Failed to extract metrics for {method_name} trial {trial}: {e}")
                    success = False
            
            end_time = time.time()
            duration = end_time - start_time
            
            return {
                'method': method_name,
                'trial': trial,
                'architecture_params': architecture_params,
                'training_params': training_params,
                'method_params': method_params,
                'success': success,
                'duration': duration,
                'metrics': metrics,
                'log_file': str(log_file),
                'output_dir': output_dir
            }
            
        except subprocess.TimeoutExpired:
            logger.error(f"Trial {trial} for {method_name} timed out")
            return {
                'method': method_name,
                'trial': trial,
                'architecture_params': architecture_params,
                'training_params': training_params,
                'method_params': method_params,
                'success': False,
                'duration': 7200,
                'metrics': {},
                'log_file': str(log_file),
                'output_dir': output_dir,
                'error': 'timeout'
            }
        except Exception as e:
            logger.error(f"Trial {trial} for {method_name} failed: {e}")
            return {
                'method': method_name,
                'trial': trial,
                'architecture_params': architecture_params,
                'training_params': training_params,
                'method_params': method_params,
                'success': False,
                'duration': time.time() - start_time,
                'metrics': {},
                'log_file': str(log_file),
                'output_dir': output_dir,
                'error': str(e)
            }
    
    def _extract_metrics(self, output_dir: str, method_name: str, trial: int) -> Dict[str, float]:
        """Extract metrics from training results."""
        output_path = Path(output_dir)
        
        # Load history file
        history_file = output_path / f"{method_name}_history.npz"
        
        # Debug: Check if file exists and list directory contents
        logger.debug(f"Looking for history file: {history_file}")
        logger.debug(f"Output directory exists: {output_path.exists()}")
        if output_path.exists():
            logger.debug(f"Directory contents: {list(output_path.iterdir())}")
        
        # Add retry mechanism for race condition
        max_retries = 5
        retry_delay = 1.0  # seconds
        
        for attempt in range(max_retries):
            if history_file.exists():
                break
            else:
                if attempt < max_retries - 1:
                    logger.debug(f"History file not found, retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                else:
                    raise FileNotFoundError(f"History file not found after {max_retries} attempts: {history_file}")
        
        # Load parameters file
        params_file = output_path / f"{method_name}_parameters.npz"
        
        metrics = {}
        
        # Load history data
        history_data = np.load(history_file)
        logger.debug(f"Available history keys for {method_name} trial {trial}: {list(history_data.keys())}")
        
        # Extract final validation metrics
        if 'val_total' in history_data and len(history_data['val_total']) > 0:
            final_val_loss = float(history_data['val_total'][-1])
            # Check if the loss is NaN or infinity
            if np.isnan(final_val_loss) or np.isinf(final_val_loss):
                logger.warning(f"Trial {trial} for {method_name} resulted in NaN/Inf loss: {final_val_loss}")
                # Return a high loss value to mark as unsuccessful
                return {'final_val_total': 1e6}
            else:
                metrics['final_val_total'] = final_val_loss
                logger.debug(f"Extracted final_val_total: {metrics['final_val_total']:.6f}")
        else:
            logger.warning(f"val_total not found or empty in history for {method_name} trial {trial}")
            # Return a high loss value to mark as unsuccessful
            return {'final_val_total': 1e6}
            
        if 'data_val' in history_data:
            metrics['final_data_val'] = float(history_data['data_val'][-1])
        if 'phys_val' in history_data:
            metrics['final_phys_val'] = float(history_data['phys_val'][-1])
        
        # Extract raw residuals if available
        raw_metrics = [k for k in history_data.keys() if k.startswith('raw_') and k.endswith('_val')]
        for metric in raw_metrics:
            if metric in history_data:
                metrics[f'final_{metric}'] = float(history_data[metric][-1])
        
        # Load final parameters if available (with retry mechanism)
        for attempt in range(max_retries):
            if params_file.exists():
                params_data = np.load(params_file)
                param_metrics = [k for k in params_data.keys() if k.startswith('param_')]
                for param in param_metrics:
                    if param in params_data:
                        metrics[f'final_{param}'] = float(params_data[param][-1])
                break
            else:
                if attempt < max_retries - 1:
                    logger.debug(f"Parameters file not found, retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                else:
                    logger.debug(f"Parameters file not found for {method_name} trial {trial}: {params_file}")
        
        logger.debug(f"Extracted metrics for {method_name} trial {trial}: {list(metrics.keys())}")
        return metrics
    
    def _objective_function(self, method_name: str, starting_point: int, trial: optuna.Trial) -> float:
        """Objective function for Optuna optimization."""
        # Suggest all parameter types
        architecture_params = self._suggest_architecture_params(trial)
        training_params = self._suggest_training_params(trial)
        method_params = self._suggest_method_params(trial, method_name)
        
        # Create trial-specific output directory
        trial_output_dir = (self.output_dir / method_name / f"starting_point_{starting_point}" / 
                           "results" / f"trial_{trial.number:03d}")
        trial_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run training
        result = self._run_training(method_name, architecture_params, training_params, 
                                  method_params, str(trial_output_dir), trial.number)
        
        # Store result in trial user attributes
        trial.set_user_attr('result', result)
        
        if result['success'] and 'final_val_total' in result['metrics']:
            # Return validation loss as objective (to minimize)
            val_loss = result['metrics']['final_val_total']
            logger.info(f"Trial {trial.number} for {method_name} SUCCESS: validation_loss={val_loss:.6f}")
            return val_loss
        else:
            # Log detailed failure information
            failure_reason = "Unknown failure"
            if not result['success']:
                if 'error' in result:
                    failure_reason = f"Training error: {result['error']}"
                else:
                    failure_reason = "Training script returned non-zero exit code"
            elif 'final_val_total' not in result['metrics']:
                failure_reason = "Missing final_val_total metric in results"
            
            logger.warning(f"Trial {trial.number} for {method_name} FAILED: {failure_reason}")
            logger.warning(f"  Architecture: {architecture_params}")
            logger.warning(f"  Training: {training_params}")
            logger.warning(f"  Method: {method_params}")
            logger.warning(f"  Duration: {result['duration']:.1f}s")
            logger.warning(f"  Log file: {result['log_file']}")
            
            # Return a high value for failed trials
            return 1e6
    
    def optimize_method(self, method_name: str, starting_point: int) -> optuna.Study:
        """Run Bayesian optimization for a specific method."""
        logger.info(f"Starting Bayesian optimization for {method_name} (starting point {starting_point})")
        
        # Create study
        study_name = f"{method_name}_starting_point_{starting_point}"
        study_dir = self.output_dir / method_name / f"starting_point_{starting_point}" / "optuna_studies"
        
        # Create sampler with different seeds for different starting points
        sampler = TPESampler(seed=starting_point * 42)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        
        # Handle force restart
        if self.force_restart:
            # Delete existing study database if it exists
            study_db_path = study_dir / "study.db"
            if study_db_path.exists():
                logger.info(f"Deleting existing study database: {study_db_path}")
                study_db_path.unlink()
        
        study = optuna.create_study(
            study_name=study_name,
            storage=f"sqlite:///{study_dir}/study.db",
            sampler=sampler,
            pruner=pruner,
            direction="minimize",
            load_if_exists=not self.force_restart  # Don't load if force restart
        )
        
        # Run optimization
        study.optimize(
            lambda trial: self._objective_function(method_name, starting_point, trial),
            n_trials=self.n_trials,
            n_jobs=1  # Bayesian optimization must be sequential within each study
        )
        
        logger.info(f"Optimization completed for {method_name} (starting point {starting_point})")
        logger.info(f"Best value: {study.best_value}")
        logger.info(f"Best parameters: {study.best_params}")
        
        return study
    
    def run_optimization(self):
        """Run Bayesian optimization for all methods."""
        logger.info(f"Starting comprehensive Bayesian optimization with {self.n_trials} trials per method")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Optimization metric: Final validation total loss (minimization)")
        logger.info(f"Parallel jobs: {self.n_jobs} (running {self.n_jobs} Bayesian optimizations in parallel)")
        
        all_results = []
        studies = {}
        
        # Create all optimization tasks
        optimization_tasks = []
        # Filter by selected methods if provided
        selected_methods = getattr(self, 'selected_methods', None)
        for method_name in self.methods.keys():
            if selected_methods and method_name not in selected_methods:
                continue
            for starting_point in range(2):
                optimization_tasks.append((method_name, starting_point + 1))
        
        logger.info(f"Total optimization tasks: {len(optimization_tasks)} (8 methods × 2 starting points)")
        
        # Run optimizations in parallel
        if self.n_jobs > 1:
            logger.info(f"Running {len(optimization_tasks)} Bayesian optimizations with {self.n_jobs} parallel jobs")
            
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                # Submit all tasks
                future_to_task = {
                    executor.submit(self._run_single_optimization, method_name, starting_point): (method_name, starting_point)
                    for method_name, starting_point in optimization_tasks
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_task):
                    method_name, starting_point = future_to_task[future]
                    try:
                        study, method_results = future.result()
                        
                        # Store results
                        if method_name not in studies:
                            studies[method_name] = {}
                        studies[method_name][f"starting_point_{starting_point}"] = study
                        
                        # Add results to global list
                        for result in method_results:
                            all_results.append(result)
                        
                        logger.info(f"✓ Completed: {method_name} (starting point {starting_point})")
                        
                    except Exception as e:
                        logger.error(f"✗ Failed: {method_name} (starting point {starting_point}): {e}")
        else:
            # Sequential execution (for debugging)
            logger.info("Running optimizations sequentially (n_jobs=1)")
            for method_name, starting_point in optimization_tasks:
                study, method_results = self._run_single_optimization(method_name, starting_point)
                
                if method_name not in studies:
                    studies[method_name] = {}
                studies[method_name][f"starting_point_{starting_point}"] = study
                
                for result in method_results:
                    all_results.append(result)
                
                logger.info(f"✓ Completed: {method_name} (starting point {starting_point})")
        
        # Save all results
        self._save_all_results(all_results)
        
        # Generate summary report
        self._generate_summary_report(all_results, studies)
        
        # Generate optimization plots
        self._generate_optimization_plots(studies)
        
        logger.info(f"\n{'='*60}")
        logger.info("Comprehensive Bayesian optimization completed!")
        logger.info(f"Results saved in: {self.output_dir}")
        logger.info(f"{'='*60}")
    
    def _run_single_optimization(self, method_name: str, starting_point: int) -> Tuple[optuna.Study, List[Dict[str, Any]]]:
        """Run a single Bayesian optimization (method + starting point)."""
        logger.info(f"Starting optimization for {method_name} (starting point {starting_point})")
        
        # Run optimization
        study = self.optimize_method(method_name, starting_point)
        
        # Extract results from study
        method_results = []
        for trial in study.trials:
            if hasattr(trial, 'user_attrs') and 'result' in trial.user_attrs:
                result = trial.user_attrs['result']
                method_results.append(result)
        
        return study, method_results
    
    def _save_method_results(self, method_name: str, results: List[Dict[str, Any]]):
        """Save results for a specific method."""
        method_dir = self.output_dir / method_name
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Save as CSV
        csv_file = method_dir / f"{method_name}_results.csv"
        df.to_csv(csv_file, index=False)
        
        # Save as JSON for easier processing
        json_file = method_dir / f"{method_name}_results.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Saved {len(results)} results for {method_name}")
    
    def _save_all_results(self, results: List[Dict[str, Any]]):
        """Save all results combined."""
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Save as CSV
        csv_file = self.output_dir / "global_results" / "all_results.csv"
        df.to_csv(csv_file, index=False)
        
        # Save as JSON
        json_file = self.output_dir / "global_results" / "all_results.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Saved {len(results)} total results")
    
    def _generate_summary_report(self, results: List[Dict[str, Any]], studies: Dict):
        """Generate a summary report of the optimization results."""
        df = pd.DataFrame(results)
        
        # Filter successful and failed runs
        successful = df[df['success'] == True]
        failed = df[df['success'] == False]
        
        # Generate summary statistics
        summary = {
            'total_trials': len(df),
            'successful_trials': len(successful),
            'failed_trials': len(failed),
            'success_rate': len(successful) / len(df) if len(df) > 0 else 0,
            'methods': {},
            'best_results': {},
            'optimization_metric': 'final_val_total (minimization)'
        }
        
        # Log failure summary
        if len(failed) > 0:
            logger.warning(f"\n{'='*60}")
            logger.warning("FAILED TRIALS SUMMARY")
            logger.warning(f"{'='*60}")
            logger.warning(f"Total failed trials: {len(failed)}")
            
            # Group failures by method
            for method_name in self.methods.keys():
                method_failed = failed[failed['method'] == method_name]
                if len(method_failed) > 0:
                    logger.warning(f"\n{method_name}: {len(method_failed)} failed trials")
                    logger.warning(f"Check individual trial logs for detailed error information.")
            
            logger.warning(f"{'='*60}")
        
        # Per-method statistics and best results
        for method_name in self.methods.keys():
            method_df = df[df['method'] == method_name]
            method_successful = method_df[method_df['success'] == True]
            
            summary['methods'][method_name] = {
                'total_trials': len(method_df),
                'successful_trials': len(method_successful),
                'success_rate': len(method_successful) / len(method_df) if len(method_df) > 0 else 0,
                'avg_duration': method_successful['duration'].mean() if len(method_successful) > 0 else 0
            }
            
            # Best results from studies
            if method_name in studies:
                best_value = float('inf')
                best_params = {}
                best_starting_point = None
                
                for starting_point, study in studies[method_name].items():
                    if study.best_value < best_value:
                        best_value = study.best_value
                        best_params = study.best_params
                        best_starting_point = starting_point
                
                summary['best_results'][method_name] = {
                    'best_value': best_value,
                    'best_params': best_params,
                    'best_starting_point': best_starting_point
                }
        
        # Save summary
        summary_file = self.output_dir / "global_results" / "optimization_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("OPTIMIZATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Optimization metric: {summary['optimization_metric']}")
        logger.info(f"Total trials: {summary['total_trials']}")
        logger.info(f"Successful trials: {summary['successful_trials']}")
        logger.info(f"Overall success rate: {summary['success_rate']:.2%}")
        
        for method_name, stats in summary['methods'].items():
            logger.info(f"\n{method_name}:")
            logger.info(f"  Trials: {stats['total_trials']}")
            logger.info(f"  Successful: {stats['successful_trials']}")
            logger.info(f"  Success rate: {stats['success_rate']:.2%}")
            logger.info(f"  Avg duration: {stats['avg_duration']:.1f}s")
            
            if method_name in summary['best_results']:
                best = summary['best_results'][method_name]
                logger.info(f"  Best validation loss: {best['best_value']:.6f}")
                logger.info(f"  Best starting point: {best['best_starting_point']}")
    
    def _generate_optimization_plots(self, studies: Dict):
        """Generate optimization plots for each method."""
        try:
            import matplotlib.pyplot as plt
            
            for method_name, method_studies in studies.items():
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                fig.suptitle(f'Bayesian Optimization Results: {method_name}', fontsize=16)
                
                # Plot 1: Optimization history
                ax1 = axes[0, 0]
                for starting_point, study in method_studies.items():
                    values = [trial.value for trial in study.trials if trial.value is not None]
                    ax1.plot(values, label=starting_point, alpha=0.7)
                ax1.set_title('Optimization History')
                ax1.set_xlabel('Trial')
                ax1.set_ylabel('Validation Loss')
                ax1.set_yscale('log')
                ax1.legend()
                ax1.grid(True)
                
                # Plot 2: Best value evolution
                ax2 = axes[0, 1]
                for starting_point, study in method_studies.items():
                    best_values = []
                    current_best = float('inf')
                    for trial in study.trials:
                        if trial.value is not None and trial.value < current_best:
                            current_best = trial.value
                        best_values.append(current_best)
                    ax2.plot(best_values, label=starting_point, alpha=0.7)
                ax2.set_title('Best Value Evolution')
                ax2.set_xlabel('Trial')
                ax2.set_ylabel('Best Validation Loss')
                ax2.set_yscale('log')
                ax2.legend()
                ax2.grid(True)
                
                # Plot 3: Parameter importance (if available)
                ax3 = axes[1, 0]
                try:
                    # Use the first study for parameter importance
                    first_study = list(method_studies.values())[0]
                    importance = optuna.importance.get_param_importances(first_study)
                    if importance:
                        params = list(importance.keys())
                        values = list(importance.values())
                        ax3.barh(params, values)
                        ax3.set_title('Parameter Importance')
                        ax3.set_xlabel('Importance')
                except Exception as e:
                    ax3.text(0.5, 0.5, f'Parameter importance\nnot available\n({e})', 
                            ha='center', va='center', transform=ax3.transAxes)
                    ax3.set_title('Parameter Importance')
                
                # Plot 4: Parameter distributions
                ax4 = axes[1, 1]
                try:
                    # Use the first study for parameter distributions
                    first_study = list(method_studies.values())[0]
                    optuna.visualization.matplotlib.plot_param_importances(first_study, ax=ax4)
                    ax4.set_title('Parameter Distributions')
                except Exception as e:
                    ax4.text(0.5, 0.5, f'Parameter distributions\nnot available\n({e})', 
                            ha='center', va='center', transform=ax4.transAxes)
                    ax4.set_title('Parameter Distributions')
                
                plt.tight_layout()
                
                # Save plot
                plot_file = self.output_dir / method_name / f"{method_name}_optimization_plots.png"
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Generated optimization plots for {method_name}")
                
        except ImportError:
            logger.warning("Matplotlib not available, skipping optimization plots")
        except Exception as e:
            logger.warning(f"Failed to generate optimization plots: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Comprehensive Bayesian Hyperparameter Optimization for PINN Methods')
    parser.add_argument('--n-trials', type=int, default=20, 
                       help='Number of trials per method (default: 100)')
    parser.add_argument('--n-jobs', type=int, default=2, 
                       help='Number of parallel Bayesian optimizations (default: 2, max 16 for full parallelization)')
    parser.add_argument('--output-dir', type=str, default='bayesian_results',
                       help='Output directory (default: bayesian_results)')
    parser.add_argument('--force-restart', action='store_true', default=True,
                       help='Force restart: delete existing studies and start fresh')
    parser.add_argument('--test-mode', action='store_true',
                       help='Run in test mode with reduced trials and epochs for quick testing')
    parser.add_argument('--methods', nargs='*', default=None,
                       help='Subset of methods to optimize (e.g., constant_weight)')
    
    args = parser.parse_args()
    
    # Reduce trials in test mode
    if args.test_mode:
        args.n_trials = min(args.n_trials, 5)  # Max 5 trials in test mode
        logger.info("Running in test mode with reduced trials and epochs")
    
    # Create and run optimization
    optimizer = BayesianHyperparameterOptimization(
        output_dir=args.output_dir,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        test_mode=args.test_mode,
        force_restart=args.force_restart
    )
    # Attach selected methods filter if provided
    optimizer.selected_methods = set(args.methods) if args.methods else None
    
    optimizer.run_optimization()


if __name__ == "__main__":
    main() 