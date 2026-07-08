#!/usr/bin/env python3
"""
Data-Driven Regularization Hyperparameter Optimization

This script performs Bayesian optimization to find the best regularization approach
and parameters for the MLP-Reg-4ch baseline. It optimizes:
- Regularization type (L2, L1, Tikhonov, Jacobian)
- Regularization strength (lambda)
- Other hyperparameters if needed

The best configuration is selected based on validation MAE (same metric as Stage 1 proxy).
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

# Optuna for Bayesian optimization
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataDrivenRegularizationOptimization:
    """
    Bayesian hyperparameter optimization for data-driven regularization baselines.
    """

    def __init__(self, output_dir: str = "data_driven_reg_optimization",
                 n_trials: int = 50, n_jobs: int = 2, test_mode: bool = False):

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.test_mode = test_mode

        # Define search space for regularization
        self.regularization_space = {
            'regularization': ['tikhonov'],  # Removed jacobian due to gradient issues
            'lambda_reg': (1e-6, 1e-2),  # Log scale regularization strength
        }

        # Results storage
        self.results = []

    def objective(self, trial):
        """Objective function for Optuna optimization."""

        # Sample hyperparameters
        regularization = trial.suggest_categorical('regularization', self.regularization_space['regularization'])
        lambda_reg = trial.suggest_float('lambda_reg', *self.regularization_space['lambda_reg'], log=True)

        # Create unique trial identifier
        trial_id = f"trial_{trial.number:03d}"

        logger.info(f"Starting trial {trial_id}: {regularization} with λ={lambda_reg:.2e}")

        # Create output directory for this trial
        trial_output_dir = self.output_dir / trial_id
        trial_output_dir.mkdir(exist_ok=True)

        try:
            # Run data-driven training with regularization
            cmd = [
                sys.executable, 'training_scripts/data_driven_training.py',
                '--output-dir', str(trial_output_dir),
                '--regularization', regularization,
                '--lambda-reg', str(lambda_reg),
                '--device', 'cpu',  # Use CPU for stability in parallel runs
            ]

            if self.test_mode:
                cmd.extend(['--epochs', '100', '--max-samples', '500'])

            # Run the training
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout

            if result.returncode != 0:
                logger.error(f"Trial {trial_id} failed: {result.stderr}")
                return float('inf')  # Return worst possible score

            # Parse results from the training output
            val_mae = self._extract_validation_mae(trial_output_dir)

            if val_mae is None:
                logger.error(f"Could not extract validation MAE for trial {trial_id}")
                return float('inf')

            # Store results
            trial_result = {
                'trial_id': trial_id,
                'regularization': regularization,
                'lambda_reg': lambda_reg,
                'val_mae': val_mae,
                'timestamp': datetime.now().isoformat()
            }

            self.results.append(trial_result)

            # Save intermediate results
            self._save_results()

            logger.info(f"Trial {trial_id} completed: MAE = {val_mae:.6f}")

            return val_mae

        except subprocess.TimeoutExpired:
            logger.error(f"Trial {trial_id} timed out")
            return float('inf')
        except Exception as e:
            logger.error(f"Trial {trial_id} failed with exception: {e}")
            return float('inf')

    def _extract_validation_mae(self, trial_output_dir: Path) -> Optional[float]:
        """Extract validation MAE from training results."""

        # Look for the model file (check both possible names for compatibility)
        model_file = None
        for possible_name in ['best_model.pth', 'data_driven_model.pth']:
            candidate = trial_output_dir / possible_name
            if candidate.exists():
                model_file = candidate
                break

        if model_file is None or not model_file.exists():
            logger.error(f"Model file not found in {trial_output_dir}")
            return None

        try:
            # Import required modules
            import torch
            from models.basicPINNv8 import ConfigurableMLP
            from training_scripts.common_utils import unified_data_preparation, ensure_double_precision

            # Load the trained model
            model = ConfigurableMLP(
                input_dim=10,  # Standard input dimension
                hidden_layers=[128, 128],  # Best PINN architecture
                output_dim=4,  # 4 acceleration outputs
                activation='elu',
                dropout_rate=0.24900923127192412,
                init_method='xavier_uniform'
            )

            model.load_state_dict(torch.load(model_file, map_location='cpu'))
            model.eval()
            model.double()  # Ensure model is in double precision

            # Load validation data
            data = unified_data_preparation(
                data_path='Data',
                max_samples=10000,  # Use reasonable sample size for validation
                synthetic=False,
                simulation_id=None,
                batch_size=1000
            )

            data = ensure_double_precision(data)

            # Set up device (CPU for consistency)
            device = torch.device('cpu')
            model.to(device)

            # Compute MAE on validation set
            model.eval()
            total_mae = 0.0
            total_samples = 0

            with torch.no_grad():
                for xb, yb in data['val']:
                    xb, yb = xb.to(device), yb.to(device)

                    # Normalize inputs
                    X_norm = (xb - data['Xmin'].to(device)) / (data['Xmax'].to(device) - data['Xmin'].to(device) + 1e-12)

                    # Get predictions
                    y_pred_norm = model(X_norm)

                    # Denormalize predictions
                    y_pred = y_pred_norm * (data['ymax'].to(device) - data['ymin'].to(device)) + data['ymin'].to(device)

                    # Compute MAE
                    mae = torch.mean(torch.abs(y_pred - yb))
                    total_mae += mae.item() * xb.size(0)  # Weight by batch size
                    total_samples += xb.size(0)

            avg_mae = total_mae / total_samples
            logger.info(f"Computed validation MAE: {avg_mae:.6f}")
            return avg_mae

        except Exception as e:
            logger.error(f"Error extracting MAE from {trial_output_dir}: {e}")
            return None

    def _save_results(self):
        """Save current results to JSON."""
        results_file = self.output_dir / 'optimization_results.json'

        # Convert to serializable format
        serializable_results = []
        for result in self.results:
            serializable_result = result.copy()
            serializable_result['lambda_reg'] = float(result['lambda_reg'])
            serializable_result['val_mae'] = float(result['val_mae'])
            serializable_results.append(serializable_result)

        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)

    def find_best_configuration(self) -> Dict[str, Any]:
        """Find the best configuration from completed trials."""

        if not self.results:
            raise ValueError("No trial results available")

        # Find trial with lowest validation MAE
        best_result = min(self.results, key=lambda x: x['val_mae'])

        return {
            'best_regularization': best_result['regularization'],
            'best_lambda_reg': best_result['lambda_reg'],
            'best_val_mae': best_result['val_mae'],
            'best_trial_id': best_result['trial_id'],
            'all_results': self.results
        }

    def run_optimization(self):
        """Run the Bayesian optimization."""

        logger.info(f"Starting data-driven regularization optimization with {self.n_trials} trials")

        # Create Optuna study
        study = optuna.create_study(
            sampler=TPESampler(),
            pruner=MedianPruner(),
            direction='minimize'  # Minimize validation MAE
        )

        # Run optimization
        study.optimize(self.objective, n_trials=self.n_trials)

        # Get best configuration
        best_config = self.find_best_configuration()

        # Save final results
        final_results = {
            'optimization_summary': {
                'total_trials': len(self.results),
                'best_configuration': {
                    'regularization': best_config['best_regularization'],
                    'lambda_reg': best_config['best_lambda_reg'],
                    'validation_mae': best_config['best_val_mae']
                },
                'optimization_completed_at': datetime.now().isoformat()
            },
            'all_trial_results': self.results
        }

        with open(self.output_dir / 'final_optimization_results.json', 'w') as f:
            json.dump(final_results, f, indent=2)

        logger.info("Optimization completed!")
        logger.info(f"Best configuration: {best_config['best_regularization']} "
                   f"with λ={best_config['best_lambda_reg']:.2e} "
                   f"(MAE={best_config['best_val_mae']:.6f})")

        return best_config


def main():
    parser = argparse.ArgumentParser(description='Data-Driven Regularization Optimization')
    parser.add_argument('--output-dir', type=str, default='data_driven_reg_optimization',
                       help='Output directory for optimization results')
    parser.add_argument('--n-trials', type=int, default=50,
                       help='Number of optimization trials')
    parser.add_argument('--n-jobs', type=int, default=2,
                       help='Number of parallel jobs')
    parser.add_argument('--test-mode', action='store_true',
                       help='Run in test mode with fewer epochs/samples')

    args = parser.parse_args()

    optimizer = DataDrivenRegularizationOptimization(
        output_dir=args.output_dir,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        test_mode=args.test_mode
    )

    best_config = optimizer.run_optimization()

    print("\n" + "="*50)
    print("OPTIMIZATION COMPLETED")
    print("="*50)
    print(f"Best Regularization: {best_config['best_regularization']}")
    print(".2e")
    print(".6f")
    print(f"Results saved to: {args.output_dir}")
    print("="*50)


if __name__ == '__main__':
    main()
