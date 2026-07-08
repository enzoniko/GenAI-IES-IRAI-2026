#!/usr/bin/env python3
"""
Evaluation script for trained Hybrid RNN-PINN model.

Loads the trained model and evaluates its performance on the normal dataset
using MAE, MSE, and DTW metrics.
"""

import sys
import os
import torch
import numpy as np
from tqdm import tqdm
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from hybrid_analysis.data_utils import prepare_sequences
from models.hybridRNNPINNv2 import RNNPINNCell
from Data.LoadData import data_paths, get_omegas


def load_and_preprocess_data():
    """
    Load and preprocess data the same way as the training script.

    Returns:
    - X: Processed input data
    - y: Target data
    - omegas: Rotation frequencies
    """
    print("Loading and preprocessing data...")

    # Load data
    X = torch.load('Data/X_normal.pth')
    y = torch.load('Data/Y_normal.pth')
    omegas = get_omegas(data_paths['normal'])

    print(f"Raw data shapes: X={X.shape}, y={y.shape}")
    print(f"Number of frequencies: {len(omegas)}")

    # Add time feature (same as training)
    t = torch.arange(0, 2e-5 * X.size(1), 2e-5).view(1, -1, 1).expand(X.size(0), -1, -1)[:, :-1, :]
    X = torch.cat((X, t), dim=2)

    print(f"After adding time feature: X={X.shape}")

    return X, y, omegas


def prepare_evaluation_sequences(X, y, omegas, num_timestamps=100, max_sequences=1000):
    """
    Prepare sequences for evaluation (similar to training but with limits).

    Parameters:
    - X, y, omegas: Preprocessed data
    - num_timestamps: Sequence length (same as training)
    - max_sequences: Maximum number of sequences to evaluate (for speed)

    Returns:
    - sequences: List of (input_seq, target_seq, speed) tuples
    """
    print("Preparing evaluation sequences...")

    sequences = prepare_sequences(
        X, y, omegas,
        separate_by_speed=False,
        num_timestamps=num_timestamps,
        num_windows=3,  # Same as training
        use_all_data=True
    )

    # Limit number of sequences if specified
    if max_sequences is not None and len(sequences) > max_sequences:
        print(f"Limiting evaluation to {max_sequences} sequences out of {len(sequences)}")
        # Take evenly spaced samples
        indices = np.linspace(0, len(sequences)-1, max_sequences, dtype=int)
        sequences = [sequences[i] for i in indices]

    print(f"Using {len(sequences)} sequences for evaluation")
    return sequences


def load_trained_model(model_path):
    """
    Load the trained model from the specified path.

    Parameters:
    - model_path: Path to the saved model (.pth file)

    Returns:
    - model: Loaded and ready-to-use model
    """
    print(f"Loading trained model from {model_path}")

    # Initialize model
    model = RNNPINNCell()

    # Load state dict
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    print("Model loaded successfully!")
    return model


def evaluate_model(model, sequences, device='cpu'):
    """
    Evaluate the model on the prepared sequences.

    Parameters:
    - model: Trained model
    - sequences: List of (input_seq, target_seq, speed) tuples
    - device: Device to run evaluation on

    Returns:
    - Dictionary with evaluation metrics
    """
    print("Running model evaluation...")

    model = model.to(device)
    all_predictions = []
    all_targets = []
    dtw_distances = []
    sequence_losses = []

    with torch.no_grad():
        for i, (input_seq, target_seq, speed) in enumerate(tqdm(sequences, desc="Evaluating sequences")):
            # Prepare batch (single sequence)
            input_seq_batch = input_seq.unsqueeze(0).to(device)  # [1, seq_len, features]
            target_seq_batch = target_seq.unsqueeze(0).to(device)  # [1, seq_len, targets]

            # Initialize hidden state
            hidden = model.init_hidden(1).to(device)

            # Store predictions
            predictions = []

            # Process sequence step by step
            for t in range(input_seq_batch.shape[1]):
                x_t = input_seq_batch[:, t, :].to(device)
                pred_t, hidden = model(x_t, hidden)
                predictions.append(pred_t.squeeze(0))

            predictions = torch.stack(predictions)  # [seq_len, output_features]

            # Store results
            all_predictions.append(predictions.cpu())
            all_targets.append(target_seq_batch.squeeze(0).cpu())

            # Calculate per-sequence metrics
            seq_mse = torch.mean((predictions - target_seq_batch.squeeze(0))**2).item()
            seq_mae = torch.mean(torch.abs(predictions - target_seq_batch.squeeze(0))).item()
            sequence_losses.append({'mse': seq_mse, 'mae': seq_mae})

            # Calculate DTW for each output variable
            seq_dtw = []
            for var_idx in range(target_seq_batch.shape[2]):
                actual_var = target_seq_batch[0, :, var_idx].cpu().numpy().astype(np.float64)
                pred_var = predictions[:, var_idx].cpu().numpy().astype(np.float64)

                try:
                    distance, _ = fastdtw(actual_var, pred_var)
                    seq_dtw.append(distance)
                except Exception as e:
                    print(f"Warning: DTW computation failed for sequence {i}, variable {var_idx}: {e}")
                    seq_dtw.append(np.nan)

            dtw_distances.append(seq_dtw)

    return {
        'predictions': all_predictions,
        'targets': all_targets,
        'sequence_losses': sequence_losses,
        'dtw_distances': dtw_distances
    }


def compute_overall_metrics(results):
    """
    Compute overall evaluation metrics from the results.

    Parameters:
    - results: Dictionary from evaluate_model

    Returns:
    - Dictionary with overall metrics
    """
    print("Computing overall metrics...")

    all_predictions = torch.cat(results['predictions'], dim=0)
    all_targets = torch.cat(results['targets'], dim=0)

    # Overall MSE and MAE
    mse = torch.mean((all_predictions - all_targets)**2).item()
    mae = torch.mean(torch.abs(all_predictions - all_targets)).item()
    rmse = np.sqrt(mse)

    # Per-variable metrics
    var_names = ['x2_ddot', 'y2_ddot', 'x3_ddot', 'y3_ddot']
    per_variable_metrics = {}

    for var_idx, var_name in enumerate(var_names):
        var_pred = all_predictions[:, var_idx]
        var_target = all_targets[:, var_idx]

        var_mse = torch.mean((var_pred - var_target)**2).item()
        var_mae = torch.mean(torch.abs(var_pred - var_target)).item()
        var_rmse = np.sqrt(var_mse)

        per_variable_metrics[var_name] = {
            'mse': var_mse,
            'mae': var_mae,
            'rmse': var_rmse
        }

    # DTW metrics
    dtw_array = np.array(results['dtw_distances'])
    dtw_mean = np.nanmean(dtw_array)
    dtw_std = np.nanstd(dtw_array)
    dtw_median = np.nanmedian(dtw_array)

    # Per-variable DTW
    per_variable_dtw = {}
    for var_idx, var_name in enumerate(var_names):
        var_dtw = dtw_array[:, var_idx]
        var_dtw = var_dtw[~np.isnan(var_dtw)]  # Remove NaN values

        if len(var_dtw) > 0:
            per_variable_dtw[var_name] = {
                'mean': np.mean(var_dtw),
                'std': np.std(var_dtw),
                'median': np.median(var_dtw)
            }
        else:
            per_variable_dtw[var_name] = {'mean': np.nan, 'std': np.nan, 'median': np.nan}

    # Sequence-level statistics
    sequence_mse = [seq['mse'] for seq in results['sequence_losses']]
    sequence_mae = [seq['mae'] for seq in results['sequence_losses']]

    return {
        'overall': {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'num_sequences': len(results['sequence_losses']),
            'total_samples': len(all_predictions)
        },
        'per_variable': per_variable_metrics,
        'dtw': {
            'overall_mean': dtw_mean,
            'overall_std': dtw_std,
            'overall_median': dtw_median
        },
        'per_variable_dtw': per_variable_dtw,
        'sequence_stats': {
            'mse_mean': np.mean(sequence_mse),
            'mse_std': np.std(sequence_mse),
            'mae_mean': np.mean(sequence_mae),
            'mae_std': np.std(sequence_mae)
        }
    }


def print_metrics_report(metrics):
    """
    Print a comprehensive metrics report.

    Parameters:
    - metrics: Dictionary from compute_overall_metrics
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION REPORT")
    print("="*60)

    print("\n📊 OVERALL PERFORMANCE:")
    print(f"  MSE: {metrics['overall']['mse']:.2f}")
    print(f"  MAE: {metrics['overall']['mae']:.2f}")
    print(f"  RMSE: {metrics['overall']['rmse']:.2f}")
    print(f"  Sequences evaluated: {metrics['overall']['num_sequences']}")
    print(f"  Total samples: {metrics['overall']['total_samples']}")

    print("\n📈 PER-VARIABLE PERFORMANCE:")
    for var_name, var_metrics in metrics['per_variable'].items():
        print(f"  {var_name}:")
        print(f"    MSE: {var_metrics['mse']:.2f}")
        print(f"    MAE: {var_metrics['mae']:.2f}")
        print(f"    RMSE: {var_metrics['rmse']:.2f}")

    print("\n🔄 DTW DISTANCE METRICS:")
    print(f"  Mean: {metrics['dtw']['overall_mean']:.2f}")
    print(f"  Std: {metrics['dtw']['overall_std']:.2f}")
    print(f"  Median: {metrics['dtw']['overall_median']:.2f}")

    print("\n🔄 PER-VARIABLE DTW:")
    for var_name, var_dtw in metrics['per_variable_dtw'].items():
        print(f"  {var_name}:")
        print(f"    Mean: {var_dtw['mean']:.2f}")
        print(f"    Std: {var_dtw['std']:.2f}")
        print(f"    Median: {var_dtw['median']:.2f}")

    print("\n📈 SEQUENCE-LEVEL STATISTICS:")
    print(f"  MSE Mean: {metrics['sequence_stats']['mse_mean']:.2f}")
    print(f"  MSE Std: {metrics['sequence_stats']['mse_std']:.2f}")
    print(f"  MAE Mean: {metrics['sequence_stats']['mae_mean']:.2f}")
    print(f"  MAE Std: {metrics['sequence_stats']['mae_std']:.2f}")

    print("\n" + "="*60)


def main():
    """Main evaluation function."""
    print("Starting Model Evaluation")
    print("="*40)

    # Debug: Check if required files exist
    print("Checking required files...")
    if not os.path.exists('Data/X_normal.pth'):
        print("❌ Data/X_normal.pth not found")
        return
    if not os.path.exists('Data/Y_normal.pth'):
        print("❌ Data/Y_normal.pth not found")
        return
    print("✅ Required data files found")

    # Check model file
    model_dir = "results/hybrid_rnn_1757803450/models"
    model_filename = "hybrid_rnn_model_final.pth"
    model_path = os.path.join(model_dir, model_filename)
    if not os.path.exists(model_path):
        print(f"❌ Model file not found at {model_path}")
        return
    print(f"✅ Model file found at {model_path}")

    # Model path is already checked above

    # Set device (prefer CUDA if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        print("\n--- Step 1: Loading and preprocessing data ---")
        X, y, omegas = load_and_preprocess_data()
        print("✅ Data loaded successfully")

        print("\n--- Step 2: Preparing evaluation sequences ---")
        sequences = prepare_evaluation_sequences(X, y, omegas, max_sequences=None)  # Use all sequences
        print(f"✅ Prepared {len(sequences)} sequences")

        print("\n--- Step 3: Loading trained model ---")
        model = load_trained_model(model_path)
        print("✅ Model loaded successfully")

        print("\n--- Step 4: Running evaluation ---")
        results = evaluate_model(model, sequences, device)
        print("✅ Evaluation completed")

        print("\n--- Step 5: Computing metrics ---")
        metrics = compute_overall_metrics(results)
        print("✅ Metrics computed")

        print("\n--- Step 6: Generating report ---")
        print_metrics_report(metrics)

        print("\n✅ Evaluation completed successfully!")

    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
