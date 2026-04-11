'''
Base Paper: Linking Physical Fidelity to Downstream Performance in Physics-Informed Fault Diagnosis

Phase 0 - Physical Foundation
A native, orchestrated pipeline extracting logic from the previous work.
'''

import os
import sys
import argparse
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Subset

from src.data.clean_mafaulda_processor import CleanMaFaulDaProcessor
from src.models.pinn import ConfigurablePINN
from src.models.relobralo_loss import ReLoBRaLoLoss
import src.configs as cfg

# ── Configuration defaults ────────────────────────────────────────────────────
RAW_DATA_DIR      = cfg.DATA_DIR_RAW
PROCESSED_DATA_DIR = cfg.DATA_DIR_PROCESSED
PINN_MODEL_PATH   = cfg.PINN_MODEL_PATH
EPOCHS            = cfg.PHASE0_TRAIN_SETTINGS['epochs']
BATCH_SIZE        = cfg.PHASE0_TRAIN_SETTINGS['batch_size']
NUM_SAMPLES       = None   # set a small int for a dry run
ROTATION          = cfg.PHASE0_TRAIN_SETTINGS['rotation_hz']
TRAIN_WINDOWS     = cfg.PHASE0_TRAIN_SETTINGS['training_windows']
TEST_WINDOWS      = cfg.PHASE0_TRAIN_SETTINGS['test_windows']
PATIENCE          = cfg.PHASE0_TRAIN_SETTINGS['early_stop_patience']
MIN_DELTA         = cfg.PHASE0_TRAIN_SETTINGS['early_stop_min_delta']


# ── Step 1: Pre-Process & Split ───────────────────────────────────────────────
def step1_prepare_dataset(raw_data_dir, processed_data_dir, rotation=None,
                           training_windows=None, test_windows=None):
    """
    Uses CleanMaFaulDaProcessor to clean raw CSVs into physical tensors and
    split them into a training set and a temporally-disjoint test set.

    Only the 'normal' class is processed here because the PINN oracle is trained
    exclusively on healthy data (the physics equations are derived from a healthy
    rotor baseline).

    Saved files:
      X_normal_<ver>_trainingset.pth  — PINN training + validation windows
      Y_normal_<ver>_trainingset.pth
      X_normal_<ver>_testset.pth      — Held-out test windows (chronological tail)
      Y_normal_<ver>_testset.pth
    """
    print("\n--- Step 1: Dataset Loading, Preparation, and Split ---")
    processor = CleanMaFaulDaProcessor(
        raw_data_dir=raw_data_dir,
        processed_data_dir=processed_data_dir
    )
    processor.run(
        category='normal',
        rotation=rotation,
        training_windows=training_windows,
        test_windows=test_windows
    )
    print("Data preparation and splitting complete.")


# ── Sequential interleaved split helper ──────────────────────────────────────
def _sequential_split(dataset, val_fraction=0.2):
    """
    Deterministic, temporally-aware train/validation split.

    Instead of random_split (which can place neighbouring windows in both sets),
    we take every k-th window for validation where k = round(1 / val_fraction).
    This ensures no two adjacent windows end up in different sets.

    Example for val_fraction=0.2 (k=5):
      val   indices: 0,  5, 10, 15, ...
      train indices: 1, 2, 3, 4,  6, 7, 8, 9,  11, ...
    """
    N = len(dataset)
    k = max(2, round(1.0 / val_fraction))       # distance between val samples
    val_indices   = list(range(0, N, k))
    train_indices = [i for i in range(N) if i % k != 0]
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


# ── Step 2: PINN Training ─────────────────────────────────────────────────────
def step2_train_pinn_oracle(processed_data_dir, output_model_path,
                             epochs=100, batch_size=256, num_samples=None,
                             patience=15, min_delta=1e-4):
    """
    Trains the ConfigurablePINN using the ReLoBRaLo dynamic loss weighting.
    Loads from the trainingset tensors produced in Step 1.
    Uses a sequential interleaved split (80/20) to avoid temporal leakage
    between training and validation windows.
    """
    print("\n--- Step 2: PINN Training & Feature Extraction Setup ---")
    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps'  if torch.backends.mps.is_available() else 'cpu'
    )
    print(f"Using device: {device}")

    # ── Load Training-Set Tensors ─────────────────────────────────────────────
    # These are the windows produced by the processor (NOT the held-out test set).
    x_path = os.path.join(processed_data_dir, 
                          f"X_normal_{cfg.DATASET_VERSION}_trainingset.pth")
    y_path = os.path.join(processed_data_dir, 
                          f"Y_normal_{cfg.DATASET_VERSION}_trainingset.pth")

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(
            f"Training-set tensors not found at:\n  {x_path}\n  {y_path}\n"
            "Run Step 1 first."
        )

    # Load directly to CPU to prevent GPU VRAM spikes during data loading
    X_raw = torch.load(x_path, map_location='cpu')
    Y_raw = torch.load(y_path, map_location='cpu')

    # ── Min-Max Normalization ─────────────────────────────────────────────────
    # Compute bounds BEFORE casting to double to save memory.
    # We compute over the full training tensor (before num_samples capping) so
    # the bounds represent the true physical range of the dataset.
    X_max = X_raw.reshape(-1, X_raw.shape[-1]).max(dim=0)[0]
    X_min = X_raw.reshape(-1, X_raw.shape[-1]).min(dim=0)[0]
    y_max = Y_raw.reshape(-1, Y_raw.shape[-1]).max(dim=0)[0]
    y_min = Y_raw.reshape(-1, Y_raw.shape[-1]).min(dim=0)[0]
    X_max, X_min = X_max.double(), X_min.double()
    y_max, y_min = y_max.double(), y_min.double()

    # Optional subsample for resource-constrained dry runs
    if num_samples is not None:
        X_raw = X_raw[:num_samples]
        Y_raw = Y_raw[:num_samples]

    X = X_raw.double()
    Y = Y_raw.double()

    X_norm = (X - X_min) / (X_max - X_min + 1e-12)
    Y_norm = (Y - y_min) / (y_max - y_min + 1e-12)

    # Normalized tensors for training and validation
    dataset = TensorDataset(X_norm, Y_norm)

    # ── Sequential Train / Validation Split ───────────────────────────────────
    # Every 5th window goes to validation — avoids putting time-adjacent
    # windows in both sets (the problem with random_split on time-series data).
    train_dataset, val_dataset = _sequential_split(dataset, val_fraction=0.2)
    print(f"  Train windows: {len(train_dataset)} | Validation windows: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    # ── Initialize PINN & Optimizer ───────────────────────────────────────────
    pinn_config = cfg.PINN_ARCH_DEFAULT
    model = ConfigurablePINN(
        unmeasured_net_config=pinn_config['unmeasured_net_config'],
        acceleration_net_config=pinn_config['acceleration_net_config'],
        param_init_config=pinn_config['param_init_config'],
        enable_mass_constraints=True
    ).to(device)

    loss_method = ReLoBRaLoLoss(enable_mass_constraints=True)
    optimizer   = optim.Adam(model.parameters(), lr=1e-4)
    scheduler   = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.5, min_lr=1e-6
    )

    best_val_loss = float('inf')
    best_model_weights = None
    epochs_without_improvement = 0

    # ── Training Loop ─────────────────────────────────────────────────────────
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            # Flatten 3D window chunks to 2D for the PINN and Loss function
            # Shape: (Batch, Window, Features) → (Batch*Window, Features)
            xb = xb.reshape(-1, xb.shape[-1])
            yb = yb.reshape(-1, yb.shape[-1])

            # Forward pass — compute the 7 ReLoBRaLo loss components
            loss_components = loss_method(
                model, xb, yb,
                X_max.to(device), X_min.to(device),
                y_max.to(device), y_min.to(device)
            )

            # Dynamically weighted backward step (ReLoBRaLo)
            total_loss = loss_method.step(loss_components, optimizer, model, xb, yb)

            if isinstance(total_loss, torch.Tensor):
                epoch_loss += total_loss.item()

        # ── Validation Pass ───────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb_val, yb_val in val_loader:
                xb_val, yb_val = xb_val.to(device), yb_val.to(device)

                xb_val = xb_val.reshape(-1, xb_val.shape[-1])
                yb_val = yb_val.reshape(-1, yb_val.shape[-1])

                loss_components = loss_method(
                    model, xb_val, yb_val,
                    X_max.to(device), X_min.to(device),
                    y_max.to(device), y_min.to(device)
                )

                # Weighted sum using current ReLoBRaLo coefficients
                weighted_val = sum(
                    loss_method.current_weights[key] * loss_components[i].item()
                    for i, key in enumerate(loss_method.loss_keys)
                    if i < len(loss_components)
                )
                val_loss += weighted_val

        avg_train = epoch_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)

        scheduler.step(avg_val)
        print(f"Epoch {epoch+1:03d}/{epochs} | Train: {avg_train:.4f} | Val: {avg_val:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Track the best model checkpoint and Early Stopping
        if avg_val < (best_val_loss - min_delta):
            best_val_loss = avg_val
            best_model_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"\n[Early Stopping] Triggered after {epoch+1} epochs. No improvement for {patience} epochs.")
            break

    # ── Save Checkpoint ───────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    
    # Save the normalization bounds separately for the rest of the pipeline
    norm_path = os.path.join(os.path.dirname(output_model_path), "normalization_metadata.pth")
    torch.save({
        'X_max': X_max.cpu(),
        'X_min': X_min.cpu(),
        'y_max': y_max.cpu(),
        'y_min': y_min.cpu()
    }, norm_path)
    print(f"Normalization metadata saved to {norm_path}")

    torch.save({
        'model_state_dict': best_model_weights,
        'X_max': X_max.cpu(),
        'X_min': X_min.cpu(),
        'y_max': y_max.cpu(),
        'y_min': y_min.cpu()
    }, output_model_path)
    print(f"\nPINN Oracle training complete. Best checkpoint saved to {output_model_path}")

    # Return norms so evaluate_on_test_set can reuse them without re-loading tensors
    return best_model_weights, X_max, X_min, y_max, y_min


# ── Step 3: Final Test Evaluation ─────────────────────────────────────────────
def evaluate_on_test_set(processed_data_dir, model_weights,
                          X_max, X_min, y_max, y_min, batch_size=256):
    """
    Evaluates the trained PINN on the held-out test set.
    The model is FROZEN (eval mode + no_grad) — this runs exactly once
    after training and reports the final unbiased performance metrics.

    Test set is the chronological tail of the recording, guaranteed to be
    temporally disjoint from the training/validation windows.
    """
    print("\n--- Step 3: Final Evaluation on Held-Out Test Set ---")
    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps'  if torch.backends.mps.is_available() else 'cpu'
    )

    # ── Load Test-Set Tensors ─────────────────────────────────────────────────
    x_test_path = os.path.join(processed_data_dir, 
                                f"X_normal_{cfg.DATASET_VERSION}_testset.pth")
    y_test_path = os.path.join(processed_data_dir, 
                                f"Y_normal_{cfg.DATASET_VERSION}_testset.pth")

    if not os.path.exists(x_test_path):
        print(f"  [WARNING] Test tensors not found at {x_test_path}. Skipping evaluation.")
        return

    X_test = torch.load(x_test_path, map_location='cpu').double()
    Y_test = torch.load(y_test_path, map_location='cpu').double()

    # Apply the same normalization bounds computed from the training set
    X_test_norm = (X_test - X_min) / (X_max - X_min + 1e-12)
    Y_test_norm = (Y_test - y_min) / (y_max - y_min + 1e-12)

    test_dataset = TensorDataset(X_test_norm, Y_test_norm)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # ── Load Best Model ───────────────────────────────────────────────────────
    pinn_config = cfg.PINN_ARCH_DEFAULT
    model = ConfigurablePINN(
        unmeasured_net_config=pinn_config['unmeasured_net_config'],
        acceleration_net_config=pinn_config['acceleration_net_config'],
        param_init_config=pinn_config['param_init_config'],
        enable_mass_constraints=True
    ).to(device)
    model.load_state_dict(model_weights)
    model.eval()  # Frozen — no gradient updates

    # ── Run Inference ─────────────────────────────────────────────────────────
    loss_method  = ReLoBRaLoLoss(enable_mass_constraints=True)
    total_mse    = 0.0
    total_phys   = 0.0
    n_batches    = 0

    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)

            # Flatten windows: (Batch, Window, Features) → (Batch*Window, Features)
            xb = xb.reshape(-1, xb.shape[-1])
            yb = yb.reshape(-1, yb.shape[-1])

            # Forward pass to get predicted acceleration
            pred_acc = model(xb)

            # Data loss: MSE between predicted and measured acceleration
            mse = torch.nn.functional.mse_loss(pred_acc.float(), yb.float())
            total_mse += mse.item()

            # Physics residuals using the loss method's component computation
            loss_components = loss_method(
                model, xb, yb,
                X_max.to(device), X_min.to(device),
                y_max.to(device), y_min.to(device)
            )
            # Physics loss = sum of the 4 residual terms (indices 1-4 in ReLoBRaLo)
            phys_loss = sum(c.item() for c in loss_components[1:5])
            total_phys += phys_loss
            n_batches  += 1

    avg_mse  = total_mse  / n_batches
    avg_phys = total_phys / n_batches

    print(f"  Test Set Results ({len(test_dataset)} windows, {n_batches} batches):")
    print(f"    Data Loss (MSE, normalized):   {avg_mse:.6f}")
    print(f"    Physics Residual Loss (total): {avg_phys:.6f}")
    print("  Test evaluation complete.")


# ── Pipeline Orchestrator ─────────────────────────────────────────────────────
def run_phase0_pipeline(raw_data_dir, processed_data_dir, output_model_path,
                         epochs, batch_size, num_samples, rotation,
                         training_windows=None, test_windows=None,
                         patience=15, min_delta=1e-4):
    step1_prepare_dataset(
        raw_data_dir=raw_data_dir,
        processed_data_dir=processed_data_dir,
        rotation=rotation,
        training_windows=training_windows,
        test_windows=test_windows
    )
    model_weights, X_max, X_min, y_max, y_min = step2_train_pinn_oracle(
        processed_data_dir=processed_data_dir,
        output_model_path=output_model_path,
        epochs=epochs,
        batch_size=batch_size,
        num_samples=num_samples,
        patience=patience,
        min_delta=min_delta
    )
    evaluate_on_test_set(
        processed_data_dir=processed_data_dir,
        model_weights=model_weights,
        X_max=X_max, X_min=X_min,
        y_max=y_max, y_min=y_min,
        batch_size=batch_size
    )


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 0: Physical Foundation Orchestrator")
    parser.add_argument("--raw_data_dir",       type=str, default=RAW_DATA_DIR)
    parser.add_argument("--processed_data_dir", type=str, default=PROCESSED_DATA_DIR)
    parser.add_argument("--output_model_path",  type=str, default=PINN_MODEL_PATH)
    parser.add_argument("--epochs",             type=int, default=EPOCHS)
    parser.add_argument("--batch_size",         type=int, default=BATCH_SIZE)
    parser.add_argument("--num_samples",        type=int, default=NUM_SAMPLES,
                        help="Limit windows used for PINN training (dry run mode)")
    parser.add_argument("--rotation",           type=int, default=ROTATION,
                        help="Rotation frequency (Hz) to select from the dataset")
    # New: explicit window counts for the data split
    parser.add_argument("--training_windows",   type=int, default=TRAIN_WINDOWS,
                        help="Number of full-rotation windows for the training set "
                             "(default: 15%% of total windows)")
    parser.add_argument("--test_windows",       type=int, default=TEST_WINDOWS,
                        help="Number of full-rotation windows held out for the test set "
                             "(default: 3%% of total windows, min 1). "
                             "Always taken from the chronological end of the recording.")
    parser.add_argument("--patience",           type=int, default=PATIENCE,
                        help="Number of epochs to wait for improvement before early stopping")
    parser.add_argument("--min_delta",          type=float, default=MIN_DELTA,
                        help="Minimum improvemnt in validation loss to reset patience")

    args = parser.parse_args()

    print("Starting Phase 0: Physical Foundation Orchestrator Pipeline.")
    print(f"  Raw data:       {args.raw_data_dir}")
    print(f"  Processed data: {args.processed_data_dir}")
    print(f"  Model output:   {args.output_model_path}")
    print(f"  Epochs: {args.epochs} | Batch size: {args.batch_size} | Rotation: {args.rotation} Hz")
    if args.training_windows:
        print(f"  Training windows (explicit): {args.training_windows}")
    if args.test_windows:
        print(f"  Test windows (explicit):     {args.test_windows}")

    run_phase0_pipeline(
        raw_data_dir=args.raw_data_dir,
        processed_data_dir=args.processed_data_dir,
        output_model_path=args.output_model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        rotation=args.rotation,
        training_windows=args.training_windows,
        test_windows=args.test_windows,
        patience=args.patience,
        min_delta=args.min_delta
    )