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
from torch.utils.data import TensorDataset, DataLoader, random_split

from src.data.clean_mafaulda_processor import CleanMaFaulDaProcessor
from src.models.pinn import ConfigurablePINN, get_default_pinn_config
from src.models.relobralo_loss import ReLoBRaLoLoss

def step1_prepare_dataset(data_dir):
    """
    Uses CleanMaFaulDaProcessor logic to clean raw CSVs into physical tensors.
    It extracts exactly 10 features (Velocities, Positions, Omega, Time) 
    while mitigating integration drift.
    """
    print("\n--- Step 1: Dataset Loading and Preparation ---")
    processor = CleanMaFaulDaProcessor(data_dir=data_dir, output_dir=data_dir)
    processor.run(category='normal') # We train the PINN oracle only on healthy data
    print("Data preparation complete.")

def step2_train_pinn_oracle(data_dir, output_model_path, epochs=100, batch_size=256, num_samples=None):
    """
    Trains the ConfigurablePINN using the ReLoBRaLo dynamic loss.
    This natively replicates the legacy training loop in a clean format.
    """
    print("\n--- Step 2: PINN Training & Feature Extraction Setup ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Processed Tensors
    x_path = os.path.join(data_dir, "v3", "X_normal_v3.pth")
    y_path = os.path.join(data_dir, "v3", "Y_normal_v3.pth")
    
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(f"Processed tensors not found at {x_path}. Run Step 1 first.")

    # Load directly to CPU to prevent GPU VRAM spikes
    X_raw = torch.load(x_path, map_location='cpu')
    Y_raw = torch.load(y_path, map_location='cpu')
    
    # Flatten (Samples, Seq_Len, Features) -> (Batch, Features)
    X = X_raw.reshape(-1, X_raw.shape[-1])
    Y = Y_raw.reshape(-1, Y_raw.shape[-1])
    
    # 2. Min-Max Normalization 
    # Calculate bounds BEFORE casting to double to save memory
    X_max, X_min = X.max(dim=0)[0], X.min(dim=0)[0]
    y_max, y_min = Y.max(dim=0)[0], Y.min(dim=0)[0]
    X_max, X_min = X_max.double(), X_min.double()
    y_max, y_min = y_max.double(), y_min.double()
    
    # Subsample for resource-constrained dry runs
    if num_samples is not None:
        limit = num_samples * 5000  # Sequence length multiplier
        X = X[:limit]
        Y = Y[:limit]
        
    X = X.double()
    Y = Y.double()
    
    X_norm = (X - X_min) / (X_max - X_min + 1e-12)
    Y_norm = (Y - y_min) / (y_max - y_min + 1e-12)
    
    dataset = TensorDataset(X_norm, Y_norm)
    
    # Split into 80% Train and 20% Validation
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 3. Initialize Oracle Model and ReLoBRaLo Loss
    pinn_config = get_default_pinn_config()
    model = ConfigurablePINN(
        unmeasured_net_config=pinn_config['unmeasured_net_config'],
        acceleration_net_config=pinn_config['acceleration_net_config'],
        param_init_config=pinn_config['param_init_config'],
        enable_mass_constraints=True
    ).to(device)
    
    loss_method = ReLoBRaLoLoss(enable_mass_constraints=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5, min_lr=1e-6)
    
    best_val_loss = float('inf')
    best_model_weights = None
    
    # 4. Clean Training Loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            
            # Forward Pass & Compute the 7 Loss Components
            loss_components = loss_method(
                model, xb, yb, 
                X_max.to(device), X_min.to(device), 
                y_max.to(device), y_min.to(device)
            )
            
            # Dynamic ReLoBRaLo Step Optimization
            total_loss = loss_method.step(loss_components, optimizer, model, xb, yb)
            
            if isinstance(total_loss, torch.Tensor):
                epoch_loss += total_loss.item()
                
        # Validation Pass
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb_val, yb_val in val_loader:
                xb_val, yb_val = xb_val.to(device), yb_val.to(device)
                
                loss_components = loss_method(
                    model, xb_val, yb_val, 
                    X_max.to(device), X_min.to(device), 
                    y_max.to(device), y_min.to(device)
                )
                
                # Compute validation loss using current ReLoBRaLo weights
                weighted_val = sum(loss_method.current_weights[key] * loss_components[i].item() 
                                   for i, key in enumerate(loss_method.loss_keys) if i < len(loss_components))
                val_loss += weighted_val
                
        avg_train = epoch_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        scheduler.step(avg_val)
        
        print(f"Epoch {epoch+1:03d}/{epochs} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            # Create a deep copy of the best weights
            best_model_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # 5. Save Final Baseline Weights for the Phase 3 SDEdit Oracle
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    torch.save({
        'model_state_dict': best_model_weights,
        'X_max': X_max.cpu(),
        'X_min': X_min.cpu(),
        'y_max': y_max.cpu(),
        'y_min': y_min.cpu()
    }, output_model_path)
    print(f"\nFeature Extraction implicit setup complete.")
    print(f"PINN Oracle weights saved to {output_model_path}")

def run_phase0_pipeline(data_dir, output_model_path, epochs, batch_size, num_samples):
    step1_prepare_dataset(data_dir=data_dir)
    step2_train_pinn_oracle(data_dir=data_dir, output_model_path=output_model_path, epochs=epochs, batch_size=batch_size, num_samples=num_samples)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 0: Physical Foundation Orchestrator")
    parser.add_argument("--data_dir", type=str, default="mafaulda-data", help="Directory containing the raw MaFaulDa dataset")
    parser.add_argument("--output_model_path", type=str, default="results/relobralo_model.pth", help="Path to save the trained PINN model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train the PINN oracle")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--num_samples", type=int, default=None, help="Limit number of trajectories to train on")
    
    args = parser.parse_args()
    
    print("Starting Phase 0: Physical Foundation Orchestrator Pipeline.")
    print(f"Configuration: Data Dir: {args.data_dir}, Output Model: {args.output_model_path}, Epochs: {args.epochs}, Batch Size: {args.batch_size}")
    
    run_phase0_pipeline(
        data_dir=args.data_dir,
        output_model_path=args.output_model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_samples=args.num_samples
    )