import os
import sys
import torch

# Ensure we can import from src/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.pinn import ConfigurablePINN, get_default_pinn_config
from src.models.relobralo_loss import ReLoBRaLoLoss, adaptive_custom_loss
import src.constants as c

def run_prediction_test():
    print("--- PINN Prediction Validation Test ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Paths — load from the trainingset tensors (never from the held-out testset)
    processed_data_dir = "data/processed-mafaulda"
    x_path = os.path.join(processed_data_dir, c.DATASET_CURRENT_VERSION, f"X_normal_{c.DATASET_CURRENT_VERSION}_trainingset.pth")
    y_path = os.path.join(processed_data_dir, c.DATASET_CURRENT_VERSION, f"Y_normal_{c.DATASET_CURRENT_VERSION}_trainingset.pth")
    model_path = "results/pinn.pth"
    
    if not os.path.exists(x_path):
        print(f"Dataset not found at {x_path}")
        return
        
    print(f"Loading data from {x_path}")
    X = torch.load(x_path, map_location='cpu', weights_only=True)
    Y = torch.load(y_path, map_location='cpu', weights_only=True)
    
    # Grab just the first window to test
    X = X[0:1].double() # shape: (1, seq, 10)
    Y = Y[0:1].double() # shape: (1, seq, 4)
    
    # Flatten
    X = X.reshape(-1, 10)
    Y = Y.reshape(-1, 4)
    
    # Model
    pinn_config = get_default_pinn_config()
    model = ConfigurablePINN(
        unmeasured_net_config=pinn_config['unmeasured_net_config'],
        acceleration_net_config=pinn_config['acceleration_net_config'],
        param_init_config=pinn_config['param_init_config'],
        enable_mass_constraints=True
    ).to(device)
    
    # Initialize bounds
    X_max = X.max(dim=0)[0]
    X_min = X.min(dim=0)[0]
    y_max = Y.max(dim=0)[0]
    y_min = Y.min(dim=0)[0]
    
    # Attempt to load weights AND the exact normalization bounds used during training
    if os.path.exists(model_path):
        print(f"Loading trained weights from {model_path}")
        # Workaround since torch.load returns the raw dict containing weights
        try:
            ckpt = torch.load(model_path, map_location=device, weights_only=True)
            model.load_state_dict(ckpt['model_state_dict'])
            X_max, X_min = ckpt['X_max'].to(device), ckpt['X_min'].to(device)
            y_max, y_min = ckpt['y_max'].to(device), ckpt['y_min'].to(device)
        except Exception as e:
            print(f"Failed to load weights properly: {e}\nUsing randomized init.")
            X_max, X_min = X_max.to(device), X_min.to(device)
            y_max, y_min = y_max.to(device), y_min.to(device)
    else:
        print("Trained weights not found, using initialized untrained weights.")
        X_max, X_min = X_max.to(device), X_min.to(device)
        y_max, y_min = y_max.to(device), y_min.to(device)
        
    X_norm = (X.to(device) - X_min) / (X_max - X_min + 1e-12)
    Y_norm = (Y.to(device) - y_min) / (y_max - y_min + 1e-12)
    
    model.eval()
    with torch.no_grad():
        # Predict normalized accelerations
        pred_norm = model(X_norm)
        
        # Denormalize predictions to get physical outputs (m/s^2)
        pred_phys = pred_norm * (y_max - y_min + 1e-12) + y_min
        Y_phys = Y.to(device)
        
        # Calculate Machine Learning Important Values (Losses)
        loss_method = ReLoBRaLoLoss(enable_mass_constraints=True)
        # adaptive_custom_loss takes normalized inputs as arguments but expects denormalization bounds!
        loss_components = loss_method(model, X_norm, Y_norm, X_max, X_min, y_max, y_min)
        
        data_loss = loss_components[0].item()
        phys_res1 = loss_components[1].item()
        phys_res2 = loss_components[2].item()
        phys_res3 = loss_components[3].item()
        phys_res4 = loss_components[4].item()
        
    print("\n" + "="*70)
    print("--- Machine Learn Important Values (PINN RMSE Losses) ---")
    print("="*70)
    print(f"Data RMSE (Normalized Diff):         {data_loss:.6f}")
    print(f"Physics Residual 1 (Force Scale x10^6): {phys_res1:.6f}")
    print(f"Physics Residual 2 (Force Scale x10^6): {phys_res2:.6f}")
    print(f"Physics Residual 3 (Mass Scale x10^6):  {phys_res3:.6f}")
    print(f"Physics Residual 4 (Mass Scale x10^6):  {phys_res4:.6f}")
    print("="*70)
    
    print("\n--- Side-by-Side Comparison (Physical accelerations m/s^2) ---")
    print(f"{'Time Step':<12} | {'Real Acc X2':<12} | {'Pred Acc X2':<12} || {'Real Acc Y2':<12} | {'Pred Acc Y2':<12}")
    print("-" * 80)
    
    # Print first 20 steps to see the side-by-side
    for i in range(20):
        r_x2 = Y_phys[i, 0].item()
        p_x2 = pred_phys[i, 0].item()
        r_y2 = Y_phys[i, 1].item()
        p_y2 = pred_phys[i, 1].item()
        print(f"Idx {i:03d}      | {r_x2:12.4f} | {p_x2:12.4f} || {r_y2:12.4f} | {p_y2:12.4f}")
        
    print("\nTest completed successfully!")

if __name__ == "__main__":
    run_prediction_test()
