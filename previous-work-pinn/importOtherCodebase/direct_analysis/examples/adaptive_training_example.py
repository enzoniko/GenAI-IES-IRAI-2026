"""
Example script demonstrating how to use the simplified adaptive optimizer.

This script shows how to:
1. Import the necessary modules
2. Load and prepare data
3. Create a PINN model
4. Train it using the adaptive optimizer
5. Visualize the results
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from direct_analysis.data_utils import prepare_data
from direct_analysis.training import train_model_adaptive
from direct_analysis.adaptive_loss import adaptive_custom_loss
from Models.basicPINNv3 import PINN

def run_adaptive_training_example():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = 'results/adaptive_example'
    os.makedirs(output_dir, exist_ok=True)
    
    model_save_path = os.path.join(output_dir, 'model.pth')
    history_path = os.path.join(output_dir, 'history.npz')
    plot_path = os.path.join(output_dir, 'training_plot.png')
    
    # Load data (adjust paths as needed)
    try:
        print("Loading and preparing data...")
        X = torch.load('Data/X_normal.pth')[:, :249998, :]
        y = torch.load('Data/Y_normal.pth')[:, :249998, :]
        
        # Add time feature to X
        t = torch.arange(0, 2e-5 * X.size(1), 2e-5).view(1, -1, 1).expand(X.size(0), -1, -1)[:, :249998, :]
        X = torch.cat((X, t), dim=2)
        
        # Reshape for training
        X = X.reshape(-1, X.size(-1))
        y = y.reshape(-1, y.size(-1))
        
        # Prepare data with smaller max_samples for faster example
        data_dict = prepare_data(X, y, batch_size=128, normalize=True, 
                                test_size=0.1, val_size=0.2, max_samples=5000)
        
        # Create model
        model = PINN().to(device)
        
        # Use a smaller learning rate to prevent numerical instability
        learning_rate = 1e-3  # Reduced from 1e-3
        
        # Train with adaptive optimizer
        print("\nTraining with adaptive optimizer...")
        model, history = train_model_adaptive(
            model=model,
            data_dict=data_dict,
            device=device,
            custom_loss_fn=adaptive_custom_loss,
            epochs=5000,  # Small number for example
            alpha=0.9,
            learning_rate=learning_rate,
            update_frequency=1,
            model_save_path=model_save_path
        )
        
        # Print final lambda values
        print("\nFinal Lagrange multipliers (lambda values):")
        print(f"λ_data: {history['data_weight'][-1]:.4f}")
        print(f"λ_phys: {history['phys_weight'][-1]:.4f}")
        print(f"λ_reg: {history['reg_weight'][-1]:.4f}")
        
        # Print final model parameters
        print("\nFinal optimized physical parameters:")
        physical_params = {}
        for name, param in model.named_parameters():
            if not name.startswith('layers'):  # Only print physical parameters, not all network weights
                physical_params[name] = param.item()
                print(f"{name}: {param.item():.6f}")
        
        # Check if training was successful by looking for NaN or Inf values
        has_valid_history = all(not np.isnan(loss) and not np.isinf(loss) for loss in history['train_loss'])
        
        if has_valid_history:
            # Save training history
            np.savez(history_path, 
                    train_loss=np.array(history['train_loss']),
                    val_loss=np.array(history['val_loss']),
                    data_weight=np.array(history['data_weight']),
                    phys_weight=np.array(history['phys_weight']),
                    reg_weight=np.array(history['reg_weight']),
                    data_loss=np.array(history['data_loss']),
                    phys_loss=np.array(history['phys_loss']),
                    reg_loss=np.array(history['reg_loss']))
            
            # Plot training results
            plt.figure(figsize=(15, 15))
            
            # Plot losses
            plt.subplot(3, 1, 1)
            plt.plot(history['train_loss'], label='Training Loss')
            plt.plot(history['val_loss'], label='Validation Loss')
            plt.yscale('log')
            plt.xlabel('Epoch')
            plt.ylabel('Loss (log scale)')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True)
            
            # Plot adaptive weights
            plt.subplot(3, 1, 2)
            plt.plot(history['data_weight'], label='Data Weight (λ_data)')
            plt.plot(history['phys_weight'], label='Physics Weight (λ_phys)')
            plt.plot(history['reg_weight'], label='Regularization Weight (λ_reg)')
            plt.xlabel('Epoch')
            plt.ylabel('Weight Value')
            plt.title('Adaptive Loss Weights')
            plt.legend()
            plt.grid(True)
            
            # Add a third subplot to track individual loss components
            # First, we need to collect this data during training
            # We'll modify history to include this information
            if 'data_loss' in history and 'phys_loss' in history and 'reg_loss' in history:
                plt.subplot(3, 1, 3)
                plt.plot(history['data_loss'], label='Data Loss')
                plt.plot(history['phys_loss'], label='Physics Loss')
                plt.plot(history['reg_loss'], label='Regularization Loss')
                plt.yscale('log')
                plt.xlabel('Epoch')
                plt.ylabel('Component Losses (log scale)')
                plt.title('Individual Loss Components')
                plt.legend()
                plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            
            print("\nTraining completed!")
            print(f"Model saved to: {model_save_path}")
            print(f"Training history saved to: {history_path}")
            print(f"Training plot saved to: {plot_path}")
        else:
            print("\nTraining encountered numerical instabilities (NaN or Inf values).")
            print("Try reducing the learning rate further or adjusting the model parameters.")
    
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please make sure the data files exist at the expected paths.")
        print("You might need to adjust the data paths in the script.")
    
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    run_adaptive_training_example() 