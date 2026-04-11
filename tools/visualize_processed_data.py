import torch
import matplotlib.pyplot as plt
import os
import sys
import argparse
import numpy as np

# Ensure we can import from src/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.configs as cfg

def visualize_dataset(category='normal', window_idx=0):
    """
    Loads and visualizes processed tensors for a given category and window.
    Shows both Physical and Normalized units if metadata is available.
    """
    data_dir = cfg.DATA_DIR_PROCESSED
    x_path = os.path.join(data_dir, f"X_{category}_{cfg.DATASET_VERSION}_trainingset.pth")
    y_path = os.path.join(data_dir, f"Y_{category}_{cfg.DATASET_VERSION}_trainingset.pth")
    norm_path = cfg.NORM_METADATA_PATH
    
    if not os.path.exists(x_path):
        print(f"Error: Tensor file not found at {x_path}")
        return

    print(f"Loading tensors for category: {category}")
    X = torch.load(x_path, map_location='cpu', weights_only=True)
    Y = torch.load(y_path, map_location='cpu', weights_only=True)
    
    metadata = None
    if os.path.exists(norm_path):
        metadata = torch.load(norm_path, map_location='cpu', weights_only=True)
        print(f"Normalization metadata loaded from {norm_path}")
    else:
        print(f"Warning: Normalization metadata not found at {norm_path}. Only physical units will be shown.")

    if window_idx >= X.shape[0]:
        print(f"Error: Window index {window_idx} out of bounds (max {X.shape[0]-1})")
        return

    window_x = X[window_idx].numpy()
    window_y = Y[window_idx].numpy()
    
    time = window_x[:, 9]
    omega = window_x[0, 8]
    
    # Calculate Normalized Y if metadata is available
    window_y_norm = None
    if metadata is not None:
        y_min = metadata['y_min'].numpy()
        y_max = metadata['y_max'].numpy()
        window_y_norm = (window_y - y_min) / (y_max - y_min + 1e-12)

    rows = 6 if window_y_norm is not None else 3
    fig, axes = plt.subplots(rows, 4, figsize=(20, 4 * rows), sharex=True)
    fig.suptitle(f"Processed MaFaulDa Signals - Category: {category.upper()} - Window: {window_idx}\nRotational Speed: {omega/(2*np.pi):.2f} Hz", fontsize=16)

    channels = ['Underhang Rad', 'Underhang Tan', 'Overhang Rad', 'Overhang Tan']
    
    for i in range(4):
        # ─── PHYSICAL UNITS ───
        # Accel
        axes[0, i].plot(time, window_y[:, i], color='tab:red')
        axes[0, i].set_title(f"Channel {i+1}: {channels[i]} (Physical)")
        if i == 0: axes[0, i].set_ylabel("Accel (m/s²)")
        axes[0, i].grid(True, alpha=0.3)
        # Vel
        axes[1, i].plot(time, window_x[:, i], color='tab:blue')
        if i == 0: axes[1, i].set_ylabel("Velocity (m/s)")
        axes[1, i].grid(True, alpha=0.3)
        # Pos
        axes[2, i].plot(time, window_x[:, i+4], color='tab:green')
        if i == 0: axes[2, i].set_ylabel("Position (m)")
        axes[2, i].grid(True, alpha=0.3)

        # ─── NORMALIZED UNITS (Training Ready) ───
        if window_y_norm is not None:
            # Normalized Accel
            axes[3, i].plot(time, window_y_norm[:, i], color='darkred', linestyle='--')
            axes[3, i].set_title(f"Channel {i+1}: Normalized")
            if i == 0: axes[3, i].set_ylabel("Accel (Norm [0,1])")
            axes[3, i].grid(True, alpha=0.3)
            
            # Note: Velocities/Positions are not currently normalized in the dataset loader
            # but we could visualize their ranges here if we add metadata for X as well.
            axes[4, i].set_visible(False) # Placeholder for future X-norm
            axes[5, i].set_visible(False)
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    os.makedirs(os.path.join(cfg.RESULTS_DIR, "plots"), exist_ok=True)
    save_path = os.path.join(cfg.RESULTS_DIR, f"plots/dataset_viz_{category}_w{window_idx}_dual.png")
    plt.savefig(save_path, dpi=300)
    print(f"Visualization saved to: {save_path}")
    plt.show()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    os.makedirs(os.path.join(cfg.RESULTS_DIR, "plots"), exist_ok=True)
    save_path = os.path.join(cfg.RESULTS_DIR, f"plots/dataset_viz_{category}_w{window_idx}.png")
    plt.savefig(save_path, dpi=300)
    print(f"Visualization saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize processed dataset tensors.")
    parser.add_argument("--category", type=str, default='normal', help="Fault category (normal, imbalance, overhang)")
    parser.add_argument("--window", type=int, default=0, help="Window index to visualize")
    
    args = parser.parse_args()
    visualize_dataset(category=args.category, window_idx=args.window)
