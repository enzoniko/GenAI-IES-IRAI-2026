"""
=============================================================================
DATASET VISUALIZATION SCRIPT
=============================================================================
This script provides comprehensive visualization and analysis of the synthetic dataset.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from data_loader import SyntheticDatasetLoader

def create_comprehensive_visualization(data_dir: str = "training_scripts/analytical_analysis/data"):
    """Create comprehensive visualization of the dataset"""
    
    # Load the dataset
    loader = SyntheticDatasetLoader(data_dir)
    summary = loader.get_dataset_summary()
    
    print("Dataset Summary:")
    print(json.dumps(summary, indent=2))
    
    # Create parameter DataFrame
    df = loader.create_parameter_dataframe()
    print(f"\nDataset shape: {df.shape}")
    print(f"Success rate: {df['success'].mean():.2%}")
    
    if df.empty:
        print("No data found!")
        return
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a large figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Parameter distributions
    ax1 = plt.subplot(3, 4, 1)
    df['M1'].hist(bins=20, alpha=0.7, ax=ax1)
    ax1.set_title('M1 Distribution')
    ax1.set_xlabel('M1 (kg)')
    ax1.set_ylabel('Frequency')
    
    ax2 = plt.subplot(3, 4, 2)
    df['Omega'].hist(bins=20, alpha=0.7, ax=ax2)
    ax2.set_title('Omega Distribution')
    ax2.set_xlabel('Omega (rad/s)')
    ax2.set_ylabel('Frequency')
    
    ax3 = plt.subplot(3, 4, 3)
    df['K1'].hist(bins=20, alpha=0.7, ax=ax3)
    ax3.set_title('K1 Distribution')
    ax3.set_xlabel('K1 (N/m)')
    ax3.set_ylabel('Frequency')
    
    ax4 = plt.subplot(3, 4, 4)
    df['E1'].hist(bins=20, alpha=0.7, ax=ax4)
    ax4.set_title('E1 Distribution')
    ax4.set_xlabel('E1 (m)')
    ax4.set_ylabel('Frequency')
    
    # 2. Parameter correlations
    ax5 = plt.subplot(3, 4, 5)
    ax5.scatter(df['M1'], df['Omega'], alpha=0.6)
    ax5.set_xlabel('M1 (kg)')
    ax5.set_ylabel('Omega (rad/s)')
    ax5.set_title('M1 vs Omega')
    
    ax6 = plt.subplot(3, 4, 6)
    ax6.scatter(df['K1'], df['D1'], alpha=0.6)
    ax6.set_xlabel('K1 (N/m)')
    ax6.set_ylabel('D1 (N·s/m)')
    ax6.set_title('K1 vs D1')
    
    ax7 = plt.subplot(3, 4, 7)
    ax7.scatter(df['Omega'], df['E1'], alpha=0.6)
    ax7.set_xlabel('Omega (rad/s)')
    ax7.set_ylabel('E1 (m)')
    ax7.set_title('Omega vs E1')
    
    ax8 = plt.subplot(3, 4, 8)
    ax8.scatter(df['M1'], df['K1'], alpha=0.6)
    ax8.set_xlabel('M1 (kg)')
    ax8.set_ylabel('K1 (N/m)')
    ax8.set_title('M1 vs K1')
    
    # 3. Parameter ranges comparison
    ax9 = plt.subplot(3, 4, 9)
    params_to_plot = ['M1', 'M2', 'M3']
    param_data = [df[param] for param in params_to_plot]
    ax9.boxplot(param_data, labels=params_to_plot)
    ax9.set_title('Mass Parameters')
    ax9.set_ylabel('Mass (kg)')
    ax9.tick_params(axis='x', rotation=45)
    
    ax10 = plt.subplot(3, 4, 10)
    params_to_plot = ['K1', 'K2']
    param_data = [df[param] for param in params_to_plot]
    ax10.boxplot(param_data, labels=params_to_plot)
    ax10.set_title('Stiffness Parameters')
    ax10.set_ylabel('Stiffness (N/m)')
    ax10.tick_params(axis='x', rotation=45)
    
    ax11 = plt.subplot(3, 4, 11)
    params_to_plot = ['D1', 'D2', 'D3']
    param_data = [df[param] for param in params_to_plot]
    ax11.boxplot(param_data, labels=params_to_plot)
    ax11.set_title('Damping Parameters')
    ax11.set_ylabel('Damping (N·s/m)')
    ax11.tick_params(axis='x', rotation=45)
    
    ax12 = plt.subplot(3, 4, 12)
    params_to_plot = ['Omega', 'E1']
    param_data = [df[param] for param in params_to_plot]
    ax12.boxplot(param_data, labels=params_to_plot)
    ax12.set_title('Speed & Imbalance')
    ax12.set_ylabel('Value')
    ax12.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.suptitle(f'Synthetic Dataset Analysis - {summary.get("conservativeness_level", "Unknown")} Level', 
                 y=1.02, fontsize=16)
    plt.show()
    
    # Print parameter statistics
    print("\nParameter Statistics:")
    print("=" * 50)
    for param in ['M1', 'M2', 'M3', 'K1', 'K2', 'D1', 'D2', 'D3', 'Omega', 'E1']:
        if param in df.columns:
            print(f"{param:8s}: {df[param].min():8.2e} to {df[param].max():8.2e} (mean: {df[param].mean():8.2e})")

def plot_sample_simulations(data_dir: str = "training_scripts/analytical_analysis/data", n_samples: int = 4):
    """Plot sample simulations from the dataset"""
    
    loader = SyntheticDatasetLoader(data_dir)
    simulations = loader.load_multiple_simulations(max_sims=n_samples)
    
    if not simulations:
        print("No simulations found!")
        return
    
    # Create subplots for each simulation
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 4*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, sim_data in enumerate(simulations):
        metadata = sim_data['metadata']
        config_id = metadata['config_id']
        
        # Plot impeller orbit
        loader.plot_simulation_orbit(sim_data, 'impeller', ax=axes[i, 0])
        axes[i, 0].set_title(f'Sim {config_id}: Impeller Orbit')
        
        # Plot bearing 1 orbit
        loader.plot_simulation_orbit(sim_data, 'bearing1', ax=axes[i, 1])
        axes[i, 1].set_title(f'Sim {config_id}: Bearing 1 Orbit')
        
        # Plot time history
        loader.plot_time_history(sim_data, 'impeller', 'position', ax=axes[i, 2])
        axes[i, 2].set_title(f'Sim {config_id}: Impeller Position')
        
        # Add parameter info
        params = metadata['parameters']
        info_text = f"M1: {params['M1']:.1f} kg\n"
        info_text += f"Omega: {params['Omega']:.0f} rad/s\n"
        info_text += f"E1: {params['E1']:.2e} m"
        axes[i, 0].text(0.02, 0.98, info_text, transform=axes[i, 0].transAxes, 
                       verticalalignment='top', fontsize=8, 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def create_parameter_space_analysis(data_dir: str = "training_scripts/analytical_analysis/data"):
    """Analyze the parameter space coverage"""
    
    loader = SyntheticDatasetLoader(data_dir)
    df = loader.create_parameter_dataframe()
    
    if df.empty:
        print("No data found!")
        return
    
    # Create correlation matrix
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    param_cols = [col for col in numeric_cols if col not in ['simulation_id', 'success']]
    
    correlation_matrix = df[param_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f')
    plt.title('Parameter Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    # Parameter space coverage
    print("\nParameter Space Coverage:")
    print("=" * 50)
    for param in param_cols:
        if param in df.columns:
            coverage = (df[param].max() - df[param].min()) / df[param].mean() * 100
            print(f"{param:8s}: {coverage:6.1f}% variation")

if __name__ == "__main__":
    # Run all visualizations
    print("Creating comprehensive dataset visualization...")
    create_comprehensive_visualization()
    
    print("\nPlotting sample simulations...")
    plot_sample_simulations()
    
    print("\nAnalyzing parameter space...")
    create_parameter_space_analysis() 