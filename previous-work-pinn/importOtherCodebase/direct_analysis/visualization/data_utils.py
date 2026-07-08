"""
Data Utilities for Visualization from Saved Data

This module provides functions for loading saved data and regenerating
plots without recomputing the model.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import time

def create_plots_from_saved_data(data_dir, output_prefix=None):
    """
    Generate plots based on previously saved data without recomputing residuals.
    
    Parameters:
    - data_dir: Directory containing saved data files (.npz)
    - output_prefix: Optional prefix for output directories, defaults to 'regenerated'
    """
    if output_prefix is None:
        timestamp = int(time.time())
        output_prefix = f'results/direct_pinn_recreated_{timestamp}'
    
    # Create output directories
    plots_dir = os.path.join(output_prefix, "plots")
    comparisons_dir = os.path.join(output_prefix, "comparisons")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(comparisons_dir, exist_ok=True)
    
    # Process boxplot data files (from plot_residuals_by_variable_and_frequency)
    print("Regenerating boxplot visualizations...")
    boxplot_files = [f for f in os.listdir(data_dir) if f.endswith('_residuals.npz')]
    
    for file in tqdm(boxplot_files, desc="Processing boxplot data files"):
        try:
            # Load data
            data = np.load(os.path.join(data_dir, file))
            
            # Extract data
            data_type = data['data_type'][0].decode('utf-8')
            variables = [v.decode('utf-8') for v in data['variables']]
            speeds = data['speeds']
            y_limits = data['y_limits']
            
            # Create figure
            num_variables = len(variables)
            num_speeds = len(speeds)
            fig, axes = plt.subplots(num_variables, 1, figsize=(15, 6 * num_variables), sharex=True)
            
            if num_variables == 1:
                axes = [axes]
                
            for var_idx, var in enumerate(variables):
                # Reconstruct boxplot data from values and indices
                if f'{var}_boxplot_values' in data and f'{var}_boxplot_indices' in data:
                    values = data[f'{var}_boxplot_values']
                    indices = data[f'{var}_boxplot_indices']
                    
                    # Group values by index
                    boxplot_data = [[] for _ in range(num_speeds)]
                    for value, idx in zip(values, indices):
                        if idx < len(boxplot_data):
                            boxplot_data[idx].append(value)
                    
                    # Convert to numpy arrays
                    boxplot_data = [np.array(group) if group else np.array([]) for group in boxplot_data]
                    
                    # Create violin and box plots
                    positions = np.arange(1, num_speeds + 1)
                    axes[var_idx].violinplot(boxplot_data, positions=positions, showextrema=True, showmedians=True)
                    axes[var_idx].boxplot(boxplot_data, positions=positions, widths=0.3, patch_artist=True, 
                                        boxprops=dict(facecolor="lightblue"))
                    
                    # Set y-axis limits based on saved limits
                    if var_idx < len(y_limits):
                        axes[var_idx].set_ylim(y_limits[var_idx][0], y_limits[var_idx][1])
                    
                    # Add statistics annotations
                    for speed_idx, speed in enumerate(speeds):
                        if f'{var}_mean' in data and speed_idx < len(data[f'{var}_mean']):
                            mean_val = data[f'{var}_mean'][speed_idx]
                            median_val = data[f'{var}_median'][speed_idx]
                            std_val = data[f'{var}_std'][speed_idx]
                            skew_val = data[f'{var}_skew'][speed_idx]
                            kurt_val = data[f'{var}_kurt'][speed_idx]
                            
                            if np.isfinite(mean_val):  # Check if we have valid statistics
                                axes[var_idx].text(
                                    x=speed_idx + 1, 
                                    y=axes[var_idx].get_ylim()[1] * 0.9, 
                                    s=(f"Mean: {mean_val:.2f}\n"
                                    f"Median: {median_val:.2f}\n"
                                    f"Std: {std_val:.2f}\n"
                                    f"Skew: {skew_val:.2f}\n"
                                    f"Kurt: {kurt_val:.2f}"),
                                    ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white")
                                )
                    
                    axes[var_idx].grid(True, which="both", linestyle="--", linewidth=0.5)
                    axes[var_idx].set_ylabel(f"Residuals for {var}")
                
            # Set x-axis labels
            axes[-1].set_xlabel("Data Type")
            axes[-1].set_xticks(np.arange(1, num_speeds + 1))
            axes[-1].set_xticklabels([data_type], rotation=45, ha='right')
            
            plt.suptitle(f"Residuals for {data_type} data using Direct PINN")
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"{data_type}_residuals.png"))
            plt.close()
            
            print(f"Recreated plot for {data_type} saved to {os.path.join(plots_dir, f'{data_type}_residuals.png')}")
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    # Regenerate normal metrics comparison plot if data exists
    print("\nRegenerating normal metrics comparison plot...")
    normal_metrics_file = os.path.join(data_dir, "normal_metrics.npz")
    if os.path.exists(normal_metrics_file):
        try:
            data = np.load(normal_metrics_file)
            
            # Set publication-quality font sizes
            plt.rcParams.update({
                'font.size': 18,
                'axes.titlesize': 18,
                'axes.labelsize': 18,
                'xtick.labelsize': 16,
                'ytick.labelsize': 16, 
                'legend.fontsize': 16,
                'figure.titlesize': 20
            })
            
            # Extract data
            speeds = data['speeds']
            mean_by_speed = data['mean']
            median_by_speed = data['median']
            std_by_speed = data['std']
            skew_by_speed = data['skew']
            kurt_by_speed = data['kurt']
            dtw_by_speed = data['dtw']
            
            # Create plot
            fig, axs = plt.subplots(2, 3, figsize=(20, 12))
            axs = axs.flatten()
            
            axs[0].bar(speeds, mean_by_speed, width=0.4)
            axs[0].set_title('Mean Residual')
            axs[0].set_xlabel('Direct PINN')
            axs[0].set_ylabel('Value')
            axs[0].grid(True)
            axs[0].set_xticks(speeds)
            axs[0].set_xticklabels(['Normal'])
            
            axs[1].bar(speeds, median_by_speed, width=0.4)
            axs[1].set_title('Median Residual')
            axs[1].set_xlabel('Direct PINN')
            axs[1].set_ylabel('Value')
            axs[1].grid(True)
            axs[1].set_xticks(speeds)
            axs[1].set_xticklabels(['Normal'])
            
            axs[2].bar(speeds, std_by_speed, width=0.4)
            axs[2].set_title('Standard Deviation of Residuals')
            axs[2].set_xlabel('Direct PINN')
            axs[2].set_ylabel('Value')
            axs[2].grid(True)
            axs[2].set_xticks(speeds)
            axs[2].set_xticklabels(['Normal'])
            
            axs[3].bar(speeds, skew_by_speed, width=0.4)
            axs[3].set_title('Skewness of Residuals')
            axs[3].set_xlabel('Direct PINN')
            axs[3].set_ylabel('Value')
            axs[3].grid(True)
            axs[3].set_xticks(speeds)
            axs[3].set_xticklabels(['Normal'])
            
            axs[4].bar(speeds, kurt_by_speed, width=0.4)
            axs[4].set_title('Kurtosis of Residuals')
            axs[4].set_xlabel('Direct PINN')
            axs[4].set_ylabel('Value')
            axs[4].grid(True)
            axs[4].set_xticks(speeds)
            axs[4].set_xticklabels(['Normal'])
            
            axs[5].bar(speeds, dtw_by_speed, width=0.4)
            axs[5].set_title('DTW Distance')
            axs[5].set_xlabel('Direct PINN')
            axs[5].set_ylabel('Value')
            axs[5].grid(True)
            axs[5].set_xticks(speeds)
            axs[5].set_xticklabels(['Normal'])
            
            plt.tight_layout()
            plt.savefig(os.path.join(comparisons_dir, "normal_metrics.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Recreated normal metrics comparison plot saved to {os.path.join(comparisons_dir, 'normal_metrics.png')}")
        except Exception as e:
            print(f"Error processing normal metrics file: {e}")
    else:
        print(f"Normal metrics file not found at {normal_metrics_file}, skipping.")
    
    # Process triple comparison data files
    print("\nRegenerating triple comparison visualizations...")
    triple_files = [f for f in os.listdir(data_dir) if f.startswith('triple_') and f.endswith('.npz')]
    
    for file in tqdm(triple_files, desc="Processing triple comparison files"):
        try:
            data = np.load(os.path.join(data_dir, file))
            
            # Extract metadata
            reference_type = data['reference_type'][0].decode('utf-8')
            fault1 = data['fault1'][0].decode('utf-8')
            fault2 = data['fault2'][0].decode('utf-8')
            
            # Parse data type mapping
            mapping_text = data['data_type_mapping'][0].decode('utf-8')
            data_type_mapping = {}
            for line in mapping_text.split('\n'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    data_type_mapping[key] = value
            
            # Data types for this comparison
            data_types_to_plot = [reference_type, fault1, fault2]
            
            # Set font sizes
            FONT_SIZE = 18
            
            # Define metrics
            standard_metrics = ['mean', 'median', 'std', 'skew', 'kurt']
            dtw_metric = 'dtw'
            
            # Metric names for legend
            metric_names = {
                'mean': 'Mean',
                'median': 'Median',
                'std': 'Std Dev',
                'skew': 'Skewness',
                'kurt': 'Kurtosis',
                'dtw': 'DTW Distance'
            }
            
            # Colors for metrics
            metric_colors = {
                'mean': 'blue',
                'median': 'green',
                'std': 'red',
                'skew': 'purple',
                'kurt': 'orange',
                'dtw': 'black'
            }
            
            # Extract metric values for each data type
            metrics_data = {}
            for dt in data_types_to_plot:
                dt_key = dt.replace("/", "_").replace("\\", "_")  # Clean key
                metrics_data[dt] = {}
                for metric in standard_metrics + [dtw_metric]:
                    key = f"{dt_key}_{metric}"
                    if key in data:
                        metrics_data[dt][metric] = data[key][0]
                    else:
                        metrics_data[dt][metric] = np.nan
            
            # Create the plot - use bar charts for Direct PINN
            fig, axs = plt.subplots(2, 3, figsize=(20, 12))
            axs = axs.flatten()
            
            # Set up x positions for bars
            x = np.arange(len(data_types_to_plot))
            width = 0.35
            
            # Plot each metric
            for i, metric in enumerate(standard_metrics + [dtw_metric]):
                values = [metrics_data[dt][metric] for dt in data_types_to_plot]
                axs[i].bar(x, values, width, color=metric_colors[metric])
                axs[i].set_title(metric_names[metric])
                axs[i].set_xticks(x)
                axs[i].set_xticklabels([data_type_mapping.get(dt, dt) for dt in data_types_to_plot])
                axs[i].grid(True, linestyle='--', alpha=0.7)
            
            plt.suptitle(f"Triple Comparison: {data_type_mapping.get(reference_type, reference_type)} vs. "
                       f"{data_type_mapping.get(fault1, fault1)} vs. {data_type_mapping.get(fault2, fault2)}",
                       fontsize=FONT_SIZE+4, y=0.98)
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
            
            # Save figure
            output_file = os.path.join(comparisons_dir, f"triple_{reference_type}_vs_{fault1}_vs_{fault2}.pdf")
            
            # Handle special characters in the filename
            for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
                output_file = output_file.replace(char, '_')
            
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Recreated triple comparison plot: {reference_type} vs {fault1} vs {fault2}")
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    print(f"\nAll plots have been regenerated in the {plots_dir} and {comparisons_dir} directories.")

def run_visualization_only(residuals_path, output_dir):
    """Run visualization on saved residuals.
    
    Parameters:
    - residuals_path: Path to the saved residuals file
    - output_dir: Directory to save visualization outputs
    """
    print(f"Loading residuals from {residuals_path}...")
    
    # Use the new load_and_standardize_residuals function
    from utils.residuals_utils import load_and_standardize_residuals
    standardized_residuals = load_and_standardize_residuals(residuals_path)
    
    # Generate plots from the residuals
    print("Generating plots...")
    plot_residuals_by_variable_and_frequency(standardized_residuals, output_dir)
    plot_metrics_for_publication(standardized_residuals, output_dir)
    
    print(f"Visualization completed. Results saved to {output_dir}") 