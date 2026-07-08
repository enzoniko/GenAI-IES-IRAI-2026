"""
Metrics Plots Module for Direct PINN Analysis

This module provides functions for creating publication-ready plots
showing metrics across different fault types.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import skew, kurtosis

def plot_metrics_for_publication(residuals_dict, output_prefix):
    """
    Generate publication-ready plots of metrics.
    
    Parameters:
    - residuals_dict: Dictionary containing residuals for different data types
    - output_prefix: Prefix for the output file
    """
    # Create data type mapping dynamically from the keys in residuals_dict
    data_types = sorted(residuals_dict.keys())
    data_type_mapping = {}
    
    # Special cases first
    if 'normal' in data_types:
        data_type_mapping['normal'] = 'N'
        data_types.remove('normal')
    if 'healthy' in data_types:
        data_type_mapping['healthy'] = 'H'
        data_types.remove('healthy')
    
    # Map remaining fault types
    for i, fault_type in enumerate(data_types, 1):
        data_type_mapping[fault_type] = f'F{i}'
    
    # Create output directories if they don't exist
    comparisons_dir = os.path.join(output_prefix, "comparisons")
    data_dir = os.path.join(output_prefix, "data")
    os.makedirs(comparisons_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # Create and save the mapping file in the data directory
    with open(os.path.join(data_dir, 'data_type_mapping.txt'), 'w') as f:
        f.write("Data Type Mapping:\n")
        for full_name, short_name in data_type_mapping.items():
            f.write(f"{full_name} -> {short_name}\n")
    
    # Generate normal-only metrics comparison plot
    if 'normal' in residuals_dict:
        plot_normal_metrics_comparison(residuals_dict, output_prefix)
    else:
        print("No 'normal' data found in residuals, skipping normal metrics comparison plot.")
    
    # Generate comparison plots between normal and fault pairs
    # Generate a list of all fault types (excluding normal/healthy)
    fault_types = [dt for dt in residuals_dict.keys() if dt not in ('normal', 'healthy')]
    
    # If normal isn't in the data, use the first available type as reference
    reference_type = 'normal' if 'normal' in residuals_dict else list(residuals_dict.keys())[0]
    
    # Create sequential pairs of fault types (not all combinations)
    fault_pairs = []
    for i in range(0, len(fault_types), 2):
        if i + 1 < len(fault_types):
            fault_pairs.append((fault_types[i], fault_types[i+1]))
        else:
            # If there's an odd number of fault types, pair the last one with the first one
            if len(fault_types) > 1:
                fault_pairs.append((fault_types[i], fault_types[0]))
    
    # Create fault pairs comparison plots
    plot_triple_comparison_plots(residuals_dict, reference_type, fault_pairs, data_type_mapping, output_prefix)

def plot_normal_metrics_comparison(residuals_dict, output_prefix):
    """
    Generate a publication-ready comparison of metrics for the 'normal' data type.
    
    Parameters:
    - residuals_dict: Dictionary containing residuals for different data types
    - output_prefix: Prefix for the output file
    """
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
    
    # Create output directories if they don't exist
    comparisons_dir = os.path.join(output_prefix, "comparisons")
    data_dir = os.path.join(output_prefix, "data")
    os.makedirs(comparisons_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # Process normal data - for direct PINN we don't have rotation speeds
    # so we'll use dummy data for consistency with other modules
    normal_data = residuals_dict['normal']
    
    # For direct PINN, we only have one data point per metric (no rotation speeds)
    speeds = [1.0]  # Dummy speed
    
    # Calculate metrics
    all_residuals = normal_data['data'].detach().cpu().numpy().flatten()
    mean_val = np.mean(all_residuals)
    median_val = np.median(all_residuals)
    std_val = np.std(all_residuals)
    skew_val = skew(all_residuals)
    kurt_val = kurtosis(all_residuals)
    
    # Get DTW distance from the residuals dictionary if available
    if 'dtw_distances' in normal_data:
        dtw_val = np.mean(normal_data['dtw_distances'])
    else:
        dtw_val = np.nan
    
    # Create arrays for plotting
    mean_by_speed = [mean_val]
    median_by_speed = [median_val]
    std_by_speed = [std_val]
    skew_by_speed = [skew_val]
    kurt_by_speed = [kurt_val]
    dtw_by_speed = [dtw_val]
    
    # Save the processed data
    save_path = os.path.join(data_dir, "normal_metrics.npz")
    np.savez(save_path, 
             speeds=speeds,
             mean=mean_by_speed,
             median=median_by_speed,
             std=std_by_speed,
             skew=skew_by_speed,
             kurt=kurt_by_speed,
             dtw=dtw_by_speed)
    
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
    
    print(f"Normal metrics comparison plot saved to {os.path.join(comparisons_dir, 'normal_metrics.png')}")

def plot_triple_comparison_plots(residuals_dict, reference_type, fault_pairs, data_type_mapping, output_prefix):
    """
    Create triple comparison plots for each pair of fault types alongside the reference type.
    Also saves the processed data for each plot to allow later customization.
    
    Parameters:
    - residuals_dict: Dictionary containing residuals for different data types
    - reference_type: The reference data type (usually 'normal')
    - fault_pairs: List of tuples, each containing a pair of fault types to compare
    - data_type_mapping: Dictionary mapping full data type names to abbreviated versions
    - output_prefix: Prefix for the output files
    """
    # Create output directories if they don't exist
    comparisons_dir = os.path.join(output_prefix, "comparisons")
    data_dir = os.path.join(output_prefix, "data")
    os.makedirs(comparisons_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # Font size for all text elements
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
    
    # Markers for metrics
    metric_markers = {
        'mean': 'o',
        'median': 's',
        'std': '^',
        'skew': 'd',
        'kurt': 'p',
        'dtw': 'x'
    }
    
    # For direct PINN, we don't have rotation speeds, so we'll use bar charts
    # Create a figure for each fault pair
    for fault1, fault2 in tqdm(fault_pairs, desc="Creating fault pair comparison plots"):
        # Skip if any of the data types is missing
        if fault1 not in residuals_dict or fault2 not in residuals_dict or reference_type not in residuals_dict:
            continue
            
        # Data types for this comparison
        data_types_to_plot = [reference_type, fault1, fault2]
        
        # Calculate metrics for each data type
        metrics_data = {}
        for dt in data_types_to_plot:
            all_residuals = residuals_dict[dt]['data'].detach().cpu().numpy().flatten()
            metrics_data[dt] = {
                'mean': np.mean(all_residuals),
                'median': np.median(all_residuals),
                'std': np.std(all_residuals),
                'skew': skew(all_residuals),
                'kurt': kurtosis(all_residuals),
                'dtw': np.mean(residuals_dict[dt]['dtw_distances']) if 'dtw_distances' in residuals_dict[dt] else np.nan
            }
        
        # Save the data for later reproduction
        filename = os.path.join(data_dir, f"triple_{reference_type}_vs_{fault1}_vs_{fault2}.npz")
        
        # Handle special characters in data_type names for the filename
        for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
            filename = filename.replace(char, '_')
        
        # Create the data to save
        save_data = {
            'reference_type': np.array([reference_type], dtype=np.str_),
            'fault1': np.array([fault1], dtype=np.str_),
            'fault2': np.array([fault2], dtype=np.str_),
        }
        
        # Save data type mapping
        mapping_data = '\n'.join([f"{k}={v}" for k, v in data_type_mapping.items()])
        save_data["data_type_mapping"] = np.array([mapping_data], dtype=np.str_)
        
        # Save metrics for each data type
        for dt in data_types_to_plot:
            dt_key = dt.replace("/", "_").replace("\\", "_")  # Replace problematic characters
            for metric in standard_metrics + [dtw_metric]:
                save_data[f"{dt_key}_{metric}"] = np.array([metrics_data[dt][metric]])
        
        # Save the data
        np.savez(filename, **save_data)
        print(f"Saved processed data to {filename}")
        
        # Create the plot - for direct PINN we'll use bar charts
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
        
        print(f"Saved triple comparison plot: {reference_type} vs {fault1} vs {fault2}") 