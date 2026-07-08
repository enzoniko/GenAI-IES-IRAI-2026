"""
Boxplot visualization module for direct PINN residuals

This module provides functions for creating boxplot and violin plot
visualizations of residuals by variable and frequency.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from scipy.stats import skew, kurtosis

def plot_residuals_by_variable_and_frequency(residuals_dict, output_prefix):
    """
    Generate summary plots with boxplots and violin plots of residuals
    for each variable across different rotation frequencies, grouped by data type.
    Also saves the processed data to allow for re-plotting later.
    
    Parameters:
    - residuals_dict: A dictionary where keys are data types and values are dictionaries 
                      containing 'data' tensor with residuals.
    - output_prefix: Prefix for the output directory to save the generated plots.
    """
    variables = ['x2_ddot', 'y2_ddot', 'x3_ddot', 'y3_ddot']
    if 'normal' in residuals_dict and 'r1' in residuals_dict['normal']:
        variables.extend(['r1', 'r2', 'r3', 'r4'])
    
    num_variables = len(variables)
    
    # Create output directories if they don't exist
    plots_dir = os.path.join(output_prefix, "plots")
    data_dir = os.path.join(output_prefix, "data")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # First pass: collect all residuals to determine global y-axis limits
    all_residuals = {var: [] for var in variables}
    for data_type, data_dict in residuals_dict.items():
        for var_idx, var in enumerate(variables):
            if var_idx < 4:  # First 4 variables are in 'data'
                all_residuals[var].extend(data_dict['data'][:, var_idx].detach().cpu().numpy())
            elif var in data_dict:  # Check if physical residuals exist
                all_residuals[var].extend(data_dict[var].detach().cpu().numpy())
    
    # Calculate global y-axis limits for each variable
    y_limits = {}
    for var in variables:
        residuals = all_residuals[var]
        if residuals:
            y_limits[var] = {
                'min': np.percentile(residuals, 1),  # Use 1st percentile for lower bound
                'max': np.percentile(residuals, 99)  # Use 99th percentile for upper bound
            }
    
    # Process each data type separately
    for data_type, data_dict in tqdm(residuals_dict.items(), desc="Plotting data types"):
        # For direct PINN, we don't have rotation speeds, so create a single group
        # This makes the boxplots consistent with hybrid analysis
        speeds = [1.0]  # Use a dummy speed
        num_speeds = 1
        
        # Create a data structure to store all processed data for this data type
        processed_data = {
            'data_type': data_type,
            'speeds': speeds,
            'variables': variables,
            'boxplot_data': {},
            'statistics': {}
        }
        
        for var_idx, var_name in enumerate(variables):
            processed_data['boxplot_data'][var_name] = []
            processed_data['statistics'][var_name] = {
                'mean': [],
                'median': [],
                'std': [],
                'skew': [],
                'kurt': []
            }
            
            # Extract residuals for this variable
            if var_idx < 4:  # First 4 variables are in 'data'
                var_residuals = data_dict['data'][:, var_idx].detach().cpu().numpy()
            elif var in data_dict:  # Check if physical residuals exist
                var_residuals = data_dict[var].detach().cpu().numpy()
            else:
                var_residuals = np.array([])
            
            processed_data['boxplot_data'][var_name].append(var_residuals)
            
            # Calculate statistics
            if len(var_residuals) > 0:
                processed_data['statistics'][var_name]['mean'].append(np.mean(var_residuals))
                processed_data['statistics'][var_name]['median'].append(np.median(var_residuals))
                processed_data['statistics'][var_name]['std'].append(np.std(var_residuals))
                processed_data['statistics'][var_name]['skew'].append(skew(var_residuals))
                processed_data['statistics'][var_name]['kurt'].append(kurtosis(var_residuals))
            else:
                # Add NaN for empty data
                for stat in ['mean', 'median', 'std', 'skew', 'kurt']:
                    processed_data['statistics'][var_name][stat].append(np.nan)
        
        # Save processed data
        filename = os.path.join(data_dir, f"{data_type}_residuals.npz")
        
        # Handle special characters in data_type names for the filename
        for char in ['\\', ':', '*', '?', '"', '<', '>', '|']:
            filename = filename.replace(char, '_')
        
        # Convert data to numpy arrays for saving
        data_to_save = {
            'data_type': np.array([data_type], dtype=np.str_),
            'speeds': np.array(speeds),
            'variables': np.array(variables, dtype=np.str_),
            'y_limits': np.array([[y_limits.get(var, {'min': -1, 'max': 1})['min'],
                                y_limits.get(var, {'min': -1, 'max': 1})['max']] for var in variables])
        }
        
        # Save boxplot data and statistics
        for var_idx, var_name in enumerate(variables):
            # Convert boxplot data to a list of values and indices
            values = []
            indices = []
            for i, data_array in enumerate(processed_data['boxplot_data'][var_name]):
                if len(data_array) > 0:
                    values.extend(data_array)
                    indices.extend([i] * len(data_array))
            
            data_to_save[f'{var_name}_boxplot_values'] = np.array(values)
            data_to_save[f'{var_name}_boxplot_indices'] = np.array(indices)
            
            # Save statistics for this variable
            for stat in ['mean', 'median', 'std', 'skew', 'kurt']:
                data_to_save[f'{var_name}_{stat}'] = np.array(processed_data['statistics'][var_name][stat])
        
        # Save the data
        np.savez(filename, **data_to_save)
        print(f"Saved processed data to {filename}")
        
        # Create plot for this data type
        fig, axes = plt.subplots(num_variables, 1, figsize=(15, 6 * num_variables), sharex=True)
        
        if num_variables == 1:
            axes = [axes]
        
        for var_idx, ax in enumerate(axes):
            # Extract data from processed data
            var = variables[var_idx]
            boxplot_data = processed_data['boxplot_data'][var]
            
            # Create violin and box plots
            positions = np.arange(1, num_speeds + 1)
            ax.violinplot(boxplot_data, positions=positions, showextrema=True, showmedians=True)
            ax.boxplot(boxplot_data, positions=positions, widths=0.3, patch_artist=True, boxprops=dict(facecolor="lightblue"))
            
            # Set y-axis limits based on global limits
            if var in y_limits:
                ax.set_ylim(y_limits[var]['min'], y_limits[var]['max'])
            
            # Annotate statistics
            if len(boxplot_data[0]) > 0:
                mean_val = processed_data['statistics'][var]['mean'][0]
                median_val = processed_data['statistics'][var]['median'][0]
                std_val = processed_data['statistics'][var]['std'][0]
                skew_val = processed_data['statistics'][var]['skew'][0]
                kurt_val = processed_data['statistics'][var]['kurt'][0]
                
                ax.text(
                    x=1, 
                    y=ax.get_ylim()[1] * 0.9, 
                    s=(f"Mean: {mean_val:.2f}\n"
                       f"Median: {median_val:.2f}\n"
                       f"Std: {std_val:.2f}\n"
                       f"Skew: {skew_val:.2f}\n"
                       f"Kurt: {kurt_val:.2f}"),
                    ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white")
                )
            
            ax.grid(True, which="both", linestyle="--", linewidth=0.5)
            ax.set_ylabel(f"Residuals for {variables[var_idx]}")
        
        # Set x-axis labels
        axes[-1].set_xlabel("Data Type")
        axes[-1].set_xticks(np.arange(1, num_speeds + 1))
        axes[-1].set_xticklabels([data_type], rotation=45, ha='right')
        
        plt.suptitle(f"Residuals for {data_type} data using Direct PINN")
        plt.tight_layout()
        
        plot_path = os.path.join(plots_dir, f"{data_type}_residuals.png")
        plt.savefig(plot_path)
        plt.close()
        
        print(f"Plot for {data_type} saved to {plot_path}") 