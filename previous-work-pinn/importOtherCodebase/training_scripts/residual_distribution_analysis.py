#!/usr/bin/env python3
"""
Residual Distribution Analysis for PINN and Data-Driven Models

This script analyzes residual properties across different operating conditions
and generates wide-format LaTeX tables showing statistical metrics (mean, std, skewness, kurtosis)
as percentage changes relative to normal condition.

Also generates a comprehensive violin plot grid showing the distribution of healthy 
(normal condition) residuals for all models in a compact format suitable for publication.

Uses Bayesian optimization models for consistency with original optimization results.

Generates tables and plots for:
- Multiple PINN models (relobralo, constant_weight, brdr, pecann, etc.)
- Data-driven models: data_standard, data_reg
- Wide-format tables: 2 per model (data residuals + physical residuals)
- Each table shows 4 residuals × 4 metrics across multiple fault conditions
- 4 comprehensive violin plots: 5×4 grids (2 model groups × 2 residual types)
- Ranking tables for MAE and STD of healthy residuals

Usage:
    # Run all models from residuals directory (full analysis)
    python residual_distribution_analysis.py --residuals-dir best_model_residuals/ --use-saved-residuals

    # Generate only violin plots (fast - skips table generation)
    python residual_distribution_analysis.py --residuals-dir best_model_residuals/ --use-saved-residuals --violin-only
    
    # Run specific model
    python residual_distribution_analysis.py --model data_standard --use-saved-residuals
    
    # Test mode (first 2 files only)
    python residual_distribution_analysis.py --residuals-dir best_model_residuals/ --test-mode
"""

import torch
import numpy as np
import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Configure matplotlib for headless operation
import matplotlib
matplotlib.use('Agg')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training_scripts.pinn_preprocessing import (
    preprocess_pinn_data,
    load_pinn_data
)


def get_condition_mapping() -> Dict[str, str]:
    """
    Map data file names to condition names used in the analysis.

    Returns:
        Dictionary mapping condition keys to data file names (without .pth extension)
    """
    return {
        'normal': 'X_normal_v3',
        'horizontal_misalignment': 'X_horizontal_misalignment_fault_1.0mm_v3',  # Using 1.0mm as representative
        'imbalance': 'X_imbalance_fault_35g_v3',  # Using 35g as representative
        'overhang_ball': 'X_overhang_ball_fault_35g_v3',  # Using 35g as representative
        'overhang_cage': 'X_overhang_cage_fault_35g_v3',  # Using 35g as representative
        'overhang_outer_race': 'X_overhang_outer_race_fault_35g_v3',  # Using 35g as representative
        'underhang_ball': 'X_underhang_ball_fault_35g_v3',  # Using 35g as representative
        'underhang_cage': 'X_underhang_cage_fault_35g_v3',  # Using 35g as representative
        'underhang_outer_race': 'X_underhang_outer_race_fault_35g_v3',  # Using 35g as representative
        'vertical_misalignment': 'X_vertical_misalignment_fault_1.27mm_v3',  # Using 1.27mm as representative
    }


def get_condition_display_names() -> Dict[str, str]:
    """
    Get display names for conditions in the LaTeX tables.

    Returns:
        Dictionary mapping condition keys to display names
    """
    return {
        'normal': 'Normal',
        'horizontal_misalignment': 'Horizontal M.',
        'imbalance': 'Imbalance',
        'overhang_ball': 'Overhang Ball',
        'overhang_cage': 'Overhang Cage',
        'overhang_outer_race': 'Overhang O. R.',
        'underhang_ball': 'Underhang Ball',
        'underhang_cage': 'Underhang Cage',
        'underhang_outer_race': 'Underhang O. R.',
        'vertical_misalignment': 'Vertical M.',
    }


def calculate_residual_statistics(residual_data: np.ndarray) -> Dict[str, float]:
    """
    Calculate statistical properties of residual data.

    Args:
        residual_data: 1D numpy array of residual values

    Returns:
        Dictionary containing mean, std, skewness, and kurtosis
    """
    # Remove any NaN or infinite values
    clean_data = residual_data[np.isfinite(residual_data)]

    if len(clean_data) == 0:
        return {
            'mean': 0.0,
            'std': 0.0,
            'skewness': 0.0,
            'kurtosis': 0.0
        }

    return {
        'mean': float(np.mean(clean_data)),
        'std': float(np.std(clean_data)),
        'skewness': float(stats.skew(clean_data)),
        'kurtosis': float(stats.kurtosis(clean_data))
    }


def calculate_percentage_change(baseline_value: float, condition_value: float) -> float:
    """
    Calculate percentage change relative to baseline.

    Args:
        baseline_value: Baseline value (normal condition)
        condition_value: Condition value

    Returns:
        Percentage change as float
    """
    if baseline_value == 0:
        return 0.0
    return ((condition_value - baseline_value) / abs(baseline_value)) * 100.0


def format_statistical_value(value: float, is_percentage: bool = False) -> str:
    """
    Format statistical values for LaTeX table.

    Args:
        value: Numerical value
        is_percentage: Whether to add % symbol

    Returns:
        Formatted string
    """
    if abs(value) < 0.01:
        format_spec = ".2e"
    else:
        format_spec = ".1f"

    formatted_value = f"{value:{format_spec}}"

    if is_percentage:
        return f"{formatted_value}\\%"
    return formatted_value


def generate_wide_format_table(model_name: str, residual_type: str, all_statistics: Dict[str, Dict[str, Dict[str, float]]]) -> str:
    """
    Generate wide-format LaTeX table for a specific model and residual type.
    
    Args:
        model_name: Name of the model
        residual_type: Either 'data' or 'phys'
        all_statistics: Dictionary with condition -> residual -> statistics mapping
    
    Returns:
        LaTeX table as string
    """
    display_names = get_condition_display_names()
    
    # Determine which residuals to include
    if residual_type == 'data':
        residual_keys = ['data_res1', 'data_res2', 'data_res3', 'data_res4']
        residual_labels = ['Data Res1', 'Data Res2', 'Data Res3', 'Data Res4']
        table_caption = f"{model_name.replace('_', ' ').title()} Model Residual Distribution Metrics for Data Residuals: Percentage Change Relative to Normal Condition"
        table_label = f"tab:residual_distribution_data_{model_name}"
    else:  # phys
        residual_keys = ['phys_res1', 'phys_res2', 'phys_res3', 'phys_res4']
        residual_labels = ['Phys Res1', 'Phys Res2', 'Phys Res3', 'Phys Res4']
        table_caption = f"{model_name.replace('_', ' ').title()} Model Residual Distribution Metrics for Physical Residuals: Percentage Change Relative to Normal Condition"
        table_label = f"tab:residual_distribution_phys_{model_name}"
    
    # Get normal (baseline) statistics
    normal_stats = all_statistics.get('normal', {})
    
    # Start LaTeX table
    latex_table = r"""\begin{table*}
    \centering
    % Caption and label for the """ + residual_type.title() + r""" Residuals table
    \caption{""" + table_caption + r"""}
    \label{""" + table_label + r"""}
    
    % Resize the table to fit the line width
    \resizebox{\linewidth}{!}{%
    \begin{tabular}{@{\extracolsep{\fill}}l *{4}{rrrr}}
        \toprule
        % --- Main Headers for each dataset, spanning 4 columns each ---
        & \multicolumn{4}{c}{\textbf{""" + residual_labels[0] + r"""}} 
        & \multicolumn{4}{c}{\textbf{""" + residual_labels[1] + r"""}} 
        & \multicolumn{4}{c}{\textbf{""" + residual_labels[2] + r"""}} 
        & \multicolumn{4}{c}{\textbf{""" + residual_labels[3] + r"""}} \\
        
        % --- Rules under the main headers to group the sub-headers ---
        \cmidrule(lr){2-5} \cmidrule(lr){6-9} \cmidrule(lr){10-13} \cmidrule(lr){14-17} 
        
        % --- Sub-headers (metrics), repeated for each dataset ---
        \textbf{Condition} 
        & \textbf{$\mu$} & \textbf{$\sigma$} & \textbf{Skew.} & \textbf{Kurt.} 
        & \textbf{$\mu$} & \textbf{$\sigma$} & \textbf{Skew.} & \textbf{Kurt.} 
        & \textbf{$\mu$} & \textbf{$\sigma$} & \textbf{Skew.} & \textbf{Kurt.} 
        & \textbf{$\mu$} & \textbf{$\sigma$} & \textbf{Skew.} & \textbf{Kurt.} \\
        \midrule
        
"""
    
    # Add rows for each condition (excluding normal)
    conditions_order = [
        'horizontal_misalignment', 'imbalance', 'overhang_ball', 'overhang_cage',
        'overhang_outer_race', 'underhang_ball', 'underhang_cage',
        'underhang_outer_race', 'vertical_misalignment'
    ]
    
    for condition in conditions_order:
        if condition not in all_statistics:
            continue
        
        condition_stats = all_statistics[condition]
        condition_display = display_names[condition]
        
        # Build row
        row_values = [condition_display]
        
        for residual_key in residual_keys:
            if residual_key not in normal_stats or residual_key not in condition_stats:
                # Add placeholders if data is missing
                row_values.extend(['--', '--', '--', '--'])
                continue
            
            # Get baseline and condition statistics
            baseline = normal_stats[residual_key]
            current = condition_stats[residual_key]
            
            # Calculate percentage changes
            mean_change = calculate_percentage_change(baseline['mean'], current['mean'])
            std_change = calculate_percentage_change(baseline['std'], current['std'])
            skew_change = calculate_percentage_change(baseline['skewness'], current['skewness'])
            kurt_change = calculate_percentage_change(baseline['kurtosis'], current['kurtosis'])
            
            # Format values (no % sign, just the number)
            row_values.extend([
                f"{mean_change:.1f}",
                f"{std_change:.1f}",
                f"{skew_change:.1f}",
                f"{kurt_change:.1f}"
            ])
        
        # Add row to table
        latex_table += "        " + " & ".join(row_values) + r" \\" + "\n"
    
    # Close table
    latex_table += r"""        \bottomrule
    \end{tabular}
    }
\end{table*}
"""
    
    return latex_table


def load_saved_residuals(residuals_path: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Load residuals from saved pinn_to_siamese_wrapper.py format and calculate statistics.

    Args:
        residuals_path: Path to the saved residuals file

    Returns:
        Dictionary mapping condition names to residual statistics (nested dict structure)
    """
    print(f"   Loading saved residuals from: {residuals_path}")
    saved_data = torch.load(residuals_path)

    # Convert back to the flat residual format and calculate statistics
    condition_statistics = {}
    residual_names = ['data_res1', 'data_res2', 'data_res3', 'data_res4',
                     'phys_res1', 'phys_res2', 'phys_res3', 'phys_res4']

    for data_type, samples in saved_data.items():
        if not samples:  # Skip empty data types
            continue

        # Use the first sample (they should all have the same structure)
        sample_data = samples[0]['data'].numpy()  # Shape: [n_samples, 8]

        # Split the 8 features back into individual residual arrays
        residuals = {
            'data_res1': sample_data[:, 0],
            'data_res2': sample_data[:, 1],
            'data_res3': sample_data[:, 2],
            'data_res4': sample_data[:, 3],
            'phys_res1': sample_data[:, 4],
            'phys_res2': sample_data[:, 5],
            'phys_res3': sample_data[:, 6],
            'phys_res4': sample_data[:, 7]
        }

        # Calculate statistics for each residual
        residual_stats = {}
        for residual_name in residual_names:
            if residual_name in residuals:
                residual_stats[residual_name] = calculate_residual_statistics(residuals[residual_name])

        # Map data_type back to condition_key using pattern matching
        condition_key = None

        # Handle special case for normal
        if data_type == 'normal':
            condition_key = 'normal'
        elif 'horizontal_misalignment' in data_type:
            condition_key = 'horizontal_misalignment'
        elif 'vertical_misalignment' in data_type:
            condition_key = 'vertical_misalignment'
        elif 'imbalance' in data_type:
            condition_key = 'imbalance'
        elif 'overhang_ball' in data_type:
            condition_key = 'overhang_ball'
        elif 'overhang_cage' in data_type:
            condition_key = 'overhang_cage'
        elif 'overhang_outer_race' in data_type:
            condition_key = 'overhang_outer_race'
        elif 'underhang_ball' in data_type:
            condition_key = 'underhang_ball'
        elif 'underhang_cage' in data_type:
            condition_key = 'underhang_cage'
        elif 'underhang_outer_race' in data_type:
            condition_key = 'underhang_outer_race'

        if condition_key:
            # Only keep one sample per condition (use the first one encountered, or a representative one)
            # For fault conditions, prefer the "standard" severity if available
            if condition_key not in condition_statistics:
                condition_statistics[condition_key] = residual_stats
                print(f"     Loaded {data_type} -> {condition_key}")
            else:
                # If we already have this condition, only replace if this is a "standard" severity
                current_data_type = None
                for dt in saved_data.keys():
                    if condition_key in str(dt).lower() or dt == 'normal':
                        # Check if this might be the current one
                        current_data_type = dt
                        break

                # Prefer certain severities as "representative"
                standard_severities = ['35g', '1.0mm', '1.27mm']  # Common representative values
                is_current_standard = any(sev in str(current_data_type or '') for sev in standard_severities)
                is_new_standard = any(sev in data_type for sev in standard_severities)

                if is_new_standard and not is_current_standard:
                    condition_statistics[condition_key] = residual_stats
                    print(f"     Replaced {current_data_type} with {data_type} -> {condition_key} (more representative)")
        else:
            print(f"     WARNING: Could not map {data_type} to condition key")

    return condition_statistics


def process_residual_analysis(model_name: str, chunk_size_mb: int = 100, max_points: int = 3000,
                            use_saved_residuals: bool = False, residuals_path: str = None) -> Tuple[Dict[str, str], Dict[str, np.ndarray]]:
    """
    Process residual analysis for a specific model.

    Args:
        model_name: Name of the model ('relobralo', 'constant_weight', 'brdr', 'pecann', 'data_standard', 'data_reg')
        chunk_size_mb: Chunk size for memory-efficient processing
        max_points: Maximum number of data points to use from each time series (default: 3000)
        use_saved_residuals: If True, load from saved residuals file instead of recomputing
        residuals_path: Path to saved residuals file (only used if use_saved_residuals=True)

    Returns:
        Tuple of (latex_tables, normal_residuals) where:
        - latex_tables: Dictionary mapping residual names to LaTeX tables
        - normal_residuals: Dictionary with residual data for normal condition
    """
    if model_name in ['data_standard', 'data_reg']:
        print(f"\nProcessing data-driven model: {model_name}")
    else:
        print(f"\nProcessing PINN model: {model_name}")

    # Get condition mappings
    condition_mapping = get_condition_mapping()
    data_dir = "Data/v3"

    # Initialize results storage
    all_statistics = {}
    normal_residuals = {}  # Store normal condition residuals for violin plots
    residual_names = ['data_res1', 'data_res2', 'data_res3', 'data_res4',
                     'phys_res1', 'phys_res2', 'phys_res3', 'phys_res4']

    if use_saved_residuals:
        # Load from saved residuals
        if not residuals_path:
            # Auto-detect path based on model name
            residuals_path = f"best_model_residuals/{model_name}_residuals.pth"
            if not os.path.exists(residuals_path):
                # Try alternative naming
                residuals_path = f"best_model_residuals/{model_name.replace('_', '')}_residuals.pth"
            if not os.path.exists(residuals_path):
                # Try data-driven naming
                if model_name == 'data_standard':
                    residuals_path = "best_model_residuals/residuals_data_driven_standard.pth"
                elif model_name == 'data_reg':
                    residuals_path = "best_model_residuals/residuals_data_driven_reg.pth"

        if not os.path.exists(residuals_path):
            print(f"   ERROR: Saved residuals file not found: {residuals_path}")
            print("   Falling back to recomputing residuals...")
            use_saved_residuals = False
        else:
            print(f"   Using saved residuals: {residuals_path}")
            all_statistics = load_saved_residuals(residuals_path)
            
            # Also extract normal condition residuals for violin plots
            saved_data = torch.load(residuals_path)
            for data_type, samples in saved_data.items():
                if data_type == 'normal' and samples:
                    sample_data = samples[0]['data'].numpy()
                    normal_residuals = {
                        'data_res1': sample_data[:, 0],
                        'data_res2': sample_data[:, 1],
                        'data_res3': sample_data[:, 2],
                        'data_res4': sample_data[:, 3],
                        'phys_res1': sample_data[:, 4],
                        'phys_res2': sample_data[:, 5],
                        'phys_res3': sample_data[:, 6],
                        'phys_res4': sample_data[:, 7]
                    }
                    break
    else:
        # Original processing logic
        pass

    # Process each condition (only if not using saved residuals)
    if not use_saved_residuals or not all_statistics:
        for condition_key, data_file in condition_mapping.items():
            data_path = os.path.join(data_dir, f"{data_file}.pth")

            if not os.path.exists(data_path):
                print(f"   WARNING: Data file not found: {data_path}")
                continue

            print(f"   Processing condition: {condition_key}")

            try:
                # Preprocess data to get residuals
                if model_name in ['data_standard', 'data_reg']:
                    print(f"      Extracting residuals from data-driven model {model_name}...")
                    residuals = preprocess_pinn_data(model_name, data_path, chunk_size_mb=chunk_size_mb)
                else:
                    print(f"      Extracting residuals from PINN model {model_name}...")
                    residuals = preprocess_pinn_data(model_name, data_path, chunk_size_mb=chunk_size_mb, use_bayesian_models=True)

                # Debug: Print residual statistics
                print("      Residual Statistics:")
                for residual_name in residual_names:
                    if residual_name in residuals:
                        res_data = residuals[residual_name]
                        print(f"         {residual_name}: mean={np.mean(res_data):.2e}, std={np.std(res_data):.2e}, "
                              f"min={np.min(res_data):.2e}, max={np.max(res_data):.2e}")

                # Check for suspiciously large residuals
                for residual_name in residual_names:
                    if residual_name in residuals:
                        res_data = residuals[residual_name]
                        if np.max(np.abs(res_data)) > 1e6:
                            print(f"      WARNING: Extremely large residuals detected in {residual_name}")
                            print(f"         Max absolute residual: {np.max(np.abs(res_data)):.2e}")
                            print(f"         This may indicate scaling or model issues")

                # Limit to first max_points data points for faster processing (default: 3000)
                for residual_name in residual_names:
                    if residual_name in residuals:
                        res_data = residuals[residual_name]
                        if hasattr(res_data, '__len__') and len(res_data) > max_points:
                            try:
                                residuals[residual_name] = res_data[:max_points]
                            except (TypeError, IndexError) as e:
                                print(f"      WARNING: Could not slice {residual_name}, using full data: {e}")

                # Calculate statistics for each residual
                condition_statistics = {}
                for residual_name in residual_names:
                    if residual_name in residuals:
                        stats_result = calculate_residual_statistics(residuals[residual_name])
                        condition_statistics[residual_name] = stats_result
                    else:
                        print(f"      WARNING: Residual {residual_name} not found in model {model_name}")

                all_statistics[condition_key] = condition_statistics

                # Store normal condition residuals for violin plots
                if condition_key == 'normal':
                    normal_residuals = residuals.copy()

            except Exception as e:
                print(f"      ERROR: Failed to process {condition_key}: {e}")
                continue

    # Generate wide-format LaTeX tables (2 per model: data and phys)
    latex_tables = {}
    
    # Generate data residuals table
    if all_statistics:
        data_table = generate_wide_format_table(model_name, 'data', all_statistics)
        latex_tables['data_residuals'] = data_table
        print(f"   Generated wide-format table for data residuals")
        
        # Generate physical residuals table (only for PINN models)
        if model_name not in ['data_reg', 'data_standard']:
            phys_table = generate_wide_format_table(model_name, 'phys', all_statistics)
            latex_tables['phys_residuals'] = phys_table
            print(f"   Generated wide-format table for physical residuals")

    return latex_tables, normal_residuals


def get_model_abbreviations() -> Dict[str, str]:
    """
    Get abbreviated names for models to save space in plots.
    
    Returns:
        Dictionary mapping full model names to abbreviations
    """
    return {
        'adaptive_lbpin': 'A-LBPIN',
        'alpinn': 'ALPinn',
        'brdr': 'BRDR',
        'constant_weight': 'ConstW',
        'dwpinn': 'DWPINN',
        'gradnorm': 'GradN',
        'pecann': 'PECANN',
        'relobralo': 'Relobr',
        'data_reg': 'Data-R',
        'data_standard': 'Data-S'
    }


def create_comprehensive_violin_plots(all_normal_residuals: Dict[str, Dict[str, np.ndarray]], output_dir: str):
    """
    Create comprehensive violin plot grids showing all models and residuals.
    Generates 4 separate plots: 2 groups of models × 2 residual types (data/physical).
    
    Args:
        all_normal_residuals: Dictionary mapping model names to normal residual data
        output_dir: Output directory for plots
    """
    # Set up the plot style
    plt.style.use('default')
    
    # Font sizes - ALL text should be 28
    plt.rcParams.update({
        'font.size': 28,
        'axes.titlesize': 28,
        'axes.labelsize': 28,
        'xtick.labelsize': 28,
        'ytick.labelsize': 28,
        'legend.fontsize': 28
    })
    
    # Organize models: PINN models first (alphabetically), then data-driven
    # Exclude *_old models
    pinn_models = sorted([m for m in all_normal_residuals.keys() 
                         if m not in ['data_reg', 'data_standard'] and not m.endswith('_old')])
    data_models = sorted([m for m in all_normal_residuals.keys() 
                         if m in ['data_reg', 'data_standard']])
    all_models = pinn_models + data_models
    
    # Get model abbreviations
    model_abbrev = get_model_abbreviations()
    
    # Split models into two groups of 5
    group1_models = all_models[:5]
    group2_models = all_models[5:]
    
    # Define residual types
    data_residuals = ['data_res1', 'data_res2', 'data_res3', 'data_res4']
    phys_residuals = ['phys_res1', 'phys_res2', 'phys_res3', 'phys_res4']
    
    data_headers = [r'$\ddot{x}_2$', r'$\ddot{y}_2$', r'$\ddot{x}_3$', r'$\ddot{y}_3$']
    phys_headers = [r'$f_A$', r'$f_B$', r'$f_C$', r'$f_D$']
    
    # Generate 4 plots
    plot_configs = [
        (group1_models, data_residuals, data_headers, 'data', 'group1'),
        (group1_models, phys_residuals, phys_headers, 'phys', 'group1'),
        (group2_models, data_residuals, data_headers, 'data', 'group2'),
        (group2_models, phys_residuals, phys_headers, 'phys', 'group2'),
    ]
    
    for models, residual_keys, col_headers, res_type, group in plot_configs:
        # Create 5×4 grid
        fig, axes = plt.subplots(5, 4, figsize=(16, 20))
        
        # Plot each model (row)
        for row_idx, model_name in enumerate(models):
            residuals = all_normal_residuals[model_name]
            is_data_driven = model_name in data_models
            
            # Skip physical residuals for data-driven models
            if is_data_driven and res_type == 'phys':
                for col_idx in range(4):
                    axes[row_idx, col_idx].axis('off')
                continue
            
            # Plot each residual (column)
            for col_idx, residual_key in enumerate(residual_keys):
                ax = axes[row_idx, col_idx]
                
                # Plot if data exists
                if residual_key in residuals:
                    data = residuals[residual_key]
                    
                    # Create violin plot
                    parts = ax.violinplot([data], positions=[0], showmeans=False, 
                                         showextrema=False, showmedians=True, widths=0.7)
                    
                    # Style the violin plot
                    for pc in parts['bodies']:
                        pc.set_facecolor('#8dd3c7')
                        pc.set_alpha(0.7)
                    
                    # Add median line styling
                    parts['cmedians'].set_color('red')
                    parts['cmedians'].set_linewidth(3)
                    
                    # Add quartile box plot overlay (minimal)
                    bp = ax.boxplot([data], positions=[0], widths=0.15, 
                                   patch_artist=True, showfliers=False,
                                   boxprops=dict(facecolor='white', alpha=0.5, linewidth=1.5),
                                   medianprops=dict(color='red', linewidth=2),
                                   whiskerprops=dict(linewidth=1.5),
                                   capprops=dict(linewidth=1.5))
                else:
                    # No data available
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=28)
                    ax.set_xlim(-1, 1)
                
                # Hide x-axis (same for all)
                ax.set_xticks([])
                ax.set_xlabel('')
                
                # Y-axis label only for first column
                if col_idx == 0:
                    abbrev = model_abbrev.get(model_name, model_name)
                    ax.set_ylabel(abbrev, fontsize=28, fontweight='bold')
                else:
                    ax.set_ylabel('')
                
                # Column headers only for first row
                if row_idx == 0:
                    ax.set_title(col_headers[col_idx], fontsize=28, fontweight='bold', pad=15)
                
                # Adjust y-axis tick labels
                ax.tick_params(axis='y', labelsize=28, width=1.5, length=6)
                
                # Grid for readability
                ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=1)
                ax.set_axisbelow(True)
                
                # Make spine thicker
                for spine in ax.spines.values():
                    spine.set_linewidth(1.5)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"healthy_residuals_violin_{group}_{res_type}.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Saved violin plot: {plot_filename}")
    
    print(f"   Generated 4 comprehensive violin plots (5×4 grids)")


def generate_ranking_tables(all_normal_residuals: Dict[str, Dict[str, np.ndarray]], output_dir: str):
    """
    Generate LaTeX tables with rankings for MAE and STD of healthy residuals.
    
    Args:
        all_normal_residuals: Dictionary mapping model names to normal residual data
        output_dir: Output directory for LaTeX files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Residual display names
    residual_display = {
        'data_res1': r'$\ddot{x}_2$',
        'data_res2': r'$\ddot{y}_2$',
        'data_res3': r'$\ddot{x}_3$',
        'data_res4': r'$\ddot{y}_3$',
        'phys_res1': r'$f_A$',
        'phys_res2': r'$f_B$',
        'phys_res3': r'$f_C$',
        'phys_res4': r'$f_D$',
    }
    
    residual_names = ['data_res1', 'data_res2', 'data_res3', 'data_res4',
                     'phys_res1', 'phys_res2', 'phys_res3', 'phys_res4']
    
    # Calculate MAE and STD for each model
    mae_values = {}
    std_values = {}
    
    for model_name, residuals in all_normal_residuals.items():
        mae_values[model_name] = {}
        std_values[model_name] = {}
        
        for res_name in residual_names:
            if res_name in residuals:
                data = residuals[res_name]
                clean_data = data[np.isfinite(data)]
                if len(clean_data) > 0:
                    mae_values[model_name][res_name] = np.mean(np.abs(clean_data))
                    std_values[model_name][res_name] = np.std(clean_data)
                else:
                    mae_values[model_name][res_name] = 0.0
                    std_values[model_name][res_name] = 0.0
    
    def rank_values(values_dict, residual_name):
        """Rank models by value for a given residual (lower is better)."""
        # Extract values for this residual
        model_values = [(model, values_dict[model].get(residual_name, float('inf'))) 
                       for model in values_dict.keys() 
                       if residual_name in values_dict[model]]
        
        # Sort by value
        model_values.sort(key=lambda x: x[1])
        
        # Assign ranks (same rank for equal values, always 3 ranks)
        ranks = {}
        if len(model_values) >= 3:
            # Get unique top 3 values
            unique_vals = sorted(set([v for _, v in model_values]))[:3]
            
            for model, val in model_values:
                if val == unique_vals[0]:
                    ranks[model] = 'first'
                elif len(unique_vals) > 1 and val == unique_vals[1]:
                    ranks[model] = 'second'
                elif len(unique_vals) > 2 and val == unique_vals[2]:
                    ranks[model] = 'third'
                else:
                    ranks[model] = None
        
        return ranks
    
    def format_value_with_rank(value, rank):
        """Format value with rank markup."""
        if rank == 'first':
            return f"\\first{{{value:.4f}}}"
        elif rank == 'second':
            return f"\\second{{{value:.4f}}}"
        elif rank == 'third':
            return f"\\third{{{value:.4f}}}"
        else:
            return f"{value:.4f}"
    
    # Generate MAE table
    mae_table = r"""\begin{table*}[t!]
  \centering
  \caption{Mean Absolute Error (MAE) for Healthy Residuals — methods ranked by magnitude per column.}
  \label{tab:healthy_mae}
  \sisetup{round-mode = places, round-precision = 4, detect-weight=true, detect-family=true}
  \resizebox{\textwidth}{!}{%
  \begin{tabular}{l""" + " S" * len(residual_names) + r"""}
    \toprule
    \textbf{Method}"""
    
    for res_name in residual_names:
        mae_table += f" & {{{residual_display[res_name]}}}"
    
    mae_table += r""" \\
    \midrule
"""
    
    # Add rows for each model
    for model_name in sorted(mae_values.keys()):
        mae_table += f"    {model_name.replace('_', '\\_')}"
        
        for res_name in residual_names:
            ranks = rank_values(mae_values, res_name)
            value = mae_values[model_name].get(res_name, 0.0)
            rank = ranks.get(model_name, None)
            mae_table += f" & {format_value_with_rank(value, rank)}"
        
        mae_table += r""" \\
"""
    
    mae_table += r"""    \bottomrule
  \end{tabular}
  }
\end{table*}
"""
    
    # Generate STD table
    std_table = r"""\begin{table*}[t!]
  \centering
  \caption{Standard Deviation for Healthy Residuals — ranked by magnitude per column.}
  \label{tab:healthy_std}
  \sisetup{round-mode = places, round-precision = 4, detect-weight=true, detect-family=true}
  \resizebox{\textwidth}{!}{%
  \begin{tabular}{l""" + " S" * len(residual_names) + r"""}
    \toprule
    \textbf{Method}"""
    
    for res_name in residual_names:
        std_table += f" & {{{residual_display[res_name]}}}"
    
    std_table += r""" \\
    \midrule
"""
    
    # Add rows for each model
    for model_name in sorted(std_values.keys()):
        std_table += f"    {model_name.replace('_', '\\_')}"
        
        for res_name in residual_names:
            ranks = rank_values(std_values, res_name)
            value = std_values[model_name].get(res_name, 0.0)
            rank = ranks.get(model_name, None)
            std_table += f" & {format_value_with_rank(value, rank)}"
        
        std_table += r""" \\
"""
    
    std_table += r"""    \bottomrule
  \end{tabular}
  }
\end{table*}
"""
    
    # Save tables to file
    output_file = os.path.join(output_dir, "healthy_residuals_ranking_tables.tex")
    with open(output_file, 'w') as f:
        f.write("% Ranking Tables for Healthy Residuals (MAE and STD)\n")
        f.write("% Generated automatically by residual_distribution_analysis.py\n")
        f.write("% Requires \\usepackage{siunitx} and \\usepackage{booktabs}\n")
        f.write("% Define ranking commands: \\newcommand{\\first}[1]{\\textbf{#1}}\n")
        f.write("% \\newcommand{\\second}[1]{\\underline{#1}}\n")
        f.write("% \\newcommand{\\third}[1]{\\textit{#1}}\n\n")
        f.write(mae_table)
        f.write("\n\n")
        f.write(std_table)
    
    print(f"   Saved ranking tables to {output_file}")


def save_latex_tables(tables_dict: Dict[str, Dict[str, str]], output_dir: str = "."):
    """
    Save LaTeX tables to files.

    Args:
        tables_dict: Dictionary mapping model names to residual tables
        output_dir: Output directory for LaTeX files
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save individual model files
    for model_name, model_tables in tables_dict.items():
        output_file = os.path.join(output_dir, f"residual_analysis_{model_name}.tex")

        with open(output_file, 'w') as f:
            f.write(f"% Residual Distribution Analysis for {model_name.replace('_', ' ').title()} Model\n")
            f.write("% Generated automatically by residual_distribution_analysis.py\n")
            f.write("% Requires \\usepackage{booktabs} for table formatting\n\n")

            for table_type, latex_table in model_tables.items():
                f.write(latex_table)
                f.write("\n")

        print(f"   Saved LaTeX tables for {model_name} to {output_file}")

    # Save combined file with all tables
    combined_file = os.path.join(output_dir, "residual_analysis_all_models.tex")
    with open(combined_file, 'w') as f:
        f.write("% Complete Residual Distribution Analysis for All Models\n")
        f.write("% Generated automatically by residual_distribution_analysis.py\n")
        f.write("% Requires \\usepackage{booktabs} for table formatting\n\n")

        for model_name, model_tables in tables_dict.items():
            f.write(f"\\section{{{model_name.replace('_', ' ').title()} Model Residual Analysis}}\n\n")

            for table_type, latex_table in model_tables.items():
                f.write(latex_table)
                f.write("\n")

    print(f"   Saved combined LaTeX file to {combined_file}")


def quick_model_test():
    """Quick test to check if models can make reasonable predictions."""
    print("QUICK MODEL TEST")
    print("-" * 30)

    models = ['relobralo', 'constant_weight', 'brdr', 'pecann']
    data_dir = "Data/v3"

    for model_name in models:
        try:
            print(f"\nTesting {model_name}...")

            # Try to load the model
            from pinn_preprocessing import load_best_model
            device = torch.device('cpu')
            model, scaling_params, config = load_best_model(model_name, device=device)

            # Create sample data
            sample_X = torch.randn(5, 10, dtype=torch.float64)  # 5 samples, 10 features

            # Test prediction
            X_norm = (sample_X - scaling_params['Xmin']) / (scaling_params['Xmax'] - scaling_params['Xmin'] + 1e-12)
            model.eval()
            with torch.no_grad():
                y_pred_norm = model(X_norm)
                y_pred = y_pred_norm * (scaling_params['ymax'] - scaling_params['ymin']) + scaling_params['ymin']

            print(".2e")
            print(".2e")

            if torch.any(torch.isinf(y_pred)) or torch.any(torch.isnan(y_pred)):
                print(f"   [FAIL] {model_name}: Invalid predictions (inf/nan)")
            elif torch.max(torch.abs(y_pred)) > 1e10:
                print(f"   [WARN] {model_name}: Extremely large predictions")
            else:
                print(f"   [OK] {model_name}: Predictions look reasonable")

        except Exception as e:
            print(f"   [ERROR] {model_name}: Failed to test - {e}")

    print("-" * 30)


def validate_model_data_compatibility():
    """Validate that models and data are compatible."""
    print("VALIDATING MODEL-DATA COMPATIBILITY")
    print("-" * 50)

    # Check if models exist
    models_dir = "results/best_models"
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    models = ['relobralo', 'constant_weight', 'brdr', 'pecann']
    data_dir = "Data/v3"
    conditions = get_condition_mapping()

    for model_name in models:
        model_path = os.path.join(models_dir, model_name, f"{model_name}_config.json")
        if not os.path.exists(model_path):
            print(f"WARNING: Model config not found: {model_path}")
            continue

        print(f"Found model: {model_name}")

        # Check if required data files exist
        missing_data = []
        for condition_key, data_file in conditions.items():
            data_path = os.path.join(data_dir, f"{data_file}.pth")
            if not os.path.exists(data_path):
                missing_data.append(f"{condition_key}: {data_file}.pth")

        if missing_data:
            print(f"   WARNING: Missing data files for {model_name}:")
            for missing in missing_data[:3]:  # Show first 3
                print(f"      - {missing}")
            if len(missing_data) > 3:
                print(f"      ... and {len(missing_data) - 3} more")
        else:
            print("   All required data files found")
    print("-" * 50)


def main():
    """Main function to run the complete residual distribution analysis.

    Uses only the first 3000 data points from each time series for faster processing
    while maintaining statistical significance.
    """
    parser = argparse.ArgumentParser(description="Generate LaTeX tables for residual distribution analysis")

    parser.add_argument("--model", type=str,
                       choices=['relobralo', 'constant_weight', 'brdr', 'pecann', 'data_standard', 'data_reg'],
                       help="Model name (optional - if not specified, runs all models). PINN: relobralo, constant_weight, brdr, pecann; Data-driven: data_standard, data_reg")
    parser.add_argument("--chunk-size-mb", type=int, default=100,
                       help="Chunk size for memory-efficient processing")
    parser.add_argument("--max-points", type=int, default=3000,
                       help="Maximum number of data points to use from each time series")
    parser.add_argument("--skip-validation", action="store_true",
                       help="Skip model-data compatibility validation")
    parser.add_argument("--use-saved-residuals", action="store_true",
                       help="Use saved residuals from best_model_residuals/ instead of recomputing")
    parser.add_argument("--residuals-path", type=str,
                       help="Path to saved residuals file (auto-detected if not specified)")
    parser.add_argument("--residuals-dir", type=str,
                       help="Directory containing all residual files to process (e.g., best_model_residuals/)")
    parser.add_argument("--output-dir", type=str, default="residual_distribution_analysis_results",
                       help="Output directory for results")
    parser.add_argument("--test-mode", action="store_true",
                       help="Test mode: process only first 2 residual files to verify pipeline")
    parser.add_argument("--violin-only", action="store_true",
                       help="Generate only violin plots (skip table generation for faster processing)")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    # Configuration: Determine which models to process
    models_to_process = []
    
    if args.residuals_dir:
        # Batch mode: Process all residual files in directory
        print(f"\nBATCH MODE: Processing all residual files in {args.residuals_dir}")
        
        if not os.path.exists(args.residuals_dir):
            print(f"ERROR: Residuals directory not found: {args.residuals_dir}")
            sys.exit(1)
        
        # Find all residual files (various naming patterns)
        import glob
        residual_files = glob.glob(os.path.join(args.residuals_dir, "*residuals*.pth"))
        
        if not residual_files:
            print(f"ERROR: No *residuals*.pth files found in {args.residuals_dir}")
            sys.exit(1)
        
        print(f"Found {len(residual_files)} residual files:")
        for rf in residual_files:
            print(f"  - {os.path.basename(rf)}")
        
        # Test mode: only process first 2 files
        if args.test_mode:
            residual_files = residual_files[:2]
            print(f"\nTEST MODE: Processing only first {len(residual_files)} files")
        
        # Extract model names from filenames
        for rf in residual_files:
            basename = os.path.basename(rf)
            # Handle various naming patterns:
            # - modelname_residuals.pth -> modelname
            # - modelname_residuals_old.pth -> modelname_old (to distinguish old versions)
            # - residuals_data_driven_reg.pth -> data_reg
            # - residuals_data_driven_standard.pth -> data_standard
            if basename.endswith('_residuals_old.pth'):
                model_name = basename.replace('_residuals_old.pth', '_old')
            elif basename.endswith('_residuals.pth'):
                model_name = basename.replace('_residuals.pth', '')
            elif basename == 'residuals_data_driven_reg.pth':
                model_name = 'data_reg'
            elif basename == 'residuals_data_driven_standard.pth':
                model_name = 'data_standard'
            else:
                # Fallback: remove .pth extension
                model_name = basename.replace('.pth', '')

            models_to_process.append((model_name, rf))
        
        model_description = f"Batch processing {len(models_to_process)} residual files"
        
    elif args.model:
        # Single model mode
        models = [args.model]
        if args.model in ['data_standard', 'data_reg']:
            model_description = f"Data-driven model: {args.model}"
        else:
            model_description = f"PINN model: {args.model}"
        
        # Convert to (model_name, residuals_path) format
        for model in models:
            residuals_path = args.residuals_path if args.residuals_path else None
            models_to_process.append((model, residuals_path))
    else:
        # Run all models by default (legacy behavior)
        models = ['relobralo', 'constant_weight', 'brdr', 'pecann', 'data_standard', 'data_reg']
        model_description = "All models (relobralo, constant_weight, brdr, pecann, data_standard, data_reg)"
        
        # Convert to (model_name, residuals_path) format
        for model in models:
            models_to_process.append((model, None))

    chunk_size_mb = args.chunk_size_mb
    max_points = args.max_points

    print("STARTING RESIDUAL DISTRIBUTION ANALYSIS")
    print("=" * 60)
    print("This script will generate:")
    print(f"   Model: {model_description}")
    if args.violin_only:
        print("   MODE: Violin plots only (skipping table generation)")
        print("   - 4 comprehensive violin plots: 5×4 grids")
        print("   - Excludes *_old model versions")
    else:
        print("   - Wide-format tables: 2 per model (data + physical residuals)")
        print("   - Each table: 4 residuals × 4 metrics (μ, σ, skewness, kurtosis)")
        print("   - 4 comprehensive violin plots: 5×4 grids")
        print("   - Ranking tables: MAE and STD for healthy residuals")
    print(f"   Output directory: {args.output_dir}")
    print("=" * 60)

    # Validate compatibility first (unless skipped or in batch mode)
    if not args.skip_validation and not args.residuals_dir:
        validate_model_data_compatibility()

        # Quick model test (skip for data-driven models)
        model_names_only = [m[0] for m in models_to_process]
        if not any(model in ['data_standard', 'data_reg'] for model in model_names_only):
            quick_model_test()

    # Initialize results storage
    all_tables = {}
    all_normal_residuals = {}

    # Process each model
    for model_name, residuals_path in models_to_process:
        try:
            # In batch mode, always use saved residuals
            use_saved = args.use_saved_residuals or args.residuals_dir
            
            if args.violin_only:
                # Violin-only mode: just load normal residuals, skip table generation
                print(f"\nLoading residuals for {model_name} (violin-only mode)...")
                
                if not residuals_path:
                    # Auto-detect path
                    residuals_path = f"best_model_residuals/{model_name}_residuals.pth"
                    if not os.path.exists(residuals_path):
                        residuals_path = f"best_model_residuals/{model_name.replace('_', '')}_residuals.pth"
                    if not os.path.exists(residuals_path):
                        if model_name == 'data_standard':
                            residuals_path = "best_model_residuals/residuals_data_driven_standard.pth"
                        elif model_name == 'data_reg':
                            residuals_path = "best_model_residuals/residuals_data_driven_reg.pth"
                
                if os.path.exists(residuals_path):
                    # Load and extract normal residuals only
                    saved_data = torch.load(residuals_path)
                    for data_type, samples in saved_data.items():
                        if data_type == 'normal' and samples:
                            sample_data = samples[0]['data'].numpy()
                            all_normal_residuals[model_name] = {
                                'data_res1': sample_data[:, 0],
                                'data_res2': sample_data[:, 1],
                                'data_res3': sample_data[:, 2],
                                'data_res4': sample_data[:, 3],
                                'phys_res1': sample_data[:, 4],
                                'phys_res2': sample_data[:, 5],
                                'phys_res3': sample_data[:, 6],
                                'phys_res4': sample_data[:, 7]
                            }
                            print(f"   Loaded normal residuals for {model_name}")
                            break
                else:
                    print(f"   WARNING: Could not find residuals file for {model_name}")
            else:
                # Full mode: generate tables and collect residuals
                model_tables, normal_residuals = process_residual_analysis(
                    model_name, chunk_size_mb, max_points,
                    use_saved_residuals=use_saved,
                    residuals_path=residuals_path
                )
                all_tables[model_name] = model_tables
                all_normal_residuals[model_name] = normal_residuals
        except Exception as e:
            print(f"ERROR: Failed to process model {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save results
    if not args.violin_only:
        # Generate tables and rankings (skip in violin-only mode)
        if all_tables:
            print("\nSaving LaTeX tables...")
            save_latex_tables(all_tables, args.output_dir)

            print("\nGenerating ranking tables for healthy residuals...")
            if all_normal_residuals:
                try:
                    generate_ranking_tables(all_normal_residuals, args.output_dir)
                except Exception as e:
                    print(f"ERROR: Failed to generate ranking tables: {e}")
                    import traceback
                    traceback.print_exc()

    # Always generate violin plots if we have residuals
    if all_normal_residuals:
        print("\nGenerating comprehensive violin plots for healthy residuals...")
        try:
            create_comprehensive_violin_plots(all_normal_residuals, args.output_dir)
        except Exception as e:
            print(f"ERROR: Failed to create comprehensive violin plots: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    
    if args.violin_only:
        print("Generated results (violin-only mode):")
        print(f"   Processed {len(all_normal_residuals)} models (excluding *_old versions)")
        print(f"   4 comprehensive violin plots (5×4 grids):")
        print(f"      - healthy_residuals_violin_group1_data.png")
        print(f"      - healthy_residuals_violin_group1_phys.png")
        print(f"      - healthy_residuals_violin_group2_data.png")
        print(f"      - healthy_residuals_violin_group2_phys.png")
    else:
        print("Generated results for:")
        pinn_models = [m for m in all_tables.keys() if m not in ['data_reg', 'data_standard']]
        data_models = [m for m in all_tables.keys() if m in ['data_reg', 'data_standard']]

        for model_name, model_tables in all_tables.items():
            if model_name in data_models:
                print(f"   - {model_name}: 1 table (data residuals)")
            else:
                print(f"   - {model_name}: 2 tables (data + phys residuals)")

        total_tables = len(pinn_models) * 2 + len(data_models) * 1
        print(f"   Total wide-format tables generated: {total_tables}")
        print(f"   4 comprehensive violin plots (5×4 grids)")
        print(f"   Ranking tables: healthy_residuals_ranking_tables.tex")
    
    print(f"   Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
