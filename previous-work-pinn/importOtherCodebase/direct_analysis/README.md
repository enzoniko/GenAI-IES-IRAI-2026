
# Direct PINN Analysis

This package provides tools for analyzing physical systems using Direct Physics-Informed Neural Networks (PINNs). The package enables data preparation, model training, residual computation, and visualization of results for physical system analysis.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Directory Structure](#directory-structure)
- [Usage](#usage)
  - [Running a Full Analysis](#running-a-full-analysis)
  - [Visualizing Saved Residuals](#visualizing-saved-residuals)
  - [Recreating Plots from Saved Data](#recreating-plots-from-saved-data)
- [Pipeline Components](#pipeline-components)
- [Output Structure](#output-structure)
- [Example Results](#example-results)

## Overview

Direct PINN Analysis is a pipeline for analyzing physical systems data using Physics-Informed Neural Networks. The package handles:

- Data loading and normalization
- PINN model training with custom physics-based loss functions
- Residual computation for different datasets (normal, abnormal, etc.)
- Visualization of model performance through various metrics and plots
- Storing and organizing results for further analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/hybrid_pinns.git
cd hybrid_pinns

# Install dependencies
pip install -r requirements.txt
```

## Directory Structure

```
direct_analysis/
├── __init__.py          # Package initialization
├── cli.py               # Command-line interface
├── data_utils.py        # Data handling utilities
├── residuals.py         # Residual computation
├── training.py          # Model training utilities
└── visualization/       # Visualization utilities
    ├── boxplots.py      # Boxplot visualization
    ├── metrics_plots.py # Publication-quality metrics plots
    └── data_utils.py    # Visualization data utilities
```

## Usage

### Running a Full Analysis

To run the complete Direct PINN analysis pipeline:

```bash
python -m direct_analysis.cli run --output-dir results/my_analysis
```

This will:
1. Load and prepare the data
2. Train a PINN model
3. Compute residuals on all datasets
4. Generate visualizations
5. Save all outputs to the specified directory

### Visualizing Saved Residuals

If you already have computed residuals and want to generate visualizations:

```bash
python -m direct_analysis.cli visualize path/to/direct_pinn_residuals.pth --output-dir results/my_visualizations
```

### Recreating Plots from Saved Data

To regenerate plots from previously saved data:

```bash
python -m direct_analysis.cli recreate path/to/data_directory --output-dir results/recreated_plots
```

You can also recreate plots programmatically:

```python
from direct_analysis.visualization.data_utils import create_plots_from_saved_data

# Regenerate plots from saved data
create_plots_from_saved_data(data_dir='path/to/data_directory')

# To use a custom output prefix:
create_plots_from_saved_data(data_dir='path/to/data_directory', output_prefix='custom')
```

## Pipeline Components

### Data Preparation
- Loads data from pre-existing `.pth` files
- Adds time feature to input data
- Normalizes features for better model performance
- Splits data into training, validation, and test sets

### Model Training
- Implements a Physics-Informed Neural Network (PINN)
- Uses custom loss functions combining data and physics-based constraints
- Implements early stopping and learning rate scheduling
- Saves the trained model and training history

### Residual Computation
- Calculates prediction residuals across multiple datasets
- Processes normal, abnormal, and other datasets as configured
- Implements optional physical residual calculation

### Visualization
- Produces boxplots of residuals by variable and frequency
- Generates publication-quality metrics plots
- Creates comparative visualizations across different data types

## Output Structure

Running the pipeline produces a structured output directory:

```
results/direct_pinn_[timestamp]/
├── models/                        # Trained models and raw results
│   ├── direct_pinn_model_final.pth  # Saved model weights
│   ├── direct_pinn_residuals.pth    # Computed residuals
│   ├── training_history.npz         # Training metrics history
│   ├── X_min.pth                    # Normalization parameters
│   ├── X_max.pth
│   ├── y_min.pth
│   └── y_max.pth
├── data/                          # Processed data for plots
├── plots/                         # Generated boxplots
├── comparisons/                   # Metrics comparison visualizations
└── README.md                      # Instructions for using the results
```

## Example Results

After running the pipeline, you'll get various visualizations:

- Boxplots showing residual distributions for each variable across frequencies
- Metrics comparisons demonstrating model performance on different datasets
- Time-series visualizations of predictions vs. ground truth

These results can be used to evaluate the model's ability to capture the underlying physics of the system and to identify anomalous behavior.

---

For additional information or contributions, please contact the repository maintainers.
