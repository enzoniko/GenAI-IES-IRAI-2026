# Anomaly Detection Module

This module provides tools for detecting anomalies in residual data from physics-informed neural networks (PINNs) and data-driven models.

## Features

- Feature extraction from time series data
- Multiple anomaly detection methods:
  - Extreme Value Theory (EVT)
  - Isolation Forest
- Preprocessing support for different residual formats:
  - Direct PINN
  - Hybrid RNN-PINN
- Grid search for optimal parameter selection
- Evaluation using standard metrics
- Visualization tools for results analysis

## Installation

This module is part of the larger repository. No separate installation is required.

## Usage

### Command Line Interface

The module can be used from the command line to perform anomaly detection on residual data:

```bash
python -m anomaly_detection.cli --residuals path/to/residuals.pth --source direct
```

#### Arguments

- `--residuals`: Path to the residuals file (.pth) [required]
- `--source`: Source type of residuals: 'direct' or 'hybrid' [default: direct]
- `--output`: Output file for the results [default: anomaly_detection_results.json]
- `--workers`: Number of worker processes for grid search [default: 4]
- `--sample-rate`: Sample rate in Hz for FFT features [default: 50000]
- `--methods`: Methods to evaluate ('evt', 'isolation_forest', or 'all') [default: all]
- `--plot-dir`: Directory for saving plots [default: anomaly_detection_plots]
- `--skip-plots`: Skip generating plots [flag]

### Programmatic Usage

You can also use the module programmatically in your Python code:

```python
from anomaly_detection import preprocess_residuals, EVTDetector, IsolationForestDetector, evaluate_detector

# Load and preprocess residuals
processed = preprocess_residuals('residuals.pth', source_type='direct')

# Split normal samples
normal_samples = processed.get('normal', [])
normal_train = normal_samples[:int(0.8 * len(normal_samples))]
normal_test = normal_samples[int(0.8 * len(normal_samples)):]

# Prepare test samples
test_samples = {'normal': normal_test}
for key, sequences in processed.items():
    if key != 'normal':
        test_samples[key] = sequences

# Create and fit a detector
detector = EVTDetector(agg_func=np.max, threshold_u_quantile=0.95, quantile=0.99)
detector.fit(normal_train)

# Evaluate the detector
metrics = evaluate_detector(detector, test_samples)
print(metrics)
```

## Methods

### Extreme Value Theory (EVT)

EVT models the extreme values of a distribution and is particularly suited for anomaly detection. It uses the Generalized Pareto Distribution (GPD) to model the tail of the distribution.

### Isolation Forest

Isolation Forest is an ensemble-based method that isolates anomalies by randomly selecting a feature and a split value. It's effective for detecting anomalies in high-dimensional spaces.

## Feature Extraction

The module provides various feature extraction methods:

- Basic statistical features (mean, median, std, kurtosis, skewness, etc.)
- FFT-based features (spectral centroid, bandwidth, roll-off, etc.)
- Wavelet-based features (energy, entropy, etc.)
- tsfresh features (optional)

## Dependencies

- numpy
- scipy
- scikit-learn
- matplotlib
- pywt (PyWavelets)
- tsfresh
- torch
- pandas 