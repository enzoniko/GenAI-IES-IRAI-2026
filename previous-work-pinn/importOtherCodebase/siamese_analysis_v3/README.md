# Siamese Neural Network for Residual Analysis v3

This module implements a Siamese neural network approach for analyzing residuals from physical models and classifying them based on both data type and rotation speed. Version 3 combines the strong F1 score performance of v0 with the enhanced visualization capabilities of v1.

## Key Features

- Uses a Siamese neural network with CNN-based encoder to create embeddings for residuals
- Employs triplet loss for training the network to distinguish between different residuals
- Provides highly accurate F1 scores from the v0 classification approach
- Advanced hierarchical visualization using t-SNE and UMAP from v1
- Three-level hierarchical labeling for comprehensive analysis:
  - Level 1: Group (normal, imbalance, overhang, etc.)
  - Level 2: Data type (normal, imbalance_fault_6g, etc.)
  - Level 3: Data type with rotation speed (normal_0.52, etc.) - except normal samples which are grouped into a single 'normal' label
- Classification metrics (reports and accuracy) for all three hierarchy levels
- Confusion matrices for levels 1 and 2 of the hierarchy
- Fine-grained classification using pairs of <data type, rotation speed> as labels
- Cross-validation to reduce overfitting and select the best model
- Early stopping to prevent overfitting
- Comprehensive evaluation and visualization tools
- Option to use either simple FFT-based features (v0) or advanced feature extraction (v1)
- Optimized embedding calculation: computes embeddings only once and reuses them for all classification and visualization tasks
- Hyperparameter optimization with randomized search and error distribution analysis
- Error-resilient pipeline with robust exception handling and fallback mechanisms
- Efficient two-phase evaluation approach: full cross-validation for initial selection followed by efficient single-split evaluation for top models

## Error Resilience

This version includes comprehensive error handling to ensure the pipeline can continue running even when encountering invalid configurations or unexpected errors:

1. **Randomized Search Resilience**: If a trial with a specific configuration fails, the search continues with the next configuration instead of crashing.

2. **Parameter Validation**: All model parameters are validated before use, with appropriate warnings and fallbacks for invalid values.

3. **Multi-level Fallback Strategy**:
   - If a specific model configuration fails during training, it's skipped
   - If all trials fail, a default model with a safe architecture is used
   - If the best model can't be recreated after search, a simpler fallback model is created
   - In case of critical errors, an emergency minimal model ensures the pipeline can still produce a result

4. **Graceful Degradation**: Rather than crashing when errors occur, the system logs detailed diagnostics and attempts to continue with a fallback approach.

5. **Specific Error Handling**:
   - Memory errors: Clear guidance on reducing batch size or feature complexity
   - CUDA errors: Suggestions for GPU memory optimization
   - Import errors: Helpful messaging about potential missing dependencies
   - File errors: Clear feedback about missing input files

6. **Comprehensive Logging**: Detailed error reports and tracebacks help diagnose issues while allowing the pipeline to continue running.

## Module Structure

- `data/`: Data processing and dataset creation
  - `processor.py`: Feature extraction from residuals with hierarchical labeling
  - `feature_extractors.py`: Advanced feature extraction methods (wavelets, statistics, etc.)
  - `parallel_features.py`: Parallel processing for advanced feature extraction
  - `dataset.py`: Triplet dataset and cross-validation utilities
- `models/`: Neural network model definitions
  - `siamese.py`: Siamese network and triplet loss implementation
  - `configurable_siamese.py`: Highly configurable Siamese network for architecture search
- `training/`: Model training utilities
  - `trainer.py`: Training loop with early stopping and cross-validation
  - `randomized_search.py`: Randomized hyperparameter search for model optimization
- `evaluation/`: Evaluation tools
  - `evaluator.py`: Comprehensive evaluation and visualization of trained models
- `cli.py`: Command-line interface

## Feature Extraction Options

This version supports two feature extraction methods:

1. **Simple FFT-based Features (v0)**: 
   - Extracts only the FFT magnitude spectrum
   - Fast and lightweight processing
   - Default option for backward compatibility

2. **Advanced Feature Extraction (v1)**:
   - Statistical features (mean, std, kurtosis, skewness, etc.)
   - FFT-based spectral features
   - Wavelet decomposition features
   - Optional tsfresh features (requires tsfresh package)
   - Parallel processing for efficiency
   - Activated with the `--advanced-features` flag

## Hyperparameter Optimization

This version includes a robust hyperparameter optimization approach:

1. **Randomized Search**: Efficiently explores a wide range of hyperparameters for network architecture and training.

2. **Architecture Exploration**: Searches through various CNN and fully-connected layer configurations.

3. **Two-Phase Evaluation Strategy**:
   - **Initial Phase**: Uses full cross-validation to identify top-performing models
   - **Final Phase**: Evaluates top models using multiple single train/validation splits instead of full cross-validation for computational efficiency 
   - Creates error distributions to assess model stability and performance consistency
   - Provides up to 30x single-split evaluations in the same computational budget as full cross-validation

4. **Detailed Outputs**:
   - Error distribution plots (PNG format)
   - Error distribution data (TXT format) with both summary statistics and raw values for paper inclusion
   - Complete model architecture and performance metrics

## Baseline Evaluation (Enhanced)

This version includes comprehensive baseline evaluation capabilities with advanced statistical analysis to assess data quality and justify the complexity of the Siamese network approach:

1. **LogisticRegression Baseline**: Closed-set classification using raw features with GridSearchCV hyperparameter optimization.

2. **PCA + KNN Baseline**: Dimensionality reduction followed by K-Nearest Neighbors, capable of open-set recognition with simulated unknown classes.

3. **Hierarchical Evaluation**: Both baselines are evaluated at all three hierarchical levels (group, data type, and detailed labels).

4. **Statistical Confidence**: 95% confidence intervals calculated using bootstrapping (1000 samples) for robust performance estimation.

5. **Open-set Recognition Metrics**:
   - Open-set F1 macro scoring to evaluate rejection of unknown fault types
   - AUROC (Area Under Receiver Operating Characteristic) for open-set detection using confidence scores

6. **Calibration Analysis**: Expected Calibration Error (ECE) calculated using k-NN confidence scores:
   - Confidence = (Number of neighbors with majority class) / K
   - Binned confidence analysis (10 bins from 0.0 to 1.0)
   - Weighted average of |mean_confidence - accuracy| per bin

7. **Per-Speed Performance Analysis**: F1 scores broken down by rotational speed to identify speed-specific performance patterns.

8. **Comprehensive Comparison**: Results are saved in a dedicated `linear_probe_analysis` subfolder with:
   - Individual method reports with best hyperparameters and all metrics
   - Comparison tables showing F1 scores, confidence intervals, and advanced metrics
   - Per-speed performance breakdowns
   - Data quality assessment insights comparing baseline performance to Siamese network results

The enhanced baseline evaluation provides a complete statistical analysis to answer: "Does the residual data quality justify using a complex Siamese network, or would simpler approaches suffice?"

> **Note on CLI Arguments**: Although parameters like `--embedding-size` and `--margin` appear in the CLI help, when running with randomized search (the default), these values are controlled by the search space defined in `randomized_search.py` and not directly from CLI arguments. This ensures proper hyperparameter exploration without user interference. The CLI arguments for these parameters would only be used if you explicitly disabled the randomized search.

## Hyperparameter Search Space

The randomized search explores a wide range of hyperparameters, including:

### Model Architecture
- **Embedding Size**: Size of the output embedding vector (16-256)
- **CNN Architecture**: 
  - Number of convolutional blocks (2-5)
  - Channel configurations for different network depths
  - Kernel sizes (3, 5, 7)
  - Pooling types (max, avg) and sizes (2, 3)
  - Stride and dilation rates
  - Padding strategies
- **Skip Connections**: Whether to use residual connections
- **Normalization**: 
  - Batch normalization, layer normalization, instance normalization, or none
  - BatchNorm parameters (momentum, epsilon)
- **Activations**: ReLU, LeakyReLU, ELU, GELU with configurable parameters
- **Fully Connected Layers**: Various configurations of hidden layer sizes

### Loss Function
- **Margin**: Triplet loss margin (0.1-2.0)
- **Distance Metric**: Squared Euclidean distance by default
- **Triplet Selection**: All, hard, semi-hard, or distance-weighted triplet mining
- **Reduction Method**: Mean or sum loss reduction
- **Label Smoothing**: Amount of label smoothing (0.0-0.2)

### Optimization
- **Optimizer**: Adam, SGD, AdamW, RMSprop
- **Weight Decay**: L2 regularization strength
- **Momentum**: For SGD and RMSprop
- **Adam Parameters**: Beta1 and Beta2 for Adam/AdamW
- **Weight Initialization**: Strategies like Kaiming, Xavier with configurable gain

### Learning Rate Scheduling
- **Scheduler Type**: ReduceLROnPlateau, CosineAnnealing, Step, or None
- **Patience**: For ReduceLROnPlateau (3-10 epochs)
- **Reduction Factor**: Amount to reduce learning rate (0.1-0.5)
- **Warmup Epochs**: Number of epochs for learning rate warmup (0-5)

## Visualization Capabilities

This version includes advanced visualization tools from v1:

1. **Multi-perspective t-SNE**: Generates t-SNE visualizations with multiple perplexity values to provide different perspectives on the data structure.

2. **Group-focused Visualizations**: Creates separate visualizations for each fault group, highlighting specific patterns within each group while maintaining context.

3. **Speed-encoded Plots**: Uses color saturation to represent rotation speed, providing an additional dimension of information in the visualizations.

4. **UMAP Visualizations**: Adds UMAP as an alternative dimensionality reduction technique with customizable parameters.

5. **Hierarchical Analysis**: Visualizes data at three different hierarchical levels to reveal patterns at varying levels of granularity.

## Evaluation Outputs

The module produces several evaluation outputs:

1. **Classification Reports**: Performance metrics (precision, recall, F1) for all three hierarchy levels
   - Level 1: Group-level metrics (normal, imbalance, overhang, etc.)
   - Level 2: Fault type metrics (normal, imbalance_fault_6g, etc.)
   - Level 3: Detailed metrics with rotation speeds (normal_0.52, etc.)

2. **Confusion Matrices**: Visual representation of classification performance
   - Level 1: Group-level confusion matrix
   - Level 2: Fault type confusion matrix

3. **Accuracy Summaries**: CSV files with accuracy for each class at levels 1 and 2

4. **Hyperparameter Search Results**:
   - Best model weights and configuration
   - Error distribution data and visualizations
   - Detailed model architecture summary

## Usage

The module can be used through the main script:

```bash
./run_siamese_analysis.py --residuals path/to/residuals.pth --source direct --output-dir results
```

### Command Line Arguments

#### Input/Output
- `--residuals`: Path to the residuals file (.pth) [required]
- `--source`: Source type of residuals: direct PINN or hybrid RNN-PINN [default: direct]
- `--output-dir`: Output directory [default: siamese_results_v3]

#### Data Processing
- `--fft-length`: Length of FFT features [default: 128]
- `--num-channels`: Number of channels in residuals [default: 4]
- `--num-triplets`: Number of triplets to generate [default: 10000]
- `--labels-to-use`: Labels to use for training (default: all)

#### Advanced Feature Extraction (v1)
- `--advanced-features`: Use advanced feature extraction from v1 [flag]
- `--sampling-rate`: Sampling rate in Hz [default: 50000]
- `--include-tsfresh`: Include tsfresh features (requires tsfresh package) [flag]
- `--wavelet`: Wavelet type to use for decomposition [default: db4]
- `--wavelet-level`: Wavelet decomposition level [default: 3]
- `--max-sequence-length`: Maximum sequence length for padding (auto-detect if None)
- `--feature-workers`: Number of worker processes for parallel feature extraction (auto-detect if None)
- `--feature-batch-size`: Batch size for parallel feature extraction [default: 10]

#### Training
- `--batch-size`: Batch size for training [default: 64]
- `--epochs`: Maximum number of training epochs [default: 100]
- `--learning-rate`: Learning rate [default: 0.001]
- `--embedding-size`: Size of embedding vector [default: 32]
- `--margin`: Margin for triplet loss [default: 1.0]
- `--n-folds`: Number of cross-validation folds [default: 5]
- `--patience`: Patience for early stopping [default: 10]
- `--workers`: Number of worker processes for data loading [default: 4]

#### Learning Rate Scheduling
- `--lr-scheduler`: Learning rate scheduler type [choices: reduce_on_plateau, cosine_annealing, step, none] [default: reduce_on_plateau]
- `--lr-factor`: Factor by which learning rate is reduced [default: 0.1]
- `--lr-patience`: Number of epochs with no improvement after which learning rate will be reduced [default: 5]
- `--lr-min`: Minimum learning rate [default: 1e-6]

#### Randomized Search
- `--num-trials`: Number of randomized search trials [default: 20]
- `--num-top-models`: Number of top models to keep and evaluate [default: 5]
- `--num-repeat-runs`: Number of repeat runs for each top model [default: 30]
- `--search-space`: Size of architecture search space to explore [default: 'small']

#### Evaluation
- `--threshold`: Similarity threshold for classification [default: 0.5]
- `--evaluate-baselines`: Evaluate baseline classifiers (LogisticRegression and PCA+KNN) for comparison [flag]
- `--baseline-test-size`: Test set size for baseline evaluation [default: 0.3]

#### Visualization
- `--visualization-level`: Which hierarchy levels to visualize ('all' or specific level number) [default: 'all']
- `--tsne-perplexity`: List of perplexity values for t-SNE [default: 30,50,70,90,110,150]
- `--umap-n-neighbors`: List of n_neighbors values for UMAP [default: 5,15,30,50,100]
- `--umap-min-dist`: List of min_dist values for UMAP [default: 0.0,0.1,0.25,0.5,0.8]
- `--reuse-embeddings`: Whether to reuse previously saved embeddings if available [flag]

#### Other
- `--device`: Device to use (cpu or cuda, default: auto-detect)
- `--seed`: Random seed for reproducibility [default: 42]
- `--verbose`: Verbosity level (0=silent, 1=progress, 2=detailed) [default: 1]

## Output

The module produces several outputs in the specified output directory:

- Trained model (`siamese_model.pt`)
- Training history plots
- Visualization subdirectory with:
  - Multi-parameter t-SNE visualizations per group
  - Multi-parameter UMAP visualizations per group
- Classification reports for all three hierarchy levels:
  - `classification_report_data_types.txt` (Level 1)
  - `classification_report_level2.txt` (Level 2)
  - `classification_report_full.txt` (Level 3)
- Confusion matrices for hierarchy levels 1 and 2:
  - `confusion_matrix_data_types.png` (Level 1)
  - `confusion_matrix_level2.png` (Level 2)
- Accuracy summary files:
  - `data_type_accuracy.csv` (Level 1)
  - `fault_type_accuracy.csv` (Level 2)
- Cached embeddings for faster reuse (`embeddings.pkl`)
- Randomized search results:
  - Best model configuration and weights
  - Error distribution plot (`error_distributions.png`)
  - Error distribution data in text format (`error_distributions.txt`) with summary statistics and raw values
  - Detailed model architecture summary
- Baseline evaluation results (when `--evaluate-baselines` is used):
  - `linear_probe_analysis/` subdirectory with method-specific results
  - `baseline_comparison.csv` and `baseline_comparison.txt` with performance comparison including confidence intervals
  - `per_speed_performance.csv` and `per_speed_performance.txt` with speed-specific F1 scores
  - Individual reports for LogisticRegression and PCA+KNN methods with ECE analysis, AUROC, and per-speed breakdowns

## Advantages Over Previous Versions

- **Compared to v0**: Adds hierarchical labeling and advanced visualization capabilities from v1
- **Compared to v1**: Maintains the superior F1 score performance of v0's classification approach
- **Compared to v2**: Simpler implementation focused on combining the best of v0 and v1
- **Improvements in v3**: 
  - Added classification metrics for all three hierarchy levels 
  - Added confusion matrices for levels 1 and 2
  - Removed redundant visualizations for cleaner output
  - Added option to use advanced feature extraction from v1
  - Optimized evaluation pipeline for better performance with batch processing
  - Efficient memory usage by computing embeddings only once and reusing them
  - Improved normal sample classification by grouping all normal samples regardless of rotation speed
  - Added hyperparameter optimization with randomized search
  - Added text-based error distribution data export for paper inclusion 
  - Enhanced top model evaluation performance by using single train/validation splits instead of full cross-validation 