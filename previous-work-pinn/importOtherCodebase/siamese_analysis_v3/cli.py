"""
Command-line interface for Siamese residual analysis
"""
import argparse
import torch
import os
import logging
import sys
import numpy as np
from typing import Dict, List, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
def setup_logging(output_dir=None):
    """Set up logging to both console and file"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    if logger.handlers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        log_file = os.path.join(output_dir, 'training.log')
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        print(f"Logging to: {log_file}")
    
    return logger

# Initialize with basic logging to console only
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Siamese Neural Network for Residuals Analysis v3")
    
    # Input/output arguments
    parser.add_argument('--residuals', type=str, default='residuals_dict.pth', 
                        help='Path to the residuals file (.pth)')
    parser.add_argument('--source', type=str, choices=['direct', 'hybrid'], default='direct',
                        help='Source type of residuals: direct PINN or hybrid RNN-PINN')
    parser.add_argument('--output-dir', type=str, default='siamese_results_v3_100000triplets_physical_4faultgroups_48samplespergroupBESTMODELALLFEATURES0', 
                        help='Output directory')
    
    # Data processing arguments
    parser.add_argument('--fft-length', type=int, default=128, 
                        help='Length of FFT features')
    parser.add_argument('--num-channels', type=int, default=4, 
                        help='Number of channels in residuals')
    parser.add_argument('--num-triplets', type=int, default=20000, 
                        help='Number of triplets to generate')
    parser.add_argument('--labels-to-use', nargs='+', 
                        help='Labels to use for training (default: all)')
    parser.add_argument('--known-classes', nargs='+', 
                        default=['normal', 'imbalance_fault', 'overhang_ball_fault', 
                                'horizontal_misalignment_fault', 'vertical_misalignment_fault'],
                        help='List of known fault classes to use for training (default: normal, imbalance_fault, overhang_ball_fault, horizontal_misalignment_fault, vertical_misalignment_fault)')
    parser.add_argument('--include-physical', action='store_true', default=False,
                        help='Include physical residuals when available (doubles channels from 4 to 8)')
    
    # Advanced feature extraction arguments (from v1)
    parser.add_argument('--advanced-features', action='store_true', default=True,
                        help='Use advanced feature extraction from v1 (wavelets, statistical features, etc.)')
    parser.add_argument('--sampling-rate', type=int, default=50000,
                        help='Sampling rate in Hz (for advanced features)')
    parser.add_argument('--include-tsfresh', action='store_true',
                        help='Include tsfresh features (requires tsfresh package)')
    parser.add_argument('--wavelet', type=str, default='db4',
                        help='Wavelet type to use for decomposition (for advanced features)')
    parser.add_argument('--wavelet-level', type=int, default=3,
                        help='Wavelet decomposition level (for advanced features)')
    parser.add_argument('--max-sequence-length', type=int, default=None,
                        help='Maximum sequence length for padding (auto-detect if None)')
    parser.add_argument('--feature-workers', type=int, default=None,
                        help='Number of worker processes for parallel feature extraction')
    parser.add_argument('--feature-batch-size', type=int, default=10,
                        help='Batch size for parallel feature extraction')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=256, 
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20, 
                        help='Maximum number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.01, 
                        help='Learning rate')
    parser.add_argument('--n-folds', type=int, default=3, 
                        help='Number of cross-validation folds')
    parser.add_argument('--patience', type=int, default=10, 
                        help='Patience for early stopping')
    parser.add_argument('--data-loader-workers', type=int, default=20, 
                        help='Number of worker processes for data loading')
    
    # Randomized search arguments
    parser.add_argument('--num-trials', type=int, default=1,
                        help='Number of randomized search trials')
    parser.add_argument('--num-top-models', type=int, default=1,
                        help='Number of top models to keep and evaluate')
    parser.add_argument('--num-repeat-runs', type=int, default=1,
                        help='Number of repeat runs for each top model')
    
    # Learning rate scheduler arguments
    parser.add_argument('--lr-scheduler', type=str, 
                        choices=['reduce_on_plateau', 'cosine_annealing', 'step', 'none'], 
                        default='reduce_on_plateau',
                        help='Learning rate scheduler type')
    parser.add_argument('--lr-factor', type=float, default=0.1,
                        help='Factor by which learning rate is reduced (for ReduceLROnPlateau)')
    parser.add_argument('--lr-patience', type=int, default=5,
                        help='Number of epochs with no improvement after which learning rate will be reduced')
    parser.add_argument('--lr-min', type=float, default=1e-6,
                        help='Minimum learning rate')
    
    # Evaluation arguments
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Similarity threshold for classification')
    parser.add_argument('--use-knn', action='store_true', default=True,
                        help='Use K-Nearest Neighbors for classification in addition to centroid-based approach')
    parser.add_argument('--knn-k-values', type=int, nargs='+', default=[5],
                        help='K values to try for KNN classification')
    parser.add_argument('--reference-samples', type=int, default=30,
                        help='Number of reference samples per class to use for KNN')

    # Baseline evaluation arguments
    parser.add_argument('--evaluate-baselines', action='store_true', default=True,
                        help='Evaluate baseline classifiers (LogisticRegression and PCA+KNN) for comparison')
    parser.add_argument('--baseline-test-size', type=float, default=0.3,
                        help='Test set size for baseline evaluation (default: 0.3)')
    parser.add_argument('--baseline-max-samples', type=int, default=500,
                        help='Maximum number of samples to use for baseline evaluation to improve speed (default: 500)')
    
    # Visualization arguments
    parser.add_argument('--visualization-level', type=str, default='all', 
                        help='Which hierarchy levels to visualize (all, 1, 2, 3)')
    parser.add_argument('--tsne-perplexity', type=float, nargs='+', default=[30, 90, 150], 
                        help='Perplexity values for t-SNE visualization')
    parser.add_argument('--umap-n-neighbors', type=int, nargs='+', default=[30, 50, 100], 
                        help='n_neighbors values for UMAP visualization')
    parser.add_argument('--umap-min-dist', type=float, nargs='+', default=[0.25, 0.5, 0.8], 
                        help='min_dist values for UMAP visualization')
    parser.add_argument('--reuse-embeddings', action='store_true', 
                        help='Reuse previously saved embeddings if available')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: auto-detect)')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    parser.add_argument('--verbose', type=int, default=2, 
                        help='Verbosity level (0=silent, 1=progress, 2=detailed)')
    
    # Argument to load a pre-trained model directly
    parser.add_argument('--load-best-model-from', type=str, default=None,
                        help='Path to a directory containing best_configuration.json and best_model.pt to load directly, skipping search.')
    
    return parser.parse_args()


def load_residuals(residuals_path: str) -> Dict[str, Any]:
    """
    Load residuals from file
    
    Args:
        residuals_path: Path to residuals file
        
    Returns:
        Dictionary of residuals
    """
    logger.info(f"Loading residuals from {residuals_path}...")
    try:
        residuals_dict = torch.load(residuals_path)
        return residuals_dict
    except Exception as e:
        logger.error(f"Error loading residuals: {e}")
        raise


def run_training(args):
    """
    Run the training process
    
    Args:
        args: Command line arguments
    """
    # Set up logging to file in output directory
    setup_logging(args.output_dir)
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    args.device = device
    
    # Load residuals
    residuals_dict = load_residuals(args.residuals)
    
    # Import necessary modules
    from siamese_analysis_v3.data.processor import preprocess_residuals, ResidualProcessor
    from siamese_analysis_v3.data.dataset import generate_triplets_from_labeled_samples, TripletDataset
    from siamese_analysis_v3.data.dataset import CrossValidationSplitter
    from siamese_analysis_v3.models.configurable_siamese import ConfigurableSiameseNetwork, TripletLoss
    from siamese_analysis_v3.training.trainer import SiameseTrainer
    from siamese_analysis_v3.training.randomized_search import randomized_search, load_best_configuration, load_best_weights, recreate_training_pipeline
    from siamese_analysis_v3.evaluation.evaluator import SiameseEvaluator

    # Create output directory and models subdirectory
    os.makedirs(args.output_dir, exist_ok=True)
    models_dir = os.path.join(args.output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Save the original working directory for later restoration
    original_cwd = os.getcwd()
    
    # Auto-adjust include_physical based on num_channels
    # If user explicitly set num_channels to 4, disable include_physical
    # If user explicitly set num_channels to 8, enable include_physical
    if args.num_channels == 4:
        if args.include_physical:
            logger.info("Auto-disabling --include-physical because --num-channels is set to 4")
            args.include_physical = False
    elif args.num_channels == 8:
        if not args.include_physical:
            logger.info("Auto-enabling --include-physical because --num-channels is set to 8")
            args.include_physical = True
    
    logger.info(f"Using num_channels={args.num_channels}, include_physical={args.include_physical}")
    
    # Preprocess residuals - organize by label (data_type_rot_speed)
    samples_by_label_eval = preprocess_residuals(residuals_dict, args.source, include_physical=args.include_physical)
    
    # Filter labels based on known classes specified via command line
    logger.info(f"Using known classes for training: {args.known_classes}")
    
    # First, log what labels we have from preprocessing
    logger.info("Available labels from preprocessing:")
    for label, samples in samples_by_label_eval.items():
        logger.info(f"  {label}: {len(samples)} samples")
    
    samples_by_label = {}
    for label, samples in samples_by_label_eval.items():
        # Check if this label matches any of the known classes
        for known_class in args.known_classes:
            if label.startswith(known_class):
                # Special handling for normal class - group all rotation speeds together
                # but take 1 sample from each rotation speed (not 1 sample total)
                if known_class == 'normal':
                    if 'normal' not in samples_by_label:
                        samples_by_label['normal'] = []
                    # Take 1 sample from this specific rotation speed
                    logger.info(f"Adding {min(2, len(samples))} samples from {label} to normal class")
                    samples_by_label['normal'].extend(samples[:2])
                else:
                    # For other fault classes, keep them separate by full label
                    if label not in samples_by_label:
                        samples_by_label[label] = []
                    samples_by_label[label].extend(samples[:48])
                break  # Found a match, no need to check other known classes
   
    # Log label statistics
    logger.info("Label statistics:")
    for label, samples in samples_by_label.items():
        logger.info(f"  {label}: {len(samples)} samples")
    
    # Create processor
    if args.advanced_features:
        logger.info("Using advanced feature extraction from v1")
        processor = ResidualProcessor(
            fft_length=args.fft_length,
            num_channels=args.num_channels,
            advanced_features=True,
            sampling_rate=args.sampling_rate,
            include_tsfresh=args.include_tsfresh,
            wavelet=args.wavelet,
            wavelet_level=args.wavelet_level,
            num_workers=args.feature_workers,
            batch_size=args.feature_batch_size
        )
    else:
        logger.info("Using simple FFT-based feature extraction")
        processor = ResidualProcessor(
            fft_length=args.fft_length,
            num_channels=args.num_channels
        )
    
    # Calculate the max sequence length based on rotation speeds
    if not args.max_sequence_length:
        max_seq_length = processor.calculate_max_sequence_length(residuals_dict)
        processor.max_sequence_length = max_seq_length
        logger.info(f"Calculated maximum sequence length: {max_seq_length}")
    else:
        processor.max_sequence_length = args.max_sequence_length
        logger.info(f"Using provided maximum sequence length: {args.max_sequence_length}")
    
    # Generate triplets
    logger.info(f"Generating {args.num_triplets} triplets...")
    anchors, positives, negatives = generate_triplets_from_labeled_samples(
        samples_by_label, processor, args.num_triplets, bypass_output_dir=args.output_dir
    )
    
    # Create dataset
    dataset = TripletDataset(anchors, positives, negatives)
    
    # Create cross-validation splitter
    cv_splitter = CrossValidationSplitter(n_splits=args.n_folds, random_state=args.seed)
    cv_splits = cv_splitter.split(dataset)

    # Determine the actual feature size from the data
    feature_size = anchors.shape[-1]
    logger.info(f"Using feature size: {feature_size} for model input")
    
    # Get the actual number of channels from the data
    actual_channels = anchors.shape[1]
    logger.info(f"Using {actual_channels} channels for model input")
    
    # Attach dataset and CV splitter to args for use in randomized search
    args.dataset = dataset
    args.cv_splitter = cv_splitter
    
    # Add batch size and workers to args for use in randomized search
    if not hasattr(args, 'batch_size'):
        args.batch_size = 64  # Default batch size
        
    if not hasattr(args, 'data_loader_workers'):
        args.data_loader_workers = 4  # Default number of data loader workers
    
    # --------------------------- RUN RANDOMIZED SEARCH ---------------------------
    logger.info("Starting randomized hyperparameter search...")
    best_model = randomized_search(args, actual_channels, feature_size)
    logger.info("Randomized search complete")
    # --------------------------END RANDOMIZED SEARCH------------------------------------------

    # Load the best configuration and the best weights from the randomized search
    best_configuration = load_best_configuration(args.output_dir)
    best_weights = load_best_weights(args.output_dir)

    # Create model with the best architecture and parameters
    model = recreate_training_pipeline(best_configuration, best_weights)
    
    # Log relevant information about the best model configuration
    logger.info(f"Best model embedding size: {best_configuration.get('embedding_size')}")
    
    # Log encoder configuration
    if 'encoder_config' in best_configuration:
        conv_layers = [l for l in best_configuration['encoder_config'] if l.get('type') == 'conv']
        channels = [l.get('out_channels') for l in conv_layers]
        logger.info(f"Best encoder config: {len(conv_layers)} conv blocks with channels {channels}")
    
    # Log FC configuration
    if 'fc_config' in best_configuration:
        linear_layers = [l for l in best_configuration['fc_config'] if l.get('type') == 'linear']
        fc_sizes = [l.get('out_features') for l in linear_layers]
        logger.info(f"Best FC config: {len(linear_layers)} linear layers with sizes {fc_sizes}")
    
    # Log if dropout was used
    if any(l.get('type') == 'dropout' for l in best_configuration.get('fc_config', [])):
        logger.info("Best model uses dropout in FC layers")
        
    logger.info(f"Best model validation loss: {best_configuration.get('avg_val_loss', 'N/A')}")
    
    # Create evaluator
    evaluator = SiameseEvaluator(
        model=model,
        processor=processor,
        device=device
    )
    
    # Evaluate and save results
    logger.info("Evaluating model...")
    evaluation_results = evaluator.evaluate_and_save(
        samples_by_label=samples_by_label_eval,
        output_dir=args.output_dir,
        threshold=args.threshold,
        visualization_level=args.visualization_level,
        tsne_perplexity_values=args.tsne_perplexity,
        umap_n_neighbors_values=args.umap_n_neighbors,
        umap_min_dist_values=args.umap_min_dist,
        reuse_embeddings=args.reuse_embeddings,
        use_knn=args.use_knn,
        knn_k_values=args.knn_k_values,
        reference_samples_per_class=args.reference_samples,
        evaluate_baselines=args.evaluate_baselines,
        baseline_test_size=args.baseline_test_size,
        baseline_max_samples=args.baseline_max_samples
    )
    
    logger.info("Training and evaluation complete!")


def run_cli():
    """Main CLI entry point"""
    # Parse arguments
    args = parse_args()
    
    try:
        # Setup logging first
        logger = setup_logging(args.output_dir)
        logger.info("Starting Siamese Neural Network Residual Analysis")
        logger.info(f"Arguments: {args}")
        
        # Run training
        run_training(args)
        
        logger.info("Analysis completed successfully!")
    except ImportError as ie:
        logger.error(f"Import error: {ie}. Please check your environment and installed packages.")
        logger.error("This may occur if optional dependencies are missing, like tsfresh or pytorch-related packages.")
        sys.exit(1)
    except FileNotFoundError as fnf:
        logger.error(f"File not found: {fnf}")
        logger.error("Please check that all required input files exist and are accessible.")
        sys.exit(1)
    except MemoryError as me:
        logger.error(f"Memory error: {me}")
        logger.error("The process ran out of memory. Try reducing batch size, num_triplets, or feature extraction parameters.")
        sys.exit(1)
    except RuntimeError as re:
        # Handle CUDA out of memory errors separately
        if "CUDA out of memory" in str(re):
            logger.error("CUDA out of memory error. Try using a smaller batch size, reducing model size, or running on CPU.")
            logger.error(f"Error details: {re}")
            sys.exit(1)
        elif "unable to open shared memory object" in str(re) or "Too many open files" in str(re):
            logger.error(f"System resource error: {re}")
            logger.error("Try reducing the number of worker processes (data_loader_workers or feature_workers).")
            sys.exit(1)
        else:
            logger.error(f"Runtime error: {re}")
            logger.exception("Detailed traceback:")
            logger.warning("Attempting to continue with fallback approaches...")
            # No sys.exit() here - let the function continue with fallbacks if possible
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.exception("Detailed traceback:")
        logger.warning("Attempting to continue with fallback approaches...")
        # No sys.exit() here - let the function continue with fallbacks if possible


if __name__ == "__main__":
    run_cli() 
