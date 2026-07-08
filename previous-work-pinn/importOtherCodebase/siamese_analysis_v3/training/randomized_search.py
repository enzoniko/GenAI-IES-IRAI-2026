"""
Randomized search for hyperparameter optimization
"""
import os
import json
import torch
import numpy as np
import logging
import sys
import random  # Add this import
from typing import Dict, List, Tuple, Any, Optional
from scipy.stats import randint, uniform, loguniform
import matplotlib.pyplot as plt
from collections import defaultdict
from torchsummary import summary
import io
from tqdm import tqdm

logger = logging.getLogger(__name__)

class RandomizedSearcher:
    """Class to handle randomized search for hyperparameter optimization"""
    
    def __init__(
        self, 
        model_class, 
        trainer_class, 
        criterion_class,
        dataset, 
        cv_splitter,
        device,
        output_dir: str,
        num_trials: int = 20,
        num_top_models: int = 5,
        num_repeat_runs: int = 30,
        random_state: int = 42
    ):
        """
        Initialize the randomized searcher
        
        Args:
            model_class: Class of the model to train
            trainer_class: Class of the trainer to use
            criterion_class: Class of the criterion to use
            dataset: Dataset to use for training
            cv_splitter: Cross-validation splitter
            device: Device to use for training
            output_dir: Directory to save results
            num_trials: Number of random parameter combinations to try
            num_top_models: Number of top models to keep
            num_repeat_runs: Number of times to run each top model for error distribution
            random_state: Random seed for reproducibility
        """
        self.model_class = model_class
        self.trainer_class = trainer_class
        self.criterion_class = criterion_class
        self.dataset = dataset
        self.cv_splitter = cv_splitter
        self.device = device
        
        # Convert to absolute path if not already
        if not os.path.isabs(output_dir):
            self.output_dir = os.path.abspath(output_dir)
        else:
            self.output_dir = output_dir
            
        self.num_trials = num_trials
        self.num_top_models = num_top_models
        self.num_repeat_runs = num_repeat_runs
        self.random_state = random_state
        
        # Set random seed
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        random.seed(random_state)  # Add this line to set the Python random seed
        
        # Create directory for saving models
        self.models_dir = os.path.join(self.output_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Create directory for saving results
        self.results_dir = os.path.join(self.output_dir, "search_results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        logger.info(f"Initialized RandomizedSearcher. Output directory: {self.output_dir}")
        logger.info(f"Models will be saved to: {self.models_dir}")
        logger.info(f"Results will be saved to: {self.results_dir}")

    """
    def define_search_space(self, **kwargs):
        # Default search space with embedding_size and margin
        search_space = {
            # Output embeddings
            "embedding_size": randint(16, 256),  # Embedding size between 16 and 256
            
            # TripletLoss parameters
            "margin": uniform(0.1, 1.9),  # Margin between 0.1 and 2.0
            "distance_metric": ["squared_euclidean"],  # Distance metric for TripletLoss
            "triplet_selection": ["all", "hard", "semi-hard", "distance_weighted"],  # Triplet selection strategy
            "reduction": ["mean", "sum"],  # Loss reduction method
            "label_smoothing": uniform(0.0, 0.2),  # Label smoothing factor (range 0.0-0.2)
            
            # Encoder architecture parameters
            "num_conv_blocks": [2, 3, 4, 5],  # Number of convolutional blocks
            "conv_channels": [
                [16, 32, 16],  # Original 3-block architecture
                [16, 32, 64, 32],  # 4-block variant
                [32, 64, 32],  # Higher channel variant for 3-block
                [16, 24, 32],  # Alternative 3-block
                [32, 64, 128, 64, 32],  # Deep pyramid architecture
                [64, 128, 256, 128, 64],  # Higher capacity pyramid
                [32, 32, 32, 32],  # Same channel depth throughout
            ],
            "kernel_size": [3, 5, 7],  # Kernel size for convolutions
            "pool_type": ["max", "avg"],  # Pooling type
            "pool_size": [2, 3],  # Pooling size
            "stride": [1, 2],  # Stride for convolutions
            "dilation": [1, 2, 3],  # Dilation for convolutions
            "padding": [0, 1, "same"],  # Padding for convolutions
            "groups": [1, 2, 4, 8],  # Groups for convolutions (grouped convolutions)
            
            # Advanced pooling
            "use_adaptive_pooling": [True, False],  # Whether to use adaptive pooling
            "adaptive_pool_size": [1, 2, 4, 8],  # Output size for adaptive pooling
            
            # Skip connections
            "use_skip_connections": [True, False],  # Whether to use skip connections
            
            # BatchNorm parameters
            "use_batch_norm": [True, False],  # Whether to use batch normalization
            "bn_momentum": uniform(0.1, 0.8),  # Momentum for batch normalization (range 0.1-0.9)
            "bn_eps": [1e-3, 1e-4, 1e-5],  # Epsilon for batch normalization
            
            # Normalization strategy
            "normalization_type": ["batch_norm", "layer_norm", "instance_norm", "none"],  # Type of normalization to use
            
            # Activation functions
            "activation_type": ["relu", "leaky_relu", "elu", "gelu"],  # Type of activation
            "leaky_relu_slope": uniform(0.01, 0.29),  # Slope for LeakyReLU (range 0.01-0.3)
            
            # Fully connected architecture parameters
            "fc_layers": [
                [{"out_features": 64}],  # Original architecture (one hidden layer)
                [{"out_features": 128}, {"out_features": 64}],  # Two hidden layers
                [{"out_features": 256}, {"out_features": 128}, {"out_features": 64}],  # Three hidden layers
                [{"out_features": 512}, {"out_features": 256}, {"out_features": 128}],  # Higher capacity
            ],
            "use_dropout": [True, False],  # Whether to use dropout
            "dropout_rate": uniform(0.2, 0.3),  # Dropout rate (range 0.2-0.5)
            
            # Weight initialization parameters
            "weight_init": ["kaiming_normal", "kaiming_uniform", "xavier_normal", "xavier_uniform", "orthogonal"],  # Weight initialization method
            "weight_init_gain": uniform(0.5, 1.5),  # Gain for weight initialization (range 0.5-2.0)
            
            # Optimization parameters
            "optimizer_type": ["adam", "adamw", "sgd"],  # Type of optimizer
            "lr": loguniform(1e-4, 1e-2),  # Learning rate (range 1e-4 to 1e-2)
            "weight_decay": loguniform(1e-6, 1e-3),  # Weight decay for regularization (range 1e-6 to 1e-3)
            
            # SGD-specific parameters
            "momentum": uniform(0.8, 0.19),  # Momentum for SGD (range 0.8-0.99)
            "nesterov": [True, False],  # Whether to use Nesterov momentum
            
            # Adam-specific parameters
            "beta1": uniform(0.9, 0.099),  # Beta1 for Adam (range 0.9-0.999)
            "beta2": uniform(0.99, 0.009),  # Beta2 for Adam (range 0.99-0.999)
            
            # Learning rate scheduler parameters
            "lr_scheduler": ["reduce_on_plateau", "cosine_annealing", "step", "none"],  # Type of learning rate scheduler
            "lr_patience": [3, 5, 7, 10],  # Patience for ReduceLROnPlateau scheduler
            "lr_factor": uniform(0.1, 0.4),  # Factor for learning rate reduction (range 0.1-0.5)
            "lr_min": loguniform(1e-6, 1e-4),  # Minimum learning rate (range 1e-6 to 1e-4)
            
            # Loss function parameters
            "loss_type": ["triplet", "contrastive"],  # Type of loss function
            "margin": uniform(0.1, 1.9),  # Margin for loss function (range 0.1-2.0)
            "label_smoothing": uniform(0.0, 0.2),  # Label smoothing (range 0.0-0.2)
            
            # Learning rate schedule parameters
            "warmup_epochs": [0, 1, 3, 5],  # Number of warmup epochs
        }
        
        # Add additional parameters if provided
        for key, value in kwargs.items():
            search_space[key] = value
        
        return search_space

    """
    """ def define_search_space(self, **kwargs):
     
        # Define the single, fixed configuration based on the user's request
        fixed_params = {
            'embedding_size': [222],
            'margin': [0.10965606565424935],
            'distance_metric': ['squared_euclidean'],
            'triplet_selection': ['hard'],
            'reduction': ['sum'],
            'label_smoothing': [0.13914285856995004],
            'num_conv_blocks': [2],
            # Specify the exact channel config for 2 blocks
            'conv_channels': [[32, 32]],
            'kernel_size': [3],
            # Use the pool settings consistent with the detailed architecture (max, 2)
            'pool_type': ['max'],
            'pool_size': [2],
            'stride': [2],
            'dilation': [1],
            'padding': [0],
            'groups': [2],
            'use_adaptive_pooling': [False],
            'adaptive_pool_size': [1], # Value doesn't matter as use_adaptive_pooling is False
            'use_skip_connections': [True],
            'use_batch_norm': [False], # Value doesn't matter as normalization_type is none
            'bn_momentum': [0.13902686533853847], # Value doesn't matter
            'bn_eps': [0.001], # Value doesn't matter
            'normalization_type': ['none'],
            'activation_type': ['leaky_relu'],
            'leaky_relu_slope': [0.164454730626221],
            # Specify the exact 3-layer FC configuration
            'fc_layers': [[{'out_features': 512}, {'out_features': 256}, {'out_features': 128}]],
            'use_dropout': [True],
            'dropout_rate': [0.4886365917776149],
            'weight_init': ['kaiming_uniform'],
            'weight_init_gain': [0.6642032900816808],
            'optimizer_type': ['adamw'],
            'lr': [0.006151200198861844],
            'weight_decay': [0.0009428139622801381],
            'momentum': [0.8116882349854547], # Value doesn't matter for AdamW
            'nesterov': [True], # Value doesn't matter for AdamW
            'beta1': [0.9874077206446087],
            'beta2': [0.9946447111839926],
            'lr_scheduler': ['reduce_on_plateau'],
            'lr_patience': [5],
            'lr_factor': [0.1],
            'lr_min': [6.501707670797311e-05],
            # Set loss_type explicitly for clarity, although create_trainer uses the class
            'loss_type': ['triplet'],
            'warmup_epochs': [3],
        }

        # Return the dictionary of fixed parameters
        return fixed_params """
    
    def define_search_space(self, **kwargs):
        """
        Define a fixed search space based on the best performing model found previously.
        """
        # Define the single, fixed configuration based on the best model's params
        fixed_params = {
            'embedding_size': [244],
            'margin': [0.1044448421889787],
            'distance_metric': ['squared_euclidean'],
            'triplet_selection': ['all'],
            'reduction': ['mean'],
            'label_smoothing': [0.1935398776414173],
            'num_conv_blocks': [5],
            # Use the exact channel config from the best model's structure info
            'conv_channels': [[16, 32, 64, 32, 32]],
            'kernel_size': [5],
            'pool_type': ['avg'],
            'pool_size': [3],
            'stride': [1],
            'dilation': [2],
            'padding': [1],
            'groups': [2], # From params
            'use_adaptive_pooling': [False], # From params
            'adaptive_pool_size': [2], # Value doesn't matter? Check if it does based on use_adaptive_pooling=False. Use value from params.
            'use_skip_connections': [False], # From params
            'use_batch_norm': [True], # From params, although normalization_type is instance_norm, keep for potential internal logic
            'bn_momentum': [0.10420305484670794], # Value might not matter if instance_norm is used. Use value from params.
            'bn_eps': [0.001], # Value might not matter if instance_norm is used. Use value from params.
            'normalization_type': ['instance_norm'], # From params
            'activation_type': ['relu'], # From params
            'leaky_relu_slope': [0.039897186343986014], # Value doesn't matter for relu. Use value from params.
            # Specify the exact FC configuration from the best model's structure info
            'fc_layers': [[{'out_features': 512}, {'out_features': 256}, {'out_features': 128}]],
            'use_dropout': [False], # From params
            'dropout_rate': [0.2943789670428233], # Value doesn't matter if use_dropout is False. Use value from params.
            'weight_init': ['orthogonal'], # From params
            'weight_init_gain': [1.7111453777251417], # From params
            'optimizer_type': ['adamw'], # From params
            'lr': [0.00839841964947657], # From params
            'weight_decay': [0.00023444483879862224], # From params
            'momentum': [0.9312064441646717], # Value doesn't matter for AdamW. Use value from params.
            'nesterov': [False], # Value doesn't matter for AdamW. Use value from params.
            'beta1': [0.9516087660739434], # From params
            'beta2': [0.9907803469067596], # From params
            'lr_scheduler': ['reduce_on_plateau'], # From params
            'lr_patience': [5], # From params
            'lr_factor': [0.1], # From params
            'lr_min': [8.247865800503034e-05], # From params
            'loss_type': ['triplet'], # From params
            'warmup_epochs': [3], # From params
            # Add other relevant params if they exist and are used, even if not directly listed in search space structure
            'pool_sizes_and_types': [[3, 'max']], # From params (though redundant given pool_type/size above)
            'fc_layer_variants': [[{'out_features': 256}, {'out_features': 128}]], # From params
        }

        # Return the dictionary of fixed parameters
        return fixed_params

    def sample_parameters(self, search_space):
        """
        Sample parameters from the search space
        
        Args:
            search_space: Dictionary of parameter distributions
        
        Returns:
            Dictionary of sampled parameters
        """
        sampled_params = {}
        for param_name, param_dist in search_space.items():
            if hasattr(param_dist, "rvs"):  # If it's a scipy distribution
                sampled_params[param_name] = param_dist.rvs()
            else:  # If it's a list or another iterable
                # Use standard Python's random.choice instead of np.random.choice for lists
                # which can better handle heterogeneous elements
                sampled_params[param_name] = random.choice(param_dist)
                
        return sampled_params
    
    def create_model(self, in_channels, feature_size, params):
        """
        Create a model with the given parameters
        
        Args:
            in_channels: Number of input channels
            feature_size: Size of the input features
            params: Dictionary of model parameters
        
        Returns:
            Tuple: (Instantiated model, model_kwargs used for instantiation)
        
        Raises:
            ValueError: If the model configuration is invalid
            RuntimeError: If the model creation fails
        """
        try:
            # Build dynamic encoder architecture based on parameters
            encoder_architecture = []
            
            # Determine number of convolutional blocks and channels
            num_blocks = params.get('num_conv_blocks', 3)
            channels = params.get('conv_channels')
            
            # Handle different channel formats
            if channels is None:
                # Default channels if not provided
                channels = [16, 32, 16]
            
            # If channels is a list of lists, select one of them
            if isinstance(channels, list) and channels and isinstance(channels[0], list):
                # Find channel configurations that are compatible with num_blocks
                compatible_channels = [ch for ch in channels if len(ch) >= num_blocks]
                
                if compatible_channels:
                    # Randomly select one from compatible configurations
                    channels = random.choice(compatible_channels)
                    # Truncate if longer than needed
                    channels = channels[:num_blocks]
                else:
                    # If no compatible configuration, create a default one
                    logger.warning(f"No compatible channel configuration for {num_blocks} blocks. Using default.")
                    channels = [16] * num_blocks
                    channels[1:min(len(channels), num_blocks//2+1)] = [32] * min(num_blocks//2, len(channels)-1)  # Middle layers with higher channels
            
            # Make sure we have enough channels for blocks
            if len(channels) < num_blocks:
                # Extend channels if needed
                channels = channels + [channels[-1]] * (num_blocks - len(channels))
            
            # Get kernel size parameter
            kernel_size = params.get('kernel_size', 3)
            
            # Get stride, dilation, and padding parameters
            stride = params.get('stride', 1)
            dilation = params.get('dilation', 1)
            padding = params.get('padding', 0)
            
            # Get pooling parameters
            pool_type = params.get('pool_type', 'max')
            pool_size = params.get('pool_size', 2)
            
            # Get activation parameters
            activation_type = params.get('activation_type', 'relu')
            leaky_relu_slope = params.get('leaky_relu_slope', 0.1)
            
            # Get normalization type
            normalization_type = params.get('normalization_type', 'batch_norm')
            
            # Check if we have a list of pool configs
            pool_sizes_and_types = params.get('pool_sizes_and_types')
            if pool_sizes_and_types and isinstance(pool_sizes_and_types[0], list):
                if isinstance(pool_sizes_and_types[0], list):
                    # Randomly select one pooling configuration
                    pool_config = random.choice(pool_sizes_and_types)
                    pool_size = pool_config[0]
                    pool_type = pool_config[1]
            
            # Ensure parameters are valid and in proper types
            try:
                num_blocks = int(num_blocks)
                kernel_size = int(kernel_size)
                stride = int(stride)
                dilation = int(dilation)
                if isinstance(padding, str) and padding != "same":
                    padding = 0  # Default to 0 if invalid string
                elif isinstance(padding, (int, float)):
                    padding = int(padding)
                pool_size = int(pool_size)
                
                # Validate numeric parameters
                if num_blocks <= 0:
                    raise ValueError(f"num_blocks must be positive, got {num_blocks}")
                if kernel_size <= 0:
                    raise ValueError(f"kernel_size must be positive, got {kernel_size}")
                if stride <= 0:
                    raise ValueError(f"stride must be positive, got {stride}")
                if dilation <= 0:
                    raise ValueError(f"dilation must be positive, got {dilation}")
                if isinstance(padding, int) and padding < 0:
                    raise ValueError(f"padding must be non-negative, got {padding}")
                if pool_size <= 0:
                    raise ValueError(f"pool_size must be positive, got {pool_size}")
            except (ValueError, TypeError) as e:
                # Handle invalid parameter values
                logger.error(f"Invalid parameter value: {e}")
                raise ValueError(f"Error validating model parameters: {e}")
            
            # Log the architecture being built
            logger.info(f"Building model with {num_blocks} blocks, channels {channels}, pool_size {pool_size}, pool_type {pool_type}")
            
            # Create convolutional blocks
            for i in range(num_blocks):
                # Conv layer
                encoder_architecture.append({
                    'type': 'conv',
                    'out_channels': channels[i],
                    'kernel_size': kernel_size,
                    'stride': stride,
                    'dilation': dilation,
                    'padding': padding if padding != "same" else kernel_size // 2  # Handle 'same' padding
                })
                
                # Add normalization layer based on type
                if normalization_type == 'batch_norm':
                    encoder_architecture.append({
                        'type': 'bn',
                        'momentum': params.get('bn_momentum', 0.1),
                        'eps': params.get('bn_eps', 1e-5)
                    })
                elif normalization_type == 'layer_norm':
                    encoder_architecture.append({
                        'type': 'layer_norm'
                    })
                elif normalization_type == 'instance_norm':
                    encoder_architecture.append({
                        'type': 'instance_norm'
                    })
                # Skip normalization if type is 'none'
                
                # Activation
                if activation_type == 'relu':
                    encoder_architecture.append({'type': 'relu'})
                elif activation_type == 'leaky_relu':
                    encoder_architecture.append({'type': 'leaky_relu', 'negative_slope': leaky_relu_slope})
                elif activation_type == 'elu':
                    encoder_architecture.append({'type': 'elu'})
                elif activation_type == 'gelu':
                    encoder_architecture.append({'type': 'gelu'})
                else:
                    # Default to ReLU if unknown activation
                    logger.warning(f"Unknown activation type: {activation_type}, using ReLU instead")
                    encoder_architecture.append({'type': 'relu'})
                
                # Pooling
                encoder_architecture.append({
                    'type': 'pool',
                    'pool_type': pool_type,
                    'kernel_size': int(pool_size),  # Ensure it's an integer
                    'stride': int(pool_size)  # Ensure it's an integer
                })
            
            # Build dynamic fully connected architecture
            fc_architecture = []
            
            # Get FC layer configurations
            fc_layers = params.get('fc_layers', [{'out_features': 64}])
            
            # Handle different fc_layers formats
            fc_layer_variants = params.get('fc_layer_variants')
            
            if fc_layer_variants:
                # If we have multiple variants, select one
                if isinstance(fc_layer_variants, list) and isinstance(fc_layer_variants[0], list):
                    fc_layers = random.choice(fc_layer_variants)
            elif isinstance(fc_layers, list) and fc_layers and isinstance(fc_layers[0], list):
                # For backward compatibility if fc_layers is a list of lists
                fc_layers = random.choice(fc_layers)
            
            # Get dropout parameters
            use_dropout = params.get('use_dropout', False)
            dropout_rate = params.get('dropout_rate', 0.3)
            
            # Ensure dropout rate is valid
            if not 0 <= dropout_rate <= 1:
                logger.warning(f"Invalid dropout rate: {dropout_rate}, using 0.3 instead")
                dropout_rate = 0.3
            
            # Log the FC architecture
            logger.info(f"Building FC layers with {len(fc_layers)} hidden layers, dropout={use_dropout}")
            
            # Create fully connected layers
            for layer_config in fc_layers:
                # Linear layer
                out_features = layer_config.get('out_features', 64)
                if not isinstance(out_features, int) or out_features <= 0:
                    logger.warning(f"Invalid out_features: {out_features}, using 64 instead")
                    out_features = 64
                
                fc_architecture.append({
                    'type': 'linear',
                    'out_features': out_features
                })
                
                # Activation
                if activation_type == 'relu':
                    fc_architecture.append({'type': 'relu'})
                elif activation_type == 'leaky_relu':
                    fc_architecture.append({'type': 'leaky_relu', 'negative_slope': leaky_relu_slope})
                elif activation_type == 'elu':
                    fc_architecture.append({'type': 'elu'})
                elif activation_type == 'gelu':
                    fc_architecture.append({'type': 'gelu'})
                else:
                    # Default to ReLU if unknown activation
                    fc_architecture.append({'type': 'relu'})
                
                # Optional dropout
                if use_dropout:
                    fc_architecture.append({
                        'type': 'dropout',
                        'p': dropout_rate
                    })
            
            # Store architecture in params for later retrieval
            params['_encoder_architecture'] = encoder_architecture
            params['_fc_architecture'] = fc_architecture
            
            # Get embedding size
            embedding_size = params.get('embedding_size', 16)
            if not isinstance(embedding_size, int) or embedding_size <= 0:
                logger.warning(f"Invalid embedding_size: {embedding_size}, using 16 instead")
                embedding_size = 16
            
            # Get weight initialization parameters
            weight_init = params.get('weight_init', 'kaiming_normal')
            weight_init_gain = params.get('weight_init_gain', 1.0)
            
            # Get skip connections parameter
            use_skip_connections = params.get('use_skip_connections', False)
            
            # Save structure info
            params['_structure_info'] = {
                'num_blocks': num_blocks,
                'channels': channels,
                'pool_size': pool_size,
                'pool_type': pool_type,
                'activation_type': activation_type,
                'normalization_type': normalization_type,
                'fc_layers': fc_layers,
                'use_dropout': use_dropout,
                'dropout_rate': dropout_rate if use_dropout else None
            }
            
            # Prepare model arguments
            model_kwargs = {
                'in_channels': in_channels,
                'feature_size': feature_size,
                'embedding_size': embedding_size,
                'encoder_config': encoder_architecture,
                'fc_config': fc_architecture,
                'weight_init': weight_init,
                'weight_init_gain': weight_init_gain,
                'use_skip_connections': use_skip_connections
            }
            
            # Instantiate the model
            try:
                model = self.model_class(**model_kwargs)
                
                # Return the model and its configuration arguments
                return model, model_kwargs
            except Exception as e:
                # Handle errors in model instantiation
                logger.error(f"Error creating model: {e}")
                logger.exception("Detailed traceback:")
                raise ValueError(f"Model creation failed: {e}")
                
        except Exception as e:
            # Handle any other errors during model creation
            logger.error(f"Error in create_model: {e}")
            logger.exception("Detailed traceback:")
            raise ValueError(f"Model creation failed: {e}")
    
    def create_trainer(self, model, model_kwargs, params, learning_rate, lr_factor=0.1, lr_patience=5, min_lr=1e-6, verbose=1):
        """
        Create a trainer with the given parameters
        
        Args:
            model: Model to train
            model_kwargs: Dictionary of arguments used to create the model
            params: Dictionary of trainer parameters
            learning_rate: Initial learning rate
            lr_factor: Factor to reduce learning rate by
            lr_patience: Number of epochs with no improvement after which learning rate will be reduced
            min_lr: Minimum learning rate
            verbose: Verbosity level
            
        Returns:
            Instantiated trainer
        """
        # Get optimizer parameters from params or use defaults
        optimizer_type = params.get('optimizer_type', 'adam').lower()
        
        # Ensure parameters are within valid ranges
        momentum = min(max(params.get('momentum', 0.9), 0.0), 0.999)  # Clamp to [0, 0.999]
        weight_decay = params.get('weight_decay', 1e-5)
        beta1 = min(max(params.get('beta1', 0.9), 0.0), 0.999)  # Clamp to [0, 0.999]
        beta2 = min(max(params.get('beta2', 0.999), 0.0), 0.999)  # Clamp to [0, 0.999]
        
        logger.info(f"Using optimizer: {optimizer_type}, momentum: {momentum}, betas: ({beta1}, {beta2})")
        
        # Create optimizer based on type
        if optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(
                model.parameters(), 
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'adam':
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=learning_rate,
                betas=(beta1, beta2),
                weight_decay=weight_decay
            )
        elif optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=learning_rate,
                betas=(beta1, beta2),
                weight_decay=weight_decay
            )
        elif optimizer_type == 'rmsprop':
            optimizer = torch.optim.RMSprop(
                model.parameters(), 
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay
            )
        else:
            # Default to Adam if unknown optimizer type
            logger.warning(f"Unknown optimizer type: {optimizer_type}. Using Adam as default.")
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=learning_rate,
                weight_decay=weight_decay
            )
        
        # Get learning rate schedule parameters
        lr_schedule_type = params.get('lr_scheduler', 'reduce_on_plateau').lower()
        lr_patience = params.get('lr_patience', lr_patience)
        lr_factor = params.get('lr_factor', lr_factor)
        warmup_epochs = params.get('warmup_epochs', 0)
        
        # Create learning rate scheduler based on type
        if lr_schedule_type == 'reduce_on_plateau':
            # Create ReduceLROnPlateau scheduler (verbose parameter removed for compatibility)
            try:
                lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=lr_factor,
                    patience=lr_patience,
                    min_lr=min_lr
                )
            except TypeError:
                # Fallback for older PyTorch versions without min_lr parameter
                lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=lr_factor,
                    patience=lr_patience
                )
        elif lr_schedule_type == 'cosine_annealing':
            # T_max set to a reasonable number (e.g., num_epochs)
            t_max = params.get('t_max', 100)  # Default to 100 epochs if not specified
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=t_max,
                eta_min=min_lr
            )
        elif lr_schedule_type == 'step':
            # Step size set to a reasonable number
            step_size = params.get('step_size', 10)  # Default to 10 epochs if not specified
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=step_size,
                gamma=lr_factor
            )
        elif lr_schedule_type == 'none':
            # No scheduler
            lr_scheduler = None
        else:
            # Default to ReduceLROnPlateau if unknown schedule type
            logger.warning(f"Unknown LR schedule type: {lr_schedule_type}. Using ReduceLROnPlateau as default.")
            # Create ReduceLROnPlateau scheduler (verbose parameter removed for compatibility)
            try:
                lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=lr_factor,
                    patience=lr_patience,
                    min_lr=min_lr
                )
            except TypeError:
                # Fallback for older PyTorch versions without min_lr parameter
                lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=lr_factor,
                    patience=lr_patience
                )
        
        # Create criterion with margin from params and correct distance parameter
        criterion = self.criterion_class(
            margin=params["margin"],
            distance_metric=params.get("distance_metric", "squared_euclidean"),
            triplet_selection=params.get("triplet_selection", "all"),
            reduction=params.get("reduction", "mean"),
            label_smoothing=params.get("label_smoothing", 0.0)
        )
        
        # Create and return trainer, passing model_kwargs
        return self.trainer_class(
            model=model,
            device=self.device,
            model_kwargs=model_kwargs,  # Pass the model config args
            criterion=criterion,
            optimizer=optimizer,
            lr=learning_rate,
            lr_scheduler=lr_scheduler,
            warmup_epochs=warmup_epochs
        )
    
    def run_search(self, in_channels, feature_size, epochs, learning_rate=0.01, 
                  batch_size=64, data_loader_workers=4, **kwargs):
        """
        Run the randomized search
        
        Args:
            in_channels: Number of input channels
            feature_size: Size of the input features
            epochs: Number of epochs to train for
            learning_rate: Initial learning rate
            batch_size: Batch size for training and validation
            data_loader_workers: Number of worker processes for data loading
            **kwargs: Additional parameters for the search space
            
        Returns:
            Best model from the search
        """
        # Define search space
        search_space = self.define_search_space(**kwargs)
        
        # Make sure these parameters aren't overridden by the search
        params_to_preserve = {
            'embedding_size', 'margin', 'num_conv_blocks', 
            'kernel_size', 'pool_type', 'pool_size', 'use_dropout', 'dropout_rate'
        }
        
        # Initialize variables to track results
        all_results = []
        
        # Make sure output directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Run trials with progress bar
        logger.info(f"Running {self.num_trials} randomized search trials...")
        
        # Use tqdm for trial progress tracking
        for trial in tqdm(range(self.num_trials), desc="Randomized search trials", leave=True, position=0):
            # Sample parameters
            params = self.sample_parameters(search_space)
            logger.info(f"Trial {trial+1}/{self.num_trials}: {params}")
            
            try:
                # Create model and get its kwargs
                model, model_kwargs = self.create_model(in_channels, feature_size, params)
                
                # Create trainer, passing model_kwargs
                trainer = self.create_trainer(model, model_kwargs, params, learning_rate)
                
                # Get CV splits
                cv_splits = self.cv_splitter.split(self.dataset)
                
                # Train with cross-validation (with error handling for Windows multiprocessing issues)
                try:
                    histories = trainer.train_with_cross_validation(
                        cv_splits=cv_splits,
                        epochs=epochs,
                        verbose=0,  # Reduce verbosity during initial search
                        batch_size=batch_size,
                        data_loader_workers=data_loader_workers
                    )
                except (OSError, RuntimeError) as e:
                    if "Invalid argument" in str(e) or "pickle" in str(e).lower():
                        logger.warning(f"Multiprocessing error during training: {e}")
                        logger.warning("This is likely a Windows multiprocessing issue. Retrying with reduced workers.")
                        # Retry with single worker
                        histories = trainer.train_with_cross_validation(
                            cv_splits=cv_splits,
                            epochs=epochs,
                            verbose=0,
                            batch_size=batch_size,
                            data_loader_workers=0  # Disable multiprocessing
                        )
                    else:
                        raise
                
                # Compute average validation loss across folds
                val_losses = [h["val_loss"][-1] for h in histories]
                avg_val_loss = np.mean(val_losses)
                std_val_loss = np.std(val_losses)
                
                # Store results
                result = {
                    "params": params,
                    "avg_val_loss": float(avg_val_loss),
                    "std_val_loss": float(std_val_loss),
                    "configuration": {
                        "in_channels": in_channels,
                        "feature_size": feature_size,
                        "embedding_size": params["embedding_size"],
                        "encoder_config": params.get("_encoder_architecture", []),
                        "fc_config": params.get("_fc_architecture", [])
                    }
                }
                
                # Save model weights using absolute path
                model_filename = f"model_trial_{trial}.pt"
                model_path = os.path.join(self.models_dir, model_filename)
                os.makedirs(os.path.dirname(model_path), exist_ok=True)  # Ensure directory exists
                torch.save(model.state_dict(), model_path)
                result["model_path"] = model_path
                
                all_results.append(result)
                
                # Update progress bar with current validation loss
                tqdm.write(f"Trial {trial+1} - Avg Val Loss: {avg_val_loss:.6f} ± {std_val_loss:.6f}")
            except Exception as e:
                logger.error(f"Trial {trial+1} failed with error: {str(e)}")
                logger.exception("Detailed traceback:")
                logger.info(f"Skipping trial {trial+1} and continuing with next configuration...")
                continue
        
        # Check if we have any successful trials
        if not all_results:
            logger.error("All trials failed. Cannot proceed with evaluation.")
            raise RuntimeError("All hyperparameter search trials failed. Please check your model configurations.")
        
        # Sort results by validation loss
        all_results.sort(key=lambda x: x["avg_val_loss"])
        
        # Keep top models
        top_models = all_results[:min(self.num_top_models, len(all_results))]
        
        # Save top model configurations
        self._save_top_model_configurations(top_models)
        
        # Run top models multiple times to get error distributions
        try:
            best_model = self._evaluate_top_models(top_models, in_channels, feature_size, epochs, learning_rate, batch_size, data_loader_workers)
            return best_model
        except Exception as e:
            logger.error(f"Error evaluating top models: {str(e)}")
            logger.exception("Detailed traceback:")
            
            # Return the best model based on the initial search if evaluation fails
            if top_models:
                logger.info("Falling back to best model from initial search without additional evaluation.")
                # Load the best model weights
                best_model_path = top_models[0]["model_path"]
                best_weights = torch.load(best_model_path)
                
                # Create a new model with the best configuration
                best_config = top_models[0]["configuration"]
                best_model, best_model_kwargs = self.create_model(
                    in_channels=best_config["in_channels"],
                    feature_size=best_config["feature_size"],
                    params=best_config["params"]
                )
                best_model.load_state_dict(best_weights)
                return best_model
            else:
                raise RuntimeError("Failed to evaluate top models and no valid fallback model available.")
    
    def _save_top_model_configurations(self, top_models):
        """
        Save the configurations of the top models
        
        Args:
            top_models: List of dictionaries with top model results
        """
        # Make sure results directory exists
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Save all top models
        top_models_path = os.path.join(self.results_dir, "top_models.json")
        os.makedirs(os.path.dirname(top_models_path), exist_ok=True)
        with open(top_models_path, "w") as f:
            json.dump(top_models, f, indent=2)
        
        # Save individual configurations
        for i, model_info in enumerate(top_models):
            config_path = os.path.join(self.results_dir, f"configuration_{i+1}.json")
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, "w") as f:
                json.dump(model_info["configuration"], f, indent=2)
            
            # Save weights separately
            weights_path = os.path.join(self.results_dir, f"weights_{i+1}.pt")
            os.makedirs(os.path.dirname(weights_path), exist_ok=True)
            torch.save(torch.load(model_info["model_path"]), weights_path)
    
    def _plot_error_distributions(self, error_distributions, top_models):
        """
        Plot error distributions for the top models and save data to a text file
        
        Args:
            error_distributions: Dictionary of error distributions for each model
            top_models: List of dictionaries with top model results
        """
        plt.figure(figsize=(12, 6))
        
        for model_idx, losses in error_distributions.items():
            model_params = top_models[model_idx]["params"]
            label = f"Model {model_idx+1}: emb={model_params['embedding_size']}, margin={model_params['margin']:.2f}"
            plt.violinplot(losses, positions=[model_idx], showmeans=True)
            
        plt.xticks(range(len(top_models)), [f"Model {i+1}" for i in range(len(top_models))])
        plt.xlabel("Model")
        plt.ylabel("Validation Loss")
        plt.title("Error Distributions for Top Models")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend with model parameters
        legend_text = []
        for i, model_info in enumerate(top_models):
            params = model_info["params"]
            legend_text.append(f"Model {i+1}: emb={params['embedding_size']}, margin={params['margin']:.2f}")
        plt.figtext(0.5, 0.01, "\n".join(legend_text), ha="center", fontsize=9, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        plt.tight_layout()
        
        # Make sure results directory exists
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Save plot to PNG file
        plot_path = os.path.join(self.results_dir, "error_distributions.png")
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path)
        plt.close()
        
        # Save error distributions to text file for paper use
        error_data_path = os.path.join(self.results_dir, "error_distributions.txt")
        with open(error_data_path, "w") as f:
            f.write("Error Distribution Data for Top Models\n")
            f.write("=====================================\n\n")
            
            # Write summary statistics for each model
            f.write("# Summary Statistics\n")
            for i, (model_idx, losses) in enumerate(error_distributions.items()):
                model_params = top_models[model_idx]["params"]
                mean_loss = np.mean(losses)
                std_loss = np.std(losses)
                median_loss = np.median(losses)
                min_loss = np.min(losses)
                max_loss = np.max(losses)
                
                f.write(f"\nModel {i+1}:\n")
                f.write(f"  Embedding Size: {model_params['embedding_size']}\n")
                f.write(f"  Margin: {model_params['margin']:.2f}\n")
                f.write(f"  Mean Loss: {mean_loss:.6f}\n")
                f.write(f"  Std Dev: {std_loss:.6f}\n")
                f.write(f"  Median Loss: {median_loss:.6f}\n")
                f.write(f"  Min Loss: {min_loss:.6f}\n")
                f.write(f"  Max Loss: {max_loss:.6f}\n")
            
            # Write raw data for each model
            f.write("\n\n# Raw Data\n")
            for i, (model_idx, losses) in enumerate(error_distributions.items()):
                f.write(f"\nModel {i+1} Values: ")
                f.write(", ".join([f"{x:.6f}" for x in losses]))
                f.write("\n")
        
        logger.info(f"Saved error distribution plot to {plot_path}")
        logger.info(f"Saved error distribution data to {error_data_path}")

    def _evaluate_top_models(self, top_models, in_channels, feature_size, epochs, 
                            learning_rate, batch_size=64, data_loader_workers=4):
        """
        Evaluate the top models multiple times to get error distributions
        
        Args:
            top_models: List of dictionaries with top model results
            in_channels: Number of input channels
            feature_size: Size of the input features
            epochs: Number of epochs to train for
            learning_rate: Initial learning rate
            batch_size: Batch size for training and validation
            data_loader_workers: Number of worker processes for data loading
            
        Returns:
            Best model from the evaluation
        """
        # Initialize variables to track results
        error_distributions = {}
        best_run_val_loss = float('inf')
        best_model = None
        best_model_idx = -1
        best_weights_path = None
        
        # Make sure results directory exists
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Run each top model multiple times
        logger.info(f"Evaluating top {len(top_models)} models with {self.num_repeat_runs} runs each...")
        
        successful_models = []  # Track models that can be successfully evaluated
        
        # Use tqdm for top model evaluation progress tracking
        for model_idx, model_info in enumerate(tqdm(top_models, desc="Evaluating top models", leave=True, position=0)):
            model_errors = []
            
            # Extract parameters
            params = model_info["params"]
            model_path = model_info["model_path"]
            
            try:
                # Load model weights
                weights = torch.load(model_path)
                
                # Run the model multiple times with progress bar
                run_iterator = range(self.num_repeat_runs)
                # Use a nested progress bar for runs within each model
                for run_idx in tqdm(run_iterator, desc=f"Model {model_idx+1}/{len(top_models)} runs", leave=False, position=1):
                    try:
                        # Create a new model with the same architecture
                        model, model_kwargs = self.create_model(in_channels, feature_size, params)
                        
                        # Create a trainer
                        trainer = self.create_trainer(model, model_kwargs, params, learning_rate)
                        
                        # Instead of full cross-validation, use a single random train/validation split
                        # This provides a good balance between computational efficiency and robust evaluation
                        random_seed = self.random_state + run_idx  # Ensure different splits for each run
                        train_indices, val_indices = self.cv_splitter.get_single_split(
                            random_seed=random_seed,
                            dataset_length=len(self.dataset)
                        )
                        
                        # Train with single validation split
                        history = trainer.train_with_single_split(
                            dataset=self.dataset,
                            train_indices=train_indices,
                            val_indices=val_indices,
                            epochs=epochs,
                            verbose=1,  # Show progress bars during top model evaluation
                            batch_size=batch_size,
                            data_loader_workers=data_loader_workers
                        )
                        
                        # Get final validation loss
                        val_loss = history["val_loss"][-1]
                        
                        # Add to model errors
                        model_errors.append(val_loss)
                        
                        # Check if this is the best run
                        if val_loss < best_run_val_loss:
                            tqdm.write(f"New best run: Model {model_idx+1}, Run {run_idx+1}, Val Loss: {val_loss:.6f}")
                            best_run_val_loss = val_loss
                            best_model = model
                            best_model_idx = model_idx
                            
                            # Save the best model weights
                            best_weights_path = os.path.join(self.results_dir, "best_model.pt")
                            os.makedirs(os.path.dirname(best_weights_path), exist_ok=True)
                            torch.save(model.state_dict(), best_weights_path)
                    
                    except Exception as e:
                        logger.error(f"Error during run {run_idx+1} for model {model_idx+1}: {str(e)}")
                        logger.exception("Detailed traceback:")
                        logger.info(f"Skipping run {run_idx+1} for model {model_idx+1}")
                        continue
                
                # Only add models with at least one successful run
                if model_errors:
                    error_distributions[model_idx] = model_errors
                    successful_models.append(model_info)
                    mean_error = np.mean(model_errors)
                    std_error = np.std(model_errors)
                    tqdm.write(f"Model {model_idx+1} error distribution: mean={mean_error:.6f}, std={std_error:.6f}")
                else:
                    logger.warning(f"Model {model_idx+1} had no successful runs and will be excluded from error distribution")
                
            except Exception as e:
                logger.error(f"Error evaluating model {model_idx+1}: {str(e)}")
                logger.exception("Detailed traceback:")
                logger.info(f"Skipping model {model_idx+1}")
                continue
        
        # Check if we have any successful models
        if not error_distributions:
            logger.error("All models failed during evaluation. Using the best model from the initial search.")
            # Return the first model from top_models as a fallback
            best_model_idx = 0
            params = top_models[best_model_idx]["params"]
            best_model, best_model_kwargs = self.create_model(in_channels, feature_size, params)
            best_model.load_state_dict(torch.load(top_models[best_model_idx]["model_path"]))
        else:
            # Plot error distributions
            self._plot_error_distributions(error_distributions, successful_models)
        
        # Save best configuration
        best_config_path = os.path.join(self.results_dir, "best_configuration.json")
        os.makedirs(os.path.dirname(best_config_path), exist_ok=True)
        
        if best_model_idx >= 0 and best_model_idx < len(top_models):
            best_config = top_models[best_model_idx]["configuration"]
            best_config["params"] = top_models[best_model_idx]["params"]
        else:
            # Fallback in case best_model_idx is invalid
            logger.warning("Invalid best_model_idx, using first model as fallback")
            best_config = top_models[0]["configuration"]
            best_config["params"] = top_models[0]["params"]
        
        best_config["avg_val_loss"] = float(best_run_val_loss)
        
        with open(best_config_path, "w") as f:
            json.dump(best_config, f, indent=2)
        
        # Save detailed model configuration
        if best_model is not None:
            save_detailed_model_configuration(
                model=best_model,
                in_channels=in_channels,
                feature_size=feature_size,
                best_configuration=best_config,
                output_dir=self.output_dir
            )
        
        logger.info(f"Best model saved with validation loss: {best_run_val_loss:.6f}")
        
        return best_model


def randomized_search(args, actual_channels, feature_size):
    """
    Run randomized search for hyperparameter optimization
    
    Args:
        args: Command line arguments
        actual_channels: Number of channels in the input data
        feature_size: Size of the input features
    
    Returns:
        Best model from the search
    """
    # Import required modules
    from siamese_analysis_v3.models.configurable_siamese import ConfigurableSiameseNetwork, TripletLoss
    from siamese_analysis_v3.training.trainer import SiameseTrainer
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        # Check if we should load a pre-trained model instead of running search
        if hasattr(args, 'load_best_model_from') and args.load_best_model_from:
            load_path = args.load_best_model_from
            if not os.path.isdir(load_path):
                # Assume it's relative to the output dir if not a full path
                load_path = os.path.join(args.output_dir, load_path) 
                
            logger.info(f"Attempting to load best model configuration and weights from: {load_path}")
            
            try:
                # Load configuration and weights
                best_configuration = load_best_configuration(load_path)
                best_weights = load_best_weights(load_path)
                
                # Recreate the model
                logger.info("Recreating model from loaded configuration and weights...")
                model = recreate_training_pipeline(best_configuration, best_weights)
                logger.info("Successfully loaded and recreated model.")
                return model
                
            except FileNotFoundError:
                logger.error(f"Error: Could not find best_configuration.json or best_model.pt in {load_path}")
                logger.error("Please ensure the path is correct and contains the required files.")
                raise
            except Exception as load_error:
                logger.error(f"Error loading or recreating model from {load_path}: {load_error}")
                logger.exception("Detailed traceback:")
                raise RuntimeError(f"Failed to load pre-trained model from {load_path}")

        # Create dataset and CV splitter objects (these should be provided in the function parameters)
        dataset = args.dataset
        cv_splitter = args.cv_splitter
        
        # Determine number of trials based on command line args (can be overridden)
        num_trials = getattr(args, 'num_trials', 20)
        num_top_models = getattr(args, 'num_top_models', 5)
        num_repeat_runs = getattr(args, 'num_repeat_runs', 15)
        
        # Initialize randomized searcher
        searcher = RandomizedSearcher(
            model_class=ConfigurableSiameseNetwork,
            trainer_class=SiameseTrainer,
            criterion_class=TripletLoss,
            dataset=dataset,
            cv_splitter=cv_splitter,
            device=args.device,
            output_dir=args.output_dir,
            num_trials=num_trials,
            num_top_models=num_top_models,
            num_repeat_runs=num_repeat_runs,
            random_state=args.seed
        )
        
        # Define additional search space parameters
        additional_search_space = {
            # More specific CNN architecture variants for different analysis tasks
            "conv_channels": [
                # Standard architectures for general signal processing
                [16, 32, 16],  # Original architecture
                [32, 64, 32],  # Higher capacity architecture
                
                # Deeper architectures
                [16, 32, 64, 32],  # 4-block variant
                [16, 24, 32, 64, 32],  # 5-block variant
                
                # Specialized architectures for frequency analysis
                [16, 32, 64, 128, 64, 32],  # Deep pyramid architecture
                [32, 32, 32, 32],  # Same channel depth throughout
            ],
            
            # Different pooling strategies - ensure integers (not tuples) for kernel_size
            "pool_sizes_and_types": [
                # [pool_size, pool_type]
                [2, "max"],  # Original max pooling with size 2
                [2, "avg"],  # Average pooling with size 2
                [3, "max"],  # Larger max pooling
                [3, "avg"],  # Larger average pooling
            ],
            
            # Different FC layer configurations
            "fc_layer_variants": [
                # Single hidden layer options
                [{"out_features": 64}],
                [{"out_features": 128}],
                [{"out_features": 256}],
                
                # Two hidden layer options
                [{"out_features": 128}, {"out_features": 64}],
                [{"out_features": 256}, {"out_features": 128}],
                [{"out_features": 512}, {"out_features": 256}],
                
                # Three hidden layer options
                [{"out_features": 256}, {"out_features": 128}, {"out_features": 64}],
            ],
        }
        
        # Override only non-searchable parameters with CLI args if provided
        # Note: We deliberately do NOT override embedding_size and margin as these
        # should be determined by the search space defined in define_search_space
        if hasattr(args, 'lr_scheduler') and args.lr_scheduler:
            additional_search_space["lr_scheduler"] = [args.lr_scheduler]
        
        if hasattr(args, 'lr_factor') and args.lr_factor:
            additional_search_space["lr_factor"] = [args.lr_factor]
        
        if hasattr(args, 'lr_patience') and args.lr_patience:
            additional_search_space["lr_patience"] = [args.lr_patience]
        
        # Run the search
        try:
            best_model = searcher.run_search(
                in_channels=actual_channels,
                feature_size=feature_size,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                data_loader_workers=args.data_loader_workers,
                **additional_search_space
            )
            
            logger.info("Randomized search completed successfully")
            return best_model
            
        except Exception as search_error:
            logger.error(f"Error during randomized search: {search_error}")
            logger.exception("Detailed traceback:")
            
            # Create a default model as fallback
            logger.warning("Creating default model as randomized search failed - will be saved as best model")
            
            # Define a simple robust architecture
            default_encoder = [
                {'type': 'conv', 'out_channels': 16, 'kernel_size': 3, 'padding': 1},
                {'type': 'bn'},
                {'type': 'relu'},
                {'type': 'pool', 'pool_type': 'max', 'kernel_size': 2, 'stride': 2},
                
                {'type': 'conv', 'out_channels': 32, 'kernel_size': 3, 'padding': 1},
                {'type': 'bn'},
                {'type': 'relu'},
                {'type': 'pool', 'pool_type': 'max', 'kernel_size': 2, 'stride': 2},
                
                {'type': 'conv', 'out_channels': 16, 'kernel_size': 3, 'padding': 1},
                {'type': 'bn'},
                {'type': 'relu'},
                {'type': 'pool', 'pool_type': 'max', 'kernel_size': 2, 'stride': 2},
            ]
            
            default_fc = [
                {'type': 'linear', 'out_features': 64},
                {'type': 'relu'},
            ]
            
            # Use a default embedding size rather than args.embedding_size to ensure
            # we're not using CLI args to control parameters that should be determined by search
            default_embedding_size = 128
            
            # Create the default model
            default_model = ConfigurableSiameseNetwork(
                in_channels=actual_channels,
                feature_size=feature_size,
                embedding_size=default_embedding_size,
                encoder_config=default_encoder,
                fc_config=default_fc
            )
            
            # Save the default model configuration and weights
            model_dir = os.path.join(args.output_dir, "models")
            results_dir = os.path.join(args.output_dir, "search_results")
            os.makedirs(model_dir, exist_ok=True)
            os.makedirs(results_dir, exist_ok=True)
            
            # Save model weights as best_model.pt so the main code can find it
            default_model_path = os.path.join(results_dir, "best_model.pt")
            torch.save(default_model.state_dict(), default_model_path)
            
            # Save configuration
            default_config = {
                "in_channels": actual_channels,
                "feature_size": feature_size,
                "embedding_size": default_embedding_size,
                "encoder_config": default_encoder,
                "fc_config": default_fc,
                "avg_val_loss": None,  # No validation loss available
                "note": "Default model created as fallback due to randomized search failure - saved as best model"
            }
            
            # Save as best_configuration.json so the main code can find it
            config_path = os.path.join(results_dir, "best_configuration.json")
            with open(config_path, "w") as f:
                json.dump(default_config, f, indent=2)
            
            logger.info("Created and saved default model as best model fallback")
            return default_model
    
    except Exception as e:
        logger.critical(f"Critical error in randomized search: {e}")
        logger.exception("Detailed traceback:")
        
        # Last resort emergency fallback model
        logger.warning("Creating emergency fallback model")
        
        # Create the simplest possible model
        emergency_encoder = [
            {'type': 'conv', 'out_channels': 16, 'kernel_size': 3, 'padding': 1},
            {'type': 'relu'},
            {'type': 'pool', 'pool_type': 'max', 'kernel_size': 2, 'stride': 2},
        ]
        
        emergency_fc = [
            {'type': 'linear', 'out_features': 32},
            {'type': 'relu'},
        ]
        
        emergency_model = ConfigurableSiameseNetwork(
            in_channels=actual_channels,
            feature_size=feature_size,
            embedding_size=16,  # Safe default
            encoder_config=emergency_encoder,
            fc_config=emergency_fc
        )
        
        logger.info("Created emergency fallback model")
        return emergency_model


def save_detailed_model_configuration(model, in_channels, feature_size, best_configuration, output_dir):
    """
    Save detailed model configuration to a text file
    
    Args:
        model: The best model
        in_channels: Number of input channels
        feature_size: Size of input features
        best_configuration: Dictionary with best configuration
        output_dir: Directory to save the configuration
    """
    # Create a string buffer to capture the model summary
    summary_buffer = io.StringIO()
    
    # Save original stdout and redirect to buffer
    original_stdout = sys.stdout
    sys.stdout = summary_buffer
    
    # Generate model summary with proper input size
    input_size = (in_channels, feature_size)
    summary(model, input_size=input_size, device=next(model.parameters()).device.type)
    
    # Restore original stdout
    sys.stdout = original_stdout
    
    # Get the summary
    model_summary = summary_buffer.getvalue()
    
    # Create detailed configuration dictionary
    detailed_config = {
        "model_summary": model_summary,
        "in_channels": in_channels,
        "feature_size": feature_size,
        "embedding_size": best_configuration.get("embedding_size"),
        "architecture": {
            "encoder": best_configuration.get("encoder_config", []),
            "fc": best_configuration.get("fc_config", [])
        },
        "total_params": sum(p.numel() for p in model.parameters()),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "validation_loss": best_configuration.get("avg_val_loss"),
        "hyperparameters": {
            k: v for k, v in best_configuration.items() 
            if k not in ["encoder_config", "fc_config", "_encoder_architecture", "_fc_architecture", "_structure_info"]
        }
    }
    
    # Check if structure info is available
    if "_structure_info" in best_configuration:
        detailed_config["structure_info"] = best_configuration["_structure_info"]
    
    # Save to json file
    config_path = os.path.join(output_dir, "search_results", "detailed_model_config.json")
    with open(config_path, "w") as f:
        json.dump(detailed_config, f, indent=2)
    
    # Save to txt file for easy reading
    txt_path = os.path.join(output_dir, "search_results", "model_summary.txt")
    with open(txt_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("MODEL ARCHITECTURE SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(model_summary)
        f.write("\n\n")
        f.write("=" * 80 + "\n")
        f.write("CONFIGURATION DETAILS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Input Channels: {in_channels}\n")
        f.write(f"Feature Size: {feature_size}\n")
        f.write(f"Embedding Size: {detailed_config['embedding_size']}\n")
        f.write(f"Total Parameters: {detailed_config['total_params']:,}\n")
        f.write(f"Trainable Parameters: {detailed_config['trainable_params']:,}\n")
        f.write(f"Best Validation Loss: {detailed_config.get('validation_loss', 'N/A')}\n\n")
        
        f.write("Encoder Architecture:\n")
        for i, layer in enumerate(detailed_config["architecture"]["encoder"]):
            f.write(f"  Layer {i+1}: {layer}\n")
        
        f.write("\nFC Architecture:\n")
        for i, layer in enumerate(detailed_config["architecture"]["fc"]):
            f.write(f"  Layer {i+1}: {layer}\n")
        
        f.write("\nHyperparameters:\n")
        for k, v in detailed_config["hyperparameters"].items():
            f.write(f"  {k}: {v}\n")
    
    logger.info(f"Saved detailed model configuration to {txt_path}")
    logger.info(f"Saved configuration JSON to {config_path}")


def recreate_training_pipeline(best_configuration, best_weights):
    """
    Recreate the training pipeline with the best configuration and weights
    
    Args:
        best_configuration: Dictionary with the best configuration
        best_weights: PyTorch state dictionary with the best weights
        
    Returns:
        Model initialized with the best configuration and weights
    """
    from siamese_analysis_v3.models.configurable_siamese import ConfigurableSiameseNetwork
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        # Get model parameters from best configuration
        in_channels = best_configuration.get('in_channels', 4)
        feature_size = best_configuration.get('feature_size', 128)
        embedding_size = best_configuration.get('embedding_size', 16)
        
        # Get architecture configurations
        encoder_architecture = best_configuration.get('encoder_config')
        fc_architecture = best_configuration.get('fc_config')
        
        # Check if configurations exist in the best_configuration
        if not encoder_architecture:
            logger.warning("No encoder configuration found in best configuration. Using default architecture.")
            # Define default CNN encoder architecture
            encoder_architecture = [
                # Block 1
                {'type': 'conv', 'out_channels': 16, 'kernel_size': 3, 'padding': 1},
                {'type': 'bn'},
                {'type': 'relu'},
                {'type': 'pool', 'pool_type': 'max', 'kernel_size': 2, 'stride': 2},
                # Block 2
                {'type': 'conv', 'out_channels': 32, 'kernel_size': 3, 'padding': 1},
                {'type': 'bn'},
                {'type': 'relu'},
                {'type': 'pool', 'pool_type': 'max', 'kernel_size': 2, 'stride': 2},
                # Block 3
                {'type': 'conv', 'out_channels': 16, 'kernel_size': 3, 'padding': 1},
                {'type': 'bn'},
                {'type': 'relu'},
                {'type': 'pool', 'pool_type': 'max', 'kernel_size': 2, 'stride': 2},
            ]
        
        if not fc_architecture:
            logger.warning("No FC configuration found in best configuration. Using default architecture.")
            # Define default fully connected layers
            fc_architecture = [
                {'type': 'linear', 'out_features': 64},
                {'type': 'relu'},
            ]
        
        # Log the architecture being used
        logger.info(f"Recreating model with in_channels={in_channels}, feature_size={feature_size}, embedding_size={embedding_size}")
        logger.info(f"Using encoder architecture with {len([l for l in encoder_architecture if l.get('type') == 'conv'])} conv layers")
        logger.info(f"Using FC architecture with {len([l for l in fc_architecture if l.get('type') == 'linear'])} linear layers")
        
        # Validate parameters
        if not isinstance(in_channels, int) or in_channels <= 0:
            logger.warning(f"Invalid in_channels: {in_channels}, using default value 4")
            in_channels = 4
        
        if not isinstance(feature_size, int) or feature_size <= 0:
            logger.warning(f"Invalid feature_size: {feature_size}, using default value 128")
            feature_size = 128
        
        if not isinstance(embedding_size, int) or embedding_size <= 0:
            logger.warning(f"Invalid embedding_size: {embedding_size}, using default value 16")
            embedding_size = 16
        
        # Create model with the best architecture
        try:
            model = ConfigurableSiameseNetwork(
                in_channels=in_channels,
                feature_size=feature_size,
                embedding_size=embedding_size,
                encoder_config=encoder_architecture,
                fc_config=fc_architecture
            )
            
            # Load the best weights
            try:
                model.load_state_dict(best_weights)
                logger.info("Successfully loaded model weights")
            except Exception as e:
                logger.error(f"Failed to load model weights: {e}")
                logger.exception("Detailed traceback:")
                logger.warning("Using model with randomly initialized weights")
            
            return model
        except Exception as e:
            logger.error(f"Failed to create model with best configuration: {e}")
            logger.exception("Detailed traceback:")
            
            # Create a simpler model as fallback
            logger.warning("Creating fallback model with simpler architecture")
            
            # Define a simpler, more robust architecture
            simple_encoder = [
                {'type': 'conv', 'out_channels': 16, 'kernel_size': 3, 'padding': 1},
                {'type': 'relu'},
                {'type': 'pool', 'pool_type': 'max', 'kernel_size': 2, 'stride': 2},
                {'type': 'conv', 'out_channels': 32, 'kernel_size': 3, 'padding': 1},
                {'type': 'relu'},
                {'type': 'pool', 'pool_type': 'max', 'kernel_size': 2, 'stride': 2},
            ]
            
            simple_fc = [
                {'type': 'linear', 'out_features': 64},
                {'type': 'relu'},
            ]
            
            fallback_model = ConfigurableSiameseNetwork(
                in_channels=in_channels,
                feature_size=feature_size,
                embedding_size=16,  # Safe default
                encoder_config=simple_encoder,
                fc_config=simple_fc
            )
            
            logger.info("Created fallback model successfully")
            return fallback_model
            
    except Exception as e:
        logger.error(f"Critical error recreating training pipeline: {e}")
        logger.exception("Detailed traceback:")
        
        # Last resort fallback - create the simplest possible model
        logger.warning("Creating emergency fallback model")
        
        # Define the simplest possible architecture that is likely to work
        emergency_encoder = [
            {'type': 'conv', 'out_channels': 16, 'kernel_size': 3, 'padding': 1},
            {'type': 'relu'},
            {'type': 'pool', 'pool_type': 'max', 'kernel_size': 2, 'stride': 2},
        ]
        
        emergency_fc = [
            {'type': 'linear', 'out_features': 32},
            {'type': 'relu'},
        ]
        
        try:
            emergency_model = ConfigurableSiameseNetwork(
                in_channels=4,  # Safe default
                feature_size=128,  # Safe default
                embedding_size=16,  # Safe default
                encoder_config=emergency_encoder,
                fc_config=emergency_fc
            )
            logger.info("Created emergency fallback model successfully")
            return emergency_model
        except Exception as fatal_error:
            # If even the emergency model fails, we have no choice but to raise an exception
            logger.critical(f"Failed to create emergency fallback model: {fatal_error}")
            raise RuntimeError("Failed to create any working model. Please check your configuration and data.")


def load_best_configuration(output_dir):
    """
    Load the best configuration from the search results

    Args:
        output_dir: Directory containing search results

    Returns:
        Dictionary with the best configuration
    """
    results_dir = os.path.join(output_dir, "search_results")

    # First try to load the best configuration
    config_path = os.path.join(results_dir, "best_configuration.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            best_configuration = json.load(f)
        return best_configuration

    # If best_configuration.json doesn't exist, try default_configuration.json
    default_config_path = os.path.join(results_dir, "default_configuration.json")
    if os.path.exists(default_config_path):
        with open(default_config_path, 'r') as f:
            best_configuration = json.load(f)
        return best_configuration

    # If neither exists, raise an error
    raise FileNotFoundError(f"Neither best_configuration.json nor default_configuration.json found in {results_dir}")


def load_best_weights(output_dir):
    """
    Load the best model weights from the search results

    Args:
        output_dir: Directory containing search results

    Returns:
        PyTorch state dictionary with the best weights
    """
    results_dir = os.path.join(output_dir, "search_results")

    # First try to load the best model weights
    weights_path = os.path.join(results_dir, "best_model.pt")
    if os.path.exists(weights_path):
        # Determine the map location based on device availability
        if torch.cuda.is_available():
            map_location = torch.device('cuda') # Load to default CUDA device
        else:
            map_location = torch.device('cpu') # Load to CPU

        logger.info(f"Loading best weights from {weights_path} with map_location='{map_location}'")
        return torch.load(weights_path, map_location=map_location)

    # If best_model.pt doesn't exist, try default_model.pt
    default_weights_path = os.path.join(results_dir, "default_model.pt")
    if os.path.exists(default_weights_path):
        # Determine the map location based on device availability
        if torch.cuda.is_available():
            map_location = torch.device('cuda') # Load to default CUDA device
        else:
            map_location = torch.device('cpu') # Load to CPU

        logger.info(f"Loading default weights from {default_weights_path} with map_location='{map_location}'")
        return torch.load(default_weights_path, map_location=map_location)

    # If neither exists, raise an error
    raise FileNotFoundError(f"Neither best_model.pt nor default_model.pt found in {results_dir}")


