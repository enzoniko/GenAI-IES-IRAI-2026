"""
Highly Configurable Siamese Neural Network Implementation using PyTorch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections.abc # To check for sequence type
import math # For initialization

class ConfigurableSiameseNetwork(nn.Module):
    """
    Configurable Siamese network that dynamically builds its architecture.

    This network uses a shared encoder for both inputs and calculates
    the intermediate feature size automatically to prevent shape mismatches
    between the encoder and the fully connected layers.
    
    Supports:
    - Flexible CNN and FC architectures
    - Multiple activation functions (ReLU, LeakyReLU, ELU, GELU)
    - BatchNorm customization
    - Advanced pooling options (standard, adaptive)
    - Multiple initialization methods
    - Skip connections (residual blocks)
    """

    def __init__(self, in_channels, feature_size, embedding_size, encoder_config, fc_config, 
                 weight_init='kaiming_normal', weight_init_gain=1.0, use_skip_connections=False):
        """
        Initializes the ConfigurableSiameseNetwork.

        Args:
            in_channels (int): Number of input channels for the time series data.
            feature_size (int): The length (time dimension) of the input sequences.
            embedding_size (int): The desired size of the final output embedding vector.
            encoder_config (list): A list of dictionaries, each specifying a layer
                                   or block for the encoder part. Supported types:
                                   - {'type': 'conv', 'out_channels': int, 'kernel_size': int,
                                     'stride': int (optional, default 1),
                                     'padding': int or str (optional, default 0),
                                     'dilation': int (optional, default 1),
                                     'groups': int (optional, default 1)}
                                   - {'type': 'bn', 'momentum': float (optional, default 0.1),
                                     'eps': float (optional, default 1e-5),
                                     'track_running_stats': bool (optional, default True)}
                                   - {'type': 'relu'}
                                   - {'type': 'leaky_relu', 'negative_slope': float (optional, default 0.1)}
                                   - {'type': 'elu', 'alpha': float (optional, default 1.0)}
                                   - {'type': 'gelu'}
                                   - {'type': 'pool', 'pool_type': 'max' or 'avg',
                                     'kernel_size': int,
                                     'stride': int (optional, defaults to kernel_size)}
                                   - {'type': 'adaptive_pool', 'pool_type': 'max' or 'avg',
                                     'output_size': int}
                                   - {'type': 'dropout', 'p': float (optional, default 0.5)}
            fc_config (list): A list of dictionaries, each specifying a layer for the
                              fully connected part (excluding the final embedding layer).
                              Supported types:
                              - {'type': 'linear', 'out_features': int}
                              - {'type': 'relu'}
                              - {'type': 'leaky_relu', 'negative_slope': float (optional, default 0.1)}
                              - {'type': 'elu', 'alpha': float (optional, default 1.0)}
                              - {'type': 'gelu'}
                              - {'type': 'dropout', 'p': float (optional, default 0.5)}
                              - {'type': 'layer_norm'}
            weight_init (str): Weight initialization method. Options:
                              - 'kaiming_normal' (default)
                              - 'kaiming_uniform'
                              - 'xavier_normal'
                              - 'xavier_uniform'
                              - 'orthogonal'
            weight_init_gain (float): Gain for weight initialization (default: 1.0)
            use_skip_connections (bool): Whether to use skip connections in the encoder (default: False)
        """
        super(ConfigurableSiameseNetwork, self).__init__()

        if not isinstance(encoder_config, collections.abc.Sequence) or not encoder_config:
             raise ValueError("encoder_config must be a non-empty list of layer configurations.")
        if not isinstance(fc_config, collections.abc.Sequence):
             raise ValueError("fc_config must be a list of layer configurations (can be empty).")

        self.in_channels = in_channels
        self.feature_size = feature_size
        self.embedding_size = embedding_size
        self.use_skip_connections = use_skip_connections
        
        # Store weight initialization parameters as instance variables for reset_parameters
        self.weight_init = weight_init
        self.weight_init_gain = weight_init_gain
        
        # Store architecture configurations for model recreation
        self.encoder_config = encoder_config
        self.fc_config = fc_config
        
        # Track layers with skip connections
        self.skip_points = []
        self.skip_layers = {}

        # --- Build Encoder Dynamically ---
        encoder_layers = []
        current_channels = in_channels
        skip_input_channels = None
        
        for i, layer_conf in enumerate(encoder_config):
            layer_type = layer_conf.get('type')
            
            # Check for skip connection start points
            if use_skip_connections and layer_type == 'conv':
                # Mark this as a potential skip input point
                if i > 0 and 'conv' in encoder_config[i-1].get('type', ''):
                    skip_input_channels = current_channels
                    self.skip_points.append(len(encoder_layers) - 1)
            
            # Handle different layer types
            if layer_type == 'conv':
                out_channels = layer_conf.get('out_channels')
                if not isinstance(out_channels, int) or out_channels <= 0:
                    raise ValueError(f"Invalid 'out_channels' in conv config: {layer_conf}")
                
                # For skip connections, we may need to adjust output channels
                if use_skip_connections and skip_input_channels is not None and i > 2:
                    # Every 2-3 layers, create a skip connection
                    if i % 3 == 0:
                        # Store skip connection parameters
                        skip_idx = len(self.skip_points) - 1
                        if skip_idx >= 0:
                            self.skip_layers[f"skip_{skip_idx}"] = nn.Conv1d(
                                skip_input_channels, out_channels, 
                                kernel_size=1, stride=1, padding=0
                            )
                
                encoder_layers.append(nn.Conv1d(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    kernel_size=layer_conf.get('kernel_size', 3), # Default kernel size 3
                    stride=layer_conf.get('stride', 1),
                    padding=layer_conf.get('padding', 0), # Default padding 0
                    dilation=layer_conf.get('dilation', 1),
                    groups=layer_conf.get('groups', 1)  # Support for grouped convolutions
                ))
                current_channels = out_channels # Update channels for next layer
            elif layer_type == 'bn':
                # BatchNorm1d requires the number of features (channels)
                if not encoder_layers or not isinstance(encoder_layers[-1], (nn.Conv1d)):
                     raise ValueError("BatchNorm1d must follow a Conv1d layer in encoder_config.")
                
                # Enhanced BatchNorm with configurable parameters
                encoder_layers.append(nn.BatchNorm1d(
                    current_channels,
                    momentum=layer_conf.get('momentum', 0.1),
                    eps=layer_conf.get('eps', 1e-5),
                    track_running_stats=layer_conf.get('track_running_stats', True)
                ))
            elif layer_type == 'layer_norm':
                # Layer normalization
                encoder_layers.append(nn.LayerNorm(current_channels))
            elif layer_type == 'instance_norm':
                # Instance normalization
                encoder_layers.append(nn.InstanceNorm1d(
                    current_channels,
                    track_running_stats=layer_conf.get('track_running_stats', False)
                ))
            elif layer_type == 'relu':
                encoder_layers.append(nn.ReLU())
            elif layer_type == 'leaky_relu':
                negative_slope = layer_conf.get('negative_slope', 0.1)
                encoder_layers.append(nn.LeakyReLU(negative_slope=negative_slope))
            elif layer_type == 'elu':
                alpha = layer_conf.get('alpha', 1.0)
                encoder_layers.append(nn.ELU(alpha=alpha))
            elif layer_type == 'gelu':
                encoder_layers.append(nn.GELU())
            elif layer_type == 'pool':
                pool_type = layer_conf.get('pool_type', 'max').lower()
                kernel_size = layer_conf.get('kernel_size')
                # Convert kernel_size from tuple to int if needed
                if isinstance(kernel_size, tuple) and len(kernel_size) == 1:
                    kernel_size = kernel_size[0]
                if not isinstance(kernel_size, int) or kernel_size <= 0:
                     raise ValueError(f"Invalid 'kernel_size' in pool config: {layer_conf}")
                stride = layer_conf.get('stride', kernel_size) # Default stride = kernel_size
                # Convert stride from tuple to int if needed
                if isinstance(stride, tuple) and len(stride) == 1:
                    stride = stride[0]
                if pool_type == 'max':
                    encoder_layers.append(nn.MaxPool1d(kernel_size=kernel_size, stride=stride))
                elif pool_type == 'avg':
                    encoder_layers.append(nn.AvgPool1d(kernel_size=kernel_size, stride=stride))
                else:
                    raise ValueError(f"Unsupported pool_type '{pool_type}' in {layer_conf}")
            elif layer_type == 'adaptive_pool':
                # Support for adaptive pooling
                pool_type = layer_conf.get('pool_type', 'max').lower()
                output_size = layer_conf.get('output_size')
                if not isinstance(output_size, int) or output_size <= 0:
                    raise ValueError(f"Invalid 'output_size' in adaptive_pool config: {layer_conf}")
                
                if pool_type == 'max':
                    encoder_layers.append(nn.AdaptiveMaxPool1d(output_size=output_size))
                elif pool_type == 'avg':
                    encoder_layers.append(nn.AdaptiveAvgPool1d(output_size=output_size))
                else:
                    raise ValueError(f"Unsupported pool_type '{pool_type}' in {layer_conf}")
            elif layer_type == 'dropout':
                 encoder_layers.append(nn.Dropout(p=layer_conf.get('p', 0.5)))
            else:
                raise ValueError(f"Unsupported encoder layer type: {layer_type} in {layer_conf}")

        # Create nn.ModuleDict for skip layers if needed
        if self.use_skip_connections and self.skip_layers:
            self.skip_layers = nn.ModuleDict(self.skip_layers)
            
        # Create main encoder
        self.encoder = nn.ModuleList(encoder_layers)

        # --- Calculate Flattened Size Dynamically ---
        # Perform a dummy forward pass to find the output size of the encoder
        self._conv_output_size = self._get_encoder_output_size(in_channels, feature_size)
        print(f"Dynamically calculated encoder output size (flattened): {self._conv_output_size}")

        # --- Build Fully Connected Layers Dynamically ---
        fc_layers = []
        # Add flatten layer first
        fc_layers.append(nn.Flatten())
        current_features = self._conv_output_size

        for layer_conf in fc_config:
            layer_type = layer_conf.get('type')
            if layer_type == 'linear':
                out_features = layer_conf.get('out_features')
                if not isinstance(out_features, int) or out_features <= 0:
                    raise ValueError(f"Invalid 'out_features' in linear config: {layer_conf}")
                fc_layers.append(nn.Linear(current_features, out_features))
                current_features = out_features # Update features for next layer
            elif layer_type == 'relu':
                fc_layers.append(nn.ReLU())
            elif layer_type == 'leaky_relu':
                negative_slope = layer_conf.get('negative_slope', 0.1)
                fc_layers.append(nn.LeakyReLU(negative_slope=negative_slope))
            elif layer_type == 'elu':
                alpha = layer_conf.get('alpha', 1.0)
                fc_layers.append(nn.ELU(alpha=alpha))
            elif layer_type == 'gelu':
                fc_layers.append(nn.GELU())
            elif layer_type == 'layer_norm':
                # Layer normalization for fully connected layers
                fc_layers.append(nn.LayerNorm(current_features))
            elif layer_type == 'dropout':
                fc_layers.append(nn.Dropout(p=layer_conf.get('p', 0.5)))
            else:
                 raise ValueError(f"Unsupported fc layer type: {layer_type} in {layer_conf}")

        # Add the final embedding layer
        fc_layers.append(nn.Linear(current_features, embedding_size))

        self.fc = nn.Sequential(*fc_layers)
        
        # Apply weight initialization
        self._initialize_weights(weight_init, weight_init_gain)

    def _initialize_weights(self, init_method, gain=1.0):
        """
        Initialize weights of the model using the specified method.
        
        Args:
            init_method (str): Initialization method to use
            gain (float): Gain factor for initialization
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                if init_method == 'kaiming_normal':
                    nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                elif init_method == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                elif init_method == 'xavier_normal':
                    nn.init.xavier_normal_(m.weight, gain=gain)
                elif init_method == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight, gain=gain)
                elif init_method == 'orthogonal':
                    nn.init.orthogonal_(m.weight, gain=gain)
                
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def reset_parameters(self):
        """
        Reset all model parameters using the same initialization method as in __init__.
        This ensures consistent reinitialization when model is reused across folds.
        """
        # Get initialization method from object state or use default
        init_method = getattr(self, 'weight_init', 'kaiming_normal')
        gain = getattr(self, 'weight_init_gain', 1.0)
        
        # Apply initialization to all layers
        self._initialize_weights(init_method, gain)
        
        # Log the reinitialization
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Model parameters reset using {init_method} initialization with gain={gain}")
        
        return self

    def _get_encoder_output_size(self, in_channels, feature_size):
        """
        Calculates the output size of the encoder by performing a dummy forward pass.

        Args:
            in_channels (int): Number of input channels.
            feature_size (int): Length of the input sequence.

        Returns:
            int: The total number of features after the encoder and flattening.

        Raises:
            ValueError: If the calculated output size is zero or negative,
                        indicating an invalid encoder configuration for the given input size.
            RuntimeError: If the dummy forward pass itself fails.
        """
        # Ensure encoder is in eval mode and no gradients are calculated
        for layer in self.encoder:
            if isinstance(layer, nn.Module):
                layer.eval()
        
        with torch.no_grad():
            # Create a dummy input tensor with the expected shape: (batch_size, channels, length)
            # Use batch_size=1 for simplicity
            dummy_input = torch.randn(1, in_channels, feature_size)
            try:
                # Forward pass through encoder
                dummy_output = self._encoder_forward(dummy_input)
                
                # Calculate the flattened size: product of all dimensions except batch
                flattened_size = dummy_output.view(dummy_output.size(0), -1).size(1)

                if flattened_size <= 0:
                    raise ValueError(
                        f"Encoder output size is non-positive ({flattened_size}) for input "
                        f"feature_size {feature_size}. This likely means the sequence length "
                        f"became too small after convolution/pooling. Review your encoder_config, "
                        f"especially pooling layers and strides, or increase feature_size."
                    )
            except Exception as e:
                raise RuntimeError(f"Error during dummy forward pass in encoder with input shape (1, {in_channels}, {feature_size}). "
                                   f"Check encoder_config. Original error: {e}") from e
        
        # Set layers back to train mode
        for layer in self.encoder:
            if isinstance(layer, nn.Module):
                layer.train()
        
        return flattened_size

    def _encoder_forward(self, x):
        """
        Forward pass through encoder with skip connections if enabled.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor after applying encoder layers.
        """
        skip_features = {}
        
        # Apply encoder layers
        for i, layer in enumerate(self.encoder):
            # Store features for skip connections
            if self.use_skip_connections and i in self.skip_points:
                skip_features[i] = x
            
            # Apply layer
            x = layer(x)
            
            # Apply skip connection if needed
            if self.use_skip_connections and i > 0 and hasattr(self, 'skip_layers'):
                for skip_idx, skip_layer_idx in enumerate(self.skip_points):
                    skip_key = f"skip_{skip_idx}"
                    
                    # Find target point for skip connection (after activation)
                    target_idx = skip_layer_idx + 3
                    
                    if i == target_idx and skip_key in self.skip_layers and skip_layer_idx in skip_features:
                        # Get features from earlier layer and transform
                        skip_x = skip_features[skip_layer_idx]
                        transformed_skip = self.skip_layers[skip_key](skip_x)
                        
                        # Add transformed features to current features (residual connection)
                        # Make sure shapes match
                        if x.shape[2:] != transformed_skip.shape[2:]:
                            # Use adaptive pooling to match spatial dimensions
                            transformed_skip = F.adaptive_avg_pool1d(transformed_skip, x.shape[2])
                        
                        x = x + transformed_skip
                
        return x

    def forward_one(self, x):
        """
        Passes a single input through the encoder and fc layers to get its embedding.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, feature_size).

        Returns:
            torch.Tensor: Output embedding tensor of shape (batch_size, embedding_size).
        """
        # Input shape check (optional but helpful)
        if x.shape[1] != self.in_channels or x.shape[2] != self.feature_size:
            print(f"Warning: Input tensor shape {x.shape} does not match expected "
                  f"shape (batch_size, {self.in_channels}, {self.feature_size}).")

        # Use encoder forward method to handle skip connections
        x = self._encoder_forward(x)
        
        x = self.fc(x)
        # L2 normalize embeddings - common practice for distance-based losses
        x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, x1, x2=None):
        """
        Forward pass for the Siamese network.

        Args:
            x1 (torch.Tensor): The first input tensor (or anchor).
            x2 (torch.Tensor): The second input tensor (or positive/negative).
                               If None, only the embedding for x1 is returned.

        Returns:
            torch.Tensor or tuple[torch.Tensor, torch.Tensor]:
                - If x2 is None, returns the embedding for x1.
                - If x2 is provided, returns a tuple (embedding1, embedding2).
        """
        embedding1 = self.forward_one(x1)

        if x2 is not None:
            embedding2 = self.forward_one(x2)
            return embedding1, embedding2

        return embedding1


class TripletLoss(nn.Module):
    """
    Triplet loss with configurable margin and distance metrics.
    Encourages the anchor-positive distance to be smaller than the
    anchor-negative distance by at least a margin.
    
    Supports multiple distance metrics and triplet selection strategies.
    """

    def __init__(self, margin=1.0, distance_metric='euclidean', reduction='mean', 
                 triplet_selection='all', mining_epsilon=0.1, label_smoothing=0.0):
        """
        Initializes the TripletLoss.

        Args:
            margin (float): The margin value. Positive value means the anchor-positive distance
                           should be less than the anchor-negative distance by at least margin.
                           Negative value can be used to enforce a specific distance between 
                           positive and negative samples. Defaults to 1.0.
            distance_metric (str): Distance metric to use. Options:
                          - 'euclidean': Standard Euclidean distance (default)
                          - 'squared_euclidean': Squared Euclidean distance (faster)
                          - 'cosine': Cosine distance (1 - cosine similarity)
                          - 'manhattan': Manhattan (L1) distance
            reduction (str): Reduction method for the loss. Options:
                           - 'mean': Mean of losses (default)
                           - 'sum': Sum of losses 
                           - 'none': No reduction
            triplet_selection (str): Method to select triplets for computing loss. Options:
                                  - 'all': Use all triplets in the batch (default)
                                  - 'hard': Only use the hardest triplet in the batch
                                  - 'semi-hard': Only use semi-hard triplets (negatives closer than margin)
                                  - 'distance_weighted': Weight triplets by their relative difficulty
            mining_epsilon (float): Small constant for numerical stability in distance weighting (default: 0.1)
            label_smoothing (float): Amount of label smoothing to apply (0.0-1.0). Default: 0.0
        """
        super(TripletLoss, self).__init__()
        if margin < 0:
            print(f"Warning: TripletLoss margin is negative ({margin}). This will enforce a minimum distance.")
        self.margin = margin
        self.distance_metric = distance_metric
        self.reduction = reduction
        self.triplet_selection = triplet_selection
        self.mining_epsilon = mining_epsilon
        self.label_smoothing = label_smoothing
        
        # Using pairwise distance for potentially better numerical stability
        if self.distance_metric == 'euclidean':
            self.pdist = nn.PairwiseDistance(p=2)  # p=2 for Euclidean distance
        elif self.distance_metric == 'manhattan':
            self.pdist = nn.PairwiseDistance(p=1)  # p=1 for Manhattan distance

    def _get_distance(self, x1, x2):
        """
        Compute distance between two sets of embeddings based on the selected metric.
        
        Args:
            x1 (torch.Tensor): First embeddings of shape (batch_size, embedding_dim)
            x2 (torch.Tensor): Second embeddings of shape (batch_size, embedding_dim)
            
        Returns:
            torch.Tensor: Distance between embeddings of shape (batch_size,)
        """
        if self.distance_metric == 'euclidean':
            return self.pdist(x1, x2)
        elif self.distance_metric == 'squared_euclidean':
            return torch.sum((x1 - x2) ** 2, dim=1)
        elif self.distance_metric == 'cosine':
            # Normalize embeddings for cosine similarity
            x1_norm = F.normalize(x1, p=2, dim=1)
            x2_norm = F.normalize(x2, p=2, dim=1)
            # Cosine similarity
            cos_sim = torch.sum(x1_norm * x2_norm, dim=1)
            # Convert to distance: 1 - similarity
            return 1.0 - cos_sim
        elif self.distance_metric == 'manhattan':
            return self.pdist(x1, x2)
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")

    def _apply_triplet_selection(self, pos_dist, neg_dist, margin_dist):
        """
        Apply triplet selection strategy.
        
        Args:
            pos_dist (torch.Tensor): Positive distances of shape (batch_size,)
            neg_dist (torch.Tensor): Negative distances of shape (batch_size,)
            margin_dist (torch.Tensor): Margin distances of shape (batch_size,)
            
        Returns:
            torch.Tensor: Selected triplet losses
        """
        batch_size = pos_dist.size(0)
        
        if self.triplet_selection == 'all':
            # Use all triplets
            return margin_dist
        elif self.triplet_selection == 'hard':
            # Only use the hardest triplet
            hardest_loss = torch.max(margin_dist)
            return hardest_loss.unsqueeze(0)  # Return as singleton tensor for reduction
        elif self.triplet_selection == 'semi-hard':
            # Only use semi-hard triplets (negatives closer than margin but not too close)
            semi_hard_mask = (neg_dist < pos_dist + self.margin) & (neg_dist > pos_dist)
            
            # If no semi-hard triplets, fall back to using all triplets
            if semi_hard_mask.sum() == 0:
                return margin_dist
            
            return margin_dist[semi_hard_mask]
        elif self.triplet_selection == 'distance_weighted':
            # Weight triplets by their relative difficulty
            # Add epsilon for numerical stability
            weights = F.softmax(margin_dist + self.mining_epsilon, dim=0)
            return margin_dist * weights * batch_size  # Scale to maintain loss magnitude
        else:
            raise ValueError(f"Unsupported triplet selection: {self.triplet_selection}")

    def forward(self, anchor, positive, negative):
        """
        Computes the triplet loss.

        Args:
            anchor (torch.Tensor): Embeddings for the anchor samples.
            positive (torch.Tensor): Embeddings for the positive samples.
            negative (torch.Tensor): Embeddings for the negative samples.

        Returns:
            torch.Tensor: The triplet loss.
        """
        # Apply label smoothing if enabled
        if self.label_smoothing > 0:
            # For triplet loss, label smoothing means adjusting the margin
            # This makes the model slightly less confident about distinctions
            smooth_margin = self.margin * (1.0 - self.label_smoothing)
        else:
            smooth_margin = self.margin
            
        # Calculate distances
        pos_dist = self._get_distance(anchor, positive)
        neg_dist = self._get_distance(anchor, negative)
        
        # Calculate triplet loss: max(0, pos_dist - neg_dist + margin)
        # Note: we use `pos_dist - neg_dist + margin` as the margin distance for selection
        margin_dist = torch.clamp(pos_dist - neg_dist + smooth_margin, min=0.0)
        
        # Apply triplet selection strategy
        selected_losses = self._apply_triplet_selection(pos_dist, neg_dist, margin_dist)
        
        # Apply reduction method
        if self.reduction == 'mean':
            return torch.mean(selected_losses)
        elif self.reduction == 'sum':
            return torch.sum(selected_losses)
        elif self.reduction == 'none':
            return selected_losses
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")


# --- Example Usage ---
if __name__ == '__main__':
    # --- Configuration ---
    INPUT_CHANNELS = 4
    FEATURE_LENGTH = 128 # Input sequence length
    EMBEDDING_DIM = 16

    # Define the encoder architecture dynamically
    # Matches the original example but defined configurably
    encoder_architecture = [
        # Block 1
        {'type': 'conv', 'out_channels': 16, 'kernel_size': 3, 'padding': 1},
        {'type': 'bn'},
        {'type': 'relu'},
        {'type': 'pool', 'pool_type': 'max', 'kernel_size': 2, 'stride': 2}, # Dim: 128 -> 64

        # Block 2
        {'type': 'conv', 'out_channels': 32, 'kernel_size': 3, 'padding': 1},
        {'type': 'bn'},
        {'type': 'relu'},
        {'type': 'pool', 'pool_type': 'max', 'kernel_size': 2, 'stride': 2}, # Dim: 64 -> 32

        # Block 3
        {'type': 'conv', 'out_channels': 16, 'kernel_size': 3, 'padding': 1},
        {'type': 'bn'},
        {'type': 'relu'},
        {'type': 'pool', 'pool_type': 'max', 'kernel_size': 2, 'stride': 2}, # Dim: 32 -> 16
    ]

    # Define the fully connected layers (excluding the final embedding layer)
    fc_architecture = [
        {'type': 'linear', 'out_features': 64},
        {'type': 'relu'},
        # The final Linear(64, EMBEDDING_DIM) is added automatically
    ]

    # --- Instantiate the Network ---
    try:
        model = ConfigurableSiameseNetwork(
            in_channels=INPUT_CHANNELS,
            feature_size=FEATURE_LENGTH,
            embedding_size=EMBEDDING_DIM,
            encoder_config=encoder_architecture,
            fc_config=fc_architecture
        )
        print("\n--- Model Architecture ---")
        print(model)
        print("Model instantiated successfully!")

        # --- Test Forward Pass ---
        print("\n--- Testing Forward Pass ---")
        # Create dummy input tensors (batch_size, channels, length)
        # Batch size = 5
        input1 = torch.randn(5, INPUT_CHANNELS, FEATURE_LENGTH)
        input2 = torch.randn(5, INPUT_CHANNELS, FEATURE_LENGTH)
        anchor = torch.randn(5, INPUT_CHANNELS, FEATURE_LENGTH)
        positive = torch.randn(5, INPUT_CHANNELS, FEATURE_LENGTH)
        negative = torch.randn(5, INPUT_CHANNELS, FEATURE_LENGTH)


        # Test single forward pass
        output1 = model(input1)
        print(f"Shape of single output: {output1.shape}") # Expected: [5, EMBEDDING_DIM]

        # Test paired forward pass
        output1_paired, output2_paired = model(input1, input2)
        print(f"Shape of paired outputs: {output1_paired.shape}, {output2_paired.shape}") # Expected: [5, EMBEDDING_DIM], [5, EMBEDDING_DIM]

        # --- Test Loss Calculation ---
        print("\n--- Testing Triplet Loss ---")
        triplet_loss_fn = TripletLoss(margin=0.5)

        # --- *** FIXED LINE BELOW *** ---
        # Get embeddings for triplet correctly
        anchor_emb, positive_emb = model(anchor, positive) # Get anchor and positive embeddings
        negative_emb = model.forward_one(negative)       # Get negative embedding separately

        loss = triplet_loss_fn(anchor_emb, positive_emb, negative_emb)
        print(f"Calculated Triplet Loss: {loss.item()}")

        # --- Example of a different configuration (Dilated CNN) ---
        print("\n--- Testing Different Configuration (Dilated CNN) ---")
        dilated_encoder_config = [
            {'type': 'conv', 'out_channels': 32, 'kernel_size': 3, 'padding': 2, 'dilation': 2}, # Dilated conv
            {'type': 'bn'},
            {'type': 'relu'},
            {'type': 'conv', 'out_channels': 64, 'kernel_size': 3, 'padding': 4, 'dilation': 4}, # More dilation
            {'type': 'bn'},
            {'type': 'relu'},
            {'type': 'pool', 'pool_type': 'avg', 'kernel_size': 4}, # Avg pooling
        ]
        # Simpler FC part for this example
        dilated_fc_config = [
             {'type': 'linear', 'out_features': 128},
             {'type': 'relu'},
        ]

        try:
            dilated_model = ConfigurableSiameseNetwork(
                in_channels=INPUT_CHANNELS,
                feature_size=FEATURE_LENGTH, # Use same feature length
                embedding_size=EMBEDDING_DIM,
                encoder_config=dilated_encoder_config,
                fc_config=dilated_fc_config
            )
            print("Dilated model instantiated successfully!")
            output_dilated = dilated_model(input1)
            print(f"Shape of dilated model output: {output_dilated.shape}")

        except (ValueError, RuntimeError) as e:
            print(f"Error instantiating dilated model: {e}")


        # --- Example of a configuration that might fail ---
        print("\n--- Testing Failing Configuration (Too much pooling) ---")
        fail_encoder_config = [
            {'type': 'pool', 'pool_type': 'max', 'kernel_size': 4, 'stride': 4}, # 128 -> 32
            {'type': 'pool', 'pool_type': 'max', 'kernel_size': 4, 'stride': 4}, # 32 -> 8
            {'type': 'pool', 'pool_type': 'max', 'kernel_size': 4, 'stride': 4}, # 8 -> 2
            {'type': 'pool', 'pool_type': 'max', 'kernel_size': 4, 'stride': 4}, # 2 -> 0/1 (problem!)
        ]
        try:
             fail_model = ConfigurableSiameseNetwork(
                in_channels=INPUT_CHANNELS,
                feature_size=FEATURE_LENGTH,
                embedding_size=EMBEDDING_DIM,
                encoder_config=fail_encoder_config,
                fc_config=[] # Empty FC config
             )
        except (ValueError, RuntimeError) as e:
             print(f"Successfully caught expected error for failing config: {e}")


    except (ValueError, RuntimeError) as e:
        print(f"\nAn error occurred during model setup or testing: {e}")

