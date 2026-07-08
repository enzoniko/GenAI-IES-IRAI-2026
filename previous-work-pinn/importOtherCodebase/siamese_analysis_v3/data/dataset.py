"""
Dataset and data loading utilities for Siamese networks
"""
import torch
import numpy as np
import os
from torch.utils.data import Dataset, Subset
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import KFold, ShuffleSplit
from tqdm import tqdm
import random
from .processor import LabeledSample
import logging

logger = logging.getLogger(__name__)


def extract_variant_name(data_type: str) -> str:
    """
    Extract variant name from data type for cleaner legend labels.
    
    Examples:
        "imbalance_fault_5g" -> "5g"
        "imbalance_fault_10g" -> "10g"
        "horizontal_misalignment_fault_0.51mm" -> "0.51mm"
        "vertical_misalignment_fault_1.27mm" -> "1.27mm"
        "overhang_ball_fault_0.18mm" -> "0.18mm"
        "normal" -> "normal"
    
    Args:
        data_type: Full data type string
        
    Returns:
        Variant name (the part after the last underscore, or the full name if no variant)
    """
    # Split by underscore and get the last part
    parts = data_type.split('_')
    if len(parts) > 1:
        # Check if the last part looks like a variant (contains numbers or units)
        last_part = parts[-1]
        if any(char.isdigit() for char in last_part) or last_part.endswith(('g', 'mm', 'hz')):
            return last_part
        elif len(parts) > 2:
            # Try the second-to-last part (e.g., for "fault_type_variant_unit")
            second_last = parts[-2]
            if any(char.isdigit() for char in second_last):
                return f"{second_last}_{last_part}"
    
    # If no clear variant found, return the original data type
    return data_type


class TripletDataset(Dataset):
    """Dataset for triplet training"""
    
    def __init__(self, anchors, positives, negatives, batch_size=64, num_workers=4):
        """
        Dataset for triplet training
        
        Args:
            anchors: Array of anchor samples
            positives: Array of positive samples
            negatives: Array of negative samples
            batch_size: Batch size for dataloaders
            num_workers: Number of worker processes for dataloaders
        """
        self.anchors = anchors
        self.positives = positives
        self.negatives = negatives
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def __len__(self):
        return len(self.anchors)
    
    def __getitem__(self, idx):
        # Convert to torch tensors if needed
        if not isinstance(self.anchors[idx], torch.Tensor):
            anchor = torch.FloatTensor(self.anchors[idx])
            positive = torch.FloatTensor(self.positives[idx])
            negative = torch.FloatTensor(self.negatives[idx])
        else:
            anchor = self.anchors[idx]
            positive = self.positives[idx]
            negative = self.negatives[idx]
            
        return {
            'anchor': anchor,
            'positive': positive,
            'negative': negative
        }


class CrossValidationSplitter:
    """Provides cross-validation splits for TripletDataset"""
    
    def __init__(self, n_splits=5, random_state=42):
        """
        Create a cross-validation splitter
        
        Args:
            n_splits: Number of folds
            random_state: Random seed
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
    def split(self, dataset: TripletDataset) -> List[Tuple[Subset, Subset]]:
        """
        Split dataset into k folds
        
        Args:
            dataset: TripletDataset to split
            
        Returns:
            List of (train_dataset, val_dataset) tuples
        """
        splits = []
        indices = np.arange(len(dataset))
        
        for train_idx, val_idx in self.kfold.split(indices):
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            splits.append((train_subset, val_subset))
            
        return splits
        
    def get_single_split(self, random_seed=None, dataset_length=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a single random train/validation split
        
        This method provides a single random train/validation split without 
        running full cross-validation. The validation set size will be 1/n_splits
        of the total dataset, similar to a single fold in cross-validation.
        
        Args:
            random_seed: Random seed for this specific split (if None, uses the instance's random_state)
            dataset_length: Length of the dataset (optional, used to create correct-sized indices)
            
        Returns:
            Tuple of (train_indices, val_indices)
        """
        if random_seed is None:
            random_seed = self.random_state
            
        # Create a ShuffleSplit with a single split
        split_ratio = 1.0 / self.n_splits  # Same validation ratio as in k-fold CV
        shuffle_split = ShuffleSplit(n_splits=1, test_size=split_ratio, random_state=random_seed)
        
        # If dataset_length is not provided, we'll have to use a sensible default
        # When this method is called from the RandomizedSearcher, we can pass the dataset length
        if dataset_length is None:
            logger.warning("Dataset length not provided for get_single_split. Using full dataset from instance.")
            # We can only do this if we have the dataset stored in the instance
            if hasattr(self, 'dataset_length') and self.dataset_length is not None:
                dataset_length = self.dataset_length
            else:
                # If we don't have the dataset length, use a reasonable default
                # This will be updated when the method is actually used with a dataset
                dataset_length = 1000
                logger.warning(f"Using default dataset_length={dataset_length} since no dataset info available")
        
        # Generate indices based on actual dataset size
        indices = np.arange(dataset_length)
        
        # Get the train/val indices
        for train_idx, val_idx in shuffle_split.split(indices):
            # Just return the first (and only) split
            return train_idx, val_idx


def generate_triplets_from_labeled_samples(
    samples_by_label: Dict[str, List[LabeledSample]], 
    processor,
    num_triplets: int,
    bypass_output_dir: str = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate triplets from labeled samples
    
    Args:
        samples_by_label: Dictionary mapping labels to lists of samples
        processor: ResidualProcessor instance
        num_triplets: Number of triplets to generate
        bypass_output_dir: Optional directory to save bypass UMAP plots (default: None)
        
    Returns:
        anchors, positives, negatives: Arrays of shape (num_triplets, channels, fft_length)
    """
    # Process all samples
    processed_data = {}
    labels = list(samples_by_label.keys())
    
    # Collect all samples first
    all_samples = []
    samples_by_index = {}
    current_index = 0
    
    for label in labels:
        samples = samples_by_label[label]
        start_idx = current_index
        end_idx = current_index + len(samples)
        
        # Extract just the data arrays
        sample_data = [s.data for s in samples]
        all_samples.extend(sample_data)
        
        # Keep track of index ranges for each label
        samples_by_index[label] = (start_idx, end_idx)
        current_index = end_idx
    
    print(f"Processing {len(all_samples)} samples from {len(labels)} labels...")

    # Process all samples at once (takes advantage of multiprocessing)
    all_processed = processor.fit_transform(all_samples)

    # Generate bypass UMAP plots from features (before Siamese training)
    if bypass_output_dir is not None:
        try:
            logger.info("Generating bypass UMAP plots from features...")

            # Create bypass output directory
            bypass_plots_dir = os.path.join(bypass_output_dir, "bypass_umap_plots")
            os.makedirs(bypass_plots_dir, exist_ok=True)

            # Reconstruct the sample hierarchy for bypass visualization
            bypass_samples_by_label = {}
            sample_idx = 0
            for label in labels:
                n_samples = len(samples_by_label[label])
                bypass_samples_by_label[label] = all_processed[sample_idx:sample_idx + n_samples]
                sample_idx += n_samples

            # Generate bypass UMAP plots
            generate_bypass_umap_plots(bypass_samples_by_label, bypass_plots_dir)

            logger.info(f"Bypass UMAP plots saved to: {bypass_plots_dir}")

        except Exception as e:
            logger.warning(f"Could not generate bypass UMAP plots: {e}")
            import traceback
            traceback.print_exc()
    
    # Organize processed samples back by label
    for label, (start_idx, end_idx) in samples_by_index.items():
        processed_data[label] = all_processed[start_idx:end_idx]
    
    # Generate triplets
    anchors = []
    positives = []
    negatives = []
    
    # Filter out labels with too few samples
    valid_labels = [label for label in labels if len(processed_data[label]) >= 2]
    
    if len(valid_labels) < 2:
        raise ValueError("Not enough valid labels with sufficient samples")
    
    for _ in tqdm(range(num_triplets), desc="Generating triplets"):
        # Select anchor label
        anchor_label = random.choice(valid_labels)
        
        # Select negative label (different from anchor)
        available_neg_labels = [label for label in valid_labels if label != anchor_label]
        negative_label = random.choice(available_neg_labels)
        
        # Select two different samples from anchor class
        anchor_idx, positive_idx = random.sample(range(len(processed_data[anchor_label])), 2)
        
        # Select a sample from negative class
        negative_idx = random.randint(0, len(processed_data[negative_label]) - 1)
        
        # Add to lists
        anchors.append(processed_data[anchor_label][anchor_idx])
        positives.append(processed_data[anchor_label][positive_idx])
        negatives.append(processed_data[negative_label][negative_idx])
    
    return np.array(anchors), np.array(positives), np.array(negatives)


def generate_bypass_umap_plots(features_by_label: Dict[str, np.ndarray], output_dir: str):
    """
    Generate UMAP plots directly from features (bypass Siamese training)

    Args:
        features_by_label: Dictionary mapping labels to feature arrays
        output_dir: Output directory for plots
    """
    try:
        # Debug: Print input information
        logger.info(f"Starting bypass UMAP generation with {len(features_by_label)} labels")
        for label, features in features_by_label.items():
            logger.info(f"  Label '{label}': {len(features)} samples, shape: {features[0].shape if len(features) > 0 else 'N/A'}")
        
        # Try to import UMAP
        try:
            from umap import UMAP
            UMAP_AVAILABLE = True
            logger.info("UMAP imported successfully via: from umap import UMAP")
        except ImportError:
            try:
                import umap
                UMAP_AVAILABLE = True
                logger.info("UMAP imported successfully via: import umap")
            except ImportError:
                logger.warning("UMAP not available, skipping bypass plots")
                return

        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        from matplotlib.colors import LinearSegmentedColormap
        from itertools import cycle
        from matplotlib import colors as mcolors

        # Set larger font sizes
        plt.rcParams.update({
            'font.size': 18,
            'axes.labelsize': 18,
            'axes.titlesize': 18,
            'xtick.labelsize': 18,
            'ytick.labelsize': 18,
            'legend.fontsize': 18,
            'figure.titlesize': 18
        })

        # Prepare data for UMAP
        all_features = []
        all_labels_level1 = []
        all_labels_level2 = []
        all_labels_level3 = []
        all_rot_speeds = []

        logger.info("Preparing bypass data for UMAP...")
        for label, features in tqdm(features_by_label.items(), desc="Preparing bypass data"):
            # features shape: (n_samples, n_channels, n_features_per_channel)
            # Need to flatten to 2D for UMAP: (n_samples, n_channels * n_features_per_channel)
            for feature in features:
                # Flatten the feature array to 1D
                if feature.ndim > 1:
                    flattened_feature = feature.flatten()
                else:
                    flattened_feature = feature
                all_features.append(flattened_feature)

                # Parse labels from the label string
                # Format: data_type_rot_speed
                parts = label.split('_')
                if len(parts) >= 3:
                    # Extract numeric part from the last component (handle units like "0.5mm", "10g")
                    last_part = parts[-1]
                    try:
                        # Try to extract numeric value from string with units
                        import re
                        numeric_match = re.search(r'([0-9]*\.?[0-9]+)', last_part)
                        if numeric_match:
                            rot_speed = float(numeric_match.group(1))
                        else:
                            rot_speed = 0.0
                    except (ValueError, AttributeError):
                        rot_speed = 0.0
                    data_type = '_'.join(parts[:-1])
                else:
                    rot_speed = 0.0
                    data_type = label

                # Create dummy LabeledSample to get hierarchical labels
                from .processor import LabeledSample
                dummy_sample = LabeledSample(
                    data=np.array([]),  # Not needed for labels
                    data_type=data_type,
                    rot_speed=rot_speed
                )

                all_labels_level1.append(dummy_sample.level1_label)
                all_labels_level2.append(dummy_sample.level2_label)
                all_labels_level3.append(dummy_sample.level3_label)
                all_rot_speeds.append(rot_speed)

        if not all_features:
            logger.warning("No features available for bypass UMAP visualization")
            return

        all_features = np.array(all_features)
        logger.info(f"Bypass UMAP input shape after flattening: {all_features.shape}")

        # Ensure 2D shape for UMAP
        if all_features.ndim != 2:
            logger.error(f"Features must be 2D for UMAP, got shape {all_features.shape}")
            return

        # Get unique groups for visualization
        unique_groups = sorted(set(all_labels_level1))

        # Generate UMAP plots for different parameter combinations
        n_neighbors_values = [15, 30, 50]
        min_dist_values = [0.1, 0.25, 0.5]

        for n_neighbors in n_neighbors_values:
            for min_dist in min_dist_values:
                logger.info(f"Fitting bypass UMAP with n_neighbors={n_neighbors}, min_dist={min_dist}")

                try:
                    # Create UMAP mapper
                    mapper = UMAP(
                        n_neighbors=n_neighbors,
                        min_dist=min_dist,
                        n_components=2,
                        metric='euclidean',
                        random_state=42,
                        verbose=False
                    )

                    # Fit and transform
                    embeddings_2d = mapper.fit_transform(all_features)

                    # Create plots for each group
                    for focus_group in unique_groups:
                        plt.figure(figsize=(14, 12))

                        # Get data types in this group
                        group_data_types = sorted(set([
                            dt for dt, g in zip(all_labels_level2, all_labels_level1)
                            if g == focus_group
                        ]))

                        if not group_data_types:
                            plt.close()
                            continue

                        # Create color map
                        color_map = {}
                        colors = plt.cm.tab20(np.linspace(0, 1, len(group_data_types)))
                        for i, dt in enumerate(group_data_types):
                            color_map[dt] = colors[i]

                        # Plot points from other groups (grayed out)
                        other_indices = [i for i, g in enumerate(all_labels_level1) if g != focus_group]
                        if other_indices:
                            plt.scatter(
                                embeddings_2d[other_indices, 0],
                                embeddings_2d[other_indices, 1],
                                c='lightgray',
                                s=30,
                                alpha=0.3,
                                label='Other groups'
                            )

                        # Plot focus group points
                        for dt in group_data_types:
                            dt_indices = [
                                i for i, (data_type, g) in enumerate(zip(all_labels_level2, all_labels_level1))
                                if data_type == dt and g == focus_group
                            ]

                            if not dt_indices:
                                continue

                            # Get colors based on rotation speed
                            dt_speeds = [all_rot_speeds[i] for i in dt_indices]
                            if max(dt_speeds) > min(dt_speeds):
                                norm_speeds = 0.3 + 0.7 * (np.array(dt_speeds) - min(dt_speeds)) / (max(dt_speeds) - min(dt_speeds))
                            else:
                                norm_speeds = np.ones_like(dt_speeds)

                            scatter_colors = []
                            base_color = color_map[dt]
                            for speed_sat in norm_speeds:
                                hsv = mcolors.rgb_to_hsv(base_color[:3])
                                hsv[1] = speed_sat
                                rgb = mcolors.hsv_to_rgb(hsv)
                                scatter_colors.append(rgb)

                            plt.scatter(
                                embeddings_2d[dt_indices, 0],
                                embeddings_2d[dt_indices, 1],
                                c=scatter_colors,
                                s=40,
                                alpha=0.8
                            )

                            # Extract variant name from data type for cleaner legend
                            # e.g., "imbalance_fault_5g" -> "5g", "horizontal_misalignment_fault_0.51mm" -> "0.51mm"
                            variant_name = extract_variant_name(dt)
                            plt.scatter([], [], c=[base_color], s=40, label=variant_name)

                        # Add colorbar for rotation speed
                        if max(all_rot_speeds) > min(all_rot_speeds):
                            sm = plt.cm.ScalarMappable(cmap=plt.cm.Greys, norm=plt.Normalize(vmin=0, vmax=1))
                            sm.set_array([])
                            cbar = plt.colorbar(sm, ax=plt.gca(), label='Rotation Frequency (normalized)', orientation='vertical', pad=0.05)
                            cbar.ax.tick_params(labelsize=18)
                            cbar.set_label('Rotation Frequency (normalized)', fontsize=18)

                        plt.title(f'Bypass UMAP - {focus_group.capitalize()}\n(n_neighbors={n_neighbors}, min_dist={min_dist})', fontsize=18)
                        plt.xlabel('UMAP dimension 1', fontsize=18)
                        plt.ylabel('UMAP dimension 2', fontsize=18)
                        plt.legend(loc='upper right', title="Data Types", fontsize=16, title_fontsize=18)
                        plt.tight_layout()

                        # Save plot
                        param_dir = os.path.join(output_dir, f"n{n_neighbors}_d{min_dist}")
                        os.makedirs(param_dir, exist_ok=True)
                        save_path = os.path.join(param_dir, f"bypass_umap_{focus_group}_group.png")
                        plt.savefig(save_path, dpi=300, bbox_inches='tight')
                        plt.close()

                except Exception as e:
                    logger.error(f"Error creating bypass UMAP visualization: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        logger.info("Bypass UMAP plots generation complete!")

    except Exception as e:
        logger.error(f"Error in bypass UMAP generation: {e}")
        import traceback
        traceback.print_exc() 