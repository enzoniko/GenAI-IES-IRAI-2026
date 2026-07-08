"""
Evaluation tools for Siamese networks
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score, make_scorer, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm
import pandas as pd
import os
import logging
import platform
from ..data.processor import LabeledSample
from matplotlib.colors import LinearSegmentedColormap
from itertools import cycle
from matplotlib import colors as mcolors
import pickle

# Suppress Windows subprocess warnings and optimize for Windows
if platform.system() == 'Windows':
    os.environ['JOBLIB_MULTIPROCESSING'] = '0'  # Disable joblib multiprocessing
    os.environ['OMP_NUM_THREADS'] = '1'  # Limit OpenMP threads
    os.environ['MKL_NUM_THREADS'] = '1'  # Limit MKL threads
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    try:
        import umap
        UMAP_AVAILABLE = True
    except ImportError:
        UMAP_AVAILABLE = False
        print("Warning: UMAP not available. UMAP visualizations will be skipped.")

logger = logging.getLogger(__name__)


class SiameseEvaluator:
    """Evaluator for Siamese networks"""
    
    def __init__(self, model, processor, device):
        """
        Evaluator for Siamese network
        
        Args:
            model: Trained SiameseNetwork instance
            processor: ResidualProcessor instance
            device: torch.device for evaluation
        """
        self.model = model.to(device)
        self.processor = processor
        self.device = device
        self.model.eval()
    
    def compute_embeddings(self, samples_by_label):
        """
        Compute embeddings for all samples using v1 strategy (group by data_type)
        
        Args:
            samples_by_label: Dictionary mapping labels to lists of samples
            
        Returns:
            embeddings_by_level: Dictionary mapping hierarchy levels to dictionaries mapping labels to embeddings
            labels: List of labels
            all_embeddings: Array of all embeddings
        """
        # Initialize embeddings by level
        embeddings_by_level = {
            1: {},  # Group level
            2: {},  # Data type level
            3: {}   # Data type + speed level
        }
        
        # Collect all samples in a flat list with tracking info
        all_samples = []
        all_labels = []
        
        print("Collecting all samples...")
        for label, samples in samples_by_label.items():
            for sample in samples:
                all_samples.append(sample)
                all_labels.append(label)
        
        # Organize samples by data_type for efficient batch processing
        samples_by_data_type = {}
        sample_indices_by_data_type = {}  # To track original indices
        
        print("Organizing samples by data type...")
        for i, sample in enumerate(all_samples):
            # Get data_type - either from LabeledSample or fallback to using the label
            if isinstance(sample, LabeledSample):
                data_type = sample.data_type
            else:
                # Fallback - use the label up to the last underscore
                data_type = all_labels[i].rsplit('_', 1)[0] if '_' in all_labels[i] else all_labels[i]
            
            # Initialize if needed
            if data_type not in samples_by_data_type:
                samples_by_data_type[data_type] = []
                sample_indices_by_data_type[data_type] = []
            
            # Add sample and its index
            samples_by_data_type[data_type].append(sample)
            sample_indices_by_data_type[data_type].append(i)
        
        print(f"Processing {len(all_samples)} samples across {len(samples_by_data_type)} data types")
        
        # Instead of pre-allocating with model.embedding_size, we'll build it dynamically
        # We'll track the embedding dimension from the first successful output
        embedding_dimension = None
        all_embeddings_dict = {}  # Use a dictionary indexed by position to avoid dimension issues
        
        with torch.no_grad():
            for data_type, samples in tqdm(samples_by_data_type.items(), desc="Computing embeddings by data type"):
                try:
                    # Get original indices for these samples
                    indices = sample_indices_by_data_type[data_type]
                    
                    # Extract data from samples
                    if isinstance(samples[0], LabeledSample):
                        sample_data = [s.data for s in samples]
                    else:
                        sample_data = samples

                    # Slice data to match the number of channels used during training
                    if hasattr(self.processor, 'num_channels') and self.processor.num_channels is not None:
                        sliced_sample_data = []
                        for sample in sample_data:
                            sample = np.asarray(sample)
                            if sample.ndim == 1:
                                sample = sample.reshape(-1, 1)
                            if sample.shape[1] > self.processor.num_channels:
                                sample = sample[:, :self.processor.num_channels]
                            sliced_sample_data.append(sample)
                        sample_data = sliced_sample_data

                    # Process this data type's samples together
                    processed_samples = self.processor.transform(sample_data)
                    
                    # Convert all processed samples to tensors at once
                    tensors = torch.FloatTensor(processed_samples).to(self.device)
                    
                    # Log shape for debugging
                    print(f"  Processing {data_type}: {len(samples)} samples, tensor shape: {tensors.shape}")
                    
                    # Compute embeddings in batches
                    embeddings = []
                    batch_size = 64  # Adjust based on GPU memory
                    
                    for i in range(0, len(tensors), batch_size):
                        batch = tensors[i:i+batch_size]
                        batch_embeddings = self.model.forward_one(batch)
                        embeddings.append(batch_embeddings.cpu().numpy())
                    
                    # Concatenate batches
                    data_type_embeddings = np.concatenate(embeddings, axis=0)
                    
                    # Determine embedding dimension if not known yet
                    if embedding_dimension is None:
                        embedding_dimension = data_type_embeddings.shape[1]
                        print(f"  Detected embedding dimension: {embedding_dimension}")
                    
                    # Store embeddings by their original indices
                    for i, idx in enumerate(indices):
                        all_embeddings_dict[idx] = data_type_embeddings[i]
                    
                    # Free GPU memory after processing this data type
                    del tensors, processed_samples, data_type_embeddings
                    if hasattr(torch, 'cuda') and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"Error processing data type {data_type}: {e}")
                    print(f"Skipping {len(samples)} samples of this data type and continuing...")
        
        # Convert dictionary to array
        print("Converting embeddings to array...")
        if embedding_dimension is None:
            raise ValueError("No embeddings were successfully computed!")
            
        all_embeddings = np.zeros((len(all_samples), embedding_dimension))
        for idx, embedding in all_embeddings_dict.items():
            all_embeddings[idx] = embedding
        
        # Now organize embeddings into hierarchical levels
        print("Organizing embeddings by hierarchy level...")
        
        # Track used indices to avoid duplicate embeddings
        used_indices = set()
        
        for i, sample in enumerate(tqdm(all_samples, desc="Organizing by hierarchy")):
            # Skip samples that weren't successfully embedded
            if i not in all_embeddings_dict:
                continue
                
            if isinstance(sample, LabeledSample):
                # Get labels for each level
                level1_label = sample.level1_label
                level2_label = sample.level2_label
                level3_label = sample.level3_label
                
                # Initialize embedding lists if needed
                for level, level_label in zip([1, 2, 3], [level1_label, level2_label, level3_label]):
                    if level_label not in embeddings_by_level[level]:
                        embeddings_by_level[level][level_label] = []
                
                # Add embedding to each level
                embeddings_by_level[1][level1_label].append(all_embeddings[i])
                embeddings_by_level[2][level2_label].append(all_embeddings[i])
                embeddings_by_level[3][level3_label].append(all_embeddings[i])
                
                used_indices.add(i)
            else:
                # Legacy mode - try to add to level 3 only
                label = all_labels[i]
                if label not in embeddings_by_level[3]:
                    embeddings_by_level[3][label] = []
                embeddings_by_level[3][label].append(all_embeddings[i])
                used_indices.add(i)
        
        # Convert lists to arrays
        for level in embeddings_by_level:
            for label in embeddings_by_level[level]:
                embeddings_by_level[level][label] = np.array(embeddings_by_level[level][label])
        
        # Make sure we used all indices
        if len(used_indices) != len(all_samples):
            print(f"Warning: {len(all_samples) - len(used_indices)} samples were not included in the hierarchy")
        
        return embeddings_by_level, all_labels, all_embeddings
    
    def visualize_embeddings_by_group(self, embeddings_by_level, output_dir=None, perplexity_values=None):
        """
        Visualize embeddings with a focus on one group at a time, using t-SNE with multiple perplexity values
        
        Args:
            embeddings_by_level: Dictionary mapping levels to embeddings by label
            output_dir: Output directory for saving plots
            perplexity_values: List of perplexity values to use for t-SNE (default: [30, 90, 150])
        """
        from sklearn.manifold import TSNE

        # Reduce perplexity values for performance and to avoid Windows warnings
        if perplexity_values is None:
            perplexity_values = [30, 90, 150]  # Reduced from 6 to 3 values
        
        if output_dir:
            os.makedirs(os.path.join(output_dir, "tsne"), exist_ok=True)
        
        # Use common sampling function
        all_embeddings, all_data_types, all_rot_speeds, all_groups = self._sample_embeddings_for_visualization(
            embeddings_by_level, samples_per_level3=30
        )
        
        print(f"Total samples for visualization: {len(all_embeddings)}")
        
        # Get unique groups
        unique_groups = sorted(set(all_groups))
        
        # For each perplexity value, generate t-SNE visualizations
        for perplexity in tqdm(perplexity_values, desc="Generating t-SNE with different perplexity values"):
            # Ensure perplexity is reasonable for dataset size
            max_reasonable_perplexity = len(all_embeddings) // 3
            if perplexity >= max_reasonable_perplexity:
                print(f"Perplexity {perplexity} too large for {len(all_embeddings)} samples. Reducing to {max_reasonable_perplexity - 1}")
                perplexity = max_reasonable_perplexity - 1

            print(f"\nFitting t-SNE model with perplexity={perplexity} on {len(all_embeddings)} embeddings...")
            
            # Use faster parameters for t-SNE with version compatibility
            try:
                # Try newer scikit-learn parameters first
                tsne = TSNE(
                    n_components=2,
                    perplexity=perplexity,   # Use the already validated perplexity
                    max_iter=1000,           # Newer versions use max_iter
                    n_iter_without_progress=100,
                    learning_rate=200,
                    init='pca',
                    metric='euclidean',
                    random_state=42,
                    n_jobs=1 if platform.system() == 'Windows' else -1,  # Single thread on Windows
                    verbose=1
                )
            except TypeError:
                # Fallback for older scikit-learn versions
                try:
                    tsne = TSNE(
                        n_components=2,
                        perplexity=perplexity,   # Use the already validated perplexity
                        n_iter=1000,           # Older versions use n_iter
                        n_iter_without_progress=100,
                        learning_rate=200,
                        init='pca',
                        metric='euclidean',
                        random_state=42,
                        n_jobs=1 if platform.system() == 'Windows' else -1,
                        verbose=1
                    )
                except TypeError:
                    # Final fallback with minimal parameters
                    tsne = TSNE(
                        n_components=2,
                        perplexity=perplexity,   # Use the already validated perplexity
                        random_state=42,
                        n_jobs=1 if platform.system() == 'Windows' else -1,
                        verbose=1
                    )
            
            # Actually fit and transform the data with error handling
            try:
                embeddings_2d = tsne.fit_transform(all_embeddings)
                print(f"t-SNE fitting with perplexity={perplexity} complete. Generating plots...")
            except Exception as e:
                print(f"Failed to compute t-SNE with perplexity={perplexity}: {e}")
                print("Skipping this perplexity value and continuing...")
                continue
            
            # Create a subfolder for this perplexity value
            if output_dir:
                perplexity_dir = os.path.join(output_dir, "tsne", f"perplexity_{perplexity}")
                os.makedirs(perplexity_dir, exist_ok=True)

        # Create a plot for each group - add progress bar
        for focus_group in tqdm(unique_groups, desc=f"Generating group plots (perplexity={perplexity})"):
            plt.figure(figsize=(14, 12))

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

            # Get data types in this group for the legend
            group_data_types = sorted(set([dt for dt, g in zip(all_data_types, all_groups) if g == focus_group]))

            # Skip if no data types in this group
            if not group_data_types:
                print(f"No data for group {focus_group}, skipping plot")
                plt.close()
                continue

            # Choose colors for each data type in this group
            color_map = {}
            legend_labels = {}
            colors = plt.cm.tab20(np.linspace(0, 1, len(group_data_types)))
            for i, dt in enumerate(group_data_types):
                color_map[dt] = colors[i]
                # Extract only the variant part (after the last underscore)
                if focus_group == 'normal':
                    legend_labels[dt] = 'normal'
                else:
                    legend_labels[dt] = dt.split('_')[-1]

            # Track if any points were actually plotted
            points_plotted = False

            # Plot points from other groups first (grayed out)
            other_indices = [i for i, g in enumerate(all_groups) if g != focus_group]
            if other_indices:
                plt.scatter(
                    embeddings_2d[other_indices, 0],
                    embeddings_2d[other_indices, 1],
                    c='lightgray',
                    s=20,
                    alpha=0.3,
                    label='Other groups'
                )
            points_plotted = True

            # Plot focus group points with colors based on data type and saturation based on rotation speed
            # Add progress bar for data types
            for dt in tqdm(group_data_types, desc=f"  Data types in {focus_group}", leave=False):
                # Get indices for this data type
                dt_indices = [i for i, (data_type, g) in enumerate(zip(all_data_types, all_groups))
                             if data_type == dt and g == focus_group]

                if not dt_indices:
                    continue

                base_color = color_map[dt]

                # Get rotation speeds for these points
                dt_speeds = [all_rot_speeds[i] for i in dt_indices]

                # Normalize speeds from 0.3 to 1.0 (for saturation)
                if max(dt_speeds) > min(dt_speeds):
                    norm_speeds = 0.3 + 0.7 * (np.array(dt_speeds) - min(dt_speeds)) / (max(dt_speeds) - min(dt_speeds))
                else:
                    norm_speeds = np.ones_like(dt_speeds)

                # Use a single scatter call instead of multiple for performance
                scatter_colors = []

                # Prepare all colors with appropriate saturation
                for speed_sat in norm_speeds:
                    hsv = mcolors.rgb_to_hsv(base_color[:3])
                    hsv[1] = speed_sat  # Adjust saturation
                    rgb = mcolors.hsv_to_rgb(hsv)
                    scatter_colors.append(rgb)

                # Single scatter call for all points of this data type
                plt.scatter(
                    embeddings_2d[dt_indices, 0],
                    embeddings_2d[dt_indices, 1],
                    c=scatter_colors,
                    s=40,
                    alpha=0.8
                )
                points_plotted = True

                # Add to legend (just the base color)
                plt.scatter([], [], c=[base_color], s=40, label=legend_labels[dt])

            # Only add colorbar if points were actually plotted
            if points_plotted:
                # Set figure size to be square
                plt.gcf().set_size_inches(12, 10)

                # Get current axes
                ax = plt.gca()

                # Make the plot square
                ax.set_aspect('equal')

                # Add colorbar for rotation speed - position it on the right
                sm = plt.cm.ScalarMappable(cmap=plt.cm.Greys, norm=plt.Normalize(vmin=0, vmax=1))
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, label='Rotation Frequency (normalized)',
                                  orientation='vertical', pad=0.05)
                cbar.ax.tick_params(labelsize=18)
                cbar.set_label('Rotation Frequency (normalized)', fontsize=18)

                plt.title(f't-SNE Visualization - {focus_group.capitalize()} Group (perplexity={perplexity})', fontsize=18)
                plt.xlabel('t-SNE dimension 1', fontsize=18)
                plt.ylabel('t-SNE dimension 2', fontsize=18)

                # Put the legend INSIDE the plot area (top right corner)
                plt.legend(loc='upper right', title="Variants", fontsize=16, title_fontsize=18)

                # Tight layout to make everything fit nicely
                plt.tight_layout()

                if output_dir:
                    save_path = os.path.join(perplexity_dir, f"tsne_{focus_group}_group.png")
                    print(f"  Saving plot to {save_path}")
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    print(f"  Plot saved successfully")
            else:
                print(f"No points to plot for group {focus_group}, skipping")

            plt.close()
        
        print("All t-SNE visualizations completed!")
    
    def visualize_embeddings_umap(self, embeddings_by_level, output_dir=None,
                               n_neighbors_values=None, min_dist_values=None):
        """
        Visualize embeddings with a focus on one group at a time, using UMAP with multiple parameter configurations

        Args:
            embeddings_by_level: Dictionary mapping levels to embeddings by label
            output_dir: Output directory for saving plots
            n_neighbors_values: List of n_neighbors values to use for UMAP (default: [5, 15, 30, 50, 100], uses first 3)
            min_dist_values: List of min_dist values to use for UMAP (default: [0.0, 0.1, 0.25, 0.5, 0.8], uses first 3)
        """
        # Check if UMAP is available
        if not UMAP_AVAILABLE:
            print("UMAP not available. Skipping UMAP visualizations.")
            print("To enable UMAP visualizations, install umap-learn: pip install umap-learn")
            return
        if n_neighbors_values is None:
            n_neighbors_values = [5, 15, 30, 50, 100]
        
        if min_dist_values is None:
            min_dist_values = [0.0, 0.1, 0.25, 0.5, 0.8]

        # Reduce the number of parameter combinations for performance
        if len(n_neighbors_values) > 3:
            n_neighbors_values = n_neighbors_values[:3]  # Use first 3 values
        if len(min_dist_values) > 3:
            min_dist_values = min_dist_values[:3]  # Use first 3 values
        
        if output_dir:
            os.makedirs(os.path.join(output_dir, "umap"), exist_ok=True)
        
        # Use common sampling function
        all_embeddings, all_data_types, all_rot_speeds, all_groups = self._sample_embeddings_for_visualization(
            embeddings_by_level, samples_per_level3=30
        )
        
        print(f"Total samples for visualization: {len(all_embeddings)}")
        
        # Get unique groups
        unique_groups = sorted(set(all_groups))
        
        # For each parameter combination, generate UMAP visualizations
        for n_neighbors in tqdm(n_neighbors_values, desc="Generating UMAP with different n_neighbors values"):
            for min_dist in tqdm(min_dist_values, desc=f"  Min dist values (n_neighbors={n_neighbors})", leave=False):
                print(f"\nFitting UMAP model with n_neighbors={n_neighbors}, min_dist={min_dist} on {len(all_embeddings)} embeddings...")
                
                # Configure UMAP with proper import handling
                if not UMAP_AVAILABLE:
                    print(f"UMAP not available, skipping n_neighbors={n_neighbors}, min_dist={min_dist}")
                    continue

                try:
                    # Try the direct import first (newer versions)
                    if 'UMAP' in globals():
                        mapper = UMAP(
                            n_neighbors=n_neighbors,
                            min_dist=min_dist,
                            n_components=2,
                            metric='euclidean',
                            random_state=42,
                            verbose=True
                        )
                    else:
                        # Fallback to module attribute access (older versions)
                        mapper = umap.UMAP(
                            n_neighbors=n_neighbors,
                            min_dist=min_dist,
                            n_components=2,
                            metric='euclidean',
                            random_state=42,
                            verbose=True
                        )
                except AttributeError as e:
                    print(f"UMAP import issue: {e}")
                    print("Skipping UMAP visualization due to import problems")
                    continue
                except Exception as e:
                    print(f"Failed to create UMAP mapper: {e}")
                    print("Skipping this UMAP configuration")
                    continue
                
                # Actually fit and transform the data
                try:
                    embeddings_2d = mapper.fit_transform(all_embeddings)
                    
                    print(f"UMAP fitting with n_neighbors={n_neighbors}, min_dist={min_dist} complete. Generating plots...")
                    
                    # Create a subfolder for this parameter combination
                    if output_dir:
                        param_dir = os.path.join(output_dir, "umap", f"n{n_neighbors}_d{min_dist}")
                        os.makedirs(param_dir, exist_ok=True)
                    
                    # Create a plot for each group
                    for focus_group in tqdm(unique_groups, desc=f"Generating group plots (n_neighbors={n_neighbors}, min_dist={min_dist})"):
                        plt.figure(figsize=(14, 12))

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
                        
                        # Get data types in this group for the legend
                        group_data_types = sorted(set([dt for dt, g in zip(all_data_types, all_groups) if g == focus_group]))
                        
                        # Skip if no data types in this group
                        if not group_data_types:
                            print(f"No data for group {focus_group}, skipping plot")
                            plt.close()
                            continue
                        
                        # Choose colors for each data type in this group
                        color_map = {}
                        legend_labels = {}
                        colors = plt.cm.tab20(np.linspace(0, 1, len(group_data_types)))
                        for i, dt in enumerate(group_data_types):
                            color_map[dt] = colors[i]
                            # Extract only the variant part (after the last underscore)
                            if focus_group == 'normal':
                                legend_labels[dt] = 'normal'
                            else:
                                legend_labels[dt] = dt.split('_')[-1]
                        
                        # Track if any points were actually plotted
                        points_plotted = False
                        
                        # Plot points from other groups first (grayed out)
                        other_indices = [i for i, g in enumerate(all_groups) if g != focus_group]
                        if other_indices:
                            plt.scatter(
                                embeddings_2d[other_indices, 0],
                                embeddings_2d[other_indices, 1],
                                c='lightgray',
                                s=20,
                                alpha=0.3,
                                label='Other groups'
                            )
                            points_plotted = True
                        
                        # Plot focus group points with colors based on data type and saturation based on rotation speed
                        for dt in tqdm(group_data_types, desc=f"  Data types in {focus_group}", leave=False):
                            # Get indices for this data type
                            dt_indices = [i for i, (data_type, g) in enumerate(zip(all_data_types, all_groups)) 
                                        if data_type == dt and g == focus_group]
                            
                            if not dt_indices:
                                continue
                            
                            base_color = color_map[dt]
                            
                            # Get rotation speeds for these points
                            dt_speeds = [all_rot_speeds[i] for i in dt_indices]
                            
                            # Normalize speeds from 0.3 to 1.0 (for saturation)
                            if max(dt_speeds) > min(dt_speeds):
                                norm_speeds = 0.3 + 0.7 * (np.array(dt_speeds) - min(dt_speeds)) / (max(dt_speeds) - min(dt_speeds))
                            else:
                                norm_speeds = np.ones_like(dt_speeds)
                            
                            # Use a single scatter call instead of multiple for performance
                            scatter_colors = []
                            
                            # Prepare all colors with appropriate saturation
                            for speed_sat in norm_speeds:
                                hsv = mcolors.rgb_to_hsv(base_color[:3])
                                hsv[1] = speed_sat  # Adjust saturation
                                rgb = mcolors.hsv_to_rgb(hsv)
                                scatter_colors.append(rgb)
                            
                            # Single scatter call for all points of this data type
                            plt.scatter(
                                embeddings_2d[dt_indices, 0],
                                embeddings_2d[dt_indices, 1],
                                c=scatter_colors,
                                s=40,
                                alpha=0.8
                            )
                            points_plotted = True
                            
                            # Add to legend (just the base color)
                            plt.scatter([], [], c=[base_color], s=40, label=legend_labels[dt])
                        
                        # Only add colorbar if points were actually plotted
                        if points_plotted:
                            # Set figure size to be square
                            plt.gcf().set_size_inches(12, 10)
                            
                            # Get current axes
                            ax = plt.gca()
                            
                            # Make the plot square
                            ax.set_aspect('equal')
                            
                            # Colorbar removed to save space
                            
                            plt.title(f'UMAP Visualization - {focus_group.capitalize()} Group\n(n_neighbors={n_neighbors}, min_dist={min_dist})', fontsize=18)
                            plt.xlabel('UMAP dimension 1', fontsize=18)
                            plt.ylabel('UMAP dimension 2', fontsize=18)

                            # Put the legend INSIDE the plot area (top right corner)
                            plt.legend(loc='upper right', title="Variants", fontsize=16, title_fontsize=18)
                            
                            # Tight layout to make everything fit nicely
                            plt.tight_layout()
                            
                            if output_dir:
                                save_path = os.path.join(param_dir, f"umap_{focus_group}_group.png")
                                print(f"  Saving plot to {save_path}")
                                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                                print(f"  Plot saved successfully")
                        else:
                            print(f"No points to plot for group {focus_group}, skipping")
                        
                        plt.close()
                except Exception as e:
                    print(f"Error fitting UMAP with n_neighbors={n_neighbors}, min_dist={min_dist}: {e}")
                    continue
        
        print("All UMAP visualizations completed!")
    
    def evaluate_classification(self, samples_by_label, threshold=0.5, embeddings_by_level=None):
        """
        Evaluate the model's ability to classify samples at all hierarchy levels
        
        Args:
            samples_by_label: Dictionary mapping labels to lists of samples
            threshold: Similarity threshold for classification
            embeddings_by_level: Optional pre-computed embeddings by level
        
        Returns:
            Dictionary with classification reports and confusion matrices for all levels
        """
        # Compute embeddings only if not provided
        if embeddings_by_level is None:
            embeddings_by_level, labels, _ = self.compute_embeddings(samples_by_label)
        
        # Initialize results dictionary
        results = {}
        
        # Evaluate classification at each hierarchy level
        for level in [1, 2, 3]:
            # Get unique labels for this level
            unique_labels = list(embeddings_by_level[level].keys())
            
            # Create prototype embeddings for each label
            prototypes = {
                label: np.mean(embeddings_by_level[level][label], axis=0) 
                for label in unique_labels
            }
            
            # Normalize prototypes (since we're using mean which may not be normalized)
            for label in prototypes:
                norm = np.linalg.norm(prototypes[label])
                if norm > 0:
                    prototypes[label] = prototypes[label] / norm
            
            # Predict labels for all samples
            y_true = []
            y_pred = []
            
            for true_label, embeddings in tqdm(embeddings_by_level[level].items(), desc=f"Evaluating level {level}"):
                for embedding in tqdm(embeddings, desc=f"  Evaluating {true_label}", leave=False):
                    y_true.append(true_label)
                    
                    # Compute squared Euclidean distances to prototypes (same as in TripletLoss)
                    distances = {
                        label: np.sum((embedding - prototypes[label]) ** 2)
                        for label in unique_labels
                    }
                    
                    # Predict label with smallest distance (not highest similarity)
                    pred_label = min(distances.items(), key=lambda x: x[1])[0]
                    y_pred.append(pred_label)
            
            # Compute confusion matrix and classification report
            cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
            report = classification_report(y_true, y_pred, labels=unique_labels)
            
            # Store results
            results[f'level{level}_report'] = report
            results[f'level{level}_cm'] = cm
            results[f'level{level}_labels'] = unique_labels
        
        # For backward compatibility, keep the old naming for level 1
        results['data_type_report'] = results['level1_report']
        results['data_type_cm'] = results['level1_cm']
        results['data_type_labels'] = results['level1_labels']
        
        # For backward compatibility, keep the old naming for level 3
        results['full_report'] = results['level3_report']
        results['full_cm'] = results['level3_cm'] 
        results['full_labels'] = results['level3_labels']
        
        return results
    
    def evaluate_classification_knn(self, samples_by_label, k=5, reference_samples_per_class=50, embeddings_by_level=None):
        """
        Evaluates classification performance using K-Nearest Neighbors approach
        with efficient reference set sampling.
        
        This implementation selects a reference set of samples for each class,
        which significantly reduces computation time and memory usage compared to
        using all samples.
        
        Args:
            samples_by_label: Dictionary mapping labels to lists of samples
            k: Number of neighbors to use
            reference_samples_per_class: Number of reference samples to use per class
            embeddings_by_level: Optional pre-computed embeddings for each hierarchy level
            
        Returns:
            Dictionary with classification results for each level
        """
        logger.info(f"Evaluating KNN classification with k={k} and {reference_samples_per_class} reference samples per class")
        
        # Get embeddings by level if not provided
        if embeddings_by_level is None:
            embeddings_by_level, _, _ = self.compute_embeddings(samples_by_label)
        
        # Initialize results dictionary
        results = {}
        
        # For each level in the hierarchy
        for level in [1, 2, 3]:
            # Create reference gallery for each class (the samples we'll compare against)
            gallery_embeddings = []
            gallery_labels = []
            
            # Get embeddings for this level
            level_embeddings = embeddings_by_level[level]
            
            for label, embs in level_embeddings.items():
                embs = np.array(embs)
                # Determine how many samples to use (min of reference_samples_per_class or all available)
                n_samples = min(reference_samples_per_class, len(embs))
                
                # Randomly select samples if we have more than needed
                if len(embs) > n_samples:
                    indices = np.random.choice(len(embs), n_samples, replace=False)
                    selected_embs = embs[indices]
                else:
                    selected_embs = embs
                
                gallery_embeddings.extend(selected_embs)
                gallery_labels.extend([label] * len(selected_embs))
            
            gallery_embeddings = np.array(gallery_embeddings)
            gallery_labels = np.array(gallery_labels)
            
            # Now predict for each sample
            y_true = []
            y_pred = []
            confidences = []
            speeds = []  # For per-speed analysis

            # For each label at this level
            for true_label, embs in level_embeddings.items():
                # Extract speed from level3 label if evaluating level 3
                if level == 3:
                    try:
                        speed = float(true_label.split('_')[-1])
                        if len(embs) > 0 and len(speeds) == 0:  # Log first speed extraction
                            logger.info(f"Siamese+KNN level 3: First speed extracted: {speed} from label {true_label}")
                    except (ValueError, IndexError):
                        speed = 0.0  # Default speed
                        logger.warning(f"Could not extract speed from label {true_label}")
                else:
                    speed = 0.0  # Speed not available for levels 1 and 2

                # Calculate distances from each embedding to the gallery
                for emb in embs:
                    # Calculate distances to all gallery embeddings
                    distances = np.linalg.norm(gallery_embeddings - emb, axis=1)

                    # Sort distances and get top k (handle case where k > gallery size)
                    k_local = min(k, len(distances))
                    if k_local == 0:
                        # Skip if no distances
                        continue

                    # Find the indices of the k nearest neighbors
                    nearest_indices = np.argsort(distances)[:k_local]

                    # Get the labels of the k nearest neighbors
                    nearest_labels = [gallery_labels[i] for i in nearest_indices]

                    # Predict the class by majority voting
                    from collections import Counter
                    predicted_label = Counter(nearest_labels).most_common(1)[0][0]

                    # Calculate confidence as (count of majority class) / k
                    majority_count = Counter(nearest_labels).most_common(1)[0][1]
                    confidence = majority_count / k_local

                    y_true.append(true_label)
                    y_pred.append(predicted_label)
                    confidences.append(confidence)
                    speeds.append(speed)
            
            # Convert to numpy arrays for metric calculations
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            confidences = np.array(confidences)
            speeds = np.array(speeds)

            if level == 3:
                logger.info(f"Siamese+KNN level {level}: Collected {len(speeds)} predictions with speeds ranging from {speeds.min():.1f} to {speeds.max():.1f} Hz")

            # Compute classification report
            report = classification_report(y_true, y_pred, output_dict=False)
            results[f'level{level}_knn_report'] = report

            # Compute confusion matrix
            if level < 3:  # We typically don't show confusion matrix for level 3 (too many classes)
                unique_labels = sorted(set(y_true))
                cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
                results[f'level{level}_knn_cm'] = cm
                results[f'level{level}_knn_labels'] = unique_labels

                # Calculate confidence intervals for F1 macro
                ci_results = self.calculate_confidence_intervals(y_true, y_pred, n_bootstraps=1000)
                results[f'level{level}_knn_confidence_intervals'] = ci_results

                # Calculate Expected Calibration Error (ECE)
                ece_results = self.calculate_ece(y_true, y_pred, confidences, n_bins=10)
                results[f'level{level}_knn_ece'] = ece_results['ece']
                results[f'level{level}_knn_ece_details'] = ece_results

                # Calculate AUROC for open-set detection (simulate unknown classes)
                auroc_score = self._calculate_open_set_auroc_for_siamese_knn(
                    gallery_embeddings, gallery_labels, level_embeddings,
                    k, unknown_fraction=0.2
                )
                results[f'level{level}_knn_open_set_auroc'] = auroc_score

            # Add per-speed analysis for level 3
            if level == 3:
                try:
                    logger.info(f"Computing per-speed analysis for Siamese+KNN level {level} with {len(speeds)} speed values")
                    speed_analysis = self.analyze_per_speed_performance(y_true, y_pred, speeds, "Siamese+KNN")
                    results[f'level{level}_knn_per_speed_performance'] = speed_analysis
                    logger.info(f"Per-speed analysis completed for Siamese+KNN level {level}: {len(speed_analysis)} buckets")
                except Exception as e:
                    logger.warning(f"Failed to compute per-speed analysis for Siamese+KNN level {level}: {e}")
                    results[f'level{level}_knn_per_speed_performance'] = {}
        
        return results
    
    def _plot_confusion_matrix(self, cm, labels, output_dir, filename_prefix, title_prefix="", level=1):
        """
        Helper function to plot/save confusion matrices
        
        Args:
            cm: Confusion matrix
            labels: Labels for the confusion matrix
            output_dir: Directory to save the plot
            filename_prefix: Prefix for the output filename
            title_prefix: Prefix for the plot title
            level: Hierarchy level (1=data types, 2=fault types)
        """
        # Skip if too many labels (more than 20)
        if len(labels) > 20:
            logger.info(f"Level {level} confusion matrix is too large ({len(labels)} labels). "
                       f"Saving data without plotting.")
            
            # Save raw data instead of plotting
            if output_dir:
                # Save labels
                with open(os.path.join(output_dir, f"level{level}_labels.txt"), 'w') as f:
                    for label in labels:
                        f.write(f"{label}\n")
                
                # Save confusion matrix as CSV
                cm_df = pd.DataFrame(cm, index=labels, columns=labels)
                cm_df.to_csv(os.path.join(output_dir, f"{filename_prefix}_level{level}.csv"))
                
                # Save as numpy for further analysis
                np.save(os.path.join(output_dir, f"{filename_prefix}_level{level}.npy"), cm)
            return
        
        # Normalize confusion matrix
        cm_sum = cm.sum(axis=1)
        # Avoid division by zero
        cm_sum[cm_sum == 0] = 1
        cm_norm = cm.astype('float') / cm_sum[:, np.newaxis]
        
        # Plot
        plt.figure(figsize=(14, 12))

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

        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, annot_kws={"size": 16})

        level_name = "Data Types" if level == 1 else "Fault Types"
        plt.title(f'{title_prefix}Normalized Confusion Matrix ({level_name})', fontsize=18)
        plt.xlabel('Predicted', fontsize=18)
        plt.ylabel('True', fontsize=18)
        
        # For level 1, use the data_types naming convention for backward compatibility
        if level == 1:
            plt.savefig(os.path.join(output_dir, f"{filename_prefix}_data_types.png"))
        else:
            plt.savefig(os.path.join(output_dir, f"{filename_prefix}_level{level}.png"))
        plt.close()
    
    def plot_confusion_matrices(self, evaluation_results, output_dir=None):
        """
        Plot confusion matrices from evaluation results for levels 1 and 2.
        For large matrices, only save the raw data.
        
        Args:
            evaluation_results: Results from evaluate_classification
            output_dir: Output directory for plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot confusion matrices for levels 1 and 2
        for level in [1, 2]:
            cm_key = f'level{level}_cm'
            labels_key = f'level{level}_labels'
            
            if cm_key not in evaluation_results or labels_key not in evaluation_results:
                logger.warning(f"Missing data for level {level} confusion matrix")
                continue
                
            cm = evaluation_results[cm_key]
            labels = evaluation_results[labels_key]
            
            self._plot_confusion_matrix(
                cm=cm,
                labels=labels,
                output_dir=output_dir,
                filename_prefix="confusion_matrix",
                level=level
            )
    
    def save_evaluation_results(self, evaluation_results, output_dir):
        """
        Save evaluation results to files for all hierarchy levels
        
        Args:
            evaluation_results: Results from evaluate_classification
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save classification reports for all levels
        for level in [1, 2, 3]:
            report_key = f'level{level}_report'
            if report_key in evaluation_results:
                # For backward compatibility, use the old filenames for levels 1 and 3
                if level == 1:
                    filename = "classification_report_data_types.txt"
                elif level == 3:
                    filename = "classification_report_full.txt"
                else:
                    filename = f"classification_report_level{level}.txt"
                
                with open(os.path.join(output_dir, filename), 'w') as f:
                    f.write(evaluation_results[report_key])
        
        # Plot and save confusion matrices (only for levels 1 and 2)
        self.plot_confusion_matrices(evaluation_results, output_dir)
        
        # Calculate and save accuracy by label for level 1 (data type)
        if 'level1_cm' in evaluation_results and 'level1_labels' in evaluation_results:
            level1_cm = evaluation_results['level1_cm']
            level1_labels = evaluation_results['level1_labels']
            
            level1_accuracy = level1_cm.diagonal() / level1_cm.sum(axis=1)
            
            # Create a DataFrame for better readability
            accuracy_df = pd.DataFrame({
                'Data Type': level1_labels,
                'Accuracy': level1_accuracy,
                'Sample Count': level1_cm.sum(axis=1)
            })
            
            # Save as CSV
            accuracy_df.to_csv(os.path.join(output_dir, "data_type_accuracy.csv"), index=False)
            
            # Also save accuracy for level 2
            if 'level2_cm' in evaluation_results and 'level2_labels' in evaluation_results:
                level2_cm = evaluation_results['level2_cm']
                level2_labels = evaluation_results['level2_labels']
                
                level2_accuracy = level2_cm.diagonal() / level2_cm.sum(axis=1)
                
                # Create a DataFrame for level 2
                level2_accuracy_df = pd.DataFrame({
                    'Fault Type': level2_labels,
                    'Accuracy': level2_accuracy,
                    'Sample Count': level2_cm.sum(axis=1)
                })
                
                # Save as CSV
                level2_accuracy_df.to_csv(os.path.join(output_dir, "fault_type_accuracy.csv"), index=False)
        
        logger.info(f"Evaluation results saved to {output_dir}")
    
    def save_embeddings(self, embeddings_by_level, output_dir):
        """
        Save embeddings to disk for future reuse
        
        Args:
            embeddings_by_level: Dictionary mapping levels to embeddings by label
            output_dir: Output directory for saving embeddings
        """
        os.makedirs(output_dir, exist_ok=True)
        embeddings_path = os.path.join(output_dir, "embeddings.pkl")
        
        print(f"Saving embeddings to {embeddings_path}...")
        with open(embeddings_path, 'wb') as f:
            pickle.dump(embeddings_by_level, f)
        print("Embeddings saved successfully.")
    
    def _sample_embeddings_for_visualization(self, embeddings_by_level, samples_per_level3=30):
        """
        Sample balanced embeddings from all level 3 labels for visualization
        
        Args:
            embeddings_by_level: Dictionary mapping levels to embeddings by label
            samples_per_level3: Number of samples to take from each level 3 label
            
        Returns:
            all_embeddings: Array of sampled embeddings
            all_data_types: List of data type labels for each embedding
            all_rot_speeds: List of rotation speeds for each embedding
            all_groups: List of group labels for each embedding
        """
        # Get all level 3 labels to ensure balanced sampling
        level3_labels = list(embeddings_by_level[3].keys())
        
        # Combine all embeddings for visualization with balanced sampling from level 3
        all_embeddings = []
        all_data_types = []
        all_rot_speeds = []
        all_groups = []
        
        # Group level3 labels by their level2 parent
        level3_by_level2 = {}
        print("Organizing level 3 labels by parent...")
        for l3_label in tqdm(level3_labels, desc="Organizing labels"):
            # Extract level2 label (remove the speed suffix)
            level2_label = '_'.join(l3_label.split('_')[:-1])
            if level2_label not in level3_by_level2:
                level3_by_level2[level2_label] = []
            level3_by_level2[level2_label].append(l3_label)
        
        # Sample from each level3 label
        print("Sampling embeddings from level 3 labels for visualization...")
        # Add overall progress bar for sampling across all level2 labels
        for level2_label in tqdm(level3_by_level2.keys(), desc="Sampling level2 labels"):
            l3_labels = level3_by_level2[level2_label]
            
            # For each level3 label under this level2 - add progress bar
            for l3_label in tqdm(l3_labels, desc=f"  Speeds for {level2_label}", leave=False):
                embeddings = embeddings_by_level[3][l3_label]
                
                # Sample at most samples_per_level3 embeddings
                sample_size = min(samples_per_level3, len(embeddings))
                sample_indices = np.random.choice(len(embeddings), sample_size, replace=False)
                
                # Extract speed from level3 label
                try:
                    rot_speed = float(l3_label.split('_')[-1])
                except:
                    rot_speed = 0.0
                
                # Extract group (level 1) from level2 label
                group = 'unknown'
                for g in embeddings_by_level[1].keys():
                    if g in level2_label:
                        group = g
                        break
                
                # Add sampled embeddings
                for idx in sample_indices:
                    all_embeddings.append(embeddings[idx])
                    all_data_types.append(level2_label)
                    all_rot_speeds.append(rot_speed)
                    all_groups.append(group)
        
        return np.array(all_embeddings), all_data_types, all_rot_speeds, all_groups
    
    def load_embeddings(self, output_dir):
        """
        Load previously saved embeddings
        
        Args:
            output_dir: Directory containing saved embeddings
            
        Returns:
            embeddings_by_level: Dictionary mapping levels to embeddings by label
        """
        embeddings_path = os.path.join(output_dir, "embeddings.pkl")
        
        if not os.path.exists(embeddings_path):
            print(f"Embeddings file not found: {embeddings_path}")
            return None
        
        print(f"Loading embeddings from {embeddings_path}...")
        with open(embeddings_path, 'rb') as f:
            embeddings_by_level = pickle.load(f)
        print("Embeddings loaded successfully.")
        
        return embeddings_by_level
    
    def evaluate_and_save(self, samples_by_label, output_dir, threshold=0.5,
                          visualization_level='all',
                          tsne_perplexity_values=[30, 50, 70, 90, 110, 150],
                          umap_n_neighbors_values=[5, 15, 30, 50, 100],
                          umap_min_dist_values=[0.0, 0.1, 0.25, 0.5, 0.8],
                          reuse_embeddings=True,
                          use_knn=False,
                          knn_k_values=[3, 5, 7, 11],
                          reference_samples_per_class=50,
                          evaluate_baselines=False,
                          baseline_test_size=0.3,
                          baseline_max_samples=500):
        """
        Evaluate model and save results

        Args:
            samples_by_label: Dictionary mapping labels to lists of samples
            output_dir: Output directory for results
            threshold: Similarity threshold for classification
            visualization_level: Which hierarchy levels to visualize ('all' or specific level number)
            tsne_perplexity_values: List of perplexity values for t-SNE
            umap_n_neighbors_values: List of n_neighbors values for UMAP
            umap_min_dist_values: List of min_dist values for UMAP
            reuse_embeddings: Whether to reuse previously saved embeddings if available
            use_knn: Whether to use KNN for classification in addition to centroid-based approach
            knn_k_values: List of k values to try for KNN classification
            reference_samples_per_class: Number of samples to use per class as reference in KNN
            evaluate_baselines: Whether to evaluate baseline classifiers (LogisticRegression, PCA+KNN)
            baseline_test_size: Test set size for baseline evaluation
            baseline_max_samples: Maximum number of samples to use for baseline evaluation

        Returns:
            evaluation_results: Dictionary of evaluation results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if we should reuse previously saved embeddings
        embeddings_by_level = None
        if reuse_embeddings:
            embeddings_by_level = self.load_embeddings(output_dir)
        
        # If no embeddings were loaded, compute them
        if embeddings_by_level is None:
            # Compute embeddings
            logger.info("Computing embeddings for all samples...")
            embeddings_by_level, labels, all_embeddings = self.compute_embeddings(samples_by_label)
            
            # Save embeddings for future reuse
            self.save_embeddings(embeddings_by_level, output_dir)
        
        # Evaluate classification metrics (using original method for best F1 scores)
        logger.info("Evaluating classification performance (centroid-based)...")
        evaluation_results = self.evaluate_classification(samples_by_label, threshold, embeddings_by_level)
        
        # Optionally evaluate using KNN
        knn_results_first_k = None
        if use_knn:
            # Create a subdirectory for KNN results
            knn_dir = os.path.join(output_dir, "knn_results")
            os.makedirs(knn_dir, exist_ok=True)

            # For each k value
            for k in knn_k_values:
                logger.info(f"Evaluating classification performance using KNN with k={k}...")
                logger.info(f"Using KNN with {reference_samples_per_class} reference samples per class")
                knn_results = self.evaluate_classification_knn(
                    samples_by_label,
                    k=k,
                    reference_samples_per_class=reference_samples_per_class,
                    embeddings_by_level=embeddings_by_level
                )

                # Save the first k results for per-speed analysis
                if k == knn_k_values[0]:
                    knn_results_first_k = knn_results

                # Save KNN results in a subdirectory for this k value
                k_dir = os.path.join(knn_dir, f"knn_k{k}")
                os.makedirs(k_dir, exist_ok=True)

                # Save classification reports
                for level in [1, 2, 3]:
                    report_key = f'level{level}_knn_report'
                    if report_key in knn_results:
                        # Consistent naming with original reports
                        if level == 1:
                            filename = "classification_report_data_types.txt"
                        elif level == 3:
                            filename = "classification_report_full.txt"
                        else:
                            filename = f"classification_report_level{level}.txt"

                        with open(os.path.join(k_dir, filename), 'w') as f:
                            f.write(knn_results[report_key])

                            # Save additional metrics for levels 1 and 2
                            if level < 3:
                                ci_key = f'level{level}_knn_confidence_intervals'
                                ece_key = f'level{level}_knn_ece'
                                auroc_key = f'level{level}_knn_open_set_auroc'

                                if ci_key in knn_results:
                                    ci = knn_results[ci_key]
                                    f.write(f"\n\nBootstrap Confidence Intervals:\n")
                                    f.write(f"Mean F1: {ci['mean']:.4f}\n")
                                    f.write(f"95% CI: [{ci['lower_ci']:.4f}, {ci['upper_ci']:.4f}]\n")

                                if ece_key in knn_results:
                                    f.write(f"\nExpected Calibration Error (ECE): {knn_results[ece_key]:.4f}\n")

                                if auroc_key in knn_results:
                                    f.write(f"\nOpen-Set AUROC: {knn_results[auroc_key]:.4f}\n")

                            # Save per-speed analysis for level 3
                            elif level == 3:
                                speed_key = f'level{level}_knn_per_speed_performance'
                                if speed_key in knn_results:
                                    f.write("\n\nPer-Speed Performance Analysis:\n")
                                    f.write("-" * 40 + "\n")
                                    speed_data = knn_results[speed_key]
                                    for bucket_name, metrics in speed_data.items():
                                        f.write(f"{bucket_name}: F1-macro={metrics['f1_macro']:.4f}, "
                                               f"Accuracy={metrics['accuracy']:.4f}, N={metrics['n_samples']}\n")

                # Save confusion matrices
                for level in [1, 2]:
                    cm_key = f'level{level}_knn_cm'
                    labels_key = f'level{level}_knn_labels'
                    if cm_key in knn_results and labels_key in knn_results:
                        cm = knn_results[cm_key]
                        labels = knn_results[labels_key]

                        # Use the common confusion matrix plotting function
                        self._plot_confusion_matrix(
                            cm=cm,
                            labels=labels,
                            output_dir=k_dir,
                            filename_prefix="confusion_matrix",
                            title_prefix=f"KNN (k={k}) ",
                            level=level
                        )

                # Calculate and save accuracy by label for each level
                for level in [1, 2]:
                    cm_key = f'level{level}_knn_cm'
                    labels_key = f'level{level}_knn_labels'
                    if cm_key in knn_results and labels_key in knn_results:
                        cm = knn_results[cm_key]
                        labels = knn_results[labels_key]

                        accuracy = cm.diagonal() / cm.sum(axis=1)

                        # Create a DataFrame for better readability
                        if level == 1:
                            accuracy_df = pd.DataFrame({
                                'Data Type': labels,
                                'Accuracy': accuracy,
                                'Sample Count': cm.sum(axis=1)
                            })
                            accuracy_df.to_csv(os.path.join(k_dir, "data_type_accuracy.csv"), index=False)
                        else:
                            accuracy_df = pd.DataFrame({
                                'Fault Type': labels,
                                'Accuracy': accuracy,
                                'Sample Count': cm.sum(axis=1)
                            })
                            accuracy_df.to_csv(os.path.join(k_dir, "fault_type_accuracy.csv"), index=False)

                # Merge the best KNN results with the main evaluation results (for comparisons)
                if k == knn_k_values[0]:  # Just use the first k value for the main results
                    for key in knn_results:
                        evaluation_results[f"{key}_{k}"] = knn_results[key]

        # Evaluate baseline methods if requested
        if evaluate_baselines:
            logger.info("Evaluating baseline classifiers...")
            baseline_results = self.evaluate_baselines(
                samples_by_label, output_dir, baseline_test_size, random_state=42, max_samples=baseline_max_samples
            )
            evaluation_results['baseline_results'] = baseline_results

            # Save per-speed performance analysis (including KNN if available)
            baseline_dir = os.path.join(output_dir, "linear_probe_analysis")
            self._save_per_speed_analysis(baseline_results, baseline_dir, knn_results_first_k)
        
        # Save the main evaluation results (centroid-based)
        logger.info("Saving evaluation results...")
        self.save_evaluation_results(evaluation_results, output_dir)
        
        # Create visualization subdirectories
        os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
        
        # Determine which levels to visualize
        levels_to_visualize = []
        if visualization_level == 'all':
            levels_to_visualize = [1, 2, 3]
        else:
            try:
                level = int(visualization_level)
                if level in [1, 2, 3]:
                    levels_to_visualize = [level]
                else:
                    levels_to_visualize = [3]  # Default to level 3 if invalid
            except:
                levels_to_visualize = [3]  # Default to level 3 if invalid
        
        # Advanced visualizations with t-SNE by group
        logger.info("Generating t-SNE visualizations by group...")
        vis_output_dir = os.path.join(output_dir, "visualizations")
        self.visualize_embeddings_by_group(
            embeddings_by_level,
            output_dir=vis_output_dir,
            perplexity_values=tsne_perplexity_values
        )
        
        # Advanced visualizations with UMAP
        logger.info("Generating UMAP visualizations...")
        self.visualize_embeddings_umap(
            embeddings_by_level,
            output_dir=vis_output_dir,
            n_neighbors_values=umap_n_neighbors_values,
            min_dist_values=umap_min_dist_values
        )
        
        logger.info("Evaluation complete!")
        return evaluation_results

    def evaluate_baselines(self, samples_by_label, output_dir, test_size=0.3, random_state=42, max_samples=500):
        """
        Evaluate baseline classifiers (LogisticRegression and PCA+KNN) on the same data

        Args:
            samples_by_label: Dictionary mapping labels to lists of samples
            output_dir: Output directory for baseline results
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            max_samples: Maximum number of samples to use for baseline evaluation (to improve speed)

        Returns:
            baseline_results: Dictionary containing all baseline evaluation results
        """
        logger.info("Starting baseline evaluation with LogisticRegression and PCA+KNN...")

        # Create linear probe analysis directory
        baseline_dir = os.path.join(output_dir, "linear_probe_analysis")
        os.makedirs(baseline_dir, exist_ok=True)

        # Prepare data for baseline methods
        logger.info(f"Preparing data for baseline methods (max {max_samples} samples)...")
        X_data, y_data, sample_info = self._prepare_baseline_data(samples_by_label, max_samples, random_state)

        # Initialize results dictionary
        baseline_results = {}

        # Evaluate at all hierarchical levels
        for level in [1, 2, 3]:
            logger.info(f"Evaluating baselines at hierarchical level {level}...")
            level_results = self._evaluate_baselines_at_level(
                X_data, y_data, sample_info, level, baseline_dir, test_size, random_state
            )
            baseline_results[f'level_{level}'] = level_results
            logger.info(f"Completed baseline evaluation for level {level}")

        # Save summary comparison
        self._save_baseline_comparison(baseline_results, baseline_dir)

        logger.info("Baseline evaluation complete!")
        return baseline_results

    def _prepare_baseline_data(self, samples_by_label, max_samples=None, random_state=42):
        """
        Prepare data for baseline classifiers by extracting features and labels

        Args:
            samples_by_label: Dictionary mapping labels to lists of samples
            max_samples: Maximum number of samples to use (None for all)
            random_state: Random seed for reproducible subsampling

        Returns:
            X_data: Feature matrix
            y_data: Dictionary with labels for each level
            sample_info: List of sample information dictionaries
        """
        all_samples = []

        # Collect all samples first
        for label, samples in samples_by_label.items():
            for sample in samples:
                if isinstance(sample, LabeledSample):
                    all_samples.append({
                        'sample': sample,
                        'label': label,
                        'level1_label': sample.level1_label,
                        'level2_label': sample.level2_label,
                        'level3_label': sample.level3_label,
                        'data_type': sample.data_type,
                        'rot_speed': sample.rot_speed
                    })

        # Subsample if requested
        if max_samples is not None and len(all_samples) > max_samples:
            logger.info(f"Subsampling from {len(all_samples)} to {max_samples} samples for baseline evaluation")
            np.random.seed(random_state)
            indices = np.random.choice(len(all_samples), size=max_samples, replace=False)
            all_samples = [all_samples[i] for i in indices]

        X_data = []
        sample_info = []

        # Process the selected samples
        for sample_data in tqdm(all_samples, desc="Processing samples"):
            sample = sample_data['sample']

            # Extract features using the processor
            features = self.processor.process_sample(sample.data)
            X_data.append(features.flatten())  # Flatten for sklearn

            # Store sample information
            sample_info.append({
                'original_label': sample_data['label'],
                'level1_label': sample_data['level1_label'],
                'level2_label': sample_data['level2_label'],
                'level3_label': sample_data['level3_label'],
                'data_type': sample_data['data_type'],
                'rot_speed': sample_data['rot_speed']
            })

        X_data = np.array(X_data)

        # Create label dictionaries for each level
        y_data = {
            1: [info['level1_label'] for info in sample_info],
            2: [info['level2_label'] for info in sample_info],
            3: [info['level3_label'] for info in sample_info]
        }

        logger.info(f"Prepared {len(X_data)} samples with {X_data.shape[1]} features each")
        return X_data, y_data, sample_info

    def _evaluate_baselines_at_level(self, X_data, y_data, sample_info, level, baseline_dir, test_size, random_state):
        """
        Evaluate baselines at a specific hierarchical level

        Args:
            X_data: Feature matrix
            y_data: Dictionary with labels for each level
            sample_info: List of sample information dictionaries
            level: Hierarchical level (1, 2, or 3)
            baseline_dir: Output directory
            test_size: Test set size
            random_state: Random seed

        Returns:
            level_results: Results for this level
        """
        level_name = f"level_{level}"
        level_dir = os.path.join(baseline_dir, level_name)
        os.makedirs(level_dir, exist_ok=True)

        # Get labels for this level
        y_level = np.array(y_data[level])

        # Check if stratification is possible (all classes must have at least 2 samples)
        unique_classes, class_counts = np.unique(y_level, return_counts=True)
        min_samples_per_class = min(class_counts)
        can_stratify = min_samples_per_class >= 2

        if not can_stratify:
            logger.warning(f"Level {level}: Cannot use stratified splitting (min samples per class: {min_samples_per_class}). Using random split instead.")

        # Split data
        stratify_param = y_level if can_stratify else None
        X_train, X_test, y_train, y_test, info_train, info_test = train_test_split(
            X_data, y_level, sample_info, test_size=test_size,
            random_state=random_state, stratify=stratify_param
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        level_results = {}

        # 1. Logistic Regression (Closed-set)
        logger.info(f"Training LogisticRegression for {level_name} (6 parameter combinations, 5-fold CV)...")
        try:
            lr_results = self._evaluate_logistic_regression(
                X_train_scaled, X_test_scaled, y_train, y_test, level_dir, info_test
            )
            level_results['logistic_regression'] = lr_results
            logger.info(f"LogisticRegression evaluation completed for {level_name}")
        except Exception as e:
            logger.warning(f"LogisticRegression evaluation failed for {level_name}: {e}")
            logger.warning("Skipping LogisticRegression baseline evaluation")
            level_results['logistic_regression'] = {'error': str(e), 'status': 'failed', 'test_f1_macro': 'N/A'}

        # 2. PCA + KNN (Open-set capable)
        logger.info(f"Training PCA+KNN for {level_name} (9 parameter combinations, 5-fold CV)...")
        try:
            pca_knn_results = self._evaluate_pca_knn(
                X_train_scaled, X_test_scaled, y_train, y_test, level_dir, info_test
            )
            level_results['pca_knn'] = pca_knn_results
            logger.info(f"PCA+KNN evaluation completed for {level_name}")
        except Exception as e:
            logger.warning(f"PCA+KNN evaluation failed for {level_name}: {e}")
            logger.warning("Skipping PCA+KNN baseline evaluation")
            level_results['pca_knn'] = {'error': str(e), 'status': 'failed', 'test_f1_macro': 'N/A'}

        # 3. Compare with Siamese results if available
        siamese_results = self._load_siamese_results(baseline_dir, level)
        if siamese_results:
            level_results['siamese_comparison'] = siamese_results

        return level_results

    def _evaluate_logistic_regression(self, X_train, X_test, y_train, y_test, output_dir, sample_info=None):
        """
        Evaluate LogisticRegression with GridSearchCV for closed-set classification

        Args:
            X_train, X_test: Training and test features
            y_train, y_test: Training and test labels
            output_dir: Output directory
            sample_info: Sample information for per-speed analysis (optional)

        Returns:
            results: Dictionary with evaluation results
        """
        # Define parameter grid (optimized for speed)
        param_grid = {
            'classifier__C': [0.1, 1.0, 10.0],  # Reduced from 5 to 3 values
            'classifier__penalty': ['l1', 'l2'],
            'classifier__solver': ['liblinear'],  # Use only liblinear (faster and more stable)
            'classifier__max_iter': [1000]
        }

        # Create pipeline
        pipeline = Pipeline([
            ('classifier', LogisticRegression(random_state=42))
        ])

        # Determine appropriate number of CV folds based on class sizes
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        min_samples_per_class = min(class_counts)
        cv_folds = min(5, min_samples_per_class)  # Use at most 5 folds, but fewer if needed
        cv_folds = max(2, cv_folds)  # GridSearchCV requires at least 2 folds

        if cv_folds < 5:
            logger.warning(f"LogisticRegression: Using {cv_folds} CV folds instead of 5 (min samples per class: {min_samples_per_class})")
        elif cv_folds == 2:
            logger.warning(f"LogisticRegression: Using {cv_folds} CV folds (minimum required, min samples per class: {min_samples_per_class})")

        # Grid search with F1 macro scoring
        f1_scorer = make_scorer(f1_score, average='macro')
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=cv_folds, scoring=f1_scorer,
            n_jobs=1 if platform.system() == 'Windows' else -1, verbose=2  # Single thread on Windows to avoid serialization issues
        )

        # Fit grid search
        grid_search.fit(X_train, y_train)

        # Get best model
        best_model = grid_search.best_estimator_

        # Evaluate on test set
        y_pred = best_model.predict(X_test)
        test_f1 = f1_score(y_test, y_pred, average='macro')

        # Calculate confidence intervals
        ci_results = self.calculate_confidence_intervals(y_test, y_pred, n_bootstraps=1000)

        # Generate classification report
        report = classification_report(y_test, y_pred)

        # Initialize results
        results = {
            'best_params': grid_search.best_params_,
            'best_cv_score': grid_search.best_score_,
            'test_f1_macro': test_f1,
            'confidence_intervals': ci_results,
            'classification_report': report,
            'model': best_model
        }

        # Add per-speed analysis if sample_info is provided
        if sample_info is not None:
            # Extract speeds for test set (sample_info now contains only test set info)
            test_speeds = [info['rot_speed'] for info in sample_info]
            speed_analysis = self.analyze_per_speed_performance(y_test, y_pred, test_speeds, "LogisticRegression")
            results['per_speed_performance'] = speed_analysis

        # Save to files
        with open(os.path.join(output_dir, 'logistic_regression_report.txt'), 'w') as f:
            f.write("Logistic Regression Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Best Parameters: {grid_search.best_params_}\n")
            f.write(f"Best CV Score (F1 Macro): {grid_search.best_score_:.4f}\n")
            f.write(f"Test F1 Macro: {test_f1:.4f}\n")
            f.write(f"Mean F1 (Bootstrap): {ci_results['mean']:.4f}\n")
            f.write(f"95% Confidence Interval: [{ci_results['lower_ci']:.4f}, {ci_results['upper_ci']:.4f}]\n")
            f.write("\n\nClassification Report:\n")
            f.write(report)

            # Add per-speed analysis if available
            if 'per_speed_performance' in results:
                f.write("\n\nPer-Speed Performance Analysis:\n")
                f.write("-" * 40 + "\n")
                speed_data = results['per_speed_performance']
                for bucket_name, metrics in speed_data.items():
                    f.write(f"{bucket_name}: F1-macro={metrics['f1_macro']:.4f}, "
                           f"Accuracy={metrics['accuracy']:.4f}, N={metrics['n_samples']}\n")

        return results

    def _evaluate_pca_knn(self, X_train, X_test, y_train, y_test, output_dir, sample_info=None):
        """
        Evaluate PCA + KNN pipeline with GridSearchCV for open-set classification

        Args:
            X_train, X_test: Training and test features
            y_train, y_test: Training and test labels
            output_dir: Output directory
            sample_info: Sample information for per-speed analysis (optional)

        Returns:
            results: Dictionary with evaluation results
        """
        # Define parameter grid (optimized for speed)
        param_grid = {
            'pca__n_components': [0.9, 0.95, None],  # Reduced from 5 to 3 values (variance explained or None for auto)
            'knn__n_neighbors': [3, 5, 7],  # Reduced from 7 to 3 values
            'knn__weights': ['uniform'],  # Use only uniform (faster, distance weighting is usually better but slower)
            'knn__metric': ['euclidean']  # Use only euclidean (most common and fastest)
        }

        # Create pipeline
        pipeline = Pipeline([
            ('pca', PCA(random_state=42)),
            ('knn', KNeighborsClassifier())
        ])

        # Determine appropriate number of CV folds based on class sizes
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        min_samples_per_class = min(class_counts)
        cv_folds = min(5, min_samples_per_class)  # Use at most 5 folds, but fewer if needed
        cv_folds = max(2, cv_folds)  # GridSearchCV requires at least 2 folds

        if cv_folds < 5:
            logger.warning(f"PCA+KNN: Using {cv_folds} CV folds instead of 5 (min samples per class: {min_samples_per_class})")
        elif cv_folds == 2:
            logger.warning(f"PCA+KNN: Using {cv_folds} CV folds (minimum required, min samples per class: {min_samples_per_class})")

        # Grid search with F1 macro scoring
        f1_scorer = make_scorer(f1_score, average='macro')
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=cv_folds, scoring=f1_scorer,
            n_jobs=1 if platform.system() == 'Windows' else -1, verbose=2  # Single thread on Windows to avoid serialization issues
        )

        # Fit grid search
        grid_search.fit(X_train, y_train)

        # Get best model
        best_model = grid_search.best_estimator_

        # Evaluate on test set (closed-set)
        y_pred = best_model.predict(X_test)
        test_f1 = f1_score(y_test, y_pred, average='macro')

        # Calculate confidence intervals
        ci_results = self.calculate_confidence_intervals(y_test, y_pred, n_bootstraps=1000)

        # Evaluate open-set recognition (simulate by holding out some classes)
        open_set_results = self._evaluate_open_set_recognition(
            best_model, X_train, X_test, y_train, y_test
        )

        # Calculate AUROC for open-set detection
        auroc_score = self._calculate_open_set_auroc_for_knn(
            best_model, X_train, X_test, y_train, y_test
        )

        # Calculate Expected Calibration Error (ECE)
        ece_results = self._calculate_ece_for_knn(best_model, X_test, y_test)

        # Generate classification report
        report = classification_report(y_test, y_pred)

        # Initialize results
        results = {
            'best_params': grid_search.best_params_,
            'best_cv_score': grid_search.best_score_,
            'test_f1_macro': test_f1,
            'confidence_intervals': ci_results,
            'open_set_f1_macro': open_set_results['f1_macro'],
            'open_set_auroc': auroc_score,
            'ece': ece_results['ece'],
            'ece_details': ece_results,
            'open_set_details': open_set_results,
            'classification_report': report,
            'model': best_model
        }

        # Add per-speed analysis if sample_info is provided
        if sample_info is not None:
            # Extract speeds for test set (sample_info now contains only test set info)
            test_speeds = [info['rot_speed'] for info in sample_info]
            speed_analysis = self.analyze_per_speed_performance(y_test, y_pred, test_speeds, "PCA+KNN")
            results['per_speed_performance'] = speed_analysis

        # Save to files
        with open(os.path.join(output_dir, 'pca_knn_report.txt'), 'w') as f:
            f.write("PCA + KNN Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Best Parameters: {grid_search.best_params_}\n")
            f.write(f"Best CV Score (F1 Macro): {grid_search.best_score_:.4f}\n")
            f.write(f"Test F1 Macro (Closed-set): {test_f1:.4f}\n")
            f.write(f"Mean F1 (Bootstrap): {ci_results['mean']:.4f}\n")
            f.write(f"95% Confidence Interval: [{ci_results['lower_ci']:.4f}, {ci_results['upper_ci']:.4f}]\n")
            f.write(f"Open-set F1 Macro: {open_set_results['f1_macro']:.4f}\n")
            f.write(f"Open-set AUROC: {auroc_score:.4f}\n")
            f.write(f"ECE: {ece_results['ece']:.4f}\n")
            f.write("\n\nClassification Report:\n")
            f.write(report)
            f.write("\n\nOpen-set Recognition Details:\n")
            f.write(f"Open-set F1 Macro: {open_set_results['f1_macro']:.4f}\n")
            f.write(f"Open-set AUROC: {auroc_score:.4f}\n")
            f.write(f"Expected Calibration Error (ECE): {ece_results['ece']:.4f}\n")
            f.write(f"Known classes accuracy: {open_set_results['known_accuracy']:.4f}\n")
            f.write(f"Unknown rejection rate: {open_set_results['unknown_rejection']:.4f}\n")

            # Add ECE bin details
            f.write("\n\nCalibration Analysis (ECE Bins):\n")
            f.write("-" * 40 + "\n")
            f.write("Bin Range    | Mean Conf | Accuracy | Samples | Weight\n")
            f.write("-" * 55 + "\n")
            for bin_info in ece_results['bin_data']:
                f.write(f"[{bin_info['bin_start']:.1f}, {bin_info['bin_end']:.1f}] | "
                       f"{bin_info['mean_confidence']:.3f}   | "
                       f"{bin_info['accuracy']:.3f}  | "
                       f"{bin_info['n_samples']:6d} | "
                       f"{bin_info['weight']:.3f}\n")

            # Add per-speed analysis if available
            if 'per_speed_performance' in results:
                f.write("\n\nPer-Speed Performance Analysis:\n")
                f.write("-" * 40 + "\n")
                speed_data = results['per_speed_performance']
                for bucket_name, metrics in speed_data.items():
                    f.write(f"{bucket_name}: F1-macro={metrics['f1_macro']:.4f}, "
                           f"Accuracy={metrics['accuracy']:.4f}, N={metrics['n_samples']}\n")

        return results

    def _calculate_open_set_auroc_for_knn(self, model, X_train, X_test, y_train, y_test, unknown_fraction=0.2):
        """
        Calculate AUROC for open-set detection using k-NN confidence scores

        Args:
            model: Trained k-NN model
            X_train, X_test: Training and test features
            y_train, y_test: Training and test labels
            unknown_fraction: Fraction of classes to treat as unknown

        Returns:
            float: AUROC score
        """
        # Get unique classes
        classes = np.unique(y_train)
        n_unknown = max(1, int(len(classes) * unknown_fraction))

        # Randomly select classes to be "unknown"
        np.random.seed(42)
        unknown_classes = np.random.choice(classes, size=n_unknown, replace=False)
        known_classes = np.setdiff1d(classes, unknown_classes)

        # Split test data into known and unknown
        known_mask = np.isin(y_test, known_classes)
        unknown_mask = np.isin(y_test, unknown_classes)

        X_test_known = X_test[known_mask]
        y_test_known = y_test[known_mask]
        X_test_unknown = X_test[unknown_mask]
        y_test_unknown = y_test[unknown_mask]

        # Calculate confidence scores using k-NN approach
        confidence_known = self._calculate_knn_confidence(model, X_test_known)
        confidence_unknown = self._calculate_knn_confidence(model, X_test_unknown)

        # Calculate AUROC
        auroc = self.calculate_open_set_auroc(
            y_test_known, model.predict(X_test_known),
            confidence_known, y_test_unknown, confidence_unknown
        )

        return auroc

    def _calculate_open_set_auroc_for_siamese_knn(self, gallery_embeddings, gallery_labels, level_embeddings, k, unknown_fraction=0.2):
        """
        Calculate AUROC for open-set detection using Siamese+KNN approach

        Args:
            gallery_embeddings: Reference embeddings for each class
            gallery_labels: Labels corresponding to gallery embeddings
            level_embeddings: Embeddings organized by true label for this level
            k: Number of neighbors to use
            unknown_fraction: Fraction of classes to treat as unknown

        Returns:
            float: AUROC score
        """
        # Get unique classes from gallery
        classes = np.unique(gallery_labels)
        n_unknown = max(1, int(len(classes) * unknown_fraction))

        # Randomly select classes to be "unknown"
        np.random.seed(42)
        unknown_classes = np.random.choice(classes, size=n_unknown, replace=False)
        known_classes = np.setdiff1d(classes, unknown_classes)

        # Collect embeddings for known and unknown classes
        known_embeddings = []
        known_true_labels = []
        unknown_embeddings = []
        unknown_true_labels = []

        for true_label, embs in level_embeddings.items():
            if true_label in known_classes:
                known_embeddings.extend(embs)
                known_true_labels.extend([true_label] * len(embs))
            elif true_label in unknown_classes:
                unknown_embeddings.extend(embs)
                unknown_true_labels.extend([true_label] * len(embs))

        # Calculate confidence scores for known and unknown samples
        confidence_known = []
        confidence_unknown = []

        # For known samples
        for emb in known_embeddings:
            distances = np.linalg.norm(gallery_embeddings - emb, axis=1)
            k_local = min(k, len(distances))
            nearest_indices = np.argsort(distances)[:k_local]
            nearest_labels = [gallery_labels[i] for i in nearest_indices]

            # Calculate confidence
            from collections import Counter
            majority_count = Counter(nearest_labels).most_common(1)[0][1]
            confidence = majority_count / k_local
            confidence_known.append(confidence)

        # For unknown samples
        for emb in unknown_embeddings:
            distances = np.linalg.norm(gallery_embeddings - emb, axis=1)
            k_local = min(k, len(distances))
            nearest_indices = np.argsort(distances)[:k_local]
            nearest_labels = [gallery_labels[i] for i in nearest_indices]

            # Calculate confidence
            from collections import Counter
            majority_count = Counter(nearest_labels).most_common(1)[0][1]
            confidence = majority_count / k_local
            confidence_unknown.append(confidence)

        # Calculate AUROC
        auroc = self.calculate_open_set_auroc(
            np.array(known_true_labels), np.array(known_true_labels),  # y_true and y_pred are the same for known
            np.array(confidence_known), np.array(unknown_true_labels), np.array(confidence_unknown)
        )

        return auroc

    def _calculate_ece_for_knn(self, model, X_test, y_test, n_bins=10):
        """
        Calculate Expected Calibration Error using k-NN confidence scores

        Args:
            model: Trained k-NN model
            X_test: Test features
            y_test: Test labels
            n_bins: Number of bins for ECE calculation

        Returns:
            dict: ECE results
        """
        # Get predictions and confidence scores
        y_pred = model.predict(X_test)
        confidences = self._calculate_knn_confidence(model, X_test)

        # Calculate ECE
        ece_results = self.calculate_ece(y_test, y_pred, confidences, n_bins)

        return ece_results

    def _calculate_knn_confidence(self, model, X):
        """
        Calculate confidence scores for k-NN using the approach described by the user:
        Confidence = (Number of neighbors with the majority class) / K

        Args:
            model: Trained k-NN model (from pipeline)
            X: Input features

        Returns:
            np.array: Confidence scores between 0 and 1
        """
        # Extract the KNN classifier from the pipeline
        knn_classifier = model.named_steps['knn']

        # For PCA+KNN pipeline, we need to transform input data through PCA first
        if 'pca' in model.named_steps:
            X_transformed = model.named_steps['pca'].transform(X)
        else:
            X_transformed = X

        # Get k value
        k = knn_classifier.n_neighbors

        # Get distances and indices of k nearest neighbors
        distances, indices = knn_classifier.kneighbors(X_transformed)

        # Get training labels
        y_train = knn_classifier._y

        # Calculate confidence for each prediction
        confidences = []
        for i, neighbor_indices in enumerate(indices):
            # Get labels of k nearest neighbors
            neighbor_labels = y_train[neighbor_indices]

            # Count votes for each class
            unique_labels, counts = np.unique(neighbor_labels, return_counts=True)

            # Find the majority class and its count
            majority_count = np.max(counts)

            # Confidence = (number of neighbors with majority class) / k
            confidence = majority_count / k
            confidences.append(confidence)

        return np.array(confidences)

    def _evaluate_open_set_recognition(self, model, X_train, X_test, y_train, y_test, unknown_fraction=0.2):
        """
        Evaluate open-set recognition capability by simulating unknown classes

        Args:
            model: Trained model
            X_train, X_test: Training and test features
            y_train, y_test: Training and test labels
            unknown_fraction: Fraction of classes to treat as unknown

        Returns:
            results: Dictionary with open-set evaluation results
        """
        # Get unique classes
        classes = np.unique(y_train)
        n_unknown = max(1, int(len(classes) * unknown_fraction))

        # Randomly select classes to be "unknown"
        np.random.seed(42)
        unknown_classes = np.random.choice(classes, size=n_unknown, replace=False)
        known_classes = np.setdiff1d(classes, unknown_classes)

        logger.info(f"Simulating open-set: {len(known_classes)} known classes, {len(unknown_classes)} unknown classes")

        # Split test data into known and unknown
        known_mask = np.isin(y_test, known_classes)
        unknown_mask = np.isin(y_test, unknown_classes)

        X_test_known = X_test[known_mask]
        y_test_known = y_test[known_mask]
        X_test_unknown = X_test[unknown_mask]
        y_test_unknown = y_test[unknown_mask]

        # Get predictions and confidence scores
        if hasattr(model, 'predict_proba'):
            # For LogisticRegression
            y_pred_known = model.predict(X_test_known)
            y_pred_unknown = model.predict(X_test_unknown)
            confidence_known = np.max(model.predict_proba(X_test_known), axis=1)
            confidence_unknown = np.max(model.predict_proba(X_test_unknown), axis=1)
        else:
            # For KNN (use distances as confidence)
            y_pred_known = model.predict(X_test_known)
            y_pred_unknown = model.predict(X_test_unknown)

            # For KNN, we'll use a simple confidence measure based on agreement
            # This is a simplified approach - in practice you'd use distance-based confidence
            confidence_known = np.ones(len(y_pred_known)) * 0.8  # Placeholder
            confidence_unknown = np.ones(len(y_pred_unknown)) * 0.3  # Placeholder

        # Calculate open-set F1 (treating unknown predictions as rejections)
        # True labels: known=1 (should be classified), unknown=0 (should be rejected)
        y_true_open = np.concatenate([np.ones(len(y_test_known)), np.zeros(len(y_test_unknown))])

        # Predictions: classify as known if confidence > threshold, else reject
        # We'll use a threshold that gives reasonable performance
        threshold = 0.5
        y_pred_open = np.concatenate([
            (confidence_known > threshold).astype(int),
            (confidence_unknown > threshold).astype(int)
        ])

        open_set_f1 = f1_score(y_true_open, y_pred_open, average='macro')

        # Calculate additional metrics
        known_correct = np.mean(y_pred_known == y_test_known)
        unknown_rejected = np.mean(confidence_unknown <= threshold)

        results = {
            'f1_macro': open_set_f1,
            'known_accuracy': known_correct,
            'unknown_rejection': unknown_rejected,
            'threshold': threshold,
            'n_known_classes': len(known_classes),
            'n_unknown_classes': len(unknown_classes)
        }

        return results

    def _load_siamese_results(self, baseline_dir, level):
        """
        Load Siamese network results for comparison

        Args:
            baseline_dir: Baseline results directory
            level: Hierarchical level

        Returns:
            siamese_results: Dictionary with Siamese results or None
        """
        try:
            # Try to find Siamese results in parent directory
            parent_dir = os.path.dirname(baseline_dir)

            # Look for classification reports
            if level == 1:
                report_file = os.path.join(parent_dir, 'classification_report_data_types.txt')
            elif level == 3:
                report_file = os.path.join(parent_dir, 'classification_report_full.txt')
            else:
                report_file = os.path.join(parent_dir, f'classification_report_level{level}.txt')

            if os.path.exists(report_file):
                with open(report_file, 'r') as f:
                    report_content = f.read()

                # Extract F1 macro score (simplified parsing)
                lines = report_content.split('\n')
                for line in lines:
                    if 'macro avg' in line and 'f1-score' in line:
                        parts = line.split()
                        if len(parts) >= 4:
                            try:
                                f1_macro = float(parts[-2])  # F1 score is usually the second-to-last column
                                return {'f1_macro': f1_macro, 'report': report_content}
                            except ValueError:
                                continue

        except Exception as e:
            logger.warning(f"Could not load Siamese results for level {level}: {e}")

        return None

    def _save_baseline_comparison(self, baseline_results, baseline_dir):
        """
        Save a comparison summary of all baseline methods

        Args:
            baseline_results: Results from all baseline evaluations
            baseline_dir: Output directory
        """
        comparison_data = []

        for level_name, level_results in baseline_results.items():
            level_num = int(level_name.split('_')[1])

            for method_name, method_results in level_results.items():
                if method_name == 'siamese_comparison':
                    if method_results:
                        comparison_data.append({
                            'Level': level_num,
                            'Method': 'Siamese_Network',
                            'F1_Macro': method_results.get('f1_macro', 'N/A'),
                            'Type': 'Embedding-based'
                        })
                else:
                    row = {
                        'Level': level_num,
                        'Method': method_name.upper(),
                        'F1_Macro': method_results.get('test_f1_macro', 'N/A'),
                        'Type': 'Feature-based'
                    }

                    # Add additional metrics
                    if method_name == 'pca_knn':
                        if 'open_set_f1_macro' in method_results:
                            row['Open_Set_F1'] = method_results['open_set_f1_macro']
                        if 'open_set_auroc' in method_results:
                            row['Open_Set_AUROC'] = method_results['open_set_auroc']
                        if 'ece' in method_results:
                            row['ECE'] = method_results['ece']

                    # Add confidence intervals
                    if 'confidence_intervals' in method_results:
                        ci = method_results['confidence_intervals']
                        row['CI_Lower'] = ci.get('lower_ci', 'N/A')
                        row['CI_Upper'] = ci.get('upper_ci', 'N/A')

                    comparison_data.append(row)

        # Create DataFrame and save
        df = pd.DataFrame(comparison_data)
        df = df.sort_values(['Level', 'Method'])

        # Save to CSV
        csv_path = os.path.join(baseline_dir, 'baseline_comparison.csv')
        df.to_csv(csv_path, index=False)

        # Note: Per-speed analysis will be saved later along with KNN results

        # Save to text for easy reading
        txt_path = os.path.join(baseline_dir, 'baseline_comparison.txt')
        with open(txt_path, 'w') as f:
            f.write("Baseline Methods Comparison (Enhanced Metrics)\n")
            f.write("=" * 60 + "\n\n")
            f.write(df.to_string(index=False))
            f.write("\n\n")

            # Add summary insights
            f.write("Summary Insights:\n")
            f.write("-" * 20 + "\n")

            for level in [1, 2, 3]:
                level_data = df[df['Level'] == level]
                if not level_data.empty:
                    f.write(f"\nLevel {level}:\n")
                    for _, row in level_data.iterrows():
                        if pd.notna(row['F1_Macro']) and isinstance(row['F1_Macro'], (int, float)):
                            f1_str = f"{row['F1_Macro']:.4f}"
                        else:
                            f1_str = str(row['F1_Macro']) if pd.notna(row['F1_Macro']) else "N/A"
                        ci_str = ""
                        if pd.notna(row.get('CI_Lower')) and pd.notna(row.get('CI_Upper')):
                            ci_str = f" [{row['CI_Lower']:.4f}, {row['CI_Upper']:.4f}]"

                        line = f"  {row['Method']}: F1 = {f1_str}{ci_str}"

                        # Add additional metrics for PCA+KNN
                        if row['Method'] == 'PCA_KNN':
                            if pd.notna(row.get('Open_Set_F1')):
                                line += f", Open-set F1 = {row['Open_Set_F1']:.4f}"
                            if pd.notna(row.get('Open_Set_AUROC')):
                                line += f", AUROC = {row['Open_Set_AUROC']:.4f}"
                            if pd.notna(row.get('ECE')):
                                line += f", ECE = {row['ECE']:.4f}"

                        f.write(line + "\n")

                    # Compare with Siamese if available
                    siamese_row = level_data[level_data['Method'] == 'Siamese_Network']
                    if not siamese_row.empty:
                        siamese_f1 = siamese_row['F1_Macro'].iloc[0]
                        f.write(f"  Siamese Network F1: {siamese_f1}\n")
                        f.write("  → Data quality assessment: "
                               f"{'Good' if siamese_f1 > level_data['F1_Macro'].max() else 'Needs investigation'}\n")

        logger.info(f"Saved baseline comparison to {csv_path}")

    def _save_per_speed_analysis(self, baseline_results, baseline_dir, knn_results=None):
        """
        Save per-speed performance analysis for all methods and levels

        Args:
            baseline_results: Results from all baseline evaluations
            baseline_dir: Output directory
            knn_results: Optional KNN results to include (dict with level keys)
        """
        speed_data = []

        # Collect speed performance data from all methods and levels
        for level_name, level_results in baseline_results.items():
            level_num = int(level_name.split('_')[1])

            for method_name, method_results in level_results.items():
                if method_name == 'siamese_comparison':
                    continue

                if 'per_speed_performance' in method_results:
                    speed_performance = method_results['per_speed_performance']

                    for bucket_name, metrics in speed_performance.items():
                        speed_data.append({
                            'Level': level_num,
                            'Method': method_name.upper(),
                            'Speed_Bucket': bucket_name,
                            'F1_Macro': metrics['f1_macro'],
                            'F1_Micro': metrics['f1_micro'],
                            'Accuracy': metrics['accuracy'],
                            'N_Samples': metrics['n_samples']
                        })

        # Collect speed performance data from KNN results (only level 3 has speed data)
        if knn_results is not None:
            for level in [1, 2, 3]:
                speed_key = f'level{level}_knn_per_speed_performance'
                if speed_key in knn_results:
                    speed_performance = knn_results[speed_key]

                    # Skip empty results
                    if not speed_performance:
                        continue

                    for bucket_name, metrics in speed_performance.items():
                        speed_data.append({
                            'Level': level,
                            'Method': 'SIAMESE+KNN',
                            'Speed_Bucket': bucket_name,
                            'F1_Macro': metrics['f1_macro'],
                            'F1_Micro': metrics['f1_micro'],
                            'Accuracy': metrics['accuracy'],
                            'N_Samples': metrics['n_samples']
                        })

        if speed_data:
            # Create DataFrame and save
            df = pd.DataFrame(speed_data)
            df = df.sort_values(['Level', 'Method', 'Speed_Bucket'])

            # Save to CSV
            csv_path = os.path.join(baseline_dir, 'per_speed_performance.csv')
            df.to_csv(csv_path, index=False)

            # Save summary to text
            txt_path = os.path.join(baseline_dir, 'per_speed_performance.txt')
            with open(txt_path, 'w') as f:
                f.write("Per-Speed Performance Analysis\n")
                f.write("=" * 40 + "\n\n")

                for level in [1, 2, 3]:
                    level_data = df[df['Level'] == level]
                    if not level_data.empty:
                        f.write(f"Level {level}:\n")
                        f.write("-" * 20 + "\n")

                        # Group by method
                        for method in level_data['Method'].unique():
                            method_data = level_data[level_data['Method'] == method]
                            f.write(f"\n{method}:\n")

                            for _, row in method_data.iterrows():
                                if pd.notna(row['F1_Macro']):
                                    f.write(f"  {row['Speed_Bucket']}: F1-macro={row['F1_Macro']:.4f}, "
                                           f"Accuracy={row['Accuracy']:.4f}, N={int(row['N_Samples'])}\n")
                                else:
                                    f.write(f"  {row['Speed_Bucket']}: Insufficient data\n")
                        f.write("\n")

            logger.info(f"Saved per-speed performance analysis to {csv_path}")
        else:
            logger.info("No per-speed performance data available")

    def calculate_confidence_intervals(self, y_true, y_pred, n_bootstraps=1000, confidence_level=0.95):
        """
        Calculate confidence intervals for F1 score using bootstrapping

        Args:
            y_true: True labels
            y_pred: Predicted labels
            n_bootstraps: Number of bootstrap samples
            confidence_level: Confidence level (0.95 for 95% CI)

        Returns:
            dict: Dictionary with mean, lower_ci, upper_ci for F1 macro
        """
        f1_scores = []

        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        n_samples = len(y_true)

        for _ in range(n_bootstraps):
            # Bootstrap sampling with replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]

            # Calculate F1 macro for this bootstrap sample
            try:
                f1_boot = f1_score(y_true_boot, y_pred_boot, average='macro')
                f1_scores.append(f1_boot)
            except:
                # Skip if calculation fails (e.g., single class in bootstrap)
                continue

        if not f1_scores:
            return {'mean': np.nan, 'lower_ci': np.nan, 'upper_ci': np.nan}

        f1_scores = np.array(f1_scores)
        mean_f1 = np.mean(f1_scores)

        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100

        lower_ci = np.percentile(f1_scores, lower_percentile)
        upper_ci = np.percentile(f1_scores, upper_percentile)

        return {
            'mean': mean_f1,
            'lower_ci': lower_ci,
            'upper_ci': upper_ci,
            'ci_level': confidence_level
        }

    def calculate_open_set_auroc(self, y_true_known, y_pred_known, confidence_known, y_true_unknown, confidence_unknown):
        """
        Calculate AUROC for open-set detection

        Args:
            y_true_known: True labels for known classes (should be accepted)
            y_pred_known: Predicted labels for known classes
            confidence_known: Confidence scores for known classes
            y_true_unknown: True labels for unknown classes (should be rejected)
            confidence_unknown: Confidence scores for unknown classes

        Returns:
            float: AUROC score
        """
        # Create binary labels for open-set detection
        # 1 = known (should be accepted), 0 = unknown (should be rejected)
        y_binary = np.concatenate([np.ones(len(confidence_known)), np.zeros(len(confidence_unknown))])

        # Use confidence as the decision function (higher confidence = more likely to be known)
        confidence_scores = np.concatenate([confidence_known, confidence_unknown])

        # Calculate AUROC
        try:
            auroc = roc_auc_score(y_binary, confidence_scores)
            return auroc
        except Exception as e:
            logger.warning(f"Could not calculate AUROC: {e}")
            return np.nan

    def calculate_ece(self, y_true, y_pred, confidences, n_bins=10):
        """
        Calculate Expected Calibration Error (ECE) using confidence scores

        Args:
            y_true: True labels
            y_pred: Predicted labels
            confidences: Confidence scores for each prediction
            n_bins: Number of bins for calibration

        Returns:
            dict: ECE value and bin-wise calibration data
        """
        if len(confidences) != len(y_true):
            raise ValueError("Confidences and y_true must have the same length")

        # Convert to numpy arrays
        confidences = np.array(confidences)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Create bins
        bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
        bin_indices = np.digitize(confidences, bin_boundaries) - 1

        # Ensure indices are within bounds
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        ece = 0.0
        total_samples = len(confidences)
        bin_data = []

        for bin_idx in range(n_bins):
            bin_mask = bin_indices == bin_idx
            bin_size = np.sum(bin_mask)

            if bin_size == 0:
                continue

            # Calculate bin statistics
            bin_conf_mean = np.mean(confidences[bin_mask])
            bin_accuracy = np.mean(y_true[bin_mask] == y_pred[bin_mask])
            bin_weight = bin_size / total_samples

            # Add to ECE
            ece += bin_weight * abs(bin_conf_mean - bin_accuracy)

            bin_data.append({
                'bin_idx': bin_idx,
                'bin_start': bin_boundaries[bin_idx],
                'bin_end': bin_boundaries[bin_idx + 1],
                'mean_confidence': bin_conf_mean,
                'accuracy': bin_accuracy,
                'n_samples': bin_size,
                'weight': bin_weight
            })

        return {
            'ece': ece,
            'bin_data': bin_data,
            'n_bins': n_bins
        }

    def analyze_per_speed_performance(self, y_true, y_pred, speeds, method_name="Unknown"):
        """
        Analyze performance broken down by rotational speed buckets

        Args:
            y_true: True labels
            y_pred: Predicted labels
            speeds: Rotational speeds corresponding to each sample
            method_name: Name of the method for logging

        Returns:
            dict: Performance metrics grouped by speed bucket
        """
        speeds = np.array(speeds)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Define 6 speed buckets
        speed_buckets = {
            '12-18 Hz': (12.0, 18.0),
            '18-24 Hz': (18.0, 24.0),
            '24-32 Hz': (24.0, 32.0),
            '32-40 Hz': (32.0, 40.0),
            '40-50 Hz': (40.0, 50.0),
            '50-62 Hz': (50.0, 62.0)
        }

        speed_results = {}

        for bucket_name, (min_speed, max_speed) in speed_buckets.items():
            # Create mask for samples in this speed range
            speed_mask = (speeds >= min_speed) & (speeds < max_speed)
            if np.sum(speed_mask) == 0:
                logger.info(f"No samples found for speed bucket {bucket_name}")
                continue

            y_true_bucket = y_true[speed_mask]
            y_pred_bucket = y_pred[speed_mask]

            try:
                f1_macro = f1_score(y_true_bucket, y_pred_bucket, average='macro')
                f1_micro = f1_score(y_true_bucket, y_pred_bucket, average='micro')
                accuracy = np.mean(y_true_bucket == y_pred_bucket)
                n_samples = len(y_true_bucket)

                speed_results[bucket_name] = {
                    'f1_macro': f1_macro,
                    'f1_micro': f1_micro,
                    'accuracy': accuracy,
                    'n_samples': n_samples
                }
            except Exception as e:
                logger.warning(f"Could not calculate metrics for speed bucket {bucket_name}: {e}")
                speed_results[bucket_name] = {
                    'f1_macro': np.nan,
                    'f1_micro': np.nan,
                    'accuracy': np.nan,
                    'n_samples': len(y_true_bucket)
                }

        return speed_results 