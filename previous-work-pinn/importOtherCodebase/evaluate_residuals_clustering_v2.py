import numpy as np
from itertools import combinations, product
import multiprocessing as mp
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import LabelEncoder
from fastdtw import fastdtw
import torch
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
import os
import pickle
import concurrent.futures

# File to store the checkpoint for DTW distance matrix computation.
CHECKPOINT_FILE = "dtw_checkpoint.pkl"

def _dtw_pair_worker(args):
    """Worker for DTW distance computation between a pair of sequences."""
    i, j, seq1, seq2, radius = args
    distance, _ = fastdtw(seq1, seq2, radius=radius)
    return (i, j, distance)

def compute_full_distance_matrix(sequences, radius=3, chunksize=10):
    """
    Compute pairwise DTW distances for all sequences (normal + faults) with checkpointing.
    """
    n = len(sequences)
    
    # Load checkpoint if available.
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "rb") as f:
            checkpoint = pickle.load(f)
        distance_matrix = checkpoint["distance_matrix"]
        computed_pairs = checkpoint["computed_pairs"]
        print(f"Resuming from checkpoint: {len(computed_pairs)} pairs already computed.")
    else:
        distance_matrix = np.zeros((n, n))
        computed_pairs = set()
    
    # Create list of unique index pairs.
    pair_indices = list(combinations(range(n), 2))
    pairs_to_compute = [
        (i, j, sequences[i], sequences[j], radius)
        for i, j in pair_indices if (i, j) not in computed_pairs
    ]
    
    total_to_compute = len(pairs_to_compute)
    print(f"Computing {total_to_compute} remaining pairs out of {len(pair_indices)} total pairs.")

    try:
        with mp.Pool() as pool:
            for result in tqdm(pool.imap_unordered(_dtw_pair_worker, pairs_to_compute, chunksize=chunksize),
                               total=total_to_compute, desc="Computing DTW Matrix"):
                i, j, dist = result
                distance_matrix[i, j] = dist
                computed_pairs.add((i, j))
                
                # Save a checkpoint every 1000 pairs.
                if len(computed_pairs) % 1000 == 0:
                    checkpoint_data = {
                        "distance_matrix": distance_matrix,
                        "computed_pairs": computed_pairs,
                    }
                    with open(CHECKPOINT_FILE, "wb") as f:
                        pickle.dump(checkpoint_data, f)
    except KeyboardInterrupt:
        print("Interrupted! Saving current checkpoint before exiting.")
        checkpoint_data = {
            "distance_matrix": distance_matrix,
            "computed_pairs": computed_pairs,
        }
        with open(CHECKPOINT_FILE, "wb") as f:
            pickle.dump(checkpoint_data, f)
        raise

    # Symmetrize the matrix.
    distance_matrix += distance_matrix.T

    # Remove the checkpoint file when complete.
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
    return distance_matrix

def extract_features(sequence):
    """Extract statistical features from a time series."""

    # Transform to numpy array
    sequence = np.array(sequence)


    return [
        np.mean(sequence, axis=0),                         # Mean
        np.std(sequence, axis=0),                          # Standard Deviation
        skew(sequence),                                    # Skewness
        kurtosis(sequence),                                # Kurtosis
        np.sqrt(np.mean(np.square(sequence), axis=0)),     # RMS (signal energy)
        np.max(np.abs(sequence), axis=0)                   # Peak Amplitude
    ]

def feature_based_clustering(residuals_dict, n_clusters=None):
    """
    Cluster residuals using extracted features + PCA for dimensionality reduction, and KMeans.
    Returns cluster labels and true labels.
    """
    all_sequences = []
    all_labels = []
    for label, sequences in residuals_dict.items():
        all_sequences.extend(sequences)
        all_labels.extend([label] * len(sequences))

    # Extract features for each sequence.
    features = [extract_features(seq) for seq in all_sequences]
    X = np.stack(features)
    X = X.reshape(X.shape[0], -1)
    X_scaled = StandardScaler().fit_transform(X)

    # Reduce dimensionality with PCA while retaining 95% variance.
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)

    if n_clusters is None:
        n_clusters = len(residuals_dict)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(X_pca)

    return cluster_labels, all_labels

def dtw_kmeans_clustering(residuals_dict, n_clusters=None):
    """
    Cluster residuals using TimeSeriesKMeans with DTW metric.
    tslearn can handle variable-length time series, so no padding is needed.
    Returns cluster labels and true labels.
    """
    all_sequences = []
    all_labels = []
    for label, sequences in residuals_dict.items():
        all_sequences.extend(sequences)
        all_labels.extend([label] * len(sequences))
    
    # Convert the list of sequences into a tslearn-friendly 3D array.
    X = to_time_series_dataset(all_sequences)

    if n_clusters is None:
        n_clusters = len(residuals_dict)

    km = TimeSeriesKMeans(
        n_clusters=n_clusters,
        metric="dtw",
        n_init=3,
        random_state=42,
        verbose=1,
        n_jobs=mp.cpu_count()
    ).fit(X)
    cluster_labels = km.labels_

    return cluster_labels, all_labels

def validate_clustering_separability(residuals_dict, eps_quantile=0.95, min_samples=5, radius=3):
    """
    Validate if DBSCAN + DTW clusters residuals into meaningful groups (normal + faults).
    Returns silhouette score and ARI.
    """
    all_sequences = []
    all_labels = []
    for label, sequences in residuals_dict.items():
        all_sequences.extend(sequences)
        all_labels.extend([label] * len(sequences))

    # Compute DTW distance matrix.
    distance_matrix = compute_full_distance_matrix(all_sequences, radius)
    positive_dists = distance_matrix[distance_matrix > 0]
    eps = np.quantile(positive_dists, eps_quantile) if positive_dists.size > 0 else 0

    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed").fit(distance_matrix)
    cluster_labels = clustering.labels_

    unique_clusters = set(cluster_labels) - {-1}
    if len(unique_clusters) < 2:
        print("Warning: Only one or zero clusters found by DBSCAN. Silhouette score undefined.")
        silhouette = -1
    else:
        silhouette = silhouette_score(distance_matrix, cluster_labels, metric="precomputed")

    le = LabelEncoder()
    true_labels = le.fit_transform(all_labels)
    ari = adjusted_rand_score(true_labels, cluster_labels)

    return silhouette, ari

def validate_all_clustering_methods(residuals_dict):
    """
    Compare three clustering approaches:
      1. DBSCAN with DTW distance (optional)
      2. Feature-based + PCA + KMeans
      3. TimeSeriesKMeans with DTW
    Evaluates each method using Silhouette Score and ARI.
    """
    all_sequences = []
    all_labels = []
    for label, sequences in residuals_dict.items():
        all_sequences.extend(sequences)
        all_labels.extend([label] * len(sequences))
    le = LabelEncoder()
    true_labels = le.fit_transform(all_labels)

    results = {}

    # Method 2: Feature-based + PCA + KMeans.
    print("Validating Feature-based clustering...")
    feature_labels, _ = feature_based_clustering(residuals_dict)
    X_features = np.stack([np.array(extract_features(seq)) for seq in all_sequences])
    X_features = X_features.reshape(X_features.shape[0], -1)
    X_scaled_features = StandardScaler().fit_transform(X_features)
    pca_features = PCA(n_components=0.95).fit_transform(X_scaled_features)
    silhouette_feature = silhouette_score(pca_features, feature_labels) if len(set(feature_labels)) >= 2 else -1
    ari_feature = adjusted_rand_score(true_labels, feature_labels)
    results['feature_pca'] = {'silhouette': silhouette_feature, 'ari': ari_feature}
    print(results['feature_pca'])

    # Method 3: TimeSeriesKMeans with DTW.
    print("Validating TimeSeriesKMeans clustering...")
    ts_kmeans_labels, _ = dtw_kmeans_clustering(residuals_dict)
    X_ts_kmeans = to_time_series_dataset(all_sequences)
    silhouette_ts_kmeans = silhouette_score(X_ts_kmeans.reshape(len(X_ts_kmeans), -1), ts_kmeans_labels) if len(set(ts_kmeans_labels)) >= 2 else -1
    ari_ts_kmeans = adjusted_rand_score(true_labels, ts_kmeans_labels)
    results['ts_kmeans'] = {'silhouette': silhouette_ts_kmeans, 'ari': ari_ts_kmeans}
    print(results['ts_kmeans'])

    # (Optionally, DBSCAN+DTW can be added here using similar logic.)
    # Save the ts_kmeans_labels
    with open('ts_kmeans_labels.pkl', 'wb') as f:
        pickle.dump(ts_kmeans_labels, f)

    return results

def process_residuals(residuals_dict, seq_length=100):
    """
    Convert residual tensors (or already segmented lists) into sequences.
    If a given key already holds a list (of variable-length segments), pass it through.
    Otherwise, split the tensor into fixed-length chunks.
    """
    processed = {}
    for key, value in tqdm(residuals_dict.items(), desc="Processing residuals"):
        # If value is already a list of segments, assume these are the segments you want.
        if isinstance(value, list):
            processed[key] = value
        else:
            data = value.cpu().numpy()
            n = data.shape[0]
            n_seqs = n // seq_length
            if n_seqs > 0:
                sequences = np.split(data[:n_seqs * seq_length], n_seqs)
            else:
                sequences = []
            processed[key] = sequences
    return processed

# -------------------- Main Execution --------------------
if __name__ == "__main__":
    # Load the residuals dictionary.
    residuals_dict = torch.load('residuals_dict.pth')

    # === Adjust residuals based on rotation speed (variable-length segmentation) ===
    # For each key, the rotation speed changes every 250,000 points.
    from Data.LoadData import data_paths, get_omegas

    for key in residuals_dict:
        print(f"Original length for {key}: {len(residuals_dict[key])}")

        # Get the file path for the key and compute the corresponding omegas.
        file_path = data_paths[key]
        omegas = get_omegas(file_path)
        omegas = omegas / (2 * np.pi)  # Convert from rad/s to Hz

        num_rotations = 1
        sampling_rate = 50000  # 50kHz

        # Compute datapoints per rotation and the needed segment lengths.
        datapoints_per_rotation = sampling_rate / omegas
        datapoints_needed = np.ceil(num_rotations * datapoints_per_rotation).to(torch.int32)

        # Process in blocks of 250,000 points.
        n_blocks = len(residuals_dict[key]) // 250000
        segments = []
        for i in range(n_blocks):
            # Use computed segment length if available; otherwise, use fallback.
            seg_length = datapoints_needed[i] if i < len(datapoints_needed) else 200
            max_segments = 1  # Adjust if more segments per block are desired.
            for j in range(max_segments):
                start_idx = i * 250000 + j * seg_length
                end_idx = start_idx + seg_length
                if end_idx <= (i + 1) * 250000:
                    segments.append(residuals_dict[key][start_idx:end_idx])
        residuals_dict[key] = segments
        print(f"Segmented into {len(segments)} samples for {key}")

    # Further process the residuals if needed.
    processed = process_residuals(residuals_dict, seq_length=100)

    # Example: Splitting normal data for train/test.
    normal_sequences = processed.get('normal', [])
    np.random.shuffle(normal_sequences)
    split_idx = int(0.8 * len(normal_sequences))
    normal_train, normal_test = normal_sequences[:split_idx], normal_sequences[split_idx:]

    # Prepare test samples.
    test_samples = {'normal': normal_test}
    for key, sequences in processed.items():
        if key != 'normal':
            test_samples[key] = sequences

    # Validate and compare clustering methods.
    results = validate_all_clustering_methods(processed)

    # Print the clustering validation results.
    print("\n=== Clustering Validation Results ===")
    for method, metrics in results.items():
        print(f"{method.upper():<20} | Silhouette: {metrics['silhouette']:.2f} | ARI: {metrics['ari']:.2f}")
