import numpy as np
from itertools import combinations, product
import multiprocessing as mp
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from fastdtw import fastdtw
import torch
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
import os
import pickle
import concurrent.futures
import pywt

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
    features = [extract_features(seq) for seq in all_sequences]
    X = np.stack(features)
    X = X.reshape(X.shape[0], -1)
    X_scaled = StandardScaler().fit_transform(X)
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

def _extract_frequency_features_1d(signal, sr=50000, roll_off_threshold=0.85):
    """Extract frequency features from a 1D signal."""
    fft_vals = np.fft.rfft(signal)
    mag = np.abs(fft_vals)
    freqs = np.fft.rfftfreq(len(signal), d=1/sr)
    mag_sum = np.sum(mag) if np.sum(mag) > 0 else 1e-10
    spectral_centroid = np.sum(freqs * mag) / mag_sum
    spectral_bandwidth = np.sqrt(np.sum(mag * (freqs - spectral_centroid) ** 2) / mag_sum)
    cumulative_energy = np.cumsum(mag)
    total_energy = cumulative_energy[-1]
    roll_off_idx = np.where(cumulative_energy >= roll_off_threshold * total_energy)[0][0]
    spectral_rolloff = freqs[roll_off_idx]
    eps = 1e-10
    spectral_flatness = (np.exp(np.mean(np.log(mag + eps))) / (np.mean(mag) + eps))
    dominant_frequency = freqs[np.argmax(mag)]
    return np.array([spectral_centroid, spectral_bandwidth, spectral_rolloff, spectral_flatness, dominant_frequency])

def extract_frequency_features(sequence, sr=50000, roll_off_threshold=0.85):
    """
    Extract frequency domain features from a time series using FFT.
    If the sequence has more than one channel, compute features per channel and average.
    Returns a feature vector:
      [spectral_centroid, spectral_bandwidth, spectral_rolloff, spectral_flatness, dominant_frequency]
    """
    sequence = np.array(sequence)
    if sequence.ndim > 1:
        # Compute features for each channel and average them.
        features_all = [_extract_frequency_features_1d(sequence[:, i], sr, roll_off_threshold)
                        for i in range(sequence.shape[1])]
        features_all = np.array(features_all)
        return np.mean(features_all, axis=0).tolist()
    else:
        return _extract_frequency_features_1d(sequence, sr, roll_off_threshold).tolist()

def frequency_domain_clustering(residuals_dict, n_clusters=None, sr=50000, roll_off_threshold=0.85):
    """
    Cluster residuals using frequency domain features.
    Returns cluster labels and true labels.
    """
    all_sequences = []
    all_labels = []
    for label, sequences in residuals_dict.items():
        all_sequences.extend(sequences)
        all_labels.extend([label] * len(sequences))
    features = [extract_frequency_features(seq, sr=sr, roll_off_threshold=roll_off_threshold)
                for seq in all_sequences]
    X = np.stack(features)
    X_scaled = StandardScaler().fit_transform(X)
    if n_clusters is None:
        n_clusters = len(residuals_dict)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    return cluster_labels, all_labels

def extract_wavelet_features(sequence, wavelet='db4', level=4):
    """
    Extract wavelet transform features from a time series.
    If the sequence is multi-channel, average the channels first.
    For each coefficient array (approximation and details), computes:
      mean, standard deviation, energy, and entropy.
    Returns a concatenated feature vector.
    """
    sequence = np.array(sequence)
    # If multi-channel, average across channels.
    if sequence.ndim > 1:
        sequence = np.mean(sequence, axis=1)
    coeffs = pywt.wavedec(sequence, wavelet, level=level)
    features = []
    for coeff in coeffs:
        mean_coeff = np.mean(coeff)
        std_coeff = np.std(coeff)
        energy_coeff = np.sum(np.square(coeff))
        abs_coeff = np.abs(coeff)
        sum_coeff = np.sum(abs_coeff)
        if sum_coeff > 0:
            prob_coeff = abs_coeff / sum_coeff
        else:
            prob_coeff = np.zeros_like(coeff)
        entropy_coeff = -np.sum(prob_coeff * np.log2(prob_coeff + 1e-10))
        features.extend([mean_coeff, std_coeff, energy_coeff, entropy_coeff])
    return features

def wavelet_domain_clustering(residuals_dict, n_clusters=None, wavelet='db4', level=4):
    """
    Cluster residuals using features extracted from the wavelet transform.
    Returns cluster labels and true labels.
    """
    all_sequences = []
    all_labels = []
    for label, sequences in residuals_dict.items():
        all_sequences.extend(sequences)
        all_labels.extend([label] * len(sequences))
    features = [extract_wavelet_features(seq, wavelet=wavelet, level=level)
                for seq in all_sequences]
    X = np.stack(features)
    X_scaled = StandardScaler().fit_transform(X)
    if n_clusters is None:
        n_clusters = len(residuals_dict)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
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

def combined_fourier_wavelet_clustering(residuals_dict, n_clusters=None, sr=50000,
                                          roll_off_threshold=0.85, wavelet='db4', level=4,
                                          n_components=0.95):
    """
    Combine Fourier (FFT) and Wavelet features, reduce dimensionality with PCA,
    and cluster using KMeans.
    
    For each sequence, the function extracts:
      - FFT-based features (spectral centroid, bandwidth, roll-off, flatness, dominant frequency)
      - Wavelet features (mean, std, energy, entropy for each decomposition coefficient)
    These features are concatenated, standardized, and then reduced via PCA before clustering.
    Returns cluster labels and true labels.
    """
    all_sequences = []
    all_labels = []
    combined_features = []
    for label, sequences in residuals_dict.items():
        for seq in sequences:
            all_sequences.append(seq)
            all_labels.append(label)
            fourier_feats = extract_frequency_features(seq, sr=sr, roll_off_threshold=roll_off_threshold)
            wavelet_feats = extract_wavelet_features(seq, wavelet=wavelet, level=level)
            combined_feats = fourier_feats + wavelet_feats
            combined_features.append(combined_feats)
    X = np.stack(combined_features)
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    if n_clusters is None:
        n_clusters = len(residuals_dict)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(X_pca)
    return cluster_labels, all_labels

def validate_all_clustering_methods(residuals_dict):
    """
    Compare multiple clustering approaches:
      1. Feature-based + PCA + KMeans (time domain features)
      2. TimeSeriesKMeans with DTW
      3. Frequency Domain Feature-based clustering
      4. Wavelet Domain Feature-based clustering
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
    
    # 1. Feature-based + PCA + KMeans.
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
    
    """ # 2. TimeSeriesKMeans with DTW.
    print("Validating TimeSeriesKMeans clustering...")
    ts_kmeans_labels, _ = dtw_kmeans_clustering(residuals_dict)
    X_ts_kmeans = to_time_series_dataset(all_sequences)
    silhouette_ts_kmeans = silhouette_score(X_ts_kmeans.reshape(len(X_ts_kmeans), -1), ts_kmeans_labels) if len(set(ts_kmeans_labels)) >= 2 else -1
    ari_ts_kmeans = adjusted_rand_score(true_labels, ts_kmeans_labels)
    results['ts_kmeans'] = {'silhouette': silhouette_ts_kmeans, 'ari': ari_ts_kmeans}
    print(results['ts_kmeans']) """
    
    # 3. Frequency Domain Feature-based clustering.
    print("Validating Frequency Domain clustering...")
    freq_labels, _ = frequency_domain_clustering(residuals_dict)
    features_freq = np.stack([np.array(extract_frequency_features(seq)) for seq in all_sequences])
    X_scaled_freq = StandardScaler().fit_transform(features_freq)
    silhouette_freq = silhouette_score(X_scaled_freq, freq_labels) if len(set(freq_labels)) >= 2 else -1
    ari_freq = adjusted_rand_score(true_labels, freq_labels)
    results['frequency_features'] = {'silhouette': silhouette_freq, 'ari': ari_freq}
    print(results['frequency_features'])
    
    # 4. Wavelet Domain Feature-based clustering.
    print("Validating Wavelet Domain clustering...")
    wavelet_labels, _ = wavelet_domain_clustering(residuals_dict)
    features_wavelet = np.stack([np.array(extract_wavelet_features(seq)) for seq in all_sequences])
    X_scaled_wavelet = StandardScaler().fit_transform(features_wavelet)
    silhouette_wavelet = silhouette_score(X_scaled_wavelet, wavelet_labels) if len(set(wavelet_labels)) >= 2 else -1
    ari_wavelet = adjusted_rand_score(true_labels, wavelet_labels)
    results['wavelet_features'] = {'silhouette': silhouette_wavelet, 'ari': ari_wavelet}
    print(results['wavelet_features'])

    # 5. Combined Fourier & Wavelet Domain clustering.
    print("Validating Combined Fourier & Wavelet Domain clustering...")
    combined_labels, _ = combined_fourier_wavelet_clustering(residuals_dict)
    # For computing silhouette, re-extract combined features.
    combined_features = []
    for seq in all_sequences:
        combined_feats = extract_frequency_features(seq) + extract_wavelet_features(seq)
        combined_features.append(combined_feats)
    X_combined = np.stack(combined_features)
    X_scaled_combined = StandardScaler().fit_transform(X_combined)
    silhouette_combined = silhouette_score(X_scaled_combined, combined_labels) if len(set(combined_labels)) >= 2 else -1
    ari_combined = adjusted_rand_score(true_labels, combined_labels)
    results['combined_fourier_wavelet'] = {'silhouette': silhouette_combined, 'ari': ari_combined}
    print(results['combined_fourier_wavelet'])
    
    # Optionally, save one of the label sets if desired.
    """ with open('ts_kmeans_labels.pkl', 'wb') as f:
        pickle.dump(ts_kmeans_labels, f) """
    
    return results

def process_residuals(residuals_dict, seq_length=100):
    """
    Convert residual tensors (or already segmented lists) into sequences.
    If a given key already holds a list (of variable-length segments), pass it through.
    Otherwise, split the tensor into fixed-length chunks.
    """
    processed = {}
    for key, value in tqdm(residuals_dict.items(), desc="Processing residuals"):
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
    from Data.LoadData import data_paths, get_omegas
    for key in residuals_dict:
        print(f"Original length for {key}: {len(residuals_dict[key])}")
        file_path = data_paths[key]
        omegas = get_omegas(file_path)
        omegas = omegas / (2 * np.pi)  # Convert from rad/s to Hz
        num_rotations = 1
        sampling_rate = 50000  # 50kHz
        datapoints_per_rotation = sampling_rate / omegas
        datapoints_needed = np.ceil(num_rotations * datapoints_per_rotation).to(torch.int32)
        n_blocks = len(residuals_dict[key]) // 250000
        segments = []
        for i in range(n_blocks):
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
    test_samples = {'normal': normal_test}
    for key, sequences in processed.items():
        if key != 'normal':
            test_samples[key] = sequences
    
    # Validate and compare clustering methods.
    results = validate_all_clustering_methods(processed)
    
    print("\n=== Clustering Validation Results ===")
    for method, metrics in results.items():
        print(f"{method.upper():<20} | Silhouette: {metrics['silhouette']:.2f} | ARI: {metrics['ari']:.2f}")
