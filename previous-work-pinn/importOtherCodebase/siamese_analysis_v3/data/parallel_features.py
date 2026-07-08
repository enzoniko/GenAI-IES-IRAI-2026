"""
Enhanced parallel feature extraction for Siamese networks
"""
import numpy as np
import concurrent.futures
from tqdm import tqdm
import os
from functools import partial
import time
from sklearn.preprocessing import StandardScaler

# Import the feature extractors
from .feature_extractors import extract_features_combined, extract_v0_features

def process_sample_batch(batch, sampling_rate, wavelet, level, include_tsfresh, max_length=None,
                         use_v0_processing=False, fft_length=128, num_channels=None):
    """
    Process a batch of samples in parallel

    Args:
        batch: List of samples to process
        sampling_rate: Sampling rate in Hz
        wavelet: Wavelet type
        level: Wavelet decomposition level
        include_tsfresh: Whether to include tsfresh features
        max_length: Maximum sequence length for padding
        use_v0_processing: Whether to use simplified v0 processing
        fft_length: Number of FFT magnitude components to extract
        num_channels: Number of channels to use (None = use all channels)

    Returns:
        List of processed feature vectors
    """
    # Slice batch samples to only use the specified number of channels
    if num_channels is not None:
        sliced_batch = []
        for sample in batch:
            sample = np.asarray(sample)
            if sample.ndim == 1:
                # Single channel data, reshape to (length, 1)
                sample = sample.reshape(-1, 1)
            # Slice to only use the first num_channels
            if sample.shape[1] > num_channels:
                sample = sample[:, :num_channels]
            sliced_batch.append(sample)
        batch = sliced_batch

    results = []
    for sample in batch:
        try:
            if use_v0_processing:
                # Extract features using simple v0 processing
                features = extract_v0_features(
                    sample, 
                    sample_rate=sampling_rate,
                    fft_length=fft_length,
                    max_length=max_length
                )
            else:
                # Extract features using the combined extractor with consistent max_length
                features = extract_features_combined(
                    sample, 
                    sample_rate=sampling_rate,
                    wavelet=wavelet,
                    level=level,
                    include_tsfresh=include_tsfresh,
                    max_length=max_length
                )
            results.append((True, features))
        except Exception as e:
            # Return a failure indicator and the error
            results.append((False, str(e)))
    
    return results

def parallel_extract_features(samples, sampling_rate=50000, wavelet='db4', level=3,
                              include_tsfresh=False, max_workers=None, batch_size=10,
                              max_length=None, use_v0_processing=False, fft_length=128, num_channels=None):
    """
    Extract features from samples in parallel

    Args:
        samples: List of samples to process
        sampling_rate: Sampling rate in Hz
        wavelet: Wavelet type
        level: Wavelet decomposition level
        include_tsfresh: Whether to include tsfresh features
        max_workers: Maximum number of worker processes (None = auto)
        batch_size: Number of samples to process per worker batch
        max_length: Maximum sequence length for padding
        use_v0_processing: Whether to use simplified v0 processing
        fft_length: Number of FFT magnitude components to extract
        num_channels: Number of channels to use (None = use all channels)

    Returns:
        Array of processed features
    """
    if max_workers is None:
        # Default to number of CPU cores - 1 (leave one for the main process)
        max_workers = max(1, os.cpu_count() - 1)

    # Slice samples to only use the specified number of channels
    if num_channels is not None:
        sliced_samples = []
        for sample in samples:
            sample = np.asarray(sample)
            if sample.ndim == 1:
                # Single channel data, reshape to (length, 1)
                sample = sample.reshape(-1, 1)
            # Slice to only use the first num_channels
            if sample.shape[1] > num_channels:
                sample = sample[:, :num_channels]
            sliced_samples.append(sample)
        samples = sliced_samples

    # Create batches
    n_samples = len(samples)
    batch_indices = list(range(0, n_samples, batch_size))
    batches = [samples[i:min(i+batch_size, n_samples)] for i in batch_indices]
    
    if use_v0_processing:
        print(f"Processing {n_samples} samples in {len(batches)} batches using v0 processing")
        print(f"Using FFT length: {fft_length}, max_length: {max_length}")
    else:
        print(f"Processing {n_samples} samples in {len(batches)} batches using advanced processing")
        print(f"Using consistent max_length: {max_length}")
    
    start_time = time.time()
    
    # Process batches in parallel
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Create a partial function with fixed arguments
        process_func = partial(
            process_sample_batch,
            sampling_rate=sampling_rate,
            wavelet=wavelet,
            level=level,
            include_tsfresh=include_tsfresh,
            max_length=max_length,
            use_v0_processing=use_v0_processing,
            fft_length=fft_length,
            num_channels=num_channels
        )
        
        # Submit all batches to the executor
        futures = {executor.submit(process_func, batch): i for i, batch in enumerate(batches)}
        
        # Collect results as they complete
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing batches"):
            batch_idx = futures[future]
            try:
                batch_results = future.result()
                results.extend(batch_results)
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                # Add error placeholders for this batch
                batch_size_actual = len(batches[batch_idx])
                results.extend([(False, str(e))] * batch_size_actual)
    
    # Process results
    features = []
    error_count = 0
    first_successful_shape = None
    
    for success, result in results:
        if success:
            if first_successful_shape is None:
                first_successful_shape = np.array(result).shape
                print(f"First successful feature shape: {first_successful_shape}")
            features.append(result)
        else:
            error_count += 1
            # Use a fallback feature vector if we have at least one successful result
            if first_successful_shape is not None:
                # Use the same dimensions as successful features
                fallback = np.zeros(first_successful_shape)
            else:
                # Reasonable default size if we have no successful extractions
                # For 2D data [channels, features_per_channel]
                num_channels = 4  # Default from args
                features_per_channel = 9 + (max_length//2 + 1) if max_length else 137  # 9 statistical + FFT bins
                fallback = np.zeros((num_channels, features_per_channel))
            features.append(fallback)
    
    elapsed_time = time.time() - start_time
    print(f"Processed {n_samples - error_count} samples successfully ({error_count} errors) in {elapsed_time:.2f} seconds")
    print(f"Average time per sample: {elapsed_time / n_samples:.4f} seconds")
    
    # Ensure all features have the same shape before creating an array
    if features:
        # Get the shape of the first feature
        expected_shape = np.array(features[0]).shape
        print(f"Expected feature shape: {expected_shape}")
        
        # Check that all features have the same shape
        for i, feature in enumerate(features):
            feature_shape = np.array(feature).shape
            if feature_shape != expected_shape:
                print(f"Warning: Feature {i} has shape {feature_shape}, expected {expected_shape}")
                # Reshape or pad the feature to the expected shape
                features[i] = np.zeros(expected_shape)
        
        # Stack once we've ensured all features have the same shape
        return np.stack(features)
    else:
        print("No features to return")
        return np.array([])

def process_label_samples(label, samples, sampling_rate, wavelet, level, include_tsfresh,
                         feature_extractor, max_length=None, use_v0_processing=False, fft_length=128, num_channels=None):
    """
    Process all samples for a single label in parallel

    Args:
        label: Label for these samples
        samples: List of sample data arrays
        sampling_rate: Sampling rate in Hz
        wavelet: Wavelet type
        level: Wavelet decomposition level
        include_tsfresh: Whether to include tsfresh features
        feature_extractor: Function to extract features
        max_length: Maximum sequence length for padding
        use_v0_processing: Whether to use simplified v0 processing
        fft_length: Number of FFT magnitude components to extract
        num_channels: Number of channels to use (None = use all channels)

    Returns:
        (label, features, scaler) tuple
    """
    try:
        if use_v0_processing:
            print(f"Processing label {label} with v0 processing (FFT only)")
            # Extract data from samples
            sample_data = []
            for sample in samples:
                if hasattr(sample, 'data'):
                    sample_data.append(sample.data)
                else:
                    sample_data.append(sample)
            
            # Extract features using v0 method
            processed = parallel_extract_features(
                sample_data,
                sampling_rate=sampling_rate,
                wavelet=wavelet,
                level=level,
                include_tsfresh=include_tsfresh,
                max_length=max_length,
                use_v0_processing=True,
                fft_length=fft_length,
                num_channels=num_channels
            )
            
            # Flatten for scaling [num_samples, channels, features] -> [num_samples, channels*features]
            reshaped = processed.reshape(processed.shape[0], -1)
            
            # Fit a scaler to standardize these features
            scaler = StandardScaler()
            # Don't transform in-place, the same scaler will be used for all labels
            scaler.fit(reshaped)
            
            return (label, processed, scaler)
        else:
            print(f"Processing label {label} with advanced feature extraction")
            # Extract data from samples
            sample_data = []
            for sample in samples:
                if hasattr(sample, 'data'):
                    sample_data.append(sample.data)
                else:
                    sample_data.append(sample)
            
            # Extract features using sophisticated method
            processed = parallel_extract_features(
                sample_data,
                sampling_rate=sampling_rate,
                wavelet=wavelet,
                level=level,
                include_tsfresh=include_tsfresh,
                max_length=max_length,
                use_v0_processing=False,
                fft_length=fft_length,
                num_channels=num_channels
            )
            
            # Flatten for scaling [num_samples, channels, features] -> [num_samples, channels*features]
            reshaped = processed.reshape(processed.shape[0], -1)
            
            # Fit a scaler to standardize these features
            scaler = StandardScaler()
            # Don't transform in-place, the same scaler will be used for all labels
            scaler.fit(reshaped)
            
            return (label, processed, scaler)
    except Exception as e:
        print(f"Error processing label {label}: {e}")
        # Return empty results
        return (label, np.array([]), None)

def parallel_process_all_labels(samples_by_label, level, sampling_rate, wavelet, wavelet_level,
                              include_tsfresh, max_workers=None, feature_extractor=None, max_length=None,
                              use_v0_processing=False, fft_length=128, num_channels=None):
    """
    Process all labeled samples in parallel

    Args:
        samples_by_label: Dictionary mapping labels to lists of samples
        level: Hierarchy level for processing (1, 2, or 3)
        sampling_rate: Sampling rate in Hz
        wavelet: Wavelet type
        wavelet_level: Wavelet decomposition level
        include_tsfresh: Whether to include tsfresh features
        max_workers: Maximum number of worker processes (None = auto)
        feature_extractor: Function to extract features
        max_length: Maximum sequence length for padding
        use_v0_processing: Whether to use simplified v0 processing
        fft_length: Number of FFT magnitude components to extract
        num_channels: Number of channels to use (None = use all channels)

    Returns:
        Tuple of (processed_data, global_scaler)
    """
    # Group samples by level
    samples_by_level = {}
    
    print(f"Grouping samples by level {level}...")
    for label, samples in samples_by_label.items():
        for sample in samples:
            if hasattr(sample, f'level{level}_label'):
                # Get the label for this level
                level_label = getattr(sample, f'level{level}_label')
                
                if level_label not in samples_by_level:
                    samples_by_level[level_label] = []
                
                samples_by_level[level_label].append(sample)
    
    print(f"Found {len(samples_by_level)} unique labels at level {level}")
    
    # Determine max_workers if not specified
    if max_workers is None:
        # At most, one process per label, but leave one core free
        max_workers = min(len(samples_by_level), max(1, os.cpu_count() - 1))
    
    print(f"Processing all labels using {max_workers} worker processes")
    start_time = time.time()
    
    # Process each label independently
    processed_data = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Create a partial function for processing
        process_func = partial(
            process_label_samples,
            sampling_rate=sampling_rate,
            wavelet=wavelet,
            level=wavelet_level,
            include_tsfresh=include_tsfresh,
            feature_extractor=feature_extractor,
            max_length=max_length,
            use_v0_processing=use_v0_processing,
            fft_length=fft_length,
            num_channels=num_channels
        )
        
        # Submit jobs
        future_to_label = {}
        for label, samples in samples_by_level.items():
            future = executor.submit(process_func, label, samples)
            future_to_label[future] = label
        
        # Collect results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_label.keys()), 
                          total=len(future_to_label), desc="Processing labels"):
            label = future_to_label[future]
            try:
                result_label, result_features, result_scaler = future.result()
                if result_features.size > 0:
                    processed_data[result_label] = result_features
                else:
                    print(f"Warning: No features extracted for label {label}")
            except Exception as e:
                print(f"Error processing label {label}: {e}")
    
    elapsed_time = time.time() - start_time
    print(f"Processed {len(processed_data)} labels in {elapsed_time:.2f} seconds")
    
    # Combine all data to fit a global scaler
    all_features = []
    for label, features in processed_data.items():
        # Flatten for scaling [num_samples, channels, features] -> [num_samples, channels*features]
        reshaped = features.reshape(features.shape[0], -1)
        all_features.append(reshaped)
    
    if all_features:
        all_features = np.vstack(all_features)
        global_scaler = StandardScaler()
        global_scaler.fit(all_features)
        
        # For v0.3, we won't apply the scaler here - that will be done later
        # when anchors/positives/negatives are created
    else:
        print("Warning: No features to fit scaler")
        global_scaler = None
    
    return processed_data, global_scaler 