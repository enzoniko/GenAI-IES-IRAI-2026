/**
 * C++ Implementation of Physics-Informed Neural Network and EVT-Based Anomaly Detection
 * for RISC-V deployment and performance measurement
 */

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <iostream>
#include <random>
#include <complex>

// Ensure cmath is included early for M_PI potentially
#include <cmath>

// Include Wavelet library header
#define _USE_MATH_DEFINES // For M_PI
// If M_PI is still not defined after including cmath with _USE_MATH_DEFINES,
// define it manually.
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include "wavelet/wavelet.h"

// Configuration parameters
const int INPUT_SIZE = 8;    // x2, y2, x3, y3, dx2, dy2, dx3, dy3
const int OUTPUT_SIZE = 4;   // ddx2, ddy2, ddx3, ddy3
const int PHYSICS_OUT = 4;   // f_A, f_B, f_C, f_D
const int HIDDEN_LAYER_SIZE = 32;
const int NUM_RESIDUALS = 4;
const int TIME_SERIES_LENGTH = 1000;
// Features: 10 Time + 5 Fourier + 16 Wavelet (Approx + 3 Detail Levels * 4 stats)
const int FEATURE_COUNT_PER_CHANNEL = 31; // Updated from 20
const int TOTAL_FEATURES = FEATURE_COUNT_PER_CHANNEL * NUM_RESIDUALS;
const int PCA_COMPONENTS = 5;

// System constants - these would be tuned to the actual system
const double M1 = 1.0, M2 = 0.5, M3 = 0.5;
const double D1 = 0.1, D2 = 0.1, D3 = 0.1;
const double K1 = 1000.0, K2 = 800.0;
const double E1 = 0.01;
const double g = 9.81;
const double Omega = 60.0; // rad/s

// Timing variables
std::chrono::time_point<std::chrono::high_resolution_clock> start_time, end_time;

/**
 * Simple neural network implementation
 */
class NeuralNetwork {
private:
    std::vector<double> W1, b1, W2, b2;
    int input_size, hidden_size, output_size;
    
public:
    NeuralNetwork(int in_size, int hidden_size, int out_size) : 
        input_size(in_size), hidden_size(hidden_size), output_size(out_size) {
#ifdef DEBUG_LOGGING
        std::cout << "Initializing NeuralNetwork (" << input_size << " -> " << hidden_size << " -> " << output_size << ")..." << std::endl;
#endif
        // Initialize with random weights - in practice, these would be pretrained
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> d(0, 0.1);
        
        W1.resize(input_size * hidden_size);
        b1.resize(hidden_size);
        W2.resize(hidden_size * output_size);
        b2.resize(output_size);
        
        for (auto& w : W1) w = d(gen);
        for (auto& b : b1) b = d(gen);
        for (auto& w : W2) w = d(gen);
        for (auto& b : b2) b = d(gen);
#ifdef DEBUG_LOGGING
        std::cout << "NeuralNetwork initialized." << std::endl;
#endif
    }
    
    // Load pre-trained weights
    void loadWeights(const std::vector<double>& weights) {
#ifdef DEBUG_LOGGING
        std::cout << "Loading weights into NeuralNetwork..." << std::endl;
#endif
        // In practice, you would load from a file
        // This is just a placeholder
        int pos = 0;
        for (size_t i = 0; i < W1.size(); i++) W1[i] = weights[pos++];
        for (size_t i = 0; i < b1.size(); i++) b1[i] = weights[pos++];
        for (size_t i = 0; i < W2.size(); i++) W2[i] = weights[pos++];
        for (size_t i = 0; i < b2.size(); i++) b2[i] = weights[pos++];
#ifdef DEBUG_LOGGING
        std::cout << "Weights loaded." << std::endl;
#endif
    }
    
    // Forward pass
    std::vector<double> forward(const std::vector<double>& x) {
        // First layer
        std::vector<double> h1(hidden_size, 0.0);
        for (int i = 0; i < hidden_size; i++) {
            for (int j = 0; j < input_size; j++) {
                h1[i] += x[j] * W1[i * input_size + j];
            }
            h1[i] += b1[i];
            // ReLU activation
            h1[i] = std::tanh(h1[i]);
        }
        
        // Output layer
        std::vector<double> output(output_size, 0.0);
        for (int i = 0; i < output_size; i++) {
            for (int j = 0; j < hidden_size; j++) {
                output[i] += h1[j] * W2[i * hidden_size + j];
            }
            output[i] += b2[i];
        }
        
        return output;
    }
};

/**
 * Physics-Informed Neural Network for Residual Generation
 */
class PhysicsInformedModel {
private:
    NeuralNetwork acc_network;
    NeuralNetwork phys_network;
    
    // Normalization parameters (Input MinMax Scaling)
    std::vector<double> min_vals;
    std::vector<double> scale_vals;
    // Denormalization parameters (Output Acceleration)
    std::vector<double> acc_min;
    std::vector<double> acc_scale;

    // Store Omega and t if needed (assuming t varies, Omega constant)
    // double current_omega; // Omega seems constant in Python code
    // double current_t; // 't' is an input feature in Python, not tracked here yet
    
public:
    PhysicsInformedModel() : 
        acc_network(INPUT_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_SIZE),
        phys_network(INPUT_SIZE, HIDDEN_LAYER_SIZE, PHYSICS_OUT) {
#ifdef DEBUG_LOGGING
        std::cout << "Initializing PhysicsInformedModel..." << std::endl;
#endif
        
        // Initialize normalization parameters (Replace with actual values if needed)
        min_vals = {-0.001, -0.001, -0.001, -0.001, -0.01, -0.01, -0.01, -0.01};
        scale_vals = {0.002, 0.002, 0.002, 0.002, 0.02, 0.02, 0.02, 0.02};
        
        // Initialize denormalization parameters for accelerations
        acc_min = {-0.1, -0.1, -0.1, -0.1};
        acc_scale = {0.2, 0.2, 0.2, 0.2};
#ifdef DEBUG_LOGGING
        std::cout << "PhysicsInformedModel initialized." << std::endl;
#endif
    }
    
    // Load pre-trained weights
    void loadWeights(const std::vector<double>& acc_weights, 
                     const std::vector<double>& phys_weights) {
#ifdef DEBUG_LOGGING
        std::cout << "Loading weights into PhysicsInformedModel..." << std::endl;
#endif
        acc_network.loadWeights(acc_weights);
        phys_network.loadWeights(phys_weights);
#ifdef DEBUG_LOGGING
        std::cout << "PhysicsInformedModel weights loaded." << std::endl;
#endif
    }
    
    // Normalize input
    std::vector<double> normalize(const std::vector<double>& x) {
        // std::cout << "Normalizing input..." << std::endl; // Can be noisy
        std::vector<double> normalized(x.size());
        for (size_t i = 0; i < x.size(); i++) {
            normalized[i] = (x[i] - min_vals[i]) / scale_vals[i];
        }
        return normalized;
    }
    
    // Denormalize output
    std::vector<double> denormalizeAccelerations(const std::vector<double>& x_norm) {
        // std::cout << "Denormalizing accelerations..." << std::endl; // Can be noisy
        std::vector<double> denormalized(x_norm.size());
        for (size_t i = 0; i < x_norm.size(); i++) {
            denormalized[i] = x_norm[i] * acc_scale[i] + acc_min[i];
        }
        return denormalized;
    }
    
    // Calculate physics residuals
    // Now accepts normalized accelerations and denormalizes internally
    std::vector<double> calculateResiduals(const std::vector<double>& state, 
                                          const std::vector<double>& accelerations_norm) { 
        // std::cout << "Calculating physics residuals..." << std::endl; // Can be noisy
        
        // Denormalize accelerations internally before using in physics equations
        std::vector<double> accelerations = denormalizeAccelerations(accelerations_norm);

        // Extract states (assuming state includes x2..y3, dx2..dy3, but not omega, t directly)
        // Python input: x2_dot, y2_dot, x3_dot, y3_dot, x2, y2, x3, y3, omega, t (10 inputs)
        // C++ input state: x2, y2, x3, y3, dx2, dy2, dx3, dy3 (8 inputs)
        // Need to handle omega and t if they vary per timestep.
        // Assuming Omega is constant and t needs to be passed or inferred if required by equations.
        // For now, assuming we only need state[0-7] and constant Omega.
        // If 't' is needed, it should be passed alongside the state. Let's assume t=0 for now for complexity.
        double t = 0.0; // Placeholder: Get actual time if needed

        double x2 = state[0], y2 = state[1], x3 = state[2], y3 = state[3];
        double dx2 = state[4], dy2 = state[5], dx3 = state[6], dy3 = state[7];
        double ddx2 = accelerations[0], ddy2 = accelerations[1];
        double ddx3 = accelerations[2], ddy3 = accelerations[3];
        
        // Get physics network output for unmeasured terms
        // Note: Python NNforUnmeasured takes 15 inputs including M1,D1,K1,K2,E1,omega,t
        // C++ phys_network takes 8 inputs (state). For complexity test, this might be ok.
        // For accuracy, the network inputs must match.
        std::vector<double> normalized_state = normalize(state); 
        std::vector<double> f_terms = phys_network.forward(normalized_state);
        
        // Calculate left-hand sides of equations matching Python residuals (Eq1=fA, Eq2=fB, etc.)
        // Eq1: K1*x2 + K2*x3 + M1*Omega^2*E1*cos(Omega*t) - fA
        // Eq2: K1*y2 + K2*y3 - M1*g + M1*Omega^2*E1*sin(Omega*t) - fB
        // Eq3: M3*ddx3 + D3*dx3 + K2*x3 - (K2/K1)*M2*ddx2 - (K2/K1)*D2*dx2 - K2*x2 - fC (Using M2/D2 like C++ original)
        // Eq4: M3*ddy3 + D3*dy3 + K2*y3 - (K2/K1)*M2*ddy2 - (K2/K1)*D2*dy2 - K2*y2 - (K2/K1)*M2*g + M3*g - fD (Using M2/D2 like C++ original)
        
        // Calculate residuals
        std::vector<double> residuals(NUM_RESIDUALS);
        residuals[0] = K1 * x2 + K2 * x3 + M1 * Omega * Omega * E1 * std::cos(Omega * t) - f_terms[0]; // Eq1 - f_A (Added trig term)
        residuals[1] = K1 * y2 + K2 * y3 - M1 * g + M1 * Omega * Omega * E1 * std::sin(Omega * t) - f_terms[1]; // Eq2 - f_B (Added trig term)
        residuals[2] = M3 * ddx3 + D3 * dx3 + K2 * x3 - (K2/K1) * M2 * ddx2 - (K2/K1) * D2 * dx2 - K2 * x2 - f_terms[2]; // Eq3 - f_C
        residuals[3] = M3 * ddy3 + D3 * dy3 + K2 * y3 - (K2/K1) * M2 * ddy2 - (K2/K1) * D2 * dy2 - K2 * y2 - (K2/K1) * M2 * g + M3 * g - f_terms[3]; // Eq4 - f_D
        
        // std::cout << "Residuals calculated." << std::endl;
        return residuals;
    }
    
    // Process a single time step
    std::vector<double> processTimeStep(const std::vector<double>& state) {
        // std::cout << "Processing time step..." << std::endl; // Very noisy
        // Normalize state
        std::vector<double> normalized_state = normalize(state);
        
        // Predict accelerations (normalized)
        std::vector<double> predicted_acc_norm = acc_network.forward(normalized_state);
        // std::vector<double> predicted_acc = denormalizeAccelerations(predicted_acc_norm); // Removed denormalization here
        
        // Calculate residuals using normalized accelerations
        return calculateResiduals(state, predicted_acc_norm); // Pass normalized accelerations
    }
    
    // Process a time series and return residuals
    std::vector<std::vector<double>> processTimeSeries(
        const std::vector<std::vector<double>>& time_series) {
#ifdef DEBUG_LOGGING
        std::cout << "Processing time series for residuals..." << std::endl;
#endif
        
        std::vector<std::vector<double>> residuals(time_series.size(), 
                                                 std::vector<double>(NUM_RESIDUALS));
        
        for (size_t t = 0; t < time_series.size(); t++) {
            residuals[t] = processTimeStep(time_series[t]);
        }
        
#ifdef DEBUG_LOGGING
        std::cout << "Time series processed. Residuals generated." << std::endl;
#endif
        return residuals;
    }
};

// --- FFT Implementation (Recursive Cooley-Tukey) ---

// Helper function to check if N is a power of 2
bool isPowerOfTwo(int n) {
    return (n > 0) && ((n & (n - 1)) == 0);
}

// Helper function to find the next power of 2
int nextPowerOfTwo(int n) {
    if (n <= 0) return 1;
    if (isPowerOfTwo(n)) return n;
    int power = 1;
    while (power < n) {
        power <<= 1;
    }
    return power;
}

// Recursive FFT for complex inputs (modifies x in-place)
void fft_recursive(std::vector<std::complex<double>>& x) {
    int N = x.size();
    if (N <= 1) return;

    // Ensure N is a power of 2 (should be handled by caller padding)
    // if (!isPowerOfTwo(N)) { 
    //     throw std::runtime_error("FFT input size must be a power of 2");
    // }

    std::vector<std::complex<double>> even(N / 2), odd(N / 2);
    for (int i = 0; i < N / 2; ++i) {
        even[i] = x[i * 2];
        odd[i] = x[i * 2 + 1];
    }

    fft_recursive(even);
    fft_recursive(odd);

    for (int k = 0; k < N / 2; ++k) {
        std::complex<double> t = std::polar(1.0, -2 * M_PI * k / N) * odd[k];
        x[k] = even[k] + t;
        x[k + N / 2] = even[k] - t;
    }
}

// FFT function for real-valued input (returns complex spectrum)
std::vector<std::complex<double>> fft_real(const std::vector<double>& data) {
    int N_orig = data.size();
    int N = nextPowerOfTwo(N_orig); // Pad to power of 2 for this simple FFT

    // Create complex vector padded with zeros
    std::vector<std::complex<double>> x(N, 0.0);
    for (int i = 0; i < N_orig; ++i) {
        x[i] = data[i];
    }

    // Perform FFT
    fft_recursive(x); 
    
    return x; // Return the full complex spectrum
}

// --- End FFT Implementation ---

/**
 * Feature extraction from residuals
 */
class FeatureExtractor {
public:
    // Extract all features from residual time series
    std::vector<double> extractFeatures(const std::vector<std::vector<double>>& residuals) {
#ifdef DEBUG_LOGGING
        std::cout << "Extracting features..." << std::endl;
#endif
        std::vector<double> features;
        features.reserve(TOTAL_FEATURES);
        
        // Process each channel
        for (int channel = 0; channel < NUM_RESIDUALS; channel++) {
            // Extract channel data
            std::vector<double> channel_data(residuals.size());
            for (size_t t = 0; t < residuals.size(); t++) {
                channel_data[t] = residuals[t][channel];
            }
            
            // Extract time-domain features
            std::vector<double> time_features = extractTimeDomainFeatures(channel_data);
            features.insert(features.end(), time_features.begin(), time_features.end());
            
            // Extract frequency-domain features
            std::vector<double> freq_features = extractFourierFeatures(channel_data);
            features.insert(features.end(), freq_features.begin(), freq_features.end());
            
            // Extract wavelet features
            // std::cout << "Extracting wavelet features for channel " << channel << "..." << std::endl; // Can be noisy
            std::vector<double> wavelet_features = extractWaveletFeatures(channel_data);
            features.insert(features.end(), wavelet_features.begin(), wavelet_features.end());
        }
#ifdef DEBUG_LOGGING
        std::cout << "Features extracted. Total features: " << features.size() << std::endl;
#endif
        return features;
    }
    
private:
    // Extract time-domain statistical features
    std::vector<double> extractTimeDomainFeatures(const std::vector<double>& data) {
        std::vector<double> features(10);
        
        // Calculate mean
        double sum = std::accumulate(data.begin(), data.end(), 0.0);
        double mean = sum / data.size();
        features[0] = mean;
        
        // Calculate median (simplified implementation)
        std::vector<double> sorted_data = data;
        std::sort(sorted_data.begin(), sorted_data.end());
        features[1] = sorted_data[data.size() / 2];
        
        // Calculate standard deviation
        double sq_sum = std::inner_product(data.begin(), data.end(), data.begin(), 0.0);
        double stdev = std::sqrt(sq_sum / data.size() - mean * mean);
        features[2] = stdev;
        
        // Max and min
        auto [min_it, max_it] = std::minmax_element(data.begin(), data.end());
        features[3] = *max_it;
        features[4] = *min_it;
        
        // Kurtosis and Skewness - simplified implementation
        double m2 = 0, m3 = 0, m4 = 0;
        for (const double& val : data) {
            double delta = val - mean;
            double delta_sq = delta * delta;
            m2 += delta_sq;
            m3 += delta_sq * delta;
            m4 += delta_sq * delta_sq;
        }
        m2 /= data.size();
        m3 /= data.size();
        m4 /= data.size();
        
        double skewness = m3 / std::pow(m2, 1.5);
        double kurtosis = m4 / (m2 * m2) - 3.0;  // Excess kurtosis
        features[5] = kurtosis;
        features[6] = skewness;
        
        // Energy
        double energy = sq_sum;
        features[7] = energy;
        
        // Shannon entropy (simplified)
        // In practice, you would calculate a proper histogram
        features[8] = std::log(stdev);  // Approximation
        
        // Peak value
        features[9] = std::abs(*max_it) > std::abs(*min_it) ? std::abs(*max_it) : std::abs(*min_it);
        
        return features;
    }
    
    // Extract Fourier features
    std::vector<double> extractFourierFeatures(const std::vector<double>& data) {
        // std::cout << "Extracting Fourier features..." << std::endl; // Can be noisy
        const int n = data.size();
        if (n == 0) return std::vector<double>(5, 0.0); // Handle empty input

        std::vector<double> features(5, 0.0); // Centroid, Bandwidth, Rolloff, Flatness, Dominant Freq
        
        // ----- Use Custom FFT Implementation -----
        // 1. Perform FFT on real data (handles padding to power of 2)
        std::vector<std::complex<double>> fft_complex_result = fft_real(data);
        int N_fft = fft_complex_result.size(); // Size after padding
        int N_out = N_fft / 2 + 1; // Number of unique complex points for real FFT

        // 2. Process the output (take first N_out points)
        std::vector<double> magnitudes(N_out);
        std::vector<double> freqs(N_out);
        double sample_rate = 1.0; // Assume sample rate of 1 Hz if not provided, adjust if needed
        for (int i = 0; i < N_out; i++) {
            magnitudes[i] = std::abs(fft_complex_result[i]); // Magnitude
            // Frequency = index * sample_rate / n
            freqs[i] = static_cast<double>(i) * sample_rate / static_cast<double>(N_fft);
        }
        
        // 3. Calculate features from magnitudes and freqs (similar to Python code)
        // Feature 1: Spectral centroid
        double weighted_sum = 0.0;
        double sum_magnitudes = 0.0;
        for (int i = 0; i < N_out; i++) {
            weighted_sum += freqs[i] * magnitudes[i]; // Use actual frequencies
            sum_magnitudes += magnitudes[i];
        }
        // Avoid division by zero
        features[0] = sum_magnitudes > 1e-10 ? weighted_sum / sum_magnitudes : 0.0;
        
        // Feature 2: Bandwidth
        double variance = 0.0;
        if (sum_magnitudes > 1e-10) {
            for (int i = 0; i < N_out; i++) {
                variance += std::pow(freqs[i] - features[0], 2) * magnitudes[i]; // Weighted variance
            }
             features[1] = std::sqrt(variance / sum_magnitudes);
        } else {
             features[1] = 0.0;
        }

        // Feature 3: Roll-off 
        double total_energy = std::inner_product(magnitudes.begin(), magnitudes.end(), magnitudes.begin(), 0.0);
        double cumsum = 0.0;
        double threshold = 0.85 * total_energy; // Using energy instead of sum_magnitudes
        int rolloff_index = N_out - 1;
        for (int i = 0; i < N_out; i++) {
            cumsum += magnitudes[i] * magnitudes[i]; // Accumulate energy (magnitude squared)
            if (cumsum >= threshold) {
                rolloff_index = i;
                break;
            }
        }
        features[2] = freqs[rolloff_index]; // Rolloff frequency
        
        // Feature 4: Spectral flatness
        double geometric_mean = 0.0;
        double epsilon = 1e-10; 
        for (int i = 0; i < N_out; i++) {
             // Average of log magnitude
            geometric_mean += std::log(magnitudes[i] + epsilon);
        }
        geometric_mean = std::exp(geometric_mean / N_out);
        double arithmetic_mean = sum_magnitudes / N_out; // Use sum_magnitudes calculated earlier
        features[3] = (arithmetic_mean > epsilon) ? (geometric_mean / arithmetic_mean) : 0.0;
        
        // Feature 5: Dominant frequency
        auto max_it = std::max_element(magnitudes.begin(), magnitudes.end());
        features[4] = freqs[std::distance(magnitudes.begin(), max_it)]; // Frequency at max magnitude

        // ----- End Custom FFT Implementation -----
        
        return features;
    }
    
    // Extract wavelet features (simplified placeholder)
    std::vector<double> extractWaveletFeatures(const std::vector<double>& data, 
                                                 const std::string& wavelet_name = "db4", 
                                                 int level = 3) {
        // std::cout << "Extracting wavelet features using " << wavelet_name << "..." << std::endl; // Can be noisy
        // ----- Use Wavelet Library Implementation -----
        if (data.empty()) return {}; // Return empty vector for empty input

        // Define Daubechies 4 filter coefficients (from standard sources)
        // Order: Lo_D, Hi_D, Lo_R, Hi_R
        // Only Lo_D and Hi_D are strictly needed for wavedec
        // Using dummy values for Lo_R, Hi_R as they aren't needed for decomposition
        const std::vector<double> Lo_D = {0.4829629131445341, 0.8365163037378079, 0.2241438680420134, -0.1294095225512603};
        const std::vector<double> Hi_D = {-0.1294095225512603, -0.2241438680420134, 0.8365163037378079, -0.4829629131445341};
        const std::vector<double> Lo_R = {0.0, 0.0, 0.0, 0.0}; // Dummy
        const std::vector<double> Hi_R = {0.0, 0.0, 0.0, 0.0}; // Dummy

        // TODO: Add support for different wavelet names if needed. Currently hardcoded to db4.
        if (wavelet_name != "db4") {
            // std::cerr << "Warning: Only db4 wavelet is currently supported in extractWaveletFeatures." << std::endl;
             // For now, proceed with db4 coefficients regardless of name for complexity test
        }

        // Create Wavelet object
        Wavelet<double> wavelet(Lo_D, Hi_D, Lo_R, Hi_R);

        // Perform multi-level wavelet decomposition
        // std::cout << "Performing Wavedec..." << std::endl; // Can be noisy
        Decomposition1D<double> decomposition = wavelet.Wavedec(data, level);
        // std::cout << "Wavedec complete." << std::endl;

        // Extract features from coefficients (matches python feature_extraction.py logic)
        std::vector<double> features;
        features.reserve((level + 1) * 4); // Reserve space: (level details + 1 approx) * 4 features each

        // Process approximation coefficients (cAn)
        const auto& app_coeffs = decomposition.GetAppcoef();
        if (!app_coeffs.empty()) {
            double mean_val = std::accumulate(app_coeffs.begin(), app_coeffs.end(), 0.0) / app_coeffs.size();
            double sq_sum = std::inner_product(app_coeffs.begin(), app_coeffs.end(), app_coeffs.begin(), 0.0);
            double std_val = std::sqrt(sq_sum / app_coeffs.size() - mean_val * mean_val);
            double energy_val = sq_sum;
            // Simplified Entropy calculation (using log(std_dev) as proxy, matching python)
            double entropy_val = std::log(std_val + 1e-8);
            features.insert(features.end(), {mean_val, std_val, energy_val, entropy_val});
        }

        // Process detail coefficients (cDn, cDn-1, ..., cD1)
        // Hypothesis: GetDetcoef might expect a 0-based index (0=cDn, 1=cDn-1, ..., level-1=cD1)
        // rather than the level number itself.
        for (int index = 0; index < level; ++index) { 
            // Calculate the corresponding level number (level, level-1, ..., 1) if needed for context
            // int current_level_number = level - index;
            const auto& det_coeffs = decomposition.GetDetcoef(index);
            if (!det_coeffs.empty()) {
                double mean_val = std::accumulate(det_coeffs.begin(), det_coeffs.end(), 0.0) / det_coeffs.size();
                double sq_sum = std::inner_product(det_coeffs.begin(), det_coeffs.end(), det_coeffs.begin(), 0.0);
                double std_val = std::sqrt(sq_sum / det_coeffs.size() - mean_val * mean_val);
                double energy_val = sq_sum;
                // Simplified Entropy calculation (NOTE: Python uses histogram-based Shannon entropy)
                double entropy_val = std::log(std_val + 1e-8);
                features.insert(features.end(), {mean_val, std_val, energy_val, entropy_val});
            } else {
                // Append zeros if coefficients are empty for this level
                 features.insert(features.end(), {0.0, 0.0, 0.0, 0.0});
            }
        }

        // ----- End Wavelet Library Implementation -----
        return features;
    }
};

/**
 * PCA dimensionality reduction
 */
class PCA {
private:
    // PCA components
    std::vector<std::vector<double>> components;
    std::vector<double> mean;
    int n_components_; // Added to store number of components
    
public:
    PCA(int n_components = PCA_COMPONENTS) : n_components_(n_components) {
#ifdef DEBUG_LOGGING
         std::cout << "Initializing PCA with " << n_components_ << " components..." << std::endl;
#endif
        // Initialize with random components - in practice, these would be pre-calculated
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> d(0, 1);
        
        components.resize(n_components_, std::vector<double>(TOTAL_FEATURES));
        mean.resize(TOTAL_FEATURES, 0.0);
        
        for (auto& component : components) {
            for (auto& val : component) {
                val = d(gen);
            }
            // Normalize component
            double norm = std::sqrt(std::inner_product(
                component.begin(), component.end(), component.begin(), 0.0));
            for (auto& val : component) {
                val /= norm;
            }
        }
#ifdef DEBUG_LOGGING
        std::cout << "PCA initialized." << std::endl;
#endif
    }
    
    // Load pre-computed PCA parameters
    void loadParameters(const std::vector<std::vector<double>>& comp, 
                        const std::vector<double>& m) {
#ifdef DEBUG_LOGGING
        std::cout << "Loading PCA parameters..." << std::endl;
#endif
        components = comp;
        mean = m;
        n_components_ = comp.size(); // Update number of components based on loaded data
#ifdef DEBUG_LOGGING
        std::cout << "PCA parameters loaded. Components: " << n_components_ << std::endl;
#endif
    }
    
    // Project data onto PCA components
    std::vector<double> transform(const std::vector<double>& x) {
        // std::cout << "Performing PCA transform..." << std::endl; // Can be noisy
        if (mean.size() != x.size() || (components.empty() && n_components_ > 0)) {
#ifdef DEBUG_LOGGING
             std::cerr << "PCA Error: Mean size (" << mean.size() 
                       << ") does not match input size (" << x.size() 
                       << ") or components not loaded properly." << std::endl;
#endif
             return {}; // Return empty or handle error appropriately
        }
        if (components.empty() && n_components_ == 0) {
#ifdef DEBUG_LOGGING
             std::cerr << "PCA Warning: No components loaded, returning empty vector." << std::endl;
#endif
             return {};
        }
        if (!components.empty() && components[0].size() != x.size()) { // Added check for non-empty components
#ifdef DEBUG_LOGGING
             std::cerr << "PCA Error: Component dimension (" << components[0].size() 
                       << ") does not match input dimension (" << x.size() << ")." << std::endl;
#endif
             return {};
        }


        std::vector<double> result(n_components_, 0.0);
        
        // Center the data
        std::vector<double> centered(x.size());
        for (size_t i = 0; i < x.size(); i++) {
            centered[i] = x[i] - mean[i];
        }
        
        // Project data onto components
        for (size_t i = 0; i < components.size(); i++) {
            // Double check sizes before accessing
             if (centered.size() != components[i].size()) {
#ifdef DEBUG_LOGGING
                  std::cerr << "PCA Transform Error: Mismatch between centered data size (" << centered.size() 
                            << ") and component " << i << " size (" << components[i].size() << ")." << std::endl;
#endif
                  return {}; // Or handle error
             }
            for (size_t j = 0; j < centered.size(); j++) {
                result[i] += centered[j] * components[i][j];
            }
        }
        
        return result;
    }
    
    // Calculate Euclidean norm of projected data
    double calculateAnomalyScore(const std::vector<double>& x) {
        std::vector<double> projected = transform(x);
        // std::cout << "Calculating anomaly score (norm of PCA projection)..." << std::endl; // Can be noisy
        if (projected.empty()) {
#ifdef DEBUG_LOGGING
            std::cerr << "PCA Anomaly Score Warning: Projection is empty, returning 0." << std::endl;
#endif
            return 0.0;
        }
        return std::sqrt(std::inner_product(
            projected.begin(), projected.end(), projected.begin(), 0.0));
    }
};

/**
 * StandardScaler implementation
 */
class StandardScaler {
private:
    std::vector<double> mean_;
    std::vector<double> scale_; // Stores standard deviation
    bool fitted_ = false;
    size_t n_features_ = 0;

public:
    StandardScaler() {}

    void fit(const std::vector<std::vector<double>>& data) {
#ifdef DEBUG_LOGGING
        std::cout << "Fitting StandardScaler..." << std::endl;
#endif
        if (data.empty()) {
#ifdef DEBUG_LOGGING
            std::cerr << "StandardScaler Warning: Fit called with empty data." << std::endl;
#endif
            return;
        }
        n_features_ = data[0].size();
        mean_.assign(n_features_, 0.0);
        scale_.assign(n_features_, 0.0);
        
        size_t n_samples = data.size();

        // Calculate mean
        for (const auto& sample : data) {
            for (size_t j = 0; j < n_features_; ++j) {
                mean_[j] += sample[j];
            }
        }
        for (size_t j = 0; j < n_features_; ++j) {
            mean_[j] /= n_samples;
        }

        // Calculate standard deviation
        for (const auto& sample : data) {
            for (size_t j = 0; j < n_features_; ++j) {
                scale_[j] += std::pow(sample[j] - mean_[j], 2);
            }
        }
        for (size_t j = 0; j < n_features_; ++j) {
            scale_[j] = std::sqrt(scale_[j] / n_samples);
            // Prevent division by zero
            if (scale_[j] < 1e-8) {
                scale_[j] = 1.0; // Or handle as appropriate
            }
        }
        fitted_ = true;
#ifdef DEBUG_LOGGING
        std::cout << "StandardScaler fitted. Features: " << n_features_ << std::endl;
#endif
    }

    std::vector<double> transform(const std::vector<double>& sample) {
        // std::cout << "StandardScaler: Transforming sample..." << std::endl; // Can be noisy
        if (!fitted_) {
            std::cerr << "StandardScaler Error: Transform called before fitting." << std::endl;
            return sample; // Or throw exception
        }
         if (sample.size() != n_features_) {
             std::cerr << "StandardScaler Error: Sample size (" << sample.size() 
                       << ") does not match fitted features (" << n_features_ << ")." << std::endl;
             return sample; // Or throw exception
         }
        std::vector<double> scaled_sample(n_features_);
        for (size_t j = 0; j < n_features_; ++j) {
            scaled_sample[j] = (sample[j] - mean_[j]) / scale_[j];
        }
        return scaled_sample;
    }
    
    std::vector<std::vector<double>> fit_transform(const std::vector<std::vector<double>>& data) {
        fit(data);
#ifdef DEBUG_LOGGING
        std::cout << "StandardScaler: Fit and transforming data..." << std::endl;
#endif
        std::vector<std::vector<double>> transformed_data;
        transformed_data.reserve(data.size());
        for(const auto& sample : data) {
            transformed_data.push_back(transform(sample));
        }
#ifdef DEBUG_LOGGING
        std::cout << "StandardScaler: Fit and transform complete." << std::endl;
#endif
        return transformed_data;
    }

    const std::vector<double>& get_mean() const { return mean_; }
    const std::vector<double>& get_scale() const { return scale_; }
};

/**
 * EVT-based anomaly detector
 */
class EVTAnomalyDetector {
private:
    double threshold;
    FeatureExtractor feature_extractor;
    StandardScaler feature_scaler; // Added scaler
    PCA pca;
    
public:
    EVTAnomalyDetector() : 
        threshold(10.0)
        {
#ifdef DEBUG_LOGGING
            std::cout << "Initializing EVTAnomalyDetector (PCA always applied)" << std::endl;
#endif
        } // PCA initialized with default components inside its constructor
    
    // Load pre-computed parameters (including scaler means/scales)
    void loadParameters(double tau, 
                        const std::vector<std::vector<double>>& pca_components,
                        const std::vector<double>& pca_mean,
                        const std::vector<double>& scaler_mean, // Add scaler params
                        const std::vector<double>& scaler_scale) // Add scaler params
                         {
#ifdef DEBUG_LOGGING
        std::cout << "Loading parameters into EVTAnomalyDetector..." << std::endl;
        std::cout << "  Threshold: " << tau << std::endl;
#endif
        threshold = tau;
        (void)scaler_mean;  // Mark as unused for now
        (void)scaler_scale; // Mark as unused for now
        // Need methods in PCA and StandardScaler to load parameters directly
        // For now, assuming PCA/Scaler are fitted elsewhere or use dummy data fit
        if (!pca_components.empty()) {
            pca.loadParameters(pca_components, pca_mean); 
        } else {
#ifdef DEBUG_LOGGING
            std::cout << "  PCA components not provided, PCA will use default/random." << std::endl;
#endif
        }
        // feature_scaler.loadParameters(scaler_mean, scaler_scale); // Need to add this method to StandardScaler or fit separately
#ifdef DEBUG_LOGGING
        std::cout << "  Note: StandardScaler parameters are not loaded directly yet, assumed fitted." << std::endl;
        std::cout << "EVTAnomalyDetector parameters loaded." << std::endl;
#endif
    }

    // Input should be a vector of samples, where each sample is a [time][channel] matrix
    void fit(const std::vector<std::vector<std::vector<double>>>& normal_samples, int sample_rate = 1) {
         // 1. Extract features from all normal samples
#ifdef DEBUG_LOGGING
         std::cout << "Fitting EVTAnomalyDetector on " << normal_samples.size() << " normal samples..." << std::endl;
#endif
        (void)sample_rate; // Mark unused parameter

        std::vector<std::vector<double>> normal_features;
        normal_features.reserve(normal_samples.size());
        for(const auto& sample_ts : normal_samples) {
             normal_features.push_back(feature_extractor.extractFeatures(sample_ts));
        }
 
#ifdef DEBUG_LOGGING
         std::cout << "Features extracted from normal samples." << std::endl;
#endif

          // 2. Fit Scaler and Scale features
          std::vector<std::vector<double>> scaled_features = feature_scaler.fit_transform(normal_features); // Fit and transform
#ifdef DEBUG_LOGGING
          std::cout << "Scaler fitted and features transformed." << std::endl;
#endif
          // 3. Fit PCA (or assume preloaded)
          // Assuming PCA class needs a fit method similar to StandardScaler
          // pca.fit(scaled_features); // Need to add fit method to PCA
#ifdef DEBUG_LOGGING
           std::cout << "PCA fitting skipped (using random or preloaded components)." << std::endl;
#endif
          // PCA is always applied now, no need for 'else' branch
 
          // TODO: Implement threshold fitting using EVT/GPD if needed, 
          // or just use the loaded/default threshold.
          // This would involve transforming features, calculating norms, finding threshold_u, 
          // fitting GPD to excesses, and calculating the final threshold.
#ifdef DEBUG_LOGGING
          std::cout << "Threshold fitting (EVT/GPD) not implemented, using default/loaded threshold: " << threshold << std::endl;
          std::cout << "EVTAnomalyDetector fitting complete." << std::endl;
#endif
     }
     
     // Detect anomaly in residual time series
     bool detectAnomaly(const std::vector<std::vector<double>>& residuals) {
#ifdef DEBUG_LOGGING
         std::cout << "Detecting anomaly..." << std::endl;
#endif
         // 1. Extract features
         std::vector<double> features = feature_extractor.extractFeatures(residuals);
         
         // 2. Scale features
         std::vector<double> scaled_features = feature_scaler.transform(features);
 
         // 3. Apply PCA and Calculate anomaly score (PCA always applied)
#ifdef DEBUG_LOGGING
         std::cout << "Features scaled. Calculating anomaly score via PCA..." << std::endl;
#endif
         double score;
         score = pca.calculateAnomalyScore(scaled_features); // PCA calculates norm internally now
 
         // 4. Compare with threshold
#ifdef DEBUG_LOGGING
         std::cout << "Anomaly score: " << score << ", Threshold: " << threshold << std::endl;
#endif
         return score > threshold;
     }
};

/**
 * Complete detection pipeline
 */
class DetectionPipeline {
private:
    PhysicsInformedModel pinn;
    EVTAnomalyDetector detector;
    
public:
    DetectionPipeline() : detector() {} // Call default constructor
    
    // Initialize with pre-trained parameters
    void initialize() {
#ifdef DEBUG_LOGGING
        std::cout << "Initializing DetectionPipeline..." << std::endl;
#endif
        // In real implementation, load from files
        std::vector<double> acc_weights(INPUT_SIZE * HIDDEN_LAYER_SIZE + HIDDEN_LAYER_SIZE + HIDDEN_LAYER_SIZE * OUTPUT_SIZE + OUTPUT_SIZE, 0.1);  // Adjusted size calculation
        std::vector<double> phys_weights(INPUT_SIZE * HIDDEN_LAYER_SIZE + HIDDEN_LAYER_SIZE + HIDDEN_LAYER_SIZE * PHYSICS_OUT + PHYSICS_OUT, 0.1); // Adjusted size calculation
#ifdef DEBUG_LOGGING
        std::cout << "Loading dummy weights for PINN..." << std::endl;
#endif
        pinn.loadWeights(acc_weights, phys_weights);
        
        // Dummy PCA/Scaler parameters (replace with actual if needed)
        // For complexity testing, fitting on dummy data might be better than loading fixed values.
        std::vector<std::vector<double>> pca_components(PCA_COMPONENTS, 
                                                      std::vector<double>(TOTAL_FEATURES, 0.1)); 
        std::vector<double> pca_mean(TOTAL_FEATURES, 0.0);
        std::vector<double> scaler_mean(TOTAL_FEATURES, 0.0);
        std::vector<double> scaler_scale(TOTAL_FEATURES, 1.0);
        // double threshold_tau = 10.0; // Example threshold - Threshold set inside detector or loaded

        // Declare as vector of [time][channel] samples
#ifdef DEBUG_LOGGING
        std::cout << "Generating dummy normal samples for detector fitting..." << std::endl;
#endif
        std::vector<std::vector<std::vector<double>>> dummy_normal_samples;
        // Generate some dummy residual data to fit scaler/PCA (e.g., 10 samples)
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> d_res(0, 0.01);
        for (int i=0; i<10; ++i) {
            std::vector<std::vector<double>> one_sample_ts(TIME_SERIES_LENGTH, std::vector<double>(NUM_RESIDUALS));
            for(int t=0; t<TIME_SERIES_LENGTH; ++t) {
                for(int c=0; c<NUM_RESIDUALS; ++c) {
                    one_sample_ts[t][c] = d_res(gen);
                }
            }
            dummy_normal_samples.push_back(one_sample_ts); // Now pushing correct type
        }
        // Fit the detector (scaler and PCA) on dummy normal data
        detector.fit(dummy_normal_samples); // Fit method unchanged, but applies PCA path internally
#ifdef DEBUG_LOGGING
        std::cout << "Detector fitting complete." << std::endl;
#endif
        // Note: detector.threshold remains the default 10.0 unless fitted by EVT logic
#ifdef DEBUG_LOGGING
        std::cout << "DetectionPipeline initialization complete." << std::endl;
#endif
    }
    
    // Generate dummy data for testing
    std::vector<std::vector<double>> generateDummyData(int length = TIME_SERIES_LENGTH) {
#ifdef DEBUG_LOGGING
        std::cout << "Generating dummy time series data (length=" << length << ")..." << std::endl;
#endif
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> d(0, 0.001);
        
        std::vector<std::vector<double>> data(length, std::vector<double>(INPUT_SIZE));
        
        for (int t = 0; t < length; t++) {
            // Generate simple sinusoidal motion with noise
            double time = t * 0.001;  // Time in seconds
            data[t][0] = 0.0005 * sin(2 * M_PI * 10 * time) + d(gen);  // x2
            data[t][1] = 0.0005 * cos(2 * M_PI * 10 * time) + d(gen);  // y2
            data[t][2] = 0.0005 * sin(2 * M_PI * 15 * time) + d(gen);  // x3
            data[t][3] = 0.0005 * cos(2 * M_PI * 15 * time) + d(gen);  // y3
            
            // Velocities are derivatives
            data[t][4] = 0.0005 * 2 * M_PI * 10 * cos(2 * M_PI * 10 * time) + d(gen);  // dx2
            data[t][5] = -0.0005 * 2 * M_PI * 10 * sin(2 * M_PI * 10 * time) + d(gen);  // dy2
            data[t][6] = 0.0005 * 2 * M_PI * 15 * cos(2 * M_PI * 15 * time) + d(gen);  // dx3
            data[t][7] = -0.0005 * 2 * M_PI * 15 * sin(2 * M_PI * 15 * time) + d(gen);  // dy3
        }
        
#ifdef DEBUG_LOGGING
        std::cout << "Dummy data generated." << std::endl;
#endif
        return data;
    }
    
    // Run detection on a sample
    bool detectAnomaly(const std::vector<std::vector<double>>& time_series) {
        start_time = std::chrono::high_resolution_clock::now();
#ifdef DEBUG_LOGGING
        std::cout << "--- Starting Anomaly Detection Run ---" << std::endl;
#endif
        
        // Generate residuals
#ifdef DEBUG_LOGGING
        std::cout << "Step 1: Generating residuals using PINN..." << std::endl;
#endif
        std::vector<std::vector<double>> residuals = pinn.processTimeSeries(time_series);
        
        // Detect anomaly
#ifdef DEBUG_LOGGING
        std::cout << "Step 2: Detecting anomaly using EVTDetector..." << std::endl;
#endif
        bool is_anomaly = detector.detectAnomaly(residuals);
        
        end_time = std::chrono::high_resolution_clock::now();
#ifdef DEBUG_LOGGING
        std::cout << "--- Anomaly Detection Run Finished ---" << std::endl;
#endif
        
        return is_anomaly;
    }
    
    // Measure execution time
    double getExecutionTimeMs() {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        return duration.count() / 1000.0;  // Convert to milliseconds
    }
};

// Main function to benchmark the detection pipeline
int main() {
    // Initialize detection pipeline
#ifdef DEBUG_LOGGING
    std::cout << "===== Starting Detection Pipeline Benchmark (DEBUG MODE) =====" << std::endl;
#else
    std::cout << "===== Starting Detection Pipeline Benchmark (Release Mode) =====" << std::endl;
#endif
    DetectionPipeline pipeline;
    pipeline.initialize(); // Initializes PINN, fits Scaler/PCA on dummy data
    
    // Generate dummy data for one detection run
#ifdef DEBUG_LOGGING
    std::cout << "\nGenerating test data..." << std::endl;
#endif
    std::vector<std::vector<double>> test_data = pipeline.generateDummyData();
    
    // Run detection multiple times to get average performance
#ifdef DEBUG_LOGGING
    std::cout << "\nStarting benchmark runs..." << std::endl;
#endif
    const int NUM_RUNS = 100;
    double total_time = 0.0;
    std::vector<double> run_times; // Vector to store individual run times
    run_times.reserve(NUM_RUNS);
    
    for (int i = 0; i < NUM_RUNS; i++) {
        // Run detection
        bool result = pipeline.detectAnomaly(test_data);
        (void)result; // Mark result as unused in Release mode to avoid warning
        
        // Accumulate time and store individual time
        double current_run_time = pipeline.getExecutionTimeMs();
        total_time += current_run_time;
        run_times.push_back(current_run_time);
        
        if (i % 10 == 0) {
#ifdef DEBUG_LOGGING
            std::cout << "  Run " << i << ": " 
                     << (result ? "Anomaly detected" : "Normal") 
                     << ", Time: " << pipeline.getExecutionTimeMs() << " ms" << std::endl;
#endif
        }
    }
    
    // Calculate average and standard deviation
    double average_time = total_time / NUM_RUNS;
    double sum_sq_diff = 0.0;
    for(double time : run_times) {
        sum_sq_diff += std::pow(time - average_time, 2);
    }
    double std_dev = std::sqrt(sum_sq_diff / NUM_RUNS); // Population standard deviation

    // Output average time and standard deviation (always print this)
#ifdef DEBUG_LOGGING
    std::cout << "\nBenchmark finished." << std::endl;
    std::cout << "Average Execution Time: " << average_time << " ms" << std::endl;
    std::cout << "Standard Deviation: " << std_dev << " ms" << std::endl;
#else
    // Release mode: Print only the numbers, separated by a space
    std::cout << average_time << " " << std_dev << std::endl; 
#endif

#ifdef DEBUG_LOGGING
    std::cout << "===== Benchmark Complete (DEBUG MODE) =====" << std::endl;
#endif
    
    return 0;
}