"""
=============================================================================
DATA LOADER V2 FOR ROTOR DYNAMICS SYNTHETIC DATASET
=============================================================================
This module provides data loading utilities for the synthetic dataset
generated using the stable equations from baseForV2.py.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt


class SyntheticDatasetLoaderV2:
    """
    Data loader for the V2 synthetic rotor dynamics dataset.
    Provides utilities for loading, filtering, and basic analysis.
    """
    
    def __init__(self, data_dir: str = "training_scripts/analytical_analysis/data_v2"):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Path to the dataset directory
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise ValueError(f"Dataset directory {data_dir} does not exist")
        
        self.summary = self._load_summary()
        self._simulation_cache = {}
    
    def _load_summary(self) -> Dict:
        """Load the generation summary"""
        summary_file = self.data_dir / "generation_summary.json"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                return json.load(f)
        return {}
    
    def get_dataset_info(self) -> Dict:
        """Get comprehensive dataset information"""
        info = {
            'dataset_path': str(self.data_dir.absolute()),
            'generation_summary': self.summary,
            'available_simulations': len(self.list_simulations()),
            'dataset_size_gb': self._calculate_dataset_size()
        }
        return info
    
    def _calculate_dataset_size(self) -> float:
        """Calculate total dataset size in GB"""
        total_size = 0
        for file_path in self.data_dir.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size / (1024**3)  # Convert to GB
    
    def list_simulations(self) -> List[int]:
        """List all available simulation IDs"""
        sim_dirs = [d for d in self.data_dir.iterdir() 
                   if d.is_dir() and d.name.startswith('simulation_')]
        sim_ids = [int(d.name.split('_')[1]) for d in sim_dirs]
        return sorted(sim_ids)
    
    def load_simulation(self, sim_id: int, use_cache: bool = True) -> Optional[Dict]:
        """
        Load a single simulation.
        
        Args:
            sim_id: Simulation ID
            use_cache: Whether to use cached data
            
        Returns:
            Dictionary containing simulation data or None if not found
        """
        if use_cache and sim_id in self._simulation_cache:
            return self._simulation_cache[sim_id]
        
        sim_dir = self.data_dir / f"simulation_{sim_id:06d}"
        
        if not sim_dir.exists():
            return None
        
        # Load metadata
        metadata_file = sim_dir / "metadata.json"
        if not metadata_file.exists():
            return None
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Load time series data
        data_file = sim_dir / "time_series.npz"
        if not data_file.exists():
            return None
        
        data = np.load(data_file, allow_pickle=True)
        
        simulation_data = {
            'metadata': metadata,
            'time': data['time'],
            'positions': data['positions'].item(),
            'velocities': data['velocities'].item(),
            'accelerations': data['accelerations'].item()
        }
        
        if use_cache:
            self._simulation_cache[sim_id] = simulation_data
        
        return simulation_data
    
    def load_multiple_simulations(self, sim_ids: List[int] = None, 
                                 max_sims: int = None) -> List[Dict]:
        """
        Load multiple simulations.
        
        Args:
            sim_ids: Specific simulation IDs to load. If None, loads all available.
            max_sims: Maximum number of simulations to load
            
        Returns:
            List of simulation data dictionaries
        """
        if sim_ids is None:
            sim_ids = self.list_simulations()
        
        if max_sims is not None:
            sim_ids = sim_ids[:max_sims]
        
        simulations = []
        for sim_id in sim_ids:
            sim_data = self.load_simulation(sim_id)
            if sim_data:
                simulations.append(sim_data)
        
        return simulations
    
    def create_parameter_dataframe(self) -> pd.DataFrame:
        """
        Create a pandas DataFrame with all simulation parameters.
        
        Returns:
            DataFrame with simulation parameters and metadata
        """
        sim_ids = self.list_simulations()
        data_list = []
        
        for sim_id in sim_ids:
            sim_data = self.load_simulation(sim_id)
            if sim_data:
                params = sim_data['metadata']['parameters']
                sim_info = sim_data['metadata'].get('simulation_info', {})
                
                row = {
                    'simulation_id': sim_id,
                    'success': True,
                    'omega_rad_s': sim_data['metadata'].get('omega_rad_s', params.get('Omega')),
                    'total_time': sim_info.get('total_time', 0),
                    'time_points': sim_info.get('time_points', 0),
                    'sampling_rate': sim_info.get('sampling_rate', 0),
                    **params
                }
                data_list.append(row)
        
        return pd.DataFrame(data_list)
    
    def filter_simulations(self, **criteria) -> List[int]:
        """
        Filter simulations based on parameter criteria.
        
        Args:
            **criteria: Parameter criteria (e.g., M1=(10, 20) for range, Omega=800 for exact)
            
        Returns:
            List of simulation IDs matching criteria
        """
        df = self.create_parameter_dataframe()
        mask = pd.Series([True] * len(df))
        
        for param, value in criteria.items():
            if param not in df.columns:
                continue
            
            if isinstance(value, tuple) and len(value) == 2:
                # Range filter
                mask &= (df[param] >= value[0]) & (df[param] <= value[1])
            else:
                # Exact or approximate match
                if isinstance(value, (int, float)):
                    # For numeric values, use small tolerance
                    tolerance = abs(value) * 0.01 if value != 0 else 1e-10
                    mask &= np.abs(df[param] - value) <= tolerance
                else:
                    # Exact match for non-numeric
                    mask &= df[param] == value
        
        return df[mask]['simulation_id'].tolist()
    
    def get_parameter_statistics(self) -> pd.DataFrame:
        """
        Get statistical summary of all parameters.
        
        Returns:
            DataFrame with parameter statistics
        """
        df = self.create_parameter_dataframe()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        param_cols = [col for col in numeric_cols 
                     if col not in ['simulation_id', 'success', 'time_points']]
        
        return df[param_cols].describe()
    
    def extract_features(self, sim_id: int, feature_type: str = 'statistical') -> Dict:
        """
        Extract features from a simulation.
        
        Args:
            sim_id: Simulation ID
            feature_type: Type of features ('statistical', 'frequency', 'all')
            
        Returns:
            Dictionary of extracted features
        """
        sim_data = self.load_simulation(sim_id)
        if not sim_data:
            return {}
        
        features = {'simulation_id': sim_id}
        
        # Use steady-state portion (last 20% of simulation)
        time = sim_data['time']
        start_idx = int(0.8 * len(time))
        
        if feature_type in ['statistical', 'all']:
            features.update(self._extract_statistical_features(sim_data, start_idx))
        
        if feature_type in ['frequency', 'all']:
            features.update(self._extract_frequency_features(sim_data, start_idx))
        
        return features
    
    def _extract_statistical_features(self, sim_data: Dict, start_idx: int) -> Dict:
        """Extract statistical features from simulation data"""
        features = {}
        
        components = ['X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3']
        data_types = ['positions', 'velocities', 'accelerations']
        
        for data_type in data_types:
            for comp in components:
                if comp in sim_data[data_type]:
                    data = sim_data[data_type][comp][start_idx:]
                    prefix = f"{data_type[:-1]}_{comp}"  # Remove 's' from positions/velocities
                    
                    features.update({
                        f"{prefix}_mean": np.mean(data),
                        f"{prefix}_std": np.std(data),
                        f"{prefix}_max": np.max(data),
                        f"{prefix}_min": np.min(data),
                        f"{prefix}_rms": np.sqrt(np.mean(data**2)),
                        f"{prefix}_peak_to_peak": np.ptp(data)
                    })
        
        return features
    
    def _extract_frequency_features(self, sim_data: Dict, start_idx: int) -> Dict:
        """Extract frequency domain features from simulation data"""
        from scipy.fft import fft, fftfreq
        
        features = {}
        
        time = sim_data['time'][start_idx:]
        if len(time) < 2:
            return features
        
        dt = time[1] - time[0]
        fs = 1.0 / dt
        omega = sim_data['metadata']['parameters']['Omega']
        rot_freq = omega / (2 * np.pi)
        
        # Analyze accelerations (most informative for machinery)
        for comp in ['X1_ddot', 'Y1_ddot', 'X2_ddot', 'Y2_ddot']:
            comp_key = comp.replace('_ddot', '')
            if comp_key in sim_data['accelerations']:
                signal = sim_data['accelerations'][comp_key][start_idx:]
                
                # Compute FFT
                fft_vals = fft(signal * np.hanning(len(signal)))
                freqs = fftfreq(len(signal), dt)
                
                # Take positive frequencies
                pos_idx = freqs > 0
                freqs_pos = freqs[pos_idx]
                magnitude = 2.0 * np.abs(fft_vals[pos_idx]) / len(signal)
                
                # Extract features at specific frequencies
                # 1X component (rotational frequency)
                idx_1x = np.argmin(np.abs(freqs_pos - rot_freq))
                features[f"{comp}_1X_amplitude"] = magnitude[idx_1x] if idx_1x < len(magnitude) else 0
                
                # 2X component
                idx_2x = np.argmin(np.abs(freqs_pos - 2*rot_freq))
                features[f"{comp}_2X_amplitude"] = magnitude[idx_2x] if idx_2x < len(magnitude) else 0
                
                # Total energy
                features[f"{comp}_total_energy"] = np.sum(magnitude**2)
                
                # Dominant frequency
                max_idx = np.argmax(magnitude)
                features[f"{comp}_dominant_freq"] = freqs_pos[max_idx] if max_idx < len(freqs_pos) else 0
                features[f"{comp}_dominant_amplitude"] = magnitude[max_idx] if max_idx < len(magnitude) else 0
        
        return features
    
    def create_feature_matrix(self, sim_ids: List[int] = None, 
                            feature_type: str = 'statistical') -> pd.DataFrame:
        """
        Create a feature matrix for machine learning.
        
        Args:
            sim_ids: Simulation IDs to include. If None, uses all available.
            feature_type: Type of features to extract
            
        Returns:
            DataFrame with features for each simulation
        """
        if sim_ids is None:
            sim_ids = self.list_simulations()
        
        feature_list = []
        for sim_id in sim_ids:
            features = self.extract_features(sim_id, feature_type)
            if features:
                feature_list.append(features)
        
        return pd.DataFrame(feature_list)
    
    def plot_simulation_summary(self, sim_id: int, save_path: str = None):
        """
        Create a summary plot for a single simulation.
        
        Args:
            sim_id: Simulation ID
            save_path: Optional path to save the plot
        """
        sim_data = self.load_simulation(sim_id)
        if not sim_data:
            print(f"Simulation {sim_id} not found")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot central disk orbit
        x1 = sim_data['positions']['X1'] * 1000  # Convert to mm
        y1 = sim_data['positions']['Y1'] * 1000
        axes[0, 0].plot(x1, y1, alpha=0.7, linewidth=1.5)
        axes[0, 0].set_title('Central Disk Orbit')
        axes[0, 0].set_xlabel('X Position (mm)')
        axes[0, 0].set_ylabel('Y Position (mm)')
        axes[0, 0].axis('equal')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot bearing orbit
        x2 = sim_data['positions']['X2'] * 1000
        y2 = sim_data['positions']['Y2'] * 1000
        axes[0, 1].plot(x2, y2, alpha=0.7, linewidth=1.5, color='orange')
        axes[0, 1].set_title('Left Bearing Orbit')
        axes[0, 1].set_xlabel('X Position (mm)')
        axes[0, 1].set_ylabel('Y Position (mm)')
        axes[0, 1].axis('equal')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot time history
        time = sim_data['time']
        axes[0, 2].plot(time, x1, label='X1', linewidth=1.5)
        axes[0, 2].plot(time, y1, label='Y1', linewidth=1.5, alpha=0.8)
        axes[0, 2].set_title('Central Disk Displacement')
        axes[0, 2].set_xlabel('Time (s)')
        axes[0, 2].set_ylabel('Displacement (mm)')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot acceleration time history
        ax1 = sim_data['accelerations']['X1'] if 'X1' in sim_data['accelerations'] else sim_data['accelerations']['X1_ddot']
        ay1 = sim_data['accelerations']['Y1'] if 'Y1' in sim_data['accelerations'] else sim_data['accelerations']['Y1_ddot']
        axes[1, 0].plot(time, ax1, label='Ax1', linewidth=1.5)
        axes[1, 0].plot(time, ay1, label='Ay1', linewidth=1.5, alpha=0.8)
        axes[1, 0].set_title('Central Disk Acceleration')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Acceleration (m/s²)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot frequency spectrum
        from scipy.fft import fft, fftfreq
        
        # Use steady-state portion
        start_idx = int(0.8 * len(time))
        time_steady = time[start_idx:]
        signal = ay1[start_idx:]
        
        if len(time_steady) > 1:
            dt = time_steady[1] - time_steady[0]
            fft_vals = fft(signal * np.hanning(len(signal)))
            freqs = fftfreq(len(signal), dt)
            
            pos_idx = freqs > 0
            freqs_pos = freqs[pos_idx]
            magnitude = 2.0 * np.abs(fft_vals[pos_idx]) / len(signal)
            
            axes[1, 1].semilogy(freqs_pos, magnitude, linewidth=1.5)
            axes[1, 1].set_xlabel('Frequency (Hz)')
            axes[1, 1].set_ylabel('Magnitude')
            axes[1, 1].set_title('Acceleration Spectrum')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_xlim(0, min(500, np.max(freqs_pos)))
        
        # Add parameter information
        params = sim_data['metadata']['parameters']
        param_text = f"Parameters:\n"
        param_text += f"M1: {params['M1']:.2f} kg\n"
        param_text += f"Ω: {params['Omega']:.0f} rad/s\n"
        param_text += f"Ks1: {params['Ks1']:.1e} N/m\n"
        param_text += f"μϵ: {params['mu_eps']:.1e} kg·m"
        
        axes[1, 2].text(0.1, 0.9, param_text, transform=axes[1, 2].transAxes,
                        verticalalignment='top', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[1, 2].set_title(f'Simulation {sim_id} Info')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()


# Convenience function for quick dataset loading
def load_dataset(data_dir: str = "training_scripts/analytical_analysis/data_v2") -> SyntheticDatasetLoaderV2:
    """
    Convenience function to load the dataset.
    
    Args:
        data_dir: Path to dataset directory
        
    Returns:
        Initialized dataset loader
    """
    return SyntheticDatasetLoaderV2(data_dir)


if __name__ == "__main__":
    # Example usage
    loader = load_dataset()
    
    print("Dataset Info:")
    info = loader.get_dataset_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print(f"\nAvailable simulations: {len(loader.list_simulations())}")
    
    # Load and display first simulation
    sim_ids = loader.list_simulations()
    if sim_ids:
        print(f"\nPlotting summary for simulation {sim_ids[0]}...")
        loader.plot_simulation_summary(sim_ids[0])