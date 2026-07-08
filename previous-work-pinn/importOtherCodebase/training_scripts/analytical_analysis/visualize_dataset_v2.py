"""
=============================================================================
ENHANCED DATASET VISUALIZATION SCRIPT V2
=============================================================================
This script provides comprehensive visualization and analysis of the synthetic dataset
generated using the stable equations from baseForV2.py. It includes advanced 
visualizations, parameter space analysis, and system dynamics insights.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.fft import fft, fftfreq
from scipy.signal import welch, spectrogram
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Optional, Tuple


class DatasetLoaderV2:
    """Enhanced data loader for the V2 dataset format"""
    
    def __init__(self, data_dir: str = "training_scripts/analytical_analysis/data_v2"):
        self.data_dir = Path(data_dir)
        self.summary = self._load_summary()
    
    def _load_summary(self) -> Dict:
        """Load the generation summary"""
        summary_file = self.data_dir / "generation_summary.json"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                return json.load(f)
        return {}
    
    def get_dataset_summary(self) -> Dict:
        """Get dataset summary information"""
        return self.summary
    
    def list_simulations(self) -> List[int]:
        """List all available simulation IDs"""
        sim_dirs = [d for d in self.data_dir.iterdir() if d.is_dir() and d.name.startswith('simulation_')]
        sim_ids = [int(d.name.split('_')[1]) for d in sim_dirs]
        return sorted(sim_ids)
    
    def load_simulation(self, sim_id: int) -> Optional[Dict]:
        """Load a single simulation"""
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
        
        return {
            'metadata': metadata,
            'time': data['time'],
            'positions': data['positions'].item(),
            'velocities': data['velocities'].item(),
            'accelerations': data['accelerations'].item()
        }
    
    def load_multiple_simulations(self, max_sims: int = 10) -> List[Dict]:
        """Load multiple simulations"""
        sim_ids = self.list_simulations()[:max_sims]
        simulations = []
        
        for sim_id in sim_ids:
            sim_data = self.load_simulation(sim_id)
            if sim_data:
                simulations.append(sim_data)
        
        return simulations
    
    def create_parameter_dataframe(self) -> pd.DataFrame:
        """Create a DataFrame with all simulation parameters"""
        sim_ids = self.list_simulations()
        data_list = []
        
        for sim_id in sim_ids:
            sim_data = self.load_simulation(sim_id)
            if sim_data:
                params = sim_data['metadata']['parameters']
                row = {
                    'simulation_id': sim_id,
                    'success': True,
                    **params
                }
                data_list.append(row)
        
        return pd.DataFrame(data_list)


class AdvancedVisualizerV2:
    """Advanced visualization tools for the V2 dataset"""
    
    def __init__(self, data_dir: str = "training_scripts/analytical_analysis/data_v2"):
        self.loader = DatasetLoaderV2(data_dir)
        self.summary = self.loader.get_dataset_summary()
        plt.style.use('default')
        sns.set_palette("husl")
    
    def create_comprehensive_overview(self):
        """Create comprehensive overview of the dataset"""
        df = self.loader.create_parameter_dataframe()
        
        if df.empty:
            print("No data found!")
            return
        
        print("Dataset Summary:")
        print(json.dumps(self.summary, indent=2, default=str))
        print(f"\nDataset shape: {df.shape}")
        print(f"Success rate: {df['success'].mean():.2%}")
        
        # Create main overview figure
        fig = plt.figure(figsize=(20, 16))
        
        # Parameter distributions
        self._plot_parameter_distributions(df, fig)
        
        # Parameter correlations
        self._plot_parameter_correlations(df, fig)
        
        # Parameter ranges
        self._plot_parameter_ranges(df, fig)
        
        plt.tight_layout()
        plt.suptitle(f'Dataset Overview - {self.summary.get("conservativeness_level", "Unknown")} Level', 
                     y=0.98, fontsize=16, fontweight='bold')
        plt.show()
        
        # Print parameter statistics
        self._print_parameter_statistics(df)
    
    def _plot_parameter_distributions(self, df: pd.DataFrame, fig):
        """Plot parameter distributions"""
        params = ['M1', 'Omega', 'Ks1', 'mu_eps']
        labels = ['M1 (kg)', 'Omega (rad/s)', 'Ks1 (N/m)', 'mu_eps (kg·m)']
        
        for i, (param, label) in enumerate(zip(params, labels)):
            ax = plt.subplot(4, 5, i + 1)
            if param in df.columns:
                df[param].hist(bins=25, alpha=0.7, ax=ax, color=f'C{i}', edgecolor='black')
                ax.set_title(f'{param} Distribution', fontweight='bold')
                ax.set_xlabel(label)
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
    
    def _plot_parameter_correlations(self, df: pd.DataFrame, fig):
        """Plot parameter correlations"""
        correlations = [
            ('M1', 'Omega', 'Mass vs Speed'),
            ('Ks1', 'Ds1', 'Stiffness vs Damping'),
            ('Omega', 'mu_eps', 'Speed vs Unbalance'),
            ('Kb', 'Db', 'Bearing Stiffness vs Damping')
        ]
        
        for i, (param1, param2, title) in enumerate(correlations):
            ax = plt.subplot(4, 5, i + 6)
            if param1 in df.columns and param2 in df.columns:
                ax.scatter(df[param1], df[param2], alpha=0.6, s=20, c=f'C{i+4}')
                ax.set_xlabel(param1)
                ax.set_ylabel(param2)
                ax.set_title(title, fontweight='bold')
                ax.grid(True, alpha=0.3)
    
    def _plot_parameter_ranges(self, df: pd.DataFrame, fig):
        """Plot parameter ranges using box plots"""
        param_groups = [
            (['M1', 'M2', 'M3'], 'Mass Parameters (kg)'),
            (['Ks1', 'Ks2'], 'Stiffness Parameters (N/m)'),
            (['Ds1', 'Ds2'], 'Damping Parameters (N·s/m)'),
            (['Kb', 'Db'], 'Bearing Parameters'),
            (['Omega'], 'Speed (rad/s)'),
            (['mu_eps'], 'Unbalance (kg·m)')
        ]
        
        for i, (params, title) in enumerate(param_groups):
            ax = plt.subplot(4, 5, i + 11)
            available_params = [p for p in params if p in df.columns]
            if available_params:
                param_data = [df[param] for param in available_params]
                bp = ax.boxplot(param_data, labels=available_params, patch_artist=True)
                
                # Color the boxes
                for patch, color in zip(bp['boxes'], plt.cm.tab10(np.linspace(0, 1, len(available_params)))):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                ax.set_title(title, fontweight='bold')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
    
    def _print_parameter_statistics(self, df: pd.DataFrame):
        """Print detailed parameter statistics"""
        print("\nParameter Statistics:")
        print("=" * 80)
        
        key_params = ['M1', 'M2', 'M3', 'Ks1', 'Ks2', 'Ds1', 'Ds2', 'Kb', 'Db', 'Omega', 'mu_eps']
        
        for param in key_params:
            if param in df.columns:
                stats = df[param].describe()
                print(f"{param:8s}: {stats['min']:8.2e} to {stats['max']:8.2e} "
                      f"(mean: {stats['mean']:8.2e}, std: {stats['std']:8.2e})")
    
    def plot_sample_dynamics(self, n_samples: int = 4):
        """Plot sample simulation dynamics with enhanced visualizations"""
        simulations = self.loader.load_multiple_simulations(max_sims=n_samples)
        
        if not simulations:
            print("No simulations found!")
            return
        
        fig, axes = plt.subplots(n_samples, 4, figsize=(20, 5*n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i, sim_data in enumerate(simulations):
            metadata = sim_data['metadata']
            config_id = metadata['config_id']
            
            # Plot orbits
            self._plot_orbit(sim_data, 'central_disk', axes[i, 0])
            axes[i, 0].set_title(f'Sim {config_id}: Central Disk Orbit', fontweight='bold')
            
            self._plot_orbit(sim_data, 'bearing', axes[i, 1])
            axes[i, 1].set_title(f'Sim {config_id}: Bearing Orbit', fontweight='bold')
            
            # Plot time history
            self._plot_time_history(sim_data, axes[i, 2])
            axes[i, 2].set_title(f'Sim {config_id}: Time History', fontweight='bold')
            
            # Plot frequency spectrum
            self._plot_frequency_spectrum(sim_data, axes[i, 3])
            axes[i, 3].set_title(f'Sim {config_id}: Frequency Spectrum', fontweight='bold')
            
            # Add parameter info
            params = metadata['parameters']
            info_text = f"M1: {params['M1']:.1f} kg\n"
            info_text += f"Ω: {params['Omega']:.0f} rad/s\n"
            info_text += f"με: {params['mu_eps']:.2e} kg·m"
            
            axes[i, 0].text(0.02, 0.98, info_text, transform=axes[i, 0].transAxes, 
                           verticalalignment='top', fontsize=9, 
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def _plot_orbit(self, sim_data: Dict, component: str, ax):
        """Plot orbit with enhanced styling"""
        if component == 'central_disk':
            x_data = sim_data['positions']['X1'] * 1000  # Convert to mm
            y_data = sim_data['positions']['Y1'] * 1000
        else:  # bearing
            x_data = sim_data['positions']['X2'] * 1000
            y_data = sim_data['positions']['Y2'] * 1000
        
        # Use steady-state portion (last 20% of data)
        start_idx = int(0.8 * len(x_data))
        x_steady = x_data[start_idx:]
        y_steady = y_data[start_idx:]
        
        # Plot orbit with color gradient
        ax.plot(x_steady, y_steady, alpha=0.7, linewidth=1.5)
        ax.scatter(x_steady[0], y_steady[0], color='green', s=30, marker='o', label='Start', zorder=5)
        ax.scatter(x_steady[-1], y_steady[-1], color='red', s=30, marker='s', label='End', zorder=5)
        
        ax.set_xlabel('X Displacement (mm)')
        ax.set_ylabel('Y Displacement (mm)')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    def _plot_time_history(self, sim_data: Dict, ax):
        """Plot time history with multiple components"""
        time = sim_data['time']
        
        # Plot central disk displacements
        ax.plot(time, sim_data['positions']['X1'] * 1000, label='X1', linewidth=1.5, alpha=0.8)
        ax.plot(time, sim_data['positions']['Y1'] * 1000, label='Y1', linewidth=1.5, alpha=0.8)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Displacement (mm)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Focus on steady-state region
        if len(time) > 1000:
            start_idx = int(0.8 * len(time))
            ax.set_xlim(time[start_idx], time[-1])
    
    def _plot_frequency_spectrum(self, sim_data: Dict, ax):
        """Plot frequency spectrum with enhanced analysis"""
        time = sim_data['time']
        signal = sim_data['accelerations']['Y1_ddot']  # Use Y acceleration
        
        # Use steady-state portion
        start_idx = int(0.8 * len(signal))
        time_steady = time[start_idx:]
        signal_steady = signal[start_idx:]
        
        if len(time_steady) > 1:
            # Calculate sampling frequency
            dt = time_steady[1] - time_steady[0]
            fs = 1.0 / dt
            
            # Compute FFT
            fft_vals = fft(signal_steady * np.hanning(len(signal_steady)))
            freqs = fftfreq(len(signal_steady), dt)
            
            # Take positive frequencies
            pos_idx = freqs > 0
            freqs_pos = freqs[pos_idx]
            magnitude = 2.0 * np.abs(fft_vals[pos_idx]) / len(signal_steady)
            
            # Plot spectrum
            ax.semilogy(freqs_pos, magnitude, linewidth=1.5)
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Magnitude')
            ax.grid(True, alpha=0.3)
            
            # Mark rotational frequency
            omega = sim_data['metadata']['parameters']['Omega']
            rot_freq = omega / (2 * np.pi)
            ax.axvline(rot_freq, color='r', linestyle='--', alpha=0.7, 
                      label=f'1X ({rot_freq:.1f} Hz)')
            ax.axvline(2*rot_freq, color='g', linestyle='--', alpha=0.7, 
                      label=f'2X ({2*rot_freq:.1f} Hz)')
            ax.legend(fontsize=8)
            
            # Limit frequency range for better visualization
            ax.set_xlim(0, min(500, fs/2))
    
    def create_parameter_correlation_analysis(self):
        """Create detailed parameter correlation analysis"""
        df = self.loader.create_parameter_dataframe()
        
        if df.empty:
            print("No data found!")
            return
        
        # Select numeric parameters
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        param_cols = [col for col in numeric_cols if col not in ['simulation_id', 'success']]
        
        # Create correlation matrix
        correlation_matrix = df[param_cols].corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='RdBu_r', 
                   center=0,
                   square=True, 
                   fmt='.2f',
                   cbar_kws={'label': 'Correlation Coefficient'})
        
        plt.title('Parameter Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Print high correlations
        print("\nHigh Correlations (|r| > 0.5):")
        print("=" * 50)
        for i in range(len(param_cols)):
            for j in range(i+1, len(param_cols)):
                corr = correlation_matrix.iloc[i, j]
                if abs(corr) > 0.5:
                    print(f"{param_cols[i]:10s} - {param_cols[j]:10s}: {corr:6.3f}")
    
    def create_comprehensive_histograms(self):
        """Create comprehensive histogram visualizations for all variables and parameters"""
        print("\n" + "="*60)
        print("CREATING COMPREHENSIVE HISTOGRAM ANALYSIS")
        print("="*60)
        
        # 1. Parameter histograms
        self._create_parameter_histograms()
        
        # 2. Variable histograms (positions, velocities, accelerations)
        self._create_variable_histograms()
        
        # 3. Statistical distribution analysis
        self._create_distribution_analysis()
    
    def _create_parameter_histograms(self):
        """Create histograms for all system parameters"""
        df = self.loader.create_parameter_dataframe()
        
        if df.empty:
            print("No data found!")
            return
        
        # Define parameter groups
        param_groups = {
            'Mass Parameters (kg)': ['M1', 'M2', 'M3'],
            'Stiffness Parameters (N/m)': ['Ks1', 'Ks2', 'Kb'],
            'Damping Parameters (N·s/m)': ['Ds1', 'Ds2', 'Db'],
            'Operational Parameters': ['Omega', 'mu_eps', 'c'],
            'Bearing Parameters': ['Kb_nl', 'g']
        }
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        axes = axes.flatten()
        
        plot_idx = 0
        all_params = []
        for group_name, params in param_groups.items():
            all_params.extend(params)
        
        # Create individual parameter histograms
        for param in all_params[:9]:  # Limit to 9 for the 3x3 grid
            if param in df.columns and plot_idx < 9:
                ax = axes[plot_idx]
                
                # Create histogram with enhanced styling
                n, bins, patches = ax.hist(df[param], bins=25, alpha=0.7, 
                                         edgecolor='black', linewidth=0.5)
                
                # Color the bars with a gradient
                colors = plt.cm.viridis(np.linspace(0, 1, len(patches)))
                for patch, color in zip(patches, colors):
                    patch.set_facecolor(color)
                
                # Add statistics
                mean_val = df[param].mean()
                std_val = df[param].std()
                ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                          label=f'Mean: {mean_val:.2e}')
                ax.axvline(mean_val + std_val, color='orange', linestyle=':', 
                          alpha=0.8, label=f'±1σ')
                ax.axvline(mean_val - std_val, color='orange', linestyle=':', alpha=0.8)
                
                ax.set_title(f'{param} Distribution', fontweight='bold')
                ax.set_xlabel(f'{param}')
                ax.set_ylabel('Frequency')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                
                plot_idx += 1
        
        # Hide unused subplots
        for idx in range(plot_idx, 9):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.suptitle('Parameter Distribution Histograms', fontsize=16, fontweight='bold', y=0.98)
        plt.show()
    
    def _create_variable_histograms(self):
        """Create histograms for all system variables (positions, velocities, accelerations)"""
        # Load sample of simulations for variable analysis
        sim_ids = self.loader.list_simulations()[:10]  # Use first 10 simulations
        
        if not sim_ids:
            print("No simulations found!")
            return
        
        # Collect data from all simulations
        all_data = {
            'positions': {'X1': [], 'Y1': [], 'X2': [], 'Y2': [], 'X3': [], 'Y3': []},
            'velocities': {'X1_dot': [], 'Y1_dot': [], 'X2_dot': [], 'Y2_dot': [], 'X3_dot': [], 'Y3_dot': []},
            'accelerations': {'X1_ddot': [], 'Y1_ddot': [], 'X2_ddot': [], 'Y2_ddot': [], 'X3_ddot': [], 'Y3_ddot': []}
        }
        
        print("Collecting variable data from simulations...")
        for sim_id in sim_ids:
            sim_data = self.loader.load_simulation(sim_id)
            if sim_data:
                # Use steady-state portion
                start_idx = int(0.8 * len(sim_data['time']))
                
                # Collect positions (convert to mm)
                for var in ['X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3']:
                    data = sim_data['positions'][var][start_idx:] * 1000  # Convert to mm
                    all_data['positions'][var].extend(data)
                
                # Collect velocities (convert to mm/s)
                for var in ['X1_dot', 'Y1_dot', 'X2_dot', 'Y2_dot', 'X3_dot', 'Y3_dot']:
                    data = sim_data['velocities'][var][start_idx:] * 1000  # Convert to mm/s
                    all_data['velocities'][var].extend(data)
                
                # Collect accelerations (keep in m/s²)
                for var in ['X1_ddot', 'Y1_ddot', 'X2_ddot', 'Y2_ddot', 'X3_ddot', 'Y3_ddot']:
                    data = sim_data['accelerations'][var][start_idx:]
                    all_data['accelerations'][var].extend(data)
        
        # Create histograms for each variable type
        self._plot_variable_type_histograms(all_data, 'positions', 'Displacement (mm)')
        self._plot_variable_type_histograms(all_data, 'velocities', 'Velocity (mm/s)')
        self._plot_variable_type_histograms(all_data, 'accelerations', 'Acceleration (m/s²)')
    
    def _plot_variable_type_histograms(self, all_data: Dict, var_type: str, ylabel: str):
        """Plot histograms for a specific variable type"""
        data = all_data[var_type]
        variables = list(data.keys())
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, var in enumerate(variables):
            if i < 6 and data[var]:  # Ensure we have data
                ax = axes[i]
                
                # Create histogram
                values = np.array(data[var])
                n, bins, patches = ax.hist(values, bins=50, alpha=0.7, 
                                         edgecolor='black', linewidth=0.5)
                
                # Color the bars
                colors = plt.cm.plasma(np.linspace(0, 1, len(patches)))
                for patch, color in zip(patches, colors):
                    patch.set_facecolor(color)
                
                # Add statistics
                mean_val = np.mean(values)
                std_val = np.std(values)
                median_val = np.median(values)
                
                ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                          label=f'Mean: {mean_val:.3f}')
                ax.axvline(median_val, color='blue', linestyle='--', linewidth=2, 
                          label=f'Median: {median_val:.3f}')
                ax.axvline(mean_val + std_val, color='orange', linestyle=':', alpha=0.8)
                ax.axvline(mean_val - std_val, color='orange', linestyle=':', alpha=0.8)
                
                # Component and coordinate info
                component = 'Central Disk' if '1' in var else ('Left Bearing' if '2' in var else 'Right Bearing')
                direction = 'X' if 'X' in var else 'Y'
                
                ax.set_title(f'{component} - {direction} {var_type.title()[:-1]}', fontweight='bold')
                ax.set_xlabel(ylabel)
                ax.set_ylabel('Frequency')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle(f'{var_type.title()} Distribution Histograms', fontsize=16, fontweight='bold', y=0.98)
        plt.show()
    
    def _create_distribution_analysis(self):
        """Create statistical distribution analysis"""
        df = self.loader.create_parameter_dataframe()
        
        if df.empty:
            return
        
        # Select key parameters for distribution analysis
        key_params = ['M1', 'Omega', 'Ks1', 'mu_eps', 'Kb', 'Db']
        available_params = [p for p in key_params if p in df.columns]
        
        if not available_params:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        from scipy import stats
        
        for i, param in enumerate(available_params[:6]):
            ax = axes[i]
            
            data = df[param].values
            
            # Create histogram
            n, bins, patches = ax.hist(data, bins=30, density=True, alpha=0.7, 
                                     color='skyblue', edgecolor='black')
            
            # Fit normal distribution
            mu, sigma = stats.norm.fit(data)
            x = np.linspace(data.min(), data.max(), 100)
            normal_pdf = stats.norm.pdf(x, mu, sigma)
            ax.plot(x, normal_pdf, 'r-', linewidth=2, label=f'Normal fit (μ={mu:.2e}, σ={sigma:.2e})')
            
            # Add distribution statistics
            skewness = stats.skew(data)
            kurtosis = stats.kurtosis(data)
            
            ax.set_title(f'{param} Distribution Analysis', fontweight='bold')
            ax.set_xlabel(param)
            ax.set_ylabel('Probability Density')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Add statistics text box
            stats_text = f'Skewness: {skewness:.3f}\nKurtosis: {kurtosis:.3f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hide unused subplots
        for idx in range(len(available_params), 6):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.suptitle('Statistical Distribution Analysis', fontsize=16, fontweight='bold', y=0.98)
        plt.show()
    
    def create_system_dynamics_analysis(self, sim_ids: List[int] = None):
        """Create comprehensive system dynamics analysis"""
        if sim_ids is None:
            sim_ids = self.loader.list_simulations()[:6]  # Analyze first 6 simulations
        
        fig = plt.figure(figsize=(18, 12))
        
        # 3D orbit plot
        ax1 = plt.subplot(2, 3, 1, projection='3d')
        self._plot_3d_orbits(sim_ids[:4], ax1)
        
        # Waterfall plot
        ax2 = plt.subplot(2, 3, 2)
        self._plot_waterfall_spectrum(sim_ids, ax2)
        
        # Phase portraits
        ax3 = plt.subplot(2, 3, 3)
        self._plot_phase_portraits(sim_ids[:3], ax3)
        
        # Poincaré maps
        ax4 = plt.subplot(2, 3, 4)
        self._plot_poincare_maps(sim_ids[:3], ax4)
        
        # Energy analysis
        ax5 = plt.subplot(2, 3, 5)
        self._plot_energy_analysis(sim_ids[:3], ax5)
        
        # Stability analysis
        ax6 = plt.subplot(2, 3, 6)
        self._plot_stability_analysis(sim_ids, ax6)
        
        plt.tight_layout()
        plt.suptitle('Advanced System Dynamics Analysis', fontsize=16, fontweight='bold', y=0.98)
        plt.show()
    
    def _plot_3d_orbits(self, sim_ids: List[int], ax):
        """Plot 3D orbits for multiple simulations"""
        colors = plt.cm.tab10(np.linspace(0, 1, len(sim_ids)))
        
        for i, sim_id in enumerate(sim_ids):
            sim_data = self.loader.load_simulation(sim_id)
            if sim_data:
                # Use steady-state data
                start_idx = int(0.8 * len(sim_data['time']))
                x = sim_data['positions']['X1'][start_idx:] * 1000
                y = sim_data['positions']['Y1'][start_idx:] * 1000
                z = sim_data['time'][start_idx:]
                
                ax.plot(x, y, z, color=colors[i], alpha=0.7, linewidth=1.5, 
                       label=f'Sim {sim_id}')
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Time (s)')
        ax.set_title('3D Orbital Evolution', fontweight='bold')
        ax.legend()
    
    def _plot_waterfall_spectrum(self, sim_ids: List[int], ax):
        """Plot waterfall spectrum for multiple simulations"""
        spectra_data = []  # Store both frequencies and spectra
        omegas = []
        
        for sim_id in sim_ids:
            sim_data = self.loader.load_simulation(sim_id)
            if sim_data:
                # Calculate spectrum
                time = sim_data['time']
                signal = sim_data['accelerations']['Y1_ddot']
                
                start_idx = int(0.8 * len(signal))
                signal_steady = signal[start_idx:]
                time_steady = time[start_idx:]
                
                if len(time_steady) > 1:
                    dt = time_steady[1] - time_steady[0]
                    freqs, psd = welch(signal_steady, fs=1/dt, nperseg=min(1024, len(signal_steady)//4))
                    
                    # Limit frequency range
                    freq_mask = freqs <= 500
                    freqs_limited = freqs[freq_mask]
                    psd_limited = psd[freq_mask]
                    
                    spectra_data.append((freqs_limited, psd_limited))
                    omegas.append(sim_data['metadata']['parameters']['Omega'])
        
        if spectra_data:
            for i, ((freqs_limited, spectrum), omega) in enumerate(zip(spectra_data, omegas)):
                # Add small offset to separate curves vertically
                offset_spectrum = spectrum + i * np.max(spectrum) * 0.1
                ax.semilogy(freqs_limited, offset_spectrum, alpha=0.7, 
                           label=f'Ω={omega:.0f}', linewidth=1.5)
        
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD (with offset)')
        ax.set_title('Waterfall Spectrum', fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_phase_portraits(self, sim_ids: List[int], ax):
        """Plot phase portraits"""
        for i, sim_id in enumerate(sim_ids):
            sim_data = self.loader.load_simulation(sim_id)
            if sim_data:
                start_idx = int(0.8 * len(sim_data['time']))
                x = sim_data['positions']['X1'][start_idx:] * 1000
                vx = sim_data['velocities']['X1_dot'][start_idx:] * 1000
                
                ax.plot(x, vx, alpha=0.7, linewidth=1.5, label=f'Sim {sim_id}')
        
        ax.set_xlabel('X Position (mm)')
        ax.set_ylabel('X Velocity (mm/s)')
        ax.set_title('Phase Portraits', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_poincare_maps(self, sim_ids: List[int], ax):
        """Plot Poincaré maps"""
        for i, sim_id in enumerate(sim_ids):
            sim_data = self.loader.load_simulation(sim_id)
            if sim_data:
                # Sample at rotational frequency
                omega = sim_data['metadata']['parameters']['Omega']
                period = 2 * np.pi / omega
                
                time = sim_data['time']
                x = sim_data['positions']['X1'] * 1000
                y = sim_data['positions']['Y1'] * 1000
                
                # Find Poincaré section points
                section_times = np.arange(time[0] + 10*period, time[-1], period)
                section_x = np.interp(section_times, time, x)
                section_y = np.interp(section_times, time, y)
                
                ax.scatter(section_x, section_y, alpha=0.7, s=10, label=f'Sim {sim_id}')
        
        ax.set_xlabel('X Position (mm)')
        ax.set_ylabel('Y Position (mm)')
        ax.set_title('Poincaré Maps', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_energy_analysis(self, sim_ids: List[int], ax):
        """Plot energy analysis"""
        for i, sim_id in enumerate(sim_ids):
            sim_data = self.loader.load_simulation(sim_id)
            if sim_data:
                time = sim_data['time']
                params = sim_data['metadata']['parameters']
                
                # Calculate kinetic energy
                vx1 = sim_data['velocities']['X1_dot']
                vy1 = sim_data['velocities']['Y1_dot']
                ke = 0.5 * params['M1'] * (vx1**2 + vy1**2)
                
                # Use steady-state portion
                start_idx = int(0.8 * len(time))
                time_steady = time[start_idx:]
                ke_steady = ke[start_idx:]
                
                ax.plot(time_steady, ke_steady, alpha=0.7, linewidth=1.5, 
                       label=f'Sim {sim_id}')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Kinetic Energy (J)')
        ax.set_title('Energy Analysis', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_stability_analysis(self, sim_ids: List[int], ax):
        """Plot stability analysis"""
        stability_metrics = []
        omega_values = []
        
        for sim_id in sim_ids:
            sim_data = self.loader.load_simulation(sim_id)
            if sim_data:
                # Calculate stability metric (RMS displacement)
                start_idx = int(0.8 * len(sim_data['time']))
                x = sim_data['positions']['X1'][start_idx:]
                y = sim_data['positions']['Y1'][start_idx:]
                
                rms_displacement = np.sqrt(np.mean(x**2 + y**2)) * 1000  # mm
                stability_metrics.append(rms_displacement)
                omega_values.append(sim_data['metadata']['parameters']['Omega'])
        
        if stability_metrics:
            ax.scatter(omega_values, stability_metrics, alpha=0.7, s=30)
            ax.set_xlabel('Rotational Speed (rad/s)')
            ax.set_ylabel('RMS Displacement (mm)')
            ax.set_title('Stability Analysis', fontweight='bold')
            ax.grid(True, alpha=0.3)


def main():
    """Main function to run all visualizations"""
    
    print("="*60)
    print("ENHANCED DATASET VISUALIZATION V2")
    print("="*60)
    
    # Initialize visualizer
    visualizer = AdvancedVisualizerV2()
    
    # 1. Comprehensive overview
    print("\n1. Creating comprehensive dataset overview...")
    visualizer.create_comprehensive_overview()
    
    # 2. Sample dynamics
    print("\n2. Plotting sample simulation dynamics...")
    visualizer.plot_sample_dynamics(n_samples=4)
    
    # 3. Parameter correlation analysis
    print("\n3. Creating parameter correlation analysis...")
    visualizer.create_parameter_correlation_analysis()
    
    # 4. Comprehensive histogram analysis
    print("\n4. Creating comprehensive histogram analysis...")
    visualizer.create_comprehensive_histograms()
    
    # 5. Advanced system dynamics analysis
    print("\n5. Creating advanced system dynamics analysis...")
    visualizer.create_system_dynamics_analysis()
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()