"""
=============================================================================
DATA LOADER FOR SYNTHETIC ROTOR DYNAMICS DATASET
=============================================================================
This module provides utilities to load and work with the generated synthetic dataset.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

class SyntheticDatasetLoader:
    """Loader for the synthetic rotor dynamics dataset"""
    
    def __init__(self, data_dir: str = "training_scripts/analytical_analysis/data"):
        self.data_dir = Path(data_dir)
        self.summary_file = self.data_dir / "generation_summary.json"
        
        # Load summary if it exists
        if self.summary_file.exists():
            with open(self.summary_file, 'r') as f:
                self.summary = json.load(f)
        else:
            self.summary = None
    
    def get_simulation_directories(self) -> List[Path]:
        """Get list of all simulation directories"""
        if not self.data_dir.exists():
            return []
        
        sim_dirs = [
            d for d in self.data_dir.iterdir() 
            if d.is_dir() and d.name.startswith("simulation_")
        ]
        return sorted(sim_dirs)
    
    def load_simulation(self, sim_dir: Union[str, Path]) -> Optional[Dict]:
        """
        Load a single simulation from its directory
        
        Args:
            sim_dir: Path to simulation directory or simulation ID
            
        Returns:
            Dictionary containing simulation data or None if loading fails
        """
        if isinstance(sim_dir, str):
            if sim_dir.isdigit():
                # Assume it's a simulation ID
                sim_dir = self.data_dir / f"simulation_{int(sim_dir):06d}"
            else:
                sim_dir = Path(sim_dir)
        
        if not sim_dir.exists():
            print(f"Simulation directory not found: {sim_dir}")
            return None
        
        try:
            # Load metadata
            metadata_file = sim_dir / "metadata.json"
            if not metadata_file.exists():
                print(f"Metadata file not found: {metadata_file}")
                return None
            
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Load time series data
            time_series_file = sim_dir / "time_series.npz"
            if not time_series_file.exists():
                print(f"Time series file not found: {time_series_file}")
                return None
            
            time_series_data = np.load(time_series_file, allow_pickle=True)
            
            # Combine metadata and time series data
            simulation_data = {
                'metadata': metadata,
                'time': time_series_data['time'],
                'positions': {
                    'X1': time_series_data['positions'].item()['X1'],
                    'Y1': time_series_data['positions'].item()['Y1'],
                    'X2': time_series_data['positions'].item()['X2'],
                    'Y2': time_series_data['positions'].item()['Y2'],
                    'X3': time_series_data['positions'].item()['X3'],
                    'Y3': time_series_data['positions'].item()['Y3']
                },
                'velocities': {
                    'X1_dot': time_series_data['velocities'].item()['X1_dot'],
                    'Y1_dot': time_series_data['velocities'].item()['Y1_dot'],
                    'X2_dot': time_series_data['velocities'].item()['X2_dot'],
                    'Y2_dot': time_series_data['velocities'].item()['Y2_dot'],
                    'X3_dot': time_series_data['velocities'].item()['X3_dot'],
                    'Y3_dot': time_series_data['velocities'].item()['Y3_dot']
                },
                'accelerations': {
                    'X1_ddot': time_series_data['accelerations'].item()['X1_ddot'],
                    'Y1_ddot': time_series_data['accelerations'].item()['Y1_ddot'],
                    'X2_ddot': time_series_data['accelerations'].item()['X2_ddot'],
                    'Y2_ddot': time_series_data['accelerations'].item()['Y2_ddot'],
                    'X3_ddot': time_series_data['accelerations'].item()['X3_ddot'],
                    'Y3_ddot': time_series_data['accelerations'].item()['Y3_ddot']
                }
            }
            
            return simulation_data
            
        except Exception as e:
            print(f"Error loading simulation from {sim_dir}: {str(e)}")
            return None
    
    def load_multiple_simulations(self, sim_ids: Optional[List[int]] = None, max_sims: Optional[int] = None) -> List[Dict]:
        """
        Load multiple simulations
        
        Args:
            sim_ids: List of simulation IDs to load (if None, loads all)
            max_sims: Maximum number of simulations to load
            
        Returns:
            List of simulation data dictionaries
        """
        sim_dirs = self.get_simulation_directories()
        
        if sim_ids is not None:
            # Load specific simulations
            selected_dirs = []
            for sim_id in sim_ids:
                sim_dir = self.data_dir / f"simulation_{sim_id:06d}"
                if sim_dir in sim_dirs:
                    selected_dirs.append(sim_dir)
        else:
            # Load all simulations
            selected_dirs = sim_dirs
        
        if max_sims is not None:
            selected_dirs = selected_dirs[:max_sims]
        
        simulations = []
        for sim_dir in selected_dirs:
            sim_data = self.load_simulation(sim_dir)
            if sim_data is not None:
                simulations.append(sim_data)
        
        return simulations
    
    def get_dataset_summary(self) -> Dict:
        """Get summary statistics of the dataset"""
        if self.summary is None:
            return {"error": "No summary file found"}
        
        sim_dirs = self.get_simulation_directories()
        actual_simulations = len(sim_dirs)
        
        summary = self.summary.copy()
        summary['actual_simulations_found'] = actual_simulations
        
        return summary
    
    def create_parameter_dataframe(self, sim_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Create a DataFrame with parameters from multiple simulations
        
        Args:
            sim_ids: List of simulation IDs to include (if None, includes all)
            
        Returns:
            DataFrame with parameters and flags for each simulation
        """
        simulations = self.load_multiple_simulations(sim_ids)
        
        if not simulations:
            return pd.DataFrame()
        
        # Extract parameters and flags
        data = []
        for sim in simulations:
            metadata = sim['metadata']
            params = metadata['parameters']
            flags = metadata['flags']
            
            row = {
                'simulation_id': metadata['config_id'],
                'success': metadata['success'],
                'timestamp': metadata['timestamp']
            }
            
            # Add parameters
            row.update(params)
            
            # Add flags
            row.update(flags)
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def plot_simulation_orbit(self, sim_data: Dict, component: str = 'impeller', 
                            steady_state_only: bool = True, ax: Optional[Axes] = None):
        """
        Plot the orbit for a specific component
        
        Args:
            sim_data: Simulation data dictionary
            component: Component to plot ('impeller', 'bearing1', 'bearing2')
            steady_state_only: If True, only plot last 1000 points
            ax: Matplotlib axes to plot on (if None, creates new figure)
        """
        if component == 'impeller':
            x_key, y_key = 'X1', 'Y1'
            title = 'Impeller Orbit'
        elif component == 'bearing1':
            x_key, y_key = 'X2', 'Y2'
            title = 'Bearing 1 Orbit'
        elif component == 'bearing2':
            x_key, y_key = 'X3', 'Y3'
            title = 'Bearing 2 Orbit'
        else:
            raise ValueError(f"Unknown component: {component}")
        
        x_data = sim_data['positions'][x_key]
        y_data = sim_data['positions'][y_key]
        
        if steady_state_only:
            x_data = x_data[-1000:]
            y_data = y_data[-1000:]
            title += " (Steady State)"
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.plot(x_data * 1e3, y_data * 1e3, 'b-', linewidth=0.5)
        ax.set_title(title)
        ax.set_xlabel(f'{x_key} (mm)')
        ax.set_ylabel(f'{y_key} (mm)')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        return ax
    
    def plot_time_history(self, sim_data: Dict, component: str = 'impeller', 
                         variable: str = 'position', ax: Optional[Axes] = None):
        """
        Plot time history for a specific component and variable
        
        Args:
            sim_data: Simulation data dictionary
            component: Component to plot ('impeller', 'bearing1', 'bearing2')
            variable: Variable to plot ('position', 'velocity', 'acceleration')
            ax: Matplotlib axes to plot on (if None, creates new figure)
        """
        if component == 'impeller':
            prefix = '1'
        elif component == 'bearing1':
            prefix = '2'
        elif component == 'bearing2':
            prefix = '3'
        else:
            raise ValueError(f"Unknown component: {component}")
        
        if variable == 'position':
            x_key, y_key = f'X{prefix}', f'Y{prefix}'
            title = f'{component.title()} Position History'
            ylabel = 'Position (mm)'
        elif variable == 'velocity':
            x_key, y_key = f'X{prefix}_dot', f'Y{prefix}_dot'
            title = f'{component.title()} Velocity History'
            ylabel = 'Velocity (m/s)'
        elif variable == 'acceleration':
            x_key, y_key = f'X{prefix}dot', f'Y{prefix}dot'
            title = f'{component.title()} Acceleration History'
            ylabel = 'Acceleration (m/s²)'
        else:
            raise ValueError(f"Unknown variable: {variable}")
        
        time_data = sim_data['time']
        
        if variable == 'position':
            x_data = sim_data['positions'][x_key]
            y_data = sim_data['positions'][y_key]
        elif variable == 'velocity':
            x_data = sim_data['velocities'][x_key]
            y_data = sim_data['velocities'][y_key]
        else:  # acceleration
            x_data = sim_data['accelerations'][x_key]
            y_data = sim_data['accelerations'][y_key]
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(time_data, x_data * 1e3, 'b-', label=f'{x_key}', linewidth=0.5)
        ax.plot(time_data, y_data * 1e3, 'r-', label=f'{y_key}', linewidth=0.5)
        ax.set_title(title)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax


def main():
    """Example usage of the data loader"""
    loader = SyntheticDatasetLoader()
    
    # Get dataset summary
    summary = loader.get_dataset_summary()
    print("Dataset Summary:")
    print(json.dumps(summary, indent=2))
    
    # Load first few simulations
    simulations = loader.load_multiple_simulations(max_sims=3)
    print(f"\nLoaded {len(simulations)} simulations")
    
    if simulations:
        # Create parameter DataFrame
        df = loader.create_parameter_dataframe()
        print(f"\nParameter DataFrame shape: {df.shape}")
        print("\nFirst few rows:")
        print(df.head())
        
        # Plot example
        sim_data = simulations[0]
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot orbits
        loader.plot_simulation_orbit(sim_data, 'impeller', ax=axes[0, 0])
        loader.plot_simulation_orbit(sim_data, 'bearing1', ax=axes[0, 1])
        
        # Plot time histories
        loader.plot_time_history(sim_data, 'impeller', 'position', ax=axes[1, 0])
        loader.plot_time_history(sim_data, 'impeller', 'velocity', ax=axes[1, 1])
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main() 