#!/usr/bin/env python3
"""
Generate synchrosqueezed wavelet spectrograms from direct PINN residuals.

This script loads residuals from the direct PINN model and generates
detailed spectrograms visualizing the time-frequency characteristics
across different rotation speeds and conditions.
"""

import os
import argparse
import glob
from common.visualization.wavelet_spectrograms import generate_spectrograms

def find_latest_direct_residuals():
    """
    Find the latest direct PINN residuals file.
    
    Returns:
    - Path to the latest residuals file
    """
    patterns = [
        "direct_pinn_residuals_*.pth",
        "direct_results/*residuals*.pth",
        "direct_analysis_results/*residuals*.pth"
    ]
    
    all_files = []
    for pattern in patterns:
        all_files.extend(glob.glob(pattern))
    
    if not all_files:
        return None
    
    # Return the most recently modified file
    return max(all_files, key=os.path.getmtime)

def main():
    """Main function for generating spectrograms."""
    parser = argparse.ArgumentParser(description="Generate spectrograms from direct PINN residuals")
    
    parser.add_argument("--residuals", type=str, help="Path to direct PINN residuals file")
    parser.add_argument("--output-dir", type=str, help="Directory to save the spectrograms")
    parser.add_argument("--normalize-residuals", action="store_true", help="Normalize residuals before processing")
    parser.add_argument("--normalize-spectogram", action="store_true", default=True, help="Normalize spectrogram")
    
    args = parser.parse_args()
    
    # Find the latest residuals file if not provided
    residuals_path = args.residuals
    if residuals_path is None:
        residuals_path = find_latest_direct_residuals()
        if residuals_path is None:
            print("Error: No direct PINN residuals file found.")
            print("Please provide the path using the --residuals argument.")
            return
    
    # Generate output directory name if not provided
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = f"direct_spectrograms_{os.path.splitext(os.path.basename(residuals_path))[0]}"
    
    # Generate the spectrograms
    output_dir = generate_spectrograms(
        residuals_path, 
        output_dir, 
        normalize_residuals=args.normalize_residuals, 
        normalize_spectogram=args.normalize_spectogram
    )
    
    print(f"Direct PINN spectrograms saved to {output_dir}")

if __name__ == "__main__":
    main() 