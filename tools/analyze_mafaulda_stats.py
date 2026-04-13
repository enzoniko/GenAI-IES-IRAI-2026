import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path

SAMPLING_RATE = 50000

def analyze_raw_mafaulda(raw_dir):
    print(f"Analyzing RAW MaFaulDa data in: {raw_dir}")
    raw_path = Path(raw_dir)
    csv_files = list(raw_path.rglob("*.csv"))
    
    if not csv_files:
        print("No raw CSV files found.")
        return

    all_stats = []
    
    for f in csv_files:
        try:
            # Extract frequency from filename
            hz_value = float(f.stem)
            window_size = int(np.round(SAMPLING_RATE / hz_value))
            
            # Get total number of rows (samples) efficiently
            # For 250k rows, we can just read first column or count lines
            num_samples = sum(1 for line in open(f))
            
            num_windows = num_samples // window_size
            
            # Construct category from path
            rel_path = f.relative_to(raw_path).parent
            category = str(rel_path).replace('/', '_').replace('-', '_')
            if not category: category = 'root'
            
            all_stats.append({
                'category': category,
                'filename': f.name,
                'hz': hz_value,
                'window_size': window_size,
                'num_samples': num_samples,
                'num_windows': num_windows
            })
        except Exception as e:
            # Skip non-mafaulda csvs or errors
            continue

    if not all_stats:
        return

    df_stats = pd.DataFrame(all_stats)
    
    print("\n--- Raw Data Metrics by Category ---")
    category_summary = df_stats.groupby('category').agg({
        'num_windows': 'sum',
        'window_size': ['min', 'max', 'mean'],
        'hz': ['min', 'max']
    }).round(1)
    print(category_summary)

    print("\n--- Project-Wide Aggregate Statistics ---")
    print(f"Total CSV Files: {len(df_stats)}")
    print(f"Total Biological Windows: {df_stats['num_windows'].sum()}")
    print(f"Window Size Range: {df_stats['window_size'].min()} to {df_stats['window_size'].max()}")
    print(f"Frequency (Hz) Range: {df_stats['hz'].min()} to {df_stats['hz'].max()}")

    # Display window size distribution for decision making
    print("\n--- Window Size Distribution (for SEQ_LENGTH selection) ---")
    bins = [0, 1000, 2000, 3000, 4000, 5000]
    dist = pd.cut(df_stats['window_size'], bins=bins).value_counts().sort_index()
    print(dist)

if __name__ == "__main__":
    analyze_raw_mafaulda("data/raw-mafaulda")
