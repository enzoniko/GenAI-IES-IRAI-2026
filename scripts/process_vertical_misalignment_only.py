"""
Process ONLY the vertical-misalignment categories, using min_window_size=3014
to ensure consistency with existing processed data.
"""
import sys
sys.path.insert(0, '.')

from pathlib import Path
from src.data.clean_mafaulda_processor import CleanMaFaulDaProcessor

RAW_DIR = 'data/raw-mafaulda'
PROCESSED_DIR = 'data/processed-mafaulda/16hz'
TARGET_HZ = 16.0
MIN_WINDOW_SIZE = 3014  # Matches existing processed tensors

processor = CleanMaFaulDaProcessor(
    raw_data_dir=RAW_DIR,
    processed_data_dir=PROCESSED_DIR
)

# Only process vertical-misalignment subdirectories
vm_base = Path(RAW_DIR) / 'vertical-misalignment'
vm_subdirs = sorted([d for d in vm_base.iterdir() if d.is_dir()])
print(f"Found {len(vm_subdirs)} vertical-misalignment subdirs: {[d.name for d in vm_subdirs]}")

for subdir in vm_subdirs:
    # Find the CSV closest to 16Hz
    csv_files = list(subdir.glob('*.csv'))
    if not csv_files:
        print(f"  WARNING: No CSV files in {subdir}")
        continue
    best_csv = min(csv_files, key=lambda f: abs(float(f.stem) - TARGET_HZ))
    print(f"\nProcessing {subdir.name}: using {best_csv.name} (closest to {TARGET_HZ}Hz)")

    # Get label name (e.g., 'vertical_misalignment_fault_0.51mm')
    label = processor.get_category_label(subdir)
    print(f"  Label: {label}")

    # Process this category
    processor.process_category(
        target_dir=subdir,
        label=label,
        specific_csv=best_csv,
        min_window_size=MIN_WINDOW_SIZE
    )

print("\nDone processing vertical-misalignment categories!")
