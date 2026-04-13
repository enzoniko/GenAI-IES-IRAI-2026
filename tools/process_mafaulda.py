import argparse
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src.configs as cfg
from src.data.clean_mafaulda_processor import CleanMaFaulDaProcessor

def main():
    parser = argparse.ArgumentParser(description="Process MaFaulDa dataset recursively.")
    parser.add_argument("--strategy", type=str, default=cfg.SIGNAL_PROCESSING_STRATEGY, 
                        help="Signal processing strategy (fft or previous_strategy)")
    parser.add_argument("--cutoff", type=float, default=cfg.CUTOFF_HZ, 
                        help="High-pass filter cutoff frequency in Hz")
    parser.add_argument("--target_hz", type=float, default=cfg.TARGET_HZ, 
                        help="Process only CSVs closest to this frequency per category.")
    parser.add_argument("--train_windows", type=int, default=None, 
                        help="Number of training windows to extract per file")
    parser.add_argument("--test_windows", type=int, default=None, 
                        help="Number of test windows to extract per file")
    
    args = parser.parse_args()
    
    processor = CleanMaFaulDaProcessor(
        raw_data_dir=cfg.DATA_DIR_RAW,
        processed_data_dir=cfg.DATA_DIR_PROCESSED,
        cutoff_hz=args.cutoff,
        strategy=args.strategy
    )
    
    print(f"Starting MaFaulDa processing...")
    print(f"Raw Dir: {cfg.DATA_DIR_RAW}")
    print(f"Processed Dir: {cfg.DATA_DIR_PROCESSED}")
    print(f"Strategy: {args.strategy}")
    print(f"Target HZ: {args.target_hz}")
    print(f"Cutoff: {args.cutoff} Hz")
    
    processor.run(
        target_hz=args.target_hz,
        training_windows=args.train_windows,
        test_windows=args.test_windows
    )
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    main()
