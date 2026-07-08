'''
Physics-Guided Counterfactual Fault Synthesis in Cyber-Physical Systems

Phase 1 - Modular Training and Component Freezing
'''

import argparse
import src.configs as cfg
from src.pipelines import run_training_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: Physics-Guided Counterfactual Fault Synthesis")
    parser.add_argument("--max_epochs", type=int, default=cfg.PHASE1_TRAIN_SETTINGS['max_epochs'], help="Maximum epochs per training phase")
    parser.add_argument("--batch_size", type=int, default=cfg.PHASE1_TRAIN_SETTINGS['batch_size'], help="Batch size for dataloader")
    parser.add_argument("--num_samples", type=int, default=1500, help="Total samples to load (reduce for quick structural debug)")
    
    args = parser.parse_args()
    print("Starting Physics-Guided Counterfactual Fault Synthesis Pipeline Phase 1.")
    print(f"Configuration: max {args.max_epochs} epochs per phase, Batch Size: {args.batch_size}, Samples: {args.num_samples}")
    
    run_training_pipeline(max_epochs=args.max_epochs, batch_size=args.batch_size, num_samples=args.num_samples)
