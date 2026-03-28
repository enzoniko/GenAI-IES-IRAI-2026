import argparse
from src.pipelines import run_training_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: Physics-Guided Counterfactual Fault Synthesis")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum epochs per training phase")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for dataloader")
    
    args = parser.parse_args()
    print("Starting Physics-Guided Counterfactual Fault Synthesis Pipeline Phase 2.")
    print(f"Configuration: max {args.max_epochs} epochs per phase, Batch Size: {args.batch_size}")
    
    run_training_pipeline(max_epochs=args.max_epochs, batch_size=args.batch_size)
