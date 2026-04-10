'''
Physics-Guided Counterfactual Fault Synthesis in Cyber-Physical Systems

Phase 2 - Deployment and SDEdit Transformation.
'''

import torch
import os
import argparse
from src.models import TSJEPA, Decoder1, Decoder2CVAE, LatentDiffusionMLP, DDPMScheduler, PriorWorkOracle
from src.data import get_dataloaders
from src.pipelines import run_training_pipeline
from src.pipelines.run_sdedit_phase2 import train_latent_diffusion, run_guided_sdedit

def get_or_train_phase1_models(device, batch_size=32, num_samples=1500):
    # Check if Phase 1 checkpoints exist
    ckpts_exist = (
        os.path.exists("results/ts_jepa.pth") and 
        os.path.exists("results/decoder1.pth") and 
        os.path.exists("results/decoder2.pth")
    )
    
    if ckpts_exist:
        print("Found existing Phase 2 checkpoints. Loading them...")
        ts_jepa = TSJEPA(in_channels=4).to(device)
        decoder1 = Decoder1(out_channels=4).to(device)
        decoder2 = Decoder2CVAE(in_channels=4).to(device)
        
        ts_jepa.load_state_dict(torch.load("results/ts_jepa.pth", map_location=device, weights_only=True))
        decoder1.load_state_dict(torch.load("results/decoder1.pth", map_location=device, weights_only=True))
        decoder2.load_state_dict(torch.load("results/decoder2.pth", map_location=device, weights_only=True))
        return ts_jepa, decoder1, decoder2
    else:
        print("Phase 2 checkpoints not found. Running Phase 2 Training Pipeline first (Max Epochs: 20 for speed)...")
        # For Hackathon speed, we use a lower max_epochs to get structural convergence
        return run_training_pipeline(max_epochs=2, batch_size=batch_size, num_samples=num_samples)

def extract_healthy_trace(val_loader, device):
    """Scans the validation loader for a real healthy signal to use as SDEdit's starting point."""
    for batch in val_loader:
        raw, _, label = batch
        idx = (label == 0).nonzero(as_tuple=True)[0]
        if len(idx) > 0:
            omega_batch = getattr(batch, 'omega', None)
            trace = raw[idx[0]:idx[0]+1].to(device)
            omega_val = omega_batch[idx[0]:idx[0]+1].to(device) if omega_batch is not None else None
            return trace, omega_val
    raise ValueError("No healthy traces found in the validation dataset to use as a starting point!")

def calibrate_oracle_targets(oracle, val_loader, device):
    """Extracts empirical clusters from validation data to set physical optimization targets."""
    print("Calculating empirical Oracle targets from real validation data...")
    oracle.eval()
    
    for batch in val_loader:
        raw, _, label = batch
        omega_batch = getattr(batch, 'omega', None)
        
        idx_1 = (label == 1).nonzero(as_tuple=True)[0]
        idx_2 = (label == 2).nonzero(as_tuple=True)[0]
        
        if len(idx_1) > 0:
            with torch.no_grad():
                if omega_batch is not None:
                    oracle.set_dynamic_omega(omega_batch[idx_1[0]:idx_1[0]+1].to(device))
                emb_1 = oracle(raw[idx_1[0]:idx_1[0]+1].to(device)).squeeze(0)
                oracle.set_target_distribution(1, emb_1)
        if len(idx_2) > 0:
            with torch.no_grad():
                if omega_batch is not None:
                    oracle.set_dynamic_omega(omega_batch[idx_2[0]:idx_2[0]+1].to(device))
                emb_2 = oracle(raw[idx_2[0]:idx_2[0]+1].to(device)).squeeze(0)
                oracle.set_target_distribution(2, emb_2)
                
        # Break once both are set
        if len(idx_1) > 0 and len(idx_2) > 0:
            break

def run_phase3_pipeline(ldm_epochs, batch_size=32, num_samples=1500):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Phase 1 Models loading
    ts_jepa, decoder1, decoder2 = get_or_train_phase1_models(device, batch_size=batch_size, num_samples=num_samples)

    # Freezing Phase 1 models to ensure no accidental updates during SDEdit optimization
    for model in [ts_jepa, decoder1, decoder2]:
        for param in model.parameters():
            param.requires_grad = False
    
    # 2. Dataset for LDM training & inference targeting
    train_loader, val_loader = get_dataloaders(batch_size=batch_size, num_samples=num_samples, val_split=0.2)
    
    # 3. Instantiate Phase 3 Modules
    ldm = LatentDiffusionMLP(z_dim=128, time_dim=64).to(device)
    oracle = PriorWorkOracle().to(device)
    scheduler = DDPMScheduler(num_train_timesteps=1000, device=device)
    
    # 4. Train LDM
    ldm = train_latent_diffusion(ts_jepa, ldm, scheduler, train_loader, val_loader, device, epochs=ldm_epochs)
    
    # 5. Extract a Healthy Trace for Synthesis Target
    healthy_trace, healthy_omega = extract_healthy_trace(val_loader, device)
    calibrate_oracle_targets(oracle, val_loader, device)

    # Pre-set the dynamic omega for the generation runs
    if healthy_omega is not None:
        oracle.set_dynamic_omega(healthy_omega)

    # 6. Execute Physics-Guided SDEdit targeted at Class 2 (Outer-Race)
    print("\n--- Initiating SDEdit Trajectory: Healthy -> Outer-Race (Class 2) ---")
    run_guided_sdedit(
        ts_jepa=ts_jepa, decoder1=decoder1, decoder2=decoder2,
        ldm=ldm, oracle=oracle, scheduler=scheduler,
        healthy_trace=healthy_trace, target_class_idx=2,
        val_loader=val_loader, device=device,
        num_inference_steps=1000, guidance_scale=0.5, strength=0.5
    )
    
    print("\n--- Initiating SDEdit Trajectory: Healthy -> Imbalance (Class 1) ---")
    run_guided_sdedit(
        ts_jepa=ts_jepa, decoder1=decoder1, decoder2=decoder2,
        ldm=ldm, oracle=oracle, scheduler=scheduler,
        healthy_trace=healthy_trace, target_class_idx=1,
        val_loader=val_loader, device=device,
        num_inference_steps=1000, guidance_scale=0.5, strength=0.5
    )

    print("\nPhase 3 Execution Completed Successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3: Physics-Guided Counterfactual Fault Synthesis loop")
    parser.add_argument("--ldm_epochs", type=int, default=50, help="Epochs to train the Latent Diffusion Model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for dataloader")
    parser.add_argument("--num_samples", type=int, default=1500, help="Total samples to load (reduce for quick structural debug)")
    args = parser.parse_args()
    
    print("Starting Physics-Guided Counterfactual Fault Synthesis Pipeline Phase 3.")
    print(f"Configuration: LDM Epochs: {args.ldm_epochs}, Batch Size: {args.batch_size}, Samples: {args.num_samples}")
    
    run_phase3_pipeline(ldm_epochs=args.ldm_epochs, batch_size=args.batch_size, num_samples=args.num_samples)
