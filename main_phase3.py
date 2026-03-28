import torch
import os
import argparse
from src.models import TSJEPA, Decoder1, Decoder2CVAE, LatentDiffusionMLP, DDPMScheduler, PriorWorkOracle
from src.data import get_dataloaders
from src.pipelines import run_training_pipeline
from src.pipelines.run_sdedit_phase3 import train_latent_diffusion, run_guided_sdedit

def get_or_train_phase2_models(device):
    # Check if Phase 2 checkpoints exist
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
        return run_training_pipeline(max_epochs=20, batch_size=32)

def main():
    parser = argparse.ArgumentParser(description="Phase 3: Physics-Guided Counterfactual Fault Synthesis loop")
    parser.add_argument("--ldm_epochs", type=int, default=50, help="Epochs to train the Latent Diffusion Model")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Phase 2 Models loading
    ts_jepa, decoder1, decoder2 = get_or_train_phase2_models(device)
    
    # Freeze Phase 2 models to be absolutely safe
    for model in [ts_jepa, decoder1, decoder2]:
        for param in model.parameters():
            param.requires_grad = False
    
    # 2. Dataset for LDM training & inference targeting
    train_loader, val_loader = get_dataloaders(batch_size=32, num_samples=1500, val_split=0.2)
    
    # 3. Instantiate Phase 3 Modules
    ldm = LatentDiffusionMLP(z_dim=128, time_dim=64).to(device)
    oracle = PriorWorkOracle(in_channels=4, seq_len=5000, embed_dim=64).to(device)
    scheduler = DDPMScheduler(num_train_timesteps=1000, device=device)
    
    # 4. Train LDM
    ldm = train_latent_diffusion(ts_jepa, ldm, scheduler, train_loader, val_loader, device, epochs=args.ldm_epochs)
    
    # 5. Extract a Healthy Trace for Synthesis Target
    healthy_trace = None
    for batch in val_loader:
        raw, _, label = batch
        # Find first healthy instance (label 0)
        idx = (label == 0).nonzero(as_tuple=True)[0]
        if len(idx) > 0:
            healthy_trace = raw[idx[0]:idx[0]+1].to(device)
            break
            
    if healthy_trace is None:
        raise ValueError("No healthy traces found in the validation dataset to use as a starting point!")
        
    print("Calculating empirical Oracle targets from real validation data...")
    oracle.eval()
    
    # Grab one real sample of each fault to set the Oracle's targets
    for batch in val_loader:
        raw, _, label = batch
        idx_1 = (label == 1).nonzero(as_tuple=True)[0]
        idx_2 = (label == 2).nonzero(as_tuple=True)[0]
        
        if len(idx_1) > 0:
            with torch.no_grad():
                emb_1 = oracle(raw[idx_1[0]:idx_1[0]+1].to(device)).squeeze(0)
                oracle.set_target_distribution(1, emb_1)
        if len(idx_2) > 0:
            with torch.no_grad():
                emb_2 = oracle(raw[idx_2[0]:idx_2[0]+1].to(device)).squeeze(0)
                oracle.set_target_distribution(2, emb_2)
                
        # Break once both are set (for a hackathon, one batch is usually enough)
        if len(idx_1) > 0 and len(idx_2) > 0:
            break

    # 6. Execute Physics-Guided SDEdit targeted at Class 2 (Outer-Race)
    print("\n--- Initiating SDEdit Trajectory: Healthy -> Outer-Race (Class 2) ---")
    run_guided_sdedit(
        ts_jepa=ts_jepa,
        decoder1=decoder1,
        decoder2=decoder2,
        ldm=ldm,
        oracle=oracle,
        scheduler=scheduler,
        healthy_trace=healthy_trace,
        target_class_idx=2,
        val_loader=val_loader,
        device=device,
        num_inference_steps=1000,
        guidance_scale=0.5,
        strength=0.5  # SDEdit starting point: start halfway into the noise process
    )
    
    # Could also execute it for Class 1 (Imbalance)
    print("\n--- Initiating SDEdit Trajectory: Healthy -> Imbalance (Class 1) ---")
    run_guided_sdedit(
        ts_jepa=ts_jepa,
        decoder1=decoder1,
        decoder2=decoder2,
        ldm=ldm,
        oracle=oracle,
        scheduler=scheduler,
        healthy_trace=healthy_trace,
        target_class_idx=1,
        val_loader=val_loader,
        device=device,
        num_inference_steps=1000,
        guidance_scale=0.5,
        strength=0.5  # SDEdit starting point
    )

    print("\nPhase 3 Execution Completed Successfully.")

if __name__ == "__main__":
    main()
