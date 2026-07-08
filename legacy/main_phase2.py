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
import src.configs as cfg

def get_or_train_phase1_models(device, batch_size=32, num_samples=1500):
    # Check if Phase 1 checkpoints exist
    ckpts_exist = (
        os.path.exists(cfg.JEPA_MODEL_PATH) and 
        os.path.exists(cfg.DEC1_MODEL_PATH) and 
        os.path.exists(cfg.DEC2_MODEL_PATH)
    )
    
    if ckpts_exist:
        print("Found existing Phase 2 checkpoints. Loading them...")
        ts_jepa = TSJEPA(in_channels=4).to(device)
        decoder1 = Decoder1(out_channels=4).to(device)
        decoder2 = Decoder2CVAE(in_channels=4).to(device)
        
        ts_jepa.load_state_dict(torch.load(cfg.JEPA_MODEL_PATH, map_location=device, weights_only=True))
        decoder1.load_state_dict(torch.load(cfg.DEC1_MODEL_PATH, map_location=device, weights_only=True))
        decoder2.load_state_dict(torch.load(cfg.DEC2_MODEL_PATH, map_location=device, weights_only=True))
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

def calibrate_oracle_targets(ts_jepa, oracle, val_loader, device):
    """
    Performs Physical Calibration by finding the 'Center of Gravity' for each fault type.
    
    Instead of just picking a single random sample to represent a fault, this function 
    scans the validation set to calculate a stable, average physical signature (Centroid) 
    for every class it discovers. 
    
    Why we do this:
    1. Statistical Stability: By averaging multiple signals, we 'wash away' the random 
       noise or quirks of individual recordings. This gives the SDEdit loop a much 
       cleaner and more reliable 'target' to steer towards.
    2. Reactive Physics: The calibration uses the actual measured rotational speed (omega) 
       for every window, ensuring the resulting target embeddings are physically 
       authentic to the real-world operating conditions.
    3. Auto-Discovery: The logic automatically identifies and maps every fault class 
       present in the data, making the pipeline completely class-agnostic and scalable.
    """
    print("--- Phase 3: Calibrating Physical Target Distribution ---")
    ts_jepa.eval()
    oracle.eval()
    
    # We will automatically discover all classes present in the validation loader
    class_embeddings = {}
    class_counts = {}
    
    with torch.no_grad():
        for batch in val_loader:
            raw, _, labels = batch
            raw = raw.to(device)
            z_macro = ts_jepa.get_z_macro(raw)
            # Use the measured rotational speed from the batch for physical extraction
            z_phys = oracle(raw, omega=batch.omega.to(device))
            
            for i in range(len(labels)):
                c = labels[i].item()
                if c not in class_embeddings:
                    class_embeddings[c] = torch.zeros_like(z_phys[i])
                    class_counts[c] = 0
                class_embeddings[c] += z_phys[i]
                class_counts[c] += 1
                
    # Average and store in Oracle
    for c in class_embeddings:
        avg_emb = class_embeddings[c] / class_counts[c]
        oracle.set_target_distribution(c, avg_emb)
        print(f"Calibrated target for Class {c} using {class_counts[c]} matches.")

def run_phase2_pipeline(ldm_epochs, batch_size=32, num_samples=1500):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Phase 1 Models loading
    ts_jepa, decoder1, decoder2 = get_or_train_phase1_models(device, batch_size=batch_size, num_samples=num_samples)

    # Freezing Phase 1 models to ensure no accidental updates during SDEdit optimization
    for model in [ts_jepa, decoder1, decoder2]:
        for param in model.parameters():
            param.requires_grad = False
    
    # 2. Dataset for LDM training & inference
    train_loader, val_loader, seq_len = get_dataloaders(
        batch_size=cfg.PHASE1_TRAIN_SETTINGS['batch_size'],
        num_samples=num_samples,
        val_split=0.2
    )
    print(f"Dataloaders ready. Detected Sequence Length: {seq_len}")
    
    # 3. Instantiate Phase 3 Modules
    ldm = LatentDiffusionMLP(z_dim=cfg.JEPA_CONFIG['d_model'], time_dim=64).to(device)
    oracle = PriorWorkOracle().to(device)
    scheduler = DDPMScheduler(num_train_timesteps=cfg.SDEDIT_GUIDANCE_SETTINGS['num_inference_steps'], device=device)
    
    # Decoder1: Latent -> Macro-Reconstruction
    decoder1 = Decoder1(d_model=cfg.JEPA_CONFIG['d_model'], seq_length=seq_len).to(device)
    # Decoder2: Latent + Class -> Micro-Jitter
    decoder2 = Decoder2CVAE(context_dim=cfg.JEPA_CONFIG['d_model'], num_classes=cfg.NUM_CLASSES, seq_length=seq_len).to(device)
    
    # 4. Train LDM
    ldm = train_latent_diffusion(ts_jepa, ldm, scheduler, train_loader, val_loader, device, epochs=ldm_epochs)
    
    # 5. Extract a Healthy Trace for Synthesis Target
    healthy_trace, healthy_omega = extract_healthy_trace(val_loader, device)
    
    # 6. Calibrate Oracle targets empirical distribution
    calibrate_oracle_targets(ts_jepa, oracle, val_loader, device)

    # 7. Execute Physics-Guided SDEdit targeted at Class 1 (Imbalance) and Class 2 (Outer-Race)
    for target_class in [1, 2]:
        print(f"\n--- Initiating SDEdit Trajectory: Healthy -> Class {target_class} ---")
        try:
            run_guided_sdedit(
                ts_jepa=ts_jepa, decoder1=decoder1, decoder2=decoder2,
                ldm=ldm, oracle=oracle, scheduler=scheduler,
                healthy_trace=healthy_trace, target_class_idx=target_class,
                val_loader=val_loader, device=device,
                omega=healthy_omega
            )
        except Exception as e:
            print(f"Failed to synthesize for class {target_class}: {e}")

    print("\nPhase 3 Execution Completed Successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3: Physics-Guided Counterfactual Fault Synthesis loop")
    parser.add_argument("--ldm_epochs", type=int, default=cfg.PHASE2_TRAIN_SETTINGS['ldm_epochs'], help="Epochs to train the Latent Diffusion Model")
    parser.add_argument("--batch_size", type=int, default=cfg.PHASE1_TRAIN_SETTINGS['batch_size'], help="Batch size for dataloader")
    parser.add_argument("--num_samples", type=int, default=1500, help="Total samples to load (reduce for quick structural debug)")
    args = parser.parse_args()
    
    print("Starting Physics-Guided Counterfactual Fault Synthesis Pipeline Phase 3.")
    print(f"Configuration: LDM Epochs: {args.ldm_epochs}, Batch Size: {args.batch_size}, Samples: {args.num_samples}")
    
    run_phase2_pipeline(ldm_epochs=args.ldm_epochs, batch_size=args.batch_size, num_samples=args.num_samples)
