import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

from src.models import LatentDiffusionMLP, PriorWorkOracle

def train_latent_diffusion(ts_jepa, ldm, scheduler, train_loader, val_loader, device, epochs=50):
    print("--- Phase 3: Training Latent Diffusion Model on z_macro ---")
    
    # Put JEPA in eval mode since it's frozen
    ts_jepa.eval()
    
    optimizer = optim.Adam(ldm.parameters(), lr=1e-3)
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from src.pipelines.train_phase2 import EarlyStopping
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    early_stopping = EarlyStopping(patience=5)
    
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        ldm.train()
        train_loss = 0.0
        
        for batch in train_loader:
            raw, _, _ = batch
            raw = raw.to(device)
            batch_size = raw.shape[0]
            
            # Extract z_macro from frozen TS-JEPA
            with torch.no_grad():
                z_macro = ts_jepa.get_z_macro(raw)
                
            # Sample random noise and random timesteps
            noise = torch.randn_like(z_macro)
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (batch_size,), device=device).long()
            
            # Add noise to the latents (forward diffusion process)
            noisy_latents = scheduler.add_noise(z_macro, noise, timesteps).float()
            
            # Predict the noise residual
            optimizer.zero_grad()
            noise_pred = ldm(noisy_latents, timesteps)
            
            loss = criterion(noise_pred, noise)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation Loop
        ldm.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                raw, _, _ = batch
                raw = raw.to(device)
                batch_size = raw.shape[0]
                z_macro = ts_jepa.get_z_macro(raw)
                noise = torch.randn_like(z_macro)
                timesteps = torch.randint(0, scheduler.num_train_timesteps, (batch_size,), device=device).long()
                noisy_latents = scheduler.add_noise(z_macro, noise, timesteps).float()
                noise_pred = ldm(noisy_latents, timesteps)
                val_loss += criterion(noise_pred, noise).item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, LDM Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
        lr_scheduler.step(avg_val_loss)
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping triggered for LDM at epoch {epoch+1}.")
            break
            
    print("LDM Training Complete.")
    return ldm

def run_guided_sdedit(ts_jepa, decoder1, decoder2, ldm, oracle, scheduler, 
                      healthy_trace, target_class_idx, val_loader, device, num_inference_steps=1000, 
                      guidance_scale=10.0, strength=0.5):
    """
    healthy_trace: A raw healthy physical signal of shape (1, 4, 5000)
    target_class_idx: The fault class to synthesize (e.g., 2 for Outer-Race)
    val_loader: Dataloader containing validation set to map background UMAP clusters
    """
    print(f"\n--- Running Guided SDEdit to Target Class: {target_class_idx} ---")
    
    ts_jepa.eval()
    decoder1.eval()
    decoder2.eval()
    ldm.eval()
    oracle.eval()
    
    t_start = int(num_inference_steps * strength)
    
    # 1. Encode healthy trace to get starting structure
    with torch.no_grad():
        z_start = ts_jepa.get_z_macro(healthy_trace)
        z_start = z_start.float()
        
    # 2. Add noise forward to t_start
    noise = torch.randn_like(z_start)
    t_start_tensor = torch.tensor([t_start], device=device).long()
    z_t = scheduler.add_noise(z_start, noise, t_start_tensor)
    
    # Load target distribution from Oracle
    target_distribution = oracle.get_target_distribution(target_class_idx).to(device)
    
    # 3. Iterative Reverse Denoising with VJP Guidance
    mse_penalty = nn.MSELoss()
    z_trajectory = [z_start.detach().cpu()]
    
    for t in reversed(range(0, t_start)):
        z_trajectory.append(z_t.detach().cpu())
        # Calculate Unconditional Score
        t_tensor = torch.tensor([t], device=device).long()
        with torch.no_grad():
            uncond_noise_pred = ldm(z_t, t_tensor)
            
        # VJP Physics Guidance
        z_t = z_t.detach()
        z_t.requires_grad_(True)
        
        # Pass through the physics graph
        pred_trace = decoder1(z_t)
        pred_emb = oracle(pred_trace)
        
        # Calculate constraint penalty
        penalty = mse_penalty(pred_emb.squeeze(0), target_distribution)
        
        # Call autograd to backpropagate penalty gradients precisely to z_t
        grad = torch.autograd.grad(penalty, z_t)[0]
        
        z_t.requires_grad_(False)
        
        # Update denoising step via DDPMScheduler using Unconditional Score
        z_t = scheduler.step(uncond_noise_pred, t, z_t)
        
        # Steer toward target physics! (Denoise step minus scale * penalty_grad)
        z_t = z_t - guidance_scale * grad
        
        if t % 100 == 0:
            print(f"Step {t:03d}/{num_inference_steps} | Oracle VJP Penalty: {penalty.item():.4f}")

    # 4. Synthesize final Fault Trace
    label_tensor = torch.tensor([target_class_idx], dtype=torch.long, device=device)
    with torch.no_grad():
        final_macro_recon = decoder1(z_t)
        sampled_fault_jitter = decoder2.sample(z_t, label_tensor)
        final_counterfactual = final_macro_recon + sampled_fault_jitter
        
        orig_macro_recon = decoder1(z_start)
        
    print("SDEdit Counterfactual synthesized successfully.")
    
    # 5. Plot the result
    os.makedirs("results", exist_ok=True)
    t_ax = torch.linspace(0, 5000/50000, 5000).numpy()
    
    fig, axes = plt.subplots(4, 1, figsize=(10, 12))
    fig.suptitle(f"Phase 3: Physics-Guided Counterfactual Synthesis\nTransition: Healthy -> Class {target_class_idx}", fontsize=16)
    
    healthy = healthy_trace[0].cpu().numpy()
    counterfactual = final_counterfactual[0].cpu().numpy()
    
    for c in range(4):
        axes[c].plot(t_ax, healthy[c], label='Original Healthy', color='blue', alpha=0.5)
        axes[c].plot(t_ax, counterfactual[c], label=f'Guided Counterfactual (Class {target_class_idx})', color='red', alpha=0.8, linestyle='dashed')
        axes[c].set_ylabel(f"Ch {c+1}")
        if c == 0:
            axes[c].legend()
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"results/sdedit_counterfactual_class_{target_class_idx}.png", dpi=300)
    plt.close()
    
    # 6. UMAP Latent Space Progressive Transformation Plot
    try:
        import umap
    except ImportError:
        print("umap-learn not installed. Skipping UMAP projection.")
        return final_counterfactual

    print("--- Computing UMAP Background and Trajectory Projection ---")
    all_z = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            r, _, labels = batch
            r = r.to(device)
            z_m = ts_jepa.get_z_macro(r)
            all_z.append(z_m.cpu())
            all_labels.append(labels)
            
    all_z = torch.cat(all_z, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    # Project trajectory
    z_trajectory.append(z_t.detach().cpu())
    z_traj_arr = torch.cat(z_trajectory, dim=0).numpy()

    # --- THE FIX: COMBINED UMAP FITTING ---
    print("Fitting UMAP on combined background + trajectory for topological accuracy...")
    import numpy as np
    import umap
    
    combined_data = np.concatenate([all_z, z_traj_arr], axis=0)
    reducer = umap.UMAP(n_components=2, random_state=42)
    combined_embedding = reducer.fit_transform(combined_data)

    # Split the embeddings back apart for plotting
    num_bg = len(all_z)
    embedding = combined_embedding[:num_bg]
    traj_embedding = combined_embedding[num_bg:]
    # --------------------------------------

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=all_labels, cmap='viridis', s=15, alpha=0.3)
    
    classes = ["Healthy", "Imbalance", "Outer-Race"]
    handles, _ = scatter.legend_elements(prop="colors")
    if len(handles) == 3:
        legend1 = plt.legend(handles, classes, title="Fault Classes", loc="upper left")
        plt.gca().add_artist(legend1)

    import numpy as np
    import matplotlib.cm as cm
    
    num_points = len(traj_embedding)
    # Generate colors from light red to dark saturated red to simulate decay/fade logic
    colors = cm.Reds(np.linspace(0.2, 1.0, num_points))
    
    # Plot the path (skipping the very first clean point so we just see the denoising path)
    plt.scatter(traj_embedding[1:, 0], traj_embedding[1:, 1], c=colors[1:], s=40, edgecolors='none', label='SDEdit Guid. Path (Decay)')
    
    # Emphasize True Start and True End
    plt.scatter(traj_embedding[0, 0], traj_embedding[0, 1], color='black', s=150, marker='X', label='True Start (Clean Healthy)')
    plt.scatter(traj_embedding[-1, 0], traj_embedding[-1, 1], color='darkred', s=200, marker='*', label=f'End (Class {target_class_idx})')
    
    plt.title(f"UMAP Progressive Transformation: Healthy to Class {target_class_idx}")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"results/umap_sdedit_trajectory_class_{target_class_idx}.png", dpi=300)
    plt.close()
    print(f"Saved UMAP trajectory progression to results/umap_sdedit_trajectory_class_{target_class_idx}.png")
    
    return final_counterfactual
