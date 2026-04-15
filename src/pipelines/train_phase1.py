import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import os
import src.configs as cfg
from src.models import TSJEPA, Decoder1, Decoder2CVAE
from src.data import get_dataloaders

class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train_phase1_tsjepa(model, train_loader, val_loader, max_epochs, device):
    print("--- Phase 1: Training TS-JEPA ---")
    optimizer = optim.Adam(model.parameters(), lr=cfg.PHASE1_TRAIN_SETTINGS['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    early_stopping = EarlyStopping(patience=cfg.PHASE1_TRAIN_SETTINGS['early_stop_patience'])
    criterion = nn.MSELoss()
    
    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            raw, _, _ = batch
            raw = raw.to(device)
            optimizer.zero_grad()
            pred_unobs, target_unobs = model(raw)
            loss = criterion(pred_unobs, target_unobs)
            loss.backward()
            optimizer.step()
            model.update_ema()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                raw, _, _ = batch
                raw = raw.to(device)
                pred_unobs, target_unobs = model(raw)
                val_loss += criterion(pred_unobs, target_unobs).item()
                
        t_loss = train_loss / len(train_loader)
        v_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{max_epochs}, TS-JEPA Train Loss: {t_loss:.4f}, Val Loss: {v_loss:.4f}")
        
        scheduler.step(v_loss)
        early_stopping(v_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered for TS-JEPA.")
            break
            
    for param in model.parameters():
        param.requires_grad = False
    return model

def train_phase1_decoder1(ts_jepa, decoder1, train_loader, val_loader, max_epochs, device):
    print("--- Phase 2: Training Decoder 1 (Deterministic) on RAW TRACES ---")
    optimizer = optim.Adam(decoder1.parameters(), lr=cfg.PHASE1_TRAIN_SETTINGS['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    early_stopping = EarlyStopping(patience=cfg.PHASE1_TRAIN_SETTINGS['early_stop_patience'])
    criterion = nn.MSELoss()
    
    for epoch in range(max_epochs):
        decoder1.train()
        train_loss = 0.0
        for batch in train_loader:
            raw, _, _ = batch  
            raw = raw.to(device)
            with torch.no_grad():
                z_macro = ts_jepa.get_z_macro(raw)
            
            optimizer.zero_grad()
            recon_raw = decoder1(z_macro)
            loss = criterion(recon_raw, raw)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        decoder1.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                raw, _, _ = batch
                raw = raw.to(device)
                z_macro = ts_jepa.get_z_macro(raw)
                recon_raw = decoder1(z_macro)
                val_loss += criterion(recon_raw, raw).item()
                
        t_loss = train_loss / len(train_loader)
        v_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{max_epochs}, Decoder 1 Train Loss: {t_loss:.4f}, Val Loss: {v_loss:.4f}")
        
        scheduler.step(v_loss)
        early_stopping(v_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered for Decoder 1.")
            break
            
    for param in decoder1.parameters():
        param.requires_grad = False
    return decoder1

def extract_residuals_and_train_decoder2(ts_jepa, decoder1, decoder2, train_loader, val_loader, max_epochs, device):
    print("--- Phase 3 & 4: Extracting Residuals & Training Decoder 2 (CVAE) ---")
    optimizer = optim.Adam(decoder2.parameters(), lr=cfg.PHASE1_TRAIN_SETTINGS['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    early_stopping = EarlyStopping(patience=cfg.PHASE1_TRAIN_SETTINGS['early_stop_patience'])
    recon_criterion = nn.MSELoss()
    
    def kl_loss_fn(mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
    for epoch in range(max_epochs):
        decoder2.train()
        train_loss = 0.0
        for batch in train_loader:
            raw, _, label = batch
            raw = raw.to(device)
            label = label.to(device)
            
            with torch.no_grad():
                z_macro = ts_jepa.get_z_macro(raw)
                recon_raw = decoder1(z_macro)
                residual = raw - recon_raw
                
            optimizer.zero_grad()
            recon_residual, mu, logvar = decoder2(residual, z_macro, label)
            
            recon_loss = recon_criterion(recon_residual, residual)
            kl_loss = kl_loss_fn(mu, logvar)
            
            loss = recon_loss + cfg.PHASE1_TRAIN_SETTINGS['beta_kl'] * (kl_loss / raw.size(0))
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        decoder2.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                raw, _, label = batch
                raw = raw.to(device)
                label = label.to(device)
                
                z_macro = ts_jepa.get_z_macro(raw)
                recon_raw = decoder1(z_macro)
                residual = raw - recon_raw
                
                recon_residual, mu, logvar = decoder2(residual, z_macro, label)
                loss = recon_criterion(recon_residual, residual) + cfg.PHASE1_TRAIN_SETTINGS['beta_kl'] * (kl_loss_fn(mu, logvar) / raw.size(0))
                val_loss += loss.item()
                
        t_loss = train_loss / len(train_loader)
        v_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{max_epochs}, CVAE Train Loss: {t_loss:.4f}, Val Loss: {v_loss:.4f}")
        
        scheduler.step(v_loss)
        early_stopping(v_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered for CVAE.")
            break
            
    return decoder2

def evaluate_pipeline(ts_jepa, decoder1, decoder2, val_loader, device):
    print("--- Phase 5: Multivariate Evaluation & Plotting ---")
    os.makedirs("results", exist_ok=True)

    # We'll plot a few representative samples if available
    # For 42 classes, we'll just pick a few to keep it manageable
    target_labels = [0, 35, 36, 37, 38] # Normal and 4 types of imbalance
    samples = {l: None for l in target_labels}
    
    for batch in val_loader:
        raws, cleans, labels = batch
        omegas = batch.omega
        for i in range(len(labels)):
            label = labels[i].item()
            if label in samples and samples[label] is None:
                samples[label] = (raws[i:i+1], cleans[i:i+1])
        if all(v is not None for v in samples.values()):
            break
            
    # mapping index to name (reverse lookup would be better but we can just use ids)
    
    for idx_label, sample_data in samples.items():
        if sample_data is None: continue
        raw, clean = sample_data
        raw, clean = raw.to(device), clean.to(device)
        label_tensor = torch.tensor([idx_label], dtype=torch.long, device=device)
        
        with torch.no_grad():
            z_macro = ts_jepa.get_z_macro(raw)
            recon_raw = decoder1(z_macro)
            residual = raw - recon_raw
            sampled_jitter = decoder2.sample(z_macro, label_tensor)
            final_synthetic_trace = recon_raw + sampled_jitter
            
        seq_len = raw.shape[-1]
        t = torch.linspace(0, seq_len/50000, seq_len).numpy()
        
        fig, axes = plt.subplots(4, 4, figsize=(24, 16))
        fig.suptitle(f"Multivariate Analysis: Label {idx_label}", fontsize=20)
        
        metric_titles = [
            "Dec1 Recon (TS-JEPA Filter)",
            "True High-Freq Residual",
            "CVAE Generated Jitter",
            "Final Superposition"
        ]
        
        for c in range(4):
            orig_c = raw[0, c].cpu().numpy()
            recon_dec1_c = recon_raw[0, c].cpu().numpy()
            res_c = residual[0, c].cpu().numpy()
            synth_jitter_c = sampled_jitter[0, c].cpu().numpy()
            synth_final_c = final_synthetic_trace[0, c].cpu().numpy()
            
            axes[c, 0].plot(t, orig_c, alpha=0.5, label='True Target')
            axes[c, 0].plot(t, recon_dec1_c, alpha=0.8, linestyle='--', color='orange', label='Dec1 Recon')
            axes[c, 0].set_ylabel(f"Channel {c+1}", fontsize=14)
            if c == 0: axes[c, 0].set_title(metric_titles[0], fontsize=14)
            axes[c, 0].legend()
            
            axes[c, 1].plot(t, res_c, color='red', alpha=0.6)
            if c == 0: axes[c, 1].set_title(metric_titles[1], fontsize=14)
            
            axes[c, 2].plot(t, res_c, color='red', alpha=0.2, label='True Res')
            axes[c, 2].plot(t, synth_jitter_c, color='green', alpha=0.7, label='CVAE Jitter')
            if c == 0: axes[c, 2].set_title(metric_titles[2], fontsize=14)
            axes[c, 2].legend()
            
            axes[c, 3].plot(t, orig_c, alpha=0.3, label='True Tr')
            axes[c, 3].plot(t, synth_final_c, color='purple', linestyle='--', alpha=0.8, label='Synth')
            if c == 0: axes[c, 3].set_title(metric_titles[3], fontsize=14)
            axes[c, 3].legend()
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
        save_path = os.path.join(cfg.RESULTS_DIR, f"evaluation_class_{idx_label}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        
    print("Evaluation completed. Saved multivariate plots to results/evaluation_class_*.png")

def plot_umap(ts_jepa, val_loader, device):
    print("--- Phase 5.5: UMAP Latent Space Projection ---")
    try:
        import umap
    except ImportError:
        print("umap-learn not installed. Skipping UMAP projection.")
        return
        
    ts_jepa.eval()
    all_z = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            raw, _, labels = batch
            raw = raw.to(device)
            z_macro = ts_jepa.get_z_macro(raw)
            all_z.append(z_macro.cpu())
            all_labels.append(labels)
            
    all_z = torch.cat(all_z, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(all_z)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=all_labels, cmap='tab20', s=15, alpha=0.8)
    
    plt.colorbar(scatter, label='Fault Class ID')
    # classes = ["Healthy", "Imbalance", "Outer-Race"]
    # handles, _ = scatter.legend_elements(prop="colors")
    # if len(handles) == 3:
    #     plt.legend(handles, classes, title="Fault Classes")
        
    plt.title("UMAP Projection of TS-JEPA z_macro Latent Space")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("results/umap_latent_space.png", dpi=300)
    plt.close()
    print("Saved UMAP projection to results/umap_latent_space.png")

def run_training_pipeline(max_epochs=100, batch_size=32, num_samples=1500):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_loader, val_loader, detected_seq_length = get_dataloaders(batch_size=batch_size, num_samples=num_samples, val_split=0.2)
    print(f"  [Pipeline] Initializing models with Sequence Length: {detected_seq_length}")
    
    ts_jepa = TSJEPA(in_channels=4).to(device)
    decoder1 = Decoder1(out_channels=4, seq_length=detected_seq_length).to(device)
    decoder2 = Decoder2CVAE(in_channels=4, seq_length=detected_seq_length).to(device)

    ts_jepa = train_phase1_tsjepa(ts_jepa, train_loader, val_loader, max_epochs, device)
    decoder1 = train_phase1_decoder1(ts_jepa, decoder1, train_loader, val_loader, max_epochs, device)
    decoder2 = extract_residuals_and_train_decoder2(ts_jepa, decoder1, decoder2, train_loader, val_loader, max_epochs, device)
    plot_umap(ts_jepa, val_loader, device)
    evaluate_pipeline(ts_jepa, decoder1, decoder2, val_loader, device)

    # Save Checkpoints explicitly for Phase 3
    print("--- Saving Phase 2 Checkpoints ---")
    torch.save(ts_jepa.state_dict(), cfg.JEPA_MODEL_PATH)
    torch.save(decoder1.state_dict(), cfg.DEC1_MODEL_PATH)
    torch.save(decoder2.state_dict(), cfg.DEC2_MODEL_PATH)
    print(f"Checkpoints saved successfully to {cfg.RESULTS_DIR}/")
    
    return ts_jepa, decoder1, decoder2
