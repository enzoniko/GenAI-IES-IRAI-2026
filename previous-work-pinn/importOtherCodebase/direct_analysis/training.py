"""
Training module for Direct PINN models

This module contains functions for training direct PINN models.
"""

import torch
from torch import optim
from tqdm import tqdm
import os
import numpy as np

class EarlyStopping:
    """
    Early stopping implementation to prevent overfitting during training.
    """
    def __init__(self, patience=10, min_delta=0):
        """
        Initialize early stopping object.
        
        Parameters:
        - patience: Number of epochs to wait after minimum was found
        - min_delta: Minimum change in loss to qualify as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, loss):
        """
        Call the early stopping object with the current validation loss.
        
        Parameters:
        - loss: Current validation loss
        
        Returns: None, but sets self.early_stop to True if stopping criteria are met
        """
        if self.best_loss is None:
            self.best_loss = loss
        elif loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

class ReduceLROnPlateau:
    """
    Learning rate scheduler that reduces the learning rate when a metric has stopped improving.
    """
    def __init__(self, optimizer, factor=0.5, patience=10, min_lr=1e-6, verbose=True):
        """
        Initialize the scheduler.
        
        Parameters:
        - optimizer: Optimizer for which the learning rate will be adjusted
        - factor: Factor by which to reduce the learning rate
        - patience: Number of epochs to wait before reducing the learning rate
        - min_lr: Minimum learning rate
        - verbose: Whether to print a message when the learning rate is reduced
        """
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose
        self.best_loss = None
        self.counter = 0

    def step(self, loss):
        """
        Update the learning rate based on the validation loss.
        
        Parameters:
        - loss: Current validation loss
        """
        if self.best_loss is None:
            self.best_loss = loss
        elif loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self._reduce_lr()

    def _reduce_lr(self):
        """
        Reduce the learning rate by the specified factor.
        """
        for param_group in self.optimizer.param_groups:
            new_lr = max(param_group['lr'] * self.factor, self.min_lr)
            if param_group['lr'] > self.min_lr:
                param_group['lr'] = new_lr
                if self.verbose:
                    print(f"Reducing learning rate to {new_lr}")
            self.counter = 0

def train_model(model, data_dict, device, custom_loss_fn, epochs=10000,
               early_stopping_patience=2000, lr_patience=1000,
               initial_alpha=1000.0, initial_beta=1.0, model_save_path=None):
    """
    Train the direct PINN model.
    
    Parameters:
    - model: The PINN model to train
    - data_dict: Dictionary containing data loaders and normalization parameters
    - device: Device to use for training
    - custom_loss_fn: Custom loss function for PINN
    - epochs: Maximum number of epochs to train
    - early_stopping_patience: Patience for early stopping
    - lr_patience: Patience for learning rate reduction
    - initial_alpha: Initial value for alpha (physics vs. data loss weight)
    - initial_beta: Initial value for beta (additional weight parameter)
    - model_save_path: Path to save the best model
    
    Returns:
    - Trained model
    - Dictionary with training statistics
    """
    train_loader = data_dict['train_loader']
    val_loader = data_dict['val_loader']
    X_min = data_dict.get('X_min', None)
    X_max = data_dict.get('X_max', None)
    y_min = data_dict.get('y_min', None)
    y_max = data_dict.get('y_max', None)
    
    if X_min is not None:
        X_min = X_min.to(device)
        X_max = X_max.to(device)
        y_min = y_min.to(device)
        y_max = y_max.to(device)
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    
    # Define callbacks
    early_stopping = EarlyStopping(patience=early_stopping_patience, min_delta=1e-4)
    lr_scheduler = ReduceLROnPlateau(optimizer, patience=lr_patience, factor=0.5, min_lr=1e-4)
    
    # Initialize training parameters
    alpha = torch.tensor(initial_alpha, requires_grad=False).to(device)
    beta = torch.tensor(initial_beta, requires_grad=False).to(device)
    best_loss = float('inf')
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'alpha': [],
        'beta': []
    }
    
    # Create a default save path if none is provided
    if model_save_path is None:
        model_save_path = 'best_direct_pinn_model.pth'
    
    # Training loop
    pbar = tqdm(range(epochs), desc="Training Direct PINN")
    for epoch in pbar:
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Compute loss
            loss = custom_loss_fn(model, X_batch, y_batch, X_max, X_min, y_max, y_min, alpha, beta)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        history['alpha'].append(alpha.item())
        history['beta'].append(beta.item())
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                loss = custom_loss_fn(model, X_batch, y_batch, X_max, X_min, y_max, y_min, alpha, beta)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        history['val_loss'].append(val_loss)
        
        # Update progress bar
        pbar.set_postfix({'train_loss': f'{train_loss:.4f}', 'val_loss': f'{val_loss:.4f}'})
        
        # Learning rate reduction
        lr_scheduler.step(val_loss)
        
        # Save the best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Epoch {epoch}: Best model saved with validation loss {best_loss:.4f}")
            
            # Update alpha and beta when model improves
            alpha = alpha * 0.99
            alpha = torch.clamp(alpha, min=800.0)
            beta = beta * 1.2
            beta = torch.clamp(beta, min=1.0, max=1000.0)
        
        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch}. Training stopped.")
            break
    
    # Load the best model
    model.load_state_dict(torch.load(model_save_path))
    
    return model, history 

def train_model_adaptive(model, data_dict, device, custom_loss_fn, epochs=10000,
                        alpha=0.9, learning_rate=1e-3, update_frequency=1, model_save_path=None,
                        early_stopping_patience=100, lr_patience=50):
    """
    Train the direct PINN model using an adaptive loss weighting algorithm.
    
    This implementation uses gradient-based adaptive weighting to balance
    different loss components, inspired by the approach in basicPINNv5.
    
    Parameters:
    - model: The PINN model to train
    - data_dict: Dictionary containing data loaders and normalization parameters
    - device: Device to use for training
    - custom_loss_fn: Custom loss function for PINN that returns multiple loss components
    - epochs: Maximum number of epochs to train
    - alpha: Weight smoothing factor (0.9 means 90% previous weight, 10% new weight)
    - learning_rate: Learning rate for manual parameter update
    - update_frequency: Frequency of weight updates (in steps)
    - model_save_path: Path to save the best model
    - early_stopping_patience: Patience for early stopping (stop after this many epochs without improvement)
    - lr_patience: Patience for learning rate reduction (reduce LR after this many epochs without improvement)
    
    Returns:
    - Trained model
    - Dictionary with training statistics
    """
    train_loader = data_dict['train_loader']
    val_loader = data_dict['val_loader']
    X_min = data_dict.get('X_min', None)
    X_max = data_dict.get('X_max', None)
    y_min = data_dict.get('y_min', None)
    y_max = data_dict.get('y_max', None)
    
    if X_min is not None:
        X_min = X_min.to(device)
        X_max = X_max.to(device)
        y_min = y_min.to(device)
        y_max = y_max.to(device)
    
    # Create a default save path if none is provided
    if model_save_path is None:
        model_save_path = 'best_direct_pinn_model.pth'
    
    # Make sure the directory exists
    os.makedirs(os.path.dirname(os.path.abspath(model_save_path)), exist_ok=True)
    
    # Initialize Lagrangian multipliers for each loss component
    lambda_data = torch.tensor(0.3, requires_grad=False).to(device)  # For data loss
    lambda_res1 = torch.tensor(0.1, requires_grad=False).to(device)  # For residual 1
    lambda_res2 = torch.tensor(0.1, requires_grad=False).to(device)  # For residual 2
    lambda_res3 = torch.tensor(0.1, requires_grad=False).to(device)  # For residual 3
    lambda_res4 = torch.tensor(0.1, requires_grad=False).to(device)  # For residual 4
    lambda_mass1 = torch.tensor(0.1, requires_grad=False).to(device) # For mass constraint 1
    lambda_mass2 = torch.tensor(0.1, requires_grad=False).to(device) # For mass constraint 2
    
    # Training history tracking
    history = {
        'train_loss': [],
        'val_loss': [],
        'data_weight': [],
        'res1_weight': [],
        'res2_weight': [],
        'res3_weight': [],
        'res4_weight': [],
        'mass1_weight': [],
        'mass2_weight': [],
        'data_loss': [],
        'res1_loss': [],
        'res2_loss': [],
        'res3_loss': [],
        'res4_loss': [],
        'mass1_loss': [],
        'mass2_loss': [],
        'learning_rate': []
    }
    
    # Norm calculation function with better numerical stability
    norm = lambda g: torch.sqrt(sum(torch.sum(p**2) for p in g if p is not None) + 1e-12)
    
    # Track training progress
    best_val_loss = float('inf')
    patience_counter = 0  # For early stopping
    lr_counter = 0  # For learning rate scheduling
    has_saved_model = False
    current_lr = learning_rate
    
    # Training loop
    step = 0
    pbar = tqdm(range(epochs), desc="Training Direct PINN (Adaptive)")
    
    for epoch in pbar:
        model.train()
        train_loss = 0.0
        train_data_loss = 0.0
        train_res1_loss = 0.0
        train_res2_loss = 0.0
        train_res3_loss = 0.0
        train_res4_loss = 0.0
        train_mass1_loss = 0.0
        train_mass2_loss = 0.0
        train_batches = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            step += 1
            
            try:
                # Forward pass
                data_loss, res1_loss, res2_loss, res3_loss, res4_loss, mass1_loss, mass2_loss = custom_loss_fn(
                    model, X_batch, y_batch, X_max, X_min, y_max, y_min
                )
                
                # Compute total loss with current weights
                total_loss = (lambda_data * data_loss + 
                             lambda_res1 * res1_loss + 
                             lambda_res2 * res2_loss + 
                             lambda_res3 * res3_loss + 
                             lambda_res4 * res4_loss + 
                             lambda_mass1 * mass1_loss + 
                             lambda_mass2 * mass2_loss)
                
                # Check for invalid loss values
                if not torch.isfinite(total_loss):
                    print(f"Warning: Non-finite loss detected at step {step}, skipping batch")
                    continue
                
                # Periodically update weights based on gradients
                if step % update_frequency == 0:
                    try:
                        # Compute gradients for the combined loss
                        grads = torch.autograd.grad(total_loss, model.parameters(), 
                                                  retain_graph=True, create_graph=True)
                        
                        # Check for NaN or Inf gradients
                        if any(torch.isnan(g).any() if g is not None else False for g in grads) or \
                           any(torch.isinf(g).any() if g is not None else False for g in grads):
                            print(f"Warning: NaN or Inf gradients at step {step}, skipping weight update")
                            continue
                        
                        # Compute gradients for each loss component
                        g_data = torch.autograd.grad(data_loss, model.parameters(), 
                                                    retain_graph=True, create_graph=True, allow_unused=True)
                        g_res1 = torch.autograd.grad(res1_loss, model.parameters(), 
                                                    retain_graph=True, create_graph=True, allow_unused=True)
                        g_res2 = torch.autograd.grad(res2_loss, model.parameters(), 
                                                    retain_graph=True, create_graph=True, allow_unused=True)
                        g_res3 = torch.autograd.grad(res3_loss, model.parameters(), 
                                                    retain_graph=True, create_graph=True, allow_unused=True)
                        g_res4 = torch.autograd.grad(res4_loss, model.parameters(), 
                                                    retain_graph=True, create_graph=True, allow_unused=True)
                        g_mass1 = torch.autograd.grad(mass1_loss, model.parameters(), 
                                                     retain_graph=True, create_graph=True, allow_unused=True)
                        g_mass2 = torch.autograd.grad(mass2_loss, model.parameters(), 
                                                     retain_graph=True, create_graph=True, allow_unused=True)
                        
                        # Calculate gradient norms
                        norm_data = norm(g_data)
                        norm_res1 = norm(g_res1)
                        norm_res2 = norm(g_res2)
                        norm_res3 = norm(g_res3)
                        norm_res4 = norm(g_res4)
                        norm_mass1 = norm(g_mass1)
                        norm_mass2 = norm(g_mass2)
                        
                        # Ensure all norms are valid
                        all_norms_valid = all(torch.isfinite(n) for n in 
                                             [norm_data, norm_res1, norm_res2, norm_res3, 
                                              norm_res4, norm_mass1, norm_mass2])
                        
                        if all_norms_valid:
                            # Calculate sum of norms for normalization
                            sum_norms = norm_data + norm_res1 + norm_res2 + norm_res3 + norm_res4 + norm_mass1 + norm_mass2
                            
                            # Compute new weights by normalizing the gradient norms
                            lambda_data_new = norm_data / sum_norms
                            lambda_res1_new = norm_res1 / sum_norms
                            lambda_res2_new = norm_res2 / sum_norms
                            lambda_res3_new = norm_res3 / sum_norms
                            lambda_res4_new = norm_res4 / sum_norms
                            lambda_mass1_new = norm_mass1 / sum_norms
                            lambda_mass2_new = norm_mass2 / sum_norms
                            
                            # Update weights with exponential moving average
                            lambda_data = alpha * lambda_data + (1 - alpha) * lambda_data_new.detach()
                            lambda_res1 = alpha * lambda_res1 + (1 - alpha) * lambda_res1_new.detach()
                            lambda_res2 = alpha * lambda_res2 + (1 - alpha) * lambda_res2_new.detach()
                            lambda_res3 = alpha * lambda_res3 + (1 - alpha) * lambda_res3_new.detach()
                            lambda_res4 = alpha * lambda_res4 + (1 - alpha) * lambda_res4_new.detach()
                            lambda_mass1 = alpha * lambda_mass1 + (1 - alpha) * lambda_mass1_new.detach()
                            lambda_mass2 = alpha * lambda_mass2 + (1 - alpha) * lambda_mass2_new.detach()
                        
                        # Apply manual parameter update with gradient clipping
                        with torch.no_grad():
                            for param, grad in zip(model.parameters(), grads):
                                if grad is not None:
                                    # Apply gradient clipping
                                    grad = torch.clamp(grad, -1.0, 1.0)
                                    param -= current_lr * grad
                    
                    except RuntimeError as e:
                        print(f"Warning: Runtime error during gradient computation: {e}")
                        continue
                
                # Accumulate loss values
                train_loss += total_loss.item()
                train_data_loss += data_loss.item()
                train_res1_loss += res1_loss.item()
                train_res2_loss += res2_loss.item()
                train_res3_loss += res3_loss.item()
                train_res4_loss += res4_loss.item()
                train_mass1_loss += mass1_loss.item()
                train_mass2_loss += mass2_loss.item()
                train_batches += 1
                
            except Exception as e:
                print(f"Warning: Unexpected error during training: {e}")
                continue
        
        # Calculate average training losses
        if train_batches > 0:
            avg_train_loss = train_loss / train_batches
            avg_data_loss = train_data_loss / train_batches
            avg_res1_loss = train_res1_loss / train_batches
            avg_res2_loss = train_res2_loss / train_batches
            avg_res3_loss = train_res3_loss / train_batches
            avg_res4_loss = train_res4_loss / train_batches
            avg_mass1_loss = train_mass1_loss / train_batches
            avg_mass2_loss = train_mass2_loss / train_batches
        else:
            print("Warning: No valid training batches in this epoch")
            avg_train_loss = float('inf')
            avg_data_loss = float('inf')
            avg_res1_loss = float('inf')
            avg_res2_loss = float('inf')
            avg_res3_loss = float('inf')
            avg_res4_loss = float('inf')
            avg_mass1_loss = float('inf')
            avg_mass2_loss = float('inf')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_data_loss = 0.0
        val_res1_loss = 0.0
        val_res2_loss = 0.0
        val_res3_loss = 0.0
        val_res4_loss = 0.0
        val_mass1_loss = 0.0
        val_mass2_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                try:
                    # Get loss components
                    data_loss, res1_loss, res2_loss, res3_loss, res4_loss, mass1_loss, mass2_loss = custom_loss_fn(
                        model, X_batch, y_batch, X_max, X_min, y_max, y_min
                    )
                    
                    # Calculate weighted loss
                    loss = (lambda_data * data_loss + 
                           lambda_res1 * res1_loss + 
                           lambda_res2 * res2_loss + 
                           lambda_res3 * res3_loss + 
                           lambda_res4 * res4_loss + 
                           lambda_mass1 * mass1_loss + 
                           lambda_mass2 * mass2_loss)
                    
                    # Skip invalid values
                    if not torch.isfinite(loss):
                        continue
                    
                    # Accumulate validation losses
                    val_loss += loss.item()
                    val_data_loss += data_loss.item()
                    val_res1_loss += res1_loss.item()
                    val_res2_loss += res2_loss.item()
                    val_res3_loss += res3_loss.item()
                    val_res4_loss += res4_loss.item()
                    val_mass1_loss += mass1_loss.item()
                    val_mass2_loss += mass2_loss.item()
                    val_batches += 1
                    
                except Exception as e:
                    print(f"Warning: Error during validation: {e}")
                    continue
        
        # Calculate average validation losses
        if val_batches > 0:
            avg_val_loss = val_loss / val_batches
            avg_val_data_loss = val_data_loss / val_batches
            avg_val_res1_loss = val_res1_loss / val_batches
            avg_val_res2_loss = val_res2_loss / val_batches
            avg_val_res3_loss = val_res3_loss / val_batches
            avg_val_res4_loss = val_res4_loss / val_batches
            avg_val_mass1_loss = val_mass1_loss / val_batches
            avg_val_mass2_loss = val_mass2_loss / val_batches
        else:
            print("Warning: No valid validation batches in this epoch")
            avg_val_loss = float('inf')
            avg_val_data_loss = float('inf')
            avg_val_res1_loss = float('inf')
            avg_val_res2_loss = float('inf')
            avg_val_res3_loss = float('inf')
            avg_val_res4_loss = float('inf')
            avg_val_mass1_loss = float('inf')
            avg_val_mass2_loss = float('inf')
        
        # Update training history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['data_weight'].append(lambda_data.item())
        history['res1_weight'].append(lambda_res1.item())
        history['res2_weight'].append(lambda_res2.item())
        history['res3_weight'].append(lambda_res3.item())
        history['res4_weight'].append(lambda_res4.item())
        history['mass1_weight'].append(lambda_mass1.item())
        history['mass2_weight'].append(lambda_mass2.item())
        history['data_loss'].append(avg_data_loss)
        history['res1_loss'].append(avg_res1_loss)
        history['res2_loss'].append(avg_res2_loss)
        history['res3_loss'].append(avg_res3_loss)
        history['res4_loss'].append(avg_res4_loss)
        history['mass1_loss'].append(avg_mass1_loss)
        history['mass2_loss'].append(avg_mass2_loss)
        history['learning_rate'].append(current_lr)
        
        # Group weights for display purposes
        phys_weight = lambda_res1 + lambda_res2 + lambda_res3 + lambda_res4
        mass_weight = lambda_mass1 + lambda_mass2
        
        # Update progress bar
        pbar.set_postfix({
            'train': f'{avg_train_loss:.2e}',
            'val': f'{avg_val_loss:.2e}',
            'λd': f'{lambda_data:.2f}',
            'λp': f'{phys_weight:.2f}',
            'λm': f'{mass_weight:.2f}',
            'lr': f'{current_lr:.2e}'
        })
        
        # Best model checkpointing
        if avg_val_loss < best_val_loss:
            print(f"\nEpoch {epoch+1}: Validation loss improved from {best_val_loss:.6e} to {avg_val_loss:.6e}")
            best_val_loss = avg_val_loss
            patience_counter = 0
            lr_counter = 0
            
            # Save model
            try:
                torch.save(model.state_dict(), model_save_path)
                has_saved_model = True
                print(f"Model saved to {model_save_path}")
                
                # Print current parameters
                print("Current physical parameters:")
                for name, param in model.named_parameters():
                    if not name.startswith('NNforUnmeasured') and not name.startswith('NNforAccelerations'):
                        print(f"  {name}: {param.item():.6f}")
            except Exception as e:
                print(f"Warning: Could not save model: {e}")
        else:
            patience_counter += 1
            lr_counter += 1
            
            # Learning rate scheduling
            if lr_counter >= lr_patience:
                old_lr = current_lr
                current_lr = max(current_lr * 0.5, 1e-6)  # Reduce by half, but not below 1e-6
                lr_counter = 0
                
                if old_lr != current_lr:
                    print(f"\nEpoch {epoch+1}: Reducing learning rate from {old_lr:.2e} to {current_lr:.2e}")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs without improvement")
                break
        
        # Generate more detailed log every few epochs
        if epoch % 10 == 0:
            print(f"\nEpoch {epoch+1} details:")
            print(f"  Data loss: {avg_data_loss:.4e} (λ={lambda_data:.2f})")
            print(f"  Physics residuals: {avg_res1_loss:.4e}, {avg_res2_loss:.4e}, {avg_res3_loss:.4e}, {avg_res4_loss:.4e}")
            print(f"  Mass residuals: {avg_mass1_loss:.4e}, {avg_mass2_loss:.4e}")
    
    # Load the best model if it was saved
    if has_saved_model:
        try:
            model.load_state_dict(torch.load(model_save_path))
            print(f"Loaded best model from {model_save_path}")
        except Exception as e:
            print(f"Warning: Could not load best model: {e}")
    
    # Create combined history from individual components
    history['phys_weight'] = [r1+r2+r3+r4 for r1,r2,r3,r4 in zip(
        history['res1_weight'], history['res2_weight'], 
        history['res3_weight'], history['res4_weight'])]
    
    history['reg_weight'] = [m1+m2 for m1,m2 in zip(
        history['mass1_weight'], history['mass2_weight'])]
    
    history['phys_loss'] = [(r1+r2+r3+r4)/4 for r1,r2,r3,r4 in zip(
        history['res1_loss'], history['res2_loss'], 
        history['res3_loss'], history['res4_loss'])]
    
    history['reg_loss'] = [(m1+m2)/2 for m1,m2 in zip(
        history['mass1_loss'], history['mass2_loss'])]
    
    # Print final model parameters
    print("\nFinal optimized model parameters:")
    for name, param in model.named_parameters():
        if not name.startswith('NNforUnmeasured') and not name.startswith('NNforAccelerations'):
            print(f"{name}: {param.item():.6f}")
    
    return model, history 