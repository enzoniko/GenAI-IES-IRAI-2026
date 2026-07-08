"""
Trainer for Siamese networks
"""
import torch
import torch.optim as optim
import numpy as np
import os
import copy
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Any, Optional
import time
import logging
import torch.nn as nn

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=5, min_delta=0.0, verbose=True):
        """
        Early stopping to prevent overfitting
        
        Args:
            patience: Number of epochs to wait after validation loss has stopped improving
            min_delta: Minimum change in validation loss to be considered as improvement
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        
    def __call__(self, val_loss):
        """
        Check if training should be stopped
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            True if training should be stopped, False otherwise
        """
        score = -val_loss
        
        if self.best_score is None:
            # First epoch
            self.best_score = score
            return False
        
        if score < self.best_score + self.min_delta:
            # Validation loss increased or didn't improve enough
            self.counter += 1
            if self.verbose:
                logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            # Validation loss improved
            self.best_score = score
            self.counter = 0
            return False
        
        return False


class ModelCheckpoint:
    """Save the best model during training"""
    
    def __init__(self, filepath, save_best_only=True, verbose=True):
        """
        Save the best model during training
        
        Args:
            filepath: Path to save the model
            save_best_only: Only save if the model is better than the previous best
            verbose: Whether to print messages
        """
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.verbose = verbose
        self.best_val_loss = np.inf
        self.best_model_state = None
        
    def __call__(self, model, val_loss):
        """
        Check if model should be saved
        
        Args:
            model: Model to save
            val_loss: Current validation loss
            
        Returns:
            True if model was saved, False otherwise
        """
        if not self.save_best_only:
            # Save model regardless of performance
            self._save_model(model)
            return True
        
        if val_loss < self.best_val_loss:
            # Validation loss improved, save model
            if self.verbose:
                logger.info(f'Validation loss improved from {self.best_val_loss:.4f} to {val_loss:.4f}')
            
            self.best_val_loss = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
            
            # Save the best model to disk
            self._save_model(model, is_best=True)
            return True
        
        return False
    
    def _save_model(self, model, is_best=False):
        """Save model to disk"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        
        # Save the current model
        if not is_best and self.save_best_only:
            # If we're only saving the best model and this isn't it, don't save
            return
            
        # For best model with save_best_only=True or any model with save_best_only=False
        torch.save(model.state_dict(), self.filepath)
        
        # Also save with 'best' in the filename if it's the best model
        if is_best:
            best_filepath = self.filepath.replace('.pt', '_best.pt')
            # Save the best model state, not the current model state
            if self.best_model_state is not None:
                torch.save(self.best_model_state, best_filepath)
            else:
                # Fallback to current model state if best_model_state is not set
                torch.save(model.state_dict(), best_filepath)
            
            if self.verbose:
                logger.info(f'Best model saved to {best_filepath}')
        elif self.verbose:
            logger.info(f'Model saved to {self.filepath}')
    
    def restore_best_model(self, model):
        """
        Restore the best model state
        
        Args:
            model: Model to restore
        """
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            if self.verbose:
                logger.info(f'Restored best model with validation loss {self.best_val_loss:.4f}')
            return model
        else:
            logger.warning('No best model state to restore')
            return model


class SiameseTrainer:
    """Trainer for Siamese networks"""
    
    def __init__(self, model, device, model_kwargs, criterion=None, optimizer=None, lr=0.001, lr_scheduler=None, warmup_epochs=0):
        """
        Trainer for Siamese network
        
        Args:
            model: SiameseNetwork instance
            device: torch.device for training
            model_kwargs: Dictionary of arguments used to initialize the model
            criterion: Loss function (defaults to TripletLoss)
            optimizer: Optimizer (defaults to Adam)
            lr: Learning rate
            lr_scheduler: Learning rate scheduler (defaults to None)
            warmup_epochs: Number of warmup epochs for learning rate (defaults to 0)
        """
        from siamese_analysis_v3.models import TripletLoss
        
        self.model = model.to(device)
        self.device = device
        self.model_kwargs = model_kwargs  # Store model initialization arguments
        self.criterion = criterion or TripletLoss()
        self.optimizer = optimizer or optim.Adam(model.parameters(), lr=lr)
        self.lr_scheduler = lr_scheduler
        if self.lr_scheduler is None:
            # Default to ReduceLROnPlateau if not provided
            self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5, 
                verbose=True, min_lr=1e-6
            )
        self.best_model_state = None
        self.best_val_loss = np.inf
        self.warmup_epochs = warmup_epochs
        self.initial_lr = lr
    
    def train_epoch(self, dataloader, epoch=None):
        """
        Train for one epoch
        
        Args:
            dataloader: DataLoader for training
            epoch: Current epoch number (for warmup calculation)
            
        Returns:
            Mean loss for the epoch
        """
        self.model.train()
        running_loss = 0.0
        
        # Apply warmup learning rate if in warmup phase
        if epoch is not None and self.warmup_epochs > 0 and epoch < self.warmup_epochs:
            # Linear warmup
            warmup_factor = (epoch + 1) / self.warmup_epochs
            current_lr = self.initial_lr * warmup_factor
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr
            
        for batch in tqdm(dataloader, desc="Training", leave=False, position=1):
            # Get data
            anchor = batch['anchor'].to(self.device)
            positive = batch['positive'].to(self.device)
            negative = batch['negative'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            anchor_embedding = self.model.forward_one(anchor)
            positive_embedding = self.model.forward_one(positive)
            negative_embedding = self.model.forward_one(negative)
            
            # Compute loss
            loss = self.criterion(anchor_embedding, positive_embedding, negative_embedding)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
        
        return running_loss / len(dataloader)
    
    def evaluate(self, dataloader):
        """
        Evaluate the model
        
        Args:
            dataloader: DataLoader for evaluation
            
        Returns:
            Mean loss for the evaluation
        """
        self.model.eval()
        running_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", leave=False, position=1):
                # Get data
                anchor = batch['anchor'].to(self.device)
                positive = batch['positive'].to(self.device)
                negative = batch['negative'].to(self.device)
                
                # Forward pass
                anchor_embedding = self.model.forward_one(anchor)
                positive_embedding = self.model.forward_one(positive)
                negative_embedding = self.model.forward_one(negative)
                
                # Compute loss
                loss = self.criterion(anchor_embedding, positive_embedding, negative_embedding)
                
                running_loss += loss.item()
        
        return running_loss / len(dataloader)
    
    def train(self, train_dataloader, val_dataloader=None, epochs=100, verbose=1, fold=None):
        """
        Train the model
        
        Args:
            train_dataloader: DataLoader for training
            val_dataloader: Optional validation DataLoader
            epochs: Number of epochs to train for
            verbose: Verbosity level (0: silent, 1: one progress bar, 2: one progress bar per epoch)
            fold: Current fold number for display in progress bar (optional)
            
        Returns:
            Dictionary with training history
        """
        history = {'train_loss': [], 'val_loss': []}
        
        # Set up progress bar for epochs
        epoch_iterator = range(epochs)
        if verbose >= 1:
            # Include fold number in description if provided
            desc = f"Fold {fold+1} Epochs" if fold is not None else "Epochs"
            epoch_iterator = tqdm(epoch_iterator, desc=desc, leave=True, position=0)
            
        # Reset learning rate to initial value after warmup
        if self.warmup_epochs > 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.initial_lr * 0.1  # Start with lower LR before warmup
        
        for epoch in epoch_iterator:
            # Train for one epoch
            train_loss = self.train_epoch(train_dataloader, epoch=epoch)
            history['train_loss'].append(train_loss)
            
            # Validate
            if val_dataloader is not None:
                val_loss = self.evaluate(val_dataloader)
                history['val_loss'].append(val_loss)
                
                # Skip LR scheduler during warmup
                if self.warmup_epochs == 0 or epoch >= self.warmup_epochs:
                    # Step learning rate scheduler
                    if self.lr_scheduler is not None:
                        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.lr_scheduler.step(val_loss)
                        else:
                            self.lr_scheduler.step()
                
                # Update best model if validation loss improved
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_model_state = copy.deepcopy(self.model.state_dict())
                
                if verbose >= 1:
                    epoch_iterator.set_postfix(
                        {'train_loss': f'{train_loss:.4f}', 'val_loss': f'{val_loss:.4f}'}
                    )
            else:
                # Step learning rate scheduler without validation
                if self.warmup_epochs == 0 or epoch >= self.warmup_epochs:
                    if self.lr_scheduler is not None and not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.lr_scheduler.step()
                
                if verbose >= 1:
                    epoch_iterator.set_postfix({'train_loss': f'{train_loss:.4f}'})
            
            # Print epoch results if not using progress bar
            if verbose == 0 and (epoch + 1) % 10 == 0:  # Print every 10 epochs if not using progress bar
                if val_dataloader is not None:
                    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")
        
        return history
    
    def train_with_cross_validation(self, cv_splits, epochs=100, callbacks=None, verbose=1, 
                                   batch_size=64, data_loader_workers=4):
        """
        Train with cross-validation
        
        Args:
            cv_splits: List of (train_dataset, val_dataset) tuples
            epochs: Maximum number of epochs
            callbacks: List of callbacks
            verbose: Verbosity level (0=silent, 1=progress bar, 2=one line per epoch)
            batch_size: Batch size for training and validation
            data_loader_workers: Number of worker processes for data loading
            
        Returns:
            List of training histories (one per fold)
        """
        histories = []
        
        for fold, (train_dataset, val_dataset) in enumerate(cv_splits):
            if verbose >= 1:
                logger.info(f"Fold {fold+1}/{len(cv_splits)}")
            
            # Reset model for each fold
            if fold > 0:
                # Get model class and parameters
                model_class = self.model.__class__
                # Use stored model_kwargs for re-instantiation
                
                # Create new model instance
                self.model = model_class(**self.model_kwargs).to(self.device)
                
                # Get optimizer class and parameters
                optimizer_class = self.optimizer.__class__
                optimizer_kwargs = {'lr': self.initial_lr}  # Add any other parameters you need
                
                # Create new optimizer instance
                self.optimizer = optimizer_class(self.model.parameters(), **optimizer_kwargs)
                
                # Reset LR scheduler if using
                if isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    # Create ReduceLROnPlateau scheduler with platform-specific verbose setting
                    import platform
                    verbose_lr = platform.system() != 'Windows'  # Enable verbose on non-Windows systems

                    try:
                        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                            self.optimizer, mode='min', factor=0.5, patience=5,
                            verbose=verbose_lr, min_lr=1e-6
                        )
                    except TypeError:
                        # Fallback for older PyTorch versions or Windows
                        try:
                            self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                                self.optimizer, mode='min', factor=0.5, patience=5,
                                verbose=verbose_lr
                            )
                        except TypeError:
                            # Final fallback without verbose
                            self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                                self.optimizer, mode='min', factor=0.5, patience=5
                            )
                elif isinstance(self.lr_scheduler, optim.lr_scheduler._LRScheduler):
                    lr_scheduler_class = self.lr_scheduler.__class__
                    lr_scheduler_kwargs = {}  # Add parameters as needed
                    self.lr_scheduler = lr_scheduler_class(self.optimizer, **lr_scheduler_kwargs)
            
            # Adjust num_workers based on platform capabilities
            import platform
            system = platform.system()
            if system == 'Windows':
                # Windows has multiprocessing limitations
                effective_workers = min(1, data_loader_workers)
                logger.warning(f"Reducing data loader workers from {data_loader_workers} to {effective_workers} for Windows compatibility")
            elif system in ['Linux', 'Darwin']:  # Linux/macOS
                # Unix-like systems can handle multiprocessing well
                effective_workers = data_loader_workers
                if data_loader_workers > 0:
                    logger.info(f"Using {effective_workers} data loader workers on {system}")
            else:
                # Unknown platform, use conservative approach
                effective_workers = min(2, data_loader_workers)
                logger.warning(f"Unknown platform {system}, using conservative multiprocessing settings")

            # Create data loaders for this fold with error handling
            try:
                train_dataloader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=effective_workers,
                    pin_memory=True,
                    drop_last=True  # To ensure triplets are always available
                )
            except Exception as e:
                logger.warning(f"Failed to create train DataLoader with {effective_workers} workers: {e}")
                logger.warning("Falling back to single-threaded data loading")
                train_dataloader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=0,  # Disable multiprocessing
                    pin_memory=False,
                    drop_last=True
                )

            try:
                val_dataloader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=effective_workers,
                    pin_memory=True
                )
            except Exception as e:
                logger.warning(f"Failed to create val DataLoader with {effective_workers} workers: {e}")
                logger.warning("Falling back to single-threaded data loading")
                val_dataloader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0,  # Disable multiprocessing
                    pin_memory=False
                )
            
            # Create checkpoint for this fold
            if callbacks is None:
                callbacks = []
                
            fold_checkpoint_filepath = f"fold_{fold+1}_checkpoint.pt"
            fold_checkpoint = ModelCheckpoint(
                filepath=fold_checkpoint_filepath,
                save_best_only=True,
                verbose=(verbose >= 2)
            )
            callbacks.append(fold_checkpoint)
            
            # Add early stopping
            early_stopping = EarlyStopping(
                patience=10,  # Can be adjusted
                verbose=(verbose >= 2)
            )
            callbacks.append(early_stopping)
            
            # Train for this fold
            history = self.train(
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                epochs=epochs,
                verbose=verbose,
                fold=fold+1
            )
            
            # Restore best model before moving to next fold
            fold_checkpoint.restore_best_model(self.model)
            
            # Save history for this fold
            histories.append(history)
            
            # Delete checkpoint file to save disk space
            if os.path.exists(fold_checkpoint_filepath):
                os.remove(fold_checkpoint_filepath)
            
            # Check if we should stop early
            if any(getattr(cb, 'early_stop', False) for cb in callbacks if hasattr(cb, 'early_stop')):
                logger.info("Early stopping triggered")
                break
        
        return histories
        
    def train_with_single_split(self, dataset, train_indices, val_indices, epochs=100, 
                             callbacks=None, verbose=1, batch_size=64, data_loader_workers=4):
        """
        Train with a single train/validation split
        
        This method trains the model using a single random train/validation split
        instead of full cross-validation, which is more efficient for multiple evaluation runs.
        
        Args:
            dataset: The complete dataset
            train_indices: Indices for training samples
            val_indices: Indices for validation samples
            epochs: Maximum number of epochs
            callbacks: List of callbacks
            verbose: Verbosity level (0=silent, 1=progress bar, 2=one line per epoch)
            batch_size: Batch size for training and validation
            data_loader_workers: Number of worker processes for data loading
            
        Returns:
            Training history dictionary
        """
        from torch.utils.data import Subset
        
        # Reset model state for each run
        # Get model class and parameters
        model_class = self.model.__class__
        # Use stored model_kwargs for re-instantiation
        
        # Create new model instance with same architecture but freshly initialized weights
        self.model = model_class(**self.model_kwargs).to(self.device)
        
        # Reset optimizer
        optimizer_class = self.optimizer.__class__
        optimizer_kwargs = {'lr': self.initial_lr}
        self.optimizer = optimizer_class(self.model.parameters(), **optimizer_kwargs)
        
        # Reset LR scheduler if using
        if isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            # Create ReduceLROnPlateau scheduler with platform-specific verbose setting
            import platform
            verbose_lr = platform.system() != 'Windows'  # Enable verbose on non-Windows systems

            try:
                self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode='min', factor=0.5, patience=5,
                    verbose=verbose_lr, min_lr=1e-6
                )
            except TypeError:
                # Fallback for older PyTorch versions or Windows
                try:
                    self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                        self.optimizer, mode='min', factor=0.5, patience=5,
                        verbose=verbose_lr
                    )
                except TypeError:
                    # Final fallback without verbose
                    self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                        self.optimizer, mode='min', factor=0.5, patience=5
                    )
        elif isinstance(self.lr_scheduler, optim.lr_scheduler._LRScheduler):
            lr_scheduler_class = self.lr_scheduler.__class__
            lr_scheduler_kwargs = {}  # Add parameters as needed
            self.lr_scheduler = lr_scheduler_class(self.optimizer, **lr_scheduler_kwargs)
        
        # Create subsets using the provided indices
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        
        # Create data loaders
        # Adjust num_workers based on platform capabilities
        import platform
        system = platform.system()
        if system == 'Windows':
            # Windows has multiprocessing limitations
            effective_workers = min(1, data_loader_workers)
            logger.warning(f"Reducing data loader workers from {data_loader_workers} to {effective_workers} for Windows compatibility")
        elif system in ['Linux', 'Darwin']:  # Linux/macOS
            # Unix-like systems can handle multiprocessing well
            effective_workers = data_loader_workers
            if data_loader_workers > 0:
                logger.info(f"Using {effective_workers} data loader workers on {system}")
        else:
            # Unknown platform, use conservative approach
            effective_workers = min(2, data_loader_workers)
            logger.warning(f"Unknown platform {system}, using conservative multiprocessing settings")

        # Create data loaders with error handling
        try:
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=effective_workers,
                pin_memory=True,
                drop_last=True  # To ensure triplets are always available
            )
        except Exception as e:
            logger.warning(f"Failed to create train DataLoader with {effective_workers} workers: {e}")
            logger.warning("Falling back to single-threaded data loading")
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,  # Disable multiprocessing
                pin_memory=False,
                drop_last=True
            )

        try:
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=effective_workers,
                pin_memory=True
            )
        except Exception as e:
            logger.warning(f"Failed to create val DataLoader with {effective_workers} workers: {e}")
            logger.warning("Falling back to single-threaded data loading")
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,  # Disable multiprocessing
                pin_memory=False
            )
        
        # Setup callbacks
        if callbacks is None:
            callbacks = []
            
        # Add model checkpoint
        checkpoint_filepath = "single_split_checkpoint.pt"
        checkpoint = ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_best_only=True,
            verbose=(verbose >= 2)
        )
        callbacks.append(checkpoint)
        
        # Add early stopping
        early_stopping = EarlyStopping(
            patience=10,  # Can be adjusted
            verbose=(verbose >= 2)
        )
        callbacks.append(early_stopping)
        
        # Train the model
        history = self.train(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            epochs=epochs,
            verbose=verbose
        )
        
        # Restore best model
        checkpoint.restore_best_model(self.model)
        
        # Delete checkpoint file to save disk space
        if os.path.exists(checkpoint_filepath):
            os.remove(checkpoint_filepath)
        
        return history
    
    def save_model(self, path):
        """Save the model"""
        # Only create directory if path has a directory component
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        """Load the model"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        return self 