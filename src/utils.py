"""
Utility classes and functions for Delta-IRIS implementation
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, Any, Union


class LossWithIntermediateLosses:
    """Container for losses with intermediate components"""
    def __init__(self, **kwargs) -> None:
        self.loss_total = sum(kwargs.values())
        self.intermediate_losses = {k: v.item() for k, v in kwargs.items()}


def init_weights(module: nn.Module) -> None:
    """Initialize model weights"""
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility"""
    import random
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    

def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, 
                   epoch: int, loss: float, path: Union[str, Path]) -> None:
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, path)


def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, 
                   path: Union[str, Path]) -> Dict[str, Any]:
    """Load model checkpoint"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return {
        'epoch': checkpoint['epoch'],
        'loss': checkpoint['loss']
    }


def plot_training_curves(train_losses: list, val_losses: list = None, 
                        save_path: Union[str, Path] = None) -> None:
    """Plot training and validation loss curves"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(train_losses, label='Training Loss', color='blue')
    if val_losses:
        ax.plot(val_losses, label='Validation Loss', color='red')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def get_device() -> torch.device:
    """Get the best available device"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():  # For Apple Silicon
        return torch.device('mps')
    else:
        return torch.device('cpu')


def print_model_summary(model: nn.Module, input_shape: tuple = None) -> None:
    """Print a summary of the model architecture"""
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {count_parameters(model):,}")
    
    if input_shape:
        # Create dummy input to get output shape
        dummy_input = torch.randn(1, *input_shape)
        try:
            with torch.no_grad():
                output = model(dummy_input)
            if isinstance(output, dict):
                print("Output shapes:")
                for key, value in output.items():
                    print(f"  {key}: {value.shape}")
            else:
                print(f"Output shape: {output.shape}")
        except Exception as e:
            print(f"Could not compute output shape: {e}")
    
    print("-" * 50)


class EarlyStopping:
    """Early stopping utility to prevent overfitting"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, 
                 restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop
        
        Returns:
            True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False


class RunningAverage:
    """Compute running average of values"""
    
    def __init__(self):
        self.total = 0.0
        self.count = 0
        
    def update(self, value: float) -> None:
        """Update with new value"""
        self.total += value
        self.count += 1
        
    @property
    def average(self) -> float:
        """Get current average"""
        return self.total / self.count if self.count > 0 else 0.0
    
    def reset(self) -> None:
        """Reset the running average"""
        self.total = 0.0
        self.count = 0
