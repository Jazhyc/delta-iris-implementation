"""
Tokenizer Utility Functions
Helper functions for spatial grids and token manipulation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Union
import math


def create_spatial_grid(obs: torch.Tensor, grid_size: int, patch_size: Optional[int] = None) -> torch.Tensor:
    """
    Create spatial grid representation from observations
    
    Args:
        obs: [batch_size, seq_len, ...] observations
        grid_size: Size of spatial grid (grid_size x grid_size patches)
        patch_size: Size of each patch (for image observations)
        
    Returns:
        spatial_grid: [batch_size, seq_len, grid_size^2, patch_dim] spatial grid
    """
    if obs.dim() == 3:
        # Vector observations
        return _create_vector_spatial_grid(obs, grid_size)
    elif obs.dim() == 5:
        # Image observations [batch, seq, C, H, W]
        return _create_image_spatial_grid(obs, grid_size, patch_size)
    else:
        raise ValueError(f"Unsupported observation shape: {obs.shape}")


def _create_vector_spatial_grid(obs: torch.Tensor, grid_size: int) -> torch.Tensor:
    """Create spatial grid from vector observations"""
    batch_size, seq_len, obs_dim = obs.shape
    num_patches = grid_size * grid_size
    
    # Create pseudo-spatial representation by repeating observations
    spatial_grid = obs.unsqueeze(-2).repeat(1, 1, num_patches, 1)  # [batch, seq, patches, obs_dim]
    
    # Add learnable positional variations
    pos_encoding = _create_positional_encoding(num_patches, obs_dim, obs.device)
    pos_encoding = pos_encoding.unsqueeze(0).unsqueeze(0)  # [1, 1, patches, obs_dim]
    
    # Scale positional encoding to be subtle
    spatial_grid = spatial_grid + 0.1 * pos_encoding
    
    return spatial_grid


def _create_image_spatial_grid(obs: torch.Tensor, grid_size: int, patch_size: Optional[int] = None) -> torch.Tensor:
    """Create spatial grid from image observations"""
    batch_size, seq_len, C, H, W = obs.shape
    
    if patch_size is None:
        patch_size = min(H, W) // grid_size
    
    # Reshape for batch processing
    obs_flat = obs.view(batch_size * seq_len, C, H, W)
    
    # Extract non-overlapping patches using unfold
    patches = F.unfold(obs_flat, kernel_size=patch_size, stride=patch_size)
    # patches shape: [batch*seq, C*patch_size^2, num_patches]
    
    # Reshape to desired format
    patches = patches.transpose(1, 2)  # [batch*seq, num_patches, C*patch_size^2]
    
    # Take only the first grid_size^2 patches
    num_patches_available = patches.shape[1]
    target_patches = grid_size * grid_size
    
    if num_patches_available >= target_patches:
        patches = patches[:, :target_patches]
    else:
        # Pad if we don't have enough patches
        padding = torch.zeros(
            batch_size * seq_len, target_patches - num_patches_available, patches.shape[2],
            device=patches.device, dtype=patches.dtype
        )
        patches = torch.cat([patches, padding], dim=1)
    
    # Reshape back to batch format
    patches = patches.view(batch_size, seq_len, target_patches, -1)
    
    return patches


def _create_positional_encoding(num_positions: int, embed_dim: int, device: torch.device) -> torch.Tensor:
    """Create sinusoidal positional encoding"""
    position = torch.arange(num_positions, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float, device=device) * 
                        -(math.log(10000.0) / embed_dim))
    
    pos_encoding = torch.zeros(num_positions, embed_dim, device=device)
    pos_encoding[:, 0::2] = torch.sin(position * div_term)
    if embed_dim > 1:
        pos_encoding[:, 1::2] = torch.cos(position * div_term[:embed_dim//2])
    
    return pos_encoding


def extract_spatial_tokens(spatial_repr: torch.Tensor, 
                          method: str = 'pool',
                          attention_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Extract tokens from spatial representation
    
    Args:
        spatial_repr: [batch_size, seq_len, grid_size^2, latent_dim] spatial representation
        method: Extraction method ('pool', 'flatten', 'attention', 'learned')
        attention_weights: Optional attention weights for weighted pooling
        
    Returns:
        tokens: Extracted tokens
    """
    batch_size, seq_len, num_patches, latent_dim = spatial_repr.shape
    
    if method == 'pool':
        # Simple average pooling
        return spatial_repr.mean(dim=2)  # [batch, seq_len, latent_dim]
    
    elif method == 'flatten':
        # Flatten all spatial tokens
        return spatial_repr.view(batch_size, seq_len, num_patches * latent_dim)
    
    elif method == 'attention':
        # Attention-weighted pooling
        if attention_weights is None:
            # Compute attention weights based on magnitude
            attention_weights = torch.norm(spatial_repr, dim=-1)  # [batch, seq_len, num_patches]
            attention_weights = F.softmax(attention_weights, dim=-1)
        
        # Apply attention weights
        weighted_repr = spatial_repr * attention_weights.unsqueeze(-1)
        return weighted_repr.sum(dim=2)  # [batch, seq_len, latent_dim]
    
    elif method == 'learned':
        # This would require a learned pooling module
        raise NotImplementedError("Learned pooling requires a pooling module")
    
    else:
        raise ValueError(f"Unknown extraction method: {method}")


def reconstruct_from_spatial_tokens(tokens: torch.Tensor, 
                                   grid_size: int,
                                   target_shape: Tuple[int, ...],
                                   method: str = 'linear') -> torch.Tensor:
    """
    Reconstruct observations from spatial tokens
    
    Args:
        tokens: Spatial tokens (various shapes depending on extraction method)
        grid_size: Original spatial grid size
        target_shape: Target observation shape
        method: Reconstruction method ('linear', 'conv', 'learned')
        
    Returns:
        reconstructed: Reconstructed observations
    """
    if method == 'linear':
        # Simple linear reconstruction
        if tokens.dim() == 3:  # [batch, seq_len, token_dim]
            batch_size, seq_len, token_dim = tokens.shape
            
            # Linear projection to target dimension
            target_dim = math.prod(target_shape[2:]) if len(target_shape) > 2 else target_shape[-1]
            
            # Create a simple linear layer (this would be learned in practice)
            linear_proj = nn.Linear(token_dim, target_dim).to(tokens.device)
            reconstructed = linear_proj(tokens)
            
            # Reshape to target shape
            if len(target_shape) > 2:
                reconstructed = reconstructed.view(batch_size, seq_len, *target_shape[2:])
            
            return reconstructed
    
    else:
        raise NotImplementedError(f"Reconstruction method '{method}' not implemented")


def compute_spatial_attention(spatial_tokens: torch.Tensor, 
                             query: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute spatial attention over token grid
    
    Args:
        spatial_tokens: [batch_size, seq_len, grid_size^2, latent_dim] spatial tokens
        query: Optional query tensor for attention
        
    Returns:
        attended_tokens: Attention-weighted tokens
        attention_weights: Attention weights [batch, seq_len, grid_size^2]
    """
    batch_size, seq_len, num_patches, latent_dim = spatial_tokens.shape
    
    if query is None:
        # Self-attention: use mean of tokens as query
        query = spatial_tokens.mean(dim=2, keepdim=True)  # [batch, seq_len, 1, latent_dim]
    
    # Compute attention scores
    # Simple dot-product attention
    scores = torch.matmul(query, spatial_tokens.transpose(-2, -1))  # [batch, seq_len, 1, num_patches]
    scores = scores.squeeze(-2)  # [batch, seq_len, num_patches]
    
    # Apply softmax
    attention_weights = F.softmax(scores / math.sqrt(latent_dim), dim=-1)
    
    # Apply attention
    attended_tokens = torch.matmul(
        attention_weights.unsqueeze(-2), spatial_tokens
    ).squeeze(-2)  # [batch, seq_len, latent_dim]
    
    return attended_tokens, attention_weights


def create_token_mask(tokens: torch.Tensor, 
                     mask_ratio: float = 0.15,
                     mask_value: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create random mask for token sequences (useful for training)
    
    Args:
        tokens: [batch_size, seq_len, ...] token sequence
        mask_ratio: Ratio of tokens to mask
        mask_value: Value to use for masked tokens
        
    Returns:
        masked_tokens: Tokens with masking applied
        mask: Boolean mask indicating masked positions
    """
    batch_size, seq_len = tokens.shape[:2]
    
    # Create random mask
    mask = torch.rand(batch_size, seq_len, device=tokens.device) < mask_ratio
    
    # Apply mask
    masked_tokens = tokens.clone()
    masked_tokens[mask] = mask_value
    
    return masked_tokens, mask


def interpolate_spatial_tokens(tokens1: torch.Tensor, 
                              tokens2: torch.Tensor, 
                              alpha: float) -> torch.Tensor:
    """
    Interpolate between two spatial token representations
    
    Args:
        tokens1: First token representation
        tokens2: Second token representation  
        alpha: Interpolation factor (0 = tokens1, 1 = tokens2)
        
    Returns:
        interpolated: Interpolated tokens
    """
    return (1 - alpha) * tokens1 + alpha * tokens2


class SpatialTokenVisualizer:
    """Utility class for visualizing spatial tokens (for debugging)"""
    
    @staticmethod
    def visualize_attention(attention_weights: torch.Tensor, 
                           grid_size: int, 
                           save_path: Optional[str] = None) -> torch.Tensor:
        """
        Visualize spatial attention weights as heatmaps
        
        Args:
            attention_weights: [batch_size, seq_len, grid_size^2] attention weights
            grid_size: Size of spatial grid
            save_path: Optional path to save visualization
            
        Returns:
            heatmaps: [batch_size, seq_len, grid_size, grid_size] attention heatmaps
        """
        batch_size, seq_len, num_patches = attention_weights.shape
        assert num_patches == grid_size * grid_size, "Attention weights don't match grid size"
        
        # Reshape to grid format
        heatmaps = attention_weights.view(batch_size, seq_len, grid_size, grid_size)
        
        # Optional: save visualization
        if save_path:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, min(seq_len, 5), figsize=(15, 3))
            if seq_len == 1:
                axes = [axes]
            
            for i in range(min(seq_len, 5)):
                axes[i].imshow(heatmaps[0, i].cpu().numpy(), cmap='hot', interpolation='nearest')
                axes[i].set_title(f'Timestep {i}')
                axes[i].axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
        
        return heatmaps
    
    @staticmethod
    def analyze_token_statistics(tokens: torch.Tensor) -> dict:
        """
        Analyze statistical properties of tokens
        
        Args:
            tokens: Token tensor
            
        Returns:
            stats: Dictionary of token statistics
        """
        stats = {
            'mean': tokens.mean().item(),
            'std': tokens.std().item(),
            'min': tokens.min().item(),
            'max': tokens.max().item(),
            'num_unique': len(torch.unique(tokens)) if tokens.dtype in [torch.long, torch.int] else None,
        }
        
        return stats
