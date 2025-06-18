"""
Spatial Tokenizer Module
Specialized spatial tokenization utilities for Delta-IRIS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


def create_spatial_grid(obs: torch.Tensor, grid_size: int) -> torch.Tensor:
    """
    Create spatial grid from observations
    
    Args:
        obs: [batch_size, seq_len, ...] observations
        grid_size: Size of spatial grid (grid_size x grid_size)
        
    Returns:
        spatial_grid: [batch_size, seq_len, grid_size^2, patch_dim]
    """
    if obs.dim() == 3:
        # Vector observations - create pseudo-spatial layout
        batch_size, seq_len, obs_dim = obs.shape
        # Repeat observation across spatial locations with slight variations
        grid_tokens = obs.unsqueeze(-2).repeat(1, 1, grid_size * grid_size, 1)
        # Add positional information
        pos_embed = create_positional_encoding(grid_size, obs_dim, obs.device)
        pos_embed = pos_embed.unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_len, 1, 1)
        grid_tokens = grid_tokens + 0.1 * pos_embed  # Slight positional variation
        return grid_tokens
    
    elif obs.dim() == 5:
        # Image observations - extract patches
        batch_size, seq_len, C, H, W = obs.shape
        return extract_image_patches(obs, grid_size)
    
    else:
        raise ValueError(f"Unsupported observation dimensionality: {obs.dim()}")


def create_positional_encoding(grid_size: int, embed_dim: int, device: torch.device) -> torch.Tensor:
    """
    Create 2D positional encoding for spatial grid
    
    Args:
        grid_size: Size of spatial grid
        embed_dim: Embedding dimension
        device: Device to create tensor on
        
    Returns:
        pos_encoding: [grid_size^2, embed_dim] positional encoding
    """
    # Create 2D position indices
    y_pos = torch.arange(grid_size, device=device).float().unsqueeze(1).repeat(1, grid_size)
    x_pos = torch.arange(grid_size, device=device).float().unsqueeze(0).repeat(grid_size, 1)
    
    # Flatten to 1D
    y_pos = y_pos.view(-1)  # [grid_size^2]
    x_pos = x_pos.view(-1)  # [grid_size^2]
    
    # Create sinusoidal encoding
    pos_encoding = torch.zeros(grid_size * grid_size, embed_dim, device=device)
    
    # Fill with sinusoidal patterns
    div_term = torch.exp(torch.arange(0, embed_dim // 2, 2, device=device).float() * 
                        -(math.log(10000.0) / (embed_dim // 2)))
    
    if embed_dim >= 2:
        pos_encoding[:, 0::4] = torch.sin(y_pos.unsqueeze(1) * div_term[:embed_dim//4])
        pos_encoding[:, 1::4] = torch.cos(y_pos.unsqueeze(1) * div_term[:embed_dim//4])
    if embed_dim >= 4:
        pos_encoding[:, 2::4] = torch.sin(x_pos.unsqueeze(1) * div_term[:embed_dim//4])
        pos_encoding[:, 3::4] = torch.cos(x_pos.unsqueeze(1) * div_term[:embed_dim//4])
    
    return pos_encoding


def extract_image_patches(images: torch.Tensor, grid_size: int) -> torch.Tensor:
    """
    Extract spatial patches from images
    
    Args:
        images: [batch_size, seq_len, C, H, W] images
        grid_size: Number of patches per dimension
        
    Returns:
        patches: [batch_size, seq_len, grid_size^2, patch_dim]
    """
    batch_size, seq_len, C, H, W = images.shape
    
    # Reshape to process all images at once
    images_flat = images.view(batch_size * seq_len, C, H, W)
    
    # Use unfold to extract patches
    patch_h = H // grid_size
    patch_w = W // grid_size
    
    # Extract patches using unfold
    patches = images_flat.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)
    # patches shape: [batch*seq, C, grid_size, grid_size, patch_h, patch_w]
    
    # Reshape to get patches as vectors
    patches = patches.contiguous().view(
        batch_size * seq_len, C, grid_size * grid_size, patch_h * patch_w
    )
    # Transpose and flatten
    patches = patches.permute(0, 2, 1, 3).contiguous()  # [batch*seq, grid_size^2, C, patch_h*patch_w]
    patches = patches.view(batch_size * seq_len, grid_size * grid_size, -1)
    
    # Reshape back to original batch structure
    patches = patches.view(batch_size, seq_len, grid_size * grid_size, -1)
    
    return patches


def extract_spatial_tokens(spatial_repr: torch.Tensor, method: str = 'flatten') -> torch.Tensor:
    """
    Extract spatial tokens from spatial representation
    
    Args:
        spatial_repr: [batch_size, seq_len, grid_size^2, latent_dim] spatial representation
        method: Method for extraction ('flatten', 'pool', 'attention')
        
    Returns:
        tokens: Extracted tokens
    """
    if method == 'flatten':
        # Flatten spatial dimensions
        batch_size, seq_len, num_patches, latent_dim = spatial_repr.shape
        return spatial_repr.view(batch_size, seq_len, num_patches * latent_dim)
    
    elif method == 'pool':
        # Pool across spatial dimensions
        return spatial_repr.mean(dim=2)  # [batch, seq_len, latent_dim]
    
    elif method == 'attention':
        # Use attention to aggregate spatial information
        # This would require a proper attention mechanism
        # For now, use weighted average
        weights = torch.softmax(spatial_repr.sum(dim=-1), dim=-1)  # [batch, seq_len, num_patches]
        return torch.sum(spatial_repr * weights.unsqueeze(-1), dim=2)
    
    else:
        raise ValueError(f"Unknown extraction method: {method}")


class SpatialAttentionPool(nn.Module):
    """
    Spatial attention pooling for aggregating spatial tokens
    """
    
    def __init__(self, latent_dim: int, num_heads: int = 4):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        
        # Query for pooling
        self.query = nn.Parameter(torch.randn(1, 1, latent_dim))
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            latent_dim, num_heads, batch_first=True
        )
        
        # Layer norm
        self.norm = nn.LayerNorm(latent_dim)
    
    def forward(self, spatial_tokens: torch.Tensor) -> torch.Tensor:
        """
        Pool spatial tokens using attention
        
        Args:
            spatial_tokens: [batch_size, seq_len, num_patches, latent_dim]
            
        Returns:
            pooled_tokens: [batch_size, seq_len, latent_dim]
        """
        batch_size, seq_len, num_patches, latent_dim = spatial_tokens.shape
        
        # Reshape for processing
        tokens_flat = spatial_tokens.view(batch_size * seq_len, num_patches, latent_dim)
        
        # Repeat query for each sequence
        query = self.query.repeat(batch_size * seq_len, 1, 1)
        
        # Apply attention pooling
        pooled_flat, _ = self.attention(query, tokens_flat, tokens_flat)
        pooled_flat = self.norm(pooled_flat)
        
        # Reshape back
        pooled_tokens = pooled_flat.view(batch_size, seq_len, latent_dim)
        
        return pooled_tokens.squeeze(-2)  # Remove the query dimension


class SpatialTokenEmbedding(nn.Module):
    """
    Embedding layer for spatial tokens with positional encoding
    """
    
    def __init__(self, vocab_size: int, embed_dim: int, grid_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        
        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        
        # Positional embedding for spatial locations
        self.pos_embed = nn.Parameter(
            torch.randn(1, grid_size * grid_size, embed_dim)
        )
        
        # Initialize embeddings
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
    
    def forward(self, spatial_tokens: torch.Tensor) -> torch.Tensor:
        """
        Embed spatial tokens with positional information
        
        Args:
            spatial_tokens: [batch_size, seq_len, grid_size^2] token indices
            
        Returns:
            embedded_tokens: [batch_size, seq_len, grid_size^2, embed_dim]
        """
        # Embed tokens
        token_embeds = self.token_embed(spatial_tokens)  # [batch, seq_len, grid_size^2, embed_dim]
        
        # Add positional embeddings
        pos_embeds = self.pos_embed.unsqueeze(1)  # [1, 1, grid_size^2, embed_dim]
        embedded_tokens = token_embeds + pos_embeds
        
        return embedded_tokens
