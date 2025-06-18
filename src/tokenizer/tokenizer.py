"""
Enhanced Delta-IRIS Tokenizer
Context-aware tokenization with spatial grids and delta encoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import logging
import math
from dataclasses import dataclass

from ..agent.config import TokenizerConfig


@dataclass
class DeltaTokenizerConfig(TokenizerConfig):
    """Enhanced configuration for Delta-IRIS tokenizer"""
    # Spatial tokenization
    spatial_grid_size: int = 8  # Split observations into spatial grids
    patch_size: int = 4  # Size of each spatial patch
    
    # Context-aware encoding
    context_length: int = 4  # Number of previous observations to use for context
    delta_encoding: bool = True  # Use delta (difference) encoding
    
    # Advanced VQ parameters
    use_exponential_moving_average: bool = True
    ema_decay: float = 0.99
    commitment_cost: float = 0.25
    
    # Spatial attention
    use_spatial_attention: bool = True
    spatial_attention_heads: int = 4


class ExponentialMovingAverage(nn.Module):
    """Exponential moving average for VQ codebook updates"""
    
    def __init__(self, num_embeddings: int, embedding_dim: int, decay: float = 0.99):
        super().__init__()
        self.decay = decay
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # Register buffers for EMA
        self.register_buffer('cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('embed_avg', torch.randn(num_embeddings, embedding_dim))
        
    def update(self, indices: torch.Tensor, encodings: torch.Tensor, embeddings: nn.Embedding):
        """Update embeddings using exponential moving average"""
        # Flatten indices and encodings
        indices_flat = indices.view(-1)
        encodings_flat = encodings.view(-1, self.embedding_dim)
        
        # Create one-hot encodings for cluster assignment
        one_hot = F.one_hot(indices_flat, self.num_embeddings).float()
        
        # Update cluster size
        cluster_size = one_hot.sum(0)
        self.cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
        
        # Update embeddings
        embed_sum = encodings_flat.T @ one_hot  # [embed_dim, num_embeddings]
        self.embed_avg.mul_(self.decay).add_(embed_sum.T, alpha=1 - self.decay)
        
        # Laplace smoothing
        n = self.cluster_size.sum()
        smoothed_cluster_size = (
            (self.cluster_size + 1e-5) / (n + self.num_embeddings * 1e-5) * n
        )
        
        # Update embedding weights
        embed_normalized = self.embed_avg / smoothed_cluster_size.unsqueeze(1)
        embeddings.weight.data.copy_(embed_normalized)


class SpatialTokenizer(nn.Module):
    """Spatial tokenization for creating spatial token grids"""
    
    def __init__(self, config: DeltaTokenizerConfig):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.grid_size = config.spatial_grid_size
        
        # Patch embedding
        patch_dim = config.obs_dim if config.obs_dim > 3 else 3 * config.patch_size * config.patch_size
        self.patch_embed = nn.Linear(patch_dim, config.latent_dim)
        
        # Positional embedding for spatial grids
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.grid_size * self.grid_size, config.latent_dim)
        )
        
        # Spatial attention if enabled
        if config.use_spatial_attention:
            self.spatial_attention = nn.MultiheadAttention(
                config.latent_dim, 
                config.spatial_attention_heads,
                batch_first=True
            )
            self.norm = nn.LayerNorm(config.latent_dim)
        
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Convert observations to spatial token grids
        
        Args:
            obs: [batch_size, seq_len, obs_dim] or [batch_size, seq_len, C, H, W] for images
            
        Returns:
            spatial_tokens: [batch_size, seq_len, grid_size^2, latent_dim]
        """
        batch_size, seq_len = obs.shape[:2]
        
        if obs.dim() == 3:
            # Vector observations - create pseudo-spatial layout
            obs_dim = obs.shape[2]
            patches = obs.unsqueeze(-2).repeat(1, 1, self.grid_size * self.grid_size, 1)
            patches = patches.view(batch_size, seq_len, self.grid_size * self.grid_size, obs_dim)
        else:
            # Image observations - extract spatial patches
            C, H, W = obs.shape[2:]
            patches = self._extract_patches(obs.view(-1, C, H, W))
            patches = patches.view(batch_size, seq_len, self.grid_size * self.grid_size, -1)
        
        # Embed patches
        spatial_tokens = self.patch_embed(patches)
        
        # Add positional embeddings
        spatial_tokens = spatial_tokens + self.pos_embed
        
        # Apply spatial attention if enabled
        if hasattr(self, 'spatial_attention'):
            # Reshape for attention: [batch*seq, grid^2, latent_dim]
            tokens_flat = spatial_tokens.view(-1, self.grid_size * self.grid_size, self.config.latent_dim)
            attended, _ = self.spatial_attention(tokens_flat, tokens_flat, tokens_flat)
            attended = self.norm(attended + tokens_flat)  # Residual connection
            spatial_tokens = attended.view(batch_size, seq_len, self.grid_size * self.grid_size, self.config.latent_dim)
        
        return spatial_tokens
    
    def _extract_patches(self, imgs: torch.Tensor) -> torch.Tensor:
        """Extract spatial patches from images"""
        # For now, use adaptive pooling to create fixed number of patches
        # In a full implementation, this would use proper patch extraction
        C = imgs.shape[1]
        patches = F.adaptive_avg_pool2d(imgs, (self.grid_size, self.grid_size))
        patches = patches.view(imgs.shape[0], C * self.grid_size * self.grid_size)
        return patches


class ContextAwareTokenizer(nn.Module):
    """Context-aware tokenization with delta encoding"""
    
    def __init__(self, config: DeltaTokenizerConfig):
        super().__init__()
        self.config = config
        self.context_length = config.context_length
        self.use_delta = config.delta_encoding
        
        # Context encoder
        self.context_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.latent_dim,
                nhead=4,
                dim_feedforward=config.hidden_dim,
                batch_first=True
            ),
            num_layers=2
        )
        
        # Delta encoding projections
        if self.use_delta:
            self.delta_proj = nn.Linear(config.latent_dim * 2, config.latent_dim)
        
    def forward(self, spatial_tokens: torch.Tensor) -> torch.Tensor:
        """
        Apply context-aware encoding with optional delta encoding
        
        Args:
            spatial_tokens: [batch_size, seq_len, grid_size^2, latent_dim]
            
        Returns:
            context_tokens: [batch_size, seq_len, grid_size^2, latent_dim] (same seq_len as input)
        """
        batch_size, seq_len, num_patches, latent_dim = spatial_tokens.shape
        
        # Process each timestep with its context
        context_tokens = []
        
        for t in range(seq_len):
            # Get context window ending at timestep t
            start_idx = max(0, t - self.context_length + 1)
            end_idx = t + 1
            context_window = spatial_tokens[:, start_idx:end_idx]  # [batch, context_len, patches, latent]
            
            # Pad context window if needed (for early timesteps)
            actual_context_len = end_idx - start_idx
            if actual_context_len < self.context_length:
                padding_len = self.context_length - actual_context_len
                padding = torch.zeros(
                    batch_size, padding_len, num_patches, latent_dim,
                    device=spatial_tokens.device, dtype=spatial_tokens.dtype
                )
                context_window = torch.cat([padding, context_window], dim=1)
            
            # Reshape for transformer: [batch * patches, context_len, latent]
            context_flat = context_window.permute(0, 2, 1, 3).reshape(-1, self.context_length, latent_dim)
            
            # Apply context encoding
            encoded_context = self.context_encoder(context_flat)
            
            # Take the last timestep as the context-aware representation
            current_repr = encoded_context[:, -1]  # [batch * patches, latent]
            current_repr = current_repr.view(batch_size, num_patches, latent_dim)
            
            # Apply delta encoding if enabled and we have previous timestep
            if self.use_delta and t > 0:
                prev_repr = context_tokens[-1]
                delta = torch.cat([current_repr, current_repr - prev_repr], dim=-1)
                current_repr = self.delta_proj(delta)
            
            context_tokens.append(current_repr)
        
        # Stack to maintain original sequence length
        return torch.stack(context_tokens, dim=1)  # [batch, seq_len, patches, latent]


class EnhancedVectorQuantizer(nn.Module):
    """Enhanced Vector Quantizer with EMA updates"""
    
    def __init__(self, config: DeltaTokenizerConfig):
        super().__init__()
        self.num_embeddings = config.codebook_size
        self.embedding_dim = config.latent_dim
        self.commitment_cost = config.commitment_cost
        self.use_ema = config.use_exponential_moving_average
        
        # Initialize embeddings
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        nn.init.uniform_(self.embeddings.weight, -1/self.num_embeddings, 1/self.num_embeddings)
        
        # EMA updates if enabled
        if self.use_ema:
            self.ema = ExponentialMovingAverage(
                self.num_embeddings, 
                self.embedding_dim, 
                config.ema_decay
            )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Quantize input tensors
        
        Args:
            x: [..., embedding_dim] input tensor
            
        Returns:
            quantized: quantized tensor with same shape as input
            indices: quantization indices
            losses: dict of loss terms
        """
        original_shape = x.shape
        x_flat = x.view(-1, self.embedding_dim)
        
        # Compute distances
        distances = torch.cdist(x_flat, self.embeddings.weight)
        indices = torch.argmin(distances, dim=1)
        
        # Get quantized vectors
        quantized_flat = self.embeddings(indices)
        quantized = quantized_flat.view(original_shape)
        
        # Compute losses
        if self.use_ema and self.training:
            # Update embeddings using EMA
            self.ema.update(indices, x_flat, self.embeddings)
            # Only commitment loss for EMA
            commitment_loss = F.mse_loss(x, quantized.detach()) * self.commitment_cost
            codebook_loss = torch.tensor(0.0, device=x.device)
        else:
            # Standard VQ losses
            commitment_loss = F.mse_loss(x, quantized.detach()) * self.commitment_cost
            codebook_loss = F.mse_loss(quantized, x.detach())
        
        # Straight-through estimator
        quantized = x + (quantized - x).detach()
        
        losses = {
            'commitment_loss': commitment_loss,
            'codebook_loss': codebook_loss,
            'total_quantizer_loss': commitment_loss + codebook_loss
        }
        
        return quantized, indices, losses


class DeltaIrisTokenizer(nn.Module):
    """
    Main Delta-IRIS Tokenizer with spatial grids and context-aware encoding
    """
    
    def __init__(self, config: DeltaTokenizerConfig):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Spatial tokenizer
        self.spatial_tokenizer = SpatialTokenizer(config)
        
        # Context-aware tokenizer
        self.context_tokenizer = ContextAwareTokenizer(config)
        
        # Enhanced vector quantizer
        self.quantizer = EnhancedVectorQuantizer(config)
        
        # Decoder to reconstruct from tokens
        self.decoder = self._build_decoder(config)
        
    def _build_decoder(self, config: DeltaTokenizerConfig) -> nn.Module:
        """Build decoder network"""
        return nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.obs_dim)
        )
    
    def encode(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Encode observations to continuous latents with spatial and context awareness
        
        Args:
            obs: [batch_size, seq_len, obs_dim] observations
            actions: [batch_size, seq_len, action_dim] actions
            
        Returns:
            latents: [batch_size, seq_len, grid_size^2, latent_dim]
        """
        # Create spatial token grids
        spatial_tokens = self.spatial_tokenizer(obs)
        
        # Apply context-aware encoding
        context_tokens = self.context_tokenizer(spatial_tokens)
        
        return context_tokens
    
    def quantize(self, latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Quantize latents to discrete tokens"""
        return self.quantizer(latents)
    
    def decode(self, quantized_latents: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Decode quantized latents back to observations
        
        Args:
            quantized_latents: [batch_size, seq_len, grid_size^2, latent_dim]
            actions: [batch_size, seq_len, action_dim] actions
            
        Returns:
            reconstructed_obs: [batch_size, seq_len, obs_dim]
        """
        batch_size, seq_len, num_patches, latent_dim = quantized_latents.shape
        
        # Pool spatial tokens to single representation per timestep
        # For now, use mean pooling - could be improved with learnable pooling
        pooled_latents = quantized_latents.mean(dim=2)  # [batch, seq_len, latent_dim]
        
        # Decode to observations
        reconstructed_obs = self.decoder(pooled_latents)
        
        return reconstructed_obs
    
    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full forward pass with spatial and context-aware tokenization
        
        Args:
            obs: [batch_size, seq_len, obs_dim] observations
            actions: [batch_size, seq_len, action_dim] actions
            
        Returns:
            Dictionary with tokens, reconstructions, and losses
        """
        # Encode with spatial and context awareness
        latents = self.encode(obs, actions)
        
        # Quantize to discrete tokens
        quantized_latents, tokens, quant_losses = self.quantize(latents)
        
        # Decode back to observations
        reconstructed_obs = self.decode(quantized_latents, actions)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstructed_obs, obs)
        
        # Total loss
        total_loss = recon_loss + quant_losses['total_quantizer_loss']
        
        losses = {
            'reconstruction_loss': recon_loss,
            **quant_losses,
            'total_tokenizer_loss': total_loss
        }
        
        return {
            'reconstructed_obs': reconstructed_obs,
            'tokens': tokens,  # [batch, seq_len, grid_size^2]
            'quantized_latents': quantized_latents,
            'continuous_latents': latents,
            'spatial_tokens': tokens,  # For compatibility
            'losses': losses
        }
    
    def get_spatial_tokens(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Get spatial tokens without gradients"""
        with torch.no_grad():
            latents = self.encode(obs, actions)
            _, tokens, _ = self.quantize(latents)
            # Ensure tokens maintain spatial structure [batch, seq, num_patches]
            batch_size, seq_len = obs.shape[:2]
            num_patches = self.config.spatial_grid_size * self.config.spatial_grid_size
            if tokens.dim() == 1:
                tokens = tokens.view(batch_size, seq_len, num_patches)
            return tokens
    
    def decode_spatial_tokens(self, tokens: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Decode spatial tokens back to observations"""
        with torch.no_grad():
            # Handle different token shapes
            if tokens.dim() == 1:
                # Flattened tokens - need to reconstruct shape
                batch_size, seq_len = actions.shape[:2]
                num_patches = self.config.spatial_grid_size * self.config.spatial_grid_size
                tokens = tokens.view(batch_size, seq_len, num_patches)
            elif tokens.dim() == 2:
                # Legacy format [batch*seq, num_patches] - expand to 3D
                batch_size, seq_len = actions.shape[:2]
                tokens = tokens.view(batch_size, seq_len, -1)
            
            # Now tokens should be [batch, seq, num_patches]
            batch_size, seq_len, num_patches = tokens.shape
            quantized_latents = self.quantizer.embeddings(tokens)
            return self.decode(quantized_latents, actions)
