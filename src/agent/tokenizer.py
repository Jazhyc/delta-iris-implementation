"""
MLP-based Tokenizer for Delta-IRIS
Adapted for one-hot encoded environments
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import logging
from .config import TokenizerConfig


class VectorQuantizer(nn.Module):
    """Vector quantization for discrete latent representation"""
    
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Initialize embeddings
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: [batch_size, seq_len, embedding_dim]
        Returns:
            quantized: quantized vectors
            indices: quantization indices  
            losses: dictionary of losses
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Flatten for quantization
        x_flat = x.view(-1, embed_dim)  # [batch_size * seq_len, embed_dim]
        
        # Compute distances to embeddings
        distances = torch.cdist(x_flat, self.embeddings.weight)  # [batch_size * seq_len, num_embeddings]
        
        # Get closest embedding indices
        indices = torch.argmin(distances, dim=1)  # [batch_size * seq_len]
        
        # Get quantized vectors
        quantized = self.embeddings(indices)  # [batch_size * seq_len, embed_dim]
        
        # Reshape back
        quantized = quantized.view(batch_size, seq_len, embed_dim)
        indices = indices.view(batch_size, seq_len)
        
        # Compute losses
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


class MLPEncoder(nn.Module):
    """MLP encoder for observations and actions"""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        
        input_dim = obs_dim + action_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, latent_dim)
        )
        
    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Encode observation-action pairs"""
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)


class MLPDecoder(nn.Module):
    """MLP decoder for reconstructing observations"""
    
    def __init__(self, latent_dim: int, action_dim: int, obs_dim: int, hidden_dim: int):
        super().__init__()
        
        input_dim = latent_dim + action_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim)
        )
        
    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Decode latent representation and action to observation"""
        x = torch.cat([latent, action], dim=-1)
        return self.net(x)


class Tokenizer(nn.Module):
    """MLP-based tokenizer for discrete latent representations"""
    
    def __init__(self, config: TokenizerConfig):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Encoder: (obs, action) -> latent
        self.encoder = MLPEncoder(
            obs_dim=config.obs_dim,
            action_dim=config.action_dim, 
            hidden_dim=config.hidden_dim,
            latent_dim=config.latent_dim
        )
        
        # Vector quantizer
        self.quantizer = VectorQuantizer(
            num_embeddings=config.codebook_size,
            embedding_dim=config.latent_dim
        )
        
        # Decoder: (quantized_latent, action) -> reconstructed_obs
        self.decoder = MLPDecoder(
            latent_dim=config.latent_dim,
            action_dim=config.action_dim,
            obs_dim=config.obs_dim,
            hidden_dim=config.hidden_dim
        )
        
    def encode(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Encode observations and actions to continuous latents"""
        return self.encoder(obs, actions)
        
    def quantize(self, latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Quantize continuous latents to discrete tokens"""
        return self.quantizer(latents)
        
    def decode(self, quantized_latents: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Decode quantized latents and actions to observations"""
        return self.decoder(quantized_latents, actions)
        
    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full forward pass: encode -> quantize -> decode
        
        Args:
            obs: [batch_size, seq_len, obs_dim] 
            actions: [batch_size, seq_len, action_dim]
            
        Returns:
            Dictionary with reconstructed observations, tokens, and losses
        """
        # Encode to continuous latents
        latents = self.encode(obs, actions)
        
        # Quantize to discrete tokens
        quantized_latents, tokens, quant_losses = self.quantize(latents)
        
        # Decode back to observations  
        reconstructed_obs = self.decode(quantized_latents, actions)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstructed_obs, obs)
        
        # Combine losses
        total_loss = recon_loss + quant_losses['total_quantizer_loss']
        
        losses = {
            'reconstruction_loss': recon_loss,
            **quant_losses,
            'total_tokenizer_loss': total_loss
        }
        
        return {
            'reconstructed_obs': reconstructed_obs,
            'tokens': tokens,
            'quantized_latents': quantized_latents,
            'continuous_latents': latents,
            'losses': losses
        }
        
    def get_tokens(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Get discrete tokens without gradients"""
        with torch.no_grad():
            latents = self.encode(obs, actions)
            _, tokens, _ = self.quantize(latents)
            return tokens
            
    def decode_tokens(self, tokens: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Decode tokens back to observations"""
        with torch.no_grad():
            # Get quantized latents from tokens
            quantized_latents = self.quantizer.embeddings(tokens)
            return self.decode(quantized_latents, actions)
            
    def _sanity_check(self, batch_data: Dict[str, torch.Tensor]):
        """Perform sanity checks on tokenizer outputs"""
        recon_obs = batch_data['reconstructed_obs']
        tokens = batch_data['tokens']
        
        # Check for NaN/Inf
        if torch.isnan(recon_obs).any():
            self.logger.warning("NaN values in reconstructed observations")
        if torch.isinf(recon_obs).any():
            self.logger.warning("Inf values in reconstructed observations")
            
        # Check token range
        if tokens.min() < 0 or tokens.max() >= self.config.codebook_size:
            self.logger.warning(f"Token values out of range: [{tokens.min()}, {tokens.max()}]")
            
        # Log reconstruction quality
        if hasattr(self, '_step_count'):
            self._step_count += 1
            if self._step_count % 100 == 0:
                recon_error = batch_data['losses']['reconstruction_loss'].item()
                self.logger.debug(f"Reconstruction error: {recon_error:.6f}")
        else:
            self._step_count = 1
