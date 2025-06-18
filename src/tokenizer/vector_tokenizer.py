"""
Vector Tokenizer for Delta-IRIS
Complete tokenizer system for vector environments using modular components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import logging

from agent.config import TokenizerConfig
from .quantizer import VectorQuantizer
from .encoder_decoder import VectorEncoder, VectorDecoder


class VectorTokenizer(nn.Module):
    """
    Complete tokenizer system for vector environments
    Combines encoder, quantizer, and decoder for end-to-end training
    """
    
    def __init__(self, config: TokenizerConfig):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Extract dimensions from config
        obs_dim = config.obs_dim
        # Handle both discrete and continuous actions
        if hasattr(config, 'num_actions') and config.num_actions is not None:
            action_dim = config.num_actions  # Use num_actions for discrete actions
        else:
            action_dim = config.action_dim  # Use action_dim for continuous actions
            
        action_dim -= 1  # Adjust for one-hot encoding
        
        latent_dim = config.latent_dim
        hidden_dim = config.hidden_dim
        
        # Initialize modular components
        self.encoder = VectorEncoder(obs_dim, action_dim, latent_dim, hidden_dim)
        self.quantizer = VectorQuantizer(
            num_embeddings=config.codebook_size,
            embedding_dim=latent_dim,
            commitment_cost=0.25
        )
        self.decoder = VectorDecoder(latent_dim, action_dim, obs_dim, hidden_dim)
        
        self.logger.info(f"Vector tokenizer initialized: obs_dim={obs_dim}, action_dim={action_dim}, latent_dim={latent_dim}")
        
    def encode(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Encode observations and actions to continuous latents"""
        return self.encoder(obs, actions)
    
    def quantize(self, latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Quantize latents to discrete tokens"""
        return self.quantizer(latents)
    
    def decode(self, quantized_latents: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Decode quantized latents and actions back to observations"""
        return self.decoder(quantized_latents, actions)
    
    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full forward pass: encode -> quantize -> decode
        
        Args:
            obs: [batch_size, seq_len, obs_dim] 
            actions: [batch_size, seq_len] discrete actions
            
        Returns:
            Dictionary with tokens, reconstructions, and losses
        """
        # Encode to continuous latents
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
            'tokens': tokens,  # [batch, seq] - single token per timestep
            'quantized_latents': quantized_latents,
            'continuous_latents': latents,
            'losses': losses
        }
    
    def get_tokens(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Get discrete tokens without gradients
        
        Args:
            obs: [batch_size, seq_len, obs_dim]
            actions: [batch_size, seq_len] discrete actions
            
        Returns:
            tokens: [batch_size, seq_len] discrete token indices
        """
        with torch.no_grad():
            latents = self.encode(obs, actions)
            _, tokens, _ = self.quantize(latents)
            return tokens
    
    def decode_tokens(self, tokens: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Decode tokens back to observations
        
        Args:
            tokens: [batch_size, seq_len] discrete token indices
            actions: [batch_size, seq_len] discrete actions
            
        Returns:
            reconstructed_obs: [batch_size, seq_len, obs_dim]
        """
        with torch.no_grad():
            # Get quantized latents from tokens
            quantized_latents = self.quantizer.embed_tokens(tokens)
            return self.decode(quantized_latents, actions)
    
    def compute_loss(self, batch, **kwargs):
        """Compute tokenizer loss for training"""
        obs = batch.observations
        actions = batch.actions
        
        # Forward pass
        outputs = self.forward(obs, actions)
        
        # Get losses
        losses = outputs['losses']
        
        # Convert to LossWithIntermediateLosses format expected by trainer
        from utils import LossWithIntermediateLosses
        return LossWithIntermediateLosses(**losses), {}
