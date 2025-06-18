"""
Tokenizer wrapper for Delta-IRIS
Uses the enhanced tokenizer from src.tokenizer.tokenizer
"""

import torch
import torch.nn as nn
from typing import Dict
import logging
from .config import TokenizerConfig

# Import enhanced tokenizer
from tokenizer.tokenizer import DeltaIrisTokenizer as EnhancedTokenizer


class Tokenizer(nn.Module):
    """Tokenizer wrapper that uses the enhanced Delta-IRIS tokenizer"""
    
    def __init__(self, config: TokenizerConfig):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Use the enhanced tokenizer
        self.logger.info("Using enhanced Delta-IRIS tokenizer")
        self.tokenizer = EnhancedTokenizer(config)
        
    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full forward pass: encode -> quantize -> decode
        
        Args:
            obs: [batch_size, seq_len, obs_dim] 
            actions: [batch_size, seq_len, action_dim]
            
        Returns:
            Dictionary with reconstructed observations, tokens, and losses
        """
        return self.tokenizer(obs, actions)
        
    def encode(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Encode observations and actions to continuous latents"""
        return self.tokenizer.encode(obs, actions)
        
    def quantize(self, latents: torch.Tensor):
        """Quantize continuous latents to discrete tokens"""
        return self.tokenizer.quantize(latents)
        
    def decode(self, quantized_latents: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Decode quantized latents and actions to observations"""
        return self.tokenizer.decode(quantized_latents, actions)
        
    def get_tokens(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Get discrete tokens without gradients - flattened for world model"""
        spatial_tokens = self.tokenizer.get_spatial_tokens(obs, actions)
        # Flatten spatial tokens to [batch_size, seq_len] by taking first token from each spatial grid
        # For non-image environments like CartPole, spatial structure isn't critical
        batch_size, seq_len, num_patches = spatial_tokens.shape
        # Take the first spatial token or average across patches
        tokens = spatial_tokens[:, :, 0]  # [batch_size, seq_len]
        return tokens
            
    def decode_tokens(self, tokens: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Decode tokens back to observations"""
        return self.tokenizer.decode_spatial_tokens(tokens, actions)
