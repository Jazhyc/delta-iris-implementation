"""
Tokenizer for Delta-IRIS
Vector environment tokenizer with modular components
"""

import torch
import torch.nn as nn
from typing import Dict
import logging
from .config import TokenizerConfig

# Import vector tokenizer
from tokenizer.vector_tokenizer import VectorTokenizer


class Tokenizer(nn.Module):
    """
    Main tokenizer class for vector environments
    Uses modular MLP-based components for encoding, quantization, and decoding
    """
    
    def __init__(self, config: TokenizerConfig):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Use vector tokenizer for all environments
        self.logger.info("Using MLP-based vector tokenizer")
        self.tokenizer = VectorTokenizer(config)
        
    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full forward pass: encode -> quantize -> decode
        
        Args:
            obs: [batch_size, seq_len, obs_dim] 
            actions: [batch_size, seq_len] discrete actions
            
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
        """
        Get discrete tokens without gradients
        
        Returns:
            tokens: [batch_size, seq_len] discrete token indices
        """
        return self.tokenizer.get_tokens(obs, actions)
            
    def decode_tokens(self, tokens: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Decode tokens back to observations"""
        return self.tokenizer.decode_tokens(tokens, actions)
    
    def compute_loss(self, batch, **kwargs):
        """Compute tokenizer loss for training"""
        return self.tokenizer.compute_loss(batch, **kwargs)

    def __repr__(self) -> str:
        return "tokenizer"
