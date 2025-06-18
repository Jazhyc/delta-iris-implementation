"""
MLP-based Tokenizer for Delta-IRIS
Adapted for one-hot encoded environments
Legacy implementation - use src.tokenizer.DeltaIrisTokenizer for enhanced features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import logging
from .config import TokenizerConfig

# Import enhanced tokenizer
try:
    from ..tokenizer import DeltaIrisTokenizer, DeltaTokenizerConfig
    ENHANCED_TOKENIZER_AVAILABLE = True
except ImportError:
    ENHANCED_TOKENIZER_AVAILABLE = False


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
            x: [..., embedding_dim]
        Returns:
            quantized: quantized vectors
            indices: quantization indices  
            losses: dictionary of losses
        """
        original_shape = x.shape
        embed_dim = original_shape[-1]
        
        # Flatten for quantization
        x_flat = x.view(-1, embed_dim)  # [total_elements, embed_dim]
        
        # Compute distances to embeddings
        distances = torch.cdist(x_flat, self.embeddings.weight)  # [total_elements, num_embeddings]
        
        # Get closest embedding indices
        indices = torch.argmin(distances, dim=1)  # [total_elements]
        
        # Get quantized vectors
        quantized = self.embeddings(indices)  # [total_elements, embed_dim]
        
        # Reshape back to original shape
        quantized = quantized.view(original_shape)
        indices = indices.view(original_shape[:-1])  # Remove last dimension (embedding_dim)
        
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
        # Ensure tensors are on the same device and dtype as the model
        obs = obs.to(dtype=self.net[0].weight.dtype, device=self.net[0].weight.device)
        action = action.to(dtype=self.net[0].weight.dtype, device=self.net[0].weight.device)
        
        # Handle different tensor shapes
        original_shape = None
        if obs.dim() == 3 and action.dim() == 3:
            # [batch, seq, dim] -> flatten for processing
            original_shape = obs.shape[:2]  # Store batch_size, seq_len
            obs = obs.view(-1, obs.shape[-1])
            action = action.view(-1, action.shape[-1])
        
        x = torch.cat([obs, action], dim=-1)
        output = self.net(x)
        
        # Reshape back to sequence format if needed
        if original_shape is not None:
            output = output.view(*original_shape, -1)
            
        return output


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
        # Ensure tensors are on the same device and dtype as the model
        latent = latent.to(dtype=self.net[0].weight.dtype, device=self.net[0].weight.device)
        action = action.to(dtype=self.net[0].weight.dtype, device=self.net[0].weight.device)
        
        # Handle different tensor shapes
        original_shape = None
        if latent.dim() == 3 and action.dim() == 3:
            # [batch, seq, dim] -> flatten for processing
            original_shape = latent.shape[:2]  # Store batch_size, seq_len
            latent = latent.view(-1, latent.shape[-1])
            action = action.view(-1, action.shape[-1])
        
        x = torch.cat([latent, action], dim=-1)
        output = self.net(x)
        
        # Reshape back to sequence format if needed
        if original_shape is not None:
            output = output.view(*original_shape, -1)
            
        return output


class Tokenizer(nn.Module):
    """MLP-based tokenizer for discrete latent representations"""
    
    def __init__(self, config: TokenizerConfig):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Try to use enhanced tokenizer if available and config supports it
        if ENHANCED_TOKENIZER_AVAILABLE and hasattr(config, 'spatial_grid_size'):
            self.logger.info("Using enhanced Delta-IRIS tokenizer")
            self._use_enhanced = True
            # Convert config to enhanced config
            enhanced_config = DeltaTokenizerConfig(**vars(config))
            self.enhanced_tokenizer = DeltaIrisTokenizer(enhanced_config)
        else:
            self.logger.info("Using legacy MLP tokenizer")
            self._use_enhanced = False
            self._init_legacy_components(config)
    
    def _init_legacy_components(self, config: TokenizerConfig):
        """Initialize legacy tokenizer components"""
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
        if self._use_enhanced:
            return self.enhanced_tokenizer.encode(obs, actions)
        else:
            return self.encoder(obs, actions)
        
    def quantize(self, latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Quantize continuous latents to discrete tokens"""
        if self._use_enhanced:
            return self.enhanced_tokenizer.quantize(latents)
        else:
            return self.quantizer(latents)
        
    def decode(self, quantized_latents: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Decode quantized latents and actions to observations"""
        if self._use_enhanced:
            return self.enhanced_tokenizer.decode(quantized_latents, actions)
        else:
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
        if self._use_enhanced:
            return self.enhanced_tokenizer.forward(obs, actions)
        else:
            return self._legacy_forward(obs, actions)
    
    def _legacy_forward(self, obs: torch.Tensor, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Legacy forward pass implementation"""
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
        if self._use_enhanced:
            return self.enhanced_tokenizer.get_spatial_tokens(obs, actions)
        else:
            with torch.no_grad():
                latents = self.encode(obs, actions)
                _, tokens, _ = self.quantize(latents)
                return tokens
            
    def decode_tokens(self, tokens: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Decode tokens back to observations"""
        if self._use_enhanced:
            return self.enhanced_tokenizer.decode_spatial_tokens(tokens, actions)
        else:
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
    
    # Enhanced tokenizer API compatibility
    def get_spatial_tokens(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Get spatial tokens (enhanced API compatibility)"""
        return self.get_tokens(obs, actions)
    
    def decode_spatial_tokens(self, tokens: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Decode spatial tokens (enhanced API compatibility)"""
        return self.decode_tokens(tokens, actions)
