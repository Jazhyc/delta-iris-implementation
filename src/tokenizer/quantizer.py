"""
Vector Quantizer for Delta-IRIS
Simplified VQ-VAE quantizer for vector environments
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class VectorQuantizer(nn.Module):
    """
    Vector Quantizer using standard VQ-VAE approach
    Maps continuous latent vectors to discrete codebook entries
    """
    
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Initialize codebook embeddings
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embeddings.weight, -1/num_embeddings, 1/num_embeddings)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Quantize input tensors using nearest neighbor in codebook
        
        Args:
            x: [batch, seq, latent_dim] input tensor
            
        Returns:
            quantized: quantized tensor with same shape as input
            indices: quantization indices [batch, seq] 
            losses: dict of VQ loss terms
        """
        original_shape = x.shape
        x_flat = x.view(-1, self.embedding_dim)
        
        # Compute distances to all codebook entries
        distances = torch.cdist(x_flat, self.embeddings.weight)
        
        # Find nearest codebook entry for each input
        indices = torch.argmin(distances, dim=1)
        
        # Get quantized vectors from codebook
        quantized_flat = self.embeddings(indices)
        quantized = quantized_flat.view(original_shape)
        
        # Compute VQ-VAE losses
        # Commitment loss: encourages encoder output to stay close to codebook
        commitment_loss = F.mse_loss(x, quantized.detach()) * self.commitment_cost
        
        # Codebook loss: updates codebook to match encoder outputs
        codebook_loss = F.mse_loss(quantized, x.detach())
        
        # Straight-through estimator: gradients flow through unchanged
        quantized = x + (quantized - x).detach()
        
        losses = {
            'commitment_loss': commitment_loss,
            'codebook_loss': codebook_loss,
            'total_quantizer_loss': commitment_loss + codebook_loss
        }
        
        # Reshape indices to match input batch/sequence structure
        indices = indices.view(original_shape[0], original_shape[1])
        
        return quantized, indices, losses
    
    def embed_tokens(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        """
        Convert discrete tokens back to continuous vectors
        
        Args:
            tokens: [batch, seq] discrete token indices
            
        Returns:
            embeddings: [batch, seq, embedding_dim] continuous vectors
        """
        return self.embeddings(tokens)
