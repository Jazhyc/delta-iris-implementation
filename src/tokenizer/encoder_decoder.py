"""
Encoder and Decoder networks for vector environments
Simple MLPs for encoding observations+actions to latents and decoding back
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorEncoder(nn.Module):
    """
    MLP encoder that maps observations + actions to continuous latent space
    """
    
    def __init__(self, obs_dim: int, action_dim: int, latent_dim: int, hidden_dim: int = 256, num_actions: int = None):
        super().__init__()
        self.obs_dim = obs_dim
        # For discrete actions, subtract 1 for one-hot encoding
        self.action_dim = num_actions if num_actions is not None else action_dim
        self.latent_dim = latent_dim
        
        # MLP encoder: (obs + action) -> latent
        input_dim = obs_dim + self.action_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Encode observations and actions to continuous latents
        
        Args:
            obs: [batch_size, seq_len, obs_dim] 
            actions: [batch_size, seq_len, action_dim] (one-hot) or [batch_size, seq_len] (discrete)
            
        Returns:
            latents: [batch_size, seq_len, latent_dim]
        """
        
        # Handle discrete actions by one-hot encoding
        if actions.dim() == 2:  # [batch, seq] discrete actions
            print(f"One-hot encoding actions to {self.action_dim} classes")
            actions = F.one_hot(actions, num_classes=self.action_dim).to(dtype=obs.dtype)
        
        # Concatenate observations and actions
        inputs = torch.cat([obs, actions], dim=-1)
        
        # Encode to latent space
        latents = self.encoder(inputs)
        
        return latents


class VectorDecoder(nn.Module):
    """
    MLP decoder that maps latents + actions back to observations
    """
    
    def __init__(self, latent_dim: int, action_dim: int, obs_dim: int, hidden_dim: int = 256, num_actions: int = None):
        super().__init__()
        # For discrete actions, always use num_actions for one-hot encoding
        self.action_dim = num_actions if num_actions is not None else action_dim
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        
        # MLP decoder: (latent + action) -> obs
        input_dim = latent_dim + self.action_dim
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim)
        )
        
    def forward(self, latents: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Decode latents and actions back to observations
        
        Args:
            latents: [batch_size, seq_len, latent_dim]
            actions: [batch_size, seq_len, action_dim] (one-hot) or [batch_size, seq_len] (discrete)
            
        Returns:
            reconstructed_obs: [batch_size, seq_len, obs_dim]
        """
        # Handle discrete actions by one-hot encoding
        if actions.dim() == 2:  # [batch, seq] discrete actions
            actions = F.one_hot(actions, num_classes=self.action_dim).to(dtype=latents.dtype)
        
        # Concatenate latents and actions
        inputs = torch.cat([latents, actions], dim=-1)
        
        # Decode to observation space
        reconstructed_obs = self.decoder(inputs)
        
        return reconstructed_obs
