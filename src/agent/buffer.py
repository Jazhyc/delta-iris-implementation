"""
GPU Memory Buffer for Delta-IRIS
Stores experience in GPU memory using BF16 precision
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass


@dataclass
class Episode:
    """Single episode storage"""
    observations: torch.Tensor  # [T, obs_dim]
    actions: torch.Tensor       # [T, action_dim] 
    rewards: torch.Tensor       # [T]
    dones: torch.Tensor         # [T]
    
    def __len__(self) -> int:
        return len(self.observations)


class ExperienceBuffer:
    """GPU-resident experience buffer with episode storage"""
    
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, 
                 device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.dtype = dtype
        
        # Initialize storage tensors
        self.observations = torch.zeros(capacity, obs_dim, device=device, dtype=dtype)
        self.actions = torch.zeros(capacity, action_dim, device=device, dtype=dtype)
        self.rewards = torch.zeros(capacity, device=device, dtype=dtype)
        self.dones = torch.zeros(capacity, device=device, dtype=torch.bool)
        
        # Episode tracking
        self.episodes: List[Episode] = []
        self.current_size = 0
        self.position = 0
        
        self.logger = logging.getLogger(__name__)
        
    def add_episode(self, episode: Episode) -> None:
        """Add a complete episode to the buffer"""
        episode_length = len(episode)
        
        # Check if episode fits in remaining capacity
        if self.position + episode_length > self.capacity:
            # Wrap around - overwrite oldest data
            remaining = self.capacity - self.position
            if remaining > 0:
                self._store_partial_episode(episode, 0, remaining, self.position)
            overflow = episode_length - remaining
            if overflow > 0:
                self._store_partial_episode(episode, remaining, episode_length, 0)
            self.position = overflow
        else:
            # Store normally
            self._store_partial_episode(episode, 0, episode_length, self.position)
            self.position += episode_length
            
        # Update current size
        self.current_size = min(self.current_size + episode_length, self.capacity)
        
        # Store episode metadata
        self.episodes.append(episode)
        
        # Sanity check
        self._sanity_check_storage()
        
        self.logger.debug(f"Added episode of length {episode_length}, buffer size: {self.current_size}")
        
    def _store_partial_episode(self, episode: Episode, start_idx: int, end_idx: int, buffer_pos: int):
        """Store a portion of an episode at the given buffer position"""
        length = end_idx - start_idx
        self.observations[buffer_pos:buffer_pos + length] = episode.observations[start_idx:end_idx]
        self.actions[buffer_pos:buffer_pos + length] = episode.actions[start_idx:end_idx]
        self.rewards[buffer_pos:buffer_pos + length] = episode.rewards[start_idx:end_idx]
        self.dones[buffer_pos:buffer_pos + length] = episode.dones[start_idx:end_idx]
        
    def sample_sequences(self, batch_size: int, sequence_length: int) -> Dict[str, torch.Tensor]:
        """Sample random sequences for training"""
        if self.current_size < sequence_length:
            raise ValueError(f"Buffer has {self.current_size} samples, need at least {sequence_length}")
            
        # Sample random starting positions
        max_start = self.current_size - sequence_length
        start_indices = torch.randint(0, max_start, (batch_size,), device=self.device)
        
        # Create batch tensors
        batch_obs = torch.zeros(batch_size, sequence_length, self.obs_dim, 
                               device=self.device, dtype=self.dtype)
        batch_actions = torch.zeros(batch_size, sequence_length, self.action_dim,
                                   device=self.device, dtype=self.dtype) 
        batch_rewards = torch.zeros(batch_size, sequence_length,
                                   device=self.device, dtype=self.dtype)
        batch_dones = torch.zeros(batch_size, sequence_length,
                                 device=self.device, dtype=torch.bool)
        
        # Fill batch
        for i, start_idx in enumerate(start_indices):
            end_idx = start_idx + sequence_length
            
            # Handle wrap-around if necessary
            if end_idx <= self.capacity:
                batch_obs[i] = self.observations[start_idx:end_idx]
                batch_actions[i] = self.actions[start_idx:end_idx]  
                batch_rewards[i] = self.rewards[start_idx:end_idx]
                batch_dones[i] = self.dones[start_idx:end_idx]
            else:
                # Wrap around case
                first_part = self.capacity - start_idx
                batch_obs[i, :first_part] = self.observations[start_idx:]
                batch_obs[i, first_part:] = self.observations[:sequence_length - first_part]
                
                batch_actions[i, :first_part] = self.actions[start_idx:]
                batch_actions[i, first_part:] = self.actions[:sequence_length - first_part]
                
                batch_rewards[i, :first_part] = self.rewards[start_idx:]
                batch_rewards[i, first_part:] = self.rewards[:sequence_length - first_part]
                
                batch_dones[i, :first_part] = self.dones[start_idx:]
                batch_dones[i, first_part:] = self.dones[:sequence_length - first_part]
        
        return {
            'observations': batch_obs,
            'actions': batch_actions,
            'rewards': batch_rewards, 
            'dones': batch_dones
        }
        
    def _sanity_check_storage(self):
        """Perform sanity checks on buffer storage"""
        assert self.current_size <= self.capacity, f"Size {self.current_size} exceeds capacity {self.capacity}"
        assert self.position < self.capacity, f"Position {self.position} exceeds capacity {self.capacity}"
        
        # Check for NaN/Inf values
        if torch.isnan(self.observations[:self.current_size]).any():
            self.logger.warning("NaN values detected in observations")
        if torch.isinf(self.observations[:self.current_size]).any():
            self.logger.warning("Inf values detected in observations")
            
    def get_stats(self) -> Dict[str, float]:
        """Get buffer statistics"""
        if self.current_size == 0:
            return {}
            
        return {
            'buffer_size': self.current_size,
            'buffer_utilization': self.current_size / self.capacity,
            'num_episodes': len(self.episodes),
            'avg_episode_length': sum(len(ep) for ep in self.episodes) / len(self.episodes) if self.episodes else 0,
            'avg_reward': self.rewards[:self.current_size].mean().item(),
            'total_reward': self.rewards[:self.current_size].sum().item(),
        }
        
    def clear(self):
        """Clear the buffer"""
        self.episodes.clear()
        self.current_size = 0
        self.position = 0
        self.logger.info("Buffer cleared")
