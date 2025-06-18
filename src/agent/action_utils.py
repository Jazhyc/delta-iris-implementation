"""
Action discretization utilities for Delta-IRIS
"""
import torch
from typing import Tuple, Optional


class ActionDiscretizer:
    """Handles discretization of continuous actions for the world model"""
    
    def __init__(self, action_dim: int, num_bins: int, action_range: Tuple[float, float]):
        """
        Args:
            action_dim: Dimension of continuous action space
            num_bins: Number of discrete bins per action dimension
            action_range: (min, max) values for action normalization
        """
        self.action_dim = action_dim
        self.num_bins = num_bins
        self.action_min, self.action_max = action_range
        
    def discretize(self, continuous_actions: torch.Tensor) -> torch.Tensor:
        """
        Discretize continuous actions into bins
        
        Args:
            continuous_actions: [..., action_dim] continuous actions
            
        Returns:
            discrete_actions: [..., action_dim] discrete action indices
        """
        # Normalize to [0, 1]
        normalized = torch.clamp(
            (continuous_actions - self.action_min) / (self.action_max - self.action_min), 
            0, 1
        )
        
        # Discretize to bins [0, num_bins-1]
        discrete = (normalized * (self.num_bins - 1)).long()
        
        return discrete
        
    def undiscretize(self, discrete_actions: torch.Tensor) -> torch.Tensor:
        """
        Convert discrete actions back to continuous values (for policy)
        
        Args:
            discrete_actions: [..., action_dim] discrete action indices
            
        Returns:
            continuous_actions: [..., action_dim] continuous actions
        """
        # Convert to [0, 1]
        normalized = discrete_actions.float() / (self.num_bins - 1)
        
        # Scale to action range
        continuous = normalized * (self.action_max - self.action_min) + self.action_min
        
        return continuous


def create_action_discretizer(env_type: str, action_dim: int, num_bins: Optional[int], 
                            action_range: Optional[Tuple[float, float]]) -> Optional[ActionDiscretizer]:
    """
    Create action discretizer based on environment type
    
    Args:
        env_type: 'continuous' or 'discrete'
        action_dim: Action space dimension
        num_bins: Number of bins for continuous actions
        action_range: Range for continuous actions
        
    Returns:
        ActionDiscretizer or None for discrete environments
    """
    if env_type == 'discrete':
        return None
    elif env_type == 'continuous':
        if num_bins is None or action_range is None:
            raise ValueError("Continuous environments require num_bins and action_range")
        return ActionDiscretizer(action_dim, num_bins, action_range)
    else:
        raise ValueError(f"Unknown environment type: {env_type}")
