"""
Configuration classes for Delta-IRIS implementation
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class TokenizerConfig:
    """Configuration for the tokenizer component"""
    obs_dim: int
    action_dim: int
    hidden_dim: int = 256
    latent_dim: int = 64
    num_tokens: int = 4
    codebook_size: int = 1024
    learning_rate: float = 1e-4


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


@dataclass
class WorldModelConfig:
    """Configuration for the world model component"""
    vocab_size: int = 1024
    action_dim: int = 1
    latent_dim: int = 64
    hidden_dim: int = 512
    num_layers: int = 4
    num_heads: int = 8
    sequence_length: int = 64
    learning_rate: float = 1e-4


@dataclass
class ActorCriticConfig:
    """Configuration for the actor-critic component"""
    obs_dim: int
    action_dim: int
    hidden_dim: int = 256
    imagination_horizon: int = 15
    gamma: float = 0.99
    lambda_gae: float = 0.95
    entropy_coef: float = 0.01
    learning_rate: float = 1e-4


@dataclass
class DataConfig:  
    """Configuration for the episode dataset and sampling"""
    sequence_length: int = 64
    batch_size: int = 32
    # Note: Dataset capacity is dynamic (episodes saved to disk)


@dataclass
class TrainerConfig:
    """Main trainer configuration"""
    # Component configs (required fields first)
    tokenizer: TokenizerConfig
    world_model: WorldModelConfig  
    actor_critic: ActorCriticConfig
    data: DataConfig  # Renamed from buffer
    
    # Training parameters (optional fields with defaults)
    epochs: int = 1000
    steps_per_epoch: int = 1000
    eval_frequency: int = 10
    device: str = "cuda"
    dtype: str = "bfloat16"  # Use BF16 precision
