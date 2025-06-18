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
class BufferConfig:  
    """Configuration for the experience buffer"""
    capacity: int = 100000
    sequence_length: int = 64
    batch_size: int = 32


@dataclass
class TrainerConfig:
    """Main trainer configuration"""
    # Component configs (required fields first)
    tokenizer: TokenizerConfig
    world_model: WorldModelConfig  
    actor_critic: ActorCriticConfig
    buffer: BufferConfig
    
    # Training parameters (optional fields with defaults)
    epochs: int = 1000
    steps_per_epoch: int = 1000
    eval_frequency: int = 10
    device: str = "cuda"
    dtype: str = "bfloat16"  # Use BF16 precision
