import hydra, wandb
from omegaconf import DictConfig, OmegaConf
from env import make_env
import os
import torch

# Import new Delta-IRIS components
from agent.config import (
    TrainerConfig, TokenizerConfig, WorldModelConfig, 
    ActorCriticConfig, DataConfig
)
from trainer import DeltaIrisTrainer

# Prevent JAX from hogging memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

def create_config_from_hydra(cfg: DictConfig) -> TrainerConfig:
    """Convert Hydra config to our internal config format"""
    
    # Set default values based on environment
    env, _ = make_env(cfg.env.name, num_envs=1)
    
    # Get observation dimension properly (handle vectorized environments)
    if hasattr(env.observation_space, 'shape') and len(env.observation_space.shape) > 0:
        obs_dim = env.observation_space.shape[-1]  # Get feature dimension
    else:
        obs_dim = env.observation_space.n
    
    # Determine if environment is continuous or discrete
    print(f"DEBUG: env.action_space = {env.action_space}")
    print(f"DEBUG: type(env.action_space) = {type(env.action_space)}")
    
    # Handle different action space types
    if hasattr(env.action_space, 'n'):
        # Standard discrete action space (e.g., Discrete(4))
        action_dim = env.action_space.n
        print(f"Environment {cfg.env.name}: obs_dim={obs_dim}, discrete action_dim={action_dim}")
    elif hasattr(env.action_space, 'nvec'):
        # MultiDiscrete action space (e.g., MultiDiscrete([2]))
        action_dim = int(env.action_space.nvec[0])  # Use first dimension
        print(f"Environment {cfg.env.name}: obs_dim={obs_dim}, multi-discrete action_dim={action_dim}")
    elif hasattr(env.action_space, 'shape'):
        # Continuous action space (e.g., Box) - not supported for now
        print(f"Environment {cfg.env.name}: continuous action space detected - not supported yet")
        raise ValueError("Continuous action spaces are not supported yet. Please use a discrete environment like CartPole-v1.")
    else:
        raise ValueError(f"Unknown action space type: {type(env.action_space)}")
    
    print(f"DEBUG: Final action_dim = {action_dim}")
    
    # Create component configs from Hydra config
    print(f"DEBUG: Creating tokenizer config with obs_dim={obs_dim}, action_dim={action_dim}")
    tokenizer_config = TokenizerConfig(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=cfg.model.tokenizer.hidden_dim,
        latent_dim=cfg.model.tokenizer.latent_dim,
        num_tokens=cfg.model.tokenizer.num_tokens,
        codebook_size=cfg.model.tokenizer.codebook_size,
        learning_rate=cfg.model.tokenizer.learning_rate
    )
    
    world_model_config = WorldModelConfig(
        vocab_size=cfg.model.world_model.vocab_size,
        action_dim=action_dim,
        latent_dim=cfg.model.world_model.latent_dim,
        hidden_dim=cfg.model.world_model.hidden_dim,
        num_layers=cfg.model.world_model.num_layers,
        num_heads=cfg.model.world_model.num_heads,
        sequence_length=cfg.model.world_model.sequence_length,
        learning_rate=cfg.model.world_model.learning_rate
    )
    
    actor_critic_config = ActorCriticConfig(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=cfg.model.actor_critic.hidden_dim,
        imagination_horizon=cfg.model.actor_critic.imagination_horizon,
        gamma=cfg.model.actor_critic.gamma,
        lambda_gae=cfg.model.actor_critic.lambda_gae,
        entropy_coef=cfg.model.actor_critic.entropy_coef,
        learning_rate=cfg.model.actor_critic.learning_rate
    )
    
    data_config = DataConfig(
        sequence_length=cfg.model.buffer.sequence_length,
        batch_size=cfg.model.buffer.batch_size
    )
    
    # Create main trainer config
    trainer_config = TrainerConfig(
        epochs=cfg.model.training.epochs,
        steps_per_epoch=cfg.model.training.steps_per_epoch,
        eval_frequency=cfg.model.training.eval_frequency,
        device=cfg.model.training.device if torch.cuda.is_available() else "cpu",
        dtype=cfg.model.training.dtype,
        tokenizer=tokenizer_config,
        world_model=world_model_config,
        actor_critic=actor_critic_config,
        data=data_config
    )
    
    return trainer_config

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    
    # Print the configuration for verification
    print("Hydra Configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Convert to internal config format
    config = create_config_from_hydra(cfg)
    
    print(f"\nUsing device: {config.device}")
    print(f"Using dtype: {config.dtype}")
    
    # Initialize Weights & Biases
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        mode=cfg.wandb.mode,
    )
    
    # Create and run trainer
    trainer = DeltaIrisTrainer(config, hydra_config=cfg)
    
    print("Starting Delta-IRIS training...")
    trainer.train()
    
    print("Training completed!")
    
if __name__ == "__main__":
    main()
