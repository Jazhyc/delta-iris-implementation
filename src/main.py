import hydra, wandb
from omegaconf import DictConfig, OmegaConf
from env import make_env
import os
import torch

# Import new Delta-IRIS components
from agent.config import (
    TrainerConfig, TokenizerConfig, WorldModelConfig, 
    ActorCriticConfig, BufferConfig
)
from trainer import DeltaIrisTrainer

# Prevent JAX from hogging memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

def create_config_from_hydra(cfg: DictConfig) -> TrainerConfig:
    """Convert Hydra config to our internal config format"""
    
    # Set default values based on environment
    env, _ = make_env(cfg.env.name, num_envs=1)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n if hasattr(env.action_space, 'n') else 1
    
    print(f"Environment {cfg.env.name}: obs_dim={obs_dim}, action_dim={action_dim}")
    
    # Create component configs
    tokenizer_config = TokenizerConfig(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=256,
        latent_dim=64,
        num_tokens=4,
        codebook_size=1024,
        learning_rate=1e-4
    )
    
    world_model_config = WorldModelConfig(
        vocab_size=1024,
        action_dim=action_dim,
        latent_dim=64,
        hidden_dim=512,
        num_layers=4,
        num_heads=8,
        sequence_length=64,
        learning_rate=1e-4
    )
    
    actor_critic_config = ActorCriticConfig(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=256,
        imagination_horizon=15,
        gamma=0.99,
        lambda_gae=0.95,
        entropy_coef=0.01,
        learning_rate=1e-4
    )
    
    buffer_config = BufferConfig(
        capacity=100000,
        sequence_length=64,
        batch_size=32
    )
    
    # Create main trainer config
    trainer_config = TrainerConfig(
        epochs=1000,
        steps_per_epoch=1000,
        eval_frequency=10,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype="bfloat16",
        tokenizer=tokenizer_config,
        world_model=world_model_config,
        actor_critic=actor_critic_config,
        buffer=buffer_config
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
    trainer = DeltaIrisTrainer(config)
    
    print("Starting Delta-IRIS training...")
    trainer.train()
    
    print("Training completed!")
    
if __name__ == "__main__":
    main()
