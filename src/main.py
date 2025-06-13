import hydra
from omegaconf import DictConfig, OmegaConf
from env import make_env
import os

# Prevent JAX from hogging memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Print the configuration for verification
    print(OmegaConf.to_yaml(cfg))
    
    # Access configuration values
    print(f"Environment: {cfg.env}")
    print(f"Model: {cfg.model}")
    print(f"Wandb project: {cfg.wandb.project}")
    
    env, _ = make_env(cfg.env.name)
    
    action_space = env.action_space
    obs_space = env.observation_space
    
    print(f"Action space: {action_space}")
    print(f"Observation space: {obs_space}")
    
if __name__ == "__main__":
    main()
