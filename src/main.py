import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Print the configuration for verification
    print(OmegaConf.to_yaml(cfg))
    
    # Access configuration values
    print(f"Environment: {cfg.env}")
    print(f"Model: {cfg.model}")
    print(f"Wandb project: {cfg.wandb.project}")
    
    # Your main application logic goes here
    
if __name__ == "__main__":
    main()
