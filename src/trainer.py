"""
Delta-IRIS Trainer
Main training loop coordinating tokenizer, world model, and actor-critic
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
import logging
import wandb
import time
from pathlib import Path

from agent.config import TrainerConfig
from agent.buffer import ExperienceBuffer, Episode
from agent.tokenizer import Tokenizer
from agent.world_model import WorldModel
from agent.actor_critic import ActorCritic
from env import make_env


class DeltaIrisTrainer:
    """Main trainer for Delta-IRIS algorithm"""
    
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = getattr(torch, config.dtype)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create environment to get observation and action dimensions
        self.env, _ = make_env("Pendulum-v1", num_envs=1)  # Single env for initialization
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n if hasattr(self.env.action_space, 'n') else 1
        
        self.logger.info(f"Environment: obs_dim={self.obs_dim}, action_dim={self.action_dim}")
        
        # Update configs with environment dimensions
        config.tokenizer.obs_dim = self.obs_dim
        config.tokenizer.action_dim = self.action_dim
        config.world_model.action_dim = self.action_dim
        config.actor_critic.obs_dim = self.obs_dim
        config.actor_critic.action_dim = self.action_dim
        
        # Initialize components
        self.tokenizer = Tokenizer(config.tokenizer).to(self.device, dtype=self.dtype)
        self.world_model = WorldModel(config.world_model).to(self.device, dtype=self.dtype)
        self.actor_critic = ActorCritic(config.actor_critic).to(self.device, dtype=self.dtype)
        
        # Initialize optimizers
        self.tokenizer_optimizer = optim.Adam(
            self.tokenizer.parameters(), 
            lr=config.tokenizer.learning_rate
        )
        self.world_model_optimizer = optim.Adam(
            self.world_model.parameters(),
            lr=config.world_model.learning_rate
        )
        self.actor_critic_optimizer = optim.Adam(
            self.actor_critic.parameters(),
            lr=config.actor_critic.learning_rate
        )
        
        # Initialize experience buffer
        self.buffer = ExperienceBuffer(
            capacity=config.buffer.capacity,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            device=self.device,
            dtype=self.dtype
        )
        
        # Create vectorized environment for data collection
        self.vec_env, _ = make_env("Pendulum-v1", num_envs=64)  # Use config from yaml
        
        # Training state
        self.epoch = 0
        self.step = 0
        
        self.logger.info("Delta-IRIS trainer initialized")
        
    def collect_experience(self, num_episodes: int = 10) -> List[Episode]:
        """Collect experience episodes using current policy"""
        episodes = []
        
        self.actor_critic.eval()
        
        with torch.no_grad():
            for episode_idx in range(num_episodes):
                obs, _ = self.vec_env.reset()
                episode_obs = [obs]
                episode_actions = []
                episode_rewards = []
                episode_dones = []
                
                done = False
                step_count = 0
                max_steps = 1000  # Prevent infinite episodes
                
                while not done and step_count < max_steps:
                    # Convert to tensor
                    obs_tensor = torch.tensor(obs, device=self.device, dtype=self.dtype)
                    
                    # Get action from policy
                    action, _, _ = self.actor_critic.get_action_and_value(obs_tensor)
                    action_np = action.cpu().numpy()
                    
                    # Step environment
                    next_obs, reward, done, truncated, info = self.vec_env.step(action_np)
                    done = done or truncated
                    
                    # Store transition
                    episode_actions.append(action_np)
                    episode_rewards.append(reward)
                    episode_dones.append(done)
                    episode_obs.append(next_obs)
                    
                    obs = next_obs
                    step_count += 1
                
                # Create episode
                episode = Episode(
                    observations=torch.tensor(episode_obs[:-1], device=self.device, dtype=self.dtype),
                    actions=torch.tensor(episode_actions, device=self.device, dtype=self.dtype),
                    rewards=torch.tensor(episode_rewards, device=self.device, dtype=self.dtype),
                    dones=torch.tensor(episode_dones, device=self.device, dtype=torch.bool)
                )
                
                episodes.append(episode)
                self.buffer.add_episode(episode)
                
        self.logger.info(f"Collected {len(episodes)} episodes")
        return episodes
        
    def train_tokenizer(self, num_steps: int = 100) -> Dict[str, float]:
        """Train the tokenizer component"""
        self.tokenizer.train()
        
        losses = []
        
        for step in range(num_steps):
            # Sample batch from buffer
            try:
                batch = self.buffer.sample_sequences(
                    self.config.buffer.batch_size,
                    self.config.buffer.sequence_length
                )
            except ValueError:
                self.logger.warning("Not enough data in buffer for tokenizer training")
                break
                
            # Forward pass
            obs = batch['observations'][:, :-1]  # All but last
            actions = batch['actions'][:, :-1]   # All but last
            
            tokenizer_output = self.tokenizer(obs, actions)
            
            # Compute loss
            total_loss = tokenizer_output['losses']['total_tokenizer_loss']
            
            # Backward pass
            self.tokenizer_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.tokenizer.parameters(), max_norm=1.0)
            self.tokenizer_optimizer.step()
            
            losses.append({k: v.item() for k, v in tokenizer_output['losses'].items()})
            
            # Sanity check
            if step % 50 == 0:
                self.tokenizer._sanity_check(tokenizer_output)
                
        # Average losses
        avg_losses = {}
        if losses:
            for key in losses[0].keys():
                avg_losses[f'tokenizer_{key}'] = sum(loss[key] for loss in losses) / len(losses)
                
        return avg_losses
        
    def train_world_model(self, num_steps: int = 100) -> Dict[str, float]:
        """Train the world model component"""
        self.world_model.train()
        self.tokenizer.eval()
        
        losses = []
        
        for step in range(num_steps):
            # Sample batch from buffer
            try:
                batch = self.buffer.sample_sequences(
                    self.config.buffer.batch_size,
                    self.config.buffer.sequence_length
                )
            except ValueError:
                self.logger.warning("Not enough data in buffer for world model training")
                break
                
            # Get tokens from tokenizer
            with torch.no_grad():
                obs = batch['observations'][:, :-1]
                actions = batch['actions'][:, :-1]
                tokens = self.tokenizer.get_tokens(obs, actions)
                
            # Prepare targets
            targets = {
                'next_tokens': tokens[:, 1:],  # Shift tokens for next-token prediction
                'rewards': batch['rewards'][:, 1:],
                'dones': batch['dones'][:, 1:].float()
            }
            
            # Forward pass
            predictions = self.world_model(tokens[:, :-1], batch['actions'][:, :-1, 0].long())
            
            # Compute loss
            loss_dict = self.world_model.compute_loss(predictions, targets)
            total_loss = loss_dict['total_world_model_loss']
            
            # Backward pass
            self.world_model_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), max_norm=1.0)
            self.world_model_optimizer.step()
            
            losses.append({k: v.item() for k, v in loss_dict.items()})
            
            # Sanity check
            if step % 50 == 0:
                self.world_model._sanity_check(predictions, targets)
                
        # Average losses
        avg_losses = {}
        if losses:
            for key in losses[0].keys():
                avg_losses[f'world_model_{key}'] = sum(loss[key] for loss in losses) / len(losses)
                
        return avg_losses
        
    def train_actor_critic(self, num_steps: int = 100) -> Dict[str, float]:
        """Train actor-critic with imagination rollouts"""
        self.actor_critic.train()
        self.world_model.eval()
        self.tokenizer.eval()
        
        losses = []
        
        for step in range(num_steps):
            # Sample initial states from buffer
            try:
                batch = self.buffer.sample_sequences(
                    self.config.buffer.batch_size,
                    1  # Just need initial states
                )
            except ValueError:
                self.logger.warning("Not enough data in buffer for actor-critic training")
                break
                
            initial_obs = batch['observations'][:, 0]  # [batch_size, obs_dim]
            
            # Perform imagination rollout
            rollout = self.actor_critic.imagination_rollout(
                self.world_model,
                self.tokenizer,
                initial_obs,
                self.config.actor_critic.imagination_horizon
            )
            
            rollout_data = rollout.get_tensors()
            
            # Compute loss
            loss_dict = self.actor_critic.compute_loss(rollout_data)
            total_loss = loss_dict['total_actor_critic_loss']
            
            # Backward pass
            self.actor_critic_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=1.0)
            self.actor_critic_optimizer.step()
            
            losses.append({k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()})
            
            # Sanity check
            if step % 50 == 0:
                self.actor_critic._sanity_check(rollout_data)
                
        # Average losses
        avg_losses = {}
        if losses:
            for key in losses[0].keys():
                avg_losses[f'actor_critic_{key}'] = sum(loss[key] for loss in losses) / len(losses)
                
        return avg_losses
        
    def evaluate(self) -> Dict[str, float]:
        """Evaluate current policy performance"""
        self.actor_critic.eval()
        
        eval_rewards = []
        eval_lengths = []
        
        with torch.no_grad():
            for _ in range(10):  # 10 evaluation episodes
                obs, _ = self.vec_env.reset()
                episode_reward = 0
                episode_length = 0
                done = False
                max_steps = 1000
                
                while not done and episode_length < max_steps:
                    obs_tensor = torch.tensor(obs, device=self.device, dtype=self.dtype)
                    action, _, _ = self.actor_critic.get_action_and_value(obs_tensor, deterministic=True)
                    
                    obs, reward, done, truncated, _ = self.vec_env.step(action.cpu().numpy())
                    done = done or truncated
                    
                    episode_reward += reward
                    episode_length += 1
                    
                eval_rewards.append(episode_reward)
                eval_lengths.append(episode_length)
                
        return {
            'eval_mean_reward': sum(eval_rewards) / len(eval_rewards),
            'eval_std_reward': torch.tensor(eval_rewards).std().item(),
            'eval_mean_length': sum(eval_lengths) / len(eval_lengths),
            'eval_min_reward': min(eval_rewards),
            'eval_max_reward': max(eval_rewards)
        }
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        start_time = time.time()
        
        # Collect experience
        self.collect_experience(num_episodes=20)
        
        # Train components
        tokenizer_losses = self.train_tokenizer(num_steps=50)
        world_model_losses = self.train_world_model(num_steps=50) 
        actor_critic_losses = self.train_actor_critic(num_steps=50)
        
        # Combine losses
        all_losses = {**tokenizer_losses, **world_model_losses, **actor_critic_losses}
        
        # Add buffer stats
        buffer_stats = self.buffer.get_stats()
        all_losses.update({f'buffer_{k}': v for k, v in buffer_stats.items()})
        
        # Add timing
        all_losses['epoch_time'] = time.time() - start_time
        
        return all_losses
        
    def train(self):
        """Main training loop"""
        self.logger.info("Starting Delta-IRIS training")
        
        # Initialize wandb if configured
        if hasattr(self.config, 'wandb'):
            wandb.init(project="DeltaIris", config=self.config.__dict__)
            
        for epoch in range(self.config.epochs):
            self.epoch = epoch
            
            # Train epoch
            losses = self.train_epoch()
            
            # Evaluate periodically
            if epoch % self.config.eval_frequency == 0:
                eval_metrics = self.evaluate()
                losses.update(eval_metrics)
                
            # Log metrics
            losses['epoch'] = epoch
            
            if hasattr(self.config, 'wandb'):
                wandb.log(losses)
                
            # Print progress
            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}: {losses}")
                
            # Save checkpoint periodically
            if epoch % 100 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")
                
        self.logger.info("Training completed")
        
    def save_checkpoint(self, filename: str):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'tokenizer': self.tokenizer.state_dict(),
            'world_model': self.world_model.state_dict(),
            'actor_critic': self.actor_critic.state_dict(),
            'tokenizer_optimizer': self.tokenizer_optimizer.state_dict(),
            'world_model_optimizer': self.world_model_optimizer.state_dict(),
            'actor_critic_optimizer': self.actor_critic_optimizer.state_dict(),
            'config': self.config
        }
        
        Path("checkpoints").mkdir(exist_ok=True)
        torch.save(checkpoint, f"checkpoints/{filename}")
        self.logger.info(f"Saved checkpoint: {filename}")
        
    def load_checkpoint(self, filename: str):
        """Load training checkpoint"""
        checkpoint = torch.load(f"checkpoints/{filename}", map_location=self.device)
        
        self.epoch = checkpoint['epoch']
        self.tokenizer.load_state_dict(checkpoint['tokenizer'])
        self.world_model.load_state_dict(checkpoint['world_model'])
        self.actor_critic.load_state_dict(checkpoint['actor_critic'])
        self.tokenizer_optimizer.load_state_dict(checkpoint['tokenizer_optimizer'])
        self.world_model_optimizer.load_state_dict(checkpoint['world_model_optimizer'])
        self.actor_critic_optimizer.load_state_dict(checkpoint['actor_critic_optimizer'])
        
        self.logger.info(f"Loaded checkpoint: {filename}")