"""
Delta-IRIS Trainer
Main training loop coordinating tokenizer, world model, and actor-critic
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm

from agent.config import TrainerConfig
from data import Episode, Batch, EpisodeDataset, BatchSampler, EpisodeCountManager, collate_segments_to_batch
from agent.tokenizer import Tokenizer
from agent.world_model import WorldModel
from agent.actor_critic import ActorCritic
from env import make_env
from wandb_logger import WandbLogger


class DeltaIrisTrainer:
    """Main trainer for Delta-IRIS algorithm"""
    
    def __init__(self, config: TrainerConfig, hydra_config=None):
        self.config = config
        self.hydra_config = hydra_config
        self.device = torch.device(config.device)
        self.dtype = getattr(torch, config.dtype)
        
        # Initialize wandb logger
        self.wandb_logger = WandbLogger()
        
        # Environment setup - use hydra config if available
        env_name = hydra_config.env.name
        self.env, _ = make_env(env_name, num_envs=1)  # Single env for initialization
        
        # Get observation dimension properly
        if hasattr(self.env.observation_space, 'shape') and len(self.env.observation_space.shape) > 0:
            # For vectorized environments, get the feature dimension (last dimension)
            self.obs_dim = self.env.observation_space.shape[-1]
        else:
            self.obs_dim = self.env.observation_space.n
        
        # Determine environment type and action space
        if hasattr(self.env.action_space, 'n'):
            # Standard discrete action space (e.g., Discrete(4))
            self.raw_action_dim = self.env.action_space.n
            self.action_dim = self.env.action_space.n
        elif hasattr(self.env.action_space, 'nvec'):
            # MultiDiscrete action space (e.g., MultiDiscrete([2]))
            self.raw_action_dim = int(self.env.action_space.nvec[0])
            self.action_dim = int(self.env.action_space.nvec[0])
        elif hasattr(self.env.action_space, 'shape'):
            # Continuous action space (e.g., Box) - not supported
            raise ValueError("Continuous action spaces are not supported yet. Please use a discrete environment.")
        else:
            raise ValueError(f"Unknown action space type: {type(self.env.action_space)}")
            
        # Update config with correct dimensions
        print(f"DEBUG: Updating configs with obs_dim={self.obs_dim}, action_dim={self.action_dim}")
        config.tokenizer.obs_dim = self.obs_dim
        config.tokenizer.action_dim = self.action_dim
        print(f"DEBUG: Tokenizer config after update: obs_dim={config.tokenizer.obs_dim}, action_dim={config.tokenizer.action_dim}")
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
        
        # Initialize experience dataset and episode count manager
        from pathlib import Path
        dataset_dir = Path("dataset")
        dataset_dir.mkdir(exist_ok=True)
        
        self.train_dataset = EpisodeDataset(
            directory=dataset_dir / 'train',
            name='train_dataset'
        )
        
        self.episode_count_manager = EpisodeCountManager(self.train_dataset)
        self.episode_count_manager.register('tokenizer', 'world_model', 'actor_critic')
        
        # Create single environment for data collection (simpler than vectorized)
        self.single_env, _ = make_env(env_name, num_envs=1)
        
        # Training state
        self.epoch = 0
        self.step = 0
        
    def collect_experience(self, num_episodes: int = 10) -> List[Episode]:
        """Collect experience episodes using current policy"""
        start_time = time.time()
        episodes = []
        episode_rewards_batch = []
        episode_lengths_batch = []
        
        self.actor_critic.eval()
        
        with torch.no_grad():
            # Add progress bar for experience collection
            episode_pbar = tqdm(range(num_episodes), desc="Collecting Experience", leave=False)
            for episode_idx in episode_pbar:
                obs, _ = self.single_env.reset()
                
                episode_obs = [obs]
                episode_actions = []
                episode_rewards = []
                episode_dones = []
                
                done = False
                step_count = 0
                max_steps = 1000  # Prevent infinite episodes
                
                while not done and step_count < max_steps:
                    # Convert obs to proper tensor format
                    obs_tensor = obs.to(device=self.device, dtype=self.dtype)
                    
                    # Get action from policy
                    action, _, _ = self.actor_critic.get_action_and_value(obs_tensor)
                    action_np = action.cpu().numpy()
                    
                    # Step environment
                    next_obs, reward, done_flag, truncated_flag, info = self.single_env.step(action_np)
                    
                    # Handle environment outputs (simple boolean logic)
                    done = bool(done_flag) or bool(truncated_flag)
                    
                    # Store transition
                    episode_actions.append(action_np)
                    episode_rewards.append(float(reward))
                    episode_dones.append(1 if done else 0)  # Convert to int for consistency
                    
                    episode_obs.append(next_obs)
                    obs = next_obs
                    step_count += 1
                
                # Create episode from collected data
                obs_tensor = torch.stack(episode_obs[:-1])
                actions_tensor = torch.stack([torch.from_numpy(action) for action in episode_actions])
                rewards_tensor = torch.tensor(episode_rewards, dtype=self.dtype)
                dones_tensor = torch.tensor(episode_dones, dtype=torch.long)  # LongTensor for consistency
                
                # Handle potential extra dimensions from vectorized environment
                if obs_tensor.dim() > 2:
                    obs_tensor = obs_tensor.squeeze(1)
                if actions_tensor.dim() > 2:
                    actions_tensor = actions_tensor.squeeze(1)
                
                episode = Episode(
                    observations=obs_tensor.to(device=self.device, dtype=self.dtype),
                    actions=actions_tensor.to(device=self.device, dtype=torch.long),
                    rewards=rewards_tensor.to(device=self.device),
                    ends=dones_tensor.to(device=self.device)
                )
                
                episodes.append(episode)
                # Add episode to dataset
                episode_id = self.train_dataset.add_episode(episode)
                self.episode_count_manager.add_episode(episode_id)
                
                # Track episode stats
                episode_reward = sum(episode_rewards)
                episode_length = len(episode_rewards)
                episode_rewards_batch.append(episode_reward)
                episode_lengths_batch.append(episode_length)
        
        collection_time = time.time() - start_time
        self.wandb_logger.log_experience_collection(
            episode_rewards_batch, episode_lengths_batch, collection_time, num_episodes
        )
        
        return episodes
        
    def train_tokenizer(self, num_steps: int = 100) -> Dict[str, float]:
        """Train the tokenizer component"""
        start_time = time.time()
        self.tokenizer.train()
        
        losses = []
        
        # Create batch sampler
        batch_sampler = BatchSampler(
            self.train_dataset,
            num_steps,
            self.config.data.batch_size,
            self.config.data.sequence_length,
            can_sample_beyond_end=True
        )
        batch_sampler.probabilities = self.episode_count_manager.compute_probabilities(
            key='tokenizer', 
            alpha=1.0  # Priority alpha
        )
        
        # Create data loader
        from torch.utils.data import DataLoader
        loader = DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            num_workers=0,  # Set to 0 for CartPole simplicity
            collate_fn=collate_segments_to_batch,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        step_pbar = tqdm(enumerate(loader), total=num_steps, desc="Training Tokenizer", leave=False)
        for step, batch in step_pbar:
            if step >= num_steps:
                break
                
            batch = batch.to(self.device)
            
            # Extract sequences (remove last observation as we predict next)
            obs = batch.observations[:, :-1]
            actions = batch.actions[:, :-1]
            
            tokenizer_output = self.tokenizer(obs, actions)
            
            total_loss = tokenizer_output['losses']['total_tokenizer_loss']
            
            self.tokenizer_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.tokenizer.parameters(), max_norm=1.0)
            self.tokenizer_optimizer.step()
            
            loss_dict = {k: v.item() for k, v in tokenizer_output['losses'].items()}
            losses.append(loss_dict)
            
            # Update episode counts for priority sampling
            for segment_id in batch.segment_ids:
                self.episode_count_manager.increment_episode_count('tokenizer', segment_id.episode_id)
            
            if step % 50 == 0 and step > 0:
                # Update probabilities for next batch
                batch_sampler.probabilities = self.episode_count_manager.compute_probabilities(
                    key='tokenizer', alpha=1.0
                )
                if hasattr(self.tokenizer, '_sanity_check'):
                    self.tokenizer._sanity_check(tokenizer_output)
        
        training_time = time.time() - start_time
        self.wandb_logger.log_tokenizer_training(losses, training_time)
        
        avg_losses = {}
        if losses:
            for key in losses[0].keys():
                avg_losses[f'tokenizer_{key}'] = sum(loss[key] for loss in losses) / len(losses)
                
        return avg_losses
        
    def train_world_model(self, num_steps: int = 100) -> Dict[str, float]:
        """Train the world model component"""
        start_time = time.time()
        self.world_model.train()
        self.tokenizer.eval()
        
        losses = []
        
        # Create batch sampler
        batch_sampler = BatchSampler(
            self.train_dataset,
            num_steps,
            self.config.data.batch_size,
            self.config.data.sequence_length,
            can_sample_beyond_end=True
        )
        batch_sampler.probabilities = self.episode_count_manager.compute_probabilities(
            key='world_model', 
            alpha=1.0
        )
        
        # Create data loader
        from torch.utils.data import DataLoader
        loader = DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            num_workers=0,
            collate_fn=collate_segments_to_batch,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        step_pbar = tqdm(enumerate(loader), total=num_steps, desc="Training World Model", leave=False)
        for step, batch in step_pbar:
            if step >= num_steps:
                break
                
            batch = batch.to(self.device)
            
            with torch.no_grad():
                obs = batch.observations[:, :-1]
                actions = batch.actions[:, :-1]
                tokens = self.tokenizer.get_tokens(obs, actions)
                
            target_seq_len = tokens.shape[1] - 1
            targets = {
                'next_tokens': tokens[:, 1:1+target_seq_len],
                'rewards': batch.rewards[:, 1:1+target_seq_len],
                'dones': batch.ends[:, 1:1+target_seq_len].float()  # Convert ends to float
            }
            
            actions_for_wm = batch.actions[:, :target_seq_len]
            actions_discrete = actions_for_wm[:, :, 0].long() if actions_for_wm.dim() > 2 else actions_for_wm.long()
            
            predictions = self.world_model(tokens[:, :-1], actions_discrete)
            
            loss_dict = self.world_model.compute_loss(predictions, targets)
            total_loss = loss_dict['total_world_model_loss']
            
            self.world_model_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), max_norm=1.0)
            self.world_model_optimizer.step()
            
            loss_values = {k: v.item() for k, v in loss_dict.items()}
            losses.append(loss_values)
            
            # Update episode counts
            for segment_id in batch.segment_ids:
                self.episode_count_manager.increment_episode_count('world_model', segment_id.episode_id)
            
            if step % 50 == 0 and step > 0:
                batch_sampler.probabilities = self.episode_count_manager.compute_probabilities(
                    key='world_model', alpha=1.0
                )
                if hasattr(self.world_model, '_sanity_check'):
                    self.world_model._sanity_check(predictions, targets)
        
        training_time = time.time() - start_time
        self.wandb_logger.log_world_model_training(losses, training_time)
        
        avg_losses = {}
        if losses:
            for key in losses[0].keys():
                avg_losses[f'world_model_{key}'] = sum(loss[key] for loss in losses) / len(losses)
                
        return avg_losses
        
    def train_actor_critic(self, num_steps: int = 100) -> Dict[str, float]:
        """Train actor-critic with imagination rollouts"""
        start_time = time.time()
        self.actor_critic.train()
        self.world_model.eval()
        self.tokenizer.eval()
        
        losses = []
        
        # Create batch sampler (smaller sequence length for actor-critic burn-in)
        batch_sampler = BatchSampler(
            self.train_dataset,
            num_steps,
            self.config.data.batch_size,
            sequence_length=1,  # Single timestep for initial state
            can_sample_beyond_end=False
        )
        batch_sampler.probabilities = self.episode_count_manager.compute_probabilities(
            key='actor_critic', 
            alpha=1.0
        )
        
        # Create data loader
        from torch.utils.data import DataLoader
        loader = DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            num_workers=0,
            collate_fn=collate_segments_to_batch,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        step_pbar = tqdm(enumerate(loader), total=num_steps, desc="Training Actor-Critic", leave=False)
        for step, batch in step_pbar:
            if step >= num_steps:
                break
                
            batch = batch.to(self.device)
            
            initial_obs = batch.observations[:, 0]
            initial_actions = batch.actions[:, 0]
            
            rollout = self.actor_critic.imagination_rollout(
                self.world_model,
                self.tokenizer,
                initial_obs,
                self.config.actor_critic.imagination_horizon
            )
            
            rollout_data = rollout.get_tensors()
            
            loss_dict = self.actor_critic.compute_loss(rollout_data)
            total_loss = loss_dict['total_actor_critic_loss']
            
            self.actor_critic_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=1.0)
            self.actor_critic_optimizer.step()
            
            loss_values = {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
            losses.append(loss_values)
            
            # Update episode counts
            for segment_id in batch.segment_ids:
                self.episode_count_manager.increment_episode_count('actor_critic', segment_id.episode_id)
            
            if step % 50 == 0 and step > 0:
                batch_sampler.probabilities = self.episode_count_manager.compute_probabilities(
                    key='actor_critic', alpha=1.0
                )
                if hasattr(self.actor_critic, '_sanity_check'):
                    self.actor_critic._sanity_check(rollout_data)
        
        training_time = time.time() - start_time
        self.wandb_logger.log_actor_critic_training(losses, training_time)
        
        avg_losses = {}
        if losses:
            for key in losses[0].keys():
                avg_losses[f'actor_critic_{key}'] = sum(loss[key] for loss in losses) / len(losses)
                
        return avg_losses
        
    def evaluate(self) -> Dict[str, float]:
        """Evaluate current policy performance"""
        start_time = time.time()
        self.actor_critic.eval()
        
        eval_rewards = []
        eval_lengths = []
        eval_actions = []
        
        with torch.no_grad():
            eval_pbar = tqdm(range(10), desc="Evaluating Policy", leave=False)
            for eval_ep in eval_pbar:
                obs, _ = self.single_env.reset()
                episode_reward = 0
                episode_length = 0
                episode_actions = []
                done = False
                max_steps = 1000
                
                while not done and episode_length < max_steps:
                    obs_tensor = obs.to(device=self.device, dtype=self.dtype)
                    action, _, _ = self.actor_critic.get_action_and_value(obs_tensor, deterministic=True)
                    
                    obs, reward, done, truncated, _ = self.single_env.step(action.cpu().numpy())
                    done = done or truncated
                    
                    # Convert tensor values to scalars
                    reward = reward.item()
                    done = done.item()
                    truncated = truncated.item()
                    
                    episode_reward += reward
                    episode_length += 1
                    episode_actions.append(action.cpu().numpy())
                    
                eval_rewards.append(episode_reward)
                eval_lengths.append(episode_length)
                eval_actions.extend(episode_actions)
        
        eval_time = time.time() - start_time
        return self.wandb_logger.log_evaluation(eval_rewards, eval_lengths, eval_actions, eval_time)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        start_time = time.time()
        
        self.collect_experience(num_episodes=20)
        
        tokenizer_losses = self.train_tokenizer(num_steps=50)
        world_model_losses = self.train_world_model(num_steps=50) 
        actor_critic_losses = self.train_actor_critic(num_steps=50)
        
        all_losses = {**tokenizer_losses, **world_model_losses, **actor_critic_losses}
        
        # Get dataset stats instead of buffer stats
        dataset_stats = {
            'num_episodes': self.train_dataset.num_episodes,
            'num_steps': self.train_dataset.num_steps,
            'cache_size': len(self.train_dataset._episode_cache)
        }
        all_losses.update({f'dataset_{k}': v for k, v in dataset_stats.items()})
        
        epoch_time = time.time() - start_time
        all_losses['epoch_time'] = epoch_time
        
        return all_losses
        
    def train(self):
        """Main training loop"""
        
        self.wandb_logger.setup()
        self.wandb_logger.log_config({
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'device': str(self.device),
            'epochs': self.config.epochs,
            'batch_size': self.config.data.batch_size,
            'sequence_length': self.config.data.sequence_length
        })
        
        best_eval_reward = float('-inf')
        
        epoch_pbar = tqdm(range(self.config.epochs), desc="Training Epochs")
        for epoch in epoch_pbar:
            self.epoch = epoch
            
            losses = self.train_epoch()
            
            eval_metrics = {}
            if epoch % self.config.eval_frequency == 0:
                eval_metrics = self.evaluate()
                losses.update(eval_metrics)
                
                current_reward = eval_metrics.get('eval_mean_reward', float('-inf'))
                if current_reward > best_eval_reward:
                    best_eval_reward = current_reward
                    self.wandb_logger.log_best_performance(best_eval_reward, epoch)
                    self.save_checkpoint(f"best_model_epoch_{epoch}.pt")
            
            # Get dataset stats instead of buffer stats
            dataset_stats = {
                'num_episodes': self.train_dataset.num_episodes,
                'num_steps': self.train_dataset.num_steps,
                'cache_size': len(self.train_dataset._episode_cache)
            }
            timing_stats = {'epoch_time': losses.get('epoch_time', 0)}
            
            # Log learning rates
            self.wandb_logger.log_learning_rates(
                self.tokenizer_optimizer.param_groups[0]['lr'],
                self.world_model_optimizer.param_groups[0]['lr'],
                self.actor_critic_optimizer.param_groups[0]['lr'],
                self.step
            )
            
            self.wandb_logger.log_epoch_summary(epoch, losses, dataset_stats, 
                                              losses.get('epoch_time', 0), timing_stats)
                
            if epoch % 100 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")
        
        final_eval = self.evaluate()
        self.wandb_logger.log_final_summary(final_eval, best_eval_reward, self.config.epochs)
        
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
        
        # Save dataset info and episode counts
        self.train_dataset.save_info()
        self.episode_count_manager.save(Path("checkpoints/episode_counts.pt"))
        
    def load_checkpoint(self, filename: str):
        """Load training checkpoint"""
        checkpoint = torch.load(f"checkpoints/{filename}", map_location=self.device, weights_only=False)
        
        self.epoch = checkpoint['epoch']
        self.tokenizer.load_state_dict(checkpoint['tokenizer'])
        self.world_model.load_state_dict(checkpoint['world_model'])
        self.actor_critic.load_state_dict(checkpoint['actor_critic'])
        self.tokenizer_optimizer.load_state_dict(checkpoint['tokenizer_optimizer'])
        self.world_model_optimizer.load_state_dict(checkpoint['world_model_optimizer'])
        self.actor_critic_optimizer.load_state_dict(checkpoint['actor_critic_optimizer'])
        
        # Load episode counts if available
        episode_counts_path = Path("checkpoints/episode_counts.pt")
        if episode_counts_path.exists():
            self.episode_count_manager.load(episode_counts_path)