"""
Actor-Critic with Imagination Training for Delta-IRIS
Trains policy and value function in imagined rollouts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Dict, Tuple, List
import logging
from .config import ActorCriticConfig


class MLPActor(nn.Module):
    """MLP-based policy network"""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass returning action logits"""
        return self.net(obs)
        
    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy"""
        logits = self.forward(obs)
        
        if deterministic:
            action = torch.argmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)[torch.arange(logits.size(0)), action]
        else:
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
        return action, log_prob


class MLPCritic(nn.Module):
    """MLP-based value function network"""
    
    def __init__(self, obs_dim: int, hidden_dim: int):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass returning state values"""
        return self.net(obs).squeeze(-1)


class ImaginationRollout:
    """Container for imagination rollout data"""
    
    def __init__(self):
        self.observations: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.dones: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []
        
    def add_step(self, obs: torch.Tensor, action: torch.Tensor, reward: torch.Tensor,
                 done: torch.Tensor, value: torch.Tensor, log_prob: torch.Tensor):
        """Add a step to the rollout"""
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)
        
    def get_tensors(self) -> Dict[str, torch.Tensor]:
        """Convert lists to tensors"""
        return {
            'observations': torch.stack(self.observations, dim=1),  # [batch, time, obs_dim]
            'actions': torch.stack(self.actions, dim=1),            # [batch, time]
            'rewards': torch.stack(self.rewards, dim=1),            # [batch, time]
            'dones': torch.stack(self.dones, dim=1),                # [batch, time]
            'values': torch.stack(self.values, dim=1),              # [batch, time]
            'log_probs': torch.stack(self.log_probs, dim=1)         # [batch, time]
        }


class ActorCritic(nn.Module):
    """Actor-Critic with imagination training capability"""
    
    def __init__(self, config: ActorCriticConfig):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Policy and value networks
        self.actor = MLPActor(config.obs_dim, config.action_dim, config.hidden_dim)
        self.critic = MLPCritic(config.obs_dim, config.hidden_dim)
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
                    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action logits and values"""
        action_logits = self.actor(obs)
        values = self.critic(obs)
        return action_logits, values
        
    def get_action_and_value(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log probability, and value for given observation"""
        action, log_prob = self.actor.get_action(obs, deterministic)
        value = self.critic(obs)
        return action, log_prob, value
        
    def compute_gae_returns(self, rewards: torch.Tensor, values: torch.Tensor, 
                           dones: torch.Tensor, next_value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE) returns
        
        Args:
            rewards: [batch, time] - rewards
            values: [batch, time] - value function estimates
            dones: [batch, time] - done flags
            next_value: [batch] - value of next state
            
        Returns:
            returns: [batch, time] - GAE returns
            advantages: [batch, time] - GAE advantages
        """
        batch_size, time_steps = rewards.shape
        
        # Compute advantages using GAE
        advantages = torch.zeros_like(rewards)
        next_advantage = 0
        
        # Append next value to values
        extended_values = torch.cat([values, next_value.unsqueeze(1)], dim=1)
        
        for t in reversed(range(time_steps)):
            # Temporal difference error
            td_error = (rewards[:, t] + 
                       self.config.gamma * extended_values[:, t + 1] * (1 - dones[:, t]) - 
                       extended_values[:, t])
            
            # GAE advantage
            advantages[:, t] = (td_error + 
                               self.config.gamma * self.config.lambda_gae * (1 - dones[:, t]) * next_advantage)
            next_advantage = advantages[:, t]
            
        # Returns are advantages + values
        returns = advantages + values
        
        return returns, advantages
        
    def imagination_rollout(self, world_model, tokenizer, initial_obs: torch.Tensor, 
                           horizon: int, last_action: torch.Tensor = None) -> ImaginationRollout:
        """
        Perform imagination rollout using world model
        
        Args:
            world_model: Trained world model for environment simulation
            tokenizer: Tokenizer for encoding/decoding observations
            initial_obs: [batch, obs_dim] - starting observations
            horizon: Number of steps to rollout
            last_action: [batch] - last action taken to reach initial_obs (optional)
            
        Returns:
            ImaginationRollout containing rollout data
        """
        batch_size = initial_obs.shape[0]
        device = initial_obs.device
        
        rollout = ImaginationRollout()
        
        # Initialize current state and token sequence
        current_obs = initial_obs
        
        with torch.no_grad():
            # Get initial tokens from observations with proper action handling
            if last_action is not None:
                # Use the provided last action for better context
                dummy_actions = last_action.unsqueeze(1)  # [batch, 1]
            else:
                # Fallback to dummy action if no context provided
                dummy_actions = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
            
            current_tokens = tokenizer.get_tokens(current_obs.unsqueeze(1), dummy_actions).squeeze(1)
            
        for step in range(horizon):
            # Get action and value from current observation
            action, log_prob, value = self.get_action_and_value(current_obs)
            
            # Prepare inputs for world model (discrete actions)
            tokens_input = current_tokens.unsqueeze(1)  # [batch, 1]
            actions_input = action.unsqueeze(1)  # [batch, 1] - discrete actions
            
            # Get world model predictions
            with torch.no_grad():
                predictions = world_model(tokens_input, actions_input)
                
                # Sample next token with temperature for exploration
                next_token_logits = predictions['next_token_logits'][:, 0]  # [batch, vocab_size]
                
                # Apply temperature for controlled exploration
                temperature = 1.0
                next_token_probs = F.softmax(next_token_logits / temperature, dim=-1)
                next_tokens = torch.multinomial(next_token_probs, 1).squeeze(1)
                
                # Get predicted reward and done with proper thresholding
                predicted_reward = predictions['reward_predictions'][:, 0]  # [batch]
                done_logits = predictions['done_predictions'][:, 0]  # [batch]
                predicted_done = torch.sigmoid(done_logits) > 0.5  # [batch]
                
                # Decode next observation from next tokens using the current action
                next_obs = tokenizer.decode_tokens(next_tokens.unsqueeze(1), action.unsqueeze(1)).squeeze(1)
            
            # Add step to rollout
            rollout.add_step(
                obs=current_obs,
                action=action,
                reward=predicted_reward,
                done=predicted_done.float(),
                value=value,
                log_prob=log_prob
            )
            
            # Update current state and maintain token sequence
            current_obs = next_obs
            current_tokens = next_tokens
            
            # Stop if all environments are done
            if torch.all(predicted_done):
                break
                
        return rollout
        
    def compute_loss(self, rollout_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute actor-critic losses from rollout data
        
        Args:
            rollout_data: Dictionary containing rollout tensors
            
        Returns:
            Dictionary of losses
        """
        observations = rollout_data['observations']
        actions = rollout_data['actions']
        rewards = rollout_data['rewards']
        dones = rollout_data['dones']
        old_values = rollout_data['values']
        old_log_probs = rollout_data['log_probs']
        
        batch_size, time_steps = rewards.shape
        
        # Get current policy outputs
        action_logits, values = self.forward(observations.view(-1, self.config.obs_dim))
        action_logits = action_logits.view(batch_size, time_steps, -1)
        values = values.view(batch_size, time_steps)
        
        # Compute current log probabilities
        action_dist = Categorical(logits=action_logits)
        current_log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy()
        
        # Compute next value (last value in sequence)
        next_value = values[:, -1]  # Use last predicted value as bootstrap
        
        # Compute GAE returns and advantages
        returns, advantages = self.compute_gae_returns(rewards, old_values, dones, next_value)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Policy loss (PPO-style, but simplified)
        ratio = torch.exp(current_log_probs - old_log_probs)
        policy_loss = -torch.mean(ratio * advantages)
        
        # Value loss  
        value_loss = F.mse_loss(values, returns)
        
        # Entropy loss (for exploration)
        entropy_loss = -torch.mean(entropy) * self.config.entropy_coef
        
        # Total loss
        total_loss = policy_loss + value_loss + entropy_loss
        
        losses = {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy_loss': entropy_loss,
            'total_actor_critic_loss': total_loss
        }
        
        # Add metrics
        losses.update({
            'mean_entropy': torch.mean(entropy),
            'mean_value': torch.mean(values),
            'mean_advantage': torch.mean(advantages),
            'mean_return': torch.mean(returns)
        })
        
        return losses
        
    def _sanity_check(self, rollout_data: Dict[str, torch.Tensor]):
        """Perform sanity checks on rollout data"""
        
        for key, tensor in rollout_data.items():
            if torch.isnan(tensor).any():
                self.logger.warning(f"NaN values in {key}")
            if torch.isinf(tensor).any():
                self.logger.warning(f"Inf values in {key}")
                
        # Check value ranges
        values = rollout_data['values']
        if values.max() > 1000 or values.min() < -1000:
            self.logger.warning(f"Value function out of reasonable range: [{values.min():.3f}, {values.max():.3f}]")
            
        # Log statistics
        if hasattr(self, '_ac_step_count'):
            self._ac_step_count += 1
            if self._ac_step_count % 100 == 0:
                rewards = rollout_data['rewards']
                self.logger.debug(f"Mean rollout reward: {rewards.mean():.3f}")
                self.logger.debug(f"Mean rollout length: {(1 - rollout_data['dones']).sum(dim=1).float().mean():.1f}")
        else:
            self._ac_step_count = 1
