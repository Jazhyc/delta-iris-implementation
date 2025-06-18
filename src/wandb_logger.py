"""
Wandb Logger for Delta-IRIS
Handles all wandb logging functionality separately from the main trainer
"""

import wandb
import numpy as np
import time
from typing import Dict, List, Any


class WandbLogger:
    """Wandb logging utility for Delta-IRIS training"""
    
    def __init__(self):
        self.step = 0
        self.episode_rewards = []
        self.episode_lengths = []
        
    def setup(self):
        """Setup wandb metric definitions"""
        try:
            # Define custom metrics with step tracking
            wandb.define_metric("step")
            wandb.define_metric("epoch")
            
            # Component-specific metrics
            wandb.define_metric("experience_collection/*", step_metric="step")
            wandb.define_metric("tokenizer/*", step_metric="step")
            wandb.define_metric("world_model/*", step_metric="step") 
            wandb.define_metric("actor_critic/*", step_metric="step")
            wandb.define_metric("evaluation/*", step_metric="step")
            wandb.define_metric("training/*", step_metric="step")
            wandb.define_metric("buffer/*", step_metric="step")
            wandb.define_metric("timing/*", step_metric="step")
            wandb.define_metric("best/*", step_metric="step")
            wandb.define_metric("losses/*", step_metric="epoch")
            
        except Exception as e:
            print(f"Warning: Could not configure wandb plots: {e}")
    
    def log_experience_collection(self, episode_rewards: List[float], episode_lengths: List[int], 
                                collection_time: float, num_episodes: int):
        """Log experience collection metrics"""
        self.episode_rewards.extend(episode_rewards)
        self.episode_lengths.extend(episode_lengths)
        
        wandb.log({
            'experience_collection/num_episodes': num_episodes,
            'experience_collection/mean_episode_reward': np.mean(episode_rewards),
            'experience_collection/std_episode_reward': np.std(episode_rewards),
            'experience_collection/min_episode_reward': np.min(episode_rewards),
            'experience_collection/max_episode_reward': np.max(episode_rewards),
            'experience_collection/mean_episode_length': np.mean(episode_lengths),
            'experience_collection/collection_time': collection_time,
            'experience_collection/episodes_per_second': num_episodes / collection_time,
            'step': self.step
        })
    
    def log_tokenizer_training(self, losses: List[Dict], training_time: float, step_offset: int = 0):
        """Log tokenizer training metrics"""
        if not losses:
            return
            
        step_losses = [loss.get('total_tokenizer_loss', 0) for loss in losses]
        
        # Log step-wise metrics every 10 steps
        for i, loss_dict in enumerate(losses):
            if i % 10 == 0:
                wandb.log({
                    'tokenizer/step_loss': loss_dict.get('total_tokenizer_loss', 0),
                    'tokenizer/reconstruction_loss': loss_dict.get('reconstruction_loss_l2', 0),
                    'tokenizer/vq_loss': loss_dict.get('vq_loss', 0),
                    'tokenizer/commitment_loss': loss_dict.get('commitment_loss', 0),
                    'tokenizer/perplexity': loss_dict.get('perplexity', 0),
                    'step': self.step + step_offset + i
                })
        
        # Log summary metrics
        wandb.log({
            'tokenizer/training_time': training_time,
            'tokenizer/avg_loss': np.mean(step_losses),
            'tokenizer/loss_std': np.std(step_losses),
            'tokenizer/final_loss': step_losses[-1] if step_losses else 0,
            'tokenizer/steps_completed': len(step_losses),
            'step': self.step
        })
    
    def log_world_model_training(self, losses: List[Dict], training_time: float, step_offset: int = 0):
        """Log world model training metrics"""
        if not losses:
            return
            
        step_losses = [loss.get('total_world_model_loss', 0) for loss in losses]
        
        # Log step-wise metrics every 10 steps
        for i, loss_dict in enumerate(losses):
            if i % 10 == 0:
                wandb.log({
                    'world_model/step_loss': loss_dict.get('total_world_model_loss', 0),
                    'world_model/next_token_loss': loss_dict.get('next_token_loss', 0),
                    'world_model/reward_loss': loss_dict.get('reward_loss', 0),
                    'world_model/done_loss': loss_dict.get('done_loss', 0),
                    'world_model/token_accuracy': loss_dict.get('token_accuracy', 0),
                    'step': self.step + step_offset + i
                })
        
        # Log summary metrics
        wandb.log({
            'world_model/training_time': training_time,
            'world_model/avg_loss': np.mean(step_losses),
            'world_model/loss_std': np.std(step_losses),
            'world_model/final_loss': step_losses[-1] if step_losses else 0,
            'world_model/steps_completed': len(step_losses),
            'step': self.step
        })
    
    def log_actor_critic_training(self, losses: List[Dict], training_time: float, step_offset: int = 0):
        """Log actor-critic training metrics"""
        if not losses:
            return
            
        step_losses = [loss.get('total_actor_critic_loss', 0) for loss in losses]
        
        # Log step-wise metrics every 10 steps
        for i, loss_dict in enumerate(losses):
            if i % 10 == 0:
                wandb.log({
                    'actor_critic/step_loss': loss_dict.get('total_actor_critic_loss', 0),
                    'actor_critic/value_loss': loss_dict.get('value_loss', 0),
                    'actor_critic/policy_loss': loss_dict.get('policy_loss', 0),
                    'actor_critic/entropy_loss': loss_dict.get('entropy_loss', 0),
                    'actor_critic/advantage_mean': loss_dict.get('advantage_mean', 0),
                    'actor_critic/advantage_std': loss_dict.get('advantage_std', 0),
                    'actor_critic/value_mean': loss_dict.get('value_mean', 0),
                    'step': self.step + step_offset + i
                })
        
        # Log summary metrics
        wandb.log({
            'actor_critic/training_time': training_time,
            'actor_critic/avg_loss': np.mean(step_losses),
            'actor_critic/loss_std': np.std(step_losses),
            'actor_critic/final_loss': step_losses[-1] if step_losses else 0,
            'actor_critic/steps_completed': len(step_losses),
            'step': self.step
        })
    
    def log_evaluation(self, eval_rewards: List[float], eval_lengths: List[int], 
                      eval_actions: List[Any], eval_time: float):
        """Log evaluation metrics"""
        eval_metrics = {
            'evaluation/mean_reward': np.mean(eval_rewards),
            'evaluation/std_reward': np.std(eval_rewards),
            'evaluation/min_reward': np.min(eval_rewards),
            'evaluation/max_reward': np.max(eval_rewards),
            'evaluation/median_reward': np.median(eval_rewards),
            'evaluation/mean_length': np.mean(eval_lengths),
            'evaluation/evaluation_time': eval_time,
            'evaluation/episodes_evaluated': len(eval_rewards),
            'step': self.step
        }
        
        wandb.log(eval_metrics)
        
        # Log action distribution
        if eval_actions:
            action_array = np.array(eval_actions)
            if action_array.ndim > 1:
                action_array = action_array.flatten()
            
            unique_actions, action_counts = np.unique(action_array, return_counts=True)
            action_dist = {f'evaluation/action_{int(action)}_freq': count/len(action_array) 
                          for action, count in zip(unique_actions, action_counts)}
            
            wandb.log(action_dist)
        
        return eval_metrics
    
    def log_epoch_summary(self, epoch: int, losses: Dict, buffer_stats: Dict, 
                         epoch_time: float, timing_stats: Dict):
        """Log epoch-level summary metrics"""
        self.step += 1
        
        # Log buffer stats
        wandb.log({
            'buffer/size': buffer_stats.get('size', 0),
            'buffer/num_episodes': buffer_stats.get('num_episodes', 0),
            'buffer/capacity_usage': buffer_stats.get('size', 0) / buffer_stats.get('capacity', 1),
            'step': self.step
        })
        
        # Log timing breakdown
        wandb.log({
            'timing/epoch_time': epoch_time,
            'timing/experience_collection_time': timing_stats.get('experience_collection_time', 0),
            'timing/tokenizer_training_time': timing_stats.get('tokenizer_training_time', 0),
            'timing/world_model_training_time': timing_stats.get('world_model_training_time', 0),
            'timing/actor_critic_training_time': timing_stats.get('actor_critic_training_time', 0),
            'step': self.step
        })
        
        # Log comprehensive epoch metrics
        epoch_log = {
            'epoch': epoch,
            'learning/epoch': epoch,
            **{f'losses/{k}': v for k, v in losses.items() if isinstance(v, (int, float))},
            'step': self.step
        }
        wandb.log(epoch_log)
        
        # Log rolling training statistics
        if len(self.episode_rewards) > 0:
            recent_rewards = self.episode_rewards[-100:]  # Last 100 episodes
            wandb.log({
                'training/recent_mean_reward': np.mean(recent_rewards),
                'training/recent_std_reward': np.std(recent_rewards),
                'training/total_episodes': len(self.episode_rewards),
                'step': self.step
            })
    
    def log_best_performance(self, reward: float, epoch: int):
        """Log best performance tracking"""
        wandb.log({
            'best/eval_reward': reward,
            'best/epoch': epoch,
            'step': self.step
        })
    
    def log_final_summary(self, final_eval: Dict, best_reward: float, total_epochs: int):
        """Log final training summary"""
        wandb.log({
            'final/eval_mean_reward': final_eval.get('evaluation/mean_reward', 0),
            'final/eval_std_reward': final_eval.get('evaluation/std_reward', 0),
            'final/total_epochs': total_epochs,
            'final/best_eval_reward': best_reward,
            'final/total_episodes_collected': len(self.episode_rewards),
            'step': self.step
        })
        
        # Create summary table
        summary_table = wandb.Table(
            columns=["Metric", "Value"],
            data=[
                ["Best Evaluation Reward", best_reward],
                ["Final Evaluation Reward", final_eval.get('evaluation/mean_reward', 0)],
                ["Total Episodes Collected", len(self.episode_rewards)],
                ["Total Epochs", total_epochs]
            ]
        )
        wandb.log({"training_summary": summary_table})
    
    def log_config(self, config_dict: Dict):
        """Log initial configuration"""
        wandb.log({
            'config/obs_dim': config_dict.get('obs_dim'),
            'config/action_dim': config_dict.get('action_dim'),
            'config/device': config_dict.get('device'),
            'config/epochs': config_dict.get('epochs'),
            'config/buffer_capacity': config_dict.get('buffer_capacity'),
            'config/batch_size': config_dict.get('batch_size'),
            'config/sequence_length': config_dict.get('sequence_length'),
            'step': 0
        })
    
    def log_model_info(self, tokenizer_params, world_model_params, actor_critic_params, total_params,
                       tokenizer, world_model, actor_critic):
        """Log model architecture and parameter information to wandb"""
        wandb.log({
            'model_info/tokenizer_parameters': tokenizer_params,
            'model_info/world_model_parameters': world_model_params,
            'model_info/actor_critic_parameters': actor_critic_params,
            'model_info/total_parameters': total_params,
            'model_info/parameters_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'step': 0
        })
        architecture_table = wandb.Table(
            columns=["Component", "Parameters", "Architecture"],
            data=[
                ["Tokenizer", f"{tokenizer_params:,}", str(tokenizer)],
                ["World Model", f"{world_model_params:,}", str(world_model)],
                ["Actor-Critic", f"{actor_critic_params:,}", str(actor_critic)],
                ["Total", f"{total_params:,}", "Complete Delta-IRIS Model"]
            ]
        )
        wandb.log({"model_architecture": architecture_table})

    def log_gradients(self, model_name, total_norm, grad_norms, step):
        """Log gradient statistics for a model"""
        wandb.log({
            f'{model_name}/grad_norm_total': total_norm,
            f'{model_name}/grad_norm_mean': np.mean(grad_norms),
            f'{model_name}/grad_norm_std': np.std(grad_norms),
            f'{model_name}/grad_norm_max': np.max(grad_norms),
            f'{model_name}/grad_norm_min': np.min(grad_norms),
            'step': step
        })

    def log_system_info(self, cpu_percent, memory, gpu_info, step):
        """Log system resource information"""
        wandb.log({
            'system/cpu_percent': cpu_percent,
            'system/memory_percent': memory.percent,
            'system/memory_available_gb': memory.available / 1024**3,
            'system/memory_used_gb': memory.used / 1024**3,
            **gpu_info,
            'step': step
        })

    def log_learning_rates(self, tokenizer_lr, world_model_lr, actor_critic_lr, step):
        """Log current learning rates"""
        wandb.log({
            'learning_rates/tokenizer': tokenizer_lr,
            'learning_rates/world_model': world_model_lr,
            'learning_rates/actor_critic': actor_critic_lr,
            'step': step
        })
