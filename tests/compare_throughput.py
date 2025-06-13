import jax
import jax.numpy as jnp
import gymnax
import gymnasium as gym
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from functools import partial
from typing import Union
from gymnasium.error import DependencyNotInstalled
from gymnasium.wrappers.array_conversion import (
    array_conversion,
    module_namespace,
)
from gymnax.wrappers.gym import GymnaxToVectorGymWrapper

device_jax = jax.devices()[0]
device_torch = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_ENVS = 2048

torch_to_jax = partial(array_conversion, xp=module_namespace(jnp), device=device_jax)
jax_to_torch = partial(array_conversion, xp=module_namespace(torch), device=device_torch)

class SimplePolicy(nn.Module):
    """Simple neural network that returns a fixed action regardless of input"""
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.linear = nn.Linear(obs_dim, action_dim)
        # Initialize weights to return fixed action
        with torch.no_grad():
            self.linear.weight.fill_(0.0)
            self.linear.bias.fill_(1.0)  # Returns [1.0] for pendulum torque
    
    def forward(self, x):
        return self.linear(x)

def benchmark_gymnax(num_steps=100_000):
    """Benchmark Gymnax Pendulum environment with parallel rollouts"""
    print("Benchmarking Gymnax Pendulum-v1 (Parallel Rollouts)...")
    
    # Calculate steps per environment to achieve total steps
    steps_per_env = num_steps // NUM_ENVS
    actual_total_steps = steps_per_env * NUM_ENVS
    print(f"Running {steps_per_env} steps per environment across {NUM_ENVS} environments")
    print(f"Total steps: {actual_total_steps} (requested: {num_steps})")
    
    # Create simple policy network
    policy = SimplePolicy(obs_dim=3, action_dim=1)  # Pendulum has 3D obs, 1D action
    policy.eval()
    
    # Move policy to the appropriate device
    policy.to(device_torch)
    
    # Create gymnax environment and wrap it
    env, env_params = gymnax.make("Pendulum-v1")
    wrapped_env = GymnaxToVectorGymWrapper(env, num_envs=NUM_ENVS, params=env_params, seed=0)
    
    # Reset all environments
    obs, _ = wrapped_env.reset(seed=0)
    
    start_time = time.time()
    
    total_episodes_done = 0
    for step in tqdm(range(steps_per_env)):
        # Convert observations to torch and get actions
        obs_torch = jax_to_torch(obs)
        with torch.no_grad():
            actions = policy(obs_torch)
            actions = torch_to_jax(actions)
        
        # Step all environments
        next_obs, rewards, terminated, truncated, infos = wrapped_env.step(actions)
        
        # Count completed episodes
        done = terminated | truncated
        total_episodes_done += np.sum(done)
        
        # Update observations (wrapper handles resets automatically)
        obs = next_obs
    
    print(f"Completed {total_episodes_done} episodes out of {actual_total_steps} total steps across {NUM_ENVS} environments")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    return total_time, actual_total_steps / total_time

def benchmark_gymnasium(num_steps=100_000):
    """Benchmark Gymnasium Pendulum environment with vectorized environments"""
    print("\nBenchmarking Gymnasium Pendulum-v1 (Vectorized)...")
    
    # Calculate steps per environment to achieve total steps
    steps_per_env = num_steps // NUM_ENVS
    actual_total_steps = steps_per_env * NUM_ENVS
    print(f"Running {steps_per_env} steps per environment across {NUM_ENVS} environments")
    print(f"Total steps: {actual_total_steps} (requested: {num_steps})")
    
    # Create simple policy network
    policy = SimplePolicy(obs_dim=3, action_dim=1)  # Pendulum has 3D obs, 1D action
    policy.eval()
    policy.to(device_torch)
    
    # Create vectorized gymnasium environments
    env = gym.vector.SyncVectorEnv([lambda: gym.make("Pendulum-v1") for _ in range(NUM_ENVS)])
    obs, _ = env.reset(seed=list(range(NUM_ENVS)))
    
    start_time = time.time()
    
    total_episodes_done = 0
    for i in tqdm(range(steps_per_env), desc="Gymnasium vectorized steps"):
        # Use network to get actions for all environments
        obs_torch = torch.from_numpy(obs).float().to(device_torch)
        with torch.no_grad():
            actions_torch = policy(obs_torch)
            actions = actions_torch.cpu().numpy()
        
        # Perform the step transition
        next_obs, rewards, terminated, truncated, _ = env.step(actions)
        done = terminated | truncated
        total_episodes_done += np.sum(done)
        
        # Update observations (vectorized env handles resets automatically)
        obs = next_obs
    
    print(f"Completed {total_episodes_done} episodes out of {actual_total_steps} total steps across {NUM_ENVS} environments")
    
    end_time = time.time()
    total_time = end_time - start_time
    env.close()
    
    return total_time, actual_total_steps / total_time

def main():
    num_steps = 10_000_000
    
    # Benchmark both environments
    gymnax_time, gymnax_sps = benchmark_gymnax(num_steps)
    gymnasium_time, gymnasium_sps = benchmark_gymnasium(num_steps)
    
    # Display results
    print("\n" + "="*50)
    print("THROUGHPUT COMPARISON RESULTS")
    print("="*50)
    print(f"Steps: {num_steps:,}")
    print()
    print(f"Gymnax Pendulum-v1:")
    print(f"  Time: {gymnax_time:.2f} seconds")
    print(f"  Throughput: {gymnax_sps:.2f} steps/sec")
    print()
    print(f"Gymnasium Pendulum-v1:")
    print(f"  Time: {gymnasium_time:.2f} seconds")
    print(f"  Throughput: {gymnasium_sps:.2f} steps/sec")
    print()
    print(f"Speedup: {gymnax_sps / gymnasium_sps:.2f}x")
    print("="*50)

if __name__ == "__main__":
    main()