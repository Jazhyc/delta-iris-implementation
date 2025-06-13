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

import ivy
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

device_jax = jax.devices()[0]
device_torch = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_ENVS = 2048

class SimplePolicy(nn.Module):
    """Simple neural network that returns a fixed action regardless of input"""
    def __init__(self, obs_dim=3, action_dim=1):
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
    
    key = jax.random.key(0)
    
    # Calculate steps per environment to achieve total steps
    steps_per_env = num_steps // NUM_ENVS
    actual_total_steps = steps_per_env * NUM_ENVS
    print(f"Running {steps_per_env} steps per environment across {NUM_ENVS} environments")
    print(f"Total steps: {actual_total_steps} (requested: {num_steps})")
    
    SimplePolicyJax = ivy.transpile(SimplePolicy, source="torch", target="jax")
    policy_jax = SimplePolicyJax(obs_dim=3, action_dim=1)
    
    # Instantiate the environment & its settings
    env, env_params = gymnax.make("Pendulum-v1")
    
    def rollout(key_input, env_params, steps_in_episode):
        """Rollout a jitted gymnax episode with lax.scan."""
        # Reset the environment
        key_reset, key_episode = jax.random.split(key_input)
        obs, state = env.reset(key_reset, env_params)
        
        
        def policy_step(state_input, tmp):
            """Step transition in jax env."""
            obs, state, key = state_input
            key, key_step = jax.random.split(key, 2)
            
            with torch.no_grad():
                action = policy_jax(obs)
            next_obs, next_state, reward, done, _ = env.step(
                key_step, state, action, env_params
            )
            carry = [next_obs, next_state, key]
            return carry, [obs, action, reward, next_obs, done]

        # Scan over episode step loop
        _, scan_out = jax.lax.scan(
            policy_step, [obs, state, key_episode], (), steps_in_episode
        )
        return scan_out

    # Vectorize the rollout function across multiple environments
    vmap_rollout = jax.vmap(rollout, in_axes=(0, None, None), out_axes=0)
    
    # Jit-compile the vectorized rollout
    jit_rollout = jax.jit(vmap_rollout, static_argnums=2)
    
    # Generate keys for parallel environments
    keys = jax.random.split(key, NUM_ENVS)
    
    start_time = time.time()
    
    # Run parallel rollouts for steps_per_env steps each
    obs, action, reward, next_obs, done = jit_rollout(keys, env_params, steps_per_env)
    
    # Get number of done episodes across all environments
    num_done = jnp.sum(done)
    print(f"Completed {num_done} episodes out of {actual_total_steps} total steps across {NUM_ENVS} environments")
    
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
    num_steps = 100_000_000
    
    # Benchmark both environments
    gymnax_time, gymnax_sps = benchmark_gymnax(num_steps)
    # gymnasium_time, gymnasium_sps = benchmark_gymnasium(num_steps)
    
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
    # print(f"Gymnasium Pendulum-v1:")
    # print(f"  Time: {gymnasium_time:.2f} seconds")
    # print(f"  Throughput: {gymnasium_sps:.2f} steps/sec")
    # print()
    # print(f"Speedup: {gymnax_sps / gymnasium_sps:.2f}x")
    # print("="*50)

if __name__ == "__main__":
    main()
