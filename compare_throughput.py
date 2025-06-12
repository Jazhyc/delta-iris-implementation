import jax
import gymnax
import gymnasium as gym
import time
import numpy as np
from tqdm import tqdm

def benchmark_gymnax(num_steps=100_000):
    """Benchmark Gymnax Pendulum environment"""
    print("Benchmarking Gymnax Pendulum-v1...")
    
    key = jax.random.key(0)
    key, key_reset, key_step = jax.random.split(key, 3)
    
    # Instantiate the environment & its settings
    env, env_params = gymnax.make("Pendulum-v1")
    
    # Reset the environment
    obs, state = env.reset(key_reset, env_params)
    
    start_time = time.time()
    
    for i in tqdm(range(num_steps), desc="Gymnax steps"):
        # Sample a random action
        key_act, key_step = jax.random.split(key_step, 2)
        action = env.action_space(env_params).sample(key_act)
        
        # Perform the step transition
        n_obs, n_state, reward, done, _ = env.step(key_step, state, action, env_params)
        
        # Reset environment if episode is done
        if done:
            obs, state = env.reset(key_step, env_params)
        else:
            obs, state = n_obs, n_state
    
    end_time = time.time()
    total_time = end_time - start_time
    
    return total_time, num_steps / total_time

def benchmark_gymnasium(num_steps=100_000):
    """Benchmark Gymnasium Pendulum environment"""
    print("\nBenchmarking Gymnasium Pendulum-v1...")
    
    env = gym.make("Pendulum-v1")
    obs, _ = env.reset(seed=0)
    
    start_time = time.time()
    
    for i in tqdm(range(num_steps), desc="Gymnasium steps"):
        # Sample a random action
        action = env.action_space.sample()
        
        # Perform the step transition
        n_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Reset environment if episode is done
        if done:
            obs, _ = env.reset()
        else:
            obs = n_obs
    
    end_time = time.time()
    total_time = end_time - start_time
    env.close()
    
    return total_time, num_steps / total_time

def main():
    num_steps = 10000
    
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
