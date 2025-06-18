"""
Debug Pendulum environment observation space
"""
import sys
sys.path.append('src')
from env import make_env

env, _ = make_env("Pendulum-v1", num_envs=1)
print(f"Environment: Pendulum-v1")
print(f"Observation space: {env.observation_space}")
print(f"Observation space shape: {env.observation_space.shape}")
print(f"Action space: {env.action_space}")
if hasattr(env.action_space, 'shape'):
    print(f"Action space shape: {env.action_space.shape}")

# Test a step
obs, _ = env.reset()
print(f"Initial observation shape: {obs.shape}")
print(f"Initial observation: {obs}")
