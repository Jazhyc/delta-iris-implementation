"""
Debug script to understand action values and ranges
"""
import torch
import sys
sys.path.append('src')

from agent.buffer import ExperienceBuffer, Episode

# Mock environment dimensions  
obs_dim = 3  # Pendulum has 3D observations
action_dim = 1  # Continuous action, discretized

device = "cpu"  # Use CPU to avoid CUDA issues
dtype = torch.float32

# Create buffer
buffer = ExperienceBuffer(
    capacity=1000,
    obs_dim=obs_dim,
    action_dim=action_dim,
    device=device,
    dtype=dtype
)

# Create mock episode with realistic Pendulum action values
episode_length = 50
episode = Episode(
    observations=torch.randn(episode_length, obs_dim, device=device, dtype=dtype),
    actions=torch.randn(episode_length, action_dim, device=device, dtype=dtype) * 2,  # Pendulum actions are in [-2, 2] roughly
    rewards=torch.randn(episode_length, device=device, dtype=dtype),
    dones=torch.zeros(episode_length, device=device, dtype=torch.bool)
)
episode.dones[-1] = True  # Mark episode end

buffer.add_episode(episode)

# Sample a batch 
batch = buffer.sample_sequences(batch_size=4, sequence_length=16)

print("Original action values (continuous):")
print(f"  Shape: {batch['actions'].shape}")
print(f"  Min: {batch['actions'].min():.3f}, Max: {batch['actions'].max():.3f}")
print(f"  Sample values: {batch['actions'][0, :5, 0]}")

# Convert to discrete actions like in the test
actions_discrete = batch['actions'][:, :, 0].long()
print(f"\nDiscrete actions (converted with .long()):")
print(f"  Shape: {actions_discrete.shape}")
print(f"  Min: {actions_discrete.min()}, Max: {actions_discrete.max()}")
print(f"  Sample values: {actions_discrete[0, :5]}")

print(f"\nWorld model expects action indices in range [0, {action_dim})")
print(f"But we have indices ranging from {actions_discrete.min()} to {actions_discrete.max()}")
print("This causes the embedding index out of bounds error!")
