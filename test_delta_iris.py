"""
Test script for Delta-IRIS implementation
"""

import torch
import sys
import os
sys.path.append('/home/jazhyc/projects/delta-iris-implementation/src')

from agent.config import TrainerConfig, TokenizerConfig, WorldModelConfig, ActorCriticConfig, BufferConfig
from agent.buffer import ExperienceBuffer, Episode
from agent.tokenizer import Tokenizer
from agent.world_model import WorldModel
from agent.actor_critic import ActorCritic

def test_components():
    """Test individual components"""
    print("Testing Delta-IRIS components...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    
    # Mock environment dimensions
    obs_dim = 3  # Pendulum has 3D observations
    action_dim = 1  # Continuous action, discretized
    
    print(f"Using device: {device}, dtype: {dtype}")
    print(f"Obs dim: {obs_dim}, Action dim: {action_dim}")
    
    # Test Tokenizer
    print("\n1. Testing Tokenizer...")
    tokenizer_config = TokenizerConfig(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=64,
        latent_dim=32,
        num_tokens=4,
        codebook_size=128,
        learning_rate=1e-4
    )
    
    tokenizer = Tokenizer(tokenizer_config).to(device, dtype=dtype)
    
    # Test forward pass
    batch_size, seq_len = 4, 8
    obs = torch.randn(batch_size, seq_len, obs_dim, device=device, dtype=dtype)
    actions = torch.randn(batch_size, seq_len, action_dim, device=device, dtype=dtype)
    
    tokenizer_output = tokenizer(obs, actions)
    print(f"   Tokenizer output keys: {list(tokenizer_output.keys())}")
    print(f"   Tokens shape: {tokenizer_output['tokens'].shape}")
    print(f"   Reconstruction loss: {tokenizer_output['losses']['reconstruction_loss']:.6f}")
    
    # Test World Model
    print("\n2. Testing World Model...")
    world_model_config = WorldModelConfig(
        vocab_size=128,
        action_dim=action_dim,
        latent_dim=32,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        sequence_length=32,
        learning_rate=1e-4
    )
    
    world_model = WorldModel(world_model_config).to(device, dtype=dtype)
    
    # Test forward pass
    tokens = tokenizer_output['tokens'][:, :-1]  # Remove last token
    actions_discrete = torch.randint(0, action_dim, (batch_size, tokens.shape[1]), device=device)  # Match token sequence length
    
    wm_predictions = world_model(tokens, actions_discrete)
    print(f"   World model prediction keys: {list(wm_predictions.keys())}")
    print(f"   Next token logits shape: {wm_predictions['next_token_logits'].shape}")
    
    # Test Actor-Critic
    print("\n3. Testing Actor-Critic...")
    ac_config = ActorCriticConfig(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=64,
        imagination_horizon=5,
        gamma=0.99,
        lambda_gae=0.95,
        entropy_coef=0.01,
        learning_rate=1e-4
    )
    
    actor_critic = ActorCritic(ac_config).to(device, dtype=dtype)
    
    # Test forward pass
    test_obs = torch.randn(batch_size, obs_dim, device=device, dtype=dtype)
    action, log_prob, value = actor_critic.get_action_and_value(test_obs)
    print(f"   Action shape: {action.shape}")
    print(f"   Value shape: {value.shape}")
    
    # Test Buffer
    print("\n4. Testing Experience Buffer...")
    buffer = ExperienceBuffer(
        capacity=1000,
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
        dtype=dtype
    )
    
    # Create mock episode
    episode_length = 50
    episode = Episode(
        observations=torch.randn(episode_length, obs_dim, device=device, dtype=dtype),
        actions=torch.randn(episode_length, action_dim, device=device, dtype=dtype),
        rewards=torch.randn(episode_length, device=device, dtype=dtype),
        dones=torch.zeros(episode_length, device=device, dtype=torch.bool)
    )
    episode.dones[-1] = True  # Mark episode end
    
    buffer.add_episode(episode)
    print(f"   Buffer size after adding episode: {buffer.current_size}")
    
    # Test sampling
    try:
        batch = buffer.sample_sequences(batch_size=2, sequence_length=10)
        print(f"   Sampled batch keys: {list(batch.keys())}")
        print(f"   Sampled observations shape: {batch['observations'].shape}")
    except ValueError as e:
        print(f"   Sampling failed (expected for small buffer): {e}")
    
    print("\nAll component tests completed successfully!")
    
    # Test integration
    print("\n5. Testing Component Integration...")
    
    # Add more episodes to buffer
    for _ in range(5):
        episode = Episode(
            observations=torch.randn(episode_length, obs_dim, device=device, dtype=dtype),
            actions=torch.randn(episode_length, action_dim, device=device, dtype=dtype),
            rewards=torch.randn(episode_length, device=device, dtype=dtype),
            dones=torch.zeros(episode_length, device=device, dtype=torch.bool)
        )
        episode.dones[-1] = True
        buffer.add_episode(episode)
    
    print(f"   Buffer size after adding more episodes: {buffer.current_size}")
    
    # Test sampling and training step
    batch = buffer.sample_sequences(batch_size=4, sequence_length=16)
    
    # Test tokenizer training step
    obs_batch = batch['observations'][:, :-1]
    actions_batch = batch['actions'][:, :-1]
    tokenizer_out = tokenizer(obs_batch, actions_batch)
    tokenizer_loss = tokenizer_out['losses']['total_tokenizer_loss']
    
    print(f"   Tokenizer loss: {tokenizer_loss:.6f}")
    
    # Test world model training step
    with torch.no_grad():
        tokens_batch = tokenizer.get_tokens(obs_batch, actions_batch)
    
    targets = {
        'next_tokens': tokens_batch[:, 1:],
        'rewards': batch['rewards'][:, 1:],
        'dones': batch['dones'][:, 1:].float()
    }
    
    # Make sure actions have the same sequence length as tokens for world model input
    actions_for_wm = batch['actions'][:, :tokens_batch.shape[1]-1, 0].long()  # Match sequence length
    wm_pred = world_model(tokens_batch[:, :-1], actions_for_wm)
    wm_losses = world_model.compute_loss(wm_pred, targets)
    wm_loss = wm_losses['total_world_model_loss']
    
    print(f"   World model loss: {wm_loss:.6f}")
    
    print("\nIntegration test completed successfully!")
    return True

if __name__ == "__main__":
    try:
        test_components()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
