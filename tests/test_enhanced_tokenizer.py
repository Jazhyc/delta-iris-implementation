#!/usr/bin/env python3
"""
Test script for the enhanced Delta-IRIS tokenizer
"""

import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tokenizer import DeltaIrisTokenizer, DeltaTokenizerConfig
from src.tokenizer.utils import create_spatial_grid, extract_spatial_tokens
from src.agent.config import TokenizerConfig
from src.agent.tokenizer import Tokenizer


def test_enhanced_tokenizer():
    """Test the enhanced Delta-IRIS tokenizer"""
    print("Testing Enhanced Delta-IRIS Tokenizer...")
    
    # Create enhanced config
    config = DeltaTokenizerConfig(
        obs_dim=4,  # CartPole observation dimension
        action_dim=2,  # CartPole action dimension
        hidden_dim=256,
        latent_dim=64,
        codebook_size=512,
        spatial_grid_size=4,
        context_length=4,
        delta_encoding=True,
        use_spatial_attention=True
    )
    
    # Create tokenizer
    tokenizer = DeltaIrisTokenizer(config)
    
    # Create test data
    batch_size = 8
    seq_len = 10
    obs = torch.randn(batch_size, seq_len, config.obs_dim)
    actions = torch.randint(0, 2, (batch_size, seq_len, config.action_dim)).float()  # Use config.action_dim
    
    print(f"Input shapes: obs={obs.shape}, actions={actions.shape}")
    
    # Test forward pass
    with torch.no_grad():
        outputs = tokenizer(obs, actions)
    
    print("Forward pass outputs:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        elif isinstance(value, dict):
            print(f"  {key}: {list(value.keys())}")
    
    # Test spatial token extraction
    spatial_tokens = tokenizer.get_spatial_tokens(obs, actions)
    print(f"Spatial tokens shape: {spatial_tokens.shape}")
    
    # Test reconstruction
    reconstructed = tokenizer.decode_spatial_tokens(spatial_tokens, actions)
    print(f"Reconstructed shape: {reconstructed.shape}")
    
    print("✓ Enhanced tokenizer test passed!")
    return True


def test_legacy_compatibility():
    """Test legacy tokenizer compatibility"""
    print("\nTesting Legacy Tokenizer Compatibility...")
    
    # Create basic config (should use legacy)
    config = TokenizerConfig(
        obs_dim=4,
        action_dim=2,
        hidden_dim=256,
        latent_dim=64,
        codebook_size=512
    )
    
    # Create tokenizer
    tokenizer = Tokenizer(config)
    
    # Create test data
    batch_size = 8
    seq_len = 10
    obs = torch.randn(batch_size, seq_len, config.obs_dim)
    actions = torch.randint(0, 2, (batch_size, seq_len, config.action_dim)).float()  # Use config.action_dim
    
    print(f"Input shapes: obs={obs.shape}, actions={actions.shape}")
    
    # Test forward pass
    with torch.no_grad():
        outputs = tokenizer(obs, actions)
    
    print("Forward pass outputs:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        elif isinstance(value, dict):
            print(f"  {key}: {list(value.keys())}")
    
    print("✓ Legacy compatibility test passed!")
    return True


def test_spatial_utilities():
    """Test spatial tokenizer utilities"""
    print("\nTesting Spatial Utilities...")
    
    # Test spatial grid creation
    obs = torch.randn(4, 8, 4)  # Vector observations
    grid = create_spatial_grid(obs, grid_size=3)
    print(f"Spatial grid shape: {grid.shape}")
    
    # Test token extraction methods
    spatial_repr = torch.randn(4, 8, 9, 64)  # [batch, seq, patches, latent]
    
    # Test different extraction methods
    pooled_tokens = extract_spatial_tokens(spatial_repr, method='pool')
    print(f"Pooled tokens shape: {pooled_tokens.shape}")
    
    flattened_tokens = extract_spatial_tokens(spatial_repr, method='flatten')
    print(f"Flattened tokens shape: {flattened_tokens.shape}")
    
    attention_tokens = extract_spatial_tokens(spatial_repr, method='attention')
    print(f"Attention tokens shape: {attention_tokens.shape}")
    
    print("✓ Spatial utilities test passed!")
    return True


def test_context_encoding():
    """Test context-aware encoding"""
    print("\nTesting Context-Aware Encoding...")
    
    from src.tokenizer.context_tokenizer import ContextAwareTokenizer
    
    # Create context tokenizer
    context_tokenizer = ContextAwareTokenizer(
        input_dim=64,
        hidden_dim=256,
        context_length=4,
        use_delta_encoding=True,
        use_memory=True
    )
    
    # Test data
    batch_size = 4
    seq_len = 8
    input_dim = 64
    inputs = torch.randn(batch_size, seq_len, input_dim)
    
    print(f"Input shape: {inputs.shape}")
    
    # Test forward pass
    with torch.no_grad():
        outputs, memory, hidden = context_tokenizer(inputs)
    
    print(f"Context outputs shape: {outputs.shape}")
    print(f"Memory shape: {memory.shape if memory is not None else None}")
    print(f"Hidden shapes: {[h.shape for h in hidden] if hidden is not None else None}")
    
    print("✓ Context encoding test passed!")
    return True


def benchmark_performance():
    """Benchmark tokenizer performance"""
    print("\nBenchmarking Performance...")
    
    config = DeltaTokenizerConfig(
        obs_dim=4,
        action_dim=2,
        hidden_dim=256,
        latent_dim=64,
        codebook_size=512,
        spatial_grid_size=4,
        context_length=4
    )
    
    tokenizer = DeltaIrisTokenizer(config)
    
    # Test data
    batch_size = 16
    seq_len = 32
    obs = torch.randn(batch_size, seq_len, config.obs_dim)
    actions = torch.randint(0, 2, (batch_size, seq_len, config.action_dim)).float()  # Use config.action_dim
    
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = tokenizer(obs, actions)
    
    # Benchmark
    import time
    num_runs = 50
    
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            outputs = tokenizer(obs, actions)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    print(f"Average forward pass time: {avg_time*1000:.2f} ms")
    print(f"Throughput: {batch_size * seq_len / avg_time:.0f} samples/second")
    
    print("✓ Performance benchmark completed!")
    return True


def main():
    """Run all tests"""
    print("Running Enhanced Tokenizer Tests...")
    print("=" * 50)
    
    tests = [
        test_enhanced_tokenizer,
        test_legacy_compatibility,
        test_spatial_utilities,
        test_context_encoding,
        benchmark_performance
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print(f"Tests completed: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    exit(main())
