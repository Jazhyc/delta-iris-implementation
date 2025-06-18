#!/usr/bin/env python3
"""
Test script for the new data infrastructure
"""
import torch
import tempfile
import shutil
from pathlib import Path

# Add src to path
import sys
sys.path.append('src')

from data import Episode, EpisodeDataset, BatchSampler, EpisodeCountManager, collate_segments_to_batch

def test_data_infrastructure():
    print("Testing Delta-IRIS data infrastructure...")
    
    # Create temporary directory for test
    temp_dir = Path(tempfile.mkdtemp())
    try:
        # Test Episode creation
        print("1. Testing Episode creation...")
        episode = Episode(
            observations=torch.randn(10, 4),  # CartPole has 4D observation space
            actions=torch.randint(0, 2, (10,)),  # CartPole has 2 actions
            rewards=torch.randn(10),
            ends=torch.zeros(10, dtype=torch.long)
        )
        episode.ends[-1] = 1  # Mark end of episode
        print(f"   Created episode with length: {len(episode)}")
        
        # Test EpisodeDataset
        print("2. Testing EpisodeDataset...")
        dataset = EpisodeDataset(temp_dir / "test_dataset", "test")
        
        # Add episodes
        episode_ids = []
        for i in range(5):
            ep = Episode(
                observations=torch.randn(10, 4),
                actions=torch.randint(0, 2, (10,)),
                rewards=torch.randn(10),
                ends=torch.zeros(10, dtype=torch.long)
            )
            ep.ends[-1] = 1
            episode_id = dataset.add_episode(ep)
            episode_ids.append(episode_id)
        
        print(f"   Added {len(episode_ids)} episodes")
        print(f"   Dataset has {dataset.num_episodes} episodes, {dataset.num_steps} steps")
        
        # Test episode loading
        print("3. Testing episode loading...")
        loaded_episode = dataset.load_episode(0)
        print(f"   Loaded episode 0 with length: {len(loaded_episode)}")
        
        # Test EpisodeCountManager
        print("4. Testing EpisodeCountManager...")
        count_manager = EpisodeCountManager(dataset)
        count_manager.register('tokenizer', 'world_model', 'actor_critic')
        
        # Add episode counts
        for episode_id in episode_ids:
            count_manager.add_episode(episode_id)
        
        # Simulate some sampling
        for _ in range(10):
            count_manager.increment_episode_count('tokenizer', 0)
            count_manager.increment_episode_count('world_model', 1)
        
        probs = count_manager.compute_probabilities('tokenizer', alpha=1.0)
        print(f"   Computed probabilities: {probs}")
        
        # Test BatchSampler
        print("5. Testing BatchSampler...")
        batch_sampler = BatchSampler(
            dataset=dataset,
            num_steps_per_epoch=3,
            batch_size=2,
            sequence_length=5,
            can_sample_beyond_end=True
        )
        batch_sampler.probabilities = probs
        
        # Sample some batches
        batches = list(batch_sampler)
        print(f"   Sampled {len(batches)} batches")
        
        # Test collate function
        print("6. Testing batch collation...")
        from torch.utils.data import DataLoader
        
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=0,
            collate_fn=collate_segments_to_batch
        )
        
        for i, batch in enumerate(loader):
            print(f"   Batch {i}: obs shape {batch.observations.shape}, actions shape {batch.actions.shape}")
            if i >= 2:  # Just test a few batches
                break
        
        # Test save/load
        print("7. Testing save/load...")
        dataset.save_info()
        count_manager.save(temp_dir / "counts.pt")
        
        # Create new instances and load
        dataset2 = EpisodeDataset(temp_dir / "test_dataset", "test2")
        count_manager2 = EpisodeCountManager(dataset2)
        count_manager2.load(temp_dir / "counts.pt")
        
        print(f"   Loaded dataset: {dataset2.num_episodes} episodes")
        print(f"   Loaded counts: {len(count_manager2.all_counts)} components")
        
        print("✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    test_data_infrastructure()
