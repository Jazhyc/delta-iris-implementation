from pathlib import Path
from typing import Dict

import numpy as np
import torch

from .dataset import EpisodeDataset


class EpisodeCountManager:
    """Manages sampling counts for episodes to implement priority sampling"""
    
    def __init__(self, dataset: EpisodeDataset) -> None:
        self.dataset = dataset
        self.all_counts: Dict[str, np.ndarray] = {}

    def load(self, path_to_checkpoint: Path) -> None:
        """Load episode counts from checkpoint"""
        self.all_counts = torch.load(path_to_checkpoint, map_location='cpu', weights_only=False)
        # Verify all count arrays match dataset size
        assert all([
            counts.shape[0] == self.dataset.num_episodes 
            for counts in self.all_counts.values()
        ])

    def save(self, path_to_checkpoint: Path) -> None:
        """Save episode counts to checkpoint"""
        torch.save(self.all_counts, path_to_checkpoint)

    def register(self, *keys: str) -> None:
        """Register new component keys for tracking"""
        assert all([key not in self.all_counts for key in keys])
        self.all_counts.update({
            key: np.zeros(self.dataset.num_episodes, dtype=np.int64) 
            for key in keys
        }) 

    def add_episode(self, episode_id: int) -> None:
        """Called when a new episode is added to extend count arrays"""
        for key, counts in self.all_counts.items():
            assert episode_id <= counts.shape[0]
            if episode_id == counts.shape[0]:
                # New episode, extend the count array
                self.all_counts[key] = np.concatenate((counts, np.zeros(1, dtype=np.int64)))
            assert self.all_counts[key].shape[0] == self.dataset.num_episodes

    def increment_episode_count(self, key: str, episode_id: int) -> None:
        """Increment the sample count for an episode"""
        assert key in self.all_counts
        self.all_counts[key][episode_id] += 1

    def compute_probabilities(self, key: str, alpha: float) -> np.ndarray:
        """Compute sampling probabilities based on inverse counts"""
        assert key in self.all_counts
        inverse_counts = 1 / (1 + self.all_counts[key])
        p = inverse_counts ** alpha
        return p / p.sum()
