from typing import Generator, List

import numpy as np
import torch

from .dataset import EpisodeDataset
from .segment import SegmentId


class BatchSampler(torch.utils.data.Sampler):
    """Sampler for creating batches of segments from episodes"""
    
    def __init__(
        self, 
        dataset: EpisodeDataset, 
        num_steps_per_epoch: int, 
        batch_size: int, 
        sequence_length: int, 
        can_sample_beyond_end: bool
    ) -> None:
        # Don't pass dataset to super() to avoid the deprecated data_source warning
        super().__init__(data_source=None)
        self.dataset = dataset
        self.probabilities = None  # Will be set by episode count manager
        self.num_steps_per_epoch = num_steps_per_epoch
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.can_sample_beyond_end = can_sample_beyond_end

    def __len__(self) -> int:
        return self.num_steps_per_epoch

    def __iter__(self) -> Generator[List[SegmentId], None, None]:
        for _ in range(self.num_steps_per_epoch):
            yield self.sample()

    def sample(self) -> List[SegmentId]:
        """Sample a batch of segment IDs"""
        # Sample episodes according to probabilities (uniform if None)
        episode_ids = np.random.choice(
            np.arange(self.dataset.num_episodes), 
            size=self.batch_size, 
            replace=True, 
            p=self.probabilities
        )
        
        # Sample random timesteps within each episode
        timesteps = np.random.randint(low=0, high=self.dataset.lengths[episode_ids])

        if self.can_sample_beyond_end:  
            # Padding allowed both before start and after end
            starts = timesteps - np.random.randint(0, self.sequence_length, len(timesteps))
            stops = starts + self.sequence_length
        else:                           
            # Padding allowed only before start
            stops = np.minimum(
                self.dataset.lengths[episode_ids], 
                timesteps + 1 + np.random.randint(0, self.sequence_length, len(timesteps))
            )
            starts = stops - self.sequence_length

        return [SegmentId(ep_id, start, stop) for ep_id, start, stop in zip(episode_ids, starts, stops)]
