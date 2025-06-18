import math
from typing import Generator, List

import torch

from .batch import Batch
from .episode import Episode
from .segment import Segment, SegmentId


def collate_segments_to_batch(segments: List[Segment]) -> Batch:
    """Collate a list of segments into a batch"""
    return Batch(
        torch.stack([s.observations for s in segments]),
        torch.stack([s.actions for s in segments]),
        torch.stack([s.rewards for s in segments]),
        torch.stack([s.ends for s in segments]),
        torch.stack([s.mask_padding for s in segments]),
        [s.id for s in segments]
    )


def make_segment(episode: Episode, segment_id: SegmentId, should_pad: bool = True) -> Segment:
    """Extract a segment from an episode, with optional padding"""
    assert segment_id.start < len(episode) and segment_id.stop > 0 and segment_id.start < segment_id.stop
    
    padding_length_right = max(0, segment_id.stop - len(episode))
    padding_length_left = max(0, -segment_id.start)
    assert padding_length_right == padding_length_left == 0 or should_pad

    def pad(x):
        """Pad tensor on left and right as needed"""
        if padding_length_right > 0:
            # Pad on the right (after the sequence)
            pad_dims = [0 for _ in range(2 * x.ndim - 1)] + [padding_length_right]
            x = torch.nn.functional.pad(x, pad_dims)
        if padding_length_left > 0:
            # Pad on the left (before the sequence)
            pad_dims = [0 for _ in range(2 * x.ndim - 2)] + [padding_length_left, 0]
            x = torch.nn.functional.pad(x, pad_dims)
        return x

    start = max(0, segment_id.start)
    stop = min(len(episode), segment_id.stop)

    return Segment(
        pad(episode.observations[start:stop]),
        pad(episode.actions[start:stop]),
        pad(episode.rewards[start:stop]),
        pad(episode.ends[start:stop]),
        mask_padding=torch.cat((
            torch.zeros(padding_length_left, dtype=torch.bool),
            torch.ones(stop - start, dtype=torch.bool),
            torch.zeros(padding_length_right, dtype=torch.bool)
        )),
        id=SegmentId(segment_id.episode_id, start, stop)
    )


class DatasetTraverser:
    """Iterate through a dataset in chunks for evaluation"""
    
    def __init__(self, dataset, batch_num_samples: int, chunk_size: int) -> None:
        self.dataset = dataset
        self.batch_num_samples = batch_num_samples
        self.chunk_size = chunk_size
        # Calculate number of batches we'll generate
        self._num_batches = math.ceil(
            sum([
                math.ceil(dataset.lengths[episode_id] / chunk_size) - 
                int(dataset.lengths[episode_id] % chunk_size == 1)  # Skip single-step chunks
                for episode_id in range(dataset.num_episodes)
            ]) / batch_num_samples
        )

    def __len__(self) -> int:
        return self._num_batches 

    def __iter__(self) -> Generator[Batch, None, None]:
        chunks = []

        for episode_id in range(self.dataset.num_episodes):
            episode = self.dataset.load_episode(episode_id)
            
            # Create chunks from this episode
            for i in range(math.ceil(len(episode) / self.chunk_size)):
                chunk = make_segment(
                    episode, 
                    SegmentId(episode_id, start=i * self.chunk_size, stop=(i + 1) * self.chunk_size), 
                    should_pad=True
                )
                chunks.append(chunk)
                
            # Remove chunks that are too small (single timestep)
            if chunks and chunks[-1].effective_size < 2:
                chunks.pop()

            # Yield batches when we have enough chunks
            while len(chunks) >= self.batch_num_samples:
                yield collate_segments_to_batch(chunks[:self.batch_num_samples])
                chunks = chunks[self.batch_num_samples:]

        # Yield remaining chunks if any
        if len(chunks) > 0:
            yield collate_segments_to_batch(chunks)
