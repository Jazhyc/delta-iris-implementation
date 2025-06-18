import hashlib
from pathlib import Path
import shutil
from typing import Dict, Optional, Union

import numpy as np
import torch

from .episode import Episode
from .segment import Segment, SegmentId
from .utils import make_segment


class EpisodeDataset(torch.utils.data.Dataset):
    """
    Dataset for storing and loading episodes from disk.
    Optimized for performance - episodes are only loaded when needed.
    """
    
    def __init__(self, directory: Path, name: str) -> None:
        super().__init__()
        self.name = name
        self.directory = Path(directory)
        self.num_episodes, self.num_steps, self.start_idx, self.lengths = None, None, None, None
        
        # Cache for recently loaded episodes (LRU-style)
        self._episode_cache = {}
        self._cache_size = 50  # Keep last 50 episodes in memory
        self._cache_order = []

        if not self.directory.is_dir():
            self._init_empty()
        else:
            self._load_info()
            print(f'({name}) {self.num_episodes} episodes, {self.num_steps} steps.')

    @property
    def info_path(self) -> Path:
        return self.directory / 'info.pt'

    @property
    def info(self) -> Dict[str, Union[int, np.ndarray]]:
        return {
            'num_episodes': self.num_episodes, 
            'num_steps': self.num_steps, 
            'start_idx': self.start_idx, 
            'lengths': self.lengths
        }

    def __len__(self) -> int:
        return self.num_steps

    def __getitem__(self, segment_id: SegmentId) -> Segment:
        return self._load_segment(segment_id)

    def _init_empty(self) -> None:
        """Initialize an empty dataset"""
        self.directory.mkdir(parents=True, exist_ok=False)
        self.num_episodes = 0
        self.num_steps = 0
        self.start_idx = np.array([], dtype=np.int64)            
        self.lengths = np.array([], dtype=np.int64)
        self.save_info()

    def _load_info(self) -> None:
        """Load dataset metadata from disk"""
        info = torch.load(self.info_path, weights_only=False)
        self.num_steps = info['num_steps']
        self.num_episodes = info['num_episodes']
        self.start_idx = info['start_idx']
        self.lengths = info['lengths']

    def save_info(self) -> None:
        """Save dataset metadata to disk"""
        torch.save(self.info, self.info_path)

    def clear(self) -> None:
        """Clear all data and start fresh"""
        if self.directory.exists():
            shutil.rmtree(self.directory)
        self._episode_cache = {}
        self._cache_order = []
        self._init_empty()

    def _get_episode_path(self, episode_id: int) -> Path:
        """Get file path for an episode with hierarchical directory structure for performance"""
        n = 3  # number of directory levels
        powers = np.arange(n)
        subfolders = list(map(int, np.floor((episode_id % 10 ** (1 + powers)) / 10 ** powers) * 10 ** powers))[::-1]
        return self.directory / '/'.join([f'{x[1]:0{n - x[0]}d}' for x in enumerate(subfolders)]) / f'{episode_id}.pt'

    def _load_segment(self, segment_id: SegmentId, should_pad: bool = True) -> Segment:
        """Load a segment from an episode"""
        episode = self.load_episode(segment_id.episode_id)
        return make_segment(episode, segment_id, should_pad)

    def load_episode(self, episode_id: int) -> Episode:
        """Load an episode from cache or disk"""
        # Check cache first
        if episode_id in self._episode_cache:
            # Move to end of cache order (LRU)
            self._cache_order.remove(episode_id)
            self._cache_order.append(episode_id)
            return self._episode_cache[episode_id]
        
        # Load from disk
        episode_data = torch.load(self._get_episode_path(episode_id), weights_only=False)
        episode = Episode(**episode_data)
        
        # Add to cache
        self._add_to_cache(episode_id, episode)
        
        return episode

    def _add_to_cache(self, episode_id: int, episode: Episode) -> None:
        """Add episode to cache with LRU eviction"""
        # Remove oldest episodes if cache is full
        while len(self._episode_cache) >= self._cache_size:
            oldest_id = self._cache_order.pop(0)
            del self._episode_cache[oldest_id]
        
        self._episode_cache[episode_id] = episode
        self._cache_order.append(episode_id)

    def add_episode(self, episode: Episode, *, episode_id: Optional[int] = None) -> int:
        """Add a new episode or extend an existing one"""
        if episode_id is None:
            # New episode
            episode_id = self.num_episodes
            self.start_idx = np.concatenate((self.start_idx, np.array([self.num_steps])))
            self.lengths = np.concatenate((self.lengths, np.array([len(episode)])))
            self.num_steps += len(episode)
            self.num_episodes += 1
        else:
            # Extend existing episode
            assert episode_id < self.num_episodes
            old_episode = self.load_episode(episode_id) 
            episode = old_episode.merge(episode)
            incr_num_steps = len(episode) - len(old_episode)
            self.lengths[episode_id] = len(episode)
            self.start_idx[episode_id + 1:] += incr_num_steps            
            self.num_steps += incr_num_steps

        # Save episode to disk
        episode_path = self._get_episode_path(episode_id)
        episode_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Atomic write using temporary file
        temp_path = episode_path.with_suffix('.tmp')
        torch.save(episode.__dict__, temp_path)
        temp_path.rename(episode_path)
        
        # Update cache
        self._add_to_cache(episode_id, episode)

        return episode_id

    def get_episode_id_from_global_idx(self, global_idx: np.ndarray) -> np.ndarray:
        """Convert global step indices to episode IDs"""
        return (np.argmax(self.start_idx.reshape(-1, 1) > global_idx, axis=0) - 1) % self.num_episodes

    def get_global_idx_from_segment_id(self, segment_id: SegmentId) -> np.ndarray:
        """Convert segment ID to global step indices"""
        start_idx = self.start_idx[segment_id.episode_id]
        return np.arange(start_idx + segment_id.start, start_idx + segment_id.stop)
