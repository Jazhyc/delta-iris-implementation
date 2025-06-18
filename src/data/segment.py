from __future__ import annotations
from dataclasses import dataclass

import torch


@dataclass
class SegmentId:
    episode_id: int
    start: int
    stop: int


@dataclass
class Segment:
    observations: torch.Tensor
    actions: torch.LongTensor
    rewards: torch.FloatTensor
    ends: torch.LongTensor
    mask_padding: torch.BoolTensor  # True for real data, False for padding
    id: SegmentId

    @property
    def effective_size(self) -> int:
        """Number of non-padded timesteps in this segment"""
        return self.mask_padding.sum().item()
