from __future__ import annotations
from dataclasses import dataclass
from typing import List

import torch

from .segment import SegmentId


@dataclass
class Batch:
    observations: torch.Tensor
    actions: torch.LongTensor
    rewards: torch.FloatTensor
    ends: torch.LongTensor
    mask_padding: torch.BoolTensor
    segment_ids: List[SegmentId]

    def pin_memory(self) -> Batch:
        def pin_if_cpu(tensor):
            # Only pin memory for CPU tensors, skip GPU tensors
            if tensor.device.type == 'cpu':
                return tensor.pin_memory()
            else:
                return tensor
        
        return Batch(**{k: v if k == 'segment_ids' else pin_if_cpu(v) for k, v in self.__dict__.items()})

    def to(self, device: torch.device) -> Batch:
        return Batch(**{k: v if k == 'segment_ids' else v.to(device) for k, v in self.__dict__.items()})
