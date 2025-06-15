"""PyTorch neural network used in RL architectures"""

import torch
import torch.nn as nn
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(
        self, # ...
    ):
        super(NeuralNetwork, self).__init__()


    def forward(
        self, state: np.ndarray, # ...
    ) -> torch.Tensor:
        pass
