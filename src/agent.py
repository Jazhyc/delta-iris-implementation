"""Base class for a RL agent"""

from abc import ABC, abstractmethod
import numpy as np

class Agent(ABC):
    def __init__(
        self, # ...
    ) -> None:
        pass
    
    @abstractmethod
    def choose_action(
        self, observation: np.ndarray, # ...
    ) -> int: # Is this always an int?
        pass

    @abstractmethod
    def learn(self) -> None:
        pass
