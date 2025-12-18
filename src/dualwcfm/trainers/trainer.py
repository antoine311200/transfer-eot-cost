from abc import ABC, abstractmethod
from typing import Any, Union
from dataclasses import dataclass

@dataclass
class TrainerConfig:
    learning_rate: float = 1e-3
    epochs: int = 1000
    batch_size: int = 64
    eval_batch_size: int = 32

class Trainer(ABC):
    @abstractmethod
    def step(self, training: bool, **kwargs) -> Union[float, tuple]:
        """Perform a single training/evaluation step

        Args:
            training: Whether to perform a training step (with backpropagation) or just evaluation

        Returns:
            Loss value as a float
        """
        pass

    @abstractmethod
    def evaluate(self, **kwargs) -> Any:
        """Evaluate the model on a validation set

        Returns:
            Loss value as a float
        """
        pass

    @abstractmethod
    def train(self, config: TrainerConfig, plot_func = None, plot_every: int = 0, print_every: int = 100, eval_every: int = 10) -> dict:
        """Train the model for a specified number of epochs

        Args:
            epochs: Number of epochs to train
            batch_size: Size of each training batch
            learning_rate: Learning rate for the optimizer
            print_every: Frequency (in steps) to print training progress
            eval_every: Frequency (in epochs) to evaluate the model

        Returns:
            Dictionary containing training and evaluation losses
        """
        pass