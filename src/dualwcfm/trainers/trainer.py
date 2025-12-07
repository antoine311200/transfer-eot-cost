from abc import ABC, abstractmethod

class Trainer(ABC):
    @abstractmethod
    def step(self, batch_size: int, training: bool) -> float:
        """Perform a single training/evaluation step

        Args:
            batch_size: Size of the batch to process
            training: Whether to perform a training step (with backpropagation) or just evaluation

        Returns:
            Loss value as a float
        """
        pass

    @abstractmethod
    def evaluate(self, batch_size: int = 1000) -> float:
        """Evaluate the model on a validation set

        Args:
            batch_size: Size of the batch to process during evaluation

        Returns:
            Loss value as a float
        """
        pass

    @abstractmethod
    def train(self, epochs: int, batch_size: int, learning_rate: float, print_every: int = 100, eval_every: int = 10) -> dict:
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