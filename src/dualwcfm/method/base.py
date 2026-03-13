from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from dualwcfm.trainer import Trainer

logger = logging.getLogger("dualwcfm")


@dataclass
class MethodConfig:
    """Base config for training methods."""
    pass

class Method(ABC):
    """Abstract training method."""

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def pretrain(self, trainer: Trainer, retrain: bool = True, save: str = None) -> None:
        """Run the pretrain phase: marginal flows + forward/backward flow models."""
        ...

    @abstractmethod
    def step_preprocess(self, trainer: Trainer, iteration: int) -> dict:
        """Build the context dict for this iteration.

        Returns a dict with at least: t, t_tensor, backward, is_image.
        """
        ...

    @abstractmethod
    def step_duals(self, trainer: Trainer, iteration: int, ctx: dict) -> None:
        """Train the dual potentials f and g."""
        ...

    @abstractmethod
    def step_flow_matching(self, trainer: Trainer, iteration: int, ctx: dict) -> None:
        """Train the conditional flow matching model."""
        ...

    @abstractmethod
    def step_postprocess(self, trainer: Trainer, iteration: int, ctx: dict) -> None:
        """Advance the coupling dataset to the next iterate."""
        ...

    @property
    def use_noisy_field(self) -> bool:
        """Whether to add noise in the vector field target during marginal training."""
        return False