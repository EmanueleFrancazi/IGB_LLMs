"""Base interfaces shared by all language-model implementations.

The project will eventually contain multiple model families. This file defines
small, explicit interfaces that each model should follow so that training,
inference, and evaluation code can be reused across architectures.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn


@dataclass
class ModelOutput:
    """Standard output returned by language models in this project.

    Attributes:
        logits: Raw next-token prediction scores with shape
            ``[batch_size, sequence_length, vocab_size]``.
        loss: Optional scalar loss, usually cross-entropy when training targets
            are provided.
        extra: Optional dictionary for model-specific diagnostics such as
            hidden states, attention statistics, or cache information.
    """

    logits: torch.Tensor
    loss: torch.Tensor | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class BaseLanguageModel(nn.Module, ABC):
    """Minimal interface expected from every language model.

    Subclasses are regular ``torch.nn.Module`` objects. They should expose a
    ``forward`` method accepting token IDs and, optionally, next-token targets.
    This keeps the interface compatible with standard language-model training
    while still allowing model-specific internals to remain explicit.
    """

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> ModelOutput:
        """Run a forward pass through the model.

        Args:
            input_ids: Integer token IDs with shape
                ``[batch_size, sequence_length]``.
            targets: Optional integer target token IDs with the same shape as
                ``input_ids``. When provided, the model should compute a scalar
                language-modeling loss.
            **kwargs: Optional model-specific arguments.

        Returns:
            A ``ModelOutput`` containing logits and, when applicable, loss.
        """
        raise NotImplementedError

    @property
    def device(self) -> torch.device:
        """Return the device where the model parameters live.

        This helper is useful for scripts that need to move input tensors to the
        same device as the model without manually inspecting parameters.
        """

        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def count_parameters(self, only_trainable: bool = True) -> int:
        """Count model parameters.

        Args:
            only_trainable: If true, count only parameters with
                ``requires_grad=True``. If false, count every parameter.

        Returns:
            Total number of selected parameters.
        """

        parameters = self.parameters()
        if only_trainable:
            parameters = (p for p in parameters if p.requires_grad)
        return sum(p.numel() for p in parameters)
