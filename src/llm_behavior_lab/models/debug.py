"""Tiny debug language model used to validate repository plumbing.

This model is intentionally simple: token embeddings, learned position
embeddings, and a linear language-modeling head. It is not intended as the
research baseline. Its purpose is to make Phase 1 executable before the real
LLaMA-style model is integrated in Phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from llm_behavior_lab.models.base import BaseLanguageModel, ModelOutput
from llm_behavior_lab.models.registry import register_model


@dataclass(frozen=True)
class DebugLMConfig:
    """Configuration for ``DebugLanguageModel``.

    Attributes:
        vocab_size: Number of tokens in the toy vocabulary.
        block_size: Maximum supported sequence length.
        embedding_dim: Token and position embedding width.
    """

    vocab_size: int = 128
    block_size: int = 32
    embedding_dim: int = 64

    def validate(self) -> None:
        """Validate configuration values early and explicitly."""

        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive.")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")


class DebugLanguageModel(BaseLanguageModel):
    """A minimal next-token model for smoke tests.

    The forward pass mirrors the shape contract expected from future real
    models: input IDs enter with shape ``[B, T]`` and logits leave with shape
    ``[B, T, vocab_size]``.
    """

    def __init__(
        self,
        vocab_size: int = 128,
        block_size: int = 32,
        embedding_dim: int = 64,
    ) -> None:
        super().__init__()
        self.config = DebugLMConfig(
            vocab_size=vocab_size,
            block_size=block_size,
            embedding_dim=embedding_dim,
        )
        self.config.validate()

        self.token_embedding = nn.Embedding(self.config.vocab_size, self.config.embedding_dim)
        self.position_embedding = nn.Embedding(self.config.block_size, self.config.embedding_dim)
        self.lm_head = nn.Linear(self.config.embedding_dim, self.config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        **_: object,
    ) -> ModelOutput:
        """Compute logits and optional language-modeling loss."""

        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must have shape [batch, sequence], got {tuple(input_ids.shape)}"
            )

        batch_size, sequence_length = input_ids.shape
        if sequence_length > self.config.block_size:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds block_size {self.config.block_size}."
            )

        positions = torch.arange(sequence_length, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, sequence_length)

        token_features = self.token_embedding(input_ids)
        position_features = self.position_embedding(positions)
        hidden_states = token_features + position_features
        logits = self.lm_head(hidden_states)

        loss = None
        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError(
                    "targets must have the same shape as input_ids, "
                    f"got targets={tuple(targets.shape)} and input_ids={tuple(input_ids.shape)}"
                )
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                targets.reshape(-1),
            )

        return ModelOutput(logits=logits, loss=loss)


register_model("debug_tiny_lm", DebugLanguageModel)
