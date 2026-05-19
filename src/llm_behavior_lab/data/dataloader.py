"""Batch construction for causal language modeling.

The project does not add a full training loop yet. This module provides only the
batching primitive needed to verify that tokenized text can be converted into
``input_ids`` and shifted ``targets`` compatible with the LLaMA-style model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch

SplitName = Literal["train", "val"]


@dataclass(frozen=True)
class CausalLMBatch:
    """A batch of causal language-modeling examples."""

    input_ids: torch.Tensor
    targets: torch.Tensor
    split: SplitName


class CausalLMBatcher:
    """Randomly sample fixed-length causal LM batches from token IDs.

    For each sampled window of length ``block_size + 1``, the first
    ``block_size`` tokens become ``input_ids`` and the next ``block_size``
    tokens become ``targets``. This creates the standard next-token prediction
    objective without implementing a training loop yet.
    """

    def __init__(
        self,
        *,
        train_ids: Sequence[int],
        val_ids: Sequence[int],
        block_size: int,
        batch_size: int,
        device: torch.device | str = "cpu",
        seed: int = 1234,
    ) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        self.block_size = block_size
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)

        self.train_data = self._to_tensor(train_ids, split="train")
        self.val_data = self._to_tensor(val_ids, split="val")

    def _to_tensor(self, token_ids: Sequence[int], *, split: SplitName) -> torch.Tensor:
        """Convert a token sequence to a CPU tensor and validate length."""

        if len(token_ids) < self.block_size + 1:
            raise ValueError(
                f"The {split} split has {len(token_ids)} tokens, but causal LM batching "
                f"requires at least block_size + 1 = {self.block_size + 1}."
            )
        return torch.tensor(token_ids, dtype=torch.long)

    def get_batch(self, split: SplitName) -> CausalLMBatch:
        """Sample one random batch from the requested split."""

        data = self.train_data if split == "train" else self.val_data
        max_start = len(data) - self.block_size - 1
        starts = torch.randint(
            low=0,
            high=max_start + 1,
            size=(self.batch_size,),
            generator=self.generator,
        )

        input_ids = torch.stack([data[start : start + self.block_size] for start in starts])
        targets = torch.stack([data[start + 1 : start + self.block_size + 1] for start in starts])

        return CausalLMBatch(
            input_ids=input_ids.to(self.device),
            targets=targets.to(self.device),
            split=split,
        )
