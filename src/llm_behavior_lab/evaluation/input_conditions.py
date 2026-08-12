"""Paired input-structure controls for the initialization experiment.

Three inputs are fed to **one** initialized model, so any difference between the
resulting guess distributions comes from the input and not from the weights:

* **real** -- the fixed corpus evaluation windows;
* **shuffled** -- exactly the same multiset of token IDs with the ordering
  destroyed;
* **gaussian** -- synthetic vectors supplied directly at the embedding boundary.

What each contrast can identify
-------------------------------

*real vs shuffled* approximately isolates **sequential ordering**. The shuffle
preserves the evaluated token IDs, their counts, the marginal token
distribution, the tokenizer, the embedding table, the window shapes, and the
positional layout. It destroys local ordering and sequential correlation, and
nothing else.

*shuffled vs gaussian* measures what is additionally associated with **discrete
token identity**: repeated lookup vectors and the unigram frequency structure,
once ordering is already gone.

*real vs gaussian* is the broadest contrast, structured token input against
unstructured continuous input. It removes far more than temporal correlation, so
**it must never be described on its own as a causal test of input correlation.**

Determinism
-----------

The permutation and the standardized Gaussian bank are drawn from their own
dedicated seeds and are **fixed across model initializations**. Redrawing them
per seed would add a second source of variation to a comparison whose whole
purpose is to vary only the weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

__all__ = [
    "INPUT_CONDITIONS",
    "EmbeddingMoments",
    "embedding_moments",
    "scaled_gaussian_embeddings",
    "shuffled_input_ids",
    "standardized_gaussian_bank",
]

#: Canonical order, used for records, figures, and console output alike.
INPUT_CONDITIONS = ("real", "shuffled", "gaussian")


def shuffled_input_ids(input_ids: torch.Tensor, *, seed: int) -> torch.Tensor:
    """Permute every evaluated token position with one fixed permutation.

    The whole ``[windows, block]`` tensor is flattened, permuted once, and
    reshaped. Permuting globally rather than within each window destroys
    ordering at every scale, including any residual structure a within-window
    shuffle would leave between windows.

    Exactly preserved: the number of input positions, the token IDs, their
    counts, the marginal token distribution of the evaluated inputs, and the
    window/block shape. Destroyed: local ordering and sequential correlation.

    Args:
        input_ids: Real evaluation windows, ``[windows, block]``.
        seed: Dedicated shuffle seed. Fixed across model initializations.

    Returns:
        A new tensor; the input is never modified in place.
    """

    if input_ids.ndim != 2:
        raise ValueError(
            f"input_ids must have shape [windows, block], got {tuple(input_ids.shape)}."
        )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    flat = input_ids.reshape(-1)
    permutation = torch.randperm(flat.numel(), generator=generator).to(flat.device)
    return flat[permutation].reshape(input_ids.shape).contiguous()


def standardized_gaussian_bank(
    *,
    num_windows: int,
    block_size: int,
    dim: int,
    seed: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Draw the fixed ``N(0, 1)`` bank shared by every model initialization.

    Standardized rather than pre-scaled: the scale that makes the condition
    comparable depends on each initialization's own embedding table, so the
    randomness is drawn once and rescaled per initialization.

    The bank is materialized in full. It is small next to the logits it feeds --
    128 windows of 64 positions at dimension 128 is about 4 MB -- and keeping the
    *same* draws across conditions and initializations matters far more than
    saving that.
    """

    if num_windows <= 0 or block_size <= 0 or dim <= 0:
        raise ValueError("num_windows, block_size, and dim must all be positive.")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    bank = torch.randn(num_windows, block_size, dim, generator=generator)
    return bank.to(torch.device(device))


@dataclass(frozen=True)
class EmbeddingMoments:
    """What the Gaussian condition was scaled to, and how it was measured.

    Four fields, all of them provenance: the scale itself, plus enough to say
    what it was taken over. A plain frozen dataclass, like every other small
    value type in this package.

    Attributes:
        mean: Realized entry-wise mean of the measured rows.
        std: Realized entry-wise standard deviation of the measured rows.
        num_rows: How many embedding rows were measured.
        rule: Human-readable description of which rows and which statistic.
    """

    mean: float
    std: float
    num_rows: int
    rule: str

    def as_dict(self) -> dict[str, object]:
        """Provenance persisted with the run."""

        return {
            "rule": self.rule,
            "mean": self.mean,
            "std": self.std,
            "num_rows": self.num_rows,
        }


def embedding_moments(
    embedding_weight: torch.Tensor,
    *,
    eligible_token_ids: Sequence[int] | None = None,
) -> EmbeddingMoments:
    """Measure the realized first two moments of an initialized embedding table.

    Scale matching uses the *realized* moments of this initialization rather than
    the nominal initializer parameters, so the Gaussian condition matches the
    weights the model actually has.

    Moments are taken over **all entries of the eligible token rows**, treating
    the table as one pool of scalars. That is the quantity that determines the
    typical magnitude of a vector entering the first block, which is what the
    Gaussian condition has to match. Structural-token rows are excluded when the
    eligible support is supplied: they are never looked up by real input, so
    including them would let tokens the experiment does not use influence the
    control.

    Args:
        embedding_weight: The ``[vocab, dim]`` embedding matrix.
        eligible_token_ids: Rows to measure. ``None`` uses every row.

    Returns:
        An :class:`EmbeddingMoments` carrying its own provenance.
    """

    if embedding_weight.ndim != 2:
        raise ValueError(
            f"embedding_weight must have shape [vocab, dim], got {tuple(embedding_weight.shape)}."
        )

    if eligible_token_ids is None:
        rows = embedding_weight
        rule = "all embedding rows, entry-wise mean and std"
    else:
        index = torch.as_tensor(list(eligible_token_ids), dtype=torch.long)
        if index.numel() == 0:
            raise ValueError("eligible_token_ids must not be empty.")
        rows = embedding_weight.index_select(0, index.to(embedding_weight.device))
        rule = "eligible embedding rows only, entry-wise mean and std"

    values = rows.detach().float().reshape(-1)
    # A single entry has no unbiased variance and would give nan. Zero is the
    # honest answer there and matches the constant-table case below: a table with
    # no spread scales the control to a constant, it does not invent one.
    std = float(values.std(unbiased=True)) if values.numel() > 1 else 0.0
    return EmbeddingMoments(
        mean=float(values.mean()),
        std=std,
        num_rows=int(rows.shape[0]),
        rule=rule,
    )


def scaled_gaussian_embeddings(
    standardized: torch.Tensor,
    moments: EmbeddingMoments,
) -> torch.Tensor:
    """Affine-scale the shared bank to one initialization's embedding scale.

    ``G = mu + sigma * Z``. The randomness is identical across initializations;
    only the scale follows the weights, so the Gaussian condition stays a
    like-for-like control rather than an arbitrary magnitude.

    A constant embedding table gives ``sigma = 0`` and therefore a constant
    control at ``mu``. That is correct and must not be special-cased into some
    arbitrary variance: nothing here divides by ``sigma``.
    """

    return moments.mean + moments.std * standardized
