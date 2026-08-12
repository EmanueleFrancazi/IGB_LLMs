"""Measure token-guess distributions at random model initialization.

The experiment this module supports asks how a randomly initialized model's
*selected token guesses* compare with the corpus token frequencies, and how much
that comparison moves when only the initialization changes.

Three sources of randomness are kept apart:

* **corpus/position sampling** -- removed entirely. Evaluation positions are
  chosen deterministically by :func:`deterministic_window_starts` and reused
  unchanged for every initialization, so no observed difference between seeds
  can come from looking at different text.
* **model initialization** -- the quantity under study. The caller seeds and
  builds one model per initialization.
* **token sampling** -- only affects the stochastic policy, is driven by its own
  generator, and is averaged over replicates *within* one initialization before
  initializations are compared.

One forward pass per initialization produces the logits that both policies then
consume, so greedy and nucleus results describe the same model state rather than
two independent draws.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from llm_behavior_lab.evaluation.guessing import (
    greedy_guess_ids,
    guess_counts,
    nucleus_guess_ids,
    restrict_to_support,
)

__all__ = [
    "EvaluationPositions",
    "InitializationMeasurement",
    "NucleusSamplingSettings",
    "build_evaluation_positions",
    "compute_evaluation_logits",
    "deterministic_window_starts",
    "measure_initialization",
]


@dataclass(frozen=True)
class NucleusSamplingSettings:
    """Parameters of the stochastic guessing policy.

    Defaults follow the reference LLaMA inference settings. They are carried as
    data rather than hard-coded inside the analysis so every persisted result
    records the policy that produced it.
    """

    temperature: float = 0.6
    top_p: float = 0.9
    seed: int = 20240601
    num_replicates: int = 8

    def validate(self) -> None:
        if self.temperature <= 0:
            raise ValueError("nucleus temperature must be positive.")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("nucleus top_p must lie in (0, 1].")
        if self.seed < 0:
            raise ValueError("nucleus seed must be non-negative.")
        if self.num_replicates <= 0:
            raise ValueError("nucleus num_replicates must be positive.")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable description for persisted metadata."""

        return {
            "method": "temperature_top_p",
            "temperature": self.temperature,
            "top_p": self.top_p,
            "sampling_seed": self.seed,
            "num_replicates": self.num_replicates,
        }


@dataclass(frozen=True)
class EvaluationPositions:
    """The fixed set of prediction positions shared by every initialization."""

    starts: tuple[int, ...]
    block_size: int
    input_ids: torch.Tensor
    target_ids: torch.Tensor

    @property
    def num_windows(self) -> int:
        """Number of evaluation windows."""

        return len(self.starts)

    @property
    def num_positions(self) -> int:
        """Number of next-token prediction positions being analyzed."""

        return self.num_windows * self.block_size

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable identity of the selected positions."""

        return {
            "strategy": "evenly_spaced_deterministic",
            "block_size": self.block_size,
            "num_windows": self.num_windows,
            "num_positions": self.num_positions,
            "first_start": self.starts[0],
            "last_start": self.starts[-1],
        }


@dataclass(frozen=True)
class InitializationMeasurement:
    """Complete per-token record for one model initialization.

    Every vector has length ``vocab_size`` and is indexed by token ID, so all
    records in an experiment align without any further bookkeeping.
    """

    model_seed: int
    greedy_counts: torch.Tensor
    nucleus_counts: torch.Tensor
    mean_predicted_probabilities: torch.Tensor
    num_positions: int

    @property
    def nucleus_counts_mean(self) -> torch.Tensor:
        """Replicate-averaged stochastic guess counts for this initialization."""

        return self.nucleus_counts.float().mean(dim=0)


def deterministic_window_starts(
    num_tokens: int,
    *,
    block_size: int,
    num_windows: int,
) -> list[int]:
    """Choose evaluation-window start offsets deterministically.

    Windows are spread as evenly as possible across the split rather than drawn
    at random. Two properties matter more here than randomness: the same
    positions must be reproducible for every initialization and every rerun, and
    they should cover the corpus uniformly so the selected-position token
    distribution stays close to the full-split one. No seed is involved, so the
    selection cannot drift when the model seed changes.

    Args:
        num_tokens: Length of the token split being sampled.
        block_size: Window length in tokens.
        num_windows: Number of windows to select.

    Returns:
        Sorted, distinct start offsets.

    Raises:
        ValueError: If the arguments are non-positive, or the split cannot
            supply that many distinct windows.
    """

    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    if num_windows <= 0:
        raise ValueError("num_windows must be positive.")

    max_start = num_tokens - block_size - 1
    if max_start < 0:
        raise ValueError(
            f"The split has {num_tokens} tokens, which cannot supply a window of "
            f"block_size + 1 = {block_size + 1} tokens."
        )
    available = max_start + 1
    if num_windows > available:
        raise ValueError(
            f"Requested {num_windows} windows but the split only supports {available} "
            "distinct start offsets. Reduce num_windows, reduce block_size, or use a "
            "larger corpus."
        )

    if num_windows == 1:
        return [0]
    step = max_start / (num_windows - 1)
    return [int(round(index * step)) for index in range(num_windows)]


def build_evaluation_positions(
    token_ids: Sequence[int],
    *,
    block_size: int,
    num_windows: int,
    device: torch.device | str = "cpu",
) -> EvaluationPositions:
    """Materialize the fixed evaluation windows and their next-token targets.

    Args:
        token_ids: Token IDs of the split being analyzed.
        block_size: Window length in tokens.
        num_windows: Number of deterministic windows.
        device: Device for the returned tensors.

    Returns:
        The shared :class:`EvaluationPositions` for the whole experiment.
    """

    starts = deterministic_window_starts(
        len(token_ids),
        block_size=block_size,
        num_windows=num_windows,
    )
    data = torch.tensor(list(token_ids), dtype=torch.long)
    inputs = torch.stack([data[start : start + block_size] for start in starts])
    targets = torch.stack([data[start + 1 : start + block_size + 1] for start in starts])
    return EvaluationPositions(
        starts=tuple(starts),
        block_size=block_size,
        input_ids=inputs.to(torch.device(device)),
        target_ids=targets.to(torch.device(device)),
    )


def compute_evaluation_logits(
    model: torch.nn.Module,
    positions: EvaluationPositions,
    *,
    vocab_size: int,
    forward_batch_size: int = 32,
) -> torch.Tensor:
    """Run the model once over the fixed positions and return restricted logits.

    The forward pass is chunked so a large evaluation-position count stays
    memory-bounded; chunking changes nothing about the result because the model
    is run without a cache and windows are independent.

    Args:
        model: Model whose forward accepts ``input_ids`` and returns ``.logits``.
        positions: The shared evaluation positions.
        vocab_size: Valid token support, normally the tokenizer vocabulary.
        forward_batch_size: Number of windows per forward pass.

    Returns:
        Logits with shape ``[num_windows, block_size, vocab_size]``.
    """

    if forward_batch_size <= 0:
        raise ValueError("forward_batch_size must be positive.")

    model.eval()
    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, positions.num_windows, forward_batch_size):
            window = positions.input_ids[start : start + forward_batch_size]
            output = model(input_ids=window)
            chunks.append(restrict_to_support(output.logits, vocab_size=vocab_size))
    return torch.cat(chunks, dim=0)


def measure_initialization(
    *,
    model_seed: int,
    logits: torch.Tensor,
    vocab_size: int,
    sampling: NucleusSamplingSettings,
    device: torch.device | str = "cpu",
) -> InitializationMeasurement:
    """Derive both guessing policies from one initialization's logits.

    The stochastic policy is replicated several times from the *same* logits.
    That is cheap next to another forward pass and it separates sampling noise
    from initialization noise: replicates are averaged here, so the caller
    compares initializations, not individual draws.

    The sampling generator is seeded from the policy seed combined with the
    model seed, so every initialization gets its own reproducible sampling
    stream and no two initializations share one.

    Args:
        model_seed: Seed that produced this initialization; recorded and mixed
            into the sampling stream.
        logits: Restricted logits from :func:`compute_evaluation_logits`.
        vocab_size: Valid token support.
        sampling: Stochastic policy parameters.
        device: Device used for the sampling generator.

    Returns:
        One :class:`InitializationMeasurement` with complete per-token vectors.
    """

    sampling.validate()
    if logits.shape[-1] != vocab_size:
        raise ValueError(
            f"logits last dimension {logits.shape[-1]} does not match vocab_size {vocab_size}. "
            "Restrict the logits to the valid support before measuring."
        )

    # Every entry except the vocabulary axis is one prediction position, so this
    # stays correct whether the caller passes [windows, block, vocab] or a
    # already-flattened [positions, vocab].
    num_positions = int(logits[..., 0].numel())

    greedy_counts = guess_counts(greedy_guess_ids(logits), vocab_size=vocab_size)

    generator = torch.Generator(device=torch.device(device))
    generator.manual_seed(sampling.seed + model_seed)
    replicate_counts = [
        guess_counts(
            nucleus_guess_ids(
                logits,
                temperature=sampling.temperature,
                top_p=sampling.top_p,
                generator=generator,
            ),
            vocab_size=vocab_size,
        )
        for _ in range(sampling.num_replicates)
    ]

    probabilities = torch.softmax(logits.float(), dim=-1)
    mean_predicted = probabilities.reshape(-1, vocab_size).mean(dim=0)

    return InitializationMeasurement(
        model_seed=model_seed,
        greedy_counts=greedy_counts,
        nucleus_counts=torch.stack(replicate_counts),
        mean_predicted_probabilities=mean_predicted,
        num_positions=num_positions,
    )
