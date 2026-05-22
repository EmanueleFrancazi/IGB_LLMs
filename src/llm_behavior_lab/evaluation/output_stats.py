"""Output-distribution statistics for language-model logits.

Phase 5 starts with transparent, lightweight statistics that describe the
randomly initialized model's output distribution. These helpers operate on
already-computed logits or probabilities and do not perform training,
checkpointing, or logging.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class OutputDistributionSummary:
    """Aggregate statistics for model output probabilities."""

    logits_shape: tuple[int, ...]
    probabilities_shape: tuple[int, ...]
    num_positions: int
    mean_entropy: float
    min_entropy: float
    max_entropy: float
    mean_top1_probability: float
    mean_topk_probability_mass: float


def logits_to_probabilities(
    logits: torch.Tensor,
    *,
    vocab_size_limit: int | None = None,
) -> torch.Tensor:
    """Convert logits to probabilities, optionally restricting the vocabulary.

    Args:
        logits: Raw model logits with shape ``[batch, sequence, vocab]``.
        vocab_size_limit: Optional number of vocabulary entries to keep before
            applying softmax. This is useful when the model has a larger output
            vocabulary than the active tokenizer can decode.

    Returns:
        Probability tensor with shape ``[batch, sequence, kept_vocab]``.
    """

    if logits.ndim != 3:
        raise ValueError(f"logits must have shape [batch, sequence, vocab], got {tuple(logits.shape)}.")

    if vocab_size_limit is not None:
        if vocab_size_limit <= 0:
            raise ValueError("vocab_size_limit must be positive when provided.")
        logits = logits[..., :vocab_size_limit]

    return torch.softmax(logits.float(), dim=-1)


def entropy_from_probabilities(probabilities: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """Compute entropy at every batch/sequence position.

    Args:
        probabilities: Tensor with shape ``[batch, sequence, vocab]``.
        eps: Small value used to avoid ``log(0)``.

    Returns:
        Entropy tensor with shape ``[batch, sequence]``.
    """

    if probabilities.ndim != 3:
        raise ValueError(
            f"probabilities must have shape [batch, sequence, vocab], got {tuple(probabilities.shape)}."
        )

    safe_probs = probabilities.clamp_min(eps)
    return -(safe_probs * safe_probs.log()).sum(dim=-1)


def topk_probability_mass(probabilities: torch.Tensor, *, k: int) -> torch.Tensor:
    """Return the probability mass assigned to the top-k tokens per position."""

    if probabilities.ndim != 3:
        raise ValueError(
            f"probabilities must have shape [batch, sequence, vocab], got {tuple(probabilities.shape)}."
        )
    if k <= 0:
        raise ValueError("k must be positive.")

    k = min(k, probabilities.shape[-1])
    values, _ = torch.topk(probabilities, k=k, dim=-1)
    return values.sum(dim=-1)


def top1_token_ids(probabilities: torch.Tensor) -> torch.Tensor:
    """Return the highest-probability token ID at each batch/sequence position."""

    if probabilities.ndim != 3:
        raise ValueError(
            f"probabilities must have shape [batch, sequence, vocab], got {tuple(probabilities.shape)}."
        )
    return torch.argmax(probabilities, dim=-1)


def top1_probability_values(probabilities: torch.Tensor) -> torch.Tensor:
    """Return the highest token probability at each batch/sequence position."""

    if probabilities.ndim != 3:
        raise ValueError(
            f"probabilities must have shape [batch, sequence, vocab], got {tuple(probabilities.shape)}."
        )
    return torch.max(probabilities, dim=-1).values


def summarize_output_distribution(
    logits: torch.Tensor,
    probabilities: torch.Tensor,
    *,
    top_k: int = 5,
) -> OutputDistributionSummary:
    """Summarize model output probabilities over all analyzed positions."""

    if logits.ndim != 3:
        raise ValueError(f"logits must have shape [batch, sequence, vocab], got {tuple(logits.shape)}.")
    if probabilities.ndim != 3:
        raise ValueError(
            f"probabilities must have shape [batch, sequence, vocab], got {tuple(probabilities.shape)}."
        )
    if logits.shape[:2] != probabilities.shape[:2]:
        raise ValueError(
            "logits and probabilities must agree on batch and sequence dimensions, "
            f"got logits={tuple(logits.shape)} and probabilities={tuple(probabilities.shape)}."
        )

    entropy = entropy_from_probabilities(probabilities)
    top1_probs = top1_probability_values(probabilities)
    topk_mass = topk_probability_mass(probabilities, k=top_k)
    num_positions = int(probabilities.shape[0] * probabilities.shape[1])

    return OutputDistributionSummary(
        logits_shape=tuple(logits.shape),
        probabilities_shape=tuple(probabilities.shape),
        num_positions=num_positions,
        mean_entropy=float(entropy.mean().item()),
        min_entropy=float(entropy.min().item()),
        max_entropy=float(entropy.max().item()),
        mean_top1_probability=float(top1_probs.mean().item()),
        mean_topk_probability_mass=float(topk_mass.mean().item()),
    )
