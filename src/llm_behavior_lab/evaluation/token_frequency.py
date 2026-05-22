"""Token-frequency utilities for baseline output analysis.

These helpers compare model-predicted probability mass with empirical token
frequencies from the current text corpus. The functions are intentionally simple
and operate on token IDs from the Phase 3 tokenizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from llm_behavior_lab.data.tokenizer import CharTokenizer


@dataclass(frozen=True)
class TokenFrequencySummary:
    """Count and frequency for one token."""

    token_id: int
    token: str
    count: int
    frequency: float


@dataclass(frozen=True)
class TokenProbabilityGap:
    """Difference between model probability and empirical frequency."""

    token_id: int
    token: str
    predicted_probability: float
    empirical_frequency: float
    gap: float


def empirical_token_counts(token_ids: Sequence[int], *, vocab_size: int) -> torch.Tensor:
    """Count token occurrences in a sequence of token IDs."""

    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive.")
    if len(token_ids) == 0:
        raise ValueError("token_ids must be non-empty.")

    token_tensor = torch.tensor(list(token_ids), dtype=torch.long)
    if torch.any(token_tensor < 0) or torch.any(token_tensor >= vocab_size):
        raise ValueError("token_ids contain values outside the vocabulary range.")

    return torch.bincount(token_tensor, minlength=vocab_size)


def empirical_token_frequencies(token_ids: Sequence[int], *, vocab_size: int) -> torch.Tensor:
    """Compute normalized empirical token frequencies."""

    counts = empirical_token_counts(token_ids, vocab_size=vocab_size).float()
    total = counts.sum()
    if total <= 0:
        raise ValueError("Cannot compute frequencies from zero token count.")
    return counts / total


def average_predicted_probabilities(probabilities: torch.Tensor) -> torch.Tensor:
    """Average predicted probabilities over batch and sequence positions."""

    if probabilities.ndim != 3:
        raise ValueError(
            f"probabilities must have shape [batch, sequence, vocab], got {tuple(probabilities.shape)}."
        )
    return probabilities.mean(dim=(0, 1))


def top_token_frequencies(
    counts: torch.Tensor,
    tokenizer: CharTokenizer,
    *,
    k: int = 10,
) -> list[TokenFrequencySummary]:
    """Return the most frequent empirical tokens."""

    if counts.ndim != 1:
        raise ValueError(f"counts must be one-dimensional, got {tuple(counts.shape)}.")
    if counts.shape[0] > tokenizer.vocab_size:
        raise ValueError("counts length cannot exceed tokenizer vocabulary size.")
    if k <= 0:
        raise ValueError("k must be positive.")

    total = counts.sum().item()
    if total <= 0:
        raise ValueError("counts must sum to a positive value.")

    k = min(k, counts.shape[0])
    values, indices = torch.topk(counts, k=k)
    summaries: list[TokenFrequencySummary] = []
    for count_tensor, token_id_tensor in zip(values, indices, strict=True):
        token_id = int(token_id_tensor.item())
        count = int(count_tensor.item())
        summaries.append(
            TokenFrequencySummary(
                token_id=token_id,
                token=tokenizer.decode([token_id]),
                count=count,
                frequency=float(count / total),
            )
        )
    return summaries


def top_probability_gaps(
    predicted_probabilities: torch.Tensor,
    empirical_frequencies: torch.Tensor,
    tokenizer: CharTokenizer,
    *,
    k: int = 10,
    largest: bool = True,
) -> list[TokenProbabilityGap]:
    """Return largest positive or negative probability-frequency gaps."""

    if predicted_probabilities.ndim != 1 or empirical_frequencies.ndim != 1:
        raise ValueError("predicted_probabilities and empirical_frequencies must be one-dimensional.")
    if predicted_probabilities.shape != empirical_frequencies.shape:
        raise ValueError(
            "predicted_probabilities and empirical_frequencies must have matching shapes, "
            f"got {tuple(predicted_probabilities.shape)} and {tuple(empirical_frequencies.shape)}."
        )
    if predicted_probabilities.shape[0] > tokenizer.vocab_size:
        raise ValueError("probability vectors cannot exceed tokenizer vocabulary size.")
    if k <= 0:
        raise ValueError("k must be positive.")

    gaps = predicted_probabilities - empirical_frequencies
    k = min(k, gaps.shape[0])
    values, indices = torch.topk(gaps, k=k, largest=largest)

    summaries: list[TokenProbabilityGap] = []
    for gap_tensor, token_id_tensor in zip(values, indices, strict=True):
        token_id = int(token_id_tensor.item())
        summaries.append(
            TokenProbabilityGap(
                token_id=token_id,
                token=tokenizer.decode([token_id]),
                predicted_probability=float(predicted_probabilities[token_id].item()),
                empirical_frequency=float(empirical_frequencies[token_id].item()),
                gap=float(gap_tensor.item()),
            )
        )
    return summaries


def kl_divergence(p: torch.Tensor, q: torch.Tensor, *, eps: float = 1e-12) -> float:
    """Compute KL(p || q) for two discrete distributions."""

    if p.shape != q.shape:
        raise ValueError(f"p and q must have the same shape, got {tuple(p.shape)} and {tuple(q.shape)}.")
    p_safe = p.float().clamp_min(eps)
    q_safe = q.float().clamp_min(eps)
    return float((p_safe * (p_safe.log() - q_safe.log())).sum().item())


def js_divergence(p: torch.Tensor, q: torch.Tensor, *, eps: float = 1e-12) -> float:
    """Compute Jensen-Shannon divergence between two distributions."""

    if p.shape != q.shape:
        raise ValueError(f"p and q must have the same shape, got {tuple(p.shape)} and {tuple(q.shape)}.")
    p_safe = p.float().clamp_min(eps)
    q_safe = q.float().clamp_min(eps)
    midpoint = 0.5 * (p_safe + q_safe)
    return 0.5 * kl_divergence(p_safe, midpoint, eps=eps) + 0.5 * kl_divergence(q_safe, midpoint, eps=eps)
