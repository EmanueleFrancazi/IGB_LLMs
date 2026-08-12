"""Token-selection policies applied to model logits.

A "guess" is the token a policy actually selects at one prediction position.
This is deliberately distinct from the model's predicted probability
distribution: averaging probabilities over positions answers "what mass does
the model place on each token", whereas counting selected guesses answers "how
often does this token actually get chosen". The two differ sharply at
initialization, and the project keeps them under separate names everywhere.

Two policies are implemented:

* :func:`greedy_guess_ids` -- deterministic ``argmax``.
* :func:`nucleus_guess_ids` -- temperature scaling followed by top-p (nucleus)
  truncation, mirroring the reference LLaMA inference behavior.

Both operate on logits already restricted to the valid token support, so a
model output dimension wider than the tokenizer vocabulary can never contribute
an undecodable token to a comparison.
"""

from __future__ import annotations

from typing import Sequence

import torch

__all__ = [
    "apply_support_mask",
    "eligible_support_mask",
    "greedy_guess_ids",
    "guess_counts",
    "guess_fractions",
    "nucleus_guess_ids",
    "restrict_to_support",
]


def restrict_to_support(logits: torch.Tensor, *, vocab_size: int) -> torch.Tensor:
    """Keep only the leading ``vocab_size`` logits.

    The model's output layer may be wider than the tokenizer vocabulary -- the
    tiny LLaMA config emits 256 logits while a character tokenizer built from a
    corpus typically has far fewer symbols. Every distribution compared in this
    package must live on the same support, so truncation happens once, here,
    before any policy is applied.

    Args:
        logits: Tensor whose last dimension indexes tokens.
        vocab_size: Number of valid token IDs to keep.

    Returns:
        A view of ``logits`` restricted to ``[..., :vocab_size]``.

    Raises:
        ValueError: If ``vocab_size`` is not positive or exceeds the available
            logits.
    """

    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive.")
    if vocab_size > logits.shape[-1]:
        raise ValueError(
            f"vocab_size {vocab_size} exceeds the model output size {logits.shape[-1]}. "
            "The tokenizer vocabulary cannot be larger than the model vocabulary."
        )
    return logits[..., :vocab_size]


def greedy_guess_ids(logits: torch.Tensor) -> torch.Tensor:
    """Return the argmax token ID at every position.

    Deterministic given the logits, so it introduces no randomness beyond the
    model initialization itself.

    Args:
        logits: Tensor with shape ``[..., vocab]`` already restricted to the
            valid support.

    Returns:
        Token IDs with shape ``logits.shape[:-1]``.
    """

    if logits.ndim < 1:
        raise ValueError("logits must have at least one dimension.")
    return torch.argmax(logits, dim=-1)


def eligible_support_mask(
    vocab_size: int,
    eligible_token_ids: Sequence[int] | None,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build a boolean mask selecting the predictive support.

    ``None`` means every token is eligible, which is the character-tokenizer
    case. Otherwise the listed canonical IDs are kept and everything else -- the
    structural tokens of a pretrained vocabulary -- is excluded. IDs are never
    renumbered: an excluded token keeps its position in every vector and simply
    holds zero.
    """

    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive.")
    if eligible_token_ids is None:
        return torch.ones(vocab_size, dtype=torch.bool, device=torch.device(device))

    ids = torch.as_tensor(list(eligible_token_ids), dtype=torch.long)
    if ids.numel() == 0:
        raise ValueError("eligible_token_ids must not be empty.")
    if int(ids.min()) < 0 or int(ids.max()) >= vocab_size:
        raise ValueError("eligible_token_ids contain values outside the vocabulary.")
    mask = torch.zeros(vocab_size, dtype=torch.bool, device=torch.device(device))
    mask[ids.to(mask.device)] = True
    return mask


def apply_support_mask(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Drive excluded tokens to ``-inf`` in place.

    After this, softmax gives them exactly zero, argmax can never choose them,
    and nucleus truncation sorts them to the tail with no mass. One masking step
    therefore covers all three consumers, which is what keeps the empirical
    comparison and both policies on the same support.
    """

    if mask.shape[0] != logits.shape[-1]:
        raise ValueError(
            f"support mask of size {mask.shape[0]} does not match the vocabulary "
            f"dimension {logits.shape[-1]}."
        )
    if bool(mask.all()):
        return logits
    return logits.masked_fill_(~mask, float("-inf"))


def nucleus_guess_ids(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    generator: torch.Generator | None = None,
    uniforms: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample one token per position with temperature and top-p truncation.

    The procedure follows the reference LLaMA inference implementation:
    probabilities are taken from temperature-scaled logits, sorted descending,
    and every token whose *preceding* cumulative mass already exceeds ``top_p``
    is removed. The surviving head is renormalized and sampled from. Because the
    comparison is against the mass strictly before each token, the single most
    probable token always survives, even when it alone exceeds ``top_p``.

    Args:
        logits: Tensor with shape ``[..., vocab]`` already restricted to the
            valid support.
        temperature: Positive scaling applied before the softmax. Lower values
            concentrate the distribution.
        top_p: Nucleus mass in ``(0, 1]``.
        generator: Optional torch generator, so sampling randomness is
            controlled separately from model-initialization randomness. Used
            only when ``uniforms`` is not supplied.
        uniforms: Optional pre-drawn values in ``[0, 1)``, one per position,
            used to invert the truncated CDF instead of calling
            ``multinomial``. Supplying them makes the draw depend only on the
            position, never on how positions were grouped into batches, which is
            what lets a streamed measurement equal an all-at-once one exactly.

    Returns:
        Token IDs with shape ``logits.shape[:-1]``.
    """

    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must lie in (0, 1].")
    if logits.ndim < 1:
        raise ValueError("logits must have at least one dimension.")

    original_shape = logits.shape[:-1]
    flat_logits = logits.reshape(-1, logits.shape[-1]).float()

    probabilities = torch.softmax(flat_logits / temperature, dim=-1)
    sorted_probabilities, sorted_indices = torch.sort(probabilities, dim=-1, descending=True)
    cumulative = torch.cumsum(sorted_probabilities, dim=-1)

    # Mass strictly before each token; the leading token always keeps 0 here.
    excluded = (cumulative - sorted_probabilities) > top_p
    sorted_probabilities = sorted_probabilities.masked_fill(excluded, 0.0)
    sorted_probabilities = sorted_probabilities / sorted_probabilities.sum(dim=-1, keepdim=True)

    if uniforms is None:
        sampled_positions = torch.multinomial(
            sorted_probabilities, num_samples=1, generator=generator
        )
    else:
        draws = uniforms.reshape(-1)
        if draws.numel() != sorted_probabilities.shape[0]:
            raise ValueError(
                f"uniforms holds {draws.numel()} values but there are "
                f"{sorted_probabilities.shape[0]} positions to sample."
            )
        kept_cumulative = torch.cumsum(sorted_probabilities, dim=-1).contiguous()
        sampled_positions = torch.searchsorted(
            kept_cumulative, draws.to(kept_cumulative.dtype).unsqueeze(-1)
        )
        # A draw at or just below the final cumulative value can land one past
        # the truncated head through rounding. Clamping to the last kept rank
        # keeps every sample inside the nucleus.
        last_kept = (~excluded).sum(dim=-1, keepdim=True) - 1
        sampled_positions = torch.minimum(sampled_positions, last_kept)

    sampled_ids = torch.gather(sorted_indices, -1, sampled_positions).squeeze(-1)
    return sampled_ids.reshape(original_shape)


def guess_counts(token_ids: torch.Tensor, *, vocab_size: int) -> torch.Tensor:
    """Count how often each token ID was selected.

    Args:
        token_ids: Any-shaped integer tensor of selected token IDs.
        vocab_size: Length of the returned count vector.

    Returns:
        A ``[vocab_size]`` tensor of non-negative integer counts.

    Raises:
        ValueError: If ``vocab_size`` is not positive or an ID falls outside it.
    """

    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive.")

    flat = token_ids.reshape(-1).to(torch.long)
    if flat.numel() == 0:
        raise ValueError("token_ids must be non-empty.")
    if bool((flat < 0).any()) or bool((flat >= vocab_size).any()):
        raise ValueError("token_ids contain values outside the valid support.")

    return torch.bincount(flat, minlength=vocab_size)


def guess_fractions(token_ids: torch.Tensor, *, vocab_size: int) -> torch.Tensor:
    """Return selected-guess frequencies, normalized over all positions.

    This is the quantity plotted and compared against empirical corpus
    frequencies. It is *not* the mean predicted probability vector.
    """

    counts = guess_counts(token_ids, vocab_size=vocab_size).float()
    return counts / counts.sum()
