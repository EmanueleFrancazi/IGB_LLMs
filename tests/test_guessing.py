"""Tests for the greedy and nucleus token-selection policies.

Sampling is checked through properties that hold for every draw -- support
membership, reproducibility, and the nucleus truncation rule -- rather than
through fixed token IDs, which would only pin down one PyTorch version's RNG.
"""

from __future__ import annotations

import torch

import pytest

from llm_behavior_lab.evaluation.guessing import (
    greedy_guess_ids,
    guess_counts,
    guess_fractions,
    nucleus_guess_ids,
    restrict_to_support,
)


def _logits(rows: list[list[float]]) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.float32)


def test_restrict_to_support_truncates_the_model_vocabulary() -> None:
    """A model wider than the tokenizer must not contribute undecodable tokens."""

    logits = _logits([[1.0, 2.0, 3.0, 4.0]])

    restricted = restrict_to_support(logits, vocab_size=2)

    assert restricted.shape == (1, 2)
    assert restricted.tolist() == [[1.0, 2.0]]


def test_restrict_to_support_rejects_a_vocabulary_larger_than_the_model() -> None:
    """That configuration is a mistake, not something to pad around."""

    with pytest.raises(ValueError, match="exceeds the model output size"):
        restrict_to_support(_logits([[1.0, 2.0]]), vocab_size=5)


def test_greedy_selects_the_argmax_at_every_position() -> None:
    """The deterministic policy, computed by hand."""

    logits = _logits([[0.1, 0.9, 0.2], [3.0, 1.0, 2.0]])

    assert greedy_guess_ids(logits).tolist() == [1, 0]


def test_greedy_is_deterministic_across_calls() -> None:
    """Greedy introduces no randomness beyond the initialization itself."""

    logits = torch.randn(64, 16)

    assert torch.equal(greedy_guess_ids(logits), greedy_guess_ids(logits))


def test_greedy_preserves_the_position_shape() -> None:
    """One guess per prediction position, with the vocabulary axis removed."""

    logits = torch.randn(4, 7, 16)

    assert greedy_guess_ids(logits).shape == (4, 7)


def test_nucleus_sampling_is_reproducible_for_one_seed() -> None:
    """Two generators seeded identically must produce identical guesses."""

    logits = torch.randn(128, 12)
    first = torch.Generator().manual_seed(4242)
    second = torch.Generator().manual_seed(4242)

    left = nucleus_guess_ids(logits, temperature=0.6, top_p=0.9, generator=first)
    right = nucleus_guess_ids(logits, temperature=0.6, top_p=0.9, generator=second)

    assert torch.equal(left, right)


def test_nucleus_sampling_differs_between_seeds() -> None:
    """Different sampling seeds must actually explore differently."""

    logits = torch.randn(256, 12)
    first = torch.Generator().manual_seed(1)
    second = torch.Generator().manual_seed(2)

    left = nucleus_guess_ids(logits, temperature=1.0, top_p=0.9, generator=first)
    right = nucleus_guess_ids(logits, temperature=1.0, top_p=0.9, generator=second)

    assert not torch.equal(left, right)


def test_nucleus_only_selects_tokens_inside_the_support() -> None:
    """Restriction happens before sampling, so no ID can escape the vocabulary."""

    logits = torch.randn(512, 9)
    generator = torch.Generator().manual_seed(7)

    guesses = nucleus_guess_ids(logits, temperature=0.6, top_p=0.9, generator=generator)

    assert int(guesses.min()) >= 0
    assert int(guesses.max()) < 9


def test_nucleus_drops_a_token_above_the_threshold() -> None:
    """Tokens whose preceding cumulative mass exceeds top_p are unreachable.

    The mass *before* each token here is about ``[0, 0.6, 0.9, 0.97]``. At
    ``top_p = 0.85`` the third token is above the threshold by a wide margin, so
    only the first two survive.
    """

    probabilities = torch.tensor([0.6, 0.3, 0.07, 0.03])
    logits = probabilities.log().unsqueeze(0).repeat(400, 1)
    generator = torch.Generator().manual_seed(11)

    guesses = nucleus_guess_ids(logits, temperature=1.0, top_p=0.85, generator=generator)

    assert set(guesses.tolist()) == {0, 1}


def test_nucleus_keeps_a_token_below_the_threshold() -> None:
    """The same distribution, with the third token comfortably inside.

    Its preceding mass of about 0.9 is below ``top_p = 0.92``, so it becomes
    reachable while the fourth token stays excluded.
    """

    probabilities = torch.tensor([0.6, 0.3, 0.07, 0.03])
    logits = probabilities.log().unsqueeze(0).repeat(600, 1)
    generator = torch.Generator().manual_seed(11)

    guesses = nucleus_guess_ids(logits, temperature=1.0, top_p=0.92, generator=generator)

    assert set(guesses.tolist()) == {0, 1, 2}


def test_the_nucleus_boundary_is_exclusive() -> None:
    """A token whose preceding mass equals top_p exactly is kept, not dropped.

    This mirrors the reference LLaMA rule, where the comparison is against the
    mass strictly preceding a token.

    The construction matters. Recovering probabilities through ``log`` and
    ``softmax`` cannot test an exact boundary: rebuilding ``[0.6, 0.3, ...]``
    that way yields a preceding mass near 0.9 that lands a few units in the last
    place on either side of it, differing between float32 backends, so the
    outcome would be arbitrary rather than wrong.

    Uniform logits avoid that entirely. ``exp(0)`` is exactly 1, the sum is
    exactly 4, and ``1/4`` is exact in binary floating point, so the preceding
    masses are exactly ``[0, 0.25, 0.5, 0.75]``. With ``top_p = 0.5`` the third
    token sits precisely on the threshold and must survive, leaving three
    reachable tokens rather than two.

    The assertion counts reachable tokens instead of naming them, because the
    probabilities are tied and which IDs occupy the kept head depends on the
    sort's tie ordering.
    """

    logits = torch.zeros(600, 4)
    generator = torch.Generator().manual_seed(11)

    guesses = nucleus_guess_ids(logits, temperature=1.0, top_p=0.5, generator=generator)

    assert len(set(guesses.tolist())) == 3


def test_nucleus_always_keeps_the_most_probable_token() -> None:
    """A token exceeding top_p on its own must still be selectable.

    Otherwise the nucleus would be empty and the policy undefined.
    """

    probabilities = torch.tensor([0.97, 0.02, 0.01])
    logits = probabilities.log().unsqueeze(0).repeat(50, 1)
    generator = torch.Generator().manual_seed(5)

    guesses = nucleus_guess_ids(logits, temperature=1.0, top_p=0.5, generator=generator)

    assert set(guesses.tolist()) == {0}


def test_a_low_temperature_concentrates_on_the_argmax() -> None:
    """Temperature must sharpen the distribution it is applied to."""

    logits = _logits([[2.0, 1.0, 0.0]]).repeat(400, 1)
    cold = nucleus_guess_ids(
        logits, temperature=0.05, top_p=1.0, generator=torch.Generator().manual_seed(3)
    )
    warm = nucleus_guess_ids(
        logits, temperature=2.0, top_p=1.0, generator=torch.Generator().manual_seed(3)
    )

    assert (cold == 0).float().mean() > (warm == 0).float().mean()
    assert (cold == 0).float().mean() > 0.95


def test_top_p_of_one_permits_the_whole_vocabulary() -> None:
    """No truncation at all is a legal, and useful, configuration."""

    logits = torch.zeros(2000, 5)
    generator = torch.Generator().manual_seed(9)

    guesses = nucleus_guess_ids(logits, temperature=1.0, top_p=1.0, generator=generator)

    assert set(guesses.tolist()) == {0, 1, 2, 3, 4}


def test_invalid_sampling_parameters_are_rejected() -> None:
    """Silently clamping them would misreport the policy that ran."""

    logits = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="temperature"):
        nucleus_guess_ids(logits, temperature=0.0, top_p=0.9)
    with pytest.raises(ValueError, match="top_p"):
        nucleus_guess_ids(logits, temperature=1.0, top_p=0.0)
    with pytest.raises(ValueError, match="top_p"):
        nucleus_guess_ids(logits, temperature=1.0, top_p=1.5)


def test_guess_counts_total_the_number_of_positions() -> None:
    """Every position contributes exactly one guess."""

    guesses = torch.tensor([[0, 1, 1], [2, 2, 2]])

    counts = guess_counts(guesses, vocab_size=4)

    assert counts.tolist() == [1, 2, 3, 0]
    assert int(counts.sum()) == 6


def test_guess_counts_reject_ids_outside_the_support() -> None:
    """An out-of-range ID means the support was not applied; fail loudly."""

    with pytest.raises(ValueError, match="outside the valid support"):
        guess_counts(torch.tensor([0, 5]), vocab_size=4)


def test_guess_fractions_sum_to_one() -> None:
    """The comparison against corpus fractions needs a normalized vector."""

    fractions = guess_fractions(torch.tensor([0, 1, 1, 3]), vocab_size=4)

    assert float(fractions.sum()) == pytest.approx(1.0)
    assert fractions.tolist() == pytest.approx([0.25, 0.5, 0.0, 0.25])
