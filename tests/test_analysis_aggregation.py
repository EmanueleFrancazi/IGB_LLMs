"""Tests for initialization-experiment aggregation.

Every expected value here is computable by hand from the small vectors defined
in the test, so a failure points at the formula rather than at a fixture.
"""

from __future__ import annotations

import numpy as np
import pytest

from llm_behavior_lab.analysis import (
    effective_support,
    js_divergence,
    mean_with_sem,
    persistent_absolute_gaps,
    ranked_profile,
    ranked_profiles,
    shannon_entropy,
    token_wise_absolute_gaps,
    top_two_gap,
    total_variation_distance,
    zero_guess_counts,
)

CORPUS = np.array([0.5, 0.3, 0.2])
#: Exactly the corpus distribution with its token identities reversed. Ranking
#: first would call this a perfect match; differencing first does not.
REVERSED = np.array([0.2, 0.3, 0.5])


def test_ranked_profile_sorts_descending() -> None:
    """Concentration profiles discard identity and decrease with rank."""

    assert ranked_profile(np.array([0.1, 0.4, 0.2])).tolist() == [0.4, 0.2, 0.1]


def test_ranked_profiles_rank_each_initialization_independently() -> None:
    """Each row is its own distribution and is sorted on its own."""

    matrix = np.array([[0.1, 0.6, 0.3], [0.5, 0.2, 0.3]])

    assert ranked_profiles(matrix).tolist() == [[0.6, 0.3, 0.1], [0.5, 0.3, 0.2]]


def test_gaps_are_taken_between_the_same_token_before_ranking() -> None:
    """The defining property of the mismatch measure.

    A reversed distribution has an identical ranked profile, so ranking before
    differencing would report a perfect match. Differencing first exposes the
    real disagreement.
    """

    gaps = token_wise_absolute_gaps(REVERSED[None, :], CORPUS)

    assert gaps.tolist() == [[0.3, 0.0, 0.3]]
    assert ranked_profile(ranked_profile(REVERSED) - ranked_profile(CORPUS)).tolist() == [
        0.0,
        0.0,
        0.0,
    ]


def test_persistent_gap_averages_fractions_before_taking_absolute_values() -> None:
    """Persistent mismatch is |mean(q) - p|, not mean(|q - p|).

    Two initializations that deviate in opposite directions cancel here while
    the typical per-initialization gap stays large. That contrast is the point
    of reporting both.
    """

    guesses = np.array([[0.7, 0.3, 0.0], [0.3, 0.3, 0.4]])

    persistent = persistent_absolute_gaps(guesses, CORPUS)
    typical_mean = token_wise_absolute_gaps(guesses, CORPUS).mean(axis=0)

    assert persistent.tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert typical_mean.tolist() == pytest.approx([0.2, 0.0, 0.2])


def test_mean_with_sem_matches_the_hand_computed_standard_error() -> None:
    """SEM is the sample standard deviation divided by sqrt(N)."""

    result = mean_with_sem(np.array([[0.3, 0.1], [0.1, 0.1]]))

    assert result.num_samples == 2
    assert result.mean.tolist() == pytest.approx([0.2, 0.1])
    # std(ddof=1) of [0.3, 0.1] is 0.1*sqrt(2); dividing by sqrt(2) leaves 0.1.
    assert result.sem.tolist() == pytest.approx([0.1, 0.0])


def test_a_single_initialization_reports_zero_error_not_nan() -> None:
    """One sample has no estimable spread; the profile must stay usable."""

    result = mean_with_sem(np.array([[0.4, 0.6]]))

    assert result.num_samples == 1
    assert result.sem.tolist() == [0.0, 0.0]
    assert not np.isnan(result.sem).any()


def test_total_variation_distance_is_half_the_absolute_difference() -> None:
    """TV of the reversed distribution is 0.5*(0.3 + 0 + 0.3)."""

    assert total_variation_distance(CORPUS, REVERSED) == pytest.approx(0.3)
    assert total_variation_distance(CORPUS, CORPUS) == pytest.approx(0.0)


def test_total_variation_distance_is_bounded_by_one() -> None:
    """Disjoint support is the maximum separation."""

    left = np.array([1.0, 0.0])
    right = np.array([0.0, 1.0])

    assert total_variation_distance(left, right) == pytest.approx(1.0)


def test_js_divergence_is_zero_for_identical_distributions() -> None:
    """And bounded above by ln 2 for disjoint ones."""

    assert js_divergence(CORPUS, CORPUS) == pytest.approx(0.0, abs=1e-9)
    assert js_divergence(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(
        np.log(2.0), abs=1e-6
    )


def test_entropy_and_effective_support_of_a_uniform_distribution() -> None:
    """A uniform distribution over four tokens has effective support four."""

    uniform = np.full(4, 0.25)

    assert shannon_entropy(uniform) == pytest.approx(np.log(4.0))
    assert effective_support(uniform) == pytest.approx(4.0)


def test_effective_support_shrinks_as_mass_concentrates() -> None:
    """The measure must be readable as 'how many tokens are really in play'."""

    concentrated = np.array([0.97, 0.01, 0.01, 0.01])

    assert effective_support(concentrated) < 1.5
    assert effective_support(concentrated) < effective_support(np.full(4, 0.25))


def test_zero_guess_counts_are_per_initialization() -> None:
    """Tokens never selected are counted separately for each initialization."""

    guesses = np.array([[0.5, 0.5, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])

    assert zero_guess_counts(guesses).tolist() == [2, 3]


def test_top_two_gap_uses_the_ranked_distribution() -> None:
    """The gap is between the largest and second-largest entries."""

    assert top_two_gap(np.array([0.1, 0.4, 0.3, 0.2])) == pytest.approx(0.1)


def test_shape_mismatches_are_rejected() -> None:
    """Silent broadcasting would corrupt a token-aligned comparison."""

    with pytest.raises(ValueError):
        token_wise_absolute_gaps(np.zeros((2, 3)), np.zeros(4))
    with pytest.raises(ValueError):
        total_variation_distance(np.zeros(3), np.zeros(4))
    with pytest.raises(ValueError):
        ranked_profiles(np.zeros(3))
