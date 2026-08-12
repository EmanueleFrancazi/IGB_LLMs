"""Tests for the uniform categorical output null.

The null exists to answer whether ranked model structure differs from what
finite sampling alone produces, so the tests concentrate on the two ways that
answer could be wrong: simulating the wrong distribution, and reporting Monte
Carlo uncertainty as if it were a model-initialization SEM.

Analytic occupancy formulas give exact expectations to check against. They are
checks, never substitutes: no analytic expression here produces the *ranked*
profile, which is the whole point of simulating.
"""

from __future__ import annotations

import numpy as np
import pytest

from llm_behavior_lab.analysis.nulls import (
    DEFAULT_NULL_REPLICATES,
    expected_occupancy_counts,
    expected_zero_frequency_count,
    expected_zero_frequency_fraction,
    simulate_uniform_null,
)


def test_a_single_token_support_is_degenerate() -> None:
    """With K=1 every draw lands on the only token."""

    summary = simulate_uniform_null(
        eligible_vocab_size=1, num_draws=5, num_replicates=3, seed=1
    )

    assert summary.ranked_mean.tolist() == [1.0]
    assert summary.zero_frequency_counts.tolist() == [0.0, 0.0, 0.0]
    assert summary.top_two_gaps.tolist() == [0.0, 0.0, 0.0]


def test_the_ranked_profile_spans_the_eligible_support() -> None:
    """One entry per eligible token, so it aligns with the model curves."""

    summary = simulate_uniform_null(
        eligible_vocab_size=37, num_draws=100, num_replicates=8, seed=2
    )

    assert summary.ranked_mean.shape == (37,)
    assert summary.ranked_low.shape == (37,)
    assert summary.ranked_high.shape == (37,)


def test_ranked_frequencies_are_non_increasing() -> None:
    """It is a ranked profile; anything else is a bug in the sort."""

    summary = simulate_uniform_null(
        eligible_vocab_size=50, num_draws=200, num_replicates=6, seed=3
    )

    assert np.all(np.diff(summary.ranked_mean) <= 1e-15)


def test_each_realization_uses_every_draw() -> None:
    """A joint multinomial spends exactly D draws, so frequencies sum to one."""

    summary = simulate_uniform_null(
        eligible_vocab_size=25, num_draws=64, num_replicates=5, seed=4
    )

    assert summary.ranked_mean.sum() == pytest.approx(1.0)


def test_the_interval_brackets_the_mean() -> None:
    """The Monte Carlo envelope must actually contain the mean profile."""

    summary = simulate_uniform_null(
        eligible_vocab_size=40, num_draws=150, num_replicates=64, seed=5
    )

    assert np.all(summary.ranked_low <= summary.ranked_high)
    assert np.all(summary.ranked_low <= summary.ranked_mean + 1e-12)
    assert np.all(summary.ranked_mean <= summary.ranked_high + 1e-12)


def test_the_null_is_reproducible_from_its_seed() -> None:
    """A figure must be regenerable from the record."""

    first = simulate_uniform_null(eligible_vocab_size=30, num_draws=90, num_replicates=6, seed=99)
    second = simulate_uniform_null(eligible_vocab_size=30, num_draws=90, num_replicates=6, seed=99)

    assert np.array_equal(first.ranked_mean, second.ranked_mean)


def test_a_different_seed_gives_different_realizations() -> None:
    """Otherwise the seed would not be doing anything."""

    first = simulate_uniform_null(eligible_vocab_size=30, num_draws=90, num_replicates=6, seed=1)
    second = simulate_uniform_null(eligible_vocab_size=30, num_draws=90, num_replicates=6, seed=2)

    assert not np.array_equal(first.ranked_mean, second.ranked_mean)


def test_analytic_zero_frequency_matches_the_simulation() -> None:
    """``E[Z] = K (1 - 1/K)^D``, checked against the joint draws."""

    eligible, draws = 2000, 1500
    summary = simulate_uniform_null(
        eligible_vocab_size=eligible, num_draws=draws, num_replicates=200, seed=7
    )

    analytic = expected_zero_frequency_count(eligible, draws)
    observed = float(summary.zero_frequency_counts.mean())

    assert observed == pytest.approx(analytic, rel=0.02)


def test_the_zero_frequency_fraction_is_the_count_over_the_support() -> None:
    """Consistency between the two ways the statistic is reported."""

    eligible, draws = 500, 400

    assert expected_zero_frequency_fraction(eligible, draws) == pytest.approx(
        expected_zero_frequency_count(eligible, draws) / eligible
    )
    assert expected_zero_frequency_fraction(eligible, draws) == pytest.approx(
        (1.0 - 1.0 / eligible) ** draws
    )


def test_occupancy_expectations_are_a_partition_of_the_support() -> None:
    """``sum_j E[N_j] = K``: every token is observed some number of times."""

    eligible, draws = 60, 40

    total = sum(expected_occupancy_counts(eligible, draws, j) for j in range(draws + 1))

    assert total == pytest.approx(eligible)


def test_occupancy_at_zero_matches_the_zero_frequency_expectation() -> None:
    """The ``j = 0`` case must reduce to the simpler formula."""

    assert expected_occupancy_counts(300, 250, 0) == pytest.approx(
        expected_zero_frequency_count(300, 250)
    )


def test_occupancy_expectations_match_a_direct_simulation() -> None:
    """``E[N_j]`` for the first few j, against explicit multinomial draws."""

    eligible, draws = 400, 300
    generator = np.random.default_rng(11)
    probabilities = np.full(eligible, 1.0 / eligible)
    observed = np.zeros(3)
    for _ in range(300):
        counts = generator.multinomial(draws, probabilities)
        for times in range(3):
            observed[times] += (counts == times).sum()
    observed /= 300

    for times in range(3):
        assert observed[times] == pytest.approx(
            expected_occupancy_counts(eligible, draws, times), rel=0.05
        )


def test_the_draw_is_joint_not_independent_binomials() -> None:
    """The property that distinguishes the two implementations.

    Independent ``Binomial(D, 1/K)`` counts do not sum to ``D``. A joint
    multinomial always does, and that dependence is what shapes the ranked
    profile. This test fails immediately if the simulation is ever replaced by
    independent marginals.
    """

    eligible, draws = 200, 150
    summary = simulate_uniform_null(
        eligible_vocab_size=eligible, num_draws=draws, num_replicates=32, seed=13
    )

    # Every realization spends exactly D draws, so every ranked profile sums to
    # one after dividing by D. Independent binomials would fluctuate around it.
    reconstructed_totals = summary.ranked_mean.sum()
    assert reconstructed_totals == pytest.approx(1.0, abs=1e-12)

    # And the counts recovered from any realization are integers summing to D.
    counts = np.random.default_rng(13).multinomial(draws, np.full(eligible, 1.0 / eligible))
    assert counts.sum() == draws


def test_the_summary_labels_its_interval_as_monte_carlo() -> None:
    """Naming is the guard against confusing it with an initialization SEM."""

    described = simulate_uniform_null(
        eligible_vocab_size=20, num_draws=50, num_replicates=16, seed=17
    ).as_dict()

    assert "monte_carlo_interval" in described
    assert "monte_carlo_replicates" in described
    assert described["effective_support_mc_low"] <= described["effective_support_mean"]
    assert described["effective_support_mean"] <= described["effective_support_mc_high"]
    assert not any("sem" in key for key in described)


def test_the_summary_carries_the_analytic_cross_check() -> None:
    """So a reader can see the simulation agreeing with theory."""

    described = simulate_uniform_null(
        eligible_vocab_size=800, num_draws=600, num_replicates=64, seed=19
    ).as_dict()

    assert described["zero_frequency_count_mean"] == pytest.approx(
        described["analytic_zero_frequency_count"], rel=0.05
    )


def test_a_subword_scale_null_is_tractable() -> None:
    """32k tokens must not be slow or memory-hungry enough to matter."""

    summary = simulate_uniform_null(
        eligible_vocab_size=31997, num_draws=8192, num_replicates=16, seed=23
    )

    assert summary.ranked_mean.shape == (31997,)
    assert summary.zero_frequency_counts.mean() == pytest.approx(
        expected_zero_frequency_count(31997, 8192), rel=0.01
    )


def test_the_null_has_a_long_zero_tail_at_subword_scale() -> None:
    """Most of the profile is exactly zero, which log axes must survive."""

    summary = simulate_uniform_null(
        eligible_vocab_size=5000, num_draws=500, num_replicates=8, seed=29
    )

    assert (summary.ranked_mean == 0.0).sum() > 4000
    assert summary.ranked_mean[0] > 0.0


def test_invalid_null_parameters_are_rejected() -> None:
    """Silently clamping would misreport what the null describes."""

    with pytest.raises(ValueError, match="eligible_vocab_size"):
        simulate_uniform_null(eligible_vocab_size=0, num_draws=10)
    with pytest.raises(ValueError, match="num_draws"):
        simulate_uniform_null(eligible_vocab_size=10, num_draws=0)
    with pytest.raises(ValueError, match="num_replicates"):
        simulate_uniform_null(eligible_vocab_size=10, num_draws=10, num_replicates=0)
    with pytest.raises(ValueError, match="interval"):
        simulate_uniform_null(eligible_vocab_size=10, num_draws=10, interval=1.5)


def test_the_default_replicate_count_is_documented_and_moderate() -> None:
    """Large enough to smooth the envelope, small enough to stay cheap."""

    assert 100 <= DEFAULT_NULL_REPLICATES <= 1000
