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


# --- Eligible support, effective support, and zero frequency -------------
#
# The zero-frequency tests below exist because of a real defect: nucleus zero
# counts were once taken from the pooled replicates, which gave that policy R
# times as many draws as greedy to reach a rare token. At a 32k vocabulary the
# statistic is dominated by draw count, so the comparison was meaningless.
# Several tests here fail if the pooled and per-replicate definitions are ever
# conflated again.

from llm_behavior_lab.analysis import (  # noqa: E402
    InitializationExperimentRecord,
    corpus_observed_zero_guess_counts,
    effective_supports,
    eligible_view,
    policy_zero_frequency,
    pooled_nucleus_zero_frequency,
    summarize_policy,
    support_summary,
)


def _record(**overrides) -> InitializationExperimentRecord:
    """A 4-eligible-token record with hand-chosen counts.

    Token 0 is structural. Greedy draws N=2; each of the two nucleus replicates
    also draws N=2, but they cover *different* tokens, so pooling reaches one
    more token than either replicate alone.
    """

    fields = {
        "corpus_counts": np.array([0, 10, 10, 10, 10]),
        "selected_target_counts": np.array([0, 1, 1, 1, 1]),
        "greedy_counts": np.array([[0, 2, 0, 0, 0]]),
        "nucleus_counts": np.array([[[0, 2, 0, 0, 0], [0, 0, 2, 0, 0]]]),
        "mean_predicted_probabilities": np.array([[0.0, 0.25, 0.25, 0.25, 0.25]]),
        "model_seeds": np.array([1]),
        "eligible_token_ids": np.array([1, 2, 3, 4]),
        "metadata": {"num_positions": 2},
    }
    fields.update(overrides)
    return InitializationExperimentRecord.build(**fields)


def test_eligible_view_drops_structural_tokens() -> None:
    """Statistics run on the support the model is scored over."""

    record = _record()

    assert record.eligible_vocab_size == 4
    assert eligible_view(record, record.corpus_counts).tolist() == [10, 10, 10, 10]
    assert record.special_token_ids.tolist() == [0]


def test_greedy_zero_frequency_counts_untouched_eligible_tokens() -> None:
    """Greedy made 2 draws on one token, so 3 of 4 eligible tokens are untouched."""

    zero = policy_zero_frequency(_record(), "greedy")

    assert zero.counts.tolist() == [3.0]
    assert zero.fractions.tolist() == pytest.approx([0.75])
    assert zero.draws_per_measurement == 2


def test_nucleus_zero_frequency_is_averaged_per_replicate() -> None:
    """Each replicate is scored on its own N draws, then the counts are averaged.

    Both replicates leave 3 of 4 tokens untouched, so the per-replicate answer is
    3 -- exactly comparable with greedy, which also had 2 draws.
    """

    zero = policy_zero_frequency(_record(), "nucleus")

    assert zero.counts.tolist() == [3.0]
    assert zero.draws_per_measurement == 2


def test_pooled_and_per_replicate_zero_frequency_differ() -> None:
    """The defect this correction fixes, made visible.

    Pooling the two replicates reaches tokens 1 and 2, leaving 2 untouched. That
    is a better-looking number obtained from twice as many draws, and it is not
    comparable with greedy's 3.
    """

    record = _record()

    per_replicate = policy_zero_frequency(record, "nucleus")
    pooled = pooled_nucleus_zero_frequency(record)

    assert per_replicate.counts.tolist() == [3.0]
    assert pooled.counts.tolist() == [2.0]
    assert pooled.draws_per_measurement == 2 * per_replicate.draws_per_measurement


def test_the_headline_nucleus_statistic_is_the_per_replicate_one() -> None:
    """The summary must not quietly report the flattering pooled figure."""

    summary = summarize_policy(_record(), "nucleus")

    assert summary.zero_frequency_count_mean == 3.0
    assert summary.pooled_zero_frequency_count_mean == 2.0
    assert summary.zero_frequency_draws == 2
    assert summary.pooled_zero_frequency_draws == 4


def test_greedy_has_no_pooled_zero_frequency_figure() -> None:
    """There is nothing to pool: greedy has one selection per position."""

    summary = summarize_policy(_record(), "greedy")

    assert summary.pooled_zero_frequency_count_mean is None
    assert summary.pooled_zero_frequency_draws is None


def test_both_policies_are_measured_at_the_same_draw_count() -> None:
    """The property that makes the two numbers comparable at all."""

    record = _record()

    greedy = policy_zero_frequency(record, "greedy")
    nucleus = policy_zero_frequency(record, "nucleus")

    assert greedy.draws_per_measurement == nucleus.draws_per_measurement


def test_zero_frequency_fraction_uses_the_eligible_support() -> None:
    """A fraction of the full vocabulary would drift with the special-token count."""

    zero = policy_zero_frequency(_record(), "greedy")

    assert zero.eligible_vocab_size == 4
    assert zero.fractions[0] == pytest.approx(zero.counts[0] / 4)


def test_zero_frequency_averages_counts_rather_than_counting_zero_averages() -> None:
    """Averaging first is precisely the bug; the order is asserted here.

    Two replicates that each miss three tokens but miss *different* ones average
    to 3. Counting zeros of the averaged fractions would give 2.
    """

    record = _record()
    averaged_fractions_zeros = int(
        (eligible_view(record, record.nucleus_fractions)[0] == 0).sum()
    )

    assert policy_zero_frequency(record, "nucleus").counts[0] == 3.0
    assert averaged_fractions_zeros == 2


def test_support_summary_keeps_four_quantities_distinct() -> None:
    """Full, eligible, corpus-observed, and effective support are different things."""

    record = _record(corpus_counts=np.array([0, 10, 10, 0, 0]))

    summary = support_summary(record)

    assert summary["vocab_size"] == 5
    assert summary["eligible_vocab_size"] == 4
    assert summary["corpus_observed_support"] == 2
    # Two equally frequent tokens: e^H = 2 exactly.
    assert summary["corpus_effective_support"] == pytest.approx(2.0)


def test_effective_support_is_reported_per_initialization() -> None:
    """So it can carry a between-initialization standard error."""

    record = _record()

    supports = effective_supports(record, "greedy")

    assert supports.shape == (1,)
    # All greedy mass on one token: e^H = 1.
    assert supports[0] == pytest.approx(1.0)


def test_effective_support_sem_is_reported() -> None:
    """Two initializations with different breadth must produce a non-zero SEM."""

    record = _record(
        greedy_counts=np.array([[0, 2, 0, 0, 0], [0, 1, 1, 0, 0]]),
        nucleus_counts=np.array(
            [[[0, 2, 0, 0, 0], [0, 0, 2, 0, 0]], [[0, 1, 1, 0, 0], [0, 1, 1, 0, 0]]]
        ),
        mean_predicted_probabilities=np.tile(np.array([0.0, 0.25, 0.25, 0.25, 0.25]), (2, 1)),
        model_seeds=np.array([1, 2]),
    )

    summary = summarize_policy(record, "greedy")

    # All mass on one token gives e^H = 1; an even split over two gives 2.
    # std(ddof=1) of [1, 2] is sqrt(0.5), so the SEM is sqrt(0.5)/sqrt(2) = 0.5.
    assert summary.effective_support_mean == pytest.approx(1.5)
    assert summary.effective_support_sem == pytest.approx(0.5)


def test_corpus_observed_zero_guess_is_a_separate_diagnostic() -> None:
    """It asks about corpus tokens only, not the whole eligible support."""

    record = _record(corpus_counts=np.array([0, 10, 10, 0, 0]))

    # Corpus uses tokens 1 and 2; greedy only ever guessed token 1.
    assert corpus_observed_zero_guess_counts(record, "greedy").tolist() == [1.0]
    # Pooled nucleus reached both, so none of the corpus tokens is unreached.
    assert corpus_observed_zero_guess_counts(record, "nucleus").tolist() == [0.0]


def test_aggregation_scales_to_a_subword_vocabulary() -> None:
    """The statistics must stay correct, and quick, at 32k tokens."""

    vocab_size, eligible_count = 32000, 31997
    eligible = np.arange(3, vocab_size)
    corpus = np.zeros(vocab_size, dtype=np.int64)
    corpus[eligible] = 1
    guesses = np.zeros((2, vocab_size), dtype=np.int64)
    guesses[:, 3:103] = 1  # 100 distinct tokens selected per initialization
    nucleus = np.zeros((2, 2, vocab_size), dtype=np.int64)
    nucleus[:, :, 3:103] = 1

    record = InitializationExperimentRecord.build(
        corpus_counts=corpus,
        selected_target_counts=corpus,
        greedy_counts=guesses,
        nucleus_counts=nucleus,
        mean_predicted_probabilities=np.tile(corpus / corpus.sum(), (2, 1)),
        model_seeds=np.array([1, 2]),
        eligible_token_ids=eligible,
        metadata={"num_positions": 100},
    )

    summary = summarize_policy(record, "greedy")

    assert record.eligible_vocab_size == eligible_count
    assert summary.zero_frequency_count_mean == eligible_count - 100
    assert summary.zero_frequency_fraction_mean == pytest.approx(
        (eligible_count - 100) / eligible_count
    )
    assert summary.effective_support_mean == pytest.approx(100.0)
