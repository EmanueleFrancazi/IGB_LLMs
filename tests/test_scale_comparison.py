"""Tests for the cross-alpha greedy comparison.

Position-wise agreement and distributional total variation answer different
questions, and the whole value of reporting both is that they can disagree. Two
conditions can produce nearly identical *marginal* guess distributions while
choosing different tokens at a large fraction of individual positions, which is
what a reorganization of output preferences looks like when the shape of the
distribution happens to be preserved. The fixture below constructs exactly that
case, so a implementation that conflated the two would fail here.

Nothing asserts what lowering alpha actually does. That is the empirical question
the scale experiment exists to answer.
"""

from __future__ import annotations

import numpy as np
import pytest

from llm_behavior_lab.analysis import (
    InitializationExperimentRecord,
    greedy_scale_comparison,
)

VOCAB_SIZE = 10
ELIGIBLE = np.arange(2, VOCAB_SIZE)
D = 16


def _record(greedy_positions: np.ndarray, alpha: float, *, with_positions: bool = True):
    """A record whose greedy behaviour is dictated by ``greedy_positions``."""

    targets = ELIGIBLE[np.arange(D) % ELIGIBLE.size]
    corpus = np.zeros(VOCAB_SIZE, dtype=np.int64)
    corpus[ELIGIBLE] = 10
    counts = np.bincount(greedy_positions, minlength=VOCAB_SIZE)

    arrays = {}
    if with_positions:
        arrays = {
            "gradient_position_indices": np.arange(D),
            "gradient_position_target_ids": targets,
            "gradient_position_greedy_ids": greedy_positions,
            "gradient_position_norms": np.ones(D),
        }

    return InitializationExperimentRecord.build(
        corpus_counts=corpus,
        selected_target_counts=np.bincount(targets, minlength=VOCAB_SIZE),
        greedy_counts=counts[None, :],
        nucleus_counts=counts[None, None, :],
        mean_predicted_probabilities=np.full((1, VOCAB_SIZE), 1.0 / VOCAB_SIZE),
        model_seeds=[1000],
        eligible_token_ids=ELIGIBLE,
        metadata={
            "initialization_scale": {"alpha": alpha},
            # Top level, exactly as the runner writes it: the zero-frequency
            # statistic reads the draw count from here.
            "num_positions": D,
            "analysis": {
                "num_positions": D,
                "gradient_analysis": {
                    "enabled": True,
                    "initialization_index": 0,
                    "covers_all_positions": True,
                },
            },
        },
        **arrays,
    )


def test_identical_greedy_behaviour_gives_perfect_agreement_and_zero_distance() -> None:
    """The pure-rescaling case: nothing about greedy moved."""

    greedy = ELIGIBLE[np.arange(D) % 4]
    comparison = greedy_scale_comparison(_record(greedy, 1.0), _record(greedy, 0.5))

    assert comparison["position_wise_greedy_agreement"]["agreement"] == 1.0
    assert comparison["position_wise_greedy_agreement"]["num_changed"] == 0
    assert comparison["distribution_total_variation"]["mean"] == pytest.approx(0.0)
    assert comparison["baseline_alpha"] == 1.0
    assert comparison["scaled_alpha"] == 0.5


def test_a_permutation_keeps_the_distribution_but_moves_every_position() -> None:
    """The case the two measures exist to separate.

    Swapping which token is chosen at which position, in a way that preserves how
    often each token is chosen overall, leaves the marginal distribution
    identical while changing a large share of individual decisions. Reporting
    only total variation would call this "no change".
    """

    baseline = ELIGIBLE[np.arange(D) % 2]            # alternating A, B, A, B...
    scaled = ELIGIBLE[(np.arange(D) + 1) % 2]        # the same two tokens, swapped

    comparison = greedy_scale_comparison(_record(baseline, 1.0), _record(scaled, 0.25))

    assert comparison["distribution_total_variation"]["mean"] == pytest.approx(0.0)
    assert comparison["position_wise_greedy_agreement"]["agreement"] == 0.0
    assert comparison["position_wise_greedy_agreement"]["num_changed"] == D


def test_a_changed_distribution_is_reported_by_total_variation() -> None:
    baseline = np.full(D, ELIGIBLE[0])
    scaled = np.full(D, ELIGIBLE[1])

    comparison = greedy_scale_comparison(_record(baseline, 1.0), _record(scaled, 0.5))

    assert comparison["distribution_total_variation"]["mean"] == pytest.approx(1.0)
    assert comparison["position_wise_greedy_agreement"]["agreement"] == 0.0


def test_support_and_zero_frequency_deltas_are_reported() -> None:
    """Concentrating onto fewer tokens lowers support and raises zero frequency."""

    spread = ELIGIBLE[np.arange(D) % ELIGIBLE.size]
    concentrated = np.full(D, ELIGIBLE[0])

    comparison = greedy_scale_comparison(_record(spread, 1.0), _record(concentrated, 0.25))

    assert comparison["effective_support"]["delta_mean"] < 0.0
    assert comparison["zero_frequency_fraction"]["delta_mean"] > 0.0
    assert comparison["zero_frequency_fraction"]["draws_per_measurement"] == D


def test_agreement_is_unavailable_without_per_position_identities() -> None:
    """Marginal counts cannot recover it, so absence is reported, not guessed."""

    greedy = ELIGIBLE[np.arange(D) % 4]
    comparison = greedy_scale_comparison(
        _record(greedy, 1.0, with_positions=False),
        _record(greedy, 0.5, with_positions=False),
    )

    agreement = comparison["position_wise_greedy_agreement"]
    assert agreement["available"] is False
    assert "gradient analysis" in agreement["reason"]
    # The distributional measures remain available and correct.
    assert comparison["distribution_total_variation"]["mean"] == pytest.approx(0.0)


def test_records_covering_different_positions_are_not_compared_position_wise() -> None:
    greedy = ELIGIBLE[np.arange(D) % 4]
    baseline = _record(greedy, 1.0)
    scaled = _record(greedy, 0.5)
    # Pretend the scaled run differentiated a different subset.
    scaled.gradient_position_indices[:] = np.arange(D) + 100

    agreement = greedy_scale_comparison(baseline, scaled)["position_wise_greedy_agreement"]

    assert agreement["available"] is False
    assert "different evaluation positions" in agreement["reason"]


def test_a_historical_record_reports_the_baseline_scale() -> None:
    """Records written before the scale experiment were run at alpha = 1."""

    greedy = ELIGIBLE[np.arange(D) % 4]
    record = InitializationExperimentRecord.build(
        corpus_counts=np.full(VOCAB_SIZE, 5, dtype=np.int64) * (np.arange(VOCAB_SIZE) >= 2),
        selected_target_counts=np.bincount(greedy, minlength=VOCAB_SIZE),
        greedy_counts=np.bincount(greedy, minlength=VOCAB_SIZE)[None, :],
        nucleus_counts=np.bincount(greedy, minlength=VOCAB_SIZE)[None, None, :],
        mean_predicted_probabilities=np.full((1, VOCAB_SIZE), 1.0 / VOCAB_SIZE),
        model_seeds=[1000],
        eligible_token_ids=ELIGIBLE,
        metadata={"num_positions": D, "analysis": {"num_positions": D}},
    )

    assert record.initialization_scale == 1.0
    assert record.raw_logit_diagnostics == []
