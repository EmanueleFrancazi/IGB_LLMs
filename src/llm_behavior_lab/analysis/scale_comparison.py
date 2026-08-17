"""Compare finished records across initialization scale.

Read-only: it takes two already-written records and reports how the greedy
behaviour of a scaled condition differs from the ``alpha = 1`` baseline. No model
is built, no forward or backward pass is run, and no new scientific definition is
introduced -- every measure here already exists in
:mod:`llm_behavior_lab.analysis.aggregation`.

**Two different questions, deliberately kept apart.**

*Position-wise greedy agreement* asks how often the two conditions pick the same
token at the same evaluation position:

.. code-block:: text

    agreement = mean_d [ greedy_alpha(d) == greedy_1(d) ]

*Distributional total variation* asks how far apart the two aggregate guess
distributions are:

.. code-block:: text

    TV(q_alpha, q_1) = 0.5 * sum_i |q_alpha(i) - q_1(i)|

They are not interchangeable, and conflating them would hide the most
interesting case. Two conditions can produce nearly identical marginal
distributions -- small TV -- while disagreeing at a large fraction of individual
positions, which is what a reorganization of output preferences looks like when
the *shape* of the guess distribution happens to be preserved.

This bears directly on the interpretive control for the scale experiment. If
lowering ``alpha`` merely rescaled all logits by a common positive factor, greedy
identities would be untouched and agreement would be exactly 1. Anything less
means the intervention changed the logit geometry itself.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from llm_behavior_lab.analysis.aggregation import (
    effective_supports,
    eligible_view,
    policy_zero_frequency,
    total_variation_distance,
)

__all__ = ["greedy_scale_comparison"]


def _position_agreement(baseline: Any, scaled: Any) -> dict[str, Any]:
    """Position-wise greedy agreement, when both records carry per-position IDs.

    The per-position greedy tokens are only persisted by the gradient analysis,
    so this is available exactly when both records ran it over the same
    positions. When they did not, the fact is reported rather than approximated
    from the marginal counts, which cannot recover it.
    """

    if not (baseline.has_position_gradients and scaled.has_position_gradients):
        return {
            "available": False,
            "reason": (
                "Per-position greedy identities are persisted by the gradient "
                "analysis; at least one record does not carry it."
            ),
        }

    baseline_positions = np.asarray(baseline.gradient_position_indices)
    scaled_positions = np.asarray(scaled.gradient_position_indices)
    if not np.array_equal(baseline_positions, scaled_positions):
        return {
            "available": False,
            "reason": (
                "The two records differentiated different evaluation positions, "
                "so a position-wise comparison would not be paired."
            ),
        }

    baseline_greedy = np.asarray(baseline.gradient_position_greedy_ids)
    scaled_greedy = np.asarray(scaled.gradient_position_greedy_ids)
    matches = baseline_greedy == scaled_greedy
    return {
        "available": True,
        "num_positions": int(matches.shape[0]),
        "agreement": float(matches.mean()),
        "num_changed": int((~matches).sum()),
    }


def greedy_scale_comparison(baseline: Any, scaled: Any, *, policy: str = "greedy") -> dict[str, Any]:
    """Compare one scaled record against the ``alpha = 1`` baseline.

    Args:
        baseline: The record produced at ``alpha = 1``.
        scaled: The record produced at a reduced scale.
        policy: Guess policy to compare. Greedy is the one the scale experiment
            is about; nucleus is accepted for completeness.

    Returns:
        A JSON-serializable comparison holding the position-wise agreement and
        the distributional measures separately.
    """

    baseline_fractions = eligible_view(baseline, baseline.policy_fractions(policy))
    scaled_fractions = eligible_view(scaled, scaled.policy_fractions(policy))
    if baseline_fractions.shape != scaled_fractions.shape:
        raise ValueError(
            "The two records must cover the same initializations and eligible "
            f"support; got {baseline_fractions.shape} and {scaled_fractions.shape}."
        )

    # Paired per initialization: row s of one condition against row s of the
    # other, which is the whole point of sharing the draw.
    per_initialization = [
        total_variation_distance(baseline_fractions[index], scaled_fractions[index])
        for index in range(baseline_fractions.shape[0])
    ]

    baseline_support = effective_supports(baseline, policy)
    scaled_support = effective_supports(scaled, policy)
    baseline_zero = policy_zero_frequency(baseline, policy)
    scaled_zero = policy_zero_frequency(scaled, policy)

    return {
        "policy": policy,
        "baseline_alpha": baseline.initialization_scale,
        "scaled_alpha": scaled.initialization_scale,
        # Distinct questions, reported separately and never merged.
        "position_wise_greedy_agreement": _position_agreement(baseline, scaled),
        "distribution_total_variation": {
            "per_initialization": np.asarray(per_initialization),
            "mean": float(np.mean(per_initialization)),
            "note": (
                "Distance between aggregate guess distributions. Small TV with low "
                "position-wise agreement means the shape is preserved while "
                "individual choices moved."
            ),
        },
        "effective_support": {
            "baseline_mean": float(baseline_support.mean()),
            "scaled_mean": float(scaled_support.mean()),
            "delta_mean": float(scaled_support.mean() - baseline_support.mean()),
        },
        "zero_frequency_fraction": {
            "baseline_mean": float(baseline_zero.fractions.mean()),
            "scaled_mean": float(scaled_zero.fractions.mean()),
            "delta_mean": float(scaled_zero.fractions.mean() - baseline_zero.fractions.mean()),
            "draws_per_measurement": int(baseline_zero.draws_per_measurement),
        },
    }
