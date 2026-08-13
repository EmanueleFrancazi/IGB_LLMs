"""Temperature-transition analysis for the nucleus sweep.

Greedy is the ``T = 0`` anchor and the finite-``D`` uniform categorical null is
the high-temperature reference. This module measures where a sweep temperature
sits between them.

Three diagnostics, each answering a different question:

* :func:`ranked_distance_to_greedy` and :func:`ranked_distance_to_uniform`
  compare **concentration shape**. Both distributions are ranked independently
  first, so token identity is deliberately discarded and rank ``r`` of one curve
  is generally a different token from rank ``r`` of the other.
* greedy **agreement** compares token identity directly: the fraction of
  evaluated positions where the sampled token equals the argmax token from the
  same logits. It is accumulated during streaming and arrives here already
  computed.

Nothing here should be read as evidence of a sharp critical temperature. The
transition is described as whatever the measurements show; monotonicity in
particular is an empirical question, not a property finite stochastic samples
under top-p truncation are guaranteed to have.

The uniform null is a **finite-D ranked occupancy reference**, not a latent model
distribution. Normalizing by it says "how concentrated is this next to what pure
chance would produce at the same draw count", nothing more.

NumPy only.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from llm_behavior_lab.analysis.aggregation import (
    _mean_sem,
    effective_support,
    eligible_view,
    ranked_profile,
    shannon_entropy,
    top_two_gap,
    total_variation_distance,
)

__all__ = [
    "TRANSITION_METRICS",
    "ranked_distance_to_greedy",
    "ranked_distance_to_uniform",
    "sweep_condition_summary",
    "sweep_summary",
    "sweep_temperature_profiles",
]

#: Reported per temperature, per condition. Named so a ranked-profile distance
#: can never be mistaken for a same-token one.
TRANSITION_METRICS = (
    "effective_support",
    "effective_support_over_null",
    "entropy",
    "zero_frequency_count",
    "zero_frequency_fraction",
    "top_two_gap",
    "tv_to_corpus",
    "tv_rank_to_greedy",
    "tv_rank_to_uniform",
    "agreement_with_greedy",
)


def ranked_distance_to_greedy(guess: np.ndarray, greedy: np.ndarray) -> float:
    """Total variation between the two **independently ranked** profiles.

    A pure concentration-shape comparison against the ``T = 0`` anchor. Not a
    same-token distance: ranking first discards which tokens carry the mass.
    """

    return total_variation_distance(ranked_profile(guess), ranked_profile(greedy))


def ranked_distance_to_uniform(guess: np.ndarray, uniform_ranked_mean: np.ndarray) -> float:
    """Total variation between the ranked guess profile and the ranked null.

    The null profile is already ranked and already averaged over Monte Carlo
    realizations. As above, this compares shape only.
    """

    ranked = ranked_profile(guess)
    reference = np.asarray(uniform_ranked_mean, dtype=np.float64)
    if ranked.shape != reference.shape:
        raise ValueError(
            f"ranked guess profile has length {ranked.shape[0]} but the null profile has "
            f"{reference.shape[0]}; both must span the eligible support."
        )
    return total_variation_distance(ranked, reference)


def sweep_temperature_profiles(record: Any, condition: str = "real") -> np.ndarray:
    """Return ``[S, T, K]`` sweep guess fractions on the eligible support."""

    counts = record.sweep_counts(condition)
    totals = counts.sum(axis=-1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("A sweep temperature produced no guesses at all.")
    return eligible_view(record, counts.astype(np.float64) / totals)


def sweep_condition_summary(record: Any, condition: str = "real") -> dict[str, Any]:
    """Per-temperature transition summary for one input condition.

    Every scalar is computed per initialization and only then averaged, so each
    reported error is the spread across initializations -- the same convention as
    everywhere else in the analysis.
    """

    if not record.has_temperature_sweep:
        return {"available": False, "condition": condition, "temperatures": [], "metrics": {}}

    temperatures = list(record.sweep_temperatures)
    profiles = sweep_temperature_profiles(record, condition)
    corpus = eligible_view(record, record.corpus_fractions)
    greedy = eligible_view(record, record.condition_policy_fractions(condition, "greedy"))
    agreement = record.sweep_agreement(condition)

    # The null profile is simulated over K and therefore already lives on the
    # eligible support; restricting it again would drop its last K-|special|
    # ranks and silently shorten the comparison.
    null_ranked = record.uniform_null["ranked_mean"] if record.has_uniform_null else None
    null_effective = effective_support(null_ranked) if null_ranked is not None else None

    metrics: dict[str, Any] = {name: {"mean": [], "sem": []} for name in TRANSITION_METRICS}
    for index, _temperature in enumerate(temperatures):
        per_initialization = {name: [] for name in TRANSITION_METRICS}
        for initialization in range(profiles.shape[0]):
            row = profiles[initialization, index]
            support = effective_support(row)
            per_initialization["effective_support"].append(support)
            per_initialization["effective_support_over_null"].append(
                support / null_effective if null_effective else float("nan")
            )
            per_initialization["entropy"].append(shannon_entropy(row))
            zeros = float((row <= 0.0).sum())
            per_initialization["zero_frequency_count"].append(zeros)
            per_initialization["zero_frequency_fraction"].append(zeros / row.shape[0])
            per_initialization["top_two_gap"].append(top_two_gap(row))
            per_initialization["tv_to_corpus"].append(total_variation_distance(corpus, row))
            per_initialization["tv_rank_to_greedy"].append(
                ranked_distance_to_greedy(row, greedy[initialization])
            )
            per_initialization["tv_rank_to_uniform"].append(
                ranked_distance_to_uniform(row, null_ranked)
                if null_ranked is not None
                else float("nan")
            )
            per_initialization["agreement_with_greedy"].append(
                float(agreement[initialization, index]) if agreement is not None else float("nan")
            )
        for name in TRANSITION_METRICS:
            mean, sem = _mean_sem(np.array(per_initialization[name], dtype=np.float64))
            metrics[name]["mean"].append(mean)
            metrics[name]["sem"].append(sem)

    return {
        "available": True,
        "condition": condition,
        "temperatures": temperatures,
        "num_initializations": int(profiles.shape[0]),
        "metrics": metrics,
    }


def sweep_summary(record: Any, conditions: Sequence[str] | None = None) -> dict[str, Any]:
    """Transition summaries for every available input condition."""

    if not record.has_temperature_sweep:
        return {"available": False, "conditions": {}}

    names = list(conditions) if conditions is not None else list(record.sweep_conditions)
    return {
        "available": True,
        "canonical_temperature": record.metadata.get("analysis", {})
        .get("sampling", {})
        .get("temperature"),
        "top_p": record.metadata.get("analysis", {}).get("sampling", {}).get("top_p"),
        "temperatures": list(record.sweep_temperatures),
        "conditions": {name: sweep_condition_summary(record, name) for name in names},
    }
