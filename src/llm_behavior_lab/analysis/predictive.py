"""Statistics of the raw predictive distribution, before any sampling policy.

At every evaluation position the model emits logits over the vocabulary. Restrict
them to the eligible predictive support, apply softmax at temperature 1, and the
result is the probability vector this module summarizes:

.. code-block:: text

    p[d, i] = softmax(z[d] restricted to the eligible support)[i]

That vector is **upstream of every sampling decision**. It carries no temperature
scaling, no top-p truncation, and no greedy or nucleus choice. It must never be
confused with the nucleus distribution, which is the same logits at ``T = 0.6``
after top-p truncation and renormalization.

Four quantities are read from the record, and the distinction between the first
and what figure 1 already shows is the point of the whole analysis:

``ranked profile`` ``Pbar_s(r)``
    Probabilities are ranked **within each position first**, and only then
    averaged at equal rank across positions. This answers "how concentrated is
    one individual next-token prediction". Figure 1 instead ranks *token guess
    frequencies accumulated across positions*, which answers "how concentrated is
    the aggregate distribution of guesses". A model could be flat at every single
    position and still produce a sharply peaked aggregate, or the reverse; the two
    are independent questions and neither implies the other.

``p_max[d]``
    The probability of the token greedy decoding selects. Greedy always takes the
    top-ranked token, but the top-ranked token need not carry much probability --
    which is exactly what this measures.

``p_target[d]``
    The probability assigned to the true next token, which is a different
    question from ``p_max``: one is confidence in the model's preference, the
    other is mass on the correct answer.

``loss[d]``
    ``-log p_target[d]``, the single-position cross-entropy.

Like the rest of :mod:`llm_behavior_lab.analysis`, this needs only NumPy.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "PROBABILITY_QUANTILES",
    "REPORTED_RANKS",
    "max_probability_summary",
    "predictive_probability_summary",
    "ranked_probability_profile",
    "target_probability_summary",
]

#: Quantiles reported for ``p_max``, chosen so both tails stay visible.
PROBABILITY_QUANTILES = (0, 5, 25, 50, 75, 95, 99, 100)

#: Ranks reported numerically. Logarithmically spaced, because a profile over
#: tens of thousands of ranks is uninformative at evenly spaced ones.
REPORTED_RANKS = (1, 2, 3, 5, 10, 100, 1000, 10000)


def _require_probabilities(record: Any) -> None:
    if not record.has_predictive_probability_analysis:
        raise ValueError(
            "This record carries no raw predictive-probability analysis. It was "
            "produced before the diagnostic existed, or with it disabled; the "
            "experiment must be rerun to obtain it."
        )


def ranked_probability_profile(record: Any) -> dict[str, np.ndarray]:
    """Mean and spread of the rank-first ranked profile across initializations.

    Each initialization already carries its own ``[K]`` profile, ranked within
    position and averaged over positions. Averaging *those* across
    initializations keeps the initialization as the independent unit, exactly as
    every other cross-initialization statistic in this package does.

    Returns:
        ``mean``, ``sem``, ``low`` and ``high`` -- each ``[K]`` -- plus ``ranks``
        starting at 1, and the per-initialization ``profiles``.
    """

    _require_probabilities(record)
    profiles = np.asarray(record.predictive_ranked_probabilities, dtype=np.float64)
    count = profiles.shape[0]
    mean = profiles.mean(axis=0)
    if count > 1:
        sem = profiles.std(axis=0, ddof=1) / np.sqrt(count)
    else:
        sem = np.zeros_like(mean)
    return {
        "ranks": np.arange(1, profiles.shape[1] + 1),
        "profiles": profiles,
        "mean": mean,
        "sem": sem,
        "low": profiles.min(axis=0),
        "high": profiles.max(axis=0),
    }


def _quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {"count": 0}
    summary: dict[str, float] = {
        "count": int(values.size),
        "mean": float(values.mean()),
    }
    for percentile, value in zip(
        PROBABILITY_QUANTILES, np.percentile(values, PROBABILITY_QUANTILES)
    ):
        summary[f"p{percentile:02d}"] = float(value)
    return summary


def max_probability_summary(record: Any) -> dict[str, Any]:
    """Describe ``p_max`` pooled and per initialization.

    Both views are reported because they answer different questions and the
    pooled one alone would be misleading: the ``I * D`` values are not one
    independent sample, since all positions within an initialization share a set
    of weights.
    """

    _require_probabilities(record)
    values = np.asarray(record.predictive_max_probabilities, dtype=np.float64)
    uniform = record.uniform_probability
    pooled = _quantiles(values)
    return {
        "pooled": pooled,
        "per_initialization_mean": values.mean(axis=1),
        "per_initialization_median": np.median(values, axis=1),
        "uniform_probability": uniform,
        "median_over_uniform": float(pooled["p50"] / uniform),
        "mean_over_uniform": float(pooled["mean"] / uniform),
    }


def target_probability_summary(record: Any) -> dict[str, Any]:
    """Describe ``p_target`` and the single-position loss derived from it.

    Kept deliberately minimal: these are persisted now because the probability
    vectors were already in hand, and their scientific reading belongs with the
    gradient analysis rather than here.
    """

    _require_probabilities(record)
    probabilities = np.asarray(record.predictive_target_probabilities, dtype=np.float64)
    losses = np.asarray(record.predictive_target_losses, dtype=np.float64)
    uniform = record.uniform_probability
    return {
        "probability": _quantiles(probabilities),
        "loss": _quantiles(losses),
        "uniform_probability": uniform,
        "uniform_loss": float(-np.log(uniform)),
        "mean_over_uniform": float(probabilities.mean() / uniform),
    }


def predictive_probability_summary(record: Any) -> dict[str, Any]:
    """Everything needed to read figures 8 and 9, and to choose their scales."""

    _require_probabilities(record)
    profile = ranked_probability_profile(record)
    mean_profile = profile["mean"]
    eligible = int(record.eligible_vocab_size)
    uniform = record.uniform_probability

    reported = {}
    for rank in REPORTED_RANKS:
        if rank <= mean_profile.shape[0]:
            reported[str(rank)] = float(mean_profile[rank - 1])

    positive = mean_profile[mean_profile > 0]
    return {
        "num_initializations": int(profile["profiles"].shape[0]),
        "num_positions": int(record.predictive_max_probabilities.shape[1]),
        "eligible_vocab_size": eligible,
        "uniform_probability": uniform,
        "protocol": record.predictive_probability_protocol,
        "ranked_profile_at_rank": reported,
        "rank1_over_uniform": float(mean_profile[0] / uniform),
        "profile_dynamic_range": (
            float(positive.max() / positive.min()) if positive.size else float("nan")
        ),
        "profile_min_positive": float(positive.min()) if positive.size else float("nan"),
        "num_zero_ranks": int((mean_profile <= 0).sum()),
        "max_probability": max_probability_summary(record),
        "target_probability": target_probability_summary(record),
    }
