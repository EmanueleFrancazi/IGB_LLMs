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
    "CUMULATIVE_DEPTHS",
    "cumulative_order_comparison",
    "ranked_mean_token_probabilities",
    "TEMPERATURE_RANKS",
    "TOP_K_MASSES",
    "temperature_confidence_summary",
    "temperature_ranked_profiles",
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


#: Ranks reported in the temperature comparison, and the top-k masses.
TEMPERATURE_RANKS = (1, 2, 5, 10, 100)
TOP_K_MASSES = (1, 10, 100)


def _require_temperatures(record: Any) -> None:
    if not record.has_temperature_confidence_analysis:
        raise ValueError(
            "This record carries no temperature-conditioned confidence analysis. "
            "It predates the diagnostic; the experiment must be rerun to obtain it."
        )


def temperature_ranked_profiles(record: Any) -> dict[str, np.ndarray]:
    """Mean and spread of the ranked profile at every diagnostic temperature.

    Returns ``[N_T, K]`` arrays: the mean across initializations, its SEM, and the
    min/max envelope, plus the temperature grid and the rank axis.
    """

    _require_temperatures(record)
    profiles = np.asarray(
        record.predictive_temperature_ranked_probabilities, dtype=np.float64
    )
    count = profiles.shape[0]
    mean = profiles.mean(axis=0)
    sem = (
        profiles.std(axis=0, ddof=1) / np.sqrt(count)
        if count > 1
        else np.zeros_like(mean)
    )
    return {
        "temperatures": np.asarray(record.confidence_temperatures, dtype=np.float64),
        "ranks": np.arange(1, profiles.shape[2] + 1),
        "profiles": profiles,
        "mean": mean,
        "sem": sem,
        "low": profiles.min(axis=0),
        "high": profiles.max(axis=0),
    }


def temperature_confidence_summary(record: Any) -> dict[str, Any]:
    """Per-temperature confidence statistics for the fixed greedy decisions.

    Every row describes the *same* greedy predictions: softmax is strictly
    increasing, so ``argmax softmax(z/T) = argmax z`` for every positive ``T``.
    Only the confidence attached to those decisions moves with temperature. This
    is what distinguishes the diagnostic from the nucleus sweep, which samples
    from the transformed distribution and therefore does change what is selected.
    """

    _require_temperatures(record)
    temperatures = np.asarray(record.confidence_temperatures, dtype=np.float64)
    maxima = np.asarray(record.predictive_temperature_max_probabilities, dtype=np.float64)
    targets = np.asarray(
        record.predictive_temperature_target_probabilities, dtype=np.float64
    )
    losses = np.asarray(record.predictive_temperature_target_losses, dtype=np.float64)
    entropy = np.asarray(record.predictive_temperature_mean_entropy, dtype=np.float64)
    profiles = temperature_ranked_profiles(record)
    uniform = record.uniform_probability

    rows = []
    for index, temperature in enumerate(temperatures):
        pooled = _quantiles(maxima[:, index, :])
        mean_profile = profiles["mean"][index]
        cumulative = np.cumsum(mean_profile)
        mean_entropy = float(entropy[:, index].mean())
        rows.append(
            {
                "temperature": float(temperature),
                "is_canonical": bool(temperature == 1.0),
                "max_probability": pooled,
                "mean_over_uniform": float(pooled["mean"] / uniform),
                "median_over_uniform": float(pooled["p50"] / uniform),
                "ranked_profile_at_rank": {
                    str(rank): float(mean_profile[rank - 1])
                    for rank in TEMPERATURE_RANKS
                    if rank <= mean_profile.shape[0]
                },
                "top_k_mass": {
                    str(k): float(cumulative[k - 1])
                    for k in TOP_K_MASSES
                    if k <= cumulative.shape[0]
                },
                "mean_predictive_entropy": mean_entropy,
                # exp of the mean entropy: the entropy-equivalent number of
                # equally likely tokens in a typical single prediction, matching
                # the project's effective-support convention.
                "effective_support": float(np.exp(mean_entropy)),
                "target_probability": _quantiles(targets[:, index, :]),
                "target_loss": _quantiles(losses[:, index, :]),
                "per_initialization_median_max": np.median(maxima[:, index, :], axis=1),
            }
        )

    return {
        "temperatures": temperatures,
        "uniform_probability": uniform,
        "eligible_vocab_size": int(record.eligible_vocab_size),
        "num_positions": int(maxima.shape[2]),
        "num_initializations": int(maxima.shape[0]),
        "greedy_identity_is_temperature_invariant": True,
        "rows": rows,
    }


#: Cumulative depths compared between the two orders of operations.
CUMULATIVE_DEPTHS = (1, 10, 100, 1000)


def _require_mean_tokens(record: Any) -> None:
    if not record.has_mean_token_probabilities:
        raise ValueError(
            "This record carries no identity-preserving mean token probabilities. "
            "It predates the diagnostic; the experiment must be rerun to obtain it."
        )


def ranked_mean_token_probabilities(record: Any) -> dict[str, np.ndarray]:
    """Figure 14: average at fixed token identity **first**, then rank.

    For every initialization independently:

    .. code-block:: text

        pbar_{s,T}(i) = mean_d p_{s,T}(d, i)        identity preserved
        B_{s,T}(r)    = sort_descending_i pbar_{s,T}(i)

    Ranking happens per initialization, before initializations are summarized.
    Averaging probabilities across initializations first and ranking afterwards
    would answer a different question, since different initializations prefer
    different tokens and averaging would wash that out.

    This is the mirror image of :func:`temperature_ranked_profiles`, which ranks
    inside each *position* before averaging. That one asks how concentrated a
    typical single prediction is; this one asks whether the **same** token
    identities are systematically favoured across many different inputs.
    """

    _require_mean_tokens(record)
    values = np.asarray(
        record.predictive_temperature_mean_token_probabilities, dtype=np.float64
    )
    eligible = np.asarray(record.eligible_token_ids, dtype=np.int64)
    # Ranked over the eligible support, matching every other ranked profile.
    restricted = values[:, :, eligible]
    profiles = np.sort(restricted, axis=2)[:, :, ::-1]

    count = profiles.shape[0]
    mean = profiles.mean(axis=0)
    sem = (
        profiles.std(axis=0, ddof=1) / np.sqrt(count)
        if count > 1
        else np.zeros_like(mean)
    )
    return {
        "temperatures": np.asarray(record.confidence_temperatures, dtype=np.float64),
        "ranks": np.arange(1, profiles.shape[2] + 1),
        "profiles": profiles,
        "mean": mean,
        "sem": sem,
        "low": profiles.min(axis=0),
        "high": profiles.max(axis=0),
        "mean_token_probabilities": restricted,
    }


def cumulative_order_comparison(record: Any) -> dict[str, Any]:
    """Top-k mass under both orders of operations, and their difference.

    ``A_T(r)`` ranks within each position and then averages; ``B_T(r)`` averages
    at fixed identity and then ranks. For each depth ``k``:

    .. code-block:: text

        C_A(T, k) = sum_{r <= k} A_T(r)
        C_B(T, k) = sum_{r <= k} B_T(r)
        Delta     = C_A - C_B

    ``Delta >= 0`` is expected on general grounds: letting the top-k identities
    vary with the position cannot capture less mass than committing to one fixed
    set of k tokens chosen after identity-preserving averaging. It is reported,
    not asserted as a headline, and the sign is checked on controlled fixtures.
    """

    _require_mean_tokens(record)
    within = temperature_ranked_profiles(record)["mean"]
    identity = ranked_mean_token_probabilities(record)["mean"]
    temperatures = np.asarray(record.confidence_temperatures, dtype=np.float64)

    rows = []
    for index, temperature in enumerate(temperatures):
        within_cumulative = np.cumsum(within[index])
        identity_cumulative = np.cumsum(identity[index])
        depths = {}
        for depth in CUMULATIVE_DEPTHS:
            if depth > within_cumulative.shape[0]:
                continue
            rank_first = float(within_cumulative[depth - 1])
            identity_first = float(identity_cumulative[depth - 1])
            depths[str(depth)] = {
                "rank_then_average": rank_first,
                "average_then_rank": identity_first,
                "delta": rank_first - identity_first,
            }
        rows.append(
            {
                "temperature": float(temperature),
                "is_canonical": float(temperature) == 1.0,
                "depths": depths,
            }
        )
    return {"temperatures": temperatures, "depths": CUMULATIVE_DEPTHS, "rows": rows}
