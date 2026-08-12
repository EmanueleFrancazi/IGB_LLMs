"""Uniform categorical output null.

The ranked guess-frequency profile of a randomly initialized model looks
structured, but so does the ranked profile of *pure chance*: draw `D` tokens
uniformly from `K` and the sorted histogram still falls away steeply, purely
because finite sampling makes some tokens luckier than others. This module
supplies the reference that separates the two.

The null asks:

    How much of the ranked structure differs from what finite sampling alone
    would produce if the underlying selection distribution were exactly uniform?

It uses the same `D` and the same eligible support `K` as the model policies, so
the comparison is like-for-like.

**The draw is joint, not marginal.** Each realization is one
``Multinomial(D; 1/K, ..., 1/K)`` vector. Simulating `K` independent
``Binomial(D, 1/K)`` variables would reproduce the correct *marginal* count
distribution while getting the ranked histogram wrong: the counts are negatively
dependent and must sum to exactly `D`. Independent binomials neither sum to `D`
nor carry that dependence, and the resulting ranked profile is systematically
too spread out.

Uncertainty here is **Monte Carlo** uncertainty of the null itself. It is not a
model-initialization SEM, and the two must never be drawn or reported as the
same kind of quantity: the null is not a model and has no initializations.

NumPy only, like the rest of :mod:`llm_behavior_lab.analysis`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "DEFAULT_NULL_REPLICATES",
    "UniformNullSummary",
    "expected_occupancy_counts",
    "expected_zero_frequency_count",
    "expected_zero_frequency_fraction",
    "simulate_uniform_null",
]

#: Monte Carlo replicates for the null. Chosen against a 32k vocabulary: 256
#: realizations give a visibly smooth mean profile and a usable 95% envelope at
#: about 65 MB of transient storage and a second or so of runtime. Raising it
#: costs linearly in both and buys little, since the envelope is already narrow
#: next to the model-versus-null separation it exists to judge.
DEFAULT_NULL_REPLICATES = 256


@dataclass(frozen=True)
class UniformNullSummary:
    """Ranked profile and scalar diagnostics of the uniform categorical null.

    Attributes:
        eligible_vocab_size: ``K``, the support the null draws from.
        num_draws: ``D``, the number of draws per realization.
        num_replicates: ``M``, Monte Carlo realizations.
        seed: Seed that produced the realizations.
        interval: Central Monte Carlo interval width, e.g. ``0.95``.
        ranked_mean: ``[K]`` mean ranked frequency profile.
        ranked_low: ``[K]`` lower bound of the pointwise Monte Carlo interval.
        ranked_high: ``[K]`` upper bound.
        effective_support: ``[M]`` per-realization ``exp(H)``.
        zero_frequency_counts: ``[M]`` per-realization unreached-token counts.
        top_two_gaps: ``[M]`` per-realization ``q_(1) - q_(2)``.
    """

    eligible_vocab_size: int
    num_draws: int
    num_replicates: int
    seed: int
    interval: float
    ranked_mean: np.ndarray
    ranked_low: np.ndarray
    ranked_high: np.ndarray
    effective_support: np.ndarray
    zero_frequency_counts: np.ndarray
    top_two_gaps: np.ndarray

    @property
    def zero_frequency_fractions(self) -> np.ndarray:
        """Per-realization unreached fraction of the eligible support."""

        return self.zero_frequency_counts / self.eligible_vocab_size

    def _interval_of(self, values: np.ndarray) -> tuple[float, float]:
        tail = (1.0 - self.interval) / 2.0
        return (
            float(np.quantile(values, tail)),
            float(np.quantile(values, 1.0 - tail)),
        )

    def as_dict(self) -> dict[str, Any]:
        """JSON-serializable scalar summary.

        Every interval here is a **Monte Carlo** interval over null
        realizations, named so it cannot be mistaken for a SEM across model
        initializations.
        """

        effective_low, effective_high = self._interval_of(self.effective_support)
        zero_low, zero_high = self._interval_of(self.zero_frequency_counts)
        gap_low, gap_high = self._interval_of(self.top_two_gaps)
        return {
            "eligible_vocab_size": self.eligible_vocab_size,
            "num_draws": self.num_draws,
            "monte_carlo_replicates": self.num_replicates,
            "seed": self.seed,
            "monte_carlo_interval": self.interval,
            "effective_support_mean": float(self.effective_support.mean()),
            "effective_support_mc_low": effective_low,
            "effective_support_mc_high": effective_high,
            "zero_frequency_count_mean": float(self.zero_frequency_counts.mean()),
            "zero_frequency_count_mc_low": zero_low,
            "zero_frequency_count_mc_high": zero_high,
            "zero_frequency_fraction_mean": float(self.zero_frequency_fractions.mean()),
            "top_two_gap_mean": float(self.top_two_gaps.mean()),
            "top_two_gap_mc_low": gap_low,
            "top_two_gap_mc_high": gap_high,
            "analytic_zero_frequency_count": expected_zero_frequency_count(
                self.eligible_vocab_size, self.num_draws
            ),
            "analytic_zero_frequency_fraction": expected_zero_frequency_fraction(
                self.eligible_vocab_size, self.num_draws
            ),
        }


def expected_zero_frequency_count(eligible_vocab_size: int, num_draws: int) -> float:
    """``E[Z] = K (1 - 1/K)^D``: tokens expected to go unreached.

    Exact for a uniform categorical chooser. Used as a deterministic check on the
    Monte Carlo implementation, never as a substitute for it -- the analytic
    expectation says nothing about the *ranked* profile.
    """

    if eligible_vocab_size <= 0:
        raise ValueError("eligible_vocab_size must be positive.")
    if num_draws < 0:
        raise ValueError("num_draws must be non-negative.")
    return float(eligible_vocab_size * (1.0 - 1.0 / eligible_vocab_size) ** num_draws)


def expected_zero_frequency_fraction(eligible_vocab_size: int, num_draws: int) -> float:
    """``(1 - 1/K)^D``: the same quantity as a fraction of the support."""

    return expected_zero_frequency_count(eligible_vocab_size, num_draws) / eligible_vocab_size


def expected_occupancy_counts(eligible_vocab_size: int, num_draws: int, times: int) -> float:
    """``E[N_j] = K C(D, j) (1/K)^j (1 - 1/K)^(D-j)``.

    Expected number of tokens observed exactly ``times`` times. The ``j = 0`` case
    reduces to :func:`expected_zero_frequency_count`.
    """

    if eligible_vocab_size <= 0:
        raise ValueError("eligible_vocab_size must be positive.")
    if num_draws < 0:
        raise ValueError("num_draws must be non-negative.")
    if times < 0:
        raise ValueError("times must be non-negative.")
    if times > num_draws:
        return 0.0

    probability = 1.0 / eligible_vocab_size
    log_term = (
        math.lgamma(num_draws + 1)
        - math.lgamma(times + 1)
        - math.lgamma(num_draws - times + 1)
        + times * math.log(probability)
        + (num_draws - times) * math.log1p(-probability)
    )
    return float(eligible_vocab_size * math.exp(log_term))


def _entropy(values: np.ndarray) -> np.ndarray:
    """Row-wise Shannon entropy in nats, zeros contributing nothing."""

    safe = np.where(values > 0.0, values, 1.0)
    return -(np.where(values > 0.0, values, 0.0) * np.log(safe)).sum(axis=-1)


def simulate_uniform_null(
    *,
    eligible_vocab_size: int,
    num_draws: int,
    num_replicates: int = DEFAULT_NULL_REPLICATES,
    seed: int = 20260812,
    interval: float = 0.95,
) -> UniformNullSummary:
    """Simulate the ranked profile of uniform categorical selection.

    Each replicate is one **joint** multinomial draw over the eligible support,
    converted to frequencies and sorted descending. Realizations are accumulated
    so the mean profile and a pointwise Monte Carlo interval can be reported.

    Args:
        eligible_vocab_size: ``K``, matching the model's predictive support.
        num_draws: ``D``, matching the model's evaluation-position count.
        num_replicates: ``M``, Monte Carlo realizations. Not ``R``: nucleus
            replicates are a different quantity entirely.
        seed: Dedicated null seed, independent of every other stream.
        interval: Central Monte Carlo interval width.

    Returns:
        A :class:`UniformNullSummary`.
    """

    if eligible_vocab_size <= 0:
        raise ValueError("eligible_vocab_size must be positive.")
    if num_draws <= 0:
        raise ValueError("num_draws must be positive.")
    if num_replicates <= 0:
        raise ValueError("num_replicates must be positive.")
    if not 0.0 < interval < 1.0:
        raise ValueError("interval must lie strictly between 0 and 1.")

    generator = np.random.default_rng(seed)
    probabilities = np.full(eligible_vocab_size, 1.0 / eligible_vocab_size)

    ranked = np.empty((num_replicates, eligible_vocab_size), dtype=np.float64)
    zero_counts = np.empty(num_replicates, dtype=np.float64)
    for replicate in range(num_replicates):
        # One joint draw: the counts are dependent and sum to exactly D.
        counts = generator.multinomial(num_draws, probabilities)
        zero_counts[replicate] = float((counts == 0).sum())
        ranked[replicate] = np.sort(counts / num_draws)[::-1]

    tail = (1.0 - interval) / 2.0
    return UniformNullSummary(
        eligible_vocab_size=eligible_vocab_size,
        num_draws=num_draws,
        num_replicates=num_replicates,
        seed=seed,
        interval=interval,
        ranked_mean=ranked.mean(axis=0),
        ranked_low=np.quantile(ranked, tail, axis=0),
        ranked_high=np.quantile(ranked, 1.0 - tail, axis=0),
        effective_support=np.exp(_entropy(ranked)),
        zero_frequency_counts=zero_counts,
        top_two_gaps=ranked[:, 0] - ranked[:, 1] if eligible_vocab_size > 1 else np.zeros(num_replicates),
    )
