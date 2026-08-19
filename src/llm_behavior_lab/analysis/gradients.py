"""Token-level aggregation of per-position parameter-gradient norms.

The record stores one exact gradient norm per evaluation position. This module
turns those into the per-token quantities the eventual figure 10 relates, and it
is the only place that aggregation is defined.

Let ``S`` be the set of positions the gradient analysis actually covered -- every
evaluation position unless a run deliberately used a window subset. Then, for
token ID ``i``:

.. code-block:: text

    n_i     = |{d in S : y_d = i}|                     target occurrences
    G_i     = mean_{d in S : y_d = i} g_d              mean gradient norm
    q_i     = |{d in S : greedy_d = i}| / |S|          greedy guess fraction
    p_i     = corpus fraction over the whole analysis split

The asymmetry between ``q_i`` and ``p_i`` is deliberate and is the one thing to
keep straight here.

``q_i`` is computed from the greedy predictions stored **at the same positions**
as the norms, not from the record's full-run ``greedy_counts``. When the analysis
covers every position the two agree exactly, and a test asserts that. When it
covers a subset they do not, and only the position-aligned version is a valid
partner for ``G_i``: both must describe the same ``S``.

``p_i`` is the opposite case. It intentionally refers to the complete analysis
split, because it is the empirical reference the whole experiment compares
against, not a property of the positions that happened to be differentiated.

Like the rest of :mod:`llm_behavior_lab.analysis`, this depends only on NumPy, so
a finished experiment is re-analyzable without PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "GRADIENT_QUANTILES",
    "OCCURRENCE_STRATA",
    "TokenGradientSummary",
    "gradient_guess_correlations",
    "gradient_guess_table",
    "gradient_observable_summary",
    "mean_probability_gradient_table",
    "nucleus_gradient_table",
    "temperature_gradient_summary",
    "temperature_gradient_table",
    "spearman_rho",
    "token_gradient_norms",
]

#: Percentiles reported for every distribution, so a scale choice is made from
#: the data rather than from an assumption about it.
GRADIENT_QUANTILES = (0, 1, 5, 25, 50, 75, 95, 99, 100)

#: Inclusive ``n_i`` bands for the precision diagnostic. ``G_i`` is a mean over
#: ``n_i`` positions, so a token seen once is far noisier than one seen a hundred
#: times; these bands make that visible without changing the primary statistic.
OCCURRENCE_STRATA = ((1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 64), (65, None))


@dataclass(frozen=True)
class TokenGradientSummary:
    """Per-token gradient aggregation over the gradient-evaluated positions.

    Both vectors are ``[V]`` and indexed by token ID, aligning with every other
    per-token vector in the record without further bookkeeping.
    """

    #: ``G_i``. NaN where the token never appears as a target in ``S``: no
    #: gradient was measured for it, which is different from a measured zero.
    mean_gradient_norm: np.ndarray
    #: ``n_i``, the number of positions in ``S`` whose target was this token.
    target_count: np.ndarray
    #: ``|S|``.
    num_positions: int
    #: Whether ``S`` is the experiment's whole evaluation-position set.
    covers_all_positions: bool
    #: Which initialization the gradients were measured on.
    initialization_index: int

    @property
    def measured_token_count(self) -> int:
        """Number of tokens that occur as a target at least once in ``S``."""

        return int((self.target_count > 0).sum())


def _require_gradients(record: Any) -> None:
    if not record.has_position_gradients:
        raise ValueError(
            "This record carries no per-position gradient analysis. Rerun the "
            "experiment with gradient analysis enabled."
        )


def _covers_all_positions(record: Any) -> bool:
    """Whether the analysis covered every evaluation position.

    Taken from the recorded protocol when present, and otherwise inferred by
    comparing the position count with the experiment's own, so a record written
    before the flag existed still reports the truth rather than a default.
    """

    recorded = record.gradient_analysis.get("covers_all_positions")
    if recorded is not None:
        return bool(recorded)
    total = record.metadata.get("analysis", {}).get("num_positions")
    if total is None:
        total = record.metadata.get("num_positions")
    if total is None:
        raise ValueError(
            "The record does not say how many evaluation positions the experiment "
            "defined, so gradient coverage cannot be established."
        )
    return int(record.gradient_position_norms.shape[0]) == int(total)


def token_gradient_norms(record: Any) -> TokenGradientSummary:
    """Aggregate per-position gradient norms into ``G_i`` and ``n_i``.

    Args:
        record: A record carrying per-position gradient arrays.

    Returns:
        The per-token summary, with ``G_i`` NaN wherever ``n_i`` is zero.

    Raises:
        ValueError: If the record carries no gradient analysis.
    """

    _require_gradients(record)
    vocab_size = record.vocab_size
    targets = np.asarray(record.gradient_position_target_ids, dtype=np.int64)
    norms = np.asarray(record.gradient_position_norms, dtype=np.float64)

    counts = np.bincount(targets, minlength=vocab_size).astype(np.int64)
    sums = np.bincount(targets, weights=norms, minlength=vocab_size)

    means = np.full(vocab_size, np.nan, dtype=np.float64)
    measured = counts > 0
    means[measured] = sums[measured] / counts[measured]

    return TokenGradientSummary(
        mean_gradient_norm=means,
        target_count=counts,
        num_positions=int(norms.shape[0]),
        covers_all_positions=_covers_all_positions(record),
        initialization_index=record.gradient_initialization_index,
    )


def gradient_guess_table(record: Any) -> dict[str, Any]:
    """Assemble the aligned per-token table relating ``G_i``, ``q_i`` and ``p_i``.

    Every column is ``[V]`` and indexed by token ID.

    ``greedy_guess_fraction`` is derived from the greedy predictions stored at
    the gradient-evaluated positions, so it and ``mean_gradient_norm`` describe
    the same position set. It is deliberately *not* ``record.greedy_fractions``,
    which always covers every position and would silently mismatch ``G_i`` for a
    subset run.

    Args:
        record: A record carrying per-position gradient arrays.

    Returns:
        A dictionary of aligned columns plus the provenance a reader needs to
        avoid mistaking a subset table for a whole-experiment one.
    """

    summary = token_gradient_norms(record)
    vocab_size = record.vocab_size
    greedy = np.asarray(record.gradient_position_greedy_ids, dtype=np.int64)

    guess_counts = np.bincount(greedy, minlength=vocab_size).astype(np.int64)
    guess_fractions = guess_counts.astype(np.float64) / float(summary.num_positions)

    return {
        "token_id": np.arange(vocab_size),
        "target_occurrence_count": summary.target_count,
        "mean_gradient_norm": summary.mean_gradient_norm,
        "greedy_guess_count": guess_counts,
        "greedy_guess_fraction": guess_fractions,
        "corpus_fraction": record.corpus_fractions,
        "eligible_mask": record.eligible_mask,
        "num_positions": summary.num_positions,
        "covers_all_positions": summary.covers_all_positions,
        "initialization_index": summary.initialization_index,
        "greedy_source": "gradient_evaluated_positions",
        "corpus_source": "whole_analysis_split",
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Rank values, giving tied entries their shared average rank.

    Average ranks are not a refinement here, they are the whole correctness
    question. Most tokens are never greedily guessed, so ``q_i`` carries one
    enormous tie block at zero; ordinal ranking would invent an arbitrary order
    inside it and report a correlation that is partly an artefact of the sort.
    """

    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    ranks = np.empty(values.shape[0], dtype=np.float64)

    start = 0
    while start < ordered.shape[0]:
        stop = start
        while stop + 1 < ordered.shape[0] and ordered[stop + 1] == ordered[start]:
            stop += 1
        ranks[order[start : stop + 1]] = 0.5 * (start + stop) + 1.0
        start = stop + 1
    return ranks


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation, tie-corrected, in NumPy alone.

    Written out rather than taken from SciPy because the analysis layer's only
    dependency is NumPy: a finished experiment must stay re-analyzable without
    installing anything further. It is the Pearson correlation of average ranks,
    which is the definition SciPy implements for tied data.

    Returns:
        ``rho`` in ``[-1, 1]``, or NaN when fewer than two points are supplied or
        either input is constant, in which case the coefficient is undefined
        rather than zero.
    """

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(
            f"spearman_rho needs matching shapes, got {x.shape} and {y.shape}."
        )
    if x.ndim != 1:
        raise ValueError("spearman_rho expects one-dimensional inputs.")
    if x.shape[0] < 2:
        return float("nan")

    rank_x = _average_ranks(x)
    rank_y = _average_ranks(y)
    rank_x -= rank_x.mean()
    rank_y -= rank_y.mean()
    denominator = np.sqrt(float((rank_x**2).sum()) * float((rank_y**2).sum()))
    if denominator == 0.0:
        return float("nan")
    return float((rank_x * rank_y).sum() / denominator)


def _quantiles(values: np.ndarray) -> dict[str, float]:
    """Summarize one distribution at the shared percentiles."""

    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"count": 0}
    percentiles = np.percentile(values, GRADIENT_QUANTILES)
    summary: dict[str, float] = {"count": int(values.size), "mean": float(values.mean())}
    for percentile, value in zip(GRADIENT_QUANTILES, percentiles):
        summary[f"p{percentile:02d}"] = float(value)
    return summary


def _plotted_mask(record: Any) -> np.ndarray:
    """Tokens carrying at least one gradient measurement, i.e. ``n_i > 0``."""

    return np.asarray(token_gradient_norms(record).target_count) > 0


def gradient_observable_summary(record: Any) -> dict[str, Any]:
    """Describe the distributions behind figure 10, before anything is plotted.

    Axis scaling and colour normalization are chosen from these numbers rather
    than assumed, so this is deliberately a separate, printable step.
    """

    table = gradient_guess_table(record)
    plotted = _plotted_mask(record)
    counts = table["target_occurrence_count"][plotted]
    norms = table["mean_gradient_norm"][plotted]
    guesses = table["greedy_guess_fraction"][plotted]
    corpus = table["corpus_fraction"][plotted]

    zero_guess = int((guesses == 0.0).sum())
    positive_norms = norms[norms > 0]
    dynamic_range = (
        float(positive_norms.max() / positive_norms.min()) if positive_norms.size else float("nan")
    )

    occurrence_histogram = {
        str(value): int((counts == value).sum()) for value in (1, 2, 3, 4, 5)
    }
    occurrence_histogram["6_to_10"] = int(((counts >= 6) & (counts <= 10)).sum())
    occurrence_histogram["11_to_100"] = int(((counts >= 11) & (counts <= 100)).sum())
    occurrence_histogram["over_100"] = int((counts > 100).sum())

    return {
        "num_positions": int(table["num_positions"]),
        "covers_all_positions": bool(table["covers_all_positions"]),
        "initialization_index": int(table["initialization_index"]),
        "num_plotted_tokens": int(plotted.sum()),
        "vocab_size": int(record.vocab_size),
        "eligible_vocab_size": int(record.eligible_vocab_size),
        "mean_gradient_norm": _quantiles(norms),
        "gradient_dynamic_range": dynamic_range,
        "greedy_guess_fraction": _quantiles(guesses),
        "zero_guess_tokens": zero_guess,
        "zero_guess_fraction": float(zero_guess / max(int(plotted.sum()), 1)),
        "positive_corpus_fraction": _quantiles(corpus[corpus > 0]),
        "nonpositive_corpus_tokens": int((corpus <= 0).sum()),
        # A token can be guessed without ever being a target, in which case it
        # has no G_i and cannot be plotted. This says how much of the greedy mass
        # the plotted tokens actually account for, so the omission is visible
        # rather than silent.
        "greedy_mass_on_plotted_tokens": float(guesses.sum()),
        "target_occurrence_count": _quantiles(counts),
        "occurrence_histogram": occurrence_histogram,
    }


def gradient_guess_correlations(record: Any) -> dict[str, Any]:
    """Rank correlations between ``G_i``, ``q_i`` and ``p_i``.

    Computed over every token with ``n_i > 0`` -- the same set figure 10 plots.
    Tokens that were never greedily guessed are **included**: ``q_i = 0`` is a
    measured outcome, and dropping it would bias the primary coefficient towards
    the tokens the model happens to favour.

    The primary coefficient is deliberately **unweighted**, even though ``G_i``
    is a mean over ``n_i`` positions and is therefore far noisier for a token
    seen once than for one seen a hundred times. Weighting would answer a
    different question and would need its own justification, so the imprecision
    is reported instead, through ``strata`` and ``sensitivity``, and left visible.
    """

    table = gradient_guess_table(record)
    plotted = _plotted_mask(record)
    counts = table["target_occurrence_count"][plotted]
    norms = table["mean_gradient_norm"][plotted]
    guesses = table["greedy_guess_fraction"][plotted]
    corpus = table["corpus_fraction"][plotted]

    strata = []
    for low, high in OCCURRENCE_STRATA:
        selected = counts >= low if high is None else (counts >= low) & (counts <= high)
        strata.append(
            {
                "min_occurrences": low,
                "max_occurrences": high,
                "num_tokens": int(selected.sum()),
                "spearman_gradient_vs_guess": spearman_rho(norms[selected], guesses[selected]),
            }
        )

    sensitivity = [
        {
            "min_occurrences": threshold,
            "num_tokens": int((counts >= threshold).sum()),
            "spearman_gradient_vs_guess": spearman_rho(
                norms[counts >= threshold], guesses[counts >= threshold]
            ),
        }
        for threshold in (1, 2, 5, 10, 20, 50)
    ]

    return {
        "num_tokens": int(plotted.sum()),
        "num_positions": int(table["num_positions"]),
        "includes_zero_guess_tokens": True,
        "weighted": False,
        "spearman_gradient_vs_guess": spearman_rho(norms, guesses),
        "spearman_gradient_vs_corpus": spearman_rho(norms, corpus),
        "spearman_guess_vs_corpus": spearman_rho(guesses, corpus),
        "strata": strata,
        "sensitivity_by_min_occurrences": sensitivity,
    }


def _require_temperature_gradients(record: Any) -> None:
    if not record.has_temperature_gradient_analysis:
        raise ValueError(
            "This record carries no temperature-conditioned gradient analysis. "
            "It predates the diagnostic; the experiment must be rerun to obtain it."
        )


def temperature_gradient_table(record: Any, temperature: float) -> dict[str, Any]:
    """The figure-10 table at one gradient temperature.

    Only ``mean_gradient_norm`` moves with temperature. ``n_i``, ``q_i`` and
    ``p_i`` are read from the same stored positions, targets and greedy IDs at
    every temperature, which is the point of the design: greedy identity is
    temperature-invariant, so any change in the scatter comes entirely from the
    gradient side.
    """

    _require_temperature_gradients(record)
    index = record.gradient_temperature_index(temperature)
    table = dict(gradient_guess_table(record))

    vocab_size = record.vocab_size
    targets = np.asarray(record.gradient_position_target_ids, dtype=np.int64)
    norms = np.asarray(record.gradient_temperature_position_norms[index], dtype=np.float64)
    counts = table["target_occurrence_count"]
    sums = np.bincount(targets, weights=norms, minlength=vocab_size)
    means = np.full(vocab_size, np.nan, dtype=np.float64)
    measured = counts > 0
    means[measured] = sums[measured] / counts[measured]

    table["mean_gradient_norm"] = means
    table["temperature"] = float(temperature)
    table["is_canonical"] = float(temperature) == 1.0
    return table


def _require_paired_position_set(record: Any, what: str) -> None:
    """Refuse to pair a whole-experiment statistic with a subset gradient table.

    ``mean_gradient_norm`` and figure 10's ``q_i`` are both computed over the
    *gradient-evaluated* positions, which may be a subset. The realized nucleus
    counts and the mean token probabilities are accumulated over **every**
    evaluation position, because they come from the main measurement loop rather
    than from the gradient pass. The two describe the same positions only when
    the gradient analysis covered all of them.

    Pairing them regardless would put a subset quantity on one axis and a
    whole-experiment quantity on the other and say nothing about it, which is
    exactly the silent mismatch ``gradient_guess_table`` was written to avoid.
    """

    if not _covers_all_positions(record):
        raise ValueError(
            f"{what} is accumulated over every evaluation position, but this "
            "record's gradient analysis covered only a subset, so the two axes "
            "would describe different position sets. Rerun with the gradient "
            "analysis covering all positions to produce this figure."
        )


def nucleus_gradient_table(record: Any, temperature: float) -> dict[str, Any]:
    """Figure 15's table: ``G_i(T)`` against the *realized* nucleus fraction.

    The stochastic-decoding sibling of figure 10. The y quantity is

    ``q_i^nuc(T) = (1/D) * sum_d 1[sampled(d, T) == i]``

    where ``sampled(d, T)`` is the single nucleus draw the experiment already
    made at that position and temperature, under the recorded ``top_p``, the
    recorded sampling seed, and the common-random-number protocol. Nothing is
    resampled here and no random number is drawn: the counts are read straight
    out of the persisted sweep, so this is the realized decision, not the
    expected nucleus distribution.

    Exactly one draw exists per position per temperature, so the counts sum to
    the position total.

    Raises:
        ValueError: If the record carries no sweep, or the gradient analysis
            covered only a subset of positions.
        KeyError: If the sweep never ran at this gradient temperature.
    """

    _require_temperature_gradients(record)
    if not record.has_temperature_sweep:
        raise ValueError(
            "This record carries no nucleus temperature sweep, so the realized "
            "nucleus guess fractions figure 15 needs were never measured."
        )
    _require_paired_position_set(record, "The realized nucleus count")

    # R = 1 is part of the definition, not an incidental property of the runs so
    # far. The statistic counts *one* realized stochastic decision per evaluation
    # position; with several replicates there is no single realized decision to
    # count, and every way of producing one -- averaging them, pooling them, or
    # picking one -- would silently answer a different question than the figure
    # asks. Refusing is the only honest option.
    replicates = record.num_replicates
    if replicates != 1:
        raise ValueError(
            f"Figure 15 measures one realized nucleus decision per evaluation "
            f"position and is defined only for R = 1; this record carries "
            f"R = {replicates} replicates. Averaging, pooling, or selecting one "
            "of them would change the statistic, so the figure is not drawn."
        )

    grid = record.sweep_temperatures
    value = float(temperature)
    if value not in grid:
        raise KeyError(
            f"Temperature {value:g} is not in the nucleus sweep grid {grid}. The "
            "sweep grid is configured independently of the gradient grid, so a "
            "gradient temperature need not have been sampled."
        )

    table = dict(temperature_gradient_table(record, temperature))
    # Row of the initialization the gradients were measured on -- never row 0 by
    # assumption, so the two axes cannot come from different weights.
    initialization = record.gradient_initialization_index
    counts = np.asarray(
        record.sweep_counts("real")[initialization, grid.index(value)], dtype=np.int64
    )
    total = float(counts.sum())
    if total <= 0:
        raise ValueError("The realized nucleus counts for this temperature are empty.")

    table["nucleus_guess_count"] = counts
    table["nucleus_guess_fraction"] = counts.astype(np.float64) / total
    table["nucleus_num_positions"] = int(total)
    table["nucleus_source"] = "realized_sweep_draw_one_per_position"
    return table


def mean_probability_gradient_table(record: Any, temperature: float) -> dict[str, Any]:
    """Figure 16's table: ``G_i(T)`` against the mean predictive probability.

    The continuous sibling of figures 10 and 15. The y quantity is

    ``pbar_i(T) = (1/D) * sum_d softmax(z_d / T)_i``

    averaged at **fixed token identity** over the same real-input positions --
    before top-p truncation and before any sampling decision. It is a global
    preference for a token identity, not a target-conditional confidence, and it
    is not averaged across initializations: the row of the initialization the
    gradients were measured on is used.

    This reuses the statistic figure 14 already persists rather than storing a
    second copy of the same numbers.

    Raises:
        ValueError: If the record carries no mean token probabilities, or the
            gradient analysis covered only a subset of positions.
        KeyError: If the confidence grid never included this gradient temperature.
    """

    _require_temperature_gradients(record)
    if not record.has_mean_token_probabilities:
        raise ValueError(
            "This record carries no identity-preserving mean token probabilities, "
            "so figure 16 has nothing to draw."
        )
    _require_paired_position_set(record, "The mean predictive probability")

    # Raises KeyError naming the recorded grid when the temperature is absent.
    index = record.temperature_index(temperature)
    initialization = record.gradient_initialization_index

    table = dict(temperature_gradient_table(record, temperature))
    table["mean_predictive_probability"] = np.asarray(
        record.predictive_temperature_mean_token_probabilities[initialization, index],
        dtype=np.float64,
    )
    table["mean_probability_source"] = "predictive_temperature_mean_token_probabilities"
    return table


def temperature_gradient_summary(record: Any) -> dict[str, Any]:
    """Per-temperature gradient statistics against the fixed guessing bias.

    ``n_i`` is computed once and shared: it counts target occurrences, which no
    temperature can change. Reporting it per temperature would suggest otherwise.
    """

    _require_temperature_gradients(record)
    grid = record.gradient_temperature_grid
    per_position = np.asarray(record.gradient_temperature_position_norms, dtype=np.float64)
    base = gradient_guess_table(record)
    plotted = base["target_occurrence_count"] > 0
    counts = base["target_occurrence_count"][plotted]
    guesses = base["greedy_guess_fraction"][plotted]
    corpus = base["corpus_fraction"][plotted]

    rows = []
    for index, temperature in enumerate(grid):
        table = temperature_gradient_table(record, temperature)
        token_norms = table["mean_gradient_norm"][plotted]
        positions = per_position[index]

        strata = []
        for low, high in OCCURRENCE_STRATA:
            selected = counts >= low if high is None else (counts >= low) & (counts <= high)
            strata.append(
                {
                    "min_occurrences": low,
                    "max_occurrences": high,
                    "num_tokens": int(selected.sum()),
                    "spearman_gradient_vs_guess": spearman_rho(
                        token_norms[selected], guesses[selected]
                    ),
                }
            )
        rows.append(
            {
                "temperature": float(temperature),
                "is_canonical": float(temperature) == 1.0,
                "position_gradient_norm": _quantiles(positions),
                "token_gradient_norm": _quantiles(token_norms),
                "num_tokens": int(plotted.sum()),
                "spearman_gradient_vs_guess": spearman_rho(token_norms, guesses),
                "spearman_gradient_vs_corpus": spearman_rho(token_norms, corpus),
                "strata": strata,
                "sensitivity_by_min_occurrences": [
                    {
                        "min_occurrences": threshold,
                        "num_tokens": int((counts >= threshold).sum()),
                        "spearman_gradient_vs_guess": spearman_rho(
                            token_norms[counts >= threshold], guesses[counts >= threshold]
                        ),
                    }
                    for threshold in (1, 2, 5, 10, 20, 50)
                ],
            }
        )

    return {
        "temperatures": np.asarray(grid, dtype=np.float64),
        "num_positions": int(per_position.shape[1]),
        "num_tokens": int(plotted.sum()),
        "greedy_is_temperature_invariant": True,
        # Occurrence counts describe targets, which no temperature can change.
        "target_occurrence_count": _quantiles(counts),
        "rows": rows,
    }
