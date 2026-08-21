"""Directional clustering of gradients grouped by the nucleus sample.

Figure 20 groups the same gradient population two ways -- by target token and by
greedy prediction -- and finds structure under both. Those are the deterministic
endpoints. This asks what happens in between: group by the token the model
actually *sampled* under nucleus sampling at temperature ``T``, and sweep ``T``.

Temperature moves one thing and one thing only, the distribution being sampled.
The uniforms are held fixed, so the stochastic draw does not change underneath
the comparison. As ``T`` falls the sample concentrates on the argmax and the
grouping approaches the greedy one; as ``T`` rises the sample spreads and classes
fragment into singletons.

That fragmentation is why the pooled statistic here must be the all-position one.
A singleton class contributes no within-class pair but is still a real position
forming between-class pairs with every other. Dropping singletons would redefine
"between" as "between non-singleton classes" -- and since the singleton fraction
moves *with temperature*, that redefinition would manufacture a trend out of the
support changing rather than the geometry changing. The support diagnostics are
reported beside every point for exactly that reason: a reader must be able to
see how much of the class structure is real at each temperature.

Nothing here re-samples. The labels come from a reproduction of the run's own
draw that has already been gated against the recorded histogram exactly; see
:mod:`llm_behavior_lab.evaluation.nucleus_labels`.

Sampling temperature is not the gradient-loss temperature. The gradients are the
``T = 1`` gradients throughout; only the *grouping* varies with ``T``.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from llm_behavior_lab.analysis.gradient_clustering import (
    DEFAULT_PERMUTATIONS,
    gradient_clustering,
    unit_sketches,
)

__all__ = ["histogram_gate", "nucleus_clustering", "support_diagnostics"]


def support_diagnostics(labels: np.ndarray, *, min_support: int = 2) -> dict[str, Any]:
    """How much class structure survives at this temperature.

    Reported with every pooled value because the two move together: a
    temperature whose classes are almost all singletons has a within-class
    statistic resting on very few positions, whatever that statistic says.
    """

    classes, counts = np.unique(labels, return_counts=True)
    qualifying = counts >= int(min_support)
    within_pairs = int((counts * (counts - 1)).sum())
    total = int(counts.sum())
    return {
        "num_represented": int(classes.size),
        "num_qualifying": int(qualifying.sum()),
        "num_singletons": int((counts == 1).sum()),
        "singleton_fraction": float((counts == 1).sum() / classes.size),
        "positions_in_qualifying": int(counts[qualifying].sum()),
        "fraction_positions_in_qualifying": float(counts[qualifying].sum() / total),
        "largest_class": int(counts.max()),
        "median_class": float(np.median(counts)),
        "num_within_pairs": within_pairs,
        "num_between_pairs": int(total ** 2 - (counts.astype(np.int64) ** 2).sum()),
    }


def nucleus_clustering(
    record: Any,
    labels_by_temperature: np.ndarray,
    temperatures: Sequence[float],
    *,
    min_support: int = 2,
    display_classes: int | None = 40,
    permutations: int = DEFAULT_PERMUTATIONS,
    permutation_seed: int = 20240918,
) -> dict[str, Any]:
    """Sweep the grouping across sampling temperature.

    Each temperature is handed to the same estimator the target and greedy
    groupings use, through its explicit-label entry point. Reusing it is the
    point: a separate implementation could drift from the all-position
    definition, and the trajectory would then be comparing two different
    statistics rather than one statistic at two temperatures.

    The target and greedy groupings are computed alongside as reference levels,
    from the same record and the same estimator, so the trajectory can be read
    against the endpoints it lies between.

    Args:
        record: A record carrying per-position gradient sketches.
        labels_by_temperature: ``[T, D]`` nucleus samples, in record position
            order, already gated against the recorded histogram.
        temperatures: The ``T`` values those rows correspond to.
        min_support: Class size that counts as qualifying, for reporting only.
        display_classes: Classes in each stored heatmap.
        permutations: Draws in the label-permutation null, per temperature.
        permutation_seed: Base seed. Each temperature offsets it by its index so
            no two temperatures share a permutation sequence, which would
            correlate their nulls and understate the spread of the trajectory.

    Returns:
        ``by_temperature`` -- one clustering result and support summary per
        temperature; ``references`` -- the target and greedy groupings.
    """

    labels_by_temperature = np.asarray(labels_by_temperature, dtype=np.int64)
    ordered = tuple(float(value) for value in temperatures)
    if labels_by_temperature.shape[0] != len(ordered):
        raise ValueError(
            f"{labels_by_temperature.shape[0]} label rows against "
            f"{len(ordered)} temperatures."
        )
    unit, usable = unit_sketches(record)
    if labels_by_temperature.shape[1] != unit.shape[0]:
        raise ValueError(
            f"The labels cover {labels_by_temperature.shape[1]} positions but the "
            f"record holds {unit.shape[0]} gradient sketches."
        )

    by_temperature = []
    for index, temperature in enumerate(ordered):
        labels = labels_by_temperature[index]
        result = gradient_clustering(
            record,
            grouping=f"nucleus T={temperature:g}",
            labels=labels,
            min_support=min_support,
            display_classes=display_classes,
            permutations=permutations,
            permutation_seed=permutation_seed + index,
        )
        result["temperature"] = float(temperature)
        # Diagnostics describe the population the statistic was computed over,
        # so they use the same zero-norm mask rather than the raw label vector.
        result["support"] = support_diagnostics(
            labels[usable], min_support=min_support
        )
        by_temperature.append(result)

    references = {
        grouping: gradient_clustering(
            record,
            grouping=grouping,
            min_support=min_support,
            display_classes=display_classes,
            permutations=permutations,
            permutation_seed=permutation_seed,
        )
        for grouping in ("target", "greedy")
    }
    return {
        "temperatures": ordered,
        "by_temperature": by_temperature,
        "references": references,
        "min_support": int(min_support),
        "permutations": int(permutations),
        "permutation_seed": int(permutation_seed),
    }


def histogram_gate(
    labels: np.ndarray,
    expected_counts: np.ndarray,
    *,
    vocab_size: int,
    temperatures: Sequence[float],
) -> dict[str, Any]:
    """Demand exact agreement with the histogram the experiment recorded.

    ``np.array_equal`` on integer counts, per temperature. Not a tolerance, not
    a correlation, not a fraction-agreeing: the counts are integers produced by
    the same deterministic procedure, so they either match or the reproduction is
    of something else.

    Raises:
        ValueError: On the first temperature whose histogram differs, reporting
            how many token IDs disagree and the largest discrepancy, so the
            failure is diagnosable rather than merely fatal.
    """

    expected = np.asarray(expected_counts, dtype=np.int64)
    if expected.shape[0] != labels.shape[0]:
        raise ValueError(
            f"{labels.shape[0]} recovered temperatures against "
            f"{expected.shape[0]} recorded histograms."
        )

    per_temperature = []
    for index, temperature in enumerate(temperatures):
        observed = np.bincount(labels[index], minlength=vocab_size).astype(np.int64)
        recorded = expected[index]
        if not np.array_equal(observed, recorded):
            differing = np.flatnonzero(observed != recorded)
            worst = int(np.abs(observed - recorded).max())
            raise ValueError(
                f"The recovered nucleus labels at T = {temperature:g} do not "
                f"reproduce the recorded histogram: {differing.size} token IDs "
                f"differ, largest discrepancy {worst}. The reconstruction "
                "describes a different model, initialization, support or "
                "sampling stream than the one measured, so nothing may be built "
                "on these labels."
            )
        per_temperature.append(
            {
                "temperature": float(temperature),
                "num_positions": int(labels[index].size),
                "num_distinct_tokens": int((observed > 0).sum()),
            }
        )
    return {"passed": True, "per_temperature": per_temperature}
