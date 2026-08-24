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
from llm_behavior_lab.analysis.temperature_pairs import temperature_pairs

__all__ = [
    "histogram_gate",
    "nucleus_clustering",
    "resolve_forward_batch_size",
    "support_diagnostics",
]


def resolve_forward_batch_size(
    analysis_metadata: dict[str, Any], override: int | None = None
) -> dict[str, Any]:
    """Which forward batch size a historical reconstruction must use.

    Batch size belongs to a run's numerical provenance, not to its performance
    tuning. The pre-drawn uniforms are indexed by position, so batching cannot
    change *which* uniform a position draws -- but the logits are the draw's
    other input, and on accelerator kernels a matrix multiplication can return
    bit-different values at different batch shapes. A token near a truncation
    boundary can then fall the other way.

    So the realized value in the record wins by default. Two things it must not
    be confused with:

    * ``runtime.forward_batch_size`` in the frozen YAML is what was *requested*;
      a real run reconstructed at the YAML's 32, or at this script's old
      hard-coded 8, when it had realized 4, and moved one position at
      ``T = 0.12``; and
    * a hard-coded default is a guess about a run it has never seen.

    An explicit override stays available -- it is what diagnosed the mismatch
    above -- but is reported as an override so an audit log cannot be misread.

    Args:
        analysis_metadata: The record's ``metadata["analysis"]`` mapping.
        override: An explicit CLI value, or ``None`` to use the realized one.

    Returns:
        ``value``, the ``source`` it came from, the ``historical`` value for
        comparison, and a one-line ``description`` for printing.

    Raises:
        ValueError: When no override is given and the record does not record a
            realized batch size. Inventing one would silently reintroduce the
            failure this function exists to prevent, so the caller is told to
            supply the value explicitly instead.
    """

    historical = analysis_metadata.get("forward_batch_size")
    historical = None if historical is None else int(historical)

    if override is not None:
        if int(override) <= 0:
            raise ValueError("forward_batch_size must be positive.")
        known = "unrecorded" if historical is None else str(historical)
        return {
            "value": int(override),
            "source": "override",
            "historical": historical,
            "description": (
                f"{int(override)} (explicit diagnostic override; historical value "
                f"{known})"
            ),
        }

    if historical is None:
        raise ValueError(
            "This record does not carry metadata['analysis']['forward_batch_size'], "
            "so the batch size it realized is unknown. Forward batching is part of "
            "the numerical provenance on accelerator kernels, so a default cannot "
            "be assumed here. Determine the value the run used and pass it with "
            "--forward-batch-size."
        )
    if historical <= 0:
        raise ValueError(
            f"The record reports a forward batch size of {historical}, which cannot "
            "be the value it ran at."
        )
    return {
        "value": historical,
        "source": "historical",
        "historical": historical,
        "description": f"{historical} (historical realized metadata)",
    }


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
    sampling_temperatures: Sequence[float],
    *,
    loss_temperatures: Sequence[float] | None = None,
    min_support: int = 2,
    display_classes: int | None = 40,
    permutations: int = DEFAULT_PERMUTATIONS,
    permutation_seed: int = 20240918,
) -> dict[str, Any]:
    """Sweep the grouping across sampling temperature.

    Each pair is ``(T_s, T_g)``: ``T_s`` shapes the distribution the grouping
    label was sampled from, ``T_g`` the loss the gradients come from. The
    gradients here are whatever the record persisted, so ``T_g`` is a statement
    about the record rather than something recomputed -- figure 22's record
    holds the ``T_g = 1`` gradients throughout. Carrying it explicitly is what
    keeps the axis honest once matched-temperature sweeps exist alongside it.

    Each pair is handed to the same estimator the target and greedy
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
        sampling_temperatures: The ``T_s`` values those rows correspond to.
        loss_temperatures: The ``T_g`` paired elementwise with them. Omitted
            means ``T_g = T_s``; figure 22's controlled design passes ``1.0``
            repeated, since its gradients are the ``T = 1`` gradients.
        min_support: Class size that counts as qualifying, for reporting only.
        display_classes: Classes in each stored heatmap.
        permutations: Draws in the label-permutation null, per temperature.
        permutation_seed: Base seed. Each pair offsets it by its index so no two
            pairs share a permutation sequence, which would correlate their
            nulls and understate the spread of the trajectory.

    Returns:
        ``by_temperature`` -- one clustering result and support summary per
        pair; ``references`` -- the target and greedy groupings; and the
        validated ``sampling_temperatures`` / ``loss_temperatures``.
    """

    pairs = temperature_pairs(sampling_temperatures, loss_temperatures)
    labels_by_temperature = np.asarray(labels_by_temperature, dtype=np.int64)
    ordered = tuple(float(value) for value in pairs["sampling_temperatures"])
    if labels_by_temperature.shape[0] != len(ordered):
        raise ValueError(
            f"{labels_by_temperature.shape[0]} label rows against "
            f"{len(ordered)} temperature pair(s)."
        )
    # One directional field per *unique* T_g, resolved once. Reading it is the
    # expensive half of a pair, and a sweep that pins T_g would otherwise pay
    # for the same field at every point.
    fields: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for unique_index, value in enumerate(pairs["unique_loss"]):
        fields[unique_index] = unit_sketches(record, loss_temperature=float(value))
    unit, usable = fields[0]
    if labels_by_temperature.shape[1] != unit.shape[0]:
        raise ValueError(
            f"The labels cover {labels_by_temperature.shape[1]} positions but the "
            f"record holds {unit.shape[0]} gradient sketches."
        )

    # Reuse, at the only level where it is sound. A pair's result depends on the
    # sampled labels (through T_s) and the gradient directions (through T_g).
    # The directions come from the record and are read once above; labels are
    # supplied per pair. So two pairs agreeing on both indices are the same
    # measurement and the second is a lookup, which is what keeps a sweep like
    # T_g = 1 repeated six times from paying six times over. Pairs that differ
    # in either coordinate are computed, since nothing about them is shared.
    by_temperature = []
    computed: dict[tuple[int, int], dict[str, Any]] = {}
    num_reused = 0
    for index, temperature in enumerate(ordered):
        loss_temperature = float(pairs["loss_temperatures"][index])
        key = (
            int(pairs["sampling_index"][index]),
            int(pairs["loss_index"][index]),
        )
        cached = computed.get(key)
        if cached is not None:
            result = dict(cached)
            result["reused_from_pair"] = int(result["pair_index"])
            num_reused += 1
        else:
            labels = labels_by_temperature[index]
            result = gradient_clustering(
                record,
                grouping=f"nucleus T_s={temperature:g} T_g={loss_temperature:g}",
                labels=labels,
                loss_temperature=loss_temperature,
                min_support=min_support,
                display_classes=display_classes,
                permutations=permutations,
                permutation_seed=permutation_seed + index,
            )
            # Diagnostics describe the population the statistic was computed
            # over, so they use this T_g's own zero-norm mask: which positions
            # have a direction at all is a property of the gradient field.
            _, field_usable = fields[key[1]]
            result["support"] = support_diagnostics(
                labels[field_usable], min_support=min_support
            )
            result["reused_from_pair"] = None
            computed[key] = result
        result["pair_index"] = index
        result["sampling_temperature"] = float(temperature)
        result["loss_temperature"] = loss_temperature
        # Kept for readers written before the pair split, where it always meant
        # the sampling temperature.
        result["temperature"] = float(temperature)
        by_temperature.append(result)

    # References belong to a gradient field, not to the sweep as a whole. With
    # T_g pinned they are the single pair of horizontal lines figure 22 draws;
    # once T_g varies there is one pair per measured T_g, and comparing a
    # nucleus point against a reference from a different field would be
    # comparing two different geometries.
    references_by_loss = {}
    for unique_index, value in enumerate(pairs["unique_loss"]):
        loss_value = float(value)
        references_by_loss[loss_value] = {
            grouping: gradient_clustering(
                record,
                grouping=grouping,
                loss_temperature=loss_value,
                min_support=min_support,
                display_classes=display_classes,
                permutations=permutations,
                permutation_seed=permutation_seed,
            )
            for grouping in ("target", "greedy")
        }
    # The historical shape, which is unambiguous exactly when T_g is pinned.
    references = references_by_loss[float(pairs["unique_loss"][0])]
    return {
        "sampling_temperatures": pairs["sampling_temperatures"],
        "loss_temperatures": pairs["loss_temperatures"],
        "loss_defaulted": pairs["loss_defaulted"],
        "num_unique_sampling": int(pairs["unique_sampling"].size),
        "num_unique_loss": int(pairs["unique_loss"].size),
        "num_pairs_reused": num_reused,
        "num_fields_read": len(fields),
        "references_by_loss_temperature": references_by_loss,
        # Historical key, always the sampling temperatures.
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

    The same three discrepancy measures -- differing bins, largest difference,
    total absolute difference -- are reported whether the gate passes or fails.
    On a pass they are all zero, and stating that explicitly is the point: a
    report that prints nothing on success cannot be distinguished from a report
    whose check never ran.

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
        difference = observed - recorded
        differing = np.flatnonzero(difference)
        worst = int(np.abs(difference).max()) if difference.size else 0
        total = int(np.abs(difference).sum())
        if not np.array_equal(observed, recorded):
            raise ValueError(
                f"The recovered nucleus labels at T = {temperature:g} do not "
                f"reproduce the recorded histogram: {differing.size} token IDs "
                f"differ, largest discrepancy {worst}, total absolute "
                f"difference {total}. The reconstruction "
                "describes a different model, initialization, support or "
                "sampling stream than the one measured, so nothing may be built "
                "on these labels."
            )
        per_temperature.append(
            {
                "temperature": float(temperature),
                "exact_match": True,
                "mismatched_bins": int(differing.size),
                "max_absolute_difference": worst,
                "sum_absolute_difference": total,
                "num_positions": int(labels[index].size),
                "num_distinct_tokens": int((observed > 0).sum()),
            }
        )
    return {"passed": True, "per_temperature": per_temperature}
