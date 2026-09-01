"""Turning per-position sketch rows into finalized metrics, once, while they exist.

A ``metrics_only`` run deletes its rows as soon as this has finished, so the
scheduling here is not a convenience -- it is the last moment at which any
row-level quantity can be computed. Everything the declared analysis contract
promises has to come out of this one pass, and anything forgotten is gone.

The work is organised around the fact that a slab is the unit of residency. For
each ``(loss temperature, map)`` pair the rows are loaded once, normalized once,
and every job that needs *that* gradient field is run against them before the
slab is released. Jobs are processed **one at a time**: target, greedy, each
nucleus grouping and the cross-partition factors never hold their class sums
simultaneously, because several high-cardinality groupings coexisting is exactly
how the workspace budget would be blown.

Which jobs exist at which temperature is fixed by the contract, not by
convenience:

* every measured ``T_g`` gets target and greedy references -- a reference
  computed in a different gradient field would be a different geometry;
* the canonical ``T_g = 1`` additionally gets the nucleus **control** points
  (all sampling temperatures against the fixed canonical field) and the
  cross-partition statistics;
* each non-canonical ``T_g`` that equals a sampling temperature gets that
  **matched** point.

At the production grid that is 26 row-permutation nulls. The cross-partition
identity null is not among them: it permutes label identities rather than rows,
so it is computed from group factors and costs nothing.

Ensemble values are the arithmetic mean over maps of the per-map values, which
is the estimator's definition; the spread across maps is the projection
uncertainty and comes from :func:`ensemble_summary`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator

import numpy as np

from llm_behavior_lab.analysis.alignment_estimators import (
    class_factors,
    class_similarity_from_factors,
    cross_factors,
    cross_matrix_from_factors,
    permutation_null,
    pooled_cross_from_factors,
    pooled_from_factors,
)
from llm_behavior_lab.analysis.sketch_estimator import ensemble_summary

__all__ = [
    "FinalizationInputs",
    "JobPlan",
    "finalize_alignment_metrics",
    "plan_jobs",
]

#: Matching tolerance for "this loss temperature is that sampling temperature".
#: The record layer owns the same constant; it is restated rather than imported
#: to keep this module free of record imports, and a test pins the two equal.
TEMPERATURE_TOLERANCE = 1e-6

CANONICAL_TEMPERATURE = 1.0


@dataclass(frozen=True)
class JobPlan:
    """One ``(grouping, loss temperature)`` null and where its labels come from."""

    #: ``"target"``, ``"greedy"``, ``"nucleus"`` or ``"cross"``.
    kind: str
    #: Index into the measured loss-temperature grid.
    loss_index: int
    #: Index into the sampling-temperature grid, for nucleus jobs only.
    sampling_index: int | None = None
    #: ``"control"`` or ``"matched"``, for nucleus jobs only.
    design: str | None = None

    @property
    def needs_row_null(self) -> bool:
        """Cross-partition permutes identities, not rows, so it needs no draws."""

        return self.kind != "cross"

    def seed_offset(self, base: int) -> int:
        """This job's permutation seed, derived from its own coordinates.

        Deliberately arithmetic on the job's integer fields and **never**
        ``hash()``: Python randomizes string hashing per process, so a seed
        derived that way would differ between two runs of the same experiment
        and the null would stop being reproducible. The offsets are spread far
        enough apart that no two jobs collide.
        """

        kind_offset = {"target": 0, "greedy": 1, "nucleus": 2, "cross": 3}[self.kind]
        design_offset = 0 if self.design != "matched" else 1
        return (
            int(base)
            + 1_000_003 * kind_offset
            + 1_009 * int(self.loss_index)
            + 17 * int(self.sampling_index if self.sampling_index is not None else 0)
            + 7 * design_offset
        )


def plan_jobs(
    loss_temperatures: np.ndarray, sampling_temperatures: np.ndarray
) -> list[JobPlan]:
    """The complete job list, derived from the two grids rather than hard-coded.

    Deriving it means a run that measured a different grid gets the jobs its own
    grid implies, and means the count can be asserted against the contract
    instead of trusted.
    """

    loss = np.asarray(loss_temperatures, dtype=np.float64)
    sampling = np.asarray(sampling_temperatures, dtype=np.float64)
    canonical = int(
        np.flatnonzero(np.abs(loss - CANONICAL_TEMPERATURE) <= TEMPERATURE_TOLERANCE)[0]
    )

    jobs: list[JobPlan] = []
    for index in range(loss.size):
        jobs.append(JobPlan("target", index))
        jobs.append(JobPlan("greedy", index))
    for sampling_index in range(sampling.size):
        jobs.append(
            JobPlan("nucleus", canonical, sampling_index=sampling_index,
                    design="control")
        )
    for sampling_index, value in enumerate(sampling):
        match = np.flatnonzero(np.abs(loss - value) <= TEMPERATURE_TOLERANCE)
        if match.size == 1:
            jobs.append(
                JobPlan("nucleus", int(match[0]), sampling_index=sampling_index,
                        design="matched")
            )
    jobs.append(JobPlan("cross", canonical))
    return jobs


@dataclass
class FinalizationInputs:
    """Everything finalization consumes besides the rows themselves."""

    loss_temperatures: np.ndarray
    #: ``[N_T, D]`` exact gradient norms, one row per measured temperature.
    norms: np.ndarray
    target_ids: np.ndarray
    greedy_ids: np.ndarray
    #: ``[S, D]`` nucleus labels captured during the run, or ``None``.
    nucleus_labels: np.ndarray | None = None
    sampling_temperatures: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    min_support: int = 2
    display_classes: int = 40
    permutations: int = 256
    permutation_seed: int = 20240918
    cross_null_seed: int = 20240919
    #: Declared `compact_factors` retentions, or empty. Retained factors are
    #: gathered during the same pass that computes the metrics -- there is no
    #: second opportunity, because the rows go immediately afterwards.
    factor_selections: tuple = ()


def _usable(norms: np.ndarray) -> np.ndarray:
    """Positions with a non-zero exact norm.

    A zero-norm gradient has no direction, so it cannot be normalized and is
    excluded -- and its exclusion is counted rather than passed over.
    """

    return np.asarray(norms, dtype=np.float64) > 0.0


def _normalize(slab: np.ndarray, norms: np.ndarray, usable: np.ndarray) -> np.ndarray:
    """``S(g_d) / ||g_d||`` for the usable rows, as a **division**."""

    rows = np.asarray(slab, dtype=np.float64)[usable]
    return rows / np.asarray(norms, dtype=np.float64)[usable][:, None]


def _grouping_result(
    unit: np.ndarray,
    labels: np.ndarray,
    *,
    min_support: int,
    display_classes: int,
    permutations: int,
    permutation_seed: int,
    want_display: bool,
) -> dict[str, Any]:
    """Pooled statistics, null and optional heatmap for one grouping, one map.

    The **population is every represented class**, singletons included:
    ``min_support`` selects what is reported and drawn, never what is measured.
    A singleton contributes no within-class pair but is a real position forming
    between-class pairs, and dropping it would redefine "between" as "between
    non-singleton classes" -- which moves with the very condition the nucleus
    sweep varies.
    """

    represented, counts = np.unique(labels, return_counts=True)
    if represented.size < 2:
        return {"available": False, "reason": "fewer than two represented classes"}

    factors = class_factors(unit, labels, represented)
    within, between = pooled_from_factors(
        factors.sums, factors.counts, factors.self_squared
    )
    null = permutation_null(
        unit, factors.counts, permutations=permutations, seed=permutation_seed
    )

    result: dict[str, Any] = {
        "available": True,
        "within": within,
        "between": between,
        "delta": within - between,
        "num_positions": int(factors.counts.sum()),
        "num_represented": int(represented.size),
        "num_classes": int((counts >= min_support).sum()),
        "num_within_pairs": int((factors.counts * (factors.counts - 1)).sum()),
        "num_between_pairs": int(
            int(factors.counts.sum()) ** 2
            - int((factors.counts.astype(np.int64) ** 2).sum())
        ),
        "null": null,
        "support": _support_diagnostics(counts, min_support),
    }

    if want_display:
        qualifying = represented[counts >= min_support]
        shown = qualifying
        if display_classes is not None and qualifying.size > display_classes:
            order = np.argsort(-counts[counts >= min_support], kind="stable")
            shown = np.sort(qualifying[order[:display_classes]])
        drawn = class_similarity_from_factors(class_factors(unit, labels, shown))
        result["display"] = {
            "classes": shown,
            "counts": drawn["counts"],
            "matrix": drawn["matrix"],
            "within_by_class": drawn["within"],
        }
    return result


def _support_diagnostics(counts: np.ndarray, min_support: int) -> dict[str, Any]:
    """How much class structure survives, which the trajectory cannot be read
    without: a falling delta may be geometry weakening or support evaporating."""

    total = float(counts.sum())
    qualifying = counts >= min_support
    return {
        "num_represented": int(counts.size),
        "num_qualifying": int(qualifying.sum()),
        "num_singletons": int((counts == 1).sum()),
        "singleton_fraction": float((counts == 1).sum() / counts.size),
        "positions_in_qualifying": int(counts[qualifying].sum()),
        "fraction_positions_in_qualifying": float(
            counts[qualifying].sum() / total if total else np.nan
        ),
        "largest_class": int(counts.max()),
        "median_class": float(np.median(counts)),
        "num_within_pairs": int((counts * (counts - 1)).sum()),
    }


def finalize_alignment_metrics(
    slabs: Iterable[tuple[int, int, np.ndarray]],
    inputs: FinalizationInputs,
    *,
    num_maps: int,
    progress: Callable[[str, int, int, float], None] | None = None,
) -> dict[str, Any]:
    """Reduce every slab to the finalized metrics, one grouping at a time.

    Args:
        slabs: Yields ``(loss index, map index, [D, K] float32)``. One slab is
            resident at a time; the iterator owns their lifetime.
        inputs: Labels, norms, grids and policy.
        num_maps: ``M``. Read from the protocol, never inferred from a shape.
        progress: ``callback(stage, done, total, elapsed)``.

    Returns:
        ``{"per_map": ..., "ensemble": ..., "jobs": ..., "seconds": ...}`` --
        the raw material the metrics writer arranges into arrays.
    """

    jobs = plan_jobs(inputs.loss_temperatures, inputs.sampling_temperatures)
    started = time.perf_counter()
    collected: dict[tuple, list[Any]] = {}
    retained: dict[str, list[Any]] = {}
    retained_cross: dict[str, Any] = {}
    last_report = started
    done = 0
    total = len(inputs.loss_temperatures) * num_maps

    for loss_index, map_index, slab in slabs:
        usable = _usable(inputs.norms[loss_index])
        unit = _normalize(slab, inputs.norms[loss_index], usable)

        for job in jobs:
            if job.loss_index != loss_index:
                continue
            key = (job.kind, job.loss_index, job.sampling_index, job.design)

            if job.kind == "cross":
                targets = np.asarray(inputs.target_ids, dtype=np.int64)[usable]
                greedy = np.asarray(inputs.greedy_ids, dtype=np.int64)[usable]
                classes = np.union1d(np.unique(targets), np.unique(greedy))
                factors = cross_factors(unit, targets, greedy, classes)
                entry = dict(pooled_cross_from_factors(factors))
                entry["display"] = _cross_display(
                    factors, targets, greedy, classes,
                    inputs.min_support, inputs.display_classes,
                )
                # The identity null permutes which greedy identity counts as
                # "the same token" as which target identity -- group vectors and
                # support are held exactly as observed -- so it is a factor-level
                # computation, not one of the 26 row nulls. The mixture panel is
                # likewise a statement about group means. Both are computed here,
                # while the rows exist, because neither can be after.
                from llm_behavior_lab.analysis.gradient_cross_partition import (
                    cross_identity_null,
                    mixture_reconstruction,
                )

                entry["null"] = cross_identity_null(
                    unit, targets, greedy,
                    permutations=inputs.permutations,
                    seed=inputs.cross_null_seed,
                )
                entry["mixture"] = mixture_reconstruction(
                    unit, targets, greedy, min_support=inputs.min_support
                )
                entry["contingency"] = {
                    "targets": classes[factors.cell_target_index],
                    "greedy": classes[factors.cell_greedy_index],
                    "counts": factors.cell_counts,
                }
                for selection in inputs.factor_selections:
                    if selection.grouping != "cross":
                        continue
                    if abs(
                        selection.loss_temperature
                        - float(inputs.loss_temperatures[loss_index])
                    ) > TEMPERATURE_TOLERANCE:
                        continue
                    slot = retained_cross.setdefault(
                        selection.label,
                        {
                            "target_classes": factors.target.classes,
                            "target_counts": factors.target.counts,
                            "greedy_classes": factors.greedy.classes,
                            "greedy_counts": factors.greedy.counts,
                            "cell_targets": factors.cell_target_index,
                            "cell_greedy": factors.cell_greedy_index,
                            "cell_counts": factors.cell_counts,
                            "target_sums": [],
                            "greedy_sums": [],
                            "cell_self_squared": [],
                        },
                    )
                    slot["target_sums"].append(factors.target.sums)
                    slot["greedy_sums"].append(factors.greedy.sums)
                    slot["cell_self_squared"].append(factors.cell_self_squared)

                collected.setdefault(key, []).append(entry)
                del factors
                continue

            labels = _labels_for(job, inputs, usable)
            if labels is None:
                continue

            for selection in inputs.factor_selections:
                if not _selection_matches(selection, job, inputs):
                    continue
                represented = np.unique(labels)
                retained.setdefault(selection.label, []).append(
                    class_factors(unit, labels, represented)
                )

            collected.setdefault(key, []).append(
                _grouping_result(
                    unit, labels,
                    min_support=inputs.min_support,
                    display_classes=inputs.display_classes,
                    permutations=inputs.permutations,
                    permutation_seed=job.seed_offset(inputs.permutation_seed),
                    want_display=(map_index == 0),
                )
            )

        del unit
        done += 1
        now = time.perf_counter()
        if progress is not None and (now - last_report >= 30.0 or done == total):
            progress("finalize", done, total, now - started)
            last_report = now

    return {
        "collected": collected,
        "retained": retained,
        "retained_cross": retained_cross,
        "jobs": jobs,
        "num_maps": num_maps,
        "seconds": time.perf_counter() - started,
        # Per temperature: a gradient field can zero out at one T_g and not at
        # another, so a single number would misreport every other row.
        "num_positions_excluded": [
            int((~_usable(row)).sum()) for row in np.asarray(inputs.norms)
        ],
    }


def _labels_for(
    job: JobPlan, inputs: FinalizationInputs, usable: np.ndarray
) -> np.ndarray | None:
    if job.kind == "target":
        return np.asarray(inputs.target_ids, dtype=np.int64)[usable]
    if job.kind == "greedy":
        return np.asarray(inputs.greedy_ids, dtype=np.int64)[usable]
    if job.kind == "nucleus":
        if inputs.nucleus_labels is None:
            return None
        return np.asarray(
            inputs.nucleus_labels[job.sampling_index], dtype=np.int64
        )[usable]
    raise ValueError(f"Unknown job kind {job.kind!r}.")


def _cross_display(
    factors, targets, greedy, classes, min_support: int, display_classes: int
) -> dict[str, Any]:
    """The bounded heatmap subset, chosen by support in **both** roles.

    Never by observed similarity: ranking cells by their value would choose the
    conclusion before drawing it.
    """

    target_counts = np.bincount(targets, minlength=int(classes.max()) + 1)[classes]
    greedy_counts = np.bincount(greedy, minlength=int(classes.max()) + 1)[classes]
    score = np.minimum(target_counts, greedy_counts)
    eligible = np.flatnonzero(score >= min_support)
    order = eligible[np.argsort(-score[eligible], kind="stable")][:display_classes]
    slots = np.sort(order)
    drawn = cross_matrix_from_factors(factors, slots, slots)
    return drawn


def summarize_across_maps(values: list[float]) -> dict[str, Any]:
    """Mean, spread and availability across maps.

    At ``M = 1`` the spread is **unavailable, not zero**: one projection cannot
    say how far another would have landed.
    """

    return ensemble_summary(np.asarray(values, dtype=np.float64))


# -- arranging the results into the artifact ---------------------------------
#
# The key names below are the artifact's public surface. They deliberately echo
# the established `nucleus_gradient_clustering.npz` and
# `gradient_cross_partition.npz` vocabulary so a reader who knows those knows
# this, and so the figures change as little as possible.


def _summary_arrays(prefix: str, values: np.ndarray, out: dict[str, Any]) -> None:
    """Per-map values plus their mean, spread and availability.

    ``sd`` and ``se`` are float64 NaN rather than ``None`` when unavailable, so
    the archive still loads with ``allow_pickle=False``. At ``M = 1`` that NaN
    means *not estimable*, which is not the same as zero and must not be read
    as one -- ``uncertainty_available`` is what distinguishes them.
    """

    summary = ensemble_summary(values)
    out[f"{prefix}_per_map"] = np.asarray(values, dtype=np.float64)
    out[prefix] = np.asarray(summary["map_mean"], dtype=np.float64)
    for name in ("sample_sd", "standard_error"):
        suffix = "sd" if name == "sample_sd" else "se"
        value = summary[name]
        out[f"{prefix}_{suffix}"] = np.asarray(
            np.nan if value is None else value, dtype=np.float64
        )


def _stack(entries: list[dict[str, Any]], *path: str) -> np.ndarray:
    """Pull one nested scalar out of each map's result, in map order."""

    values = []
    for entry in entries:
        node: Any = entry
        for key in path:
            node = node[key]
        values.append(float(node))
    return np.asarray(values, dtype=np.float64)


def build_metrics_arrays(result: dict[str, Any], inputs: FinalizationInputs) -> dict[str, Any]:
    """Arrange the collected per-map results into the artifact's arrays."""

    collected = result["collected"]
    num_maps = result["num_maps"]
    loss = np.asarray(inputs.loss_temperatures, dtype=np.float64)
    out: dict[str, Any] = {
        "reference_loss_temperatures": loss,
        "map_count": np.asarray(num_maps, dtype=np.int64),
        "degrees_of_freedom": np.asarray(max(num_maps - 1, 0), dtype=np.int64),
        "uncertainty_available": np.asarray(num_maps > 1),
        "min_support": np.asarray(inputs.min_support, dtype=np.int64),
        "permutations": np.asarray(inputs.permutations, dtype=np.int64),
        "permutation_seed": np.asarray(inputs.permutation_seed, dtype=np.int64),
        "num_positions_excluded": np.asarray(
            result["num_positions_excluded"], dtype=np.int64
        ),
    }

    # -- target and greedy references at every measured loss temperature
    for grouping in ("target", "greedy"):
        available = np.zeros(loss.size, dtype=bool)
        buffers = {
            name: np.full((loss.size, num_maps), np.nan)
            for name in ("within", "between", "delta", "null_delta_mean")
        }
        scalars = {
            name: np.full(loss.size, np.nan)
            for name in ("null_delta_std", "null_delta_low", "null_delta_high",
                         "num_positions", "num_represented", "num_classes",
                         "num_within_pairs", "num_between_pairs")
        }
        for index in range(loss.size):
            entries = collected.get((grouping, index, None, None))
            if not entries or not entries[0].get("available"):
                continue
            available[index] = True
            buffers["within"][index] = _stack(entries, "within")
            buffers["between"][index] = _stack(entries, "between")
            buffers["delta"][index] = (
                buffers["within"][index] - buffers["between"][index]
            )
            buffers["null_delta_mean"][index] = _stack(entries, "null", "delta_mean")
            for name in ("null_delta_std", "null_delta_low", "null_delta_high"):
                key = name.replace("null_", "")
                scalars[name][index] = float(np.mean(_stack(entries, "null", key)))
            for name in ("num_positions", "num_represented", "num_classes",
                         "num_within_pairs", "num_between_pairs"):
                scalars[name][index] = entries[0][name]
            if "display" in entries[0]:
                display = entries[0]["display"]
                out[f"display_{grouping}_classes"] = display["classes"]
                out[f"display_{grouping}_counts"] = display["counts"]
                out[f"display_{grouping}_matrix"] = display["matrix"]
                out[f"display_{grouping}_within_by_class"] = display["within_by_class"]

        for name, values in buffers.items():
            _summary_arrays(f"reference_{grouping}_{name}", values, out)
        for name, values in scalars.items():
            out[f"reference_{grouping}_{name}"] = values
        out[f"reference_{grouping}_available"] = available

    # -- nucleus control and matched, in one pair-indexed layout
    pairs = [
        job for job in result["jobs"]
        if job.kind == "nucleus" and (
            "nucleus", job.loss_index, job.sampling_index, job.design
        ) in collected
    ]
    if pairs:
        sampling = np.asarray(inputs.sampling_temperatures, dtype=np.float64)
        count = len(pairs)
        out["nucleus_sampling_temperatures"] = np.asarray(
            [sampling[job.sampling_index] for job in pairs], dtype=np.float64
        )
        out["nucleus_loss_temperatures"] = np.asarray(
            [loss[job.loss_index] for job in pairs], dtype=np.float64
        )
        # Fixed-width Unicode, never object dtype: the archive must load with
        # allow_pickle=False.
        out["nucleus_design"] = np.asarray(
            [job.design for job in pairs], dtype="<U8"
        )
        buffers = {
            name: np.full((count, num_maps), np.nan)
            for name in ("within", "between", "delta", "null_delta_mean")
        }
        support_names = (
            "num_represented", "num_qualifying", "num_singletons",
            "singleton_fraction", "positions_in_qualifying",
            "fraction_positions_in_qualifying", "largest_class", "median_class",
            "num_within_pairs",
        )
        support = {name: np.full(count, np.nan) for name in support_names}
        nulls = {
            name: np.full(count, np.nan)
            for name in ("null_delta_std", "null_delta_low", "null_delta_high")
        }
        available = np.zeros(count, dtype=bool)
        for slot, job in enumerate(pairs):
            entries = collected[
                ("nucleus", job.loss_index, job.sampling_index, job.design)
            ]
            if not entries[0].get("available"):
                continue
            available[slot] = True
            buffers["within"][slot] = _stack(entries, "within")
            buffers["between"][slot] = _stack(entries, "between")
            buffers["delta"][slot] = buffers["within"][slot] - buffers["between"][slot]
            buffers["null_delta_mean"][slot] = _stack(entries, "null", "delta_mean")
            for name in nulls:
                nulls[name][slot] = float(
                    np.mean(_stack(entries, "null", name.replace("null_", "")))
                )
            for name in support_names:
                support[name][slot] = entries[0]["support"][name]
            if "display" in entries[0]:
                out[f"nucleus_display_classes_{slot}"] = entries[0]["display"]["classes"]
                out[f"nucleus_display_matrix_{slot}"] = entries[0]["display"]["matrix"]
                out[f"nucleus_display_counts_{slot}"] = entries[0]["display"]["counts"]
        for name, values in buffers.items():
            _summary_arrays(f"nucleus_{name}", values, out)
        for name, values in nulls.items():
            out[f"nucleus_{name}"] = values
        for name, values in support.items():
            out[f"nucleus_support_{name}"] = values
        out["nucleus_available"] = available

    # -- cross-partition
    cross_key = next(
        (key for key in collected if key[0] == "cross"), None
    )
    if cross_key is not None:
        entries = collected[cross_key]
        for name in ("c_same", "c_different", "delta_cross"):
            _summary_arrays(f"cross_{name}", _stack(entries, name), out)
        first = entries[0]
        out["cross_num_same_pairs"] = np.asarray(first["num_same_pairs"], np.int64)
        out["cross_num_different_pairs"] = np.asarray(
            first["num_different_pairs"], np.int64
        )
        out["cross_num_true_positive_positions"] = np.asarray(
            first["num_true_positive_positions"], np.int64
        )
        out["cross_contingency_targets"] = first["contingency"]["targets"]
        out["cross_contingency_greedy"] = first["contingency"]["greedy"]
        out["cross_contingency_counts"] = first["contingency"]["counts"]
        for name in ("delta_mean", "delta_std", "delta_low", "delta_high"):
            out[f"cross_null_{name}"] = np.asarray(
                float(np.mean([entry["null"][name] for entry in entries])),
                dtype=np.float64,
            )
        out["cross_null_permutations"] = np.asarray(
            first["null"]["permutations"], dtype=np.int64
        )
        out["cross_null_seed"] = np.asarray(first["null"]["seed"], dtype=np.int64)

        mixture = first["mixture"]
        out["cross_mixture_tokens"] = mixture["tokens"]
        out["cross_mixture_support"] = mixture["support"]
        out["cross_mixture_similarity"] = mixture["similarity"]
        out["cross_mixture_residual_norm"] = mixture["residual_norm"]
        for name in ("median_similarity", "iqr_similarity",
                     "support_weighted_similarity"):
            _summary_arrays(
                f"cross_mixture_{name}",
                np.asarray([entry["mixture"][name] for entry in entries]),
                out,
            )

        display = first["display"]
        out["cross_displayed_classes"] = display["target_classes"]
        out["cross_displayed_matrix"] = display["matrix"]
        out["cross_displayed_contingency"] = display["contingency"]
        out["cross_displayed_target_counts"] = display["target_counts"]
        out["cross_displayed_greedy_counts"] = display["greedy_counts"]

    # -- retained compact factors, when the mode declared any
    if inputs.factor_selections:
        from llm_behavior_lab.analysis.compact_factors import factor_arrays

        out.update(
            factor_arrays(
                inputs.factor_selections,
                result.get("retained", {}),
                cross_by_selection=result.get("retained_cross", {}),
            )
        )

    return out


def _selection_matches(selection, job: JobPlan, inputs: FinalizationInputs) -> bool:
    """Whether a declared retention wants this job's factors."""

    if selection.grouping != job.kind:
        return False
    if abs(
        selection.loss_temperature - float(inputs.loss_temperatures[job.loss_index])
    ) > TEMPERATURE_TOLERANCE:
        return False
    if job.kind != "nucleus":
        return True
    if job.sampling_index is None:
        return False
    measured = float(inputs.sampling_temperatures[job.sampling_index])
    return abs(selection.sampling_temperature - measured) <= TEMPERATURE_TOLERANCE
