"""Reading finalized metrics in the shapes the figures already expect.

The figures were written against the dictionaries the row-based analyses return.
A ``metrics_only`` record has the same numbers and none of the rows, so rather
than rewrite four figures against a second vocabulary, this adapts the artifact
back into those shapes. The plotting code then does not change at all, which is
the point: a figure that renders from a v12 record and a figure that renders
from a v13 artifact must be the *same* figure, or the two are not comparable.

Nothing here computes a statistic. Every value is read out of the artifact; if a
quantity is missing the view says so rather than deriving it, because deriving it
would mean recomputing from rows that no longer exist -- and silently falling
back to that is the specific failure this whole change exists to prevent.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "clustering_view",
    "cross_partition_view",
    "nucleus_view",
    "sanity_view",
]


class MetricsUnavailable(RuntimeError):
    """A finalized record was asked for something its artifact does not carry.

    Raised rather than falling through to a row-based computation. On a
    ``metrics_only`` record there are no rows to fall back to, so a silent
    fallback would either crash far from the cause or -- worse -- quietly
    recompute from a *different* record's arrays.
    """


def _index_of(metrics: Any, loss_temperature: float | None) -> int:
    """Row of the reference axis for one loss temperature."""

    temperatures = np.asarray(metrics["reference_loss_temperatures"], dtype=np.float64)
    target = 1.0 if loss_temperature is None else float(loss_temperature)
    matches = np.flatnonzero(np.abs(temperatures - target) <= 1e-6)
    if matches.size != 1:
        listed = ", ".join(f"{value:g}" for value in temperatures)
        raise MetricsUnavailable(
            f"No finalized metrics at loss temperature T_g = {target:g}; the "
            f"artifact carries {listed}. A reference from a different gradient "
            "field describes a different geometry, so it is not substituted."
        )
    return int(matches[0])


def clustering_view(
    metrics: Any, grouping: str, *, loss_temperature: float | None = None
) -> dict[str, Any]:
    """Figure 20's per-grouping result, read from the artifact.

    Returns the same keys :func:`gradient_clustering` does -- ``population``,
    ``null``, ``display`` -- so the figure is unchanged.
    """

    if grouping not in ("target", "greedy"):
        raise ValueError(f"grouping must be target or greedy; got {grouping!r}.")
    index = _index_of(metrics, loss_temperature)
    prefix = f"reference_{grouping}"
    if not bool(np.asarray(metrics[f"{prefix}_available"])[index]):
        raise MetricsUnavailable(
            f"The {grouping} grouping has no finalized result at this loss "
            "temperature; it was not measurable in the run that produced it."
        )

    def value(name: str) -> float:
        return float(np.asarray(metrics[f"{prefix}_{name}"])[index])

    display_key = f"display_{grouping}_matrix"
    if display_key not in metrics:
        raise MetricsUnavailable(
            f"The artifact carries no display heatmap for the {grouping} "
            "grouping, so figure 20 cannot be drawn from it."
        )

    counts = np.asarray(metrics[f"display_{grouping}_counts"])
    return {
        "grouping": grouping,
        "min_support": int(np.asarray(metrics["min_support"])),
        "num_positions": int(value("num_positions")),
        "num_positions_excluded": int(
            np.asarray(metrics["num_positions_excluded"])[index]
        ),
        "sketch_dimension": int(np.asarray(metrics["sketch_dimension"]))
        if "sketch_dimension" in metrics
        else int(np.asarray(metrics[display_key]).shape[0]),
        "loss_temperature": loss_temperature,
        "population": {
            "classes": np.asarray(metrics[f"display_{grouping}_classes"]),
            "counts": counts,
            "num_classes": int(value("num_classes")),
            "num_represented": int(value("num_represented")),
            "num_positions": int(value("num_positions")),
            "within": value("within"),
            "between": value("between"),
            "delta": value("delta"),
            "num_within_pairs": int(value("num_within_pairs")),
            "num_between_pairs": int(value("num_between_pairs")),
        },
        "null": {
            "permutations": int(np.asarray(metrics["permutations"])),
            "seed": int(np.asarray(metrics["permutation_seed"])),
            "delta_mean": value("null_delta_mean"),
            "delta_std": value("null_delta_std"),
            "delta_low": value("null_delta_low"),
            "delta_high": value("null_delta_high"),
            "within_mean": value("within"),
            "between_mean": value("between"),
        },
        "display": {
            "classes": np.asarray(metrics[f"display_{grouping}_classes"]),
            "counts": counts,
            "matrix": np.asarray(metrics[display_key]),
            "within_by_class": np.asarray(
                metrics[f"display_{grouping}_within_by_class"]
            ),
            "num_classes": int(counts.size),
            "selection": "finalized display subset",
        },
        "permutation_seed": int(np.asarray(metrics["permutation_seed"])),
    }


def clustering_spread(metrics: Any, grouping: str, *, loss_temperature=None):
    """The across-map spread figure 20 annotates with, or ``None`` at one map."""

    if not bool(np.asarray(metrics["uncertainty_available"])):
        return None
    index = _index_of(metrics, loss_temperature)
    maps = int(np.asarray(metrics["map_count"]))
    spread = {}
    for name in ("within", "between", "delta"):
        prefix = f"reference_{grouping}_{name}"
        spread[name] = {
            "map_mean": float(np.asarray(metrics[prefix])[index]),
            "sample_sd": float(np.asarray(metrics[f"{prefix}_sd"])[index]),
            "standard_error": float(np.asarray(metrics[f"{prefix}_se"])[index]),
            "degrees_of_freedom": maps - 1,
            "uncertainty_available": True,
            "map_count": maps,
        }
    return spread


def nucleus_view(metrics: Any, *, design: str = "control") -> dict[str, Any]:
    """Figure 22's result dict, read from the artifact.

    Mirrors what :func:`load_nucleus_clustering_artifact` returns, so figure 22
    consumes one shape whichever source it came from.
    """

    designs = np.asarray(metrics["nucleus_design"])
    slots = np.flatnonzero(designs == design)
    if slots.size == 0:
        raise MetricsUnavailable(
            f"The artifact carries no {design!r} nucleus points. Control and "
            "matched are different designs -- one is not a substitute for the "
            "other."
        )

    sampling = np.asarray(metrics["nucleus_sampling_temperatures"])[slots]
    loss = np.asarray(metrics["nucleus_loss_temperatures"])[slots]
    order = np.argsort(sampling, kind="stable")
    slots, sampling, loss = slots[order], sampling[order], loss[order]

    by_temperature = []
    for position, slot in enumerate(slots):
        support = {
            name: float(
                np.asarray(metrics[f"nucleus_support_{name}"])[slot]
            )
            for name in (
                "num_represented", "num_qualifying", "num_singletons",
                "singleton_fraction", "positions_in_qualifying",
                "fraction_positions_in_qualifying", "largest_class",
                "median_class", "num_within_pairs",
            )
        }
        entry = {
            "temperature": float(sampling[position]),
            "loss_temperature": float(loss[position]),
            # The per-point fields the figure annotates with. Named as the v12
            # artifact names them, so one figure reads either source.
            "num_positions": int(
                np.asarray(metrics["reference_target_num_positions"])[
                    _index_of(metrics, None)
                ]
            ),
            "sketch_dimension": int(
                np.asarray(metrics["display_target_matrix"]).shape[0]
            ),
            "min_support": int(np.asarray(metrics["min_support"])),
            "population": {"delta": float(np.asarray(metrics["nucleus_delta"])[slot])},
            "null": {
                "delta_mean": float(
                    np.asarray(metrics["nucleus_null_delta_mean"])[slot]
                ),
                "delta_low": float(
                    np.asarray(metrics["nucleus_null_delta_low"])[slot]
                ),
                "delta_high": float(
                    np.asarray(metrics["nucleus_null_delta_high"])[slot]
                ),
            },
            "support": support,
        }
        key = f"nucleus_display_matrix_{slot}"
        if key in metrics:
            entry["display"] = {
                "classes": np.asarray(metrics[f"nucleus_display_classes_{slot}"]),
                "matrix": np.asarray(metrics[key]),
            }
        by_temperature.append(entry)

    references = {
        grouping: clustering_view(metrics, grouping)["population"]
        for grouping in ("target", "greedy")
    }
    temperatures = np.asarray(metrics["reference_loss_temperatures"])
    references_by_loss = {
        float(value): {
            grouping: {
                "population": clustering_view(
                    metrics, grouping, loss_temperature=float(value)
                )["population"]
            }
            for grouping in ("target", "greedy")
        }
        for value in temperatures
    }

    return {
        "sampling_temperatures": sampling,
        # Emitted for parity with the v12 artifact loader, which supplies both.
        "temperatures": sampling,
        "loss_temperatures": loss,
        "by_temperature": by_temperature,
        "references": {name: {"population": block} for name, block in references.items()},
        "references_by_loss_temperature": references_by_loss,
        "min_support": int(np.asarray(metrics["min_support"])),
        "permutations": int(np.asarray(metrics["permutations"])),
        "map_count": int(np.asarray(metrics["map_count"])),
        "uncertainty_available": bool(np.asarray(metrics["uncertainty_available"])),
        "uncertainty": {
            "delta_se": np.asarray(metrics["nucleus_delta_se"])[slots],
            "delta_sd": np.asarray(metrics["nucleus_delta_sd"])[slots],
        },
    }


def cross_partition_view(metrics: Any) -> dict[str, Any]:
    """Figure 24's inputs, read from the artifact instead of recomputed."""

    if "cross_delta_cross" not in metrics:
        raise MetricsUnavailable(
            "The artifact carries no cross-partition results, so figure 24 "
            "cannot be drawn from it."
        )
    return {
        "pooled": {
            "c_same": float(np.asarray(metrics["cross_c_same"])),
            "c_different": float(np.asarray(metrics["cross_c_different"])),
            "delta_cross": float(np.asarray(metrics["cross_delta_cross"])),
            "num_same_pairs": int(np.asarray(metrics["cross_num_same_pairs"])),
            "num_different_pairs": int(
                np.asarray(metrics["cross_num_different_pairs"])
            ),
            "num_true_positive_positions": int(
                np.asarray(metrics["cross_num_true_positive_positions"])
            ),
        },
        "displayed": {
            "classes": np.asarray(metrics["cross_displayed_classes"]),
            "matrix": np.asarray(metrics["cross_displayed_matrix"]),
            "contingency": np.asarray(metrics["cross_displayed_contingency"]),
            "target_counts": np.asarray(metrics["cross_displayed_target_counts"]),
            "greedy_counts": np.asarray(metrics["cross_displayed_greedy_counts"]),
        },
        "contingency": {
            "targets": np.asarray(metrics["cross_contingency_targets"]),
            "greedy": np.asarray(metrics["cross_contingency_greedy"]),
            "counts": np.asarray(metrics["cross_contingency_counts"]),
        },
        "null": {
            "permutations": int(np.asarray(metrics["cross_null_permutations"])),
            "seed": int(np.asarray(metrics["cross_null_seed"])),
            "delta_mean": float(np.asarray(metrics["cross_null_delta_mean"])),
            "delta_std": float(np.asarray(metrics["cross_null_delta_std"])),
            "delta_low": float(np.asarray(metrics["cross_null_delta_low"])),
            "delta_high": float(np.asarray(metrics["cross_null_delta_high"])),
        },
        "mixture": {
            "tokens": np.asarray(metrics["cross_mixture_tokens"]),
            "support": np.asarray(metrics["cross_mixture_support"]),
            "similarity": np.asarray(metrics["cross_mixture_similarity"]),
            "residual_norm": np.asarray(metrics["cross_mixture_residual_norm"]),
            "median_similarity": float(
                np.asarray(metrics["cross_mixture_median_similarity"])
            ),
            "iqr_similarity": float(
                np.asarray(metrics["cross_mixture_iqr_similarity"])
            ),
            "support_weighted_similarity": float(
                np.asarray(metrics["cross_mixture_support_weighted_similarity"])
            ),
            "num_classes": int(np.asarray(metrics["cross_mixture_similarity"]).size),
            "metric": (
                "projected-space cosine between sketch-space mean directions"
            ),
        },
        "map_count": int(np.asarray(metrics["map_count"])),
        "min_support": int(np.asarray(metrics["min_support"])),
        "num_positions": int(
            np.asarray(metrics["reference_target_num_positions"])[
                _index_of(metrics, None)
            ]
        ),
    }


def sanity_view(metrics: Any) -> dict[str, Any]:
    """Figure 23's report, read from the authoritative artifact.

    The v13 sanity summary lives inside the metrics artifact rather than in a
    second file, so there is exactly one v13 artifact to keep in step. The
    legacy ``sanity/countsketch_fidelity.npz`` remains the v12 source.
    """

    if "sanity_exact_cosine_matrix" not in metrics:
        raise MetricsUnavailable(
            "This run carries no CountSketch fidelity summary; the sanity check "
            "is off by default and must be requested."
        )
    exact = np.asarray(metrics["sanity_exact_cosine_matrix"])
    production = np.asarray(metrics["sanity_production_cosine_matrix"])
    report = {
        "num_gradients": int(exact.shape[0]),
        "gradient_dtype": "float32",
        "accumulation_dtype": "float64",
        "exact_cosines": exact,
        "production_cosines": production,
        "selected_position_indices": np.asarray(
            metrics["sanity_selected_position_indices"]
        ),
        "selected_target_ids": np.asarray(metrics["sanity_selected_target_ids"]),
        "selected_greedy_ids": np.asarray(metrics["sanity_selected_greedy_ids"]),
        "production_dimension": int(np.asarray(metrics["sanity_dimension"])),
        "production_seed": int(np.asarray(metrics["sanity_seed"])),
        "map_semantics": "production",
        "sensitivity": [],
    }
    return report
