#!/usr/bin/env python3
"""Derive the target-versus-greedy cross-partition statistics for a record.

The initialization-distribution record persists the sketches and both label
vectors; everything figure 24 draws follows from those by NumPy alone. That is
deliberate: the derivation must never require a second backward pass, and the
existing record must never be modified. This script therefore *reads* the record
and writes a separate artifact beside it, adding no array to the record and
leaving its version untouched.

The permutation null dominates the cost -- 256 draws over a full contingency --
so it is computed once here rather than each time a figure is drawn.

Usage::

    python scripts/write_cross_partition_artifact.py <run>/analyses

writes ``<run>/analyses/gradient_cross_partition.npz``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Allow running from the repository root before editable installation.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from llm_behavior_lab.analysis import load_record  # noqa: E402
from llm_behavior_lab.analysis.gradient_clustering import (  # noqa: E402
    unit_sketches,
    unit_sketches_per_map,
)
from llm_behavior_lab.analysis.nucleus_clustering_artifact import (  # noqa: E402
    summary_arrays,
    uncertainty_arrays,
)
from llm_behavior_lab.analysis.sketch_estimator import (  # noqa: E402
    ensemble_summary,
)
from llm_behavior_lab.analysis.gradient_cross_partition import (  # noqa: E402
    cross_partition_per_map,
    contingency_summary,
    cross_identity_null,
    cross_partition_matrix,
    mixture_reconstruction,
    pooled_cross_statistic,
)

#: Name of the derived artifact, inside the record's own directory.
ARTIFACT_NAME = "gradient_cross_partition.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "record_dir", type=Path, help="The run's analyses directory."
    )
    parser.add_argument(
        "--name",
        default="initialization_distribution",
        help="Record stem inside the analyses directory.",
    )
    parser.add_argument(
        "--display-classes",
        type=int,
        default=40,
        help="Tokens to keep in the stored heatmap, by min(target, greedy) support.",
    )
    parser.add_argument(
        "--min-support",
        type=int,
        default=2,
        help="Support a token needs in BOTH roles before it can be displayed.",
    )
    parser.add_argument(
        "--permutations", type=int, default=256, help="Identity-permutation draws."
    )
    return parser.parse_args()


def _cross_partition_arrays(
    record, pooled, null, contingency, mixture, drawn,
    shown, shown_target_counts, shown_greedy_counts, targets, greedy,
    num_excluded: int, min_support: int,
) -> dict:
    """Map completed analysis results onto the artifact's stored fields.

    Extracted from ``main`` for the same reason as the nucleus seam: the mapping
    is where a field can be wired to the wrong statistic, and it must be testable
    without a tokenizer or a model. ``main`` calls this and nothing else builds
    the payload.
    """

    # -- additive uncertainty ------------------------------------------------
    #
    # Point estimates above are untouched: `pooled` and `mixture` read the
    # ensemble embedding and remain the production values. What follows is the
    # per-map spread behind them, so map uncertainty is not discarded here --
    # this is the boundary where the scientific output is written.
    #
    # At M = 1 the per-map value *is* the point estimate, so it is reshaped
    # rather than recomputed.
    map_count = int(record.sketch_map_count)
    if map_count == 1:
        spread = {
            "c_same_per_map": np.asarray([pooled["c_same"]], dtype=float),
            "c_different_per_map": np.asarray([pooled["c_different"]], dtype=float),
            "delta_cross_per_map": np.asarray([pooled["delta_cross"]], dtype=float),
            "mixture_median_similarity_per_map": np.asarray(
                [mixture["median_similarity"]], dtype=float
            ),
        }
    else:
        per_map_rows, per_map_usable = unit_sketches_per_map(record)
        spread_result = cross_partition_per_map(
            per_map_rows[per_map_usable], targets, greedy,
            min_support=min_support,
        )
        spread = {
            key: np.asarray(spread_result[key], dtype=float)
            for key in (
                "c_same_per_map",
                "c_different_per_map",
                "delta_cross_per_map",
                "mixture_median_similarity_per_map",
            )
        }

    uncertainty: dict = dict(spread)
    for name in ("c_same", "c_different", "delta_cross"):
        uncertainty.update(
            summary_arrays(name, ensemble_summary(spread[f"{name}_per_map"]))
        )
    mixture_summary = ensemble_summary(spread["mixture_median_similarity_per_map"])
    uncertainty.update(summary_arrays("mixture_median_similarity", mixture_summary))
    # The arithmetic mean of the per-map *ratios*, kept apart from the plug-in
    # `mixture_median_similarity` above. A ratio of averages is not the average
    # of ratios, and the plug-in value stays the production point estimate.
    # The plug-in scalar itself, stored so the per-map fields beside it are
    # interpretable from the archive alone. It was previously only printed in
    # the run summary, never persisted.
    uncertainty["mixture_median_similarity"] = np.asarray(
        mixture["median_similarity"], dtype=float
    )
    uncertainty["mixture_median_similarity_map_mean"] = np.asarray(
        mixture_summary["map_mean"], dtype=float
    )
    uncertainty.update(uncertainty_arrays(map_count))

    arrays = dict(
        displayed_classes=shown,
        displayed_matrix=drawn["matrix"],
        displayed_contingency=drawn["contingency"],
        displayed_target_counts=shown_target_counts,
        displayed_greedy_counts=shown_greedy_counts,
        c_same=pooled["c_same"],
        c_different=pooled["c_different"],
        delta_cross=pooled["delta_cross"],
        **uncertainty,
        num_same_pairs=pooled["num_same_pairs"],
        num_different_pairs=pooled["num_different_pairs"],
        null_delta_mean=null["delta_mean"],
        null_delta_std=null["delta_std"],
        null_delta_low=null["delta_low"],
        null_delta_high=null["delta_high"],
        null_permutations=null["permutations"],
        null_seed=null["seed"],
        mixture_tokens=mixture["tokens"],
        mixture_support=mixture["support"],
        mixture_similarity=mixture["similarity"],
        contingency_counts=contingency["counts"],
        contingency_targets=contingency["target_ids"],
        contingency_greedy=contingency["greedy_ids"],

    )
    return arrays


def _write_cross_partition_artifact(path, arrays: dict):
    """Write the payload as a compressed NPZ. The only place that serializes it."""

    np.savez_compressed(path, **arrays)
    return path


def main() -> None:
    args = parse_args()
    record = load_record(args.record_dir, name=args.name)
    if not record.has_gradient_position_sketches:
        raise SystemExit(
            "This record carries no gradient sketches, so the cross-partition "
            "geometry cannot be derived from it."
        )

    rows, usable = unit_sketches(record)
    targets = np.asarray(record.gradient_position_target_ids, dtype=np.int64)[usable]
    greedy = np.asarray(record.gradient_position_greedy_ids, dtype=np.int64)[usable]
    rows = rows[usable]
    num_excluded = int(usable.size - usable.sum())

    pooled = pooled_cross_statistic(rows, targets, greedy)
    null = cross_identity_null(rows, targets, greedy, permutations=args.permutations)
    contingency = contingency_summary(targets, greedy)
    mixture = mixture_reconstruction(rows, targets, greedy, min_support=args.min_support)

    # Support only, in both roles. Ranking by similarity would choose the
    # conclusion before drawing it.
    classes = pooled["classes"]
    ceiling = int(classes.max()) + 1
    target_counts = np.bincount(targets, minlength=ceiling)[classes]
    greedy_counts = np.bincount(greedy, minlength=ceiling)[classes]
    score = np.minimum(target_counts, greedy_counts)
    eligible = np.flatnonzero(score >= args.min_support)
    order = eligible[np.argsort(-score[eligible], kind="stable")][: args.display_classes]
    shown = np.sort(classes[order])
    # Recomputed against ``shown`` rather than reindexed through ``order``, so
    # the stored counts cannot drift out of step with the stored axis.
    shown_target_counts = np.bincount(targets, minlength=ceiling)[shown]
    shown_greedy_counts = np.bincount(greedy, minlength=ceiling)[shown]
    drawn = cross_partition_matrix(rows, targets, greedy, shown, shown)

    arrays = _cross_partition_arrays(
        record, pooled, null, contingency, mixture, drawn,
        shown, shown_target_counts, shown_greedy_counts, targets, greedy,
        num_excluded, args.min_support,
    )
    destination = args.record_dir / ARTIFACT_NAME
    _write_cross_partition_artifact(destination, arrays)

    summary = {
        "num_positions": int(rows.shape[0]),
        "num_positions_excluded": num_excluded,
        "num_true_positive_positions": contingency["num_true_positive_positions"],
        "true_positive_fraction": contingency["true_positive_fraction"],
        "num_target_classes": contingency["num_target_classes"],
        "num_greedy_classes": contingency["num_greedy_classes"],
        "num_occupied_cells": contingency["num_cells"],
        "c_same": pooled["c_same"],
        "c_different": pooled["c_different"],
        "delta_cross": pooled["delta_cross"],
        "null_delta_mean": null["delta_mean"],
        "null_delta_low": null["delta_low"],
        "null_delta_high": null["delta_high"],
        "null_permutations": null["permutations"],
        "mixture_median_similarity": mixture["median_similarity"],
        "mixture_metric": mixture["metric"],
        "displayed_classes": int(shown.size),
        "min_support": args.min_support,
    }
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    print(f"\nArtifact written: {destination}")


if __name__ == "__main__":
    main()
