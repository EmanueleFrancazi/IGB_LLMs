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
from llm_behavior_lab.analysis.gradient_clustering import unit_sketches  # noqa: E402
from llm_behavior_lab.analysis.gradient_cross_partition import (  # noqa: E402
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

    destination = args.record_dir / ARTIFACT_NAME
    np.savez_compressed(
        destination,
        displayed_classes=shown,
        displayed_matrix=drawn["matrix"],
        displayed_contingency=drawn["contingency"],
        displayed_target_counts=shown_target_counts,
        displayed_greedy_counts=shown_greedy_counts,
        c_same=pooled["c_same"],
        c_different=pooled["c_different"],
        delta_cross=pooled["delta_cross"],
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
