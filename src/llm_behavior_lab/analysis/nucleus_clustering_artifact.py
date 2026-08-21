"""Reading back the cached nucleus-clustering results.

The clustering itself is cheap; recovering *which* position sampled which token
is not, because it needs a forward pass over every evaluation position. So it is
done once by ``scripts/write_nucleus_clustering_artifact.py`` -- which gates the
recovered labels against the recorded histogram before computing anything -- and
cached beside the record.

This module only reads that cache back into the structure
:func:`~llm_behavior_lab.analysis.nucleus_clustering.nucleus_clustering` returns,
so a figure cannot tell whether it was handed fresh results or stored ones. It
performs no gating of its own: an artifact exists only if the gate already
passed, and re-deriving a weaker check here would suggest otherwise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["ARTIFACT_NAME", "load_nucleus_clustering_artifact"]

#: Where the writer leaves its output, relative to the record directory.
ARTIFACT_NAME = "nucleus_gradient_clustering.npz"


def load_nucleus_clustering_artifact(record_dir: str | Path) -> dict[str, Any]:
    """Rebuild the sweep result from ``<record_dir>/nucleus_gradient_clustering.npz``.

    Raises:
        FileNotFoundError: When no artifact has been written for this run. The
            caller decides whether that is fatal; for the whole-set renderer it
            simply means this figure is not part of that record's set.
    """

    path = Path(record_dir) / ARTIFACT_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"No nucleus-clustering artifact at {path}. Write one with "
            "scripts/write_nucleus_clustering_artifact.py, which recovers the "
            "per-position nucleus samples and gates them against the recorded "
            "histogram."
        )

    with np.load(path) as data:
        temperatures = tuple(float(value) for value in data["temperatures"])
        support_fields = (
            "num_represented", "num_qualifying", "num_singletons",
            "singleton_fraction", "positions_in_qualifying",
            "fraction_positions_in_qualifying", "largest_class", "median_class",
            "num_within_pairs", "num_between_pairs",
        )
        by_temperature = []
        for index, temperature in enumerate(temperatures):
            by_temperature.append(
                {
                    "temperature": temperature,
                    "grouping": f"nucleus T={temperature:g}",
                    "population": {
                        "delta": float(data["delta"][index]),
                        "within": float(data["within"][index]),
                        "between": float(data["between"][index]),
                    },
                    "null": {
                        "delta_mean": float(data["null_delta_mean"][index]),
                        "delta_low": float(data["null_delta_low"][index]),
                        "delta_high": float(data["null_delta_high"][index]),
                    },
                    "support": {
                        field: data[f"support_{field}"][index].item()
                        for field in support_fields
                    },
                    "display": {
                        "classes": data[f"display_classes_{index}"],
                        "matrix": data[f"display_matrix_{index}"],
                        "counts": data[f"display_counts_{index}"],
                    },
                    "num_positions": int(data["num_positions"]),
                    "num_positions_excluded": int(data["num_positions_excluded"]),
                    "sketch_dimension": int(data["sketch_dimension"]),
                }
            )
        references = {
            grouping: {
                "population": {"delta": float(data[f"reference_{grouping}_delta"])},
                "null": {
                    "delta_low": float(data[f"reference_{grouping}_null_low"]),
                    "delta_high": float(data[f"reference_{grouping}_null_high"]),
                },
            }
            for grouping in ("target", "greedy")
        }
        labels = data["nucleus_labels"]
        min_support = int(data["min_support"])
        permutations = int(data["permutations"])
        permutation_seed = int(data["permutation_seed"])

    return {
        "temperatures": temperatures,
        "by_temperature": by_temperature,
        "references": references,
        "nucleus_labels": labels,
        "min_support": min_support,
        "permutations": permutations,
        "permutation_seed": permutation_seed,
    }
