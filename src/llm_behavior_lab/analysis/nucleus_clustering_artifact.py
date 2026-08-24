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

from llm_behavior_lab.analysis.temperature_pairs import (
    describe_pairing,
    validate_pair_aligned,
)

__all__ = ["ARTIFACT_NAME", "load_nucleus_clustering_artifact"]

#: Where the writer leaves its output, relative to the record directory.
ARTIFACT_NAME = "nucleus_gradient_clustering.npz"

#: Loss temperature attributed to an artifact written before ``T_s`` and ``T_g``
#: were stored separately. Every such artifact came from the canonical ``T = 1``
#: gradients -- that is what figure 22 has always reported in its subtitle -- so
#: this states the established reading rather than inventing a new one. It is
#: never written back: the stored artifact is left exactly as it was.
HISTORICAL_LOSS_TEMPERATURE = 1.0


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
        # An artifact from before the pair split stores only "temperatures",
        # which always meant the sampling temperatures.
        explicit = "sampling_temperatures" in data.files
        if explicit:
            sampling = np.asarray(data["sampling_temperatures"], dtype=float)
            loss = np.asarray(data["loss_temperatures"], dtype=float)
        else:
            sampling = np.asarray(data["temperatures"], dtype=float)
            loss = np.full(sampling.shape, HISTORICAL_LOSS_TEMPERATURE, dtype=float)
        validate_pair_aligned(
            sampling, loss,
            delta=data["delta"], within=data["within"], between=data["between"],
            null_delta_mean=data["null_delta_mean"],
        )
        temperatures = tuple(float(value) for value in sampling)
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
                    "sampling_temperature": temperature,
                    "loss_temperature": float(loss[index]),
                    "grouping": (
                        f"nucleus T_s={temperature:g} T_g={float(loss[index]):g}"
                    ),
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
        "sampling_temperatures": sampling,
        "loss_temperatures": loss,
        "pair_metadata": "explicit" if explicit else "historical_fallback",
        "pairing": describe_pairing(sampling, loss),
        # Historical key, always the sampling temperatures.
        "temperatures": temperatures,
        "by_temperature": by_temperature,
        "references": references,
        "nucleus_labels": labels,
        "min_support": min_support,
        "permutations": permutations,
        "permutation_seed": permutation_seed,
    }
