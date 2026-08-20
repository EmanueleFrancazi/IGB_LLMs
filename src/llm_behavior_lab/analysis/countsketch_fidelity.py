"""How well does the CountSketch reproduce true gradient directions?

Figure 20 reports estimated cosines between per-position gradients, some of them
large. Those estimates come from a projection, so the projection itself needs
measuring against the thing it approximates. That is what this module does, and
it is deliberately a **methodological check**, not a scientific result: it says
nothing about whether gradients cluster, only about whether the instrument that
reports clustering can be trusted at ``K = 512``.

Three questions are kept apart throughout, because conflating them would be easy
and wrong:

*empirical clustering*
    do real token subgroups share gradient directions? (figure 20)
*label-permutation null*
    would that separation survive shuffling the labels? (``gradient_clustering``)
*projection error*
    does the sketch estimate the cosine of the full gradients? (here)

A note on the word **exact**. It means "computed from the full, unprojected
gradient vectors" -- not infinite precision. Captured gradients are float32, and
every dot product here accumulates in float64, which is stated rather than
implied.

The estimator under test is production's, unchanged::

    cos_hat(a, b) = <S(g_a), S(g_b)> / (||g_a|| ||g_b||)

with one fixed sketch per seed, applied to every gradient, and the **exact**
norms in the denominator.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

__all__ = [
    "ALTERNATE_SKETCH_SEEDS",
    "SENSITIVITY_DIMENSIONS",
    "count_sketch_matrix",
    "cosine_from_gram",
    "fidelity_report",
    "pair_type_breakdown",
    "select_sanity_positions",
    "sketch_estimated_cosines",
]

#: Deterministic alternate realizations for the offline multi-seed check. The
#: production seed is deliberately excluded: it is reported on its own, because
#: the experiment used that one realization and its error is not a draw from a
#: distribution the reader gets to average over.
ALTERNATE_SKETCH_SEEDS = (101, 202, 303, 404, 505, 606, 707, 808)

#: Sketch widths compared offline, production's among them.
SENSITIVITY_DIMENSIONS = (128, 256, 512, 1024)


def select_sanity_positions(
    target_ids: np.ndarray,
    greedy_ids: np.ndarray,
    *,
    count: int = 8,
) -> np.ndarray:
    """Choose a small deterministic position subset with useful pair structure.

    Selection uses **only** labels and position order. It never looks at a
    sketch value, an exact cosine, or any measure of directional coherence --
    doing so would choose the positions by the answer and make the validation
    circular. A test asserts that perturbing the sketches cannot move the
    selection.

    The rule, in order:

    1. take the most frequent target class, and keep its two lowest position
       indices -- this guarantees a same-target pair;
    2. take the most frequent greedy class, and keep its two lowest position
       indices -- this guarantees a same-greedy pair;
    3. fill the remainder from the lowest-indexed positions whose target and
       greedy labels are both unused so far, giving pairs that share neither
       label;
    4. if that is still short, fill by ascending position index.

    Ties in class frequency break toward the smaller token ID, so the result is
    identical on every machine and every rerun.
    """

    target_ids = np.asarray(target_ids, dtype=np.int64)
    greedy_ids = np.asarray(greedy_ids, dtype=np.int64)
    if target_ids.shape != greedy_ids.shape:
        raise ValueError("target and greedy label arrays must have equal length.")

    def most_frequent(labels: np.ndarray) -> int:
        values, counts = np.unique(labels, return_counts=True)
        # -counts then values: most frequent first, smallest ID on a tie.
        return int(values[np.lexsort((values, -counts))[0]])

    chosen: list[int] = []

    def take(candidates: np.ndarray, how_many: int) -> None:
        for position in candidates:
            if len(chosen) >= count:
                return
            if int(position) not in chosen:
                chosen.append(int(position))
                how_many -= 1
                if how_many <= 0:
                    return

    take(np.flatnonzero(target_ids == most_frequent(target_ids)), 2)
    take(np.flatnonzero(greedy_ids == most_frequent(greedy_ids)), 2)

    used_targets = {int(target_ids[p]) for p in chosen}
    used_greedy = {int(greedy_ids[p]) for p in chosen}
    fresh = [
        index
        for index in range(target_ids.size)
        if int(target_ids[index]) not in used_targets
        and int(greedy_ids[index]) not in used_greedy
    ]
    take(np.asarray(fresh, dtype=np.int64), count - len(chosen))
    take(np.arange(target_ids.size), count - len(chosen))

    return np.array(sorted(chosen), dtype=np.int64)


def count_sketch_matrix(
    num_parameters: int, dimension: int, seed: int, *, tensor_sizes: Sequence[int] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild one fixed CountSketch map as bucket and sign vectors.

    Mirrors the production construction: a generator seeded per parameter tensor
    from ``seed + 1000003 * index``, drawing buckets and signs once. Passing the
    production ``tensor_sizes`` therefore reproduces the production projection
    exactly; omitting them gives a single-block map, which is all the offline
    alternate-seed and ``K``-sweep analyses need.
    """

    sizes = list(tensor_sizes) if tensor_sizes else [int(num_parameters)]
    if sum(sizes) != int(num_parameters):
        raise ValueError("tensor_sizes must sum to num_parameters.")

    buckets, signs = [], []
    for index, size in enumerate(sizes):
        generator = np.random.default_rng(seed + 1000003 * index)
        buckets.append(generator.integers(0, dimension, size=size, dtype=np.int64))
        signs.append(generator.integers(0, 2, size=size).astype(np.float64) * 2.0 - 1.0)
    return np.concatenate(buckets), np.concatenate(signs)


def _project(gradients: np.ndarray, buckets: np.ndarray, signs: np.ndarray, dimension: int):
    """Apply one fixed map to every gradient: ``S(g)[k] = sum_{h(j)=k} s(j) g_j``."""

    out = np.zeros((gradients.shape[0], dimension), dtype=np.float64)
    for row in range(gradients.shape[0]):
        np.add.at(out[row], buckets, gradients[row].astype(np.float64) * signs)
    return out


def cosine_from_gram(gradients: np.ndarray, norms: np.ndarray) -> np.ndarray:
    """Full-gradient cosine matrix, accumulated in float64.

    "Exact" here means unprojected, not infinitely precise: the stored vectors
    are float32 and the products below are float64.
    """

    values = np.asarray(gradients, dtype=np.float64)
    norms = np.asarray(norms, dtype=np.float64)
    return (values @ values.T) / np.outer(norms, norms)


def sketch_estimated_cosines(
    gradients: np.ndarray, norms: np.ndarray, *, dimension: int, seed: int,
    tensor_sizes: Sequence[int] | None = None,
) -> np.ndarray:
    """The production estimator: sketched inner products over **exact** norms."""

    buckets, signs = count_sketch_matrix(
        gradients.shape[1], dimension, seed, tensor_sizes=tensor_sizes
    )
    projected = _project(gradients, buckets, signs, dimension)
    norms = np.asarray(norms, dtype=np.float64)
    return (projected @ projected.T) / np.outer(norms, norms)


def _errors(exact: np.ndarray, estimated: np.ndarray) -> dict[str, Any]:
    """Error summary over unique non-self pairs."""

    size = exact.shape[0]
    upper = np.triu_indices(size, k=1)
    difference = estimated[upper] - exact[upper]
    absolute = np.abs(difference)
    correlation = float("nan")
    if difference.size >= 3 and np.std(exact[upper]) > 0 and np.std(estimated[upper]) > 0:
        correlation = float(np.corrcoef(exact[upper], estimated[upper])[0, 1])
    return {
        "num_pairs": int(difference.size),
        "exact_min": float(exact[upper].min()) if difference.size else float("nan"),
        "exact_max": float(exact[upper].max()) if difference.size else float("nan"),
        "exact_median": float(np.median(exact[upper])) if difference.size else float("nan"),
        "estimated_min": float(estimated[upper].min()) if difference.size else float("nan"),
        "estimated_max": float(estimated[upper].max()) if difference.size else float("nan"),
        "mean_signed_error": float(difference.mean()) if difference.size else float("nan"),
        "mean_absolute_error": float(absolute.mean()) if difference.size else float("nan"),
        "median_absolute_error": float(np.median(absolute)) if difference.size else float("nan"),
        "rmse": float(np.sqrt((difference ** 2).mean())) if difference.size else float("nan"),
        "max_absolute_error": float(absolute.max()) if difference.size else float("nan"),
        "pearson_correlation": correlation,
    }


def pair_type_breakdown(
    exact: np.ndarray, estimated: np.ndarray,
    target_ids: np.ndarray, greedy_ids: np.ndarray,
) -> dict[str, dict[str, Any]]:
    """Errors split by whether a pair shares its target and/or greedy label.

    A non-overlapping decomposition of the unique non-self pairs. Descriptive
    only: with a handful of positions some categories hold one or two pairs, and
    nothing inferential should be read into those.
    """

    size = exact.shape[0]
    rows, columns = np.triu_indices(size, k=1)
    same_target = np.asarray(target_ids)[rows] == np.asarray(target_ids)[columns]
    same_greedy = np.asarray(greedy_ids)[rows] == np.asarray(greedy_ids)[columns]

    categories = {
        "same_target_and_greedy": same_target & same_greedy,
        "same_target_only": same_target & ~same_greedy,
        "same_greedy_only": ~same_target & same_greedy,
        "different_both": ~same_target & ~same_greedy,
    }
    out = {}
    for name, mask in categories.items():
        difference = estimated[rows, columns][mask] - exact[rows, columns][mask]
        out[name] = {
            "num_pairs": int(mask.sum()),
            "mean_signed_error": float(difference.mean()) if mask.any() else float("nan"),
            "mean_absolute_error": (
                float(np.abs(difference).mean()) if mask.any() else float("nan")
            ),
        }
    return out


def fidelity_report(
    gradients: np.ndarray,
    norms: np.ndarray,
    target_ids: np.ndarray,
    greedy_ids: np.ndarray,
    *,
    production_dimension: int,
    production_seed: int,
    tensor_sizes: Sequence[int] | None = None,
    alternate_seeds: Sequence[int] = ALTERNATE_SKETCH_SEEDS,
    sensitivity_dimensions: Sequence[int] = SENSITIVITY_DIMENSIONS,
) -> dict[str, Any]:
    """Everything the sanity check reports, computed offline from stored vectors.

    No model execution of any kind happens here: the alternate seeds and the
    ``K`` sweep reuse the same captured gradients, which is the entire reason for
    capturing them.
    """

    exact = cosine_from_gram(gradients, norms)
    production = sketch_estimated_cosines(
        gradients, norms, dimension=production_dimension,
        seed=production_seed, tensor_sizes=tensor_sizes,
    )

    alternates = []
    for seed in alternate_seeds:
        estimated = sketch_estimated_cosines(
            gradients, norms, dimension=production_dimension, seed=seed,
        )
        summary = _errors(exact, estimated)
        summary["seed"] = int(seed)
        alternates.append(summary)

    sensitivity = []
    for dimension in sensitivity_dimensions:
        per_seed = [
            _errors(
                exact,
                sketch_estimated_cosines(
                    gradients, norms, dimension=dimension, seed=seed
                ),
            )
            for seed in alternate_seeds[:4]
        ]
        sensitivity.append({
            "dimension": int(dimension),
            "num_seeds": len(per_seed),
            "mean_absolute_error": float(
                np.mean([entry["mean_absolute_error"] for entry in per_seed])
            ),
            "rmse": float(np.mean([entry["rmse"] for entry in per_seed])),
            "max_absolute_error": float(
                np.max([entry["max_absolute_error"] for entry in per_seed])
            ),
        })

    signed = [entry["mean_signed_error"] for entry in alternates]
    return {
        "num_gradients": int(gradients.shape[0]),
        "gradient_dtype": str(np.asarray(gradients).dtype),
        "accumulation_dtype": "float64",
        "production_dimension": int(production_dimension),
        "production_seed": int(production_seed),
        "exact_cosines": exact,
        "production_cosines": production,
        "production": _errors(exact, production),
        "pair_types": pair_type_breakdown(exact, production, target_ids, greedy_ids),
        "alternate_seeds": alternates,
        "alternate_summary": {
            "num_seeds": len(alternates),
            "signed_error_mean": float(np.mean(signed)) if signed else float("nan"),
            "signed_error_min": float(np.min(signed)) if signed else float("nan"),
            "signed_error_max": float(np.max(signed)) if signed else float("nan"),
            "mean_absolute_error": float(
                np.mean([entry["mean_absolute_error"] for entry in alternates])
            ) if alternates else float("nan"),
            "rmse": float(np.mean([entry["rmse"] for entry in alternates]))
            if alternates else float("nan"),
        },
        "sensitivity": sensitivity,
    }
