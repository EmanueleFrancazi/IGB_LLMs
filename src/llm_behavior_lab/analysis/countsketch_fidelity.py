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
    "deviation_report",
    "estimator_comparison",
    "projected_space_cosines",
    "SANITY_ARTIFACT",
    "fidelity_report",
    "load_fidelity_artifact",
    "sanity_artifact_path",
    "pair_type_breakdown",
    "subgroup_delta",
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
    greedy_ids: np.ndarray | None = None,
    *,
    count: int = 12,
) -> np.ndarray:
    """Choose a small deterministic position subset with useful pair structure.

    ``greedy_ids`` is optional because of *when* selection has to happen. Target
    labels are corpus data and are known before any model runs; greedy labels
    only exist once the forward pass has been done, and by then the loop is
    already streaming. Selecting on targets alone keeps the whole thing to a
    single pass with no gradient recomputed, and the greedy pair composition is
    reported afterwards rather than guaranteed in advance.

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
    if greedy_ids is None:
        greedy_ids = np.full_like(target_ids, -1)
    else:
        greedy_ids = np.asarray(greedy_ids, dtype=np.int64)
        if target_ids.shape != greedy_ids.shape:
            raise ValueError("target and greedy label arrays must have equal length.")
    has_greedy = bool((greedy_ids >= 0).any())

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

    # Two from the commonest target class guarantees a same-target pair; a
    # third widens within-class coverage once the subset is larger than eight.
    take(np.flatnonzero(target_ids == most_frequent(target_ids)), 3 if count >= 12 else 2)
    if has_greedy:
        take(np.flatnonzero(greedy_ids == most_frequent(greedy_ids)), 2)
    # The second most frequent target class gives a second within-class group,
    # and therefore same-target pairs that are not all from one token.
    remaining = target_ids[[p for p in range(target_ids.size) if p not in chosen]]
    if remaining.size:
        values, counts = np.unique(remaining, return_counts=True)
        second = int(values[np.lexsort((values, -counts))[0]])
        take(np.flatnonzero(target_ids == second), 2)

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
    """Build one fixed CountSketch map as bucket and sign vectors.

    **This does not reproduce the production map.** Production draws from
    ``torch.Generator`` and this draws from NumPy's PCG64; the same integer seed
    gives two structurally unrelated maps, which is exactly the bug this warning
    exists to prevent recurring. To project with the production map, obtain it
    from ``_GradientSketcher.numpy_map()`` and pass it as ``sketch_map``.

    What this is for is the *hypothetical* realizations: the alternate-seed check
    and the ``K`` sweep, which ask what a different draw of the construction
    would have given. Any deterministic generator serves there, because no
    comparison is being made against a stored production sketch.
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
    sketch_map: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """The production estimator: sketched inner products over **exact** norms.

    ``sketch_map`` supplies an explicit ``(buckets, signs)`` pair, which is how
    the production map is used: it comes from the sketcher that produced the
    experiment's sketches rather than being re-derived here. Without it a map is
    built from ``seed``, which is correct for hypothetical realizations and
    **wrong** for anything compared against a stored production sketch.
    """

    if sketch_map is not None:
        buckets, signs = sketch_map
        if buckets.shape[0] != gradients.shape[1]:
            raise ValueError(
                "The supplied sketch map does not cover the gradient dimension."
            )
    else:
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


def subgroup_delta(cosines: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Within minus between on one cosine matrix, over the sanity subset.

    The same statistic figure 20 reports, applied here to both the exact and the
    sketched matrices so the two can be differenced. Per-pair error is not the
    scientific observable; ``delta`` is, and a projection could in principle be
    noisy per pair while preserving the within-versus-between geometry, or the
    reverse.

    Strictly a subset diagnostic. With a dozen positions this ``delta`` is not an
    estimate of the full-experiment value and must not be read as one -- only the
    *difference* between the exact and sketched versions is meaningful.

    ``None`` when the subset lacks either a within-group or a between-group pair,
    rather than a fabricated number.
    """

    labels = np.asarray(labels)
    size = cosines.shape[0]
    rows, columns = np.triu_indices(size, k=1)
    same = labels[rows] == labels[columns]
    if not same.any() or same.all():
        return {"available": False, "reason": "no within-group or no between-group pair"}
    within = float(cosines[rows, columns][same].mean())
    between = float(cosines[rows, columns][~same].mean())
    return {
        "available": True,
        "within": within,
        "between": between,
        "delta": within - between,
        "num_within_pairs": int(same.sum()),
        "num_between_pairs": int((~same).sum()),
    }


def fidelity_report(
    gradients: np.ndarray,
    norms: np.ndarray,
    target_ids: np.ndarray,
    greedy_ids: np.ndarray,
    *,
    production_dimension: int,
    production_seed: int,
    tensor_sizes: Sequence[int] | None = None,
    production_map: tuple[np.ndarray, np.ndarray] | None = None,
    map_factory: Any = None,
    alternate_seeds: Sequence[int] = ALTERNATE_SKETCH_SEEDS,
    sensitivity_dimensions: Sequence[int] = SENSITIVITY_DIMENSIONS,
) -> dict[str, Any]:
    """Everything the sanity check reports, computed offline from stored vectors.

    No model execution of any kind happens here: the alternate seeds and the
    ``K`` sweep reuse the same captured gradients, which is the entire reason for
    capturing them.

    ``map_factory(dimension, seed) -> (buckets, signs)`` decides what "another
    realization" means, and it matters. Pass the production builder and the
    alternate seeds and the ``K`` sweep answer *how much would this result move
    under another draw of the production construction* -- which is the robustness
    question. Leave it out and they fall back to a generic NumPy map, which is a
    valid CountSketch experiment but a different question, and must not be
    reported as production robustness.
    """

    def build(dimension: int, seed: int):
        if map_factory is None:
            return None
        return map_factory(dimension, seed)

    exact = cosine_from_gram(gradients, norms)
    # The production row must use the experiment's own map, not a re-derivation.
    production = sketch_estimated_cosines(
        gradients, norms, dimension=production_dimension,
        seed=production_seed, tensor_sizes=tensor_sizes, sketch_map=production_map,
    )

    # The scientific observable, computed both ways on the same subset.
    deltas = {}
    for name, labels in (("target", target_ids), ("greedy", greedy_ids)):
        exact_delta = subgroup_delta(exact, labels)
        sketch_delta = subgroup_delta(production, labels)
        entry = {"exact": exact_delta, "sketch": sketch_delta}
        if exact_delta["available"] and sketch_delta["available"]:
            entry["delta_error"] = sketch_delta["delta"] - exact_delta["delta"]
        deltas[name] = entry

    alternates = []
    for seed in alternate_seeds:
        estimated = sketch_estimated_cosines(
            gradients, norms, dimension=production_dimension, seed=seed,
            sketch_map=build(production_dimension, seed),
        )
        summary = _errors(exact, estimated)
        summary["seed"] = int(seed)
        # How much the subgroup statistic itself moves with the realization.
        for name, labels in (("target", target_ids), ("greedy", greedy_ids)):
            observed = subgroup_delta(exact, labels)
            projected = subgroup_delta(estimated, labels)
            summary[f"{name}_delta_error"] = (
                projected["delta"] - observed["delta"]
                if observed["available"] and projected["available"]
                else float("nan")
            )
        alternates.append(summary)

    sensitivity = []
    for dimension in sensitivity_dimensions:
        per_seed = [
            _errors(
                exact,
                sketch_estimated_cosines(
                    gradients, norms, dimension=dimension, seed=seed,
                    sketch_map=build(dimension, seed),
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
        "map_semantics": "production" if map_factory is not None else "generic_numpy",
        "exact_cosines": exact,
        "production_cosines": production,
        "production": _errors(exact, production),
        "pair_types": pair_type_breakdown(exact, production, target_ids, greedy_ids),
        "subgroup_deltas": deltas,
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
        "alternate_delta_errors": {
            name: [
                entry[f"{name}_delta_error"] for entry in alternates
                if np.isfinite(entry[f"{name}_delta_error"])
            ]
            for name in ("target", "greedy")
        },
    }


#: Where a run keeps its fidelity artifact, relative to the run root.
SANITY_ARTIFACT = ("sanity", "countsketch_fidelity.npz")


def sanity_artifact_path(record_dir) -> "Any":
    """Resolve a run's fidelity artifact from its ``analyses`` directory.

    The renderer is handed ``<run>/analyses`` because that is where the main
    record lives, while this artifact sits beside it under ``<run>/sanity``.
    Resolving from the run root keeps that relationship in one place instead of
    spreading path arithmetic through the renderer.
    """

    from pathlib import Path

    return Path(record_dir).parent.joinpath(*SANITY_ARTIFACT)


def load_fidelity_artifact(path) -> dict[str, Any]:
    """Rebuild the report the fidelity figure consumes from a saved artifact.

    Only what the figure needs is reconstructed. The artifact deliberately holds
    derived results rather than the full gradients, so nothing here recomputes a
    projection or touches a model.
    """

    from pathlib import Path

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No CountSketch fidelity artifact at {path}.")

    with np.load(path) as data:
        exact = np.asarray(data["exact_cosine_matrix"], dtype=np.float64)
        production = np.asarray(data["production_cosine_matrix"], dtype=np.float64)
        report: dict[str, Any] = {
            "num_gradients": int(exact.shape[0]),
            "gradient_dtype": "float32",
            "accumulation_dtype": "float64",
            "production_dimension": int(data["k_values"][
                int(np.argmin(np.abs(np.asarray(data["k_values"]) - 512)))
            ]) if "k_values" in data else 512,
            "production_seed": 20240917,
            "exact_cosines": exact,
            "production_cosines": production,
            "production": _errors(exact, production),
            "selected_position_indices": np.asarray(data["selected_position_indices"]),
            "selected_target_ids": np.asarray(data["selected_target_ids"]),
            "selected_greedy_ids": np.asarray(data["selected_greedy_ids"]),
            "sensitivity": [
                {
                    "dimension": int(dimension),
                    "mean_absolute_error": float(mae),
                    "rmse": float(rmse),
                }
                for dimension, mae, rmse in zip(
                    data["k_values"], data["k_mae"], data["k_rmse"]
                )
            ] if "k_values" in data else [],
        }
        report["pair_types"] = pair_type_breakdown(
            exact, production,
            report["selected_target_ids"], report["selected_greedy_ids"],
        )
        for name in ("target", "greedy"):
            key = f"{name}_delta_error"
            if key in data:
                report.setdefault("subgroup_deltas", {})[name] = {
                    "delta_error": float(np.asarray(data[key]).reshape(-1)[0])
                }
    return report


def projected_space_cosines(production: np.ndarray) -> np.ndarray:
    """The bounded projected-space cosine, derived from the production matrix.

    The production estimator divides by the **exact** norms, so its diagonal is
    ``||S(g)||^2 / ||g||^2`` rather than 1 -- and that is exactly what is needed
    to convert it::

        cos_proj(a, b) = production(a, b) / sqrt(production(a, a) production(b, b))
                       = <S(a), S(b)> / (||S(a)|| ||S(b)||)

    So no sketch vectors and no rerun are required: the persisted matrix already
    carries the projected norms in its diagonal.

    This is a genuine cosine between two projected vectors, hence confined to
    ``[-1, 1]``. That is a property of the comparator, not evidence that it is
    the better estimator: it replaces an exact denominator with a noisy one, and
    which wins is an empirical question the fidelity report answers.
    """

    diagonal = np.diag(production).astype(np.float64)
    if np.any(diagonal <= 0.0):
        raise ValueError(
            "A projected sketch has zero norm, so its projected-space cosine is "
            "undefined."
        )
    scale = np.sqrt(np.outer(diagonal, diagonal))
    return production / scale


def deviation_report(exact: np.ndarray, estimated: np.ndarray) -> dict[str, Any]:
    """Range and tail behaviour of one estimator against the exact cosines.

    MAE and RMSE describe the middle of the error distribution; these describe
    its edges, which is what matters for reading an individual heatmap cell.

    The out-of-range counts are the point of this for the production estimator:
    dividing a sketched inner product by exact norms is not a cosine and is not
    confined to ``[-1, 1]``. Values outside it are a measurable property of the
    estimator, reported rather than clipped away -- clipping would hide the very
    thing being quantified.

    Percentiles are NumPy's linear interpolation. On a handful of pairs they are
    order statistics with a label, not confidence bounds, so the pair count is
    reported beside them.
    """

    size = exact.shape[0]
    upper = np.triu_indices(size, k=1)
    exact_pairs = exact[upper]
    estimated_pairs = estimated[upper]
    signed = estimated_pairs - exact_pairs
    absolute = np.abs(signed)
    outside = (estimated_pairs < -1.0) | (estimated_pairs > 1.0)

    return {
        "num_pairs": int(exact_pairs.size),
        "exact_min": float(exact_pairs.min()),
        "exact_max": float(exact_pairs.max()),
        "estimated_min": float(estimated_pairs.min()),
        "estimated_max": float(estimated_pairs.max()),
        "signed_min": float(signed.min()),
        "signed_max": float(signed.max()),
        "bias": float(signed.mean()),
        "mean_absolute_error": float(absolute.mean()),
        "median_absolute_error": float(np.median(absolute)),
        "p95_absolute_error": float(np.percentile(absolute, 95)),
        "p99_absolute_error": float(np.percentile(absolute, 99)),
        "max_absolute_error": float(absolute.max()),
        "rmse": float(np.sqrt((signed ** 2).mean())),
        "num_below_minus_one": int((estimated_pairs < -1.0).sum()),
        "num_above_plus_one": int((estimated_pairs > 1.0).sum()),
        "fraction_outside_unit_interval": float(outside.mean()),
        "pearson_correlation": (
            float(np.corrcoef(exact_pairs, estimated_pairs)[0, 1])
            if exact_pairs.size >= 3
            and np.std(exact_pairs) > 0
            and np.std(estimated_pairs) > 0
            else float("nan")
        ),
    }


def estimator_comparison(exact: np.ndarray, production: np.ndarray) -> dict[str, Any]:
    """Both estimators measured against the same exact cosines.

    The question is whether the bounded comparator is actually more faithful, or
    whether swapping an exact denominator for a random one costs more than
    boundedness gains. Reported side by side rather than decided here: changing
    the production estimator is a scientific decision, not an implementation one.
    """

    projected = projected_space_cosines(production)
    return {
        "exact_norm_estimator": deviation_report(exact, production),
        "projected_space_estimator": deviation_report(exact, projected),
        "projected_cosines": projected,
    }
