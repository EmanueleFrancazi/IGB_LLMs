"""Directional structure of per-position gradients, by token subgroup.

The question is whether gradients cluster: are two positions that share a token
label more directionally alike than two positions that do not? Magnitude has
already been measured; this is about direction only, so every gradient is
divided by its own exact norm before anything is compared and the magnitudes
play no further part.

Gradients reach here as count sketches rather than as full parameter vectors --
the exact ``[D, P]`` matrix is about 1.1 TB in float32 at experiment scale.

The scaling matters and is easy to get wrong. Each sketch is divided by the
**exact** gradient norm the experiment already persists, not by its own length:

.. code-block:: text

    u_d = sketch(g_d) / ||g_d||          NOT sketch(g_d) / ||sketch(g_d)||

With that choice ``E<u_a, u_b> = cos(g_a, g_b)`` exactly, because the sketch is
unbiased in the inner product and the divisor is a constant rather than another
random quantity. Normalizing the sketch to unit length instead would give a
ratio of two correlated estimators, which is only consistent as ``K`` grows and
carries an ``O(1/K)`` bias -- acceptable, but strictly worse, and it silently
changes what the diagonal identity has to subtract.

The consequence is that ``||u_d||`` is **not** 1. It is ``||sketch(g_d)|| /
||g_d||``, which concentrates near 1 but fluctuates with the projection. Every
within-class identity therefore subtracts the measured ``sum_d ||u_d||^2`` and
never the class count.

Individual pairwise values carry the projection's own noise and should not be
read alone; the pooled class-level means are what this module reports.

Two groupings are computed and kept apart:

``target``
    the true next token ``y_d``. Positions sharing a target share a *loss*.
``greedy``
    the token the initialized model predicts, ``argmax z_d``. Positions sharing
    a greedy prediction share a *decision*.

They answer different questions and can disagree, which is itself informative.

Nothing here assumes clustering exists. A null, a negative and a positive result
all come out as the same descriptive numbers, and the permutation reference says
how much apparent structure the same gradients and the same class-size
distribution produce with the labels shuffled.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "GROUPINGS",
    "class_similarity_matrix",
    "DEFAULT_PERMUTATIONS",
    "clustering_summary",
    "gradient_clustering",
    "unit_sketches",
]

#: The two subgroup definitions, in the order reports present them.
GROUPINGS = ("target", "greedy")


def unit_sketches(record: Any) -> tuple[np.ndarray, np.ndarray]:
    """``[D, K]`` sketches divided by the exact gradient norms, and a usable mask.

    The divisor is ``||g_d||`` from the experiment's own per-position norms, so
    the inner product of two rows is an **unbiased** estimator of the cosine
    between the corresponding true gradients. The rows are therefore only
    approximately unit length, and callers must never assume otherwise.

    A position whose exact gradient norm is zero has no direction and is
    excluded rather than scaled into a fabricated one.
    """

    if not has_gradient_sketches(record):
        raise ValueError(
            "This record carries no per-position gradient sketches, so directional "
            "clustering cannot be measured. Rerun with the sketch enabled."
        )
    if record.gradient_position_norms is None:
        raise ValueError(
            "Scaling the sketches needs the exact per-position gradient norms, "
            "which this record does not carry."
        )
    sketches = np.asarray(record.gradient_position_sketches, dtype=np.float64)
    exact = np.asarray(record.gradient_position_norms, dtype=np.float64)
    if exact.shape[0] != sketches.shape[0]:
        raise ValueError(
            "The sketch and the gradient-norm arrays describe different positions."
        )
    usable = exact > 0.0
    scaled = np.zeros_like(sketches)
    scaled[usable] = sketches[usable] / exact[usable, None]
    return scaled, usable


def has_gradient_sketches(record: Any) -> bool:
    """Whether the record carries per-position gradient sketches."""

    return getattr(record, "gradient_position_sketches", None) is not None


def _labels(record: Any, grouping: str) -> np.ndarray:
    if grouping == "target":
        return np.asarray(record.gradient_position_target_ids, dtype=np.int64)
    if grouping == "greedy":
        return np.asarray(record.gradient_position_greedy_ids, dtype=np.int64)
    raise ValueError(f"grouping must be one of {GROUPINGS}; got {grouping!r}.")


def _class_sums(
    unit: np.ndarray, labels: np.ndarray, classes: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-class vector sum, membership count, and sum of squared self norms.

    The third return value is what makes the within-class identity correct once
    the rows are no longer unit length: the self terms of ``||S_i||^2`` are the
    measured ``sum_{d in i} ||u_d||^2``, not ``n_i``.
    """

    sums = np.zeros((classes.size, unit.shape[1]), dtype=np.float64)
    counts = np.zeros(classes.size, dtype=np.int64)
    self_squared = np.zeros(classes.size, dtype=np.float64)
    row_squared = np.einsum("ij,ij->i", unit, unit)
    position = {int(token): index for index, token in enumerate(classes)}
    for row, label in enumerate(labels):
        index = position.get(int(label))
        if index is not None:
            sums[index] += unit[row]
            counts[index] += 1
            self_squared[index] += row_squared[row]
    return sums, counts, self_squared


def class_similarity_matrix(
    unit: np.ndarray, labels: np.ndarray, classes: np.ndarray
) -> dict[str, Any]:
    """The token x token mean-cosine matrix, diagonal included honestly.

    Cell ``(i, j)`` with ``i != j`` is the mean cosine over every cross pair, and
    cell ``(i, i)`` is the mean cosine over every **distinct** pair inside class
    ``i``. The diagonal is therefore a measurement, not the trivial ``1`` a
    self-similarity convention would put there -- which is the whole point, since
    the diagonal is exactly where within-class coherence would show up.

    Both are computed from class sums rather than by forming pairs::

        sum_{a != b in i} <u_a, u_b> = ||S_i||^2 - sum_{d in i} ||u_d||^2
        sum_{a in i, b in j} <u_a, u_b> = <S_i, S_j>

    exactly, turning an ``O(n^2 K)`` computation into ``O(n K)``.

    The subtracted term is the **measured** sum of squared row norms. Using
    ``n_i`` there would be a silent assumption that every row has unit length,
    which is false here: the rows are sketches divided by the exact gradient
    norm, so their lengths scatter around 1. A test plants rows whose norms are
    deliberately far from 1 and fails if ``n_i`` is used.

    A class with a single member has no distinct pair, so its diagonal is NaN.
    """

    sums, counts, self_squared = _class_sums(unit, labels, classes)
    gram = sums @ sums.T
    matrix = np.full((classes.size, classes.size), np.nan, dtype=np.float64)

    pairs = counts[:, None] * counts[None, :]
    off = pairs > 0
    matrix[off] = gram[off] / pairs[off]

    diagonal_pairs = counts * (counts - 1)
    measurable = diagonal_pairs > 0
    diagonal = np.full(classes.size, np.nan, dtype=np.float64)
    diagonal[measurable] = (
        np.diag(gram)[measurable] - self_squared[measurable]
    ) / diagonal_pairs[measurable]
    np.fill_diagonal(matrix, diagonal)

    return {
        "classes": classes,
        "counts": counts,
        "matrix": matrix,
        "within": diagonal,
        "self_squared": self_squared,
        "num_within_measurable": int(measurable.sum()),
    }


def _summary_from_matrix(result: dict[str, Any]) -> dict[str, Any]:
    """Pooled within, pooled between, and their difference."""

    counts = result["counts"]
    matrix = result["matrix"]
    within = result["within"]

    # Pooled over pairs, not over classes: a class with 400 members should not
    # count the same as one with 2 when the question is about typical pairs.
    within_pairs = counts * (counts - 1)
    usable = np.isfinite(within) & (within_pairs > 0)
    within_mean = (
        float((within[usable] * within_pairs[usable]).sum() / within_pairs[usable].sum())
        if usable.any()
        else float("nan")
    )

    off = ~np.eye(counts.size, dtype=bool)
    cross_pairs = (counts[:, None] * counts[None, :]).astype(np.float64)
    finite = off & np.isfinite(matrix) & (cross_pairs > 0)
    between_mean = (
        float((matrix[finite] * cross_pairs[finite]).sum() / cross_pairs[finite].sum())
        if finite.any()
        else float("nan")
    )

    return {
        "within": within_mean,
        "between": between_mean,
        "delta": within_mean - between_mean,
        "num_classes": int(counts.size),
        "num_within_measurable": result["num_within_measurable"],
        "num_within_pairs": int(within_pairs.sum()),
        "num_between_pairs": int(cross_pairs[finite].sum()) if finite.any() else 0,
    }


#: Permutations drawn for the null. Enough to read a 2.5-97.5% interval off the
#: order statistics without leaning on any distributional assumption.
DEFAULT_PERMUTATIONS = 256


def _pooled_from_blocks(
    sums: np.ndarray, counts: np.ndarray, self_squared: np.ndarray
) -> tuple[float, float]:
    """Pooled within and between similarity, without forming the C x C matrix.

    Summing over all ordered pairs gives ``||T||^2`` with ``T = sum_i S_i``, and
    the within-class part is ``sum_i ||S_i||^2``. Subtracting one from the other
    leaves the between-class part, and removing the self terms from the first
    leaves the distinct within-class part::

        within  = (sum_i ||S_i||^2 - sum_i Q_i) / sum_i n_i (n_i - 1)
        between = (||T||^2 - sum_i ||S_i||^2) / ((sum_i n_i)^2 - sum_i n_i^2)

    That is the same number the matrix route produces -- a test asserts it -- at
    ``O(C K)`` instead of ``O(C^2 K)``, which is what makes a 256-permutation
    null affordable at experiment scale.
    """

    class_squared = np.einsum("ij,ij->i", sums, sums)
    total = sums.sum(axis=0)
    within_pairs = float((counts * (counts - 1)).sum())
    between_pairs = float(counts.sum() ** 2 - (counts.astype(np.float64) ** 2).sum())

    within = (
        float((class_squared.sum() - self_squared.sum()) / within_pairs)
        if within_pairs > 0
        else float("nan")
    )
    between = (
        float((float(total @ total) - class_squared.sum()) / between_pairs)
        if between_pairs > 0
        else float("nan")
    )
    return within, between


def _permutation_null(
    unit: np.ndarray,
    counts: np.ndarray,
    *,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    """Null distribution of ``delta`` under label permutation.

    Only the class **sizes** are kept; which position carries which label is
    reshuffled. So the null says how much apparent clustering follows from the
    class-size distribution and the gradients' overall geometry alone, with token
    identity carrying no information.

    Permuting the rows and cutting them into blocks of the observed sizes is the
    same thing as permuting the labels, and lets each draw cost one pass.

    Reported as a mean and a 2.5-97.5% interval from the order statistics. No
    p-value: with one experiment and dependent positions a p-value here would
    read as far stronger evidence than the design supports.
    """

    generator = np.random.default_rng(seed)
    row_squared = np.einsum("ij,ij->i", unit, unit)
    edges = np.concatenate([[0], np.cumsum(counts)])
    deltas = np.empty(permutations, dtype=np.float64)
    withins = np.empty(permutations, dtype=np.float64)
    betweens = np.empty(permutations, dtype=np.float64)

    for draw in range(permutations):
        order = generator.permutation(edges[-1])
        shuffled = unit[order]
        sums = np.add.reduceat(shuffled, edges[:-1], axis=0)
        self_squared = np.add.reduceat(row_squared[order], edges[:-1])
        within, between = _pooled_from_blocks(sums, counts, self_squared)
        withins[draw] = within
        betweens[draw] = between
        deltas[draw] = within - between

    low, high = np.percentile(deltas, [2.5, 97.5])
    return {
        "permutations": int(permutations),
        "seed": int(seed),
        "within_mean": float(withins.mean()),
        "between_mean": float(betweens.mean()),
        "delta_mean": float(deltas.mean()),
        "delta_std": float(deltas.std(ddof=1)) if permutations > 1 else float("nan"),
        "delta_low": float(low),
        "delta_high": float(high),
    }


def gradient_clustering(
    record: Any,
    *,
    grouping: str = "target",
    min_support: int = 2,
    max_classes: int | None = None,
    permutations: int = DEFAULT_PERMUTATIONS,
    permutation_seed: int = 20240918,
) -> dict[str, Any]:
    """Directional clustering of gradients under one subgroup definition.

    Args:
        record: A record carrying per-position gradient sketches.
        grouping: ``"target"`` or ``"greedy"``.
        min_support: Smallest class size kept. Two is the smallest that admits a
            within-class pair at all; anything larger is a readability choice and
            is reported, never silent.
        max_classes: Optionally keep only the most frequent classes, which is
            what makes the matrix drawable. ``None`` keeps every class above
            ``min_support``.
        permutations: Draws in the label-permutation null.
        permutation_seed: Seed of that null.

    Returns:
        The matrix, the per-class coherence, the pooled summary, and the null
        distribution of ``delta`` under label permutation.
    """

    unit, usable = unit_sketches(record)
    labels = _labels(record, grouping)
    if labels.shape[0] != unit.shape[0]:
        raise ValueError(
            "The sketch and the label arrays describe different position counts."
        )
    labels = labels[usable]
    unit = unit[usable]

    tokens, counts = np.unique(labels, return_counts=True)
    keep = counts >= int(min_support)
    tokens, counts = tokens[keep], counts[keep]
    if max_classes is not None and tokens.size > max_classes:
        order = np.argsort(-counts, kind="stable")[: int(max_classes)]
        order = np.sort(order)
        tokens, counts = tokens[order], counts[order]
    if tokens.size < 2:
        raise ValueError(
            f"Only {tokens.size} class(es) reach min_support={min_support}; a "
            "between-class comparison needs at least two."
        )

    observed = class_similarity_matrix(unit, labels, tokens)

    # The null needs only the rows that belong to a kept class, in any order.
    keep_rows = np.isin(labels, tokens)
    null = _permutation_null(
        unit[keep_rows],
        observed["counts"],
        permutations=permutations,
        seed=permutation_seed,
    )

    return {
        "grouping": grouping,
        "min_support": int(min_support),
        "num_positions": int(usable.sum()),
        "num_positions_excluded": int((~usable).sum()),
        "sketch_dimension": int(unit.shape[1]),
        "classes": tokens,
        "counts": counts,
        "matrix": observed["matrix"],
        "within_by_class": observed["within"],
        "observed": _summary_from_matrix(observed),
        "null": null,
        "permutation_seed": int(permutation_seed),
    }


def clustering_summary(
    record: Any, *, min_support: int = 2, max_classes: int | None = None
) -> dict[str, Any]:
    """Both groupings side by side, with their permutation references."""

    return {
        grouping: gradient_clustering(
            record, grouping=grouping, min_support=min_support, max_classes=max_classes
        )
        for grouping in GROUPINGS
    }
