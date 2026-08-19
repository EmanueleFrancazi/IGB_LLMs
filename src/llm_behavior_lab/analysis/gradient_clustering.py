"""Directional structure of per-position gradients, by token subgroup.

The question is whether gradients cluster: are two positions that share a token
label more directionally alike than two positions that do not? Magnitude has
already been measured; this is about direction only, so every gradient is
normalized to the unit sphere before anything is compared and the magnitudes
play no further part.

Gradients reach here as count sketches rather than as full parameter vectors --
the exact ``[D, P]`` matrix is about 1.1 PB at experiment scale. The sketch
preserves inner products in expectation, so class-level mean cosines are
unbiased estimates of the mean cosines of the true gradients. Individual
pairwise values carry the projection's own noise and should not be read alone.

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
    "clustering_summary",
    "gradient_clustering",
    "unit_sketches",
]

#: The two subgroup definitions, in the order reports present them.
GROUPINGS = ("target", "greedy")


def unit_sketches(record: Any) -> tuple[np.ndarray, np.ndarray]:
    """``[D, K]`` direction-only sketches, plus a mask of usable positions.

    A zero sketch has no direction. That happens only if a gradient vanished
    exactly, which at initialization it does not, but it is excluded rather than
    normalized into a fabricated direction.
    """

    if not has_gradient_sketches(record):
        raise ValueError(
            "This record carries no per-position gradient sketches, so directional "
            "clustering cannot be measured. Rerun with the sketch enabled."
        )
    sketches = np.asarray(record.gradient_position_sketches, dtype=np.float64)
    norms = np.linalg.norm(sketches, axis=1)
    usable = norms > 0.0
    unit = np.zeros_like(sketches)
    unit[usable] = sketches[usable] / norms[usable, None]
    return unit, usable


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
) -> tuple[np.ndarray, np.ndarray]:
    """Per-class sum of unit sketches, and per-class membership counts."""

    sums = np.zeros((classes.size, unit.shape[1]), dtype=np.float64)
    counts = np.zeros(classes.size, dtype=np.int64)
    position = {int(token): index for index, token in enumerate(classes)}
    for row, label in enumerate(labels):
        index = position.get(int(label))
        if index is not None:
            sums[index] += unit[row]
            counts[index] += 1
    return sums, counts


def class_similarity_matrix(
    unit: np.ndarray, labels: np.ndarray, classes: np.ndarray
) -> dict[str, Any]:
    """The token x token mean-cosine matrix, diagonal included honestly.

    Cell ``(i, j)`` with ``i != j`` is the mean cosine over every cross pair, and
    cell ``(i, i)`` is the mean cosine over every **distinct** pair inside class
    ``i``. The diagonal is therefore a measurement, not the trivial ``1`` a
    self-similarity convention would put there -- which is the whole point, since
    the diagonal is exactly where within-class coherence would show up.

    Both are computed from class sums rather than by forming pairs. For unit
    vectors, ``sum_{a != b in i} <u_a, u_b> = ||S_i||^2 - n_i``, and
    ``sum_{a in i, b in j} <u_a, u_b> = <S_i, S_j>``. That is exact, not an
    approximation, and turns an ``O(n^2 K)`` computation into ``O(n K)``.

    A class with a single member has no distinct pair, so its diagonal is NaN.
    """

    sums, counts = _class_sums(unit, labels, classes)
    gram = sums @ sums.T
    matrix = np.full((classes.size, classes.size), np.nan, dtype=np.float64)

    pairs = counts[:, None] * counts[None, :]
    off = pairs > 0
    matrix[off] = gram[off] / pairs[off]

    diagonal_pairs = counts * (counts - 1)
    measurable = diagonal_pairs > 0
    diagonal = np.full(classes.size, np.nan, dtype=np.float64)
    diagonal[measurable] = (
        np.diag(gram)[measurable] - counts[measurable]
    ) / diagonal_pairs[measurable]
    np.fill_diagonal(matrix, diagonal)

    return {
        "classes": classes,
        "counts": counts,
        "matrix": matrix,
        "within": diagonal,
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


def _permuted_labels(labels: np.ndarray, seed: int) -> np.ndarray:
    """Shuffle labels while keeping the class-size distribution exactly.

    A permutation of the label vector preserves every class size, so the null
    isolates the grouping itself: any ``delta`` it produces comes from class
    sizes and from the gradients' overall geometry, not from token identity
    meaning anything.
    """

    generator = np.random.default_rng(seed)
    return generator.permutation(labels)


def gradient_clustering(
    record: Any,
    *,
    grouping: str = "target",
    min_support: int = 2,
    max_classes: int | None = None,
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
        permutation_seed: Seed of the label-permutation reference.

    Returns:
        The matrix, the per-class coherence, the pooled summary, and the same
        summary recomputed on permuted labels.
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
    permuted = class_similarity_matrix(
        unit, _permuted_labels(labels, permutation_seed), tokens
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
        "permuted": _summary_from_matrix(permuted),
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
