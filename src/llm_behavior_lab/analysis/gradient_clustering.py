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


def unit_sketches(
    record: Any, *, loss_temperature: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """``[D, K]`` sketches divided by the exact gradient norms, and a usable mask.

    The divisor is ``||g_d||`` from the experiment's own per-position norms, so
    the inner product of two rows is an **unbiased** estimator of the cosine
    between the corresponding true gradients. The rows are therefore only
    approximately unit length, and callers must never assume otherwise.

    A position whose exact gradient norm is zero has no direction and is
    excluded rather than scaled into a fabricated one.

    ``loss_temperature`` selects which measured gradient field to read. Omitting
    it keeps the canonical field, which is what every caller written before the
    loss-temperature axis existed wants. Requesting a temperature the record
    never measured raises rather than falling back; see
    :mod:`llm_behavior_lab.analysis.directional_fields`.
    """

    if loss_temperature is not None:
        from llm_behavior_lab.analysis.directional_fields import directional_field

        field = directional_field(record, loss_temperature)
        sketches = np.asarray(field["sketches"], dtype=np.float64)
        exact = np.asarray(field["norms"], dtype=np.float64)
        if exact.shape[0] != sketches.shape[0]:
            raise ValueError(
                "The sketch and the gradient-norm arrays describe different "
                "positions."
            )
        usable = exact > 0.0
        scaled = np.zeros_like(sketches)
        scaled[usable] = sketches[usable] / exact[usable, None]
        return scaled, usable

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
    labels: np.ndarray | None = None,
    loss_temperature: float | None = None,
    min_support: int = 2,
    display_classes: int | None = None,
    permutations: int = DEFAULT_PERMUTATIONS,
    permutation_seed: int = 20240918,
) -> dict[str, Any]:
    """Directional clustering of gradients under one subgroup definition.

    Two populations are kept strictly apart, and conflating them was a real
    defect in an earlier version of this function.

    The **analysis population** is every class with at least ``min_support``
    members. Pooled within, pooled between, ``delta`` and the permutation null
    are all computed over it, so they describe the experiment rather than a
    plotting choice.

    The **display subset** is the ``display_classes`` most supported classes,
    chosen deterministically. It exists only so a heatmap is legible and it
    **never** feeds the pooled statistic. Previously one argument controlled
    both, which silently reduced the reported effect to whatever happened to fit
    on the axes.

    Args:
        record: A record carrying per-position gradient sketches.
        grouping: ``"target"`` or ``"greedy"``, unless ``labels`` is supplied,
            in which case it is only the name this grouping reports itself under.
        labels: One label per position, in record order, replacing the
            record-derived grouping. This is how a grouping the record does not
            persist -- the nucleus sample at a given temperature -- reuses this
            estimator, its population definition and its null rather than a
            parallel implementation that could drift from them. Labels must be
            supplied for **every** position, before the zero-norm mask is
            applied, so they line up with the sketches by construction.
        min_support: Smallest class size counted as *qualifying* for reporting
            and for the heatmap diagonal. It does **not** restrict the pooled
            statistic, which spans every position: a singleton simply has no
            within-class pair to contribute.
        display_classes: How many of the most supported classes to build the
            heatmap matrix from. ``None`` builds it over the whole population,
            which is only sensible when that population is small.
        permutations: Draws in the label-permutation null.
        permutation_seed: Seed of that null.

    Returns:
        ``population`` -- the pooled statistic over every qualifying class;
        ``null`` -- the permutation null over that same population;
        ``display`` -- the matrix and per-class coherence of the drawn subset.
    """

    unit, usable = unit_sketches(record, loss_temperature=loss_temperature)
    labels = (
        _labels(record, grouping)
        if labels is None
        else np.asarray(labels, dtype=np.int64)
    )
    if labels.shape[0] != unit.shape[0]:
        raise ValueError(
            "The sketch and the label arrays describe different position counts."
        )
    labels = labels[usable]
    unit = unit[usable]

    represented, represented_counts = np.unique(labels, return_counts=True)
    keep = represented_counts >= int(min_support)
    tokens, counts = represented[keep], represented_counts[keep]
    # Under the all-position definition these are two different requirements:
    # between-class pairs need two represented classes, singletons included,
    # while within-class pairs need at least one class with a pair to give.
    if represented.size < 2:
        raise ValueError(
            f"Only {represented.size} represented class(es); a between-class "
            "comparison needs at least two."
        )
    if tokens.size < 1:
        raise ValueError(
            f"No class reaches min_support={min_support}, so there is no "
            "within-class pair to compare against."
        )

    # -- population: EVERY position, including singleton classes.
    #
    # A singleton has no within-class pair, so it contributes nothing to the
    # within numerator or denominator -- but it is still a real position that
    # forms between-class pairs with everything else, and dropping it would
    # silently redefine "between" as "between non-singleton classes". That
    # matters most where the singleton fraction moves with the condition being
    # compared, which is exactly the nucleus-temperature case.
    sums, population_counts, self_squared = _class_sums(unit, labels, represented)
    within, between = _pooled_from_blocks(sums, population_counts, self_squared)
    total = int(population_counts.sum())
    population = {
        "classes": represented,
        "counts": population_counts,
        "num_classes": int(tokens.size),
        "num_represented": int(represented.size),
        "num_positions": total,
        "within": within,
        "between": between,
        "delta": within - between,
        "num_within_pairs": int((population_counts * (population_counts - 1)).sum()),
        "num_between_pairs": int(
            total ** 2 - (population_counts.astype(np.int64) ** 2).sum()
        ),
    }

    # The null permutes labels over the same complete population, so singleton
    # between-pairs are present on both sides of the comparison.
    null = _permutation_null(
        unit, population_counts,
        permutations=permutations, seed=permutation_seed,
    )

    # -- display: a legible subset, deterministic, and statistically inert
    shown = tokens
    if display_classes is not None and tokens.size > display_classes:
        order = np.argsort(-counts, kind="stable")[: int(display_classes)]
        shown = np.sort(tokens[order])
    drawn = class_similarity_matrix(unit, labels, shown)
    display = {
        "classes": shown,
        "counts": drawn["counts"],
        "matrix": drawn["matrix"],
        "within_by_class": drawn["within"],
        "num_classes": int(shown.size),
        "selection": (
            "all qualifying classes"
            if shown.size == tokens.size
            else f"{shown.size} most supported of {tokens.size} qualifying classes"
        ),
    }

    return {
        "grouping": grouping,
        "min_support": int(min_support),
        "num_positions": int(usable.sum()),
        "num_positions_excluded": int((~usable).sum()),
        "sketch_dimension": int(unit.shape[1]),
        "loss_temperature": loss_temperature,
        "population": population,
        "null": null,
        "display": display,
        "permutation_seed": int(permutation_seed),
    }


def clustering_summary(
    record: Any, *, min_support: int = 2, display_classes: int | None = None
) -> dict[str, Any]:
    """Both groupings side by side, with their permutation references."""

    return {
        grouping: gradient_clustering(
            record, grouping=grouping, min_support=min_support,
            display_classes=display_classes,
        )
        for grouping in GROUPINGS
    }
