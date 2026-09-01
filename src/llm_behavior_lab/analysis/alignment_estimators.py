"""The factorized CountSketch alignment estimators, in one place.

Every directional statistic this project reports is a bilinear form in the
normalized sketch rows, and every one of them can be computed from two per-class
factors instead of from the rows themselves::

    T_{m,a} = sum_{i in a} z_{im}          the class vector sum
    Q_{m,a} = sum_{i in a} ||z_{im}||^2    the measured self term

    A_aa = (1/M) sum_m (||T_{m,a}||^2 - Q_{m,a}) / (n_a (n_a - 1))    n_a >= 2
    A_ab = (1/M) sum_m  T_{m,a} . T_{m,b}  / (n_a n_b)                a != b

with ``z_{im} = S_m(g_i) / ||g_i||`` -- the sketch over the **exact** gradient
norm, so the rows are only approximately unit length and ``Q`` is a measured
quantity. Subtracting ``n_a`` there instead of ``Q_{m,a}`` is the mistake this
module exists to make unavailable; a fixture with ``Q != n`` fails loudly if it
ever creeps back.

This is the same estimator :mod:`llm_behavior_lab.analysis.gradient_clustering`
computes on the concatenated ensemble embedding
``v_i = M^{-1/2} [z_{i1} | ... | z_{iM}]``: that identity gives
``||T_a^v||^2 = (1/M) sum_m ||T_{m,a}||^2`` and ``Q_a^v = (1/M) sum_m Q_{m,a}``
exactly. The factorized route reaches the same numbers without ever building the
``[D, M*K]`` array, which is about a gibibyte at experiment scale.

Reduction epoch
---------------

Class sums here are formed **blockwise**: rows are laid out in contiguous
per-class blocks and each block is reduced with ``ndarray.sum``, which NumPy
evaluates pairwise. The established v12 route uses ``np.add.reduceat``, which
accumulates sequentially. The statistic, the draws, the estimator and the
permutation sequence are identical; only the float64 accumulation order inside a
class sum differs.

That is a deliberate, reviewed change, recorded as :data:`REDUCTION_CONVENTION`
and :data:`NUMERICAL_EPOCH` so a v13 number is never silently compared against a
pre-epoch one. It was adopted because ``np.add.reduceat`` along axis 0 was
measured at 0.6 GB/s -- about twelve times below memory bandwidth, and 85% of the
permutation null's cost -- while the reported statistics moved by at most
7.6e-14 relative, roughly thirteen times inside the declared 1e-12 tolerance.
**v12 analysis keeps its exact ``reduceat`` route, bitwise unchanged.**

Both callers reduce **selected** rows into per-class blocks, so
:func:`block_factors` indexes and sums in one step rather than materializing a
reordered copy first. That is not a micro-optimization: gathering a ``[D, K]``
block and then reducing it reads the data, writes it, and reads it again, where
indexing inside the reduction reads it once. Measured at experiment scale it is
1.8x faster and it removes a 256 MiB buffer, while producing bitwise the same
values -- the same rows are summed in the same order either way.

An earlier draft also batched runs of equal-sized blocks into one vectorized
reduction. That was exact too, but it bought a further 1.1x for a
run-length-encoding loop here plus a size-major row layout and a scatter-back in
both callers, so it was removed in favour of the single clear loop below.

NumPy only. No I/O, no file iteration, no record objects: this module is the
single owner of the arithmetic and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "ESTIMATOR_CONVENTION",
    "REDUCTION_CONVENTION",
    "NUMERICAL_EPOCH",
    "ClassFactors",
    "CrossFactors",
    "block_factors",
    "class_factors",
    "class_similarity_from_factors",
    "cross_factors",
    "cross_matrix_from_factors",
    "permutation_null",
    "permutation_orders",
    "pooled_cross_from_factors",
    "pooled_from_factors",
]

#: How a pairwise similarity is defined. Names the ``Q``-corrected, exact-norm,
#: arithmetic-mean-over-maps estimator above.
ESTIMATOR_CONVENTION = "countsketch_pairwise_mean_q_self/v1"

#: How a class sum is accumulated. See the module docstring.
REDUCTION_CONVENTION = "blockwise_class_sum/v1"

#: Identifies the summation-order change relative to the v12 ``reduceat`` route.
#: A comparison between results carrying different epochs is a comparison
#: between two different accumulations and must be made under the declared
#: tolerance, never asserted bitwise.
NUMERICAL_EPOCH = "alignment_reduction_epoch/v1"


def block_factors(
    rows: np.ndarray,
    row_squared: np.ndarray,
    index: np.ndarray,
    counts: Any,
    *,
    sums_out: np.ndarray | None = None,
    self_out: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``T`` and ``Q`` for blocks cut out of ``rows`` by ``index``.

    ``index`` lists row numbers grouped so that block ``b`` owns the next
    ``counts[b]`` of them. Both callers already have such a list: the point
    estimate sorts rows by class, and the null permutes them.

    The rows are indexed **inside** the reduction rather than gathered into a
    reordered copy first. Same rows, same order, bitwise the same sums -- but one
    pass over the data instead of three, and no ``[D, K]`` scratch buffer.

    ``T`` and ``Q`` come out together because they are always wanted together and
    share the indexing work; splitting them would index every block twice.

    Returns:
        ``([C, K], [C])`` float64.
    """

    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"rows must be [D, K]; got shape {values.shape}.")
    squared = np.asarray(row_squared, dtype=np.float64)
    if squared.shape != (values.shape[0],):
        raise ValueError(
            f"row_squared must have shape ({values.shape[0]},); got {squared.shape}."
        )
    sizes = np.asarray(counts, dtype=np.int64)
    if sizes.ndim != 1:
        raise ValueError(f"counts must be one-dimensional; got shape {sizes.shape}.")
    if np.any(sizes < 0):
        raise ValueError("counts must be non-negative.")
    selection = np.asarray(index, dtype=np.int64)
    total = int(sizes.sum())
    if total != selection.size:
        raise ValueError(
            f"counts sum to {total} but index lists {selection.size} rows; the "
            "blocks must exactly tile the selection."
        )

    if sums_out is None:
        sums_out = np.empty((sizes.size, values.shape[1]), dtype=np.float64)
    elif sums_out.shape != (sizes.size, values.shape[1]):
        raise ValueError(
            f"sums_out must have shape {(sizes.size, values.shape[1])}; "
            f"got {sums_out.shape}."
        )
    if self_out is None:
        self_out = np.empty(sizes.size, dtype=np.float64)
    elif self_out.shape != (sizes.size,):
        raise ValueError(
            f"self_out must have shape {(sizes.size,)}; got {self_out.shape}."
        )

    start = 0
    for block, size in enumerate(sizes):
        members = selection[start : start + size]
        sums_out[block] = values[members].sum(axis=0)
        self_out[block] = squared[members].sum()
        start += size
    return sums_out, self_out


@dataclass(frozen=True)
class ClassFactors:
    """``T_{m,a}``, ``Q_{m,a}`` and the class sizes, for one map.

    These are gradient sketches, summed. Naming them ``sums`` rather than
    anything more abstract is deliberate: a reader deciding whether they may be
    persisted needs to see what they are.
    """

    #: ``[C]`` class identifiers, in the order every array here is indexed by.
    classes: np.ndarray
    #: ``[C]`` members per class, ``n_a``.
    counts: np.ndarray
    #: ``[C, K]`` class vector sums, ``T_{m,a}``.
    sums: np.ndarray
    #: ``[C]`` measured self terms, ``Q_{m,a}``. **Never** ``n_a``.
    self_squared: np.ndarray

    @property
    def num_classes(self) -> int:
        return int(self.classes.size)

    @property
    def num_positions(self) -> int:
        return int(self.counts.sum())


def class_factors(unit: np.ndarray, labels: Any, classes: Any) -> ClassFactors:
    """Per-class factors for one map, under :data:`REDUCTION_CONVENTION`.

    Rows whose label is not in ``classes`` are excluded, which is what lets the
    display subset reuse this without redefining the population.

    Rows are sorted into per-class blocks by class index, stably, so a class's
    rows are summed in their original order -- the same order the v12 route
    accumulates them in.

    Args:
        unit: ``[D, K]`` sketches already divided by the exact gradient norms.
        labels: ``[D]`` one label per row.
        classes: ``[C]`` the classes to accumulate, in the order to report them.
    """

    rows = np.asarray(unit, dtype=np.float64)
    if rows.ndim != 2:
        raise ValueError(f"unit must be [D, K]; got shape {rows.shape}.")
    label_values = np.asarray(labels, dtype=np.int64)
    class_values = np.asarray(classes, dtype=np.int64)
    if label_values.shape[0] != rows.shape[0]:
        raise ValueError(
            f"unit has {rows.shape[0]} rows but labels has "
            f"{label_values.shape[0]} entries."
        )
    if np.unique(class_values).size != class_values.size:
        raise ValueError("classes must not repeat.")

    # Position of each row's label within `classes`, or -1 when absent.
    ordering = np.argsort(class_values, kind="stable")
    ordered = class_values[ordering]
    slot = np.searchsorted(ordered, label_values)
    slot = np.clip(slot, 0, ordered.size - 1) if ordered.size else slot
    matched = ordered.size > 0
    index = np.full(label_values.shape, -1, dtype=np.int64)
    if matched:
        hit = ordered[slot] == label_values
        index[hit] = ordering[slot[hit]]

    keep = index >= 0
    kept_index = index[keep]
    counts = np.bincount(kept_index, minlength=class_values.size).astype(np.int64)

    order = np.flatnonzero(keep)[np.argsort(kept_index, kind="stable")]
    sums, self_squared = block_factors(
        rows, np.einsum("ij,ij->i", rows, rows), order, counts
    )

    return ClassFactors(
        classes=class_values, counts=counts, sums=sums, self_squared=self_squared
    )


def pooled_from_factors(
    sums: np.ndarray, counts: np.ndarray, self_squared: np.ndarray
) -> tuple[float, float]:
    """Pooled within- and between-class similarity, without a ``C x C`` matrix.

    Summing over all ordered pairs gives ``||T||^2`` with ``T = sum_a T_a``, and
    the within-class part is ``sum_a ||T_a||^2``. Subtracting one from the other
    leaves the between-class part, and removing the self terms from the first
    leaves the distinct within-class part::

        within  = (sum_a ||T_a||^2 - sum_a Q_a) / sum_a n_a (n_a - 1)
        between = (||T||^2 - sum_a ||T_a||^2) / ((sum_a n_a)^2 - sum_a n_a^2)

    ``O(C K)`` instead of ``O(C^2 K)``, which is what makes a 256-draw null
    affordable at experiment scale.

    Pooled over **pairs**, not over classes: a class of 400 should not weigh the
    same as one of 2 when the question is about a typical pair. Singletons
    contribute no within-pair but real between-pairs, so they are not dropped.

    ``nan`` where the corresponding pair count is zero -- an unavailable
    statistic, never a zero standing in for one.
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


def class_similarity_from_factors(factors: ClassFactors) -> dict[str, Any]:
    """The ``C x C`` mean-similarity matrix, diagonal included honestly.

    Cell ``(a, b)`` with ``a != b`` is the mean over every cross pair; cell
    ``(a, a)`` is the mean over every **distinct** pair inside class ``a``. The
    diagonal is a measurement, not the trivial ``1`` a self-similarity
    convention would put there -- which is the whole point, since the diagonal is
    exactly where within-class coherence shows up.

    A class with one member has no distinct pair, so its diagonal is ``nan``.

    Intended for the bounded display subset. It is ``O(C^2 K)`` and the pooled
    statistics deliberately do not go through it.
    """

    sums = factors.sums
    counts = factors.counts
    gram = sums @ sums.T
    matrix = np.full((counts.size, counts.size), np.nan, dtype=np.float64)

    pairs = counts[:, None] * counts[None, :]
    off = pairs > 0
    matrix[off] = gram[off] / pairs[off]

    diagonal_pairs = counts * (counts - 1)
    measurable = diagonal_pairs > 0
    diagonal = np.full(counts.size, np.nan, dtype=np.float64)
    diagonal[measurable] = (
        np.diag(gram)[measurable] - factors.self_squared[measurable]
    ) / diagonal_pairs[measurable]
    np.fill_diagonal(matrix, diagonal)

    return {
        "classes": factors.classes,
        "counts": counts,
        "matrix": matrix,
        "within": diagonal,
        "self_squared": factors.self_squared,
        "num_within_measurable": int(measurable.sum()),
    }


def permutation_orders(
    counts: np.ndarray, *, permutations: int, seed: int
) -> list[np.ndarray]:
    """The permutation index draws for a null, generated once.

    Every map is handed **the same draws**. The across-map spread is meant to
    isolate projection randomness; if each map drew its own permutations that
    spread would also carry permutation noise and would overstate the projection
    error.

    The sequence is exactly the one the v12 route produces -- same generator,
    same seed, same order of calls -- so a null is comparable across epochs even
    though its accumulation is not bitwise identical.
    """

    generator = np.random.default_rng(seed)
    total = int(np.sum(counts))
    return [generator.permutation(total) for _ in range(permutations)]


def permutation_null(
    unit: np.ndarray,
    counts: np.ndarray,
    *,
    permutations: int,
    seed: int,
    draw_batch: int = 32,
) -> dict[str, Any]:
    """Null distribution of ``delta`` under label permutation.

    Only the class **sizes** are kept; which position carries which label is
    reshuffled. The null therefore says how much apparent clustering follows from
    the class-size distribution and the gradients' overall geometry alone, with
    token identity carrying no information. Permuting the rows and cutting them
    into blocks of the observed sizes is the same thing as permuting the labels,
    and lets each draw cost one pass.

    Reported as a mean and a 2.5-97.5% interval from the order statistics. **No
    p-value**: with one experiment and dependent positions a p-value here would
    read as far stronger evidence than the design supports.

    Two implementation choices, neither of which may change a value:

    * the draw's rows are indexed inside :func:`block_factors` rather than
      gathered into a reordered ``[D, K]`` copy first -- one pass over the data
      instead of three, and no scratch buffer at all.
    * draws are pulled from the generator in bounded batches rather than all at
      once -- ``256 * D`` int64 indices is 67 MiB at experiment scale, for no
      reason. Sequential draws from one generator give the identical sequence.

    Args:
        unit: ``[D, K]`` normalized sketches for one map.
        counts: ``[C]`` observed class sizes.
        permutations: Draws.
        seed: Seed of the draw sequence.
        draw_batch: How many orders are resident at once. Memory only.
    """

    rows = np.asarray(unit, dtype=np.float64)
    sizes = np.asarray(counts, dtype=np.int64)
    if permutations < 1:
        raise ValueError(f"permutations must be positive; got {permutations}.")
    if draw_batch < 1:
        raise ValueError(f"draw_batch must be positive; got {draw_batch}.")
    total = int(sizes.sum())
    if total != rows.shape[0]:
        raise ValueError(
            f"counts sum to {total} but unit carries {rows.shape[0]} rows."
        )

    row_squared = np.einsum("ij,ij->i", rows, rows)
    sums_buffer = np.empty((sizes.size, rows.shape[1]), dtype=np.float64)
    self_buffer = np.empty(sizes.size, dtype=np.float64)

    deltas = np.empty(permutations, dtype=np.float64)
    withins = np.empty(permutations, dtype=np.float64)
    betweens = np.empty(permutations, dtype=np.float64)

    generator = np.random.default_rng(seed)
    drawn = 0
    while drawn < permutations:
        batch = min(draw_batch, permutations - drawn)
        orders = [generator.permutation(total) for _ in range(batch)]
        for order in orders:
            block_factors(
                rows, row_squared, order, sizes,
                sums_out=sums_buffer, self_out=self_buffer,
            )
            within, between = pooled_from_factors(sums_buffer, sizes, self_buffer)
            withins[drawn] = within
            betweens[drawn] = between
            deltas[drawn] = within - between
            drawn += 1
        del orders

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


# -- cross-partition factors -------------------------------------------------
#
# The target grouping and the greedy grouping partition the *same* population:
# every position carries both labels, so it sits in one target class and one
# greedy class simultaneously. Cell (i, j) of the cross matrix is the mean
# similarity between target class i and greedy class j, and every occupied cell
# -- not only the diagonal -- has to subtract the positions it shares, or a cell
# would silently include a position compared against itself.


@dataclass(frozen=True)
class CrossFactors:
    """Everything the cross-partition statistics need, and nothing per-position.

    The contingency is held **sparsely**. A dense ``[C, C]`` table over a subword
    vocabulary is about 8 GB and almost entirely zero; at most ``D`` cells are
    occupied, so the occupied ones are listed instead. That is not only a memory
    choice: the dense route is what makes the current v12 cross-partition
    analysis take minutes.
    """

    #: Class factors of the two groupings over one shared class set.
    target: ClassFactors
    greedy: ClassFactors
    #: ``[cells]`` indices into ``target.classes`` / ``greedy.classes``.
    cell_target_index: np.ndarray
    cell_greedy_index: np.ndarray
    #: ``[cells]`` occupancy ``m_ij`` and self term ``Q_ij``.
    cell_counts: np.ndarray
    cell_self_squared: np.ndarray
    #: ``sum_d ||u_d||^2`` over every included position, i.e. ``sum_ij Q_ij``.
    total_self_squared: float

    @property
    def classes(self) -> np.ndarray:
        return self.target.classes


def cross_factors(
    unit: np.ndarray, targets: Any, greedy: Any, classes: Any
) -> CrossFactors:
    """Target and greedy factors plus the sparse contingency, in one pass.

    ``Q_ij`` needs only each row's squared norm, never the row itself, which is
    why the cross-partition statistics survive the rows being discarded.

    Args:
        unit: ``[D, K]`` sketches divided by the exact gradient norms.
        targets: ``[D]`` true-target labels.
        greedy: ``[D]`` greedy-prediction labels.
        classes: The shared class set both groupings are indexed by. Shared on
            purpose: "the same token in its two roles" is only a meaningful cell
            if one index means one token on both axes.
    """

    rows = np.asarray(unit, dtype=np.float64)
    target_values = np.asarray(targets, dtype=np.int64)
    greedy_values = np.asarray(greedy, dtype=np.int64)
    class_values = np.asarray(classes, dtype=np.int64)
    if target_values.shape != greedy_values.shape:
        raise ValueError(
            "targets and greedy must describe the same positions; got "
            f"{target_values.shape} and {greedy_values.shape}."
        )

    target_factors = class_factors(rows, target_values, class_values)
    greedy_factors = class_factors(rows, greedy_values, class_values)

    lookup = {int(token): index for index, token in enumerate(class_values)}
    target_index = np.array(
        [lookup.get(int(value), -1) for value in target_values], dtype=np.int64
    )
    greedy_index = np.array(
        [lookup.get(int(value), -1) for value in greedy_values], dtype=np.int64
    )
    inside = (target_index >= 0) & (greedy_index >= 0)

    row_squared = np.einsum("ij,ij->i", rows, rows)
    pairs = np.stack([target_index[inside], greedy_index[inside]], axis=1)
    cells, inverse, counts = np.unique(
        pairs, axis=0, return_inverse=True, return_counts=True
    )
    self_squared = np.bincount(
        inverse, weights=row_squared[inside], minlength=cells.shape[0]
    )

    return CrossFactors(
        target=target_factors,
        greedy=greedy_factors,
        cell_target_index=cells[:, 0] if cells.size else np.zeros(0, dtype=np.int64),
        cell_greedy_index=cells[:, 1] if cells.size else np.zeros(0, dtype=np.int64),
        cell_counts=counts.astype(np.int64),
        cell_self_squared=np.asarray(self_squared, dtype=np.float64),
        total_self_squared=float(row_squared[inside].sum()),
    )


def pooled_cross_from_factors(factors: CrossFactors) -> dict[str, Any]:
    """Same-token versus different-token cross alignment, over every position.

    ``delta_cross`` asks whether pairing a token's target role with the *same*
    token's greedy role yields more alignment than pairing it with an arbitrary
    other token's greedy role.

    Computed from sufficient statistics, and -- unlike the v12 route -- without
    ever forming a dense ``[C, C]`` contingency. The same-token part needs only
    the **diagonal** cells, and the different-token part follows by subtracting
    it from the totals::

        same_num   = sum_i <T_i^Y, T_i^G> - sum_i Q_ii
        same_pairs = sum_i n_i^Y n_i^G     - sum_i m_ii
        total_num  = <sum_i T_i^Y, sum_j T_j^G> - sum_ij Q_ij
        total_pairs= (sum_i n_i^Y)(sum_j n_j^G) - sum_ij m_ij

    ``sum_ij Q_ij`` is just the total squared row norm and ``sum_ij m_ij`` the
    position count, because every position lies in exactly one cell.
    """

    target_sums = factors.target.sums
    greedy_sums = factors.greedy.sums
    target_counts = factors.target.counts
    greedy_counts = factors.greedy.counts

    diagonal = factors.cell_target_index == factors.cell_greedy_index
    diagonal_counts = int(factors.cell_counts[diagonal].sum())
    diagonal_self = float(factors.cell_self_squared[diagonal].sum())

    same_numerator = float(
        np.einsum("ij,ij->", target_sums, greedy_sums) - diagonal_self
    )
    same_pairs = float(
        (target_counts.astype(np.float64) * greedy_counts).sum() - diagonal_counts
    )

    total_numerator = float(
        target_sums.sum(axis=0) @ greedy_sums.sum(axis=0) - factors.total_self_squared
    )
    total_pairs = float(
        float(target_counts.sum()) * float(greedy_counts.sum())
        - float(factors.cell_counts.sum())
    )
    different_numerator = total_numerator - same_numerator
    different_pairs = total_pairs - same_pairs

    same = same_numerator / same_pairs if same_pairs > 0 else float("nan")
    different = (
        different_numerator / different_pairs if different_pairs > 0 else float("nan")
    )
    return {
        "classes": factors.classes,
        "c_same": same,
        "c_different": different,
        "delta_cross": same - different,
        "num_same_pairs": int(same_pairs),
        "num_different_pairs": int(different_pairs),
        "num_true_positive_positions": diagonal_counts,
    }


def cross_matrix_from_factors(
    factors: CrossFactors, target_slots: Any, greedy_slots: Any
) -> dict[str, Any]:
    """Self-excluded mean similarity between target class ``i`` and greedy ``j``::

        C_ij = (T_i^Y . T_j^G - Q_ij) / (n_i^Y n_j^G - m_ij)

    The subtraction removes every position belonging to both groups, in **every**
    cell. Without it an off-diagonal cell would silently include each shared
    position compared with itself -- and at initialization the two labels differ
    almost everywhere, so those cells are the ones being read.

    ``nan`` where the denominator vanishes: a cell with no cross pair left after
    self-exclusion has nothing to average.

    Dense over the supplied slots, which is why they are the bounded display
    subset rather than the whole class set.
    """

    target_slots = np.asarray(target_slots, dtype=np.int64)
    greedy_slots = np.asarray(greedy_slots, dtype=np.int64)

    target_sums = factors.target.sums[target_slots]
    greedy_sums = factors.greedy.sums[greedy_slots]
    target_counts = factors.target.counts[target_slots]
    greedy_counts = factors.greedy.counts[greedy_slots]

    shape = (target_slots.size, greedy_slots.size)
    counts = np.zeros(shape, dtype=np.int64)
    self_squared = np.zeros(shape, dtype=np.float64)

    target_position = {int(slot): index for index, slot in enumerate(target_slots)}
    greedy_position = {int(slot): index for index, slot in enumerate(greedy_slots)}
    for cell in range(factors.cell_counts.size):
        row = target_position.get(int(factors.cell_target_index[cell]))
        column = greedy_position.get(int(factors.cell_greedy_index[cell]))
        if row is None or column is None:
            continue
        counts[row, column] = int(factors.cell_counts[cell])
        self_squared[row, column] = float(factors.cell_self_squared[cell])

    numerator = target_sums @ greedy_sums.T - self_squared
    denominator = (
        target_counts[:, None].astype(np.float64) * greedy_counts[None, :] - counts
    )
    matrix = np.full(shape, np.nan, dtype=np.float64)
    usable = denominator > 0
    matrix[usable] = numerator[usable] / denominator[usable]

    return {
        "matrix": matrix,
        "target_classes": factors.classes[target_slots],
        "greedy_classes": factors.classes[greedy_slots],
        "target_counts": target_counts,
        "greedy_counts": greedy_counts,
        "contingency": counts,
        "self_squared": self_squared,
        "num_usable_cells": int(usable.sum()),
    }
