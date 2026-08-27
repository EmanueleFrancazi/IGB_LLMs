"""How the target-conditioned and greedy-conditioned groupings relate.

Figure 20 shows two token-indexed directional families over the *same* gradient
population: one grouping positions by the true target ``y_d``, the other by the
model's greedy prediction ``h_d``. Both look coherent. That raises an obvious
question this module exists to answer descriptively -- are they the same
structure seen twice, two roughly orthogonal structures, or something else?

The vocabulary matters here. These are two **partitions of one population**, not
bases and not independent samples. Every position carries both labels at once, so
it sits in exactly one target group and exactly one greedy group simultaneously.
At initialization ``y_d == h_d`` almost never holds, but that says the two
partitions *disagree*, not that they fail to overlap: the overlaps are precisely
the contingency cells ``m_ij``, and they are what makes a naive cross product
wrong.

Concretely, for target group ``A_i`` and greedy group ``B_j``::

    U_i^Y . U_j^G  =  sum over every (a in A_i, b in B_j) pair

which includes ``a == b`` for every position in ``A_i ∩ B_j``. Those self terms
contribute ``||u_d||^2`` -- not a comparison between two gradients at all -- and
they appear in **off-diagonal** cells too, wherever a position has target ``i``
and greedy ``j``. Subtracting only the ``i == j`` true positives would leave most
of them in place.

Nothing here fits a generative model. A reading like ``u_d ~ a_{y_d} + b_{h_d} +
noise`` is an interpretation aid; the mixture diagnostic below asks only how much
of the greedy grouping's mean directions is already implied by the target
grouping plus the contingency table.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "contingency_summary",
    "cross_partition_matrix",
    "cross_identity_null",
    "cross_partition_per_map",
    "mixture_reconstruction",
    "pooled_cross_statistic",
]


def cross_partition_per_map(
    rows_per_map: Any,
    targets: Any,
    greedy: Any,
    *,
    min_support: int = 2,
) -> dict[str, Any]:
    """``c_same``, ``c_different``, ``delta_cross`` and the mixture ratio per map.

    The point estimates stay with :func:`pooled_cross_statistic` and
    :func:`mixture_reconstruction`, which read the ensemble embedding; this is
    the spread beside them. Every map sees the same positions, the same target
    and greedy label vectors and the same support filtering, so the variation is
    projection randomness.

    Each map goes through the same sufficient-statistic route the single-map path
    uses -- group sums and the contingency diagonal -- so no ``O(D^2)`` matrix is
    formed at any map count.

    **On the mixture ratio.** ``mixture_median_similarity`` is a *ratio* of
    bilinear quantities, so the plug-in value computed on the ensemble embedding
    is **not** the mean of the per-map ratios; the square root in each norm makes
    the two differ. Both are returned, named apart, and the plug-in value remains
    the production point estimate -- at ``M = 1`` it coincides with the quantity
    every existing result used. The per-map spread is a projection diagnostic for
    the ratio, not a standard error of the plug-in value.

    Args:
        rows_per_map: ``[D, M, W]`` normalized sketches with the map axis kept.
        targets: ``[D]`` target token IDs.
        greedy: ``[D]`` greedy token IDs.
        min_support: Passed through to :func:`mixture_reconstruction`.
    """

    from llm_behavior_lab.analysis.sketch_estimator import ensemble_summary

    rows_per_map = np.asarray(rows_per_map, dtype=np.float64)
    if rows_per_map.ndim != 3:
        raise ValueError(
            "rows_per_map must be [positions, maps, buckets]; got "
            f"{rows_per_map.ndim} dimensions."
        )
    maps = int(rows_per_map.shape[1])

    same = np.empty(maps, dtype=np.float64)
    different = np.empty(maps, dtype=np.float64)
    mixture = np.empty(maps, dtype=np.float64)
    for index in range(maps):
        rows = rows_per_map[:, index, :]
        pooled = pooled_cross_statistic(rows, targets, greedy)
        same[index] = pooled["c_same"]
        different[index] = pooled["c_different"]
        mixture[index] = mixture_reconstruction(
            rows, targets, greedy, min_support=min_support
        )["median_similarity"]
    delta = same - different

    return {
        "map_count": maps,
        "c_same_per_map": same,
        "c_different_per_map": different,
        "delta_cross_per_map": delta,
        "mixture_median_similarity_per_map": mixture,
        "c_same": ensemble_summary(same),
        "c_different": ensemble_summary(different),
        "delta_cross": ensemble_summary(delta),
        # Named to keep it distinct from the plug-in production estimate.
        "mixture_median_similarity_map_summary": ensemble_summary(mixture),
    }


def _group_sums(
    rows: np.ndarray, labels: np.ndarray, classes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-class vector sums and counts, in the order of ``classes``."""

    index = {int(token): position for position, token in enumerate(classes)}
    sums = np.zeros((classes.size, rows.shape[1]), dtype=np.float64)
    counts = np.zeros(classes.size, dtype=np.int64)
    for row, label in enumerate(labels):
        position = index.get(int(label))
        if position is not None:
            sums[position] += rows[row]
            counts[position] += 1
    return sums, counts


def _contingency(
    targets: np.ndarray,
    greedy: np.ndarray,
    target_classes: np.ndarray,
    greedy_classes: np.ndarray,
    rows: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``m_ij`` counts and, optionally, ``Q_ij`` self-term sums.

    ``Q_ij`` is the sum of ``||u_d||^2`` over positions in ``A_i ∩ B_j``. It is
    the correction the cross similarity needs, and it is nonzero for every
    occupied cell rather than only the diagonal.
    """

    target_index = {int(t): i for i, t in enumerate(target_classes)}
    greedy_index = {int(g): j for j, g in enumerate(greedy_classes)}
    counts = np.zeros((target_classes.size, greedy_classes.size), dtype=np.int64)
    self_squared = np.zeros((target_classes.size, greedy_classes.size), dtype=np.float64)
    squared = None if rows is None else np.einsum("ij,ij->i", rows, rows)
    for row in range(targets.size):
        i = target_index.get(int(targets[row]))
        j = greedy_index.get(int(greedy[row]))
        if i is None or j is None:
            continue
        counts[i, j] += 1
        if squared is not None:
            self_squared[i, j] += squared[row]
    return counts, self_squared


def cross_partition_matrix(
    rows: np.ndarray,
    targets: np.ndarray,
    greedy: np.ndarray,
    target_classes: np.ndarray,
    greedy_classes: np.ndarray,
) -> dict[str, Any]:
    """Self-excluded mean similarity between target group ``i`` and greedy group ``j``.

    ::

        C_ij = ( U_i^Y . U_j^G  -  Q_ij ) / ( n_i^Y n_j^G  -  m_ij )

    The subtraction removes every position that belongs to both groups, in every
    cell. Without it an off-diagonal cell would silently include each shared
    position compared with itself.

    NaN where the denominator vanishes -- a cell with no cross pair left after
    self-exclusion has nothing to average.
    """

    target_sums, target_counts = _group_sums(rows, targets, target_classes)
    greedy_sums, greedy_counts = _group_sums(rows, greedy, greedy_classes)
    counts, self_squared = _contingency(
        targets, greedy, target_classes, greedy_classes, rows
    )

    numerator = target_sums @ greedy_sums.T - self_squared
    denominator = (
        target_counts[:, None].astype(np.float64) * greedy_counts[None, :] - counts
    )
    matrix = np.full(numerator.shape, np.nan, dtype=np.float64)
    usable = denominator > 0
    matrix[usable] = numerator[usable] / denominator[usable]

    return {
        "matrix": matrix,
        "target_classes": target_classes,
        "greedy_classes": greedy_classes,
        "target_counts": target_counts,
        "greedy_counts": greedy_counts,
        "contingency": counts,
        "self_squared": self_squared,
        "num_usable_cells": int(usable.sum()),
    }


def pooled_cross_statistic(
    rows: np.ndarray, targets: np.ndarray, greedy: np.ndarray
) -> dict[str, Any]:
    """Same-token versus different-token cross alignment, over every position.

    Pooled over pairs rather than over classes, and computed from sufficient
    statistics: the same-token part needs only the diagonal of the contingency
    table, and the different-token part follows by subtracting it from the totals.
    No ``O(D^2)`` matrix is formed.

    ``delta_cross`` is the contrast. It asks whether pairing a token's target role
    with the *same* token's greedy role yields more alignment than pairing it with
    an arbitrary other token's greedy role.
    """

    classes = np.union1d(np.unique(targets), np.unique(greedy))
    target_sums, target_counts = _group_sums(rows, targets, classes)
    greedy_sums, greedy_counts = _group_sums(rows, greedy, classes)
    counts, self_squared = _contingency(targets, greedy, classes, classes, rows)

    diagonal_counts = np.diag(counts)
    diagonal_self = np.diag(self_squared)
    same_numerator = float(
        np.einsum("ij,ij->", target_sums, greedy_sums) - diagonal_self.sum()
    )
    same_pairs = float(
        (target_counts.astype(np.float64) * greedy_counts).sum() - diagonal_counts.sum()
    )

    total_numerator = float(target_sums.sum(axis=0) @ greedy_sums.sum(axis=0)
                            - self_squared.sum())
    total_pairs = float(
        target_counts.sum() * greedy_counts.sum() - counts.sum()
    )
    different_numerator = total_numerator - same_numerator
    different_pairs = total_pairs - same_pairs

    same = same_numerator / same_pairs if same_pairs > 0 else float("nan")
    different = (
        different_numerator / different_pairs if different_pairs > 0 else float("nan")
    )
    return {
        "classes": classes,
        "c_same": same,
        "c_different": different,
        "delta_cross": same - different,
        "num_same_pairs": int(same_pairs),
        "num_different_pairs": int(different_pairs),
        "num_true_positive_positions": int(diagonal_counts.sum()),
    }


def cross_identity_null(
    rows: np.ndarray,
    targets: np.ndarray,
    greedy: np.ndarray,
    *,
    permutations: int = 256,
    seed: int = 20240919,
) -> dict[str, Any]:
    """Null for ``delta_cross`` that destroys only the identity correspondence.

    Every group vector, every support count and all internal geometry are kept
    exactly as observed. What is permuted is which greedy-token identity is
    treated as "the same token" as which target identity, so the null asks
    precisely one thing: does matching a token to *itself* across the two roles
    beat matching it to an arbitrary other token?

    Not a causal test, and no p-value: one experiment, dependent positions.
    """

    classes = np.union1d(np.unique(targets), np.unique(greedy))
    target_sums, target_counts = _group_sums(rows, targets, classes)
    greedy_sums, greedy_counts = _group_sums(rows, greedy, classes)
    counts, self_squared = _contingency(targets, greedy, classes, classes, rows)

    products = target_sums @ greedy_sums.T
    pair_counts = target_counts[:, None].astype(np.float64) * greedy_counts[None, :]
    total_numerator = float(products.sum() - self_squared.sum())
    total_pairs = float(pair_counts.sum() - counts.sum())

    generator = np.random.default_rng(seed)
    deltas = np.empty(permutations, dtype=np.float64)
    order = np.arange(classes.size)
    for draw in range(permutations):
        shuffled = generator.permutation(order)
        same_numerator = float(
            products[order, shuffled].sum() - self_squared[order, shuffled].sum()
        )
        same_pairs = float(
            pair_counts[order, shuffled].sum() - counts[order, shuffled].sum()
        )
        different_pairs = total_pairs - same_pairs
        if same_pairs <= 0 or different_pairs <= 0:
            deltas[draw] = np.nan
            continue
        deltas[draw] = same_numerator / same_pairs - (
            (total_numerator - same_numerator) / different_pairs
        )

    finite = deltas[np.isfinite(deltas)]
    low, high = (
        np.percentile(finite, [2.5, 97.5]) if finite.size else (np.nan, np.nan)
    )
    return {
        "permutations": int(permutations),
        "seed": int(seed),
        "delta_mean": float(finite.mean()) if finite.size else float("nan"),
        "delta_std": float(finite.std(ddof=1)) if finite.size > 1 else float("nan"),
        "delta_low": float(low),
        "delta_high": float(high),
        "num_valid_draws": int(finite.size),
    }


def contingency_summary(targets: np.ndarray, greedy: np.ndarray) -> dict[str, Any]:
    """Sparse target-by-greedy contingency structure and its basic shape.

    Stored sparsely: a dense 32k-by-32k table would be 8 GB and is almost
    entirely zero.
    """

    targets = np.asarray(targets, dtype=np.int64)
    greedy = np.asarray(greedy, dtype=np.int64)
    pairs, counts = np.unique(
        np.stack([targets, greedy], axis=1), axis=0, return_counts=True
    )
    correct = int((targets == greedy).sum())
    return {
        "target_ids": pairs[:, 0],
        "greedy_ids": pairs[:, 1],
        "counts": counts,
        "num_target_classes": int(np.unique(targets).size),
        "num_greedy_classes": int(np.unique(greedy).size),
        "num_cells": int(counts.size),
        "num_true_positive_positions": correct,
        "true_positive_fraction": float(correct / targets.size),
        "num_positions": int(targets.size),
    }


def mixture_reconstruction(
    rows: np.ndarray,
    targets: np.ndarray,
    greedy: np.ndarray,
    *,
    min_support: int = 2,
) -> dict[str, Any]:
    """Can greedy-group mean directions be rebuilt by remixing target ones?

    For greedy token ``j``, the contingency table says which target groups its
    positions came from, so it predicts a mean direction::

        mu_hat_j = sum_i P(y = i | h = j) mu_i^Y

    against the observed ``mu_j^G``. If greedy structure were nothing but a
    reshuffling of target structure, the two would agree closely.

    Similarity is reported as a **projected-space cosine between sketch-space
    mean vectors**, and the limitation is worth stating: these are means of
    sketched directions, so the denominator uses sketch norms rather than exact
    gradient norms -- unlike the production pairwise estimator, no exact norm
    exists for a mean of several gradients. It is descriptive.

    This is not "variance explained". It is one directional agreement per greedy
    class, summarized. An imperfect reconstruction is not by itself evidence of a
    second independent geometric component.
    """

    classes = np.union1d(np.unique(targets), np.unique(greedy))
    target_sums, target_counts = _group_sums(rows, targets, classes)
    greedy_sums, greedy_counts = _group_sums(rows, greedy, classes)
    counts, _ = _contingency(targets, greedy, classes, classes)

    target_means = np.zeros_like(target_sums)
    has_target = target_counts > 0
    target_means[has_target] = target_sums[has_target] / target_counts[has_target, None]

    similarities, residuals, supports, tokens = [], [], [], []
    for j, token in enumerate(classes):
        if greedy_counts[j] < min_support:
            continue
        weights = counts[:, j].astype(np.float64) / float(greedy_counts[j])
        predicted = weights @ target_means
        observed = greedy_sums[j] / float(greedy_counts[j])
        scale = np.linalg.norm(predicted) * np.linalg.norm(observed)
        if scale <= 0:
            continue
        similarities.append(float(predicted @ observed / scale))
        residuals.append(float(np.linalg.norm(observed - predicted)))
        supports.append(int(greedy_counts[j]))
        tokens.append(int(token))

    similarities = np.asarray(similarities)
    supports_array = np.asarray(supports, dtype=np.float64)
    return {
        "tokens": np.asarray(tokens, dtype=np.int64),
        "support": np.asarray(supports, dtype=np.int64),
        "similarity": similarities,
        "residual_norm": np.asarray(residuals),
        "num_classes": int(similarities.size),
        "median_similarity": (
            float(np.median(similarities)) if similarities.size else float("nan")
        ),
        "iqr_similarity": (
            float(np.percentile(similarities, 75) - np.percentile(similarities, 25))
            if similarities.size
            else float("nan")
        ),
        "support_weighted_similarity": (
            float((similarities * supports_array).sum() / supports_array.sum())
            if similarities.size
            else float("nan")
        ),
        "metric": "projected-space cosine between sketch-space mean directions",
    }
