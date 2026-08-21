"""Tests for the target-versus-greedy cross-partition geometry.

Everything is checked against explicit pair enumeration on fixtures small enough
to write out by hand. The self-exclusion is the part that is easy to get wrong
and the part that matters most, so it is tested from several directions --
including the case the formula exists for: a position whose target and greedy
labels **differ** still lands in an off-diagonal contingency cell and would be
compared against itself there unless subtracted.
"""

from __future__ import annotations

import numpy as np
import pytest

from llm_behavior_lab.analysis.gradient_cross_partition import (
    contingency_summary,
    cross_identity_null,
    cross_partition_matrix,
    mixture_reconstruction,
    pooled_cross_statistic,
)

K = 16


def _rows(count: int, seed: int = 3) -> np.ndarray:
    generator = np.random.default_rng(seed)
    values = generator.normal(size=(count, K))
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def _brute_cross(rows, targets, greedy, target_token, greedy_token):
    """Mean over (a in A_i, b in B_j) with a != b, by explicit enumeration."""

    left = np.flatnonzero(targets == target_token)
    right = np.flatnonzero(greedy == greedy_token)
    values = [
        float(rows[a] @ rows[b]) for a in left for b in right if a != b
    ]
    return float(np.mean(values)) if values else float("nan")


def test_the_cross_matrix_matches_explicit_enumeration() -> None:
    rows = _rows(9)
    targets = np.array([1, 1, 2, 2, 3, 3, 1, 2, 3], dtype=np.int64)
    greedy = np.array([2, 3, 1, 3, 1, 2, 3, 1, 2], dtype=np.int64)
    classes = np.array([1, 2, 3], dtype=np.int64)

    result = cross_partition_matrix(rows, targets, greedy, classes, classes)

    for i, target_token in enumerate(classes):
        for j, greedy_token in enumerate(classes):
            expected = _brute_cross(rows, targets, greedy, target_token, greedy_token)
            actual = result["matrix"][i, j]
            if np.isnan(expected):
                assert np.isnan(actual)
            else:
                assert actual == pytest.approx(expected, abs=1e-12)


def test_an_off_diagonal_cell_needs_self_exclusion_too() -> None:
    """The case the formula exists for.

    Position 0 has target 1 and greedy 2, so it belongs to A_1 and to B_2 at
    once. Cell (1, 2) is off-diagonal, yet the naive product still compares that
    position with itself. Subtracting only the i == j true positives would leave
    the error in place.
    """

    rows = _rows(4)
    targets = np.array([1, 1, 3, 3], dtype=np.int64)
    greedy = np.array([2, 4, 2, 4], dtype=np.int64)
    target_classes = np.array([1, 3], dtype=np.int64)
    greedy_classes = np.array([2, 4], dtype=np.int64)

    result = cross_partition_matrix(
        rows, targets, greedy, target_classes, greedy_classes
    )

    # No position has target == greedy anywhere in this fixture.
    assert not np.any(targets == greedy)
    # Yet cell (target 1, greedy 2) contains position 0 in both groups.
    assert result["contingency"][0, 0] == 1
    assert result["self_squared"][0, 0] == pytest.approx(float(rows[0] @ rows[0]))

    # Naive, self-inclusive value differs from the correct one.
    naive = float(rows[targets == 1].sum(axis=0) @ rows[greedy == 2].sum(axis=0)) / 4.0
    correct = result["matrix"][0, 0]
    assert correct == pytest.approx(
        _brute_cross(rows, targets, greedy, 1, 2), abs=1e-12
    )
    assert not np.isclose(naive, correct)


def test_a_cell_with_no_remaining_cross_pair_is_undefined() -> None:
    """One position shared by both groups and nothing else leaves no pair."""

    rows = _rows(1)
    targets = np.array([5], dtype=np.int64)
    greedy = np.array([7], dtype=np.int64)

    result = cross_partition_matrix(
        rows, targets, greedy, np.array([5]), np.array([7])
    )

    assert result["matrix"].shape == (1, 1)
    assert np.isnan(result["matrix"][0, 0])
    assert result["num_usable_cells"] == 0


def test_the_pooled_statistic_matches_brute_force() -> None:
    rows = _rows(10, seed=11)
    targets = np.array([1, 1, 2, 2, 3, 3, 1, 2, 3, 1], dtype=np.int64)
    greedy = np.array([2, 3, 1, 3, 1, 2, 3, 1, 2, 1], dtype=np.int64)

    result = pooled_cross_statistic(rows, targets, greedy)

    same, different = [], []
    for a in range(rows.shape[0]):
        for b in range(rows.shape[0]):
            if a == b:
                continue
            value = float(rows[a] @ rows[b])
            (same if targets[a] == greedy[b] else different).append(value)

    assert result["num_same_pairs"] == len(same)
    assert result["num_different_pairs"] == len(different)
    assert result["c_same"] == pytest.approx(float(np.mean(same)), abs=1e-12)
    assert result["c_different"] == pytest.approx(float(np.mean(different)), abs=1e-12)
    assert result["delta_cross"] == pytest.approx(
        float(np.mean(same)) - float(np.mean(different)), abs=1e-12
    )


def test_a_singleton_class_contributes_cross_pairs() -> None:
    """Unlike within-class statistics, a singleton still pairs across roles."""

    rows = _rows(3)
    targets = np.array([1, 2, 3], dtype=np.int64)
    greedy = np.array([3, 1, 2], dtype=np.int64)

    result = pooled_cross_statistic(rows, targets, greedy)

    # Every class has one member in each role, and no position is a true positive.
    assert result["num_true_positive_positions"] == 0
    assert result["num_same_pairs"] + result["num_different_pairs"] == 3 * 3 - 3


def test_the_identity_null_preserves_supports_and_is_deterministic() -> None:
    rows = _rows(12, seed=5)
    targets = np.array([1, 1, 2, 2, 3, 3, 4, 4, 1, 2, 3, 4], dtype=np.int64)
    greedy = np.array([2, 3, 1, 4, 1, 2, 3, 1, 4, 3, 4, 2], dtype=np.int64)

    first = cross_identity_null(rows, targets, greedy, permutations=32)
    second = cross_identity_null(rows, targets, greedy, permutations=32)

    assert first == second
    assert first["permutations"] == 32
    assert first["num_valid_draws"] > 0
    assert first["delta_low"] <= first["delta_mean"] <= first["delta_high"]


def test_the_contingency_summary_is_sparse_and_correct() -> None:
    targets = np.array([1, 1, 2, 2, 3], dtype=np.int64)
    greedy = np.array([2, 2, 1, 3, 3], dtype=np.int64)

    summary = contingency_summary(targets, greedy)

    assert summary["num_positions"] == 5
    assert summary["num_target_classes"] == 3
    assert summary["num_greedy_classes"] == 3
    assert summary["num_cells"] == 4          # (1,2)x2, (2,1), (2,3), (3,3)
    assert summary["num_true_positive_positions"] == 1     # position 4
    assert summary["true_positive_fraction"] == pytest.approx(0.2)
    assert int(summary["counts"].sum()) == 5


def test_mixture_reconstruction_is_exact_when_greedy_groups_are_target_groups() -> None:
    """If a greedy group is exactly one target group, the remix must recover it."""

    rows = _rows(6, seed=9)
    targets = np.array([1, 1, 1, 2, 2, 2], dtype=np.int64)
    # Greedy relabels the same blocks, so P(y|h) is a point mass each time.
    greedy = np.array([7, 7, 7, 8, 8, 8], dtype=np.int64)

    result = mixture_reconstruction(rows, targets, greedy)

    assert result["num_classes"] == 2
    assert np.allclose(result["similarity"], 1.0, atol=1e-9)
    assert np.allclose(result["residual_norm"], 0.0, atol=1e-9)


def test_mixture_reconstruction_reports_support_and_a_named_metric() -> None:
    rows = _rows(10, seed=13)
    targets = np.array([1, 1, 2, 2, 3, 3, 1, 2, 3, 1], dtype=np.int64)
    greedy = np.array([7, 8, 7, 8, 7, 8, 7, 8, 7, 8], dtype=np.int64)

    result = mixture_reconstruction(rows, targets, greedy)

    assert set(result["tokens"].tolist()) == {7, 8}
    assert result["support"].sum() == 10
    assert -1.0 - 1e-9 <= result["median_similarity"] <= 1.0 + 1e-9
    # Named honestly: this is not variance explained.
    assert "cosine" in result["metric"]
    assert "variance" not in result["metric"]


def test_no_dense_contingency_matrix_is_built_for_the_pooled_statistic() -> None:
    """Sufficient statistics only: 2000 classes must not allocate 2000^2 cells."""

    generator = np.random.default_rng(2)
    rows = _rows(400, seed=4)
    targets = generator.integers(0, 200, size=400).astype(np.int64)
    greedy = generator.integers(0, 200, size=400).astype(np.int64)

    result = pooled_cross_statistic(rows, targets, greedy)

    assert np.isfinite(result["c_same"])
    assert np.isfinite(result["c_different"])
    assert result["num_same_pairs"] + result["num_different_pairs"] == 400 * 400 - 400
