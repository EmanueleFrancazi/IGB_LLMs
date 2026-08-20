"""Tests for the CountSketch fidelity sanity check.

These are NumPy-only: everything after gradient capture is offline arithmetic on
stored vectors, which is the point of capturing them. The capture path itself
needs PyTorch and is covered in ``tests/test_gradient_sketch.py``.

The central property under test is not that the sketch is accurate -- that is
what the check measures -- but that the machinery measuring it is correct and,
above all, **not circular**: position selection must not be able to see any
similarity value.
"""

from __future__ import annotations

import numpy as np
import pytest

from llm_behavior_lab.analysis.countsketch_fidelity import (
    ALTERNATE_SKETCH_SEEDS,
    SENSITIVITY_DIMENSIONS,
    cosine_from_gram,
    count_sketch_matrix,
    fidelity_report,
    pair_type_breakdown,
    select_sanity_positions,
    sketch_estimated_cosines,
)

P = 4096


def _gradients(rows: int = 8, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    values = generator.normal(size=(rows, P)).astype(np.float32)
    norms = np.linalg.norm(values.astype(np.float64), axis=1)
    return values, norms


def _labels(rows: int = 8):
    targets = np.array([5, 5, 9, 9, 3, 4, 6, 7][:rows], dtype=np.int64)
    greedy = np.array([2, 8, 2, 1, 0, 4, 6, 7][:rows], dtype=np.int64)
    return targets, greedy


# -- selection must not be circular ------------------------------------------


def test_selection_is_deterministic() -> None:
    targets, greedy = _labels()
    first = select_sanity_positions(targets, greedy, count=6)
    second = select_sanity_positions(targets, greedy, count=6)

    assert np.array_equal(first, second)
    assert first.size == 6
    assert np.array_equal(first, np.sort(first))


def test_selection_cannot_see_similarity_values() -> None:
    """The validation would be circular if it could. Labels only."""

    targets, greedy = _labels()
    baseline = select_sanity_positions(targets, greedy, count=6)

    # The function has no parameter through which a gradient or sketch could
    # enter, so wildly different geometry cannot move the answer.
    for seed in (1, 2, 3):
        _gradients(seed=seed)
        assert np.array_equal(select_sanity_positions(targets, greedy, count=6), baseline)


def test_selection_yields_the_required_pair_types() -> None:
    targets, greedy = _labels()

    chosen = select_sanity_positions(targets, greedy, count=6)
    picked_targets, picked_greedy = targets[chosen], greedy[chosen]

    rows, columns = np.triu_indices(chosen.size, k=1)
    same_target = (picked_targets[rows] == picked_targets[columns]).sum()
    same_greedy = (picked_greedy[rows] == picked_greedy[columns]).sum()
    neither = (
        (picked_targets[rows] != picked_targets[columns])
        & (picked_greedy[rows] != picked_greedy[columns])
    ).sum()

    assert same_target >= 1
    assert same_greedy >= 1
    assert neither >= 1


# -- the sketch map ----------------------------------------------------------


def test_the_same_seed_reproduces_the_same_map() -> None:
    first = count_sketch_matrix(P, 128, seed=42)
    second = count_sketch_matrix(P, 128, seed=42)

    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


def test_different_seeds_give_different_but_deterministic_maps() -> None:
    buckets_a, signs_a = count_sketch_matrix(P, 128, seed=1)
    buckets_b, signs_b = count_sketch_matrix(P, 128, seed=2)

    assert not np.array_equal(buckets_a, buckets_b)
    assert np.array_equal(count_sketch_matrix(P, 128, seed=2)[0], buckets_b)
    assert set(np.unique(signs_a)).issubset({-1.0, 1.0})


def test_one_map_is_shared_by_every_gradient_in_a_seed() -> None:
    """Redrawing per gradient would destroy the inner-product property."""

    values, norms = _gradients(rows=4)
    estimated = sketch_estimated_cosines(values, norms, dimension=256, seed=11)

    # Symmetric, and the diagonal is ||S(g)||^2 / ||g||^2 -- not forced to 1.
    assert np.allclose(estimated, estimated.T, atol=1e-12)
    assert not np.allclose(np.diag(estimated), 1.0)


# -- the estimator ------------------------------------------------------------


def test_exact_cosine_matches_a_brute_force_computation() -> None:
    values, norms = _gradients(rows=5)

    exact = cosine_from_gram(values, norms)

    for a in range(5):
        for b in range(5):
            expected = float(
                np.dot(values[a].astype(np.float64), values[b].astype(np.float64))
                / (norms[a] * norms[b])
            )
            assert exact[a, b] == pytest.approx(expected, rel=1e-12)
    assert np.allclose(np.diag(exact), 1.0, atol=1e-9)


def test_the_estimator_divides_by_exact_norms_not_sketch_norms() -> None:
    values, norms = _gradients(rows=4)

    estimated = sketch_estimated_cosines(values, norms, dimension=256, seed=5)

    # Scaling a gradient scales its sketch and its exact norm together, so every
    # estimated cosine is unchanged. Dividing by sketch norms would also be
    # invariant, so the discriminating check is the next one.
    scaled = values * np.array([1.0, 4.0, 0.25, 9.0])[:, None]
    scaled_norms = np.linalg.norm(scaled.astype(np.float64), axis=1)
    rescaled = sketch_estimated_cosines(scaled, scaled_norms, dimension=256, seed=5)
    assert np.allclose(estimated, rescaled, atol=1e-9)

    # Feeding wrong norms must change the answer, proving the norms are used.
    wrong = sketch_estimated_cosines(values, norms * 2.0, dimension=256, seed=5)
    assert not np.allclose(estimated, wrong)


def test_a_wide_sketch_approaches_the_exact_cosines() -> None:
    values, norms = _gradients(rows=6)
    exact = cosine_from_gram(values, norms)

    narrow = sketch_estimated_cosines(values, norms, dimension=64, seed=3)
    wide = sketch_estimated_cosines(values, norms, dimension=4096, seed=3)

    upper = np.triu_indices(6, k=1)
    assert np.abs(wide[upper] - exact[upper]).mean() < np.abs(
        narrow[upper] - exact[upper]
    ).mean()


# -- reporting ---------------------------------------------------------------


def test_the_report_excludes_self_pairs_and_computes_errors_correctly() -> None:
    values, norms = _gradients(rows=5)
    targets, greedy = _labels(rows=5)

    report = fidelity_report(
        values, norms, targets, greedy,
        production_dimension=256, production_seed=20240917,
        alternate_seeds=(1, 2), sensitivity_dimensions=(64, 256),
    )
    production = report["production"]

    assert production["num_pairs"] == 10          # C(5,2), self-pairs excluded
    exact = report["exact_cosines"]
    estimated = report["production_cosines"]
    upper = np.triu_indices(5, k=1)
    difference = estimated[upper] - exact[upper]
    assert production["mean_signed_error"] == pytest.approx(float(difference.mean()))
    assert production["mean_absolute_error"] == pytest.approx(
        float(np.abs(difference).mean())
    )
    assert production["rmse"] == pytest.approx(
        float(np.sqrt((difference ** 2).mean()))
    )
    assert production["max_absolute_error"] == pytest.approx(
        float(np.abs(difference).max())
    )


def test_pair_types_partition_the_non_self_pairs() -> None:
    values, norms = _gradients(rows=6)
    targets, greedy = _labels(rows=6)
    exact = cosine_from_gram(values, norms)
    estimated = sketch_estimated_cosines(values, norms, dimension=256, seed=1)

    breakdown = pair_type_breakdown(exact, estimated, targets, greedy)

    assert sum(entry["num_pairs"] for entry in breakdown.values()) == 15   # C(6,2)
    assert set(breakdown) == {
        "same_target_and_greedy", "same_target_only",
        "same_greedy_only", "different_both",
    }


def test_alternate_seeds_and_sensitivity_reuse_the_same_gradients() -> None:
    """No model recomputation: everything is offline on the stored vectors."""

    values, norms = _gradients(rows=6)
    targets, greedy = _labels(rows=6)

    report = fidelity_report(
        values, norms, targets, greedy,
        production_dimension=512, production_seed=20240917,
    )

    assert report["alternate_summary"]["num_seeds"] == len(ALTERNATE_SKETCH_SEEDS)
    assert [entry["dimension"] for entry in report["sensitivity"]] == list(
        SENSITIVITY_DIMENSIONS
    )
    # The production seed is reported separately, not folded into the alternates.
    assert 20240917 not in [entry["seed"] for entry in report["alternate_seeds"]]
    assert report["gradient_dtype"] == "float32"
    assert report["accumulation_dtype"] == "float64"


def test_the_report_is_deterministic() -> None:
    values, norms = _gradients(rows=5)
    targets, greedy = _labels(rows=5)

    first = fidelity_report(
        values, norms, targets, greedy,
        production_dimension=256, production_seed=7,
        alternate_seeds=(1, 2), sensitivity_dimensions=(128,),
    )
    second = fidelity_report(
        values, norms, targets, greedy,
        production_dimension=256, production_seed=7,
        alternate_seeds=(1, 2), sensitivity_dimensions=(128,),
    )

    assert first["production"] == second["production"]
    assert np.array_equal(first["production_cosines"], second["production_cosines"])


def test_error_falls_as_the_sketch_widens() -> None:
    """Contextualizes K = 512 rather than asserting a threshold."""

    values, norms = _gradients(rows=8)
    targets, greedy = _labels(rows=8)

    report = fidelity_report(
        values, norms, targets, greedy,
        production_dimension=512, production_seed=20240917,
    )
    errors = {
        entry["dimension"]: entry["mean_absolute_error"]
        for entry in report["sensitivity"]
    }

    assert errors[1024] < errors[128]


# -- the scientific observable, not just per-pair error -----------------------


def test_subgroup_delta_matches_a_brute_force_computation() -> None:
    from llm_behavior_lab.analysis.countsketch_fidelity import subgroup_delta

    values, norms = _gradients(rows=6)
    labels = np.array([1, 1, 1, 2, 2, 3], dtype=np.int64)
    cosines = cosine_from_gram(values, norms)

    result = subgroup_delta(cosines, labels)

    rows, columns = np.triu_indices(6, k=1)
    same = labels[rows] == labels[columns]
    assert result["within"] == pytest.approx(float(cosines[rows, columns][same].mean()))
    assert result["between"] == pytest.approx(
        float(cosines[rows, columns][~same].mean())
    )
    assert result["num_within_pairs"] == int(same.sum())
    assert result["num_between_pairs"] == int((~same).sum())


def test_subgroup_delta_is_unavailable_without_both_pair_kinds() -> None:
    """Reported as unavailable rather than fabricated."""

    from llm_behavior_lab.analysis.countsketch_fidelity import subgroup_delta

    values, norms = _gradients(rows=4)
    cosines = cosine_from_gram(values, norms)

    assert subgroup_delta(cosines, np.array([7, 7, 7, 7]))["available"] is False
    assert subgroup_delta(cosines, np.array([1, 2, 3, 4]))["available"] is False
    assert subgroup_delta(cosines, np.array([1, 1, 2, 3]))["available"] is True


def test_the_report_carries_exact_and_sketched_deltas_and_their_error() -> None:
    values, norms = _gradients(rows=8)
    targets = np.array([1, 1, 1, 2, 2, 3, 4, 5], dtype=np.int64)
    greedy = np.array([9, 9, 8, 8, 7, 7, 6, 5], dtype=np.int64)

    report = fidelity_report(
        values, norms, targets, greedy,
        production_dimension=512, production_seed=20240917,
        alternate_seeds=(1, 2, 3), sensitivity_dimensions=(256, 512),
    )

    for name in ("target", "greedy"):
        entry = report["subgroup_deltas"][name]
        assert entry["exact"]["available"] and entry["sketch"]["available"]
        assert entry["delta_error"] == pytest.approx(
            entry["sketch"]["delta"] - entry["exact"]["delta"]
        )
        # The realization-to-realization spread of that same statistic.
        assert len(report["alternate_delta_errors"][name]) == 3
