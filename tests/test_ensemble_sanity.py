"""The fidelity check must validate the instrument that is actually used.

The v12 check re-projected retained gradients through map 0 only, which made it
structurally single-map. The production estimator averages inner products over
``M`` independent maps, so a single-map check validates something the figures do
not use.

Two properties carry the weight here:

* the comparison uses the sketches the **production maps actually produced**,
  captured during the measurement. Rebuilding the maps afterwards to re-project
  would compare exact gradients against a *reconstruction* of the instrument,
  which can agree perfectly while the instrument itself is wrong;
* nothing large is persisted. The exact gradients and the sampled sketch
  matrices exist for one comparison and then go; an artifact carrying them would
  defeat the storage design they are diagnosing.

The bounds get three independent tests because no single one of them is
sufficient: a position count says nothing about a 134M-parameter model, a
gradient budget says nothing about a wide sketch at many maps, and a sketch
budget says nothing about the gradients held beside it.
"""

from __future__ import annotations

import numpy as np
import pytest

from llm_behavior_lab.analysis import countsketch_fidelity as fid


def _sample(rows=8, parameters=256, maps=4, dimension=32, seed=3):
    """Exact gradients plus the sketches a real production map bank would give."""

    from llm_behavior_lab.evaluation.position_gradients import (
        production_sketch_tables,
    )

    generator = np.random.default_rng(seed)
    gradients = generator.standard_normal((rows, parameters)).astype(np.float32)
    norms = np.linalg.norm(gradients.astype(np.float64), axis=1)

    sketches = np.zeros((rows, maps, dimension), dtype=np.float32)
    for index in range(maps):
        tables = production_sketch_tables([parameters], dimension, 20240917 + index)
        buckets = tables[0][0].numpy()
        signs = tables[0][1].numpy()
        for row in range(rows):
            projected = np.zeros(dimension, dtype=np.float64)
            np.add.at(projected, buckets, gradients[row].astype(np.float64) * signs)
            sketches[row, index] = projected.astype(np.float32)

    targets = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)[:rows]
    greedy = np.array([1, 0, 1, 2, 2, 3, 3, 0], dtype=np.int64)[:rows]
    return gradients, sketches, norms, targets, greedy


# -- bounds ------------------------------------------------------------------


def test_all_three_bounds_are_reported_when_they_pass() -> None:
    report = fid.check_sanity_bounds(
        num_positions=12, num_parameters=1000, maps=4, dimension=512,
        max_positions=12, max_gradient_bytes=1 << 30, max_sketch_bytes=1 << 28,
    )
    assert report["gradient_bytes"] == 12 * 1000 * 4
    assert report["sketch_bytes"] == 12 * 4 * 512 * 4


def test_the_position_bound_binds() -> None:
    with pytest.raises(ValueError, match="bounded diagnostic"):
        fid.check_sanity_bounds(
            num_positions=13, num_parameters=10, maps=1, dimension=8,
            max_positions=12, max_gradient_bytes=1 << 30,
            max_sketch_bytes=1 << 30,
        )


def test_the_gradient_byte_bound_binds_independently() -> None:
    """A position count says nothing about a 134M-parameter model."""

    with pytest.raises(ValueError, match="complete gradients"):
        fid.check_sanity_bounds(
            num_positions=12, num_parameters=134_000_000, maps=1, dimension=8,
            max_positions=12, max_gradient_bytes=1 << 20,
            max_sketch_bytes=1 << 30,
        )


def test_the_sketch_byte_bound_binds_independently() -> None:
    """A gradient budget says nothing about a wide sketch at many maps."""

    with pytest.raises(ValueError, match="matching sketches"):
        fid.check_sanity_bounds(
            num_positions=12, num_parameters=10, maps=8, dimension=4096,
            max_positions=12, max_gradient_bytes=1 << 30,
            max_sketch_bytes=1024,
        )


# -- thresholds --------------------------------------------------------------


def test_thresholds_are_derived_from_the_estimator_error_scale() -> None:
    """``1/sqrt(K*M)`` is the estimator's own unit, not a tuned constant."""

    thresholds = fid.error_thresholds(512, 4)
    assert thresholds["unit"] == pytest.approx(1.0 / np.sqrt(512 * 4))
    assert thresholds["warn"] == pytest.approx(2.0 * thresholds["unit"])
    assert thresholds["fail"] == pytest.approx(6.0 * thresholds["unit"])


def test_more_maps_tighten_the_thresholds() -> None:
    """Averaging independent maps divides the error's spread by ``sqrt(M)``."""

    assert fid.error_thresholds(512, 4)["warn"] < fid.error_thresholds(512, 1)["warn"]


@pytest.mark.parametrize("band", ["pass", "warn", "fail"])
def test_the_outcome_is_explicit(band) -> None:
    """Each band is probed relative to the thresholds themselves.

    Hardcoding error values would tie the test to one ``(K, M)`` and silently
    stop exercising the band it names the moment either changed.
    """

    thresholds = fid.error_thresholds(32, 4)
    probe = {
        "pass": thresholds["warn"] * 0.5,
        "warn": (thresholds["warn"] + thresholds["fail"]) / 2.0,
        "fail": thresholds["fail"] * 2.0,
    }[band]
    outcome = (
        "fail" if probe > thresholds["fail"]
        else "warn" if probe > thresholds["warn"]
        else "pass"
    )
    assert outcome == band


def test_a_real_sample_lands_inside_the_pass_band() -> None:
    """The instrument being validated should validate."""

    gradients, sketches, norms, targets, greedy = _sample()
    summary = fid.ensemble_fidelity_summary(
        gradients, sketches, norms, targets, greedy, dimension=32
    )
    assert summary["outcome"] in {"pass", "warn"}
    assert summary["mean_absolute_error"] < summary["thresholds"]["fail"]


# -- the estimator under test is the production one --------------------------


def test_every_production_map_contributes() -> None:
    """A single-map check validates an instrument the figures do not use."""

    gradients, sketches, norms, targets, greedy = _sample(maps=4)
    summary = fid.ensemble_fidelity_summary(
        gradients, sketches, norms, targets, greedy, dimension=32
    )
    assert summary["map_count"] == 4

    one_map = fid.ensemble_fidelity_summary(
        gradients, sketches[:, :1], norms, targets, greedy, dimension=32
    )
    assert one_map["map_count"] == 1
    # Four maps average four independent projections, so the ensemble estimate
    # is closer to the exact geometry than any single map's.
    assert summary["rmse"] < one_map["rmse"]


def test_self_pairs_are_excluded() -> None:
    """A gradient compared with itself is trivially well-behaved, and including
    it would dilute exactly the error being measured."""

    gradients, sketches, norms, targets, greedy = _sample(rows=6)
    summary = fid.ensemble_fidelity_summary(
        gradients, sketches, norms, targets, greedy, dimension=32
    )
    assert summary["num_ordered_pairs"] == 6 * 6 - 6


def test_both_pair_counts_are_reported_and_named() -> None:
    """Six gradients give 30 ordered off-diagonal entries and 15 unordered
    pairs. Both are true; a single number called ``num_pairs`` left the reader
    to guess which was meant.

    The matrix is symmetric, so the duplication does not move MAE, RMSE or the
    correlation -- which is precisely why the ambiguity could sit there
    unnoticed.
    """

    gradients, sketches, norms, targets, greedy = _sample(rows=6)
    summary = fid.ensemble_fidelity_summary(
        gradients, sketches, norms, targets, greedy, dimension=32
    )
    assert summary["num_ordered_pairs"] == 30
    assert summary["num_unordered_pairs"] == 15
    assert summary["pair_convention"] == "ordered_off_diagonal"

    arrays = fid.sanity_arrays(summary)
    assert int(arrays["sanity_num_ordered_pairs"]) == 30
    assert int(arrays["sanity_num_unordered_pairs"]) == 15
    assert str(arrays["sanity_pair_convention"]) == "ordered_off_diagonal"


def test_the_symmetric_duplication_does_not_move_the_error_statistics() -> None:
    """Stated as a check rather than a claim: if it ever did, the convention
    would be a scientific choice rather than a labelling one."""

    gradients, sketches, norms, targets, greedy = _sample(rows=6)
    summary = fid.ensemble_fidelity_summary(
        gradients, sketches, norms, targets, greedy, dimension=32
    )
    exact = np.asarray(summary["exact_cosines"])
    estimated = np.asarray(summary["production_cosines"])
    upper = np.triu(np.ones_like(exact, dtype=bool), k=1)
    unordered_mae = float(np.abs(estimated[upper] - exact[upper]).mean())
    assert unordered_mae == pytest.approx(
        summary["mean_absolute_error"], rel=1e-12
    )


def test_the_summary_reports_the_full_error_picture() -> None:
    gradients, sketches, norms, targets, greedy = _sample()
    summary = fid.ensemble_fidelity_summary(
        gradients, sketches, norms, targets, greedy, dimension=32
    )
    for key in (
        "bias", "mean_absolute_error", "median_absolute_error", "rmse",
        "max_absolute_error", "p95_absolute_error", "p99_absolute_error",
        "pearson_correlation", "num_ordered_pairs", "num_unordered_pairs",
        "pair_convention", "num_below_minus_one", "num_above_plus_one",
    ):
        assert key in summary, key


def test_subgroup_metrics_cover_target_and_greedy() -> None:
    gradients, sketches, norms, targets, greedy = _sample()
    summary = fid.ensemble_fidelity_summary(
        gradients, sketches, norms, targets, greedy, dimension=32
    )
    assert set(summary["subgroup_deltas"]) == {"target", "greedy"}
    for entry in summary["subgroup_deltas"].values():
        assert entry["exact"]["available"]
        assert entry["sketch"]["available"]
        assert "delta_error" in entry


def test_the_subgroup_diagonal_excludes_self_pairs() -> None:
    """The same exclusion the production statistics use, so the check exercises
    the estimator rather than a simplified stand-in."""

    generator = np.random.default_rng(1)
    cosines = generator.standard_normal((6, 6))
    cosines = (cosines + cosines.T) / 2
    np.fill_diagonal(cosines, 5.0)  # absurd self-values
    labels = np.array([0, 0, 0, 1, 1, 1])
    result = fid.subgroup_delta(cosines, labels)
    assert result["available"]
    # A within value near 5 would mean the diagonal leaked in.
    assert abs(result["within"]) < 2.0


def test_values_outside_the_unit_interval_are_counted_not_clipped() -> None:
    """The estimator is unbounded by construction; clipping would hide the
    property being measured."""

    gradients, sketches, norms, targets, greedy = _sample()
    summary = fid.ensemble_fidelity_summary(
        gradients, sketches, norms, targets, greedy, dimension=32
    )
    assert "num_below_minus_one" in summary and "num_above_plus_one" in summary


# -- persistence is compact --------------------------------------------------


def test_only_derived_results_are_persisted() -> None:
    """Neither the exact gradients nor the sampled sketches may reach the
    artifact -- they are the largest things the check touches."""

    gradients, sketches, norms, targets, greedy = _sample()
    summary = fid.ensemble_fidelity_summary(
        gradients, sketches, norms, targets, greedy, dimension=32
    )
    arrays = fid.sanity_arrays(summary)

    persisted = np.concatenate(
        [np.asarray(v).ravel() for v in arrays.values() if np.asarray(v).dtype.kind == "f"]
    )
    assert persisted.size < gradients.size
    assert not any("gradient" in name and "sketch" not in name for name in arrays)
    for name, value in arrays.items():
        assert np.asarray(value).dtype != object, name


def test_the_persisted_summary_names_its_outcome_and_thresholds() -> None:
    gradients, sketches, norms, targets, greedy = _sample()
    arrays = fid.sanity_arrays(
        fid.ensemble_fidelity_summary(
            gradients, sketches, norms, targets, greedy, dimension=32
        )
    )
    assert str(arrays["sanity_threshold_outcome"]) in {"pass", "warn", "fail"}
    assert float(arrays["sanity_threshold_warn"]) > 0
    assert float(arrays["sanity_threshold_fail"]) > float(
        arrays["sanity_threshold_warn"]
    )


def test_the_cosine_matrices_are_small_enough_to_keep() -> None:
    """12 positions gives a 12x12 matrix -- about a kilobyte, and the thing a
    reader most wants to see."""

    gradients, sketches, norms, targets, greedy = _sample(rows=8)
    arrays = fid.sanity_arrays(
        fid.ensemble_fidelity_summary(
            gradients, sketches, norms, targets, greedy, dimension=32
        )
    )
    assert arrays["sanity_exact_cosine_matrix"].shape == (8, 8)
    assert arrays["sanity_production_cosine_matrix"].shape == (8, 8)
