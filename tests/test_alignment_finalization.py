"""Finalization is the last moment any row-level quantity can be computed.

A ``metrics_only`` run deletes its sketches immediately afterwards, so anything
this pass forgets is gone for good -- there is no cheaper recomputation and no
fallback. That makes two things worth testing far more carefully than the
arithmetic, which is already covered where it lives:

* **the job list is complete.** It is derived from the measured grids rather
  than hard-coded, so a run gets the jobs its own grid implies; the production
  grid must still come out at the 26 the contract declares.
* **the results match the established route.** Every consumer-facing quantity is
  compared against `gradient_clustering` / `gradient_cross_partition` under the
  declared epoch tolerance, with NaN and availability patterns matched exactly.

The seed derivation gets its own attention. An earlier draft derived permutation
seeds through ``hash()``, which Python randomizes per process -- the null would
have been irreproducible between two runs of the same experiment, and nothing
about the output would have looked wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from llm_behavior_lab.analysis import alignment_finalization as fin
from llm_behavior_lab.analysis.alignment_estimators import (
    class_factors,
    permutation_null,
    pooled_from_factors,
)

RTOL = 1e-12
ATOL = 1e-14

PRODUCTION_LOSS = np.array([0.12, 0.24, 0.36, 0.48, 0.60, 1.00, 1.20])
PRODUCTION_SAMPLING = np.array([0.12, 0.24, 0.36, 0.48, 0.60, 1.20])


def assert_matching(actual, expected, *, what: str) -> None:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    finite = np.isfinite(expected)
    assert np.array_equal(finite, np.isfinite(actual)), f"{what}: availability"
    np.testing.assert_allclose(
        actual[finite], expected[finite], rtol=RTOL, atol=ATOL, err_msg=what
    )


# -- the job plan ------------------------------------------------------------


def test_the_production_grid_yields_the_declared_twenty_six_nulls() -> None:
    """The number the benchmark priced and the contract enumerates.

    7 loss temperatures x (target + greedy) = 14 references, 6 nucleus control
    points at the canonical field, 6 matched points -- and the cross-partition
    job, which is *not* a row null.
    """

    jobs = fin.plan_jobs(PRODUCTION_LOSS, PRODUCTION_SAMPLING)
    row_nulls = [job for job in jobs if job.needs_row_null]
    assert len(row_nulls) == 26
    assert sum(job.kind == "target" for job in jobs) == 7
    assert sum(job.kind == "greedy" for job in jobs) == 7
    assert sum(job.design == "control" for job in jobs) == 6
    assert sum(job.design == "matched" for job in jobs) == 6
    assert sum(job.kind == "cross" for job in jobs) == 1


def test_the_cross_partition_job_needs_no_row_permutations() -> None:
    """Its null permutes label identities, not rows, so it is computed from
    group factors and never touches the slab twice."""

    jobs = fin.plan_jobs(PRODUCTION_LOSS, PRODUCTION_SAMPLING)
    cross = [job for job in jobs if job.kind == "cross"]
    assert len(cross) == 1
    assert not cross[0].needs_row_null


def test_control_points_all_sit_at_the_canonical_field() -> None:
    """`T_g` is pinned at 1 for the control design: only the grouping is heated.

    A control point computed in another gradient field would be a different
    geometry wearing the same label.
    """

    jobs = fin.plan_jobs(PRODUCTION_LOSS, PRODUCTION_SAMPLING)
    canonical = int(np.flatnonzero(PRODUCTION_LOSS == 1.00)[0])
    for job in jobs:
        if job.design == "control":
            assert job.loss_index == canonical


def test_matched_points_sit_at_their_own_sampling_temperature() -> None:
    jobs = fin.plan_jobs(PRODUCTION_LOSS, PRODUCTION_SAMPLING)
    for job in jobs:
        if job.design == "matched":
            assert PRODUCTION_LOSS[job.loss_index] == pytest.approx(
                PRODUCTION_SAMPLING[job.sampling_index]
            )


def test_a_sampling_temperature_never_measured_gets_no_matched_point() -> None:
    """No matched point may be invented for a field the run did not measure."""

    jobs = fin.plan_jobs(np.array([1.0, 0.5]), np.array([0.5, 0.9]))
    matched = [job for job in jobs if job.design == "matched"]
    assert len(matched) == 1
    assert matched[0].loss_index == 1


def test_the_plan_follows_the_measured_grid_not_a_hard_coded_one() -> None:
    jobs = fin.plan_jobs(np.array([1.0]), np.array([]))
    assert sum(job.needs_row_null for job in jobs) == 2  # target + greedy only


# -- seed derivation ---------------------------------------------------------


def test_seeds_are_derived_arithmetically_not_by_hashing() -> None:
    """``hash()`` is randomized per process; a seed built from it would make the
    null irreproducible between two runs of the same experiment."""

    job = fin.JobPlan("nucleus", 3, sampling_index=2, design="matched")
    assert job.seed_offset(1000) == job.seed_offset(1000)
    import subprocess
    import sys

    script = (
        "import sys; sys.path.insert(0, 'src');"
        "from llm_behavior_lab.analysis.alignment_finalization import JobPlan;"
        "print(JobPlan('nucleus', 3, sampling_index=2, design='matched')"
        ".seed_offset(1000))"
    )
    seeds = {
        subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, check=True,
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
        ).stdout.strip()
        for seed in ("0", "1")
    }
    assert len(seeds) == 1


def test_distinct_jobs_get_distinct_seeds() -> None:
    """Two groupings sharing a draw sequence would correlate their nulls."""

    jobs = fin.plan_jobs(PRODUCTION_LOSS, PRODUCTION_SAMPLING)
    seeds = [job.seed_offset(20240918) for job in jobs if job.needs_row_null]
    assert len(set(seeds)) == len(seeds)


# -- results against the established route -----------------------------------


def _inputs(positions=48, buckets=6, maps=2, seed=5):
    generator = np.random.default_rng(seed)
    loss = np.array([0.6, 1.0])
    sampling = np.array([0.6])
    norms = generator.uniform(0.5, 2.0, size=(loss.size, positions))
    rows = generator.standard_normal(
        (loss.size, positions, maps, buckets)
    ).astype(np.float32)
    inputs = fin.FinalizationInputs(
        loss_temperatures=loss,
        norms=norms,
        target_ids=generator.integers(0, 5, size=positions),
        greedy_ids=generator.integers(0, 5, size=positions),
        nucleus_labels=generator.integers(0, 4, size=(1, positions)),
        sampling_temperatures=sampling,
        permutations=16,
    )
    slabs = [
        (t, m, rows[t, :, m, :])
        for t in range(loss.size)
        for m in range(maps)
    ]
    return inputs, slabs, rows, maps


def test_reference_values_match_a_direct_computation() -> None:
    """The scheduler must not perturb what the estimators produce."""

    inputs, slabs, rows, maps = _inputs()
    result = fin.finalize_alignment_metrics(slabs, inputs, num_maps=maps)
    arrays = fin.build_metrics_arrays(result, inputs)

    for grouping, labels in (
        ("target", inputs.target_ids), ("greedy", inputs.greedy_ids)
    ):
        for loss_index in range(inputs.loss_temperatures.size):
            usable = inputs.norms[loss_index] > 0.0
            for map_index in range(maps):
                unit = (
                    rows[loss_index, :, map_index, :].astype(np.float64)[usable]
                    / inputs.norms[loss_index][usable][:, None]
                )
                classes = np.unique(labels[usable])
                factors = class_factors(unit, labels[usable], classes)
                within, between = pooled_from_factors(
                    factors.sums, factors.counts, factors.self_squared
                )
                assert_matching(
                    arrays[f"reference_{grouping}_within_per_map"][loss_index, map_index],
                    within, what=f"{grouping} within",
                )
                assert_matching(
                    arrays[f"reference_{grouping}_delta_per_map"][loss_index, map_index],
                    within - between, what=f"{grouping} delta",
                )


def test_the_ensemble_is_the_mean_over_maps() -> None:
    inputs, slabs, _, maps = _inputs()
    result = fin.finalize_alignment_metrics(slabs, inputs, num_maps=maps)
    arrays = fin.build_metrics_arrays(result, inputs)
    assert_matching(
        arrays["reference_target_delta"],
        arrays["reference_target_delta_per_map"].mean(axis=1),
        what="ensemble mean",
    )


def test_the_null_uncertainty_is_persisted() -> None:
    """`gradient_clustering_per_map` computes the spread of the null mean; the
    established artifact never stored it, so a reader could not see it."""

    inputs, slabs, _, maps = _inputs()
    arrays = fin.build_metrics_arrays(
        fin.finalize_alignment_metrics(slabs, inputs, num_maps=maps), inputs
    )
    for key in (
        "reference_target_null_delta_mean_per_map",
        "reference_target_null_delta_mean_sd",
        "reference_target_null_delta_mean_se",
        "nucleus_null_delta_mean_sd",
        "nucleus_null_delta_mean_se",
    ):
        assert key in arrays, key


def test_single_map_uncertainty_is_unavailable_not_zero() -> None:
    """One projection cannot say how far another would have landed."""

    inputs, slabs, _, _ = _inputs(maps=1)
    arrays = fin.build_metrics_arrays(
        fin.finalize_alignment_metrics(slabs, inputs, num_maps=1), inputs
    )
    assert bool(arrays["uncertainty_available"]) is False
    assert np.isnan(arrays["reference_target_delta_sd"]).all()
    assert not np.isnan(arrays["reference_target_delta"]).all()


def test_cross_partition_matches_the_established_route() -> None:
    from llm_behavior_lab.analysis.gradient_cross_partition import (
        pooled_cross_statistic,
    )

    inputs, slabs, rows, maps = _inputs()
    arrays = fin.build_metrics_arrays(
        fin.finalize_alignment_metrics(slabs, inputs, num_maps=maps), inputs
    )
    canonical = int(np.flatnonzero(inputs.loss_temperatures == 1.0)[0])
    usable = inputs.norms[canonical] > 0.0
    for map_index in range(maps):
        unit = (
            rows[canonical, :, map_index, :].astype(np.float64)[usable]
            / inputs.norms[canonical][usable][:, None]
        )
        legacy = pooled_cross_statistic(
            unit, inputs.target_ids[usable], inputs.greedy_ids[usable]
        )
        for name in ("c_same", "c_different", "delta_cross"):
            assert_matching(
                arrays[f"cross_{name}_per_map"][map_index], legacy[name],
                what=f"cross {name}",
            )


def test_nucleus_points_carry_both_designs_and_their_support() -> None:
    inputs, slabs, _, maps = _inputs()
    arrays = fin.build_metrics_arrays(
        fin.finalize_alignment_metrics(slabs, inputs, num_maps=maps), inputs
    )
    designs = set(arrays["nucleus_design"].tolist())
    assert designs == {"control", "matched"}
    assert arrays["nucleus_design"].dtype.kind == "U"
    assert "nucleus_support_singleton_fraction" in arrays
    assert "nucleus_support_fraction_positions_in_qualifying" in arrays


def test_no_array_has_object_dtype() -> None:
    """The archive is read with ``allow_pickle=False``."""

    inputs, slabs, _, maps = _inputs()
    arrays = fin.build_metrics_arrays(
        fin.finalize_alignment_metrics(slabs, inputs, num_maps=maps), inputs
    )
    for name, values in arrays.items():
        assert np.asarray(values).dtype != object, name


def test_positions_excluded_is_reported_per_temperature() -> None:
    """A field can zero out at one ``T_g`` and not another, so one number would
    misreport every other row."""

    inputs, slabs, _, maps = _inputs()
    inputs.norms[0][:3] = 0.0
    arrays = fin.build_metrics_arrays(
        fin.finalize_alignment_metrics(slabs, inputs, num_maps=maps), inputs
    )
    assert arrays["num_positions_excluded"].tolist() == [3, 0]


def test_zero_norm_rows_are_excluded_from_the_statistics() -> None:
    inputs, slabs, _, maps = _inputs()
    inputs.norms[1][:4] = 0.0
    arrays = fin.build_metrics_arrays(
        fin.finalize_alignment_metrics(slabs, inputs, num_maps=maps), inputs
    )
    canonical = int(np.flatnonzero(inputs.loss_temperatures == 1.0)[0])
    assert arrays["reference_target_num_positions"][canonical] == 44
