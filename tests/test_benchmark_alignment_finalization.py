"""The finalization benchmark must measure the production null, not a lookalike.

A benchmark is only evidence if the thing it times is the thing that will run.
The permutation null is the one part of finalization whose cost decides whether
the whole lifecycle is viable, so the harness carries two obligations that these
tests pin:

* its buffered, batched loop is **bitwise** the established
  :func:`_permutation_null` -- same draw sequence, same values, no tolerance
  spent anywhere;
* the faster ``blockwise`` reduction is reported as a *measured option* and is
  never the default, because it changes the float64 accumulation order inside a
  class sum.

The class-size profiles are generated here rather than read from a campaign
record, so these run on a fresh checkout. They are also the reason the timing is
meaningful: ``add.reduceat`` cost depends on block structure, and a few thousand
uneven classes behave nothing like 32768 singletons.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "benchmark_alignment_finalization.py"
)


def _module():
    """Load the script the way the other script tests do."""

    spec = importlib.util.spec_from_file_location(
        "benchmark_alignment_finalization", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


benchmark = _module()


# -- the loop under test is the production loop ------------------------------


def test_the_buffered_loop_is_bitwise_the_established_null() -> None:
    """No tolerance is spent here, and none may be."""

    result = benchmark.check_numerical_equivalence()
    assert result["identical"]
    assert result["mismatched_keys"] == []


def test_the_batched_draw_sequence_is_the_established_sequence() -> None:
    """Batching bounds how many orders are resident, never which ones."""

    assert benchmark.check_numerical_equivalence()["draw_sequence_identical"]


@pytest.mark.parametrize("batch", [1, 3, 5, 32, 64])
def test_every_batch_size_gives_the_same_result(batch) -> None:
    """The batch size is a memory knob. If it were a scientific one it would be
    a defect, so it is checked at sizes that do and do not divide the draw
    count."""

    generator = np.random.default_rng(3)
    counts = np.array([4, 2, 1, 3], dtype=np.int64)
    unit = generator.standard_normal((int(counts.sum()), 5))
    reference = benchmark.permutation_null_buffered(
        unit, counts, permutations=16, seed=5, draw_batch=16
    )
    assert benchmark.permutation_null_buffered(
        unit, counts, permutations=16, seed=5, draw_batch=batch
    ) == reference


def test_a_caller_supplied_gather_buffer_changes_nothing() -> None:
    generator = np.random.default_rng(4)
    counts = np.array([3, 3, 2], dtype=np.int64)
    unit = generator.standard_normal((8, 6))
    own = benchmark.permutation_null_buffered(
        unit, counts, permutations=12, seed=9, draw_batch=4
    )
    supplied = benchmark.permutation_null_buffered(
        unit, counts, permutations=12, seed=9, draw_batch=4,
        gather=np.empty_like(unit),
    )
    assert own == supplied


# -- the fast reduction is an option, not a default --------------------------


def test_the_default_reduction_is_the_production_one() -> None:
    assert benchmark.parse_args([]).reduction == "reduceat"
    assert benchmark.REDUCTIONS[0] == "reduceat"


def test_the_blockwise_reduction_really_does_move_the_result() -> None:
    """Pinned as a *difference*, deliberately.

    If this ever starts passing as bitwise equality, the two reductions have
    become the same computation and the review question this option exists to
    raise has quietly evaporated -- which is exactly when a silent adoption
    would slip through.
    """

    generator = np.random.default_rng(6)
    counts = np.array([300, 200, 120, 1], dtype=np.int64)
    unit = generator.standard_normal((int(counts.sum()), 64))
    exact = benchmark.permutation_null_buffered(
        unit, counts, permutations=8, seed=2, draw_batch=8, reduction="reduceat"
    )
    fast = benchmark.permutation_null_buffered(
        unit, counts, permutations=8, seed=2, draw_batch=8, reduction="blockwise"
    )
    assert exact != fast
    # ... but only in the last bits: the statistic is the same one.
    assert fast["delta_mean"] == pytest.approx(exact["delta_mean"], rel=1e-12)
    assert fast["delta_low"] == pytest.approx(exact["delta_low"], rel=1e-12)
    assert fast["delta_high"] == pytest.approx(exact["delta_high"], rel=1e-12)


def _same_including_nan(left: dict, right: dict) -> bool:
    """Dict equality that treats NaN in the same slot as agreement.

    Plain ``==`` cannot express this: ``nan != nan``, so two identical results
    full of NaN compare unequal. The comparison that matters is the one the
    plan's tolerance rule states -- the NaN *pattern* must match exactly, and
    the finite entries must agree.
    """

    if set(left) != set(right):
        return False
    for key, value in left.items():
        other = right[key]
        if isinstance(value, float) and np.isnan(value):
            if not (isinstance(other, float) and np.isnan(other)):
                return False
            continue
        if isinstance(other, float) and np.isnan(other):
            return False
        if value != other:
            return False
    return True


def test_singleton_classes_reduce_identically_under_both_strategies() -> None:
    """A block of one has no summation order to disagree about.

    It also has no within-class pair, so every within-derived statistic is NaN.
    That is the measurement, not a failure -- the high-cardinality bound exists
    to price the reduction, not to produce a clustering number.
    """

    generator = np.random.default_rng(8)
    counts = np.ones(16, dtype=np.int64)
    unit = generator.standard_normal((16, 9))
    exact = benchmark.permutation_null_buffered(
        unit, counts, permutations=8, seed=1, draw_batch=8, reduction="reduceat"
    )
    fast = benchmark.permutation_null_buffered(
        unit, counts, permutations=8, seed=1, draw_batch=8, reduction="blockwise"
    )
    assert _same_including_nan(exact, fast)
    assert np.isnan(exact["within_mean"])
    assert np.isnan(exact["delta_mean"])
    assert not np.isnan(exact["between_mean"])


def test_the_comparison_reports_reported_statistics_not_intermediates() -> None:
    comparison = benchmark.compare_reductions(
        positions=512, buckets=16, draws=8, seed=1
    )
    assert set(comparison["differences"]) == {
        "within_mean", "between_mean", "delta_mean",
        "delta_std", "delta_low", "delta_high",
    }
    assert comparison["max_relative"] < 1e-12


# -- class-size profiles -----------------------------------------------------


@pytest.mark.parametrize("profile", benchmark.PROFILES)
def test_every_profile_partitions_the_positions(profile) -> None:
    counts = benchmark.class_counts(profile, 4096, seed=1)
    assert int(counts.sum()) == 4096
    assert (counts > 0).all()


def test_the_profiles_span_the_regimes_they_claim_to() -> None:
    """The four regimes must actually differ in block structure, or timing four
    of them measures one thing three times."""

    classes = {
        profile: benchmark.class_counts(profile, 8192, seed=1).size
        for profile in benchmark.PROFILES
    }
    assert classes["greedy"] < classes["target"] < classes["nucleus"]
    assert classes["singleton"] == 8192


def test_the_greedy_profile_is_near_degenerate() -> None:
    """At initialization a handful of tokens absorb most greedy predictions."""

    counts = benchmark.class_counts("greedy", 8192, seed=1)
    assert np.sort(counts)[-5:].sum() >= 0.75 * 8192


def test_profiles_are_deterministic() -> None:
    assert np.array_equal(
        benchmark.class_counts("target", 2048, seed=3),
        benchmark.class_counts("target", 2048, seed=3),
    )


def test_an_unknown_profile_is_refused() -> None:
    with pytest.raises(SystemExit, match="Unknown profile"):
        benchmark.main(["--profiles", "not-a-profile", "--positions", "64"])


# -- normalization order -----------------------------------------------------


def test_rows_are_divided_by_the_exact_norms_not_multiplied_by_reciprocals() -> None:
    """The established numbers come from a division. The two do not round
    alike, and the difference is exactly the kind that goes unnoticed."""

    generator = np.random.default_rng(2)
    slab = generator.standard_normal((64, 8)).astype(np.float32)
    norms = generator.uniform(0.5, 2.0, size=64)
    expected = np.asarray(slab, dtype=np.float64) / norms[:, None]
    assert np.array_equal(benchmark.normalized_rows(slab, norms), expected)


# -- reporting ---------------------------------------------------------------


def test_the_job_accounting_counts_the_map_axis_once() -> None:
    """`jobs x M` map-jobs. Counting M twice would overstate the cost fourfold
    and counting it never would understate it by the same factor."""

    args = benchmark.parse_args(["--maps", "4"])
    assert args.jobs == benchmark.PRODUCTION_JOBS == 26
    assert args.jobs * args.maps == 104


def test_the_collection_baseline_is_the_recorded_one() -> None:
    """The budget denominator is a measured run, not a guess. It comes from the
    tracked logbook metadata, so it is auditable."""

    import json

    metadata = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "docs" / "experiment_logbook" / "supporting_material" / "metadata.json"
        ).read_text(encoding="utf-8")
    )
    recorded = metadata["analysis"]["gradient_analysis"]["seconds"]
    assert benchmark.parse_args([]).collection_seconds == pytest.approx(recorded)


def test_the_run_exits_non_zero_when_the_regression_ceiling_is_exceeded() -> None:
    """The hard limit has to be machine-readable, not just printed."""

    status = benchmark.main([
        "--positions", "2048", "--buckets", "64", "--timed-draws", "2",
        "--profiles", "target", "--ceiling-seconds", "0.001",
    ])
    assert status == 1


def test_the_run_exits_zero_when_the_ceiling_is_met() -> None:
    status = benchmark.main([
        "--positions", "2048", "--buckets", "64", "--timed-draws", "2",
        "--profiles", "target", "--ceiling-seconds", "1e7",
    ])
    assert status == 0


def test_the_stretch_target_is_reported_but_does_not_gate() -> None:
    """The 10% figure became a reported target rather than a hard limit.

    A run comfortably inside the ceiling must still succeed while missing the
    stretch target, or the two have been conflated again.
    """

    status = benchmark.main([
        "--positions", "2048", "--buckets", "64", "--timed-draws", "2",
        "--profiles", "target",
        "--collection-seconds", "1.0", "--ceiling-seconds", "1e7",
    ])
    assert status == 0


def test_the_accounting_drops_the_gather_buffer_on_the_production_path() -> None:
    """The production reduction indexes rows in place, so claiming a ``[D, K]``
    scratch buffer would overstate the workspace by 256 MiB at experiment
    scale."""

    production = benchmark.time_profile(
        "target", positions=512, buckets=16, timed_draws=1, draw_batch=4,
        seed=1, reduction="blockwise",
    )
    legacy = benchmark.time_profile(
        "target", positions=512, buckets=16, timed_draws=1, draw_batch=4,
        seed=1, reduction="reduceat",
    )
    assert "gather_buffer_f64" not in production["accounted_bytes"]
    assert "gather_buffer_f64" in legacy["accounted_bytes"]
    assert sum(production["accounted_bytes"].values()) < sum(
        legacy["accounted_bytes"].values()
    )
