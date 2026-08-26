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

import sys

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


# -- renderer integration ----------------------------------------------------


def _write_artifact(run_root, rows: int = 8):
    """A compact synthetic artifact in the layout a real run produces."""

    values, norms = _gradients(rows=rows)
    targets = np.array([1, 1, 2, 2, 3, 4, 5, 6][:rows], dtype=np.int64)
    greedy = np.array([9, 8, 9, 7, 6, 5, 4, 3][:rows], dtype=np.int64)
    report = fidelity_report(
        values, norms, targets, greedy,
        production_dimension=512, production_seed=20240917,
        alternate_seeds=(1, 2), sensitivity_dimensions=(128, 512),
    )
    destination = run_root / "sanity"
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination / "countsketch_fidelity.npz",
        selected_position_indices=np.arange(rows),
        selected_target_ids=targets,
        selected_greedy_ids=greedy,
        exact_norms=norms,
        exact_cosine_matrix=report["exact_cosines"],
        production_cosine_matrix=report["production_cosines"],
        k_values=np.asarray([entry["dimension"] for entry in report["sensitivity"]]),
        k_mae=np.asarray(
            [entry["mean_absolute_error"] for entry in report["sensitivity"]]
        ),
        k_rmse=np.asarray([entry["rmse"] for entry in report["sensitivity"]]),
    )
    return destination / "countsketch_fidelity.npz"


def test_the_artifact_resolves_from_the_analyses_directory(tmp_path) -> None:
    from llm_behavior_lab.analysis.countsketch_fidelity import sanity_artifact_path

    written = _write_artifact(tmp_path)

    assert sanity_artifact_path(tmp_path / "analyses") == written
    assert sanity_artifact_path(tmp_path / "analyses").is_file()


def test_a_missing_artifact_raises_rather_than_returning_empty(tmp_path) -> None:
    """The all-figures path catches this; --only must say what is wrong."""

    from llm_behavior_lab.analysis.countsketch_fidelity import (
        load_fidelity_artifact,
        sanity_artifact_path,
    )

    with pytest.raises(FileNotFoundError, match="fidelity artifact"):
        load_fidelity_artifact(sanity_artifact_path(tmp_path / "analyses"))


def test_the_loaded_report_reproduces_the_saved_matrices(tmp_path) -> None:
    from llm_behavior_lab.analysis.countsketch_fidelity import load_fidelity_artifact

    written = _write_artifact(tmp_path)

    report = load_fidelity_artifact(written)

    assert report["num_gradients"] == 8
    assert report["production_cosines"].shape == (8, 8)
    assert report["accumulation_dtype"] == "float64"
    # Errors are recomputed from the stored matrices, not re-projected.
    assert report["production"]["num_pairs"] == 28
    assert set(report["pair_types"]) == {
        "same_target_and_greedy", "same_target_only",
        "same_greedy_only", "different_both",
    }


def test_the_fidelity_figure_routes_to_sanity_checks(tmp_path) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    from llm_behavior_lab.analysis.countsketch_fidelity import load_fidelity_artifact
    from llm_behavior_lab.analysis.figures import plot_countsketch_fidelity

    written = _write_artifact(tmp_path)
    report = load_fidelity_artifact(written)

    paths = plot_countsketch_fidelity(report, tmp_path / "figures")

    assert len(paths) == 1
    assert paths[0].parent.name == "sanity_checks"
    assert paths[0].name == "figure23_countsketch_fidelity.svg"
    assert paths[0].stat().st_size > 0


def test_figure_23_is_categorised_as_a_sanity_check() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    from llm_behavior_lab.analysis.figures import figure_category

    assert figure_category("figure23_countsketch_fidelity") == "sanity_checks"
    # And the existing routing is untouched.
    assert figure_category("figure20_gradient_directional_clustering") == "main"
    assert figure_category("figure0_sampling_adequacy") == "sanity_checks"


# -- robustness must use production semantics ---------------------------------


def test_the_map_factory_drives_alternates_and_the_k_sweep() -> None:
    """Alternate seeds and K sensitivity must use the supplied construction."""

    values, norms = _gradients(rows=6)
    targets, greedy = _labels(rows=6)
    seen = []

    def factory(dimension, seed):
        seen.append((dimension, seed))
        generator = np.random.default_rng(90000 + seed + dimension)
        buckets = generator.integers(0, dimension, size=values.shape[1])
        signs = generator.integers(0, 2, size=values.shape[1]) * 2.0 - 1.0
        return buckets, signs

    report = fidelity_report(
        values, norms, targets, greedy,
        production_dimension=512, production_seed=20240917,
        map_factory=factory,
        alternate_seeds=(1, 2), sensitivity_dimensions=(128, 512),
    )

    # Every alternate seed and every K went through the factory.
    assert (512, 1) in seen and (512, 2) in seen
    assert (128, 1) in seen and (512, 1) in seen
    assert report["map_semantics"] == "production"


def test_without_a_factory_the_report_says_the_maps_are_generic() -> None:
    """So a generic run can never be mistaken for production robustness."""

    values, norms = _gradients(rows=5)
    targets, greedy = _labels(rows=5)

    report = fidelity_report(
        values, norms, targets, greedy,
        production_dimension=256, production_seed=20240917,
        alternate_seeds=(1,), sensitivity_dimensions=(256,),
    )

    assert report["map_semantics"] == "generic_numpy"


def test_the_production_row_ignores_the_factory_and_uses_the_given_map() -> None:
    """The production result is the experiment's own realization, not a rebuild."""

    values, norms = _gradients(rows=5)
    targets, greedy = _labels(rows=5)
    generator = np.random.default_rng(4)
    supplied = (
        generator.integers(0, 128, size=values.shape[1]),
        generator.integers(0, 2, size=values.shape[1]) * 2.0 - 1.0,
    )

    report = fidelity_report(
        values, norms, targets, greedy,
        production_dimension=128, production_seed=20240917,
        production_map=supplied,
        map_factory=lambda dimension, seed: (
            np.zeros(values.shape[1], dtype=np.int64),
            np.ones(values.shape[1]),
        ),
        alternate_seeds=(1,), sensitivity_dimensions=(128,),
    )

    direct = sketch_estimated_cosines(
        values, norms, dimension=128, seed=0, sketch_map=supplied
    )
    assert np.allclose(report["production_cosines"], direct, rtol=1e-12)


# -- runner lifecycle ---------------------------------------------------------


def _runner_source() -> str:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    return (root / "scripts" / "run_initialization_distribution_experiment.py").read_text()


def test_the_fidelity_write_happens_after_the_run_directory_exists() -> None:
    """The artifact needs a run directory, and `run` is created late in main().

    The write was originally placed beside the gradient loop, hundreds of lines
    before ``ExperimentRun.create``, so a real sanity run raised
    ``UnboundLocalError`` the moment it captured its gradients. Unit-testing the
    writer in isolation could never have caught that -- the ordering *is* the
    bug -- so this checks the control flow in the runner itself.
    """

    import re

    source = _runner_source()
    lines = source.splitlines()

    def line_of(pattern: str, *, skip_def: bool = False) -> int:
        for number, text in enumerate(lines, start=1):
            if pattern in text and not (skip_def and text.strip().startswith("def ")):
                return number
        raise AssertionError(f"{pattern!r} not found in the runner")

    created = line_of("run = ExperimentRun.create")
    written = line_of("_write_countsketch_fidelity(run,", skip_def=True)
    computed = line_of("gradient_result = compute_position_gradient_norms")

    assert computed < created, "the gradient loop still runs before the run exists"
    assert created < written, (
        "the fidelity artifact is written before `run` is assigned; "
        f"create at line {created}, write at line {written}"
    )

    # And exactly one call site, so a stray earlier one cannot creep back.
    calls = [
        number
        for number, text in enumerate(lines, start=1)
        if "_write_countsketch_fidelity(run," in text
        and not text.strip().startswith("def ")
    ]
    assert calls == [written]


def test_the_captured_gradients_are_released_after_the_write() -> None:
    """~393 MiB must not survive the analysis that consumes it."""

    source = _runner_source()

    assert "_write_countsketch_fidelity(run, gradient_result, protocol)" in source
    # The release follows the write, in that order.
    write_at = source.index("_write_countsketch_fidelity(run, gradient_result, protocol)")
    release_at = source.index("exact_gradients=None")
    assert write_at < release_at


def test_the_writer_still_guards_on_sanity_being_enabled() -> None:
    """Disabled by default: no artifact, no buffers, nothing written."""

    source = _runner_source()

    assert "if gradient_result is not None and gradient_result.exact_gradients is not None:" in source
    assert 'action="store_true"' in source
    assert "--countsketch-fidelity-sanity" in source


def test_the_sanity_destination_resolves_against_a_real_experiment_run(tmp_path):
    """Exercise the actual ExperimentRun object, not the runner's source text.

    WRITTEN BUT NOT EXECUTED in a NumPy-only workspace: ``experiment.run``
    imports torch directly, so the real class cannot be built without it.

    It earns its place on the server anyway. The earlier ordering test read the
    file and so could not see that ``run.directory`` does not exist -- the real
    run exposes its root only as ``run.paths.run_dir`` -- and only touching the
    real object catches that.
    """

    pytest.importorskip("torch", reason="ExperimentRun imports torch")

    from llm_behavior_lab.analysis.countsketch_fidelity import sanity_artifact_path
    from llm_behavior_lab.experiment.config import ExperimentSettings
    from llm_behavior_lab.experiment.run import ExperimentRun

    run = ExperimentRun.create(
        ExperimentSettings(name="countsketch_sanity_test", output_dir=tmp_path),
        run_id="testrun",
    )

    assert run.paths.run_dir.is_dir()
    assert run.paths.analyses_dir.parent == run.paths.run_dir
    assert not hasattr(run, "directory"), (
        "the runner must not depend on an ExperimentRun.directory attribute"
    )

    destination = run.paths.run_dir / "sanity"
    destination.mkdir(parents=True, exist_ok=True)
    artifact = destination / "countsketch_fidelity.npz"
    np.savez_compressed(artifact, probe=np.arange(3))

    assert artifact.parent.parent == run.paths.run_dir
    assert sanity_artifact_path(run.paths.analyses_dir) == artifact
    assert sanity_artifact_path(run.paths.analyses_dir).is_file()


def test_the_runner_uses_the_canonical_run_root() -> None:
    """Pins the attribute that actually exists, so the crash cannot return."""

    source = _runner_source()

    assert 'run.paths.run_dir / "sanity"' in source
    assert "run.directory" not in source


# -- estimator range and deviation diagnostics --------------------------------


def test_the_projected_cosine_is_derivable_from_the_production_matrix() -> None:
    """No sketches and no rerun: the diagonal already carries ||S(g)||."""

    from llm_behavior_lab.analysis.countsketch_fidelity import (
        count_sketch_matrix,
        projected_space_cosines,
    )

    values, norms = _gradients(rows=6)
    production = sketch_estimated_cosines(values, norms, dimension=256, seed=9)

    derived = projected_space_cosines(production)

    # Recompute it the direct way and require agreement.
    buckets, signs = count_sketch_matrix(values.shape[1], 256, 9)
    projected = np.zeros((6, 256))
    for row in range(6):
        np.add.at(projected[row], buckets, values[row].astype(np.float64) * signs)
    lengths = np.linalg.norm(projected, axis=1)
    direct = (projected @ projected.T) / np.outer(lengths, lengths)

    assert np.allclose(derived, direct, rtol=1e-10, atol=1e-12)


def test_the_projected_cosine_is_bounded_and_the_production_one_need_not_be() -> None:
    """Boundedness is a property of the comparator, not proof it is better."""

    from llm_behavior_lab.analysis.countsketch_fidelity import projected_space_cosines

    values, norms = _gradients(rows=6)
    production = sketch_estimated_cosines(values, norms, dimension=32, seed=4)
    projected = projected_space_cosines(production)

    assert projected.min() >= -1.0 - 1e-9
    assert projected.max() <= 1.0 + 1e-9
    assert np.allclose(np.diag(projected), 1.0, atol=1e-9)
    # The production estimator carries no such guarantee; its diagonal is
    # ||S(g)||^2 / ||g||^2, which is not 1.
    assert not np.allclose(np.diag(production), 1.0)


def test_identical_vectors_give_a_projected_cosine_of_one() -> None:
    from llm_behavior_lab.analysis.countsketch_fidelity import projected_space_cosines

    values, _ = _gradients(rows=1)
    duplicated = np.vstack([values, values])
    norms = np.linalg.norm(duplicated.astype(np.float64), axis=1)

    projected = projected_space_cosines(
        sketch_estimated_cosines(duplicated, norms, dimension=128, seed=3)
    )

    assert np.allclose(projected, 1.0, atol=1e-9)


def test_the_deviation_report_counts_out_of_range_estimates_without_clipping() -> None:
    """Values outside [-1, 1] are a measured property, not something to hide."""

    from llm_behavior_lab.analysis.countsketch_fidelity import deviation_report

    exact = np.array([[1.0, 0.2], [0.2, 1.0]])
    estimated = np.array([[1.0, 1.4], [1.4, 1.0]])

    report = deviation_report(exact, estimated)

    assert report["num_pairs"] == 1
    assert report["estimated_max"] == pytest.approx(1.4)   # not clipped
    assert report["num_above_plus_one"] == 1
    assert report["num_below_minus_one"] == 0
    assert report["fraction_outside_unit_interval"] == pytest.approx(1.0)
    assert report["bias"] == pytest.approx(1.2)
    assert report["max_absolute_error"] == pytest.approx(1.2)


def test_deviation_percentiles_are_ordered_and_report_their_support() -> None:
    from llm_behavior_lab.analysis.countsketch_fidelity import deviation_report

    values, norms = _gradients(rows=8)
    exact = cosine_from_gram(values, norms)
    estimated = sketch_estimated_cosines(values, norms, dimension=128, seed=6)

    report = deviation_report(exact, estimated)

    assert report["num_pairs"] == 28
    assert (
        report["median_absolute_error"]
        <= report["p95_absolute_error"]
        <= report["p99_absolute_error"]
        <= report["max_absolute_error"]
    )
    assert report["signed_min"] <= report["bias"] <= report["signed_max"]


def test_both_estimators_are_measured_against_the_same_exact_matrix() -> None:
    from llm_behavior_lab.analysis.countsketch_fidelity import estimator_comparison

    values, norms = _gradients(rows=7)
    exact = cosine_from_gram(values, norms)
    production = sketch_estimated_cosines(values, norms, dimension=256, seed=8)

    comparison = estimator_comparison(exact, production)

    assert set(comparison) == {
        "exact_norm_estimator", "projected_space_estimator", "projected_cosines",
    }
    for name in ("exact_norm_estimator", "projected_space_estimator"):
        assert comparison[name]["num_pairs"] == 21
    # The bounded one cannot report out-of-range values, by construction.
    assert comparison["projected_space_estimator"]["fraction_outside_unit_interval"] == 0.0


# -- methodology: sketch width against independent map ensembles --------------
#
# The design question is where a fixed coordinate budget should go. These tests
# guard the machinery that answers it: folding must be an identity rather than a
# subsample, ensembles must be plain arithmetic means, and every configuration
# must see the same gradients and the same pairs.


def _production_factory():
    """The real production map builder, as the sweep must be given it."""

    from llm_behavior_lab.evaluation.position_gradients import production_sketch_map

    sizes = [1024, 2048, 1024]
    assert sum(sizes) == P
    return sizes, lambda dimension, seed: production_sketch_map(sizes, dimension, seed)


def test_folding_uses_every_source_bucket_exactly_once() -> None:
    from llm_behavior_lab.analysis.countsketch_fidelity import fold_sketches

    # A one-hot per bucket: folding must place each of them in exactly one
    # target bucket, so the folded matrix is a partition indicator.
    wide = np.eye(64, dtype=np.float64)
    folded = fold_sketches(wide, 16)

    assert folded.shape == (64, 16)
    assert np.array_equal(folded.sum(axis=1), np.ones(64))
    # Each target bucket receives exactly K_max / K sources -- nothing dropped,
    # nothing counted twice.
    assert np.array_equal(folded.sum(axis=0), np.full(16, 4.0))


def test_folding_is_not_coordinate_subsampling() -> None:
    from llm_behavior_lab.analysis.countsketch_fidelity import fold_sketches

    wide = np.arange(1, 17, dtype=np.float64).reshape(1, 16)
    folded = fold_sketches(wide, 4)

    # Subsampling would return four of the sixteen entries; folding returns sums
    # over residue classes, so the total mass is preserved exactly.
    assert folded.sum() == wide.sum()
    assert np.array_equal(folded, np.array([[1 + 5 + 9 + 13, 2 + 6 + 10 + 14,
                                             3 + 7 + 11 + 15, 4 + 8 + 12 + 16]]))


def test_folding_a_sketch_equals_applying_the_folded_bucket_map() -> None:
    from llm_behavior_lab.analysis.countsketch_fidelity import (
        fold_sketches,
        sketch_gradients,
    )

    _, factory = _production_factory()
    values, _ = _gradients(rows=5)
    buckets, signs = factory(1024, 4242)

    wide = sketch_gradients(values, buckets, signs, 1024)
    folded = fold_sketches(wide, 256)
    direct = sketch_gradients(values, buckets % 256, signs, 256)

    assert np.allclose(folded, direct, rtol=0, atol=1e-9)


def test_a_folded_width_must_divide_the_source_width() -> None:
    from llm_behavior_lab.analysis.countsketch_fidelity import fold_sketches

    with pytest.raises(ValueError, match="must divide"):
        fold_sketches(np.zeros((2, 96)), 64)


def test_the_production_map_is_unchanged_by_this_work() -> None:
    """The historical realization must stay bit-identical, folding aside."""

    sizes, factory = _production_factory()
    buckets, signs = factory(512, 20240917)

    from llm_behavior_lab.evaluation.position_gradients import production_sketch_map

    expected_buckets, expected_signs = production_sketch_map(sizes, 512, 20240917)
    assert np.array_equal(buckets, expected_buckets)
    assert np.array_equal(signs, expected_signs)
    assert set(np.unique(signs)) == {-1.0, 1.0}


def test_power_of_two_widths_fold_out_of_the_wider_draw() -> None:
    """An observed property of this construction, pinned so a change is noticed.

    ``torch.randint`` over a power-of-two range takes the low bits, so the map
    drawn directly at ``K`` happens to equal the ``K_max`` map folded. Nothing in
    the sweep relies on it -- folding is done explicitly -- but it means the
    folded ``K = 512`` row IS the historical production map rather than merely a
    statistical twin, which is worth knowing when reading the results.
    """

    sizes, factory = _production_factory()
    wide, wide_signs = factory(4096, 20240917)
    for dimension in (512, 1024, 2048):
        narrow, narrow_signs = factory(dimension, 20240917)
        assert np.array_equal(wide % dimension, narrow)
        assert np.array_equal(wide_signs, narrow_signs)


def test_the_fast_projection_matches_the_scatter_add_projection() -> None:
    from llm_behavior_lab.analysis.countsketch_fidelity import (
        _project,
        sketch_gradients,
    )

    _, factory = _production_factory()
    values, _ = _gradients(rows=4)
    buckets, signs = factory(256, 11)

    assert np.allclose(
        sketch_gradients(values, buckets, signs, 256),
        _project(values, buckets, signs, 256),
        rtol=0, atol=1e-9,
    )


def test_the_single_map_ensemble_reproduces_the_production_estimator() -> None:
    from llm_behavior_lab.analysis.countsketch_fidelity import (
        cosines_from_sketches,
        sketch_gradients,
    )

    _, factory = _production_factory()
    values, norms = _gradients(rows=6)
    buckets, signs = factory(512, 20240917)

    through_sketches = cosines_from_sketches(
        sketch_gradients(values, buckets, signs, 512), norms
    )
    production = sketch_estimated_cosines(
        values, norms, dimension=512, seed=20240917, sketch_map=(buckets, signs)
    )

    assert np.allclose(through_sketches, production, rtol=0, atol=1e-9)


def test_the_ensemble_estimate_is_the_arithmetic_mean_of_its_members() -> None:
    from llm_behavior_lab.analysis.countsketch_fidelity import (
        METHODOLOGY_SEEDS,
        methodology_sweep,
    )

    _, factory = _production_factory()
    values, norms = _gradients(rows=6)
    targets, greedy = _labels(rows=6)

    report = methodology_sweep(
        values, norms, targets, greedy,
        map_factory=factory, k_max=512, dimensions=(256, 512),
        ensemble_sizes=(1, 2), seeds=METHODOLOGY_SEEDS[:4],
    )

    cell = next(
        entry for entry in report["grid"]
        if entry["dimension"] == 256 and entry["ensemble_size"] == 2
    )
    members = report["per_map_cosines"][256]
    expected = (members[0] + members[1]) / 2.0
    first = cell["ensembles"][0]
    from llm_behavior_lab.analysis.countsketch_fidelity import deviation_report

    assert first["seeds"] == tuple(METHODOLOGY_SEEDS[:2])
    assert first["deviation"] == deviation_report(report["exact_cosines"], expected)


def test_disjoint_ensembles_never_share_a_map() -> None:
    from llm_behavior_lab.analysis.countsketch_fidelity import (
        METHODOLOGY_SEEDS,
        methodology_sweep,
    )

    _, factory = _production_factory()
    values, norms = _gradients(rows=5)
    targets, greedy = _labels(rows=5)

    report = methodology_sweep(
        values, norms, targets, greedy,
        map_factory=factory, k_max=512, dimensions=(512,),
        ensemble_sizes=(2,), seeds=METHODOLOGY_SEEDS[:6],
    )

    cell = report["grid"][0]
    assert cell["num_ensembles"] == 3
    seen = [seed for member in cell["ensembles"] for seed in member["seeds"]]
    assert len(seen) == len(set(seen))


def test_the_seed_bank_excludes_the_production_and_alternate_seeds() -> None:
    from llm_behavior_lab.analysis.countsketch_fidelity import METHODOLOGY_SEEDS

    assert 20240917 not in METHODOLOGY_SEEDS
    assert not set(METHODOLOGY_SEEDS) & set(ALTERNATE_SKETCH_SEEDS)
    assert len(set(METHODOLOGY_SEEDS)) == len(METHODOLOGY_SEEDS)
    # Every studied ensemble size has to divide the bank, or an M would be
    # measured over fewer disjoint ensembles than intended.
    for size in (1, 2, 4, 8):
        assert len(METHODOLOGY_SEEDS) % size == 0


def test_different_seeds_give_genuinely_different_realizations() -> None:
    from llm_behavior_lab.analysis.countsketch_fidelity import (
        METHODOLOGY_SEEDS,
        cosines_from_sketches,
        sketch_gradients,
    )

    _, factory = _production_factory()
    values, norms = _gradients(rows=6)

    matrices = []
    for seed in METHODOLOGY_SEEDS[:3]:
        buckets, signs = factory(256, seed)
        matrices.append(cosines_from_sketches(
            sketch_gradients(values, buckets, signs, 256), norms
        ))

    upper = np.triu_indices(6, 1)
    for i in range(3):
        for j in range(i + 1, 3):
            assert not np.allclose(matrices[i], matrices[j])
            # Errors from independent maps must not line up.
            assert abs(np.corrcoef(matrices[i][upper], matrices[j][upper])[0, 1]) < 0.999


def test_every_configuration_shares_the_same_gradients_and_pair_count() -> None:
    from llm_behavior_lab.analysis.countsketch_fidelity import (
        METHODOLOGY_SEEDS,
        methodology_sweep,
    )

    _, factory = _production_factory()
    values, norms = _gradients(rows=7)
    targets, greedy = _labels(rows=7)

    report = methodology_sweep(
        values, norms, targets, greedy,
        map_factory=factory, k_max=1024, dimensions=(256, 512, 1024),
        ensemble_sizes=(1, 2, 4), seeds=METHODOLOGY_SEEDS[:8],
    )

    assert report["num_gradients"] == 7
    for entry in report["grid"]:
        assert entry["budget"] == entry["dimension"] * entry["ensemble_size"]
        for member in entry["ensembles"]:
            assert member["deviation"]["num_pairs"] == 21
            assert member["statistics"]["cross"]["num_same_pairs"] + \
                member["statistics"]["cross"]["num_different_pairs"] == 7 * 6


def test_the_sweep_never_clips_estimates() -> None:
    from llm_behavior_lab.analysis.countsketch_fidelity import (
        METHODOLOGY_SEEDS,
        methodology_sweep,
    )

    _, factory = _production_factory()
    # Deliberately narrow, so some estimate must leave the unit interval.
    values, norms = _gradients(rows=8)
    targets, greedy = _labels(rows=8)

    report = methodology_sweep(
        values, norms, targets, greedy,
        map_factory=factory, k_max=8, dimensions=(4, 8),
        ensemble_sizes=(1,), seeds=METHODOLOGY_SEEDS[:4],
    )

    escaped = [
        member["deviation"]["fraction_outside_unit_interval"] > 0
        for entry in report["grid"] for member in entry["ensembles"]
    ]
    assert any(escaped), "an 8-bucket sketch should leave [-1, 1] somewhere"


def test_class_means_from_a_cosine_matrix_match_the_row_based_version() -> None:
    from llm_behavior_lab.analysis.countsketch_fidelity import class_means_from_cosines
    from llm_behavior_lab.analysis.gradient_clustering import class_similarity_matrix

    generator = np.random.default_rng(3)
    # Row norms deliberately far from 1, which is the case the sketch produces.
    unit = generator.normal(size=(9, 16)) * generator.uniform(0.3, 2.0, size=(9, 1))
    labels = np.array([1, 1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int64)
    classes = np.unique(labels)

    expected = class_similarity_matrix(unit, labels, classes)["matrix"]
    actual = class_means_from_cosines(unit @ unit.T, labels, classes)

    assert np.allclose(actual, expected, rtol=0, atol=1e-12, equal_nan=True)


def test_a_single_member_class_has_no_measurable_diagonal() -> None:
    from llm_behavior_lab.analysis.countsketch_fidelity import class_means_from_cosines

    cosines = np.eye(3)
    labels = np.array([1, 2, 2], dtype=np.int64)
    matrix = class_means_from_cosines(cosines, labels, np.array([1, 2]))

    assert np.isnan(matrix[0, 0])
    assert np.isfinite(matrix[1, 1])


def test_cross_partition_from_a_cosine_matrix_matches_the_pooled_statistic() -> None:
    from llm_behavior_lab.analysis.countsketch_fidelity import (
        cross_partition_from_cosines,
    )
    from llm_behavior_lab.analysis.gradient_cross_partition import (
        pooled_cross_statistic,
    )

    generator = np.random.default_rng(11)
    rows = generator.normal(size=(10, 12))
    targets = np.array([1, 1, 2, 3, 3, 2, 4, 4, 1, 2], dtype=np.int64)
    greedy = np.array([2, 3, 2, 1, 4, 4, 3, 1, 1, 3], dtype=np.int64)

    expected = pooled_cross_statistic(rows, targets, greedy)
    actual = cross_partition_from_cosines(rows @ rows.T, targets, greedy)

    assert actual["num_same_pairs"] == expected["num_same_pairs"]
    assert actual["num_different_pairs"] == expected["num_different_pairs"]
    for key in ("c_same", "c_different", "delta_cross"):
        assert actual[key] == pytest.approx(expected[key], rel=0, abs=1e-12)


def test_final_statistics_use_the_same_pairs_on_exact_and_estimated() -> None:
    from llm_behavior_lab.analysis.countsketch_fidelity import (
        final_statistics,
        sketch_estimated_cosines,
    )

    _, factory = _production_factory()
    values, norms = _gradients(rows=8)
    targets, greedy = _labels(rows=8)
    buckets, signs = factory(256, 5)

    exact = final_statistics(cosine_from_gram(values, norms), targets, greedy)
    estimated = final_statistics(
        sketch_estimated_cosines(values, norms, dimension=256, seed=5,
                                 sketch_map=(buckets, signs)),
        targets, greedy,
    )

    for name in ("target", "greedy"):
        assert exact[name]["num_within_pairs"] == estimated[name]["num_within_pairs"]
        assert exact[name]["num_between_pairs"] == estimated[name]["num_between_pairs"]
    assert exact["cross"]["num_same_pairs"] == estimated["cross"]["num_same_pairs"]
    assert exact["cross"]["num_different_pairs"] == estimated["cross"]["num_different_pairs"]


def test_the_sweep_is_deterministic() -> None:
    from llm_behavior_lab.analysis.countsketch_fidelity import (
        METHODOLOGY_SEEDS,
        methodology_sweep,
    )

    _, factory = _production_factory()
    values, norms = _gradients(rows=6)
    targets, greedy = _labels(rows=6)
    arguments = dict(
        map_factory=factory, k_max=512, dimensions=(256, 512),
        ensemble_sizes=(1, 2), seeds=METHODOLOGY_SEEDS[:4],
    )

    first = methodology_sweep(values, norms, targets, greedy, **arguments)
    second = methodology_sweep(values, norms, targets, greedy, **arguments)

    assert np.array_equal(first["exact_cosines"], second["exact_cosines"])
    assert first["production"]["deviation"] == second["production"]["deviation"]
    # repr rather than ==: this fixture has no position whose target is another's
    # greedy token, so ``c_same`` is legitimately NaN, and NaN != NaN would read
    # as nondeterminism.
    assert repr(first["grid"]) == repr(second["grid"])


def test_an_absent_cross_pair_kind_is_reported_as_nan_not_fabricated() -> None:
    from llm_behavior_lab.analysis.countsketch_fidelity import (
        cross_partition_from_cosines,
    )

    cosines = np.full((3, 3), 0.5)
    # No position's target is any position's greedy token.
    result = cross_partition_from_cosines(
        cosines, np.array([1, 1, 2]), np.array([7, 8, 9])
    )

    assert result["num_same_pairs"] == 0
    assert np.isnan(result["c_same"])
    assert np.isnan(result["delta_cross"])
    assert result["num_different_pairs"] == 6
