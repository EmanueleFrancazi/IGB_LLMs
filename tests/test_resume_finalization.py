"""A finalization failure must cost minutes, not hours.

Collection is the expensive half. When finalization or publication fails after a
successful collection, the rows are hours of GPU time that can still answer every
question -- so the store keeps them and the work is finished from them later.

Refusing to resume a store whose rows were correctly deleted after publication is
necessary but proves nothing about recoverability. What proves it is this: seal a
store, fail the finalization, resume from the manifest, and get **byte-identical**
metrics to an uninterrupted run. Anything less than byte-identical would mean the
resumed path is a second implementation, and the two would drift.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from llm_behavior_lab.analysis.alignment_finalization import (
    FinalizationInputs,
    build_metrics_arrays,
    finalize_alignment_metrics,
)
from llm_behavior_lab.evaluation import sketch_store as ss
from llm_behavior_lab.evaluation.sketch_store import (
    SketchStoreLayout,
    StoreState,
    TemporarySketchStore,
)

LAYOUT = SketchStoreLayout(
    num_temperatures=2, num_positions=24, num_maps=3, num_buckets=6
)


def _inputs(seed: int = 11) -> FinalizationInputs:
    generator = np.random.default_rng(seed)
    return FinalizationInputs(
        loss_temperatures=np.array([0.6, 1.0]),
        norms=generator.uniform(0.5, 2.0, size=(2, LAYOUT.num_positions)),
        target_ids=generator.integers(0, 4, size=LAYOUT.num_positions),
        greedy_ids=generator.integers(0, 4, size=LAYOUT.num_positions),
        nucleus_labels=generator.integers(0, 3, size=(1, LAYOUT.num_positions)),
        sampling_temperatures=np.array([0.6]),
        permutations=8,
    )


def _filled_store(tmp_path: Path, *, seed: int = 5) -> TemporarySketchStore:
    store = TemporarySketchStore.create(
        manifest_dir=tmp_path / "tmp", bulk_dir=tmp_path / "tmp",
        layout=LAYOUT, run_id="resume",
    )
    generator = np.random.default_rng(seed)
    for temperature in range(LAYOUT.num_temperatures):
        for position in range(LAYOUT.num_positions):
            store.write_row(
                temperature, position,
                generator.standard_normal(
                    (LAYOUT.num_maps, LAYOUT.num_buckets)
                ).astype(np.float32),
            )
    store.seal()
    return store


def _finalize(store: TemporarySketchStore, inputs: FinalizationInputs):
    return build_metrics_arrays(
        finalize_alignment_metrics(
            store.iter_slabs(), inputs, num_maps=LAYOUT.num_maps
        ),
        inputs,
    )


# -- the failure keeps the measurement ---------------------------------------


def test_a_finalization_failure_leaves_a_recoverable_store(tmp_path) -> None:
    store = _filled_store(tmp_path)
    store.begin_finalization()
    store.mark_recoverable("metrics validation failed")

    assert store.state is StoreState.RECOVERABLE
    assert store.directory.exists()
    store.validate()


def test_a_recoverable_store_reopens_from_its_manifest_alone(tmp_path) -> None:
    """The manifest is the only handle: at the point a store is created there
    may be no run directory to look in."""

    store = _filled_store(tmp_path)
    store.begin_finalization()
    store.mark_recoverable("interrupted")

    reopened = ss.open_store(store.manifest_path)
    assert reopened.state is StoreState.RECOVERABLE
    reopened.validate()
    assert reopened.layout == LAYOUT


# -- the resumed result is the uninterrupted result --------------------------


def test_resuming_produces_byte_identical_metrics(tmp_path) -> None:
    """The property that makes resume trustworthy.

    Not "close" and not "statistically equivalent": the same rows through the
    same code must give the same bytes, or the resumed path has become a second
    implementation that will drift from the first.
    """

    inputs = _inputs()

    uninterrupted = _filled_store(tmp_path / "a")
    expected = _finalize(uninterrupted, inputs)

    interrupted = _filled_store(tmp_path / "b")
    interrupted.begin_finalization()
    interrupted.mark_recoverable("crashed before publication")
    resumed = ss.open_store(interrupted.manifest_path)
    resumed.validate()
    actual = _finalize(resumed, inputs)

    assert set(actual) == set(expected)
    for name, values in expected.items():
        left, right = np.asarray(actual[name]), np.asarray(values)
        assert left.dtype == right.dtype, name
        assert left.shape == right.shape, name
        if left.dtype.kind == "f":
            finite = np.isfinite(right)
            assert np.array_equal(finite, np.isfinite(left)), name
            assert np.array_equal(left[finite], right[finite]), name
        else:
            assert np.array_equal(left, right), name


def test_the_rows_are_unchanged_by_the_failed_attempt(tmp_path) -> None:
    """A failed finalization must not consume or mutate what it read."""

    store = _filled_store(tmp_path)
    before = {(t, m): np.array(slab) for t, m, slab in store.iter_slabs()}

    store.begin_finalization()
    store.mark_recoverable("failed")

    reopened = ss.open_store(store.manifest_path)
    for temperature, index, slab in reopened.iter_slabs():
        assert np.array_equal(slab, before[(temperature, index)])


# -- what resume refuses -----------------------------------------------------


def test_a_published_and_deleted_store_cannot_be_resumed(tmp_path) -> None:
    """Correct, but it is the recoverable case above that proves the design."""

    store = _filled_store(tmp_path)
    store.begin_finalization()
    store.mark_published()
    store.discard()

    reopened = ss.open_store(store.manifest_path)
    assert reopened.state is StoreState.DELETED
    with pytest.raises(RuntimeError, match="not readable"):
        list(reopened.iter_slabs())


def test_a_failed_collection_cannot_be_resumed(tmp_path) -> None:
    """Its rows are incomplete, so there is nothing to finish from."""

    store = TemporarySketchStore.create(
        manifest_dir=tmp_path / "tmp", bulk_dir=tmp_path / "tmp",
        layout=LAYOUT, run_id="partial",
    )
    store.write_row(
        0, 0, np.zeros((LAYOUT.num_maps, LAYOUT.num_buckets), np.float32)
    )
    store.fail_collection("died mid-collection")

    reopened = ss.open_store(store.manifest_path)
    assert reopened.state is StoreState.FAILED_COLLECTION
    with pytest.raises(RuntimeError, match="not readable"):
        list(reopened.iter_slabs())


def test_a_corrupted_recoverable_store_is_caught_before_it_is_used(tmp_path) -> None:
    """Resume validates before computing, so a damaged slab surfaces as a
    damaged slab rather than as a plausible number."""

    store = _filled_store(tmp_path)
    store.begin_finalization()
    store.mark_recoverable("interrupted")

    path = next(store.directory.glob("sketch_T00_M00.f32"))
    blob = bytearray(path.read_bytes())
    blob[0] ^= 0xFF
    path.write_bytes(bytes(blob))

    reopened = ss.open_store(store.manifest_path)
    with pytest.raises(ValueError, match="no longer matches the digest"):
        reopened.validate()


# -- the script's own refusals -----------------------------------------------


def test_the_resume_script_reuses_a_valid_published_pair(tmp_path) -> None:
    """A crash between the metrics rename and the record rename leaves a good
    artifact. Recomputing it would waste the time resume exists to save."""

    import importlib.util
    import sys

    from llm_behavior_lab.analysis.alignment_metrics import (
        AlignmentMetrics,
        write_alignment_metrics,
    )

    store = _filled_store(tmp_path)
    store.begin_finalization()
    store.mark_recoverable("crashed after the metrics were published")

    analyses = tmp_path / "analyses"
    analyses.mkdir()
    write_alignment_metrics(
        analyses,
        AlignmentMetrics(
            arrays={"reference_target_delta": np.array([0.5])},
            provenance={"storage_mode": "metrics_only"},
        ),
    )

    path = (
        Path(__file__).resolve().parents[1]
        / "scripts" / "resume_gradient_finalization.py"
    )
    spec = importlib.util.spec_from_file_location("resume_script", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    status = module.main([str(store.manifest_path), "--analyses-dir", str(analyses)])
    assert status == 0


def test_the_resume_script_refuses_a_conflicting_pair(tmp_path, capsys) -> None:
    """Two runs writing one directory is not something to guess about."""

    import importlib.util
    import sys

    from llm_behavior_lab.analysis.alignment_metrics import (
        METRICS_NPZ_NAME,
        AlignmentMetrics,
        write_alignment_metrics,
    )

    store = _filled_store(tmp_path)
    store.begin_finalization()
    store.mark_recoverable("crashed")

    analyses = tmp_path / "analyses"
    analyses.mkdir()
    write_alignment_metrics(
        analyses,
        AlignmentMetrics(
            arrays={"reference_target_delta": np.array([0.5])},
            provenance={"storage_mode": "metrics_only"},
        ),
    )
    blob = bytearray((analyses / METRICS_NPZ_NAME).read_bytes())
    blob[-1] ^= 0xFF
    (analyses / METRICS_NPZ_NAME).write_bytes(bytes(blob))

    path = (
        Path(__file__).resolve().parents[1]
        / "scripts" / "resume_gradient_finalization.py"
    )
    spec = importlib.util.spec_from_file_location("resume_script2", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    assert module.main(
        [str(store.manifest_path), "--analyses-dir", str(analyses)]
    ) == 2
    assert "Refusing to overwrite" in capsys.readouterr().err
