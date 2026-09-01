"""Publishing a finalized run must be all-or-nothing across four files.

The record's metadata carries the metrics artifact's hashes, so "this run is
finalized" is one fact spread over four files. For a ``metrics_only`` run the
rows are deleted immediately afterwards, which means a partially published
bundle is not a recoverable inconvenience -- it is a result that looks complete,
reads without complaint, and is wrong.

Every test here injects a failure at a different point and asserts the same
thing: no reachable state exposes a loadable record whose metrics are absent or
mismatched. The commit marker is the record's ``.json``, published last, because
``load_record`` requires it -- so a crash before that rename leaves a run that
does not load, which is the correct way to be incomplete.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from llm_behavior_lab.analysis import alignment_publication as pub
from llm_behavior_lab.analysis.alignment_metrics import (
    METRICS_JSON_NAME,
    METRICS_NPZ_NAME,
    AlignmentMetrics,
    load_alignment_metrics,
    load_finalized_record,
)
from llm_behavior_lab.analysis.records import RECORD_VERSION, load_record

VOCAB = 4
INITS = 2


def _metrics(**overrides) -> AlignmentMetrics:
    arrays = {
        "reference_target_delta": np.array([0.11, 0.22]),
        "cross_delta_cross": np.array(0.05),
        "map_count": np.array(4),
    }
    arrays.update(overrides.pop("arrays", {}))
    return AlignmentMetrics(
        arrays=arrays, provenance={"storage_mode": "metrics_only"}
    )


def _record_factory(reference):
    from llm_behavior_lab.analysis import InitializationExperimentRecord

    rng = np.random.default_rng(0)
    return InitializationExperimentRecord.build(
        corpus_counts=np.array([9, 7, 5, 3]),
        selected_target_counts=np.array([4, 3, 2, 1]),
        greedy_counts=rng.integers(1, 9, size=(INITS, VOCAB)),
        nucleus_counts=rng.integers(1, 9, size=(INITS, 1, VOCAB)),
        mean_predicted_probabilities=rng.dirichlet(np.ones(VOCAB), size=INITS),
        model_seeds=np.array([1, 2]),
        metadata={
            "num_positions": 8,
            "tokens": list("abcd"),
            "record_version": RECORD_VERSION,
            "analysis": {
                "gradient_analysis": {
                    "gradient_alignment": {
                        "storage_mode": "metrics_only",
                        "metrics_artifact": reference,
                    }
                }
            },
        },
    )


def _publish(directory) -> pub.PublicationResult:
    return pub.publish_finalized_run(
        directory, record_factory=_record_factory, metrics=_metrics()
    )


# -- the happy path ----------------------------------------------------------


def test_a_published_run_loads_with_its_metrics_attached(tmp_path) -> None:
    _publish(tmp_path)
    record = load_finalized_record(tmp_path)

    assert record.has_finalized_gradient_metrics
    assert record.storage_mode == "metrics_only"
    assert record.alignment_metrics["reference_target_delta"].tolist() == [0.11, 0.22]


def test_exactly_the_four_files_are_published(tmp_path) -> None:
    result = _publish(tmp_path)
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
        pub.PUBLISH_ORDER
    )
    assert result.published == pub.PUBLISH_ORDER


def test_no_staging_directory_survives(tmp_path) -> None:
    _publish(tmp_path)
    assert not any(path.name.startswith(".staging") for path in tmp_path.iterdir())


def test_the_record_json_is_published_last(tmp_path, monkeypatch) -> None:
    """It is the commit marker: ``load_record`` requires it, so its arrival is
    what makes the run loadable at all."""

    renamed: list[Path] = []
    original = os.replace

    def recording(src, dst):
        renamed.append(Path(dst))
        return original(src, dst)

    monkeypatch.setattr(os, "replace", recording)
    _publish(tmp_path)

    # Filtered to the run directory: the atomic writes into staging rename to
    # the same basenames, and counting those would measure the staging writer
    # rather than the publication order.
    published = [
        path.name for path in renamed
        if path.parent == tmp_path and path.name in pub.PUBLISH_ORDER
    ]
    assert published == list(pub.PUBLISH_ORDER)
    assert published[-1] == pub.COMMIT_MARKER


def test_dependencies_are_published_before_the_things_that_name_them(
    tmp_path, monkeypatch
) -> None:
    """The JSON names and hashes the NPZ; the record names and hashes both."""

    order: list[Path] = []
    original = os.replace

    def recording(src, dst):
        order.append(Path(dst))
        return original(src, dst)

    monkeypatch.setattr(os, "replace", recording)
    _publish(tmp_path)

    published = [
        path.name for path in order
        if path.parent == tmp_path and path.name in pub.PUBLISH_ORDER
    ]
    assert published.index(METRICS_NPZ_NAME) < published.index(METRICS_JSON_NAME)
    assert published.index(METRICS_JSON_NAME) < published.index(
        f"{pub.RECORD_NAME}.npz"
    )


# -- fault injection at each publication boundary ----------------------------


def _fail_on(monkeypatch, target: str):
    """Make the rename of ``target`` fail, as a crash at that instant would."""

    original = os.replace

    def failing(src, dst):
        if Path(dst).name == target:
            raise OSError(f"injected failure publishing {target}")
        return original(src, dst)

    monkeypatch.setattr(os, "replace", failing)


@pytest.mark.parametrize("target", pub.PUBLISH_ORDER)
def test_a_failure_at_any_rename_never_leaves_a_complete_looking_run(
    tmp_path, monkeypatch, target
) -> None:
    """The invariant, stated once and checked at every boundary.

    Either the run does not load at all, or it loads *with* metrics that match
    the hashes it records. There is no third outcome.
    """

    _fail_on(monkeypatch, target)
    with pytest.raises(OSError, match="injected failure"):
        _publish(tmp_path)

    monkeypatch.undo()
    try:
        record = load_finalized_record(tmp_path)
    except (FileNotFoundError, ValueError):
        return  # did not load: correctly incomplete
    assert record.has_finalized_gradient_metrics


@pytest.mark.parametrize("target", pub.PUBLISH_ORDER[:-1])
def test_a_failure_before_the_commit_marker_leaves_an_unloadable_run(
    tmp_path, monkeypatch, target
) -> None:
    """Anything short of the final rename must not load as a finished run."""

    _fail_on(monkeypatch, target)
    with pytest.raises(OSError):
        _publish(tmp_path)

    monkeypatch.undo()
    assert not (tmp_path / pub.COMMIT_MARKER).exists()
    with pytest.raises(FileNotFoundError):
        load_record(tmp_path)


def test_an_orphan_record_archive_does_not_load(tmp_path) -> None:
    """A crash between the record's two files leaves an ``.npz`` with no
    ``.json``. That must read as absent, not as an empty result."""

    _publish(tmp_path)
    (tmp_path / f"{pub.RECORD_NAME}.json").unlink()
    with pytest.raises(FileNotFoundError):
        load_record(tmp_path)


# -- staging validation ------------------------------------------------------


def test_validation_failure_publishes_nothing(tmp_path, monkeypatch) -> None:
    """The bundle is checked in staging, so a bad one never becomes visible."""

    monkeypatch.setattr(
        pub, "_validate_bundle",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad bundle")),
    )
    with pytest.raises(ValueError, match="bad bundle"):
        _publish(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_a_record_that_does_not_reference_its_metrics_is_refused(tmp_path) -> None:
    """Two files that happen to sit together are not a bundle."""

    def unreferencing(reference):
        return _record_factory({})

    with pytest.raises(ValueError, match="does not reference the metrics"):
        pub.publish_finalized_run(
            tmp_path, record_factory=unreferencing, metrics=_metrics()
        )
    assert list(tmp_path.iterdir()) == []


def test_a_hash_mismatch_between_record_and_metrics_is_refused(tmp_path) -> None:
    """The failure this prevents is a record read beside another run's metrics,
    which would look entirely well-formed."""

    def stale(reference):
        return _record_factory(dict(reference, npz_sha256="0" * 64))

    with pytest.raises(ValueError, match="published by different runs"):
        pub.publish_finalized_run(
            tmp_path, record_factory=stale, metrics=_metrics()
        )
    assert list(tmp_path.iterdir()) == []


def test_a_corrupted_metrics_archive_is_caught_on_the_defensive_reread(
    tmp_path, monkeypatch
) -> None:
    """Staging validated the contents; this catches a bad publication."""

    calls = {"n": 0}
    original = pub._validate_bundle

    def once(directory, reference):
        calls["n"] += 1
        if calls["n"] == 1:
            return original(directory, reference)
        blob = bytearray((directory / METRICS_NPZ_NAME).read_bytes())
        blob[-1] ^= 0xFF
        (directory / METRICS_NPZ_NAME).write_bytes(bytes(blob))
        return original(directory, reference)

    monkeypatch.setattr(pub, "_validate_bundle", once)
    with pytest.raises(ValueError, match="hashes to"):
        _publish(tmp_path)


# -- republication -----------------------------------------------------------


def test_republishing_replaces_the_whole_bundle(tmp_path) -> None:
    _publish(tmp_path)
    pub.publish_finalized_run(
        tmp_path, record_factory=_record_factory,
        metrics=_metrics(arrays={"cross_delta_cross": np.array(0.99)}),
    )
    record = load_finalized_record(tmp_path)
    assert record.alignment_metrics.scalar("cross_delta_cross") == pytest.approx(0.99)
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
        pub.PUBLISH_ORDER
    )


def test_the_reference_returned_matches_what_was_published(tmp_path) -> None:
    result = _publish(tmp_path)
    loaded = load_alignment_metrics(tmp_path, expected=result.metrics_reference)
    assert loaded.reference == result.metrics_reference
