"""The metrics artifact is the only copy of results that cannot be recomputed.

A ``metrics_only`` run deletes its per-position sketches once finalization
publishes. From that moment the artifact *is* the measurement: there is no row
to fall back on, no cheaper way to recompute it, and a corrupted or mismatched
pair is not a nuisance but a lost experiment. Every check here exists because
failing loudly at load time is the only remaining opportunity to notice.

Three properties carry that weight:

* the pair is **self-checking** -- the JSON records the NPZ's size and digest, so
  a half-published or mismatched combination is refused rather than analysed;
* the pair is **cross-checkable against its record**, so a record and a metrics
  artifact published by different runs cannot be read as one result;
* the archive is loadable with ``allow_pickle=False``, which an object-dtype
  member would silently break -- silently at *write* time, and fatally at read
  time, long after the rows are gone.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from llm_behavior_lab.analysis import alignment_metrics as am
from llm_behavior_lab.analysis.alignment_estimators import (
    ESTIMATOR_CONVENTION,
    NUMERICAL_EPOCH,
    REDUCTION_CONVENTION,
)


def _metrics(**overrides) -> am.AlignmentMetrics:
    arrays = {
        "reference_target_delta": np.array([0.13, 0.24, 0.11]),
        "reference_loss_temperatures": np.array([0.12, 0.6, 1.0]),
        "display_target_matrix": np.arange(9, dtype=np.float64).reshape(3, 3),
        "cross_delta_cross": np.array(0.0421),
        "nucleus_available": np.array([True, True, False]),
        "map_count": np.array(4),
    }
    arrays.update(overrides.pop("arrays", {}))
    provenance = {
        "storage_mode": "metrics_only",
        "declared_analyses": ["target", "greedy", "nucleus_control"],
        "permutation_seed": 20240918,
    }
    provenance.update(overrides.pop("provenance", {}))
    return am.AlignmentMetrics(arrays=arrays, provenance=provenance)


# -- round trip --------------------------------------------------------------


def test_a_published_pair_round_trips(tmp_path) -> None:
    original = _metrics()
    am.write_alignment_metrics(tmp_path, original)
    loaded = am.load_alignment_metrics(tmp_path)

    assert set(loaded.arrays) == set(original.arrays)
    for name, values in original.arrays.items():
        assert np.array_equal(loaded[name], values), name
    assert loaded.provenance["storage_mode"] == "metrics_only"
    assert loaded.schema_version == am.ALIGNMENT_METRICS_SCHEMA_VERSION


def test_scalars_survive_as_zero_dimensional_arrays(tmp_path) -> None:
    """The convention `gradient_cross_partition.npz` already uses: a scalar is a
    0-d array, not a length-one vector that later has to be squeezed."""

    am.write_alignment_metrics(tmp_path, _metrics())
    loaded = am.load_alignment_metrics(tmp_path)
    assert loaded["cross_delta_cross"].shape == ()
    assert loaded.scalar("cross_delta_cross") == pytest.approx(0.0421)


def test_the_archive_loads_without_pickle(tmp_path) -> None:
    am.write_alignment_metrics(tmp_path, _metrics())
    with np.load(tmp_path / am.METRICS_NPZ_NAME, allow_pickle=False) as archive:
        assert "reference_target_delta" in archive.files


def test_the_written_pair_is_the_documented_pair(tmp_path) -> None:
    reference = am.write_alignment_metrics(tmp_path, _metrics())
    assert (tmp_path / am.METRICS_NPZ_NAME).is_file()
    assert (tmp_path / am.METRICS_JSON_NAME).is_file()
    assert reference["npz_name"] == am.METRICS_NPZ_NAME
    assert reference["json_name"] == am.METRICS_JSON_NAME
    assert reference["npz_bytes"] > 0 and reference["json_bytes"] > 0
    assert len(reference["npz_sha256"]) == 64
    assert len(reference["json_sha256"]) == 64


# -- object dtype is refused, at write time ----------------------------------


def test_an_object_array_is_refused_before_it_reaches_disk(tmp_path) -> None:
    """Refused at construction, not at load.

    ``np.savez`` accepts an object array happily and the archive then raises on
    load -- by which time the run has ended and the rows are gone.
    """

    with pytest.raises(ValueError, match="object dtype"):
        _metrics(arrays={"policy": np.array(["a", None], dtype=object)})
    assert not (tmp_path / am.METRICS_NPZ_NAME).exists()


def test_fixed_width_unicode_is_accepted(tmp_path) -> None:
    """Strings belong in the JSON, but where an array is genuinely wanted a
    fixed-width Unicode dtype loads without pickle."""

    metrics = _metrics(arrays={"nucleus_design": np.array(["control", "matched"])})
    am.write_alignment_metrics(tmp_path, metrics)
    loaded = am.load_alignment_metrics(tmp_path)
    assert list(loaded["nucleus_design"]) == ["control", "matched"]


# -- the conventions are stamped from their owner ----------------------------


def test_the_conventions_are_stamped_from_the_estimator_module(tmp_path) -> None:
    am.write_alignment_metrics(tmp_path, _metrics())
    loaded = am.load_alignment_metrics(tmp_path)
    assert loaded.conventions == {
        "estimator_convention": ESTIMATOR_CONVENTION,
        "reduction_convention": REDUCTION_CONVENTION,
        "numerical_epoch": NUMERICAL_EPOCH,
    }


def test_a_convention_that_disagrees_with_this_build_is_refused(tmp_path) -> None:
    """An artifact must name the accumulation that actually produced it."""

    metrics = _metrics(provenance={"reduction_convention": "sequential_reduceat/v0"})
    with pytest.raises(ValueError, match="reduction_convention"):
        am.write_alignment_metrics(tmp_path, metrics)


# -- self-checking -----------------------------------------------------------


def test_a_corrupted_archive_is_detected(tmp_path) -> None:
    am.write_alignment_metrics(tmp_path, _metrics())
    path = tmp_path / am.METRICS_NPZ_NAME
    blob = bytearray(path.read_bytes())
    blob[-1] ^= 0xFF
    path.write_bytes(bytes(blob))
    with pytest.raises(ValueError, match="hashes to"):
        am.load_alignment_metrics(tmp_path)


def test_a_replaced_archive_is_detected(tmp_path) -> None:
    """A different but individually valid NPZ must not pass. This is the
    realistic corruption -- two runs' artifacts mixed in one directory, not a
    flipped bit."""

    am.write_alignment_metrics(tmp_path, _metrics())
    np.savez_compressed(
        tmp_path / am.METRICS_NPZ_NAME, reference_target_delta=np.array([9.9])
    )
    with pytest.raises(ValueError, match="hashes to"):
        am.load_alignment_metrics(tmp_path)


def test_a_missing_archive_is_reported_as_an_incomplete_pair(tmp_path) -> None:
    am.write_alignment_metrics(tmp_path, _metrics())
    (tmp_path / am.METRICS_NPZ_NAME).unlink()
    with pytest.raises(FileNotFoundError, match="incomplete"):
        am.load_alignment_metrics(tmp_path)


def test_a_missing_provenance_means_the_run_was_not_finalized(tmp_path) -> None:
    am.write_alignment_metrics(tmp_path, _metrics())
    (tmp_path / am.METRICS_JSON_NAME).unlink()
    with pytest.raises(FileNotFoundError, match="not finalized"):
        am.load_alignment_metrics(tmp_path)


def test_a_newer_schema_is_refused(tmp_path) -> None:
    am.write_alignment_metrics(tmp_path, _metrics())
    path = tmp_path / am.METRICS_JSON_NAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = am.ALIGNMENT_METRICS_SCHEMA_VERSION + 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        am.load_alignment_metrics(tmp_path)


# -- cross-reference against the record --------------------------------------


def test_a_matching_record_reference_validates(tmp_path) -> None:
    reference = am.write_alignment_metrics(tmp_path, _metrics())
    loaded = am.load_alignment_metrics(tmp_path, expected=reference)
    assert loaded.reference == reference


def test_a_record_expecting_different_metrics_is_refused(tmp_path) -> None:
    """The failure this prevents is a record read together with some other
    run's metrics, which would look entirely well-formed."""

    reference = am.write_alignment_metrics(tmp_path, _metrics())
    stale = dict(reference, npz_sha256="0" * 64)
    with pytest.raises(ValueError, match="published by different runs"):
        am.load_alignment_metrics(tmp_path, expected=stale)


# -- publication order and durability ----------------------------------------


def test_the_provenance_is_written_after_the_archive_it_names(monkeypatch, tmp_path) -> None:
    """Order is a correctness property, not a style choice.

    Reversed, a crash between the two renames would leave a JSON promising an
    archive that is absent or stale, which reads as a complete artifact.
    """

    renamed: list[str] = []
    import os as os_module

    original = os_module.replace

    def recording(src, dst):
        renamed.append(os_module.fspath(dst))
        return original(src, dst)

    monkeypatch.setattr(os_module, "replace", recording)
    am.write_alignment_metrics(tmp_path, _metrics())
    published = [name for name in renamed if not name.rsplit("/", 1)[-1].startswith(".")]
    assert published[0].endswith(am.METRICS_NPZ_NAME)
    assert published[1].endswith(am.METRICS_JSON_NAME)


def test_the_temporary_file_is_flushed_before_the_rename(monkeypatch, tmp_path) -> None:
    """A durable rename over undurable contents produces a correctly named,
    truncated file -- the one outcome an atomic write must not have."""

    import os as os_module

    events: list[str] = []
    original_fsync = os_module.fsync
    original_replace = os_module.replace

    def recording_fsync(fd):
        events.append("fsync")
        return original_fsync(fd)

    def recording_replace(src, dst):
        events.append("replace")
        return original_replace(src, dst)

    monkeypatch.setattr(os_module, "fsync", recording_fsync)
    monkeypatch.setattr(os_module, "replace", recording_replace)
    am.write_alignment_metrics(tmp_path, _metrics())
    assert events[0] == "fsync"
    assert "replace" in events
    assert events.index("fsync") < events.index("replace")


def test_no_temporary_files_are_left_behind(tmp_path) -> None:
    am.write_alignment_metrics(tmp_path, _metrics())
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        am.METRICS_JSON_NAME, am.METRICS_NPZ_NAME,
    ]


def test_republishing_replaces_cleanly(tmp_path) -> None:
    am.write_alignment_metrics(tmp_path, _metrics())
    second = _metrics(arrays={"cross_delta_cross": np.array(0.5)})
    am.write_alignment_metrics(tmp_path, second)
    loaded = am.load_alignment_metrics(tmp_path)
    assert loaded.scalar("cross_delta_cross") == pytest.approx(0.5)
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        am.METRICS_JSON_NAME, am.METRICS_NPZ_NAME,
    ]


# -- hashing helper ----------------------------------------------------------


def test_the_file_digest_is_streamed_not_slurped(tmp_path) -> None:
    """The same helper hashes multi-gibibyte sketch slabs, so it must not read a
    file whole."""

    path = tmp_path / "blob.bin"
    payload = bytes(range(256)) * 8192
    path.write_bytes(payload)
    import hashlib

    assert am.sha256_of_file(path) == hashlib.sha256(payload).hexdigest()
    assert am._HASH_BLOCK_BYTES < len(payload)
