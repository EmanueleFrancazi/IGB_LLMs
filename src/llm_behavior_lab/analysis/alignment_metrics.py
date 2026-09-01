"""The finalized gradient-alignment metrics artifact: schema, writing, reading.

A ``metrics_only`` run keeps no per-position sketches, so everything the
directional analyses report has to be computed while the rows still exist and
then written down. This module owns that artifact -- its shape, its atomic
publication and its validation -- and owns **no mathematics**. The estimators
live in :mod:`llm_behavior_lab.analysis.alignment_estimators`; a formula
restated here would be a second place for the two to disagree.

The artifact is a pair, deliberately:

``gradient_alignment_metrics.npz``
    every numeric result, scalars included as zero-dimensional arrays -- the
    convention ``gradient_cross_partition.npz`` already uses. Loaded with
    ``allow_pickle=False``, so **no object-dtype member may ever be written**;
    strings live in the JSON side or as fixed-width Unicode arrays.

``gradient_alignment_metrics.json``
    schema version, conventions, policies, selections, gate outcomes -- and the
    NPZ's size and SHA-256.

Why a pair rather than one file: the numbers want a binary container and the
provenance wants to be readable without NumPy. Why the JSON carries the NPZ's
hash: it makes the pair self-checking, so a half-published or mismatched
combination is detected on load instead of being quietly analysed.

Publication order and durability are not incidental. The JSON is written after
the NPZ it describes, both are flushed before their rename, and the containing
directory is synced afterwards. A reader that finds the JSON is guaranteed to
find the NPZ it names, with the contents it names.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from llm_behavior_lab.analysis.alignment_estimators import (
    ESTIMATOR_CONVENTION,
    NUMERICAL_EPOCH,
    REDUCTION_CONVENTION,
)
from llm_behavior_lab.analysis.records import _atomic_write_bytes, fsync_directory

__all__ = [
    "ALIGNMENT_METRICS_SCHEMA_VERSION",
    "METRICS_JSON_NAME",
    "METRICS_NPZ_NAME",
    "AlignmentMetrics",
    "load_alignment_metrics",
    "load_finalized_record",
    "metrics_artifact_reference",
    "sha256_of_file",
    "write_alignment_metrics",
]

#: Schema of the metrics artifact. Deliberately independent of
#: ``RECORD_VERSION`` and of the sketch-protocol schema: a metrics-layout change
#: should not force a record-version bump, and a record change should not
#: invalidate a metrics reader that still understands the layout.
ALIGNMENT_METRICS_SCHEMA_VERSION = 1

METRICS_NPZ_NAME = "gradient_alignment_metrics.npz"
METRICS_JSON_NAME = "gradient_alignment_metrics.json"

#: Read in 1 MiB blocks. The artifact is a couple of mebibytes, so this never
#: matters for it -- but the same helper hashes temporary sketch slabs, which are
#: gibibytes, and reading one of those whole would defeat the point.
_HASH_BLOCK_BYTES = 1024 * 1024


def sha256_of_file(path: str | Path) -> str:
    """Streaming SHA-256 of a file, as lowercase hex."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(_HASH_BLOCK_BYTES)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _check_storable(name: str, values: Any) -> np.ndarray:
    """Refuse anything an ``allow_pickle=False`` reader could not load back.

    Object arrays are the trap: ``np.savez`` accepts them happily and the
    resulting archive then raises on load, long after the run that produced it
    has ended and the rows it was derived from have been deleted.
    """

    array = np.asarray(values)
    if array.dtype == object:
        raise ValueError(
            f"Metrics array {name!r} has object dtype. The artifact is read with "
            "allow_pickle=False, so it would be unloadable. Put strings and "
            "labels in the JSON side, or store them as fixed-width Unicode."
        )
    return array


@dataclass(frozen=True)
class AlignmentMetrics:
    """One run's finalized alignment metrics, already validated.

    ``arrays`` is the numeric payload and ``provenance`` the readable one. They
    are kept apart rather than merged so that the question "what was measured"
    can be answered without NumPy, and the question "what were the numbers"
    without parsing prose.
    """

    arrays: dict[str, np.ndarray]
    provenance: dict[str, Any]
    schema_version: int = ALIGNMENT_METRICS_SCHEMA_VERSION
    #: Filled in by :func:`write_alignment_metrics` and :func:`load_alignment_metrics`;
    #: ``None`` for an artifact held only in memory.
    reference: dict[str, Any] | None = field(default=None)

    def __post_init__(self) -> None:
        for name, values in self.arrays.items():
            _check_storable(name, values)

    def __getitem__(self, name: str) -> np.ndarray:
        return self.arrays[name]

    def __contains__(self, name: str) -> bool:
        return name in self.arrays

    def scalar(self, name: str) -> float:
        """A zero-dimensional entry as a Python float."""

        return float(np.asarray(self.arrays[name]).reshape(()))

    @property
    def conventions(self) -> dict[str, str]:
        """The estimator, reduction and epoch this artifact was produced under.

        Present on every artifact so a stored number always says which
        accumulation produced it, and a cross-epoch comparison can refuse rather
        than silently proceed.
        """

        return {
            "estimator_convention": self.provenance["estimator_convention"],
            "reduction_convention": self.provenance["reduction_convention"],
            "numerical_epoch": self.provenance["numerical_epoch"],
        }


def _stamped_provenance(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Provenance with the conventions filled in from their single owner.

    A caller may pass them explicitly -- a migration might -- but a mismatch with
    the module that actually did the arithmetic is refused rather than recorded.
    """

    payload = dict(provenance)
    for key, owned in (
        ("estimator_convention", ESTIMATOR_CONVENTION),
        ("reduction_convention", REDUCTION_CONVENTION),
        ("numerical_epoch", NUMERICAL_EPOCH),
    ):
        declared = payload.setdefault(key, owned)
        if declared != owned:
            raise ValueError(
                f"{key} is {declared!r} but this build computes {owned!r}. A "
                "metrics artifact must name the accumulation that produced it."
            )
    payload["schema_version"] = ALIGNMENT_METRICS_SCHEMA_VERSION
    return payload


def write_alignment_metrics(
    directory: str | Path, metrics: AlignmentMetrics
) -> dict[str, Any]:
    """Publish the pair atomically and return its artifact reference.

    Order matters and is not an implementation detail: the NPZ is renamed into
    place first, then the JSON that names and hashes it. A reader that sees the
    JSON therefore always sees the NPZ it describes. The reverse order would
    make a crash between the two look like a complete artifact whose numbers
    happened to be missing.

    Returns:
        The reference block to embed in the record's metadata -- both filenames,
        both sizes and both SHA-256 digests -- so the record and the metrics can
        be cross-checked on load.
    """

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {name: _check_storable(name, values)
               for name, values in metrics.arrays.items()}

    npz_path = directory / METRICS_NPZ_NAME
    _atomic_write_bytes(npz_path, lambda path: np.savez_compressed(path, **payload))
    npz_bytes = npz_path.stat().st_size
    npz_digest = sha256_of_file(npz_path)

    provenance = _stamped_provenance(metrics.provenance)
    provenance["npz_name"] = METRICS_NPZ_NAME
    provenance["npz_bytes"] = int(npz_bytes)
    provenance["npz_sha256"] = npz_digest

    json_path = directory / METRICS_JSON_NAME
    _atomic_write_bytes(
        json_path,
        lambda path: path.write_text(
            json.dumps(provenance, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        ),
    )
    fsync_directory(directory)

    return metrics_artifact_reference(directory)


def metrics_artifact_reference(directory: str | Path) -> dict[str, Any]:
    """Names, sizes and digests of a published pair, for the record metadata."""

    directory = Path(directory)
    npz_path = directory / METRICS_NPZ_NAME
    json_path = directory / METRICS_JSON_NAME
    return {
        "npz_name": METRICS_NPZ_NAME,
        "npz_bytes": int(npz_path.stat().st_size),
        "npz_sha256": sha256_of_file(npz_path),
        "json_name": METRICS_JSON_NAME,
        "json_bytes": int(json_path.stat().st_size),
        "json_sha256": sha256_of_file(json_path),
    }


def load_alignment_metrics(
    directory: str | Path, *, expected: Mapping[str, Any] | None = None
) -> AlignmentMetrics:
    """Load and validate a published pair.

    Validation is not optional and not deferred. A ``metrics_only`` record has
    no rows to fall back on, so an artifact that fails any of these checks is
    the only copy of results that can no longer be recomputed -- reporting that
    loudly is the whole point.

    Args:
        directory: Where the pair lives.
        expected: Optional reference block from the record's metadata. When
            given, both digests must match, which is what ties a record to the
            exact metrics it was published with.

    Raises:
        FileNotFoundError: If either half is missing.
        ValueError: On a schema this reader does not understand, a digest
            mismatch, or a cross-reference disagreement.
    """

    directory = Path(directory)
    json_path = directory / METRICS_JSON_NAME
    npz_path = directory / METRICS_NPZ_NAME
    if not json_path.is_file():
        raise FileNotFoundError(
            f"No alignment-metrics provenance at {json_path}. A run without it "
            "was not finalized, whatever else is present."
        )
    if not npz_path.is_file():
        raise FileNotFoundError(
            f"Alignment-metrics provenance at {json_path} names {METRICS_NPZ_NAME}, "
            "which is absent. The pair is incomplete."
        )

    provenance = json.loads(json_path.read_text(encoding="utf-8"))
    schema = provenance.get("schema_version")
    if schema != ALIGNMENT_METRICS_SCHEMA_VERSION:
        raise ValueError(
            f"Alignment-metrics schema_version {schema!r} at {json_path}; this "
            f"reader understands {ALIGNMENT_METRICS_SCHEMA_VERSION}. A newer "
            "artifact may lay its arrays out differently, so it is not read."
        )

    digest = sha256_of_file(npz_path)
    if digest != provenance.get("npz_sha256"):
        raise ValueError(
            f"{npz_path} hashes to {digest} but its provenance records "
            f"{provenance.get('npz_sha256')}. The pair does not belong together."
        )
    size = npz_path.stat().st_size
    if size != provenance.get("npz_bytes"):
        raise ValueError(
            f"{npz_path} is {size} bytes but its provenance records "
            f"{provenance.get('npz_bytes')}."
        )

    if expected is not None:
        for key in ("npz_sha256", "json_sha256"):
            wanted = expected.get(key)
            found = digest if key == "npz_sha256" else sha256_of_file(json_path)
            if wanted is not None and wanted != found:
                raise ValueError(
                    f"The record expects {key} {wanted} but the artifact at "
                    f"{directory} has {found}. The record and the metrics beside "
                    "it were published by different runs."
                )

    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}

    return AlignmentMetrics(
        arrays=arrays,
        provenance=provenance,
        schema_version=ALIGNMENT_METRICS_SCHEMA_VERSION,
        reference=metrics_artifact_reference(directory),
    )


def load_finalized_record(directory: str | Path):
    """Load a record together with the metrics published beside it.

    This lives here rather than in :mod:`records` on purpose. That module is the
    schema layer and imports nothing from this package -- a property a test
    enforces -- so having it reach for the metrics reader would create the exact
    cycle the layering exists to prevent. The dependency runs one way: this
    module knows about records, records knows about nobody.

    A record whose metadata names a metrics artifact **must** be able to produce
    it. A ``metrics_only`` record has no rows to fall back on, so a dangling or
    mismatched reference is reported here rather than surfacing later as
    analyses that quietly appear unavailable.
    """

    import dataclasses

    from llm_behavior_lab.analysis.records import load_record

    directory = Path(directory)
    record = load_record(directory)
    reference = (
        record.gradient_analysis.get("gradient_alignment", {}).get("metrics_artifact")
    )
    if not reference:
        return record
    return dataclasses.replace(
        record,
        alignment_metrics=load_alignment_metrics(directory, expected=reference),
    )
