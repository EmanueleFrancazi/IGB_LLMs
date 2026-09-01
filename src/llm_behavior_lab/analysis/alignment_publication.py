"""Publishing a finalized run as one transaction over four files.

Atomic per-file writes are not enough here. The record's metadata carries the
metrics artifact's hashes, so "record present" and "metrics present and matching"
are one fact spread across four files, and any ordering that can expose half of
it produces a run that looks complete and is not -- which, for a ``metrics_only``
run whose rows have been deleted, is indistinguishable from a correct result
until someone reads the numbers.

The sequence:

1. write all four files into a private staging directory **under their final
   names**, so the authoritative loaders run against them unmodified;
2. validate the complete bundle *there*, record and metrics together;
3. publish by rename in dependency order -- metrics NPZ, metrics JSON, record
   NPZ, record JSON;
4. re-read from the real location as a defensive check.

The record's ``.json`` is renamed last and is therefore the commit marker:
:func:`load_record` requires it, so a crash before that rename leaves a run that
does not load -- correctly incomplete rather than apparently complete. A crash
after it leaves a bundle whose metrics were already validated at step 2.

Nothing here deletes anything. Releasing the temporary sketch store is the
caller's decision and must happen only after step 4 succeeds, because the rows
are the only thing that could regenerate what step 3 published.
"""

from __future__ import annotations

import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_behavior_lab.analysis.alignment_metrics import (
    METRICS_JSON_NAME,
    METRICS_NPZ_NAME,
    AlignmentMetrics,
    load_alignment_metrics,
    metrics_artifact_reference,
    write_alignment_metrics,
)
from llm_behavior_lab.analysis.records import fsync_directory

__all__ = ["PublicationResult", "publish_finalized_run"]

RECORD_NAME = "initialization_distribution"

#: The order files become visible. Dependencies first: nothing may reference a
#: file that is not yet in place.
PUBLISH_ORDER = (
    METRICS_NPZ_NAME,
    METRICS_JSON_NAME,
    f"{RECORD_NAME}.npz",
    f"{RECORD_NAME}.json",
)

#: The last rename. Its arrival is what makes the run loadable at all.
COMMIT_MARKER = PUBLISH_ORDER[-1]


@dataclass(frozen=True)
class PublicationResult:
    directory: Path
    metrics_reference: dict[str, Any]
    published: tuple[str, ...]


def publish_finalized_run(
    directory: str | Path,
    *,
    record_factory,
    metrics: AlignmentMetrics,
) -> PublicationResult:
    """Stage, validate, publish and re-read a finalized run.

    Args:
        directory: The run's ``analyses`` directory.
        record_factory: ``reference -> record``. Called with the metrics
            artifact reference so the record can embed the hashes it will later
            be validated against. A factory rather than a finished record
            because the record cannot be built until the metrics have been
            written and hashed.
        metrics: The finalized metrics to publish.

    Raises:
        Exception: Whatever validation raises, with nothing published. The
            caller must treat that as recoverable and keep its rows.
    """

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    staging = directory / f".staging.{uuid.uuid4().hex[:12]}"
    staging.mkdir()

    try:
        reference = write_alignment_metrics(staging, metrics)
        record = record_factory(reference)
        record.save(staging, name=RECORD_NAME)

        # Validate the complete bundle before anything becomes visible. Both
        # halves, together, through the real loaders -- a record that validates
        # alone says nothing about whether the metrics beside it are the ones it
        # names.
        _validate_bundle(staging, reference)

        for name in PUBLISH_ORDER:
            os.replace(staging / name, directory / name)
        fsync_directory(directory)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    # Defensive re-read from the real location. The bundle was already validated
    # in staging, so this is checking the publication rather than the contents.
    _validate_bundle(directory, reference)

    return PublicationResult(
        directory=directory,
        metrics_reference=metrics_artifact_reference(directory),
        published=PUBLISH_ORDER,
    )


def _validate_bundle(directory: Path, reference: dict[str, Any]) -> None:
    """Load the record and its metrics together, and check they belong together."""

    from llm_behavior_lab.analysis.alignment_metrics import load_finalized_record

    load_alignment_metrics(directory, expected=reference)
    record = load_finalized_record(directory)
    if not record.has_finalized_gradient_metrics:
        raise ValueError(
            f"The record at {directory} does not reference the metrics published "
            "beside it, so the two would be readable only by accident."
        )
