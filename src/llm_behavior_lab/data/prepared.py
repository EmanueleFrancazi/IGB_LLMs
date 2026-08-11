"""Prepared dataset directories and their manifests.

A prepared dataset is a directory holding exactly two files::

    <name>/text.txt        the corpus, ready to tokenize
    <name>/manifest.json   what it is and how it was produced

The manifest is written last and only ever appears inside a directory that is
moved into place atomically, so its presence is the completeness signal: a
directory without a manifest is an interrupted preparation, never a usable
dataset.

The manifest also records the dataset identity that produced the text. When a
configuration later disagrees with it -- a different split, revision, or size
limit -- the mismatch is reported instead of silently reusing the old corpus.

Paths are deliberately absent from the manifest so a prepared directory can be
moved or copied between machines.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_behavior_lab.data.config import DatasetConfig
from llm_behavior_lab.data.errors import DatasetIntegrityError

TEXT_FILENAME = "text.txt"
MANIFEST_FILENAME = "manifest.json"
PREPARED_DIR_NAME = "prepared"
PARTIAL_SUFFIX = ".partial"

MANIFEST_VERSION = 1

#: Fields that must agree between a manifest and the configuration using it.
IDENTITY_FIELDS = (
    "source",
    "repo_id",
    "subset",
    "split",
    "revision",
    "text_field",
    "max_examples",
    "max_characters",
    "document_separator",
    "streaming",
)


def prepared_directory(root: str | Path, name: str) -> Path:
    """Return the prepared-dataset directory for ``name`` beneath ``root``."""

    return Path(root) / PREPARED_DIR_NAME / name


def dataset_identity(dataset: DatasetConfig) -> dict[str, Any]:
    """Extract the identity fields recorded in and compared against manifests."""

    return {field: getattr(dataset, field) for field in IDENTITY_FIELDS}


def text_digest(text: str) -> str:
    """Return the SHA-256 digest of the corpus text."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_manifest(directory: str | Path) -> dict[str, Any] | None:
    """Read a manifest, returning ``None`` when it is absent or unreadable.

    An unreadable manifest is treated exactly like a missing one: the directory
    is an incomplete preparation and will be rebuilt rather than trusted.
    """

    path = Path(directory) / MANIFEST_FILENAME
    if not path.is_file():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return None
    return manifest if isinstance(manifest, dict) else None


def describe_mismatch(manifest: dict[str, Any], dataset: DatasetConfig) -> str | None:
    """Describe the first identity field where a manifest and config disagree.

    Returns:
        A human-readable description, or ``None`` when the manifest matches.
    """

    if manifest.get("name") != dataset.name:
        return f"name: prepared as {manifest.get('name')!r}, requested {dataset.name!r}"
    for field, expected in dataset_identity(dataset).items():
        recorded = manifest.get(field)
        if recorded != expected:
            return f"{field}: prepared with {recorded!r}, requested {expected!r}"
    return None


def load_prepared(
    directory: str | Path,
    dataset: DatasetConfig,
    *,
    verify: bool = False,
) -> tuple[Path, dict[str, Any]] | None:
    """Load a prepared directory when it holds usable data for ``dataset``.

    Args:
        directory: Candidate prepared-dataset directory.
        dataset: Dataset identity the caller wants.
        verify: Re-check the recorded digest against the file on disk.

    Returns:
        ``(text_path, manifest)`` when the directory is complete and matches, or
        ``None`` when it is absent or incomplete and should simply be skipped.

    Raises:
        DatasetIntegrityError: If the directory is complete but was prepared for
            a different dataset identity, or if ``verify`` finds a changed file.
    """

    directory = Path(directory)
    text_path = directory / TEXT_FILENAME
    manifest = read_manifest(directory)
    if manifest is None or not text_path.is_file():
        return None

    mismatch = describe_mismatch(manifest, dataset)
    if mismatch is not None:
        raise DatasetIntegrityError(
            f"Prepared data in {directory} does not match the requested dataset.\n"
            f"  {mismatch}\n"
            "Re-prepare it with --force-refresh, or point the config at a different "
            "dataset name."
        )

    if verify:
        recorded = manifest.get("sha256")
        actual = text_digest(text_path.read_text(encoding="utf-8"))
        if recorded != actual:
            raise DatasetIntegrityError(
                f"Prepared data in {directory} has changed since it was prepared.\n"
                f"  recorded sha256 {recorded}\n"
                f"  actual   sha256 {actual}\n"
                "Re-prepare it with --force-refresh."
            )

    return text_path, manifest


def build_manifest(
    dataset: DatasetConfig,
    text: str,
    *,
    num_documents: int | None = None,
    prepared_by: str | None = None,
) -> dict[str, Any]:
    """Build the manifest recorded alongside a prepared corpus."""

    manifest: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "name": dataset.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "num_characters": len(text),
        "num_documents": num_documents,
        "sha256": text_digest(text),
    }
    manifest.update(dataset_identity(dataset))
    if prepared_by:
        manifest["prepared_by"] = prepared_by
    return manifest


def write_prepared(
    directory: str | Path,
    text: str,
    dataset: DatasetConfig,
    *,
    num_documents: int | None = None,
    prepared_by: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write a prepared dataset directory atomically.

    The corpus and its manifest are written into a sibling staging directory
    which is then moved into place, so an interrupted call can never leave a
    directory that looks complete.

    Args:
        directory: Destination prepared-dataset directory.
        text: Corpus contents.
        dataset: Dataset identity recorded in the manifest.
        num_documents: Optional number of source documents used.
        prepared_by: Optional command that produced the data.
        overwrite: Replace an existing prepared directory.

    Returns:
        The manifest that was written.

    Raises:
        FileExistsError: If the destination exists and ``overwrite`` is false.
    """

    directory = Path(directory)
    if directory.exists() and not overwrite:
        raise FileExistsError(f"Prepared dataset already exists: {directory}")

    directory.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        dataset, text, num_documents=num_documents, prepared_by=prepared_by
    )
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{directory.name}.", suffix=PARTIAL_SUFFIX, dir=directory.parent
        )
    )
    try:
        (staging / TEXT_FILENAME).write_text(text, encoding="utf-8")
        (staging / MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if directory.exists():
            shutil.rmtree(directory)
        os.replace(staging, directory)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def clean_partial_directories(parent: str | Path) -> list[Path]:
    """Remove staging directories left behind by an interrupted preparation.

    Returns:
        The directories that were removed.
    """

    parent = Path(parent)
    if not parent.is_dir():
        return []
    removed: list[Path] = []
    for candidate in parent.iterdir():
        if candidate.is_dir() and candidate.name.endswith(PARTIAL_SUFFIX):
            shutil.rmtree(candidate, ignore_errors=True)
            removed.append(candidate)
    return removed
