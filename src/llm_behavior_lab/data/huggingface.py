"""Hugging Face dataset acquisition.

This is the only module in the project that can reach the network, and it does
so only when a caller explicitly permits it. ``datasets`` is imported lazily, so
the package remains optional: installing it is required to use an external
dataset and irrelevant to everything else.

The same function serves two situations, distinguished by one flag:

* ``allow_network=False`` reuses a copy already in the local Hugging Face cache
  and cannot fetch anything, which is what makes no-download and offline modes
  able to reuse data that is already present.
* ``allow_network=True`` may download, and announces what it is about to do
  before any network activity happens.

Either way the result is written as a prepared dataset directory, so subsequent
runs resolve from disk without consulting Hugging Face at all.
"""

from __future__ import annotations

import os
from importlib import metadata, util
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from llm_behavior_lab.data.config import DatasetConfig
from llm_behavior_lab.data.errors import DatasetAcquisitionError
from llm_behavior_lab.data.prepared import clean_partial_directories, write_prepared

HF_CACHE_DIR_NAME = "hf"
INSTALL_HINT = 'python3 -m pip install -e ".[hf]"'


def hf_cache_directory(data_root: str | Path) -> Path:
    """Return the Hugging Face cache location beneath the project data root."""

    return Path(data_root) / HF_CACHE_DIR_NAME


def datasets_available() -> bool:
    """Report whether the optional ``datasets`` package can be imported.

    Uses the import system's metadata rather than importing the package, so a
    negative answer costs nothing.
    """

    return util.find_spec("datasets") is not None


def _import_load_dataset() -> Callable[..., Any]:
    """Import ``datasets.load_dataset`` lazily with an actionable failure."""

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise DatasetAcquisitionError(
            "Hugging Face datasets require the optional dependency.\n"
            f"Install it with: {INSTALL_HINT}"
        ) from exc
    return load_dataset


def _installed_version(package: str) -> str | None:
    """Return an installed package version without importing the package."""

    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _numeric_prefix(version: str) -> tuple[int, ...]:
    """Parse the leading numeric components of a version string."""

    parts: list[int] = []
    for chunk in version.replace("-", ".").split("."):
        digits = "".join(character for character in chunk if character.isdigit())
        if not digits:
            break
        parts.append(int(digits))
        if len(parts) == 3:
            break
    return tuple(parts)


def check_dependency_environment() -> None:
    """Detect a NumPy/SciPy combination known to break ``datasets`` imports.

    Older SciPy builds compiled against NumPy 1.x fail to import beside NumPy
    2.x with errors that look like dataset problems but are environment
    problems. Catching it here keeps the message honest.
    """

    numpy_version = _installed_version("numpy")
    scipy_version = _installed_version("scipy")
    if not numpy_version or not scipy_version:
        return

    numpy_parts = _numeric_prefix(numpy_version)
    scipy_parts = _numeric_prefix(scipy_version)
    if not numpy_parts or len(scipy_parts) < 2:
        return

    if numpy_parts[0] >= 2 and scipy_parts[:2] < (1, 11):
        raise DatasetAcquisitionError(
            f"This environment has NumPy {numpy_version} with SciPy {scipy_version}. "
            "SciPy builds older than 1.11 are incompatible with NumPy 2.x and can "
            "fail while importing Hugging Face datasets.\n"
            f"Refresh the optional dependencies with: {INSTALL_HINT}"
        )


def concatenate_text_examples(
    rows: Iterable[Mapping[str, Any]],
    *,
    text_field: str,
    max_examples: int | None = None,
    max_characters: int | None = None,
    document_separator: str = "\n\n",
) -> tuple[str, int]:
    """Join dataset rows into one corpus, honouring the configured limits.

    Empty rows are skipped. Both limits are applied while iterating so a bounded
    corpus never requires materializing an unbounded one.

    Args:
        rows: Iterable of dataset rows.
        text_field: Field holding text in each row.
        max_examples: Optional maximum number of rows to inspect.
        max_characters: Optional maximum number of characters to keep.
        document_separator: String placed between kept documents.

    Returns:
        The corpus text and the number of documents it came from.

    Raises:
        DatasetAcquisitionError: If a row lacks ``text_field`` or the limits
            select nothing.
    """

    chunks: list[str] = []
    characters = 0
    used = 0

    for index, row in enumerate(rows):
        if max_examples is not None and index >= max_examples:
            break
        if text_field not in row:
            raise DatasetAcquisitionError(
                f"Dataset rows do not contain the field {text_field!r}. "
                "Set dataset.text_field to the field holding the text."
            )
        value = row[text_field]
        if value is None:
            continue
        text = str(value)
        if not text.strip():
            continue

        if max_characters is not None:
            remaining = max_characters - characters
            if remaining <= 0:
                break
            text = text[:remaining]

        chunks.append(text)
        characters += len(text)
        used += 1

        if max_characters is not None and characters >= max_characters:
            break

    corpus = document_separator.join(chunks)
    if not corpus:
        raise DatasetAcquisitionError(
            "Loading produced an empty corpus. Check dataset.split, "
            "dataset.text_field, and the configured limits."
        )
    return corpus, used


def describe_source(dataset: DatasetConfig) -> str:
    """Render the external dataset identity for user-facing messages."""

    parts = [dataset.repo_id or "<unset>"]
    if dataset.subset:
        parts.append(f"subset={dataset.subset}")
    parts.append(f"split={dataset.split}")
    parts.append(f"revision={dataset.revision or 'latest'}")
    return ", ".join(parts)


def _load_rows(dataset: DatasetConfig, cache_dir: Path, *, allow_network: bool) -> Any:
    """Call ``load_dataset`` with caching and offline behavior made explicit."""

    check_dependency_environment()
    load_dataset = _import_load_dataset()

    kwargs: dict[str, Any] = {
        "split": dataset.split,
        "cache_dir": str(cache_dir),
    }
    if dataset.revision:
        kwargs["revision"] = dataset.revision

    previous = os.environ.get("HF_HUB_OFFLINE")
    if not allow_network:
        os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        if dataset.subset:
            return load_dataset(dataset.repo_id, dataset.subset, **kwargs)
        return load_dataset(dataset.repo_id, **kwargs)
    finally:
        if not allow_network:
            if previous is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = previous


def materialize(
    dataset: DatasetConfig,
    destination: str | Path,
    *,
    cache_dir: str | Path,
    allow_network: bool,
    reporter: Callable[[str], None] | None = None,
    overwrite: bool = False,
    prepared_by: str | None = None,
) -> dict[str, Any]:
    """Produce a prepared dataset directory from a Hugging Face dataset.

    Args:
        dataset: Dataset identity to obtain.
        destination: Prepared-dataset directory to create.
        cache_dir: Hugging Face cache location.
        allow_network: Permit fetching. When false, only a cached copy is used
            and no network access can occur.
        reporter: Receives user-facing progress messages.
        overwrite: Replace an existing prepared directory.
        prepared_by: Command recorded in the manifest.

    Returns:
        The manifest of the prepared dataset.

    Raises:
        DatasetAcquisitionError: If loading or preparation fails.
    """

    destination = Path(destination)
    cache_dir = Path(cache_dir)
    announce = reporter or (lambda _message: None)

    if allow_network:
        announce(
            f"Acquiring dataset {dataset.name!r}\n"
            f"  source     : huggingface {describe_source(dataset)}\n"
            f"  limits     : max_examples={dataset.max_examples}, "
            f"max_characters={dataset.max_characters}\n"
            f"  cache      : {cache_dir}\n"
            f"  destination: {destination}\n"
            "  This will download data. Use --offline or --no-download to prevent it."
        )
        cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        rows = _load_rows(dataset, cache_dir, allow_network=allow_network)
    except DatasetAcquisitionError:
        raise
    except Exception as exc:
        raise DatasetAcquisitionError(
            f"Failed to load Hugging Face dataset {describe_source(dataset)}.\n"
            f"  {type(exc).__name__}: {exc}"
        ) from exc

    text, num_documents = concatenate_text_examples(
        rows,
        text_field=dataset.text_field,
        max_examples=dataset.max_examples,
        max_characters=dataset.max_characters,
        document_separator=dataset.document_separator,
    )

    clean_partial_directories(destination.parent)
    manifest = write_prepared(
        destination,
        text,
        dataset,
        num_documents=num_documents,
        prepared_by=prepared_by,
        overwrite=overwrite,
    )
    announce(
        f"Prepared dataset {dataset.name!r}: {manifest['num_characters']} characters "
        f"from {num_documents} documents at {destination}"
    )
    return manifest
