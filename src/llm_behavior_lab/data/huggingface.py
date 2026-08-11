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
import sys
from contextlib import contextmanager
from importlib import util
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from llm_behavior_lab.data.config import DatasetConfig
from llm_behavior_lab.data.errors import DatasetAcquisitionError
from llm_behavior_lab.data.prepared import write_prepared

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


@contextmanager
def _offline_mode():
    """Enable Hugging Face offline mode for the duration of a call.

    ``HF_HUB_OFFLINE`` is the documented mechanism, and it is set here so that
    any subprocess inherits it. It is not sufficient on its own: both
    ``huggingface_hub`` and ``datasets`` read it into a module constant at
    import time, and the loader is imported before this runs, so setting only
    the variable would leave offline mode disabled. The corresponding constants
    are therefore set too, and everything is restored afterwards.

    This makes the offline guarantee explicit rather than assumed. It still
    relies on the libraries honouring their own offline constant; there is no
    independent network barrier here.
    """

    import huggingface_hub.constants as hub_constants

    previous_env = os.environ.get("HF_HUB_OFFLINE")
    previous_hub = hub_constants.HF_HUB_OFFLINE
    os.environ["HF_HUB_OFFLINE"] = "1"
    hub_constants.HF_HUB_OFFLINE = True

    datasets_config = sys.modules.get("datasets.config")
    previous_datasets = getattr(datasets_config, "HF_HUB_OFFLINE", None)
    if datasets_config is not None:
        datasets_config.HF_HUB_OFFLINE = True

    try:
        yield
    finally:
        if previous_env is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = previous_env
        hub_constants.HF_HUB_OFFLINE = previous_hub
        if datasets_config is not None and previous_datasets is not None:
            datasets_config.HF_HUB_OFFLINE = previous_datasets


def describe_resolved(rows: Any) -> dict[str, Any]:
    """Summarize what the loader actually returned, for the manifest.

    The requested identity says what was asked for; this says what came back.
    Everything is read from the object already in hand -- no extra network call
    is made, so nothing here works differently offline. Fields the loader does
    not expose are recorded as ``None`` rather than omitted, so a reader can
    tell "unavailable" from "not recorded".

    The upstream commit SHA is deliberately not fetched: obtaining it needs a
    Hub lookup, which would add a network operation to a path that may be
    running offline. Pin ``dataset.revision`` when strict upstream
    reproducibility matters.
    """

    info = getattr(rows, "info", None)
    return {
        "fingerprint": getattr(rows, "_fingerprint", None),
        "version": str(getattr(info, "version", None)) if info is not None else None,
        "config_name": getattr(rows, "config_name", None),
        "split": str(getattr(rows, "split", None)) if getattr(rows, "split", None) else None,
        "download_size": getattr(info, "download_size", None) if info is not None else None,
        "dataset_size": getattr(info, "dataset_size", None) if info is not None else None,
    }


def _load_rows(
    dataset: DatasetConfig,
    cache_dir: Path,
    *,
    allow_network: bool,
    force_redownload: bool = False,
) -> Any:
    """Call ``load_dataset`` with caching and offline behavior made explicit."""

    load_dataset = _import_load_dataset()

    kwargs: dict[str, Any] = {
        "split": dataset.split,
        "cache_dir": str(cache_dir),
    }
    if dataset.revision:
        kwargs["revision"] = dataset.revision
    if dataset.streaming:
        kwargs["streaming"] = True
    if force_redownload:
        kwargs["download_mode"] = "force_redownload"

    if allow_network:
        if dataset.subset:
            return load_dataset(dataset.repo_id, dataset.subset, **kwargs)
        return load_dataset(dataset.repo_id, **kwargs)

    with _offline_mode():
        if dataset.subset:
            return load_dataset(dataset.repo_id, dataset.subset, **kwargs)
        return load_dataset(dataset.repo_id, **kwargs)


def materialize(
    dataset: DatasetConfig,
    destination: str | Path,
    *,
    cache_dir: str | Path,
    allow_network: bool,
    reporter: Callable[[str], None] | None = None,
    overwrite: bool = False,
    prepared_by: str | None = None,
    force_redownload: bool = False,
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
        force_redownload: Ask the loader to re-fetch rather than reuse its own
            cache. Only meaningful together with ``allow_network``.

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
            f"  streaming  : {dataset.streaming}\n"
            f"  cache      : {cache_dir}\n"
            f"  destination: {destination}\n"
            "  This will download data. Use --offline or --no-download to prevent it."
        )
        cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        rows = _load_rows(
            dataset,
            cache_dir,
            allow_network=allow_network,
            force_redownload=force_redownload,
        )
    except DatasetAcquisitionError:
        raise
    except Exception as exc:
        raise DatasetAcquisitionError(
            f"Failed to load Hugging Face dataset {describe_source(dataset)}.\n"
            f"  {type(exc).__name__}: {exc}"
        ) from exc

    resolved = describe_resolved(rows)
    text, num_documents = concatenate_text_examples(
        rows,
        text_field=dataset.text_field,
        max_examples=dataset.max_examples,
        max_characters=dataset.max_characters,
        document_separator=dataset.document_separator,
    )

    manifest = write_prepared(
        destination,
        text,
        dataset,
        num_documents=num_documents,
        prepared_by=prepared_by,
        overwrite=overwrite,
        resolved=resolved,
    )
    announce(
        f"Prepared dataset {dataset.name!r}: {manifest['num_characters']} characters "
        f"from {num_documents} documents at {destination}"
    )
    return manifest
