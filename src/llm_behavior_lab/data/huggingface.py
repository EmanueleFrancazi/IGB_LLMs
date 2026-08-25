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

import gc
import os
import sys
import threading
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

    ``max_characters`` bounds the **final corpus**, separators included, so
    ``len(result) <= max_characters`` always holds. A separator is charged only
    when a document actually follows another, and no trailing separator is
    added, so the budget is never spent on text that is not in the result.

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
    separator_length = len(document_separator)

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
            # A separator is only spent when this document follows another, so
            # the running total always equals the length of the joined corpus.
            overhead = separator_length if chunks else 0
            remaining = max_characters - characters - overhead
            if remaining <= 0:
                break
            text = text[:remaining]
            characters += overhead

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


#: Loader calls are serialized because enabling offline mode means mutating
#: process-global state. Without this, a cache-only call could force a
#: concurrent acquisition offline, and overlapping calls could restore the
#: flags in the wrong order and leave the process permanently offline.
_LOADER_LOCK = threading.Lock()

#: Environment variables that select offline mode. Both are set so a
#: subprocess inherits the intent whichever library reads it.
_OFFLINE_ENV_VARS = ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")

#: Module constants that decide offline mode at call time. The libraries read
#: the environment once at import, so setting only the variables is not enough.
#: Releases differ in which of these exist: older ``datasets`` uses
#: ``HF_DATASETS_OFFLINE``, newer releases also carry the Hub-style name. Only
#: attributes the installed version already defines are touched, so nothing
#: meaningless is created.
_OFFLINE_CONSTANTS = (
    ("huggingface_hub.constants", "HF_HUB_OFFLINE"),
    ("datasets.config", "HF_HUB_OFFLINE"),
    ("datasets.config", "HF_DATASETS_OFFLINE"),
)


@contextmanager
def _offline_mode():
    """Enable Hugging Face offline mode for the duration of a call.

    ``HF_HUB_OFFLINE`` is the documented mechanism and is set here so that any
    subprocess inherits it. It is not sufficient on its own: the libraries read
    it into module constants at import time and the loader is imported before
    this runs, so the constants are set too and restored afterwards, on normal
    exit and on exceptions alike.

    Enforcement still belongs to those libraries; there is no independent
    network barrier here. Callers must hold :data:`_LOADER_LOCK`, because the
    state being changed is process-global.

    ``huggingface_hub`` ships with the optional ``hf`` extra, so it may be
    absent. Its constant is only worth setting when it is installed, and the
    loop below already skips a module that is not in :data:`sys.modules`, so
    its absence is a no-op rather than an error. Only that absence is
    tolerated: an import that fails for any other reason -- a broken install,
    or a dependency of the package itself missing -- still propagates, because
    that is a real fault rather than a package the caller chose not to install.
    """

    try:
        import huggingface_hub.constants  # noqa: F401  ensure it is in sys.modules
    except ModuleNotFoundError as exc:
        if exc.name not in ("huggingface_hub", "huggingface_hub.constants"):
            raise

    saved_env = {name: os.environ.get(name) for name in _OFFLINE_ENV_VARS}
    for name in _OFFLINE_ENV_VARS:
        os.environ[name] = "1"

    saved_constants = []
    for module_name, attribute in _OFFLINE_CONSTANTS:
        module = sys.modules.get(module_name)
        if module is None or not hasattr(module, attribute):
            continue
        saved_constants.append((module, attribute, getattr(module, attribute)))
        setattr(module, attribute, True)

    try:
        yield
    finally:
        for module, attribute, previous in saved_constants:
            setattr(module, attribute, previous)
        for name, previous in saved_env.items():
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous


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

    def call() -> Any:
        if dataset.subset:
            return load_dataset(dataset.repo_id, dataset.subset, **kwargs)
        return load_dataset(dataset.repo_id, **kwargs)

    # Held for acquisition too: otherwise a concurrent cache-only call could
    # switch the global flags underneath a fetch that is meant to use the network.
    with _LOADER_LOCK:
        if allow_network:
            return call()
        with _offline_mode():
            return call()


def _release_stream() -> None:
    """Release what a partially consumed streamed dataset still holds.

    A streaming dataset that is abandoned part-way -- which is exactly what the
    configured limits make this module do -- can keep native resources alive
    until interpreter shutdown. In some dependency combinations releasing them
    that late aborts the process with ``SIGABRT`` *after* preparation has
    already succeeded, turning a correct run into a failed-looking one.

    Dropping the ordinary Python references is measurably not enough; collecting
    while the interpreter is still healthy is. The mechanism lives in the
    dependency stack rather than here, and which component owns it is not
    established, so this is deliberately scoped to streaming loads and does
    nothing else. ``data/README.md`` records what was observed.
    """

    gc.collect()


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
            cache. Only meaningful together with ``allow_network``, and only for
            non-streaming loads: streaming bypasses the download-and-prepare
            step this option governs, so a streamed refresh simply reopens the
            source rather than invalidating a download cache.

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

    try:
        # Both of these need the dataset object: provenance reads its
        # attributes, concatenation iterates it. Nothing afterwards does.
        resolved = describe_resolved(rows)
        text, num_documents = concatenate_text_examples(
            rows,
            text_field=dataset.text_field,
            max_examples=dataset.max_examples,
            max_characters=dataset.max_characters,
            document_separator=dataset.document_separator,
        )
    finally:
        # ``rows`` is the only reference this module keeps: the loop's iterator
        # lives inside concatenate_text_examples and is released when it
        # returns. Dropping it here and collecting frees a partially consumed
        # stream now rather than at interpreter shutdown -- see
        # :func:`_release_stream` for why that matters.
        del rows
        if dataset.streaming:
            _release_stream()

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
