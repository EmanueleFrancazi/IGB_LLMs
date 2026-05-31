"""Dataset-source loading utilities.

The early project phases used a single local text file. This module keeps that
path intact while adding a small config-driven abstraction for future dataset
sources. The downstream pipeline still receives plain text, so tokenization,
splitting, and batching remain unchanged.

External Hugging Face datasets are optional. The loader performs a small
preflight dependency check and wraps common NumPy/SciPy binary-compatibility
errors with an actionable message, because the ``datasets`` package may import
optional scientific-Python packages while preparing dataset builders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Mapping

from llm_behavior_lab.data.text_dataset import load_text_file


@dataclass(frozen=True)
class LoadedTextDataset:
    """Text loaded from a configured dataset source.

    Attributes:
        source_type: Dataset source kind, such as ``"local_text"`` or
            ``"huggingface"``.
        source_name: Human-readable path or dataset identifier.
        text: Concatenated text consumed by the tokenizer.
        num_examples: Number of documents/examples used when known.
        metadata: Additional source-specific information for diagnostics.
    """

    source_type: str
    source_name: str
    text: str
    num_examples: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class DatasetSourceError(RuntimeError):
    """Raised when a configured dataset source cannot be loaded cleanly."""


def resolve_path(path_value: str | Path, *, repo_root: str | Path | None = None) -> Path:
    """Resolve a possibly relative path.

    Args:
        path_value: Path from config.
        repo_root: Optional repository root used for relative paths.

    Returns:
        Absolute or current-working-directory-relative ``Path``.
    """

    path = Path(path_value)
    if path.is_absolute() or repo_root is None:
        return path
    return Path(repo_root) / path


def load_text_dataset_from_config(
    config: Mapping[str, Any],
    *,
    repo_root: str | Path | None = None,
) -> LoadedTextDataset:
    """Load text according to a data config dictionary.

    Expected local config:

    ```yaml
    dataset:
      source_type: local_text
      path: data/raw/tiny_corpus.txt
    ```

    Expected Hugging Face config:

    ```yaml
    dataset:
      source_type: huggingface
      name: Salesforce/wikitext
      subset: wikitext-2-raw-v1
      split: train[:1000]
      text_field: text
      max_examples: 1000
    ```
    """

    if "dataset" not in config:
        raise KeyError("Data config must contain a top-level 'dataset' section.")

    dataset_config = config["dataset"]
    if not isinstance(dataset_config, Mapping):
        raise TypeError("Config field 'dataset' must be a dictionary.")

    source_type = str(dataset_config.get("source_type", "local_text")).strip().lower()
    if source_type == "local_text":
        return load_local_text_from_config(dataset_config, repo_root=repo_root)
    if source_type == "huggingface":
        return load_huggingface_text_from_config(dataset_config)

    raise ValueError(
        f"Unsupported dataset source_type={source_type!r}. "
        "Supported values are 'local_text' and 'huggingface'."
    )


def load_local_text_from_config(
    dataset_config: Mapping[str, Any],
    *,
    repo_root: str | Path | None = None,
) -> LoadedTextDataset:
    """Load a local UTF-8 text dataset from config."""

    if "path" not in dataset_config:
        raise KeyError("Local text datasets require dataset.path.")

    path = resolve_path(dataset_config["path"], repo_root=repo_root)
    text = load_text_file(path)
    return LoadedTextDataset(
        source_type="local_text",
        source_name=str(path),
        text=text,
        num_examples=1,
        metadata={"path": str(path)},
    )


def _parse_version_prefix(version: str) -> tuple[int, int, int]:
    """Parse the numeric prefix of a package version without extra dependencies."""

    parts: list[int] = []
    for raw_part in version.replace("-", ".").split("."):
        digits = "".join(character for character in raw_part if character.isdigit())
        if digits == "":
            break
        parts.append(int(digits))
        if len(parts) == 3:
            break
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


def _installed_version(package_name: str) -> str | None:
    """Return an installed package version without importing the package."""

    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def _dependency_fix_message() -> str:
    """Return the recommended command for external dataset dependencies."""

    return (
        "Install or refresh the optional dataset dependencies with:\n"
        "  python3 -m pip install --upgrade --force-reinstall -e '.[dev,hf]'\n"
        "or, minimally:\n"
        "  python3 -m pip install --upgrade 'numpy<2' 'scipy>=1.11.4' 'datasets>=2.19'"
    )


def _check_huggingface_dependency_environment() -> None:
    """Detect common NumPy/SciPy incompatibilities before loading datasets.

    The Hugging Face ``datasets`` library may import SciPy helper modules while
    constructing or streaming dataset builders. If an old SciPy build compiled
    against NumPy 1.x is present beside NumPy 2.x, imports can fail with errors
    such as ``AttributeError: _ARRAY_API not found`` or
    ``ImportError: numpy.core.multiarray failed to import``. This check catches
    the most common problematic combination early and turns it into an
    actionable project-level error.
    """

    numpy_version = _installed_version("numpy")
    scipy_version = _installed_version("scipy")
    if numpy_version is None or scipy_version is None:
        return

    numpy_major, _, _ = _parse_version_prefix(numpy_version)
    scipy_major, scipy_minor, _ = _parse_version_prefix(scipy_version)
    scipy_is_old_for_numpy_2 = numpy_major >= 2 and (scipy_major, scipy_minor) < (1, 11)

    if scipy_is_old_for_numpy_2:
        raise DatasetSourceError(
            "Hugging Face dataset loading is likely to fail because this environment "
            f"has NumPy {numpy_version} together with SciPy {scipy_version}. "
            "Older SciPy wheels/extensions are not compatible with NumPy 2.x and can "
            "raise '_ARRAY_API not found' or 'numpy.core.multiarray failed to import'.\n\n"
            f"{_dependency_fix_message()}"
        )


def _import_load_dataset():
    """Import Hugging Face datasets lazily.

    The core project should remain usable without the optional dependency. Users
    only need ``datasets`` when they request a Hugging Face source.
    """

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exact message tested indirectly
        raise ImportError(
            "Hugging Face dataset loading requires the optional dependency. "
            "Install it with: python3 -m pip install -e '.[hf]'"
        ) from exc
    return load_dataset


def _is_numpy_scipy_binary_error(exc: BaseException) -> bool:
    """Return true for common NumPy/SciPy extension incompatibility failures."""

    message = " ".join(
        str(item) for item in (exc, getattr(exc, "__cause__", ""), getattr(exc, "__context__", ""))
    )
    patterns = [
        "_ARRAY_API not found",
        "numpy.core.multiarray failed to import",
        "compiled using NumPy 1.x cannot be run in NumPy 2",
    ]
    return any(pattern in message for pattern in patterns)


def load_huggingface_text_from_config(dataset_config: Mapping[str, Any]) -> LoadedTextDataset:
    """Load and concatenate text from a Hugging Face dataset.

    The loader intentionally supports limiting by examples and/or characters so
    early experiments do not accidentally process huge corpora.
    """

    dataset_name = str(dataset_config.get("name", "")).strip()
    if not dataset_name:
        raise KeyError("Hugging Face datasets require dataset.name.")

    subset = dataset_config.get("subset")
    split = str(dataset_config.get("split", "train"))
    text_field = str(dataset_config.get("text_field", "text"))
    streaming = bool(dataset_config.get("streaming", False))
    max_examples = dataset_config.get("max_examples", 1000)
    max_characters = dataset_config.get("max_characters")
    document_separator = str(dataset_config.get("document_separator", "\n\n"))

    if max_examples is not None:
        max_examples = int(max_examples)
        if max_examples <= 0:
            raise ValueError("dataset.max_examples must be positive when provided.")
    if max_characters is not None:
        max_characters = int(max_characters)
        if max_characters <= 0:
            raise ValueError("dataset.max_characters must be positive when provided.")

    _check_huggingface_dependency_environment()
    load_dataset = _import_load_dataset()
    load_kwargs: dict[str, Any] = {"split": split, "streaming": streaming}
    try:
        if subset is not None:
            dataset = load_dataset(dataset_name, str(subset), **load_kwargs)
        else:
            dataset = load_dataset(dataset_name, **load_kwargs)
    except Exception as exc:  # pragma: no cover - environment-specific path
        if _is_numpy_scipy_binary_error(exc):
            raise DatasetSourceError(
                "Hugging Face dataset loading failed because NumPy/SciPy binary "
                "dependencies in the current environment are incompatible. This is an "
                "environment issue, not a WikiText-2 data issue.\n\n"
                f"{_dependency_fix_message()}"
            ) from exc
        raise DatasetSourceError(
            f"Failed to load Hugging Face dataset {dataset_name!r} "
            f"subset={subset!r} split={split!r}. Original error: {exc}"
        ) from exc

    text, num_examples = concatenate_text_examples(
        dataset,
        text_field=text_field,
        max_examples=max_examples,
        max_characters=max_characters,
        document_separator=document_separator,
    )

    subset_name = None if subset is None else str(subset)
    return LoadedTextDataset(
        source_type="huggingface",
        source_name=dataset_name,
        text=text,
        num_examples=num_examples,
        metadata={
            "name": dataset_name,
            "subset": subset_name,
            "split": split,
            "text_field": text_field,
            "streaming": streaming,
            "max_examples": max_examples,
            "max_characters": max_characters,
        },
    )


def concatenate_text_examples(
    examples: Iterable[Mapping[str, Any]],
    *,
    text_field: str,
    max_examples: int | None = 1000,
    max_characters: int | None = None,
    document_separator: str = "\n\n",
) -> tuple[str, int]:
    """Concatenate text examples with explicit limits.

    Args:
        examples: Iterable of dataset rows.
        text_field: Field containing text in each row.
        max_examples: Optional maximum number of rows to inspect.
        max_characters: Optional maximum number of output characters.
        document_separator: String placed between non-empty documents.

    Returns:
        Pair ``(text, num_examples_used)``.
    """

    if max_examples is not None and max_examples <= 0:
        raise ValueError("max_examples must be positive when provided.")
    if max_characters is not None and max_characters <= 0:
        raise ValueError("max_characters must be positive when provided.")

    chunks: list[str] = []
    characters = 0
    used_examples = 0

    for row_index, row in enumerate(examples):
        if max_examples is not None and row_index >= max_examples:
            break
        if text_field not in row:
            raise KeyError(f"Dataset row does not contain text_field={text_field!r}.")

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
        used_examples += 1

        if max_characters is not None and characters >= max_characters:
            break

    joined = document_separator.join(chunks)
    if not joined:
        raise ValueError("Dataset loading produced empty text. Check split, field, and limits.")
    return joined, used_examples
