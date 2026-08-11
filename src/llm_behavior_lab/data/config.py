"""Dataset configuration and runtime resolution policy.

The dataset layer keeps two kinds of information apart:

* :class:`DatasetConfig` describes *which* dataset an experiment uses. It comes
  from tracked YAML, is snapshotted alongside a run, and is part of
  reproducibility.
* :class:`ResolutionPolicy` describes *how* one invocation may obtain that
  dataset. It comes from command-line flags and the environment, is
  machine-specific, and is deliberately absent from tracked configuration.

Keeping them separate means a configuration file never carries a machine path
and never decides whether a process may reach the network.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from llm_behavior_lab.data.errors import DatasetConfigError, UnsupportedDatasetSourceError

LOCAL_TEXT_SOURCE = "local_text"
HUGGINGFACE_SOURCE = "huggingface"
SUPPORTED_SOURCES = (LOCAL_TEXT_SOURCE, HUGGINGFACE_SOURCE)

DATA_ROOT_ENV_VAR = "LLM_BEHAVIOR_LAB_DATA_ROOT"
DATA_ROOT_DIR_NAME = "llm-behavior-lab"

# Dataset names become directory components for prepared copies, so they are
# restricted to characters that are safe on every supported platform.
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

#: Keys inside ``dataset:`` that this layer owns and parses.
DATASET_KEYS = (
    "name",
    "source",
    "path",
    "repo_id",
    "subset",
    "split",
    "revision",
    "text_field",
    "max_examples",
    "max_characters",
    "document_separator",
    "streaming",
    "verify",
)

#: Keys inside ``dataset:`` owned by other pipeline stages. They are read
#: elsewhere -- ``val_fraction`` by the train/validation split -- so they are
#: legal here even though this layer ignores them. Anything outside both tuples
#: is treated as a mistake rather than silently dropped, because a misspelling
#: such as ``max_character`` would otherwise change what corpus is prepared
#: without any sign that it did.
FOREIGN_DATASET_KEYS = ("val_fraction",)


@dataclass(frozen=True)
class DatasetConfig:
    """Identity of the dataset an experiment uses.

    Attributes:
        name: Stable identifier, also used as a directory component for prepared
            copies.
        source: Dataset source kind, one of :data:`SUPPORTED_SOURCES`.
        path: Local text path. Required for ``local_text`` sources, and usable as
            an explicit override for any source.
        repo_id: External dataset identifier. Required for ``huggingface``.
        subset: Optional dataset subset or configuration name.
        split: Dataset split, or a split slice such as ``"train[:1000]"``.
        revision: Optional pinned revision of an external dataset.
        text_field: Field holding text in external dataset rows.
        max_examples: Optional cap on the number of rows consumed.
        max_characters: Optional cap on the number of characters kept.
        document_separator: String placed between concatenated documents.
        streaming: Iterate the source lazily while preparing, so the whole split
            need not be materialized first. Useful for large upstream splits.
        verify: Whether prepared-data checksums are re-verified on load.
    """

    name: str
    source: str = LOCAL_TEXT_SOURCE
    path: str | None = None
    repo_id: str | None = None
    subset: str | None = None
    split: str = "train"
    revision: str | None = None
    text_field: str = "text"
    max_examples: int | None = None
    max_characters: int | None = None
    document_separator: str = "\n\n"
    streaming: bool = False
    verify: bool = False

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Check that the configured dataset identity is usable.

        Raises:
            UnsupportedDatasetSourceError: If ``source`` is not supported.
            DatasetConfigError: If any other field is missing or invalid.
        """

        if not self.name.strip():
            raise DatasetConfigError("dataset.name must be non-empty.")
        if _SAFE_NAME.fullmatch(self.name) is None:
            raise DatasetConfigError(
                "dataset.name must start with a letter or digit and contain only "
                f"letters, digits, '.', '-', and '_'; got {self.name!r}. The name is "
                "used as a directory component for prepared copies."
            )

        if self.source not in SUPPORTED_SOURCES:
            raise UnsupportedDatasetSourceError(
                f"Unsupported dataset.source={self.source!r}. "
                f"Supported sources are: {', '.join(SUPPORTED_SOURCES)}."
            )

        if self.source == LOCAL_TEXT_SOURCE and not (self.path or "").strip():
            raise DatasetConfigError(
                f"dataset.path is required for source={LOCAL_TEXT_SOURCE!r}."
            )
        if self.source == HUGGINGFACE_SOURCE and not (self.repo_id or "").strip():
            raise DatasetConfigError(
                f"dataset.repo_id is required for source={HUGGINGFACE_SOURCE!r}."
            )

        if not self.split.strip():
            raise DatasetConfigError("dataset.split must be non-empty.")
        if not self.text_field.strip():
            raise DatasetConfigError("dataset.text_field must be non-empty.")

        for field_name, value in (
            ("max_examples", self.max_examples),
            ("max_characters", self.max_characters),
        ):
            if value is not None and value <= 0:
                raise DatasetConfigError(
                    f"dataset.{field_name} must be positive when provided."
                )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "DatasetConfig":
        """Build and validate a ``DatasetConfig`` from a data configuration.

        Expected structure::

            dataset:
              name: tiny_local_text
              source: local_text          # optional; defaults to local_text
              path: data/raw/tiny_corpus.txt

        Keys owned by other pipeline stages, listed in
        :data:`FOREIGN_DATASET_KEYS`, are accepted and ignored so existing
        configuration files keep working unchanged. Any other unrecognised key
        is rejected: silently dropping a misspelling such as ``max_character``
        would change which corpus is prepared with no sign that it had.

        Args:
            config: Full data configuration containing a ``dataset`` section.

        Returns:
            A validated ``DatasetConfig``.

        Raises:
            DatasetConfigError: If the section is missing, is not a dictionary,
                or describes an unusable dataset.
        """

        section = config.get("dataset")
        if not isinstance(section, Mapping):
            raise DatasetConfigError(
                "Data config must contain a dictionary section named 'dataset'."
            )
        if "name" not in section:
            raise DatasetConfigError("dataset.name is required.")

        unknown = sorted(set(section) - set(DATASET_KEYS) - set(FOREIGN_DATASET_KEYS))
        if unknown:
            raise DatasetConfigError(
                "Unknown key(s) in the dataset config: " + ", ".join(unknown) + ".\n"
                "Dataset settings: " + ", ".join(DATASET_KEYS) + ".\n"
                "Keys owned by other stages: " + ", ".join(FOREIGN_DATASET_KEYS) + ".\n"
                "Check for a misspelling; unknown keys are rejected because silently "
                "ignoring one can change which corpus is prepared."
            )

        def optional_str(key: str) -> str | None:
            value = section.get(key)
            return None if value is None else str(value)

        def optional_int(key: str) -> int | None:
            value = section.get(key)
            return None if value is None else int(value)

        return cls(
            name=str(section["name"]),
            source=str(section.get("source", LOCAL_TEXT_SOURCE)),
            path=optional_str("path"),
            repo_id=optional_str("repo_id"),
            subset=optional_str("subset"),
            split=str(section.get("split", "train")),
            revision=optional_str("revision"),
            text_field=str(section.get("text_field", "text")),
            max_examples=optional_int("max_examples"),
            max_characters=optional_int("max_characters"),
            document_separator=str(section.get("document_separator", "\n\n")),
            streaming=bool(section.get("streaming", False)),
            verify=bool(section.get("verify", False)),
        )


@dataclass(frozen=True)
class ResolutionPolicy:
    """How one invocation may obtain a dataset.

    The defaults are deliberately conservative: a policy constructed without
    arguments can never reach the network. User-facing workflows opt in to
    acquisition explicitly.

    Attributes:
        data_root: External root for prepared and cached data. ``None`` selects
            the default described by :func:`resolve_data_root`.
        offline: Forbid all network access. Acquisition is impossible while this
            is set, regardless of ``allow_download``.
        allow_download: Permit obtaining data that is not already available.
            Reusing data that is already present never requires this.
        force_refresh: Re-acquire and replace an existing prepared copy.
    """

    data_root: Path | None = None
    offline: bool = False
    allow_download: bool = False
    force_refresh: bool = False

    def __post_init__(self) -> None:
        self.validate()

    @property
    def may_acquire(self) -> bool:
        """Whether this invocation may obtain data that is not already present."""

        return self.allow_download and not self.offline

    def validate(self) -> None:
        """Check that the requested policy is self-consistent.

        Raises:
            DatasetConfigError: If a refresh is requested while acquisition is
                impossible.
        """

        if self.force_refresh and not self.may_acquire:
            raise DatasetConfigError(
                "force_refresh requires acquisition to be permitted, but "
                f"offline={self.offline} and allow_download={self.allow_download}."
            )


def resolve_data_root(explicit: str | Path | None = None) -> Path:
    """Select the external root for prepared and cached dataset material.

    Precedence:

    1. ``explicit``, typically a command-line argument
    2. the ``LLM_BEHAVIOR_LAB_DATA_ROOT`` environment variable
    3. ``$XDG_CACHE_HOME/llm-behavior-lab``
    4. ``~/.cache/llm-behavior-lab``

    The default is always outside the repository, so large downloads cannot land
    in a tracked path. The directory is not created here; callers create it only
    when they are about to write.

    Args:
        explicit: Optional caller-supplied root. A leading ``~`` is expanded.

    Returns:
        The selected root. Relative values are returned unchanged and are
        therefore interpreted against the current working directory.
    """

    if explicit is not None:
        return Path(explicit).expanduser()

    configured = os.environ.get(DATA_ROOT_ENV_VAR, "").strip()
    if configured:
        return Path(configured).expanduser()

    xdg_cache = os.environ.get("XDG_CACHE_HOME", "").strip()
    if xdg_cache:
        return Path(xdg_cache).expanduser() / DATA_ROOT_DIR_NAME

    return Path.home() / ".cache" / DATA_ROOT_DIR_NAME
