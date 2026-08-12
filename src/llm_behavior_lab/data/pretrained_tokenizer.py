"""Pretrained subword tokenizer support.

This module loads **tokenizer artifacts only**. It calls
``AutoTokenizer.from_pretrained`` and nothing else: no ``AutoModel``, no
``AutoModelForCausalLM``, no pipeline, no pretrained model constructor. The
language model in this project stays the repository's own randomly initialized
tiny LLaMA, and the tokenizer is the only pretrained component anywhere in the
experiment.

That boundary matters for interpretation as much as for download size. A
pretrained tokenizer already encodes linguistic structure learned from its own
training corpus, so an experiment using one measures a randomly initialized
model *over a realistic vocabulary* -- not a completely unlearned text-processing
system. See ``src/README.md`` for how that shapes what the results may claim.

``transformers`` is an optional dependency, imported lazily, so the character
workflows never require it.

Two policies are made explicit rather than inherited from library defaults:

* **Acquisition is off unless asked for.** ``allow_download`` must be set, and
  the fetch is announced before it happens.
* **Ordinary corpus text is encoded without structural tokens.** No BOS or EOS
  is inserted, because a corpus is being measured, not a prompt assembled.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from llm_behavior_lab.data.errors import DatasetAcquisitionError, DatasetNotAvailableError

__all__ = [
    "INSTALL_HINT",
    "PretrainedTokenizer",
    "TOKENIZER_CACHE_DIR_NAME",
    "load_pretrained_tokenizer",
    "tokenizer_cache_directory",
    "transformers_available",
]

#: Tokenizer artifacts live beside the dataset caches, under the same external
#: data root, so nothing downloadable ever lands inside the repository.
TOKENIZER_CACHE_DIR_NAME = "tokenizers"

INSTALL_HINT = 'python3 -m pip install -e ".[tokenizers]"'


def tokenizer_cache_directory(data_root: str | Path) -> Path:
    """Return the tokenizer cache location beneath the project data root."""

    return Path(data_root) / TOKENIZER_CACHE_DIR_NAME


def transformers_available() -> bool:
    """Report whether the optional ``transformers`` package can be imported."""

    from importlib import util

    return util.find_spec("transformers") is not None


def _import_auto_tokenizer() -> Any:
    """Import ``AutoTokenizer`` lazily with an actionable failure.

    This is the single seam through which the optional dependency is reached,
    which also makes it the single place tests patch to stay offline.
    """

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise DatasetAcquisitionError(
            "Pretrained tokenizers require the optional dependency.\n"
            f"Install it with: {INSTALL_HINT}"
        ) from exc
    return AutoTokenizer


@dataclass(frozen=True)
class PretrainedTokenizer:
    """Adapter presenting a Hugging Face tokenizer as a project tokenizer.

    Canonical token IDs are preserved exactly as the pretrained tokenizer
    defines them. Structural IDs are *excluded* from the predictive support
    rather than removed or renumbered, so an ID means the same thing here as it
    does upstream and a record can always be compared with the tokenizer that
    produced it.

    Attributes:
        tokenizer: The wrapped Hugging Face tokenizer.
        identifier: Hub identifier it was loaded from.
        requested_revision: Revision asked for, or ``None`` for the default.
        add_special_tokens: Whether ``encode`` inserts structural tokens. False
            for corpus measurement.
        source: Where the files came from, for provenance.
    """

    tokenizer: Any
    identifier: str
    requested_revision: str | None = None
    add_special_tokens: bool = False
    source: str = "unknown"
    _eligible: tuple[int, ...] = field(default=(), repr=False)

    @property
    def vocab_size(self) -> int:
        """Full canonical vocabulary size, including any added tokens.

        ``len(tokenizer)`` is used rather than ``tokenizer.vocab_size`` because
        the latter excludes tokens added after training, which would leave the
        model's output layer too narrow to cover every reachable ID.
        """

        return len(self.tokenizer)

    @property
    def special_token_ids(self) -> tuple[int, ...]:
        """Structural IDs, sorted and de-duplicated.

        Collected from ``all_special_ids`` plus any explicitly registered
        additional special tokens. IDs at or beyond the vocabulary size are
        dropped: a malformed entry must not silently shrink the support.
        """

        collected: set[int] = set()
        for value in getattr(self.tokenizer, "all_special_ids", ()) or ():
            if value is not None:
                collected.add(int(value))
        added = getattr(self.tokenizer, "added_tokens_decoder", None) or {}
        for token_id, token in added.items():
            if getattr(token, "special", False):
                collected.add(int(token_id))
        size = self.vocab_size
        return tuple(sorted(value for value in collected if 0 <= value < size))

    @property
    def eligible_token_ids(self) -> tuple[int, ...]:
        """Canonical IDs the model may be scored on, in increasing order."""

        if self._eligible:
            return self._eligible
        excluded = set(self.special_token_ids)
        return tuple(index for index in range(self.vocab_size) if index not in excluded)

    @property
    def special_tokens(self) -> dict[str, Any]:
        """Readable name-to-ID map of the structural tokens."""

        mapping = getattr(self.tokenizer, "special_tokens_map", None) or {}
        described: dict[str, Any] = {}
        for name, token in mapping.items():
            if isinstance(token, (list, tuple)):
                described[name] = {
                    str(item): self._token_to_id(str(item)) for item in token
                }
            else:
                described[name] = {str(token): self._token_to_id(str(token))}
        return described

    def _token_to_id(self, token: str) -> int | None:
        convert = getattr(self.tokenizer, "convert_tokens_to_ids", None)
        if convert is None:
            return None
        try:
            value = convert(token)
        except Exception:  # pragma: no cover - defensive, tokenizer-specific
            return None
        return None if value is None else int(value)

    def encode(self, text: str) -> list[int]:
        """Encode ordinary corpus text.

        ``add_special_tokens`` is passed explicitly rather than left to the
        library default, which inserts BOS for many models. A corpus being
        measured must not gain structural tokens it does not contain.
        """

        return list(
            self.tokenizer.encode(text, add_special_tokens=self.add_special_tokens)
        )

    def decode(self, token_ids: Sequence[int]) -> str:
        """Decode token IDs back to text, keeping any structural tokens visible."""

        return self.tokenizer.decode(list(token_ids), skip_special_tokens=False)

    def token_repr(self, token_id: int) -> str:
        """Readable representation of one token, for figure labels.

        The raw piece is used rather than the decoded string, so the subword
        boundary marker stays visible instead of being silently turned into a
        space.
        """

        convert = getattr(self.tokenizer, "convert_ids_to_tokens", None)
        if convert is None:
            return repr(self.decode([token_id]))
        return repr(convert(int(token_id)))

    def describe(self) -> dict[str, Any]:
        """Provenance recorded alongside experiment results.

        The three identifiers that are easy to conflate are kept apart:
        ``requested_revision`` is what was asked for, ``resolved_revision`` is
        what the local files actually are when that is honestly available, and
        ``source`` says where they came from. A missing resolved revision is
        recorded as ``None`` rather than back-filled with a guess.
        """

        from importlib import metadata

        try:
            library_version = metadata.version("transformers")
        except Exception:  # pragma: no cover - metadata is environment-specific
            library_version = None

        return {
            "type": "pretrained",
            "backend": "transformers.AutoTokenizer",
            "identifier": self.identifier,
            "requested_revision": self.requested_revision,
            "resolved_revision": getattr(self.tokenizer, "_commit_hash", None),
            "tokenizer_class": type(self.tokenizer).__name__,
            "transformers_version": library_version,
            "vocab_size": self.vocab_size,
            "eligible_vocab_size": len(self.eligible_token_ids),
            "special_token_ids": list(self.special_token_ids),
            "special_tokens": self.special_tokens,
            "adds_special_tokens_on_encode": self.add_special_tokens,
            "source": self.source,
        }


def load_pretrained_tokenizer(
    *,
    identifier: str,
    revision: str | None = None,
    data_root: str | Path | None = None,
    allow_download: bool = False,
    offline: bool = False,
    add_special_tokens: bool = False,
    reporter: Callable[[str], None] | None = None,
) -> PretrainedTokenizer:
    """Load a pretrained tokenizer, from cache or -- when permitted -- the Hub.

    Only tokenizer files are ever requested. Loading is attempted from the local
    cache first, so a cached tokenizer works with no network at all; acquisition
    happens only when it is explicitly permitted, and is announced first.

    Args:
        identifier: Hub identifier, e.g. ``mistralai/Mistral-7B-v0.1``.
        revision: Optional pinned revision. Pin it for a serious experiment.
        data_root: External root holding the tokenizer cache.
        allow_download: Permit fetching files that are not already cached.
        offline: Forbid all network access; implies no acquisition.
        add_special_tokens: Whether corpus encoding inserts structural tokens.
        reporter: Receives a one-line announcement before any network use.

    Returns:
        The loaded :class:`PretrainedTokenizer`.

    Raises:
        DatasetAcquisitionError: If the optional dependency is missing or the
            load itself fails.
        DatasetNotAvailableError: If the tokenizer is not cached and acquisition
            is not permitted.
    """

    from llm_behavior_lab.data.config import resolve_data_root

    auto_tokenizer = _import_auto_tokenizer()
    cache_dir = tokenizer_cache_directory(resolve_data_root(data_root))
    announce = reporter or (lambda _message: None)

    common: dict[str, Any] = {"cache_dir": str(cache_dir)}
    if revision:
        common["revision"] = revision

    # Cache-only first. This is the whole offline story: a tokenizer already on
    # disk loads without the network regardless of the policy flags.
    try:
        tokenizer = auto_tokenizer.from_pretrained(
            identifier, local_files_only=True, **common
        )
    except Exception as cached_error:
        may_acquire = allow_download and not offline
        if not may_acquire:
            blocker = (
                "offline mode is enabled"
                if offline
                else "downloads are not permitted for this invocation"
            )
            raise DatasetNotAvailableError(
                f"Tokenizer {identifier!r} is not available locally and {blocker}.\n"
                f"  checked cache: {cache_dir}\n"
                f"  revision     : {revision or 'default'}\n"
                "Obtain it once with acquisition permitted, for example by rerunning "
                "the same command without --offline or --no-download. Only tokenizer "
                "files are downloaded; no model weights are fetched.\n"
                f"If the optional dependency is missing, install it with: {INSTALL_HINT}"
            ) from cached_error

        announce(
            f"Acquiring tokenizer {identifier!r}\n"
            f"  revision : {revision or 'default (not pinned)'}\n"
            f"  cache    : {cache_dir}\n"
            "  Tokenizer files only; no model weights are downloaded.\n"
            "  This will use the network. Use --offline or --no-download to prevent it."
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            tokenizer = auto_tokenizer.from_pretrained(identifier, **common)
        except Exception as exc:
            raise DatasetAcquisitionError(
                f"Failed to obtain tokenizer {identifier!r}.\n"
                f"  {type(exc).__name__}: {exc}"
            ) from exc
        source = f"downloaded to {cache_dir}"
    else:
        source = f"local cache {cache_dir}"

    loaded = PretrainedTokenizer(
        tokenizer=tokenizer,
        identifier=identifier,
        requested_revision=revision,
        add_special_tokens=add_special_tokens,
        source=source,
    )
    if not loaded.eligible_token_ids:
        raise DatasetAcquisitionError(
            f"Tokenizer {identifier!r} reports no eligible tokens after excluding "
            "special IDs, which cannot be right. Check the tokenizer files."
        )
    return loaded


def tokenizer_cache_footprint(data_root: str | Path | None = None) -> dict[str, Any]:
    """Summarize what the tokenizer cache holds, for reporting.

    Used by the experiment script to state the on-disk cost honestly: this is
    disk usage, not bytes transferred.
    """

    from llm_behavior_lab.data.config import resolve_data_root

    cache_dir = tokenizer_cache_directory(resolve_data_root(data_root))
    if not cache_dir.exists():
        return {"path": str(cache_dir), "exists": False, "num_files": 0, "bytes": 0}
    total = 0
    files = 0
    for root, _directories, names in os.walk(cache_dir):
        for name in names:
            path = Path(root) / name
            if path.is_file() and not path.is_symlink():
                total += path.stat().st_size
                files += 1
    return {"path": str(cache_dir), "exists": True, "num_files": files, "bytes": total}
