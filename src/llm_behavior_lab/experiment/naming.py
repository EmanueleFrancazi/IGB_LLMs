"""Interpretable run-directory names.

A run identifier used to be a timestamp and a random code, which is
collision-safe and completely opaque: telling two experiments apart meant
opening ``metadata.json`` in each. These helpers add the few axes that actually
distinguish one run from another at a glance, while keeping the random suffix
that makes collisions impossible.

    20260812-140112__wikitext2-raw-train1k__mistral-7b-v0.1-32k__llama-tiny-32k__N8192-I4-R4__9bb2a015

Only the axes worth scanning a directory listing for are included: what data,
which tokenizer, which model, and the three counts that determine the
statistical resolution. Temperature, top-p, seeds, block size, and revisions stay
in the metadata and config snapshots, where they are readable and where nothing
truncates them.

Every component is derived from resolved runtime metadata. Nothing here restates
scientific configuration as a hard-coded string, so a name cannot drift away from
the run it labels.

This module deliberately imports nothing beyond the standard library: naming is
pure string handling and should stay testable without the modelling stack.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any, Mapping

__all__ = [
    "COMPONENT_MAX_LENGTH",
    "compose_run_id",
    "dataset_slug",
    "model_slug",
    "run_timestamp",
    "short_run_code",
    "slugify",
    "tokenizer_slug",
    "vocabulary_suffix",
]

#: Each component is capped so a full name stays comfortably inside filesystem
#: limits even with five components joined.
COMPONENT_MAX_LENGTH = 24

_UNSAFE = re.compile(r"[^A-Za-z0-9.]+")


def slugify(value: Any, *, max_length: int = COMPONENT_MAX_LENGTH) -> str:
    """Reduce a value to a lowercase, filesystem-safe component.

    Runs of anything outside letters, digits, and ``.`` collapse to a single
    hyphen; the result is trimmed of leading and trailing separators and
    truncated. An empty or entirely unsafe value becomes ``"unknown"`` rather
    than an empty path component.
    """

    text = _UNSAFE.sub("-", str(value)).strip("-.").lower()
    if not text:
        return "unknown"
    return text[:max_length].strip("-.") or "unknown"


def vocabulary_suffix(vocab_size: int | None) -> str:
    """Render a vocabulary size compactly: ``39`` stays, ``32000`` becomes ``32k``.

    The distinction between a character run and a subword run is the single most
    useful thing in the name, and it is exactly the vocabulary size.
    """

    if not vocab_size or vocab_size <= 0:
        return ""
    if vocab_size >= 1000:
        thousands = vocab_size / 1000
        rendered = f"{thousands:.0f}" if abs(thousands - round(thousands)) < 0.05 else f"{thousands:.1f}"
        return f"{rendered}k"
    return str(vocab_size)


def dataset_slug(dataset_name: Any) -> str:
    """Component identifying the corpus."""

    return slugify(dataset_name)


def tokenizer_slug(description: Mapping[str, Any] | None) -> str:
    """Component identifying the tokenizer, derived from its own provenance.

    A character tokenizer becomes ``char-39``. A pretrained one uses the final
    segment of its Hub identifier plus the vocabulary size, so
    ``mistralai/Mistral-7B-v0.1`` at 32000 tokens becomes
    ``mistral-7b-v0.1-32k``. The identifier is never assumed: it comes from the
    tokenizer's ``describe()`` output, so a different backend renames itself
    automatically.
    """

    described = dict(description or {})
    vocab = vocabulary_suffix(described.get("vocab_size"))
    kind = str(described.get("type", "unknown"))

    if kind == "pretrained":
        identifier = str(described.get("identifier", "pretrained"))
        base = identifier.rsplit("/", 1)[-1]
        name = slugify(base, max_length=COMPONENT_MAX_LENGTH - len(vocab) - 1)
    else:
        name = slugify(kind, max_length=COMPONENT_MAX_LENGTH - len(vocab) - 1)

    return f"{name}-{vocab}" if vocab else name


def model_slug(model_name: Any, vocab_size: int | None = None) -> str:
    """Component identifying the model, including its output vocabulary.

    The vocabulary belongs here too: the same architecture at 256 and at 32000
    tokens is a materially different experiment.
    """

    vocab = vocabulary_suffix(vocab_size)
    name = slugify(model_name, max_length=COMPONENT_MAX_LENGTH - len(vocab) - 1)
    return f"{name}-{vocab}" if vocab else name


def run_timestamp(moment: datetime | None = None) -> str:
    """UTC timestamp, sortable and readable: ``20260812-140112``."""

    moment = moment or datetime.now(timezone.utc)
    return moment.astimezone(timezone.utc).strftime("%Y%m%d-%H%M%S")


def short_run_code() -> str:
    """Random suffix that keeps two runs in the same second distinct."""

    return uuid.uuid4().hex[:8]


def compose_run_id(
    *,
    dataset: Any,
    tokenizer: Mapping[str, Any] | None,
    model_name: Any,
    model_vocab_size: int | None,
    num_positions: int,
    num_initializations: int,
    num_replicates: int,
    moment: datetime | None = None,
    code: str | None = None,
) -> str:
    """Build the full run-directory name.

    Args:
        dataset: Resolved dataset name.
        tokenizer: The tokenizer's ``describe()`` mapping.
        model_name: Registered model name.
        model_vocab_size: The model's output vocabulary.
        num_positions: Evaluation positions, ``N``.
        num_initializations: Independent initializations, ``I``.
        num_replicates: Nucleus replicates per initialization, ``R``.
        moment: Timestamp, defaulting to now. Supplied by tests for determinism.
        code: Collision-safe suffix, defaulting to a fresh random one.

    Returns:
        A name of the form
        ``<timestamp>__<dataset>__<tokenizer>__<model>__N..-I..-R..__<code>``.
    """

    counts = f"N{int(num_positions)}-I{int(num_initializations)}-R{int(num_replicates)}"
    return "__".join(
        [
            run_timestamp(moment),
            dataset_slug(dataset),
            tokenizer_slug(tokenizer),
            model_slug(model_name, model_vocab_size),
            counts,
            code or short_run_code(),
        ]
    )
