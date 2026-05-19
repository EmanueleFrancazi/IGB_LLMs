"""Text loading and splitting utilities.

This module keeps Phase 3 deliberately small: it loads a local text file,
encodes it with a tokenizer, and creates deterministic train/validation splits.
The resulting token lists can then be consumed by the causal LM batcher.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class TokenSplits:
    """Token IDs split into train and validation portions."""

    train_ids: list[int]
    val_ids: list[int]


def load_text_file(path: str | Path) -> str:
    """Load a UTF-8 text file."""

    resolved_path = Path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Text dataset file does not exist: {resolved_path}")

    text = resolved_path.read_text(encoding="utf-8")
    if not text:
        raise ValueError(f"Text dataset file is empty: {resolved_path}")
    return text


def split_token_ids(
    token_ids: Sequence[int],
    *,
    val_fraction: float = 0.1,
    min_train_tokens: int = 2,
    min_val_tokens: int = 2,
) -> TokenSplits:
    """Create deterministic train/validation token splits.

    The split is contiguous: the first part becomes training data and the final
    part becomes validation data. This is simple, reproducible, and sufficient
    for early pipeline checks.
    """

    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1.")

    total_tokens = len(token_ids)
    if total_tokens < min_train_tokens + min_val_tokens:
        raise ValueError(
            "Not enough tokens to create train/validation splits: "
            f"got {total_tokens}, need at least {min_train_tokens + min_val_tokens}."
        )

    val_size = max(min_val_tokens, int(total_tokens * val_fraction))
    train_size = total_tokens - val_size
    if train_size < min_train_tokens:
        raise ValueError(
            f"Train split would contain {train_size} tokens, but at least "
            f"{min_train_tokens} are required."
        )

    return TokenSplits(
        train_ids=list(token_ids[:train_size]),
        val_ids=list(token_ids[train_size:]),
    )
