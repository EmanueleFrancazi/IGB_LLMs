"""Simple tokenizer utilities for early text-only experiments.

Phase 3 intentionally starts with a character-level tokenizer. This keeps the
pipeline transparent and removes any dependency on external tokenizer files or
network downloads. Later phases can add BPE, SentencePiece, or Hugging Face
compatible tokenizers behind the same basic encode/decode interface.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CharTokenizer:
    """Deterministic character-level tokenizer.

    The vocabulary is built from the sorted set of characters appearing in a
    corpus. Sorting makes token IDs reproducible for the same text.
    """

    stoi: dict[str, int]
    itos: list[str]

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        """Build a tokenizer from all unique characters in ``text``."""

        if not text:
            raise ValueError("Cannot build a tokenizer from empty text.")

        vocab = sorted(set(text))
        stoi = {character: index for index, character in enumerate(vocab)}
        return cls(stoi=stoi, itos=vocab)

    @property
    def vocab_size(self) -> int:
        """Return the number of tokens in the vocabulary."""

        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        """Convert text into token IDs.

        Raises:
            ValueError: If the text contains a character that is not in this
                tokenizer's vocabulary.
        """

        token_ids: list[int] = []
        for character in text:
            if character not in self.stoi:
                raise ValueError(f"Character {character!r} is not in the tokenizer vocabulary.")
            token_ids.append(self.stoi[character])
        return token_ids

    def decode(self, token_ids: list[int] | tuple[int, ...]) -> str:
        """Convert token IDs back into text."""

        characters: list[str] = []
        for token_id in token_ids:
            if token_id < 0 or token_id >= self.vocab_size:
                raise ValueError(
                    f"Token ID {token_id} is outside the vocabulary range [0, {self.vocab_size})."
                )
            characters.append(self.itos[token_id])
        return "".join(characters)
