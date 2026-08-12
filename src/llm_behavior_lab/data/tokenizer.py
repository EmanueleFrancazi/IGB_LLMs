"""Tokenizer interface and the character-level implementation.

The project started character-level, which keeps the pipeline transparent and
needs no external files. A realistic experiment needs a subword vocabulary of
tens of thousands of tokens, so the two live behind one small interface --
:class:`Tokenizer` -- with exactly two implementations: :class:`CharTokenizer`
here and ``PretrainedTokenizer`` in
:mod:`llm_behavior_lab.data.pretrained_tokenizer`.

The interface is a ``Protocol`` rather than a base class or a registry. Two
implementations do not justify inheritance machinery, and a protocol keeps
``CharTokenizer`` exactly what it already was: a plain frozen dataclass that
nothing needs to subclass.

Beyond encode/decode, the interface carries what a large pretrained vocabulary
forces the analysis to know:

* ``special_token_ids`` -- structural IDs such as BOS, EOS, or padding, which
  are not ordinary corpus tokens;
* ``eligible_token_ids`` -- the predictive support, i.e. every ID a model may
  legitimately be scored on. Canonical IDs are **never** renumbered, so an
  excluded ID simply never appears in this list;
* ``describe()`` -- provenance recorded with every experiment.

A character tokenizer has no special tokens, so its eligible support is its
whole vocabulary and these additions cost it nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    """What the experiment requires of any tokenizer."""

    @property
    def vocab_size(self) -> int:
        """Size of the full canonical vocabulary."""

    @property
    def special_token_ids(self) -> tuple[int, ...]:
        """Structural IDs excluded from the predictive support."""

    @property
    def eligible_token_ids(self) -> tuple[int, ...]:
        """Canonical IDs a model may be scored on, in increasing order."""

    def encode(self, text: str) -> list[int]:
        """Convert text to token IDs without adding structural tokens."""

    def decode(self, token_ids: Sequence[int]) -> str:
        """Convert token IDs back to text."""

    def token_repr(self, token_id: int) -> str:
        """Human-readable representation of one token, for labels."""

    def describe(self) -> dict[str, Any]:
        """Provenance recorded alongside experiment results."""


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

    @property
    def special_token_ids(self) -> tuple[int, ...]:
        """A character vocabulary has no structural tokens."""

        return ()

    @property
    def eligible_token_ids(self) -> tuple[int, ...]:
        """Every character is an ordinary corpus token."""

        return tuple(range(self.vocab_size))

    def token_repr(self, token_id: int) -> str:
        """Readable form of one token; whitespace stays visible."""

        return repr(self.itos[token_id])

    def describe(self) -> dict[str, Any]:
        """Provenance for a tokenizer derived entirely from the corpus."""

        return {
            "type": "char",
            "vocab_size": self.vocab_size,
            "eligible_vocab_size": self.vocab_size,
            "special_token_ids": [],
            "special_tokens": {},
            "adds_special_tokens_on_encode": False,
            "source": "derived from the analysis corpus",
        }

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

    def decode(self, token_ids: Sequence[int]) -> str:
        """Convert token IDs back into text."""

        characters: list[str] = []
        for token_id in token_ids:
            if token_id < 0 or token_id >= self.vocab_size:
                raise ValueError(
                    f"Token ID {token_id} is outside the vocabulary range [0, {self.vocab_size})."
                )
            characters.append(self.itos[token_id])
        return "".join(characters)


TOKENIZER_TYPES = ("char", "pretrained")


def build_tokenizer(
    tokenizer_config: dict[str, Any] | None,
    *,
    text: str,
    data_root: Any = None,
    allow_download: bool = False,
    offline: bool = False,
    reporter: Any = None,
) -> Tokenizer:
    """Build the tokenizer named by a data config's ``tokenizer`` section.

    Expected structure::

        tokenizer:
          type: char                        # or: pretrained

        tokenizer:
          type: pretrained
          identifier: mistralai/Mistral-7B-v0.1
          revision: null                    # pin for a serious experiment
          add_special_tokens: false

    ``char`` derives its vocabulary from ``text`` and needs nothing external.
    ``pretrained`` loads tokenizer files only, through the optional dependency,
    and never touches model weights.

    Args:
        tokenizer_config: The ``tokenizer`` section; ``None`` means character.
        text: Corpus text, used only by the character tokenizer.
        data_root: External root under which tokenizer artifacts are cached.
        allow_download: Permit fetching tokenizer files that are not cached.
        offline: Forbid all network access.
        reporter: Receives a one-line announcement before any network use.

    Returns:
        A tokenizer satisfying :class:`Tokenizer`.

    Raises:
        ValueError: If the configured type is unknown.
    """

    section = dict(tokenizer_config or {})
    tokenizer_type = str(section.get("type", "char"))

    if tokenizer_type == "char":
        return CharTokenizer.from_text(text)

    if tokenizer_type == "pretrained":
        # Imported here so the character-only project never needs the optional
        # dependency, and importing this module never imports transformers.
        from llm_behavior_lab.data.pretrained_tokenizer import load_pretrained_tokenizer

        return load_pretrained_tokenizer(
            identifier=str(section["identifier"]),
            revision=section.get("revision"),
            data_root=data_root,
            allow_download=allow_download,
            offline=offline,
            add_special_tokens=bool(section.get("add_special_tokens", False)),
            reporter=reporter,
        )

    raise ValueError(
        f"Unknown tokenizer.type {tokenizer_type!r}. Supported types: "
        + ", ".join(TOKENIZER_TYPES)
        + "."
    )
