"""Tests for the tokenizer interface and its two implementations.

The pretrained backend is exercised through a fake that mimics the small part of
the Hugging Face tokenizer surface this project uses. No test reaches the
network, none requires ``transformers`` to be installed, and none may load a
pretrained model: that last point is asserted directly, because the whole
interpretation of the experiment depends on the model staying random.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_behavior_lab.data import errors
from llm_behavior_lab.data import pretrained_tokenizer as pretrained
from llm_behavior_lab.data.pretrained_tokenizer import (
    PretrainedTokenizer,
    load_pretrained_tokenizer,
    tokenizer_cache_directory,
)
from llm_behavior_lab.data.tokenizer import CharTokenizer, Tokenizer, build_tokenizer

CORPUS = "hello tokenizer world"


class FakeHFTokenizer:
    """The slice of the Hugging Face tokenizer surface this project uses."""

    def __init__(self, *, size: int = 16, special: dict[str, int] | None = None) -> None:
        self._size = size
        self._special = special or {"bos_token": 1, "eos_token": 2, "unk_token": 0}
        self.encode_calls: list[dict] = []

    def __len__(self) -> int:
        return self._size

    @property
    def all_special_ids(self) -> list[int]:
        return sorted(self._special.values())

    @property
    def special_tokens_map(self) -> dict[str, str]:
        return {name: f"<{name}>" for name in self._special}

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._special[token.strip("<>")]

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return f"▁piece{token_id}"

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        self.encode_calls.append({"text": text, "add_special_tokens": add_special_tokens})
        body = [(index % (self._size - 3)) + 3 for index, _ in enumerate(text)]
        return [1, *body, 2] if add_special_tokens else body

    def decode(self, token_ids, skip_special_tokens: bool = False) -> str:
        return "".join(f"[{int(value)}]" for value in token_ids)


def install_fake_tokenizer(monkeypatch, *, fake=None, fail_local: bool = False) -> list[dict]:
    """Replace the AutoTokenizer seam and record how it was called."""

    calls: list[dict] = []
    instance = fake or FakeHFTokenizer()

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(identifier, **kwargs):
            calls.append({"identifier": identifier, **kwargs})
            if fail_local and kwargs.get("local_files_only"):
                raise OSError("not cached")
            return instance

    monkeypatch.setattr(pretrained, "_import_auto_tokenizer", lambda: FakeAutoTokenizer)
    return calls


# -- the character tokenizer is unchanged --------------------------------


def test_char_tokenizer_round_trip_is_unchanged() -> None:
    """The original behaviour must survive the interface being added."""

    tokenizer = CharTokenizer.from_text(CORPUS)

    assert tokenizer.decode(tokenizer.encode(CORPUS)) == CORPUS
    assert tokenizer.vocab_size == len(set(CORPUS))


def test_char_tokenizer_has_no_special_tokens() -> None:
    """Every character is an ordinary corpus token, so nothing is excluded."""

    tokenizer = CharTokenizer.from_text(CORPUS)

    assert tokenizer.special_token_ids == ()
    assert tokenizer.eligible_token_ids == tuple(range(tokenizer.vocab_size))


def test_char_tokenizer_describes_itself() -> None:
    """Provenance must record that the vocabulary came from the corpus."""

    described = CharTokenizer.from_text(CORPUS).describe()

    assert described["type"] == "char"
    assert described["adds_special_tokens_on_encode"] is False
    assert described["eligible_vocab_size"] == described["vocab_size"]


def test_both_implementations_satisfy_the_interface() -> None:
    """One protocol, two implementations, no base class."""

    assert isinstance(CharTokenizer.from_text(CORPUS), Tokenizer)
    assert isinstance(PretrainedTokenizer(tokenizer=FakeHFTokenizer(), identifier="x"), Tokenizer)


# -- the pretrained adapter ----------------------------------------------


def test_canonical_ids_are_preserved_and_specials_excluded() -> None:
    """Structural IDs leave the support; the rest keep their canonical numbers."""

    adapter = PretrainedTokenizer(tokenizer=FakeHFTokenizer(size=10), identifier="fake/model")

    assert adapter.vocab_size == 10
    assert adapter.special_token_ids == (0, 1, 2)
    assert adapter.eligible_token_ids == (3, 4, 5, 6, 7, 8, 9)


def test_corpus_encoding_adds_no_structural_tokens() -> None:
    """A corpus is measured, not a prompt assembled."""

    fake = FakeHFTokenizer()
    adapter = PretrainedTokenizer(tokenizer=fake, identifier="fake/model", add_special_tokens=False)

    encoded = adapter.encode("abc")

    assert fake.encode_calls[-1]["add_special_tokens"] is False
    assert not set(encoded).intersection(adapter.special_token_ids)


def test_special_tokens_can_be_requested_explicitly() -> None:
    """The policy is a choice that is recorded, not a hidden default."""

    fake = FakeHFTokenizer()
    adapter = PretrainedTokenizer(tokenizer=fake, identifier="fake/model", add_special_tokens=True)

    adapter.encode("abc")

    assert fake.encode_calls[-1]["add_special_tokens"] is True
    assert adapter.describe()["adds_special_tokens_on_encode"] is True


def test_vocab_size_counts_added_tokens() -> None:
    """``len(tokenizer)`` covers tokens added after training; ``vocab_size`` may not.

    Using the narrower number would leave the model's output layer unable to
    represent every reachable ID.
    """

    fake = FakeHFTokenizer(size=40)
    fake.vocab_size = 32  # what the base tokenizer would report

    assert PretrainedTokenizer(tokenizer=fake, identifier="fake/model").vocab_size == 40


def test_token_repr_keeps_subword_markers_visible() -> None:
    """Decoded text would hide the leading-space marker; the raw piece does not."""

    adapter = PretrainedTokenizer(tokenizer=FakeHFTokenizer(), identifier="fake/model")

    assert "▁" in adapter.token_repr(5)


def test_provenance_separates_requested_and_resolved_revision() -> None:
    """Three identifiers that must never be conflated."""

    fake = FakeHFTokenizer()
    fake._commit_hash = "abc123"
    described = PretrainedTokenizer(
        tokenizer=fake, identifier="fake/model", requested_revision="main", source="local cache /x"
    ).describe()

    assert described["identifier"] == "fake/model"
    assert described["requested_revision"] == "main"
    assert described["resolved_revision"] == "abc123"
    assert described["source"] == "local cache /x"
    assert described["eligible_vocab_size"] == described["vocab_size"] - 3


def test_provenance_records_an_absent_revision_as_none() -> None:
    """A missing resolved revision is recorded, never back-filled with a guess."""

    described = PretrainedTokenizer(tokenizer=FakeHFTokenizer(), identifier="fake/model").describe()

    assert described["requested_revision"] is None
    assert described["resolved_revision"] is None


# -- loading, caching, and offline behaviour ------------------------------


def test_a_cached_tokenizer_loads_without_permission_to_download(monkeypatch, tmp_path) -> None:
    """Reuse of local files never requires acquisition to be permitted."""

    calls = install_fake_tokenizer(monkeypatch)

    loaded = load_pretrained_tokenizer(
        identifier="fake/model", data_root=tmp_path, allow_download=False
    )

    assert calls[0]["local_files_only"] is True
    assert len(calls) == 1
    assert "local cache" in loaded.source


def test_a_missing_tokenizer_offline_fails_with_guidance(monkeypatch, tmp_path) -> None:
    """The error must say what is missing, where it looked, and what to do."""

    install_fake_tokenizer(monkeypatch, fail_local=True)

    with pytest.raises(errors.DatasetNotAvailableError) as excinfo:
        load_pretrained_tokenizer(
            identifier="fake/model", data_root=tmp_path, allow_download=True, offline=True
        )

    message = str(excinfo.value)
    assert "offline mode is enabled" in message
    assert str(tokenizer_cache_directory(tmp_path)) in message
    assert "no model weights" in message


def test_a_missing_tokenizer_without_download_permission_is_refused(monkeypatch, tmp_path) -> None:
    """Acquisition never happens as a silent library default."""

    install_fake_tokenizer(monkeypatch, fail_local=True)

    with pytest.raises(errors.DatasetNotAvailableError, match="downloads are not permitted"):
        load_pretrained_tokenizer(
            identifier="fake/model", data_root=tmp_path, allow_download=False
        )


def test_acquisition_is_announced_before_the_network_is_used(monkeypatch, tmp_path) -> None:
    """The user learns what is about to happen, and that weights are not part of it."""

    install_fake_tokenizer(monkeypatch, fail_local=True)
    messages: list[str] = []

    load_pretrained_tokenizer(
        identifier="fake/model",
        data_root=tmp_path,
        allow_download=True,
        reporter=messages.append,
    )

    assert len(messages) == 1
    assert "Tokenizer files only; no model weights are downloaded." in messages[0]


def test_acquisition_falls_back_to_the_network_only_after_the_cache(monkeypatch, tmp_path) -> None:
    """Local first, then the Hub -- never the other way round."""

    calls = install_fake_tokenizer(monkeypatch, fail_local=True)

    load_pretrained_tokenizer(identifier="fake/model", data_root=tmp_path, allow_download=True)

    assert calls[0]["local_files_only"] is True
    assert "local_files_only" not in calls[1]


def test_the_revision_is_passed_through_when_pinned(monkeypatch, tmp_path) -> None:
    """A pinned revision is what makes a serious experiment reproducible."""

    calls = install_fake_tokenizer(monkeypatch)

    load_pretrained_tokenizer(identifier="fake/model", revision="v1.0", data_root=tmp_path)

    assert calls[0]["revision"] == "v1.0"


def test_tokenizer_artifacts_live_outside_the_repository(tmp_path) -> None:
    """Nothing downloadable may land in a tracked path."""

    cache = tokenizer_cache_directory(tmp_path)

    assert cache == tmp_path / "tokenizers"
    assert Path(__file__).resolve().parents[1] not in cache.parents


def test_a_missing_optional_dependency_names_the_extra(monkeypatch) -> None:
    """The character workflow must not require the pretrained dependency."""

    monkeypatch.setitem(__import__("sys").modules, "transformers", None)

    with pytest.raises(errors.DatasetAcquisitionError, match=r'pip install -e ".\[tokenizers\]"'):
        pretrained._import_auto_tokenizer()


def test_no_pretrained_model_is_ever_loaded() -> None:
    """The model stays random; only the tokenizer is pretrained.

    Asserted against the parsed source rather than the raw text, so the module
    may freely *document* what it does not do while the test still catches a
    real call. This is an interpretation guarantee, not an implementation
    detail: any of these entry points would quietly turn the experiment into
    something else.
    """

    import ast

    tree = ast.parse(Path(pretrained.__file__).read_text(encoding="utf-8"))
    forbidden = {
        "AutoModel",
        "AutoModelForCausalLM",
        "AutoModelForSeq2SeqLM",
        "AutoModelWithLMHead",
        "pipeline",
    }

    referenced: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            referenced.add(node.id)
        elif isinstance(node, ast.Attribute):
            referenced.add(node.attr)
        elif isinstance(node, ast.ImportFrom):
            referenced.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            referenced.update(alias.name for alias in node.names)

    assert not (referenced & forbidden), (
        "the tokenizer backend must never reference a pretrained model loader: "
        f"{sorted(referenced & forbidden)}"
    )
    assert "AutoTokenizer" in referenced


# -- build_tokenizer dispatch ---------------------------------------------


def test_build_tokenizer_defaults_to_the_character_implementation() -> None:
    """An absent section keeps the historical behaviour."""

    assert isinstance(build_tokenizer(None, text=CORPUS), CharTokenizer)
    assert isinstance(build_tokenizer({"type": "char"}, text=CORPUS), CharTokenizer)


def test_build_tokenizer_dispatches_to_the_pretrained_backend(monkeypatch, tmp_path) -> None:
    """Two implementations, one explicit branch, no registry."""

    install_fake_tokenizer(monkeypatch)

    built = build_tokenizer(
        {"type": "pretrained", "identifier": "fake/model", "add_special_tokens": False},
        text=CORPUS,
        data_root=tmp_path,
    )

    assert isinstance(built, PretrainedTokenizer)
    assert built.add_special_tokens is False


def test_an_unknown_tokenizer_type_is_rejected() -> None:
    """Silently defaulting would change which vocabulary an experiment used."""

    with pytest.raises(ValueError, match="Unknown tokenizer.type"):
        build_tokenizer({"type": "sentencepiece"}, text=CORPUS)
