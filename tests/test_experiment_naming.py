"""Tests for run-directory naming and tokenizer provenance reporting.

Two housekeeping concerns that turned out to be the same concern: a run's
identity has to be derived from resolved metadata, and it has to say the same
thing everywhere it appears. A pretrained run once announced itself as ``char``
on the console while its persisted metadata correctly said ``pretrained``, which
is the worst kind of wrong because the vocabulary size printed beside it looked
plausible.

The naming module imports nothing beyond the standard library, so these tests
stay fast and independent of the modelling stack.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from llm_behavior_lab.experiment.naming import (
    COMPONENT_MAX_LENGTH,
    compose_run_id,
    dataset_slug,
    model_slug,
    run_timestamp,
    short_run_code,
    slugify,
    tokenizer_slug,
    vocabulary_suffix,
)

MOMENT = datetime(2026, 8, 12, 14, 1, 12, tzinfo=timezone.utc)

CHAR_TOKENIZER = {"type": "char", "vocab_size": 39, "eligible_vocab_size": 39}
MISTRAL_TOKENIZER = {
    "type": "pretrained",
    "identifier": "mistralai/Mistral-7B-v0.1",
    "requested_revision": None,
    "vocab_size": 32000,
    "eligible_vocab_size": 31997,
}

#: Every component of a run name must survive an ordinary filesystem untouched.
SAFE_NAME = re.compile(r"[A-Za-z0-9_.\-]+")


def _character_run(**overrides) -> str:
    fields = {
        "dataset": "tiny_local_text",
        "tokenizer": CHAR_TOKENIZER,
        "model_name": "llama_tiny",
        "model_vocab_size": 256,
        "num_positions": 256,
        "num_initializations": 3,
        "num_replicates": 2,
        "moment": MOMENT,
        "code": "ef0cd1b9",
    }
    fields.update(overrides)
    return compose_run_id(**fields)


def _subword_run(**overrides) -> str:
    fields = {
        "dataset": "wikitext2_raw_train1k",
        "tokenizer": MISTRAL_TOKENIZER,
        "model_name": "llama_tiny",
        "model_vocab_size": 32000,
        "num_positions": 8192,
        "num_initializations": 4,
        "num_replicates": 4,
        "moment": MOMENT,
        "code": "9bb2a015",
    }
    fields.update(overrides)
    return compose_run_id(**fields)


# -- slug formation -------------------------------------------------------


def test_slugify_produces_a_filesystem_safe_component() -> None:
    """Separators, spaces, and slashes must never reach a path component."""

    assert slugify("Salesforce/WikiText 2 (raw)") == "salesforce-wikitext-2-raw"[:COMPONENT_MAX_LENGTH]
    assert SAFE_NAME.fullmatch(slugify("a b/c\\d"))
    assert slugify("a b/c\\d") == "a-b-c-d"


def test_slugify_never_returns_an_empty_component() -> None:
    """An empty path component would silently collapse the directory name."""

    assert slugify("") == "unknown"
    assert slugify("///") == "unknown"
    assert slugify(None) == "none"


def test_slugify_bounds_the_component_length() -> None:
    """Five components are joined, so each must stay short."""

    assert len(slugify("x" * 500)) == COMPONENT_MAX_LENGTH


def test_vocabulary_suffix_compresses_large_sizes() -> None:
    """The character-versus-subword distinction is exactly the vocabulary size."""

    assert vocabulary_suffix(39) == "39"
    assert vocabulary_suffix(999) == "999"
    assert vocabulary_suffix(32000) == "32k"
    assert vocabulary_suffix(50257) == "50.3k"
    assert vocabulary_suffix(0) == ""
    assert vocabulary_suffix(None) == ""


def test_tokenizer_slug_distinguishes_character_from_pretrained() -> None:
    """The slug is derived from provenance, not from a hard-coded name."""

    assert tokenizer_slug(CHAR_TOKENIZER) == "char-39"
    assert tokenizer_slug(MISTRAL_TOKENIZER) == "mistral-7b-v0.1-32k"


def test_tokenizer_slug_follows_a_different_identifier() -> None:
    """A different backend must rename itself with no code change."""

    assert tokenizer_slug(
        {"type": "pretrained", "identifier": "openai-community/gpt2", "vocab_size": 50257}
    ) == "gpt2-50.3k"


def test_tokenizer_slug_survives_missing_provenance() -> None:
    """A malformed description must not produce an unusable path."""

    assert tokenizer_slug(None) == "unknown"
    assert SAFE_NAME.fullmatch(tokenizer_slug({}))


def test_model_slug_includes_the_output_vocabulary() -> None:
    """The same architecture at 256 and 32000 tokens is a different experiment."""

    assert model_slug("llama_tiny", 256) == "llama-tiny-256"
    assert model_slug("llama_tiny", 32000) == "llama-tiny-32k"


def test_dataset_slug_is_derived_from_the_resolved_name() -> None:
    """Naming follows the dataset the run actually used."""

    assert dataset_slug("wikitext2_raw_train1k") == "wikitext2-raw-train1k"


# -- the composed name ----------------------------------------------------


def test_the_character_run_name_carries_its_identity() -> None:
    """A directory listing should be readable without opening metadata."""

    name = _character_run()

    assert name == (
        "20260812-140112__tiny-local-text__char-39__llama-tiny-256__N256-I3-R2__ef0cd1b9"
    )


def test_the_subword_run_name_carries_its_identity() -> None:
    """Same grammar, visibly different experiment."""

    name = _subword_run()

    assert name == (
        "20260812-140112__wikitext2-raw-train1k__mistral-7b-v0.1-32k"
        "__llama-tiny-32k__N8192-I4-R4__9bb2a015"
    )


def test_the_name_is_deterministic_for_fixed_inputs() -> None:
    """Only the timestamp and the random code may vary."""

    assert _subword_run() == _subword_run()


def test_the_counts_that_set_statistical_resolution_are_present() -> None:
    """N, I, and R are what a reader compares between runs."""

    assert "N8192-I4-R4" in _subword_run()
    assert "N256-I3-R2" in _character_run()


def test_hyperparameters_beyond_the_chosen_axes_stay_out_of_the_name() -> None:
    """Temperature, top-p, seeds, and revisions belong in metadata.

    Encoding them here would make the name unreadable and would duplicate
    configuration that the snapshots already hold verbatim.
    """

    name = _subword_run()

    for absent in ("temperature", "0.6", "top_p", "seed", "20240601", "block"):
        assert absent not in name


def test_every_generated_name_is_filesystem_safe() -> None:
    """Including under hostile inputs."""

    hostile = compose_run_id(
        dataset="a/b c:d*e",
        tokenizer={"type": "pretrained", "identifier": "x y/Z:!", "vocab_size": 32000},
        model_name="m n/o",
        model_vocab_size=32000,
        num_positions=1,
        num_initializations=1,
        num_replicates=1,
        moment=MOMENT,
        code="0000ffff",
    )

    assert SAFE_NAME.fullmatch(hostile), hostile


def test_the_name_length_stays_bounded() -> None:
    """Long identifiers must not produce an unusable path."""

    name = compose_run_id(
        dataset="d" * 300,
        tokenizer={"type": "pretrained", "identifier": "org/" + "t" * 300, "vocab_size": 32000},
        model_name="m" * 300,
        model_vocab_size=32000,
        num_positions=99999,
        num_initializations=99,
        num_replicates=99,
        moment=MOMENT,
        code="0000ffff",
    )

    assert len(name) < 160


def test_run_codes_are_unique_enough_to_prevent_collisions() -> None:
    """Two runs in the same second must still get distinct directories."""

    codes = {short_run_code() for _ in range(2000)}

    assert len(codes) == 2000
    assert all(SAFE_NAME.fullmatch(code) for code in codes)


def test_the_timestamp_is_utc_and_sortable() -> None:
    """Lexicographic order must match chronological order."""

    earlier = run_timestamp(datetime(2026, 8, 12, 9, 0, 0, tzinfo=timezone.utc))
    later = run_timestamp(datetime(2026, 8, 12, 14, 1, 12, tzinfo=timezone.utc))

    assert earlier == "20260812-090000"
    assert later == "20260812-140112"
    assert earlier < later


def test_a_run_name_is_usable_as_a_directory(tmp_path) -> None:
    """The end-to-end property all the normalization exists for."""

    for name in (_character_run(), _subword_run()):
        (tmp_path / name).mkdir()
        assert (tmp_path / name).is_dir()


# -- tokenizer provenance reporting ---------------------------------------


def _experiment_script():
    """Import the experiment script the way the other script tests do."""

    path = Path(__file__).resolve().parents[1] / "scripts" / "run_initialization_distribution_experiment.py"
    spec = importlib.util.spec_from_file_location("initialization_distribution_script", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_a_character_run_reports_character_tokenization() -> None:
    """The historical behaviour, which must not regress in the other direction."""

    line = _experiment_script()._describe_tokenizer_line(CHAR_TOKENIZER)

    assert line.startswith("char")
    assert "39" in line
    assert "pretrained" not in line


def test_a_pretrained_run_reports_its_backend_and_identifier() -> None:
    """The reported bug: this printed 'char, vocab size 32000'."""

    line = _experiment_script()._describe_tokenizer_line(MISTRAL_TOKENIZER)

    assert line.startswith("pretrained")
    assert "mistralai/Mistral-7B-v0.1" in line
    assert "32000" in line
    assert "char" not in line


def test_the_reported_identity_is_derived_not_hard_coded() -> None:
    """A different tokenizer must report itself with no code change."""

    line = _experiment_script()._describe_tokenizer_line(
        {"type": "pretrained", "identifier": "openai-community/gpt2", "vocab_size": 50257}
    )

    assert "openai-community/gpt2" in line
    assert "mistral" not in line.lower()


def test_the_reported_identity_surfaces_an_unpinned_revision() -> None:
    """Reproducibility depends on it, so it belongs in the summary."""

    assert "revision=unpinned" in _experiment_script()._describe_tokenizer_line(MISTRAL_TOKENIZER)
    assert "revision=v0.3" in _experiment_script()._describe_tokenizer_line(
        {**MISTRAL_TOKENIZER, "requested_revision": "v0.3"}
    )


def test_the_displayed_identity_matches_what_gets_persisted() -> None:
    """One mapping feeds the console line, the run name, and the metadata.

    That shared origin is the actual fix: the previous bug was possible only
    because the printed line had its own independent notion of the tokenizer.
    """

    script = _experiment_script()

    line = script._describe_tokenizer_line(MISTRAL_TOKENIZER)
    name = compose_run_id(
        dataset="wikitext2_raw_train1k",
        tokenizer=MISTRAL_TOKENIZER,
        model_name="llama_tiny",
        model_vocab_size=32000,
        num_positions=8192,
        num_initializations=4,
        num_replicates=4,
        moment=MOMENT,
        code="9bb2a015",
    )

    assert MISTRAL_TOKENIZER["type"] in line
    assert "mistral" in name
    assert str(MISTRAL_TOKENIZER["vocab_size"]) in line
    assert "32k" in name


def test_the_eligible_support_is_shown_when_it_differs() -> None:
    """31997 of 32000 is the number the statistics actually use."""

    line = _experiment_script()._describe_tokenizer_line(MISTRAL_TOKENIZER)

    assert "31997 eligible" in line
    # A character vocabulary has no structural tokens, so no qualifier is added.
    assert "eligible" not in _experiment_script()._describe_tokenizer_line(CHAR_TOKENIZER)
