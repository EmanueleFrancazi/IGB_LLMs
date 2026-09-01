"""Tests for the persisted initialization-experiment record.

The record is the only durable output of the experiment, so these tests focus on
the two properties a later re-analysis depends on: every vector stays aligned by
token ID, and nothing is lost on the way to disk and back.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from llm_behavior_lab.analysis import (
    RECORD_VERSION,
    InitializationExperimentRecord,
    load_record,
)

VOCAB_SIZE = 5
NUM_INITIALIZATIONS = 3
NUM_REPLICATES = 2


def _record(**overrides) -> InitializationExperimentRecord:
    """Build a small, fully specified record."""

    rng = np.random.default_rng(0)
    fields = {
        "corpus_counts": np.array([50, 40, 30, 20, 10]),
        "selected_target_counts": np.array([5, 4, 3, 2, 1]),
        "greedy_counts": rng.integers(0, 20, size=(NUM_INITIALIZATIONS, VOCAB_SIZE)),
        "nucleus_counts": rng.integers(
            0, 20, size=(NUM_INITIALIZATIONS, NUM_REPLICATES, VOCAB_SIZE)
        ),
        "mean_predicted_probabilities": rng.dirichlet(
            np.ones(VOCAB_SIZE), size=NUM_INITIALIZATIONS
        ),
        "model_seeds": np.array([1000, 1001, 1002]),
        "metadata": {"num_positions": 40, "tokens": list("abcde")},
    }
    fields.update(overrides)
    return InitializationExperimentRecord.build(**fields)


def test_token_ids_are_derived_as_the_canonical_axis() -> None:
    """Position in every array is the token ID; nothing else defines alignment."""

    record = _record()

    assert record.token_ids.tolist() == [0, 1, 2, 3, 4]
    assert record.vocab_size == VOCAB_SIZE
    assert record.num_initializations == NUM_INITIALIZATIONS
    assert record.num_replicates == NUM_REPLICATES


def test_round_trip_preserves_every_array_and_the_metadata(tmp_path) -> None:
    """A reloaded record must be indistinguishable from the original."""

    record = _record()
    record.save(tmp_path)
    reloaded = load_record(tmp_path)

    assert np.array_equal(reloaded.corpus_counts, record.corpus_counts)
    assert np.array_equal(reloaded.selected_target_counts, record.selected_target_counts)
    assert np.array_equal(reloaded.greedy_counts, record.greedy_counts)
    assert np.array_equal(reloaded.nucleus_counts, record.nucleus_counts)
    assert np.allclose(reloaded.mean_predicted_probabilities, record.mean_predicted_probabilities)
    assert np.array_equal(reloaded.model_seeds, record.model_seeds)
    assert reloaded.metadata["num_positions"] == 40
    assert reloaded.tokens == ("a", "b", "c", "d", "e")


def test_round_trip_preserves_token_alignment_under_a_permutation(tmp_path) -> None:
    """A per-token signature must survive the write unchanged.

    Distinct values per token would expose any reordering introduced by the
    archive format.
    """

    corpus = np.arange(1, VOCAB_SIZE + 1) * 7
    greedy = np.stack([np.arange(VOCAB_SIZE) + offset for offset in (100, 200, 300)])
    record = _record(corpus_counts=corpus, greedy_counts=greedy)
    record.save(tmp_path)

    reloaded = load_record(tmp_path)

    assert reloaded.corpus_counts.tolist() == corpus.tolist()
    assert reloaded.greedy_counts.tolist() == greedy.tolist()


def test_saved_metadata_is_human_readable_json(tmp_path) -> None:
    """The protocol must be inspectable without NumPy."""

    _record(
        metadata={
            "num_positions": 40,
            "tokens": list("abcde"),
            "record_version": RECORD_VERSION,
        }
    ).save(tmp_path)

    payload = json.loads((tmp_path / "initialization_distribution.json").read_text(encoding="utf-8"))

    assert payload["num_positions"] == 40
    assert payload["record_version"] == RECORD_VERSION


def test_a_record_that_declares_no_version_saves_without_one(tmp_path) -> None:
    """Absence is a real value, not a gap to be filled in.

    Records predating the field carry no version, and `save()` must not invent
    one for them -- stamping the module constant is exactly how a re-saved v11
    archive used to come back claiming to be current.
    """

    record = _record()
    assert record.record_version is None

    record.save(tmp_path)
    payload = json.loads(
        (tmp_path / "initialization_distribution.json").read_text(encoding="utf-8")
    )

    assert "record_version" not in payload
    assert load_record(tmp_path).record_version is None


def test_two_files_are_written_with_deterministic_names(tmp_path) -> None:
    """Reruns overwrite rather than accumulate."""

    _record().save(tmp_path)
    _record().save(tmp_path)

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "initialization_distribution.json",
        "initialization_distribution.npz",
    ]


def test_fractions_are_normalized_per_row() -> None:
    """Every distribution the analysis consumes must sum to one."""

    record = _record()

    assert record.corpus_fractions.sum() == pytest.approx(1.0)
    assert record.selected_target_fractions.sum() == pytest.approx(1.0)
    assert record.greedy_fractions.sum(axis=1).tolist() == pytest.approx([1.0] * NUM_INITIALIZATIONS)
    assert record.nucleus_fractions.sum(axis=1).tolist() == pytest.approx([1.0] * NUM_INITIALIZATIONS)


def test_nucleus_fractions_average_replicates_within_an_initialization() -> None:
    """Replicates are collapsed before initializations are ever compared."""

    counts = np.array([[[10, 0], [0, 10]]])
    record = _record(
        corpus_counts=np.array([1, 1]),
        selected_target_counts=np.array([1, 1]),
        greedy_counts=np.array([[5, 5]]),
        nucleus_counts=counts,
        mean_predicted_probabilities=np.array([[0.5, 0.5]]),
        model_seeds=np.array([7]),
        metadata={"num_positions": 10, "tokens": ["a", "b"]},
    )

    assert record.nucleus_replicate_fractions.tolist() == [[[1.0, 0.0], [0.0, 1.0]]]
    assert record.nucleus_fractions.tolist() == [[0.5, 0.5]]


def test_policy_fractions_selects_the_requested_policy() -> None:
    """And rejects an unknown one rather than defaulting silently."""

    record = _record()

    assert np.array_equal(record.policy_fractions("greedy"), record.greedy_fractions)
    assert np.array_equal(record.policy_fractions("nucleus"), record.nucleus_fractions)
    with pytest.raises(ValueError, match="Unknown policy"):
        record.policy_fractions("beam")


def test_misaligned_arrays_are_rejected_at_construction() -> None:
    """A shape error must fail loudly instead of silently broadcasting."""

    with pytest.raises(ValueError, match="vocabulary"):
        _record(nucleus_counts=np.zeros((NUM_INITIALIZATIONS, NUM_REPLICATES, VOCAB_SIZE + 1)))
    with pytest.raises(ValueError, match="initializations"):
        _record(greedy_counts=np.zeros((NUM_INITIALIZATIONS + 1, VOCAB_SIZE)))
    with pytest.raises(ValueError, match="tokens"):
        _record(metadata={"num_positions": 40, "tokens": ["a", "b"]})


def test_repeated_model_seeds_are_rejected() -> None:
    """Two identical seeds are the same initialization, not two samples."""

    with pytest.raises(ValueError, match="distinct"):
        _record(model_seeds=np.array([1000, 1000, 1002]))


def test_loading_an_incomplete_record_directory_fails_clearly(tmp_path) -> None:
    """A missing half of the record must not load as an empty result."""

    record = _record()
    record.save(tmp_path)
    (tmp_path / "initialization_distribution.npz").unlink()

    with pytest.raises(FileNotFoundError, match="arrays are missing"):
        load_record(tmp_path)


# --- Eligible predictive support (record version 2) ----------------------


def test_eligible_token_ids_default_to_the_whole_vocabulary() -> None:
    """A vocabulary without structural tokens needs no extra bookkeeping."""

    record = _record()

    assert record.eligible_token_ids.tolist() == list(range(VOCAB_SIZE))
    assert record.eligible_vocab_size == VOCAB_SIZE
    assert record.special_token_ids.tolist() == []


def test_excluded_ids_keep_their_canonical_positions() -> None:
    """Exclusion removes tokens from the support, never renumbers the rest."""

    record = _record(
        corpus_counts=np.array([0, 40, 30, 20, 10]),
        eligible_token_ids=np.array([1, 2, 3, 4]),
    )

    assert record.vocab_size == 5
    assert record.eligible_vocab_size == 4
    assert record.special_token_ids.tolist() == [0]
    assert record.eligible_mask.tolist() == [False, True, True, True, True]


def test_corpus_mass_on_an_excluded_token_is_rejected() -> None:
    """Structural tokens in the corpus mean the encoding policy was wrong."""

    with pytest.raises(ValueError, match="Excluded token IDs carry corpus counts"):
        _record(
            corpus_counts=np.array([5, 40, 30, 20, 10]),
            eligible_token_ids=np.array([1, 2, 3, 4]),
        )


def test_corpus_observed_support_counts_only_tokens_that_occur() -> None:
    """Distinct from both the full vocabulary and the eligible support."""

    record = _record(
        corpus_counts=np.array([0, 40, 30, 0, 0]),
        eligible_token_ids=np.array([1, 2, 3, 4]),
    )

    assert record.corpus_observed_support == 2


def test_eligible_support_round_trips(tmp_path) -> None:
    """The support must survive persistence, or later analysis silently widens."""

    record = _record(
        corpus_counts=np.array([0, 40, 30, 20, 10]),
        eligible_token_ids=np.array([1, 2, 3, 4]),
    )
    record.save(tmp_path)

    reloaded = load_record(tmp_path)

    assert reloaded.eligible_token_ids.tolist() == [1, 2, 3, 4]
    assert reloaded.eligible_vocab_size == 4


def test_a_version_one_record_loads_with_a_full_support(tmp_path) -> None:
    """Older records predate exclusion, so every token was eligible.

    Defaulting reproduces their original meaning exactly, which is cheaper and
    less error-prone than a migration for a format with no structural tokens.
    """

    record = _record()
    record.save(tmp_path)
    archive = tmp_path / "initialization_distribution.npz"
    arrays = {key: value for key, value in np.load(archive).items()}
    del arrays["eligible_token_ids"]
    np.savez_compressed(archive, **arrays)

    reloaded = load_record(tmp_path)

    assert reloaded.eligible_token_ids.tolist() == list(range(VOCAB_SIZE))


def test_out_of_range_eligible_ids_are_rejected() -> None:
    """A malformed support must not silently shrink or widen the comparison."""

    with pytest.raises(ValueError, match="outside the vocabulary"):
        _record(eligible_token_ids=np.array([0, 1, VOCAB_SIZE]))
    with pytest.raises(ValueError, match="distinct"):
        _record(eligible_token_ids=np.array([0, 0, 1]))


# --- Input conditions and the uniform null (record version 3) -------------


def _extended(**overrides) -> InitializationExperimentRecord:
    """A record carrying both version-3 additions."""

    fields = {
        "condition_greedy_counts": {
            "shuffled": np.zeros((NUM_INITIALIZATIONS, VOCAB_SIZE), dtype=np.int64),
            "gaussian": np.zeros((NUM_INITIALIZATIONS, VOCAB_SIZE), dtype=np.int64),
        },
        "condition_nucleus_counts": {
            "shuffled": np.zeros(
                (NUM_INITIALIZATIONS, NUM_REPLICATES, VOCAB_SIZE), dtype=np.int64
            ),
            "gaussian": np.zeros(
                (NUM_INITIALIZATIONS, NUM_REPLICATES, VOCAB_SIZE), dtype=np.int64
            ),
        },
        "uniform_null": {
            "ranked_mean": np.array([0.4, 0.3, 0.2, 0.1, 0.0]),
            "ranked_low": np.array([0.35, 0.25, 0.15, 0.05, 0.0]),
            "ranked_high": np.array([0.45, 0.35, 0.25, 0.15, 0.0]),
        },
    }
    for store in ("condition_greedy_counts", "condition_nucleus_counts"):
        for values in fields[store].values():
            values[..., 1] = 40 if store == "condition_greedy_counts" else 40
    fields.update(overrides)
    return _record(**fields)


def test_available_conditions_lists_real_first() -> None:
    """``real`` is always present and always first."""

    assert _extended().available_conditions == ("real", "shuffled", "gaussian")
    assert _record().available_conditions == ("real",)


def test_the_real_condition_is_not_stored_twice() -> None:
    """It is served from the primary arrays instead."""

    record = _extended()

    assert "real" not in record.condition_greedy_counts
    assert np.array_equal(
        record.condition_policy_fractions("real", "greedy"), record.greedy_fractions
    )
    assert np.array_equal(
        record.condition_policy_fractions("real", "nucleus"), record.nucleus_fractions
    )


def test_duplicating_the_real_condition_is_rejected() -> None:
    """Two sources of truth for one condition is a bug waiting to happen."""

    with pytest.raises(ValueError, match="must not be duplicated"):
        _extended(
            condition_greedy_counts={
                "real": np.zeros((NUM_INITIALIZATIONS, VOCAB_SIZE), dtype=np.int64)
            }
        )


def test_condition_arrays_must_match_the_primary_shapes() -> None:
    """Every condition covers the same initializations and support."""

    with pytest.raises(ValueError, match="every condition must cover"):
        _extended(
            condition_greedy_counts={
                "shuffled": np.zeros((NUM_INITIALIZATIONS + 1, VOCAB_SIZE), dtype=np.int64)
            }
        )


def test_input_conditions_and_the_null_round_trip(tmp_path) -> None:
    """Both additions must survive persistence, or a replot loses them."""

    record = _extended()
    record.save(tmp_path)

    reloaded = load_record(tmp_path)

    assert reloaded.available_conditions == ("real", "shuffled", "gaussian")
    assert np.array_equal(
        reloaded.condition_greedy_counts["gaussian"], record.condition_greedy_counts["gaussian"]
    )
    assert np.allclose(reloaded.uniform_null["ranked_high"], record.uniform_null["ranked_high"])
    assert reloaded.has_input_structure and reloaded.has_uniform_null


def test_a_record_without_the_additions_still_loads(tmp_path) -> None:
    """Version 1 and 2 records predate both and must keep working."""

    _record().save(tmp_path)

    reloaded = load_record(tmp_path)

    assert reloaded.available_conditions == ("real",)
    assert not reloaded.has_input_structure
    assert not reloaded.has_uniform_null


def test_requesting_a_missing_condition_names_what_is_available() -> None:
    """A silent fallback would compare the wrong thing."""

    with pytest.raises(KeyError, match="available"):
        _record().condition_counts("gaussian", "greedy")


def test_nucleus_condition_fractions_average_replicates() -> None:
    """Matching how the real condition is summarised, so pairs are comparable."""

    record = _extended()

    fractions = record.condition_policy_fractions("shuffled", "nucleus")

    assert fractions.shape == (NUM_INITIALIZATIONS, VOCAB_SIZE)
    assert fractions.sum(axis=1).tolist() == pytest.approx([1.0] * NUM_INITIALIZATIONS)
