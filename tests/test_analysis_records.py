"""Tests for the persisted initialization-experiment record.

The record is the only durable output of the experiment, so these tests focus on
the two properties a later re-analysis depends on: every vector stays aligned by
token ID, and nothing is lost on the way to disk and back.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from llm_behavior_lab.analysis import InitializationExperimentRecord, load_record

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

    _record().save(tmp_path)

    payload = json.loads((tmp_path / "initialization_distribution.json").read_text(encoding="utf-8"))

    assert payload["num_positions"] == 40
    assert payload["record_version"] >= 1


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
