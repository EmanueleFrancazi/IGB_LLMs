"""Tests for per-position gradient persistence and token-level aggregation.

These exercise the read side only, so they need NumPy and nothing else. The
property they mostly defend is the one that is easy to get wrong once a run is
allowed to cover a subset of windows: ``G_i`` and ``q_i`` must describe the same
set of evaluation positions, while ``p_i`` deliberately does not.
"""

from __future__ import annotations

import numpy as np
import pytest

from llm_behavior_lab.analysis import (
    InitializationExperimentRecord,
    gradient_guess_table,
    load_record,
    token_gradient_norms,
)

VOCAB_SIZE = 6
BLOCK_SIZE = 2
NUM_WINDOWS = 3
NUM_POSITIONS = BLOCK_SIZE * NUM_WINDOWS
#: Target token at each of the six evaluation positions.
TARGETS = np.array([2, 3, 2, 4, 3, 2], dtype=np.int64)
#: Greedy prediction at the same six positions.
GREEDY = np.array([5, 5, 4, 5, 2, 5], dtype=np.int64)
NORMS = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)


def _corpus_counts() -> np.ndarray:
    # Token 0 is structural and must never carry corpus mass.
    return np.array([0, 7, 11, 13, 17, 19], dtype=np.int64)


def _build(*, gradient_positions=None, num_initializations: int = 2, **overrides):
    """Assemble a small record, optionally carrying a gradient analysis."""

    selected = np.bincount(TARGETS, minlength=VOCAB_SIZE).astype(np.int64)
    greedy_counts = np.stack(
        [
            np.bincount(GREEDY, minlength=VOCAB_SIZE).astype(np.int64),
            np.array([0, 0, 1, 1, 2, 2], dtype=np.int64),
        ]
    )[:num_initializations]

    gradient_arrays: dict[str, np.ndarray] = {}
    metadata_analysis: dict[str, object] = {"num_positions": NUM_POSITIONS}
    if gradient_positions is not None:
        indices = np.asarray(gradient_positions, dtype=np.int64)
        gradient_arrays = {
            "gradient_position_indices": indices,
            "gradient_position_target_ids": TARGETS[indices],
            "gradient_position_greedy_ids": GREEDY[indices],
            "gradient_position_norms": NORMS[indices],
        }
        metadata_analysis["gradient_analysis"] = {
            "initialization_index": 0,
            "covers_all_positions": len(indices) == NUM_POSITIONS,
            "softmax_support": "eligible",
        }
    gradient_arrays.update(overrides)

    return InitializationExperimentRecord.build(
        corpus_counts=_corpus_counts(),
        selected_target_counts=selected,
        greedy_counts=greedy_counts,
        nucleus_counts=greedy_counts[:, None, :],
        mean_predicted_probabilities=np.full(
            (num_initializations, VOCAB_SIZE), 1.0 / VOCAB_SIZE
        ),
        model_seeds=np.arange(num_initializations) + 1000,
        eligible_token_ids=np.arange(1, VOCAB_SIZE),
        metadata={"analysis": metadata_analysis},
        **gradient_arrays,
    )


# -- persistence -------------------------------------------------------------


def test_record_without_gradients_still_loads(tmp_path) -> None:
    """The arrays are optional: every earlier record keeps working untouched."""

    record = _build()
    assert record.has_position_gradients is False

    record.save(tmp_path)
    reloaded = load_record(tmp_path)

    assert reloaded.has_position_gradients is False
    assert reloaded.gradient_position_norms is None
    with pytest.raises(ValueError, match="no per-position gradient analysis"):
        token_gradient_norms(reloaded)


def test_gradient_arrays_survive_a_round_trip(tmp_path) -> None:
    record = _build(gradient_positions=range(NUM_POSITIONS))
    record.save(tmp_path)
    reloaded = load_record(tmp_path)

    assert reloaded.has_position_gradients is True
    assert np.array_equal(reloaded.gradient_position_target_ids, TARGETS)
    assert np.array_equal(reloaded.gradient_position_greedy_ids, GREEDY)
    assert np.array_equal(reloaded.gradient_position_norms, NORMS)
    assert reloaded.gradient_initialization_index == 0
    assert reloaded.gradient_analysis["softmax_support"] == "eligible"


def test_gradient_arrays_are_all_or_nothing() -> None:
    """A record cannot hold norms without the positions they belong to."""

    with pytest.raises(ValueError, match="all-or-nothing"):
        _build(gradient_position_norms=NORMS)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"gradient_position_norms": NORMS[:-1]}, "same positions"),
        ({"gradient_position_norms": np.full(NUM_POSITIONS, -1.0)}, "non-negative"),
        ({"gradient_position_norms": np.full(NUM_POSITIONS, np.inf)}, "finite"),
        ({"gradient_position_indices": np.zeros(NUM_POSITIONS, dtype=np.int64)}, "distinct"),
        (
            {"gradient_position_target_ids": np.full(NUM_POSITIONS, VOCAB_SIZE)},
            "outside the vocabulary",
        ),
        (
            {"gradient_position_greedy_ids": np.full(NUM_POSITIONS, VOCAB_SIZE)},
            "outside the vocabulary",
        ),
    ],
)
def test_invalid_gradient_arrays_are_rejected(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        _build(gradient_positions=range(NUM_POSITIONS), **overrides)


def test_initialization_index_must_exist() -> None:
    record = _build(gradient_positions=range(NUM_POSITIONS))
    record.metadata["analysis"]["gradient_analysis"]["initialization_index"] = 7
    with pytest.raises(ValueError, match="outside the 2 recorded initializations"):
        record.validate()


# -- aggregation -------------------------------------------------------------


def test_token_gradient_norms_are_the_per_token_means() -> None:
    """G_i and n_i, hand-computed from six positions."""

    summary = token_gradient_norms(_build(gradient_positions=range(NUM_POSITIONS)))

    # token 2 at positions 0, 2, 5 -> (1 + 3 + 6) / 3
    # token 3 at positions 1, 4    -> (2 + 5) / 2
    # token 4 at position 3        -> 4
    assert summary.target_count.tolist() == [0, 0, 3, 2, 1, 0]
    assert summary.mean_gradient_norm[2] == pytest.approx(10.0 / 3.0)
    assert summary.mean_gradient_norm[3] == pytest.approx(3.5)
    assert summary.mean_gradient_norm[4] == pytest.approx(4.0)
    assert np.isnan(summary.mean_gradient_norm[[0, 1, 5]]).all()
    assert int(summary.target_count.sum()) == NUM_POSITIONS
    assert summary.num_positions == NUM_POSITIONS
    assert summary.covers_all_positions is True
    assert summary.measured_token_count == 3


def test_target_counts_match_the_selected_targets_when_all_positions_are_used() -> None:
    """With full coverage, n_i is the experiment's own target histogram."""

    record = _build(gradient_positions=range(NUM_POSITIONS))
    summary = token_gradient_norms(record)

    assert np.array_equal(summary.target_count, record.selected_target_counts)


def test_guess_fractions_match_the_full_run_when_all_positions_are_used() -> None:
    """Position-aligned q_i reproduces greedy_counts exactly at full coverage."""

    record = _build(gradient_positions=range(NUM_POSITIONS))
    table = gradient_guess_table(record)

    assert np.array_equal(table["greedy_guess_count"], record.greedy_counts[0])
    assert np.allclose(table["greedy_guess_fraction"], record.greedy_fractions[0])
    assert table["greedy_guess_fraction"].sum() == pytest.approx(1.0)


def test_guess_fractions_follow_the_gradient_subset_not_the_full_run() -> None:
    """On a subset, q_i must describe S -- the same positions as G_i.

    This is the regression this test exists for: reading the record's full-run
    ``greedy_fractions`` instead would pair a subset ``G_i`` with a whole-run
    ``q_i``, and the two would silently describe different position sets.
    """

    subset = [0, 1, 2, 3]
    record = _build(gradient_positions=subset)
    table = gradient_guess_table(record)

    assert table["covers_all_positions"] is False
    assert table["num_positions"] == len(subset)
    # Positions 0-3 guess 5, 5, 4, 5.
    assert table["greedy_guess_count"].tolist() == [0, 0, 0, 0, 1, 3]
    assert table["greedy_guess_fraction"][5] == pytest.approx(0.75)
    assert not np.allclose(table["greedy_guess_fraction"], record.greedy_fractions[0])
    # G_i likewise covers only the subset: token 2 is now positions 0 and 2.
    assert table["mean_gradient_norm"][2] == pytest.approx(2.0)
    assert table["target_occurrence_count"].tolist() == [0, 0, 2, 1, 1, 0]


def test_corpus_fractions_stay_on_the_whole_split() -> None:
    """p_i is deliberately not restricted to the gradient-evaluated positions."""

    record = _build(gradient_positions=[0, 1])
    table = gradient_guess_table(record)

    assert np.array_equal(table["corpus_fraction"], record.corpus_fractions)
    assert table["corpus_source"] == "whole_analysis_split"
    assert table["greedy_source"] == "gradient_evaluated_positions"


def test_table_columns_align_on_the_token_axis() -> None:
    record = _build(gradient_positions=range(NUM_POSITIONS))
    table = gradient_guess_table(record)

    for column in (
        "token_id",
        "target_occurrence_count",
        "mean_gradient_norm",
        "greedy_guess_count",
        "greedy_guess_fraction",
        "corpus_fraction",
        "eligible_mask",
    ):
        assert table[column].shape == (VOCAB_SIZE,), column
    assert table["token_id"].tolist() == list(range(VOCAB_SIZE))
    assert table["initialization_index"] == 0
