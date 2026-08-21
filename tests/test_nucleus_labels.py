"""Tests for recovering the sweep's per-position nucleus samples.

The gate is the important part. Reproducing labels means re-running a forward
pass and re-deriving a draw the experiment made earlier, and there are several
ways for that to silently describe a different experiment: a different sampling
stream, a different support mask, a different initialization. The recorded
histogram is the one piece of the original draw that *was* persisted, so it is
the only available check -- and it has to be an exact one.

The gate is pure NumPy and lives in the analysis layer, so these run wherever
the analysis suite runs -- including where PyTorch is not installed. Recovering
the labels needs a model and is exercised on the machine that has one.
"""

from __future__ import annotations

import numpy as np
import pytest

from llm_behavior_lab.analysis.nucleus_clustering import histogram_gate

VOCAB = 10


def _labels_and_counts(rows: list[list[int]]):
    labels = np.asarray(rows, dtype=np.int64)
    counts = np.stack([np.bincount(row, minlength=VOCAB) for row in labels])
    return labels, counts


def test_the_gate_passes_on_an_exact_reproduction() -> None:
    labels, counts = _labels_and_counts([[2, 2, 3, 4], [2, 3, 3, 5]])

    result = histogram_gate(
        labels, counts, vocab_size=VOCAB, temperatures=[0.12, 1.20]
    )

    assert result["passed"] is True
    assert [entry["temperature"] for entry in result["per_temperature"]] == [0.12, 1.20]
    assert result["per_temperature"][0]["num_positions"] == 4
    assert result["per_temperature"][0]["num_distinct_tokens"] == 3


def test_a_single_position_moving_between_tokens_fails_the_gate() -> None:
    """The smallest possible discrepancy must still abort.

    Two positions swapping labels leaves the histogram unchanged and is
    genuinely undetectable here; one position *moving* changes two counts by one
    each, and that is the smallest thing the gate can see. It must not be
    treated as close enough.
    """

    labels, counts = _labels_and_counts([[2, 2, 3, 4]])
    counts = counts.copy()
    counts[0][2] -= 1
    counts[0][3] += 1

    with pytest.raises(ValueError, match="do not reproduce the recorded histogram"):
        histogram_gate(labels, counts, vocab_size=VOCAB, temperatures=[0.6])


def test_the_failure_names_the_temperature_and_the_scale_of_the_disagreement() -> None:
    """A fatal gate still has to be diagnosable."""

    labels, counts = _labels_and_counts([[2, 2, 3], [4, 4, 5]])
    counts = counts.copy()
    counts[1][4] = 0
    counts[1][7] = 2

    with pytest.raises(ValueError) as failure:
        histogram_gate(labels, counts, vocab_size=VOCAB, temperatures=[0.12, 1.20])

    message = str(failure.value)
    assert "T = 1.2" in message
    assert "2 token IDs differ" in message
    assert "largest discrepancy 2" in message


def test_the_gate_checks_every_temperature_not_only_the_first() -> None:
    labels, counts = _labels_and_counts([[2, 2, 3], [4, 4, 5]])
    counts = counts.copy()
    counts[1][5] += 3

    with pytest.raises(ValueError, match="T = 1.2"):
        histogram_gate(labels, counts, vocab_size=VOCAB, temperatures=[0.12, 1.20])


def test_a_histogram_count_mismatch_is_rejected_before_comparing() -> None:
    labels, counts = _labels_and_counts([[2, 2, 3], [4, 4, 5]])

    with pytest.raises(ValueError, match="recovered temperatures"):
        histogram_gate(labels, counts[:1], vocab_size=VOCAB, temperatures=[0.12])


def test_the_gate_is_not_fooled_by_a_matching_total() -> None:
    """Same number of positions, different distribution, still a failure."""

    labels, counts = _labels_and_counts([[2, 2, 2, 3]])
    wrong = np.zeros_like(counts)
    wrong[0][2] = 2
    wrong[0][3] = 2
    assert wrong.sum() == counts.sum()

    with pytest.raises(ValueError):
        histogram_gate(labels, wrong, vocab_size=VOCAB, temperatures=[0.6])
