"""Tests for the temperature-resolved directional field.

Two things have to hold. The **canonical slice** must be the canonical field --
not a second measurement of it that happens to agree -- because a directional
answer at ``T = 1`` must not depend on which array was read. And a record that
never measured ``T_g != 1`` must **fail** when asked for one, rather than
quietly handing back the canonical directions: temperature enters the loss, so
``g(T)`` is the gradient of a different objective, and substituting ``g(1)``
would answer a different question while looking like a result.

NumPy only. Producing the sketches needs PyTorch and is covered in
``tests/test_gradient_sketch.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from llm_behavior_lab.analysis import InitializationExperimentRecord
from llm_behavior_lab.analysis.directional_fields import (
    available_loss_temperatures,
    directional_field,
    loss_temperature_index,
)

VOCAB = 12
ELIGIBLE = np.arange(2, VOCAB)
K = 16
TEMPERATURES = (0.12, 0.60, 1.0, 1.20)


def _record(
    num_positions: int = 20,
    *,
    temperature_sketches: bool = True,
    seed: int = 4,
):
    """A record with, or without, the temperature-resolved sketch field."""

    generator = np.random.default_rng(seed)
    targets = generator.choice(ELIGIBLE, size=num_positions)
    greedy = generator.choice(ELIGIBLE, size=num_positions)
    corpus = np.zeros(VOCAB, dtype=np.int64)
    corpus[ELIGIBLE] = 9

    # Deliberately different per temperature, so reading the wrong row shows up.
    per_temperature = np.stack([
        generator.normal(size=(num_positions, K)) + 10.0 * index
        for index in range(len(TEMPERATURES))
    ]).astype(np.float32)
    norms = np.stack([
        np.full(num_positions, 1.0 + index) for index in range(len(TEMPERATURES))
    ])
    canonical = TEMPERATURES.index(1.0)

    return InitializationExperimentRecord.build(
        corpus_counts=corpus,
        selected_target_counts=np.bincount(targets, minlength=VOCAB),
        greedy_counts=np.bincount(greedy, minlength=VOCAB)[None, :],
        nucleus_counts=np.bincount(greedy, minlength=VOCAB)[None, None, :],
        mean_predicted_probabilities=np.full((1, VOCAB), 1.0 / VOCAB),
        model_seeds=[1000],
        eligible_token_ids=ELIGIBLE,
        metadata={
            "num_positions": num_positions,
            "analysis": {
                "num_positions": num_positions,
                "gradient_analysis": {
                    "enabled": True,
                    "initialization_index": 0,
                    "covers_all_positions": True,
                    "gradient_sketch": {"dimension": K, "seed": 1},
                },
            },
        },
        gradient_position_indices=np.arange(num_positions),
        gradient_position_target_ids=targets.astype(np.int64),
        gradient_position_greedy_ids=greedy.astype(np.int64),
        gradient_position_norms=norms[canonical],
        gradient_position_sketches=per_temperature[canonical],
        gradient_temperatures=np.asarray(TEMPERATURES),
        gradient_temperature_position_norms=norms,
        gradient_temperature_position_sketches=(
            per_temperature if temperature_sketches else None
        ),
    )


# -- shape and alignment -----------------------------------------------------


def test_the_field_spans_temperatures_positions_and_sketch_width() -> None:
    record = _record(num_positions=20)

    sketches = record.gradient_temperature_position_sketches

    assert sketches.shape == (len(TEMPERATURES), 20, K)
    assert record.gradient_temperature_position_norms.shape == (len(TEMPERATURES), 20)
    assert record.gradient_temperatures.shape == (len(TEMPERATURES),)
    assert record.has_temperature_gradient_sketches


def test_each_temperature_row_pairs_its_own_sketches_with_its_own_norms() -> None:
    record = _record()

    for index, temperature in enumerate(TEMPERATURES):
        field = directional_field(record, temperature)
        assert field["index"] == index
        assert field["loss_temperature"] == temperature
        assert np.array_equal(
            field["sketches"], record.gradient_temperature_position_sketches[index]
        )
        assert np.array_equal(
            field["norms"], record.gradient_temperature_position_norms[index]
        )


# -- the canonical slice IS the canonical field ------------------------------


def test_the_canonical_slice_equals_the_canonical_fields() -> None:
    record = _record()

    field = directional_field(record, 1.0)

    assert np.array_equal(field["sketches"], record.gradient_position_sketches)
    assert np.array_equal(field["norms"], record.gradient_position_norms)


def test_a_canonical_slice_that_disagrees_is_rejected_at_construction() -> None:
    """Two independently measured T = 1 fields would be a coincidence to check.

    Building one is refused outright, so nothing downstream has to wonder which
    of two disagreeing canonical arrays it should have believed.
    """

    record = _record()
    sketches = np.array(record.gradient_temperature_position_sketches, copy=True)
    canonical = TEMPERATURES.index(1.0)
    sketches[canonical, 0, 0] += 1.0

    with pytest.raises(ValueError, match="canonical row"):
        InitializationExperimentRecord(
            **{
                **record.__dict__,
                "gradient_temperature_position_sketches": sketches,
            }
        )


def test_the_temperature_field_requires_the_canonical_one() -> None:
    record = _record()

    with pytest.raises(ValueError, match="without gradient_position_sketches"):
        InitializationExperimentRecord(
            **{**record.__dict__, "gradient_position_sketches": None}
        )


def test_a_misshapen_temperature_field_is_rejected() -> None:
    record = _record()
    truncated = np.asarray(record.gradient_temperature_position_sketches)[:2]

    with pytest.raises(ValueError, match=r"\[temperatures, positions, K\]"):
        InitializationExperimentRecord(
            **{**record.__dict__, "gradient_temperature_position_sketches": truncated}
        )


# -- records from before the field existed -----------------------------------


def test_a_record_without_the_field_still_loads_and_serves_the_canonical_one() -> None:
    record = _record(temperature_sketches=False)

    assert not record.has_temperature_gradient_sketches
    assert record.has_gradient_position_sketches
    assert available_loss_temperatures(record) == (1.0,)

    field = directional_field(record, 1.0)

    assert field["source"] == "canonical"
    assert np.array_equal(field["sketches"], record.gradient_position_sketches)


def test_a_record_without_the_field_refuses_any_other_loss_temperature() -> None:
    """No silent fallback: this is the failure mode worth being loud about."""

    record = _record(temperature_sketches=False)

    with pytest.raises(ValueError, match="No measured gradient-direction field"):
        directional_field(record, 0.12)


def test_the_refusal_lists_what_was_measured() -> None:
    record = _record()

    with pytest.raises(ValueError) as failure:
        directional_field(record, 0.36)

    message = str(failure.value)
    assert "T_g = 0.36" in message
    assert "0.12, 0.6, 1, 1.2" in message
    assert "cannot fall back" in message


def test_a_record_with_no_sketches_at_all_says_so() -> None:
    record = _record()
    stripped = InitializationExperimentRecord(
        **{
            **record.__dict__,
            "gradient_position_sketches": None,
            "gradient_temperature_position_sketches": None,
        }
    )

    assert available_loss_temperatures(stripped) == ()
    with pytest.raises(ValueError, match="no gradient direction information"):
        directional_field(stripped, 1.0)


# -- matching -----------------------------------------------------------------


def test_a_float_round_trip_still_matches_its_measured_temperature() -> None:
    record = _record()

    index = loss_temperature_index(record, float(np.float32(0.12)))

    assert index == 0


def test_the_tolerance_cannot_alias_two_grid_temperatures() -> None:
    """0.12 apart is the grid step; the tolerance is far below it."""

    from llm_behavior_lab.analysis.directional_fields import (
        TEMPERATURE_MATCH_TOLERANCE,
    )

    assert TEMPERATURE_MATCH_TOLERANCE < 0.12 / 1000.0
    record = _record()
    with pytest.raises(ValueError, match="No measured gradient-direction field"):
        directional_field(record, 0.12 + 1e-3)


def test_no_sketch_norm_normalization_is_introduced() -> None:
    """The denominator stays the exact full-parameter norm.

    The fixture's sketch rows have norms far from 1, so a field that had started
    dividing by the projected norm would show up immediately.
    """

    record = _record()

    field = directional_field(record, 0.60)

    index = TEMPERATURES.index(0.60)
    assert np.array_equal(
        field["norms"], record.gradient_temperature_position_norms[index]
    )
    projected = np.linalg.norm(field["sketches"], axis=1)
    assert not np.allclose(field["norms"], projected)
