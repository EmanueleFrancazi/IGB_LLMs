"""Tests for paired sampling and loss temperatures.

The two temperatures were one word for a long time, and the risk in separating
them is not that the arithmetic breaks -- it is that a sweep silently measures a
different set of pairs than the one that was requested. So these concentrate on
the ways that could happen: a length mismatch quietly broadcast, a default that
is not really a copy, a product where a pairing was meant, and an artifact whose
result arrays have drifted out of step with its temperature arrays.
"""

from __future__ import annotations

import numpy as np
import pytest

from llm_behavior_lab.analysis.temperature_pairs import (
    describe_pairing,
    temperature_pairs,
    validate_pair_aligned,
)


# -- the default: T_g = T_s --------------------------------------------------


def test_omitting_the_loss_temperatures_copies_the_sampling_ones() -> None:
    pairs = temperature_pairs([0.12, 0.24, 0.36])

    assert np.array_equal(pairs["sampling_temperatures"], [0.12, 0.24, 0.36])
    assert np.array_equal(pairs["loss_temperatures"], [0.12, 0.24, 0.36])
    assert pairs["loss_defaulted"] is True
    assert pairs["matched"] is True
    assert pairs["num_pairs"] == 3


def test_the_default_is_a_copy_not_a_shared_array() -> None:
    """Aliasing would let a later edit to one silently change the other."""

    sampling = np.array([0.12, 0.24])
    pairs = temperature_pairs(sampling)
    pairs["loss_temperatures"][0] = 9.0

    assert pairs["sampling_temperatures"][0] == 0.12
    assert sampling[0] == 0.12


# -- explicit pairing --------------------------------------------------------


def test_equal_length_arrays_are_paired_elementwise() -> None:
    pairs = temperature_pairs([0.12, 0.24, 0.36], [1.0, 2.0, 3.0])

    assert list(zip(pairs["sampling_temperatures"], pairs["loss_temperatures"])) == [
        (0.12, 1.0), (0.24, 2.0), (0.36, 3.0),
    ]
    assert pairs["loss_defaulted"] is False
    assert pairs["matched"] is False


def test_the_controlled_figure_22_design_is_expressible() -> None:
    """T_s varying with T_g pinned at 1, which is what figure 22 measured."""

    sampling = [0.12, 0.24, 0.36, 0.48, 0.60, 1.20]
    pairs = temperature_pairs(sampling, [1.0] * 6)

    assert pairs["num_pairs"] == 6
    assert np.all(pairs["loss_temperatures"] == 1.0)
    assert pairs["num_pairs"] == pairs["unique_sampling"].size
    assert pairs["unique_loss"].size == 1
    assert describe_pairing(
        pairs["sampling_temperatures"], pairs["loss_temperatures"]
    ) == "T_g = 1 fixed"


def test_pairing_is_elementwise_and_never_a_product() -> None:
    """Three sampling values against three loss values are three measurements."""

    pairs = temperature_pairs([0.12, 0.24, 0.36], [1.0, 1.0, 2.0])

    assert pairs["num_pairs"] == 3          # not 3 x 2
    assert pairs["unique_sampling"].size == 3
    assert pairs["unique_loss"].size == 2


def test_repeated_values_are_grouped_for_reuse() -> None:
    pairs = temperature_pairs([0.12, 0.12, 0.60], [1.0, 1.0, 1.0])

    assert np.array_equal(pairs["unique_sampling"], [0.12, 0.60])
    assert np.array_equal(pairs["unique_loss"], [1.0])
    # The first two pairs are the same measurement; the third is not.
    keys = list(zip(pairs["sampling_index"], pairs["loss_index"]))
    assert keys[0] == keys[1] != keys[2]


# -- rejections --------------------------------------------------------------


def test_mismatched_lengths_are_rejected_rather_than_broadcast() -> None:
    with pytest.raises(ValueError, match="same length"):
        temperature_pairs([0.12, 0.24, 0.36], [1.0])


def test_a_single_loss_temperature_is_not_silently_broadcast() -> None:
    """The convenient guess is exactly the one that would measure the wrong sweep."""

    with pytest.raises(ValueError):
        temperature_pairs([0.12, 0.24], 1.0)


@pytest.mark.parametrize(
    "sampling,loss",
    [([], None), ([], [1.0]), ([0.12], [])],
)
def test_empty_arrays_are_rejected(sampling, loss) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        temperature_pairs(sampling, loss)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_a_non_positive_sampling_temperature_is_rejected(bad) -> None:
    with pytest.raises(ValueError, match="sampling_temperatures must be positive"):
        temperature_pairs([0.12, bad])


@pytest.mark.parametrize("bad", [0.0, -0.5])
def test_a_non_positive_loss_temperature_is_rejected(bad) -> None:
    with pytest.raises(ValueError, match="loss_temperatures must be positive"):
        temperature_pairs([0.12, 0.24], [1.0, bad])


def test_a_non_finite_temperature_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        temperature_pairs([0.12, np.inf])


# -- what a reader has to check ----------------------------------------------


def test_per_pair_arrays_must_share_the_pair_dimension() -> None:
    sampling = np.array([0.12, 0.24, 0.36])
    loss = np.ones(3)

    assert validate_pair_aligned(sampling, loss, delta=np.zeros(3)) == 3

    with pytest.raises(ValueError, match="'delta' has leading dimension 2"):
        validate_pair_aligned(sampling, loss, delta=np.zeros(2))


def test_temperature_arrays_of_different_lengths_are_caught_on_read() -> None:
    with pytest.raises(ValueError, match="paired elementwise"):
        validate_pair_aligned(np.array([0.12, 0.24]), np.array([1.0]))


def test_the_pairing_description_names_the_three_real_cases() -> None:
    matched = np.array([0.12, 0.60])
    assert describe_pairing(matched, matched) == "T_g = T_s (matched)"
    assert describe_pairing(matched, np.ones(2)) == "T_g = 1 fixed"
    assert describe_pairing(matched, np.array([1.0, 2.0])) == (
        "T_g varies independently of T_s"
    )
