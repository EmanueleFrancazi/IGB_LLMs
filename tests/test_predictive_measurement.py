"""Tests for computing the raw predictive-probability diagnostics.

These run the real measurement path, so they need PyTorch. They check the things
only the measurement can get wrong: that the diagnostics describe the same
softmax the rest of the experiment uses, that they agree with an independent
all-at-once reference, and -- most importantly -- that switching them on changes
nothing else. The guessing counts, the sampling draws, and the sweep must be
bit-for-bit what they were, because this is an additional observation and not a
protocol change.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from llm_behavior_lab.evaluation.guessing import (
    apply_support_mask,
    eligible_support_mask,
    greedy_guess_ids,
    restrict_to_support,
)
from llm_behavior_lab.evaluation.init_distribution import (
    NucleusSamplingSettings,
    build_evaluation_positions,
    compute_evaluation_logits,
    measure_initialization,
)
from llm_behavior_lab.models.llama.config import LlamaConfig
from llm_behavior_lab.models.llama.model import LlamaForCausalLM
from llm_behavior_lab.utils import seed_everything

VOCAB_SIZE = 24
INELIGIBLE = (0, 1)
ELIGIBLE = tuple(index for index in range(VOCAB_SIZE) if index not in INELIGIBLE)
BLOCK_SIZE = 5
NUM_WINDOWS = 4
SWEEP = (0.3, 0.6, 1.2)
TOKENS = [2 + (index * 5) % (VOCAB_SIZE - 2) for index in range(90)]
FORWARD_BATCH = 2

# -- numerical contract -------------------------------------------------------
#
# The observable is float32-derived on purpose. The model emits float32 logits
# and ``softmax(logits.float())`` is a float32 vector; only the *accumulation*
# across positions is float64, which is what a sum over tens of thousands of
# positions requires. Converting the stored result to float64 does not make it a
# float64 measurement, and a test must not demand accuracy the quantity never had.
#
# Measured float32 behaviour, from which these numbers are taken:
#
#   float32 eps                                        1.19e-07
#   |sum(p) - 1| for one softmax vector, K = 31997     3e-08 median, 1.4e-07 max
#   relative dp from a 1e-6 logit perturbation         7e-07 median, 5.8e-06 max
#
# A perturbation that size is exactly what different forward batching produces:
# a different matmul reduction order changes the logits in their last bits.
#
#: Two independently computed float32 probability paths. Loose enough to absorb
#: float32 rounding and batching-induced logit differences, and still five orders
#: of magnitude tighter than any wrong rank, token, or temperature would be.
PROBABILITY_RTOL = 1e-5
PROBABILITY_ATOL = 1e-8

#: |sum(p) - 1| over K float32-derived probabilities. About eight times float32
#: eps, and the same contract the record validation applies. A profile that was
#: genuinely unnormalized would miss by far more than this.
NORMALIZATION_ATOL = 1e-6

#: Both sides read from one accumulator, so only float64 summation order differs.
FLOAT64_RTOL = 1e-9


def _model(seed: int = 77) -> LlamaForCausalLM:
    seed_everything(seed)
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=VOCAB_SIZE,
            dim=32,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            multiple_of=16,
            max_batch_size=4,
            max_seq_len=8,
        )
    )


def _positions():
    return build_evaluation_positions(TOKENS, block_size=BLOCK_SIZE, num_windows=NUM_WINDOWS)


def _settings(**overrides) -> NucleusSamplingSettings:
    fields = {
        "temperature": 0.6,
        "top_p": 0.9,
        "seed": 909,
        "num_replicates": 1,
        "common_random_numbers": True,
    }
    fields.update(overrides)
    return NucleusSamplingSettings(**fields)


def _measure(model, positions, **overrides):
    parameters = {
        "model_seed": 1000,
        "vocab_size": VOCAB_SIZE,
        "sampling": _settings(),
        "eligible_token_ids": ELIGIBLE,
        "forward_batch_size": 2,
    }
    parameters.update(overrides)
    return measure_initialization(model, positions, **parameters)


def _reference_logits(model, positions, *, forward_batch_size: int = FORWARD_BATCH):
    """``[W, block, V]`` support-masked logits, batched like the measurement.

    The batch size is matched on purpose. Two forward passes over the same
    windows in differently sized batches use different matmul reduction orders
    and produce logits that differ in their last float32 bits, so comparing a
    measurement against a reference computed at another batch size would test
    forward reproducibility rather than the aggregation being checked. Batching
    invariance is worth testing, and is tested separately below at a tolerance
    that suits it.
    """

    logits = compute_evaluation_logits(
        model, positions, vocab_size=VOCAB_SIZE, forward_batch_size=forward_batch_size
    )
    mask = eligible_support_mask(VOCAB_SIZE, ELIGIBLE)
    return apply_support_mask(restrict_to_support(logits, vocab_size=VOCAB_SIZE).clone(), mask)


def _reference_probabilities(model, positions) -> torch.Tensor:
    """``[D, V]`` raw T=1 probabilities from the same logits, all at once.

    Deliberately unlike the streamed path -- every position materialized first,
    which is only viable at this size -- so agreement is evidence rather than a
    restatement of the same code.
    """

    return torch.softmax(_reference_logits(model, positions).float(), dim=-1).reshape(
        -1, VOCAB_SIZE
    )


# -- the diagnostics describe the right distribution -------------------------


def test_the_profile_matches_an_all_at_once_reference() -> None:
    """Rank within position, then average, computed independently."""

    model, positions = _model(), _positions()
    measurement = _measure(model, positions, collect_probability_statistics=True)

    probabilities = _reference_probabilities(model, positions)
    ordered = torch.sort(probabilities, dim=-1, descending=True).values[:, : len(ELIGIBLE)]
    expected = ordered.double().mean(dim=0)

    assert torch.allclose(
        measurement.ranked_probability_profile,
        expected,
        rtol=PROBABILITY_RTOL,
        atol=PROBABILITY_ATOL,
    )


def test_ranking_happens_within_positions_not_after_averaging() -> None:
    """The averaged-then-ranked profile is a different vector, and is not used.

    On a real model the two are numerically close, so this asserts the exact
    quantity rather than a plausible-looking one.
    """

    model, positions = _model(), _positions()
    measurement = _measure(model, positions, collect_probability_statistics=True)

    probabilities = _reference_probabilities(model, positions)
    average_first = torch.sort(probabilities.mean(dim=0), descending=True).values[: len(ELIGIBLE)]

    assert not torch.allclose(
        measurement.ranked_probability_profile, average_first.double(), rtol=1e-6
    )
    # Ranking first can only concentrate: the per-position maximum averages to at
    # least the maximum of the averages.
    assert measurement.ranked_probability_profile[0] >= average_first[0].double() - 1e-12


def test_the_profile_satisfies_its_invariants() -> None:
    measurement = _measure(_model(), _positions(), collect_probability_statistics=True)
    profile = measurement.ranked_probability_profile

    assert profile.shape == (len(ELIGIBLE),)
    assert torch.all(torch.isfinite(profile))
    assert torch.all(profile >= 0)
    assert torch.all(torch.diff(profile) <= 1e-12)
    assert float(profile.sum()) == pytest.approx(1.0, abs=NORMALIZATION_ATOL)


def test_the_maximum_is_the_probability_of_the_greedy_token() -> None:
    """Argmax identity: the stored maximum belongs to the greedy prediction."""

    model, positions = _model(), _positions()
    measurement = _measure(model, positions, collect_probability_statistics=True)

    masked = _reference_logits(model, positions)
    probabilities = torch.softmax(masked.float(), dim=-1).reshape(-1, VOCAB_SIZE)
    greedy = greedy_guess_ids(masked).reshape(-1)

    # The greedy token really is the argmax of the probability vector.
    assert torch.equal(probabilities.argmax(dim=-1), greedy)
    expected = probabilities.gather(-1, greedy.unsqueeze(-1)).squeeze(-1).double()
    assert torch.allclose(
        measurement.max_probabilities, expected, rtol=PROBABILITY_RTOL, atol=PROBABILITY_ATOL
    )
    assert torch.allclose(
        measurement.max_probabilities,
        probabilities.max(dim=-1).values.double(),
        rtol=PROBABILITY_RTOL,
        atol=PROBABILITY_ATOL,
    )
    assert float(measurement.max_probabilities.min()) >= 0.0
    assert float(measurement.max_probabilities.max()) <= 1.0


def test_rank_one_equals_the_mean_maximum() -> None:
    measurement = _measure(_model(), _positions(), collect_probability_statistics=True)

    # Both sides come from the same accumulator, so only float64 summation order
    # separates them and the tolerance can stay very tight.
    assert float(measurement.ranked_probability_profile[0]) == pytest.approx(
        float(measurement.max_probabilities.mean()), rel=FLOAT64_RTOL
    )


def test_the_target_probability_is_the_true_next_token() -> None:
    model, positions = _model(), _positions()
    measurement = _measure(model, positions, collect_probability_statistics=True)

    probabilities = _reference_probabilities(model, positions)
    targets = positions.target_ids.reshape(-1)
    expected = probabilities.gather(-1, targets.unsqueeze(-1)).squeeze(-1).double()

    assert torch.allclose(
        measurement.target_probabilities,
        expected,
        rtol=PROBABILITY_RTOL,
        atol=PROBABILITY_ATOL,
    )


def test_the_loss_is_the_negative_log_target_probability() -> None:
    measurement = _measure(_model(), _positions(), collect_probability_statistics=True)

    assert torch.allclose(
        measurement.target_losses,
        -torch.log(measurement.target_probabilities),
        rtol=1e-12,
    )


def test_the_targets_are_the_corpus_targets_even_for_a_shuffled_input() -> None:
    """A condition changes the input, never the reference the loss is against."""

    model, positions = _model(), _positions()
    shuffled = positions.input_ids.flip(0)
    measurement = _measure(
        model, positions, input_ids=shuffled, collect_probability_statistics=True
    )

    assert measurement.target_probabilities.shape == (positions.num_positions,)
    assert torch.all(measurement.target_probabilities > 0)


# -- switching it on must change nothing else --------------------------------


@pytest.mark.parametrize("sweep", [(), SWEEP])
def test_collecting_probabilities_does_not_change_any_existing_output(sweep) -> None:
    """Identical greedy, nucleus, sweep and mass, with the sweep off and on.

    This is the guarantee that makes the diagnostic safe to add to a protocol
    that has already produced published numbers.
    """

    positions = _positions()
    without = _measure(_model(), positions, sweep_temperatures=sweep)
    with_statistics = _measure(
        _model(), positions, sweep_temperatures=sweep, collect_probability_statistics=True
    )

    assert torch.equal(without.greedy_counts, with_statistics.greedy_counts)
    assert torch.equal(without.nucleus_counts, with_statistics.nucleus_counts)
    assert torch.equal(
        without.mean_predicted_probabilities, with_statistics.mean_predicted_probabilities
    )
    assert without.num_positions == with_statistics.num_positions
    if sweep:
        assert torch.equal(without.sweep_counts, with_statistics.sweep_counts)
        assert torch.equal(without.sweep_agreement, with_statistics.sweep_agreement)
    else:
        assert with_statistics.sweep_counts is None


@pytest.mark.parametrize("sweep", [(), SWEEP])
def test_the_diagnostics_are_identical_with_and_without_the_sweep(sweep) -> None:
    """The probability statistics are upstream of sampling, so the sweep is irrelevant."""

    positions = _positions()
    baseline = _measure(_model(), positions, collect_probability_statistics=True)
    other = _measure(
        _model(), positions, sweep_temperatures=sweep, collect_probability_statistics=True
    )

    assert torch.equal(baseline.ranked_probability_profile, other.ranked_probability_profile)
    assert torch.equal(baseline.max_probabilities, other.max_probabilities)
    assert torch.equal(baseline.target_probabilities, other.target_probabilities)


def test_the_diagnostics_do_not_disturb_the_sampling_stream() -> None:
    """Nucleus draws must be unchanged: no generator is touched by this path."""

    positions = _positions()
    settings = _settings(num_replicates=3, common_random_numbers=False)
    without = _measure(_model(), positions, sampling=settings)
    with_statistics = _measure(
        _model(), positions, sampling=settings, collect_probability_statistics=True
    )

    assert torch.equal(without.nucleus_counts, with_statistics.nucleus_counts)


def test_the_statistics_are_off_by_default() -> None:
    """Existing callers keep their exact behaviour and cost."""

    measurement = _measure(_model(), _positions())

    assert measurement.ranked_probability_profile is None
    assert measurement.max_probabilities is None
    assert measurement.target_probabilities is None
    assert measurement.target_losses is None


def test_batching_changes_no_discrete_outcome() -> None:
    """The forward batch size is a memory knob, and the *decisions* are exact.

    Everything discrete must be bit-for-bit identical: which token greedy picks,
    how often each is picked, how many positions were measured. Those carry no
    floating-point ambiguity and are asserted with equality.
    """

    positions = _positions()
    small = _measure(
        _model(), positions, forward_batch_size=1, collect_probability_statistics=True
    )
    large = _measure(
        _model(), positions, forward_batch_size=NUM_WINDOWS, collect_probability_statistics=True
    )

    assert torch.equal(small.greedy_counts, large.greedy_counts)
    assert torch.equal(small.nucleus_counts, large.nucleus_counts)
    assert small.num_positions == large.num_positions


def test_batching_moves_probabilities_only_within_float32_noise() -> None:
    """The continuous values agree to float32, which is all they can agree to.

    Different batch sizes give the matmuls different reduction orders, so the
    logits differ in their last float32 bits and the probabilities inherit that.
    Demanding exactness here would be demanding bitwise forward reproducibility,
    which is a different property from the aggregation being correct -- and one
    the hardware does not offer.
    """

    positions = _positions()
    small = _measure(
        _model(), positions, forward_batch_size=1, collect_probability_statistics=True
    )
    large = _measure(
        _model(), positions, forward_batch_size=NUM_WINDOWS, collect_probability_statistics=True
    )

    for name in (
        "ranked_probability_profile",
        "max_probabilities",
        "target_probabilities",
        "temperature_ranked_probabilities",
        "temperature_max_probabilities",
    ):
        assert torch.allclose(
            getattr(small, name),
            getattr(large, name),
            rtol=PROBABILITY_RTOL,
            atol=PROBABILITY_ATOL,
        ), name


def test_ineligible_tokens_receive_no_probability() -> None:
    """The support restriction is what makes the profile span exactly K ranks."""

    model, positions = _model(), _positions()
    measurement = _measure(model, positions, collect_probability_statistics=True)
    probabilities = _reference_probabilities(model, positions)

    assert measurement.ranked_probability_profile.shape[0] == len(ELIGIBLE)
    for token_id in INELIGIBLE:
        assert float(probabilities[:, token_id].abs().max()) == 0.0
    assert np.isclose(
        float(measurement.ranked_probability_profile.sum()), 1.0, atol=NORMALIZATION_ATOL
    )


# -- temperature-conditioned confidence --------------------------------------


def test_the_temperature_grid_holds_greedy_identity_fixed() -> None:
    """argmax is temperature-invariant, so every temperature shares one greedy set."""

    model, positions = _model(), _positions()
    measurement = _measure(model, positions, collect_probability_statistics=True)

    masked = _reference_logits(model, positions)
    greedy = greedy_guess_ids(masked).reshape(-1)
    flat = masked.reshape(-1, VOCAB_SIZE).float()

    for index, temperature in enumerate(measurement.confidence_temperatures):
        probabilities = torch.softmax(flat / temperature, dim=-1)
        # Exact, and it must stay exact: the identity of the greedy token is a
        # discrete outcome, not a float comparison, and its invariance across
        # temperature is the claim the whole experiment rests on.
        assert torch.equal(probabilities.argmax(dim=-1), greedy), temperature
        expected = probabilities.gather(-1, greedy.unsqueeze(-1)).squeeze(-1).double()
        assert torch.allclose(
            measurement.temperature_max_probabilities[index],
            expected,
            rtol=PROBABILITY_RTOL,
            atol=PROBABILITY_ATOL,
        ), temperature
        assert torch.allclose(
            measurement.temperature_max_probabilities[index],
            probabilities.max(dim=-1).values.double(),
            rtol=PROBABILITY_RTOL,
            atol=PROBABILITY_ATOL,
        ), temperature


def test_every_temperature_profile_matches_a_direct_reference() -> None:
    """Gathering through one logits sort equals sorting each temperature directly."""

    model, positions = _model(), _positions()
    measurement = _measure(model, positions, collect_probability_statistics=True)

    flat = _reference_logits(model, positions).reshape(-1, VOCAB_SIZE).float()

    for index, temperature in enumerate(measurement.confidence_temperatures):
        probabilities = torch.softmax(flat / temperature, dim=-1)
        ordered = torch.sort(probabilities, dim=-1, descending=True).values[:, : len(ELIGIBLE)]
        assert torch.allclose(
            measurement.temperature_ranked_probabilities[index],
            ordered.double().mean(dim=0),
            rtol=PROBABILITY_RTOL,
            atol=PROBABILITY_ATOL,
        ), temperature


def test_the_canonical_slice_is_the_canonical_arrays() -> None:
    """T = 1 in the grid and the T = 1 fields are literally the same numbers."""

    measurement = _measure(_model(), _positions(), collect_probability_statistics=True)
    index = measurement.confidence_temperatures.index(1.0)

    assert torch.equal(
        measurement.temperature_ranked_probabilities[index],
        measurement.ranked_probability_profile,
    )
    assert torch.equal(
        measurement.temperature_max_probabilities[index], measurement.max_probabilities
    )
    assert torch.equal(
        measurement.temperature_target_probabilities[index], measurement.target_probabilities
    )
    assert torch.equal(
        measurement.temperature_target_losses[index], measurement.target_losses
    )


def test_lower_temperature_sharpens_the_measured_confidence() -> None:
    """Confidence rises monotonically as temperature falls, at fixed decisions."""

    measurement = _measure(_model(), _positions(), collect_probability_statistics=True)
    order = np.argsort(np.asarray(measurement.confidence_temperatures))
    medians = [
        float(measurement.temperature_max_probabilities[index].median()) for index in order
    ]
    entropy = [float(measurement.temperature_mean_entropy[index]) for index in order]

    assert all(later < earlier for earlier, later in zip(medians, medians[1:]))
    assert all(later > earlier for earlier, later in zip(entropy, entropy[1:]))


@pytest.mark.parametrize("sweep", [(), SWEEP])
def test_the_temperature_grid_is_independent_of_the_nucleus_sweep(sweep) -> None:
    """A confidence diagnostic must not depend on whether sampling ran."""

    positions = _positions()
    baseline = _measure(_model(), positions, collect_probability_statistics=True)
    other = _measure(
        _model(), positions, sweep_temperatures=sweep, collect_probability_statistics=True
    )

    assert torch.equal(
        baseline.temperature_ranked_probabilities, other.temperature_ranked_probabilities
    )
    assert torch.equal(
        baseline.temperature_max_probabilities, other.temperature_max_probabilities
    )
    assert torch.equal(baseline.temperature_mean_entropy, other.temperature_mean_entropy)


def test_the_temperature_grid_leaves_greedy_and_nucleus_counts_alone() -> None:
    positions = _positions()
    without = _measure(_model(), positions, sweep_temperatures=SWEEP)
    with_grid = _measure(
        _model(), positions, sweep_temperatures=SWEEP, collect_probability_statistics=True
    )

    assert torch.equal(without.greedy_counts, with_grid.greedy_counts)
    assert torch.equal(without.nucleus_counts, with_grid.nucleus_counts)
    assert torch.equal(without.sweep_counts, with_grid.sweep_counts)


def test_a_non_positive_confidence_temperature_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        _measure(
            _model(),
            _positions(),
            collect_probability_statistics=True,
            confidence_temperatures=(0.5, 0.0),
        )
