"""Tests for the paired input-structure controls.

The comparison only means anything if exactly one thing differs between
conditions. Most of these tests hold that up: the shuffle preserves the token
multiset and nothing else, the Gaussian bank and the permutation are fixed
across initializations, and the embedding bypass changes the input and no part
of the network.
"""

from __future__ import annotations

import pytest
import torch

from llm_behavior_lab.evaluation.init_distribution import (
    NucleusSamplingSettings,
    build_evaluation_positions,
    measure_initialization,
    sampling_uniforms,
)
from llm_behavior_lab.evaluation.input_conditions import (
    INPUT_CONDITIONS,
    embedding_moments,
    scaled_gaussian_embeddings,
    shuffled_input_ids,
    standardized_gaussian_bank,
)
from llm_behavior_lab.models.llama.config import LlamaConfig
from llm_behavior_lab.models.llama.model import LlamaForCausalLM
from llm_behavior_lab.utils import seed_everything

VOCAB_SIZE = 32
TOKENS = [(index * 7) % VOCAB_SIZE for index in range(400)]


def _model(vocab_size: int = VOCAB_SIZE, dim: int = 32) -> LlamaForCausalLM:
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=vocab_size,
            dim=dim,
            n_layers=1,
            n_heads=2,
            n_kv_heads=1,
            multiple_of=8,
            norm_eps=1e-5,
            max_batch_size=8,
            max_seq_len=32,
        )
    )


def _settings(**overrides) -> NucleusSamplingSettings:
    fields = {"temperature": 0.6, "top_p": 0.9, "seed": 555, "num_replicates": 1}
    fields.update(overrides)
    return NucleusSamplingSettings(**fields)


# -- shuffled condition ---------------------------------------------------


def test_the_shuffle_preserves_the_exact_token_multiset() -> None:
    """The defining property: same tokens, same counts, different order."""

    original = torch.tensor([[1, 2, 3, 4], [5, 5, 6, 7]])

    shuffled = shuffled_input_ids(original, seed=11)

    assert torch.equal(
        torch.sort(original.reshape(-1)).values, torch.sort(shuffled.reshape(-1)).values
    )
    assert torch.bincount(original.reshape(-1), minlength=8).tolist() == torch.bincount(
        shuffled.reshape(-1), minlength=8
    ).tolist()


def test_the_shuffle_preserves_the_window_shape() -> None:
    """Positions, windows, and the positional layout must be untouched."""

    original = torch.arange(24).reshape(6, 4)

    assert shuffled_input_ids(original, seed=3).shape == original.shape


def test_the_shuffle_changes_the_order() -> None:
    """Otherwise the condition would be a copy of the real one."""

    original = torch.arange(64).reshape(8, 8)

    assert not torch.equal(shuffled_input_ids(original, seed=5), original)


def test_the_shuffle_is_deterministic_for_a_seed() -> None:
    """Fixed across initializations, so the control never moves."""

    original = torch.arange(64).reshape(8, 8)

    assert torch.equal(
        shuffled_input_ids(original, seed=5), shuffled_input_ids(original, seed=5)
    )
    assert not torch.equal(
        shuffled_input_ids(original, seed=5), shuffled_input_ids(original, seed=6)
    )


def test_the_shuffle_does_not_mutate_its_input() -> None:
    """The real condition must survive being used to build the shuffled one."""

    original = torch.arange(24).reshape(6, 4)
    before = original.clone()

    shuffled_input_ids(original, seed=9)

    assert torch.equal(original, before)


def test_the_shuffle_is_global_rather_than_within_window() -> None:
    """A within-window shuffle would leave structure between windows.

    With windows that are internally constant, a within-window permutation would
    be a no-op; a global one is not.
    """

    original = torch.tensor([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]])

    assert not torch.equal(shuffled_input_ids(original, seed=17), original)


# -- Gaussian condition ---------------------------------------------------


def test_the_gaussian_bank_has_the_expected_shape() -> None:
    """It must cover the same windows and positions as the real input."""

    bank = standardized_gaussian_bank(num_windows=5, block_size=4, dim=8, seed=1)

    assert bank.shape == (5, 4, 8)


def test_the_gaussian_bank_is_standardized() -> None:
    """Standardized once, rescaled per initialization."""

    bank = standardized_gaussian_bank(num_windows=64, block_size=32, dim=16, seed=2)

    assert float(bank.mean()) == pytest.approx(0.0, abs=0.02)
    assert float(bank.std()) == pytest.approx(1.0, abs=0.02)


def test_the_gaussian_bank_is_fixed_across_initializations() -> None:
    """Redrawing per seed would add a second source of variation."""

    first = standardized_gaussian_bank(num_windows=4, block_size=4, dim=8, seed=7)
    second = standardized_gaussian_bank(num_windows=4, block_size=4, dim=8, seed=7)

    assert torch.equal(first, second)
    assert not torch.equal(
        first, standardized_gaussian_bank(num_windows=4, block_size=4, dim=8, seed=8)
    )


def test_embedding_moments_measure_the_realized_table() -> None:
    """Scale matching follows the weights this initialization actually has."""

    weight = torch.full((10, 4), 3.0)
    weight[0] = 100.0  # a structural row that must be excluded

    moments = embedding_moments(weight, eligible_token_ids=list(range(1, 10)))

    assert moments.mean == pytest.approx(3.0)
    assert moments.std == pytest.approx(0.0, abs=1e-6)
    assert moments.num_rows == 9
    assert "eligible" in moments.rule


def test_embedding_moments_can_use_every_row() -> None:
    """The character case, where no token is structural."""

    moments = embedding_moments(torch.zeros(6, 3))

    assert moments.num_rows == 6
    assert "all embedding rows" in moments.rule


def test_gaussian_scaling_matches_the_embedding_moments() -> None:
    """``G = mu + sigma * Z``, so the control shares the input scale."""

    generator = torch.Generator().manual_seed(3)
    weight = torch.randn(64, 16, generator=generator) * 0.05 + 0.2
    moments = embedding_moments(weight)
    bank = standardized_gaussian_bank(num_windows=32, block_size=16, dim=16, seed=4)

    scaled = scaled_gaussian_embeddings(bank, moments)

    assert float(scaled.mean()) == pytest.approx(moments.mean, abs=0.01)
    assert float(scaled.std()) == pytest.approx(moments.std, rel=0.05)


def test_gaussian_scaling_needs_no_token_ids() -> None:
    """The condition exists precisely to remove discrete token identity."""

    bank = standardized_gaussian_bank(num_windows=2, block_size=3, dim=4, seed=1)

    scaled = scaled_gaussian_embeddings(bank, embedding_moments(torch.zeros(5, 4)))

    assert scaled.shape == (2, 3, 4)
    assert scaled.dtype.is_floating_point


# -- the model embedding seam ---------------------------------------------


def test_the_token_path_is_unchanged() -> None:
    """Existing callers must behave exactly as before."""

    seed_everything(1)
    model = _model()
    tokens = torch.tensor([[1, 2, 3, 4]])

    output = model(input_ids=tokens)

    assert output.logits.shape == (1, 4, VOCAB_SIZE)


def test_the_embedding_path_produces_the_same_shapes() -> None:
    """Downstream semantics are identical; only the input differs."""

    seed_everything(1)
    model = _model()
    embeds = torch.randn(2, 4, 32)

    output = model(inputs_embeds=embeds)

    assert output.logits.shape == (2, 4, VOCAB_SIZE)


def test_the_two_input_forms_are_mutually_exclusive() -> None:
    """Ambiguity here would silently decide which condition was measured."""

    seed_everything(1)
    model = _model()

    with pytest.raises(ValueError, match="exactly one of input_ids or inputs_embeds"):
        model(input_ids=torch.tensor([[1, 2]]), inputs_embeds=torch.randn(1, 2, 32))
    with pytest.raises(ValueError, match="exactly one of input_ids or inputs_embeds"):
        model()


def test_feeding_the_looked_up_embeddings_reproduces_the_token_path() -> None:
    """The bypass really is only the lookup.

    Passing exactly what the embedding table would have produced must give the
    same logits, which is what makes the Gaussian condition comparable.
    """

    seed_everything(2)
    model = _model()
    tokens = torch.tensor([[3, 1, 4, 1]])

    with torch.no_grad():
        from_tokens = model(input_ids=tokens).logits
        from_embeds = model(inputs_embeds=model.tok_embeddings(tokens)).logits

    assert torch.allclose(from_tokens, from_embeds, atol=1e-6)


def test_malformed_embeddings_are_rejected() -> None:
    """Shape errors must fail loudly rather than broadcast."""

    seed_everything(1)
    model = _model()

    with pytest.raises(ValueError, match=r"\[batch, sequence, dim\]"):
        model(inputs_embeds=torch.randn(4, 32))
    with pytest.raises(ValueError, match="does not match the model dimension"):
        model(inputs_embeds=torch.randn(1, 4, 7))


# -- pairing and common random numbers ------------------------------------


def test_the_same_uniforms_are_reused_across_initializations() -> None:
    """Common random numbers: the draw depends on the position alone."""

    settings = _settings()

    first = sampling_uniforms(settings, model_seed=1000, num_positions=64)
    second = sampling_uniforms(settings, model_seed=2000, num_positions=64)

    assert torch.equal(first, second)


def test_the_historical_per_initialization_stream_is_still_available() -> None:
    """Old configurations keep their behaviour."""

    settings = _settings(common_random_numbers=False)

    first = sampling_uniforms(settings, model_seed=1000, num_positions=64)
    second = sampling_uniforms(settings, model_seed=2000, num_positions=64)

    assert not torch.equal(first, second)


def test_one_model_serves_all_three_conditions() -> None:
    """The weights must not change between conditions."""

    seed_everything(5)
    model = _model()
    positions = build_evaluation_positions(TOKENS, block_size=8, num_windows=4)
    before = model.tok_embeddings.weight.clone()

    common = {
        "model_seed": 5,
        "vocab_size": VOCAB_SIZE,
        "sampling": _settings(),
        "forward_batch_size": 2,
    }
    real = measure_initialization(model, positions, **common)
    shuffled = measure_initialization(
        model, positions, input_ids=shuffled_input_ids(positions.input_ids, seed=1), **common
    )
    moments = embedding_moments(model.tok_embeddings.weight)
    bank = standardized_gaussian_bank(num_windows=4, block_size=8, dim=32, seed=2)
    gaussian = measure_initialization(
        model, positions, inputs_embeds=scaled_gaussian_embeddings(bank, moments), **common
    )

    assert torch.equal(model.tok_embeddings.weight, before)
    for measurement in (real, shuffled, gaussian):
        assert measurement.greedy_counts.shape == (VOCAB_SIZE,)
        assert int(measurement.greedy_counts.sum()) == positions.num_positions
        assert int(measurement.nucleus_counts.sum()) == positions.num_positions


def test_conditions_actually_differ() -> None:
    """If they did not, the comparison would be measuring nothing."""

    seed_everything(6)
    model = _model()
    positions = build_evaluation_positions(TOKENS, block_size=8, num_windows=6)
    common = {"model_seed": 6, "vocab_size": VOCAB_SIZE, "sampling": _settings()}

    real = measure_initialization(model, positions, **common)
    gaussian = measure_initialization(
        model,
        positions,
        inputs_embeds=scaled_gaussian_embeddings(
            standardized_gaussian_bank(num_windows=6, block_size=8, dim=32, seed=2),
            embedding_moments(model.tok_embeddings.weight),
        ),
        **common,
    )

    assert not torch.equal(real.greedy_counts, gaussian.greedy_counts)


def test_the_batch_size_does_not_change_a_condition_result() -> None:
    """Common random numbers must survive streaming, for every condition."""

    seed_everything(7)
    model = _model()
    positions = build_evaluation_positions(TOKENS, block_size=8, num_windows=6)
    embeds = scaled_gaussian_embeddings(
        standardized_gaussian_bank(num_windows=6, block_size=8, dim=32, seed=2),
        embedding_moments(model.tok_embeddings.weight),
    )

    results = [
        measure_initialization(
            model,
            positions,
            model_seed=7,
            vocab_size=VOCAB_SIZE,
            sampling=_settings(),
            forward_batch_size=size,
            inputs_embeds=embeds,
        )
        for size in (1, 3, 6)
    ]

    for other in results[1:]:
        assert torch.equal(results[0].greedy_counts, other.greedy_counts)
        assert torch.equal(results[0].nucleus_counts, other.nucleus_counts)


def test_r_equals_one_gives_one_assignment_per_position() -> None:
    """Both policies then summarise the same D assignments."""

    seed_everything(8)
    model = _model()
    positions = build_evaluation_positions(TOKENS, block_size=8, num_windows=4)

    measurement = measure_initialization(
        model, positions, model_seed=8, vocab_size=VOCAB_SIZE, sampling=_settings()
    )

    assert measurement.nucleus_counts.shape == (1, VOCAB_SIZE)
    assert int(measurement.nucleus_counts.sum()) == positions.num_positions
    assert int(measurement.greedy_counts.sum()) == positions.num_positions


def test_a_mismatched_condition_shape_is_rejected() -> None:
    """The conditions must cover the same positions for the draws to align."""

    seed_everything(9)
    model = _model()
    positions = build_evaluation_positions(TOKENS, block_size=8, num_windows=4)

    with pytest.raises(ValueError, match="does not match the evaluation windows"):
        measure_initialization(
            model,
            positions,
            model_seed=9,
            vocab_size=VOCAB_SIZE,
            sampling=_settings(),
            input_ids=torch.zeros(2, 8, dtype=torch.long),
        )


def test_the_canonical_condition_order_is_stable() -> None:
    """Records, figures, and console output all rely on it."""

    assert INPUT_CONDITIONS == ("real", "shuffled", "gaussian")
