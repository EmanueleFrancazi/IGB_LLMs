"""Tests for exact per-position parameter-gradient norms.

The observable is easy to state and easy to get subtly wrong, so most of these
tests pin one specific way it could silently degrade:

* differentiating the window-averaged loss instead of one position's loss;
* differentiating the wrong position, or against the wrong target;
* leaving the structural tokens in the softmax denominator, or letting the
  ``-inf`` mask produce ``NaN``;
* covering only part of the parameters;
* mutating the model, its buffers, its ``.grad`` state, or its train/eval mode.

Two of them are hand-checkable rather than self-referential: the per-position
losses must average to the model's own window loss, and the gradient block
belonging to the output projection must equal an outer product whose Frobenius
norm is a product of two vector norms.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from llm_behavior_lab.analysis import (
    InitializationExperimentRecord,
    gradient_guess_table,
    token_gradient_norms,
)
from llm_behavior_lab.evaluation import empirical_token_counts
from llm_behavior_lab.evaluation.guessing import eligible_support_mask
from llm_behavior_lab.evaluation.init_distribution import (
    NucleusSamplingSettings,
    measure_initialization,
)
from llm_behavior_lab.evaluation.init_distribution import build_evaluation_positions
from llm_behavior_lab.evaluation.position_gradients import (
    compute_position_gradient_norms,
    evenly_spaced_indices,
    masked_evaluation_logits,
    single_position_losses,
)
from llm_behavior_lab.models.llama.config import LlamaConfig
from llm_behavior_lab.models.llama.model import LlamaForCausalLM
from llm_behavior_lab.utils import seed_everything

REPO_ROOT = Path(__file__).resolve().parents[1]
VOCAB_SIZE = 32
BLOCK_SIZE = 6
NUM_WINDOWS = 3
#: Stand-ins for the structural tokens of a pretrained vocabulary.
INELIGIBLE = (0, 1)
ELIGIBLE = tuple(index for index in range(VOCAB_SIZE) if index not in INELIGIBLE)
#: A corpus that never emits a structural token, matching the runner's guarantee.
TOKENS = [2 + (index * 7) % (VOCAB_SIZE - 2) for index in range(120)]


def _build_model(seed: int = 4242) -> LlamaForCausalLM:
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
    return build_evaluation_positions(
        TOKENS, block_size=BLOCK_SIZE, num_windows=NUM_WINDOWS
    )


def _measure(model, positions, **kwargs):
    return compute_position_gradient_norms(
        model,
        positions,
        vocab_size=VOCAB_SIZE,
        eligible_token_ids=ELIGIBLE,
        **kwargs,
    )


def _reference_norms(model, positions, window_indices) -> list[float]:
    """Independent implementation: one fresh forward per position, no reuse.

    Deliberately unlike the streamed implementation -- no retained graph, no
    shared forward pass, no preallocated buffers -- so agreement is evidence
    rather than a restatement.
    """

    mask = eligible_support_mask(VOCAB_SIZE, ELIGIBLE)
    parameters = [p for p in model.parameters() if p.requires_grad]
    norms: list[float] = []
    for window in window_indices:
        for offset in range(BLOCK_SIZE):
            logits = masked_evaluation_logits(
                model,
                positions.input_ids[window : window + 1],
                vocab_size=VOCAB_SIZE,
                mask=mask,
            )
            loss = F.cross_entropy(
                logits[0, offset : offset + 1],
                positions.target_ids[window, offset : offset + 1],
            )
            grads = torch.autograd.grad(loss, parameters)
            total = sum(float(g.double().pow(2).sum()) for g in grads)
            norms.append(math.sqrt(total))
    return norms


# -- what is being differentiated -------------------------------------------


def test_single_position_losses_average_to_the_window_loss() -> None:
    """Each ell_d is one position's loss, and their mean is the model's loss.

    This is the property that separates the requested observable from the
    existing seam: ``model(..., targets=...)`` returns the window mean, which is
    *not* what any gradient here is taken of.
    """

    model = _build_model()
    positions = _positions()
    model.eval()

    with torch.no_grad():
        losses = single_position_losses(
            model,
            positions.input_ids,
            positions.target_ids,
            vocab_size=VOCAB_SIZE,
        )
        window_loss = model(
            input_ids=positions.input_ids, targets=positions.target_ids
        ).loss

    assert losses.shape == positions.target_ids.shape
    assert torch.allclose(losses.mean(), window_loss, rtol=1e-6, atol=1e-7)
    # And they are genuinely per-position, not a broadcast constant.
    assert float(losses.std()) > 0.0


def test_loop_losses_match_the_single_position_helper() -> None:
    """The differentiated scalar is the documented ell_d, position by position."""

    model = _build_model()
    positions = _positions()
    result = _measure(model, positions)

    with torch.no_grad():
        expected = single_position_losses(
            model,
            positions.input_ids,
            positions.target_ids,
            vocab_size=VOCAB_SIZE,
            eligible_token_ids=ELIGIBLE,
        ).reshape(-1)

    # A float32 kernel may reduce a one-row and an eighteen-row log-softmax
    # slightly differently, so this is a semantic check, not a bitwise one: a
    # wrong position or target would be off by whole nats, not by an ulp.
    assert torch.allclose(result.losses, expected.double(), rtol=1e-6, atol=1e-9)


def test_eligible_support_defines_the_probability() -> None:
    """The softmax denominator excludes structural tokens, and stays finite."""

    model = _build_model()
    positions = _positions()
    model.eval()

    with torch.no_grad():
        raw = model(input_ids=positions.input_ids).logits
        mask = eligible_support_mask(VOCAB_SIZE, ELIGIBLE)
        masked = raw.masked_fill(~mask, float("-inf"))
        expected = -torch.log_softmax(masked, dim=-1).gather(
            -1, positions.target_ids.unsqueeze(-1)
        ).squeeze(-1)
        actual = single_position_losses(
            model,
            positions.input_ids,
            positions.target_ids,
            vocab_size=VOCAB_SIZE,
            eligible_token_ids=ELIGIBLE,
        )

    assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-7)
    # The unrestricted loss is a different number, so the mask is doing work.
    with torch.no_grad():
        unrestricted = single_position_losses(
            model, positions.input_ids, positions.target_ids, vocab_size=VOCAB_SIZE
        )
    assert not torch.allclose(actual, unrestricted, rtol=1e-9, atol=1e-12)

    result = _measure(model, positions)
    assert bool(torch.isfinite(result.gradient_norms).all())
    assert bool((result.gradient_norms > 0).all())


def test_masked_vocabulary_rows_receive_exactly_zero_gradient() -> None:
    """The -inf mask contributes no gradient rather than a NaN one."""

    model = _build_model()
    positions = _positions()
    mask = eligible_support_mask(VOCAB_SIZE, ELIGIBLE)

    logits = masked_evaluation_logits(
        model, positions.input_ids[:1], vocab_size=VOCAB_SIZE, mask=mask
    )
    loss = F.cross_entropy(logits[0, 2:3], positions.target_ids[0, 2:3])
    (gradient,) = torch.autograd.grad(loss, [model.output.weight])

    assert bool(torch.isfinite(gradient).all())
    for token_id in INELIGIBLE:
        assert bool((gradient[token_id] == 0).all())


def test_ineligible_target_is_rejected() -> None:
    """A target outside the support would make ell_d infinite; fail instead."""

    model = _build_model()
    positions = _positions()
    poisoned = positions.target_ids.clone()
    poisoned[0, 0] = INELIGIBLE[0]

    with pytest.raises(ValueError, match="outside the predictive support"):
        single_position_losses(
            model,
            positions.input_ids,
            poisoned,
            vocab_size=VOCAB_SIZE,
            eligible_token_ids=ELIGIBLE,
        )


# -- is the norm the right norm ---------------------------------------------


def test_output_projection_block_matches_the_analytic_outer_product() -> None:
    """||dL/dW_out||_F == ||p - e_y|| * ||h_d||, computed by hand.

    The gradient of a single-position cross-entropy with respect to the output
    projection is the outer product ``(p - e_y) h_d^T``, whose Frobenius norm
    factorizes. Checking it pins three things at once: the target token, the
    position whose hidden state is used, and the masked softmax.
    """

    model = _build_model()
    positions = _positions()
    mask = eligible_support_mask(VOCAB_SIZE, ELIGIBLE)
    offset = 3

    captured: dict[str, torch.Tensor] = {}
    handle = model.norm.register_forward_hook(
        lambda _module, _inputs, output: captured.__setitem__("hidden", output)
    )
    try:
        logits = masked_evaluation_logits(
            model, positions.input_ids[:1], vocab_size=VOCAB_SIZE, mask=mask
        )
    finally:
        handle.remove()

    target = positions.target_ids[0, offset]
    loss = F.cross_entropy(logits[0, offset : offset + 1], target.reshape(1))
    (gradient,) = torch.autograd.grad(loss, [model.output.weight])

    probabilities = torch.softmax(logits[0, offset].detach(), dim=-1)
    residual = probabilities.clone()
    residual[target] -= 1.0
    hidden = captured["hidden"][0, offset].detach()
    expected = float(residual.norm()) * float(hidden.norm())

    assert float(gradient.norm()) == pytest.approx(expected, rel=1e-5, abs=1e-8)


def test_norms_match_an_independent_reference_implementation() -> None:
    """The streamed measurement equals a fresh-forward-per-position reference."""

    model = _build_model()
    positions = _positions()

    result = _measure(model, positions)
    reference = _reference_norms(model, positions, range(NUM_WINDOWS))

    assert result.num_positions == NUM_WINDOWS * BLOCK_SIZE
    assert torch.allclose(
        result.gradient_norms,
        torch.tensor(reference, dtype=torch.float64),
        rtol=1e-9,
        atol=1e-12,
    )


def test_norm_covers_every_trainable_parameter() -> None:
    """The reported parameter count is the model's own, measured at runtime."""

    model = _build_model()
    result = _measure(model, _positions())

    assert result.parameter_count == model.count_parameters()
    assert result.num_parameter_tensors == len(list(model.parameters()))
    assert result.definition.endswith("with_respect_to_all_trainable_parameters")
    assert result.softmax_support == "eligible"


def test_position_indices_and_targets_are_aligned() -> None:
    """Every entry points back at the evaluation position it came from."""

    model = _build_model()
    positions = _positions()
    result = _measure(model, positions)

    expected_indices = torch.arange(NUM_WINDOWS * BLOCK_SIZE, dtype=torch.long)
    assert torch.equal(result.position_indices, expected_indices)
    assert torch.equal(result.target_ids, positions.target_ids.reshape(-1))
    assert bool((result.greedy_ids.unsqueeze(-1) != torch.tensor(INELIGIBLE)).all())


# -- the model must come out exactly as it went in --------------------------


def _state_snapshot(model):
    return (
        {name: tensor.detach().clone() for name, tensor in model.named_parameters()},
        {name: tensor.detach().clone() for name, tensor in model.named_buffers()},
    )


def _assert_state_unchanged(model, snapshot) -> None:
    parameters, buffers = snapshot
    for name, tensor in model.named_parameters():
        assert torch.equal(tensor, parameters[name]), f"parameter {name} changed"
    for name, tensor in model.named_buffers():
        assert torch.equal(tensor, buffers[name]), f"buffer {name} changed"


@pytest.mark.parametrize("training", [False, True])
def test_parameters_buffers_grads_and_mode_are_preserved(training: bool) -> None:
    """No weight, buffer, gradient, or mode is touched by the measurement."""

    model = _build_model()
    positions = _positions()
    model.train(training)

    # A pre-existing gradient is the sharpest probe: autograd.grad must not
    # accumulate into it, and nothing here may call zero_grad.
    sentinel = model.output.weight
    sentinel.grad = torch.full_like(sentinel, 0.25)
    sentinel_copy = sentinel.grad.clone()
    untouched = model.tok_embeddings.weight
    assert untouched.grad is None

    snapshot = _state_snapshot(model)
    _measure(model, positions)

    _assert_state_unchanged(model, snapshot)
    assert model.training is training
    assert sentinel.grad is not None
    assert torch.equal(sentinel.grad, sentinel_copy)
    assert untouched.grad is None


def test_training_mode_is_restored_after_a_failure(monkeypatch) -> None:
    """An exception mid-measurement must not leave the model in eval mode."""

    model = _build_model()
    positions = _positions()
    model.train(True)

    def explode(*_args, **_kwargs):
        raise RuntimeError("induced failure")

    monkeypatch.setattr(
        "llm_behavior_lab.evaluation.position_gradients.greedy_guess_ids", explode
    )

    with pytest.raises(RuntimeError, match="induced failure"):
        _measure(model, positions)

    assert model.training is True


# -- determinism, isolation, and window selection ---------------------------


def test_windows_are_independent_and_the_measurement_is_deterministic() -> None:
    """Batching windows together cannot change any position's norm."""

    model = _build_model()
    positions = _positions()

    together = _measure(model, positions)
    separate = torch.cat(
        [
            _measure(model, positions, window_indices=[window]).gradient_norms
            for window in range(NUM_WINDOWS)
        ]
    )
    repeated = _measure(model, positions)

    assert torch.equal(together.gradient_norms, separate)
    assert torch.equal(together.gradient_norms, repeated.gradient_norms)
    assert torch.equal(together.greedy_ids, repeated.greedy_ids)


def test_window_subset_is_traceable_to_its_positions() -> None:
    """A subset keeps the flat position indices of the full evaluation grid."""

    model = _build_model()
    positions = _positions()
    subset = _measure(model, positions, num_windows=2)

    assert subset.num_windows == 2
    assert subset.num_positions == 2 * BLOCK_SIZE
    chosen = evenly_spaced_indices(NUM_WINDOWS, 2)
    expected = torch.tensor(
        [window * BLOCK_SIZE + offset for window in chosen for offset in range(BLOCK_SIZE)],
        dtype=torch.long,
    )
    assert torch.equal(subset.position_indices, expected)

    full = _measure(model, positions)
    lookup = {int(index): value for index, value in zip(full.position_indices, full.gradient_norms)}
    for index, value in zip(subset.position_indices, subset.gradient_norms):
        assert lookup[int(index)] == value


def test_evenly_spaced_indices_defaults_to_every_window() -> None:
    """No reduction of the scientific position set happens by default."""

    assert evenly_spaced_indices(5) == (0, 1, 2, 3, 4)
    assert evenly_spaced_indices(5, 5) == (0, 1, 2, 3, 4)
    assert evenly_spaced_indices(5, 1) == (0,)

    for count in range(1, 6):
        chosen = evenly_spaced_indices(5, count)
        assert len(chosen) == count
        assert len(set(chosen)) == count
        assert list(chosen) == sorted(chosen)
        assert chosen[0] == 0
        if count > 1:
            assert chosen[-1] == 4

    with pytest.raises(ValueError):
        evenly_spaced_indices(3, 4)
    with pytest.raises(ValueError):
        evenly_spaced_indices(3, 0)


def test_invalid_window_indices_are_rejected() -> None:
    model = _build_model()
    positions = _positions()

    with pytest.raises(ValueError, match="distinct"):
        _measure(model, positions, window_indices=[0, 0])
    with pytest.raises(ValueError, match=r"\[0, 3\)"):
        _measure(model, positions, window_indices=[NUM_WINDOWS])
    with pytest.raises(ValueError, match="must not be empty"):
        _measure(model, positions, window_indices=[])


# -- alignment with the guessing observables --------------------------------


def _record_with_gradients(model, positions, result, *, forward_batch_size: int):
    """Build a record the way the runner does, so the two agree by construction."""

    measurement = measure_initialization(
        model,
        positions,
        model_seed=1000,
        vocab_size=VOCAB_SIZE,
        sampling=NucleusSamplingSettings(
            temperature=0.6, top_p=0.9, seed=7, num_replicates=1, common_random_numbers=True
        ),
        eligible_token_ids=ELIGIBLE,
        forward_batch_size=forward_batch_size,
    )
    corpus_counts = empirical_token_counts(TOKENS, vocab_size=VOCAB_SIZE).numpy()
    selected = empirical_token_counts(
        positions.target_ids.reshape(-1).tolist(), vocab_size=VOCAB_SIZE
    ).numpy()
    return InitializationExperimentRecord.build(
        corpus_counts=corpus_counts,
        selected_target_counts=selected,
        greedy_counts=measurement.greedy_counts.numpy()[None, :],
        nucleus_counts=measurement.nucleus_counts.numpy()[None, :, :],
        mean_predicted_probabilities=(
            measurement.mean_predicted_probabilities.numpy()[None, :]
        ),
        model_seeds=[1000],
        eligible_token_ids=list(ELIGIBLE),
        metadata={
            "analysis": {
                "num_positions": positions.num_positions,
                "gradient_analysis": result.as_metadata(
                    enabled=True,
                    initialization_index=0,
                    covers_all_positions=(
                        result.num_positions == positions.num_positions
                    ),
                ),
            }
        },
        gradient_position_indices=result.position_indices.numpy(),
        gradient_position_target_ids=result.target_ids.numpy(),
        gradient_position_greedy_ids=result.greedy_ids.numpy(),
        gradient_position_norms=result.gradient_norms.numpy(),
    )


def test_stored_greedy_ids_reproduce_the_full_run_guess_counts() -> None:
    """q_i on all positions equals the experiment's own greedy histogram.

    The ordinary measurement batches several windows per forward pass while the
    gradient path uses one, so this is simultaneously a batch-invariance check on
    the argmax. If it ever fails, an argmax tie flipped with the batch shape --
    that is a finding to report, not a tolerance to loosen.
    """

    model = _build_model()
    positions = _positions()
    result = _measure(model, positions)
    record = _record_with_gradients(model, positions, result, forward_batch_size=2)

    counts = np.bincount(result.greedy_ids.numpy(), minlength=VOCAB_SIZE)
    assert np.array_equal(counts, record.greedy_counts[0])

    table = gradient_guess_table(record)
    assert table["covers_all_positions"] is True
    assert np.allclose(table["greedy_guess_fraction"], record.greedy_fractions[0])


def test_target_counts_reproduce_the_selected_targets() -> None:
    """n_i on all positions equals the experiment's own target histogram."""

    model = _build_model()
    positions = _positions()
    record = _record_with_gradients(
        model, positions, _measure(model, positions), forward_batch_size=2
    )

    summary = token_gradient_norms(record)
    assert np.array_equal(summary.target_count, record.selected_target_counts)
    assert summary.num_positions == positions.num_positions


def test_a_window_subset_does_not_borrow_the_full_run_guesses() -> None:
    """On a subset, q_i must come from the differentiated positions alone."""

    model = _build_model()
    positions = _positions()
    subset = _measure(model, positions, num_windows=1)
    record = _record_with_gradients(model, positions, subset, forward_batch_size=2)

    table = gradient_guess_table(record)
    assert table["covers_all_positions"] is False
    assert table["num_positions"] == BLOCK_SIZE
    expected = np.bincount(subset.greedy_ids.numpy(), minlength=VOCAB_SIZE)
    assert np.array_equal(table["greedy_guess_count"], expected)
    assert int(table["target_occurrence_count"].sum()) == BLOCK_SIZE


# -- runner protocol resolution ---------------------------------------------


def _load_runner():
    """Load the experiment script's protocol resolver without running it."""

    path = REPO_ROOT / "scripts" / "run_initialization_distribution_experiment.py"
    spec = importlib.util.spec_from_file_location("initialization_distribution_script", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module._resolve_protocol


def _args(**overrides) -> argparse.Namespace:
    fields = {
        "num_initializations": None, "num_windows": None, "block_size": None,
        "num_replicates": None, "split": None, "forward_batch_size": None,
        "temperatures": None, "no_temperature_sweep": False,
        "no_uniform_null": False, "no_input_structure": False,
        "gradient_analysis": False, "no_gradient_analysis": False,
        "gradient_windows": None,
    }
    fields.update(overrides)
    return argparse.Namespace(**fields)


def test_gradient_analysis_is_off_unless_asked_for() -> None:
    """It costs a backward pass per position, so it is never a silent default."""

    resolve = _load_runner()

    assert resolve({}, _args())["gradient_analysis_enabled"] is False
    assert (
        resolve({"gradient_analysis": {"enabled": False}}, _args())[
            "gradient_analysis_enabled"
        ]
        is False
    )


def test_gradient_analysis_can_be_enabled_from_either_side() -> None:
    resolve = _load_runner()

    assert resolve({}, _args(gradient_analysis=True))["gradient_analysis_enabled"] is True
    assert (
        resolve({"gradient_analysis": {"enabled": True}}, _args())[
            "gradient_analysis_enabled"
        ]
        is True
    )


def test_the_command_line_can_veto_an_enabled_config() -> None:
    resolve = _load_runner()

    protocol = resolve(
        {"gradient_analysis": {"enabled": True}},
        _args(gradient_analysis=True, no_gradient_analysis=True),
    )
    assert protocol["gradient_analysis_enabled"] is False


def test_gradient_window_subset_resolution() -> None:
    """Defaults to every window; the override is explicit on both sides."""

    resolve = _load_runner()

    assert resolve({}, _args())["gradient_num_windows"] is None
    assert (
        resolve({"gradient_analysis": {"num_windows": 8}}, _args())[
            "gradient_num_windows"
        ]
        == 8
    )
    assert (
        resolve({"gradient_analysis": {"num_windows": 8}}, _args(gradient_windows=3))[
            "gradient_num_windows"
        ]
        == 3
    )
    assert resolve({}, _args())["gradient_initialization_index"] == 0
