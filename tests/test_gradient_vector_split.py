"""Tests for the T = 1 correct-vs-wrong parameter-gradient vector split.

WRITTEN BUT NOT EXECUTED LOCALLY. Every case in this module needs PyTorch,
which is not installed in the development workspace, so these have never been
run here. They are written to be run on the cluster.

What the diagnostic exists for: the norm-mass split says how much gradient
magnitude each group generates, but the first optimizer step follows
``g_correct + g_wrong``, and ``sum_d ||g_d||`` is not ``||sum_d g_d||``. The gap
between them is cancellation, which is precisely what the cosine measures.

The reference test recomputes both aggregate vectors directly -- one
``autograd.grad`` per position, summed into two plain lists -- and compares them
elementwise with the streaming accumulator. That is an independent
implementation of the same definition, not a restatement of it.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
import torch.nn.functional as F

from llm_behavior_lab.evaluation.init_distribution import build_evaluation_positions
from llm_behavior_lab.evaluation.position_gradients import (
    CANONICAL_GRADIENT_TEMPERATURE,
    compute_position_gradient_norms,
)

VOCAB = 12
BLOCK = 4
WINDOWS = 3
D = WINDOWS * BLOCK

#: A deterministic token stream, long enough for three spread windows plus the
#: one-token lookahead the targets need.
TOKENS = [(index * 7 + 3) % VOCAB for index in range(64)]


class TinyModel(torch.nn.Module):
    """A deterministic two-layer model small enough to differentiate by hand."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.embedding = torch.nn.Embedding(VOCAB, 8)
        self.output = torch.nn.Linear(8, VOCAB, bias=False)
        with torch.no_grad():
            self.embedding.weight.copy_(
                torch.randn(VOCAB, 8, generator=generator, dtype=torch.float64)
            )
            self.output.weight.copy_(
                torch.randn(VOCAB, 8, generator=generator, dtype=torch.float64)
            )
        self.double()

    def forward(self, input_ids: torch.Tensor):
        hidden = self.embedding(input_ids)
        logits = self.output(hidden)
        return type("Output", (), {"logits": logits})()


def _positions():
    """The shared windows, built the way the experiment builds them."""

    return build_evaluation_positions(
        TOKENS, block_size=BLOCK, num_windows=WINDOWS
    )


def _greedy_ids(model, positions) -> torch.Tensor:
    """``[WINDOWS, BLOCK]`` argmax predictions of the untrained model."""

    with torch.no_grad():
        return torch.stack(
            [
                model(positions.input_ids[window : window + 1]).logits[0].argmax(dim=-1)
                for window in range(WINDOWS)
            ]
        )


def _positions_with_correct(model, num_correct: int):
    """Windows whose targets give a known correct/wrong split.

    The real corpus targets of a random model are almost all wrong, which would
    leave ``g_correct`` empty and the interesting half of the comparison
    untested. Only the *targets* are overridden here -- the windows, starts and
    block size stay exactly as the experiment built them -- so the accumulator
    still sees a well-formed ``EvaluationPositions`` and the split it partitions
    on is deterministic rather than left to chance.
    """

    positions = _positions()
    greedy = _greedy_ids(model, positions)
    with torch.no_grad():
        argmin = torch.stack(
            [
                model(positions.input_ids[window : window + 1]).logits[0].argmin(dim=-1)
                for window in range(WINDOWS)
            ]
        )

    # First `num_correct` flat positions agree with greedy; the rest disagree.
    targets = argmin.clone()
    flat = targets.reshape(-1)
    flat_greedy = greedy.reshape(-1)
    flat[:num_correct] = flat_greedy[:num_correct]
    assert int((flat == flat_greedy).sum()) == num_correct
    return dataclasses.replace(positions, target_ids=flat.reshape(WINDOWS, BLOCK))


def _reference_vectors(model, positions):
    """Sum g_d directly into two buffers, independently of the accumulator."""

    parameters = [p for p in model.parameters() if p.requires_grad]
    correct = [torch.zeros_like(p, dtype=torch.float64) for p in parameters]
    wrong = [torch.zeros_like(p, dtype=torch.float64) for p in parameters]
    counts = [0, 0]

    for window in range(positions.input_ids.shape[0]):
        for offset in range(positions.input_ids.shape[1]):
            logits = model(positions.input_ids[window : window + 1]).logits[0, offset]
            target = positions.target_ids[window, offset]
            loss = F.cross_entropy(logits[None, :] / CANONICAL_GRADIENT_TEMPERATURE,
                                   target[None])
            grads = torch.autograd.grad(loss, parameters, allow_unused=True)
            is_correct = int(logits.argmax()) == int(target)
            buffers = correct if is_correct else wrong
            for buffer, gradient in zip(buffers, grads):
                buffer.add_(gradient.detach().double())
            counts[0 if is_correct else 1] += 1
    return correct, wrong, counts


def _flat(vectors) -> torch.Tensor:
    return torch.cat([vector.reshape(-1) for vector in vectors])


def test_accumulated_vectors_match_a_direct_recomputation() -> None:
    """The streaming split must equal an independent per-position summation."""

    model = TinyModel()
    positions = _positions_with_correct(model, num_correct=5)

    result = compute_position_gradient_norms(
        model, positions, vocab_size=VOCAB, vector_split=True
    )
    correct, wrong, counts = _reference_vectors(model, positions)

    split = result.vector_split
    assert split is not None
    assert [split["num_correct"], split["num_wrong"]] == counts

    reference_correct = _flat(correct)
    reference_wrong = _flat(wrong)
    assert split["norm_correct"] == pytest.approx(
        float(reference_correct.norm()), rel=1e-12
    )
    assert split["norm_wrong"] == pytest.approx(float(reference_wrong.norm()), rel=1e-12)
    assert split["norm_total"] == pytest.approx(
        float((reference_correct + reference_wrong).norm()), rel=1e-12
    )
    assert split["dot"] == pytest.approx(
        float(reference_correct @ reference_wrong), rel=1e-12
    )


def test_counts_partition_the_evaluated_positions() -> None:
    model = TinyModel()
    positions = _positions_with_correct(model, num_correct=5)
    result = compute_position_gradient_norms(
        model, positions, vocab_size=VOCAB, vector_split=True
    )
    split = result.vector_split

    assert split["num_correct"] == 5
    assert split["num_wrong"] == D - 5
    assert split["num_correct"] + split["num_wrong"] == D
    assert split["num_correct"] + split["num_wrong"] == result.num_positions


def test_the_triangle_inequality_holds() -> None:
    model = TinyModel()
    split = compute_position_gradient_norms(
        model, _positions_with_correct(model, num_correct=5),
        vocab_size=VOCAB, vector_split=True
    ).vector_split

    assert split["norm_total"] <= split["norm_correct"] + split["norm_wrong"] + 1e-9
    # And the reverse triangle inequality, which catches a sign error.
    assert split["norm_total"] >= abs(split["norm_correct"] - split["norm_wrong"]) - 1e-9


def test_the_cosine_is_consistent_with_the_norms_and_dot() -> None:
    model = TinyModel()
    split = compute_position_gradient_norms(
        model, _positions_with_correct(model, num_correct=5),
        vocab_size=VOCAB, vector_split=True
    ).vector_split

    assert split["cosine"] is not None
    if split["cosine"] is not None:
        expected = split["dot"] / (split["norm_correct"] * split["norm_wrong"])
        assert split["cosine"] == pytest.approx(expected, rel=1e-12)
        assert -1.0 - 1e-9 <= split["cosine"] <= 1.0 + 1e-9
        # ||a+b||^2 = ||a||^2 + 2 a.b + ||b||^2 ties the four scalars together.
        assert split["norm_total"] ** 2 == pytest.approx(
            split["norm_correct"] ** 2
            + 2.0 * split["dot"]
            + split["norm_wrong"] ** 2,
            rel=1e-9,
        )


def test_the_cosine_is_undefined_rather_than_zero_when_a_group_is_empty() -> None:
    """No direction means no angle; 0.0 would assert unmeasured orthogonality."""

    model = TinyModel()
    positions = _positions_with_correct(model, num_correct=0)

    split = compute_position_gradient_norms(
        model, positions, vocab_size=VOCAB, vector_split=True
    ).vector_split

    assert split["num_correct"] == 0
    assert split["norm_correct"] == pytest.approx(0.0)
    assert split["cosine"] is None


def test_the_diagnostic_is_off_by_default() -> None:
    """Existing callers must keep their exact behaviour and cost."""

    result = compute_position_gradient_norms(TinyModel(), _positions(), vocab_size=VOCAB)

    assert result.vector_split is None
    assert "vector_split" not in result.as_metadata()


def test_enabling_the_split_does_not_change_the_norms() -> None:
    """The scalar observable must be bitwise identical with the flag on."""

    model = TinyModel()
    positions = _positions()

    without = compute_position_gradient_norms(model, positions, vocab_size=VOCAB)
    with_split = compute_position_gradient_norms(
        model, positions, vocab_size=VOCAB, vector_split=True
    )

    assert torch.equal(without.gradient_norms, with_split.gradient_norms)
    assert torch.equal(
        without.temperature_gradient_norms, with_split.temperature_gradient_norms
    )
    assert torch.equal(without.greedy_ids, with_split.greedy_ids)
    assert torch.equal(without.target_ids, with_split.target_ids)


def test_only_the_canonical_temperature_is_accumulated() -> None:
    """One buffer pair, not one per temperature."""

    model = TinyModel()
    positions = _positions_with_correct(model, num_correct=5)
    grid = (0.5, CANONICAL_GRADIENT_TEMPERATURE, 2.0)

    split = compute_position_gradient_norms(
        model, positions, vocab_size=VOCAB, temperatures=grid, vector_split=True
    ).vector_split

    assert split["temperature"] == CANONICAL_GRADIENT_TEMPERATURE
    # Accumulating every temperature would triple the counts here.
    assert split["num_correct"] + split["num_wrong"] == D


def test_the_metadata_block_carries_every_reported_field() -> None:
    model = TinyModel()
    payload = compute_position_gradient_norms(
        model, _positions_with_correct(model, num_correct=5),
        vocab_size=VOCAB, vector_split=True
    ).as_metadata()

    assert set(payload["vector_split"]) == {
        "temperature", "norm_correct", "norm_wrong", "norm_total",
        "dot", "cosine", "num_correct", "num_wrong",
    }


# -- model state and RNG, extending the existing sentinel contract ------------


def test_the_split_leaves_model_state_and_rng_untouched() -> None:
    """Parameters, buffers, an existing .grad sentinel, mode and RNG all survive."""

    model = TinyModel()
    positions = _positions()

    parameters = {name: p.detach().clone() for name, p in model.named_parameters()}
    buffers = {name: b.detach().clone() for name, b in model.named_buffers()}

    # A non-zero sentinel: autograd.grad must not touch .grad at all.
    sentinel = torch.full_like(model.output.weight, 3.5)
    model.output.weight.grad = sentinel.clone()
    assert model.embedding.weight.grad is None

    model.train()
    rng_state = torch.get_rng_state()

    compute_position_gradient_norms(
        model, positions, vocab_size=VOCAB, vector_split=True
    )

    for name, parameter in model.named_parameters():
        assert torch.equal(parameter.detach(), parameters[name]), name
    for name, buffer in model.named_buffers():
        assert torch.equal(buffer.detach(), buffers[name]), name
    assert torch.equal(model.output.weight.grad, sentinel)
    assert model.embedding.weight.grad is None
    assert model.training is True
    assert torch.equal(torch.get_rng_state(), rng_state)


def test_the_entry_mode_is_restored_from_eval_too() -> None:
    model = TinyModel()
    model.eval()

    compute_position_gradient_norms(
        model, _positions(), vocab_size=VOCAB, vector_split=True
    )

    assert model.training is False


# -- additive backward compatibility of the new CLI option -------------------


def _runner_module():
    """Import the experiment runner by path, as the other runner tests do."""

    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "runner_for_vector_split_test",
        root / "scripts" / "run_initialization_distribution_experiment.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_namespace_predating_the_option_still_resolves() -> None:
    """The option is additive: older callers must not become AttributeErrors.

    A namespace built before ``--gradient-vector-split`` existed simply has no
    such attribute. Reading it directly turned every such caller into an
    AttributeError that had nothing to do with what it was doing, which is the
    regression this guards.
    """

    import argparse

    module = _runner_module()
    parser_namespace = argparse.Namespace(
        num_initializations=None, num_windows=None, block_size=None,
        num_replicates=None, split=None, forward_batch_size=None,
        temperatures=None, no_temperature_sweep=False,
        no_uniform_null=False, no_input_structure=False,
        gradient_analysis=False, no_gradient_analysis=False,
        gradient_windows=None,
    )
    assert not hasattr(parser_namespace, "gradient_vector_split")

    protocol = module._resolve_protocol({}, parser_namespace)

    assert protocol["gradient_vector_split"] is False


def test_the_option_can_still_be_enabled_from_the_cli_and_the_config() -> None:
    """The compatibility shim must not disable the feature itself."""

    import argparse

    module = _runner_module()

    def namespace(**overrides):
        fields = dict(
            num_initializations=None, num_windows=None, block_size=None,
            num_replicates=None, split=None, forward_batch_size=None,
            temperatures=None, no_temperature_sweep=False,
            no_uniform_null=False, no_input_structure=False,
            gradient_analysis=False, no_gradient_analysis=False,
            gradient_windows=None,
        )
        fields.update(overrides)
        return argparse.Namespace(**fields)

    from_cli = module._resolve_protocol({}, namespace(gradient_vector_split=True))
    assert from_cli["gradient_vector_split"] is True

    from_config = module._resolve_protocol(
        {"gradient_analysis": {"vector_split": True}}, namespace()
    )
    assert from_config["gradient_vector_split"] is True

    off_by_default = module._resolve_protocol(
        {}, namespace(gradient_vector_split=False)
    )
    assert off_by_default["gradient_vector_split"] is False
