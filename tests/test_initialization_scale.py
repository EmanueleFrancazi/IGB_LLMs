"""Tests for the paired initialization-scale intervention.

The intervention is one multiplier on the audited zero-centred random weights.
What makes it a *controlled* experiment is the pairing: the three conditions come
from one draw, so signs, directions and relative structure are shared and only
magnitude differs. Reseeding per scale would give three unrelated models and
answer a different question, so most of these tests pin the pairing rather than
the arithmetic.

Nothing here asserts what lower scale does to the model's behaviour. That is the
empirical question the experiment exists to answer.
"""

from __future__ import annotations

import pytest
import torch

from llm_behavior_lab.models.initialization_scale import (
    DETERMINISTIC_PARAMETER_SUFFIXES,
    classify_parameters,
    initialization_scale_report,
    scale_initialization,
)
from llm_behavior_lab.models.llama.config import LlamaConfig
from llm_behavior_lab.models.llama.model import LlamaForCausalLM
from llm_behavior_lab.utils import seed_everything

SEED = 1000


def _model(seed: int = SEED) -> LlamaForCausalLM:
    """A model built exactly as the runner builds one."""

    seed_everything(seed)
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            dim=32,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            multiple_of=16,
            max_batch_size=4,
            max_seq_len=8,
        )
    )


def _snapshot(model) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in model.named_parameters()}


# -- the audit, asserted rather than assumed ---------------------------------


def test_the_architecture_has_no_bias_parameters() -> None:
    """Every linear is bias=False, so 'biases at zero' matches all history.

    If a bias were ever added, the scale experiment would silently stop
    reproducing the historical baseline at alpha = 1, so this is checked rather
    than remembered.
    """

    model = _model()
    biases = [name for name, _ in model.named_parameters() if name.endswith("bias")]

    assert biases == []


def test_only_the_normalization_gains_are_left_unscaled() -> None:
    """The deterministic parameters are exactly the RMSNorm gains."""

    model = _model()
    classified = classify_parameters(model)
    unscaled = sorted(name for name, scaled in classified.items() if not scaled)

    assert unscaled
    assert all(
        any(name.endswith(suffix) for suffix in DETERMINISTIC_PARAMETER_SUFFIXES)
        for name in unscaled
    )
    for name in unscaled:
        assert torch.equal(
            dict(model.named_parameters())[name], torch.ones_like(
                dict(model.named_parameters())[name]
            )
        )


def test_a_new_tensor_would_be_scaled_by_default() -> None:
    """Classification excludes deliberately, so an addition is not missed.

    A newly added random tensor left unscaled would break the pairing with no
    visible symptom; defaulting to scaled fails in the safer direction.
    """

    model = _model()
    classified = classify_parameters(model)

    assert classified["tok_embeddings.weight"] is True
    assert classified["output.weight"] is True
    assert classified["layers.0.attention.wq.weight"] is True
    assert classified["norm.weight"] is False


# -- the intervention --------------------------------------------------------


def test_alpha_one_is_a_literal_no_op() -> None:
    """Not a multiply by 1.0: the tensors are not touched at all."""

    model = _model()
    before = _snapshot(model)

    report = scale_initialization(model, 1.0)

    assert report["is_no_op"] is True
    for name, tensor in model.named_parameters():
        assert torch.equal(tensor, before[name]), name


@pytest.mark.parametrize("alpha", [0.5, 0.25])
def test_scaled_weights_are_exactly_alpha_times_the_paired_baseline(alpha) -> None:
    """The same draw, one multiply: pairing holds tensor by tensor."""

    baseline = _snapshot(_model())
    scaled = _model()
    scale_initialization(scaled, alpha)

    classified = classify_parameters(scaled)
    for name, tensor in scaled.named_parameters():
        if classified[name]:
            assert torch.equal(tensor, baseline[name] * alpha), name
        else:
            assert torch.equal(tensor, baseline[name]), name


@pytest.mark.parametrize("alpha", [0.5, 0.25])
def test_signs_and_directions_are_shared_across_scales(alpha) -> None:
    """Only magnitude differs: every sign matches the baseline's."""

    baseline = _snapshot(_model())
    scaled = _model()
    scale_initialization(scaled, alpha)

    for name, tensor in scaled.named_parameters():
        assert torch.equal(torch.sign(tensor), torch.sign(baseline[name])), name


def test_each_scale_derives_from_a_pristine_draw_not_cumulatively() -> None:
    """0.25 must be one step from the baseline, never 0.5 applied twice.

    Both routes reach the same numbers here, but only the independent one stays
    correct if the multiplier is ever changed, so the construction is pinned.
    """

    baseline = _snapshot(_model())
    independent = _model()
    scale_initialization(independent, 0.25)

    cumulative = _model()
    scale_initialization(cumulative, 0.5)
    scale_initialization(cumulative, 0.5)

    weight = dict(independent.named_parameters())["tok_embeddings.weight"]
    assert torch.equal(weight, baseline["tok_embeddings.weight"] * 0.25)
    # The cumulative route reaches the same place only by coincidence of 0.5^2.
    assert torch.allclose(
        dict(cumulative.named_parameters())["tok_embeddings.weight"], weight
    )


def test_a_tied_parameter_would_not_be_scaled_twice() -> None:
    """Guarded rather than assumed: this architecture ties nothing.

    Tying the head to the embedding is a common variant, and scaling one storage
    under two names would silently square the factor.
    """

    model = _model()
    model.output.weight = model.tok_embeddings.weight  # deliberately tie them
    baseline = model.tok_embeddings.weight.detach().clone()

    scale_initialization(model, 0.5)

    assert torch.equal(model.tok_embeddings.weight, baseline * 0.5)
    assert model.output.weight.data_ptr() == model.tok_embeddings.weight.data_ptr()


def test_a_non_positive_scale_is_rejected() -> None:
    for alpha in (0.0, -1.0):
        with pytest.raises(ValueError, match="alpha must be positive"):
            scale_initialization(_model(), alpha)


# -- reporting ---------------------------------------------------------------


def test_the_report_is_per_group_and_claims_no_single_sigma() -> None:
    """The architecture has several initializer families; one number would lie."""

    baseline = _model()
    captured = [(name, tensor.detach().clone()) for name, tensor in baseline.named_parameters()]
    scaled = _model()
    scale_initialization(scaled, 0.5)
    report = initialization_scale_report(scaled, 0.5, baseline=captured)

    assert report["has_single_sigma_w"] is False
    assert report["variance_factor"] == pytest.approx(0.25)
    groups = {group["name"]: group for group in report["groups"]}

    # The embedding and the linears start at genuinely different scales.
    embedding = groups["tok_embeddings.weight"]
    linear = groups["layers.0.attention.wq.weight"]
    assert embedding["baseline_std"] > 5.0 * linear["baseline_std"]

    for group in report["groups"]:
        if group["scaled"]:
            assert group["effective_std"] == pytest.approx(
                0.5 * group["baseline_std"], rel=1e-5
            )
        else:
            assert group["effective_std"] == pytest.approx(group["baseline_std"])


@pytest.mark.parametrize("alpha", [1.0, 0.5, 0.25])
def test_empirical_standard_deviations_follow_the_expected_ratio(alpha) -> None:
    """Measured, not only asserted from the multiplication."""

    captured = [(name, tensor.detach().clone()) for name, tensor in _model().named_parameters()]
    scaled = _model()
    scale_initialization(scaled, alpha)
    report = initialization_scale_report(scaled, alpha, baseline=captured)

    for group in report["groups"]:
        if group["scaled"] and group["baseline_std"] > 0:
            assert group["effective_std"] / group["baseline_std"] == pytest.approx(
                alpha, rel=1e-5
            )


def test_buffers_are_never_touched() -> None:
    """Rotary frequencies and the KV cache are not parameters and must not move."""

    model = _model()
    before = {name: tensor.detach().clone() for name, tensor in model.named_buffers()}

    scale_initialization(model, 0.25)

    for name, tensor in model.named_buffers():
        assert torch.equal(tensor, before[name]), name


# -- protocol resolution: the historical contract ----------------------------


def _load_runner():
    """Load the experiment script's protocol resolver without running it."""

    import importlib.util
    import sys
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_initialization_distribution_experiment.py"
    )
    spec = importlib.util.spec_from_file_location("initialization_distribution_script", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _historical_args(**overrides):
    """A namespace as it looked *before* the scale option existed.

    Deliberately minimal, and deliberately without ``initialization_scale``: this
    is the shape every protocol test written before the scale work still hands to
    the resolver, and it must keep resolving.
    """

    import argparse

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


def test_a_namespace_without_the_scale_option_resolves_to_the_baseline() -> None:
    """Absent means alpha = 1, because that is the historical condition.

    The scale intervention is a literal no-op at 1, so a protocol constructed
    before the option existed describes exactly the unscaled baseline. Failing
    instead would make every pre-existing protocol test depend on an option that
    has nothing to do with what it asserts.
    """

    resolve = _load_runner()._resolve_protocol

    protocol = resolve({}, _historical_args())

    assert protocol["initialization_scale"] == 1.0


def test_an_explicit_scale_is_honoured() -> None:
    resolve = _load_runner()._resolve_protocol

    protocol = resolve({}, _historical_args(initialization_scale=0.5))

    assert protocol["initialization_scale"] == 0.5


def test_the_fallback_equals_the_parser_default() -> None:
    """The two must agree, or an omitted flag would mean two different things."""

    import sys

    module = _load_runner()
    argv = sys.argv
    try:
        sys.argv = ["run_initialization_distribution_experiment.py"]
        parsed = module.parse_args()
    finally:
        sys.argv = argv

    from_parser = module._resolve_protocol({}, parsed)["initialization_scale"]
    from_fallback = module._resolve_protocol({}, _historical_args())["initialization_scale"]

    assert parsed.initialization_scale == 1.0
    assert from_parser == from_fallback == 1.0


def test_historical_gradient_and_sweep_resolution_are_unchanged() -> None:
    """The compatibility fallback must not disturb anything else it touches."""

    resolve = _load_runner()._resolve_protocol

    bare = resolve({}, _historical_args())
    assert bare["gradient_analysis_enabled"] is False
    assert bare["gradient_initialization_index"] == 0
    assert bare["gradient_num_windows"] is None
    assert bare["temperature_sweep_enabled"] is False
    assert bare["sweep_temperatures"] == ()

    enabled = resolve(
        {
            "gradient_analysis": {"enabled": True, "num_windows": 8},
            "temperature_sweep": {"enabled": True, "temperatures": [0.3, 0.6]},
        },
        _historical_args(),
    )
    assert enabled["gradient_analysis_enabled"] is True
    assert enabled["gradient_num_windows"] == 8
    assert enabled["temperature_sweep_enabled"] is True
    assert enabled["sweep_temperatures"] == (0.3, 0.6)
    assert enabled["initialization_scale"] == 1.0
