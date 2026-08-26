"""Shape, tying, initialization and registry tests for the GPT-2-style model.

The GPT family exists to be *different* from the LLaMA family, so most of what
is pinned here is the difference: biases where LLaMA has none, a learned
positional table where LLaMA has rotary frequencies, and one tied tensor where
LLaMA has two. Anything normalized toward LLaMA would quietly remove the
variation the arm was added to test.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from llm_behavior_lab.models import build_model_from_config, list_models
from llm_behavior_lab.models.base import BaseLanguageModel, ModelOutput
from llm_behavior_lab.models.gpt import GPTConfig, GPTForCausalLM
from llm_behavior_lab.models.initialization_scale import (
    classify_parameters,
    deterministic_parameter_suffixes,
    scale_initialization,
)
from llm_behavior_lab.utils import seed_everything

REPO_ROOT = Path(__file__).resolve().parents[1]
GPT2_12X768_CONFIG_PATH = REPO_ROOT / "configs" / "model" / "gpt2_12x768.yaml"


def _small(**overrides) -> GPTForCausalLM:
    """A cheap model for behavioural checks that do not depend on the shape."""

    settings = {
        "vocab_size": 64,
        "dim": 32,
        "n_layers": 2,
        "n_heads": 4,
        "max_seq_len": 16,
        "bias": True,
        "dropout": 0.0,
    }
    settings.update(overrides)
    seed_everything(1000)
    return GPTForCausalLM(GPTConfig(**settings))


@pytest.fixture(scope="module")
def gpt2_12x768():
    """The campaign arm, built once from the tracked config file.

    Module scoped: 110M parameters is expensive enough that rebuilding it per
    test would dominate the suite, and every assertion describes the same
    construction. The numbers come from the real builder reading the real file,
    not from re-implementing the layout in the test.
    """

    config = yaml.safe_load(GPT2_12X768_CONFIG_PATH.read_text(encoding="utf-8"))
    return config, build_model_from_config(config)


# -- registry and configuration ----------------------------------------------


def test_the_gpt_family_is_registered_under_one_name() -> None:
    """One family name; size belongs to the config file, not the registry."""

    registered = list_models()

    assert "gpt2" in registered
    assert [name for name in registered if name.startswith("gpt")] == ["gpt2"]


def test_the_tracked_config_builds_through_the_registry(gpt2_12x768) -> None:
    config, model = gpt2_12x768

    assert config["model"]["name"] == "gpt2"
    assert isinstance(model, GPTForCausalLM)
    assert isinstance(model, BaseLanguageModel)


def test_canonical_configuration_fields_produce_the_intended_architecture(
    gpt2_12x768,
) -> None:
    """The project-facing field names map onto the nanoGPT layout."""

    config, model = gpt2_12x768
    params = config["model"]["params"]

    assert params["dim"] == 768
    assert params["n_layers"] == 12
    assert params["n_heads"] == 12
    assert params["max_seq_len"] == 1024
    assert params["bias"] is True
    assert params["dropout"] == 0.0

    assert len(model.transformer.h) == 12
    assert model.transformer.wte.weight.shape == (32_000, 768)
    # The learned positional table is a parameter table, sized by max_seq_len.
    assert model.transformer.wpe.weight.shape == (1024, 768)
    assert model.transformer.h[0].attn.c_attn.weight.shape == (2304, 768)
    assert model.transformer.h[0].mlp.c_fc.weight.shape == (3072, 768)
    assert model.transformer.h[0].attn.n_heads == 12
    assert model.transformer.h[0].attn.head_dim == 64


def test_an_invalid_configuration_is_rejected() -> None:
    for overrides, message in (
        ({"dim": 30, "n_heads": 4}, "divisible"),
        ({"n_layers": 0}, "n_layers must be positive"),
        ({"max_seq_len": 0}, "max_seq_len must be positive"),
        ({"dropout": 1.0}, r"dropout must lie in \[0, 1\)"),
    ):
        settings = {
            "vocab_size": 64,
            "dim": 32,
            "n_layers": 2,
            "n_heads": 4,
            "max_seq_len": 16,
        }
        settings.update(overrides)
        with pytest.raises((ValueError, TypeError), match=message):
            GPTConfig(**settings).validate()


# -- parameter budget and tying ----------------------------------------------


def test_the_parameter_budget_matches_the_reference_decomposition(gpt2_12x768) -> None:
    """Unique trainable parameters and tensors, both pinned exactly.

    A mistyped width or a dropped bias would still build and still run; only the
    counts reveal it. The per-group split is asserted alongside the total so a
    compensating pair of errors cannot pass.
    """

    _, model = gpt2_12x768
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]

    block_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith("transformer.h.")
    )

    assert model.transformer.wte.weight.numel() == 24_576_000
    assert model.transformer.wpe.weight.numel() == 786_432
    assert block_parameters == 85_054_464
    assert model.transformer.ln_f.weight.numel() + model.transformer.ln_f.bias.numel() == 1_536
    assert model.count_parameters() == 110_418_432
    assert len(trainable) == 148


def test_the_embedding_and_head_are_one_parameter(gpt2_12x768) -> None:
    """Tying is object identity, not equal values.

    Everything downstream that walks parameters -- the exact gradient norm, the
    CountSketch tensor ordering, the initialization-scale audit -- must see the
    shared tensor once and only once.
    """

    _, model = gpt2_12x768

    assert model.transformer.wte.weight is model.lm_head.weight
    assert model.tok_embeddings.weight is model.output.weight

    names = [
        name
        for name, parameter in model.named_parameters()
        if parameter is model.lm_head.weight
    ]
    assert names == ["transformer.wte.weight"]


def test_a_tied_tensor_is_serialized_under_both_names(gpt2_12x768) -> None:
    """PyTorch does not de-duplicate state dicts, and neither does GPT-2.

    Recorded rather than worked around: 149 state-dict keys against 148
    parameter tensors is what a tied checkpoint looks like, and a round trip has
    to leave the tie intact.
    """

    _, model = gpt2_12x768
    state = model.state_dict()

    assert len(state) == 149
    assert state["lm_head.weight"].data_ptr() == state["transformer.wte.weight"].data_ptr()


def test_a_state_dict_round_trip_preserves_the_tie() -> None:
    """Loading has to reach the shared storage and leave it shared.

    The target is perturbed first, so the assertions below cannot pass merely
    because two models built from the same seed already agree.
    """

    source = _small()
    target = _small()
    with torch.no_grad():
        for parameter in target.parameters():
            parameter.fill_(0.0)
    assert not torch.equal(target.transformer.wte.weight, source.transformer.wte.weight)

    target.load_state_dict(source.state_dict())

    assert target.transformer.wte.weight is target.lm_head.weight
    assert torch.equal(target.transformer.wte.weight, source.transformer.wte.weight)
    assert torch.equal(target.lm_head.weight, source.lm_head.weight)


def test_the_causal_mask_buffers_stay_out_of_the_state_dict() -> None:
    """Non-persistent, like the LLaMA family's rotary and cache buffers."""

    model = _small()
    state = model.state_dict()

    buffers = [name for name, _ in model.named_buffers()]
    assert buffers
    assert all(name not in state for name in buffers)


# -- shared structural surface -----------------------------------------------


def test_the_shared_surface_points_at_the_canonical_modules(gpt2_12x768) -> None:
    """Read-only views, registered nowhere, so no alias is introduced."""

    _, model = gpt2_12x768

    assert model.tok_embeddings is model.transformer.wte
    assert model.layers is model.transformer.h
    assert model.norm is model.transformer.ln_f
    assert model.output is model.lm_head
    assert isinstance(model.layers, nn.ModuleList)

    # The properties are class attributes, so they never became submodules.
    module_names = {name for name, _ in model.named_modules()}
    assert "tok_embeddings" not in module_names
    assert "layers" not in module_names
    assert "norm" not in module_names
    assert "output" not in module_names


# -- forward behaviour -------------------------------------------------------


def test_forward_shape_and_scalar_loss(gpt2_12x768) -> None:
    _, model = gpt2_12x768
    input_ids = torch.randint(0, 32_000, (2, 64))
    targets = torch.randint(0, 32_000, (2, 64))

    output = model(input_ids=input_ids, targets=targets)

    assert isinstance(output, ModelOutput)
    assert output.logits.shape == (2, 64, 32_000)
    assert output.loss is not None
    assert output.loss.ndim == 0
    # At initialization the loss should sit near log(vocab_size); a wide band,
    # because this pins "the head is not degenerate", not a distribution.
    loss = float(output.loss.detach())
    assert 0.5 * math.log(32_000) < loss < 2.0 * math.log(32_000)


def test_exactly_one_input_form_is_accepted() -> None:
    model = _small()
    input_ids = torch.randint(0, 64, (1, 8))

    with pytest.raises(ValueError, match="exactly one"):
        model(input_ids=input_ids, inputs_embeds=model.tok_embeddings(input_ids))
    with pytest.raises(ValueError, match="exactly one"):
        model()


def test_the_inputs_embeds_path_matches_the_token_path() -> None:
    """The synthetic-input control must differ from real tokens in one place.

    Feeding the embedding lookup's own output back in has to reproduce the token
    path exactly; if it did not, the Gaussian input condition would be comparing
    two different networks rather than two different inputs.
    """

    model = _small()
    model.eval()
    input_ids = torch.randint(0, 64, (2, 8))

    with torch.no_grad():
        from_ids = model(input_ids=input_ids).logits
        from_embeds = model(inputs_embeds=model.tok_embeddings(input_ids)).logits

    assert torch.equal(from_ids, from_embeds)


def test_positional_embeddings_are_applied_on_the_inputs_embeds_path() -> None:
    """The bypass covers the token lookup only.

    Zeroing the positional table has to change the result, which it can only do
    if the table was being added on this path in the first place.
    """

    model = _small()
    model.eval()
    embeds = torch.randn(2, 8, model.config.dim)

    with torch.no_grad():
        before = model(inputs_embeds=embeds).logits
        model.transformer.wpe.weight.zero_()
        after = model(inputs_embeds=embeds).logits

    assert not torch.allclose(before, after)


def test_a_sequence_longer_than_the_positional_table_is_rejected() -> None:
    model = _small(max_seq_len=16)

    model(input_ids=torch.randint(0, 64, (1, 16)))
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        model(input_ids=torch.randint(0, 64, (1, 17)))


def test_attention_is_causal() -> None:
    """A token may not influence any logit at an earlier position."""

    model = _small()
    model.eval()
    original = torch.randint(0, 64, (1, 8))
    edited = original.clone()
    edited[0, 5] = (original[0, 5] + 1) % 64

    with torch.no_grad():
        before = model(input_ids=original).logits
        after = model(input_ids=edited).logits

    assert torch.equal(before[:, :5], after[:, :5])
    assert not torch.allclose(before[:, 5:], after[:, 5:])


# -- initialization and the deterministic-suffix policy ----------------------


def test_the_gpt_class_declares_its_own_deterministic_suffixes(gpt2_12x768) -> None:
    _, model = gpt2_12x768
    suffixes = deterministic_parameter_suffixes(model)

    assert suffixes == GPTForCausalLM.DETERMINISTIC_PARAMETER_SUFFIXES
    # Explicit suffixes, not a blanket rule that would catch a future random bias.
    assert ".bias" not in suffixes
    assert "bias" not in suffixes


def test_every_norm_gain_bias_and_zeroed_linear_bias_is_left_unscaled(
    gpt2_12x768,
) -> None:
    """98 tensors: 8 per block over 12 blocks, plus the final norm's gain and bias."""

    _, model = gpt2_12x768
    classified = classify_parameters(model)
    unscaled = {name for name, scaled in classified.items() if not scaled}

    expected = set()
    for index in range(12):
        prefix = f"transformer.h.{index}."
        expected.update(
            {
                prefix + "ln_1.weight",
                prefix + "ln_1.bias",
                prefix + "ln_2.weight",
                prefix + "ln_2.bias",
                prefix + "attn.c_attn.bias",
                prefix + "attn.c_proj.bias",
                prefix + "mlp.c_fc.bias",
                prefix + "mlp.c_proj.bias",
            }
        )
    expected.update({"transformer.ln_f.weight", "transformer.ln_f.bias"})

    assert unscaled == expected
    assert len(unscaled) == 98

    # The stochastic tensors, tied vocabulary matrix included, still scale.
    assert classified["transformer.wte.weight"] is True
    assert classified["transformer.wpe.weight"] is True
    assert classified["transformer.h.0.attn.c_attn.weight"] is True
    assert classified["transformer.h.0.mlp.c_proj.weight"] is True


def test_the_declared_policy_agrees_with_the_realized_initialization(
    gpt2_12x768,
) -> None:
    """Every tensor the policy calls deterministic really is a constant.

    The policy classifies by name and never reads a value; this test is the
    other direction, checking that the names it lists describe what the
    constructor actually did. A LayerNorm gain is exactly ones, every listed
    bias is exactly zeros, and nothing the policy leaves scalable is constant.
    """

    _, model = gpt2_12x768
    classified = classify_parameters(model)
    tensors = dict(model.named_parameters())

    for name, scaled in classified.items():
        parameter = tensors[name].detach()
        if scaled:
            assert parameter.unique().numel() > 1, f"{name} is constant but scalable"
        elif name.endswith(".bias"):
            assert torch.equal(parameter, torch.zeros_like(parameter)), name
        else:
            assert torch.equal(parameter, torch.ones_like(parameter)), name


def test_residual_projections_carry_the_depth_scaled_initialization(
    gpt2_12x768,
) -> None:
    """GPT-2 scales c_proj by 1/sqrt(2 * n_layers); everything else uses 0.02.

    Checked as a ratio of empirical standard deviations over tensors with
    millions of entries, where the estimate is precise to well under a percent,
    so the tolerance below is loose rather than marginal.
    """

    _, model = gpt2_12x768
    expected_ratio = 1.0 / math.sqrt(2 * 12)

    for index in (0, 6, 11):
        block = model.transformer.h[index]
        ordinary = float(block.mlp.c_fc.weight.detach().std())
        residual = float(block.mlp.c_proj.weight.detach().std())
        assert ordinary == pytest.approx(0.02, rel=0.05)
        assert residual / ordinary == pytest.approx(expected_ratio, rel=0.05)

        attention_residual = float(block.attn.c_proj.weight.detach().std())
        assert attention_residual / ordinary == pytest.approx(expected_ratio, rel=0.05)


def test_scaling_leaves_the_deterministic_tensors_and_scales_the_tie_once() -> None:
    """The Stage 2 mechanism, exercised on the real class rather than a stand-in."""

    model = _small()
    before = {name: tensor.detach().clone() for name, tensor in model.named_parameters()}
    baseline_vocabulary = model.transformer.wte.weight.detach().clone()

    applied = scale_initialization(model, 0.5)

    after = dict(model.named_parameters())
    for name in (
        "transformer.ln_f.weight",
        "transformer.ln_f.bias",
        "transformer.h.0.ln_1.weight",
        "transformer.h.0.attn.c_attn.bias",
        "transformer.h.0.mlp.c_proj.bias",
    ):
        assert torch.equal(after[name], before[name]), name

    # Halved, not quartered: one storage under two names.
    assert torch.equal(model.transformer.wte.weight, baseline_vocabulary * 0.5)
    assert model.lm_head.weight is model.transformer.wte.weight
    assert applied["scaled_parameters"].count("transformer.wte.weight") == 1
    assert "lm_head.weight" not in applied["scaled_parameters"]


def test_alpha_one_is_a_literal_no_op_for_the_gpt_family() -> None:
    model = _small()
    before = {name: tensor.detach().clone() for name, tensor in model.named_parameters()}

    applied = scale_initialization(model, 1.0)

    assert applied["is_no_op"] is True
    for name, tensor in model.named_parameters():
        assert torch.equal(tensor, before[name]), name


def test_construction_is_deterministic_under_a_fixed_seed() -> None:
    """Two builds from one seed must agree bitwise, tied draws included."""

    first = _small()
    second = _small()

    for (name, left), (_, right) in zip(
        first.named_parameters(), second.named_parameters()
    ):
        assert torch.equal(left, right), name


def test_the_model_reports_its_device() -> None:
    assert _small().device == torch.device("cpu")
