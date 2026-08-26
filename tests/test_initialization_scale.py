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
from torch import nn

from llm_behavior_lab.models.gpt import GPTConfig, GPTForCausalLM
from llm_behavior_lab.models.initialization_scale import (
    DEFAULT_INITIALIZATION_NOTE,
    DETERMINISTIC_PARAMETER_SUFFIXES,
    classify_parameters,
    deterministic_parameter_suffixes,
    initialization_note,
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


# -- per-family deterministic sets -------------------------------------------
#
# A second architecture family names its deterministic tensors differently and
# has more of them: LayerNorm carries a bias as well as a gain, and its linear
# biases are initialized to zero rather than drawn. Scaling a LayerNorm gain is
# a different intervention from scaling a random draw -- it changes the
# normalization -- so the set has to follow the family.
#
# The stand-in below is a classification fixture, not a model: it has no
# forward. It exists so this stage can be reviewed without the GPT family.


class _DeclaringFamily(nn.Module):
    """A GPT-2-shaped parameter layout that declares its own deterministic set.

    Named to match the upstream layout the future family will use, because the
    declaration is a set of *name suffixes* and a test on exact top-level names
    would not exercise that. ``c_proj.bias`` deliberately occurs twice, under
    ``attn`` and under ``mlp``, so one suffix has to cover both.
    """

    DETERMINISTIC_PARAMETER_SUFFIXES = (
        "ln_1.weight",
        "ln_1.bias",
        "ln_f.weight",
        "ln_f.bias",
        "c_attn.bias",
        "c_proj.bias",
        "c_fc.bias",
    )

    def __init__(self, dim: int = 8, vocab_size: int = 16) -> None:
        super().__init__()
        self.wte = nn.Embedding(vocab_size, dim)
        self.wpe = nn.Embedding(4, dim)
        self.ln_1 = nn.LayerNorm(dim)
        self.attn = nn.ModuleDict(
            {"c_attn": nn.Linear(dim, 3 * dim), "c_proj": nn.Linear(dim, dim)}
        )
        self.mlp = nn.ModuleDict(
            {"c_fc": nn.Linear(dim, 4 * dim), "c_proj": nn.Linear(4 * dim, dim)}
        )
        self.ln_f = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        # Tied, as GPT-2 is. One storage, reachable under two names.
        self.wte.weight = self.lm_head.weight
        for module in self.modules():
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)


def _declaring_family(seed: int = SEED) -> _DeclaringFamily:
    seed_everything(seed)
    return _DeclaringFamily()


def test_the_llama_family_resolves_to_the_module_default() -> None:
    """LLaMA declares nothing, so it runs the historical code path unchanged.

    Identity, not equality: falling through to the module constant is what makes
    the existing behaviour identical by construction rather than by transcribing
    the same three suffixes into a second place where they could drift.
    """

    assert deterministic_parameter_suffixes(_model()) is DETERMINISTIC_PARAMETER_SUFFIXES


def test_llama_classification_is_unchanged_in_content_and_order() -> None:
    """Regression: the same names, the same values, the same insertion order."""

    model = _model()
    expected = {
        name: not any(
            name.endswith(suffix) for suffix in DETERMINISTIC_PARAMETER_SUFFIXES
        )
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    classified = classify_parameters(model)

    assert classified == expected
    assert list(classified) == list(expected)
    assert sum(classified.values()) == sum(expected.values())


def test_llama_scale_report_lists_are_unchanged() -> None:
    """The serialized audit keeps its content and its ordering."""

    model = _model()
    applied = scale_initialization(model, 0.5)
    names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]

    assert applied["scaled_parameters"] == [
        name
        for name in names
        if not any(name.endswith(suffix) for suffix in DETERMINISTIC_PARAMETER_SUFFIXES)
    ]
    assert applied["unscaled_parameters"] == [
        name
        for name in names
        if any(name.endswith(suffix) for suffix in DETERMINISTIC_PARAMETER_SUFFIXES)
    ]
    assert applied["is_no_op"] is False
    assert applied["num_scaled_tensors"] == len(applied["scaled_parameters"])


def test_a_declaring_family_excludes_its_norms_and_zeroed_biases() -> None:
    """Every LayerNorm gain and bias, and every zero-initialized linear bias."""

    model = _declaring_family()
    classified = classify_parameters(model)
    unscaled = sorted(name for name, scaled in classified.items() if not scaled)

    assert unscaled == [
        "attn.c_attn.bias",
        "attn.c_proj.bias",
        "ln_1.bias",
        "ln_1.weight",
        "ln_f.bias",
        "ln_f.weight",
        "mlp.c_fc.bias",
        "mlp.c_proj.bias",
    ]
    # The stochastic tensors, including the tied vocabulary matrix, still scale.
    assert classified["wte.weight"] is True
    assert classified["wpe.weight"] is True
    assert classified["attn.c_attn.weight"] is True
    assert classified["mlp.c_proj.weight"] is True


def test_a_declaring_family_leaves_its_deterministic_tensors_untouched() -> None:
    """The declaration has to survive the intervention, not just the audit."""

    model = _declaring_family()
    before = _snapshot(model)

    scale_initialization(model, 0.5)

    after = dict(model.named_parameters())
    for name in ("ln_1.weight", "ln_1.bias", "ln_f.weight", "ln_f.bias"):
        assert torch.equal(after[name], before[name]), name
    for name in ("attn.c_attn.bias", "attn.c_proj.bias", "mlp.c_fc.bias", "mlp.c_proj.bias"):
        assert torch.equal(after[name], before[name]), name
    assert torch.equal(after["wpe.weight"], before["wpe.weight"] * 0.5)


def test_a_declaring_family_scales_a_tied_tensor_exactly_once() -> None:
    """One storage under two names must not pick up alpha squared."""

    model = _declaring_family()
    assert model.wte.weight.data_ptr() == model.lm_head.weight.data_ptr()
    baseline = model.wte.weight.detach().clone()

    applied = scale_initialization(model, 0.5)

    assert torch.equal(model.wte.weight, baseline * 0.5)
    assert model.lm_head.weight.data_ptr() == model.wte.weight.data_ptr()
    # named_parameters() de-duplicates, so the storage is audited under one name.
    assert applied["scaled_parameters"].count("wte.weight") == 1
    assert "lm_head.weight" not in applied["scaled_parameters"]


def test_alpha_one_is_a_literal_no_op_for_a_declaring_family() -> None:
    model = _declaring_family()
    before = _snapshot(model)

    applied = scale_initialization(model, 1.0)

    assert applied["is_no_op"] is True
    for name, tensor in model.named_parameters():
        assert torch.equal(tensor, before[name]), name


def test_classification_never_reads_tensor_values() -> None:
    """Intent, not realization.

    A stochastic tensor that happens to hold all ones is still stochastic, and a
    deterministic gain overwritten with a random draw is still deterministic.
    Deciding from the values would make the audit seed-dependent and would
    misreport a draw that landed near a constant.
    """

    model = _declaring_family()
    with torch.no_grad():
        model.mlp.c_fc.weight.fill_(1.0)
        model.ln_1.weight.normal_()

    classified = classify_parameters(model)

    assert classified["mlp.c_fc.weight"] is True
    assert classified["ln_1.weight"] is False


def test_an_undeclared_module_falls_back_to_the_default() -> None:
    """The declaration is opt-in; anything else keeps the historical set."""

    plain = nn.Linear(4, 4)

    assert deterministic_parameter_suffixes(plain) is DETERMINISTIC_PARAMETER_SUFFIXES


def test_an_explicitly_empty_declaration_is_honoured() -> None:
    """Declaring nothing and declaring 'nothing is deterministic' differ."""

    model = _declaring_family()
    model.DETERMINISTIC_PARAMETER_SUFFIXES = ()

    assert deterministic_parameter_suffixes(model) == ()
    assert all(classify_parameters(model).values())


def test_a_bare_string_declaration_is_rejected() -> None:
    """The dangerous shape: iterating it would match single characters."""

    model = _declaring_family()
    model.DETERMINISTIC_PARAMETER_SUFFIXES = "ln_f.weight"

    with pytest.raises(TypeError, match="not a single string"):
        classify_parameters(model)


def test_a_declaration_of_non_strings_is_rejected() -> None:
    model = _declaring_family()
    model.DETERMINISTIC_PARAMETER_SUFFIXES = ("ln_f.weight", "")

    with pytest.raises(TypeError, match="non-empty string"):
        classify_parameters(model)


# -- initialization provenance ------------------------------------------------
#
# The note is persisted into every run's metadata and outlives the run. A model
# that inherited another family's note would misdescribe the very thing a
# cross-architecture comparison varies, so the resolution is pinned here in the
# same way the deterministic set is.


def _gpt(**overrides) -> GPTForCausalLM:
    settings = {
        "vocab_size": 32,
        "dim": 16,
        "n_layers": 2,
        "n_heads": 4,
        "max_seq_len": 8,
        "bias": True,
        "dropout": 0.0,
    }
    settings.update(overrides)
    seed_everything(SEED)
    return GPTForCausalLM(GPTConfig(**settings))


def test_the_llama_family_resolves_to_the_default_note() -> None:
    """Identity again: LLaMA declares nothing and falls through."""

    assert initialization_note(_model()) is DEFAULT_INITIALIZATION_NOTE


def test_the_default_note_describes_this_repositorys_llama_implementation() -> None:
    """Every claim in it is checkable against the model, and is checked here.

    A provenance string is only worth persisting if it is true, so the assertions
    below read the actual model rather than trusting the prose.
    """

    note = DEFAULT_INITIALIZATION_NOTE
    model = _model()

    assert "normal_(0,1)" in note
    assert "kaiming_uniform_(a=sqrt(5))" in note
    assert "no bias parameters at all" in note
    assert "RMSNorm gains are initialized to one and are not scaled" in note
    assert "untied" in note
    # It must not claim the upstream explicit initialization this repository
    # deliberately does not use.
    assert "0.02" in note and "deliberately retained" in note

    assert [name for name, _ in model.named_parameters() if "bias" in name] == []
    assert model.tok_embeddings.weight.data_ptr() != model.output.weight.data_ptr()
    for name, parameter in model.named_parameters():
        if any(name.endswith(s) for s in DETERMINISTIC_PARAMETER_SUFFIXES):
            assert torch.equal(parameter, torch.ones_like(parameter)), name


def test_the_gpt_family_declares_its_own_note() -> None:
    model = _gpt()

    note = initialization_note(model)

    assert note == GPTForCausalLM.INITIALIZATION_NOTE
    assert note != DEFAULT_INITIALIZATION_NOTE


def test_the_gpt_note_states_its_convention_and_makes_no_llama_claim() -> None:
    """None of the three LLaMA-only claims may survive into a GPT record."""

    note = initialization_note(_gpt())

    for claim in (
        "kaiming_uniform",
        "normal_(0,1)",
        "no bias parameters at all",
        "RMSNorm",
        "untied",
    ):
        assert claim not in note, claim

    for stated in (
        "normal_(0, 0.02)",
        "0.02/sqrt(2 * n_layers)",
        "LayerNorm gains are initialized to one",
        "LayerNorm biases to zero",
        "every linear bias",
        "share one parameter",
        "scaled exactly once",
    ):
        assert stated in note, stated


def test_the_scale_report_note_is_the_resolved_one_for_each_family() -> None:
    """One source: the report cannot disagree with the resolver."""

    llama = _model()
    gpt = _gpt()

    assert scale_initialization(llama, 0.5)["note"] == initialization_note(llama)
    assert scale_initialization(gpt, 0.5)["note"] == initialization_note(gpt)
    assert scale_initialization(_model(), 1.0)["note"] == DEFAULT_INITIALIZATION_NOTE


def test_a_malformed_note_declaration_is_rejected() -> None:
    """Fail loudly rather than persist misleading or empty provenance."""

    model = _gpt()

    model.INITIALIZATION_NOTE = 42
    with pytest.raises(TypeError, match="must be a string"):
        scale_initialization(model, 0.5)

    model.INITIALIZATION_NOTE = "   "
    with pytest.raises(TypeError, match="must not be empty"):
        scale_initialization(model, 0.5)


def test_the_scale_report_keys_are_unchanged() -> None:
    """No serialized key added, removed or renamed by this change."""

    applied = scale_initialization(_model(), 0.5)

    assert sorted(applied) == [
        "alpha",
        "is_no_op",
        "note",
        "num_scaled_elements",
        "num_scaled_tensors",
        "scaled_parameters",
        "unscaled_parameters",
    ]


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


# -- sketch-width resolution -------------------------------------------------
#
# The estimator's width reaches a record's metadata and its persisted sketch
# shape, so an omitted, mistyped or ignored value is a scientific problem rather
# than a usability one.


def test_an_omitted_sketch_width_falls_back_to_the_default() -> None:
    """The parser's default and the resolver's fallback must agree."""

    import sys

    module = _load_runner()
    argv = sys.argv
    try:
        sys.argv = ["run_initialization_distribution_experiment.py"]
        parsed = module.parse_args()
    finally:
        sys.argv = argv

    # The flag itself no longer carries the width; the resolver supplies it.
    assert parsed.sketch_dimension is None
    assert module._resolve_protocol({}, parsed)["sketch_dimension"] == (
        module.DEFAULT_SKETCH_DIMENSION
    )


def test_an_explicit_sketch_width_wins_over_the_config() -> None:
    module = _load_runner()
    config = {"gradient_analysis": {"sketch_dimension": 128}}

    resolved = module._resolve_protocol(
        config, _historical_args(sketch_dimension=64)
    )["sketch_dimension"]

    assert resolved == 64


def test_the_config_supplies_the_width_when_the_flag_is_omitted() -> None:
    """Previously unreachable: the flag's default used to shadow the config.

    With ``default=DEFAULT_SKETCH_DIMENSION`` the left operand of the old ``or``
    chain was always truthy, so ``gradient_analysis.sketch_dimension`` could
    never take effect. It is real configuration now.
    """

    module = _load_runner()
    config = {"gradient_analysis": {"sketch_dimension": 128}}

    resolved = module._resolve_protocol(
        config, _historical_args(sketch_dimension=None)
    )["sketch_dimension"]

    assert resolved == 128


def test_a_zero_sketch_width_is_refused_rather_than_silently_replaced() -> None:
    """Zero is falsy, so the old ``or`` chain turned it into 512.

    A run would then have recorded a width nobody asked for, and the persisted
    sketch would have had a shape the command line did not describe.
    """

    module = _load_runner()

    with pytest.raises(ValueError, match="positive number of buckets"):
        module._resolve_protocol({}, _historical_args(sketch_dimension=0))


def test_a_negative_sketch_width_is_refused() -> None:
    module = _load_runner()

    with pytest.raises(ValueError, match="positive number of buckets"):
        module._resolve_protocol({}, _historical_args(sketch_dimension=-8))


def test_existing_valid_configurations_resolve_unchanged() -> None:
    """No tracked config sets a width, so every one of them still means 512."""

    module = _load_runner()

    for config in ({}, {"gradient_analysis": {}}, {"gradient_analysis": {"sketch": True}}):
        resolved = module._resolve_protocol(config, _historical_args())
        assert resolved["sketch_dimension"] == module.DEFAULT_SKETCH_DIMENSION
