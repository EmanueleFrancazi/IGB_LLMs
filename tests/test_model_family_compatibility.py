"""The measurement pipeline must be model-generic across architecture families.

Every test here runs a **production** helper -- ``measure_initialization``,
``compute_position_gradient_norms``, ``compute_per_layer_gradient_norms``,
``build_evaluation_positions``, the input-condition constructors -- against a
tiny GPT and a tiny LLaMA, and asserts the same contract for both. Nothing
reimplements a measurement here; where a test looks like it is computing
something, it is checking a production result.

The models are deliberately minuscule. This is compatibility validation: it
answers "does the pipeline reach every parameter of a GPT model and produce
finite, correctly shaped output", not "is the estimator any good". Sketch
widths and position counts are chosen to make the shapes checkable and the CPU
cost negligible, and none of the numbers here is a scientific quantity.

Running the 110M and 134M production arms through a CPU gradient test would cost
minutes per assertion and would measure the same contract, so the tracked
configuration files are exercised statically instead -- see the configuration
section.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
import yaml

from llm_behavior_lab.data.dataloader import CausalLMBatch
from llm_behavior_lab.evaluation.gradient_norms import compute_per_layer_gradient_norms
from llm_behavior_lab.evaluation.init_distribution import (
    NucleusSamplingSettings,
    build_evaluation_positions,
    measure_initialization,
)
from llm_behavior_lab.evaluation.input_conditions import (
    embedding_moments,
    scaled_gaussian_embeddings,
    shuffled_input_ids,
    standardized_gaussian_bank,
)
from llm_behavior_lab.evaluation.position_gradients import compute_position_gradient_norms
from llm_behavior_lab.models import build_model, build_model_from_config, list_models
from llm_behavior_lab.models.initialization_scale import (
    initialization_note,
    initialization_scale_report,
    scale_initialization,
)
from llm_behavior_lab.utils import describe_device, seed_everything

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs" / "model"

#: Fields the experiment runner reads straight off ``model.params`` before any
#: model exists, so they cannot be supplied by the model object.
RUNNER_CONFIG_FIELDS = ("vocab_size", "dim", "max_seq_len")

#: Attributes the runner and the diagnostics reach for on the model object.
RUNNER_MODEL_ATTRIBUTES = ("tok_embeddings", "layers")

VOCAB_SIZE = 24
BLOCK_SIZE = 6
#: Four windows against a forward batch of two, so "once per forward pass" is an
#: assertion about batching rather than a coincidence of a single batch.
NUM_WINDOWS = 4
FORWARD_BATCH_SIZE = 2

#: Structural IDs held out of the predictive support, mirroring the pretrained
#: tokenizer's ``<unk>``/``<s>``/``</s>`` exclusion at a scale that runs offline
#: with no ``transformers`` dependency. IDs are never renumbered: an excluded
#: token keeps its index in every vector and simply holds zero.
EXCLUDED_TOKEN_IDS = (0, 1, 2)
ELIGIBLE_TOKEN_IDS = tuple(range(len(EXCLUDED_TOKEN_IDS), VOCAB_SIZE))


def _tiny_gpt():
    seed_everything(1000)
    return build_model(
        "gpt2",
        vocab_size=VOCAB_SIZE,
        dim=16,
        n_layers=2,
        n_heads=4,
        # Deliberately larger than BLOCK_SIZE, mirroring the production arm's
        # 1024 against a 64-token window: most positional rows go unused.
        max_seq_len=16,
        bias=True,
        dropout=0.0,
    )


def _tiny_llama():
    seed_everything(1000)
    return build_model(
        "llama_tiny",
        vocab_size=VOCAB_SIZE,
        dim=16,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        multiple_of=8,
        max_batch_size=4,
        max_seq_len=16,
    )


#: The two families, exercised through identical production calls.
FAMILIES = pytest.mark.parametrize(
    "build",
    [pytest.param(_tiny_gpt, id="gpt2"), pytest.param(_tiny_llama, id="llama")],
)

#: Synthetic corpus drawn only from eligible IDs, as the runner requires: it
#: refuses to run when the encoded corpus contains a structural token, because
#: the empirical distribution and the predictive support would then disagree.
CORPUS = [
    len(EXCLUDED_TOKEN_IDS) + (index * 7 + 3) % len(ELIGIBLE_TOKEN_IDS)
    for index in range(120)
]


def _positions():
    return build_evaluation_positions(
        CORPUS, block_size=BLOCK_SIZE, num_windows=NUM_WINDOWS
    )


def _sampling():
    return NucleusSamplingSettings(
        temperature=0.6,
        top_p=0.9,
        seed=20240601,
        num_replicates=1,
        common_random_numbers=True,
    )


def _measure(model, positions, *, eligible_token_ids=None, **condition):
    """One production measurement call, shared by every input condition."""

    return measure_initialization(
        model,
        positions,
        model_seed=1000,
        vocab_size=VOCAB_SIZE,
        sampling=_sampling(),
        eligible_token_ids=eligible_token_ids,
        forward_batch_size=FORWARD_BATCH_SIZE,
        device="cpu",
        **condition,
    )


def _gaussian_for(model, positions):
    """The Gaussian condition, built exactly as the runner builds it."""

    dim = model.tok_embeddings.weight.shape[1]
    bank = standardized_gaussian_bank(
        num_windows=positions.num_windows,
        block_size=BLOCK_SIZE,
        dim=dim,
        seed=60002,
    )
    moments = embedding_moments(
        model.tok_embeddings.weight, eligible_token_ids=ELIGIBLE_TOKEN_IDS
    )
    return scaled_gaussian_embeddings(bank, moments)


def _experiment_script():
    """Import the runner the way the other script tests do."""

    path = REPO_ROOT / "scripts" / "run_initialization_distribution_experiment.py"
    spec = importlib.util.spec_from_file_location(
        "initialization_distribution_script_compat", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# -- configuration and runner access ------------------------------------------


@pytest.mark.parametrize(
    "config_name", ["llama_12x768.yaml", "gpt2_12x768.yaml", "tiny_llama_32k.yaml"]
)
def test_tracked_configs_expose_the_fields_the_runner_reads(config_name) -> None:
    """The runner reads these three off the YAML, before any model exists.

    It cannot ask the model, because the Gaussian bank is drawn before the first
    initialization is built. So the field names are a contract between every
    family's config schema and the runner, and no alias layer exists to repair a
    family that spells them differently.
    """

    config = yaml.safe_load((CONFIG_DIR / config_name).read_text(encoding="utf-8"))
    params = config["model"]["params"]

    for field in RUNNER_CONFIG_FIELDS:
        assert field in params, f"{config_name} lacks {field}"
        assert isinstance(params[field], int)
        assert params[field] > 0


def test_the_gpt_arm_accepts_the_protocol_window() -> None:
    """block_size 64 against max_seq_len 1024, the runner's static check.

    The comparison is mirrored rather than invoked: the runner performs it as
    straight-line code inside ``main()``, which cannot be called without running
    a whole experiment. The values on both sides are read from the tracked files
    the runner would read, so a change to either is still caught here.
    """

    experiment = yaml.safe_load(
        (REPO_ROOT / "configs" / "experiment" / "initialization_distribution.yaml")
        .read_text(encoding="utf-8")
    )
    block_size = experiment["positions"]["block_size"]
    gpt = yaml.safe_load((CONFIG_DIR / "gpt2_12x768.yaml").read_text(encoding="utf-8"))
    llama = yaml.safe_load((CONFIG_DIR / "llama_12x768.yaml").read_text(encoding="utf-8"))

    assert block_size == 64
    assert block_size <= gpt["model"]["params"]["max_seq_len"]
    assert block_size <= llama["model"]["params"]["max_seq_len"]


def test_the_runner_resolves_both_arms_through_one_construction_seam() -> None:
    """``build_model_from_config`` is the runner's only way to make a model.

    Asserted as object identity against the package function, plus registry
    membership for both family names. The tracked arms are not constructed here:
    that costs a quarter of a billion parameters for a fact
    ``test_llama_shapes`` and ``test_gpt_shapes`` already establish by building
    each one from its own YAML.
    """

    module = _experiment_script()

    assert module.build_model_from_config is build_model_from_config

    registered = set(list_models())
    for config_name, expected_name in (
        ("gpt2_12x768.yaml", "gpt2"),
        ("llama_12x768.yaml", "llama"),
    ):
        config = yaml.safe_load((CONFIG_DIR / config_name).read_text(encoding="utf-8"))
        assert config["model"]["name"] == expected_name
        assert expected_name in registered


def _scale_orchestrator():
    """Import the initialization-scale driver the way the runner is imported."""

    path = REPO_ROOT / "scripts" / "run_initialization_scale_experiment.py"
    spec = importlib.util.spec_from_file_location("initialization_scale_driver", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _parent_manifest(tmp_path, extra_argv=()):
    """Run the orchestrator in dry-run mode and return the manifest it wrote.

    Dry run launches no child process, so this exercises the real manifest
    construction without running an experiment.
    """

    module = _scale_orchestrator()
    argv = sys.argv
    try:
        sys.argv = [
            "run_initialization_scale_experiment.py",
            "--dry-run",
            "--output-root",
            str(tmp_path),
            "--run-id",
            "manifest_test",
            *extra_argv,
        ]
        module.main()
    finally:
        sys.argv = argv
    return json.loads((tmp_path / "manifest_test" / "manifest.json").read_text())


def test_the_parent_manifest_makes_no_model_specific_initialization_claim(
    tmp_path,
) -> None:
    """The driver never builds a model, so it cannot describe one truthfully.

    It used to assert which tensor classes are scaled, that RMSNorm gains are the
    deterministic ones, that no biases exist, and that linears are
    kaiming-uniform. Every one of those is false for the GPT arm, and the driver
    accepts ``--model-config`` as a passthrough argument, so it could be pointed
    at that arm.
    """

    manifest = _parent_manifest(
        tmp_path, ("--model-config", "configs/model/gpt2_12x768.yaml")
    )
    pairing = manifest["pairing"]

    for removed in (
        "scaled_parameter_classes",
        "unscaled_parameter_classes",
        "bias_parameters",
        "sigma_w_note",
    ):
        assert removed not in pairing, removed

    # Removed outright, not blanked: an empty list or a placeholder string would
    # keep the ambiguous semantics while looking answered.
    serialized = json.dumps(manifest)
    for claim in ("kaiming_uniform", "RMSNorm", "normal_(0,1)", "bias=False"):
        assert claim not in serialized, claim


def test_the_parent_manifest_points_at_the_child_metadata(tmp_path) -> None:
    """Orchestration facts here; model provenance where the model was built."""

    manifest = _parent_manifest(tmp_path)
    provenance = manifest["initialization_provenance"]

    assert provenance["source"] == "child_run_metadata"
    assert provenance["metadata_file"] == "metadata.json"
    assert provenance["metadata_path"] == "initialization_scale"

    # The orchestration facts it *can* state are still stated.
    assert manifest["scales"] == [1.0, 0.5, 0.25]
    assert manifest["pairing"]["alpha_1_is_literal_no_op"] is True
    assert "conditions" in manifest


def test_the_manifest_retains_its_orchestration_fields(tmp_path) -> None:
    """What the driver genuinely knows, it still records.

    ``conditions``, ``scales`` and ``alpha_1_is_literal_no_op`` are facts about
    the orchestration itself -- which multipliers were requested, which child
    processes ran, and that alpha = 1 is a literal no-op in the intervention.
    None of them describes a model, so none of them moved.
    """

    manifest = _parent_manifest(tmp_path)

    assert isinstance(manifest["conditions"], list)
    assert isinstance(manifest["scales"], list)
    assert manifest["pairing"]["alpha_1_is_literal_no_op"] is True
    assert manifest["pairing"]["same_underlying_random_draw"] is True


@FAMILIES
def test_the_persisted_single_sigma_flag_comes_from_the_shared_report(build) -> None:
    """The runner persists what the report computed, not a literal beside it.

    This is also what makes the parent manifest's compatibility mirror currently
    truthful: both families supported today resolve ``False``, so the fixed
    literal the driver writes agrees with what every child run derives from its
    own model.

    It is a statement about **these two families**, not a guarantee about future
    ones. A family with a single initializer standard deviation everywhere would
    resolve ``True``, its child metadata would say so, and the mirror would then
    be wrong -- which is exactly why the mirror is documented as temporary and
    the child value is the authoritative one.
    """

    model = build()
    report = initialization_scale_report(model, 1.0)

    assert "has_single_sigma_w" in report
    assert report["has_single_sigma_w"] is False


def test_the_runner_consumes_the_shared_provenance_helpers() -> None:
    """The runner must resolve provenance, not restate it.

    Object identity against the package functions: if the runner ever grew its
    own copy of either -- as it previously carried its own hand-written
    initialization prose -- a record could describe a run differently from the
    scale report attached to the same run.
    """

    module = _experiment_script()

    assert module.scale_initialization is scale_initialization
    assert module.describe_device is describe_device


@FAMILIES
def test_the_persisted_note_follows_the_model_that_was_built(build) -> None:
    """What the runner writes is what the resolver returned for this model.

    The runner persists ``scale_applied["note"]`` under both
    ``initialization_scale.note`` and ``initialization_scale.applied.note``, so
    the two are the same string by construction rather than by agreement between
    two authors.
    """

    model = build()
    applied = scale_initialization(model, 1.0)

    assert applied["note"] == initialization_note(model)
    assert applied["is_no_op"] is True


@FAMILIES
def test_the_model_exposes_the_attributes_the_pipeline_reaches_for(build) -> None:
    """No family branch anywhere: one attribute name, both architectures."""

    model = build()

    for attribute in RUNNER_MODEL_ATTRIBUTES:
        assert hasattr(model, attribute), attribute

    # The Gaussian condition reads the embedding table through this exact path.
    assert model.tok_embeddings.weight.ndim == 2
    assert len(model.layers) == 2


# -- input conditions ---------------------------------------------------------


@FAMILIES
def test_every_input_condition_runs_through_the_production_path(build) -> None:
    """Real, shuffled and Gaussian, built and measured exactly as the runner does."""

    model = build()
    positions = _positions()
    dim = model.tok_embeddings.weight.shape[1]

    shuffled = shuffled_input_ids(positions.input_ids, seed=60001)
    bank = standardized_gaussian_bank(
        num_windows=positions.num_windows,
        block_size=BLOCK_SIZE,
        dim=dim,
        seed=60002,
    )
    moments = embedding_moments(model.tok_embeddings.weight, eligible_token_ids=None)
    gaussian = scaled_gaussian_embeddings(bank, moments)

    real = _measure(model, positions)
    shuffled_measurement = _measure(model, positions, input_ids=shuffled)
    gaussian_measurement = _measure(model, positions, inputs_embeds=gaussian)

    for measurement in (real, shuffled_measurement, gaussian_measurement):
        counts = measurement.greedy_counts
        assert counts.shape == (VOCAB_SIZE,)
        assert int(counts.sum()) == positions.num_positions
        assert torch.isfinite(counts.double()).all()

    # The shuffle preserves the multiset of inputs and the window shape.
    assert shuffled.shape == positions.input_ids.shape
    assert torch.equal(
        torch.sort(shuffled.reshape(-1)).values,
        torch.sort(positions.input_ids.reshape(-1)).values,
    )
    # Scaled to this initialization's own embedding moments, per family.
    assert gaussian.shape == (positions.num_windows, BLOCK_SIZE, dim)
    assert gaussian.dtype == torch.float32
    assert torch.isfinite(gaussian).all()


@FAMILIES
def test_the_gaussian_condition_produces_finite_logits_and_a_scalar_loss(build) -> None:
    """Shape, dtype and device handling on the synthetic-input boundary."""

    model = build()
    model.eval()
    positions = _positions()
    dim = model.tok_embeddings.weight.shape[1]
    moments = embedding_moments(model.tok_embeddings.weight, eligible_token_ids=None)
    gaussian = scaled_gaussian_embeddings(
        standardized_gaussian_bank(
            num_windows=positions.num_windows, block_size=BLOCK_SIZE, dim=dim, seed=60002
        ),
        moments,
    )

    with torch.no_grad():
        output = model(inputs_embeds=gaussian, targets=positions.target_ids)

    assert output.logits.shape == (positions.num_windows, BLOCK_SIZE, VOCAB_SIZE)
    assert output.logits.dtype == torch.float32
    assert output.logits.device == model.device
    assert torch.isfinite(output.logits).all()
    assert output.loss is not None
    assert output.loss.ndim == 0
    assert torch.isfinite(output.loss)


def _observe_forward_interface(model, run):
    """Record which input kwarg each forward received, and every ``wpe`` call.

    Instrumentation only: both hooks observe and return ``None``, so the real
    model computes the real forward. Nothing here reimplements the pass.

    Returns:
        ``(kwargs_per_forward, positional_call_shapes)``.
    """

    observed: list[list[str]] = []
    positional_calls: list[tuple[int, ...]] = []

    def record_inputs(module, args, kwargs):
        observed.append(sorted(name for name, value in kwargs.items() if value is not None))

    def record_positional(module, inputs, output):
        positional_calls.append(tuple(output.shape))

    input_hook = model.register_forward_pre_hook(record_inputs, with_kwargs=True)
    positional_hook = model.transformer.wpe.register_forward_hook(record_positional)
    try:
        run()
    finally:
        input_hook.remove()
        positional_hook.remove()
    return observed, positional_calls


def test_the_measurement_path_selects_the_inputs_embeds_interface_for_gaussian() -> None:
    """Stage 3 proved the model's bypass is correct; this proves it is chosen.

    ``measure_initialization`` must reach the GPT model through ``inputs_embeds``
    for the Gaussian condition and through ``input_ids`` for the real one, and
    the learned positional module must be invoked either way. Observed with
    forward hooks on the production call, so the assertion is structural rather
    than inferred from whether an argmax happened to move.
    """

    model = _tiny_gpt()
    positions = _positions()
    gaussian = _gaussian_for(model, positions)

    gaussian_kwargs, gaussian_positional = _observe_forward_interface(
        model, lambda: _measure(model, positions, inputs_embeds=gaussian)
    )
    real_kwargs, real_positional = _observe_forward_interface(
        model, lambda: _measure(model, positions)
    )

    expected_forwards = -(-positions.num_windows // FORWARD_BATCH_SIZE)
    assert expected_forwards > 1, "one batch would not test per-forward behaviour"

    # The condition chooses the interface, and nothing else does.
    assert gaussian_kwargs == [["inputs_embeds"]] * expected_forwards
    assert real_kwargs == [["input_ids"]] * expected_forwards

    # The learned positional table is consulted once per forward on both paths:
    # the bypass covers the token lookup and nothing else.
    positional_shape = (BLOCK_SIZE, model.config.dim)
    assert gaussian_positional == [positional_shape] * expected_forwards
    assert real_positional == [positional_shape] * expected_forwards


# -- eligible predictive support ----------------------------------------------


@FAMILIES
def test_a_restricted_eligible_support_is_honoured_in_every_condition(build) -> None:
    """The support must be the same token universe for every family and condition.

    Cross-model ``Delta`` values are only comparable when the softmax denominator
    is the same set of tokens, so this is the one thing that genuinely has to be
    harmonized across arms. It is tokenizer-defined and reaches the model through
    ``measure_initialization``; a family-specific path here would break the
    comparison silently.
    """

    model = build()
    positions = _positions()
    excluded = list(EXCLUDED_TOKEN_IDS)
    eligible = list(ELIGIBLE_TOKEN_IDS)

    # One support object, shared by all three conditions, exactly as the runner
    # shares its single ``eligible_token_ids`` across them.
    support = ELIGIBLE_TOKEN_IDS
    conditions = {
        "real": {},
        "shuffled": {"input_ids": shuffled_input_ids(positions.input_ids, seed=60001)},
        "gaussian": {"inputs_embeds": _gaussian_for(model, positions)},
    }

    for label, condition in conditions.items():
        measurement = _measure(
            model, positions, eligible_token_ids=support, **condition
        )

        # No renumbering: the vectors still span the full canonical vocabulary.
        assert measurement.greedy_counts.shape == (VOCAB_SIZE,), label
        assert measurement.mean_predicted_probabilities.shape == (VOCAB_SIZE,), label

        # An excluded token can never be selected by either policy.
        assert int(measurement.greedy_counts[excluded].sum()) == 0, label
        assert int(measurement.nucleus_counts[:, excluded].sum()) == 0, label
        # Masked to -inf before the softmax, so its probability is exactly zero.
        assert torch.equal(
            measurement.mean_predicted_probabilities[excluded],
            torch.zeros(len(excluded), dtype=measurement.mean_predicted_probabilities.dtype),
        ), label

        # Totals are unchanged: every position still produced a guess, from the
        # eligible set alone.
        assert int(measurement.greedy_counts.sum()) == positions.num_positions, label
        assert int(measurement.greedy_counts[eligible].sum()) == positions.num_positions, label


# -- gradient reachability and tied parameters --------------------------------


@FAMILIES
def test_a_single_position_loss_reaches_every_trainable_parameter(build) -> None:
    """The guard in the production gradient path must succeed for both families.

    ``compute_position_gradient_norms`` raises when any parameter comes back
    unreached, so a completed call with a full-length result *is* the evidence.
    """

    model = build()
    positions = _positions()

    result = compute_position_gradient_norms(
        model,
        positions,
        vocab_size=VOCAB_SIZE,
        eligible_token_ids=None,
        num_windows=1,
        temperatures=(0.5, 1.0),
        gradient_sketch=True,
        sketch_dimension=8,
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    assert result.parameter_count == sum(p.numel() for p in trainable)
    assert result.num_parameter_tensors == len(trainable)
    assert result.num_positions == BLOCK_SIZE
    assert result.temperature_gradient_norms.shape == (2, BLOCK_SIZE)
    assert torch.isfinite(result.temperature_gradient_norms).all()
    assert (result.temperature_gradient_norms > 0).all()


def test_the_gpt_positional_table_is_reached_with_zero_unused_rows() -> None:
    """``grad is None`` would abort the measurement; zero rows are fine.

    Under the campaign protocol most of ``wpe`` never participates. What matters
    is that autograd still returns a tensor for it -- the every-parameter-reached
    guard rejects ``None`` -- and that the untouched rows are exactly zero rather
    than absent.
    """

    model = _tiny_gpt()
    model.eval()
    positions = _positions()
    logits = model(input_ids=positions.input_ids[:1]).logits
    loss = torch.nn.functional.cross_entropy(
        logits[0, 2:3], positions.target_ids[0, 2:3]
    )

    parameters = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, parameters, allow_unused=True)

    assert all(gradient is not None for gradient in grads)

    names = [name for name, p in model.named_parameters() if p.requires_grad]
    wpe_grad = grads[names.index("transformer.wpe.weight")]
    assert wpe_grad.shape == (16, 16)
    # Position 2's loss sees positions 0..2 only; rows beyond the window are
    # untouched and rows past the sequence never exist in the graph at all.
    assert torch.equal(wpe_grad[BLOCK_SIZE:], torch.zeros_like(wpe_grad[BLOCK_SIZE:]))
    assert not torch.equal(wpe_grad[:3], torch.zeros_like(wpe_grad[:3]))


def test_the_tied_parameter_is_counted_once_everywhere() -> None:
    """Enumeration, exact-gradient accounting and the sketch map must agree."""

    model = _tiny_gpt()
    positions = _positions()

    result = compute_position_gradient_norms(
        model,
        positions,
        vocab_size=VOCAB_SIZE,
        eligible_token_ids=None,
        num_windows=1,
        temperatures=(1.0,),
        gradient_sketch=True,
        sketch_dimension=8,
    )

    names = [name for name, p in model.named_parameters() if p.requires_grad]
    assert names.count("transformer.wte.weight") == 1
    assert "lm_head.weight" not in names

    vocabulary = model.tok_embeddings.weight.numel()
    assert result.parameter_count == sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    # Counted once, not twice: a double count would show up as one extra
    # vocabulary matrix in the total and in the sketch domain.
    assert sum(result.sketch_tensor_sizes) == result.parameter_count
    assert result.sketch_tensor_sizes.count(vocabulary) == 1
    assert len(result.sketch_tensor_sizes) == result.num_parameter_tensors


@FAMILIES
def test_a_minimal_countsketch_measurement_is_finite_and_correctly_shaped(build) -> None:
    """Compatibility only: eight buckets says nothing about estimator quality."""

    model = build()
    positions = _positions()
    temperatures = (0.5, 1.0)

    result = compute_position_gradient_norms(
        model,
        positions,
        vocab_size=VOCAB_SIZE,
        eligible_token_ids=None,
        num_windows=1,
        temperatures=temperatures,
        gradient_sketch=True,
        sketch_dimension=8,
    )

    assert result.temperature_gradient_sketches.shape == (2, BLOCK_SIZE, 8)
    assert torch.isfinite(result.temperature_gradient_sketches).all()
    assert result.gradient_sketches.shape == (BLOCK_SIZE, 8)
    # The canonical row is taken from the temperature stack, never re-measured.
    assert torch.equal(
        result.gradient_sketches,
        result.temperature_gradient_sketches[temperatures.index(1.0)],
    )
    assert result.sketch_protocol["dimension"] == 8
    assert result.sketch_protocol["parameter_count"] == result.parameter_count

    buckets, signs = result.sketch_map
    assert buckets.shape == (result.parameter_count,)
    assert signs.shape == (result.parameter_count,)
    assert set(signs.tolist()) <= {-1.0, 1.0}
    assert int(buckets.min()) >= 0 and int(buckets.max()) < 8


# -- layer-gradient reporting -------------------------------------------------


@FAMILIES
def test_layer_gradient_reporting_consumes_model_layers_without_a_branch(build) -> None:
    """One entry per block, finite, in stack order, for either family."""

    model = build()
    positions = _positions()
    batch = CausalLMBatch(
        input_ids=positions.input_ids,
        targets=positions.target_ids,
        split="train",
    )

    result = compute_per_layer_gradient_norms(model, [batch])

    assert len(result.layer_norms) == len(model.layers)
    assert [entry.layer_index for entry in result.layer_norms] == [0, 1]
    assert all(entry.squared_l2_norm > 0 for entry in result.layer_norms)
    assert all(
        torch.isfinite(torch.tensor(entry.squared_l2_norm))
        for entry in result.layer_norms
    )
    assert torch.isfinite(torch.tensor(result.mean_loss))
    assert result.num_batches == 1


# -- the parent manifest's provenance pointer --------------------------------
#
# The pointer is descriptive provenance: it documents where authoritative
# per-child initialization data lives. No tracked code reads it, so nothing
# here asserts that a consumer follows it -- only that the driver writes it.


def test_the_parent_manifest_keeps_a_labelled_compatibility_mirror(tmp_path) -> None:
    """One model-specific field survives, deliberately and with its terms stated.

    ``pairing.has_single_sigma_w`` is a fixed literal that an existing local
    validation script still reads. Removing it would break that script the moment
    this driver wrote a manifest, so it stays until the script is made portable
    and updated to read child provenance. What matters is that the manifest says
    so: the pointer records that this is a mirror, not a derived value, and names
    the authoritative source.
    """

    manifest = _parent_manifest(tmp_path)
    provenance = manifest["initialization_provenance"]

    assert manifest["pairing"]["has_single_sigma_w"] is False

    mirror = provenance["legacy_compatibility_mirror"]
    assert "retained temporarily" in mirror
    assert "NOT derived from a model" in mirror
    assert "metadata.json -> initialization_scale -> has_single_sigma_w" in mirror
    assert "Remove the mirror" in mirror

    # The authoritative location is still the child metadata, unchanged.
    assert provenance["source"] == "child_run_metadata"
    assert provenance["metadata_path"] == "initialization_scale"


def test_the_deferred_layerwise_statement_is_model_generic(tmp_path) -> None:
    """No fixed block count, no Tiny-LLaMA-specific claim."""

    deferred = _parent_manifest(tmp_path)["deferred"]["layerwise_gradient_stability"]

    assert "architecture-aware" in deferred
    for claim in ("two transformer blocks", "two", "shallow", "LLaMA"):
        assert claim not in deferred, claim
