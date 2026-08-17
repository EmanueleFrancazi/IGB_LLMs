"""Compare token guesses at random initialization with corpus token frequencies.

The experiment holds everything fixed except the model-initialization seed:
one corpus, one tokenizer, one split, and one deterministic set of evaluation
positions are computed before any model exists, and every initialization is
measured on exactly those positions.

For each initialization the model runs once over the fixed positions, and both
guessing policies read the same logits:

* greedy -- ``argmax``, deterministic given the initialization;
* nucleus -- temperature scaling plus top-p truncation, replicated several times
  from those same logits so sampling noise can be averaged out within an
  initialization before initializations are compared.

The result is one record holding complete per-token vectors, written into a
standard experiment-run directory. Figures are produced from that record by
:mod:`llm_behavior_lab.analysis.figures`; nothing here draws anything.
"""

from __future__ import annotations

import argparse
import resource
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

# Allow running from the repository root before editable installation.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from llm_behavior_lab.analysis import (  # noqa: E402
    DEFAULT_NULL_REPLICATES,
    InitializationExperimentRecord,
    paired_concentration_differences,
    paired_condition_distances,
    simulate_uniform_null,
    pooled_nucleus_zero_frequency,
    sampling_adequacy,
    summarize_policy,
    support_summary,
    within_initialization_sampling_spread,
)
from llm_behavior_lab.data import (  # noqa: E402
    DatasetConfig,
    add_dataset_arguments,
    resolution_policy_from_args,
    resolve_dataset,
    resolve_repo_path,
    split_token_ids,
)
from llm_behavior_lab.data.tokenizer import build_tokenizer  # noqa: E402
from llm_behavior_lab.evaluation import empirical_token_counts  # noqa: E402
from llm_behavior_lab.evaluation.input_conditions import (  # noqa: E402
    INPUT_CONDITIONS,
    embedding_moments,
    scaled_gaussian_embeddings,
    shuffled_input_ids,
    standardized_gaussian_bank,
)
from llm_behavior_lab.evaluation.init_distribution import (  # noqa: E402
    NucleusSamplingSettings,
    build_evaluation_positions,
    measure_initialization,
)
from llm_behavior_lab.evaluation.position_gradients import (  # noqa: E402
    compute_position_gradient_norms,
)
from llm_behavior_lab.experiment import ExperimentRun, experiment_settings_from_config  # noqa: E402
from llm_behavior_lab.experiment.naming import compose_run_id  # noqa: E402
from llm_behavior_lab.models import build_model_from_config  # noqa: E402
from llm_behavior_lab.models.initialization_scale import (  # noqa: E402
    initialization_scale_report,
    scale_initialization,
)
from llm_behavior_lab.utils import get_device, seed_everything  # noqa: E402


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load a YAML config file."""

    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"Expected config dictionary, got {type(config)!r}")
    return config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Measure token-guess distributions across random model initializations."
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=REPO_ROOT / "configs" / "data" / "tiny_text.yaml",
        help="Path to the data YAML config.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=REPO_ROOT / "configs" / "model" / "tiny_llama.yaml",
        help="Path to the model YAML config.",
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=REPO_ROOT / "configs" / "experiment" / "initialization_distribution.yaml",
        help="Path to the experiment protocol config.",
    )
    parser.add_argument(
        "--num-initializations",
        type=int,
        default=None,
        help="Override the number of independent model initializations.",
    )
    parser.add_argument(
        "--num-windows",
        type=int,
        default=None,
        help="Override the number of deterministic evaluation windows.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="Override the evaluation window length in tokens.",
    )
    parser.add_argument(
        "--num-replicates",
        type=int,
        default=None,
        help="Override the number of stochastic sampling replicates per initialization.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val"],
        default=None,
        help="Override the analysis split.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the experiment output root.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional explicit run ID. Existing runs are never overwritten.",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default=None,
        help="Optional run notes overriding the experiment config.",
    )
    parser.add_argument(
        "--forward-batch-size",
        type=int,
        default=None,
        help=(
            "Windows per forward pass. Affects memory only, never the result. "
            "Lower it for a large vocabulary: the transient logits are "
            "batch x block x vocab, and sorting for nucleus truncation costs "
            "several multiples of that again."
        ),
    )
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=None,
        metavar="T",
        help=(
            "Override the nucleus temperature sweep, e.g. --temperatures 0.12 0.6 1.2. "
            "All values must be positive; greedy is the T=0 anchor and is always computed "
            "by argmax. Does not affect the canonical sampling.temperature."
        ),
    )
    parser.add_argument(
        "--no-temperature-sweep",
        action="store_true",
        help="Skip the multi-temperature nucleus sweep.",
    )
    parser.add_argument(
        "--no-uniform-null",
        action="store_true",
        help="Skip the uniform categorical output null.",
    )
    parser.add_argument(
        "--gradient-analysis",
        action="store_true",
        help=(
            "Measure the exact per-position parameter-gradient norm for one "
            "initialization. Costs one backward pass per evaluation position, so "
            "it is off unless asked for."
        ),
    )
    parser.add_argument(
        "--no-gradient-analysis",
        action="store_true",
        help="Skip the per-position gradient analysis even if the config enables it.",
    )
    parser.add_argument(
        "--gradient-windows",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Measure gradients on a deterministic evenly spaced subset of N "
            "evaluation windows instead of all of them. This reduces the "
            "scientific position set; the default measures every window."
        ),
    )
    parser.add_argument(
        "--no-input-structure",
        action="store_true",
        help="Skip the shuffled and Gaussian input conditions.",
    )
    parser.add_argument(
        "--initialization-scale",
        type=float,
        default=1.0,
        metavar="ALPHA",
        help=(
            "Multiply every audited zero-centred random weight by ALPHA, leaving "
            "the deterministic RMSNorm gains untouched. 1.0 is a literal no-op. "
            "Applied to a freshly built model, so each scale derives from the "
            "same pristine draw and the conditions are paired."
        ),
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip figure generation. The record is written either way.",
    )
    add_dataset_arguments(parser)
    return parser.parse_args()


def _resolve_temperatures(values: Any) -> tuple[float, ...]:
    """Validate and normalize the sweep temperatures.

    Duplicates are collapsed and the user's order is preserved, so a config or
    command line reads back exactly as written. Zero and negative values are
    rejected: greedy is the T=0 anchor and is computed by argmax, never by a
    zero-temperature softmax, which is undefined.
    """

    resolved: list[float] = []
    for value in values or ():
        try:
            temperature = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"temperature_sweep.temperatures contains a non-numeric entry {value!r}."
            ) from error
        if not temperature > 0.0:
            raise ValueError(
                f"temperature_sweep.temperatures must be strictly positive; got {temperature}. "
                "Greedy is the T=0 anchor and is computed by argmax."
            )
        if temperature not in resolved:
            resolved.append(temperature)
    return tuple(resolved)


def _describe_tokenizer_line(description: dict[str, Any]) -> str:
    """Render the resolved tokenizer identity for the console summary.

    Derived entirely from the tokenizer's own ``describe()`` output. Hard-coding
    the kind here is what previously made a Mistral run announce itself as
    ``char``: the persisted metadata was right and only the printed line lied,
    which is the worst version of the bug because the number beside it looked
    plausible.
    """

    kind = str(description.get("type", "unknown"))
    vocab = description.get("vocab_size")
    eligible = description.get("eligible_vocab_size")
    parts = [kind]
    if kind == "pretrained" and description.get("identifier"):
        parts.append(str(description["identifier"]))
        revision = description.get("requested_revision")
        parts.append(f"revision={revision or 'unpinned'}")
    rendered = ", ".join(parts)
    if eligible is not None and vocab is not None and eligible != vocab:
        return f"{rendered}, vocab size {vocab} ({eligible} eligible)"
    return f"{rendered}, vocab size {vocab}"


def _resolve_protocol(experiment_config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Merge the experiment config with command-line overrides."""

    initialization = experiment_config.get("initialization", {})
    positions = experiment_config.get("positions", {})
    sampling = experiment_config.get("sampling", {})
    runtime = experiment_config.get("runtime", {})
    null = experiment_config.get("uniform_null", {})
    sweep = experiment_config.get("temperature_sweep", {})
    structure = experiment_config.get("input_structure", {})
    gradients = experiment_config.get("gradient_analysis", {})

    return {
        "num_initializations": int(
            args.num_initializations
            if args.num_initializations is not None
            else initialization.get("num_initializations", 12)
        ),
        "base_seed": int(initialization.get("base_seed", 1000)),
        "seed_stride": int(initialization.get("seed_stride", 1)),
        "num_windows": int(
            args.num_windows if args.num_windows is not None else positions.get("num_windows", 512)
        ),
        "block_size": args.block_size if args.block_size is not None else positions.get("block_size"),
        "split": args.split if args.split is not None else str(positions.get("split", "train")),
        "temperature": float(sampling.get("temperature", 0.6)),
        "top_p": float(sampling.get("top_p", 0.9)),
        "sampling_seed": int(sampling.get("seed", 20240601)),
        "num_replicates": int(
            args.num_replicates
            if args.num_replicates is not None
            else sampling.get("num_replicates", 8)
        ),
        "forward_batch_size": int(
            args.forward_batch_size
            if args.forward_batch_size is not None
            else runtime.get("forward_batch_size", 32)
        ),
        # Absent from a historical config means the historical per-initialization
        # stream. The current protocol selects the new behaviour explicitly.
        "common_random_numbers": bool(sampling.get("common_random_numbers", False)),
        "uniform_null_enabled": bool(null.get("enabled", True)) and not args.no_uniform_null,
        "uniform_null_seed": int(null.get("seed", 20260812)),
        "uniform_null_replicates": int(null.get("replicates", DEFAULT_NULL_REPLICATES)),
        "input_structure_enabled": bool(structure.get("enabled", True))
        and not args.no_input_structure,
        "temperature_sweep_enabled": (
            (bool(sweep.get("enabled", False)) or args.temperatures is not None)
            and not args.no_temperature_sweep
        ),
        "sweep_temperatures": _resolve_temperatures(
            args.temperatures if args.temperatures is not None else sweep.get("temperatures", ())
        ),
        "shuffle_seed": int(structure.get("shuffle_seed", 60001)),
        "gaussian_seed": int(structure.get("gaussian_seed", 60002)),
        # A config written before this analysis existed simply does not carry it.
        "gradient_analysis_enabled": (
            (bool(gradients.get("enabled", False)) or args.gradient_analysis)
            and not args.no_gradient_analysis
        ),
        "initialization_scale": float(args.initialization_scale),
        "gradient_initialization_index": int(gradients.get("initialization_index", 0)),
        "gradient_num_windows": (
            args.gradient_windows
            if args.gradient_windows is not None
            else gradients.get("num_windows")
        ),
    }


def main() -> None:
    """Run the multi-initialization guess-distribution experiment."""

    args = parse_args()
    data_config = load_yaml_config(args.data_config)
    model_config = load_yaml_config(args.model_config)
    experiment_config = load_yaml_config(args.experiment_config)
    protocol = _resolve_protocol(experiment_config, args)

    if protocol["num_initializations"] <= 0:
        raise ValueError("--num-initializations must be positive.")
    if protocol["seed_stride"] <= 0:
        raise ValueError("initialization.seed_stride must be positive.")
    if protocol["gradient_analysis_enabled"] and not (
        0 <= protocol["gradient_initialization_index"] < protocol["num_initializations"]
    ):
        raise ValueError(
            f"gradient_analysis.initialization_index "
            f"{protocol['gradient_initialization_index']} is outside the "
            f"{protocol['num_initializations']} requested initializations."
        )

    batching_config = data_config["batching"]
    block_size = int(protocol["block_size"] or batching_config["block_size"])
    protocol["block_size"] = block_size

    device = get_device(str(data_config.get("runtime", {}).get("device", "auto")))

    # ---- fixed corpus, tokenizer, split, and evaluation positions ----------
    resolved = resolve_dataset(
        DatasetConfig.from_config(data_config),
        resolution_policy_from_args(args),
        repo_root=REPO_ROOT,
    )

    text = resolved.read_text()
    policy = resolution_policy_from_args(args)
    tokenizer = build_tokenizer(
        data_config.get("tokenizer"),
        text=text,
        data_root=policy.data_root,
        allow_download=policy.allow_download,
        offline=policy.offline,
        reporter=print,
    )
    token_ids = tokenizer.encode(text)
    splits = split_token_ids(
        token_ids,
        val_fraction=float(data_config["dataset"].get("val_fraction", 0.1)),
        min_train_tokens=block_size + 1,
        min_val_tokens=block_size + 1,
    )
    split_ids = splits.train_ids if protocol["split"] == "train" else splits.val_ids

    model_params = model_config["model"]["params"]
    model_vocab_size = int(model_params["vocab_size"])
    tokenizer_description = tokenizer.describe()
    is_pretrained = tokenizer_description.get("type") == "pretrained"
    if tokenizer.vocab_size > model_vocab_size:
        raise ValueError(
            f"Tokenizer vocab size {tokenizer.vocab_size} exceeds model vocab size "
            f"{model_vocab_size}. Increase model.params.vocab_size."
        )
    if is_pretrained and tokenizer.vocab_size != model_vocab_size:
        # A wider model would waste an embedding row per unused ID and put
        # unreachable logits in every comparison. For a pretrained vocabulary the
        # two must agree exactly; the character tokenizer keeps the historical
        # "may be narrower" behaviour, since its vocabulary depends on the text.
        raise ValueError(
            f"Tokenizer {tokenizer_description.get('identifier')!r} has vocabulary "
            f"{tokenizer.vocab_size} but the model config declares {model_vocab_size}. "
            "A pretrained tokenizer requires an exact match; use "
            "configs/model/tiny_llama_32k.yaml or set model.params.vocab_size to "
            f"{tokenizer.vocab_size}."
        )

    eligible_token_ids = tokenizer.eligible_token_ids
    special_token_ids = set(tokenizer.special_token_ids)
    if special_token_ids.intersection(token_ids):
        raise ValueError(
            "The encoded corpus contains structural token IDs "
            f"{sorted(special_token_ids.intersection(token_ids))[:5]}, so the empirical "
            "distribution and the predictive support would disagree. Set "
            "tokenizer.add_special_tokens to false."
        )
    if block_size > int(model_params["max_seq_len"]):
        raise ValueError(
            f"block_size {block_size} exceeds model max_seq_len {model_params['max_seq_len']}."
        )

    positions = build_evaluation_positions(
        split_ids,
        block_size=block_size,
        num_windows=protocol["num_windows"],
        device=device,
    )

    # The primary empirical reference is the whole analysis split, not the text
    # as a whole: mixing in validation tokens would compare the model against
    # data the split deliberately excludes.
    corpus_counts = empirical_token_counts(split_ids, vocab_size=tokenizer.vocab_size)
    selected_counts = empirical_token_counts(
        positions.target_ids.reshape(-1).tolist(),
        vocab_size=tokenizer.vocab_size,
    )

    sampling = NucleusSamplingSettings(
        temperature=protocol["temperature"],
        top_p=protocol["top_p"],
        seed=protocol["sampling_seed"],
        num_replicates=protocol["num_replicates"],
        common_random_numbers=protocol["common_random_numbers"],
    )
    sampling.validate()
    if protocol["input_structure_enabled"]:
        # Fail rather than silently reinterpreting the request. The paired
        # comparison is defined at one stochastic realization per position, held
        # fixed across conditions; overriding either half would quietly change
        # what was measured.
        if sampling.num_replicates != 1:
            raise ValueError(
                "The input-structure comparison requires exactly one nucleus replicate "
                f"(R=1), but the protocol asks for R={sampling.num_replicates}. Set "
                "sampling.num_replicates to 1, or disable the comparison with "
                "--no-input-structure."
            )
        if not sampling.common_random_numbers:
            raise ValueError(
                "The input-structure comparison requires common random numbers, so every "
                "input condition sees the same sampling draws, but the protocol has "
                "sampling.common_random_numbers disabled. Set it to true, or disable the "
                "comparison with --no-input-structure."
            )

    print("Initialization-distribution experiment")
    print(f"Dataset: {resolved.name} ({resolved.source}, via {resolved.route})")
    print(f"Dataset path: {resolved.path}")
    print(f"Tokenizer: {_describe_tokenizer_line(tokenizer_description)}")
    print(f"Analysis split: {protocol['split']} ({len(split_ids)} tokens)")
    print(
        f"Evaluation positions: {positions.num_positions} "
        f"({positions.num_windows} windows x {block_size} tokens, deterministic)"
    )
    print(f"Initializations: {protocol['num_initializations']}")
    print(
        f"Nucleus policy: temperature={sampling.temperature}, top_p={sampling.top_p}, "
        f"replicates={sampling.num_replicates}, sampling seed={sampling.seed}"
    )
    print(f"Device: {device}")
    if protocol["initialization_scale"] != 1.0:
        print(
            f"Initialization scale: alpha = {protocol['initialization_scale']} "
            f"(variance x {protocol['initialization_scale'] ** 2:g}); "
            "applied to the audited random weights only"
        )

    # ---- one measurement per initialization -------------------------------
    model_seeds: list[int] = []
    measurements = []
    condition_measurements: dict[str, list] = {"shuffled": [], "gaussian": []}
    embedding_moment_log: list[dict[str, Any]] = []
    parameter_count = 0
    gradient_result = None
    scale_applied: dict[str, Any] = {}
    scale_report: dict[str, Any] = {}
    logit_diagnostics: list[dict[str, Any]] = []

    shuffled_ids = None
    gaussian_bank = None
    if protocol["input_structure_enabled"]:
        # Drawn once, before any model exists, and reused for every
        # initialization: the controls must not vary with the weights.
        shuffled_ids = shuffled_input_ids(positions.input_ids, seed=protocol["shuffle_seed"])
        gaussian_bank = standardized_gaussian_bank(
            num_windows=positions.num_windows,
            block_size=block_size,
            dim=int(model_params["dim"]),
            seed=protocol["gaussian_seed"],
            device=device,
        )
        print(
            f"Input conditions: {', '.join(INPUT_CONDITIONS)} "
            f"(shuffle seed {protocol['shuffle_seed']}, Gaussian seed {protocol['gaussian_seed']})"
        )

    sweep_temperatures = (
        protocol["sweep_temperatures"] if protocol["temperature_sweep_enabled"] else ()
    )
    if sweep_temperatures:
        print(
            "Temperature sweep:\n"
            f"  T = {', '.join(f'{value:g}' for value in sweep_temperatures)}\n"
            f"  top_p = {sampling.top_p} (fixed)\n"
            f"  R = {sampling.num_replicates}\n"
            f"  common random numbers: {'yes' if sampling.common_random_numbers else 'no'}\n"
            "  greedy is the T=0 anchor; the same logits and the same uniforms serve "
            "every temperature"
        )

    started = time.perf_counter()
    for index in range(protocol["num_initializations"]):
        model_seed = protocol["base_seed"] + index * protocol["seed_stride"]
        seed_everything(model_seed)
        model = build_model_from_config(model_config).to(device)
        # Captured before the intervention so the per-group report measures the
        # baseline rather than inferring it by dividing the scaled tensors.
        baseline_parameters = (
            [(name, parameter.detach().clone()) for name, parameter in model.named_parameters()]
            if index == 0
            else None
        )
        scale_applied = scale_initialization(model, protocol["initialization_scale"])
        if index == 0:
            scale_report = initialization_scale_report(
                model, protocol["initialization_scale"], baseline=baseline_parameters
            )
        del baseline_parameters
        parameter_count = model.count_parameters()
        # Streamed: logits exist one batch at a time and are folded into
        # per-token counters, so memory does not grow with the position count.
        def measure(**condition_inputs):
            return measure_initialization(
                model,
                positions,
                model_seed=model_seed,
                vocab_size=tokenizer.vocab_size,
                sampling=sampling,
                eligible_token_ids=eligible_token_ids,
                forward_batch_size=protocol["forward_batch_size"],
                device=device,
                sweep_temperatures=sweep_temperatures,
                **condition_inputs,
            )

        # The raw T=1 predictive diagnostics are collected for the real input
        # only: they describe the model's predictive geometry on the corpus, and
        # computing them for the control conditions would triple their cost
        # without being part of the question.
        measurements.append(measure(collect_probability_statistics=True))
        if measurements[-1].logit_diagnostics is not None:
            logit_diagnostics.append(
                dict(measurements[-1].logit_diagnostics, model_seed=model_seed)
            )
        if (
            protocol["gradient_analysis_enabled"]
            and index == protocol["gradient_initialization_index"]
        ):
            # Measured on the model object that was just measured above, not on a
            # re-seeded reconstruction of it: the observable describes *this*
            # initialization. The call leaves every parameter, buffer, gradient,
            # and the train/eval mode exactly as it found them.
            gradient_result = compute_position_gradient_norms(
                model,
                positions,
                vocab_size=tokenizer.vocab_size,
                eligible_token_ids=eligible_token_ids,
                num_windows=protocol["gradient_num_windows"],
            )
            rate = gradient_result.num_positions / max(gradient_result.seconds, 1e-9)
            print(
                f"    gradient analysis: {gradient_result.num_positions:,} positions "
                f"over {gradient_result.parameter_count:,} parameters in "
                f"{gradient_result.seconds:,.1f}s ({rate:,.1f} positions/s)"
            )
        if protocol["input_structure_enabled"]:
            # Same model object, same weights, same nucleus uniforms: only the
            # input changes, so any difference is attributable to the input.
            condition_measurements["shuffled"].append(measure(input_ids=shuffled_ids))
            moments = embedding_moments(
                model.tok_embeddings.weight, eligible_token_ids=eligible_token_ids
            )
            embedding_moment_log.append(moments.as_dict())
            condition_measurements["gaussian"].append(
                measure(inputs_embeds=scaled_gaussian_embeddings(gaussian_bank, moments))
            )
        model_seeds.append(model_seed)
        del model
        print(
            f"  initialization {index + 1}/{protocol['num_initializations']} "
            f"(seed {model_seed}) measured"
        )
    elapsed = time.perf_counter() - started
    peak_rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    # ---- assemble the record ----------------------------------------------
    def stacked(attribute: str, source=None) -> Any:
        items = measurements if source is None else source
        return torch.stack([getattr(item, attribute) for item in items]).cpu().numpy()

    null_summary = None
    if protocol["uniform_null_enabled"]:
        null_summary = simulate_uniform_null(
            eligible_vocab_size=len(eligible_token_ids),
            num_draws=positions.num_positions,
            num_replicates=protocol["uniform_null_replicates"],
            seed=protocol["uniform_null_seed"],
        )

    if gradient_result is None:
        gradient_metadata: dict[str, Any] = {"enabled": False}
    else:
        gradient_index = protocol["gradient_initialization_index"]
        gradient_metadata = gradient_result.as_metadata(
            enabled=True,
            initialization_index=gradient_index,
            model_seed=model_seeds[gradient_index],
            input_condition="real",
            guessing_policy="greedy",
            covers_all_positions=(
                gradient_result.num_positions == positions.num_positions
            ),
        )

    settings = experiment_settings_from_config(experiment_config)
    output_dir = settings.output_dir if args.output_dir is None else args.output_dir
    settings = replace(
        settings,
        output_dir=resolve_repo_path(output_dir),
        notes=settings.notes if args.notes is None else args.notes,
    )

    metadata = {
        "experiment_type": "initialization_distribution",
        "initialization_scale": {
            "alpha": protocol["initialization_scale"],
            "variance_factor": protocol["initialization_scale"] ** 2,
            "is_no_op": protocol["initialization_scale"] == 1.0,
            "has_single_sigma_w": False,
            "note": (
                "alpha is a global multiplier on every audited zero-centred random "
                "weight. The architecture has no single sigma_w: the embedding is "
                "normal_(0,1) and every linear is kaiming_uniform_(a=sqrt(5)) with a "
                "fan-in dependent scale. Deterministic RMSNorm gains are not scaled, "
                "and the architecture contains no bias parameters at all."
            ),
            "applied": scale_applied,
            "parameter_groups": scale_report,
        },
        "device": str(device),
        "model_name": model_config["model"]["name"],
        "model_parameter_count": parameter_count,
        "model_vocab_size": model_vocab_size,
        "dataset_path": str(resolved.path),
        "dataset": resolved.provenance(),
        "tokenizer": tokenizer_description,
        "analysis": {
            "split": protocol["split"],
            "split_token_count": len(split_ids),
            "corpus_token_count": int(corpus_counts.sum().item()),
            "num_positions": positions.num_positions,
            "evaluation_positions": positions.as_dict(),
            "num_initializations": protocol["num_initializations"],
            "model_seeds": model_seeds,
            "sampling": sampling.as_dict(),
            "input_conditions": (
                list(INPUT_CONDITIONS) if protocol["input_structure_enabled"] else ["real"]
            ),
            "shuffle_seed": protocol["shuffle_seed"] if protocol["input_structure_enabled"] else None,
            "gaussian_seed": protocol["gaussian_seed"] if protocol["input_structure_enabled"] else None,
            "gaussian_embedding_moments": embedding_moment_log,
            "temperature_sweep": {
                "enabled": bool(sweep_temperatures),
                "temperatures": list(sweep_temperatures),
                "canonical_temperature": sampling.temperature,
                "top_p": sampling.top_p,
                "shares_logits_with_canonical": True,
                "shares_uniforms_with_canonical": True,
            },
            "predictive_probabilities": {
                "enabled": True,
                "from_raw_logits": True,
                "support": "eligible",
                "eligible_vocab_size": len(eligible_token_ids),
                "temperature": 1.0,
                "before_top_p": True,
                "before_sampling": True,
                "input_condition": "real",
                "uniform_probability": 1.0 / len(eligible_token_ids),
                "note": (
                    "Softmax of the raw logits restricted to the eligible support, "
                    "at temperature 1, before any top-p truncation and before any "
                    "greedy or nucleus selection. NOT the nucleus distribution."
                ),
                "ranked_profile_order": "rank_within_position_then_average_over_positions",
                "retains_full_probability_tensor": False,
                "confidence_temperatures": list(measurements[0].confidence_temperatures),
                "confidence_temperature_note": (
                    "Diagnostic only: softmax(z/T) is measured, never sampled and "
                    "never truncated. argmax softmax(z/T) = argmax z for every "
                    "positive T, so every temperature describes the SAME greedy "
                    "decisions. This is not the stochastic nucleus sweep."
                ),
            },
            # Describes z itself, not z/T, so it is stored once per
            # initialization rather than once per temperature.
            "raw_logits": logit_diagnostics,
            "gradient_analysis": gradient_metadata,
            "vocab_size": tokenizer.vocab_size,
            "eligible_vocab_size": len(eligible_token_ids),
            "excluded_special_token_ids": sorted(special_token_ids),
            "forward_batch_size": protocol["forward_batch_size"],
            "transient_logits_shape": [
                min(protocol["forward_batch_size"], positions.num_windows),
                block_size,
                tokenizer.vocab_size,
            ],
            "retains_all_position_logits": False,
        },
        "runtime": {
            "measurement_seconds": round(elapsed, 3),
            "peak_rss_mib": round(peak_rss_mib, 1),
        },
    }

    run_id = args.run_id or compose_run_id(
        dataset=resolved.name,
        tokenizer=tokenizer_description,
        model_name=model_config["model"]["name"],
        model_vocab_size=model_vocab_size,
        num_positions=positions.num_positions,
        num_initializations=protocol["num_initializations"],
        num_replicates=sampling.num_replicates,
    )
    run = ExperimentRun.create(
        settings,
        run_id=run_id,
        repo_root=REPO_ROOT,
        metadata=metadata,
    )
    run.snapshot_configs(
        model_config=model_config,
        data_config=data_config,
        experiment_config=experiment_config,
    )

    record_metadata = dict(metadata)
    record_metadata["num_positions"] = positions.num_positions
    record_metadata["tokens"] = [
        tokenizer.token_repr(token_id) for token_id in range(tokenizer.vocab_size)
    ]
    record_metadata["run_id"] = run.run_id
    if null_summary is not None:
        record_metadata["uniform_null"] = null_summary.as_dict()
    record_metadata["experiment_name"] = run.experiment_name

    def sweep_arrays(attribute: str) -> dict[str, Any]:
        if not sweep_temperatures:
            return {}
        collected = {"real": np.stack([getattr(m, attribute).numpy() for m in measurements])}
        for name, items in condition_measurements.items():
            if items:
                collected[name] = np.stack([getattr(m, attribute).numpy() for m in items])
        return collected

    record = InitializationExperimentRecord.build(
        corpus_counts=corpus_counts.cpu().numpy(),
        selected_target_counts=selected_counts.cpu().numpy(),
        greedy_counts=stacked("greedy_counts"),
        nucleus_counts=stacked("nucleus_counts"),
        mean_predicted_probabilities=stacked("mean_predicted_probabilities"),
        model_seeds=model_seeds,
        eligible_token_ids=list(eligible_token_ids),
        metadata=record_metadata,
        condition_greedy_counts={
            name: stacked("greedy_counts", items)
            for name, items in condition_measurements.items()
            if items
        },
        condition_nucleus_counts={
            name: stacked("nucleus_counts", items)
            for name, items in condition_measurements.items()
            if items
        },
        sweep_counts_by_condition=sweep_arrays("sweep_counts"),
        sweep_agreement_by_condition=sweep_arrays("sweep_agreement"),
        predictive_ranked_probabilities=stacked("ranked_probability_profile"),
        predictive_max_probabilities=stacked("max_probabilities"),
        predictive_target_probabilities=stacked("target_probabilities"),
        predictive_target_losses=stacked("target_losses"),
        predictive_temperatures=np.asarray(measurements[0].confidence_temperatures),
        predictive_temperature_ranked_probabilities=stacked("temperature_ranked_probabilities"),
        predictive_temperature_max_probabilities=stacked("temperature_max_probabilities"),
        predictive_temperature_target_probabilities=stacked(
            "temperature_target_probabilities"
        ),
        predictive_temperature_target_losses=stacked("temperature_target_losses"),
        predictive_temperature_mean_entropy=stacked("temperature_mean_entropy"),
        predictive_temperature_mean_token_probabilities=stacked(
            "temperature_mean_token_probabilities"
        ),
        gradient_position_indices=(
            None if gradient_result is None else gradient_result.position_indices.numpy()
        ),
        gradient_position_target_ids=(
            None if gradient_result is None else gradient_result.target_ids.numpy()
        ),
        gradient_position_greedy_ids=(
            None if gradient_result is None else gradient_result.greedy_ids.numpy()
        ),
        gradient_position_norms=(
            None if gradient_result is None else gradient_result.gradient_norms.numpy()
        ),
        gradient_temperatures=(
            None if gradient_result is None else np.asarray(gradient_result.temperatures)
        ),
        gradient_temperature_position_norms=(
            None
            if gradient_result is None
            else gradient_result.temperature_gradient_norms.numpy()
        ),
        uniform_null=(
            {}
            if null_summary is None
            else {
                "ranked_mean": null_summary.ranked_mean,
                "ranked_low": null_summary.ranked_low,
                "ranked_high": null_summary.ranked_high,
            }
        ),
    )
    record.save(run.paths.analyses_dir)

    # ---- scalar summaries through the existing metric pipeline ------------
    adequacy = sampling_adequacy(record)
    run.evaluation_metrics.log(
        step=0,
        stage="initialization",
        split=protocol["split"],
        metrics={
            "selected_position_tv_to_split": adequacy["total_variation_distance"],
            "selected_position_js_to_split": adequacy["js_divergence"],
            "evaluated_positions": adequacy["num_positions"],
            "vocabulary_size": adequacy["vocab_size"],
            "vocabulary_fraction_represented": adequacy["fraction_of_vocabulary_represented"],
        },
    )
    summaries = {policy: summarize_policy(record, policy) for policy in ("greedy", "nucleus")}
    for policy, summary in summaries.items():
        run.evaluation_metrics.log(
            step=0,
            stage=f"initialization_guess_policy_{policy}",
            split=protocol["split"],
            metrics=summary.as_dict(),
        )
    sampling_spread = within_initialization_sampling_spread(record)
    run.evaluation_metrics.log(
        step=0,
        stage="initialization_sampling_variability",
        split=protocol["split"],
        metrics=sampling_spread,
    )
    run.save_analysis_json(
        "initialization_distribution_summary.json",
        {
            "sampling_adequacy": adequacy,
            "policies": {policy: summary.as_dict() for policy, summary in summaries.items()},
            "sampling_variability": sampling_spread,
        },
    )

    support = support_summary(record)
    print("\nVocabulary and support (four different questions, four numbers):")
    print(f"  V full                : {support['vocab_size']:,}")
    print(f"  V eligible            : {support['eligible_vocab_size']:,}")
    print(f"  V corpus-observed     : {support['corpus_observed_support']:,}")
    print(f"  corpus effective N_eff: {support['corpus_effective_support']:,.2f}")

    print("\nEmpirical sampling adequacy of the evaluation positions:")
    print(f"  TV(split, selected targets): {adequacy['total_variation_distance']:.6f}")
    print(f"  JS(split, selected targets): {adequacy['js_divergence']:.6f}")
    print(f"  selected effective N_eff   : {adequacy['selected_effective_support']:,.2f}")
    print(
        f"  eligible vocabulary represented in targets: "
        f"{adequacy['tokens_represented_in_selection']:,}/{adequacy['eligible_vocab_size']:,}"
    )

    for policy, summary in summaries.items():
        print(
            f"\n{policy} guessing policy "
            f"(mean ± SEM across {summary.num_initializations} initializations):"
        )
        print(f"  TV(corpus, guesses)  : {summary.total_variation_mean:.6f} ± {summary.total_variation_sem:.6f}")
        print(
            f"  effective support    : {summary.effective_support_mean:,.3f}"
            f" ± {summary.effective_support_sem:,.3f} tokens"
        )
        print(
            f"  zero-frequency tokens: {summary.zero_frequency_count_mean:,.2f}"
            f" ± {summary.zero_frequency_count_sem:,.2f}"
            f"  ({summary.zero_frequency_fraction_mean:.2%})"
            f"  over {summary.zero_frequency_draws:,} draws"
            + ("  [per replicate]" if policy == "nucleus" else "")
        )
        print(f"  q(1) - q(2)          : {summary.top_two_gap_mean:.6f} ± {summary.top_two_gap_sem:.6f}")
        print(f"  max typical |q-p|    : {summary.typical_gap_max_mean:.6f}")
        print(f"  max persistent       : {summary.persistent_gap_max:.6f}")
        print(f"  corpus tokens never guessed: {summary.corpus_observed_zero_guess_mean:,.1f}")

    pooled = pooled_nucleus_zero_frequency(record)
    print(
        f"\nNucleus pooled coverage diagnostic (NOT comparable with greedy): "
        f"{pooled.counts.mean():,.1f} tokens unreached over "
        f"{pooled.draws_per_measurement:,} pooled draws"
    )
    print(f"\nRuntime: {elapsed:.1f}s | peak RSS: {peak_rss_mib:,.0f} MiB")
    print(f"Model parameters: {parameter_count:,}")
    print("\nStochastic sampling variability (nucleus policy):")
    if sampling_spread.get("estimated"):
        print(
            f"  mean within-initialization std of TV: "
            f"{sampling_spread['mean_within_initialization_std_tv']:.6f}"
        )
        print(
            f"  between-initialization std of TV: "
            f"{sampling_spread['between_initialization_std_tv']:.6f}"
        )
    else:
        print(f"  within-initialization stochastic variability: {sampling_spread['reason']}")
        print(
            "  the nucleus SEM across initializations therefore describes the combined "
            "initialization + one fixed sampling realization"
        )

    if null_summary is not None:
        described = null_summary.as_dict()
        print(
            f"\nUniform categorical output null "
            f"(K={described['eligible_vocab_size']:,}, D={described['num_draws']:,}, "
            f"M={described['monte_carlo_replicates']} Monte Carlo replicates):"
        )
        print(
            f"  zero-frequency: {described['zero_frequency_count_mean']:,.1f} "
            f"({described['zero_frequency_fraction_mean']:.2%}); "
            f"analytic K(1-1/K)^D = {described['analytic_zero_frequency_count']:,.1f}"
        )
        print(
            f"  effective support: {described['effective_support_mean']:,.1f} "
            f"[{described['effective_support_mc_low']:,.1f}, "
            f"{described['effective_support_mc_high']:,.1f}] Monte Carlo interval"
        )
        print(f"  top1-top2 gap: {described['top_two_gap_mean']:.6f}")

    if protocol["input_structure_enabled"]:
        print("\nInput-structure comparison (paired within each initialization):")
        for policy in ("greedy", "nucleus"):
            distances = paired_condition_distances(record, policy)
            differences = paired_concentration_differences(record, policy)
            print(f"  {policy}:")
            for name, values in distances["pairs"].items():
                label = name.replace("tv_", "").replace("_vs_", " vs ")
                print(f"    same-token TV {label}: {values['mean']:.6f} ± {values['sem']:.6f}")
            for pair, measures in differences["pairs"].items():
                support = measures["effective_support"]
                zero = measures["zero_frequency_fraction"]
                print(
                    f"    delta {pair}: N_eff {support['mean']:+,.2f} ± {support['sem']:,.2f}, "
                    f"zero-frac {zero['mean']:+.4f} ± {zero['sem']:.4f}"
                )

    if sweep_temperatures:
        from llm_behavior_lab.analysis import sweep_summary

        summary = sweep_summary(record)
        print("\nTemperature transition (real input, mean across initializations):")
        print(
            f"  {'T':>6} {'N_eff':>10} {'/null':>7} {'zero%':>7} {'TV(corp)':>9} "
            f"{'TVrk_greedy':>12} {'TVrk_unif':>10} {'agree':>7}"
        )
        metrics = summary["conditions"]["real"]["metrics"]
        for index, temperature in enumerate(sweep_temperatures):
            print(
                f"  {temperature:>6g} {metrics['effective_support']['mean'][index]:>10,.1f}"
                f" {metrics['effective_support_over_null']['mean'][index]:>7.3f}"
                f" {metrics['zero_frequency_fraction']['mean'][index]:>6.1%}"
                f" {metrics['tv_to_corpus']['mean'][index]:>9.4f}"
                f" {metrics['tv_rank_to_greedy']['mean'][index]:>12.4f}"
                f" {metrics['tv_rank_to_uniform']['mean'][index]:>10.4f}"
                f" {metrics['agreement_with_greedy']['mean'][index]:>7.3f}"
            )
        others = [name for name in summary["conditions"] if name != "real"]
        if others:
            print("  N_eff / N_eff(null) by input condition:")
            for name in ("real", *others):
                ratios = summary["conditions"][name]["metrics"]["effective_support_over_null"]["mean"]
                print(f"    {name:<9} " + "  ".join(f"{value:.3f}" for value in ratios))
        run.save_analysis_json("temperature_sweep_summary.json", summary)

    if record.has_predictive_probability_analysis:
        from llm_behavior_lab.analysis import predictive_probability_summary

        predictive = predictive_probability_summary(record)
        uniform = predictive["uniform_probability"]
        print(
            "\nRaw predictive distribution (T = 1, eligible support, before top-p "
            "and before any sampling decision):"
        )
        print(
            f"  ranked WITHIN each position, then averaged over positions "
            f"(not figure 1's across-position ranking)"
        )
        ranks = predictive["ranked_profile_at_rank"]
        print("  " + "  ".join(f"P({rank})={value:.4g}" for rank, value in ranks.items()))
        print(
            f"  uniform 1/K = {uniform:.4g};  rank 1 is "
            f"{predictive['rank1_over_uniform']:.2f}x uniform;  "
            f"profile dynamic range {predictive['profile_dynamic_range']:.4g}"
        )
        pooled = predictive["max_probability"]["pooled"]
        print(
            f"  p_max: min {pooled['p00']:.4g}, p05 {pooled['p05']:.4g}, "
            f"p25 {pooled['p25']:.4g}, median {pooled['p50']:.4g}, "
            f"p75 {pooled['p75']:.4g}, p95 {pooled['p95']:.4g}, "
            f"p99 {pooled['p99']:.4g}, max {pooled['p100']:.4g}"
        )
        print(
            f"  p_max mean {pooled['mean']:.4g} "
            f"({predictive['max_probability']['mean_over_uniform']:.2f}x uniform); "
            "greedy always takes the top token, which is not the same as that "
            "token carrying much mass"
        )
        target = predictive["target_probability"]
        print(
            f"  p_target mean {target['probability']['mean']:.4g}, "
            f"loss mean {target['loss']['mean']:.4f} "
            f"(uniform log K = {target['uniform_loss']:.4f})"
        )

    if gradient_result is not None:
        from llm_behavior_lab.analysis import gradient_guess_table

        table = gradient_guess_table(record)
        measured = table["target_occurrence_count"] > 0
        norms = table["mean_gradient_norm"][measured]
        print(
            f"\nPer-position parameter-gradient norms "
            f"(initialization {gradient_metadata['initialization_index']}, "
            f"seed {gradient_metadata['model_seed']}, real input, "
            f"{gradient_metadata['softmax_support']} support):"
        )
        print(
            f"  positions differentiated : {table['num_positions']:,}"
            f"  ({'all' if table['covers_all_positions'] else 'SUBSET of'} "
            f"{positions.num_positions:,})"
        )
        print(f"  parameters in the norm   : {gradient_metadata['parameter_count']:,}")
        print(f"  mean single-position loss: {gradient_metadata['mean_loss']:.6f}")
        print(
            f"  G_i over {int(measured.sum()):,} tokens with targets: "
            f"min {norms.min():.6g}, median {np.median(norms):.6g}, max {norms.max():.6g}"
        )
        print(
            f"  measurement time         : {gradient_result.seconds:,.1f}s "
            f"({table['num_positions'] / max(gradient_result.seconds, 1e-9):,.1f} positions/s)"
        )

    figure_paths: list[Path] = []
    if not args.no_figures:
        try:
            from llm_behavior_lab.analysis.figures import generate_all_figures
        except ImportError:
            print(
                "\nmatplotlib is not installed, so no figures were produced. "
                'Install it with: python3 -m pip install -e ".[analysis]"'
            )
        else:
            figure_paths = generate_all_figures(record, run.paths.run_dir / "figures")

    print("\nExperiment completed successfully.")
    print(f"Run directory: {run.paths.run_dir}")
    print(f"Record: {run.paths.analyses_dir / 'initialization_distribution.npz'}")
    for path in figure_paths:
        print(f"Figure: {path}")


if __name__ == "__main__":
    main()
