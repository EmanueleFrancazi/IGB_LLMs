"""Measure what per-position parameter-gradient norms actually cost.

This is a **performance benchmark**, not a scientific measurement. It builds the
same corpus, tokenizer, split, evaluation positions, and model architecture the
experiment uses, and times
:func:`~llm_behavior_lab.evaluation.position_gradients.compute_position_gradient_norms`
on a handful of windows. It writes no record and produces no figure.

Two things follow from it being a benchmark rather than an experiment.

The initialization seed is a knob, not a claim. Gradient-norm *timing* does not
depend on the weights, so this script does not assert that its model is the
experiment's first initialization; it prints the seed it used and leaves the
comparison at that.

Several window counts are timed rather than one, so the scaling can be *checked*
instead of assumed. The projected cost of a full run is printed as an
extrapolation and labelled as one.

A note on the memory columns. CUDA peaks can be reset between measurements, so
they are reported per row. Process RSS cannot: ``ru_maxrss`` is a high-water mark
for the whole process, so it is reported once, at the end, for the run as a
whole. Reporting it per row would produce a monotone column that looks like
per-size growth and is not.

Sketch memory, and why ``M`` is the axis that matters
-----------------------------------------------------

The CountSketch is off by default here, which keeps every historical invocation
of this script comparable. Enabling it with ``--gradient-sketch`` is what makes
the memory columns describe the configuration a campaign actually runs, because
the map tables are **device resident**:

.. code-block:: text

    per map, per parameter:  int32 bucket (4 B) + int8 sign (1 B) = 5 B
    device map memory     =  M * 5 * P bytes         <- scales with M, not K

``K`` changes the *width of the sketch output* and therefore the host-side
``[N_T, D, K]`` array and the analysis width; it does not change the map tables,
which have one entry per parameter regardless of how many buckets those entries
point into. So doubling ``K`` costs host storage, while adding a map replica
costs device memory.

``M > 1`` is now **measured rather than projected**. ``--sketch-maps`` is passed
straight through to
:func:`~llm_behavior_lab.evaluation.position_gradients.compute_position_gradient_norms`,
which builds the full bank and projects every gradient through all of it from one
backward pass, so the CUDA columns include whatever the replicas really cost --
tables and transients alike. The analytical figures printed beside them,

.. code-block:: text

    total device map tables       =  M * 5 * P bytes
    incremental against M = 1     =  (M - 1) * 5 * P bytes

are a **cross-check on the measurement, not a substitute for it**: they account
for the persistent tables only and say nothing about construction transients.
Where an earlier revision of this file projected the ``M > 1`` peak, read the
measured row instead.

None of this makes benchmark output scientific evidence. It is timing and memory
for a configuration; it writes no record and answers no research question.
"""

from __future__ import annotations

import argparse
import inspect
import resource
import sys
from pathlib import Path

import torch

# Allow running from the repository root before editable installation.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from llm_behavior_lab.data import (  # noqa: E402
    DatasetConfig,
    add_dataset_arguments,
    resolution_policy_from_args,
    resolve_dataset,
    split_token_ids,
)
from llm_behavior_lab.data.tokenizer import build_tokenizer  # noqa: E402
from llm_behavior_lab.evaluation.init_distribution import (  # noqa: E402
    build_evaluation_positions,
)
from llm_behavior_lab.evaluation.position_gradients import (  # noqa: E402
    DEFAULT_SKETCH_DIMENSION,
    GRADIENT_TEMPERATURES,
    compute_position_gradient_norms,
)

# Deliberately reaching for a private constant. This benchmark lives in the
# same repository as the estimator and exists to report what that estimator
# actually costs, so it tracks the production device representation rather
# than restating it -- a hard-coded copy here went stale once already. The
# device dtypes stay private because they are an implementation choice.
from llm_behavior_lab.evaluation.position_gradients import (  # noqa: E402
    _SKETCH_MAP_DEVICE_BYTES_PER_PARAMETER,
    _validated_map_count,
)
from llm_behavior_lab.models import build_model_from_config  # noqa: E402
from llm_behavior_lab.utils import get_device, load_yaml_config, seed_everything  # noqa: E402

#: The position count the extrapolation is quoted for: the standing protocol of
#: 512 windows x 64 tokens.
FULL_EXPERIMENT_POSITIONS = 32768

#: The production sketch seed, read from the production signature rather than
#: copied, so the benchmark cannot drift from the estimator it is timing.
DEFAULT_SKETCH_SEED = inspect.signature(
    compute_position_gradient_norms
).parameters["sketch_seed"].default


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark exact per-position parameter-gradient norms. Performance "
            "only: writes no record and draws no figure."
        )
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=REPO_ROOT / "configs" / "data" / "tiny_text.yaml",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=REPO_ROOT / "configs" / "model" / "tiny_llama.yaml",
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=REPO_ROOT / "configs" / "experiment" / "initialization_distribution.yaml",
    )
    parser.add_argument(
        "--window-counts",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        metavar="N",
        help=(
            "Window counts to time, in order. Several values let the scaling be "
            "checked rather than assumed from a single point."
        ),
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="Evaluation window length. Defaults to the experiment protocol.",
    )
    parser.add_argument(
        "--model-seed",
        type=int,
        default=None,
        help=(
            "Initialization seed. Timing does not depend on the weights; this "
            "defaults to the experiment config's initialization.base_seed only so "
            "the benchmark is reproducible."
        ),
    )
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=None,
        metavar="T",
        help=(
            "Loss temperatures to time. Defaults to the full grid. Pass 1.0 alone "
            "to time the canonical baseline for a like-for-like comparison."
        ),
    )
    parser.add_argument(
        "--split",
        choices=["train", "val"],
        default=None,
        help="Analysis split. Defaults to the experiment protocol.",
    )
    sketch = parser.add_argument_group("count sketch")
    sketch.add_argument(
        "--gradient-sketch",
        action="store_true",
        help=(
            "Also project every gradient through the production count sketch, as "
            "a scientific run does. Off by default, which keeps historical "
            "invocations of this benchmark comparable -- but the map tables are "
            "device resident, so the memory columns only describe a campaign "
            "configuration when this is on."
        ),
    )
    sketch.add_argument(
        "--sketch-dimension",
        type=int,
        default=None,
        metavar="K",
        help=(
            "Sketch width. Resolved like the runner does it: this flag, then the "
            "experiment config's gradient_analysis.sketch_dimension, then the "
            f"production default ({DEFAULT_SKETCH_DIMENSION})."
        ),
    )
    sketch.add_argument(
        "--sketch-maps",
        type=int,
        # None, not 1, for the same reason as --sketch-dimension above: only
        # ``is None`` can tell "not asked for" from "asked for explicitly", and
        # an explicit --sketch-maps 0 must be refused rather than silently
        # replaced by the default.
        default=None,
        metavar="M",
        help=(
            "Independent map replicas M, measured rather than projected. Every "
            "map projects the same gradient from one backward pass, so timing "
            "moves only by the projection work; the device-resident map tables "
            f"grow by {_SKETCH_MAP_DEVICE_BYTES_PER_PARAMETER} bytes per "
            "parameter per map. Requires --gradient-sketch above 1. Defaults to "
            "1, which is the historical measurement."
        ),
    )
    sketch.add_argument(
        "--sketch-seed",
        type=int,
        default=None,
        help=(
            "Seed for the production map construction. Benchmark-only; the "
            "runner has no such option and always uses the production default "
            f"({DEFAULT_SKETCH_SEED}). Timing does not depend on it."
        ),
    )
    parser.add_argument(
        "--allow-narrow-vocabulary",
        action="store_true",
        help=(
            "Permit a tokenizer narrower than the model's output head. Refused "
            "by default: logits are truncated to the tokenizer vocabulary before "
            "the loss, so the output-layer gradient -- the dominant cost -- is "
            "measured over a fraction of the head and the result is not a "
            "production cost estimate. Required by the legacy tiny-character "
            "pairing, which must acknowledge what it is measuring."
        ),
    )
    add_dataset_arguments(parser)
    return parser.parse_args()


def _resolve_sketch(args: argparse.Namespace, experiment_config: dict) -> dict:
    """Resolve the sketch configuration, in the runner's own precedence order.

    The runner reads ``gradient_analysis.sketch`` and
    ``gradient_analysis.sketch_dimension`` from the experiment config and lets
    the command line override the width, so the same vocabulary is used here
    rather than a second one invented for the benchmark:

    .. code-block:: text

        enabled : --gradient-sketch  OR  gradient_analysis.sketch
        K       : --sketch-dimension  >  gradient_analysis.sketch_dimension  >  default
        seed    : --sketch-seed       >  production default
        M       : --sketch-maps       >  gradient_analysis.sketch_maps       >  1

    ``M`` reads the experiment config for the same reason ``K`` does. This script
    exists to report what a campaign configuration costs, and the map tables are
    device resident -- so a config that asks for four maps must not be timed at
    one here, which would understate device memory by exactly the quantity the
    benchmark was run to find.

    Raises:
        ValueError: If ``K`` is not positive, if ``M`` is not an integer of at
            least one, or if ``M > 1`` is requested with the sketch off -- in
            which case no map would be built at all and the measurement would
            silently be of something else.
    """

    gradients = experiment_config.get("gradient_analysis", {}) or {}

    enabled = bool(args.gradient_sketch) or bool(gradients.get("sketch", False))
    # ``is None``, not ``or``: an explicit ``--sketch-dimension 0`` is a mistake
    # worth reporting, and ``or`` would silently replace it with the default.
    dimension = int(
        args.sketch_dimension
        if args.sketch_dimension is not None
        else gradients.get("sketch_dimension", DEFAULT_SKETCH_DIMENSION)
    )
    seed = int(args.sketch_seed if args.sketch_seed is not None else DEFAULT_SKETCH_SEED)
    # The estimator's own rule, not a second opinion about it, so this parser
    # cannot accept a count the measurement below would refuse.
    maps = _validated_map_count(
        args.sketch_maps
        if args.sketch_maps is not None
        else gradients.get("sketch_maps", 1),
        name="--sketch-maps",
    )

    if dimension < 1:
        raise ValueError(f"--sketch-dimension must be positive; got {dimension}.")
    if maps != 1 and not enabled:
        raise ValueError(
            f"--sketch-maps {maps} was requested but the count sketch is off, so "
            "no map would be built and the memory columns would describe a "
            "single-map run under a label that says "
            f"{maps}. Pass --gradient-sketch, or leave --sketch-maps at 1."
        )

    return {"enabled": enabled, "dimension": dimension, "maps": maps, "seed": seed}


def _check_vocabulary_match(
    tokenizer, model_params: dict, *, allow_narrow: bool = False
) -> str | None:
    """Refuse any vocabulary pairing whose timing would not be a cost estimate.

    Two of the three conditions are the experiment runner's own, stated as
    properties of the tokenizer rather than of any model family: a vocabulary
    wider than the model's output is impossible, and a *pretrained* vocabulary
    must match the model exactly.

    The third is stricter than the runner, deliberately. The runner allows a
    corpus-derived vocabulary to be **narrower** than the model -- its size
    depends on the text -- and for a scientific run that is a legitimate
    configuration. For a timing run it is a trap: logits are truncated to the
    tokenizer's vocabulary before the loss is taken, so a 32000-row output head
    measured through a ~150-symbol character vocabulary backpropagates into
    about half a percent of the head. A benchmark result is used as a production
    cost estimate, so a warning is not enough; narrowing is refused unless the
    caller says ``allow_narrow``, and even then the header states plainly that
    the number is not representative of the full output head.

    Args:
        tokenizer: The built tokenizer.
        model_params: The model config's ``params`` mapping.
        allow_narrow: Permit a narrower corpus-derived vocabulary, from
            ``--allow-narrow-vocabulary``. The legacy tiny-character pairing
            needs it, which is the point: it has to be asked for.

    Returns:
        A warning to print in the header when narrowing was permitted, or
        ``None`` when the vocabularies match exactly.

    Raises:
        ValueError: On either runner condition, or on narrowing without
            ``allow_narrow``.
    """

    model_vocab_size = int(model_params["vocab_size"])
    description = tokenizer.describe()

    if tokenizer.vocab_size > model_vocab_size:
        raise ValueError(
            f"Tokenizer vocab size {tokenizer.vocab_size} exceeds model vocab size "
            f"{model_vocab_size}. Increase model.params.vocab_size."
        )
    if description.get("type") == "pretrained" and tokenizer.vocab_size != model_vocab_size:
        raise ValueError(
            f"Tokenizer {description.get('identifier')!r} has vocabulary "
            f"{tokenizer.vocab_size} but the model config declares "
            f"{model_vocab_size}. A pretrained tokenizer requires an exact match."
        )

    if tokenizer.vocab_size == model_vocab_size:
        return None

    share = tokenizer.vocab_size / model_vocab_size
    if not allow_narrow:
        raise ValueError(
            f"The tokenizer supplies {tokenizer.vocab_size:,} of the model's "
            f"{model_vocab_size:,} output rows ({share:.1%}). Logits are truncated "
            "to the tokenizer vocabulary before the loss, so the output-layer "
            "gradient -- the dominant cost -- would be measured over a fraction "
            "of the head and the result would not be a production cost estimate. "
            "Pair the model with a tokenizer of matching size, or pass "
            "--allow-narrow-vocabulary to measure the reduced output head "
            "deliberately."
        )
    return (
        f"WARNING: --allow-narrow-vocabulary is in effect. The tokenizer supplies "
        f"{tokenizer.vocab_size:,} of the model's {model_vocab_size:,} output rows "
        f"({share:.1%}). This timing is NOT representative of the full output "
        "head and must not be used as a production cost estimate."
    )


def _cuda_peaks(device: torch.device) -> tuple[float, float] | None:
    """Return (allocated, reserved) CUDA peaks in MiB, or ``None`` on CPU."""

    if device.type != "cuda":
        return None
    return (
        torch.cuda.max_memory_allocated(device) / (1024**2),
        torch.cuda.max_memory_reserved(device) / (1024**2),
    )


def main() -> None:
    """Time the gradient measurement at several window counts."""

    args = parse_args()
    data_config = load_yaml_config(args.data_config)
    model_config = load_yaml_config(args.model_config)
    experiment_config = load_yaml_config(args.experiment_config)

    # Resolved here rather than beside its first use further down. An
    # unusable count -- zero maps, or replicas with the sketch off -- is a
    # command-line mistake, and the run should end on it before the dataset is
    # resolved, the tokenizer built and the corpus encoded, not after.
    sketch = _resolve_sketch(args, experiment_config)

    temperatures = tuple(args.temperatures) if args.temperatures else GRADIENT_TEMPERATURES
    window_counts = sorted(set(int(value) for value in args.window_counts))
    if not window_counts or window_counts[0] <= 0:
        raise ValueError("--window-counts must be positive integers.")

    positions_config = experiment_config.get("positions", {})
    block_size = int(
        args.block_size
        or positions_config.get("block_size")
        or data_config["batching"]["block_size"]
    )
    split_name = args.split or str(positions_config.get("split", "train"))
    model_seed = int(
        args.model_seed
        if args.model_seed is not None
        else experiment_config.get("initialization", {}).get("base_seed", 1000)
    )

    device = get_device(str(data_config.get("runtime", {}).get("device", "auto")))

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
    split_ids = splits.train_ids if split_name == "train" else splits.val_ids

    positions = build_evaluation_positions(
        split_ids,
        block_size=block_size,
        num_windows=max(window_counts),
        device=device,
    )

    vocabulary_warning = _check_vocabulary_match(
        tokenizer,
        model_config["model"]["params"],
        allow_narrow=args.allow_narrow_vocabulary,
    )

    seed_everything(model_seed)
    model = build_model_from_config(model_config).to(device)
    parameter_count = model.count_parameters()
    num_parameter_tensors = len(list(model.parameters()))
    before = {name: tensor.detach().clone() for name, tensor in model.named_parameters()}

    print("Per-position gradient benchmark -- PERFORMANCE ONLY, not a measurement")
    print(f"Dataset: {resolved.name} ({resolved.source}, via {resolved.route})")
    print(f"Tokenizer vocabulary: {tokenizer.vocab_size:,}")
    print(f"Model: {model_config['model']['name']}, seed {model_seed} (timing is seed-independent)")
    print(f"Parameters in the norm: {parameter_count:,} across {num_parameter_tensors} tensors")
    print(f"Device: {device}")
    print(f"Block size: {block_size} -> {block_size} positions per window")
    print(
        f"Loss temperatures: {len(temperatures)} "
        f"({', '.join(f'{value:g}' for value in temperatures)})"
    )
    print(
        "One backward pass per position PER TEMPERATURE, from a single retained "
        "forward graph per window."
    )
    if sketch["enabled"]:
        # Per map, per parameter. Total is what the run holds; incremental is
        # what the replicas cost against the single-map configuration every
        # earlier measurement was taken at, which is the number a memory gate is
        # actually read against.
        per_map_bytes = _SKETCH_MAP_DEVICE_BYTES_PER_PARAMETER * parameter_count
        map_bytes = sketch["maps"] * per_map_bytes
        incremental_bytes = (sketch["maps"] - 1) * per_map_bytes
        print(
            f"Count sketch: ON  K = {sketch['dimension']}, M = {sketch['maps']}, "
            f"seed {sketch['seed']}"
        )
        print(
            f"  device map tables: {sketch['maps']} x {_SKETCH_MAP_DEVICE_BYTES_PER_PARAMETER} B "
            f"x {parameter_count:,} "
            f"parameters = {map_bytes / (1024 ** 2):,.1f} MiB, included in the "
            "CUDA columns below"
        )
        print(
            f"  incremental against M = 1: {incremental_bytes / (1024 ** 2):,.1f} MiB "
            "of persistent tables. Analytical: it counts the tables only, and the "
            "measured columns below are what actually happened."
        )
    else:
        print(
            "Count sketch: OFF (default). The device-resident map tables are not "
            "allocated and the projection is not timed, so the CUDA columns "
            "below UNDERSTATE a campaign configuration by about "
            f"{_SKETCH_MAP_DEVICE_BYTES_PER_PARAMETER * parameter_count / (1024 ** 2):,.1f}"
            " MiB. Pass --gradient-sketch to measure it."
        )
    if vocabulary_warning is not None:
        print()
        print(vocabulary_warning)
    print()

    header = (
        f"{'windows':>8} {'positions':>10} {'seconds':>10} {'positions/s':>12}"
        f" {'backwards/s':>12}"
    )
    if device.type == "cuda":
        header += f" {'CUDA alloc':>12} {'CUDA resvd':>12}"
    print(header)

    measurements: list[tuple[int, int, float, float]] = []
    for count in window_counts:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        result = compute_position_gradient_norms(
            model,
            positions,
            vocab_size=tokenizer.vocab_size,
            eligible_token_ids=tokenizer.eligible_token_ids,
            num_windows=count,
            temperatures=temperatures,
            gradient_sketch=sketch["enabled"],
            sketch_dimension=sketch["dimension"],
            sketch_seed=sketch["seed"],
            sketch_maps=sketch["maps"],
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if sketch["enabled"]:
            # Asserted rather than assumed: the map tables are built inside this
            # call, so a wrong shape here would mean the peaks above described
            # something other than the requested configuration.
            #
            # The replica axis is present only above one map, matching the
            # estimator's rank-conditional contract exactly. Writing the M = 1
            # shape as [.., 1, K] here would pass for the wrong reason.
            expected = (
                (len(temperatures), result.num_positions, sketch["dimension"])
                if sketch["maps"] == 1
                else (
                    len(temperatures),
                    result.num_positions,
                    sketch["maps"],
                    sketch["dimension"],
                )
            )
            if tuple(result.temperature_gradient_sketches.shape) != expected:
                raise RuntimeError(
                    "Sketch shape "
                    f"{tuple(result.temperature_gradient_sketches.shape)} does not "
                    f"match the requested {expected}."
                )
            if len(result.sketch_tensor_sizes) != num_parameter_tensors:
                raise RuntimeError(
                    f"The sketch covered {len(result.sketch_tensor_sizes)} tensors "
                    f"but the model has {num_parameter_tensors}."
                )
        rate = result.num_positions / max(result.seconds, 1e-9)
        measurements.append((count, result.num_positions, result.seconds, rate))

        row = (
            f"{count:>8} {result.num_positions:>10,} "
            f"{result.seconds:>10.3f} {rate:>12,.2f}"
            f" {rate * len(temperatures):>12,.2f}"
        )
        peaks = _cuda_peaks(device)
        if peaks is not None:
            row += f" {peaks[0]:>10,.1f}MiB {peaks[1]:>10,.1f}MiB"
        print(row)

    after = {name: tensor.detach() for name, tensor in model.named_parameters()}
    unchanged = all(torch.equal(after[name], value) for name, value in before.items())

    # ru_maxrss is a process-level high-water mark and cannot be reset between the
    # measurements above, so it describes the whole run, not any single row.
    peak_rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print()
    print(f"Process peak RSS (high-water mark over the whole benchmark, not per-size): {peak_rss_mib:,.1f} MiB")
    print(f"Model parameters unchanged after the benchmark: {unchanged}")

    print("\nObserved scaling (seconds relative to the smallest window count):")
    base_count, _, base_seconds, _ = measurements[0]
    for count, _, seconds, _ in measurements:
        expected = count / base_count
        observed = seconds / max(base_seconds, 1e-9)
        print(
            f"  {count:>4} windows: {observed:>6.2f}x time for {expected:>6.2f}x work "
            f"({'linear' if abs(observed - expected) <= 0.25 * expected else 'NON-LINEAR'})"
        )

    _, _, _, best_rate = measurements[-1]
    projected = FULL_EXPERIMENT_POSITIONS / max(best_rate, 1e-9)
    print()
    print("EXTRAPOLATION (measured positions/s from the largest window count,")
    print("assumes linear scaling in positions -- this figure was NOT measured):")
    print(
        f"  D = {FULL_EXPERIMENT_POSITIONS:,} positions x {len(temperatures)} "
        f"temperatures -> {projected:,.0f} s ({projected / 60:,.1f} min) on {device}"
    )
    print(
        f"  per-temperature share: {projected / len(temperatures):,.0f} s "
        f"({projected / len(temperatures) / 60:,.1f} min)"
    )


if __name__ == "__main__":
    main()
