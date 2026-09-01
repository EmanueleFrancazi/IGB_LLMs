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
import dataclasses
import math
import resource
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Allow running from the repository root before editable installation.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from llm_behavior_lab.analysis import (  # noqa: E402
    DEFAULT_NULL_REPLICATES,
    RECORD_VERSION,
    InitializationExperimentRecord,
    simulate_uniform_null,
    sampling_adequacy,
    summarize_policy,
    within_initialization_sampling_spread,
)
from llm_behavior_lab.analysis.records import STORAGE_MODES  # noqa: E402
from llm_behavior_lab.analysis.reporting import (  # noqa: E402
    report_gradient_norms,
    report_input_structure,
    report_policy_summaries,
    report_predictive_probabilities,
    report_runtime,
    report_sampling_adequacy,
    report_sampling_variability,
    report_support,
    report_temperature_sweep,
    report_uniform_null,
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
    CANONICAL_GRADIENT_TEMPERATURE,
    GRADIENT_TEMPERATURES,
    DEFAULT_SKETCH_DIMENSION,
    compute_position_gradient_norms,
)

# Deliberately reaching for a private helper, the same way the benchmark reaches
# for the private device-bytes constant. What counts as a legal map count is the
# measurement's rule, and the CLI exists to refuse the illegal ones *early*
# rather than to invent a second opinion about them: a value this parser accepted
# and the estimator then rejected would fail hours into a run, and a value the
# parser rejected and the estimator would have accepted is a CLI that cannot
# reach its own measurement.
from llm_behavior_lab.evaluation.position_gradients import (  # noqa: E402
    _validated_map_count,
)
from llm_behavior_lab.experiment import ExperimentRun, experiment_settings_from_config  # noqa: E402
from llm_behavior_lab.experiment.naming import compose_run_id  # noqa: E402
from llm_behavior_lab.models import build_model_from_config  # noqa: E402
from llm_behavior_lab.models.initialization_scale import (  # noqa: E402
    initialization_scale_report,
    scale_initialization,
)
from llm_behavior_lab.utils import (  # noqa: E402
    describe_device,
    get_device,
    load_yaml_config,
    seed_everything,
)


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
        "--gradient-vector-split",
        action="store_true",
        help=(
            "Also accumulate the summed parameter gradients of correctly and "
            "incorrectly assigned positions at T = 1, reporting their norms, dot "
            "product and cosine. Adds no backward pass; holds two float64 "
            "parameter-shaped buffers (about 131 MiB at 8.6M parameters)."
        ),
    )
    parser.add_argument(
        "--countsketch-fidelity-sanity",
        action="store_true",
        help=(
            "Small methodological run only. Retain the complete gradients of a "
            "deterministic handful of positions, compare the production "
            "CountSketch against their exact cosines, and write a compact "
            "sanity artifact. Never enable this for a scientific run."
        ),
    )
    parser.add_argument(
        "--gradient-sketch",
        action="store_true",
        help=(
            "Also record a deterministic count sketch of every position's "
            "gradient at every measured loss temperature, so directional "
            "clustering by token subgroup can be measured afterwards. Adds no "
            "backward pass; persists one compact vector per position and "
            "temperature (about 470 MB at D = 32768, K = 512, 7 temperatures)."
        ),
    )
    parser.add_argument(
        "--sketch-dimension",
        type=int,
        # None, not the width itself: the resolution below distinguishes "not
        # asked for" from "asked for explicitly", which a default of 512 cannot.
        default=None,
        metavar="K",
        help=(
            "Width of that sketch. Omitted, the experiment config's "
            "gradient_analysis.sketch_dimension applies, then the default "
            f"({DEFAULT_SKETCH_DIMENSION})."
        ),
    )
    parser.add_argument(
        "--sketch-storage",
        choices=STORAGE_MODES,
        default=None,
        help=(
            "What a run keeps of its per-position sketches. 'metrics_only' (the "
            "default) finalizes every declared statistic and then retains "
            "neither rows nor factors -- about 2 MiB per arm instead of 3.5 GiB. "
            "'per_position' keeps the full exploratory row surface at that cost "
            "and is bounded by --sketch-storage-max-bytes. All three modes "
            "finalize metrics; the mode says what was ADDITIONALLY kept."
        ),
    )
    parser.add_argument(
        "--sketch-storage-max-bytes",
        type=int,
        default=512 * 1024 * 1024,
        help=(
            "Ceiling on retained per-position sketches. A production arm is "
            "about 3.5 GiB, so the default refuses one outright: keeping rows at "
            "experiment scale is a deliberate diagnostic choice, not a default."
        ),
    )
    parser.add_argument(
        "--fidelity-max-positions",
        type=int,
        default=12,
        help="Bound on retained sanity positions.",
    )
    parser.add_argument(
        "--fidelity-max-gradient-bytes",
        type=int,
        default=1024 ** 3,
        help=(
            "Bound on the retained complete gradients. Independent of the "
            "position bound: a count says nothing about a 134M-parameter model."
        ),
    )
    parser.add_argument(
        "--fidelity-max-sketch-bytes",
        type=int,
        default=256 * 1024 * 1024,
        help=(
            "Bound on the retained sketches. Independent again: a gradient "
            "budget says nothing about a wide sketch at many maps."
        ),
    )
    parser.add_argument(
        "--sketch-factors",
        default=None,
        help=(
            "Which subgroup gradient-sketch factors compact_factors retains, "
            "e.g. 'target@1.0,greedy@1.0,cross@1.0,nucleus:0.6@0.6'. Explicit "
            "by design: retaining factors decides which questions stay askable "
            "afterwards, so there is no wildcard and no default set."
        ),
    )
    parser.add_argument(
        "--sketch-factors-max-bytes",
        type=int,
        default=512 * 1024 * 1024,
        help="Ceiling on the retained factor arrays.",
    )
    parser.add_argument(
        "--gradient-temp-dir",
        default=None,
        help=(
            "Where the temporary sketch slabs live. The durable manifest always "
            "stays in the run directory, so losing node-local scratch remains "
            "diagnosable from the run alone."
        ),
    )
    parser.add_argument(
        "--gradient-temp-max-bytes",
        type=int,
        default=8 * 1024 ** 3,
        help="Ceiling on the temporary sketch store.",
    )
    parser.add_argument(
        "--sketch-maps",
        type=int,
        # None for the same reason as --sketch-dimension above: the resolution
        # must be able to tell "not asked for" from "asked for explicitly".
        default=None,
        metavar="M",
        help=(
            "Number of independent CountSketch maps M to project every gradient "
            "through. Omitted, the experiment config's "
            "gradient_analysis.sketch_maps applies, then the default (1), which "
            "is exactly the historical single-map measurement. Above one, each "
            "gradient is projected through every map from the SAME backward "
            "pass, so the backward-pass count does not change, and the persisted "
            "sketch arrays gain a replica axis. Requires --gradient-sketch. Each "
            "extra map costs device-resident map tables and widens the record."
        ),
    )
    parser.add_argument(
        "--gradient-temperatures",
        type=float,
        nargs="+",
        default=None,
        metavar="T",
        help=(
            "Loss temperatures T_g to measure gradient fields at, in the order "
            "they are measured and persisted. Defaults to the established grid "
            "0.12 0.24 0.36 0.48 0.60 1.00 1.20. Duplicates collapse; the "
            "canonical T = 1 is added if omitted, since the record's canonical "
            "norm and sketch fields are that row. This selects what is "
            "MEASURED; which (T_s, T_g) pairs are analysed is chosen later by "
            "scripts/write_nucleus_clustering_artifact.py."
        ),
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


def _resolve_sketch_dimension(
    args: argparse.Namespace, gradients: dict[str, Any]
) -> int:
    """Resolve the sketch width: command line, then config, then the default.

    ``None``-based, not truthiness-based. The previous ``or`` chain had two
    defects at once. An explicit ``--sketch-dimension 0`` is falsy, so it fell
    through and silently became 512 instead of being refused -- a run would then
    have recorded a width nobody asked for. And because the flag's own default
    used to be 512 rather than ``None``, the left operand was *always* truthy,
    which made ``gradient_analysis.sketch_dimension`` dead configuration that
    could never take effect. Neither is reachable now.

    Raises:
        ValueError: If the resolved width is not positive. A sketch of zero or
            negative buckets is not a narrower measurement, it is not a
            measurement.
    """

    requested = getattr(args, "sketch_dimension", None)
    dimension = int(
        requested
        if requested is not None
        else gradients.get("sketch_dimension", DEFAULT_SKETCH_DIMENSION)
    )
    if dimension < 1:
        raise ValueError(
            f"sketch_dimension must be a positive number of buckets; got {dimension}."
        )
    return dimension


def _resolve_sketch_maps(args: argparse.Namespace, gradients: dict[str, Any]) -> int:
    """Resolve the map count ``M``: command line, then config, then the default.

    Same shape as :func:`_resolve_sketch_dimension`, and ``None``-based for the
    same reason: only ``is None`` distinguishes "not asked for" from "asked for
    explicitly", and truthiness cannot -- an explicit ``--sketch-maps 0`` is
    falsy and an ``or`` chain would silently hand back a single map instead of
    refusing a count nobody can measure.

    The default is 1, so a command line that says nothing about maps measures
    exactly what it always did, and ``--sketch-maps 1`` is indistinguishable
    from omitting the flag.

    Validation is the measurement's own, via ``_validated_map_count``, so the
    CLI cannot accept a value the estimator would refuse or refuse one it would
    accept. **The count is never inferred** -- not from the sketch width, not
    from an array's rank.

    Raises:
        ValueError: If the resolved count is boolean, non-integral, or below one.
    """

    requested = getattr(args, "sketch_maps", None)
    return _validated_map_count(
        requested if requested is not None else gradients.get("sketch_maps", 1),
        name="--sketch-maps",
    )


def _validate_sketch_map_request(protocol: dict[str, Any]) -> None:
    """Refuse a map count the rest of the run cannot honour, before it starts.

    Both refusals below would eventually happen anyway -- the estimator raises on
    the first, and the Stage 8b2-c writer guard raises on the second -- but they
    would happen *after* the dataset, the tokenizer, the model and, in the second
    case, the entire gradient measurement. At campaign scale that is hours spent
    reaching a guaranteed failure, so they are hoisted to argument resolution,
    which runs before anything is loaded or built. The downstream guards stay
    exactly where they are: this is the early exit, not a replacement for them.
    """

    map_count = protocol["sketch_maps"]
    if map_count == 1:
        return

    if not protocol["gradient_sketch"]:
        raise ValueError(
            f"--sketch-maps {map_count} was requested but the gradient sketch is "
            "off, so no map would be built at all. Pass --gradient-sketch (or set "
            "gradient_analysis.sketch in the experiment config), or leave "
            "--sketch-maps at 1."
        )

    # The M>1 refusal is gone: the check now compares against the sketches every
    # production map actually produced, captured live during the measurement,
    # so it validates the ensemble estimator the figures use rather than map 0
    # alone.


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

    protocol = {
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
        # A config written before this analysis existed simply does not carry it,
        # and neither does a namespace built before the flag was added -- so the
        # attribute is read defensively, the same way initialization_scale is.
        # Reaching for it directly turns every older caller into an
        # AttributeError that has nothing to do with what it was doing.
        "countsketch_fidelity_sanity": bool(
            getattr(args, "countsketch_fidelity_sanity", False)
        ),
        "gradient_sketch": (
            bool(gradients.get("sketch", False))
            or bool(getattr(args, "gradient_sketch", False))
        ),
        "sketch_dimension": _resolve_sketch_dimension(args, gradients),
        # Beside the width, resolved the same way, and carried at every M
        # including 1 -- a run that measured one map should say so explicitly
        # rather than leave it to be inferred from an array's shape.
        "sketch_maps": _resolve_sketch_maps(args, gradients),
        # metrics_only is the production default: every declared statistic is
        # finalized while the rows exist, and then the rows go. A mode chosen on
        # the command line wins over the config, as every other sketch knob does.
        "sketch_storage": (
            args.sketch_storage
            if getattr(args, "sketch_storage", None) is not None
            else gradients.get("sketch_storage", "metrics_only")
        ),
        "sketch_storage_max_bytes": int(
            getattr(args, "sketch_storage_max_bytes", 512 * 1024 * 1024)
        ),
        "fidelity_max_positions": int(getattr(args, "fidelity_max_positions", 12)),
        "fidelity_max_gradient_bytes": int(
            getattr(args, "fidelity_max_gradient_bytes", 1024 ** 3)
        ),
        "fidelity_max_sketch_bytes": int(
            getattr(args, "fidelity_max_sketch_bytes", 256 * 1024 * 1024)
        ),
        "sketch_factors": getattr(args, "sketch_factors", None),
        "sketch_factors_max_bytes": int(
            getattr(args, "sketch_factors_max_bytes", 512 * 1024 * 1024)
        ),
        "gradient_temp_dir": getattr(args, "gradient_temp_dir", None),
        "gradient_temp_max_bytes": int(
            getattr(args, "gradient_temp_max_bytes", 8 * 1024 ** 3)
        ),
        "gradient_vector_split": (
            bool(gradients.get("vector_split", False))
            or bool(getattr(args, "gradient_vector_split", False))
        ),
        "gradient_analysis_enabled": (
            (bool(gradients.get("enabled", False)) or args.gradient_analysis)
            and not args.no_gradient_analysis
        ),
        # Absent from a namespace built before this option existed means the
        # historical condition, which is exactly alpha = 1: the scale intervention
        # is a no-op there. The fallback must match the parser default above, and
        # a test pins the two together.
        "initialization_scale": float(getattr(args, "initialization_scale", 1.0)),
        "gradient_initialization_index": int(gradients.get("initialization_index", 0)),
        "gradient_num_windows": (
            args.gradient_windows
            if args.gradient_windows is not None
            else gradients.get("num_windows")
        ),
        # Which LOSS temperatures the gradient fields are measured at. Selecting
        # (T_s, T_g) pairs from what was measured is the paired nucleus
        # analysis's job, not this one's.
        "gradient_temperatures": _resolve_gradient_temperatures(
            getattr(args, "gradient_temperatures", None),
            gradients.get("temperatures"),
        ),
    }
    # Cross-option checks belong here rather than in main(): every caller that
    # resolves a protocol -- the runner, and the tests that resolve one directly
    # -- gets the same refusal, and main() reaches this before it loads a dataset
    # or builds a model.
    if protocol["sketch_storage"] == "compact_factors" and not protocol[
        "sketch_factors"
    ]:
        raise ValueError(
            "compact_factors retains an explicitly declared factor set, so "
            "--sketch-factors is required. Retaining nothing is what "
            "metrics_only already does."
        )
    if protocol["sketch_storage"] != "compact_factors" and protocol[
        "sketch_factors"
    ]:
        raise ValueError(
            "--sketch-factors only applies to --sketch-storage compact_factors; "
            "no other mode persists subgroup factors."
        )
    if protocol["sketch_storage"] not in STORAGE_MODES:
        raise ValueError(
            f"Unknown --sketch-storage {protocol['sketch_storage']!r}; expected "
            f"one of {', '.join(STORAGE_MODES)}."
        )
    _validate_sketch_map_request(protocol)
    return protocol


def _resolve_gradient_temperatures(
    requested: Any, from_config: Any = None
) -> tuple[float, ...]:
    """The loss temperatures to measure gradient fields at.

    Omitted keeps the established seven-value grid, so a run that says nothing
    about temperatures measures exactly what it always did.

    Precedence is resolved here, on ``None`` rather than on truthiness. An
    explicitly empty request is a mistake worth reporting, and ``requested or
    from_config`` would silently turn it into "nothing was asked for" and hand
    back the default grid -- which is exactly what it did.

    The canonical ``T = 1`` is required rather than optional: the record's
    canonical norm and sketch fields are that row, and figure 20 and the vector
    split are defined on it. A grid that leaves it out is completed rather than
    rejected, since the omission is far more likely a slip than a request to
    abandon every canonical analysis.

    Duplicates collapse to their first occurrence, and the surviving order is
    the order measured and persisted.
    """

    if requested is None:
        requested = from_config
    if requested is None:
        return GRADIENT_TEMPERATURES

    values = [float(value) for value in requested]
    if not values:
        raise ValueError("--gradient-temperatures must not be empty.")
    for value in values:
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"Every gradient temperature must be finite and positive; got "
                f"{value:g}. Softmax at T = 0 is undefined, and greedy is the "
                "T -> 0 limit rather than a temperature that can be measured."
            )
    grid = tuple(dict.fromkeys(values))
    if CANONICAL_GRADIENT_TEMPERATURE not in grid:
        grid = grid + (CANONICAL_GRADIENT_TEMPERATURE,)
    return grid


def _measured_map_count(gradient_result) -> int:
    """How many production maps the measurement actually built.

    Read from the protocol the measurement wrote, never from the request that
    asked for it and never from an array's rank. The request is what someone
    typed; this is what happened, and only the second belongs in a record or in
    a decision about how to store one.

    A missing key means an in-memory result from before the map count was
    recorded. Those were single-map by construction, so they read as ``M = 1``.
    """

    protocol = getattr(gradient_result, "sketch_protocol", None) or {}
    return int(protocol.get("map_count", 1))


def _write_countsketch_fidelity(run, gradient_result, protocol) -> None:
    """Run the offline fidelity analysis and persist only its derived results.

    The complete gradients exist solely for the duration of this call. They are
    hundreds of megabytes and nothing downstream needs them once the exact
    cosines, the production estimate, the alternate seeds and the K sweep have
    been computed, so they are dropped here rather than written to disk.
    """

    import numpy as np

    from llm_behavior_lab.analysis.countsketch_fidelity import fidelity_report
    from llm_behavior_lab.evaluation.position_gradients import production_sketch_map

    # Map-count gate, first: before any reconstruction and before the artifact
    # directory is touched, so a refusal cannot leave a half-written fidelity
    # result behind.
    #
    # At M > 1 the check reads the sketches the production maps actually
    # produced, captured live beside the exact gradients, so there is no
    # reconstruction to be map-0 or two-dimensional about. What used to be
    # refused here is now the ordinary path.
    fidelity_map_count = _measured_map_count(gradient_result)

    gradients = gradient_result.exact_gradients.numpy()
    indices = gradient_result.exact_positions.numpy()
    lookup = {int(value): row for row, value in enumerate(
        gradient_result.position_indices.numpy()
    )}
    rows = [lookup[int(index)] for index in indices]
    targets = gradient_result.target_ids.numpy()[rows]
    greedy = gradient_result.greedy_ids.numpy()[rows]
    norms = gradient_result.gradient_norms.numpy()[rows]

    # Integrity gate. A mismatch here means the flattening order, the position
    # alignment or the capture point is wrong, and every fidelity number that
    # followed would be meaningless.
    recomputed = np.linalg.norm(gradients.astype(np.float64), axis=1)
    drift = float(np.max(np.abs(recomputed - norms) / np.maximum(norms, 1e-30)))
    if drift > 1e-5:
        raise ValueError(
            f"Recomputed norms disagree with the recorded exact norms by "
            f"{drift:.3e}; capture alignment is wrong, so fidelity cannot be "
            "interpreted."
        )

    if gradient_result.sketch_map is None:
        raise ValueError(
            "The fidelity check needs the production sketch map, so the run must "
            "also enable the gradient sketch."
        )
    buckets, signs = gradient_result.sketch_map
    sketch_rows = gradient_result.gradient_sketches.numpy()[rows]
    rebuilt = np.zeros_like(sketch_rows, dtype=np.float64)
    for row in range(gradients.shape[0]):
        np.add.at(rebuilt[row], buckets, gradients[row].astype(np.float64) * signs)
    scale = np.maximum(np.abs(sketch_rows).max(), 1e-30)
    sketch_drift = float(np.max(np.abs(rebuilt - sketch_rows)) / scale)
    if sketch_drift > 1e-4:
        raise ValueError(
            f"The production sketch could not be reconstructed from the captured "
            f"gradients (relative drift {sketch_drift:.3e}); flattening order, "
            "map or alignment is wrong."
        )

    report = fidelity_report(
        gradients, norms, targets, greedy,
        production_dimension=protocol["sketch_dimension"],
        production_seed=20240917,
        production_map=gradient_result.sketch_map,
        # Alternate seeds and the K sweep must be other realizations of the
        # production construction, not of a different RNG, or they answer a
        # question nobody asked.
        map_factory=lambda dimension, seed: production_sketch_map(
            gradient_result.sketch_tensor_sizes, dimension, seed
        ),
    )
    production = report["production"]
    print(
        f"    countsketch fidelity: {report['num_gradients']} gradients, "
        f"{production['num_pairs']} pairs, MAE {production['mean_absolute_error']:.5f}, "
        f"RMSE {production['rmse']:.5f}, bias {production['mean_signed_error']:+.5f}, "
        f"norm drift {drift:.2e}, sketch drift {sketch_drift:.2e}"
    )
    counts = "  ".join(
        f"{name}={entry['num_pairs']}" for name, entry in report["pair_types"].items()
    )
    print(f"      pair types: {counts}")
    for name, entry in report["subgroup_deltas"].items():
        if not (entry["exact"]["available"] and entry["sketch"]["available"]):
            print(f"      {name} subgroup delta: unavailable on selected subset")
            continue
        print(
            f"      {name} subgroup: exact delta {entry['exact']['delta']:+.5f}  "
            f"sketch delta {entry['sketch']['delta']:+.5f}  "
            f"delta error {entry['delta_error']:+.5f}"
        )
    for name, errors in report["alternate_delta_errors"].items():
        if errors:
            print(
                f"      {name} delta error across {len(errors)} alternate seeds: "
                f"[{min(errors):+.5f}, {max(errors):+.5f}]"
            )
    print(f"      map semantics: {report['map_semantics']}")
    print(f"      {'K':>6} {'MAE':>10} {'RMSE':>10}")
    for entry in report["sensitivity"]:
        print(
            f"      {entry['dimension']:>6} {entry['mean_absolute_error']:>10.5f} "
            f"{entry['rmse']:>10.5f}"
        )

    payload = {
        "selected_position_indices": indices,
        "selected_target_ids": targets,
        "selected_greedy_ids": greedy,
        "exact_norms": norms,
        "exact_cosine_matrix": report["exact_cosines"],
        "production_cosine_matrix": report["production_cosines"],
        "alternate_seeds": np.asarray(
            [entry["seed"] for entry in report["alternate_seeds"]]
        ),
        "alternate_mae": np.asarray(
            [entry["mean_absolute_error"] for entry in report["alternate_seeds"]]
        ),
        "alternate_bias": np.asarray(
            [entry["mean_signed_error"] for entry in report["alternate_seeds"]]
        ),
        "alternate_rmse": np.asarray(
            [entry["rmse"] for entry in report["alternate_seeds"]]
        ),
        "k_values": np.asarray(
            [entry["dimension"] for entry in report["sensitivity"]]
        ),
        "k_mae": np.asarray(
            [entry["mean_absolute_error"] for entry in report["sensitivity"]]
        ),
        "k_rmse": np.asarray([entry["rmse"] for entry in report["sensitivity"]]),
        "norm_drift": np.asarray([drift]),
        "sketch_drift": np.asarray([sketch_drift]),
        "map_semantics": np.asarray([report["map_semantics"]]),
    }
    for name, entry in report["subgroup_deltas"].items():
        for side in ("exact", "sketch"):
            values = entry[side]
            if values["available"]:
                for key in ("within", "between", "delta"):
                    payload[f"{name}_{side}_{key}"] = np.asarray([values[key]])
        if "delta_error" in entry:
            payload[f"{name}_delta_error"] = np.asarray([entry["delta_error"]])

    # RunPaths carries the canonical run root; deriving it from a sibling path
    # or inventing a new attribute would be a second way to say the same thing.
    destination = run.paths.run_dir / "sanity"
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "countsketch_fidelity.npz", **payload)
    print(f"    countsketch fidelity artifact: {destination / 'countsketch_fidelity.npz'}")


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
    # Scoped here so the publication step below can see them whether or not the
    # gradient analysis ran at all.
    sketch_store = None
    row_sink = None
    sweep_labels = None
    nucleus_gate_outcome = None
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
                # Only for the initialization whose gradients are measured: the
                # labels exist to group *those* gradients, and capturing them for
                # every initialization would keep twelve copies of something only
                # one of them can use.
                collect_sweep_labels=(
                    protocol["gradient_analysis_enabled"]
                    and index == protocol["gradient_initialization_index"]
                ),
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
            # The nucleus labels this initialization actually drew, kept rather
            # than only tallied. Recovering them afterwards costs a forward pass
            # per sampling temperature -- about five hours for a production arm.
            #
            # They come from the same draw on the same logits that produced the
            # recorded counts, so the gate below is expected to pass. It still
            # runs: the gate is a check on the data, and a reconstruction that
            # does not reproduce the recorded histogram is a reconstruction of
            # some other model. Replacing a check with an argument is how that
            # protection gets lost.
            if measurements[-1].sweep_labels is not None:
                from llm_behavior_lab.analysis.nucleus_clustering import (
                    histogram_gate,
                )

                sweep_labels = measurements[-1].sweep_labels.numpy()
                histogram_gate(
                    sweep_labels,
                    measurements[-1].sweep_counts.numpy(),
                    vocab_size=tokenizer.vocab_size,
                    temperatures=sweep_temperatures,
                )
                nucleus_gate_outcome = {
                    "passed": True,
                    "comparison": "exact_integer_equality",
                    "temperatures": [float(value) for value in sweep_temperatures],
                    "model_seed": int(model_seed),
                    "forward_batch_size": int(protocol["forward_batch_size"]),
                    "top_p": float(sampling.top_p),
                    "sampling_seed": int(sampling.seed),
                    "source": "captured_during_run",
                }
                print(
                    f"    nucleus labels captured for "
                    f"{len(sweep_temperatures)} sampling temperatures; "
                    "histogram gate passed"
                )

            # Measured on the model object that was just measured above, not on a
            # re-seeded reconstruction of it: the observable describes *this*
            # initialization. The call leaves every parameter, buffer, gradient,
            # and the train/eval mode exactly as it found them.
            # Selected before the loop from target labels and position index
            # alone -- both known without running the model, and neither able to
            # see a similarity value, which is what keeps the check honest.
            sanity_positions = None
            if protocol["countsketch_fidelity_sanity"]:
                from llm_behavior_lab.analysis.countsketch_fidelity import (
                    select_sanity_positions,
                )
                from llm_behavior_lab.evaluation.position_gradients import (
                    DEFAULT_SANITY_POSITIONS,
                )

                sanity_positions = select_sanity_positions(
                    positions.target_ids.reshape(-1).cpu().numpy(),
                    None,
                    count=DEFAULT_SANITY_POSITIONS,
                )
                print(
                    f"    countsketch fidelity sanity: capturing "
                    f"{len(sanity_positions)} exact gradients"
                )
            # metrics_only and compact_factors stream their rows to bounded
            # disk; per_position keeps the historical in-memory surface, capped,
            # because at experiment scale it is 3.5 GiB.
            sketch_store = None
            if protocol["gradient_sketch"] and protocol["sketch_storage"] == (
                "per_position"
            ):
                # The rows are kept, so the sink is the historical in-memory
                # array -- but the run still finalizes, because the mode says
                # what was ADDITIONALLY retained, never whether the analysis ran.
                from llm_behavior_lab.evaluation.sketch_store import (
                    InMemoryRowSink,
                    SketchStoreLayout,
                )

                measured_positions = (
                    positions.num_positions
                    if protocol["gradient_num_windows"] is None
                    else protocol["gradient_num_windows"] * positions.block_size
                )
                retained_bytes = (
                    len(protocol["gradient_temperatures"]) * measured_positions
                    * protocol["sketch_maps"] * protocol["sketch_dimension"] * 4
                )
                if retained_bytes > protocol["sketch_storage_max_bytes"]:
                    raise ValueError(
                        f"per_position would retain {retained_bytes:,} bytes "
                        f"({retained_bytes / 1024 ** 3:.2f} GiB) but the limit is "
                        f"{protocol['sketch_storage_max_bytes']:,}. Keeping rows "
                        "at experiment scale is a deliberate diagnostic choice; "
                        "raise --sketch-storage-max-bytes to make it one."
                    )
                row_sink = InMemoryRowSink(
                    SketchStoreLayout(
                        num_temperatures=len(protocol["gradient_temperatures"]),
                        num_positions=measured_positions,
                        num_maps=protocol["sketch_maps"],
                        num_buckets=protocol["sketch_dimension"],
                    )
                )
            elif protocol["gradient_sketch"]:
                from llm_behavior_lab.evaluation.sketch_store import (
                    SketchStoreLayout,
                    TemporarySketchStore,
                    preflight_storage,
                )

                store_layout = SketchStoreLayout(
                    num_temperatures=len(protocol["gradient_temperatures"]),
                    num_positions=positions.num_positions
                    if protocol["gradient_num_windows"] is None
                    else protocol["gradient_num_windows"] * positions.block_size,
                    num_maps=protocol["sketch_maps"],
                    num_buckets=protocol["sketch_dimension"],
                )
                # The run directory does not exist yet -- `ExperimentRun.create`
                # runs after the measurement -- so the durable manifest is
                # anchored to the output directory, which does. Its path is
                # recorded in the published record, so a store left behind on
                # lost node-local scratch is still traceable from the run.
                manifest_dir = resolve_repo_path(
                    experiment_settings_from_config(experiment_config).output_dir
                    if args.output_dir is None
                    else args.output_dir
                ) / "_gradient_tmp"
                bulk_dir = (
                    Path(protocol["gradient_temp_dir"])
                    if protocol["gradient_temp_dir"]
                    else manifest_dir
                )
                store_preflight = preflight_storage(
                    store_layout, bulk_dir,
                    max_bytes=protocol["gradient_temp_max_bytes"],
                    reserve_bytes=2 * 1024 ** 3,
                )
                row_sink = sketch_store = TemporarySketchStore.create(
                    manifest_dir=manifest_dir, bulk_dir=bulk_dir,
                    layout=store_layout,
                    run_id=str(args.run_id or "pending"),
                    preflight=store_preflight,
                )
                print(
                    f"    temporary sketch store: "
                    f"{store_layout.total_bytes / 1024 ** 2:,.0f} MiB in "
                    f"{store_layout.num_slabs} slabs"
                )

            gradient_result = compute_position_gradient_norms(
                model,
                positions,
                vocab_size=tokenizer.vocab_size,
                eligible_token_ids=eligible_token_ids,
                num_windows=protocol["gradient_num_windows"],
                temperatures=protocol["gradient_temperatures"],
                vector_split=protocol["gradient_vector_split"],
                gradient_sketch=protocol["gradient_sketch"],
                sketch_dimension=protocol["sketch_dimension"],
                sketch_maps=protocol["sketch_maps"],
                exact_gradient_positions=sanity_positions,
                row_sink=row_sink,
            )
            if gradient_result.vector_split is not None:
                split = gradient_result.vector_split
                cosine = split["cosine"]
                print(
                    f"    vector split (T = {split['temperature']:g}): "
                    f"||g_correct|| = {split['norm_correct']:.6g} "
                    f"({split['num_correct']:,} positions), "
                    f"||g_wrong|| = {split['norm_wrong']:.6g} "
                    f"({split['num_wrong']:,} positions)\n"
                    f"      ||g_correct + g_wrong|| = {split['norm_total']:.6g}, "
                    f"dot = {split['dot']:.6g}, "
                    f"cos = {'undefined' if cosine is None else f'{cosine:.6f}'}"
                )
            rate = gradient_result.num_positions / max(gradient_result.seconds, 1e-9)
            print(
                f"    gradient analysis: {gradient_result.num_positions:,} positions "
                f"over {gradient_result.parameter_count:,} parameters in "
                f"{gradient_result.seconds:,.1f}s ({rate:,.1f} positions/s)"
            )
            # Read back off the protocol the measurement wrote, not off the
            # request: this line is evidence about what happened, and restating
            # the request here would make it agree with itself by construction.
            if gradient_result.sketch_protocol is not None:
                measured_protocol = gradient_result.sketch_protocol
                print(
                    f"    count sketch: K = {measured_protocol['dimension']}, "
                    f"M = {measured_protocol['map_count']}, "
                    f"base seed {measured_protocol['base_seed']}"
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
            # Both of these are resolved from the model that was actually built
            # rather than written out here. The note and scale_applied["note"]
            # used to be two separate hand-written descriptions of one fact, both
            # of them LLaMA-specific; a GPT run would have inherited claims about
            # kaiming-uniform linears and an architecture with no biases, neither
            # of which is true of it. has_single_sigma_w was a hard-coded literal
            # beside a report that computes the same thing: identical today for
            # both families, but free to drift apart the moment one is not.
            "has_single_sigma_w": scale_report["has_single_sigma_w"],
            "note": scale_applied["note"],
            "applied": scale_applied,
            "parameter_groups": scale_report,
        },
        "device": str(device),
        # Additive. str(device) alone cannot say which physical GPU ran this when
        # a campaign is split across two: "cuda" under CUDA_VISIBLE_DEVICES=0 and
        # under =1 record the same string, and two identical cards report the
        # same name.
        "device_provenance": describe_device(device),
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
    # Declared by the writer, not left to be stamped on the way out.
    #
    # `save()` writes exactly this value anyway, so the persisted record is
    # unchanged. What changes is the *in-memory* record between `build()` and
    # `save()`: validation runs at construction, and the multi-map layout is
    # legal only at record_version 12, so a freshly built M > 1 record with no
    # declared version was refused by its own schema before it could be written.
    # A record that cannot state its schema until it reaches disk is not
    # self-describing, and every other writer in this project -- the tests
    # included -- states it up front.
    #
    # It is stamped at every map count, not only above one: this is the writer
    # saying which schema it wrote, which is true of the M = 1 records too.
    # `record_metadata` is a copy, so the run's own `metadata.json` is untouched.
    record_metadata["record_version"] = RECORD_VERSION
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
        # Conditional on the measured map count, and this is the whole of the
        # multi-map storage decision.
        #
        # At M = 1 the canonical [D, K] block is stored exactly as it always has
        # been. At M > 1 it is *omitted*: the canonical sketch is the canonical
        # row of the temperature array below, reachable through
        # `record.per_map_sketches()`, and record v12 refuses a physically stored
        # canonical array above one map. Storing one anyway would re-admit the
        # possibility that two canonical representations disagree, and would
        # duplicate about 512 MiB per campaign record to do it.
        #
        # Note what is *not* written here: map 0 alone. A [D, K] slice of a
        # four-map measurement is not the canonical sketch of that measurement,
        # it is one quarter of it, and it would load without complaint.
        gradient_position_sketches=(
            None
            if gradient_result is None
            or gradient_result.gradient_sketches is None
            or _measured_map_count(gradient_result) != 1
            else gradient_result.gradient_sketches.numpy()
        ),
        # [N_T, D, K] at M = 1 and [N_T, D, M, K] above it, under the same field
        # name deliberately: a released v11 loader raises on the rank it does not
        # understand, where a new name would have loaded silently with the
        # sketches reported absent.
        #
        # At M = 1 the canonical field above is a row of this one, so the two
        # cannot disagree; the record validates that they do not.
        gradient_temperature_position_sketches=(
            None
            if gradient_result is None
            or gradient_result.temperature_gradient_sketches is None
            else gradient_result.temperature_gradient_sketches.numpy()
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
    # -- finalize and publish -------------------------------------------------
    #
    # Everything the declared analysis contract promises is computed here, while
    # the rows still exist, and the whole bundle is published as one transaction.
    # Only once that bundle has been re-read from its real location are the rows
    # released: they are the only thing that could regenerate what was published.
    if row_sink is not None and gradient_result is not None:
        import dataclasses as _dataclasses

        from llm_behavior_lab.analysis.alignment_estimators import (
            ESTIMATOR_CONVENTION,
            NUMERICAL_EPOCH,
            REDUCTION_CONVENTION,
        )
        from llm_behavior_lab.analysis.alignment_finalization import (
            FinalizationInputs,
            build_metrics_arrays,
            finalize_alignment_metrics,
        )
        from llm_behavior_lab.analysis.alignment_metrics import AlignmentMetrics
        from llm_behavior_lab.analysis.alignment_publication import (
            publish_finalized_run,
        )

        if sketch_store is not None:
            sketch_store.begin_finalization()
        try:
            finalization_started = time.perf_counter()

            # The bounded fidelity summary, computed from the sketches every
            # production map actually produced. It goes into the authoritative
            # artifact rather than a second file, so there is one v13 artifact
            # to keep in step.
            #
            # The gradients and their sketches are the largest things this run
            # still holds -- hundreds of megabytes at experiment scale -- and
            # they are released in the `finally` below on every path, success or
            # failure, before anything else is attempted.
            sanity_summary = None
            sanity_selection = None
            if (
                gradient_result.exact_gradients is not None
                and gradient_result.exact_sketches is not None
            ):
                from llm_behavior_lab.analysis.countsketch_fidelity import (
                    check_sanity_bounds,
                    ensemble_fidelity_summary,
                )

                try:
                    exact = gradient_result.exact_gradients.numpy()
                    sampled = gradient_result.exact_sketches.numpy()
                    lookup = {
                        int(value): row
                        for row, value in enumerate(
                            gradient_result.position_indices.numpy()
                        )
                    }
                    rows = [
                        lookup[int(index)]
                        for index in gradient_result.exact_positions.numpy()
                    ]
                    check_sanity_bounds(
                        num_positions=exact.shape[0],
                        num_parameters=exact.shape[1],
                        maps=protocol["sketch_maps"],
                        dimension=protocol["sketch_dimension"],
                        max_positions=protocol["fidelity_max_positions"],
                        max_gradient_bytes=protocol["fidelity_max_gradient_bytes"],
                        max_sketch_bytes=protocol["fidelity_max_sketch_bytes"],
                    )
                    sanity_summary = ensemble_fidelity_summary(
                        exact, sampled,
                        gradient_result.gradient_norms.numpy()[rows],
                        gradient_result.target_ids.numpy()[rows],
                        gradient_result.greedy_ids.numpy()[rows],
                        dimension=protocol["sketch_dimension"],
                    )
                    sanity_selection = {
                        "positions": gradient_result.exact_positions.numpy(),
                        "targets": gradient_result.target_ids.numpy()[rows],
                        "greedy": gradient_result.greedy_ids.numpy()[rows],
                    }
                    print(
                        f"    countsketch fidelity ({sanity_summary['map_count']} "
                        f"maps): MAE {sanity_summary['mean_absolute_error']:.5f}, "
                        f"outcome {sanity_summary['outcome']}"
                    )
                finally:
                    # Released here, not after publication: nothing downstream
                    # needs them and holding them through finalization would
                    # double the run's peak for no reason.
                    gradient_result = dataclasses.replace(
                        gradient_result, exact_gradients=None, exact_sketches=None
                    )
                    exact = sampled = None

            factor_selections = ()
            if protocol["sketch_factors"]:
                from llm_behavior_lab.analysis.compact_factors import (
                    estimate_factor_bytes,
                    parse_factor_selection,
                )

                factor_selections = tuple(
                    parse_factor_selection(protocol["sketch_factors"])
                )
                # Estimated against the realized class count, before anything is
                # built: float64 factors at a subword vocabulary are hundreds of
                # megabytes per selection.
                realized_classes = int(
                    np.unique(gradient_result.target_ids.numpy()).size
                )
                factor_bytes = estimate_factor_bytes(
                    factor_selections,
                    num_classes=realized_classes,
                    num_maps=protocol["sketch_maps"],
                    num_buckets=protocol["sketch_dimension"],
                )
                if factor_bytes > protocol["sketch_factors_max_bytes"]:
                    raise ValueError(
                        f"The declared factor selection would retain "
                        f"{factor_bytes:,} bytes but the limit is "
                        f"{protocol['sketch_factors_max_bytes']:,}. Narrow the "
                        "selection or raise --sketch-factors-max-bytes."
                    )
                print(
                    f"    retaining {len(factor_selections)} factor selection(s), "
                    f"about {factor_bytes / 1024 ** 2:,.1f} MiB"
                )

            inputs = FinalizationInputs(
                loss_temperatures=np.asarray(gradient_result.temperatures),
                norms=gradient_result.temperature_gradient_norms.numpy(),
                target_ids=gradient_result.target_ids.numpy(),
                greedy_ids=gradient_result.greedy_ids.numpy(),
                nucleus_labels=sweep_labels,
                sampling_temperatures=np.asarray(sweep_temperatures, dtype=float),
                factor_selections=factor_selections,
            )
            finalized = finalize_alignment_metrics(
                row_sink.iter_slabs(), inputs,
                num_maps=protocol["sketch_maps"],
                progress=lambda stage, done, total, elapsed: print(
                    f"    {stage}: {done}/{total} slabs, {elapsed:.0f}s"
                ),
            )
            arrays = build_metrics_arrays(finalized, inputs)
            if sanity_summary is not None:
                from llm_behavior_lab.analysis.countsketch_fidelity import (
                    sanity_arrays,
                )

                arrays.update(
                    sanity_arrays(
                        sanity_summary,
                        position_indices=sanity_selection["positions"],
                        target_ids=sanity_selection["targets"],
                        greedy_ids=sanity_selection["greedy"],
                        seed=protocol["sketch_seed"]
                        if "sketch_seed" in protocol else 20240917,
                    )
                )
            finalization_seconds = time.perf_counter() - finalization_started

            def _factory(reference):
                block = {
                    "storage_mode": protocol["sketch_storage"],
                    "alignment_metrics_schema_version": 1,
                    "estimator_convention": ESTIMATOR_CONVENTION,
                    "reduction_convention": REDUCTION_CONVENTION,
                    "numerical_epoch": NUMERICAL_EPOCH,
                    "metrics_artifact": reference,
                    # One authoritative home for the operational limits that
                    # have no other owner. `gradient_temp_max_bytes`,
                    # `reserve_bytes` and `safety_factor` are deliberately NOT
                    # repeated here: `temporary_store.preflight` already owns
                    # them, and two copies of one fact is one too many.
                    #
                    # The resolved temp directory is recorded as a *policy*, not
                    # a path. A completed record must stay portable, and the
                    # absolute location -- which may be node-local scratch that
                    # no longer exists -- belongs in the live manifest, where it
                    # is operationally necessary, not in the published artifact.
                    "storage_limits": {
                        "sketch_storage_max_bytes": protocol[
                            "sketch_storage_max_bytes"
                        ],
                        "sketch_factors_max_bytes": protocol[
                            "sketch_factors_max_bytes"
                        ],
                        "temp_dir_policy": (
                            "explicit_override"
                            if protocol["gradient_temp_dir"]
                            else "output_root_gradient_tmp"
                        ),
                        "temp_dir_overridden": bool(protocol["gradient_temp_dir"]),
                    },
                    "finalization_status": "complete",
                    "factor_selection": [
                        selection.as_dict() for selection in factor_selections
                    ],
                    "finalization_seconds": round(finalization_seconds, 3),
                    # The compact, path-free form: a completed run must stay
                    # portable, and the bulk directory it names is deleted
                    # moments later. The full manifest keeps its absolute paths
                    # under the temporary root, where they are only useful while
                    # a store still exists to point at.
                    "temporary_store": (
                        None if sketch_store is None
                        else sketch_store.compact_manifest()
                    ),
                    "nucleus_histogram_gate": nucleus_gate_outcome,
                    "sanity": (
                        None if sanity_summary is None
                        else {
                            "enabled": True,
                            "map_count": sanity_summary["map_count"],
                            "num_ordered_pairs": sanity_summary[
                                "num_ordered_pairs"
                            ],
                            "num_unordered_pairs": sanity_summary[
                                "num_unordered_pairs"
                            ],
                            "pair_convention": sanity_summary["pair_convention"],
                            "outcome": sanity_summary["outcome"],
                            "thresholds": sanity_summary["thresholds"],
                            "bounds": {
                                "max_positions": protocol["fidelity_max_positions"],
                                "max_gradient_bytes": protocol[
                                    "fidelity_max_gradient_bytes"
                                ],
                                "max_sketch_bytes": protocol[
                                    "fidelity_max_sketch_bytes"
                                ],
                            },
                        }
                    ),
                }
                metadata = dict(record_metadata)
                analysis = dict(metadata.get("analysis", {}))
                gradient = dict(analysis.get("gradient_analysis", {}))
                gradient["gradient_alignment"] = block
                analysis["gradient_analysis"] = gradient
                metadata["analysis"] = analysis
                return _dataclasses.replace(record, metadata=metadata)

            publish_finalized_run(
                run.paths.analyses_dir,
                record_factory=_factory,
                metrics=AlignmentMetrics(
                    arrays=arrays,
                    provenance={
                        "storage_mode": protocol["sketch_storage"],
                        "declared_analyses": sorted(
                            {job.kind for job in finalized["jobs"]}
                        ),
                        "permutations": inputs.permutations,
                        "permutation_seed": inputs.permutation_seed,
                        "min_support": inputs.min_support,
                        "factor_selection": [
                            selection.as_dict() for selection in factor_selections
                        ],
                    },
                ),
            )
        except BaseException as error:
            # Collection succeeded; only finalization did not. The rows are
            # hours of measurement and are kept so the run can be finalized
            # again -- never discarded because a later step failed.
            if sketch_store is not None:
                sketch_store.mark_recoverable(f"{type(error).__name__}: {error}")
                print(
                    f"    finalization failed: {error}\n"
                    f"    rows retained at {sketch_store.directory}"
                )
            raise
        if sketch_store is not None:
            sketch_store.mark_published()
            sketch_store.discard()
        print(f"    finalized in {finalization_seconds:.1f}s; rows released")
    else:
        record.save(run.paths.analyses_dir)

    # After the record, because the artifact belongs to a run directory and that
    # only exists once ExperimentRun.create has run. gradient_result is still the
    # object the gradient loop produced -- it is read a few lines above to build
    # the record -- so nothing is recomputed and no second backward pass happens.
    # The captured vectors are released as soon as this returns.
    # Legacy path only. A v13 run has already folded the fidelity summary into
    # the authoritative metrics artifact and released the buffers, so reaching
    # here means no finalization ran and the separate file is the only home the
    # summary has.
    if gradient_result is not None and gradient_result.exact_gradients is not None:
        try:
            _write_countsketch_fidelity(run, gradient_result, protocol)
        finally:
            gradient_result = dataclasses.replace(
                gradient_result, exact_gradients=None, exact_sketches=None
            )

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

    report_support(record)
    report_sampling_adequacy(adequacy)
    report_policy_summaries(record, summaries)
    report_runtime(elapsed, peak_rss_mib, parameter_count)
    report_sampling_variability(sampling_spread)

    if null_summary is not None:
        report_uniform_null(null_summary)

    if protocol["input_structure_enabled"]:
        report_input_structure(record)

    if sweep_temperatures:
        from llm_behavior_lab.analysis import sweep_summary

        summary = sweep_summary(record)
        report_temperature_sweep(summary, sweep_temperatures)
        run.save_analysis_json("temperature_sweep_summary.json", summary)

    if record.has_predictive_probability_analysis:
        report_predictive_probabilities(record)

    if gradient_result is not None:
        report_gradient_norms(
            record, gradient_result, gradient_metadata, positions.num_positions
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
