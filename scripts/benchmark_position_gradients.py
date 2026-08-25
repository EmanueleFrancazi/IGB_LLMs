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
"""

from __future__ import annotations

import argparse
import resource
import sys
import time
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
    GRADIENT_TEMPERATURES,
    compute_position_gradient_norms,
)
from llm_behavior_lab.models import build_model_from_config  # noqa: E402
from llm_behavior_lab.utils import get_device, load_yaml_config, seed_everything  # noqa: E402

#: The position count the extrapolation is quoted for: the standing protocol of
#: 512 windows x 64 tokens.
FULL_EXPERIMENT_POSITIONS = 32768


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
    add_dataset_arguments(parser)
    return parser.parse_args()


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
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
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
