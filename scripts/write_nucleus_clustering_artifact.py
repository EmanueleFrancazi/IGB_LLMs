#!/usr/bin/env python3
"""Recover a run's nucleus samples and cluster its gradients by them.

Figure 20 groups one gradient population by target token and by greedy
prediction. This produces the grouping in between: the token actually sampled at
each nucleus sweep temperature. The record does not persist those labels -- the
experiment streams logits and keeps only per-token counts -- so they are
recovered by re-running the forward pass over the same positions with the same
initialization and the same pre-drawn uniforms.

**Nothing is taken on trust.** The recovered labels are gated against the
histogram the experiment recorded, per temperature, with exact integer equality.
Any mismatch aborts before a single statistic is computed: a reconstruction that
does not reproduce the recorded counts is a reconstruction of some other model.

Only a forward pass is required. No gradient is recomputed, and the record is
read but never modified -- the results land in a separate artifact beside it.

The protocol is read from the record's own metadata rather than re-supplied on
the command line, so the reproduction cannot silently use a different sampling
seed, support or temperature grid than the run being reproduced. The configs are
still needed for the architecture and the corpus.

Point the config flags at the run's **own** snapshots under ``<run>/config/``
rather than at ``configs/``. The repository copies may have moved on since the
run; the snapshots are what it was actually launched with::

    python scripts/write_nucleus_clustering_artifact.py <run>/analyses \\
        --data-config <run>/config/data_config.yaml \\
        --model-config <run>/config/model_config.yaml

writes ``<run>/analyses/nucleus_gradient_clustering.npz``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from llm_behavior_lab.analysis import load_record  # noqa: E402
from llm_behavior_lab.analysis.nucleus_clustering import (  # noqa: E402
    histogram_gate,
    nucleus_clustering,
)
from llm_behavior_lab.data import (  # noqa: E402
    DatasetConfig,
    resolution_policy_from_args,
    resolve_dataset,
    split_token_ids,
)
from llm_behavior_lab.data.tokenizer import build_tokenizer  # noqa: E402
from llm_behavior_lab.evaluation.init_distribution import (  # noqa: E402
    NucleusSamplingSettings,
    build_evaluation_positions,
)
from llm_behavior_lab.evaluation.nucleus_labels import (  # noqa: E402
    nucleus_position_labels,
)
from llm_behavior_lab.models import build_model_from_config  # noqa: E402
from llm_behavior_lab.models.initialization_scale import (  # noqa: E402
    scale_initialization,
)
from llm_behavior_lab.utils import get_device, seed_everything  # noqa: E402

#: Name of the derived artifact, inside the record's own directory.
ARTIFACT_NAME = "nucleus_gradient_clustering.npz"


def load_yaml_config(path: Path) -> dict:
    import yaml

    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record_dir", type=Path, help="The run's analyses directory.")
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument(
        "--name",
        default="initialization_distribution",
        help="Record stem inside the analyses directory.",
    )
    parser.add_argument(
        "--forward-batch-size",
        type=int,
        default=8,
        help=(
            "Windows per forward pass. A throughput knob only: the draw is "
            "inverted from position-indexed uniforms, so batching cannot move a "
            "sample. If it somehow did, the histogram gate would catch it."
        ),
    )
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--display-classes", type=int, default=40)
    parser.add_argument("--permutations", type=int, default=256)
    parser.add_argument(
        "--data-root", type=Path, default=None, help="Dataset resolution root."
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record = load_record(args.record_dir, name=args.name)
    if not record.has_gradient_position_sketches:
        raise SystemExit(
            "This record carries no gradient sketches, so there is nothing to "
            "group by the nucleus sample."
        )

    analysis = record.metadata["analysis"]
    sweep = analysis["temperature_sweep"]
    if not sweep.get("enabled"):
        raise SystemExit(
            "This run has no temperature sweep, so it recorded no nucleus "
            "histograms to gate a reproduction against."
        )
    temperatures = [float(value) for value in sweep["temperatures"]]

    sampling_metadata = analysis["sampling"]
    sampling = NucleusSamplingSettings(
        temperature=float(sampling_metadata["temperature"]),
        top_p=float(sampling_metadata["top_p"]),
        seed=int(sampling_metadata["sampling_seed"]),
        num_replicates=int(sampling_metadata["num_replicates"]),
        common_random_numbers=bool(sampling_metadata["common_random_numbers"]),
    )

    gradient_metadata = analysis["gradient_analysis"]
    initialization_index = int(gradient_metadata["initialization_index"])
    model_seed = int(analysis["model_seeds"][initialization_index])

    # -- rebuild exactly what the run evaluated ------------------------------
    data_config = load_yaml_config(args.data_config)
    model_config = load_yaml_config(args.model_config)
    device = get_device(str(data_config.get("runtime", {}).get("device", "auto")))

    policy = resolution_policy_from_args(args)
    resolved = resolve_dataset(
        DatasetConfig.from_config(data_config), policy, repo_root=REPO_ROOT
    )
    text = resolved.read_text()
    tokenizer = build_tokenizer(
        data_config.get("tokenizer"),
        text=text,
        data_root=policy.data_root,
        allow_download=policy.allow_download,
        offline=policy.offline,
        reporter=print,
    )
    token_ids = tokenizer.encode(text)
    evaluation = analysis["evaluation_positions"]
    block_size = int(evaluation["block_size"])
    splits = split_token_ids(
        token_ids,
        val_fraction=float(data_config["dataset"].get("val_fraction", 0.1)),
        min_train_tokens=block_size + 1,
        min_val_tokens=block_size + 1,
    )
    split_ids = splits.train_ids if analysis["split"] == "train" else splits.val_ids
    positions = build_evaluation_positions(
        split_ids,
        block_size=block_size,
        num_windows=int(evaluation["num_windows"]),
        device=device,
    )
    if positions.num_positions != int(analysis["num_positions"]):
        raise SystemExit(
            f"Rebuilt {positions.num_positions} evaluation positions but the run "
            f"measured {analysis['num_positions']}. The corpus, split or window "
            "count differs, so this is not the same experiment."
        )

    seed_everything(model_seed)
    model = build_model_from_config(model_config).to(device)
    scale_initialization(model, float(record.metadata["initialization_scale"]["alpha"]))

    print(
        f"Recovering nucleus labels: initialization {initialization_index} "
        f"(seed {model_seed}), {positions.num_positions:,} positions, "
        f"T = {', '.join(f'{value:g}' for value in temperatures)}"
    )
    recovered = nucleus_position_labels(
        model,
        positions,
        model_seed=model_seed,
        vocab_size=tokenizer.vocab_size,
        sampling=sampling,
        eligible_token_ids=tokenizer.eligible_token_ids,
        temperatures=temperatures,
        forward_batch_size=args.forward_batch_size,
        device=device,
    )

    # -- the gate, before anything is computed from the labels ---------------
    gate = histogram_gate(
        recovered["labels"],
        record.sweep_counts("real")[initialization_index],
        vocab_size=tokenizer.vocab_size,
        temperatures=recovered["temperatures"],
    )
    print(
        "Histogram gate passed: the recovered labels reproduce the recorded "
        f"counts exactly at all {len(gate['per_temperature'])} temperatures."
    )

    # -- clustering ----------------------------------------------------------
    result = nucleus_clustering(
        record,
        recovered["labels"],
        recovered["temperatures"],
        min_support=args.min_support,
        display_classes=args.display_classes,
        permutations=args.permutations,
    )

    arrays: dict[str, np.ndarray] = {
        "temperatures": np.asarray(result["temperatures"], dtype=float),
        # Stored rather than re-inferred when read back: the displayed matrix is
        # a class-by-class block, so nothing about the sketch width or the
        # position count can be recovered from its shape.
        "num_positions": np.array(result["by_temperature"][0]["num_positions"]),
        "num_positions_excluded": np.array(
            result["by_temperature"][0]["num_positions_excluded"]
        ),
        "sketch_dimension": np.array(
            result["by_temperature"][0]["sketch_dimension"]
        ),
        "min_support": np.array(result["min_support"]),
        "permutations": np.array(result["permutations"]),
        "permutation_seed": np.array(result["permutation_seed"]),
        "nucleus_labels": recovered["labels"],
        "delta": np.array([e["population"]["delta"] for e in result["by_temperature"]]),
        "within": np.array([e["population"]["within"] for e in result["by_temperature"]]),
        "between": np.array(
            [e["population"]["between"] for e in result["by_temperature"]]
        ),
        "null_delta_mean": np.array(
            [e["null"]["delta_mean"] for e in result["by_temperature"]]
        ),
        "null_delta_low": np.array(
            [e["null"]["delta_low"] for e in result["by_temperature"]]
        ),
        "null_delta_high": np.array(
            [e["null"]["delta_high"] for e in result["by_temperature"]]
        ),
    }
    for field in (
        "num_represented", "num_qualifying", "num_singletons", "singleton_fraction",
        "positions_in_qualifying", "fraction_positions_in_qualifying",
        "largest_class", "median_class", "num_within_pairs", "num_between_pairs",
    ):
        arrays[f"support_{field}"] = np.array(
            [entry["support"][field] for entry in result["by_temperature"]]
        )
    for index, entry in enumerate(result["by_temperature"]):
        arrays[f"display_classes_{index}"] = entry["display"]["classes"]
        arrays[f"display_matrix_{index}"] = entry["display"]["matrix"]
        arrays[f"display_counts_{index}"] = entry["display"]["counts"]
    for grouping in ("target", "greedy"):
        reference = result["references"][grouping]
        arrays[f"reference_{grouping}_delta"] = np.array(
            reference["population"]["delta"]
        )
        arrays[f"reference_{grouping}_null_low"] = np.array(
            reference["null"]["delta_low"]
        )
        arrays[f"reference_{grouping}_null_high"] = np.array(
            reference["null"]["delta_high"]
        )

    destination = args.record_dir / ARTIFACT_NAME
    np.savez_compressed(destination, **arrays)

    summary = {
        "initialization_index": initialization_index,
        "model_seed": model_seed,
        "num_positions": int(positions.num_positions),
        "histogram_gate": "exact match at every temperature",
        "temperatures": list(result["temperatures"]),
        "permutations": result["permutations"],
        "by_temperature": [
            {
                "temperature": entry["temperature"],
                "delta": entry["population"]["delta"],
                "within": entry["population"]["within"],
                "between": entry["population"]["between"],
                "null_delta_low": entry["null"]["delta_low"],
                "null_delta_high": entry["null"]["delta_high"],
                "num_qualifying_classes": entry["support"]["num_qualifying"],
                "singleton_fraction": entry["support"]["singleton_fraction"],
                "fraction_positions_in_qualifying": entry["support"][
                    "fraction_positions_in_qualifying"
                ],
            }
            for entry in result["by_temperature"]
        ],
        "references": {
            grouping: result["references"][grouping]["population"]["delta"]
            for grouping in ("target", "greedy")
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    print(f"\nArtifact written: {destination}")


if __name__ == "__main__":
    main()
