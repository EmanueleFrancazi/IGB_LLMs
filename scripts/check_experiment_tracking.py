"""Smoke-test Phase 6 experiment logging and checkpoint infrastructure.

The script creates one local experiment run, snapshots configs and metadata,
logs scalar and array metrics, saves an initialized model checkpoint, discovers
the latest checkpoint, restores it into a second compatible model, and verifies
that every restored parameter matches the saved model.

No training updates are performed.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from llm_behavior_lab.experiment import ExperimentRun, experiment_settings_from_config  # noqa: E402
from llm_behavior_lab.models import build_model_from_config  # noqa: E402
from llm_behavior_lab.utils import format_parameter_count, get_device, load_yaml_config, seed_everything  # noqa: E402


def resolve_repo_path(path_value: str | Path) -> Path:
    """Resolve a path relative to the repository root."""

    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Check Phase 6 experiment tracking.")
    parser.add_argument(
        "--model-config",
        type=Path,
        default=REPO_ROOT / "configs" / "model" / "tiny_llama.yaml",
        help="Path to the model YAML config.",
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=REPO_ROOT / "configs" / "data" / "tiny_text.yaml",
        help="Path to the data YAML config.",
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=REPO_ROOT / "configs" / "experiment" / "phase6_smoke.yaml",
        help="Path to the experiment YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output-root override.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional explicit run ID. Existing directories are not overwritten.",
    )
    return parser.parse_args()


def _state_dicts_match(first: torch.nn.Module, second: torch.nn.Module) -> bool:
    """Return true when two model state dictionaries match exactly."""

    first_state = first.state_dict()
    second_state = second.state_dict()
    if first_state.keys() != second_state.keys():
        return False
    return all(torch.equal(first_state[name].cpu(), second_state[name].cpu()) for name in first_state)


def main() -> None:
    """Run the Phase 6 persistence round-trip check."""

    args = parse_args()
    model_config = load_yaml_config(args.model_config)
    data_config = load_yaml_config(args.data_config)
    experiment_config = load_yaml_config(args.experiment_config)
    settings = experiment_settings_from_config(experiment_config)

    output_dir = settings.output_dir if args.output_dir is None else args.output_dir
    settings = replace(settings, output_dir=resolve_repo_path(output_dir))

    runtime_config = data_config.get("runtime", model_config.get("runtime", {}))
    seed = int(runtime_config.get("seed", settings.seed))
    device = get_device(str(runtime_config.get("device", "auto")))
    seed_everything(seed)

    model = build_model_from_config(model_config).to(device)
    model.eval()
    parameter_count = model.count_parameters()

    run = ExperimentRun.create(
        replace(settings, seed=seed),
        run_id=args.run_id,
        repo_root=REPO_ROOT,
        metadata={
            "experiment_type": "phase6_smoke_test",
            "device": str(device),
            "model_name": model_config["model"]["name"],
            "model_parameter_count": parameter_count,
        },
    )
    snapshots = run.snapshot_configs(
        model_config=model_config,
        data_config=data_config,
        experiment_config=experiment_config,
    )

    training_record = run.training_metrics.log(
        step=0,
        stage="smoke_test",
        split="train",
        metrics={"loss": 5.0, "learning_rate": 0.0},
    )
    array_artifact = run.array_metrics.save(
        metric_name="example_layer_values",
        step=0,
        stage="smoke_test",
        split="train",
        arrays={
            "layer_indices": torch.arange(3),
            "values": torch.tensor([1.0, 0.5, 0.25]),
        },
    )
    evaluation_record = run.evaluation_metrics.log(
        step=0,
        stage="smoke_test",
        split="validation",
        metrics={"output_entropy": 3.5, "top1_concentration": 0.2},
        artifacts={"example_layer_values": array_artifact.relative_path},
    )
    run.save_analysis_json(
        "smoke_test_summary.json",
        {
            "training_record": training_record,
            "evaluation_record": evaluation_record,
            "array_artifact": array_artifact.relative_path,
        },
    )

    checkpoint = run.checkpoints.save(
        model=model,
        step=0,
        model_config=model_config,
        additional_state={"smoke_test": True},
    )
    latest = run.checkpoints.latest_checkpoint()
    if latest is None or latest.path != checkpoint.path:
        raise AssertionError("Latest checkpoint discovery did not return the saved checkpoint.")

    restored_model = build_model_from_config(model_config).to("cpu")
    payload = run.checkpoints.restore(
        model=restored_model,
        checkpoint=latest,
        map_location="cpu",
    )
    if not _state_dicts_match(model.to("cpu"), restored_model):
        raise AssertionError("Restored model parameters do not match the saved model.")

    loaded_arrays = run.array_metrics.load(array_artifact.path)
    if not torch.equal(torch.from_numpy(loaded_arrays["values"]), torch.tensor([1.0, 0.5, 0.25])):
        raise AssertionError("Array artifact round-trip failed.")

    print("Phase 6 experiment tracking check completed successfully.")
    print(f"Run directory: {run.paths.run_dir}")
    print(f"Metadata path: {run.paths.metadata_path}")
    print(f"Config snapshots: {[str(path) for path in snapshots.values()]}")
    print(f"Training metrics: {run.training_metrics.path}")
    print(f"Evaluation metrics: {run.evaluation_metrics.path}")
    print(f"Array artifact: {array_artifact.path}")
    print(f"Checkpoint: {checkpoint.path}")
    print(f"Latest checkpoint: {latest.path}")
    print(f"Restored global step: {payload['global_step']}")
    print(f"Parameter count: {parameter_count} ({format_parameter_count(parameter_count)})")
    print("Checkpoint parameter round-trip: verified")


if __name__ == "__main__":
    main()
