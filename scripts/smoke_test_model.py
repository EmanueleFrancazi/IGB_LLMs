"""Smoke test for Phase 1 model plumbing.

This script verifies that we can:

1. load a YAML config,
2. seed random-number generators,
3. select a device,
4. build a registered model,
5. run a forward pass,
6. compute language-modeling loss, and
7. report parameter counts and tensor shapes.

It intentionally uses the tiny debug model, not the future research baseline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

# Allow running the script from the repository root before installation.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from llm_behavior_lab.models import build_model_from_config, list_models  # noqa: E402
from llm_behavior_lab.utils import format_parameter_count, get_device, seed_everything  # noqa: E402


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load a YAML config file and return a dictionary."""

    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise TypeError(f"Expected YAML config to contain a dictionary, got {type(config)!r}")

    return config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Run a Phase 1 model smoke test.")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "model" / "tiny_debug.yaml",
        help="Path to a YAML model config.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of synthetic sequences in the smoke-test batch.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=16,
        help="Synthetic sequence length. Must be <= model block_size.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the smoke test."""

    args = parse_args()
    config = load_yaml_config(args.config)

    runtime_config = config.get("runtime", {})
    seed = int(runtime_config.get("seed", 1234))
    device_name = str(runtime_config.get("device", "auto"))

    seed_everything(seed)
    device = get_device(device_name)

    model = build_model_from_config(config).to(device)
    model.eval()

    model_config = config["model"]
    model_name = model_config["name"]
    vocab_size = int(model_config["params"]["vocab_size"])
    block_size = int(model_config["params"]["block_size"])

    if args.sequence_length > block_size:
        raise ValueError(
            f"sequence_length={args.sequence_length} exceeds block_size={block_size}."
        )

    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(args.batch_size, args.sequence_length),
        device=device,
    )
    targets = torch.randint(
        low=0,
        high=vocab_size,
        size=(args.batch_size, args.sequence_length),
        device=device,
    )

    with torch.no_grad():
        output = model(input_ids=input_ids, targets=targets)

    parameter_count = model.count_parameters()

    print("Smoke test completed successfully.")
    print(f"Available registered models: {list_models()}")
    print(f"Selected model: {model_name}")
    print(f"Device: {device}")
    print(f"Parameter count: {parameter_count} ({format_parameter_count(parameter_count)})")
    print(f"Input shape: {tuple(input_ids.shape)}")
    print(f"Target shape: {tuple(targets.shape)}")
    print(f"Logits shape: {tuple(output.logits.shape)}")
    print(f"Loss shape: {tuple(output.loss.shape) if output.loss is not None else None}")
    print(f"Loss value: {output.loss.item():.6f}" if output.loss is not None else "Loss value: None")


if __name__ == "__main__":
    main()
