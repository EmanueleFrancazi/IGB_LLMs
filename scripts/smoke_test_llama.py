"""Sanity check for the Phase 2 LLaMA-style model integration.

This script intentionally uses synthetic token IDs rather than a tokenizer or
real dataset. Tokenization and data loading belong to Phase 3.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

# Allow running from the repository root before editable installation.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from llm_behavior_lab.models import build_model_from_config, list_models  # noqa: E402
from llm_behavior_lab.utils import format_parameter_count, get_device, seed_everything  # noqa: E402


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

    parser = argparse.ArgumentParser(description="Run a Phase 2 LLaMA sanity check.")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "model" / "tiny_llama.yaml",
        help="Path to the tiny LLaMA YAML config.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Synthetic batch size.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=16,
        help="Synthetic sequence length. Must be <= max_seq_len.",
    )
    return parser.parse_args()


def main() -> None:
    """Instantiate the model, run a forward pass, and validate shapes."""

    args = parse_args()
    config = load_yaml_config(args.config)

    runtime_config = config.get("runtime", {})
    seed = int(runtime_config.get("seed", 1234))
    device_name = str(runtime_config.get("device", "auto"))

    seed_everything(seed)
    device = get_device(device_name)

    model = build_model_from_config(config).to(device)
    model.eval()

    model_params = config["model"]["params"]
    model_name = config["model"]["name"]
    vocab_size = int(model_params["vocab_size"])
    max_seq_len = int(model_params["max_seq_len"])

    if args.sequence_length > max_seq_len:
        raise ValueError(
            f"sequence_length={args.sequence_length} exceeds max_seq_len={max_seq_len}."
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

    expected_logits_shape = (args.batch_size, args.sequence_length, vocab_size)
    if tuple(output.logits.shape) != expected_logits_shape:
        raise AssertionError(
            f"Expected logits shape {expected_logits_shape}, got {tuple(output.logits.shape)}."
        )
    if output.loss is None or output.loss.ndim != 0:
        raise AssertionError("Expected scalar loss when targets are provided.")

    n_params = model.count_parameters()

    print("LLaMA Phase 2 sanity check completed successfully.")
    print(f"Available registered models: {list_models()}")
    print(f"Selected model: {model_name}")
    print(f"Device: {device}")
    print(f"Parameter count: {n_params} ({format_parameter_count(n_params)})")
    print(f"Input shape: {tuple(input_ids.shape)}")
    print(f"Target shape: {tuple(targets.shape)}")
    print(f"Logits shape: {tuple(output.logits.shape)}")
    print(f"Loss shape: {tuple(output.loss.shape)}")
    print(f"Loss value: {output.loss.item():.6f}")


if __name__ == "__main__":
    main()
