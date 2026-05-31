"""Check local and standard dataset configuration options.

This script demonstrates the unified data-source path added before Phase 6. It
loads either a local text file or an optional Hugging Face dataset, builds the
current character tokenizer, creates causal LM batches, and optionally verifies
model compatibility with one forward pass.
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

from llm_behavior_lab.data import (  # noqa: E402
    CausalLMBatcher,
    CharTokenizer,
    load_text_dataset_from_config,
    split_token_ids,
)
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

    parser = argparse.ArgumentParser(description="Check configured dataset options.")
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
        "--skip-model-check",
        action="store_true",
        help="Only check dataset loading/tokenization/batching; skip model forward pass.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the dataset-option check."""

    args = parse_args()
    data_config = load_yaml_config(args.data_config)
    model_config = load_yaml_config(args.model_config)

    runtime_config = data_config.get("runtime", model_config.get("runtime", {}))
    seed = int(runtime_config.get("seed", 1234))
    device_name = str(runtime_config.get("device", "auto"))

    seed_everything(seed)
    device = get_device(device_name)

    dataset_config = data_config["dataset"]
    batching_config = data_config["batching"]
    val_fraction = float(dataset_config.get("val_fraction", 0.1))
    batch_size = int(batching_config["batch_size"])
    block_size = int(batching_config["block_size"])

    loaded_dataset = load_text_dataset_from_config(data_config, repo_root=REPO_ROOT)
    tokenizer = CharTokenizer.from_text(loaded_dataset.text)
    token_ids = tokenizer.encode(loaded_dataset.text)
    splits = split_token_ids(
        token_ids,
        val_fraction=val_fraction,
        min_train_tokens=block_size + 1,
        min_val_tokens=block_size + 1,
    )

    model_params = model_config["model"]["params"]
    model_vocab_size = int(model_params["vocab_size"])
    model_max_seq_len = int(model_params["max_seq_len"])
    if tokenizer.vocab_size > model_vocab_size:
        raise ValueError(
            f"Tokenizer vocab size {tokenizer.vocab_size} exceeds model vocab size "
            f"{model_vocab_size}. Increase model.params.vocab_size."
        )
    if block_size > model_max_seq_len:
        raise ValueError(
            f"Data block_size {block_size} exceeds model max_seq_len {model_max_seq_len}."
        )

    batcher = CausalLMBatcher(
        train_ids=splits.train_ids,
        val_ids=splits.val_ids,
        block_size=block_size,
        batch_size=batch_size,
        device=device,
        seed=seed,
    )
    train_batch = batcher.get_batch("train")
    val_batch = batcher.get_batch("val")

    output = None
    parameter_count = None
    if not args.skip_model_check:
        model = build_model_from_config(model_config).to(device)
        model.eval()
        with torch.no_grad():
            output = model(input_ids=train_batch.input_ids, targets=train_batch.targets)
        parameter_count = model.count_parameters()

    print("Dataset options check completed successfully.")
    print(f"Available registered models: {list_models()}")
    print(f"Dataset source type: {loaded_dataset.source_type}")
    print(f"Dataset source name: {loaded_dataset.source_name}")
    print(f"Dataset metadata: {loaded_dataset.metadata}")
    print(f"Raw examples used: {loaded_dataset.num_examples}")
    print(f"Raw text characters: {len(loaded_dataset.text)}")
    print("Tokenizer type: char")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Total token count: {len(token_ids)}")
    print(f"Train token count: {len(splits.train_ids)}")
    print(f"Validation token count: {len(splits.val_ids)}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Block size: {block_size}")
    print(f"Train input shape: {tuple(train_batch.input_ids.shape)}")
    print(f"Train target shape: {tuple(train_batch.targets.shape)}")
    print(f"Validation input shape: {tuple(val_batch.input_ids.shape)}")
    print(f"Validation target shape: {tuple(val_batch.targets.shape)}")
    if output is not None and parameter_count is not None:
        print(f"Model name: {model_config['model']['name']}")
        print(f"Model vocab size: {model_vocab_size}")
        print(f"Parameter count: {parameter_count} ({format_parameter_count(parameter_count)})")
        print(f"Logits shape: {tuple(output.logits.shape)}")
        print(f"Loss shape: {tuple(output.loss.shape) if output.loss is not None else None}")
        if output.loss is not None:
            print(f"Loss value: {output.loss.item():.6f}")


if __name__ == "__main__":
    main()
