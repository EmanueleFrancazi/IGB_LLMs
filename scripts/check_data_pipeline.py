"""Phase 3 data-pipeline and model-compatibility check.

This script verifies that local text can be loaded, tokenized, split into
train/validation sets, batched into causal language-modeling examples, and fed
through the Phase 2 LLaMA-style model.

It intentionally does not train the model. Training starts in a later phase.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Allow running from the repository root before editable installation.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from llm_behavior_lab.data import (  # noqa: E402
    CausalLMBatcher,
    CharTokenizer,
    DatasetConfig,
    add_dataset_arguments,
    resolution_policy_from_args,
    resolve_dataset,
    split_token_ids,
)
from llm_behavior_lab.models import build_model_from_config, list_models  # noqa: E402
from llm_behavior_lab.utils import format_parameter_count, get_device, load_yaml_config, seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Run the Phase 3 data-pipeline check.")
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
        help="Only check dataset loading, tokenization, and batching.",
    )
    add_dataset_arguments(parser)
    return parser.parse_args()


def main() -> None:
    """Run the data-to-model compatibility check."""

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

    resolved = resolve_dataset(
        DatasetConfig.from_config(data_config),
        resolution_policy_from_args(args),
        repo_root=REPO_ROOT,
    )
    text_path = resolved.path
    text = resolved.read_text()
    tokenizer = CharTokenizer.from_text(text)
    token_ids = tokenizer.encode(text)
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

        expected_logits_shape = (batch_size, block_size, model_vocab_size)
        if tuple(output.logits.shape) != expected_logits_shape:
            raise AssertionError(
                f"Expected logits shape {expected_logits_shape}, got {tuple(output.logits.shape)}."
            )
        if output.loss is None or output.loss.ndim != 0:
            raise AssertionError("Expected scalar loss when targets are provided.")

        parameter_count = model.count_parameters()

    print("Phase 3 data pipeline check completed successfully.")
    print(f"Available registered models: {list_models()}")
    print(f"Dataset name: {resolved.name}")
    print(f"Dataset source: {resolved.source}")
    print(f"Dataset resolved via: {resolved.route}")
    print(f"Dataset path: {text_path}")
    print(f"Raw text characters: {len(text)}")
    print(f"Tokenizer type: char")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Total token count: {len(token_ids)}")
    print(f"Train token count: {len(splits.train_ids)}")
    print(f"Validation token count: {len(splits.val_ids)}")
    print(f"Device: {device}")
    print(f"Train input shape: {tuple(train_batch.input_ids.shape)}")
    print(f"Train target shape: {tuple(train_batch.targets.shape)}")
    print(f"Validation input shape: {tuple(val_batch.input_ids.shape)}")
    print(f"Validation target shape: {tuple(val_batch.targets.shape)}")
    if output is None:
        print("Model check: skipped")
    else:
        print(f"Model name: {model_config['model']['name']}")
        print(f"Model vocab size: {model_vocab_size}")
        print(
            f"Parameter count: {parameter_count} "
            f"({format_parameter_count(parameter_count)})"
        )
        print(f"Logits shape: {tuple(output.logits.shape)}")
        print(f"Loss shape: {tuple(output.loss.shape)}")
        print(f"Loss value: {output.loss.item():.6f}")
    print(f"Decoded first training input preview: {tokenizer.decode(train_batch.input_ids[0].tolist())!r}")
    print(f"Decoded first training target preview: {tokenizer.decode(train_batch.targets[0].tolist())!r}")


if __name__ == "__main__":
    main()
