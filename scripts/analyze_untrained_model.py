"""Analyze baseline behavior of the untrained LLaMA-style model.

Phase 5 measures initialization-time model behavior. The script loads the tiny
text corpus, builds the character tokenizer, samples causal LM windows, runs the
randomly initialized model in inference mode, and prints simple output
statistics plus token-frequency comparisons.

This script intentionally does not train, checkpoint, or persist experiment
logs. Those features are planned for later phases.
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
    load_text_file,
    split_token_ids,
)
from llm_behavior_lab.evaluation import analyze_untrained_outputs  # noqa: E402
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


def resolve_repo_path(path_value: str | Path) -> Path:
    """Resolve relative paths against the repository root."""

    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Analyze the untrained LLaMA-style model.")
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
        "--split",
        choices=["train", "val"],
        default="train",
        help="Token split to sample analysis windows from.",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=4,
        help="Number of random batches/windows to analyze.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top tokens/gaps to display.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=3,
        help="Number of example positions for detailed top-k predictions.",
    )
    return parser.parse_args()


def _format_token(token: str) -> str:
    """Make whitespace tokens readable in printed reports."""

    return repr(token)


def main() -> None:
    """Run the Phase 5 baseline analysis."""

    args = parse_args()
    if args.num_batches <= 0:
        raise ValueError("--num-batches must be positive.")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")

    data_config = load_yaml_config(args.data_config)
    model_config = load_yaml_config(args.model_config)

    runtime_config = data_config.get("runtime", model_config.get("runtime", {}))
    seed = int(runtime_config.get("seed", 1234))
    device_name = str(runtime_config.get("device", "auto"))

    seed_everything(seed)
    device = get_device(device_name)

    dataset_config = data_config["dataset"]
    batching_config = data_config["batching"]
    text_path = resolve_repo_path(dataset_config["path"])
    val_fraction = float(dataset_config.get("val_fraction", 0.1))
    batch_size = int(batching_config["batch_size"])
    block_size = int(batching_config["block_size"])

    text = load_text_file(text_path)
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

    sampled_batches = [batcher.get_batch(args.split) for _ in range(args.num_batches)]
    input_ids = torch.cat([batch.input_ids for batch in sampled_batches], dim=0)

    model = build_model_from_config(model_config).to(device)
    model.eval()

    with torch.no_grad():
        output = model(input_ids=input_ids)

    result = analyze_untrained_outputs(
        logits=output.logits,
        input_ids=input_ids,
        empirical_token_ids=token_ids,
        tokenizer=tokenizer,
        top_k=args.top_k,
        max_examples=args.max_examples,
    )

    parameter_count = model.count_parameters()
    output_summary = result.output_summary

    print("Phase 5 untrained-model analysis completed successfully.")
    print("Note: the model is randomly initialized. These numbers describe baseline behavior, not quality.")
    print(f"Available registered models: {list_models()}")
    print(f"Dataset path: {text_path}")
    print(f"Raw text characters: {len(text)}")
    print(f"Tokenizer type: char")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Total token count: {len(token_ids)}")
    print(f"Train token count: {len(splits.train_ids)}")
    print(f"Validation token count: {len(splits.val_ids)}")
    print(f"Analysis split: {args.split}")
    print(f"Analyzed batches: {args.num_batches}")
    print(f"Analyzed windows: {input_ids.shape[0]}")
    print(f"Analyzed positions: {output_summary.num_positions}")
    print(f"Device: {device}")
    print(f"Model name: {model_config['model']['name']}")
    print(f"Model vocab size: {model_vocab_size}")
    print(f"Parameter count: {parameter_count} ({format_parameter_count(parameter_count)})")
    print(f"Input tensor shape: {tuple(input_ids.shape)}")
    print(f"Logits shape: {output_summary.logits_shape}")
    print(f"Probability tensor shape: {output_summary.probabilities_shape}")
    print(f"Mean output entropy: {output_summary.mean_entropy:.6f}")
    print(f"Min output entropy: {output_summary.min_entropy:.6f}")
    print(f"Max output entropy: {output_summary.max_entropy:.6f}")
    print(f"Mean top-1 probability: {output_summary.mean_top1_probability:.6f}")
    print(f"Mean top-{args.top_k} probability mass: {output_summary.mean_topk_probability_mass:.6f}")
    print(f"Top-1 assignment concentration: {result.top1_concentration:.6f}")
    print(f"KL(predicted || empirical): {result.kl_predicted_to_empirical:.6f}")
    print(f"JS(predicted, empirical): {result.js_predicted_empirical:.6f}")

    print("\nExample top-k next-token predictions:")
    for example in result.topk_examples:
        print(
            f"  batch={example.batch_index}, position={example.position_index}, "
            f"input_token_id={example.input_token_id}, input_token={_format_token(example.input_token)}"
        )
        for rank, prediction in enumerate(example.predictions, start=1):
            print(
                f"    {rank}. token_id={prediction.token_id:<3} "
                f"token={_format_token(prediction.token):<8} probability={prediction.probability:.6f}"
            )

    print("\nMost frequent empirical dataset tokens:")
    for item in result.empirical_top_tokens:
        print(
            f"  token_id={item.token_id:<3} token={_format_token(item.token):<8} "
            f"count={item.count:<4} frequency={item.frequency:.6f}"
        )

    print("\nMost frequent top-1 predicted tokens:")
    for item in result.top1_predicted_tokens:
        print(
            f"  token_id={item.token_id:<3} token={_format_token(item.token):<8} "
            f"count={item.count:<4} frequency={item.frequency:.6f}"
        )

    print("\nLargest positive probability-frequency gaps:")
    for item in result.positive_probability_gaps:
        print(
            f"  token_id={item.token_id:<3} token={_format_token(item.token):<8} "
            f"predicted={item.predicted_probability:.6f} "
            f"empirical={item.empirical_frequency:.6f} gap={item.gap:.6f}"
        )

    print("\nLargest negative probability-frequency gaps:")
    for item in result.negative_probability_gaps:
        print(
            f"  token_id={item.token_id:<3} token={_format_token(item.token):<8} "
            f"predicted={item.predicted_probability:.6f} "
            f"empirical={item.empirical_frequency:.6f} gap={item.gap:.6f}"
        )


if __name__ == "__main__":
    main()
