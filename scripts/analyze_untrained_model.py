"""Analyze baseline behavior of the untrained LLaMA-style model.

Phase 5 measures initialization-time model behavior. The script loads the tiny
text corpus, builds the character tokenizer, samples causal LM windows, runs the
randomly initialized model in inference mode, and prints simple output
statistics plus token-frequency comparisons.

This refinement also optionally computes per-layer squared L2 gradient norms
with respect to decoder-block outputs. That gradient diagnostic is disabled by
default because it requires a backward pass. Enable it with
``--compute-grad-norms``.

This script intentionally does not train, checkpoint, or introduce full
experiment logging. Those features are planned for later phases.
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
from llm_behavior_lab.evaluation import (  # noqa: E402
    analyze_untrained_outputs,
    compute_per_layer_gradient_norms,
    save_gradient_norm_result,
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
    parser.add_argument(
        "--compute-grad-norms",
        action="store_true",
        help="Compute per-layer squared L2 gradient norms. Disabled by default.",
    )
    parser.add_argument(
        "--grad-norm-num-batches",
        type=int,
        default=1,
        help="Number of batches to use for gradient-norm analysis when enabled.",
    )
    parser.add_argument(
        "--grad-norm-eps",
        type=float,
        default=1e-12,
        help="Numerical stabilizer used in log(g_l + eps) trend fitting.",
    )
    parser.add_argument(
        "--grad-norm-output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "phase5_gradient_norms",
        help="Directory where gradient-norm JSON and CSV files are saved when enabled.",
    )
    return parser.parse_args()


def _format_token(token: str) -> str:
    """Make whitespace tokens readable in printed reports."""

    return repr(token)


def _format_optional_float(value: float | None) -> str:
    """Format optional fit diagnostics for printing."""

    if value is None:
        return "not available"
    return f"{value:.6f}"


def main() -> None:
    """Run the Phase 5 baseline analysis."""

    args = parse_args()
    if args.num_batches <= 0:
        raise ValueError("--num-batches must be positive.")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")
    if args.compute_grad_norms and args.grad_norm_num_batches <= 0:
        raise ValueError("--grad-norm-num-batches must be positive when gradient norms are enabled.")
    if args.compute_grad_norms and args.grad_norm_eps <= 0:
        raise ValueError("--grad-norm-eps must be positive when gradient norms are enabled.")

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
    print(f"Dataset source type: {loaded_dataset.source_type}")
    print(f"Dataset source name: {loaded_dataset.source_name}")
    print(f"Dataset metadata: {loaded_dataset.metadata}")
    print(f"Raw examples used: {loaded_dataset.num_examples}")
    print(f"Raw text characters: {len(loaded_dataset.text)}")
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

    if args.compute_grad_norms:
        print("\nComputing per-layer gradient norms...")
        grad_batches = [batcher.get_batch(args.split) for _ in range(args.grad_norm_num_batches)]
        grad_result = compute_per_layer_gradient_norms(
            model,
            grad_batches,
            eps=args.grad_norm_eps,
        )
        metadata = {
            "data_config": str(args.data_config),
            "model_config": str(args.model_config),
            "dataset_source_type": loaded_dataset.source_type,
            "dataset_source_name": loaded_dataset.source_name,
            "model_name": model_config["model"]["name"],
            "analysis_split": args.split,
            "seed": seed,
            "device": str(device),
            "batch_size": batch_size,
            "block_size": block_size,
            "grad_norm_num_batches": args.grad_norm_num_batches,
        }
        json_path, csv_path = save_gradient_norm_result(
            grad_result,
            args.grad_norm_output_dir,
            metadata=metadata,
        )

        print("\nPer-layer gradient norm diagnostic:")
        print(f"  Definition: {grad_result.definition}")
        print(f"  Gradient batches: {grad_result.num_batches}")
        print(f"  Mean gradient-diagnostic loss: {grad_result.mean_loss:.6f}")
        print(
            "  Squared L2 gradient norms: "
            f"{[round(item.squared_l2_norm, 6) for item in grad_result.layer_norms]}"
        )
        print(
            "  Log squared L2 gradient norms: "
            f"{[round(item.log_squared_l2_norm, 6) for item in grad_result.layer_norms]}"
        )
        print(f"  Log-linear slope: {grad_result.trend_fit.slope:.6f}")
        print(f"  Log-linear intercept: {grad_result.trend_fit.intercept:.6f}")
        print(
            "  Slope standard error: "
            f"{_format_optional_float(grad_result.trend_fit.slope_standard_error)}"
        )
        print(f"  R^2: {_format_optional_float(grad_result.trend_fit.r_squared)}")
        print(f"  Saved gradient JSON: {json_path}")
        print(f"  Saved gradient CSV: {csv_path}")


if __name__ == "__main__":
    main()
