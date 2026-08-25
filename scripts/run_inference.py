"""Run a Phase 4 inference sanity check.

The model is still untrained, so generated text is not expected to be meaningful.
This script validates the inference path: prompt encoding, model-ready tensors,
logits extraction, top-k probabilities, and short greedy or sampling generation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from the repository root before editable installation.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from llm_behavior_lab.data import (  # noqa: E402
    CharTokenizer,
    DatasetConfig,
    add_dataset_arguments,
    resolution_policy_from_args,
    resolve_dataset,
)
from llm_behavior_lab.inference import (  # noqa: E402
    extract_logits,
    extract_next_token_logits,
    generate_text,
    next_token_probabilities,
    prepare_prompt_tensor,
    top_k_predictions,
)
from llm_behavior_lab.models import build_model_from_config, list_models  # noqa: E402
from llm_behavior_lab.utils import format_parameter_count, get_device, load_yaml_config, seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Run the Phase 4 inference check.")
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
        "--prompt",
        type=str,
        default="The ",
        help="Prompt to encode with the character tokenizer.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4,
        help="Number of new tokens to generate.",
    )
    parser.add_argument(
        "--strategy",
        choices=["greedy", "sample"],
        default="greedy",
        help="Decoding strategy.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature. Must be positive.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Optional top-k sampling filter. Used only with --strategy sample.",
    )
    parser.add_argument(
        "--num-top-predictions",
        type=int,
        default=5,
        help="Number of next-token predictions to print.",
    )
    add_dataset_arguments(parser)
    return parser.parse_args()


def main() -> None:
    """Run the inference sanity check."""

    args = parse_args()
    data_config = load_yaml_config(args.data_config)
    model_config = load_yaml_config(args.model_config)

    runtime_config = data_config.get("runtime", model_config.get("runtime", {}))
    seed = int(runtime_config.get("seed", 1234))
    device_name = str(runtime_config.get("device", "auto"))

    seed_everything(seed)
    device = get_device(device_name)

    resolved = resolve_dataset(
        DatasetConfig.from_config(data_config),
        resolution_policy_from_args(args),
        repo_root=REPO_ROOT,
    )
    text = resolved.read_text()
    tokenizer = CharTokenizer.from_text(text)

    model_params = model_config["model"]["params"]
    model_vocab_size = int(model_params["vocab_size"])
    max_context_length = int(model_params["max_seq_len"])
    if tokenizer.vocab_size > model_vocab_size:
        raise ValueError(
            f"Tokenizer vocab size {tokenizer.vocab_size} exceeds model vocab size "
            f"{model_vocab_size}. Increase model.params.vocab_size."
        )

    model = build_model_from_config(model_config).to(device)
    model.eval()

    prompt_batch = prepare_prompt_tensor(
        args.prompt,
        tokenizer,
        max_context_length=max_context_length,
        device=device,
    )
    logits = extract_logits(model, prompt_batch.input_ids)
    next_logits = extract_next_token_logits(
        model,
        prompt_batch.input_ids,
        vocab_size_limit=tokenizer.vocab_size,
    )
    probabilities = next_token_probabilities(next_logits, temperature=args.temperature)
    predictions = top_k_predictions(
        probabilities,
        tokenizer,
        k=args.num_top_predictions,
    )[0]

    generation = generate_text(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        max_context_length=max_context_length,
        device=device,
        strategy=args.strategy,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=seed,
    )

    parameter_count = model.count_parameters()

    print("Phase 4 inference check completed successfully.")
    print(f"Available registered models: {list_models()}")
    print(f"Dataset name: {resolved.name}")
    print(f"Dataset source: {resolved.source}")
    print(f"Dataset resolved via: {resolved.route}")
    print(f"Dataset path: {resolved.path}")
    print(f"Tokenizer type: char")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Model name: {model_config['model']['name']}")
    print(f"Model vocab size: {model_vocab_size}")
    print(f"Device: {device}")
    print(f"Parameter count: {parameter_count} ({format_parameter_count(parameter_count)})")
    print(f"Prompt: {args.prompt!r}")
    print(f"Encoded prompt token IDs: {prompt_batch.token_ids}")
    print(f"Input tensor shape: {tuple(prompt_batch.input_ids.shape)}")
    print(f"Full logits shape: {tuple(logits.shape)}")
    print(f"Next-token logits shape after tokenizer-vocab restriction: {tuple(next_logits.shape)}")
    print("Top-k next-token predictions:")
    for rank, prediction in enumerate(predictions, start=1):
        print(
            f"  {rank}. token_id={prediction.token_id:<3} "
            f"token={prediction.token!r:<6} probability={prediction.probability:.6f}"
        )
    print(f"Decoding strategy: {args.strategy}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-k sampling filter: {args.top_k}")
    print(f"Generated token IDs: {generation.generated_token_ids}")
    print(f"New token IDs: {generation.new_token_ids}")
    print(f"Decoded generated text: {generation.decoded_text!r}")
    print("Note: the model is untrained, so generated text is expected to be random or meaningless.")


if __name__ == "__main__":
    main()
