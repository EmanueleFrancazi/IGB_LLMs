"""Analyze baseline behavior of the untrained LLaMA-style model.

Phase 5 measures initialization-time model behavior. The script loads the tiny
text corpus, builds the character tokenizer, samples causal LM windows, runs the
randomly initialized model in inference mode, and prints output statistics plus
token-frequency comparisons.

Per-layer squared L2 gradient norms remain optional because they require a
backward pass. Phase 6 additionally makes structured persistence optional with
``--persist-run``. Without that flag, the script preserves the original
print-only behavior.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, replace
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
    DatasetConfig,
    add_dataset_arguments,
    resolution_policy_from_args,
    resolve_dataset,
    resolve_repo_path,
    split_token_ids,
)
from llm_behavior_lab.evaluation import (  # noqa: E402
    analyze_untrained_outputs,
    average_predicted_probabilities,
    compute_per_layer_gradient_norms,
    empirical_token_counts,
    empirical_token_frequencies,
    gradient_norm_result_to_dict,
    logits_to_probabilities,
    save_gradient_norm_result,
    top1_token_ids,
)
from llm_behavior_lab.experiment import (  # noqa: E402
    ExperimentRun,
    experiment_settings_from_config,
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
        help=(
            "Legacy output directory for standalone gradient JSON/CSV files. "
            "Used only when --persist-run is not enabled."
        ),
    )
    parser.add_argument(
        "--persist-run",
        action="store_true",
        help="Persist configs, metadata, metrics, arrays, analyses, and an initialized checkpoint.",
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=REPO_ROOT / "configs" / "experiment" / "untrained_baseline.yaml",
        help="Phase 6 experiment-persistence config used with --persist-run.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional explicit run ID. Existing runs are never overwritten.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output root override for persisted experiment runs.",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default=None,
        help="Optional run notes overriding experiment-config notes.",
    )
    add_dataset_arguments(parser)
    return parser.parse_args()


def _format_token(token: str) -> str:
    """Make whitespace tokens readable in printed reports."""

    return repr(token)


def _format_optional_float(value: float | None) -> str:
    """Format optional fit diagnostics for printing."""

    if value is None:
        return "not available"
    return f"{value:.6f}"


def _persist_analysis_run(
    *,
    args: argparse.Namespace,
    data_config: dict[str, Any],
    model_config: dict[str, Any],
    experiment_config: dict[str, Any],
    seed: int,
    device: torch.device,
    text_path: Path,
    tokenizer: CharTokenizer,
    token_ids: list[int],
    input_ids: torch.Tensor,
    logits: torch.Tensor,
    model: torch.nn.Module,
    parameter_count: int,
    analysis_result: Any,
    gradient_result: Any | None,
    dataset_provenance: dict[str, Any] | None = None,
) -> ExperimentRun:
    """Persist one initialization-analysis run through Phase 6 interfaces."""

    settings = experiment_settings_from_config(experiment_config)
    output_dir = settings.output_dir if args.output_dir is None else args.output_dir
    settings = replace(
        settings,
        output_dir=resolve_repo_path(output_dir),
        seed=seed,
        notes=settings.notes if args.notes is None else args.notes,
    )

    run = ExperimentRun.create(
        settings,
        run_id=args.run_id,
        repo_root=REPO_ROOT,
        metadata={
            "experiment_type": "untrained_model_analysis",
            "device": str(device),
            "model_name": model_config["model"]["name"],
            "model_parameter_count": parameter_count,
            "dataset_path": str(text_path),
            "dataset": dataset_provenance or {},
            # Derived from the tokenizer rather than asserted here, so the
            # recorded identity cannot drift from the one actually used.
            "tokenizer": tokenizer.describe(),
            "analysis": {
                "split": args.split,
                "num_batches": args.num_batches,
                "top_k": args.top_k,
                "max_examples": args.max_examples,
                "compute_grad_norms": bool(args.compute_grad_norms),
            },
        },
    )
    run.snapshot_configs(
        model_config=model_config,
        data_config=data_config,
        experiment_config=experiment_config,
    )

    probabilities = logits_to_probabilities(logits, vocab_size_limit=tokenizer.vocab_size)
    mean_predicted_probabilities = average_predicted_probabilities(probabilities)
    # Same device-alignment policy as analyze_untrained_outputs: the corpus
    # vector is built on the CPU from token IDs, the model's probabilities live
    # on the model's device, and the subtraction below is where they meet.
    # Persistence is unaffected either way -- the array store already moves
    # tensors to the CPU before saving.
    empirical_frequencies = empirical_token_frequencies(
        token_ids,
        vocab_size=tokenizer.vocab_size,
    ).to(mean_predicted_probabilities.device)
    top1_ids = top1_token_ids(probabilities).reshape(-1).tolist()
    top1_counts = empirical_token_counts(top1_ids, vocab_size=tokenizer.vocab_size)
    probability_frequency_gaps = mean_predicted_probabilities - empirical_frequencies

    distribution_artifact = run.array_metrics.save(
        metric_name="untrained_output_distributions",
        step=0,
        stage="initialization",
        split=args.split,
        arrays={
            "mean_predicted_probabilities": mean_predicted_probabilities,
            "empirical_token_frequencies": empirical_frequencies,
            "probability_frequency_gaps": probability_frequency_gaps,
            "top1_assignment_counts": top1_counts,
        },
    )

    checkpoint_info = None
    if settings.checkpointing.enabled:
        checkpoint_info = run.checkpoints.save(
            model=model,
            step=0,
            model_config=model_config,
            additional_state={
                "stage": "initialization",
                "analysis_split": args.split,
                "tokenizer_vocab_size": tokenizer.vocab_size,
            },
        )

    checkpoint_id = None if checkpoint_info is None else checkpoint_info.checkpoint_id
    output_summary = analysis_result.output_summary
    run.evaluation_metrics.log(
        step=0,
        stage="initialization",
        split=args.split,
        checkpoint_id=checkpoint_id,
        metrics={
            "mean_output_entropy": output_summary.mean_entropy,
            "min_output_entropy": output_summary.min_entropy,
            "max_output_entropy": output_summary.max_entropy,
            "mean_top1_probability": output_summary.mean_top1_probability,
            "mean_topk_probability_mass": output_summary.mean_topk_probability_mass,
            "top1_concentration": analysis_result.top1_concentration,
            "kl_predicted_to_empirical": analysis_result.kl_predicted_to_empirical,
            "js_predicted_empirical": analysis_result.js_predicted_empirical,
            "analyzed_positions": output_summary.num_positions,
        },
        artifacts={"output_distributions": distribution_artifact.relative_path},
    )

    analysis_payload = asdict(analysis_result)
    analysis_payload["array_artifact"] = distribution_artifact.relative_path
    analysis_payload["checkpoint_id"] = checkpoint_id
    run.save_analysis_json("untrained_analysis.json", analysis_payload)

    if gradient_result is not None:
        gradient_artifact = run.array_metrics.save(
            metric_name="per_layer_gradient_norms",
            step=0,
            stage="initialization",
            split=args.split,
            checkpoint_id=checkpoint_id,
            arrays={
                "layer_indices": [item.layer_index for item in gradient_result.layer_norms],
                "squared_l2_norms": [item.squared_l2_norm for item in gradient_result.layer_norms],
                "log_squared_l2_norms": [
                    item.log_squared_l2_norm for item in gradient_result.layer_norms
                ],
            },
        )
        run.evaluation_metrics.log(
            step=0,
            stage="initialization_gradient_diagnostic",
            split=args.split,
            checkpoint_id=checkpoint_id,
            metrics={
                "gradient_diagnostic_mean_loss": gradient_result.mean_loss,
                "gradient_log_slope": gradient_result.trend_fit.slope,
                "gradient_log_intercept": gradient_result.trend_fit.intercept,
                "gradient_slope_standard_error": gradient_result.trend_fit.slope_standard_error,
                "gradient_r_squared": gradient_result.trend_fit.r_squared,
            },
            artifacts={"per_layer_gradient_norms": gradient_artifact.relative_path},
        )
        gradient_payload = gradient_norm_result_to_dict(
            gradient_result,
            metadata={
                "array_artifact": gradient_artifact.relative_path,
                "checkpoint_id": checkpoint_id,
            },
        )
        run.save_analysis_json("gradient_norm_analysis.json", gradient_payload)

    return run


def main() -> None:
    """Run the Phase 5 baseline analysis with optional Phase 6 persistence."""

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
    experiment_config = load_yaml_config(args.experiment_config) if args.persist_run else None

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
    print(f"Dataset name: {resolved.name}")
    print(f"Dataset source: {resolved.source}")
    print(f"Dataset resolved via: {resolved.route}")
    print(f"Dataset path: {text_path}")
    print(f"Raw text characters: {len(text)}")
    print("Tokenizer type: char")
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

    gradient_result = None
    if args.compute_grad_norms:
        print("\nComputing per-layer gradient norms...")
        grad_batches = [batcher.get_batch(args.split) for _ in range(args.grad_norm_num_batches)]
        gradient_result = compute_per_layer_gradient_norms(
            model,
            grad_batches,
            eps=args.grad_norm_eps,
        )

        print("\nPer-layer gradient norm diagnostic:")
        print(f"  Definition: {gradient_result.definition}")
        print(f"  Gradient batches: {gradient_result.num_batches}")
        print(f"  Mean gradient-diagnostic loss: {gradient_result.mean_loss:.6f}")
        print(
            "  Squared L2 gradient norms: "
            f"{[round(item.squared_l2_norm, 6) for item in gradient_result.layer_norms]}"
        )
        print(
            "  Log squared L2 gradient norms: "
            f"{[round(item.log_squared_l2_norm, 6) for item in gradient_result.layer_norms]}"
        )
        print(f"  Log-linear slope: {gradient_result.trend_fit.slope:.6f}")
        print(f"  Log-linear intercept: {gradient_result.trend_fit.intercept:.6f}")
        print(
            "  Slope standard error: "
            f"{_format_optional_float(gradient_result.trend_fit.slope_standard_error)}"
        )
        print(f"  R^2: {_format_optional_float(gradient_result.trend_fit.r_squared)}")

        if not args.persist_run:
            metadata = {
                "data_config": str(args.data_config),
                "model_config": str(args.model_config),
                "dataset_path": str(text_path),
                "dataset_name": resolved.name,
                "dataset_route": resolved.route,
                "model_name": model_config["model"]["name"],
                "analysis_split": args.split,
                "seed": seed,
                "device": str(device),
                "batch_size": batch_size,
                "block_size": block_size,
                "grad_norm_num_batches": args.grad_norm_num_batches,
            }
            json_path, csv_path = save_gradient_norm_result(
                gradient_result,
                args.grad_norm_output_dir,
                metadata=metadata,
            )
            print(f"  Saved gradient JSON: {json_path}")
            print(f"  Saved gradient CSV: {csv_path}")

    if args.persist_run:
        if experiment_config is None:
            raise RuntimeError("Experiment config was not loaded for a persisted run.")
        run = _persist_analysis_run(
            args=args,
            data_config=data_config,
            model_config=model_config,
            experiment_config=experiment_config,
            seed=seed,
            device=device,
            text_path=text_path,
            tokenizer=tokenizer,
            token_ids=token_ids,
            input_ids=input_ids,
            logits=output.logits,
            model=model,
            parameter_count=parameter_count,
            analysis_result=result,
            gradient_result=gradient_result,
            dataset_provenance=resolved.provenance(),
        )
        print("\nStructured experiment persistence completed successfully.")
        print(f"Run directory: {run.paths.run_dir}")
        print(f"Metadata: {run.paths.metadata_path}")
        print(f"Evaluation metrics: {run.evaluation_metrics.path}")
        latest_checkpoint = run.checkpoints.latest_checkpoint()
        if latest_checkpoint is not None:
            print(f"Initialized checkpoint: {latest_checkpoint.path}")


if __name__ == "__main__":
    main()
