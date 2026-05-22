"""Baseline analysis for randomly initialized language models.

The functions in this module summarize what an untrained model predicts before
any optimization has occurred. The outputs are intentionally lightweight and
human-readable so they can later be tracked over checkpoints once training and
logging infrastructure are added.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from llm_behavior_lab.data.tokenizer import CharTokenizer
from llm_behavior_lab.evaluation.output_stats import (
    OutputDistributionSummary,
    logits_to_probabilities,
    summarize_output_distribution,
    top1_token_ids,
)
from llm_behavior_lab.evaluation.token_frequency import (
    TokenFrequencySummary,
    TokenProbabilityGap,
    average_predicted_probabilities,
    empirical_token_counts,
    empirical_token_frequencies,
    js_divergence,
    kl_divergence,
    top_probability_gaps,
    top_token_frequencies,
)
from llm_behavior_lab.inference import TopKPrediction, top_k_predictions


@dataclass(frozen=True)
class TopKPositionExample:
    """Top-k predictions for one analyzed batch/sequence position."""

    batch_index: int
    position_index: int
    input_token_id: int
    input_token: str
    predictions: list[TopKPrediction]


@dataclass(frozen=True)
class UntrainedAnalysisResult:
    """Container for baseline initialization-analysis results."""

    output_summary: OutputDistributionSummary
    empirical_top_tokens: list[TokenFrequencySummary]
    top1_predicted_tokens: list[TokenFrequencySummary]
    positive_probability_gaps: list[TokenProbabilityGap]
    negative_probability_gaps: list[TokenProbabilityGap]
    topk_examples: list[TopKPositionExample]
    kl_predicted_to_empirical: float
    js_predicted_empirical: float
    top1_concentration: float


def summarize_top1_predictions(
    probabilities: torch.Tensor,
    tokenizer: CharTokenizer,
    *,
    k: int = 10,
) -> tuple[list[TokenFrequencySummary], float]:
    """Summarize how often each token is the model's top-1 prediction."""

    top1_ids = top1_token_ids(probabilities).reshape(-1).tolist()
    counts = empirical_token_counts(top1_ids, vocab_size=tokenizer.vocab_size)
    summaries = top_token_frequencies(counts, tokenizer, k=k)
    total = counts.sum().item()
    concentration = float(counts.max().item() / total) if total > 0 else 0.0
    return summaries, concentration


def collect_topk_examples(
    probabilities: torch.Tensor,
    input_ids: torch.Tensor,
    tokenizer: CharTokenizer,
    *,
    k: int = 5,
    max_examples: int = 3,
) -> list[TopKPositionExample]:
    """Collect top-k predictions for the first few analyzed positions."""

    if probabilities.ndim != 3:
        raise ValueError(
            f"probabilities must have shape [batch, sequence, vocab], got {tuple(probabilities.shape)}."
        )
    if input_ids.shape != probabilities.shape[:2]:
        raise ValueError(
            "input_ids must match probabilities batch/sequence dimensions, "
            f"got input_ids={tuple(input_ids.shape)} and probabilities={tuple(probabilities.shape)}."
        )
    if max_examples < 0:
        raise ValueError("max_examples must be non-negative.")

    flat_probs = probabilities.reshape(-1, probabilities.shape[-1])
    flat_inputs = input_ids.reshape(-1)
    predictions = top_k_predictions(flat_probs, tokenizer, k=k)

    examples: list[TopKPositionExample] = []
    sequence_length = input_ids.shape[1]
    for flat_index in range(min(max_examples, flat_probs.shape[0])):
        batch_index = flat_index // sequence_length
        position_index = flat_index % sequence_length
        input_token_id = int(flat_inputs[flat_index].item())
        examples.append(
            TopKPositionExample(
                batch_index=batch_index,
                position_index=position_index,
                input_token_id=input_token_id,
                input_token=tokenizer.decode([input_token_id]),
                predictions=predictions[flat_index],
            )
        )
    return examples


def analyze_untrained_outputs(
    *,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    empirical_token_ids: Sequence[int],
    tokenizer: CharTokenizer,
    top_k: int = 5,
    max_examples: int = 3,
) -> UntrainedAnalysisResult:
    """Analyze untrained model outputs against empirical token frequencies."""

    probabilities = logits_to_probabilities(logits, vocab_size_limit=tokenizer.vocab_size)
    output_summary = summarize_output_distribution(logits, probabilities, top_k=top_k)

    empirical_counts = empirical_token_counts(empirical_token_ids, vocab_size=tokenizer.vocab_size)
    empirical_frequencies = empirical_token_frequencies(
        empirical_token_ids,
        vocab_size=tokenizer.vocab_size,
    )
    predicted_probabilities = average_predicted_probabilities(probabilities)

    top1_summaries, top1_concentration = summarize_top1_predictions(
        probabilities,
        tokenizer,
        k=top_k,
    )

    return UntrainedAnalysisResult(
        output_summary=output_summary,
        empirical_top_tokens=top_token_frequencies(empirical_counts, tokenizer, k=top_k),
        top1_predicted_tokens=top1_summaries,
        positive_probability_gaps=top_probability_gaps(
            predicted_probabilities,
            empirical_frequencies,
            tokenizer,
            k=top_k,
            largest=True,
        ),
        negative_probability_gaps=top_probability_gaps(
            predicted_probabilities,
            empirical_frequencies,
            tokenizer,
            k=top_k,
            largest=False,
        ),
        topk_examples=collect_topk_examples(
            probabilities,
            input_ids,
            tokenizer,
            k=top_k,
            max_examples=max_examples,
        ),
        kl_predicted_to_empirical=kl_divergence(predicted_probabilities, empirical_frequencies),
        js_predicted_empirical=js_divergence(predicted_probabilities, empirical_frequencies),
        top1_concentration=top1_concentration,
    )
