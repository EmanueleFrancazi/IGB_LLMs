"""Inference utilities for logits inspection and lightweight generation."""

from llm_behavior_lab.inference.generation import (
    DecodeStrategy,
    GenerationResult,
    PromptBatch,
    TopKPrediction,
    extract_logits,
    extract_next_token_logits,
    generate_text,
    next_token_probabilities,
    prepare_prompt_tensor,
    select_next_token,
    top_k_predictions,
)

__all__ = [
    "DecodeStrategy",
    "GenerationResult",
    "PromptBatch",
    "TopKPrediction",
    "extract_logits",
    "extract_next_token_logits",
    "generate_text",
    "next_token_probabilities",
    "prepare_prompt_tensor",
    "select_next_token",
    "top_k_predictions",
]
