"""Inference and lightweight text-generation utilities.

Phase 4 adds model-output inspection and short generation without introducing a
training loop or checkpointing. The utilities are intentionally small and work
with the current character-level tokenizer plus any model implementing the
project's ``BaseLanguageModel`` interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from llm_behavior_lab.data.tokenizer import CharTokenizer
from llm_behavior_lab.models.base import BaseLanguageModel

DecodeStrategy = Literal["greedy", "sample"]


@dataclass(frozen=True)
class PromptBatch:
    """Tokenized prompt prepared for model inference."""

    prompt: str
    token_ids: list[int]
    input_ids: torch.Tensor


@dataclass(frozen=True)
class TopKPrediction:
    """Human-readable representation of one next-token prediction."""

    token_id: int
    token: str
    probability: float


@dataclass(frozen=True)
class GenerationResult:
    """Result returned by ``generate_text``."""

    prompt: str
    prompt_token_ids: list[int]
    generated_token_ids: list[int]
    new_token_ids: list[int]
    decoded_text: str


def prepare_prompt_tensor(
    prompt: str,
    tokenizer: CharTokenizer,
    *,
    max_context_length: int,
    device: torch.device | str,
    truncate_from_left: bool = True,
) -> PromptBatch:
    """Encode a prompt and create a model-ready ``[1, sequence]`` tensor.

    Args:
        prompt: Text prompt to encode.
        tokenizer: Tokenizer used to convert text to token IDs.
        max_context_length: Maximum number of prompt tokens to keep.
        device: Device where the tensor should be placed.
        truncate_from_left: If true, keep the most recent tokens when the prompt
            exceeds the context length. This is the usual behavior for causal LM
            generation.

    Returns:
        ``PromptBatch`` containing the original prompt, retained token IDs, and
        tensor input.
    """

    if not prompt:
        raise ValueError("prompt must be non-empty.")
    if max_context_length <= 0:
        raise ValueError("max_context_length must be positive.")

    token_ids = tokenizer.encode(prompt)
    if len(token_ids) > max_context_length:
        if truncate_from_left:
            token_ids = token_ids[-max_context_length:]
        else:
            token_ids = token_ids[:max_context_length]

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=torch.device(device))
    return PromptBatch(prompt=prompt, token_ids=token_ids, input_ids=input_ids)


def extract_logits(
    model: BaseLanguageModel,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Run a model in eval/no-grad mode and return full logits.

    Args:
        model: Causal language model implementing the project base interface.
        input_ids: Token IDs with shape ``[batch, sequence]``.

    Returns:
        Full logits with shape ``[batch, sequence, vocab_size]``.
    """

    model.eval()
    with torch.no_grad():
        output = model(input_ids=input_ids)
    return output.logits


def extract_next_token_logits(
    model: BaseLanguageModel,
    input_ids: torch.Tensor,
    *,
    vocab_size_limit: int | None = None,
) -> torch.Tensor:
    """Return logits for the next token after the final input position.

    ``vocab_size_limit`` is useful when the model vocabulary is larger than the
    active tokenizer vocabulary. In the current Phase 4 setup, the tiny LLaMA
    config has vocab size 256 while the character tokenizer has fewer valid
    symbols, so generation should restrict choices to decodable token IDs.
    """

    logits = extract_logits(model, input_ids)
    next_token_logits = logits[:, -1, :]
    if vocab_size_limit is not None:
        if vocab_size_limit <= 0:
            raise ValueError("vocab_size_limit must be positive when provided.")
        next_token_logits = next_token_logits[:, :vocab_size_limit]
    return next_token_logits


def next_token_probabilities(
    next_token_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Convert next-token logits to probabilities with temperature scaling."""

    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    return F.softmax(next_token_logits.float() / temperature, dim=-1)


def top_k_predictions(
    probabilities: torch.Tensor,
    tokenizer: CharTokenizer,
    *,
    k: int = 5,
) -> list[list[TopKPrediction]]:
    """Return top-k token predictions for each batch item.

    Args:
        probabilities: Tensor with shape ``[batch, vocab]``.
        tokenizer: Tokenizer used to decode token IDs.
        k: Number of predictions to return per batch item.

    Returns:
        A nested list with one prediction list per batch item.
    """

    if probabilities.ndim != 2:
        raise ValueError(
            f"probabilities must have shape [batch, vocab], got {tuple(probabilities.shape)}."
        )
    if k <= 0:
        raise ValueError("k must be positive.")

    k = min(k, probabilities.shape[-1])
    values, indices = torch.topk(probabilities, k=k, dim=-1)

    rows: list[list[TopKPrediction]] = []
    for row_values, row_indices in zip(values, indices, strict=True):
        row: list[TopKPrediction] = []
        for value, token_id_tensor in zip(row_values, row_indices, strict=True):
            token_id = int(token_id_tensor.item())
            token = tokenizer.decode([token_id])
            row.append(
                TopKPrediction(
                    token_id=token_id,
                    token=token,
                    probability=float(value.item()),
                )
            )
        rows.append(row)
    return rows


def _apply_top_k_filter(logits: torch.Tensor, top_k: int | None) -> torch.Tensor:
    """Mask logits outside the top-k set before sampling."""

    if top_k is None:
        return logits
    if top_k <= 0:
        raise ValueError("top_k must be positive when provided.")

    top_k = min(top_k, logits.shape[-1])
    values, _ = torch.topk(logits, k=top_k, dim=-1)
    threshold = values[:, [-1]]
    return logits.masked_fill(logits < threshold, float("-inf"))


def select_next_token(
    next_token_logits: torch.Tensor,
    *,
    strategy: DecodeStrategy = "greedy",
    temperature: float = 1.0,
    top_k: int | None = None,
    generator: torch.Generator | None = None,
) -> int:
    """Select one next-token ID from logits using greedy or sampling decoding."""

    if next_token_logits.ndim != 2 or next_token_logits.shape[0] != 1:
        raise ValueError(
            "select_next_token currently expects next_token_logits with shape [1, vocab]."
        )
    if strategy not in {"greedy", "sample"}:
        raise ValueError("strategy must be either 'greedy' or 'sample'.")

    if strategy == "greedy":
        return int(torch.argmax(next_token_logits, dim=-1).item())

    filtered_logits = _apply_top_k_filter(next_token_logits, top_k)
    probabilities = next_token_probabilities(filtered_logits, temperature=temperature)
    sampled = torch.multinomial(probabilities, num_samples=1, generator=generator)
    return int(sampled.item())


def generate_text(
    model: BaseLanguageModel,
    tokenizer: CharTokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    max_context_length: int,
    device: torch.device | str,
    strategy: DecodeStrategy = "greedy",
    temperature: float = 1.0,
    top_k: int | None = None,
    seed: int | None = None,
) -> GenerationResult:
    """Generate a short continuation from a prompt.

    The implementation recomputes the full context at every step. This is simple
    and reliable for Phase 4. The model's KV-cache path can be used later when
    generation performance becomes important.
    """

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative.")

    device = torch.device(device)
    prompt_batch = prepare_prompt_tensor(
        prompt,
        tokenizer,
        max_context_length=max_context_length,
        device=device,
    )
    generated_ids = list(prompt_batch.token_ids)
    new_token_ids: list[int] = []

    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

    for _ in range(max_new_tokens):
        context_ids = generated_ids[-max_context_length:]
        input_ids = torch.tensor([context_ids], dtype=torch.long, device=device)
        next_logits = extract_next_token_logits(
            model,
            input_ids,
            vocab_size_limit=tokenizer.vocab_size,
        )
        next_token_id = select_next_token(
            next_logits,
            strategy=strategy,
            temperature=temperature,
            top_k=top_k,
            generator=generator,
        )
        generated_ids.append(next_token_id)
        new_token_ids.append(next_token_id)

    decoded_text = tokenizer.decode(generated_ids)
    return GenerationResult(
        prompt=prompt,
        prompt_token_ids=prompt_batch.token_ids,
        generated_token_ids=generated_ids,
        new_token_ids=new_token_ids,
        decoded_text=decoded_text,
    )
