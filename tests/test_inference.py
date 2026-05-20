"""Tests for Phase 4 inference utilities."""

import torch

from llm_behavior_lab.data import CharTokenizer
from llm_behavior_lab.inference import (
    extract_logits,
    extract_next_token_logits,
    generate_text,
    next_token_probabilities,
    prepare_prompt_tensor,
    select_next_token,
    top_k_predictions,
)
from llm_behavior_lab.models import build_model


def _build_tiny_model(vocab_size: int = 32):
    """Build a small LLaMA-style model for inference tests."""

    model = build_model(
        "llama_tiny",
        vocab_size=vocab_size,
        dim=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        multiple_of=16,
        max_batch_size=2,
        max_seq_len=16,
    )
    model.eval()
    return model


def test_prepare_prompt_tensor_truncates_from_left() -> None:
    """Prompt preparation should encode, truncate, and add a batch dimension."""

    tokenizer = CharTokenizer.from_text("abcdef")
    batch = prepare_prompt_tensor(
        "abcdef",
        tokenizer,
        max_context_length=3,
        device="cpu",
    )

    assert batch.token_ids == tokenizer.encode("def")
    assert batch.input_ids.shape == (1, 3)
    assert batch.input_ids.dtype == torch.long


def test_logits_and_probabilities_shapes() -> None:
    """Logits extraction and probability conversion should preserve expected shapes."""

    tokenizer = CharTokenizer.from_text("hello world")
    model = _build_tiny_model(vocab_size=32)
    prompt_batch = prepare_prompt_tensor(
        "hello",
        tokenizer,
        max_context_length=16,
        device="cpu",
    )

    logits = extract_logits(model, prompt_batch.input_ids)
    next_logits = extract_next_token_logits(
        model,
        prompt_batch.input_ids,
        vocab_size_limit=tokenizer.vocab_size,
    )
    probabilities = next_token_probabilities(next_logits)

    assert logits.shape == (1, 5, 32)
    assert next_logits.shape == (1, tokenizer.vocab_size)
    assert probabilities.shape == (1, tokenizer.vocab_size)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(1), atol=1e-6)


def test_top_k_predictions_are_decodable() -> None:
    """Top-k predictions should include token IDs, token strings, and probabilities."""

    tokenizer = CharTokenizer.from_text("abc")
    probabilities = torch.tensor([[0.1, 0.7, 0.2]])

    predictions = top_k_predictions(probabilities, tokenizer, k=2)[0]

    assert [prediction.token_id for prediction in predictions] == [1, 2]
    assert [prediction.token for prediction in predictions] == ["b", "c"]
    assert predictions[0].probability > predictions[1].probability


def test_select_next_token_greedy() -> None:
    """Greedy decoding should select the maximum-logit token."""

    logits = torch.tensor([[0.0, 3.0, 1.0]])
    assert select_next_token(logits, strategy="greedy") == 1


def test_generate_text_returns_decodable_ids() -> None:
    """Generation should restrict output to tokenizer-decodable IDs."""

    text = "The quick brown fox"
    tokenizer = CharTokenizer.from_text(text)
    model = _build_tiny_model(vocab_size=64)

    result = generate_text(
        model,
        tokenizer,
        "The ",
        max_new_tokens=4,
        max_context_length=16,
        device="cpu",
        strategy="greedy",
    )

    assert result.prompt_token_ids == tokenizer.encode("The ")
    assert len(result.new_token_ids) == 4
    assert len(result.generated_token_ids) == len(result.prompt_token_ids) + 4
    assert result.decoded_text == tokenizer.decode(result.generated_token_ids)
