"""Tests for the Phase 3 text data pipeline."""

import torch

from llm_behavior_lab.data import CausalLMBatcher, CharTokenizer, split_token_ids


def test_char_tokenizer_round_trip() -> None:
    """Encoding then decoding should recover the original text."""

    text = "hello language models"
    tokenizer = CharTokenizer.from_text(text)
    token_ids = tokenizer.encode(text)

    assert tokenizer.vocab_size == len(set(text))
    assert tokenizer.decode(token_ids) == text


def test_split_token_ids_lengths() -> None:
    """Train and validation splits should cover all tokens exactly once."""

    token_ids = list(range(100))
    splits = split_token_ids(token_ids, val_fraction=0.2)

    assert len(splits.train_ids) == 80
    assert len(splits.val_ids) == 20
    assert splits.train_ids + splits.val_ids == token_ids


def test_causal_lm_batcher_shapes_and_shift() -> None:
    """The batcher should create shifted input/target pairs."""

    train_ids = list(range(50))
    val_ids = list(range(50, 100))
    batcher = CausalLMBatcher(
        train_ids=train_ids,
        val_ids=val_ids,
        block_size=8,
        batch_size=4,
        device="cpu",
        seed=1234,
    )

    batch = batcher.get_batch("train")

    assert batch.input_ids.shape == (4, 8)
    assert batch.targets.shape == (4, 8)
    assert batch.input_ids.dtype == torch.long
    assert batch.targets.dtype == torch.long
    assert torch.equal(batch.targets[:, :-1], batch.input_ids[:, 1:])
