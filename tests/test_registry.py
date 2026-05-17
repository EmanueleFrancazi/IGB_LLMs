"""Tests for model registration and the tiny debug model."""

import torch

from llm_behavior_lab.models import build_model, build_model_from_config, list_models
from llm_behavior_lab.models.base import BaseLanguageModel, ModelOutput


def test_debug_model_is_registered() -> None:
    """Phase 1 should register the smoke-test model by default."""

    assert "debug_tiny_lm" in list_models()


def test_build_debug_model_forward_shapes() -> None:
    """The debug model should satisfy the project shape contract."""

    batch_size = 2
    sequence_length = 8
    vocab_size = 32

    model = build_model(
        "debug_tiny_lm",
        vocab_size=vocab_size,
        block_size=sequence_length,
        embedding_dim=16,
    )

    assert isinstance(model, BaseLanguageModel)

    input_ids = torch.randint(0, vocab_size, (batch_size, sequence_length))
    targets = torch.randint(0, vocab_size, (batch_size, sequence_length))

    output = model(input_ids=input_ids, targets=targets)

    assert isinstance(output, ModelOutput)
    assert output.logits.shape == (batch_size, sequence_length, vocab_size)
    assert output.loss is not None
    assert output.loss.ndim == 0
    assert model.count_parameters() > 0


def test_build_model_from_config() -> None:
    """A dictionary config should be enough to construct a registered model."""

    config = {
        "model": {
            "name": "debug_tiny_lm",
            "params": {
                "vocab_size": 16,
                "block_size": 4,
                "embedding_dim": 8,
            },
        }
    }

    model = build_model_from_config(config)
    input_ids = torch.randint(0, 16, (1, 4))
    output = model(input_ids=input_ids)

    assert output.logits.shape == (1, 4, 16)
    assert output.loss is None
