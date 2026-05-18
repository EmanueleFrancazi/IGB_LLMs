"""Shape and registry tests for the Phase 2 LLaMA-style model."""

import torch

from llm_behavior_lab.models import build_model, build_model_from_config, list_models
from llm_behavior_lab.models.base import BaseLanguageModel, ModelOutput
from llm_behavior_lab.models.llama import LlamaConfig, LlamaForCausalLM


def test_llama_models_are_registered() -> None:
    """The LLaMA builders should be available through the shared registry."""

    registered = list_models()
    assert "llama" in registered
    assert "llama_tiny" in registered


def test_llama_forward_shape_and_loss() -> None:
    """The tiny LLaMA model should satisfy the project LM shape contract."""

    batch_size = 2
    sequence_length = 8
    vocab_size = 64

    model = build_model(
        "llama_tiny",
        vocab_size=vocab_size,
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        multiple_of=32,
        max_batch_size=4,
        max_seq_len=sequence_length,
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


def test_llama_build_model_from_config() -> None:
    """A dictionary config should instantiate a tiny LLaMA model."""

    config = {
        "model": {
            "name": "llama_tiny",
            "params": {
                "vocab_size": 32,
                "dim": 32,
                "n_layers": 1,
                "n_heads": 4,
                "n_kv_heads": 2,
                "multiple_of": 16,
                "max_batch_size": 2,
                "max_seq_len": 8,
            },
        }
    }

    model = build_model_from_config(config)
    output = model(torch.randint(0, 32, (1, 8)))

    assert output.logits.shape == (1, 8, 32)
    assert output.loss is None


def test_llama_cache_path_one_token() -> None:
    """The integration keeps a one-token cached inference pathway available."""

    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            dim=32,
            n_layers=1,
            n_heads=4,
            n_kv_heads=2,
            multiple_of=16,
            max_batch_size=2,
            max_seq_len=8,
        )
    )

    token_0 = torch.randint(0, 32, (1, 1))
    token_1 = torch.randint(0, 32, (1, 1))

    out_0 = model(token_0, start_pos=0, use_cache=True)
    out_1 = model(token_1, start_pos=1, use_cache=True)

    assert out_0.logits.shape == (1, 1, 32)
    assert out_1.logits.shape == (1, 1, 32)
