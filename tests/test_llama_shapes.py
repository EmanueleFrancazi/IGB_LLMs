"""Shape and registry tests for the Phase 2 LLaMA-style model."""

from pathlib import Path

import pytest
import torch
import yaml

from llm_behavior_lab.models import build_model, build_model_from_config, list_models
from llm_behavior_lab.models.base import BaseLanguageModel, ModelOutput
from llm_behavior_lab.models.llama import LlamaConfig, LlamaForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[1]
LLAMA_12X768_CONFIG_PATH = REPO_ROOT / "configs" / "model" / "llama_12x768.yaml"


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


@pytest.fixture(scope="module")
def llama_12x768():
    """The 12x768 arm, built once from the tracked config file.

    Module scoped because this is the largest model the suite constructs -- about
    134M parameters and roughly half a gigabyte of float32 -- and every assertion
    below describes the same construction. Building it once is what keeps the
    parameter-budget check honest: the numbers come from the real builder reading
    the real file, not from re-implementing the sizing rule in the test.
    """

    config = yaml.safe_load(LLAMA_12X768_CONFIG_PATH.read_text(encoding="utf-8"))
    return config, build_model_from_config(config)


def test_llama_12x768_uses_the_existing_family_builder(llama_12x768) -> None:
    """Size comes from the config file, not from a size-encoding registry alias."""

    config, model = llama_12x768

    assert config["model"]["name"] == "llama"
    assert isinstance(model, LlamaForCausalLM)


def test_llama_12x768_parameter_budget_leaves_the_vocabulary_a_minority(
    llama_12x768,
) -> None:
    """The realized budget must match the reference arithmetic exactly.

    A wrong ``multiple_of`` or a mistyped dimension would still build and still
    run; only the count reveals it. The per-group split is asserted alongside the
    total so a compensating pair of errors cannot pass.

    The split is also what the arm exists for. Together the assertions below fix
    the vocabulary share at 49,152,000 / 134,105,856 = 36.65%, against 95.41% for
    the 2-layer baseline: the transformer blocks now hold most of the parameters.
    That ratio is not asserted separately because these numbers already determine
    it, and a second check of the same fact could not fail independently.
    """

    _, model = llama_12x768
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]

    block_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith("layers.")
    )

    assert model.tok_embeddings.weight.numel() == 24_576_000
    assert model.output.weight.numel() == 24_576_000
    assert block_parameters == 84_953_088
    assert model.norm.weight.numel() == 768
    assert model.count_parameters() == 134_105_856
    assert len(trainable) == 111


def test_llama_12x768_architecture_fields(llama_12x768) -> None:
    """Depth, feed-forward width, attention layout, and the absence of biases."""

    _, model = llama_12x768

    assert len(model.layers) == 12
    # 4*768 = 3072 -> int(2*3072/3) = 2048 -> already a multiple of 256.
    assert model.layers[0].feed_forward.w1.out_features == 2048
    assert model.layers[0].feed_forward.w2.in_features == 2048

    attention = model.layers[0].attention
    # n_kv_heads == n_heads is standard multi-head attention, not grouped-query.
    assert attention.n_heads_q == 12
    assert attention.n_kv_heads == 12
    assert attention.n_rep == 1
    assert attention.head_dim == 64

    assert [name for name, _ in model.named_parameters() if "bias" in name] == []

    # Untied, unlike the upstream reference: the two vocabulary matrices are
    # distinct storages and are counted separately in the budget above.
    assert model.tok_embeddings.weight.data_ptr() != model.output.weight.data_ptr()


def test_llama_12x768_forward_shape(llama_12x768) -> None:
    """A full evaluation window produces logits over the whole vocabulary."""

    _, model = llama_12x768
    input_ids = torch.randint(0, 32_000, (2, 64))

    output = model(input_ids=input_ids)

    assert output.logits.shape == (2, 64, 32_000)
    assert output.loss is None
