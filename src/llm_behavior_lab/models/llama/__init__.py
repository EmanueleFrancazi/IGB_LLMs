"""LLaMA-style model package."""

from llm_behavior_lab.models.llama.config import LlamaConfig
from llm_behavior_lab.models.llama.model import (
    LlamaDecoderBlock,
    LlamaFeedForward,
    LlamaForCausalLM,
    LlamaRMSNorm,
    LlamaSelfAttention,
    apply_rotary_embeddings,
    build_llama_model,
    precompute_rotary_frequencies,
    repeat_kv,
)
from llm_behavior_lab.models.registry import register_model

# Register both a generic name and the tiny baseline name. They currently use
# the same builder; the config file determines the actual size.
register_model("llama", build_llama_model)
register_model("llama_tiny", build_llama_model)

__all__ = [
    "LlamaConfig",
    "LlamaForCausalLM",
    "LlamaDecoderBlock",
    "LlamaSelfAttention",
    "LlamaFeedForward",
    "LlamaRMSNorm",
    "build_llama_model",
    "precompute_rotary_frequencies",
    "apply_rotary_embeddings",
    "repeat_kv",
]
