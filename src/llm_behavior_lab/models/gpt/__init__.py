"""GPT-2-style model package."""

from llm_behavior_lab.models.gpt.config import GPTConfig
from llm_behavior_lab.models.gpt.model import (
    GPTBlock,
    GPTForCausalLM,
    GPTLayerNorm,
    GPTMLP,
    GPTSelfAttention,
    build_gpt_model,
)
from llm_behavior_lab.models.registry import register_model

# One family name. Size comes from the config file, as it does for the LLaMA
# family, so a new shape is a new YAML rather than a new registry entry.
register_model("gpt2", build_gpt_model)

__all__ = [
    "GPTConfig",
    "GPTForCausalLM",
    "GPTBlock",
    "GPTSelfAttention",
    "GPTMLP",
    "GPTLayerNorm",
    "build_gpt_model",
]
