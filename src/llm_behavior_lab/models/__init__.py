"""Model package exports and default registrations."""

from llm_behavior_lab.models.base import BaseLanguageModel, ModelOutput
from llm_behavior_lab.models.debug import DebugLanguageModel

# Importing the llama package registers the LLaMA-style builders.
from llm_behavior_lab.models.llama import LlamaConfig, LlamaForCausalLM  # noqa: F401
from llm_behavior_lab.models.registry import (
    build_model,
    build_model_from_config,
    get_model_builder,
    list_models,
    register_model,
)

__all__ = [
    "BaseLanguageModel",
    "ModelOutput",
    "DebugLanguageModel",
    "LlamaConfig",
    "LlamaForCausalLM",
    "build_model",
    "build_model_from_config",
    "get_model_builder",
    "list_models",
    "register_model",
]
