"""Model package exports and default registrations."""

from llm_behavior_lab.models.base import BaseLanguageModel, ModelOutput

# Importing a family package registers its builders.
from llm_behavior_lab.models.gpt import GPTConfig, GPTForCausalLM  # noqa: F401
from llm_behavior_lab.models.llama import LlamaConfig, LlamaForCausalLM  # noqa: F401
from llm_behavior_lab.models.initialization_scale import (
    DETERMINISTIC_PARAMETER_SUFFIXES,
    classify_parameters,
    initialization_scale_report,
    scale_initialization,
)
from llm_behavior_lab.models.registry import (
    build_model,
    build_model_from_config,
    get_model_builder,
    list_models,
    register_model,
)

__all__ = [
    "DETERMINISTIC_PARAMETER_SUFFIXES",
    "BaseLanguageModel",
    "ModelOutput",
    "GPTConfig",
    "GPTForCausalLM",
    "LlamaConfig",
    "LlamaForCausalLM",
    "build_model",
    "classify_parameters",
    "initialization_scale_report",
    "scale_initialization",
    "build_model_from_config",
    "get_model_builder",
    "list_models",
    "register_model",
]
