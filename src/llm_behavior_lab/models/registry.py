"""Simple model registry.

The registry decouples scripts from concrete model classes. Training and
inference code can request a model by name, while each model implementation
registers itself in one place.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from llm_behavior_lab.models.base import BaseLanguageModel

ModelBuilder = Callable[..., BaseLanguageModel]

_MODEL_REGISTRY: dict[str, ModelBuilder] = {}


def register_model(name: str, builder: ModelBuilder, *, overwrite: bool = False) -> None:
    """Register a model builder under a string name.

    Args:
        name: Public model identifier used in config files.
        builder: Callable that returns a ``BaseLanguageModel`` instance.
        overwrite: Whether to allow replacing an existing registration.

    Raises:
        ValueError: If the name is empty or already registered.
    """

    normalized_name = name.strip()
    if not normalized_name:
        raise ValueError("Model name must be a non-empty string.")

    if normalized_name in _MODEL_REGISTRY and not overwrite:
        raise ValueError(f"Model '{normalized_name}' is already registered.")

    _MODEL_REGISTRY[normalized_name] = builder


def get_model_builder(name: str) -> ModelBuilder:
    """Return the registered builder for ``name``.

    Raises:
        KeyError: If the requested model is not registered.
    """

    try:
        return _MODEL_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(list_models()) or "<none>"
        raise KeyError(
            f"Unknown model '{name}'. Available models: {available}"
        ) from exc


def build_model(name: str, **model_kwargs: Any) -> BaseLanguageModel:
    """Instantiate a registered model.

    Args:
        name: Registered model name.
        **model_kwargs: Keyword arguments forwarded to the model builder.

    Returns:
        A ``BaseLanguageModel`` instance.
    """

    builder = get_model_builder(name)
    return builder(**model_kwargs)


def build_model_from_config(config: dict[str, Any]) -> BaseLanguageModel:
    """Build a model from a dictionary loaded from a config file.

    Expected format:

    ```yaml
    model:
      name: debug_tiny_lm
      params:
        vocab_size: 128
        block_size: 32
        embedding_dim: 64
    ```
    """

    if "model" not in config:
        raise KeyError("Config must contain a top-level 'model' section.")

    model_config = config["model"]
    name = model_config.get("name")
    if name is None:
        raise KeyError("Config section 'model' must contain a 'name' field.")

    params = model_config.get("params", {})
    if not isinstance(params, dict):
        raise TypeError("Config field 'model.params' must be a dictionary.")

    return build_model(name, **params)


def list_models() -> list[str]:
    """Return registered model names in sorted order."""

    return sorted(_MODEL_REGISTRY.keys())
