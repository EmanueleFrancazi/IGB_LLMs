"""Parameter-counting utilities."""

from __future__ import annotations

from torch import nn


def count_parameters(model: nn.Module, *, only_trainable: bool = True) -> int:
    """Count parameters in a PyTorch module."""

    parameters = model.parameters()
    if only_trainable:
        parameters = (p for p in parameters if p.requires_grad)
    return sum(p.numel() for p in parameters)


def format_parameter_count(num_parameters: int) -> str:
    """Format a parameter count using compact units."""

    if num_parameters < 1_000:
        return str(num_parameters)
    if num_parameters < 1_000_000:
        return f"{num_parameters / 1_000:.2f}K"
    if num_parameters < 1_000_000_000:
        return f"{num_parameters / 1_000_000:.2f}M"
    return f"{num_parameters / 1_000_000_000:.2f}B"
