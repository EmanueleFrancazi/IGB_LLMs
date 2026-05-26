"""Per-layer gradient-norm diagnostics for initialization analysis.

This module computes squared L2 norms of gradients with respect to transformer
layer outputs. The diagnostic is intended for Phase 5 baseline analysis: it
checks whether gradients are roughly stable through depth before any training.

The implementation uses forward hooks, so it does not require changing the
LLaMA-style model code. For each decoder block, we retain the gradient of the
block output tensor and measure ``sum(grad ** 2)`` after backpropagating a
causal language-modeling loss.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch import nn

from llm_behavior_lab.data.dataloader import CausalLMBatch
from llm_behavior_lab.models.base import BaseLanguageModel


@dataclass(frozen=True)
class LayerGradientNorm:
    """Squared L2 gradient norm for one transformer layer."""

    layer_index: int
    squared_l2_norm: float
    log_squared_l2_norm: float


@dataclass(frozen=True)
class GradientTrendFit:
    """Log-linear fit for per-layer gradient norms.

    The fitted model is ``log(g_l + eps) = intercept + slope * layer_index``.
    """

    slope: float
    intercept: float
    slope_standard_error: float | None
    r_squared: float | None
    eps: float


@dataclass(frozen=True)
class GradientNormResult:
    """Complete result for a per-layer gradient-norm diagnostic."""

    layer_norms: list[LayerGradientNorm]
    trend_fit: GradientTrendFit
    mean_loss: float
    num_batches: int
    definition: str


def _get_transformer_layers(model: BaseLanguageModel) -> Sequence[nn.Module]:
    """Return the model's transformer layers.

    The current LLaMA-style model stores decoder blocks in ``model.layers``.
    Keeping this helper small makes it easy to extend later to model families
    with different names for their block stack.
    """

    layers = getattr(model, "layers", None)
    if layers is None:
        raise AttributeError(
            "Gradient-norm analysis expects the model to expose a 'layers' attribute."
        )
    if not isinstance(layers, Sequence) and not isinstance(layers, nn.ModuleList):
        raise TypeError("model.layers must be a sequence or torch.nn.ModuleList.")
    if len(layers) == 0:
        raise ValueError("model.layers must contain at least one layer.")
    return layers


def fit_log_gradient_trend(
    squared_l2_norms: Sequence[float],
    *,
    eps: float = 1e-12,
) -> GradientTrendFit:
    """Fit a log-linear trend to gradient norms across layers.

    Args:
        squared_l2_norms: Raw per-layer squared L2 gradient norms.
        eps: Stabilizer added before taking the logarithm.

    Returns:
        ``GradientTrendFit`` containing slope, intercept, optional standard
        error, and optional R^2.
    """

    if eps <= 0:
        raise ValueError("eps must be positive.")
    if len(squared_l2_norms) == 0:
        raise ValueError("squared_l2_norms must be non-empty.")

    y = torch.tensor([math.log(float(value) + eps) for value in squared_l2_norms], dtype=torch.float64)
    x = torch.arange(len(squared_l2_norms), dtype=torch.float64)

    x_mean = x.mean()
    y_mean = y.mean()
    centered_x = x - x_mean
    centered_y = y - y_mean
    ss_xx = torch.sum(centered_x.square())

    if float(ss_xx.item()) == 0.0:
        slope = 0.0
        intercept = float(y_mean.item())
        slope_standard_error = None
        r_squared = None
    else:
        slope_tensor = torch.sum(centered_x * centered_y) / ss_xx
        intercept_tensor = y_mean - slope_tensor * x_mean
        fitted = intercept_tensor + slope_tensor * x
        residuals = y - fitted
        ss_res = torch.sum(residuals.square())
        ss_tot = torch.sum(centered_y.square())

        slope = float(slope_tensor.item())
        intercept = float(intercept_tensor.item())
        r_squared = None if float(ss_tot.item()) == 0.0 else float((1.0 - ss_res / ss_tot).item())

        degrees_of_freedom = len(squared_l2_norms) - 2
        if degrees_of_freedom > 0:
            residual_variance = ss_res / degrees_of_freedom
            slope_standard_error = float(torch.sqrt(residual_variance / ss_xx).item())
        else:
            slope_standard_error = None

    return GradientTrendFit(
        slope=slope,
        intercept=intercept,
        slope_standard_error=slope_standard_error,
        r_squared=r_squared,
        eps=eps,
    )


def _register_layer_output_hooks(
    layers: Sequence[nn.Module],
    activations: dict[int, torch.Tensor],
) -> list[torch.utils.hooks.RemovableHandle]:
    """Attach hooks that retain gradients on each layer output tensor."""

    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(layer_index: int):
        def hook(_: nn.Module, __: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            if not torch.is_tensor(output):
                raise TypeError("Expected each transformer layer to return a tensor output.")
            output.retain_grad()
            activations[layer_index] = output

        return hook

    for layer_index, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(layer_index)))
    return handles


def compute_per_layer_gradient_norms(
    model: BaseLanguageModel,
    batches: Iterable[CausalLMBatch],
    *,
    eps: float = 1e-12,
) -> GradientNormResult:
    """Compute per-layer squared L2 gradient norms for a set of batches.

    The diagnostic measures gradients with respect to each decoder block's
    output residual-stream tensor. For each batch, it computes the causal LM
    loss, backpropagates, and accumulates ``sum(grad ** 2)`` for each layer
    output. The returned norms are averaged across batches.
    """

    if eps <= 0:
        raise ValueError("eps must be positive.")

    layers = _get_transformer_layers(model)
    num_layers = len(layers)
    accumulated_norms = torch.zeros(num_layers, dtype=torch.float64)
    total_loss = 0.0
    num_batches = 0
    activations: dict[int, torch.Tensor] = {}
    handles = _register_layer_output_hooks(layers, activations)

    try:
        model.eval()
        for batch in batches:
            num_batches += 1
            activations.clear()
            model.zero_grad(set_to_none=True)

            output = model(input_ids=batch.input_ids, targets=batch.targets)
            if output.loss is None:
                raise RuntimeError("Model output did not contain a loss; targets are required.")

            total_loss += float(output.loss.detach().item())
            output.loss.backward()

            for layer_index in range(num_layers):
                activation = activations.get(layer_index)
                if activation is None:
                    raise RuntimeError(f"No activation was captured for layer {layer_index}.")
                if activation.grad is None:
                    raise RuntimeError(f"No gradient was retained for layer {layer_index}.")
                accumulated_norms[layer_index] += activation.grad.detach().double().pow(2).sum().cpu()
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)

    if num_batches == 0:
        raise ValueError("At least one batch is required for gradient-norm analysis.")

    averaged_norms = accumulated_norms / num_batches
    raw_norm_values = [float(value.item()) for value in averaged_norms]
    trend_fit = fit_log_gradient_trend(raw_norm_values, eps=eps)

    layer_norms = [
        LayerGradientNorm(
            layer_index=layer_index,
            squared_l2_norm=value,
            log_squared_l2_norm=math.log(value + eps),
        )
        for layer_index, value in enumerate(raw_norm_values)
    ]

    return GradientNormResult(
        layer_norms=layer_norms,
        trend_fit=trend_fit,
        mean_loss=total_loss / num_batches,
        num_batches=num_batches,
        definition="squared_l2_norm_of_gradient_wrt_decoder_block_output",
    )


def gradient_norm_result_to_dict(
    result: GradientNormResult,
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert a gradient-norm result to a JSON-serializable dictionary."""

    payload: dict[str, Any] = {
        "definition": result.definition,
        "mean_loss": result.mean_loss,
        "num_batches": result.num_batches,
        "layer_indices": [item.layer_index for item in result.layer_norms],
        "squared_l2_norms": [item.squared_l2_norm for item in result.layer_norms],
        "log_squared_l2_norms": [item.log_squared_l2_norm for item in result.layer_norms],
        "trend_fit": asdict(result.trend_fit),
        "per_layer": [asdict(item) for item in result.layer_norms],
    }
    if metadata is not None:
        payload["metadata"] = metadata
    return payload


def save_gradient_norm_result(
    result: GradientNormResult,
    output_dir: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
    json_filename: str = "gradient_norms.json",
    csv_filename: str = "gradient_norms.csv",
) -> tuple[Path, Path]:
    """Save gradient-norm results as JSON summary and per-layer CSV."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_path = output_path / json_filename
    csv_path = output_path / csv_filename

    payload = gradient_norm_result_to_dict(result, metadata=metadata)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["layer_index", "squared_l2_norm", "log_squared_l2_norm"],
        )
        writer.writeheader()
        for item in result.layer_norms:
            writer.writerow(asdict(item))

    return json_path, csv_path
