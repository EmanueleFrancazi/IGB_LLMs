"""Tests for Phase 5 gradient-norm diagnostics."""

import json
import math

import torch

from llm_behavior_lab.data.dataloader import CausalLMBatch
from llm_behavior_lab.evaluation import (
    compute_per_layer_gradient_norms,
    fit_log_gradient_trend,
    save_gradient_norm_result,
)
from llm_behavior_lab.models import build_model


def _build_gradient_test_model():
    """Build a tiny model with enough layers to test trend diagnostics."""

    model = build_model(
        "llama_tiny",
        vocab_size=32,
        dim=32,
        n_layers=3,
        n_heads=4,
        n_kv_heads=2,
        multiple_of=16,
        max_batch_size=4,
        max_seq_len=8,
    )
    return model


def test_fit_log_gradient_trend_recovers_simple_slope() -> None:
    """A perfect exponential sequence should have the expected log-linear slope."""

    norms = [1.0, math.e, math.e**2]
    fit = fit_log_gradient_trend(norms, eps=1e-12)

    assert abs(fit.slope - 1.0) < 1e-6
    assert abs(fit.intercept) < 1e-6
    assert fit.slope_standard_error is not None
    assert fit.slope_standard_error < 1e-6
    assert fit.r_squared is not None
    assert abs(fit.r_squared - 1.0) < 1e-6


def test_compute_per_layer_gradient_norms_and_save(tmp_path) -> None:
    """Gradient diagnostics should return one nonnegative value per model layer."""

    torch.manual_seed(1234)
    model = _build_gradient_test_model()
    input_ids = torch.randint(0, 32, (2, 8))
    targets = torch.randint(0, 32, (2, 8))
    batch = CausalLMBatch(input_ids=input_ids, targets=targets, split="train")

    result = compute_per_layer_gradient_norms(model, [batch], eps=1e-12)

    assert result.num_batches == 1
    assert result.mean_loss > 0
    assert len(result.layer_norms) == 3
    assert len({item.layer_index for item in result.layer_norms}) == 3
    assert all(item.squared_l2_norm >= 0 for item in result.layer_norms)
    assert all(math.isfinite(item.log_squared_l2_norm) for item in result.layer_norms)
    assert math.isfinite(result.trend_fit.slope)
    assert math.isfinite(result.trend_fit.intercept)

    json_path, csv_path = save_gradient_norm_result(
        result,
        tmp_path,
        metadata={"test": True},
    )

    assert json_path.exists()
    assert csv_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["metadata"] == {"test": True}
    assert payload["layer_indices"] == [0, 1, 2]
    assert len(payload["squared_l2_norms"]) == 3
