"""Tests for the per-position gradient count sketch.

WRITTEN BUT NOT EXECUTED LOCALLY. Every case here needs PyTorch, which is not
installed in the development workspace. They are written to be run on the
cluster.

The sketch is only worth having if it preserves the quantity the clustering
analysis reads out of it, so the central test compares cosines estimated from
sketches against cosines of the **exact** gradients on a model small enough to
hold them. The tolerance is derived from the projection's own error scale rather
than tuned until it passes: a count sketch estimates inner products with a
relative error of order ``1/sqrt(K)``.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
import torch.nn.functional as F

from llm_behavior_lab.evaluation.init_distribution import build_evaluation_positions
from llm_behavior_lab.evaluation.position_gradients import (
    CANONICAL_GRADIENT_TEMPERATURE,
    DEFAULT_SKETCH_DIMENSION,
    _GradientSketcher,
    compute_position_gradient_norms,
)

VOCAB = 12
BLOCK = 4
WINDOWS = 4
D = WINDOWS * BLOCK
TOKENS = [(index * 5 + 2) % VOCAB for index in range(64)]


class TinyModel(torch.nn.Module):
    """Small enough that every exact per-position gradient can be held."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.embedding = torch.nn.Embedding(VOCAB, 8)
        self.output = torch.nn.Linear(8, VOCAB, bias=False)
        with torch.no_grad():
            self.embedding.weight.copy_(
                torch.randn(VOCAB, 8, generator=generator, dtype=torch.float64)
            )
            self.output.weight.copy_(
                torch.randn(VOCAB, 8, generator=generator, dtype=torch.float64)
            )
        self.double()

    def forward(self, input_ids: torch.Tensor):
        return type("Output", (), {"logits": self.output(self.embedding(input_ids))})()


def _positions():
    return build_evaluation_positions(TOKENS, block_size=BLOCK, num_windows=WINDOWS)


def _exact_gradients(model, positions) -> torch.Tensor:
    """``[D, P]`` exact flattened gradients -- only possible on a tiny model."""

    parameters = [p for p in model.parameters() if p.requires_grad]
    rows = []
    for window in range(WINDOWS):
        for offset in range(BLOCK):
            logits = model(positions.input_ids[window : window + 1]).logits[0, offset]
            loss = F.cross_entropy(
                logits[None, :] / CANONICAL_GRADIENT_TEMPERATURE,
                positions.target_ids[window, offset][None],
            )
            grads = torch.autograd.grad(loss, parameters, allow_unused=True)
            rows.append(torch.cat([g.detach().double().reshape(-1) for g in grads]))
    return torch.stack(rows)


def _cosines(rows: torch.Tensor) -> torch.Tensor:
    unit = rows / rows.norm(dim=1, keepdim=True)
    return unit @ unit.T


def test_the_sketch_preserves_cosines_within_its_error_scale() -> None:
    """The measurement that matters: sketch cosines vs exact-gradient cosines."""

    model = TinyModel()
    positions = _positions()

    result = compute_position_gradient_norms(
        model, positions, vocab_size=VOCAB, gradient_sketch=True,
        sketch_dimension=DEFAULT_SKETCH_DIMENSION,
    )
    exact = _cosines(_exact_gradients(model, positions))
    estimated = _cosines(result.gradient_sketches.double())

    off = ~torch.eye(D, dtype=torch.bool)
    error = (estimated[off] - exact[off]).abs()
    # A count sketch's inner-product error scales as 1/sqrt(K); at K = 512 that
    # is about 4%. Allowing 3x that is a generous but still meaningful bound.
    assert float(error.mean()) < 0.15
    assert float(error.max()) < 0.40


def test_a_wider_sketch_is_more_accurate() -> None:
    """Accuracy must actually improve with K, or the projection is not working."""

    model = TinyModel()
    positions = _positions()
    exact = _cosines(_exact_gradients(model, positions))
    off = ~torch.eye(D, dtype=torch.bool)

    errors = {}
    for dimension in (32, 1024):
        result = compute_position_gradient_norms(
            model, positions, vocab_size=VOCAB, gradient_sketch=True,
            sketch_dimension=dimension,
        )
        estimated = _cosines(result.gradient_sketches.double())
        errors[dimension] = float((estimated[off] - exact[off]).abs().mean())

    assert errors[1024] < errors[32]


def test_the_sketch_is_an_unbiased_inner_product_estimator() -> None:
    """E<sketch(a), sketch(b)> = <a, b>: check the mean over many seeds."""

    generator = torch.Generator().manual_seed(3)
    left = torch.randn(2048, generator=generator, dtype=torch.float64)
    right = torch.randn(2048, generator=generator, dtype=torch.float64)
    truth = float(left @ right)

    estimates = []
    for seed in range(200):
        sketcher = _GradientSketcher([left], dimension=64, seed=seed)
        estimates.append(
            float(sketcher.project([left]) @ sketcher.project([right]))
        )

    mean = sum(estimates) / len(estimates)
    spread = (sum((value - mean) ** 2 for value in estimates) / len(estimates)) ** 0.5
    # The mean of 200 draws should sit well inside the standard error.
    assert abs(mean - truth) < 3.0 * spread / len(estimates) ** 0.5


def test_the_projection_is_deterministic_across_runs() -> None:
    model = TinyModel()
    positions = _positions()

    first = compute_position_gradient_norms(
        model, positions, vocab_size=VOCAB, gradient_sketch=True
    )
    second = compute_position_gradient_norms(
        model, positions, vocab_size=VOCAB, gradient_sketch=True
    )

    assert torch.equal(first.gradient_sketches, second.gradient_sketches)


def test_a_different_seed_gives_a_different_projection() -> None:
    model = TinyModel()
    positions = _positions()

    default = compute_position_gradient_norms(
        model, positions, vocab_size=VOCAB, gradient_sketch=True
    )
    other = compute_position_gradient_norms(
        model, positions, vocab_size=VOCAB, gradient_sketch=True, sketch_seed=99
    )

    assert not torch.equal(default.gradient_sketches, other.gradient_sketches)


def test_sketching_does_not_change_the_existing_results() -> None:
    """The established observables must be bitwise identical with it on."""

    model = TinyModel()
    positions = _positions()

    without = compute_position_gradient_norms(model, positions, vocab_size=VOCAB)
    with_sketch = compute_position_gradient_norms(
        model, positions, vocab_size=VOCAB, gradient_sketch=True
    )

    assert torch.equal(without.gradient_norms, with_sketch.gradient_norms)
    assert torch.equal(
        without.temperature_gradient_norms, with_sketch.temperature_gradient_norms
    )
    assert torch.equal(without.greedy_ids, with_sketch.greedy_ids)
    assert torch.equal(without.target_ids, with_sketch.target_ids)
    assert without.gradient_sketches is None


def test_the_sketch_is_off_by_default() -> None:
    result = compute_position_gradient_norms(TinyModel(), _positions(), vocab_size=VOCAB)

    assert result.gradient_sketches is None
    assert result.sketch_protocol is None
    assert "gradient_sketch" not in result.as_metadata()


def test_the_sketch_shape_and_protocol_are_recorded() -> None:
    result = compute_position_gradient_norms(
        TinyModel(), _positions(), vocab_size=VOCAB, gradient_sketch=True
    )

    assert result.gradient_sketches.shape == (D, DEFAULT_SKETCH_DIMENSION)
    protocol = result.as_metadata()["gradient_sketch"]
    assert protocol["dimension"] == DEFAULT_SKETCH_DIMENSION
    assert protocol["temperature"] == CANONICAL_GRADIENT_TEMPERATURE
    assert protocol["projection"] == "count_sketch_signed_feature_hashing"


def test_only_the_canonical_temperature_is_sketched() -> None:
    """One row per position, not one per temperature."""

    result = compute_position_gradient_norms(
        TinyModel(), _positions(), vocab_size=VOCAB,
        temperatures=(0.5, CANONICAL_GRADIENT_TEMPERATURE, 2.0),
        gradient_sketch=True,
    )

    assert result.gradient_sketches.shape[0] == D


def test_sketching_leaves_model_state_and_rng_untouched() -> None:
    """It must not consume the global RNG or disturb the model."""

    model = TinyModel()
    positions = _positions()
    parameters = {name: p.detach().clone() for name, p in model.named_parameters()}
    sentinel = torch.full_like(model.output.weight, 2.25)
    model.output.weight.grad = sentinel.clone()
    model.train()
    rng_state = torch.get_rng_state()

    compute_position_gradient_norms(
        model, positions, vocab_size=VOCAB, gradient_sketch=True, vector_split=True
    )

    for name, parameter in model.named_parameters():
        assert torch.equal(parameter.detach(), parameters[name]), name
    assert torch.equal(model.output.weight.grad, sentinel)
    assert model.training is True
    assert torch.equal(torch.get_rng_state(), rng_state)


def test_clustering_recovers_planted_structure_from_real_gradients() -> None:
    """End to end: identical targets should give more aligned gradients.

    Positions sharing a target share a loss, so this is the weakest form of the
    hypothesis the diagnostic exists to test -- and it must be visible through
    the sketch, not only in the exact gradients.
    """

    import numpy as np

    from llm_behavior_lab.analysis.gradient_clustering import class_similarity_matrix

    model = TinyModel()
    base = _positions()
    # Two target classes, each repeated across many positions.
    targets = base.target_ids.clone().reshape(-1)
    targets[: D // 2] = 3
    targets[D // 2 :] = 7
    positions = dataclasses.replace(
        base, target_ids=targets.reshape(WINDOWS, BLOCK)
    )

    result = compute_position_gradient_norms(
        model, positions, vocab_size=VOCAB, gradient_sketch=True
    )
    sketches = result.gradient_sketches.double().numpy()
    unit = sketches / np.linalg.norm(sketches, axis=1, keepdims=True)
    labels = targets.numpy()

    matrix = class_similarity_matrix(unit, labels, np.array([3, 7]))["matrix"]

    assert np.isfinite(matrix).all()
    # Within-class must exceed between-class for both classes.
    assert matrix[0, 0] > matrix[0, 1]
    assert matrix[1, 1] > matrix[0, 1]


def test_sketched_aggregates_match_the_exact_vector_split() -> None:
    """Cross-check the sketch against the exact correct/wrong aggregate vectors.

    The vector-split diagnostic already computes ``||g_correct||``,
    ``||g_wrong||``, ``||g_correct + g_wrong||``, their dot product and cosine
    from the true gradients. Summing the *sketches* over the same two groups must
    reproduce all five within the projection's error, which is the most direct
    evidence available that the sketch carries the geometry it claims to.

    The sketch is linear, so ``sum_d sketch(g_d) = sketch(sum_d g_d)``: the
    comparison is between one sketch of an aggregate and that aggregate's exact
    norm, and its error scale is the usual ``1/sqrt(K)``.
    """

    import numpy as np

    model = TinyModel()
    positions = _positions()

    result = compute_position_gradient_norms(
        model, positions, vocab_size=VOCAB,
        gradient_sketch=True, sketch_dimension=2048, vector_split=True,
    )
    exact = result.vector_split
    assert exact is not None

    # Sketches are stored scaled by nothing; rebuild the group aggregates.
    sketches = result.gradient_sketches.double().numpy()
    correct = np.asarray(result.greedy_ids) == np.asarray(result.target_ids)
    left = sketches[correct].sum(axis=0)
    right = sketches[~correct].sum(axis=0)

    estimates = {
        "norm_correct": float(np.linalg.norm(left)),
        "norm_wrong": float(np.linalg.norm(right)),
        "norm_total": float(np.linalg.norm(left + right)),
        "dot": float(left @ right),
    }
    scale = max(estimates["norm_correct"], estimates["norm_wrong"], 1e-30)
    for key in ("norm_correct", "norm_wrong", "norm_total"):
        relative = abs(estimates[key] - exact[key]) / max(abs(exact[key]), 1e-30)
        assert relative < 0.15, (key, estimates[key], exact[key])
    # The dot product is the noisiest of the five; compare on the norm scale.
    assert abs(estimates["dot"] - exact["dot"]) / (scale ** 2) < 0.15

    if exact["cosine"] is not None:
        estimated_cosine = estimates["dot"] / (
            estimates["norm_correct"] * estimates["norm_wrong"]
        )
        assert abs(estimated_cosine - exact["cosine"]) < 0.15
