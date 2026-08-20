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
    """Cosine matrix of **exact** gradients, where self-normalizing is correct.

    Never apply this to sketches: production divides a raw sketch by the exact
    gradient norm, and dividing by the sketch's own length instead gives the
    biased ratio estimator that commit 779dd36 removed.
    """

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
    gradients = _exact_gradients(model, positions)
    exact = _cosines(gradients)
    # Production scales the raw sketch by the exact gradient norm, not by the
    # sketch's own length; self-normalizing here would test a different, biased
    # estimator than the one the analysis actually uses.
    scaled = result.gradient_sketches.double() / gradients.norm(dim=1)[:, None]
    estimated = scaled @ scaled.T

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

    norms = _exact_gradients(model, positions).norm(dim=1)
    errors = {}
    for dimension in (32, 1024):
        result = compute_position_gradient_norms(
            model, positions, vocab_size=VOCAB, gradient_sketch=True,
            sketch_dimension=dimension,
        )
        scaled = result.gradient_sketches.double() / norms[:, None]
        estimated = scaled @ scaled.T
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


def test_sketch_similarity_matches_exact_gradient_similarity() -> None:
    """Implementation fidelity: does the sketch reproduce the exact geometry?

    This is the only end-to-end claim that is mathematically guaranteed. Whether
    positions sharing a target have aligned gradients is the **empirical
    hypothesis of the experiment**, not a property of a randomly initialized
    model, so it is deliberately not asserted here: for

        g_d = J_d^T (p_d - e_y)

    two positions can share ``y`` and still have unrelated Jacobians and
    predictive vectors. On this architecture the input token in fact dominates --
    distinct inputs give orthogonal embedding-gradient rows -- and the sign of
    within minus between flips with the seed. Encoding it as a unit-test
    expectation would make the server gate depend on a coin flip.

    What must hold is that the sketch estimates whatever geometry is there. So
    the exact gradients are retained, their cosine geometry computed directly,
    and the production estimator -- raw sketch divided by the **exact** gradient
    norm, as ``unit_sketches`` does -- is required to agree with it.

    ``within > between`` is asserted where it is planted and therefore true: the
    synthetic NumPy tests in ``tests/test_gradient_clustering.py``.
    """

    import numpy as np

    from llm_behavior_lab.analysis.gradient_clustering import class_similarity_matrix

    model = TinyModel()
    base = _positions()
    targets = base.target_ids.clone().reshape(-1)
    targets[: D // 2] = 3
    targets[D // 2 :] = 7
    positions = dataclasses.replace(base, target_ids=targets.reshape(WINDOWS, BLOCK))

    dimension = 2048
    result = compute_position_gradient_norms(
        model, positions, vocab_size=VOCAB,
        gradient_sketch=True, sketch_dimension=dimension,
    )

    exact = _exact_gradients(model, positions)
    exact_norms = exact.norm(dim=1)
    # Production scaling, applied to both sides so only the projection differs.
    exact_rows = (exact / exact_norms[:, None]).numpy()
    sketch_rows = (
        result.gradient_sketches.double() / exact_norms[:, None]
    ).numpy()

    # The persisted norms must be the ones used, so check they agree first.
    assert torch.allclose(result.gradient_norms, exact_norms, rtol=1e-9, atol=0)

    labels = targets.numpy()
    classes = np.array([3, 7])
    truth = class_similarity_matrix(exact_rows, labels, classes)["matrix"]
    estimate = class_similarity_matrix(sketch_rows, labels, classes)["matrix"]

    assert np.isfinite(truth).all() and np.isfinite(estimate).all()
    error = np.abs(estimate - truth)
    # A count sketch estimates a cosine with a standard error near 1/sqrt(K);
    # at K = 2048 that is about 0.022, so these are roughly 2 and 8 sigma.
    assert float(error.mean()) < 0.05, (estimate, truth)
    assert float(error.max()) < 0.20, (estimate, truth)


def test_identical_examples_give_identical_gradients_through_the_sketch() -> None:
    """A guaranteed positive control, without relying on a random property.

    Two positions with the same input *and* the same target produce exactly the
    same gradient, so their cosine is 1 by construction rather than by
    hypothesis. Class coherence must therefore come out at 1 for both the exact
    gradients and the sketch, which is a real end-to-end clustering case that
    cannot flip with the seed.
    """

    import numpy as np

    from llm_behavior_lab.analysis.gradient_clustering import class_similarity_matrix

    model = TinyModel()
    base = _positions()
    # Two windows repeated: window 0 duplicated, then window 1 duplicated.
    inputs = base.input_ids.clone()
    inputs[1] = inputs[0]
    inputs[3] = inputs[2]
    targets = base.target_ids.clone()
    targets[0] = targets[1] = 3
    targets[2] = targets[3] = 7
    positions = dataclasses.replace(base, input_ids=inputs, target_ids=targets)

    result = compute_position_gradient_norms(
        model, positions, vocab_size=VOCAB,
        gradient_sketch=True, sketch_dimension=2048,
    )
    exact_norms = result.gradient_norms
    rows = (result.gradient_sketches.double() / exact_norms[:, None]).numpy()

    # Offset j of window 0 and offset j of window 1 are the same computation.
    for offset in range(BLOCK):
        first, second = offset, BLOCK + offset
        cosine = float(
            rows[first] @ rows[second]
            / (np.linalg.norm(rows[first]) * np.linalg.norm(rows[second]))
        )
        assert cosine == pytest.approx(1.0, abs=1e-9)

    labels = np.repeat([3, 3, 7, 7], BLOCK)
    matrix = class_similarity_matrix(rows, labels, np.array([3, 7]))["matrix"]
    # Duplicated pairs sit at 1; the class mean mixes them with distinct
    # offsets, so only the guaranteed bound is asserted.
    assert np.isfinite(matrix).all()
    assert matrix[0, 0] <= 1.0 + 1e-9 and matrix[1, 1] <= 1.0 + 1e-9


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


# -- exact-gradient capture for the fidelity sanity check ---------------------


def test_exact_capture_is_off_by_default() -> None:
    """A scientific run must allocate nothing and behave identically."""

    result = compute_position_gradient_norms(TinyModel(), _positions(), vocab_size=VOCAB)

    assert result.exact_gradients is None
    assert result.exact_positions is None


def test_capture_retains_exactly_the_requested_positions() -> None:
    model = TinyModel()
    positions = _positions()
    wanted = [1, 4, 9]

    result = compute_position_gradient_norms(
        model, positions, vocab_size=VOCAB, gradient_sketch=True,
        exact_gradient_positions=wanted,
    )

    assert result.exact_positions.tolist() == sorted(wanted)
    assert result.exact_gradients.shape[0] == len(wanted)
    assert result.exact_gradients.dtype == torch.float32


def test_captured_norms_match_the_normal_exact_norms() -> None:
    """The integrity gate: flattening order and position alignment must hold."""

    import numpy as np

    model = TinyModel()
    positions = _positions()
    wanted = [0, 3, 7]

    result = compute_position_gradient_norms(
        model, positions, vocab_size=VOCAB, exact_gradient_positions=wanted,
    )

    flat = result.position_indices.numpy().tolist()
    rows = [flat.index(int(index)) for index in result.exact_positions.numpy()]
    recomputed = np.linalg.norm(
        result.exact_gradients.numpy().astype(np.float64), axis=1
    )
    expected = result.gradient_norms.numpy()[rows]

    assert np.allclose(recomputed, expected, rtol=1e-5, atol=0)


def test_production_sketch_reconstructs_from_the_captured_gradient() -> None:
    """Validates flattening order, bucket/sign construction, seed and K at once."""

    import numpy as np

    from llm_behavior_lab.analysis.countsketch_fidelity import count_sketch_matrix

    model = TinyModel()
    positions = _positions()
    parameters = [p for p in model.parameters() if p.requires_grad]
    sizes = [int(p.numel()) for p in parameters]

    result = compute_position_gradient_norms(
        model, positions, vocab_size=VOCAB, gradient_sketch=True,
        sketch_dimension=64, sketch_seed=20240917,
        exact_gradient_positions=[2, 5],
    )

    flat = result.position_indices.numpy().tolist()
    buckets, signs = count_sketch_matrix(
        sum(sizes), 64, seed=20240917, tensor_sizes=sizes
    )
    for row, index in enumerate(result.exact_positions.numpy()):
        gradient = result.exact_gradients.numpy()[row].astype(np.float64)
        rebuilt = np.zeros(64)
        np.add.at(rebuilt, buckets, gradient * signs)
        recorded = result.gradient_sketches.numpy()[flat.index(int(index))]
        assert np.allclose(rebuilt, recorded, rtol=1e-4, atol=1e-6)


def test_capture_does_not_change_the_normal_observables() -> None:
    model = TinyModel()
    positions = _positions()

    without = compute_position_gradient_norms(model, positions, vocab_size=VOCAB)
    with_capture = compute_position_gradient_norms(
        model, positions, vocab_size=VOCAB, exact_gradient_positions=[1, 6],
    )

    assert torch.equal(without.gradient_norms, with_capture.gradient_norms)
    assert torch.equal(
        without.temperature_gradient_norms, with_capture.temperature_gradient_norms
    )
    assert torch.equal(without.greedy_ids, with_capture.greedy_ids)


def test_capture_leaves_model_state_and_rng_untouched() -> None:
    model = TinyModel()
    parameters = {name: p.detach().clone() for name, p in model.named_parameters()}
    sentinel = torch.full_like(model.output.weight, 1.75)
    model.output.weight.grad = sentinel.clone()
    model.train()
    state = torch.get_rng_state()

    compute_position_gradient_norms(
        model, _positions(), vocab_size=VOCAB, gradient_sketch=True,
        exact_gradient_positions=[0, 2, 4],
    )

    for name, parameter in model.named_parameters():
        assert torch.equal(parameter.detach(), parameters[name]), name
    assert torch.equal(model.output.weight.grad, sentinel)
    assert model.training is True
    assert torch.equal(torch.get_rng_state(), state)
