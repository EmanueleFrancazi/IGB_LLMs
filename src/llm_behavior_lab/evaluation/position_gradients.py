"""Exact per-position parameter-gradient norms at a fixed initialization.

For one evaluation position ``d`` whose true next token is ``y_d``:

.. code-block:: text

    z_d       = model logits at position d, restricted to the tokenizer vocabulary
    z_d^elig  = z_d with non-eligible (structural) IDs driven to -inf
    ell_d     = -log softmax(z_d^elig)[y_d]     single position, eligible support
    g_d       = || grad_theta ell_d ||_2        theta = every requires_grad parameter
    greedy_d  = argmax z_d^elig                 same masked logits, same forward

``g_d`` is the **exact** global L2 norm over every trainable parameter. It is not
a logit gradient, not a hidden-state gradient, not an LM-head-only gradient, and
not the gradient of a window-averaged loss. No cheaper proxy is substituted
anywhere in this module.

**Not to be confused with** :mod:`llm_behavior_lab.evaluation.gradient_norms`,
which is a different Phase 5 diagnostic: it measures squared norms of gradients
with respect to *decoder block output activations*, via forward hooks, taken from
the *window-averaged* loss. That module answers "are gradients stable through
depth"; this one answers "how hard does one position's true-token loss pull on
the parameters".

Two properties make the measurement affordable and safe.

*Affordable*: position ``d``'s logits depend only on tokens at or before ``d`` in
its own window, so one forward pass per window serves every position in that
window; the graph is retained and re-differentiated once per position. Nothing
shaped ``[num_positions, num_parameters]`` is ever materialized -- at ``D=32768``
over 8.6M parameters that would be about 1.1 TB in float32. Live at any instant are one
window's graph and one gradient set; only ``[D_g]``-sized vectors survive.

*Safe*: gradients are taken with :func:`torch.autograd.grad`, which returns them
instead of accumulating into ``parameter.grad``. There is no optimizer, no
``zero_grad``, no clipping, and no in-place weight operation, so the model's
parameters, buffers, and existing ``.grad`` state are left exactly as found. The
entry ``training``/``eval`` mode is captured and restored even if the measurement
raises.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F

from llm_behavior_lab.evaluation.guessing import (
    apply_support_mask,
    eligible_support_mask,
    greedy_guess_ids,
    restrict_to_support,
)
from llm_behavior_lab.evaluation.init_distribution import EvaluationPositions

__all__ = [
    "CANONICAL_GRADIENT_TEMPERATURE",
    "GRADIENT_DEFINITION",
    "GRADIENT_TEMPERATURES",
    "PositionGradientResult",
    "DEFAULT_SKETCH_DIMENSION",
    "compute_position_gradient_norms",
    "evenly_spaced_indices",
    "masked_evaluation_logits",
    "single_position_losses",
]

#: Persisted with every result so a stored number can never be read without the
#: definition that produced it.
GRADIENT_DEFINITION = (
    "l2_norm_of_gradient_of_single_position_next_token_cross_entropy"
    "_with_respect_to_all_trainable_parameters"
)

#: Temperatures at which the loss itself is defined. The six established sweep
#: values plus ``T = 1``, the canonical baseline.
#:
#: Temperature enters the **loss**, not a post-hoc rescaling of a result:
#: ``ell_T(d) = -log softmax(z_d / T)[y_d]``. The gradient therefore changes with
#: ``T`` through two routes at once -- the explicit ``1/T`` in
#: ``d ell_T / d z_i = (p_T(i) - 1[i = y]) / T`` and the sharpening of ``p_T``
#: itself. Both are intended. No compensating ``T^2`` factor is applied; this is
#: plain temperature-scaled cross entropy, not a distillation loss.
#:
#: Greedy identity is untouched: softmax is strictly increasing, so
#: ``argmax softmax(z/T) = argmax z`` for every positive ``T``. The guessing bias
#: is held fixed while the learning signal varies, which is the whole design.
GRADIENT_TEMPERATURES = (0.12, 0.24, 0.36, 0.48, 0.60, 1.00, 1.20)

#: The canonical baseline inside that grid.
CANONICAL_GRADIENT_TEMPERATURE = 1.00

#: Width of the per-position gradient sketch. 512 buckets keep the persisted
#: array to about 67 MB at D = 32768 in float32, and a count sketch preserves
#: inner products with a relative error of order ``1/sqrt(K)`` -- roughly 4% here
#: -- which is fine for comparing group means but not for trusting any single
#: pairwise cosine. Validated rather than assumed: see the sketch tests.
#:
#: Runtime cost is **not yet measured**. Each position adds one scatter-add over
#: every parameter on top of a backward pass that already dominates it, so the
#: overhead is expected to be small relative to the roughly 2027 s the gradient
#: analysis took at D = 32768 -- but that expectation stands until the matched
#: baseline and sketch smokes are timed on the cluster.
DEFAULT_SKETCH_DIMENSION = 512


@dataclass(frozen=True)
class PositionGradientResult:
    """Per-position gradient norms for one model initialization.

    Every vector has length ``D_g``, the number of evaluated positions, and the
    four are aligned entry by entry. ``position_indices`` is the flat index
    ``window * block_size + offset`` into the experiment's evaluation grid, so a
    subset of windows stays traceable back to the positions it came from.

    ``losses`` is carried for validation and for the console sanity check (at
    initialization the mean should sit near ``log`` of the eligible support). It
    is deliberately *not* persisted: the record schema stores the observables the
    analysis is defined on, and this one is recoverable from a rerun.
    """

    position_indices: torch.Tensor
    target_ids: torch.Tensor
    greedy_ids: torch.Tensor
    #: ``[N_T, D_g]`` exact norms, one row per temperature.
    temperature_gradient_norms: torch.Tensor
    #: ``[N_T, D_g]`` the losses those gradients were taken of.
    temperature_losses: torch.Tensor
    #: The temperature grid, in the order the rows are stored.
    temperatures: tuple[float, ...]
    #: ``[D_g]`` canonical ``T = 1`` slice, the established observable.
    gradient_norms: torch.Tensor
    losses: torch.Tensor
    parameter_count: int
    num_parameter_tensors: int
    num_windows: int
    block_size: int
    seconds: float
    definition: str = GRADIENT_DEFINITION
    softmax_support: str = "eligible"
    #: Scalars of the canonical-temperature correct-vs-wrong vector split, or
    #: ``None`` when the diagnostic was not requested.
    vector_split: dict[str, Any] | None = None
    #: ``[D_g, K]`` count sketch of each position's canonical gradient, or
    #: ``None`` when sketching was not requested.
    gradient_sketches: torch.Tensor | None = None
    #: Provenance of that projection, or ``None``.
    sketch_protocol: dict[str, Any] | None = None

    @property
    def canonical_index(self) -> int:
        """Row holding the canonical ``T = 1`` observable."""

        return self.temperatures.index(CANONICAL_GRADIENT_TEMPERATURE)

    @property
    def num_positions(self) -> int:
        """Number of evaluated positions, ``D_g``."""

        return int(self.gradient_norms.shape[0])

    def as_metadata(self, **extra: Any) -> dict[str, Any]:
        """Return a JSON-serializable description of what was measured."""

        payload: dict[str, Any] = {
            "definition": self.definition,
            "softmax_support": self.softmax_support,
            "includes_all_trainable_parameters": True,
            "parameter_count": self.parameter_count,
            "num_parameter_tensors": self.num_parameter_tensors,
            "num_positions": self.num_positions,
            "num_windows": self.num_windows,
            "block_size": self.block_size,
            "temperatures": list(self.temperatures),
            "canonical_temperature": CANONICAL_GRADIENT_TEMPERATURE,
            "temperature_enters_the_loss": True,
            "compensating_t_squared_factor": False,
            "mean_loss": float(self.losses.mean().item()),
            "mean_gradient_norm": float(self.gradient_norms.mean().item()),
            "seconds": round(self.seconds, 3),
        }
        if self.vector_split is not None:
            payload["vector_split"] = dict(self.vector_split)
        if self.sketch_protocol is not None:
            payload["gradient_sketch"] = dict(self.sketch_protocol)
        payload.update(extra)
        return payload


class _GradientSketcher:
    """Deterministic count sketch of a full parameter gradient.

    The whole point is that ``g_d`` cannot be kept. At 8.6M parameters and 32768
    positions the exact matrix is about 1.1 TB in float32, so directional structure has to
    be measured through a projection that is cheap to apply and preserves the
    only thing being asked about: inner products between gradients.

    A count sketch does exactly that. Every parameter coordinate is assigned,
    once, a bucket ``h(i) in [0, K)`` and a sign ``s(i) in {-1, +1}``, and the
    sketch is ``sketch[h(i)] += s(i) * g[i]``. The signs make the estimator
    unbiased: for any two gradients ``E<sketch(a), sketch(b)> = <a, b>``, with a
    variance that falls as ``1/K``. Cosines computed from sketches are therefore
    unbiased in the inner product and accurate enough for class-level means,
    which is what the clustering analysis consumes.

    Why not a dense Gaussian projection: it would need a ``[P, K]`` matrix, about
    17 GB at ``P = 8.6M, K = 512``. The sketch needs two ``[P]`` tables instead
    and applies in one scatter-add.

    Determinism is structural. Buckets and signs come from a
    :class:`torch.Generator` seeded once from ``seed``, drawn per parameter
    tensor in a fixed order, and never touched again -- so two runs of the same
    experiment produce bitwise the same projection. It draws from its own
    generator and never from the global RNG, so enabling the sketch cannot shift
    any sampling stream in the experiment around it.
    """

    def __init__(
        self,
        parameters: Sequence[torch.Tensor],
        *,
        dimension: int = DEFAULT_SKETCH_DIMENSION,
        seed: int = 20240917,
    ) -> None:
        if dimension < 1:
            raise ValueError(f"sketch dimension must be positive; got {dimension}.")
        self.dimension = int(dimension)
        self.seed = int(seed)
        self.buckets: list[torch.Tensor] = []
        self.signs: list[torch.Tensor] = []
        for index, parameter in enumerate(parameters):
            # One generator per tensor, seeded from the run seed and the tensor's
            # position, so the tables do not depend on how many tensors precede
            # it and stay stable if an unrelated buffer is ever added.
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.seed + 1000003 * index)
            count = parameter.numel()
            self.buckets.append(
                torch.randint(
                    0, self.dimension, (count,), generator=generator, dtype=torch.long
                ).to(parameter.device)
            )
            signs = torch.randint(
                0, 2, (count,), generator=generator, dtype=torch.int8
            ).to(parameter.device)
            self.signs.append(signs.double() * 2.0 - 1.0)

    def project(self, grads: Sequence[torch.Tensor]) -> torch.Tensor:
        """Sketch one gradient set into a ``[dimension]`` float64 vector."""

        sketch = torch.zeros(
            self.dimension, dtype=torch.float64, device=self.signs[0].device
        )
        for gradient, bucket, sign in zip(grads, self.buckets, self.signs):
            sketch.index_add_(0, bucket, gradient.detach().double().reshape(-1) * sign)
        return sketch


class _VectorSplitAccumulator:
    """Streaming ``g_correct`` and ``g_wrong`` at the canonical temperature.

    Norm mass answers "how much gradient magnitude does each group generate";
    it cannot answer "where does the first update actually point", because
    ``sum_d ||g_d||`` is not ``||sum_d g_d||`` and the difference is exactly
    however much the individual gradients cancel. That needs the vector sums,
    which is what this accumulates.

    Two parameter-shaped buffers are held, nothing per position: each ``g_d``
    is added into one of them and released with the gradient set that produced
    it. Peak cost is therefore two copies of the parameters -- about 131 MiB at
    8.6M parameters -- and is independent of ``D``.

    float64 throughout, and deliberately with no float32 fallback. The quantity
    is a sum over tens of thousands of terms whose cosine is the thing being
    measured; cancellation is the signal, so accumulating it in the precision
    that loses cancellation would quietly destroy the answer. If the memory ever
    becomes a real problem it should be reported and decided on, not silently
    traded away.

    Only the canonical ``T = 1`` grid row is accumulated: that is the objective
    training will actually use, and one buffer pair per temperature would cost
    seven times the memory for a question nobody asked.
    """

    def __init__(self, parameters: Sequence[torch.Tensor]) -> None:
        self.correct = [
            torch.zeros_like(parameter, dtype=torch.float64) for parameter in parameters
        ]
        self.wrong = [
            torch.zeros_like(parameter, dtype=torch.float64) for parameter in parameters
        ]
        self.num_correct = 0
        self.num_wrong = 0

    def add(self, grads: Sequence[torch.Tensor], *, is_correct: bool) -> None:
        """Accumulate one position's gradient into the matching group."""

        target = self.correct if is_correct else self.wrong
        for buffer, gradient in zip(target, grads):
            buffer.add_(gradient.detach().double())
        if is_correct:
            self.num_correct += 1
        else:
            self.num_wrong += 1

    def summary(self) -> dict[str, Any]:
        """Reduce the two buffers to the reported scalars."""

        squared_correct = torch.zeros((), dtype=torch.float64)
        squared_wrong = torch.zeros((), dtype=torch.float64)
        squared_total = torch.zeros((), dtype=torch.float64)
        dot = torch.zeros((), dtype=torch.float64)
        for correct, wrong in zip(self.correct, self.wrong):
            squared_correct += correct.pow(2).sum().cpu()
            squared_wrong += wrong.pow(2).sum().cpu()
            squared_total += (correct + wrong).pow(2).sum().cpu()
            dot += (correct * wrong).sum().cpu()

        norm_correct = float(squared_correct.sqrt())
        norm_wrong = float(squared_wrong.sqrt())
        product = norm_correct * norm_wrong
        return {
            "temperature": CANONICAL_GRADIENT_TEMPERATURE,
            "norm_correct": norm_correct,
            "norm_wrong": norm_wrong,
            "norm_total": float(squared_total.sqrt()),
            "dot": float(dot),
            # Undefined when either group is empty or its sum vanishes; a cosine
            # of 0.0 would claim orthogonality that was never measured.
            "cosine": float(dot) / product if product > 0.0 else None,
            "num_correct": self.num_correct,
            "num_wrong": self.num_wrong,
        }


def evenly_spaced_indices(total: int, count: int | None = None) -> tuple[int, ...]:
    """Choose ``count`` window indices spread evenly over ``range(total)``.

    Deterministic and seed-free, matching how the evaluation windows themselves
    are chosen: the same subset must be reproducible for every rerun. ``None`` --
    the default everywhere -- means every window, so no reduction of the
    scientific position set happens unless a caller explicitly asks for one.

    Args:
        total: Number of available windows.
        count: Number of windows to keep, or ``None`` for all of them.

    Returns:
        Sorted, distinct window indices.

    Raises:
        ValueError: If ``total`` is not positive, or ``count`` is outside
            ``[1, total]``.
    """

    if total <= 0:
        raise ValueError("total must be positive.")
    if count is None or count == total:
        return tuple(range(total))
    if count <= 0:
        raise ValueError("count must be positive.")
    if count > total:
        raise ValueError(
            f"Requested {count} windows but only {total} are available."
        )
    if count == 1:
        return (0,)
    step = (total - 1) / (count - 1)
    return tuple(int(round(index * step)) for index in range(count))


def _trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    """Return every parameter the gradient norm is taken over."""

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError(
            "The model exposes no trainable parameters, so a parameter-gradient "
            "norm is undefined."
        )
    return parameters


def _require_every_parameter_reached(
    grads: Sequence[torch.Tensor | None],
    parameters: Sequence[torch.nn.Parameter],
) -> None:
    """Fail if a parameter received no gradient at all.

    ``allow_unused=True`` is passed so autograd reports an unreachable parameter
    as ``None`` instead of raising an opaque error. For a decoder-only model
    every parameter lies on the path from any position's logits, so a ``None``
    means the graph is not what this module assumes -- silently treating it as a
    zero contribution would understate the norm without saying so.
    """

    missing = [index for index, gradient in enumerate(grads) if gradient is None]
    if missing:
        raise RuntimeError(
            f"{len(missing)} of {len(parameters)} trainable parameters received no "
            "gradient from the single-position loss, so the reported norm would "
            "not cover every parameter. The model's graph is not what this "
            "measurement assumes."
        )


def masked_evaluation_logits(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    vocab_size: int,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Run one forward pass and return support-masked logits, graph intact.

    The restriction and masking sequence is the one
    :func:`~llm_behavior_lab.evaluation.init_distribution.measure_initialization`
    applies, in the same order, so the gradient observable and the guessing
    observables read literally the same masked logits rather than two
    independently-derived versions of them.

    The clone matters twice over: the mask is applied in place, and the model may
    hand back a view of its own buffer. Cloning is differentiable, so the graph
    is unaffected.

    Args:
        model: Model whose forward accepts ``input_ids`` and returns ``.logits``.
        input_ids: Token windows shaped ``[batch, block]``.
        vocab_size: Tokenizer vocabulary; logits wider than this are truncated.
        mask: Boolean ``[vocab_size]`` predictive-support mask.

    Returns:
        Masked logits shaped ``[batch, block, vocab_size]``.
    """

    output = model(input_ids=input_ids)
    logits = restrict_to_support(output.logits, vocab_size=vocab_size)
    return apply_support_mask(logits.clone(), mask)


def single_position_losses(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    vocab_size: int,
    eligible_token_ids: Sequence[int] | None = None,
) -> torch.Tensor:
    """Return ``ell_d`` for every position of every supplied window.

    This is the quantity the gradient is taken of: **one position's** next-token
    cross-entropy, not the window mean the model's own ``forward`` returns when
    given ``targets``. The two are related by ``mean(single_position_losses(...))
    == model(input_ids, targets=...).loss`` over an unrestricted support, which
    is exactly what the accompanying test asserts.

    Args:
        model: Model whose forward accepts ``input_ids`` and returns ``.logits``.
        input_ids: Token windows shaped ``[batch, block]``.
        target_ids: Next-token targets with the same shape.
        vocab_size: Tokenizer vocabulary.
        eligible_token_ids: Predictive support. ``None`` means every token.

    Returns:
        Losses shaped like ``target_ids``.
    """

    if input_ids.shape != target_ids.shape:
        raise ValueError(
            f"input_ids {tuple(input_ids.shape)} and target_ids "
            f"{tuple(target_ids.shape)} must have the same shape."
        )

    mask = eligible_support_mask(vocab_size, eligible_token_ids, device=input_ids.device)
    _check_targets_eligible(target_ids, mask)
    logits = masked_evaluation_logits(model, input_ids, vocab_size=vocab_size, mask=mask)
    losses = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        target_ids.reshape(-1),
        reduction="none",
    )
    return losses.reshape(target_ids.shape)


def _check_targets_eligible(target_ids: torch.Tensor, mask: torch.Tensor) -> None:
    """Fail loudly if a target token is outside the predictive support.

    An ineligible target has exactly zero probability under the masked softmax,
    so ``ell_d`` would be ``+inf`` and ``g_d`` meaningless. The experiment runner
    already refuses a corpus containing structural tokens; this is the backstop
    for any other caller.
    """

    flat = target_ids.reshape(-1)
    if bool((flat < 0).any()) or bool((flat >= mask.shape[0]).any()):
        raise ValueError("target_ids contain values outside the vocabulary.")
    ineligible = flat[~mask.to(flat.device)[flat]]
    if ineligible.numel():
        offenders = sorted(set(int(value) for value in ineligible.tolist()))[:5]
        raise ValueError(
            f"Target token IDs {offenders} are outside the predictive support, so "
            "-log P(y_d | context_d) is infinite. The corpus must not contain "
            "structural tokens."
        )


def compute_position_gradient_norms(
    model: torch.nn.Module,
    positions: EvaluationPositions,
    *,
    vocab_size: int,
    eligible_token_ids: Sequence[int] | None = None,
    window_indices: Sequence[int] | None = None,
    num_windows: int | None = None,
    temperatures: Sequence[float] = GRADIENT_TEMPERATURES,
    vector_split: bool = False,
    gradient_sketch: bool = False,
    sketch_dimension: int = DEFAULT_SKETCH_DIMENSION,
    sketch_seed: int = 20240917,
    progress: Callable[[int, int], None] | None = None,
) -> PositionGradientResult:
    """Measure ``g_d`` exactly, one evaluation position at a time.

    One forward pass per window feeds ``block_size`` backward passes taken from
    the retained graph. Peak memory is therefore one window's activations plus
    one full gradient set, independent of how many positions are evaluated.

    The model is never modified. Gradients come from :func:`torch.autograd.grad`,
    so ``parameter.grad`` is not touched, and the entry ``training`` mode is
    restored on every exit path.

    Args:
        model: Model whose forward accepts ``input_ids`` and returns ``.logits``.
        positions: The experiment's shared evaluation positions.
        vocab_size: Tokenizer vocabulary.
        eligible_token_ids: Predictive support. ``None`` means every token.
        window_indices: Explicit window subset. Defaults to every window.
        num_windows: Convenience alternative to ``window_indices``: keep this
            many evenly spaced windows. Ignored when ``window_indices`` is given.
        vector_split: Also accumulate ``g_correct`` and ``g_wrong``, the summed
            parameter gradients of correctly and incorrectly assigned positions
            at the canonical temperature. Off by default, so every existing
            caller keeps its exact cost and behaviour. Adds no backward pass --
            it reuses each gradient set already computed for the norm -- but
            holds two float64 parameter-shaped buffers, about 131 MiB at 8.6M
            parameters.
        gradient_sketch: Also record a deterministic count sketch of every
            position's canonical gradient, so directional structure can be
            measured afterwards. Off by default. Adds no backward pass.
        sketch_dimension: Width ``K`` of that sketch.
        sketch_seed: Seed of the projection. Fixed across runs so two
            experiments produce comparable sketches; drawn from its own
            generator, never the global RNG.
        progress: Optional ``callback(windows_done, windows_total)``.

    Returns:
        A :class:`PositionGradientResult` with one entry per evaluated position.
    """

    if window_indices is None:
        window_indices = evenly_spaced_indices(positions.num_windows, num_windows)
    else:
        window_indices = tuple(int(index) for index in window_indices)
        if not window_indices:
            raise ValueError("window_indices must not be empty.")
        if len(set(window_indices)) != len(window_indices):
            raise ValueError("window_indices must be distinct.")
        if min(window_indices) < 0 or max(window_indices) >= positions.num_windows:
            raise ValueError(
                f"window_indices must lie in [0, {positions.num_windows})."
            )

    grid = tuple(dict.fromkeys(float(value) for value in temperatures))
    if not grid:
        raise ValueError("temperatures must not be empty.")
    if any(value <= 0.0 for value in grid):
        raise ValueError(
            "Every gradient temperature must be positive; softmax(z/T) is "
            "undefined at T = 0."
        )
    if CANONICAL_GRADIENT_TEMPERATURE not in grid:
        raise ValueError(
            f"The canonical baseline T = {CANONICAL_GRADIENT_TEMPERATURE} must be "
            "present in the grid; it is the established observable every other "
            "temperature is read against."
        )

    device = positions.input_ids.device
    mask = eligible_support_mask(vocab_size, eligible_token_ids, device=device)
    _check_targets_eligible(positions.target_ids[list(window_indices)], mask)

    parameters = _trainable_parameters(model)
    block_size = positions.block_size
    total_positions = len(window_indices) * block_size

    position_indices = torch.empty(total_positions, dtype=torch.long)
    target_ids = torch.empty(total_positions, dtype=torch.long)
    greedy_ids = torch.empty(total_positions, dtype=torch.long)
    # [N_T, D_g] scalars only. A [D_g, N_T, parameters] tensor is never formed:
    # at 32768 positions, 7 temperatures and 8.6M parameters that would be about
    # 7.9 TB in float32. One gradient set exists at a time and is reduced to a scalar.
    gradient_norms = torch.empty((len(grid), total_positions), dtype=torch.float64)
    losses = torch.empty((len(grid), total_positions), dtype=torch.float64)
    split = _VectorSplitAccumulator(parameters) if vector_split else None
    sketcher = (
        _GradientSketcher(parameters, dimension=sketch_dimension, seed=sketch_seed)
        if gradient_sketch
        else None
    )
    sketches = (
        torch.empty((total_positions, sketch_dimension), dtype=torch.float64)
        if gradient_sketch
        else None
    )
    canonical_temperature_index = grid.index(CANONICAL_GRADIENT_TEMPERATURE)

    was_training = model.training
    started = time.perf_counter()
    try:
        model.eval()
        cursor = 0
        for done, window in enumerate(window_indices, start=1):
            window_inputs = positions.input_ids[window : window + 1]
            window_targets = positions.target_ids[window : window + 1]
            logits = masked_evaluation_logits(
                model, window_inputs, vocab_size=vocab_size, mask=mask
            )[0]
            # Read from the same masked logits as the loss, in the same forward
            # pass, so the guess fractions are position-aligned with the norms
            # by construction rather than by a second, separately-batched pass.
            window_greedy = greedy_guess_ids(logits.detach())

            last_offset = block_size - 1
            last_temperature = len(grid) - 1
            for offset in range(block_size):
                position_indices[cursor] = window * block_size + offset
                target_ids[cursor] = window_targets[0, offset]
                greedy_ids[cursor] = window_greedy[offset]

                for index, temperature in enumerate(grid):
                    # Temperature scales the logits *inside* the loss, so the
                    # gradient really is the gradient of a different objective --
                    # not a rescaled version of one. Dividing by exactly 1.0 is
                    # the identity in IEEE-754, so the canonical row is bitwise
                    # the established T = 1 observable rather than a near copy.
                    loss = F.cross_entropy(
                        logits[offset : offset + 1] / temperature,
                        window_targets[0, offset : offset + 1],
                    )
                    # Not .backward(): autograd.grad returns the gradients and
                    # leaves every parameter.grad exactly as the caller left it.
                    # The one forward graph for this window is retained until the
                    # final temperature of the final position uses it.
                    grads = torch.autograd.grad(
                        loss,
                        parameters,
                        retain_graph=not (
                            offset == last_offset and index == last_temperature
                        ),
                        allow_unused=True,
                    )
                    _require_every_parameter_reached(grads, parameters)
                    # Accumulated in float64 on whichever device the gradients
                    # live on; an 8.6M-parameter sum of squares loses meaningful
                    # precision in float32.
                    squared = None
                    for gradient in grads:
                        contribution = gradient.detach().double().pow(2).sum()
                        squared = contribution if squared is None else squared + contribution

                    gradient_norms[index, cursor] = squared.sqrt().cpu()
                    losses[index, cursor] = loss.detach().double().cpu()
                    if sketcher is not None and index == canonical_temperature_index:
                        # Same gradient set the norm came from; no second
                        # backward pass and nothing full-sized is retained.
                        sketches[cursor] = sketcher.project(grads).cpu()
                    if split is not None and index == canonical_temperature_index:
                        # The same gradient set the norm was taken from, before
                        # it is released. No second backward pass.
                        split.add(
                            grads,
                            is_correct=bool(
                                int(greedy_ids[cursor]) == int(target_ids[cursor])
                            ),
                        )
                    del grads, squared, loss
                cursor += 1

            del logits, window_greedy
            if progress is not None:
                progress(done, len(window_indices))
    finally:
        model.train(was_training)
    seconds = time.perf_counter() - started

    canonical = grid.index(CANONICAL_GRADIENT_TEMPERATURE)
    return PositionGradientResult(
        position_indices=position_indices,
        target_ids=target_ids,
        greedy_ids=greedy_ids,
        temperature_gradient_norms=gradient_norms,
        temperature_losses=losses,
        temperatures=grid,
        gradient_norms=gradient_norms[canonical],
        losses=losses[canonical],
        vector_split=None if split is None else split.summary(),
        gradient_sketches=sketches,
        sketch_protocol=(
            None
            if sketcher is None
            else {
                "dimension": sketcher.dimension,
                "seed": sketcher.seed,
                "temperature": CANONICAL_GRADIENT_TEMPERATURE,
                "projection": "count_sketch_signed_feature_hashing",
                "preserves": "inner_products_in_expectation",
                "parameter_count": sum(p.numel() for p in parameters),
            }
        ),
        parameter_count=sum(parameter.numel() for parameter in parameters),
        num_parameter_tensors=len(parameters),
        num_windows=len(window_indices),
        block_size=block_size,
        seconds=seconds,
    )
