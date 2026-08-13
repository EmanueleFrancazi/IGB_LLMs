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
over 8.6M parameters that would be about 1.1 PB. Live at any instant are one
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
    "GRADIENT_DEFINITION",
    "PositionGradientResult",
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
    gradient_norms: torch.Tensor
    losses: torch.Tensor
    parameter_count: int
    num_parameter_tensors: int
    num_windows: int
    block_size: int
    seconds: float
    definition: str = GRADIENT_DEFINITION
    softmax_support: str = "eligible"

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
            "mean_loss": float(self.losses.mean().item()),
            "mean_gradient_norm": float(self.gradient_norms.mean().item()),
            "seconds": round(self.seconds, 3),
        }
        payload.update(extra)
        return payload


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

    device = positions.input_ids.device
    mask = eligible_support_mask(vocab_size, eligible_token_ids, device=device)
    _check_targets_eligible(positions.target_ids[list(window_indices)], mask)

    parameters = _trainable_parameters(model)
    block_size = positions.block_size
    total_positions = len(window_indices) * block_size

    position_indices = torch.empty(total_positions, dtype=torch.long)
    target_ids = torch.empty(total_positions, dtype=torch.long)
    greedy_ids = torch.empty(total_positions, dtype=torch.long)
    gradient_norms = torch.empty(total_positions, dtype=torch.float64)
    losses = torch.empty(total_positions, dtype=torch.float64)

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

            for offset in range(block_size):
                loss = F.cross_entropy(
                    logits[offset : offset + 1],
                    window_targets[0, offset : offset + 1],
                )
                # Not .backward(): autograd.grad returns the gradients and leaves
                # every parameter.grad exactly as the caller left it.
                grads = torch.autograd.grad(
                    loss,
                    parameters,
                    retain_graph=offset + 1 < block_size,
                    allow_unused=True,
                )
                _require_every_parameter_reached(grads, parameters)
                # Accumulated in float64 on whichever device the gradients live
                # on; a 8.6M-parameter sum of squares loses meaningful precision
                # in float32.
                squared = None
                for gradient in grads:
                    contribution = gradient.detach().double().pow(2).sum()
                    squared = contribution if squared is None else squared + contribution

                position_indices[cursor] = window * block_size + offset
                target_ids[cursor] = window_targets[0, offset]
                greedy_ids[cursor] = window_greedy[offset]
                gradient_norms[cursor] = squared.sqrt().cpu()
                losses[cursor] = loss.detach().double().cpu()
                cursor += 1
                del grads, squared, loss

            del logits, window_greedy
            if progress is not None:
                progress(done, len(window_indices))
    finally:
        model.train(was_training)
    seconds = time.perf_counter() - started

    return PositionGradientResult(
        position_indices=position_indices,
        target_ids=target_ids,
        greedy_ids=greedy_ids,
        gradient_norms=gradient_norms,
        losses=losses,
        parameter_count=sum(parameter.numel() for parameter in parameters),
        num_parameter_tensors=len(parameters),
        num_windows=len(window_indices),
        block_size=block_size,
        seconds=seconds,
    )
