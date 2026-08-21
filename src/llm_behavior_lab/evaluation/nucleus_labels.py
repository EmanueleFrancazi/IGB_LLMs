"""Per-position nucleus samples, reproduced from a completed experiment.

The experiment streams its logits and keeps only ``[vocab]``-sized counters, so
the record holds *how many* positions sampled each token at each sweep
temperature but not *which* position sampled what. Grouping gradients by the
nucleus sample therefore needs those labels recovered.

They are recovered rather than re-drawn. Two properties of the original design
make that possible without touching the record:

* the draw is inverted from **pre-drawn uniforms indexed by position**, so a
  position's sample depends on the position alone and not on how positions were
  batched (:func:`sampling_uniforms`); and
* every sweep temperature reuses replicate 0's uniforms and one shared sort
  (:func:`nucleus_guess_ids_by_temperature`), so temperature is the only thing
  that varies.

Reproduction still requires a forward pass, because logits are not persisted --
but only a forward pass. No gradient is recomputed and nothing is written back
into the record.

Whether the reproduction is faithful is not assumed. :func:`histogram_gate`
compares the recovered labels' histogram with the one the experiment actually
recorded and demands **exact integer equality**. A near match is a failure: if
the reconstruction had drifted -- a different seed, a different support mask, a
different initialization -- the histogram would move, and any clustering built
on top of the labels would describe a model that was never measured.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from llm_behavior_lab.analysis.nucleus_clustering import histogram_gate
from llm_behavior_lab.evaluation.guessing import (
    apply_support_mask,
    eligible_support_mask,
    nucleus_guess_ids_by_temperature,
    restrict_to_support,
)
from llm_behavior_lab.evaluation.init_distribution import (
    NucleusSamplingSettings,
    sampling_uniforms,
)

#: Re-exported so a caller recovering labels can gate them without reaching
#: into the analysis layer. The gate itself is pure NumPy and lives there,
#: so it can be tested where PyTorch is not installed.
__all__ = ["histogram_gate", "nucleus_position_labels"]


def nucleus_position_labels(
    model: Any,
    positions: Any,
    *,
    model_seed: int,
    vocab_size: int,
    sampling: NucleusSamplingSettings,
    eligible_token_ids: Sequence[int] | np.ndarray,
    temperatures: Sequence[float],
    forward_batch_size: int,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Recover the sweep's per-position samples for one initialization.

    The loop mirrors the measurement loop exactly -- same restriction, same
    in-place support mask on a cloned tensor, same batch order, same uniform
    slice per batch -- because anything that differs would change the logits and
    so the samples. What differs is only what is kept: labels per position
    instead of counts per token.

    Args:
        model: The initialized model, already on ``device``.
        positions: Evaluation positions, as the experiment built them.
        model_seed: Initialization seed, mixed into the sampling stream exactly
            as the experiment mixed it.
        vocab_size: Tokenizer vocabulary size, for the support restriction.
        sampling: The run's nucleus policy.
        eligible_token_ids: The sampled support.
        temperatures: The sweep temperatures to recover, in report order.
        forward_batch_size: Windows per forward pass.
        device: Device to run on.

    Returns:
        ``labels`` shaped ``[num_temperatures, num_positions]`` (int64) and the
        ``temperatures`` they correspond to.
    """

    sampling.validate()
    if forward_batch_size <= 0:
        raise ValueError("forward_batch_size must be positive.")
    ordered = tuple(dict.fromkeys(float(value) for value in temperatures))
    if not ordered:
        raise ValueError("temperatures must not be empty.")

    device = torch.device(device)
    mask = eligible_support_mask(vocab_size, eligible_token_ids, device=device)
    # Replicate 0 -- the stream the sweep itself consumed. Any other replicate
    # would be a different, equally valid experiment, and not this one.
    uniforms = sampling_uniforms(
        sampling,
        model_seed=model_seed,
        num_positions=positions.num_positions,
        device=device,
    )[0]

    labels = np.empty((len(ordered), positions.num_positions), dtype=np.int64)
    model.eval()
    consumed = 0
    with torch.no_grad():
        for start in range(0, positions.num_windows, forward_batch_size):
            output = model(
                input_ids=positions.input_ids[start : start + forward_batch_size]
            )
            logits = restrict_to_support(output.logits, vocab_size=vocab_size)
            logits = apply_support_mask(logits.clone(), mask)
            batch_positions = int(logits[..., 0].numel())
            sampled = nucleus_guess_ids_by_temperature(
                logits,
                temperatures=ordered,
                top_p=sampling.top_p,
                uniforms=uniforms[consumed : consumed + batch_positions],
            )
            for index, temperature in enumerate(ordered):
                labels[index, consumed : consumed + batch_positions] = (
                    sampled[temperature].reshape(-1).cpu().numpy()
                )
            consumed += batch_positions

    if consumed != positions.num_positions:
        raise ValueError(
            f"Recovered {consumed} positions but the experiment covered "
            f"{positions.num_positions}."
        )
    return {"labels": labels, "temperatures": ordered}
