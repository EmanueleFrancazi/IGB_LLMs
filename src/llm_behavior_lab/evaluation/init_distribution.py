"""Measure token-guess distributions at random model initialization.

The experiment this module supports asks how a randomly initialized model's
*selected token guesses* compare with the corpus token frequencies, and how much
that comparison moves when only the initialization changes.

Three sources of randomness are kept apart:

* **corpus/position sampling** -- removed entirely. Evaluation positions are
  chosen deterministically by :func:`deterministic_window_starts` and reused
  unchanged for every initialization, so no observed difference between seeds
  can come from looking at different text.
* **model initialization** -- the quantity under study. The caller seeds and
  builds one model per initialization.
* **token sampling** -- only affects the stochastic policy, is driven by its own
  generator, and is averaged over replicates *within* one initialization before
  initializations are compared.

Both policies consume the same logits, so greedy and nucleus results describe
one model state rather than two independent draws.

Memory is flat in the number of evaluation positions. The model is run batch by
batch and each batch's ``[batch, block, vocab]`` logits are folded into
``[vocab]``-sized counters and then discarded; no accumulator is ever indexed by
position. That matters at a realistic vocabulary, where holding every position's
logits would cost gigabytes -- 32768 positions over 32000 tokens is about
3.9 GiB. Sampling randomness is drawn up front and indexed by position, so a
streamed measurement equals an all-at-once one exactly rather than approximately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from llm_behavior_lab.evaluation.guessing import (
    apply_support_mask,
    eligible_support_mask,
    greedy_guess_ids,
    guess_counts,
    nucleus_guess_ids,
    restrict_to_support,
)

__all__ = [
    "EvaluationPositions",
    "InitializationMeasurement",
    "NucleusSamplingSettings",
    "build_evaluation_positions",
    "compute_evaluation_logits",
    "deterministic_window_starts",
    "measure_from_logits",
    "measure_initialization",
    "sampling_uniforms",
]


@dataclass(frozen=True)
class NucleusSamplingSettings:
    """Parameters of the stochastic guessing policy.

    Defaults follow the reference LLaMA inference settings. They are carried as
    data rather than hard-coded inside the analysis so every persisted result
    records the policy that produced it.

    **These generic defaults are backward-compatible, not the current protocol.**
    They reproduce what a configuration written before the null-model
    integration meant, so a config that omits either field keeps its historical
    behaviour: ``num_replicates = 8``, and a sampling stream mixed with the model
    seed. A scientific protocol is selected by the experiment configuration
    naming its values explicitly, never by inheriting a class default -- see
    ``configs/experiment/initialization_distribution.yaml``, which sets
    ``num_replicates: 1`` and ``common_random_numbers: true``.

    ``num_replicates`` is how many stochastic draws each position gets. At 1,
    greedy and nucleus each produce exactly ``D`` assignments, so both are
    summarised at the same sample size. Above 1 the per-replicate zero-frequency
    correction applies.

    ``common_random_numbers`` makes the sampling draws depend on the position
    alone, so every input condition *and* every model initialization sees the
    same underlying uniforms. That is what conditions the input-structure
    comparison on one fixed stochastic realization instead of letting sampling
    noise move between conditions. False -- the generic default -- gives each
    initialization its own stream, which is what every run before this
    integration did.
    """

    temperature: float = 0.6
    top_p: float = 0.9
    seed: int = 20240601
    num_replicates: int = 8
    common_random_numbers: bool = False

    def validate(self) -> None:
        if self.temperature <= 0:
            raise ValueError("nucleus temperature must be positive.")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("nucleus top_p must lie in (0, 1].")
        if self.seed < 0:
            raise ValueError("nucleus seed must be non-negative.")
        if self.num_replicates <= 0:
            raise ValueError("nucleus num_replicates must be positive.")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable description for persisted metadata."""

        return {
            "method": "temperature_top_p",
            "temperature": self.temperature,
            "top_p": self.top_p,
            "sampling_seed": self.seed,
            "num_replicates": self.num_replicates,
            "common_random_numbers": self.common_random_numbers,
        }


@dataclass(frozen=True)
class EvaluationPositions:
    """The fixed set of prediction positions shared by every initialization."""

    starts: tuple[int, ...]
    block_size: int
    input_ids: torch.Tensor
    target_ids: torch.Tensor

    @property
    def num_windows(self) -> int:
        """Number of evaluation windows."""

        return len(self.starts)

    @property
    def num_positions(self) -> int:
        """Number of next-token prediction positions being analyzed."""

        return self.num_windows * self.block_size

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable identity of the selected positions."""

        return {
            "strategy": "evenly_spaced_deterministic",
            "block_size": self.block_size,
            "num_windows": self.num_windows,
            "num_positions": self.num_positions,
            "first_start": self.starts[0],
            "last_start": self.starts[-1],
        }


@dataclass(frozen=True)
class InitializationMeasurement:
    """Complete per-token record for one model initialization.

    Every vector has length ``vocab_size`` and is indexed by token ID, so all
    records in an experiment align without any further bookkeeping.
    """

    model_seed: int
    greedy_counts: torch.Tensor
    nucleus_counts: torch.Tensor
    mean_predicted_probabilities: torch.Tensor
    num_positions: int

    @property
    def nucleus_counts_mean(self) -> torch.Tensor:
        """Replicate-averaged stochastic guess counts for this initialization."""

        return self.nucleus_counts.float().mean(dim=0)


def deterministic_window_starts(
    num_tokens: int,
    *,
    block_size: int,
    num_windows: int,
) -> list[int]:
    """Choose evaluation-window start offsets deterministically.

    Windows are spread as evenly as possible across the split rather than drawn
    at random. Two properties matter more here than randomness: the same
    positions must be reproducible for every initialization and every rerun, and
    they should cover the corpus uniformly so the selected-position token
    distribution stays close to the full-split one. No seed is involved, so the
    selection cannot drift when the model seed changes.

    Args:
        num_tokens: Length of the token split being sampled.
        block_size: Window length in tokens.
        num_windows: Number of windows to select.

    Returns:
        Sorted, distinct start offsets.

    Raises:
        ValueError: If the arguments are non-positive, or the split cannot
            supply that many distinct windows.
    """

    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    if num_windows <= 0:
        raise ValueError("num_windows must be positive.")

    max_start = num_tokens - block_size - 1
    if max_start < 0:
        raise ValueError(
            f"The split has {num_tokens} tokens, which cannot supply a window of "
            f"block_size + 1 = {block_size + 1} tokens."
        )
    available = max_start + 1
    if num_windows > available:
        raise ValueError(
            f"Requested {num_windows} windows but the split only supports {available} "
            "distinct start offsets. Reduce num_windows, reduce block_size, or use a "
            "larger corpus."
        )

    if num_windows == 1:
        return [0]
    step = max_start / (num_windows - 1)
    return [int(round(index * step)) for index in range(num_windows)]


def build_evaluation_positions(
    token_ids: Sequence[int],
    *,
    block_size: int,
    num_windows: int,
    device: torch.device | str = "cpu",
) -> EvaluationPositions:
    """Materialize the fixed evaluation windows and their next-token targets.

    Args:
        token_ids: Token IDs of the split being analyzed.
        block_size: Window length in tokens.
        num_windows: Number of deterministic windows.
        device: Device for the returned tensors.

    Returns:
        The shared :class:`EvaluationPositions` for the whole experiment.
    """

    starts = deterministic_window_starts(
        len(token_ids),
        block_size=block_size,
        num_windows=num_windows,
    )
    data = torch.tensor(list(token_ids), dtype=torch.long)
    inputs = torch.stack([data[start : start + block_size] for start in starts])
    targets = torch.stack([data[start + 1 : start + block_size + 1] for start in starts])
    return EvaluationPositions(
        starts=tuple(starts),
        block_size=block_size,
        input_ids=inputs.to(torch.device(device)),
        target_ids=targets.to(torch.device(device)),
    )


def compute_evaluation_logits(
    model: torch.nn.Module,
    positions: EvaluationPositions,
    *,
    vocab_size: int,
    forward_batch_size: int = 32,
) -> torch.Tensor:
    """Return restricted logits for **every** position at once.

    .. warning::
       This materializes ``[num_windows, block_size, vocab_size]``. At a
       realistic subword vocabulary that is enormous -- 32768 positions over
       32000 tokens is about 3.9 GiB in float32 -- so the experiment does not
       use it. It exists as the small, obviously-correct reference that
       :func:`measure_initialization` is tested against, and is appropriate only
       for small vocabularies or few positions.

    Args:
        model: Model whose forward accepts ``input_ids`` and returns ``.logits``.
        positions: The shared evaluation positions.
        vocab_size: Valid token support, normally the tokenizer vocabulary.
        forward_batch_size: Number of windows per forward pass.

    Returns:
        Logits with shape ``[num_windows, block_size, vocab_size]``.
    """

    if forward_batch_size <= 0:
        raise ValueError("forward_batch_size must be positive.")

    model.eval()
    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, positions.num_windows, forward_batch_size):
            window = positions.input_ids[start : start + forward_batch_size]
            output = model(input_ids=window)
            chunks.append(restrict_to_support(output.logits, vocab_size=vocab_size))
    return torch.cat(chunks, dim=0)


def sampling_uniforms(
    sampling: NucleusSamplingSettings,
    *,
    model_seed: int,
    num_positions: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Pre-draw one uniform per replicate and position.

    Drawing the randomness up front, indexed by position, is what makes a
    streamed measurement identical to an all-at-once one: a position's draw
    depends on the position, never on which batch it happened to fall in.

    The cost is negligible next to the logits it replaces -- 8 replicates over
    32768 positions is about 1 MB -- and each replicate gets its own stream, so
    replicates stay independent and reproducible.

    Args:
        sampling: Stochastic policy parameters.
        model_seed: Initialization seed, mixed in so no two initializations
            share a sampling stream.
        num_positions: Number of prediction positions in the experiment.
        device: Device the draws are placed on.

    Returns:
        Uniform draws with shape ``[num_replicates, num_positions]``.
    """

    sampling.validate()
    if num_positions <= 0:
        raise ValueError("num_positions must be positive.")

    device = torch.device(device)
    rows = []
    for replicate in range(sampling.num_replicates):
        generator = torch.Generator(device=device)
        # With common random numbers the stream depends on the replicate alone,
        # so the same position draws the same uniform in every condition and
        # every initialization. Otherwise it is distinct per (initialization,
        # replicate); the multiplier exceeds any plausible replicate count, so no
        # two pairs collide.
        offset = 0 if sampling.common_random_numbers else model_seed
        generator.manual_seed((sampling.seed + offset) * 1_000_003 + replicate)
        rows.append(torch.rand(num_positions, generator=generator, device=device))
    return torch.stack(rows)


def _accumulate_batch(
    logits: torch.Tensor,
    *,
    sampling: NucleusSamplingSettings,
    uniforms: torch.Tensor,
    greedy_counts: torch.Tensor,
    nucleus_counts: torch.Tensor,
    probability_sum: torch.Tensor,
    vocab_size: int,
) -> None:
    """Fold one batch of logits into the running per-token counters.

    Everything derived from a batch is derived here, from the same masked
    logits, before the batch is discarded: greedy guesses, every nucleus
    replicate, and the probability mass. That is what allows the caller to keep
    only ``[vocab]``-sized state.
    """

    greedy_counts += guess_counts(greedy_guess_ids(logits), vocab_size=vocab_size)
    for replicate in range(sampling.num_replicates):
        sampled = nucleus_guess_ids(
            logits,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            uniforms=uniforms[replicate],
        )
        nucleus_counts[replicate] += guess_counts(sampled, vocab_size=vocab_size)
    # float64 because a 32k-token sum over tens of thousands of positions loses
    # meaningful precision in float32.
    probability_sum += (
        torch.softmax(logits.float(), dim=-1).reshape(-1, vocab_size).sum(dim=0).double()
    )


def measure_initialization(
    model: torch.nn.Module,
    positions: EvaluationPositions,
    *,
    model_seed: int,
    vocab_size: int,
    sampling: NucleusSamplingSettings,
    eligible_token_ids: Sequence[int] | None = None,
    forward_batch_size: int = 32,
    device: torch.device | str = "cpu",
    input_ids: torch.Tensor | None = None,
    inputs_embeds: torch.Tensor | None = None,
) -> InitializationMeasurement:
    """Measure one initialization without ever holding all logits.

    The model is run batch by batch. Each batch produces a transient
    ``[batch, block, vocab]`` tensor from which greedy guesses, every nucleus
    replicate, and the probability mass are taken; only ``[vocab]``-sized
    counters survive the iteration. No accumulator is ever indexed by position,
    so memory is flat in the number of evaluation positions.

    Both policies read the *same* logits, as before: batching changes when the
    logits are discarded, not what they are.

    Args:
        model: Model whose forward accepts ``input_ids`` and returns ``.logits``.
        positions: The shared evaluation positions.
        model_seed: Seed that produced this initialization.
        vocab_size: Full canonical vocabulary size.
        sampling: Stochastic policy parameters.
        eligible_token_ids: Predictive support. ``None`` means every token, the
            character-tokenizer case.
        forward_batch_size: Windows per forward pass. Affects memory only.
        device: Device for the counters and the sampling draws.
        input_ids: Token windows to feed instead of ``positions.input_ids``, for
            the shuffled condition. Must match the real windows' shape, so the
            position count and the sampling draws line up.
        inputs_embeds: Synthetic ``[windows, block, dim]`` vectors fed at the
            embedding boundary instead of any token IDs, for the Gaussian
            condition. Mutually exclusive with ``input_ids``.

    Returns:
        One :class:`InitializationMeasurement` with complete per-token vectors.

    Note:
        The evaluation *targets* are never altered by a condition. Only the
        model input changes; the corpus reference stays what it always was.
    """

    sampling.validate()
    if forward_batch_size <= 0:
        raise ValueError("forward_batch_size must be positive.")

    device = torch.device(device)
    mask = eligible_support_mask(vocab_size, eligible_token_ids, device=device)
    uniforms = sampling_uniforms(
        sampling,
        model_seed=model_seed,
        num_positions=positions.num_positions,
        device=device,
    )

    greedy_counts = torch.zeros(vocab_size, dtype=torch.long, device=device)
    nucleus_counts = torch.zeros(
        (sampling.num_replicates, vocab_size), dtype=torch.long, device=device
    )
    probability_sum = torch.zeros(vocab_size, dtype=torch.float64, device=device)

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("Provide at most one of input_ids or inputs_embeds.")
    source_ids = positions.input_ids if input_ids is None else input_ids
    if inputs_embeds is None and tuple(source_ids.shape) != tuple(positions.input_ids.shape):
        raise ValueError(
            f"input_ids shape {tuple(source_ids.shape)} does not match the evaluation "
            f"windows {tuple(positions.input_ids.shape)}; the conditions must cover the "
            "same positions for the sampling draws to align."
        )
    if inputs_embeds is not None and tuple(inputs_embeds.shape[:2]) != tuple(
        positions.input_ids.shape
    ):
        raise ValueError(
            f"inputs_embeds covers {tuple(inputs_embeds.shape[:2])} positions but the "
            f"evaluation windows are {tuple(positions.input_ids.shape)}."
        )

    model.eval()
    consumed = 0
    with torch.no_grad():
        for start in range(0, positions.num_windows, forward_batch_size):
            if inputs_embeds is None:
                output = model(input_ids=source_ids[start : start + forward_batch_size])
            else:
                output = model(
                    inputs_embeds=inputs_embeds[start : start + forward_batch_size]
                )
            logits = restrict_to_support(output.logits, vocab_size=vocab_size)
            # Cloned because the mask is applied in place and the model may hand
            # back a view of its own buffer.
            logits = apply_support_mask(logits.clone(), mask)
            batch_positions = int(logits[..., 0].numel())
            _accumulate_batch(
                logits,
                sampling=sampling,
                uniforms=uniforms[:, consumed : consumed + batch_positions],
                greedy_counts=greedy_counts,
                nucleus_counts=nucleus_counts,
                probability_sum=probability_sum,
                vocab_size=vocab_size,
            )
            consumed += batch_positions
            del logits, output

    if consumed != positions.num_positions:
        raise RuntimeError(
            f"Measured {consumed} positions but the experiment defines "
            f"{positions.num_positions}."
        )

    return InitializationMeasurement(
        model_seed=model_seed,
        greedy_counts=greedy_counts.cpu(),
        nucleus_counts=nucleus_counts.cpu(),
        mean_predicted_probabilities=(probability_sum / consumed).float().cpu(),
        num_positions=consumed,
    )


def measure_from_logits(
    *,
    model_seed: int,
    logits: torch.Tensor,
    vocab_size: int,
    sampling: NucleusSamplingSettings,
    eligible_token_ids: Sequence[int] | None = None,
    device: torch.device | str = "cpu",
) -> InitializationMeasurement:
    """Reference measurement taking all logits at once.

    Small and obviously correct, and therefore the thing
    :func:`measure_initialization` is tested against. Because both consume the
    same pre-drawn uniforms, the two agree exactly rather than approximately.

    Not used by the experiment: at a realistic vocabulary the caller would have
    to materialize every position's logits to call it.
    """

    sampling.validate()
    if logits.shape[-1] != vocab_size:
        raise ValueError(
            f"logits last dimension {logits.shape[-1]} does not match vocab_size {vocab_size}. "
            "Restrict the logits to the valid support before measuring."
        )

    device = torch.device(device)
    num_positions = int(logits[..., 0].numel())
    mask = eligible_support_mask(vocab_size, eligible_token_ids, device=logits.device)
    masked = apply_support_mask(logits.clone(), mask)

    greedy_counts = torch.zeros(vocab_size, dtype=torch.long, device=masked.device)
    nucleus_counts = torch.zeros(
        (sampling.num_replicates, vocab_size), dtype=torch.long, device=masked.device
    )
    probability_sum = torch.zeros(vocab_size, dtype=torch.float64, device=masked.device)
    _accumulate_batch(
        masked,
        sampling=sampling,
        uniforms=sampling_uniforms(
            sampling,
            model_seed=model_seed,
            num_positions=num_positions,
            device=device,
        ).to(masked.device),
        greedy_counts=greedy_counts,
        nucleus_counts=nucleus_counts,
        probability_sum=probability_sum,
        vocab_size=vocab_size,
    )

    return InitializationMeasurement(
        model_seed=model_seed,
        greedy_counts=greedy_counts.cpu(),
        nucleus_counts=nucleus_counts.cpu(),
        mean_predicted_probabilities=(probability_sum / num_positions).float().cpu(),
        num_positions=num_positions,
    )
