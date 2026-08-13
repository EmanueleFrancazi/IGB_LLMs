"""Token-level aggregation of per-position parameter-gradient norms.

The record stores one exact gradient norm per evaluation position. This module
turns those into the per-token quantities the eventual figure 8 relates, and it
is the only place that aggregation is defined.

Let ``S`` be the set of positions the gradient analysis actually covered -- every
evaluation position unless a run deliberately used a window subset. Then, for
token ID ``i``:

.. code-block:: text

    n_i     = |{d in S : y_d = i}|                     target occurrences
    G_i     = mean_{d in S : y_d = i} g_d              mean gradient norm
    q_i     = |{d in S : greedy_d = i}| / |S|          greedy guess fraction
    p_i     = corpus fraction over the whole analysis split

The asymmetry between ``q_i`` and ``p_i`` is deliberate and is the one thing to
keep straight here.

``q_i`` is computed from the greedy predictions stored **at the same positions**
as the norms, not from the record's full-run ``greedy_counts``. When the analysis
covers every position the two agree exactly, and a test asserts that. When it
covers a subset they do not, and only the position-aligned version is a valid
partner for ``G_i``: both must describe the same ``S``.

``p_i`` is the opposite case. It intentionally refers to the complete analysis
split, because it is the empirical reference the whole experiment compares
against, not a property of the positions that happened to be differentiated.

Like the rest of :mod:`llm_behavior_lab.analysis`, this depends only on NumPy, so
a finished experiment is re-analyzable without PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "TokenGradientSummary",
    "gradient_guess_table",
    "token_gradient_norms",
]


@dataclass(frozen=True)
class TokenGradientSummary:
    """Per-token gradient aggregation over the gradient-evaluated positions.

    Both vectors are ``[V]`` and indexed by token ID, aligning with every other
    per-token vector in the record without further bookkeeping.
    """

    #: ``G_i``. NaN where the token never appears as a target in ``S``: no
    #: gradient was measured for it, which is different from a measured zero.
    mean_gradient_norm: np.ndarray
    #: ``n_i``, the number of positions in ``S`` whose target was this token.
    target_count: np.ndarray
    #: ``|S|``.
    num_positions: int
    #: Whether ``S`` is the experiment's whole evaluation-position set.
    covers_all_positions: bool
    #: Which initialization the gradients were measured on.
    initialization_index: int

    @property
    def measured_token_count(self) -> int:
        """Number of tokens that occur as a target at least once in ``S``."""

        return int((self.target_count > 0).sum())


def _require_gradients(record: Any) -> None:
    if not record.has_position_gradients:
        raise ValueError(
            "This record carries no per-position gradient analysis. Rerun the "
            "experiment with gradient analysis enabled."
        )


def _covers_all_positions(record: Any) -> bool:
    """Whether the analysis covered every evaluation position.

    Taken from the recorded protocol when present, and otherwise inferred by
    comparing the position count with the experiment's own, so a record written
    before the flag existed still reports the truth rather than a default.
    """

    recorded = record.gradient_analysis.get("covers_all_positions")
    if recorded is not None:
        return bool(recorded)
    total = record.metadata.get("analysis", {}).get("num_positions")
    if total is None:
        total = record.metadata.get("num_positions")
    if total is None:
        raise ValueError(
            "The record does not say how many evaluation positions the experiment "
            "defined, so gradient coverage cannot be established."
        )
    return int(record.gradient_position_norms.shape[0]) == int(total)


def token_gradient_norms(record: Any) -> TokenGradientSummary:
    """Aggregate per-position gradient norms into ``G_i`` and ``n_i``.

    Args:
        record: A record carrying per-position gradient arrays.

    Returns:
        The per-token summary, with ``G_i`` NaN wherever ``n_i`` is zero.

    Raises:
        ValueError: If the record carries no gradient analysis.
    """

    _require_gradients(record)
    vocab_size = record.vocab_size
    targets = np.asarray(record.gradient_position_target_ids, dtype=np.int64)
    norms = np.asarray(record.gradient_position_norms, dtype=np.float64)

    counts = np.bincount(targets, minlength=vocab_size).astype(np.int64)
    sums = np.bincount(targets, weights=norms, minlength=vocab_size)

    means = np.full(vocab_size, np.nan, dtype=np.float64)
    measured = counts > 0
    means[measured] = sums[measured] / counts[measured]

    return TokenGradientSummary(
        mean_gradient_norm=means,
        target_count=counts,
        num_positions=int(norms.shape[0]),
        covers_all_positions=_covers_all_positions(record),
        initialization_index=record.gradient_initialization_index,
    )


def gradient_guess_table(record: Any) -> dict[str, Any]:
    """Assemble the aligned per-token table relating ``G_i``, ``q_i`` and ``p_i``.

    Every column is ``[V]`` and indexed by token ID.

    ``greedy_guess_fraction`` is derived from the greedy predictions stored at
    the gradient-evaluated positions, so it and ``mean_gradient_norm`` describe
    the same position set. It is deliberately *not* ``record.greedy_fractions``,
    which always covers every position and would silently mismatch ``G_i`` for a
    subset run.

    Args:
        record: A record carrying per-position gradient arrays.

    Returns:
        A dictionary of aligned columns plus the provenance a reader needs to
        avoid mistaking a subset table for a whole-experiment one.
    """

    summary = token_gradient_norms(record)
    vocab_size = record.vocab_size
    greedy = np.asarray(record.gradient_position_greedy_ids, dtype=np.int64)

    guess_counts = np.bincount(greedy, minlength=vocab_size).astype(np.int64)
    guess_fractions = guess_counts.astype(np.float64) / float(summary.num_positions)

    return {
        "token_id": np.arange(vocab_size),
        "target_occurrence_count": summary.target_count,
        "mean_gradient_norm": summary.mean_gradient_norm,
        "greedy_guess_count": guess_counts,
        "greedy_guess_fraction": guess_fractions,
        "corpus_fraction": record.corpus_fractions,
        "eligible_mask": record.eligible_mask,
        "num_positions": summary.num_positions,
        "covers_all_positions": summary.covers_all_positions,
        "initialization_index": summary.initialization_index,
        "greedy_source": "gradient_evaluated_positions",
        "corpus_source": "whole_analysis_split",
    }
