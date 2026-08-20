"""Persisted records for the initialization-distribution experiment.

One experiment produces a single record directory holding two files::

    <name>.npz     every per-token vector, token-ID aligned
    <name>.json    the protocol that produced them

The split is deliberate. The arrays are bulky and numeric; the metadata is small
and must stay readable without loading NumPy. Keeping complete per-token vectors
-- rather than the top-N summaries the console analysis prints -- is what makes
the record re-analyzable later without rerunning any model.

This module, and the rest of :mod:`llm_behavior_lab.analysis`, depends only on
NumPy. Re-analyzing a finished experiment therefore needs neither PyTorch nor a
GPU, which matters because analysis is revisited far more often than it is
produced.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "InitializationExperimentRecord",
    "RECORD_VERSION",
    "load_record",
]

#: Version 2 added ``eligible_token_ids``. Version 3 added the optional
#: input-condition counts and the uniform-null profile. Version 5 added the
#: optional per-position gradient arrays. Version 6 added the optional raw
#: predictive-probability diagnostics, and version 7 the temperature-conditioned
#: greedy-confidence grid, and version 8 the temperature-conditioned gradient
#: norms, and version 9 the identity-preserving mean token probabilities.
#: Older records load unchanged: version 1 predates special-token
#: exclusion, so every token was eligible -- exactly the
#: default applied when that array is absent -- and every later addition is
#: optional, so its absence simply means the run did not carry that analysis.
RECORD_VERSION = 10

_ARRAY_NAMES = (
    "token_ids",
    "corpus_counts",
    "selected_target_counts",
    "greedy_counts",
    "nucleus_counts",
    "mean_predicted_probabilities",
    "model_seeds",
)

#: Present from version 2 onward; defaulted when loading an older record.
_OPTIONAL_ARRAY_NAMES = ("eligible_token_ids",)

#: Version 3 additions, stored under flat prefixed keys so the archive stays a
#: plain name-to-array mapping. ``real`` is not stored again here: it is already
#: ``greedy_counts`` / ``nucleus_counts``.
_CONDITION_GREEDY_PREFIX = "condition_greedy__"
_CONDITION_NUCLEUS_PREFIX = "condition_nucleus__"
_NULL_PREFIX = "uniform_null__"

#: Version 4 additions. ``sweep_counts__<condition>`` is ``[S, T, V]`` and
#: ``sweep_agreement__<condition>`` is ``[S, T]``; the temperatures themselves
#: live in metadata so the arrays stay purely numeric.
_SWEEP_COUNTS_PREFIX = "sweep_counts__"
_SWEEP_AGREEMENT_PREFIX = "sweep_agreement__"

#: Version 5 additions. All four are ``[D_g]`` and aligned entry by entry, where
#: ``D_g`` is the number of positions the gradient analysis covered -- every
#: evaluation position unless a run deliberately used a window subset. They are
#: present or absent together; a run without the analysis stores none of them.
#:
#: Raw per-position values are stored rather than pre-aggregated per-token sums,
#: because ``records`` owns the format and ``aggregation`` owns every statistic.
#: ``G_i`` and ``n_i`` are therefore derived, never persisted twice, and a later
#: re-aggregation stays possible without recomputing a single gradient. The cost
#: is about 0.8 MB at ``D_g = 32768``.
_GRADIENT_ARRAY_NAMES = (
    "gradient_position_indices",
    "gradient_position_target_ids",
    "gradient_position_greedy_ids",
    "gradient_position_norms",
)

#: Version 6 additions: sufficient statistics of the **raw ``T = 1`` predictive
#: distribution**, before any sampling policy. ``[I, K]`` for the ranked profile
#: and ``[I, D]`` for the three per-position vectors. Present or absent together.
#:
#: The full ``[I, D, K]`` probability tensor is deliberately never stored -- at
#: 12 initializations, 32768 positions and 31997 eligible tokens it would be
#: about 50 GiB, against roughly 12 MB for these statistics.
_PROBABILITY_ARRAY_NAMES = (
    "predictive_ranked_probabilities",
    "predictive_max_probabilities",
    "predictive_target_probabilities",
    "predictive_target_losses",
)

#: Version 7 additions: the same statistics evaluated at a grid of diagnostic
#: temperatures, ``[I, N_T, K]`` for the ranked profiles and ``[I, N_T, D]`` for
#: the per-position vectors, plus the grid itself and the mean entropy.
#:
#: These are a **confidence** diagnostic, not a sampling one. Nothing is drawn
#: and nothing is truncated; temperature only reshapes the logits into a
#: probability vector whose geometry is measured. Because softmax is strictly
#: increasing, every temperature here describes the *same* greedy decisions.
#:
#: The ``T = 1`` slice duplicates the version 6 arrays on purpose: it keeps the
#: grid self-contained and lets validation assert the two agree exactly.
_TEMPERATURE_ARRAY_NAMES = (
    "predictive_temperatures",
    "predictive_temperature_ranked_probabilities",
    "predictive_temperature_max_probabilities",
    "predictive_temperature_target_probabilities",
    "predictive_temperature_target_losses",
    "predictive_temperature_mean_entropy",
)

#: The unscaled reference inside that grid.
CANONICAL_TEMPERATURE = 1.0

#: Version 8 additions. ``[N_T]`` temperatures and ``[N_T, D_g]`` exact gradient
#: norms, where temperature enters the **loss** rather than rescaling a result:
#: ``ell_T(d) = -log softmax(z_d/T)[y_d]``. The canonical ``T = 1`` row is also
#: kept under the version 5 name, and validation asserts the two agree exactly.
#:
#: Small: 7 temperatures over 32768 positions is about 1.8 MB. The expense of
#: this analysis is backward passes, not storage.
_GRADIENT_TEMPERATURE_ARRAY_NAMES = (
    "gradient_temperatures",
    "gradient_temperature_position_norms",
)

#: Version 9 addition: ``[I, N_T, V]`` mean probability **at fixed token
#: identity**, ``pbar_{s,T}(i) = mean_d p_{s,T}(d, i)``.
#:
#: This is the opposite order of operations from the ranked profile: identity is
#: preserved while averaging, and ranking happens afterwards. Comparing the two
#: separates within-prediction concentration from a persistent preference for
#: particular tokens. Its ``T = 1`` row generalizes ``mean_predicted_probabilities``
#: rather than duplicating that logic, and validation asserts the two agree.
#:
#: The ranked figure-14 profile is derived from this deterministically and is not
#: stored again. About 21.5 MB at 12 initializations, 7 temperatures, V = 32000 --
#: against roughly 50 GiB for the full ``[I, N_T, D, V]`` tensor, which is never
#: formed.
_MEAN_TOKEN_ARRAY_NAME = "predictive_temperature_mean_token_probabilities"

#: Version 10 addition: ``[D_g, K]`` count sketch of every evaluated position's
#: canonical-temperature parameter gradient.
#:
#: Direction only is the question, so a projection that preserves inner products
#: is enough and the exact ``[D_g, P]`` matrix -- about 1.1 TB in float32 at
#: experiment scale -- is never formed. About 67 MB at ``D_g = 32768, K = 512`` in float32,
#: which is why it is stored at that precision: the count sketch's own
#: ``1/sqrt(K)`` error dominates float32 rounding by orders of magnitude.
#:
#: Optional, like every array added since version 5: a record written without it
#: stays valid and every figure that does not need it still renders.
_GRADIENT_SKETCH_ARRAY_NAME = "gradient_position_sketches"


def _atomic_write_bytes(path: Path, write) -> Path:
    """Write via a temporary file in the destination directory, then rename.

    The temporary file keeps the destination's extension: ``np.savez_compressed``
    appends ``.npz`` when a path lacks it, which would leave the archive beside
    the file that gets renamed into place.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        write(temporary_path)
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return path


def _optional_array(values: Any, dtype: Any) -> np.ndarray | None:
    """Convert an optional sequence to an array, preserving ``None``.

    ``None`` means "this run did not carry that analysis" and must survive as
    ``None``; an empty array would claim the analysis ran and found nothing.
    """

    if values is None:
        return None
    return np.asarray(values, dtype=dtype)


def _fractions(counts: np.ndarray) -> np.ndarray:
    """Normalize counts along the last axis, guarding against empty totals."""

    totals = counts.sum(axis=-1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("Cannot normalize counts whose total is zero.")
    return counts.astype(np.float64) / totals


@dataclass(frozen=True)
class InitializationExperimentRecord:
    """Complete per-token result of one multi-initialization experiment.

    Every vector is indexed by token ID over the same valid support, so no
    alignment step is ever needed downstream. Shapes, with ``S``
    initializations, ``R`` sampling replicates, and ``V`` valid tokens:

    ==============================  =========  ====================================
    field                           shape      meaning
    ==============================  =========  ====================================
    ``token_ids``                   ``[V]``    canonical alignment key
    ``corpus_counts``               ``[V]``    whole analysis split
    ``selected_target_counts``      ``[V]``    next-token targets at the analyzed
                                               positions only
    ``greedy_counts``               ``[S,V]``  argmax guesses per initialization
    ``nucleus_counts``              ``[S,R,V]``sampled guesses, per replicate
    ``mean_predicted_probabilities````[S,V]``  mean predictive mass, *not* guesses
    ``model_seeds``                 ``[S]``    initialization seed per row
    ``eligible_token_ids``          ``[E]``    canonical IDs forming the
                                               predictive support
    ==============================  =========  ====================================

    Three vocabulary sizes must not be confused, and the record keeps all three
    recoverable: the **full** vocabulary ``V``, the **eligible** predictive
    support ``E`` after removing structural tokens, and the **corpus-observed**
    support, the eligible tokens that actually occur.

    One optional group is indexed by *position* rather than by token: the
    ``gradient_position_*`` vectors, each ``[D_g]``, holding the exact
    single-position parameter-gradient norm and the target and greedy token at
    every position the gradient analysis covered. They are token-aggregated by
    :mod:`llm_behavior_lab.analysis.gradients`, never here.
    """

    token_ids: np.ndarray
    corpus_counts: np.ndarray
    selected_target_counts: np.ndarray
    greedy_counts: np.ndarray
    nucleus_counts: np.ndarray
    mean_predicted_probabilities: np.ndarray
    model_seeds: np.ndarray
    eligible_token_ids: np.ndarray
    metadata: dict[str, Any]
    #: Input condition -> ``[S, V]`` greedy counts. Excludes ``real``.
    condition_greedy_counts: dict[str, np.ndarray] = field(default_factory=dict)
    #: Input condition -> ``[S, R, V]`` nucleus counts. Excludes ``real``.
    condition_nucleus_counts: dict[str, np.ndarray] = field(default_factory=dict)
    #: ``ranked_mean`` / ``ranked_low`` / ``ranked_high`` over the eligible support.
    uniform_null: dict[str, np.ndarray] = field(default_factory=dict)
    #: Input condition -> ``[S, T, V]`` nucleus counts, one row per sweep
    #: temperature. Absent when no sweep ran.
    sweep_counts_by_condition: dict[str, np.ndarray] = field(default_factory=dict)
    #: Input condition -> ``[S, T]`` fraction of positions agreeing with greedy.
    sweep_agreement_by_condition: dict[str, np.ndarray] = field(default_factory=dict)
    #: ``[D_g]`` flat ``window * block_size + offset`` index of each position the
    #: gradient analysis covered. ``None`` when the run carried no such analysis.
    gradient_position_indices: np.ndarray | None = None
    #: ``[D_g]`` true next token ``y_d`` at each of those positions.
    gradient_position_target_ids: np.ndarray | None = None
    #: ``[D_g]`` greedy argmax at each of those positions, read from the same
    #: support-masked logits as the loss. This is what ``q_i`` is derived from,
    #: so the guess fractions describe the same position set as the norms.
    gradient_position_greedy_ids: np.ndarray | None = None
    #: ``[D_g]`` exact ``|| grad_theta ell_d ||_2`` over all trainable parameters.
    gradient_position_norms: np.ndarray | None = None
    #: ``[I, K]`` mean probability at each *within-position* rank. Ranked first,
    #: averaged second -- not the ranking of an aggregate distribution.
    predictive_ranked_probabilities: np.ndarray | None = None
    #: ``[I, D]`` probability of the greedy token at each evaluated position.
    predictive_max_probabilities: np.ndarray | None = None
    #: ``[I, D]`` probability assigned to the true next token.
    predictive_target_probabilities: np.ndarray | None = None
    #: ``[I, D]`` single-position cross-entropy ``-log p_target``.
    predictive_target_losses: np.ndarray | None = None
    #: ``[I, N_T, V]`` mean probability at fixed token identity (figure 14).
    predictive_temperature_mean_token_probabilities: np.ndarray | None = None
    gradient_position_sketches: np.ndarray | None = None
    #: ``[N_T]`` temperatures at which the loss itself was defined.
    gradient_temperatures: np.ndarray | None = None
    #: ``[N_T, D_g]`` exact full-parameter gradient norm at each temperature.
    gradient_temperature_position_norms: np.ndarray | None = None
    #: ``[N_T]`` diagnostic temperatures, in the order the arrays are indexed by.
    predictive_temperatures: np.ndarray | None = None
    #: ``[I, N_T, K]`` ranked profile at each diagnostic temperature.
    predictive_temperature_ranked_probabilities: np.ndarray | None = None
    #: ``[I, N_T, D]`` probability of the greedy token at each temperature.
    predictive_temperature_max_probabilities: np.ndarray | None = None
    #: ``[I, N_T, D]`` probability of the true next token at each temperature.
    predictive_temperature_target_probabilities: np.ndarray | None = None
    #: ``[I, N_T, D]`` ``-log p_target`` at each temperature.
    predictive_temperature_target_losses: np.ndarray | None = None
    #: ``[I, N_T]`` mean per-position predictive entropy, in nats.
    predictive_temperature_mean_entropy: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.validate()

    # -- shape helpers ---------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Number of valid tokens shared by every vector."""

        return int(self.token_ids.shape[0])

    @property
    def num_initializations(self) -> int:
        """Number of independent model initializations."""

        return int(self.greedy_counts.shape[0])

    @property
    def num_replicates(self) -> int:
        """Number of stochastic sampling replicates per initialization."""

        return int(self.nucleus_counts.shape[1])

    @property
    def tokens(self) -> tuple[str, ...]:
        """Human-readable token strings, when recorded. Metadata only.

        Token ID remains the canonical key; these are for labelling figures.
        """

        recorded = self.metadata.get("tokens")
        if recorded is None:
            return ()
        return tuple(str(token) for token in recorded)

    @property
    def eligible_vocab_size(self) -> int:
        """Size of the predictive support, after excluding structural tokens."""

        return int(self.eligible_token_ids.shape[0])

    @property
    def eligible_mask(self) -> np.ndarray:
        """Boolean ``[V]`` mask selecting the predictive support."""

        mask = np.zeros(self.vocab_size, dtype=bool)
        mask[self.eligible_token_ids] = True
        return mask

    @property
    def special_token_ids(self) -> np.ndarray:
        """Canonical IDs excluded from the predictive support."""

        return np.flatnonzero(~self.eligible_mask)

    @property
    def corpus_observed_support(self) -> int:
        """Eligible tokens that actually occur in the analysis split.

        Distinct from both the full vocabulary and the eligible support: a 32k
        subword vocabulary will contain many tokens a modest corpus never uses.
        """

        return int((self.corpus_counts[self.eligible_mask] > 0).sum())

    # -- distributions ---------------------------------------------------

    @property
    def corpus_fractions(self) -> np.ndarray:
        """``p``: empirical token fractions over the whole analysis split."""

        return _fractions(self.corpus_counts)

    @property
    def selected_target_fractions(self) -> np.ndarray:
        """``p_selected``: target fractions at the analyzed positions only."""

        return _fractions(self.selected_target_counts)

    @property
    def greedy_fractions(self) -> np.ndarray:
        """``q_greedy``: selected-guess fractions, one row per initialization."""

        return _fractions(self.greedy_counts)

    @property
    def nucleus_replicate_fractions(self) -> np.ndarray:
        """Per-replicate stochastic guess fractions, shape ``[S, R, V]``."""

        return _fractions(self.nucleus_counts)

    @property
    def nucleus_fractions(self) -> np.ndarray:
        """``q_nucleus``: replicate-averaged guess fractions per initialization.

        Replicates are averaged *within* an initialization so that the
        between-initialization comparison treats the initialization as the
        independent unit rather than the individual sampling draw.
        """

        return self.nucleus_replicate_fractions.mean(axis=1)

    @property
    def available_conditions(self) -> tuple[str, ...]:
        """Input conditions this record carries, ``real`` always first."""

        extra = [name for name in ("shuffled", "gaussian") if name in self.condition_greedy_counts]
        return ("real", *extra)

    @property
    def has_input_structure(self) -> bool:
        """Whether the paired input-structure comparison is available."""

        return len(self.available_conditions) > 1

    @property
    def has_uniform_null(self) -> bool:
        """Whether a simulated uniform-null profile is available."""

        return "ranked_mean" in self.uniform_null

    @property
    def has_temperature_sweep(self) -> bool:
        """Whether a multi-temperature nucleus sweep was recorded."""

        return bool(self.sweep_counts_by_condition) and bool(self.sweep_temperatures)

    @property
    def has_predictive_probability_analysis(self) -> bool:
        """Whether raw ``T = 1`` predictive-probability statistics were recorded."""

        return self.predictive_ranked_probabilities is not None

    @property
    def predictive_probability_protocol(self) -> dict[str, Any]:
        """What the recorded probabilities are, empty when none were recorded.

        Carries the facts a reader needs to avoid mistaking these for the
        nucleus distribution: raw logits, eligible support, ``T = 1``, before
        top-p, before any sampling decision.
        """

        return dict(self.metadata.get("analysis", {}).get("predictive_probabilities", {}))

    @property
    def uniform_probability(self) -> float:
        """``1 / K``: the probability each token gets under an exact uniform."""

        return 1.0 / float(self.eligible_vocab_size)

    @property
    def initialization_scale(self) -> float:
        """The scale multiplier ``alpha`` this record was produced at.

        Read from metadata, defaulting to 1.0 so every record written before the
        scale experiment reports the baseline it was in fact run at rather than
        an absent value.
        """

        recorded = self.metadata.get("initialization_scale", {})
        return float(recorded.get("alpha", 1.0)) if recorded else 1.0

    @property
    def raw_logit_diagnostics(self) -> list[dict[str, Any]]:
        """Per-initialization summaries of the finite eligible logits."""

        return list(self.metadata.get("analysis", {}).get("raw_logits", []))

    @property
    def has_mean_token_probabilities(self) -> bool:
        """Whether identity-preserving mean token probabilities were recorded."""

        return self.predictive_temperature_mean_token_probabilities is not None

    @property
    def has_temperature_gradient_analysis(self) -> bool:
        """Whether gradient norms were measured across a temperature grid."""

        return self.gradient_temperature_position_norms is not None

    @property
    def gradient_temperature_grid(self) -> tuple[float, ...]:
        """Temperatures at which the gradient loss was defined."""

        if self.gradient_temperatures is None:
            return ()
        return tuple(float(value) for value in self.gradient_temperatures)

    def gradient_temperature_index(self, temperature: float) -> int:
        """Row holding one gradient temperature."""

        grid = self.gradient_temperature_grid
        for index, value in enumerate(grid):
            if value == float(temperature):
                return index
        raise KeyError(f"Temperature {temperature} is not in the gradient grid {grid}.")

    @property
    def has_temperature_confidence_analysis(self) -> bool:
        """Whether the temperature-conditioned confidence grid was recorded."""

        return self.predictive_temperatures is not None

    @property
    def confidence_temperatures(self) -> tuple[float, ...]:
        """The diagnostic temperatures, in the order every array is indexed by."""

        if self.predictive_temperatures is None:
            return ()
        return tuple(float(value) for value in self.predictive_temperatures)

    def temperature_index(self, temperature: float) -> int:
        """Position of one diagnostic temperature within the grid."""

        grid = self.confidence_temperatures
        for index, value in enumerate(grid):
            if value == float(temperature):
                return index
        raise KeyError(f"Temperature {temperature} is not in the recorded grid {grid}.")

    @property
    def has_position_gradients(self) -> bool:
        """Whether per-position parameter-gradient norms were recorded."""

        return self.gradient_position_norms is not None

    @property
    def has_gradient_position_sketches(self) -> bool:
        """Whether per-position gradient sketches were recorded.

        Directional analysis needs both the sketches and the exact norms that
        scale them, so both are required here rather than the array alone.
        """

        return (
            self.gradient_position_sketches is not None
            and self.gradient_position_norms is not None
        )

    @property
    def gradient_analysis(self) -> dict[str, Any]:
        """Protocol of the gradient analysis, empty when none was recorded."""

        return dict(self.metadata.get("analysis", {}).get("gradient_analysis", {}))

    @property
    def gradient_initialization_index(self) -> int:
        """Which initialization the gradients were measured on.

        Read from the recorded protocol rather than assumed to be row 0, so a
        table built from these arrays can never be paired with the wrong
        initialization's guesses.
        """

        if not self.has_position_gradients:
            raise ValueError("This record carries no per-position gradient analysis.")
        return int(self.gradient_analysis.get("initialization_index", 0))

    @property
    def sweep_temperatures(self) -> tuple[float, ...]:
        """Sweep temperatures, in the order they were run and persisted."""

        recorded = self.metadata.get("analysis", {}).get("temperature_sweep", {})
        return tuple(float(value) for value in recorded.get("temperatures", ()))

    @property
    def sweep_conditions(self) -> tuple[str, ...]:
        """Input conditions the sweep covers, ``real`` first."""

        extra = [c for c in ("shuffled", "gaussian") if c in self.sweep_counts_by_condition]
        return ("real", *extra) if "real" in self.sweep_counts_by_condition else tuple(extra)

    def sweep_counts(self, condition: str = "real") -> np.ndarray:
        """``[S, T, V]`` sweep counts for one input condition."""

        if condition not in self.sweep_counts_by_condition:
            raise KeyError(
                f"Record has no sweep counts for condition {condition!r}; "
                f"available: {self.sweep_conditions}."
            )
        return self.sweep_counts_by_condition[condition]

    def sweep_agreement(self, condition: str = "real") -> np.ndarray | None:
        """``[S, T]`` greedy agreement, or ``None`` when it was not recorded."""

        return self.sweep_agreement_by_condition.get(condition)

    def condition_counts(self, condition: str, policy: str) -> np.ndarray:
        """Raw counts for one input condition and policy.

        ``real`` is served from the primary arrays rather than duplicated, so a
        record never holds the same numbers twice.
        """

        if condition == "real":
            return self.greedy_counts if policy == "greedy" else self.nucleus_counts
        store = (
            self.condition_greedy_counts if policy == "greedy" else self.condition_nucleus_counts
        )
        if condition not in store:
            raise KeyError(
                f"Record has no {policy!r} counts for condition {condition!r}; "
                f"available: {self.available_conditions}."
            )
        return store[condition]

    def condition_policy_fractions(self, condition: str, policy: str) -> np.ndarray:
        """``[S, V]`` guess fractions for one input condition and policy.

        Nucleus replicates are averaged within an initialization exactly as for
        the real condition, so the comparison across conditions is like-for-like.
        """

        if policy not in ("greedy", "nucleus"):
            raise ValueError(f"Unknown policy {policy!r}; expected 'greedy' or 'nucleus'.")
        counts = self.condition_counts(condition, policy)
        fractions = _fractions(counts)
        return fractions if policy == "greedy" else fractions.mean(axis=1)

    def policy_fractions(self, policy: str) -> np.ndarray:
        """Return the ``[S, V]`` guess fractions for ``"greedy"`` or ``"nucleus"``."""

        if policy == "greedy":
            return self.greedy_fractions
        if policy == "nucleus":
            return self.nucleus_fractions
        raise ValueError(f"Unknown policy {policy!r}; expected 'greedy' or 'nucleus'.")

    # -- validation and persistence --------------------------------------

    def validate(self) -> None:
        """Check that every vector aligns on the same token support."""

        vocab_size = self.token_ids.shape[0]
        if self.token_ids.ndim != 1 or vocab_size == 0:
            raise ValueError("token_ids must be a non-empty one-dimensional array.")
        if not np.array_equal(self.token_ids, np.arange(vocab_size)):
            raise ValueError(
                "token_ids must be the contiguous range [0, vocab_size); the record "
                "format uses position in the array as the token ID."
            )
        for name, expected in (
            ("corpus_counts", (vocab_size,)),
            ("selected_target_counts", (vocab_size,)),
        ):
            if getattr(self, name).shape != expected:
                raise ValueError(f"{name} must have shape {expected}.")

        num_inits = self.greedy_counts.shape[0]
        if self.greedy_counts.shape != (num_inits, vocab_size):
            raise ValueError("greedy_counts must have shape [initializations, vocab].")
        if self.mean_predicted_probabilities.shape != (num_inits, vocab_size):
            raise ValueError(
                "mean_predicted_probabilities must have shape [initializations, vocab]."
            )
        if self.nucleus_counts.ndim != 3 or self.nucleus_counts.shape[0] != num_inits:
            raise ValueError(
                "nucleus_counts must have shape [initializations, replicates, vocab]."
            )
        if self.nucleus_counts.shape[2] != vocab_size:
            raise ValueError("nucleus_counts must share the vocabulary dimension.")
        if self.model_seeds.shape != (num_inits,):
            raise ValueError("model_seeds must have one entry per initialization.")
        if len(set(self.model_seeds.tolist())) != num_inits:
            raise ValueError("model_seeds must be distinct; repeats are not independent.")

        if self.eligible_token_ids.ndim != 1 or self.eligible_token_ids.shape[0] == 0:
            raise ValueError("eligible_token_ids must be a non-empty one-dimensional array.")
        if self.eligible_token_ids.shape[0] > vocab_size:
            raise ValueError("eligible_token_ids cannot exceed the vocabulary size.")
        if int(self.eligible_token_ids.min()) < 0 or int(self.eligible_token_ids.max()) >= vocab_size:
            raise ValueError("eligible_token_ids contain values outside the vocabulary.")
        if len(set(self.eligible_token_ids.tolist())) != self.eligible_token_ids.shape[0]:
            raise ValueError("eligible_token_ids must be distinct.")
        excluded = np.setdiff1d(np.arange(vocab_size), self.eligible_token_ids)
        if excluded.size and int(self.corpus_counts[excluded].sum()) != 0:
            raise ValueError(
                "Excluded token IDs carry corpus counts, so the empirical distribution "
                "and the predictive support disagree. Corpus encoding must not emit "
                "structural tokens."
            )

        for name, store in (
            ("condition_greedy_counts", self.condition_greedy_counts),
            ("condition_nucleus_counts", self.condition_nucleus_counts),
        ):
            for condition, array in store.items():
                if condition == "real":
                    raise ValueError(
                        "The 'real' condition lives in the primary arrays and must not be "
                        f"duplicated in {name}."
                    )
                expected = (
                    self.greedy_counts.shape
                    if name == "condition_greedy_counts"
                    else self.nucleus_counts.shape
                )
                if array.shape != expected:
                    raise ValueError(
                        f"{name}[{condition!r}] has shape {array.shape}, expected {expected}; "
                        "every condition must cover the same initializations and support."
                    )
        for condition, array in self.sweep_counts_by_condition.items():
            if array.ndim != 3 or array.shape[0] != num_inits or array.shape[2] != vocab_size:
                raise ValueError(
                    f"sweep_counts[{condition!r}] must have shape "
                    f"[initializations, temperatures, vocab]; got {array.shape}."
                )
        for condition, array in self.sweep_agreement_by_condition.items():
            if array.ndim != 2 or array.shape[0] != num_inits:
                raise ValueError(
                    f"sweep_agreement[{condition!r}] must have shape "
                    f"[initializations, temperatures]; got {array.shape}."
                )
        for key, array in self.uniform_null.items():
            if array.ndim != 1:
                raise ValueError(f"uniform_null[{key!r}] must be one-dimensional.")

        self._validate_position_gradients(vocab_size, num_inits)
        self._validate_predictive_probabilities(num_inits)
        self._validate_gradient_sketches()
        self._validate_temperature_confidence(num_inits)
        self._validate_temperature_gradients()
        self._validate_mean_token_probabilities(num_inits, vocab_size)

        tokens = self.metadata.get("tokens")
        if tokens is not None and len(tokens) != vocab_size:
            raise ValueError("metadata['tokens'] must have one entry per valid token ID.")

    def _validate_position_gradients(self, vocab_size: int, num_inits: int) -> None:
        """Check the optional per-position gradient arrays.

        They are all-or-nothing: a record holding norms without the positions and
        targets they belong to could not be aggregated, and one holding indices
        without norms would silently produce an empty analysis.
        """

        present = {
            name: getattr(self, name)
            for name in _GRADIENT_ARRAY_NAMES
            if getattr(self, name) is not None
        }
        if not present:
            return
        missing = sorted(set(_GRADIENT_ARRAY_NAMES) - set(present))
        if missing:
            raise ValueError(
                "Per-position gradient arrays are all-or-nothing; missing: "
                + ", ".join(missing)
            )

        lengths = {name: array.shape for name, array in present.items()}
        for name, shape in lengths.items():
            if len(shape) != 1:
                raise ValueError(f"{name} must be one-dimensional, got shape {shape}.")
        if len(set(shape[0] for shape in lengths.values())) != 1:
            raise ValueError(
                "Every per-position gradient array must cover the same positions; "
                f"got lengths {[int(shape[0]) for shape in lengths.values()]}."
            )
        if self.gradient_position_norms.shape[0] == 0:
            raise ValueError("Per-position gradient arrays must be non-empty.")

        indices = self.gradient_position_indices
        if indices.min() < 0:
            raise ValueError("gradient_position_indices must be non-negative.")
        if np.unique(indices).shape[0] != indices.shape[0]:
            raise ValueError(
                "gradient_position_indices must be distinct; each evaluation "
                "position is measured at most once."
            )
        for name in ("gradient_position_target_ids", "gradient_position_greedy_ids"):
            token_ids = getattr(self, name)
            if token_ids.min() < 0 or token_ids.max() >= vocab_size:
                raise ValueError(f"{name} contain values outside the vocabulary.")

        norms = self.gradient_position_norms
        if not np.all(np.isfinite(norms)):
            raise ValueError("gradient_position_norms must all be finite.")
        if np.any(norms < 0):
            raise ValueError("gradient_position_norms must be non-negative.")

        recorded = self.gradient_analysis.get("initialization_index")
        if recorded is not None and not 0 <= int(recorded) < num_inits:
            raise ValueError(
                f"gradient_analysis.initialization_index {recorded} is outside the "
                f"{num_inits} recorded initializations."
            )

    def _validate_gradient_sketches(self) -> None:
        """The sketch must describe exactly the gradient-evaluated positions."""

        if self.gradient_position_sketches is None:
            return
        if self.gradient_position_norms is None:
            raise ValueError(
                "gradient_position_sketches was given without the per-position "
                "gradient analysis it projects."
            )
        sketches = np.asarray(self.gradient_position_sketches)
        if sketches.ndim != 2:
            raise ValueError("gradient_position_sketches must be [positions, K].")
        if sketches.shape[0] != int(self.gradient_position_norms.shape[0]):
            raise ValueError(
                "gradient_position_sketches must have one row per gradient-"
                "evaluated position."
            )
        if not np.all(np.isfinite(sketches)):
            raise ValueError("gradient_position_sketches must be finite.")

    def _validate_predictive_probabilities(self, num_inits: int) -> None:
        """Check the optional raw-predictive-probability statistics.

        The invariants are the ones that make the ranked profile meaningful: it
        must be a genuine descending ranking of a probability vector, and its
        first rank must agree with the separately stored greedy-token
        probability. A profile that failed either would look plausible on a plot
        while describing something else.
        """

        present = {
            name: getattr(self, name)
            for name in _PROBABILITY_ARRAY_NAMES
            if getattr(self, name) is not None
        }
        if not present:
            return
        missing = sorted(set(_PROBABILITY_ARRAY_NAMES) - set(present))
        if missing:
            raise ValueError(
                "Predictive-probability arrays are all-or-nothing; missing: "
                + ", ".join(missing)
            )

        profile = self.predictive_ranked_probabilities
        if profile.ndim != 2 or profile.shape[0] != num_inits:
            raise ValueError(
                "predictive_ranked_probabilities must have shape "
                f"[initializations, eligible_vocab]; got {profile.shape}."
            )
        if profile.shape[1] != self.eligible_vocab_size:
            raise ValueError(
                f"predictive_ranked_probabilities spans {profile.shape[1]} ranks but the "
                f"eligible support has {self.eligible_vocab_size} tokens."
            )

        per_position = {
            name: array
            for name, array in present.items()
            if name != "predictive_ranked_probabilities"
        }
        num_positions = self.predictive_max_probabilities.shape[1]
        for name, array in per_position.items():
            if array.ndim != 2 or array.shape != (num_inits, num_positions):
                raise ValueError(
                    f"{name} must have shape [initializations, positions] "
                    f"({num_inits}, {num_positions}); got {array.shape}."
                )

        if not np.all(np.isfinite(profile)):
            raise ValueError("predictive_ranked_probabilities must all be finite.")
        if np.any(profile < 0.0):
            raise ValueError("predictive_ranked_probabilities must be non-negative.")
        # Ranked, so non-increasing by construction. A tiny negative tolerance
        # absorbs float64 noise without admitting a genuinely unsorted profile.
        if np.any(np.diff(profile, axis=1) > 1e-12):
            raise ValueError(
                "predictive_ranked_probabilities must be non-increasing with rank; "
                "the profile is ranked within each position before averaging."
            )
        totals = profile.sum(axis=1)
        if not np.allclose(totals, 1.0, rtol=0.0, atol=1e-6):
            raise ValueError(
                "Each ranked predictive profile must sum to 1 over the eligible "
                f"support; got totals in [{totals.min():.9f}, {totals.max():.9f}]."
            )

        for name in ("predictive_max_probabilities", "predictive_target_probabilities"):
            array = getattr(self, name)
            if np.any(array < 0.0) or np.any(array > 1.0):
                raise ValueError(f"{name} must lie in [0, 1].")
        rank_one = profile[:, 0]
        observed = self.predictive_max_probabilities.mean(axis=1)
        if not np.allclose(rank_one, observed, rtol=1e-6, atol=1e-9):
            raise ValueError(
                "Rank 1 of each ranked profile must equal the mean stored maximum "
                "probability for that initialization; the two disagree, so the "
                "profile and the per-position statistics describe different data."
            )

    def _validate_temperature_confidence(self, num_inits: int) -> None:
        """Check the optional temperature-conditioned confidence grid.

        The invariant worth the most here is the last one: the ``T = 1`` slice
        must equal the canonical arrays exactly. The grid duplicates that slice
        deliberately, and a duplicate that silently drifted would put two
        different numbers behind the same name.
        """

        present = {
            name: getattr(self, name)
            for name in _TEMPERATURE_ARRAY_NAMES
            if getattr(self, name) is not None
        }
        if not present:
            return
        missing = sorted(set(_TEMPERATURE_ARRAY_NAMES) - set(present))
        if missing:
            raise ValueError(
                "Temperature-confidence arrays are all-or-nothing; missing: "
                + ", ".join(missing)
            )

        temperatures = self.predictive_temperatures
        if temperatures.ndim != 1 or temperatures.shape[0] == 0:
            raise ValueError("predictive_temperatures must be a non-empty 1-D array.")
        if np.any(temperatures <= 0.0):
            raise ValueError(
                "Every diagnostic temperature must be positive; softmax(z/T) is "
                "undefined at T = 0, and greedy is already the T -> 0 limit."
            )
        if np.unique(temperatures).shape[0] != temperatures.shape[0]:
            raise ValueError("predictive_temperatures must be distinct.")

        count = temperatures.shape[0]
        profiles = self.predictive_temperature_ranked_probabilities
        if profiles.shape != (num_inits, count, self.eligible_vocab_size):
            raise ValueError(
                "predictive_temperature_ranked_probabilities must have shape "
                f"[initializations, temperatures, eligible_vocab]; got {profiles.shape}."
            )
        num_positions = self.predictive_temperature_max_probabilities.shape[2]
        for name in (
            "predictive_temperature_max_probabilities",
            "predictive_temperature_target_probabilities",
            "predictive_temperature_target_losses",
        ):
            array = getattr(self, name)
            if array.shape != (num_inits, count, num_positions):
                raise ValueError(
                    f"{name} must have shape [initializations, temperatures, positions] "
                    f"({num_inits}, {count}, {num_positions}); got {array.shape}."
                )
        if self.predictive_temperature_mean_entropy.shape != (num_inits, count):
            raise ValueError(
                "predictive_temperature_mean_entropy must have shape "
                f"[initializations, temperatures]; got "
                f"{self.predictive_temperature_mean_entropy.shape}."
            )

        if not np.all(np.isfinite(profiles)) or np.any(profiles < 0.0):
            raise ValueError(
                "predictive_temperature_ranked_probabilities must be finite and "
                "non-negative."
            )
        if np.any(np.diff(profiles, axis=2) > 1e-12):
            raise ValueError(
                "Every temperature's ranked profile must be non-increasing with rank."
            )
        totals = profiles.sum(axis=2)
        if not np.allclose(totals, 1.0, rtol=0.0, atol=1e-6):
            raise ValueError(
                "Every temperature's ranked profile must sum to 1 over the eligible "
                f"support; got totals in [{totals.min():.9f}, {totals.max():.9f}]."
            )
        for name in (
            "predictive_temperature_max_probabilities",
            "predictive_temperature_target_probabilities",
        ):
            array = getattr(self, name)
            if np.any(array < 0.0) or np.any(array > 1.0):
                raise ValueError(f"{name} must lie in [0, 1].")
        if not np.allclose(
            profiles[:, :, 0],
            self.predictive_temperature_max_probabilities.mean(axis=2),
            rtol=1e-6,
            atol=1e-9,
        ):
            raise ValueError(
                "Rank 1 of every temperature's ranked profile must equal that "
                "temperature's mean maximum probability."
            )

        if self.predictive_ranked_probabilities is None:
            return
        canonical = np.flatnonzero(temperatures == CANONICAL_TEMPERATURE)
        if canonical.size != 1:
            return
        index = int(canonical[0])
        for grid_name, canonical_name in (
            ("predictive_temperature_ranked_probabilities", "predictive_ranked_probabilities"),
            ("predictive_temperature_max_probabilities", "predictive_max_probabilities"),
            (
                "predictive_temperature_target_probabilities",
                "predictive_target_probabilities",
            ),
            ("predictive_temperature_target_losses", "predictive_target_losses"),
        ):
            if not np.array_equal(
                getattr(self, grid_name)[:, index], getattr(self, canonical_name)
            ):
                raise ValueError(
                    f"{grid_name} at T = 1 differs from {canonical_name}; the grid's "
                    "canonical slice and the canonical arrays must be identical."
                )

    def _validate_temperature_gradients(self) -> None:
        """Check the optional temperature-conditioned gradient norms.

        The invariant that matters most is the last: the canonical row must equal
        the version 5 array exactly. Temperature enters the loss here, so a drift
        between the two would mean the established observable and its own
        baseline slice disagree.
        """

        present = {
            name: getattr(self, name)
            for name in _GRADIENT_TEMPERATURE_ARRAY_NAMES
            if getattr(self, name) is not None
        }
        if not present:
            return
        missing = sorted(set(_GRADIENT_TEMPERATURE_ARRAY_NAMES) - set(present))
        if missing:
            raise ValueError(
                "Temperature-gradient arrays are all-or-nothing; missing: "
                + ", ".join(missing)
            )
        if self.gradient_position_norms is None:
            raise ValueError(
                "Temperature-conditioned gradient norms require the per-position "
                "gradient arrays they are aligned with."
            )

        temperatures = self.gradient_temperatures
        norms = self.gradient_temperature_position_norms
        if temperatures.ndim != 1 or temperatures.shape[0] == 0:
            raise ValueError("gradient_temperatures must be a non-empty 1-D array.")
        if np.any(temperatures <= 0.0):
            raise ValueError("Every gradient temperature must be positive.")
        if np.unique(temperatures).shape[0] != temperatures.shape[0]:
            raise ValueError("gradient_temperatures must be distinct.")
        expected = (temperatures.shape[0], self.gradient_position_norms.shape[0])
        if norms.shape != expected:
            raise ValueError(
                "gradient_temperature_position_norms must have shape "
                f"[temperatures, positions] {expected}; got {norms.shape}."
            )
        if not np.all(np.isfinite(norms)) or np.any(norms < 0.0):
            raise ValueError(
                "gradient_temperature_position_norms must be finite and non-negative."
            )

        canonical = np.flatnonzero(temperatures == CANONICAL_TEMPERATURE)
        if canonical.size != 1:
            raise ValueError(
                f"The canonical baseline T = {CANONICAL_TEMPERATURE} must appear "
                "exactly once in gradient_temperatures."
            )
        if not np.array_equal(norms[int(canonical[0])], self.gradient_position_norms):
            raise ValueError(
                "The canonical gradient temperature row differs from "
                "gradient_position_norms; the established T = 1 observable and its "
                "own baseline slice must be identical."
            )

    def _validate_mean_token_probabilities(self, num_inits: int, vocab_size: int) -> None:
        """Check the identity-preserving mean token probabilities.

        The invariant that earns its keep is the last one: the ``T = 1`` row must
        agree with ``mean_predicted_probabilities``, which has always held the
        same quantity. Generalizing that field rather than adding a second
        implementation is only safe if the two provably coincide.
        """

        values = self.predictive_temperature_mean_token_probabilities
        if values is None:
            return
        if self.predictive_temperatures is None:
            raise ValueError(
                "Mean token probabilities require the temperature grid they are "
                "indexed by."
            )
        count = self.predictive_temperatures.shape[0]
        if values.shape != (num_inits, count, vocab_size):
            raise ValueError(
                "predictive_temperature_mean_token_probabilities must have shape "
                f"[initializations, temperatures, vocab] "
                f"({num_inits}, {count}, {vocab_size}); got {values.shape}."
            )
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError(
                "predictive_temperature_mean_token_probabilities must be finite "
                "and non-negative."
            )
        totals = values.sum(axis=2)
        if not np.allclose(totals, 1.0, rtol=0.0, atol=1e-6):
            raise ValueError(
                "Each mean token-probability vector must sum to 1; got totals in "
                f"[{totals.min():.9f}, {totals.max():.9f}]."
            )

        canonical = np.flatnonzero(self.predictive_temperatures == CANONICAL_TEMPERATURE)
        if canonical.size != 1:
            return
        index = int(canonical[0])
        if not np.allclose(
            values[:, index].astype(np.float32),
            self.mean_predicted_probabilities,
            rtol=1e-6,
            atol=1e-9,
        ):
            raise ValueError(
                "The T = 1 mean token probabilities differ from "
                "mean_predicted_probabilities; the grid generalizes that field "
                "and the two must describe the same quantity."
            )

    def save(self, directory: str | Path, *, name: str = "initialization_distribution") -> Path:
        """Write the record as ``<name>.npz`` plus ``<name>.json``.

        Returns:
            The directory the two files were written to.
        """

        directory = Path(directory)
        arrays = {
            name: getattr(self, name) for name in _ARRAY_NAMES + _OPTIONAL_ARRAY_NAMES
        }
        for condition, values in self.condition_greedy_counts.items():
            arrays[f"{_CONDITION_GREEDY_PREFIX}{condition}"] = values
        for condition, values in self.condition_nucleus_counts.items():
            arrays[f"{_CONDITION_NUCLEUS_PREFIX}{condition}"] = values
        for key, values in self.uniform_null.items():
            arrays[f"{_NULL_PREFIX}{key}"] = values
        for condition, values in self.sweep_counts_by_condition.items():
            arrays[f"{_SWEEP_COUNTS_PREFIX}{condition}"] = values
        for condition, values in self.sweep_agreement_by_condition.items():
            arrays[f"{_SWEEP_AGREEMENT_PREFIX}{condition}"] = values
        for optional_name in (
            _GRADIENT_ARRAY_NAMES
            + _PROBABILITY_ARRAY_NAMES
            + _TEMPERATURE_ARRAY_NAMES
            + _GRADIENT_TEMPERATURE_ARRAY_NAMES
            + (_MEAN_TOKEN_ARRAY_NAME, _GRADIENT_SKETCH_ARRAY_NAME)
        ):
            values = getattr(self, optional_name)
            if values is not None:
                arrays[optional_name] = values
        _atomic_write_bytes(
            directory / f"{name}.npz",
            lambda path: np.savez_compressed(path, **arrays),
        )
        payload = dict(self.metadata)
        payload["record_version"] = RECORD_VERSION
        _atomic_write_bytes(
            directory / f"{name}.json",
            lambda path: path.write_text(
                json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            ),
        )
        return directory

    @classmethod
    def from_parts(
        cls,
        arrays: Mapping[str, np.ndarray],
        metadata: Mapping[str, Any],
    ) -> "InitializationExperimentRecord":
        """Rebuild a record from a loaded array mapping and metadata."""

        missing = sorted(set(_ARRAY_NAMES) - set(arrays))
        if missing:
            raise ValueError("Record is missing array(s): " + ", ".join(missing))
        loaded = {name: np.asarray(arrays[name]) for name in _ARRAY_NAMES}
        if "eligible_token_ids" in arrays:
            eligible = np.asarray(arrays["eligible_token_ids"])
        else:
            # Version 1 predates special-token exclusion: every token was
            # eligible, so defaulting preserves the original meaning exactly.
            eligible = np.arange(loaded["token_ids"].shape[0])
        def collect(prefix: str) -> dict[str, np.ndarray]:
            return {
                key[len(prefix) :]: np.asarray(value)
                for key, value in arrays.items()
                if key.startswith(prefix)
            }

        optional = {
            name: np.asarray(arrays[name])
            for name in (
                _GRADIENT_ARRAY_NAMES
                + _PROBABILITY_ARRAY_NAMES
                + _TEMPERATURE_ARRAY_NAMES
                + _GRADIENT_TEMPERATURE_ARRAY_NAMES
                + (_MEAN_TOKEN_ARRAY_NAME, _GRADIENT_SKETCH_ARRAY_NAME)
            )
            if name in arrays
        }

        return cls(
            **loaded,
            eligible_token_ids=eligible,
            metadata=dict(metadata),
            condition_greedy_counts=collect(_CONDITION_GREEDY_PREFIX),
            condition_nucleus_counts=collect(_CONDITION_NUCLEUS_PREFIX),
            uniform_null=collect(_NULL_PREFIX),
            sweep_counts_by_condition=collect(_SWEEP_COUNTS_PREFIX),
            sweep_agreement_by_condition=collect(_SWEEP_AGREEMENT_PREFIX),
            **optional,
        )

    @classmethod
    def build(
        cls,
        *,
        corpus_counts: Sequence[float] | np.ndarray,
        selected_target_counts: Sequence[float] | np.ndarray,
        greedy_counts: Sequence[Sequence[float]] | np.ndarray,
        nucleus_counts: np.ndarray,
        mean_predicted_probabilities: np.ndarray,
        model_seeds: Sequence[int] | np.ndarray,
        metadata: Mapping[str, Any],
        eligible_token_ids: Sequence[int] | np.ndarray | None = None,
        condition_greedy_counts: Mapping[str, np.ndarray] | None = None,
        condition_nucleus_counts: Mapping[str, np.ndarray] | None = None,
        uniform_null: Mapping[str, np.ndarray] | None = None,
        sweep_counts_by_condition: Mapping[str, np.ndarray] | None = None,
        sweep_agreement_by_condition: Mapping[str, np.ndarray] | None = None,
        gradient_position_indices: Sequence[int] | np.ndarray | None = None,
        gradient_position_target_ids: Sequence[int] | np.ndarray | None = None,
        gradient_position_greedy_ids: Sequence[int] | np.ndarray | None = None,
        gradient_position_norms: Sequence[float] | np.ndarray | None = None,
        predictive_ranked_probabilities: np.ndarray | None = None,
        predictive_max_probabilities: np.ndarray | None = None,
        predictive_target_probabilities: np.ndarray | None = None,
        predictive_target_losses: np.ndarray | None = None,
        predictive_temperatures: np.ndarray | None = None,
        predictive_temperature_ranked_probabilities: np.ndarray | None = None,
        predictive_temperature_max_probabilities: np.ndarray | None = None,
        predictive_temperature_target_probabilities: np.ndarray | None = None,
        predictive_temperature_target_losses: np.ndarray | None = None,
        predictive_temperature_mean_entropy: np.ndarray | None = None,
        gradient_temperatures: np.ndarray | None = None,
        gradient_temperature_position_norms: np.ndarray | None = None,
        predictive_temperature_mean_token_probabilities: np.ndarray | None = None,
        gradient_position_sketches: np.ndarray | None = None,
    ) -> "InitializationExperimentRecord":
        """Assemble a record, deriving the canonical ``token_ids`` axis.

        ``eligible_token_ids`` defaults to the whole vocabulary, which is the
        correct answer for a character tokenizer and for any vocabulary without
        structural tokens.
        """

        corpus = np.asarray(corpus_counts)
        eligible = (
            np.arange(corpus.shape[0])
            if eligible_token_ids is None
            else np.asarray(eligible_token_ids)
        )
        return cls(
            token_ids=np.arange(corpus.shape[0]),
            corpus_counts=corpus,
            selected_target_counts=np.asarray(selected_target_counts),
            greedy_counts=np.asarray(greedy_counts),
            nucleus_counts=np.asarray(nucleus_counts),
            mean_predicted_probabilities=np.asarray(mean_predicted_probabilities),
            model_seeds=np.asarray(model_seeds),
            eligible_token_ids=eligible,
            metadata=dict(metadata),
            condition_greedy_counts={
                key: np.asarray(value) for key, value in (condition_greedy_counts or {}).items()
            },
            condition_nucleus_counts={
                key: np.asarray(value) for key, value in (condition_nucleus_counts or {}).items()
            },
            uniform_null={key: np.asarray(value) for key, value in (uniform_null or {}).items()},
            sweep_counts_by_condition={
                key: np.asarray(value) for key, value in (sweep_counts_by_condition or {}).items()
            },
            sweep_agreement_by_condition={
                key: np.asarray(value)
                for key, value in (sweep_agreement_by_condition or {}).items()
            },
            gradient_position_indices=_optional_array(gradient_position_indices, np.int64),
            gradient_position_target_ids=_optional_array(
                gradient_position_target_ids, np.int64
            ),
            gradient_position_greedy_ids=_optional_array(
                gradient_position_greedy_ids, np.int64
            ),
            gradient_position_norms=_optional_array(gradient_position_norms, np.float64),
            predictive_ranked_probabilities=_optional_array(
                predictive_ranked_probabilities, np.float64
            ),
            predictive_max_probabilities=_optional_array(
                predictive_max_probabilities, np.float64
            ),
            predictive_target_probabilities=_optional_array(
                predictive_target_probabilities, np.float64
            ),
            predictive_target_losses=_optional_array(predictive_target_losses, np.float64),
            predictive_temperatures=_optional_array(predictive_temperatures, np.float64),
            predictive_temperature_ranked_probabilities=_optional_array(
                predictive_temperature_ranked_probabilities, np.float64
            ),
            predictive_temperature_max_probabilities=_optional_array(
                predictive_temperature_max_probabilities, np.float64
            ),
            predictive_temperature_target_probabilities=_optional_array(
                predictive_temperature_target_probabilities, np.float64
            ),
            predictive_temperature_target_losses=_optional_array(
                predictive_temperature_target_losses, np.float64
            ),
            predictive_temperature_mean_entropy=_optional_array(
                predictive_temperature_mean_entropy, np.float64
            ),
            gradient_temperatures=_optional_array(gradient_temperatures, np.float64),
            gradient_temperature_position_norms=_optional_array(
                gradient_temperature_position_norms, np.float64
            ),
            gradient_position_sketches=_optional_array(
                gradient_position_sketches, np.float32
            ),
            predictive_temperature_mean_token_probabilities=_optional_array(
                predictive_temperature_mean_token_probabilities, np.float64
            ),
        )


def load_record(
    directory: str | Path,
    *,
    name: str = "initialization_distribution",
) -> InitializationExperimentRecord:
    """Load a record previously written by :meth:`InitializationExperimentRecord.save`."""

    directory = Path(directory)
    arrays_path = directory / f"{name}.npz"
    metadata_path = directory / f"{name}.json"
    if not arrays_path.is_file():
        raise FileNotFoundError(f"Record arrays are missing: {arrays_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Record metadata is missing: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    return InitializationExperimentRecord.from_parts(arrays, metadata)
