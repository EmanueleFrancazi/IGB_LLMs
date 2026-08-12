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
#: input-condition counts and the uniform-null profile. Older records load
#: unchanged: version 1 predates special-token exclusion, so every token was
#: eligible -- exactly the default applied when that array is absent -- and the
#: version 3 additions are optional, so their absence simply means the run did
#: not carry those comparisons.
RECORD_VERSION = 3

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
        for key, array in self.uniform_null.items():
            if array.ndim != 1:
                raise ValueError(f"uniform_null[{key!r}] must be one-dimensional.")

        tokens = self.metadata.get("tokens")
        if tokens is not None and len(tokens) != vocab_size:
            raise ValueError("metadata['tokens'] must have one entry per valid token ID.")

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

        return cls(
            **loaded,
            eligible_token_ids=eligible,
            metadata=dict(metadata),
            condition_greedy_counts=collect(_CONDITION_GREEDY_PREFIX),
            condition_nucleus_counts=collect(_CONDITION_NUCLEUS_PREFIX),
            uniform_null=collect(_NULL_PREFIX),
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
