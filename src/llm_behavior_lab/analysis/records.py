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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "InitializationExperimentRecord",
    "RECORD_VERSION",
    "load_record",
]

#: Bumped only when the array set or its meaning changes incompatibly.
RECORD_VERSION = 1

_ARRAY_NAMES = (
    "token_ids",
    "corpus_counts",
    "selected_target_counts",
    "greedy_counts",
    "nucleus_counts",
    "mean_predicted_probabilities",
    "model_seeds",
)


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
    ==============================  =========  ====================================
    """

    token_ids: np.ndarray
    corpus_counts: np.ndarray
    selected_target_counts: np.ndarray
    greedy_counts: np.ndarray
    nucleus_counts: np.ndarray
    mean_predicted_probabilities: np.ndarray
    model_seeds: np.ndarray
    metadata: dict[str, Any]

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

        tokens = self.metadata.get("tokens")
        if tokens is not None and len(tokens) != vocab_size:
            raise ValueError("metadata['tokens'] must have one entry per valid token ID.")

    def save(self, directory: str | Path, *, name: str = "initialization_distribution") -> Path:
        """Write the record as ``<name>.npz`` plus ``<name>.json``.

        Returns:
            The directory the two files were written to.
        """

        directory = Path(directory)
        arrays = {field: getattr(self, field) for field in _ARRAY_NAMES}
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
        return cls(
            **{name: np.asarray(arrays[name]) for name in _ARRAY_NAMES},
            metadata=dict(metadata),
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
    ) -> "InitializationExperimentRecord":
        """Assemble a record, deriving the canonical ``token_ids`` axis."""

        corpus = np.asarray(corpus_counts)
        return cls(
            token_ids=np.arange(corpus.shape[0]),
            corpus_counts=corpus,
            selected_target_counts=np.asarray(selected_target_counts),
            greedy_counts=np.asarray(greedy_counts),
            nucleus_counts=np.asarray(nucleus_counts),
            mean_predicted_probabilities=np.asarray(mean_predicted_probabilities),
            model_seeds=np.asarray(model_seeds),
            metadata=dict(metadata),
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
