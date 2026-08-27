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
RECORD_VERSION = 12

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

#: Tolerance for matching a requested loss temperature against a measured one.
#:
#: Owned here, in the layer with no intra-package imports, because the record
#: accessor needs it and :mod:`llm_behavior_lab.analysis.directional_fields`
#: already imports :data:`CANONICAL_TEMPERATURE` from this module -- resolving it
#: the other way round would be a cycle. ``directional_fields`` re-exports this
#: name, so its existing importers are unaffected.
#:
#: Temperatures originate as module constants or as CLI-parsed floats of the same
#: literals, so exact equality usually holds; the tolerance covers a value that
#: has passed through float32 somewhere, which moves 0.12 by about 1.5e-9. The
#: grid steps by 0.12, five orders of magnitude above this, so no two
#: temperatures anyone would request can alias onto each other.
TEMPERATURE_MATCH_TOLERANCE = 1e-6

#: Sketch-protocol schema this record layer understands.
#:
#: Deliberately separate from :data:`RECORD_VERSION`. The record schema and the
#: sketch protocol can move independently, and conflating them would force a
#: record-version bump for a protocol-only change or hide a protocol change
#: behind an unrelated one. A *missing* schema version means the legacy protocol
#: written before replicas existed, which is v1 by definition and can only ever
#: describe a single map.
SKETCH_PROTOCOL_SCHEMA_VERSION = 2

#: Required keys and their exact permitted values under schema v2. Exact values,
#: not free text: these describe how the arrays beside them must be *read*, so a
#: reader that finds an unexpected value is looking at something it does not
#: know how to interpret and must say so rather than guess.
_SKETCH_PROTOCOL_V2_EXACT = {
    "canonical_relationship": "slice_of_temperature_array",
    "estimator": "mean_of_per_map_inner_products_over_exact_norms",
    "accumulation_dtype": "float64",
    "storage_dtype": "float32",
}

#: Axis names of ``gradient_temperature_position_sketches`` at each map count.
_SKETCH_AXES_SINGLE_MAP = ["temperature", "position", "bucket"]
_SKETCH_AXES_MULTI_MAP = ["temperature", "position", "map", "bucket"]


def _integral_map_count(value: Any, *, where: str) -> int:
    """Return ``value`` as a positive ``int`` map count, or explain why not.

    ``bool`` is refused explicitly: ``True == 1`` in Python, so it would
    otherwise pass as a single map and read as though somebody had asked a
    yes/no question and been handed a count.

    This deliberately restates the rule that
    ``evaluation.position_gradients._validated_map_count`` applies at
    measurement time rather than importing it. The analysis layer is
    **NumPy-only** by design -- a finished experiment must be re-readable
    without PyTorch installed -- and importing the evaluation module here would
    pull torch into every record load. The duplication is three lines across a
    deliberate architectural boundary; the tests assert both accept and reject
    the same values.
    """

    try:
        count = int(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{where} must be an integer of at least one; got {value!r}."
        ) from None
    if isinstance(value, bool) or count != value or count < 1:
        raise ValueError(
            f"{where} must be an integer of at least one; got {value!r}."
        )
    return count


def _require_supported_record_version(version: Any) -> None:
    """Refuse a record written by a newer version of this project.

    ``record_version`` has been written since version 2 and, until now, never
    read -- so a forward-incompatible archive would previously have loaded with
    its unknown arrays quietly discarded. A missing value stays supported: it
    means a record from before the field existed.

    Raises:
        ValueError: If the version is malformed, or newer than this reader.
    """

    if version is None:
        return
    # An integral type, not merely something that survives int(). A float, even
    # an integral one, means the metadata was built by something that did not
    # treat this as a version number, and `bool` is Integral in Python -- True
    # would otherwise read as version 1.
    if isinstance(version, bool) or not isinstance(version, (int, np.integer)):
        raise ValueError(
            f"record_version must be an integer; got {version!r}."
        )
    value = int(version)
    if value < 1:
        # Version 1 is the oldest this format ever had -- it predates
        # `eligible_token_ids`. There is no version 0, so a non-positive value
        # is malformed metadata rather than an ancient record.
        raise ValueError(
            f"record_version must be positive; got {value}. Version 1 is the "
            "oldest this format has ever used, so there is no record this could "
            "legitimately describe."
        )
    if value > RECORD_VERSION:
        raise ValueError(
            f"This record declares record_version {value}, but this version of "
            f"the project reads up to {RECORD_VERSION}. It was written by a "
            "newer writer whose arrays may be laid out differently, so it "
            "cannot be read here. Upgrade rather than loading it partially."
        )


def _resolve_sketch_map_count(
    *,
    protocol: Mapping[str, Any],
    record_version: Any,
    has_canonical_array: bool,
    temperature_rank: int | None,
) -> int | None:
    """The number of production CountSketch maps this record's sketches carry.

    **The single implementation of the map-count rules.** Both
    :meth:`InitializationExperimentRecord.validate` and
    :attr:`InitializationExperimentRecord.sketch_map_count` route through it, so
    the two cannot drift into disagreeing about which records are legal.

    Metadata is authoritative throughout. The array rank is *validated against*
    the declared count and never used to discover it: a rank is evidence about
    storage, not a statement about what was measured, and a record that has to be
    guessed at is a record that can be guessed wrong.

    Args:
        protocol: The ``gradient_sketch`` protocol block; empty when absent.
        record_version: ``record_version`` from the metadata, or ``None``.
        has_canonical_array: Whether ``gradient_position_sketches`` is present.
        temperature_rank: ``ndim`` of ``gradient_temperature_position_sketches``,
            or ``None`` when that array is absent.

    Returns:
        The map count, or ``None`` when the record carries no sketch surface at
        all -- which is a legitimate record, not an error.

    Raises:
        ValueError: For any combination the truth table refuses.
    """

    schema = protocol.get("schema_version")
    declared = protocol.get("map_count")
    has_sketches = has_canonical_array or temperature_rank is not None

    if schema is not None:
        try:
            schema_value = int(schema)
            unsupported = isinstance(schema, bool) or schema_value != schema
        except (TypeError, ValueError):
            schema_value, unsupported = None, True
        if unsupported or schema_value != SKETCH_PROTOCOL_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported sketch-protocol schema_version {schema!r}. This "
                f"reader understands version {SKETCH_PROTOCOL_SCHEMA_VERSION}; a "
                "higher one was written by a newer version of this project and "
                "may lay its arrays out differently, so it cannot be read here."
            )

    if not has_sketches:
        # A record may legitimately carry no sketches at all -- gradient analysis
        # off, or norms without direction. That is not an error; it simply has no
        # map surface, and `sketch_map_count` raises when asked.
        return None

    if temperature_rank == 4:
        # The multi-map layout. All three conditions, together: the rank alone
        # must never be enough to establish it.
        if record_version != RECORD_VERSION:
            raise ValueError(
                "A four-dimensional gradient_temperature_position_sketches array "
                f"requires record_version {RECORD_VERSION}; got "
                f"{record_version!r}."
            )
        if schema != SKETCH_PROTOCOL_SCHEMA_VERSION:
            raise ValueError(
                "A four-dimensional gradient_temperature_position_sketches array "
                f"requires sketch-protocol schema_version "
                f"{SKETCH_PROTOCOL_SCHEMA_VERSION}; got {schema!r}."
            )
        if declared is None:
            raise ValueError(
                "A multi-map sketch record must declare map_count explicitly; the "
                "array's map axis is validated against it, never used to infer it."
            )
        count = _integral_map_count(declared, where="sketch_protocol.map_count")
        if count == 1:
            raise ValueError(
                "map_count is 1 but the temperature sketch array is "
                "four-dimensional. A single-map record stores "
                "[temperatures, positions, K]."
            )
    elif schema == SKETCH_PROTOCOL_SCHEMA_VERSION:
        if record_version != RECORD_VERSION:
            raise ValueError(
                f"A schema-v{SKETCH_PROTOCOL_SCHEMA_VERSION} sketch protocol "
                f"requires record_version {RECORD_VERSION}; got {record_version!r}."
            )
        if declared is None:
            raise ValueError(
                f"A schema-v{SKETCH_PROTOCOL_SCHEMA_VERSION} sketch protocol with "
                "sketch arrays must declare map_count explicitly. Defaulting is "
                "reserved for genuinely legacy protocols, so a current record "
                "cannot masquerade as one."
            )
        count = _integral_map_count(declared, where="sketch_protocol.map_count")
        if count != 1:
            raise ValueError(
                f"map_count is {count} but the stored sketches are not in the "
                "multi-map layout; a record above one map stores a "
                "four-dimensional temperature array."
            )
    else:
        # Legacy protocol: no schema version. It predates replicas and can only
        # ever describe a single map.
        #
        # **The legacy-wrapper rule**, and it is deliberate.
        #
        # This does *not* additionally require record_version < 12, and must not.
        # `save()` stamps the current RECORD_VERSION onto whatever it writes, so
        # loading an old record and re-saving it -- an ordinary thing to do --
        # produces a **v12 container around a schema-version-absent legacy
        # protocol**. Requiring schema v2 of every v12 record would make that
        # archive unloadable, which is a backward-compatibility break, not a
        # safety property. An earlier draft did exactly that and broke
        # `test_a_record_carrying_sketches_round_trips`.
        #
        # What is still guaranteed, and where:
        #   * a *newly measured* v12 record always emits schema 2, because the
        #     measurement writes it unconditionally;
        #   * a schema-absent protocol is confined to the historical 2-D/3-D
        #     M = 1 layouts, which cannot express more than one map at all;
        #   * every 4-D layout demands record 12 *and* schema 2 *and* an explicit
        #     map_count > 1, checked above;
        #   * schema v2 itself never defaults the count.
        # So the masquerade this used to guard against -- a current writer
        # quietly omitting the count -- is prevented where it actually bites.
        if declared is None:
            count = 1
        else:
            count = _integral_map_count(declared, where="sketch_protocol.map_count")
            if count != 1:
                raise ValueError(
                    f"map_count is {count} without a sketch-protocol "
                    "schema_version. A legacy protocol predates replicas and "
                    "cannot describe more than one map."
                )

    if count > 1 and has_canonical_array:
        raise ValueError(
            "gradient_position_sketches must be absent when map_count is above "
            "one. The canonical field is derived from the canonical row of the "
            "temperature array so there is exactly one canonical representation; "
            "storing a second one re-admits the possibility that they disagree."
        )
    return count


def _validate_sketch_protocol_v2(protocol: Mapping[str, Any], map_count: int) -> None:
    """Check the schema-v2 required key set, types and exact values."""

    for key, expected in _SKETCH_PROTOCOL_V2_EXACT.items():
        if key not in protocol:
            raise ValueError(f"sketch_protocol.{key} is required under schema v2.")
        if protocol[key] != expected:
            raise ValueError(
                f"sketch_protocol.{key} must be {expected!r} under schema v2; got "
                f"{protocol[key]!r}."
            )

    storage = protocol.get("canonical_storage")
    expected_storage = "stored" if map_count == 1 else "derived"
    if storage != expected_storage:
        raise ValueError(
            f"sketch_protocol.canonical_storage must be {expected_storage!r} at "
            f"map_count {map_count}; got {storage!r}."
        )

    axes = protocol.get("temperature_sketch_axes")
    expected_axes = (
        _SKETCH_AXES_SINGLE_MAP if map_count == 1 else _SKETCH_AXES_MULTI_MAP
    )
    # Type first: `list(...)` on an int raises TypeError, and a schema violation
    # should surface as the ValueError every other check here raises.
    if not isinstance(axes, (list, tuple)):
        raise ValueError(
            "sketch_protocol.temperature_sketch_axes must be a list of axis "
            f"names; got {axes!r}."
        )
    if list(axes) != expected_axes:
        raise ValueError(
            "sketch_protocol.temperature_sketch_axes must be "
            f"{expected_axes} at map_count {map_count}; got {axes!r}."
        )

    if "recomputed_per_map" not in protocol:
        raise ValueError(
            "sketch_protocol.recomputed_per_map is required under schema v2."
        )
    if protocol["recomputed_per_map"] is not False:
        raise ValueError(
            "sketch_protocol.recomputed_per_map must be False: every map projects "
            "the same gradient from one backward pass, and a record claiming "
            "otherwise describes a different measurement."
        )

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

#: Version 11 addition: ``[N_T, D_g, K]`` count sketch of every evaluated
#: position's gradient at every measured loss temperature.
#:
#: The norms already spanned this axis, but a norm is a scalar and a direction
#: is not recoverable from one, so a directional analysis at ``T_g != 1`` needs
#: its own projection. Each row comes from the same backward pass as the
#: matching row of ``gradient_temperature_position_norms``.
#:
#: About 470 MB at ``N_T = 7, D_g = 32768, K = 512`` in float32 -- much the
#: largest array in the record, and the reason it stays optional: a run needing
#: only the canonical direction should not pay for it.
#:
#: Its canonical ``T = 1`` slice is the same measured field as
#: ``gradient_position_sketches``, and validation asserts they are equal, just
#: as it already does for the norms.
_GRADIENT_TEMPERATURE_SKETCH_ARRAY_NAME = "gradient_temperature_position_sketches"


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
    #: ``[N_T, D_g, K]`` count sketch of each position's gradient at each
    #: temperature. Its canonical row equals ``gradient_position_sketches``.
    gradient_temperature_position_sketches: np.ndarray | None = None
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
    def has_temperature_gradient_sketches(self) -> bool:
        """Whether directional information exists beyond the canonical ``T``.

        A record without this carries direction at ``T = 1`` only. It is still a
        complete record; it simply cannot answer a question about ``T_g != 1``,
        and has to say so rather than substituting the canonical field.
        """

        return (
            self.gradient_temperature_position_sketches is not None
            and self.gradient_temperatures is not None
            and self.gradient_position_norms is not None
        )

    @property
    def has_gradient_position_sketches(self) -> bool:
        """Whether a canonical per-position gradient sketch is **available**.

        Directional analysis needs both the sketches and the exact norms that
        scale them, so both are required here rather than the array alone.

        Availability, not physical presence. A multi-map record deliberately
        stores no canonical array -- the canonical field is the canonical row of
        the temperature array -- and reporting ``False`` for one would silently
        switch off figures 20 and 24 and both artifact writers on exactly the
        records the replica work exists to produce.
        """

        if self.gradient_position_norms is None:
            return False
        if self.gradient_position_sketches is not None:
            return True
        # Derivable: multi-map storage with the canonical temperature measured.
        temperature = self.gradient_temperature_position_sketches
        if temperature is None or temperature.ndim != 4:
            return False
        if self.gradient_temperatures is None:
            return False
        try:
            self._canonical_sketch_temperature_index()
        except ValueError:
            return False
        return True

    def _canonical_sketch_temperature_index(self) -> int:
        """Row of the temperature axis holding the canonical ``T = 1``."""

        return self._sketch_temperature_index(None)

    def _sketch_temperature_index(self, loss_temperature: float | None) -> int:
        """Resolve a loss temperature to a row of the temperature sketch axis.

        Owned here rather than delegated to
        :mod:`llm_behavior_lab.analysis.directional_fields`, which imports from
        this module: the reverse direction would be a cycle. The matching rule
        and its error behaviour are the established ones -- absolute tolerance
        :data:`TEMPERATURE_MATCH_TOLERANCE`, no interpolation, no fallback to the
        canonical field, and an ambiguous match refused rather than resolved.
        """

        requested = (
            float(CANONICAL_TEMPERATURE)
            if loss_temperature is None
            else float(loss_temperature)
        )
        if self.gradient_temperatures is None:
            # Canonical-only record: the canonical field is all there is.
            if abs(requested - CANONICAL_TEMPERATURE) <= TEMPERATURE_MATCH_TOLERANCE:
                return 0
            raise ValueError(
                f"No measured gradient-direction field exists for loss "
                f"temperature T_g = {requested:g}. This record measured the "
                f"canonical T = {CANONICAL_TEMPERATURE:g} only. Directions at one "
                "temperature say nothing about another, so this cannot fall back "
                "or interpolate."
            )
        measured = np.asarray(self.gradient_temperatures, dtype=np.float64)
        close = np.flatnonzero(
            np.abs(measured - requested) <= TEMPERATURE_MATCH_TOLERANCE
        )
        if close.size == 0:
            listed = ", ".join(f"{value:g}" for value in measured)
            raise ValueError(
                f"No measured gradient-direction field exists for loss "
                f"temperature T_g = {requested:g}. Measured: {listed}."
            )
        if close.size > 1:
            raise ValueError(
                f"Loss temperature T_g = {requested:g} matches {close.size} "
                "measured temperatures, which cannot be resolved unambiguously."
            )
        return int(close[0])

    def per_map_sketches(self, loss_temperature: float | None = None) -> np.ndarray:
        """``[D, M, K]`` gradient sketches at one loss temperature.

        One shape at every map count, so a consumer never branches on storage.
        ``M = 1`` records -- legacy and current alike -- present their historical
        two-dimensional array as ``[D, 1, K]``; multi-map records return the
        chosen temperature row directly.

        Every return is a **view**: a length-one map axis is added by reshaping,
        never by copying, so asking for the uniform shape costs nothing. The
        arrays are large enough at campaign scale that a defensive copy here
        would be a real cost paid on every call.

        Args:
            loss_temperature: Which measured loss temperature to read. ``None``
                means the canonical ``T = 1``.

        Returns:
            ``[D, M, K]``.

        Raises:
            ValueError: If the record carries no sketches, or never measured a
                direction at this temperature.
        """

        count = self._resolved_map_count()
        if count is None or self.gradient_position_norms is None:
            raise ValueError(
                "This record carries no per-position gradient sketches, so no "
                "directional field can be read from it."
            )
        index = self._sketch_temperature_index(loss_temperature)

        if count > 1:
            return np.asarray(self.gradient_temperature_position_sketches)[index]

        canonical_index = self._canonical_sketch_temperature_index()
        if index == canonical_index and self.gradient_position_sketches is not None:
            # The historical canonical array, reshaped rather than copied.
            sketches = np.asarray(self.gradient_position_sketches)
        else:
            sketches = np.asarray(self.gradient_temperature_position_sketches)[index]
        return sketches.reshape(sketches.shape[0], 1, sketches.shape[1])

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

    @property
    def _sketch_protocol(self) -> dict[str, Any]:
        """The ``gradient_sketch`` protocol block, empty when absent."""

        return dict(self.gradient_analysis.get("gradient_sketch", {}) or {})

    def _resolved_map_count(self) -> int | None:
        """This record's map count via the shared resolver, or ``None``."""

        temperature = self.gradient_temperature_position_sketches
        return _resolve_sketch_map_count(
            protocol=self._sketch_protocol,
            record_version=self.metadata.get("record_version"),
            has_canonical_array=self.gradient_position_sketches is not None,
            temperature_rank=None if temperature is None else int(temperature.ndim),
        )

    @property
    def sketch_map_count(self) -> int:
        """Number of independent production CountSketch maps, ``M``.

        Read from metadata through the shared resolver -- never inferred from an
        array's rank or width.

        Raises:
            ValueError: If this record carries no sketch surface. Returning 1
                would let a sketch-free record answer a question about maps it
                never had.
        """

        count = self._resolved_map_count()
        if count is None:
            raise ValueError(
                "This record carries no gradient sketches, so it has no map "
                "count. Check has_gradient_position_sketches first."
            )
        return count

    def _validate_gradient_sketches(self) -> None:
        """The sketch must describe exactly the gradient-evaluated positions."""

        # Resolving here validates the whole metadata/array combination once,
        # for every record, whether or not any sketch array is present.
        map_count = self._resolved_map_count()
        protocol = self._sketch_protocol
        if (
            map_count is not None
            and protocol.get("schema_version") == SKETCH_PROTOCOL_SCHEMA_VERSION
        ):
            _validate_sketch_protocol_v2(protocol, map_count)

        if self.gradient_position_sketches is None:
            return
        if map_count is not None and map_count > 1:
            # Unreachable through the resolver, which already refuses this
            # combination; kept as a local invariant so the canonical-storage
            # rule is stated where the canonical array is validated.
            raise ValueError(
                "gradient_position_sketches must be absent when map_count is "
                "above one."
            )
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
        self._validate_temperature_gradient_sketches(int(canonical[0]))

    def _validate_temperature_gradient_sketches(self, canonical_index: int) -> None:
        """Check the temperature-resolved directional field, if present.

        The invariant worth stating is the last one, and it is the same one the
        norms already carry: the canonical slice must *be* the canonical field,
        not merely resemble it. Two independently measured ``T = 1`` gradient
        fields that happen to agree would be a coincidence to re-establish on
        every run; one measured field read twice cannot disagree.
        """

        sketches = self.gradient_temperature_position_sketches
        if sketches is None:
            return
        map_count = self._resolved_map_count() or 1
        temperatures = self.gradient_temperatures

        if map_count > 1:
            # Multi-map: temperature array only, and the canonical field is the
            # canonical row of it rather than a second stored copy. There is no
            # slice-equality check because there is nothing to compare against --
            # which is the point: one representation cannot disagree with itself.
            if sketches.ndim != 4:
                raise ValueError(
                    "gradient_temperature_position_sketches must be "
                    "[temperatures, positions, maps, K] when map_count is above "
                    f"one; got {sketches.ndim} dimensions."
                )
            expected_head = (
                temperatures.shape[0],
                int(self.gradient_position_norms.shape[0]),
            )
            if sketches.shape[:2] != expected_head:
                raise ValueError(
                    "gradient_temperature_position_sketches must have shape "
                    f"[temperatures, positions, maps, K] with leading "
                    f"{expected_head}; got {sketches.shape}."
                )
            if int(sketches.shape[2]) != map_count:
                raise ValueError(
                    f"The map axis has length {int(sketches.shape[2])} but "
                    f"sketch_protocol.map_count declares {map_count}. The count "
                    "is authoritative and the axis is checked against it."
                )
            if not np.all(np.isfinite(sketches)):
                raise ValueError(
                    "gradient_temperature_position_sketches must be finite."
                )
            return

        if self.gradient_position_sketches is None:
            raise ValueError(
                "gradient_temperature_position_sketches was given without "
                "gradient_position_sketches, so its canonical slice has nothing "
                "to be checked against."
            )
        expected = (
            temperatures.shape[0],
            int(self.gradient_position_norms.shape[0]),
            int(np.asarray(self.gradient_position_sketches).shape[1]),
        )
        if sketches.shape != expected:
            raise ValueError(
                "gradient_temperature_position_sketches must have shape "
                f"[temperatures, positions, K] {expected}; got {sketches.shape}."
            )
        if not np.all(np.isfinite(sketches)):
            raise ValueError(
                "gradient_temperature_position_sketches must be finite."
            )
        if not np.array_equal(
            sketches[canonical_index], np.asarray(self.gradient_position_sketches)
        ):
            raise ValueError(
                "The canonical row of gradient_temperature_position_sketches "
                "differs from gradient_position_sketches. They must be the same "
                "measured field: a directional analysis at T = 1 has to give the "
                "same answer whichever array it reads."
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
            + (
                _MEAN_TOKEN_ARRAY_NAME,
                _GRADIENT_SKETCH_ARRAY_NAME,
                _GRADIENT_TEMPERATURE_SKETCH_ARRAY_NAME,
            )
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

        # Forward-version guard, first: **before** the optional-array whitelist
        # below silently drops every name this reader does not know. A newer
        # writer's record may carry arrays that change how the ones we do
        # recognize must be read, so loading it partially would be worse than
        # refusing it -- the result would look complete and be wrong. Placed in
        # `from_parts` rather than `load_record` because this is the single
        # funnel: archive loads and direct in-memory construction both arrive
        # here.
        _require_supported_record_version(metadata.get("record_version"))

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
                + (
                _MEAN_TOKEN_ARRAY_NAME,
                _GRADIENT_SKETCH_ARRAY_NAME,
                _GRADIENT_TEMPERATURE_SKETCH_ARRAY_NAME,
            )
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
        gradient_temperature_position_sketches: np.ndarray | None = None,
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
            gradient_temperature_position_sketches=_optional_array(
                gradient_temperature_position_sketches, np.float32
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
