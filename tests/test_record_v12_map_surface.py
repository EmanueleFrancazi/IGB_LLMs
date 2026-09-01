"""Record v12: one map-count rule, one canonical representation, one accessor.

Stage 8b2 taught the measurement path to build several independent CountSketch
maps. This is the record layer catching up: how a multi-map measurement is
stored, how it is refused when it is malformed, and how every consumer reads it
through a single shape whatever ``M`` was.

Three properties carry the substage.

* **Metadata is authoritative.** ``M`` comes from
  ``sketch_protocol["map_count"]``, never from an array's rank or width. A rank
  is evidence about storage; it is not a statement about what was measured, and
  a record that has to be guessed at can be guessed wrong. Ranks are *validated
  against* the declared count.
* **One canonical representation.** A multi-map record stores only the
  temperature array; the canonical field is its canonical row. Storing a second
  copy would re-admit exactly the disagreement v11's slice-equality check exists
  to prevent -- and would cost 512 MiB at campaign scale.
* **Legacy records keep loading.** Pre-sketch records, v11 records, and
  Stage-8b2-era records all still load, as ``M = 1``, with every field value
  bitwise what it was.

The runner is untouched here and still measures at ``M = 1``; the multi-map
records below are synthetic. Making them *valid* is this substage's whole job.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from llm_behavior_lab.analysis import records as rec
from llm_behavior_lab.analysis.records import (
    RECORD_VERSION,
    SKETCH_PROTOCOL_SCHEMA_VERSION,
    InitializationExperimentRecord,
    load_record,
)

VOCAB = 16
ELIGIBLE = np.arange(3, VOCAB)
D = 6
K = 4
GRID = np.asarray([0.6, 1.0], dtype=np.float64)
CANONICAL_ROW = 1


def _protocol(map_count: int | None, *, schema: int | None = None, **overrides):
    """A sketch protocol block, shaped like the one the measurement emits."""

    protocol: dict = {"dimension": K, "seed": 20240917}
    if map_count is not None:
        protocol["map_count"] = map_count
    if schema is not None:
        protocol["schema_version"] = schema
        protocol.update(
            {
                "canonical_storage": "stored" if map_count == 1 else "derived",
                "canonical_relationship": "slice_of_temperature_array",
                "temperature_sketch_axes": (
                    ["temperature", "position", "bucket"]
                    if map_count == 1
                    else ["temperature", "position", "map", "bucket"]
                ),
                "estimator": "mean_of_per_map_inner_products_over_exact_norms",
                "accumulation_dtype": "float64",
                "storage_dtype": "float32",
                "recomputed_per_map": False,
            }
        )
    protocol.update(overrides)
    return protocol


def _build(
    *,
    protocol=None,
    record_version=RECORD_VERSION,
    canonical=None,
    temperature=None,
    with_gradients=True,
    grid=GRID,
):
    """Assemble a record directly, so malformed combinations can be tested."""

    targets = np.arange(D) % len(ELIGIBLE) + 3
    corpus = np.zeros(VOCAB, dtype=np.int64)
    corpus[ELIGIBLE] = 11
    metadata: dict = {"num_positions": D}
    if record_version is not None:
        metadata["record_version"] = record_version
    if with_gradients:
        analysis: dict = {"enabled": True, "initialization_index": 0}
        if protocol is not None:
            analysis["gradient_sketch"] = protocol
        metadata["analysis"] = {"gradient_analysis": analysis}

    extra: dict = {}
    if with_gradients:
        extra.update(
            gradient_position_indices=np.arange(D),
            gradient_position_target_ids=targets,
            gradient_position_greedy_ids=targets,
            gradient_position_norms=np.ones(D),
        )
        if grid is not None:
            extra.update(
                gradient_temperatures=grid,
                gradient_temperature_position_norms=np.ones((len(grid), D)),
            )
    if canonical is not None:
        extra["gradient_position_sketches"] = canonical
    if temperature is not None:
        extra["gradient_temperature_position_sketches"] = temperature

    return InitializationExperimentRecord.build(
        corpus_counts=corpus,
        selected_target_counts=np.bincount(targets, minlength=VOCAB),
        greedy_counts=np.bincount(targets, minlength=VOCAB)[None, :],
        nucleus_counts=np.bincount(targets, minlength=VOCAB)[None, None, :],
        mean_predicted_probabilities=np.full((1, VOCAB), 1.0 / VOCAB),
        model_seeds=[1000],
        eligible_token_ids=ELIGIBLE,
        metadata=metadata,
        **extra,
    )


def _single_map(*, protocol=None, **kwargs):
    temperature = np.arange(len(GRID) * D * K, dtype=np.float32).reshape(
        len(GRID), D, K
    )
    return _build(
        protocol=protocol
        if protocol is not None
        else _protocol(1, schema=SKETCH_PROTOCOL_SCHEMA_VERSION),
        canonical=temperature[CANONICAL_ROW],
        temperature=temperature,
        **kwargs,
    )


def _reload(record, metadata_overrides=None):
    """Round-trip through ``from_parts``, which is where the guard lives."""

    arrays = {
        name: getattr(record, name)
        for name in rec._ARRAY_NAMES
        if getattr(record, name) is not None
    }
    for name in (
        rec._GRADIENT_ARRAY_NAMES
        + rec._GRADIENT_TEMPERATURE_ARRAY_NAMES
        + (rec._GRADIENT_SKETCH_ARRAY_NAME, rec._GRADIENT_TEMPERATURE_SKETCH_ARRAY_NAME)
    ):
        value = getattr(record, name, None)
        if value is not None:
            arrays[name] = value
    arrays["eligible_token_ids"] = record.eligible_token_ids
    metadata = dict(record.metadata)
    metadata.update(metadata_overrides or {})
    return InitializationExperimentRecord.from_parts(arrays, metadata)


def _multi_map(maps=4, *, protocol=None, **kwargs):
    temperature = np.arange(
        len(GRID) * D * maps * K, dtype=np.float32
    ).reshape(len(GRID), D, maps, K)
    return _build(
        protocol=protocol
        if protocol is not None
        else _protocol(maps, schema=SKETCH_PROTOCOL_SCHEMA_VERSION),
        temperature=temperature,
        **kwargs,
    )


# -- version and forward guard ------------------------------------------------


def test_the_record_version_is_thirteen() -> None:
    """Bumped for the v13 storage-mode surface.

    The multi-map sketch layout introduced at 12 is unchanged, which is why the
    schema-v2 gate accepts both versions rather than pinning to whatever this
    build happens to be -- pinning is what would have made every existing v12
    record unreadable the moment this constant moved.
    """

    assert RECORD_VERSION == 13
    assert rec._SKETCH_V2_RECORD_VERSIONS == (12, 13)


def test_a_newer_record_version_is_refused() -> None:
    """Checked at the load seam, which is where a foreign archive arrives."""

    with pytest.raises(ValueError, match="newer writer"):
        _reload(_single_map(), {"record_version": RECORD_VERSION + 1})


def test_a_newer_record_version_is_refused_from_disk(tmp_path) -> None:
    record = _single_map()
    record.save(tmp_path)
    path = tmp_path / "initialization_distribution.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["record_version"] = RECORD_VERSION + 1
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="newer writer"):
        load_record(tmp_path)


@pytest.mark.parametrize(
    "bad", ["12", 12.5, 12.0, True, False, object(), [12], None.__class__]
)
def test_a_non_integral_record_version_is_refused(bad) -> None:
    """An integral *type*, not merely something ``int()`` survives.

    ``12.0`` is refused too: metadata that stores a version as a float was not
    built by something treating it as a version number. ``bool`` is ``Integral``
    in Python, so ``True`` would otherwise read as version 1.
    """

    with pytest.raises(ValueError, match="record_version must be an integer"):
        rec._require_supported_record_version(bad)


@pytest.mark.parametrize("bad", [0, -1, -5])
def test_a_non_positive_record_version_is_refused(bad) -> None:
    """There is no version 0.

    The format's own history starts at version 1, which predates
    ``eligible_token_ids``; a non-positive value is malformed metadata, not an
    ancient record.
    """

    with pytest.raises(ValueError, match="must be positive"):
        rec._require_supported_record_version(bad)


@pytest.mark.parametrize("good", [1, 5, 11, RECORD_VERSION, np.int64(11)])
def test_every_real_record_version_is_accepted(good) -> None:
    rec._require_supported_record_version(good)


def test_a_missing_record_version_is_still_supported() -> None:
    """Records predate the field; absence means old, not malformed."""

    rec._require_supported_record_version(None)


def test_the_guard_runs_before_unknown_arrays_are_discarded() -> None:
    """Order matters, and it is asserted rather than assumed.

    ``from_parts`` filters optional arrays through a fixed whitelist, so a newer
    writer's arrays would be dropped silently. A record loaded that way would
    look complete and be wrong, which is worse than refusing it -- so the guard
    has to come first.
    """

    record = _single_map()
    arrays = {
        name: getattr(record, name)
        for name in rec._ARRAY_NAMES
        if getattr(record, name) is not None
    }
    arrays["an_array_this_reader_does_not_know"] = np.zeros(3)
    metadata = dict(record.metadata)
    metadata["record_version"] = RECORD_VERSION + 1

    with pytest.raises(ValueError, match="newer writer"):
        InitializationExperimentRecord.from_parts(arrays, metadata)


# -- the resolver truth table --------------------------------------------------


def test_a_legacy_record_with_no_protocol_loads_as_one_map() -> None:
    record = _build(
        protocol=None,
        record_version=11,
        canonical=np.zeros((D, K), dtype=np.float32),
    )
    assert record.sketch_map_count == 1


def test_a_missing_record_version_loads_as_one_map() -> None:
    record = _build(
        protocol=_protocol(None),
        record_version=None,
        canonical=np.zeros((D, K), dtype=np.float32),
    )
    assert record.sketch_map_count == 1


def test_a_stage_8b2_era_explicit_single_map_loads() -> None:
    """`map_count: 1` with no schema version -- what 8b2-b wrote."""

    record = _build(
        protocol=_protocol(1),
        record_version=11,
        canonical=np.zeros((D, K), dtype=np.float32),
    )
    assert record.sketch_map_count == 1


def test_a_legacy_protocol_may_not_declare_multiple_maps() -> None:
    """A protocol that predates replicas cannot describe them."""

    with pytest.raises(ValueError, match="legacy protocol predates replicas"):
        _build(
            protocol=_protocol(4),
            record_version=11,
            canonical=np.zeros((D, K), dtype=np.float32),
        )


def test_the_legacy_wrapper_rule_end_to_end(tmp_path) -> None:
    """A v12 *container* may wrap a schema-absent legacy M=1 protocol.

    The full chain, in one place, because the rule is easy to tighten by
    accident: an earlier draft required schema 2 of every v12 record and made
    re-saved legacy archives unloadable.
    """

    legacy = _build(
        protocol=_protocol(None),
        record_version=11,
        canonical=np.zeros((D, K), dtype=np.float32),
    )
    assert legacy.sketch_map_count == 1

    legacy.save(tmp_path)
    reloaded = load_record(tmp_path)

    # The container keeps the version it was written at -- re-saving a v11
    # record must not promote it to whatever this build happens to be ...
    assert reloaded.record_version == 11
    assert reloaded.metadata["record_version"] == 11
    # ... while the protocol inside it still carries no schema version ...
    assert "schema_version" not in reloaded.gradient_analysis["gradient_sketch"]
    # ... and it still reads as a single map.
    assert reloaded.sketch_map_count == 1
    assert reloaded.per_map_sketches().shape == (D, 1, K)


def test_a_v12_schema_absent_record_may_not_declare_multiple_maps() -> None:
    """The legacy wrapper is confined to M = 1, at any record version."""

    with pytest.raises(ValueError, match="legacy protocol predates replicas"):
        _build(
            protocol=_protocol(4),
            record_version=RECORD_VERSION,
            canonical=np.zeros((D, K), dtype=np.float32),
        )


def test_a_v12_schema_absent_four_dimensional_array_is_refused() -> None:
    """A schema-absent protocol can never reach the multi-map layout."""

    with pytest.raises(ValueError, match="requires sketch-protocol schema_version"):
        _multi_map(record_version=RECORD_VERSION, protocol=_protocol(4))


def test_re_saving_a_legacy_record_does_not_make_it_unloadable(tmp_path) -> None:
    """`save()` preserves the record's own version.

    Loading an old record and writing it back out is ordinary, and it must not
    turn a valid archive into one this reader refuses -- nor into one that
    *claims* to be current. An earlier implementation stamped the module
    constant here, so a round trip silently relabelled v11 content as v12; the
    masquerade guard lives where it belongs instead, in the schema rules.
    """

    _build(
        protocol=_protocol(None),
        record_version=11,
        canonical=np.zeros((D, K), dtype=np.float32),
    ).save(tmp_path)
    reloaded = load_record(tmp_path)

    assert reloaded.record_version == 11
    assert reloaded.metadata["record_version"] == 11
    assert reloaded.sketch_map_count == 1


@pytest.mark.parametrize(
    "record_version, schema",
    [(11, SKETCH_PROTOCOL_SCHEMA_VERSION), (RECORD_VERSION, None)],
    ids=["wrong-record-version", "missing-schema"],
)
def test_a_four_dimensional_array_needs_every_condition(record_version, schema) -> None:
    """Record 12 **and** schema 2 **and** an explicit count above one."""

    with pytest.raises(ValueError):
        _multi_map(
            record_version=record_version,
            protocol=_protocol(4, schema=schema),
        )


def test_a_four_dimensional_array_needs_an_explicit_count() -> None:
    protocol = _protocol(4, schema=SKETCH_PROTOCOL_SCHEMA_VERSION)
    protocol.pop("map_count")
    with pytest.raises(ValueError, match="must declare map_count explicitly"):
        _multi_map(protocol=protocol)


def test_schema_v2_with_sketches_requires_an_explicit_count() -> None:
    protocol = _protocol(1, schema=SKETCH_PROTOCOL_SCHEMA_VERSION)
    protocol.pop("map_count")
    with pytest.raises(ValueError, match="must declare map_count explicitly"):
        _build(protocol=protocol, canonical=np.zeros((D, K), dtype=np.float32))


def test_schema_v2_requires_record_version_twelve() -> None:
    with pytest.raises(ValueError, match="requires record_version"):
        _build(
            protocol=_protocol(1, schema=SKETCH_PROTOCOL_SCHEMA_VERSION),
            record_version=11,
            canonical=np.zeros((D, K), dtype=np.float32),
        )


@pytest.mark.parametrize("bad", [0, -1, 2.5, True, "4", None])
def test_a_malformed_map_count_is_refused(bad) -> None:
    protocol = _protocol(4, schema=SKETCH_PROTOCOL_SCHEMA_VERSION)
    protocol["map_count"] = bad
    with pytest.raises(ValueError):
        _multi_map(protocol=protocol)


@pytest.mark.parametrize("bad", [3, 1, "2", 2.5, True])
def test_an_unsupported_protocol_schema_version_is_refused(bad) -> None:
    protocol = _protocol(1, schema=SKETCH_PROTOCOL_SCHEMA_VERSION)
    protocol["schema_version"] = bad
    with pytest.raises(ValueError, match="schema_version"):
        _build(protocol=protocol, canonical=np.zeros((D, K), dtype=np.float32))


def test_the_two_layers_agree_on_which_counts_are_legal() -> None:
    """The record layer restates the rule the measurement layer applies.

    Deliberate duplication across a hard boundary -- the analysis package is
    NumPy-only so a finished experiment reads without PyTorch -- so the two are
    pinned to the same answers here rather than trusted to stay aligned.
    """

    from llm_behavior_lab.evaluation.position_gradients import _validated_map_count

    for good in (1, 2, 4, np.int64(3)):
        assert rec._integral_map_count(good, where="x") == _validated_map_count(
            good, name="x"
        )
    for bad in (0, -1, 2.5, True, "4", None):
        with pytest.raises(ValueError):
            rec._integral_map_count(bad, where="x")
        with pytest.raises(ValueError):
            _validated_map_count(bad, name="x")


# -- physical layout -----------------------------------------------------------


def test_single_map_physical_shapes_are_the_historical_ones() -> None:
    record = _single_map()

    assert record.gradient_position_sketches.shape == (D, K)
    assert record.gradient_temperature_position_sketches.shape == (len(GRID), D, K)
    assert record.sketch_map_count == 1


def test_multi_map_stores_only_the_temperature_array() -> None:
    record = _multi_map(maps=4)

    assert record.gradient_position_sketches is None
    assert record.gradient_temperature_position_sketches.shape == (len(GRID), D, 4, K)
    assert record.sketch_map_count == 4


@pytest.mark.parametrize("shape", [(D, K), (D, 4, K)], ids=["2d", "3d"])
def test_a_physical_canonical_array_is_refused_at_multiple_maps(shape) -> None:
    """Either form, because either would be a second canonical representation."""

    with pytest.raises(ValueError, match="must be absent when map_count is above one"):
        _multi_map(maps=4, canonical=np.zeros(shape, dtype=np.float32))


def test_the_map_axis_must_match_the_declared_count() -> None:
    temperature = np.zeros((len(GRID), D, 3, K), dtype=np.float32)
    with pytest.raises(ValueError, match="map axis has length 3"):
        _build(
            protocol=_protocol(4, schema=SKETCH_PROTOCOL_SCHEMA_VERSION),
            temperature=temperature,
        )


def test_a_single_map_count_with_a_four_dimensional_array_is_refused() -> None:
    with pytest.raises(ValueError, match="map_count is 1"):
        _multi_map(maps=4, protocol=_protocol(1, schema=SKETCH_PROTOCOL_SCHEMA_VERSION))


def test_a_multi_map_record_round_trips_through_disk(tmp_path) -> None:
    record = _multi_map(maps=4)
    record.save(tmp_path)
    reloaded = load_record(tmp_path)

    assert reloaded.sketch_map_count == 4
    assert reloaded.gradient_position_sketches is None
    assert np.array_equal(
        reloaded.gradient_temperature_position_sketches,
        record.gradient_temperature_position_sketches,
    )


def test_a_canonical_only_multi_map_record_is_valid() -> None:
    """``N_T = 1``: the grid holds the canonical temperature alone."""

    grid = np.asarray([1.0])
    temperature = np.zeros((1, D, 4, K), dtype=np.float32)
    record = _build(
        protocol=_protocol(4, schema=SKETCH_PROTOCOL_SCHEMA_VERSION),
        temperature=temperature,
        grid=grid,
    )

    assert record.sketch_map_count == 4
    assert record.per_map_sketches().shape == (D, 4, K)


# -- protocol v2 emission and validation ----------------------------------------


@pytest.mark.parametrize("maps", [1, 4])
def test_the_measurement_emits_every_schema_v2_key(maps) -> None:
    """The emitted protocol must satisfy the validator that will read it."""

    import torch

    from llm_behavior_lab.evaluation.init_distribution import build_evaluation_positions
    from llm_behavior_lab.evaluation.position_gradients import (
        compute_position_gradient_norms,
    )

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(12, 8)
            self.output = torch.nn.Linear(8, 12, bias=False)
            self.double()

        def forward(self, input_ids):
            return type(
                "Output", (), {"logits": self.output(self.embedding(input_ids))}
            )()

    positions = build_evaluation_positions(
        [(index * 5 + 2) % 12 for index in range(64)], block_size=4, num_windows=2
    )
    protocol = compute_position_gradient_norms(
        TinyModel(),
        positions,
        vocab_size=12,
        temperatures=(1.0,),
        gradient_sketch=True,
        sketch_dimension=K,
        sketch_maps=maps,
    ).sketch_protocol

    assert protocol["schema_version"] == SKETCH_PROTOCOL_SCHEMA_VERSION
    assert protocol["canonical_storage"] == ("stored" if maps == 1 else "derived")
    assert protocol["canonical_relationship"] == "slice_of_temperature_array"
    assert protocol["temperature_sketch_axes"] == (
        ["temperature", "position", "bucket"]
        if maps == 1
        else ["temperature", "position", "map", "bucket"]
    )
    assert protocol["estimator"] == "mean_of_per_map_inner_products_over_exact_norms"
    assert protocol["accumulation_dtype"] == "float64"
    assert protocol["storage_dtype"] == "float32"
    assert protocol["recomputed_per_map"] is False
    # Stage 8b2 keys survive untouched.
    assert protocol["seed"] == 20240917
    assert protocol["map_count"] == maps
    assert json.loads(json.dumps(protocol)) == protocol


def test_the_emitted_schema_version_matches_the_record_layer() -> None:
    from llm_behavior_lab.evaluation import position_gradients as pg

    assert pg._SKETCH_PROTOCOL_SCHEMA_VERSION == SKETCH_PROTOCOL_SCHEMA_VERSION


@pytest.mark.parametrize(
    "key, value",
    [
        ("canonical_relationship", "independent_measurement"),
        ("estimator", "mean_of_sketch_vectors"),
        ("accumulation_dtype", "float32"),
        ("storage_dtype", "float64"),
        ("canonical_storage", "derived"),
        ("temperature_sketch_axes", ["temperature", "position", "map", "bucket"]),
        ("recomputed_per_map", True),
    ],
)
def test_a_contradictory_schema_v2_field_is_refused(key, value) -> None:
    protocol = _protocol(1, schema=SKETCH_PROTOCOL_SCHEMA_VERSION)
    protocol[key] = value
    with pytest.raises(ValueError, match=key):
        _single_map(protocol=protocol)


@pytest.mark.parametrize(
    "key, value",
    [
        ("canonical_relationship", 7),
        ("estimator", ["a", "list"]),
        ("accumulation_dtype", None),
        ("storage_dtype", {"dtype": "float32"}),
        ("canonical_storage", 1),
        ("temperature_sketch_axes", "temperature,position,bucket"),
        ("temperature_sketch_axes", 3),
        ("recomputed_per_map", 0),
        ("recomputed_per_map", "false"),
    ],
)
def test_a_schema_v2_field_of_the_wrong_type_is_refused(key, value) -> None:
    """Types, not only values.

    ``recomputed_per_map`` is checked with ``is not False`` specifically so a
    falsy ``0`` cannot stand in for the boolean, and the axis list is type-checked
    before it is compared so a non-sequence raises ValueError like every other
    schema violation rather than a bare TypeError.
    """

    protocol = _protocol(1, schema=SKETCH_PROTOCOL_SCHEMA_VERSION)
    protocol[key] = value
    with pytest.raises(ValueError, match=key):
        _single_map(protocol=protocol)


@pytest.mark.parametrize(
    "key",
    [
        "canonical_relationship",
        "estimator",
        "accumulation_dtype",
        "storage_dtype",
        "recomputed_per_map",
    ],
)
def test_a_missing_schema_v2_field_is_refused(key) -> None:
    protocol = _protocol(1, schema=SKETCH_PROTOCOL_SCHEMA_VERSION)
    protocol.pop(key)
    with pytest.raises(ValueError, match=key):
        _single_map(protocol=protocol)


# -- the accessor ---------------------------------------------------------------


def test_per_map_sketches_is_uniform_at_every_map_count() -> None:
    assert _single_map().per_map_sketches().shape == (D, 1, K)
    assert _multi_map(maps=4).per_map_sketches().shape == (D, 4, K)


def test_the_single_map_canonical_request_reads_the_historical_array() -> None:
    record = _single_map()
    result = record.per_map_sketches()

    assert np.array_equal(result[:, 0, :], record.gradient_position_sketches)


def test_a_noncanonical_temperature_gets_a_length_one_map_axis() -> None:
    record = _single_map()
    result = record.per_map_sketches(loss_temperature=0.6)

    assert result.shape == (D, 1, K)
    assert np.array_equal(
        result[:, 0, :], record.gradient_temperature_position_sketches[0]
    )


def test_the_multi_map_accessor_selects_the_requested_temperature() -> None:
    record = _multi_map(maps=4)
    stored = record.gradient_temperature_position_sketches

    assert np.array_equal(record.per_map_sketches(loss_temperature=0.6), stored[0])
    assert np.array_equal(record.per_map_sketches(), stored[CANONICAL_ROW])


def test_the_accessor_shares_storage_with_the_record_array() -> None:
    """Views of the *intended* array, not merely of something.

    ``base is not None`` would pass for a view of any array at all, including a
    copy's. ``shares_memory`` is what actually establishes that reading through
    the accessor reads the record's own storage -- which is the property that
    makes the uniform ``[D, M, K]`` shape free at campaign scale, where these
    arrays are gigabytes.
    """

    single = _single_map()
    multi = _multi_map(maps=4)

    # M = 1, canonical: the historical two-dimensional array.
    assert np.shares_memory(
        single.per_map_sketches(), single.gradient_position_sketches
    )
    # M = 1, a specific non-canonical temperature: that row of the grid.
    assert np.shares_memory(
        single.per_map_sketches(loss_temperature=0.6),
        single.gradient_temperature_position_sketches,
    )
    # M > 1, canonical: the canonical row of the temperature array.
    assert np.shares_memory(
        multi.per_map_sketches(), multi.gradient_temperature_position_sketches
    )
    # M > 1, a specific non-canonical temperature.
    assert np.shares_memory(
        multi.per_map_sketches(loss_temperature=0.6),
        multi.gradient_temperature_position_sketches,
    )


def test_an_unmeasured_temperature_is_refused() -> None:
    with pytest.raises(ValueError, match="No measured gradient-direction field"):
        _single_map().per_map_sketches(loss_temperature=0.24)


def test_the_accessor_refuses_a_record_without_sketches() -> None:
    record = _build(protocol=None, canonical=None, temperature=None)

    with pytest.raises(ValueError, match="no per-position gradient sketches"):
        record.per_map_sketches()


# -- sketch_map_count edges ------------------------------------------------------


def test_sketch_map_count_raises_without_a_sketch_surface() -> None:
    """Returning 1 would answer a question the record never had."""

    record = _build(protocol=None, canonical=None, temperature=None)

    with pytest.raises(ValueError, match="no gradient sketches"):
        record.sketch_map_count


def test_a_version_twelve_record_without_sketches_is_valid() -> None:
    """No sketches means no protocol is required, and that is not an error."""

    record = _build(protocol=None, canonical=None, temperature=None)

    assert record.has_gradient_position_sketches is False
    assert record.has_position_gradients is True


def test_a_pre_sketch_record_still_loads(tmp_path) -> None:
    record = _build(
        protocol=None,
        record_version=11,
        canonical=None,
        temperature=None,
        with_gradients=False,
        grid=None,
    )
    record.save(tmp_path)
    reloaded = load_record(tmp_path)

    assert reloaded.has_position_gradients is False
    assert reloaded.has_gradient_position_sketches is False


# -- the derivable-aware predicate -------------------------------------------------


def test_the_predicate_is_true_for_a_physically_stored_canonical() -> None:
    assert _single_map().has_gradient_position_sketches is True


def test_the_predicate_is_true_when_the_canonical_is_derivable() -> None:
    """The whole point: a multi-map record must not disable figure 20 or 24."""

    record = _multi_map(maps=4)

    assert record.gradient_position_sketches is None
    assert record.has_gradient_position_sketches is True


def test_the_predicate_is_false_without_the_exact_norms() -> None:
    """Direction needs the divisor as well as the sketch."""

    record = _multi_map(maps=4)
    object.__setattr__(record, "gradient_position_norms", None)

    assert record.has_gradient_position_sketches is False


def test_the_predicate_is_false_when_the_canonical_temperature_is_unmeasured() -> None:
    record = _multi_map(maps=4)
    object.__setattr__(record, "gradient_temperatures", np.asarray([0.6, 0.24]))

    assert record.has_gradient_position_sketches is False


# -- temperature-matching ownership --------------------------------------------------


def test_directional_fields_still_exposes_the_tolerance() -> None:
    """Its two importers must keep working after the constant moved."""

    from llm_behavior_lab.analysis import directional_fields

    assert directional_fields.TEMPERATURE_MATCH_TOLERANCE == 1e-6
    assert (
        directional_fields.TEMPERATURE_MATCH_TOLERANCE
        is rec.TEMPERATURE_MATCH_TOLERANCE
    )


def test_the_record_layer_imports_nothing_from_the_package() -> None:
    """No cycle: `directional_fields` imports `records`, never the reverse."""

    source = (rec.__file__ or "")
    text = open(source, encoding="utf-8").read()

    assert "from llm_behavior_lab" not in text
    assert "import llm_behavior_lab" not in text


def test_the_tolerance_is_applied_when_matching() -> None:
    record = _single_map()
    nudged = 0.6 + 5e-7

    assert np.array_equal(
        record.per_map_sketches(loss_temperature=nudged),
        record.per_map_sketches(loss_temperature=0.6),
    )


def test_an_ambiguous_temperature_match_is_refused() -> None:
    record = _single_map()
    object.__setattr__(record, "gradient_temperatures", np.asarray([0.6, 0.6]))

    with pytest.raises(ValueError, match="matches 2 measured temperatures"):
        record.per_map_sketches(loss_temperature=0.6)
