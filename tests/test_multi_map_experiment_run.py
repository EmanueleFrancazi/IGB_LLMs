"""A production ``M > 1`` record, produced by the production runner, on CPU.

Every earlier stage of this work exercised multi-map code through *synthetic*
records: arrays assembled in a test, handed straight to the schema. That proves
the schema and the consumers, and it proves nothing about the path an actual
experiment takes -- resolution, measurement, record assembly, ``save()``, the
authoritative loader, analysis, figures. This module closes that gap by running
the real entry point and reading what it wrote.

The single fact worth stating plainly: **at ``M > 1`` the canonical
``gradient_position_sketches`` array is not written at all.** It is derived from
the canonical row of the temperature array, so there is exactly one canonical
representation and no possibility of two disagreeing -- and at campaign scale it
avoids duplicating roughly half a gigabyte per record. The tests below check the
*physical* archive for that, not just the accessor, because an accessor can
paper over a stored array that should not exist.

The other half is that ``M = 1`` did not move. Not "passes its tests": produces
a bit-identical archive whether or not the new option is mentioned on the
command line.

Everything here is CPU-only, offline, and writes exclusively to pytest temporary
directories.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from llm_behavior_lab.analysis.records import RECORD_VERSION, load_record

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_initialization_distribution_experiment.py"
DATA_CONFIG = REPO_ROOT / "configs" / "data" / "tiny_text.yaml"
MODEL_CONFIG = REPO_ROOT / "configs" / "model" / "tiny_llama.yaml"
EXPERIMENT_CONFIG = REPO_ROOT / "configs" / "experiment" / "initialization_distribution.yaml"

#: The measured loss-temperature grid. Two values, one of which must be the
#: canonical T = 1: the record's canonical fields are that row, and the
#: multi-map canonical sketch is derived from it.
TEMPERATURES = ("0.6", "1.0")
WINDOWS = 4
BLOCK = 8
POSITIONS = WINDOWS * BLOCK
WIDTH = 8

CANONICAL_KEY = "gradient_position_sketches"
TEMPERATURE_KEY = "gradient_temperature_position_sketches"


def _runner():
    spec = importlib.util.spec_from_file_location(
        "initialization_distribution_script", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _cpu_data_config(directory: Path) -> Path:
    """The tracked tiny-corpus config, pinned to CPU.

    Copied rather than edited in place: the test must never select an
    accelerator and must never modify a tracked file.
    """

    config = yaml.safe_load(DATA_CONFIG.read_text(encoding="utf-8"))
    config["runtime"]["device"] = "cpu"
    destination = directory / "tiny_text_cpu.yaml"
    destination.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return destination


def _run(directory: Path, run_id: str, extra=(), *, module=None) -> Path:
    """Drive the real runner once and return its analyses directory."""

    module = module or _runner()
    directory.mkdir(parents=True, exist_ok=True)
    argv = [
        "run_initialization_distribution_experiment.py",
        "--data-config", str(_cpu_data_config(directory)),
        "--model-config", str(MODEL_CONFIG),
        "--experiment-config", str(EXPERIMENT_CONFIG),
        "--num-initializations", "1",
        "--num-windows", str(WINDOWS),
        "--block-size", str(BLOCK),
        "--num-replicates", "1",
        "--no-temperature-sweep",
        "--no-uniform-null",
        "--no-input-structure",
        "--gradient-analysis",
        "--gradient-sketch",
        "--sketch-dimension", str(WIDTH),
        "--gradient-temperatures", *TEMPERATURES,
        "--no-figures",
        "--offline",
        "--output-dir", str(directory),
        "--run-id", run_id,
        *extra,
    ]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sys, "argv", argv)
        module.main()
    return directory / "initialization_distribution" / run_id / "analyses"


def _arrays(analyses: Path) -> dict:
    """Every array in the archive, read the way the loaders read it."""

    with np.load(
        analyses / "initialization_distribution.npz", allow_pickle=False
    ) as data:
        return {key: data[key] for key in data.files}


@pytest.fixture(scope="module")
def four_map_run(tmp_path_factory):
    """One M = 4 run, reused by every assertion about it."""

    directory = tmp_path_factory.mktemp("m4")
    return _run(directory, "four_maps", ["--sketch-maps", "4"])


@pytest.fixture(scope="module")
def single_map_run(tmp_path_factory):
    """One M = 1 run with the option omitted entirely."""

    directory = tmp_path_factory.mktemp("m1")
    return _run(directory, "one_map")


# -- forwarding ---------------------------------------------------------------


def test_the_runner_forwards_the_requested_map_count(tmp_path) -> None:
    """The number reaches the estimator, not merely the console.

    A run could print ``M = 4``, measure one map, and produce an archive that
    looks single-map for the mundane reason that it is. Spying on the call is
    the only place that distinction is visible before the arrays exist.
    """

    from llm_behavior_lab.evaluation.position_gradients import (
        compute_position_gradient_norms,
    )

    module = _runner()
    captured = {}

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return compute_position_gradient_norms(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(module, "compute_position_gradient_norms", spy)
        _run(tmp_path, "spied", ["--sketch-maps", "4"], module=module)

    assert captured["sketch_maps"] == 4
    assert captured["gradient_sketch"] is True
    assert captured["sketch_dimension"] == WIDTH


def test_an_omitted_option_forwards_a_single_map_explicitly(tmp_path) -> None:
    """Explicitly 1, not silently absent.

    Letting the argument default inside the estimator would mean the runner
    never states what it asked for, and a later change to that default would
    silently change what production measures.
    """

    from llm_behavior_lab.evaluation.position_gradients import (
        compute_position_gradient_norms,
    )

    module = _runner()
    captured = {}

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return compute_position_gradient_norms(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(module, "compute_position_gradient_norms", spy)
        _run(tmp_path, "spied_default", module=module)

    assert captured["sketch_maps"] == 1


# -- the physical archive at M > 1 --------------------------------------------


def test_the_canonical_sketch_key_is_absent_from_the_archive(four_map_run) -> None:
    """Checked on the NPZ itself, not through an accessor.

    ``per_map_sketches`` returns the canonical block at every map count, so it
    cannot distinguish "derived" from "stored". Only the key list can.
    """

    keys = set(_arrays(four_map_run))

    assert CANONICAL_KEY not in keys
    assert TEMPERATURE_KEY in keys


def test_map_zero_is_not_written_into_the_canonical_field(four_map_run) -> None:
    """The subtle wrong answer, ruled out explicitly.

    A ``[D, K]`` slice of a four-map measurement is not that measurement's
    canonical sketch -- it is a quarter of it -- and it would load without
    complaint, giving every downstream estimate a silently reduced map budget.
    """

    record = load_record(four_map_run)

    assert record.gradient_position_sketches is None
    assert CANONICAL_KEY not in set(_arrays(four_map_run))


def test_the_temperature_array_carries_the_replica_axis(four_map_run) -> None:
    stored = _arrays(four_map_run)[TEMPERATURE_KEY]

    assert stored.shape == (len(TEMPERATURES), POSITIONS, 4, WIDTH)
    assert stored.dtype == np.float32


def test_the_archive_holds_no_object_dtype_array(four_map_run) -> None:
    """``np.load`` runs with ``allow_pickle=False``; object dtype would not load."""

    for key, value in _arrays(four_map_run).items():
        assert value.dtype != object, key


# -- the record, through the authoritative loader ------------------------------


def test_the_record_loads_and_reports_its_map_count(four_map_run) -> None:
    record = load_record(four_map_run)

    assert record.sketch_map_count == 4
    assert record.metadata["record_version"] == RECORD_VERSION


def test_the_map_count_comes_from_metadata_not_from_the_array(four_map_run) -> None:
    """Stated where it can be read, not left to be inferred from a shape."""

    protocol = load_record(four_map_run).metadata["analysis"]["gradient_analysis"][
        "gradient_sketch"
    ]

    assert protocol["map_count"] == 4
    assert protocol["schema_version"] == 2
    assert protocol["canonical_storage"] == "derived"
    assert protocol["canonical_relationship"] == "slice_of_temperature_array"
    assert protocol["temperature_sketch_axes"] == [
        "temperature", "position", "map", "bucket",
    ]
    assert protocol["recomputed_per_map"] is False
    assert len(protocol["map_seeds"]) == 4


def test_the_accessor_is_uniform_at_every_temperature(four_map_run) -> None:
    record = load_record(four_map_run)

    assert record.per_map_sketches().shape == (POSITIONS, 4, WIDTH)
    for temperature in (0.6, 1.0):
        assert record.per_map_sketches(loss_temperature=temperature).shape == (
            POSITIONS, 4, WIDTH,
        )


def test_the_canonical_accessor_is_the_canonical_temperature_row(four_map_run) -> None:
    """Derivation, asserted as equality rather than assumed."""

    record = load_record(four_map_run)
    grid = np.asarray(record.gradient_temperatures)
    canonical = int(np.argmin(np.abs(grid - 1.0)))

    assert np.array_equal(
        record.per_map_sketches(),
        record.gradient_temperature_position_sketches[canonical],
    )


def test_the_availability_predicates_stay_true_when_derived(four_map_run) -> None:
    """Without this a multi-map record would silently disable figures 20 and 24
    and both artifact writers -- the array is absent, but the sketch is not.
    """

    record = load_record(four_map_run)

    assert record.has_gradient_position_sketches is True
    assert record.has_temperature_gradient_sketches is True


# -- analysis and figures on the produced record -------------------------------


def test_the_core_clustering_analysis_runs_on_the_produced_record(four_map_run) -> None:
    import importlib

    gc = importlib.import_module("llm_behavior_lab.analysis.gradient_clustering")
    record = load_record(four_map_run)

    rows, usable = gc.unit_sketches_per_map(record)
    ensemble = gc.unit_sketches(record)

    assert rows.shape == (POSITIONS, 4, WIDTH)
    # The 1/sqrt(M) concatenation: one embedding whose Gram is the ensemble
    # estimate, which is why every bilinear consumer needs no formula change.
    assert ensemble[0].shape == (POSITIONS, 4 * WIDTH)
    assert usable.dtype == np.bool_


def test_per_map_uncertainty_is_available_with_m_minus_one_degrees(
    four_map_run,
) -> None:
    import importlib

    gc = importlib.import_module("llm_behavior_lab.analysis.gradient_clustering")
    record = load_record(four_map_run)

    summary = gc.clustering_summary_per_map(record)

    for grouping in ("target", "greedy"):
        block = summary[grouping]
        assert block["map_count"] == 4
        assert block["delta_per_map"].shape == (4,)
        assert block["delta"]["degrees_of_freedom"] == 3
        assert block["delta"]["uncertainty_available"] is True
        assert block["delta"]["standard_error"] == pytest.approx(
            block["delta"]["sample_sd"] / 2.0
        )


def test_figures_twenty_and_twenty_four_render(four_map_run, tmp_path) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    from llm_behavior_lab.analysis import figures

    record = load_record(four_map_run)

    twenty = figures.plot_gradient_directional_clustering(record, tmp_path)
    twenty_four = figures.plot_cross_partition_geometry(record, tmp_path)

    assert twenty and twenty_four
    for path in [*twenty, *twenty_four]:
        assert path.exists() and path.stat().st_size > 0


# -- the record declares its schema before it reaches disk ----------------------


def test_a_sketched_run_declares_its_record_version_at_build_time(
    single_map_run, four_map_run
) -> None:
    """A regression test for a defect this stage found in the shipped runner.

    ``save()`` has always stamped ``record_version`` on the way out, so the
    *persisted* record was correct and nobody noticed that the in-memory one
    never carried it. Stage 8b2-b then made the measurement emit
    ``schema_version: 2``, and Stage 8c-a made schema 2 require
    ``record_version == 12``. Validation runs inside ``build()`` -- before
    ``save()`` -- so from that point on **every sketched production run failed
    at record construction**, at both map counts, with "A schema-v2 sketch
    protocol requires record_version 12; got None".

    It survived review because nothing drove the runner end to end with the
    sketch enabled: the schema tests built records directly and stated the
    version themselves, which is what the writer should have been doing.

    So the assertion is not about a metadata key. It is that a run which
    enables the sketch produces a record at all, at every map count, and that
    the record says which schema it was written under.
    """

    for analyses in (single_map_run, four_map_run):
        payload = json.loads(
            (analyses / "initialization_distribution.json").read_text(encoding="utf-8")
        )
        protocol = payload["analysis"]["gradient_analysis"]["gradient_sketch"]

        assert payload["record_version"] == RECORD_VERSION
        assert protocol["schema_version"] == 2
        assert load_record(analyses).metadata["record_version"] == RECORD_VERSION


# -- M = 1 is untouched --------------------------------------------------------


def test_a_single_map_run_keeps_the_historical_layout(single_map_run) -> None:
    stored = _arrays(single_map_run)

    assert stored[CANONICAL_KEY].shape == (POSITIONS, WIDTH)
    assert stored[TEMPERATURE_KEY].shape == (len(TEMPERATURES), POSITIONS, WIDTH)
    assert load_record(single_map_run).sketch_map_count == 1


def test_the_single_map_protocol_still_declares_stored_storage(
    single_map_run,
) -> None:
    protocol = load_record(single_map_run).metadata["analysis"]["gradient_analysis"][
        "gradient_sketch"
    ]

    assert protocol["map_count"] == 1
    assert protocol["canonical_storage"] == "stored"
    assert protocol["temperature_sketch_axes"] == [
        "temperature", "position", "bucket",
    ]


def test_the_canonical_row_still_equals_the_stored_canonical_array(
    single_map_run,
) -> None:
    """The historical invariant: the two representations cannot disagree."""

    record = load_record(single_map_run)
    grid = np.asarray(record.gradient_temperatures)
    canonical = int(np.argmin(np.abs(grid - 1.0)))

    assert np.array_equal(
        record.gradient_position_sketches,
        record.gradient_temperature_position_sketches[canonical],
    )


def test_omitting_the_option_and_asking_for_one_map_produce_the_same_archive(
    tmp_path,
) -> None:
    """Bit-identical, every array, in both directions.

    The reference every existing record and the frozen CountSketch fixture were
    produced against is the command line that does not mention maps. An option
    that perturbed it -- by a different RNG draw order, an extra allocation, a
    reshaped array -- would invalidate exactly the comparison it exists to
    enable, and would do it silently.
    """

    omitted = _arrays(_run(tmp_path / "a", "omitted"))
    explicit = _arrays(_run(tmp_path / "b", "explicit", ["--sketch-maps", "1"]))

    assert set(omitted) == set(explicit)
    for key, value in omitted.items():
        assert value.shape == explicit[key].shape, key
        assert value.dtype == explicit[key].dtype, key
        assert np.array_equal(value, explicit[key]), key


def test_the_two_single_map_runs_agree_on_everything_but_run_identity(
    tmp_path,
) -> None:
    """Metadata too, not only arrays.

    Run identity, timings and the host environment differ between any two runs
    and say nothing about the option; everything else must match, including the
    complete sketch protocol.
    """

    def prune(analyses: Path) -> dict:
        payload = json.loads(
            (analyses / "initialization_distribution.json").read_text(encoding="utf-8")
        )
        payload.pop("run_id", None)
        payload.pop("created_at", None)
        payload.pop("environment", None)
        payload.pop("runtime", None)
        payload["analysis"]["gradient_analysis"].pop("seconds", None)
        return payload

    omitted = prune(_run(tmp_path / "a", "omitted"))
    explicit = prune(_run(tmp_path / "b", "explicit", ["--sketch-maps", "1"]))

    assert omitted == explicit
    assert omitted["analysis"]["gradient_analysis"]["gradient_sketch"]["map_count"] == 1
    assert omitted["record_version"] == RECORD_VERSION


def test_the_backward_pass_count_is_independent_of_the_map_count(
    single_map_run, four_map_run
) -> None:
    """``D x N_T`` at both, which is the governing claim of the replica design.

    Every map projects the same gradient from one ``autograd.grad`` result. A
    record whose cost scaled with ``M`` would describe a different measurement,
    one in which projection and recomputation variance are confounded -- which
    is precisely the quantity the replicas exist to separate. The persisted
    norms are one scalar per (temperature, position) backward, so their shape
    is the visible consequence.
    """

    one = load_record(single_map_run)
    four = load_record(four_map_run)

    assert one.gradient_temperature_position_norms.shape == (
        len(TEMPERATURES), POSITIONS,
    )
    assert (
        four.gradient_temperature_position_norms.shape
        == one.gradient_temperature_position_norms.shape
    )


# -- an M = 4 run cannot land on an M = 1 run -----------------------------------


def test_two_runs_cannot_share_an_automatic_identity(tmp_path) -> None:
    """Nothing about the map count needs to enter the run name.

    A run id is a UTC timestamp, the dataset, tokenizer and model slugs, the
    three resolution counts, **and a random suffix**. Two runs differing only in
    map count therefore get different directories for the same reason any two
    runs do, and adding an ``M`` component would rename every historical run
    pattern to solve a problem the suffix already solves.
    """

    from llm_behavior_lab.experiment.naming import compose_run_id

    common = dict(
        dataset="wikitext2_raw_train1k",
        tokenizer={"type": "pretrained", "identifier": "mistralai/Mistral-7B-v0.1",
                   "vocab_size": 32000},
        model_name="llama_tiny",
        model_vocab_size=32000,
        num_positions=32768,
        num_initializations=1,
        num_replicates=1,
    )

    assert compose_run_id(**common) != compose_run_id(**common)


def test_an_explicit_identity_refuses_to_overwrite(tmp_path) -> None:
    """And when the identity *is* pinned, the second run is refused.

    The dangerous shape is a four-map run silently replacing a single-map
    record that a figure or an artifact already refers to. It cannot happen:
    the run directory is created with ``exist_ok=False``.
    """

    first = _run(tmp_path, "pinned")
    before = _arrays(first)

    with pytest.raises(FileExistsError, match="will not be overwritten"):
        _run(tmp_path, "pinned", ["--sketch-maps", "4"])

    after = _arrays(first)
    assert set(before) == set(after)
    assert CANONICAL_KEY in after
    assert after[CANONICAL_KEY].shape == (POSITIONS, WIDTH)
    assert np.array_equal(before[CANONICAL_KEY], after[CANONICAL_KEY])


# -- facts the Stage 8c-d v11-loader probe needs -------------------------------


def test_the_multi_map_archive_is_shaped_for_the_released_loader_probe(
    four_map_run,
) -> None:
    """The archive must be able to make a released v11 loader fail *loudly*.

    Reusing the existing field name is the whole mechanism. A new array name
    would have loaded silently in released code, with the sketches reported
    absent and every directional analysis quietly disabled; the same name at a
    rank that code cannot interpret is the only loud-failure channel available.

    This asserts the archive's shape, not the outcome of the probe -- that is
    Stage 8c-d's, run once by hand against the source at ``410803c``. It stays
    out of the suite deliberately: it would depend on Git history and break in a
    shallow clone or an exported tarball.
    """

    arrays = _arrays(four_map_run)
    payload = json.loads(
        (four_map_run / "initialization_distribution.json").read_text(encoding="utf-8")
    )
    protocol = payload["analysis"]["gradient_analysis"]["gradient_sketch"]

    assert TEMPERATURE_KEY in arrays
    assert arrays[TEMPERATURE_KEY].ndim == 4
    assert CANONICAL_KEY not in arrays
    assert payload["record_version"] == RECORD_VERSION == 12
    assert protocol["schema_version"] == 2
    assert protocol["map_count"] == 4
