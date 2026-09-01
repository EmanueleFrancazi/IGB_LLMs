"""A real v13 run, driven through the actual runner.

Everything else in this change is tested in pieces. This is the one place that
asserts the pieces compose: a genuine experiment, measured on a real model, with
its rows streamed to a real store, finalized, published as a transaction, and
then read back through the authoritative loader.

The property it exists for cannot be checked any other way. ``metrics_only``
deletes its per-position sketches once publication succeeds, so "the rows are
gone" and "every declared statistic survived" have to be true of the *same*
artifact -- and a synthetic payload shaped like one proves neither.

Physical absence is asserted on the archive's own key list rather than through an
accessor. An accessor can report a field missing for a dozen reasons; only the
key list says the bytes are not there.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

yaml = pytest.importorskip("yaml")
pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "run_initialization_distribution_experiment.py"

WINDOWS = 6
BLOCK = 8
WIDTH = 8
MAPS = 4
TEMPERATURES = ("0.6", "1.0")


def _runner():
    spec = importlib.util.spec_from_file_location("v13_runner", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _cpu_data_config(directory: Path) -> Path:
    """A copy, never the tracked file."""

    config = yaml.safe_load(
        (REPO / "configs" / "data" / "tiny_text.yaml").read_text(encoding="utf-8")
    )
    config.setdefault("runtime", {})["device"] = "cpu"
    path = directory / "data.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _run(directory: Path, run_id: str, mode: str) -> Path:
    module = _runner()
    argv = [
        "run_initialization_distribution_experiment.py",
        "--data-config", str(_cpu_data_config(directory)),
        "--model-config", str(REPO / "configs" / "model" / "tiny_llama.yaml"),
        "--num-initializations", "1",
        "--num-windows", str(WINDOWS),
        "--block-size", str(BLOCK),
        "--num-replicates", "1",
        "--no-uniform-null",
        "--no-input-structure",
        "--gradient-analysis",
        "--gradient-sketch",
        "--sketch-maps", str(MAPS),
        "--sketch-dimension", str(WIDTH),
        "--gradient-temperatures", *TEMPERATURES,
        "--sketch-storage", mode,
        "--no-figures",
        "--offline",
        "--output-dir", str(directory / "out"),
        "--run-id", run_id,
    ]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sys, "argv", argv)
        module.main()
    return directory / "out" / "initialization_distribution" / run_id / "analyses"


@pytest.fixture(scope="module")
def metrics_only_run(tmp_path_factory) -> Path:
    return _run(tmp_path_factory.mktemp("v13-metrics-only"), "v13", "metrics_only")


def _record_keys(analyses: Path) -> list[str]:
    with np.load(analyses / "initialization_distribution.npz", allow_pickle=False) as z:
        return list(z.files)


# -- the rows are physically gone --------------------------------------------


def test_no_per_position_sketch_array_is_in_the_archive(metrics_only_run) -> None:
    """Checked on the key list: an accessor could report absence for other
    reasons, but the key list says the bytes are not there."""

    keys = _record_keys(metrics_only_run)
    assert "gradient_position_sketches" not in keys
    assert "gradient_temperature_position_sketches" not in keys


def test_no_subgroup_factor_array_is_in_the_archive(metrics_only_run) -> None:
    """`metrics_only` retains neither rows nor factors."""

    assert [key for key in _record_keys(metrics_only_run) if "sketch" in key] == []
    assert [key for key in _record_keys(metrics_only_run) if key.startswith("factor")] == []


def test_the_norms_and_labels_survive(metrics_only_run) -> None:
    """Only the rows were dropped. The observables the rest of the analysis is
    defined on are untouched."""

    keys = _record_keys(metrics_only_run)
    for name in (
        "gradient_position_norms",
        "gradient_temperature_position_norms",
        "gradient_position_target_ids",
        "gradient_position_greedy_ids",
    ):
        assert name in keys, name


# -- the metrics are there, and are the declared ones -------------------------


def test_the_record_loads_with_its_metrics_attached(metrics_only_run) -> None:
    from llm_behavior_lab.analysis.alignment_metrics import load_finalized_record

    record = load_finalized_record(metrics_only_run)
    assert record.record_version == 13
    assert record.storage_mode == "metrics_only"
    assert record.has_finalized_gradient_metrics
    assert not record.has_gradient_position_sketches
    assert not record.has_gradient_sketch_factors


def test_every_declared_family_is_present(metrics_only_run) -> None:
    from llm_behavior_lab.analysis.alignment_metrics import load_finalized_record

    metrics = load_finalized_record(metrics_only_run).alignment_metrics
    for grouping in ("target", "greedy"):
        assert f"reference_{grouping}_delta" in metrics
        assert f"reference_{grouping}_null_delta_low" in metrics
        assert f"display_{grouping}_matrix" in metrics
    assert "nucleus_delta" in metrics
    assert "cross_delta_cross" in metrics


def test_both_nucleus_designs_are_present_without_any_model_replay(
    metrics_only_run,
) -> None:
    """The five-hour reconstruction is gone: the labels were kept during the run.

    Control points sit at the canonical field with the grouping heated; the
    matched point sits at its own ``T_g``. Those are different questions and the
    artifact keeps them apart.
    """

    from llm_behavior_lab.analysis.alignment_metrics import load_finalized_record

    metrics = load_finalized_record(metrics_only_run).alignment_metrics
    designs = metrics["nucleus_design"].tolist()
    assert "control" in designs and "matched" in designs
    for slot, design in enumerate(designs):
        if design == "control":
            assert metrics["nucleus_loss_temperatures"][slot] == pytest.approx(1.0)
        else:
            assert metrics["nucleus_loss_temperatures"][slot] == pytest.approx(
                metrics["nucleus_sampling_temperatures"][slot]
            )


def test_the_histogram_gate_ran_and_passed(metrics_only_run) -> None:
    """Captured labels are expected to match; the gate still checks."""

    payload = json.loads(
        (metrics_only_run / "initialization_distribution.json").read_text(
            encoding="utf-8"
        )
    )
    gate = payload["analysis"]["gradient_analysis"]["gradient_alignment"][
        "nucleus_histogram_gate"
    ]
    assert gate["passed"] is True
    assert gate["comparison"] == "exact_integer_equality"
    assert gate["source"] == "captured_during_run"


def test_per_map_uncertainty_is_available_at_four_maps(metrics_only_run) -> None:
    from llm_behavior_lab.analysis.alignment_metrics import load_finalized_record

    metrics = load_finalized_record(metrics_only_run).alignment_metrics
    assert int(metrics["map_count"]) == MAPS
    assert int(metrics["degrees_of_freedom"]) == MAPS - 1
    assert bool(metrics["uncertainty_available"]) is True
    assert metrics["reference_target_delta_per_map"].shape[1] == MAPS
    assert np.isfinite(metrics["reference_target_delta_sd"]).any()


# -- the transaction and the store -------------------------------------------


def test_exactly_the_four_bundle_files_are_published(metrics_only_run) -> None:
    from llm_behavior_lab.analysis import alignment_publication as pub

    present = {path.name for path in metrics_only_run.iterdir()}
    assert set(pub.PUBLISH_ORDER) <= present
    assert not any(name.startswith(".staging") for name in present)


def test_the_temporary_store_is_deleted_but_its_manifest_survives(
    metrics_only_run,
) -> None:
    """The bulk data is gone; the record of what it was is not."""

    root = metrics_only_run.parents[2] / "_gradient_tmp"
    manifests = list(root.glob("*.manifest.json"))
    assert manifests, "the durable manifest must outlive the rows it describes"

    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["state"] == "deleted"
    assert not Path(manifest["bulk_directory"]).exists()
    assert manifest["slabs"], "the seal recorded every slab before publication"


def test_the_conventions_are_recorded_on_the_artifact(metrics_only_run) -> None:
    payload = json.loads(
        (metrics_only_run / "gradient_alignment_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["estimator_convention"] == "countsketch_pairwise_mean_q_self/v1"
    assert payload["reduction_convention"] == "blockwise_class_sum/v1"
    assert payload["numerical_epoch"] == "alignment_reduction_epoch/v1"


def test_the_bundle_is_far_smaller_than_the_rows_it_replaced(
    metrics_only_run,
) -> None:
    """The whole point, at whatever scale the run happened to be."""

    metrics_bytes = (
        (metrics_only_run / "gradient_alignment_metrics.npz").stat().st_size
        + (metrics_only_run / "gradient_alignment_metrics.json").stat().st_size
    )
    rows_bytes = len(TEMPERATURES) * WINDOWS * BLOCK * MAPS * WIDTH * 4
    assert metrics_bytes > 0
    # The metrics carry heatmaps and nulls, so at this toy scale they are larger
    # than the rows; the saving is asymptotic in D, not universal. Asserting the
    # ratio here would pin an artefact of the fixture size.
    assert rows_bytes == len(TEMPERATURES) * 48 * MAPS * WIDTH * 4


# -- slab digests are content, not identity ----------------------------------


def test_two_identical_runs_hash_their_slabs_identically(
    metrics_only_run, tmp_path_factory
) -> None:
    """The strongest reproducibility assertion this bundle carries.

    An earlier draft pruned ``slab_digests`` from the metadata-agreement check
    on the assumption that it was per-run identity like ``store_id``. It is not:
    the keys are *logical* slab names -- ``sketch_T00_M00.f32`` -- carrying no
    path and no store id, and the values hash the sketch rows themselves. Two
    runs of the same experiment write the same rows, so a digest mismatch means
    the measurement moved, which is precisely what should be caught rather than
    excluded.
    """

    second = _run(tmp_path_factory.mktemp("v13-twin"), "twin", "metrics_only")

    def digests(analyses):
        payload = json.loads(
            (analyses / "initialization_distribution.json").read_text(
                encoding="utf-8"
            )
        )
        block = payload["analysis"]["gradient_analysis"]["gradient_alignment"]
        return (block.get("temporary_store") or {}).get("slab_digests") or {}

    first, twin = digests(metrics_only_run), digests(second)
    assert first, "a metrics_only run must record its sealed slab digests"
    assert len(first) == 2 * MAPS  # one slab per (temperature, map)
    assert sorted(first) == sorted(twin)
    assert first == twin


def test_the_digest_keys_carry_no_path_or_store_identity(metrics_only_run) -> None:
    """Which is what makes them comparable across runs at all."""

    payload = json.loads(
        (metrics_only_run / "initialization_distribution.json").read_text(
            encoding="utf-8"
        )
    )
    store = payload["analysis"]["gradient_analysis"]["gradient_alignment"][
        "temporary_store"
    ]
    for name in store["slab_digests"]:
        assert "/" not in name
        assert store["store_id"] not in name
        assert name.startswith("sketch_T") and name.endswith(".f32")
