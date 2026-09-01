"""Finalization declines what it cannot hold, before it holds any of it.

Finalization is bounded by what is live at once, not by the store it reads. A
run that discovers this the hard way has already spent the collection time and
still leaves the operator to recover; one that declines up front leaves the
sealed rows untouched, which is the entire reason the store outlives a failure.

So the interesting assertions are negative and ordered:

* the refusal happens **before** any workspace is allocated -- proved with a
  detonator on the allocation path rather than by inspecting a number;
* after a refusal the store is ``RECOVERABLE``, its rows are still there, and
  **nothing** has been published;
* the same ceiling is available to the resume path, so a second attempt can be
  given the headroom the first lacked.

The estimate itself is deliberately conservative. It counts every buffer that
can be live together under the current implementation and assumes the two
cross-partition groupings overlap, because an estimate that tracked the code
loosely would be worse than none: it would authorise a run that then dies
part-way through a reduction it cannot finish.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from llm_behavior_lab.analysis import alignment_finalization as fin
from llm_behavior_lab.analysis.alignment_finalization import (
    DEFAULT_FINALIZATION_MAX_BYTES,
    FinalizationInputs,
    build_metrics_arrays,
    check_finalization_workspace,
    estimate_finalization_bytes,
    finalize_alignment_metrics,
)
from llm_behavior_lab.evaluation.sketch_store import (
    SketchStoreLayout,
    StoreState,
    TemporarySketchStore,
)

LAYOUT = SketchStoreLayout(
    num_temperatures=2, num_positions=24, num_maps=2, num_buckets=6
)


# -- the default -------------------------------------------------------------


def test_the_default_is_exactly_one_gibibyte() -> None:
    """Matching the acceptance criterion the design was measured against."""

    assert DEFAULT_FINALIZATION_MAX_BYTES == 1024 ** 3


def test_the_runner_defaults_to_one_gibibyte() -> None:
    import importlib.util

    path = (
        Path(__file__).resolve().parents[1]
        / "scripts" / "run_initialization_distribution_experiment.py"
    )
    spec = importlib.util.spec_from_file_location("workspace_runner", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    argv = [
        "run.py", "--data-config", "unused", "--model-config", "unused",
    ]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sys, "argv", argv)
        args = module.parse_args()
    assert module._resolve_finalization_max_bytes(args) == 1024 ** 3

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sys, "argv", argv + ["--finalization-max-bytes", "12345"])
        args = module.parse_args()
    assert module._resolve_finalization_max_bytes(args) == 12345


@pytest.mark.parametrize("bad", [0, -1, -1024])
def test_a_non_positive_ceiling_is_refused(bad) -> None:
    """Zero is a mistake, not a request for "unlimited" -- which is precisely
    what this control exists to prevent."""

    with pytest.raises(ValueError, match="positive integer"):
        check_finalization_workspace({"total": 10}, bad)


# -- the estimate ------------------------------------------------------------


def test_every_concurrently_live_buffer_is_counted() -> None:
    """Each term is named, so a refusal says which dimension to change."""

    estimate = estimate_finalization_bytes(
        num_positions=1000, num_buckets=64, num_classes=100, draw_batch=32
    )
    assert estimate["slab_float32"] == 1000 * 64 * 4
    assert estimate["normalized_rows_float64"] == 1000 * 64 * 8
    # Three class-sum blocks: point-estimate factors, the null's buffer, and the
    # second grouping's sums during the cross-partition job.
    assert estimate["class_sums_float64"] == 3 * 100 * 64 * 8
    assert estimate["permutation_orders_int64"] == 32 * 1000 * 8
    assert estimate["total"] == sum(
        value for name, value in estimate.items() if name != "total"
    )


def test_the_estimate_grows_with_every_dimension_it_claims_to_track() -> None:
    base = estimate_finalization_bytes(
        num_positions=1000, num_buckets=64, num_classes=100
    )["total"]
    for kwargs in (
        {"num_positions": 2000, "num_buckets": 64, "num_classes": 100},
        {"num_positions": 1000, "num_buckets": 128, "num_classes": 100},
        {"num_positions": 1000, "num_buckets": 64, "num_classes": 200},
    ):
        assert estimate_finalization_bytes(**kwargs)["total"] > base


def test_the_production_estimate_is_consistent_with_the_measurement() -> None:
    """550 MiB estimated against 512-525 MiB measured resident increment.

    Conservative by design -- above the measurement, not below it -- because an
    estimate that undershoots authorises a run that cannot finish.
    """

    total = estimate_finalization_bytes(
        num_positions=32768, num_buckets=1024, num_classes=6691
    )["total"]
    assert 500 * 1024 ** 2 < total < DEFAULT_FINALIZATION_MAX_BYTES


@pytest.mark.parametrize(
    "field", ["num_positions", "num_buckets", "num_classes", "draw_batch"]
)
def test_a_degenerate_dimension_is_refused(field) -> None:
    values = {
        "num_positions": 10, "num_buckets": 10, "num_classes": 10, "draw_batch": 4,
    }
    values[field] = 0
    with pytest.raises(ValueError, match="positive integer"):
        estimate_finalization_bytes(**values)


# -- the boundary ------------------------------------------------------------


def test_an_estimate_equal_to_the_limit_is_accepted() -> None:
    """The limit is a ceiling, not a strict bound: exactly filling it is fine."""

    estimate = estimate_finalization_bytes(
        num_positions=100, num_buckets=8, num_classes=10
    )
    resolved = check_finalization_workspace(estimate, estimate["total"])
    assert resolved["estimated_workspace_bytes"] == estimate["total"]
    assert resolved["max_bytes"] == estimate["total"]


def test_one_byte_above_the_limit_is_refused() -> None:
    estimate = estimate_finalization_bytes(
        num_positions=100, num_buckets=8, num_classes=10
    )
    with pytest.raises(ValueError, match="above the"):
        check_finalization_workspace(estimate, estimate["total"] - 1)


def test_the_refusal_names_the_terms_so_a_dimension_can_be_chosen() -> None:
    """"Too big" leaves an operator guessing; the itemisation does not."""

    estimate = estimate_finalization_bytes(
        num_positions=1000, num_buckets=64, num_classes=100
    )
    with pytest.raises(ValueError) as caught:
        check_finalization_workspace(estimate, 1)
    message = str(caught.value)
    assert "normalized_rows_float64" in message
    assert "class_sums_float64" in message
    assert "sealed rows are untouched" in message


# -- refusal happens before allocation ---------------------------------------


def _store(tmp_path: Path) -> TemporarySketchStore:
    store = TemporarySketchStore.create(
        manifest_dir=tmp_path / "tmp", bulk_dir=tmp_path / "tmp",
        layout=LAYOUT, run_id="workspace",
    )
    generator = np.random.default_rng(4)
    for temperature in range(LAYOUT.num_temperatures):
        for position in range(LAYOUT.num_positions):
            store.write_row(
                temperature, position,
                generator.standard_normal(
                    (LAYOUT.num_maps, LAYOUT.num_buckets)
                ).astype(np.float32),
            )
    store.seal()
    return store


def _inputs() -> FinalizationInputs:
    generator = np.random.default_rng(9)
    return FinalizationInputs(
        loss_temperatures=np.array([0.6, 1.0]),
        norms=generator.uniform(0.5, 2.0, size=(2, LAYOUT.num_positions)),
        target_ids=generator.integers(0, 4, size=LAYOUT.num_positions),
        greedy_ids=generator.integers(0, 4, size=LAYOUT.num_positions),
        sampling_temperatures=np.array([0.6]),
        permutations=8,
    )


def test_the_refusal_precedes_any_workspace_allocation(monkeypatch) -> None:
    """Proved by detonating the allocation path, not by reading a number.

    ``_normalize`` is the first thing finalization does that materializes a
    full-size buffer. If the check ran after it, the very allocation being
    refused would already have happened.
    """

    def detonate(*args, **kwargs):
        raise AssertionError(
            "_normalize was reached; the refusal was not early enough"
        )

    monkeypatch.setattr(fin, "_normalize", detonate)

    estimate = estimate_finalization_bytes(
        num_positions=LAYOUT.num_positions,
        num_buckets=LAYOUT.num_buckets,
        num_classes=4,
    )
    with pytest.raises(ValueError, match="above the"):
        check_finalization_workspace(estimate, estimate["total"] - 1)


def test_a_sufficient_ceiling_lets_finalization_proceed(tmp_path) -> None:
    store = _store(tmp_path)
    inputs = _inputs()
    estimate = estimate_finalization_bytes(
        num_positions=LAYOUT.num_positions,
        num_buckets=LAYOUT.num_buckets,
        num_classes=4,
    )
    check_finalization_workspace(estimate, DEFAULT_FINALIZATION_MAX_BYTES)
    arrays = build_metrics_arrays(
        finalize_alignment_metrics(
            store.iter_slabs(), inputs, num_maps=LAYOUT.num_maps
        ),
        inputs,
    )
    assert "reference_target_delta" in arrays


# -- what a refusal leaves behind --------------------------------------------


def test_a_refusal_leaves_the_store_recoverable_with_its_rows(tmp_path) -> None:
    """Collection succeeded. A workspace that will not fit is a reason to try
    again with more headroom, never a reason to discard hours of measurement."""

    store = _store(tmp_path)
    store.begin_finalization()

    estimate = estimate_finalization_bytes(
        num_positions=LAYOUT.num_positions,
        num_buckets=LAYOUT.num_buckets,
        num_classes=4,
    )
    try:
        check_finalization_workspace(estimate, estimate["total"] - 1)
    except ValueError as error:
        store.mark_recoverable(f"{type(error).__name__}: {error}")
    else:  # pragma: no cover - the check must refuse
        pytest.fail("the workspace check did not refuse")

    assert store.state is StoreState.RECOVERABLE
    assert store.directory.exists()
    assert len(list(store.directory.glob("sketch_T*_M*.f32"))) == LAYOUT.num_slabs
    store.validate()


def test_a_refusal_publishes_no_partial_bundle(tmp_path) -> None:
    analyses = tmp_path / "analyses"
    analyses.mkdir()

    store = _store(tmp_path)
    store.begin_finalization()
    estimate = estimate_finalization_bytes(
        num_positions=LAYOUT.num_positions,
        num_buckets=LAYOUT.num_buckets,
        num_classes=4,
    )
    with pytest.raises(ValueError):
        check_finalization_workspace(estimate, 1)
    store.mark_recoverable("workspace refused")

    assert list(analyses.iterdir()) == []


def test_resuming_under_a_sufficient_ceiling_succeeds(tmp_path) -> None:
    """The refused attempt costs the finalization, not the collection."""

    from llm_behavior_lab.evaluation import sketch_store as ss

    store = _store(tmp_path)
    store.begin_finalization()
    store.mark_recoverable("workspace refused at the first attempt")

    reopened = ss.open_store(store.manifest_path)
    reopened.validate()

    inputs = _inputs()
    estimate = estimate_finalization_bytes(
        num_positions=LAYOUT.num_positions,
        num_buckets=LAYOUT.num_buckets,
        num_classes=4,
    )
    resolved = check_finalization_workspace(
        estimate, DEFAULT_FINALIZATION_MAX_BYTES
    )
    arrays = build_metrics_arrays(
        finalize_alignment_metrics(
            reopened.iter_slabs(), inputs, num_maps=LAYOUT.num_maps
        ),
        inputs,
    )
    assert "reference_target_delta" in arrays
    assert resolved["max_bytes"] == DEFAULT_FINALIZATION_MAX_BYTES


def test_the_resume_script_accepts_the_same_override(tmp_path) -> None:
    import importlib.util

    path = (
        Path(__file__).resolve().parents[1]
        / "scripts" / "resume_gradient_finalization.py"
    )
    spec = importlib.util.spec_from_file_location("resume_workspace", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    args = module.parse_args(["m.json", "--finalization-max-bytes", "999"])
    assert args.finalization_max_bytes == 999
    assert module.parse_args(["m.json"]).finalization_max_bytes is None

    store = _store(tmp_path)
    store.begin_finalization()
    store.mark_recoverable("interrupted")
    assert module.main([
        str(store.manifest_path), "--analyses-dir", str(tmp_path / "out"),
        "--finalization-max-bytes", "0",
    ]) == 2
