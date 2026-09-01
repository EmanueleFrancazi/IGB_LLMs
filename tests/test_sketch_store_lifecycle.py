"""The temporary store must never lose a measurement, nor keep one it may delete.

A collection run is hours of GPU time; a finalization run is minutes of CPU. The
store exists to make that asymmetry safe, and almost every test here is about a
failure path rather than a success path:

* an **incomplete** collection is worthless, so its bulk data is removed and the
  manifest survives to say why;
* a **complete** collection whose finalization failed is irreplaceable, so its
  bulk data is kept and the run can be finalized again;
* bulk data is deleted **only** once the results are published and re-read.

The deletion guards get the same attention. Recursive removal driven by a path
read out of a file is exactly how a bookkeeping bug becomes data loss, so the
path is proved to be the one the manifest names -- beneath the configured root,
not a symlink, carrying a sentinel for this run -- before anything is removed.

Nothing here mocks the mathematics or the filesystem; the stores are real, small,
and written to ``tmp_path``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from llm_behavior_lab.evaluation import sketch_store as ss
from llm_behavior_lab.evaluation.sketch_store import (
    SketchStoreLayout,
    StoreState,
    TemporarySketchStore,
)

LAYOUT = SketchStoreLayout(
    num_temperatures=2, num_positions=6, num_maps=3, num_buckets=4
)


def _store(tmp_path: Path, layout: SketchStoreLayout = LAYOUT) -> TemporarySketchStore:
    return TemporarySketchStore.create(
        manifest_dir=tmp_path / "run" / "tmp",
        bulk_dir=tmp_path / "run" / "tmp",
        layout=layout,
        run_id="run-1",
    )


def _fill(store: TemporarySketchStore, layout: SketchStoreLayout = LAYOUT) -> None:
    generator = np.random.default_rng(0)
    for temperature in range(layout.num_temperatures):
        for position in range(layout.num_positions):
            store.write_row(
                temperature,
                position,
                generator.standard_normal(
                    (layout.num_maps, layout.num_buckets)
                ).astype(np.float32),
            )


# -- layout and preflight ----------------------------------------------------


def test_the_byte_estimate_is_exact_not_approximate() -> None:
    layout = SketchStoreLayout(
        num_temperatures=7, num_positions=32768, num_maps=4, num_buckets=1024
    )
    assert ss.estimate_store_bytes(layout) == 7 * 32768 * 4 * 1024 * 4
    assert layout.slab_bytes == 32768 * 1024 * 4
    assert layout.num_slabs == 28


def test_the_estimate_covers_one_map_and_many_maps() -> None:
    single = SketchStoreLayout(
        num_temperatures=7, num_positions=1000, num_maps=1, num_buckets=512
    )
    quad = SketchStoreLayout(
        num_temperatures=7, num_positions=1000, num_maps=4, num_buckets=512
    )
    assert quad.total_bytes == 4 * single.total_bytes


def test_a_store_above_the_byte_limit_is_refused(tmp_path) -> None:
    with pytest.raises(ValueError, match="configured limit"):
        ss.preflight_storage(
            LAYOUT, tmp_path, max_bytes=8, reserve_bytes=0
        )


def test_a_store_that_would_not_fit_on_disk_is_refused(tmp_path, monkeypatch) -> None:
    """Refused before measurement rather than hours in."""

    monkeypatch.setattr(ss, "free_space_bytes", lambda _: 32)
    with pytest.raises(ValueError, match="are free on the filesystem"):
        ss.preflight_storage(LAYOUT, tmp_path, max_bytes=0, reserve_bytes=1024)


def test_the_reserve_and_safety_factor_both_bind(tmp_path, monkeypatch) -> None:
    """The estimate alone is not the requirement: a run that exactly fills the
    disk has left nothing for the record it still has to write."""

    estimate = ss.estimate_store_bytes(LAYOUT)
    monkeypatch.setattr(ss, "free_space_bytes", lambda _: estimate + 8)
    with pytest.raises(ValueError, match="safety factor and reserve"):
        ss.preflight_storage(
            LAYOUT, tmp_path, max_bytes=0, reserve_bytes=4096, safety_factor=1.25
        )


def test_a_viable_store_reports_its_numbers(tmp_path) -> None:
    report = ss.preflight_storage(
        LAYOUT, tmp_path, max_bytes=1 << 30, reserve_bytes=0
    )
    assert report["estimated_bytes"] == LAYOUT.total_bytes
    assert report["free_bytes"] > 0


@pytest.mark.parametrize("field", ["num_temperatures", "num_positions", "num_maps"])
def test_a_degenerate_layout_is_refused(field) -> None:
    values = {
        "num_temperatures": 2, "num_positions": 3, "num_maps": 1, "num_buckets": 4,
    }
    values[field] = 0
    with pytest.raises(ValueError, match="positive integer"):
        SketchStoreLayout(**values)


def test_the_storage_dtype_is_not_a_knob() -> None:
    """float32 quantization is part of the measurement protocol."""

    with pytest.raises(ValueError, match="float32"):
        SketchStoreLayout(
            num_temperatures=1, num_positions=1, num_maps=1, num_buckets=1,
            dtype="float64",
        )


# -- allocation and writing --------------------------------------------------


def test_every_slab_is_preallocated_at_its_exact_size(tmp_path) -> None:
    store = _store(tmp_path)
    paths = sorted(store.directory.glob("sketch_T*_M*.f32"))
    assert len(paths) == LAYOUT.num_slabs
    for path in paths:
        assert path.stat().st_size == LAYOUT.slab_bytes


def test_the_reservation_mode_is_recorded_rather_than_assumed(tmp_path) -> None:
    """``truncate`` may leave a sparse file, so a store must say which it got
    instead of implying a guarantee it did not obtain."""

    store = _store(tmp_path)
    assert store.manifest["reservation"] in {"fallocate", "truncate"}


def test_rows_are_written_by_index_not_appended(tmp_path) -> None:
    """Position order is structural, so an out-of-order caller is still correct."""

    store = _store(tmp_path)
    generator = np.random.default_rng(1)
    rows = {
        position: generator.standard_normal(
            (LAYOUT.num_maps, LAYOUT.num_buckets)
        ).astype(np.float32)
        for position in range(LAYOUT.num_positions)
    }
    for temperature in range(LAYOUT.num_temperatures):
        for position in reversed(range(LAYOUT.num_positions)):
            store.write_row(temperature, position, rows[position])
    store.seal()

    for temperature, index, slab in store.iter_slabs():
        for position in range(LAYOUT.num_positions):
            assert np.array_equal(slab[position], rows[position][index])


def test_a_wrongly_shaped_block_is_refused(tmp_path) -> None:
    store = _store(tmp_path)
    with pytest.raises(ValueError, match="for one position"):
        store.write_row(0, 0, np.zeros((LAYOUT.num_maps + 1, LAYOUT.num_buckets)))


@pytest.mark.parametrize("position", [-1, LAYOUT.num_positions])
def test_an_out_of_range_position_is_refused(tmp_path, position) -> None:
    store = _store(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        store.write_row(
            0, position, np.zeros((LAYOUT.num_maps, LAYOUT.num_buckets), np.float32)
        )


def test_writing_to_a_sealed_store_is_refused(tmp_path) -> None:
    store = _store(tmp_path)
    _fill(store)
    store.seal()
    with pytest.raises(RuntimeError, match="only a collecting store"):
        store.write_row(
            0, 0, np.zeros((LAYOUT.num_maps, LAYOUT.num_buckets), np.float32)
        )


# -- the manifest lives outside the bulk directory ---------------------------


def test_the_manifest_is_a_sibling_of_the_bulk_directory(tmp_path) -> None:
    store = _store(tmp_path)
    assert store.manifest_path.parent == store.directory.parent
    assert store.manifest_path.parent != store.directory


def test_the_manifest_survives_deletion_of_the_bulk_directory(tmp_path) -> None:
    """The directory is what cleanup removes; the manifest is what explains the
    removal. Sharing a fate would destroy the diagnosis with the data."""

    store = _store(tmp_path)
    store.fail_collection("disk gave out")
    assert not store.directory.exists()
    assert store.manifest_path.is_file()
    manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    assert manifest["state"] == StoreState.FAILED_COLLECTION.value
    assert manifest["error"] == "disk gave out"


def test_the_manifest_records_the_separately_resolved_bulk_path(tmp_path) -> None:
    """``--gradient-temp-dir`` may put the bulk data on node-local scratch. If
    that scratch is lost, the run must still say where it went."""

    store = TemporarySketchStore.create(
        manifest_dir=tmp_path / "run" / "tmp",
        bulk_dir=tmp_path / "scratch",
        layout=LAYOUT,
        run_id="run-2",
    )
    assert store.manifest_path.parent == tmp_path / "run" / "tmp"
    assert Path(store.manifest["bulk_directory"]).parent == (tmp_path / "scratch")


# -- sealing -----------------------------------------------------------------


def test_a_sealed_store_records_sizes_rows_and_digests(tmp_path) -> None:
    store = _store(tmp_path)
    _fill(store)
    slabs = store.seal()
    assert store.state is StoreState.COLLECTED
    assert len(slabs) == LAYOUT.num_slabs
    for entry in slabs.values():
        assert entry["bytes"] == LAYOUT.slab_bytes
        assert entry["rows"] == LAYOUT.num_positions
        assert len(entry["sha256"]) == 64


def test_sealing_an_incomplete_store_fails_and_removes_the_rows(tmp_path) -> None:
    """A store missing rows cannot be finalized, so keeping it would only invite
    someone to try."""

    store = _store(tmp_path)
    store.write_row(
        0, 0, np.zeros((LAYOUT.num_maps, LAYOUT.num_buckets), np.float32)
    )
    with pytest.raises(ValueError, match="rows but the layout"):
        store.seal()
    assert store.state is StoreState.FAILED_COLLECTION
    assert not store.directory.exists()


def test_an_unsealed_store_is_not_readable(tmp_path) -> None:
    """Partial writes are possible; the protection is that nothing reads a store
    that has not proved itself complete."""

    store = _store(tmp_path)
    with pytest.raises(RuntimeError, match="not readable"):
        list(store.iter_slabs())


def test_a_corrupted_slab_is_detected_on_validation(tmp_path) -> None:
    store = _store(tmp_path)
    _fill(store)
    store.seal()
    store.validate()

    path = next(store.directory.glob("sketch_T00_M00.f32"))
    blob = bytearray(path.read_bytes())
    blob[0] ^= 0xFF
    path.write_bytes(bytes(blob))
    with pytest.raises(ValueError, match="no longer matches the digest"):
        store.validate()


def test_a_truncated_slab_is_detected_on_validation(tmp_path) -> None:
    store = _store(tmp_path)
    _fill(store)
    store.seal()
    path = next(store.directory.glob("sketch_T00_M00.f32"))
    with open(path, "r+b") as handle:
        handle.truncate(LAYOUT.slab_bytes - 4)
    with pytest.raises(ValueError, match="was sealed at"):
        store.validate()


def test_a_missing_slab_is_detected_on_validation(tmp_path) -> None:
    store = _store(tmp_path)
    _fill(store)
    store.seal()
    next(store.directory.glob("sketch_T00_M00.f32")).unlink()
    with pytest.raises(ValueError, match="is missing"):
        store.validate()


# -- reading -----------------------------------------------------------------


def test_slabs_are_yielded_one_at_a_time_in_order(tmp_path) -> None:
    store = _store(tmp_path)
    _fill(store)
    store.seal()
    seen = [(t, m) for t, m, _ in store.iter_slabs()]
    assert seen == [
        (t, m)
        for t in range(LAYOUT.num_temperatures)
        for m in range(LAYOUT.num_maps)
    ]


def test_a_slab_round_trips_the_float32_rows_exactly(tmp_path) -> None:
    """The projection quantizes to float32; the store must not re-quantize or
    widen anything on the way through."""

    store = _store(tmp_path)
    generator = np.random.default_rng(3)
    written = {}
    for temperature in range(LAYOUT.num_temperatures):
        for position in range(LAYOUT.num_positions):
            block = generator.standard_normal(
                (LAYOUT.num_maps, LAYOUT.num_buckets)
            ).astype(np.float32)
            written[(temperature, position)] = block
            store.write_row(temperature, position, block)
    store.seal()

    for temperature, index, slab in store.iter_slabs():
        assert slab.dtype == np.float32
        for position in range(LAYOUT.num_positions):
            assert np.array_equal(
                slab[position], written[(temperature, position)][index]
            )


# -- the failure asymmetry ---------------------------------------------------


def test_a_finalization_failure_keeps_the_rows(tmp_path) -> None:
    """The property the whole design exists for: collection succeeded, so the
    measurement must survive a downstream failure."""

    store = _store(tmp_path)
    _fill(store)
    store.seal()
    store.begin_finalization()
    store.mark_recoverable("metrics validation failed")

    assert store.state is StoreState.RECOVERABLE
    assert store.directory.exists()
    assert len(list(store.directory.glob("sketch_T*_M*.f32"))) == LAYOUT.num_slabs
    store.validate()


def test_a_collection_failure_removes_the_rows(tmp_path) -> None:
    store = _store(tmp_path)
    store.write_row(
        0, 0, np.zeros((LAYOUT.num_maps, LAYOUT.num_buckets), np.float32)
    )
    store.fail_collection("cuda died")
    assert store.state is StoreState.FAILED_COLLECTION
    assert not store.directory.exists()


def test_rows_may_not_be_discarded_before_publication(tmp_path) -> None:
    """"The results exist somewhere readable" is the only thing that makes
    deleting hours of measurement safe."""

    store = _store(tmp_path)
    _fill(store)
    store.seal()
    with pytest.raises(RuntimeError, match="Only a published run"):
        store.discard()
    assert store.directory.exists()


def test_publication_then_discard_removes_the_rows(tmp_path) -> None:
    store = _store(tmp_path)
    _fill(store)
    store.seal()
    store.begin_finalization()
    store.mark_published()
    store.discard()
    assert store.state is StoreState.DELETED
    assert not store.directory.exists()
    assert store.manifest_path.is_file()


def test_finalizing_an_unsealed_store_is_refused(tmp_path) -> None:
    store = _store(tmp_path)
    with pytest.raises(RuntimeError, match="must be sealed first"):
        store.begin_finalization()


# -- exception paths ---------------------------------------------------------


def test_an_exception_during_collection_removes_the_rows(tmp_path) -> None:
    store = _store(tmp_path)
    with pytest.raises(RuntimeError, match="boom"):
        with store:
            store.write_row(
                0, 0, np.zeros((LAYOUT.num_maps, LAYOUT.num_buckets), np.float32)
            )
            raise RuntimeError("boom")
    assert store.state is StoreState.FAILED_COLLECTION
    assert not store.directory.exists()


def test_a_keyboard_interrupt_during_collection_is_handled_like_any_failure(
    tmp_path,
) -> None:
    """Interrupts arrive during collection as often as errors do, and a store
    abandoned mid-write must not be left looking sealed."""

    store = _store(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        with store:
            store.write_row(
                0, 0, np.zeros((LAYOUT.num_maps, LAYOUT.num_buckets), np.float32)
            )
            raise KeyboardInterrupt
    assert store.state is StoreState.FAILED_COLLECTION
    assert not store.directory.exists()


def test_an_interrupt_after_sealing_keeps_the_rows(tmp_path) -> None:
    store = _store(tmp_path)
    _fill(store)
    store.seal()
    with pytest.raises(KeyboardInterrupt):
        with store:
            raise KeyboardInterrupt
    assert store.state is StoreState.RECOVERABLE
    assert store.directory.exists()


def test_a_clean_exit_leaves_a_sealed_store_alone(tmp_path) -> None:
    store = _store(tmp_path)
    with store:
        _fill(store)
        store.seal()
    assert store.state is StoreState.COLLECTED
    assert store.directory.exists()


# -- deletion safety ---------------------------------------------------------


def test_a_directory_outside_the_root_is_never_deleted(tmp_path) -> None:
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / ss.SENTINEL_NAME).write_text(
        json.dumps({"run_id": "run-1", "store_id": "abc"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="does not lie beneath"):
        ss.remove_store_directory(
            outside, root=tmp_path / "root", run_id="run-1", store_id="abc"
        )
    assert outside.exists()


def test_the_root_itself_is_never_deleted(tmp_path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(ValueError, match="temporary root or a parent"):
        ss.remove_store_directory(root, root=root, run_id="r", store_id="s")
    assert root.exists()


def test_a_symlinked_path_is_never_deleted(tmp_path) -> None:
    """A manifest path that has become a symlink no longer names what was
    recorded, and following it would delete something else entirely."""

    root = tmp_path / "root"
    root.mkdir()
    real = tmp_path / "precious"
    real.mkdir()
    (real / ss.SENTINEL_NAME).write_text(
        json.dumps({"run_id": "r", "store_id": "s"}), encoding="utf-8"
    )
    link = root / "store"
    link.symlink_to(real, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        ss.remove_store_directory(link, root=root, run_id="r", store_id="s")
    assert real.exists()


def test_a_directory_without_a_sentinel_is_never_deleted(tmp_path) -> None:
    root = tmp_path / "root"
    target = root / "store"
    target.mkdir(parents=True)
    with pytest.raises(ValueError, match="sentinel"):
        ss.remove_store_directory(target, root=root, run_id="r", store_id="s")
    assert target.exists()


def test_a_sentinel_naming_another_run_is_never_deleted(tmp_path) -> None:
    """Guards against a reused run id or a manifest edited by hand."""

    root = tmp_path / "root"
    target = root / "store"
    target.mkdir(parents=True)
    (target / ss.SENTINEL_NAME).write_text(
        json.dumps({"run_id": "other", "store_id": "s"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="its sentinel names run"):
        ss.remove_store_directory(target, root=root, run_id="r", store_id="s")
    assert target.exists()


def test_deleting_an_absent_directory_is_not_an_error(tmp_path) -> None:
    """Cleanup runs on paths that may already have been cleaned."""

    assert not ss.remove_store_directory(
        tmp_path / "root" / "gone", root=tmp_path / "root", run_id="r", store_id="s"
    )


# -- stale-store detection ---------------------------------------------------


def test_a_live_store_is_not_reported_stale(tmp_path) -> None:
    store = _store(tmp_path)
    assert ss.find_stale_stores(store.manifest_path.parent) == []


def test_a_dead_process_store_is_reported_with_its_disposition(tmp_path) -> None:
    store = _store(tmp_path)
    _fill(store)
    store.seal()

    manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    manifest["process"]["pid"] = 999_999_999
    manifest["process"]["start_time"] = 1
    store.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    stale = ss.find_stale_stores(store.manifest_path.parent)
    assert len(stale) == 1
    assert stale[0]["state"] == StoreState.COLLECTED.value
    # A sealed store is a complete measurement: reported, never auto-removed.
    assert stale[0]["precious"] is True
    assert stale[0]["disposable"] is False


def test_an_abandoned_incomplete_store_is_marked_disposable(tmp_path) -> None:
    store = _store(tmp_path)
    manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    manifest["process"]["pid"] = 999_999_999
    manifest["process"]["start_time"] = 1
    store.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    stale = ss.find_stale_stores(store.manifest_path.parent)
    assert stale[0]["disposable"] is True
    assert stale[0]["precious"] is False


def test_a_store_from_another_host_is_treated_as_live(tmp_path) -> None:
    """Unknown means live. Deleting a running job's store is far worse than
    leaving a dead one for a human."""

    store = _store(tmp_path)
    manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    manifest["process"]["hostname"] = "some-other-node"
    store.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert ss.find_stale_stores(store.manifest_path.parent) == []


def test_a_store_from_a_previous_boot_is_not_live(tmp_path) -> None:
    store = _store(tmp_path)
    manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    if manifest["process"].get("boot_id") is None:
        pytest.skip("boot id unavailable on this platform")
    manifest["process"]["boot_id"] = "00000000-0000-0000-0000-000000000000"
    store.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert len(ss.find_stale_stores(store.manifest_path.parent)) == 1


def test_an_unreadable_manifest_is_skipped_rather_than_raising(tmp_path) -> None:
    root = tmp_path / "tmp"
    root.mkdir()
    (root / "broken.manifest.json").write_text("{not json", encoding="utf-8")
    assert ss.find_stale_stores(root) == []


# -- reopening ---------------------------------------------------------------


def test_a_sealed_store_can_be_reopened_from_its_manifest(tmp_path) -> None:
    """Resume reads the manifest, not the process that wrote it."""

    store = _store(tmp_path)
    _fill(store)
    store.seal()

    reopened = ss.open_store(store.manifest_path)
    assert reopened.state is StoreState.COLLECTED
    assert reopened.layout == LAYOUT
    reopened.validate()
    assert len(list(reopened.iter_slabs())) == LAYOUT.num_slabs


def test_a_reopened_store_reads_the_same_rows(tmp_path) -> None:
    store = _store(tmp_path)
    _fill(store)
    store.seal()
    original = {(t, m): np.array(slab) for t, m, slab in store.iter_slabs()}

    reopened = ss.open_store(store.manifest_path)
    for temperature, index, slab in reopened.iter_slabs():
        assert np.array_equal(slab, original[(temperature, index)])


# -- the measurement loop talks to a sink, not to storage --------------------
#
# `compute_position_gradient_norms` must not learn about run directories,
# manifests or storage modes. These pin that it emits identical rows whatever
# the destination is, and that a store destination really does avoid the
# [N_T, D, M, K] allocation that made a production arm unscalable.


def _tiny_measurement(row_sink=None, *, maps: int):
    torch = pytest.importorskip("torch")
    from llm_behavior_lab.evaluation.init_distribution import (
        build_evaluation_positions,
    )
    from llm_behavior_lab.evaluation.position_gradients import (
        compute_position_gradient_norms,
    )

    vocab, block, windows, dimension = 12, 4, 2, 6
    tokens = [(index * 5 + 2) % vocab for index in range(64)]

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            generator = torch.Generator().manual_seed(3)
            self.embed = torch.nn.Embedding(vocab, 8)
            self.out = torch.nn.Linear(8, vocab, bias=False)
            with torch.no_grad():
                self.embed.weight.copy_(
                    torch.randn(vocab, 8, generator=generator)
                )
                self.out.weight.copy_(torch.randn(vocab, 8, generator=generator))
            self.double()

        def forward(self, input_ids):
            logits = self.out(self.embed(input_ids))
            return type("Output", (), {"logits": logits})()

    positions = build_evaluation_positions(
        tokens, block_size=block, num_windows=windows
    )
    return compute_position_gradient_norms(
        TinyModel(), positions, vocab_size=vocab,
        temperatures=(0.6, 1.0), gradient_sketch=True,
        sketch_dimension=dimension, sketch_maps=maps, row_sink=row_sink,
    ), (2, windows * block, maps, dimension)


@pytest.mark.parametrize("maps", [1, 4])
def test_a_store_sink_receives_exactly_the_in_memory_rows(tmp_path, maps) -> None:
    """The destination must not change a single value.

    Same seeds, same maps, same order: the store and the in-memory sink are two
    places to put identical bytes, asserted bitwise.
    """

    in_memory, shape = _tiny_measurement(None, maps=maps)
    layout = SketchStoreLayout(
        num_temperatures=shape[0], num_positions=shape[1],
        num_maps=shape[2], num_buckets=shape[3],
    )
    store = TemporarySketchStore.create(
        manifest_dir=tmp_path / "tmp", bulk_dir=tmp_path / "tmp",
        layout=layout, run_id="measure",
    )
    _tiny_measurement(store, maps=maps)

    reference = in_memory.temperature_gradient_sketches.numpy()
    if maps == 1:
        reference = reference[:, :, None, :]
    for temperature, index, slab in store.iter_slabs():
        assert np.array_equal(slab, reference[temperature, :, index, :])


def test_a_store_sink_leaves_no_sketch_arrays_on_the_result(tmp_path) -> None:
    """The point of the store: the 3.5 GiB array is never built.

    A result that still carried the rows would put them straight back into the
    record, which is the cost this whole lifecycle exists to remove.
    """

    layout = SketchStoreLayout(
        num_temperatures=2, num_positions=8, num_maps=4, num_buckets=6
    )
    store = TemporarySketchStore.create(
        manifest_dir=tmp_path / "tmp", bulk_dir=tmp_path / "tmp",
        layout=layout, run_id="measure",
    )
    result, _ = _tiny_measurement(store, maps=4)

    assert result.temperature_gradient_sketches is None
    assert result.gradient_sketches is None
    # The norms, labels and protocol are unaffected -- only the rows moved.
    assert result.temperature_gradient_norms.shape == (2, 8)
    assert result.sketch_protocol["map_count"] == 4


def test_the_store_is_sealed_by_the_measurement(tmp_path) -> None:
    """Sealing is what makes the rows readable, so the measurement must do it
    rather than leaving the caller to remember."""

    layout = SketchStoreLayout(
        num_temperatures=2, num_positions=8, num_maps=2, num_buckets=6
    )
    store = TemporarySketchStore.create(
        manifest_dir=tmp_path / "tmp", bulk_dir=tmp_path / "tmp",
        layout=layout, run_id="measure",
    )
    _tiny_measurement(store, maps=2)
    assert store.state is StoreState.COLLECTED
    store.validate()


def test_a_sink_without_the_sketch_enabled_is_refused(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    from llm_behavior_lab.evaluation.init_distribution import (
        build_evaluation_positions,
    )
    from llm_behavior_lab.evaluation.position_gradients import (
        compute_position_gradient_norms,
    )

    positions = build_evaluation_positions(
        [(i * 5 + 2) % 12 for i in range(64)], block_size=4, num_windows=2
    )
    layout = SketchStoreLayout(
        num_temperatures=2, num_positions=8, num_maps=1, num_buckets=6
    )
    store = TemporarySketchStore.create(
        manifest_dir=tmp_path / "tmp", bulk_dir=tmp_path / "tmp",
        layout=layout, run_id="measure",
    )

    class Bare(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.out = torch.nn.Linear(4, 12, bias=False)

        def forward(self, input_ids):  # pragma: no cover - never reached
            raise AssertionError("measurement started despite the refusal")

    with pytest.raises(ValueError, match="no row would ever be projected"):
        compute_position_gradient_norms(
            Bare(), positions, vocab_size=12, temperatures=(1.0,),
            gradient_sketch=False, row_sink=store,
        )
