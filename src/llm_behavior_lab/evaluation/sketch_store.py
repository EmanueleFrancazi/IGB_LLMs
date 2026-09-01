"""The run-scoped temporary store for per-position gradient sketches.

A production arm projects ``N_T * D * M`` sketch rows. Held in memory or written
into the record they are about 3.5 GiB, which is what makes the current lifecycle
unscalable. They are needed only until finalization has reduced them to metrics,
so this module gives them somewhere bounded to live in between, and takes
responsibility for their being gone afterwards.

Two properties drive every design choice here.

**Measurement is expensive; finalization is not.** A collection run is hours of
GPU time. A finalization failure must therefore never destroy the rows -- the
store survives so the run can be finalized again -- while an *incomplete*
collection is worthless and is cleaned up immediately. That asymmetry is the
state machine below, and it is the reason the durable manifest lives beside the
bulk directory rather than inside it: deleting the data must never delete the
record of why.

**Only a sealed store may be read.** Writes go into preallocated slabs by row
index, and ``memmap`` assignment is not crash-atomic, so a half-written store can
contain anything. Nothing guards against that at read time because nothing needs
to: a store is readable only after :meth:`TemporarySketchStore.seal` has checked
every slab's size and row count and recorded its digest.

The layout is one slab per ``(temperature, map)`` pair, each a flat ``[D, K]``
float32 stream. One loss temperature of one map is what finalization loads at a
time, so a slab is exactly the unit of work -- 128 MiB at experiment scale,
against 3.5 GiB for the whole store.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

import numpy as np

__all__ = [
    "SENTINEL_NAME",
    "StoreState",
    "InMemoryRowSink",
    "SketchStoreLayout",
    "TemporarySketchStore",
    "estimate_store_bytes",
    "find_stale_stores",
    "free_space_bytes",
    "preflight_storage",
    "open_store",
    "remove_store_directory",
]

#: Marks a directory as ours. Cleanup refuses to delete a path without it, so a
#: manifest pointing somewhere unexpected -- a hand-edited path, a reused run id,
#: a symlink -- cannot turn into a recursive delete of something else.
SENTINEL_NAME = ".gradient_sketch_store"

#: Rows flushed between manifest updates. Small enough that a crash loses little
#: bookkeeping, large enough that the manifest is not rewritten per position.
DEFAULT_FLUSH_EVERY = 1024


class StoreState(str, Enum):
    """Lifecycle of one store. Persisted, so it survives the process."""

    CREATING = "creating"
    COLLECTING = "collecting"
    #: Sealed and validated. The only state a reader may consume.
    COLLECTED = "collected"
    FINALIZING = "finalizing"
    #: Collection succeeded, finalization or publication did not. The bulk data
    #: is **kept**: it represents hours of measurement and can be finalized
    #: again.
    RECOVERABLE = "recoverable"
    #: Collection itself failed. The bulk data is incomplete and worthless, so it
    #: is removed; the manifest stays to explain why.
    FAILED_COLLECTION = "failed_collection"
    PUBLISHED = "published"
    #: Published and cleaned up. Manifest only.
    DELETED = "deleted"


#: States whose bulk data may be removed without losing anything irreplaceable.
_DISPOSABLE = frozenset(
    {StoreState.CREATING, StoreState.COLLECTING, StoreState.FAILED_COLLECTION,
     StoreState.PUBLISHED, StoreState.DELETED}
)

#: States that hold a complete measurement. Never removed automatically.
_PRECIOUS = frozenset({StoreState.COLLECTED, StoreState.FINALIZING,
                       StoreState.RECOVERABLE})


@dataclass(frozen=True)
class SketchStoreLayout:
    """Shapes the store is built for. Fixed at creation and never inferred."""

    num_temperatures: int
    num_positions: int
    num_maps: int
    num_buckets: int
    dtype: str = "float32"

    def __post_init__(self) -> None:
        for name in ("num_temperatures", "num_positions", "num_maps", "num_buckets"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer; got {value!r}.")
        if self.dtype != "float32":
            raise ValueError(
                f"Sketch rows are stored as float32; got {self.dtype!r}. The "
                "quantization is part of the measurement protocol, not a knob."
            )

    @property
    def itemsize(self) -> int:
        return int(np.dtype(self.dtype).itemsize)

    @property
    def slab_bytes(self) -> int:
        """One ``(temperature, map)`` slab: ``D * K`` values."""

        return self.num_positions * self.num_buckets * self.itemsize

    @property
    def num_slabs(self) -> int:
        return self.num_temperatures * self.num_maps

    @property
    def total_bytes(self) -> int:
        return self.slab_bytes * self.num_slabs


def estimate_store_bytes(layout: SketchStoreLayout) -> int:
    """Exact byte count the store will occupy. Not an approximation."""

    return layout.total_bytes


def free_space_bytes(directory: str | Path) -> int:
    """Free bytes on the filesystem holding ``directory`` (or its nearest parent)."""

    path = Path(directory)
    while not path.exists() and path != path.parent:
        path = path.parent
    return int(shutil.disk_usage(path).free)


def preflight_storage(
    layout: SketchStoreLayout,
    directory: str | Path,
    *,
    max_bytes: int,
    reserve_bytes: int,
    safety_factor: float = 1.25,
) -> dict[str, Any]:
    """Refuse an impossible store **before** anything expensive happens.

    Two independent gates, because either alone is insufficient: a configured
    byte limit expresses what the operator is willing to spend, and free space
    expresses what the machine can actually provide. A run that passes one and
    fails the other must not start.

    Raises:
        ValueError: If the store would exceed ``max_bytes``, or if free space
            would not cover ``safety_factor`` times the estimate plus
            ``reserve_bytes``.
    """

    estimate = estimate_store_bytes(layout)
    if max_bytes > 0 and estimate > max_bytes:
        raise ValueError(
            f"The temporary sketch store would need {estimate:,} bytes "
            f"({estimate / 1024 ** 3:.2f} GiB) but the configured limit is "
            f"{max_bytes:,}. Raise the limit deliberately, reduce the sketch "
            "width, the map count or the position count, or choose a storage "
            "mode that does not retain rows."
        )
    free = free_space_bytes(directory)
    required = int(estimate * safety_factor) + int(reserve_bytes)
    if free < required:
        raise ValueError(
            f"The temporary sketch store needs about {estimate:,} bytes and "
            f"{required:,} including the safety factor and reserve, but only "
            f"{free:,} are free on the filesystem holding {directory}. "
            "Refusing before measurement rather than failing hours in."
        )
    return {
        "estimated_bytes": estimate,
        "free_bytes": free,
        "required_bytes": required,
        "max_bytes": int(max_bytes),
        "reserve_bytes": int(reserve_bytes),
        "safety_factor": float(safety_factor),
    }


def _process_identity() -> dict[str, Any]:
    """Enough to tell "this process" from "a different process, same PID".

    A PID alone is not identity: they are reused, and a stale manifest whose PID
    now belongs to something else would otherwise look alive -- or, worse, a
    live store would look stale and be deleted. Host, boot id and process start
    time together make reuse detectable.
    """

    identity: dict[str, Any] = {"pid": os.getpid(), "hostname": socket.gethostname()}
    try:
        identity["boot_id"] = (
            Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
        )
    except OSError:
        identity["boot_id"] = None
    try:
        stat = Path(f"/proc/{os.getpid()}/stat").read_text(encoding="utf-8")
        identity["start_time"] = int(stat.rsplit(")", 1)[1].split()[19])
    except (OSError, IndexError, ValueError):
        identity["start_time"] = None
    return identity


def _process_is_live(identity: dict[str, Any]) -> bool:
    """Whether the process a manifest names is still running.

    **Unknown means live.** If the host differs, or the boot id or start time
    were unavailable when the manifest was written, this cannot tell reuse from
    survival -- and deleting a live run's store is far worse than leaving a dead
    one's behind for a human to remove.
    """

    current = _process_identity()
    if identity.get("hostname") != current["hostname"]:
        return True
    if identity.get("boot_id") is None or current["boot_id"] is None:
        return True
    if identity.get("boot_id") != current["boot_id"]:
        # A different boot on this host: nothing from before it is running.
        return False
    pid = identity.get("pid")
    if not isinstance(pid, int):
        return True
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        start = int(stat.rsplit(")", 1)[1].split()[19])
    except (OSError, IndexError, ValueError):
        return False
    recorded = identity.get("start_time")
    if recorded is None:
        return True
    return start == recorded


def remove_store_directory(
    path: str | Path, *, root: str | Path, run_id: str, store_id: str
) -> bool:
    """Delete a store directory, after proving it is the one the manifest names.

    Recursive deletion driven by a path read from a file is exactly the
    operation that turns a bookkeeping bug into data loss, so every one of these
    must hold before anything is removed:

    * the resolved path lies beneath the resolved temporary root;
    * it is not the root itself, nor any parent of it;
    * neither it nor any component above it is a symlink;
    * it carries a sentinel naming this run and this store.

    Returns:
        Whether a directory was removed. A missing directory is not an error --
        cleanup is idempotent by design, because it runs on paths that may
        already have been cleaned.
    """

    target = Path(path)
    root_path = Path(root).resolve()
    if not target.exists():
        return False

    resolved = target.resolve()
    if resolved == root_path or resolved in root_path.parents:
        raise ValueError(
            f"Refusing to delete {resolved}: it is the temporary root or a "
            "parent of it."
        )
    if root_path not in resolved.parents:
        raise ValueError(
            f"Refusing to delete {resolved}: it does not lie beneath the "
            f"configured temporary root {root_path}."
        )
    probe = target
    while True:
        if probe.is_symlink():
            raise ValueError(
                f"Refusing to delete {target}: {probe} is a symlink, so the "
                "path does not necessarily name what the manifest recorded."
            )
        if probe.resolve() == root_path:
            break
        parent = probe.parent
        if parent == probe:
            break
        probe = parent

    sentinel = target / SENTINEL_NAME
    if not sentinel.is_file():
        raise ValueError(
            f"Refusing to delete {target}: no {SENTINEL_NAME} sentinel, so this "
            "directory was not created by this module."
        )
    marker = json.loads(sentinel.read_text(encoding="utf-8"))
    if marker.get("run_id") != run_id or marker.get("store_id") != store_id:
        raise ValueError(
            f"Refusing to delete {target}: its sentinel names run "
            f"{marker.get('run_id')!r} / store {marker.get('store_id')!r}, not "
            f"{run_id!r} / {store_id!r}."
        )
    shutil.rmtree(target)
    return True


def find_stale_stores(manifest_root: str | Path) -> list[dict[str, Any]]:
    """Manifests in ``manifest_root`` whose process is no longer running.

    Reports rather than acts. Whether a stale store may be deleted depends on
    its state, and that decision belongs to the caller: a stale ``COLLECTED``
    store is hours of measurement waiting to be finalized, not litter.
    """

    root = Path(manifest_root)
    if not root.is_dir():
        return []
    stale = []
    for path in sorted(root.glob("*.manifest.json")):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        identity = manifest.get("process", {})
        if _process_is_live(identity):
            continue
        state = manifest.get("state")
        manifest["manifest_path"] = str(path)
        manifest["disposable"] = state in {value.value for value in _DISPOSABLE}
        manifest["precious"] = state in {value.value for value in _PRECIOUS}
        stale.append(manifest)
    return stale


class TemporarySketchStore:
    """Preallocated slabs plus a durable manifest, with an explicit lifecycle.

    Created by :meth:`create`, written through :meth:`write_row`, closed by
    :meth:`seal`, read through :meth:`iter_slabs`, and removed by
    :meth:`discard` only once its results are safely published.

    The manifest is a **sibling file**, not a member of the bulk directory. That
    is deliberate: the directory is what cleanup deletes, and the manifest is
    what explains the deletion, so the two must not share a fate.
    """

    def __init__(
        self,
        *,
        manifest_path: Path,
        directory: Path,
        root: Path,
        layout: SketchStoreLayout,
        run_id: str,
        store_id: str,
        manifest: dict[str, Any],
    ) -> None:
        self.manifest_path = manifest_path
        self.directory = directory
        self.root = root
        self.layout = layout
        self.run_id = run_id
        self.store_id = store_id
        self._manifest = manifest
        self._slabs: dict[tuple[int, int], np.memmap] = {}
        self._written = np.zeros(
            (layout.num_temperatures, layout.num_maps), dtype=np.int64
        )
        self._since_flush = 0

    # -- construction --------------------------------------------------------

    @classmethod
    def create(
        cls,
        *,
        manifest_dir: str | Path,
        bulk_dir: str | Path,
        layout: SketchStoreLayout,
        run_id: str,
        preflight: dict[str, Any] | None = None,
        checksum: bool = True,
    ) -> "TemporarySketchStore":
        """Allocate the slabs and write the manifest.

        ``manifest_dir`` and ``bulk_dir`` are separate so that ``--gradient-temp-dir``
        can move the bulk data onto node-local scratch while the manifest stays
        in the run directory. Losing the scratch then remains diagnosable from
        the run alone, which is the whole point of splitting them.
        """

        manifest_root = Path(manifest_dir)
        manifest_root.mkdir(parents=True, exist_ok=True)
        bulk_root = Path(bulk_dir)
        bulk_root.mkdir(parents=True, exist_ok=True)

        store_id = uuid.uuid4().hex[:12]
        stem = f"gradient_sketches.{run_id}.{store_id}"
        directory = bulk_root / stem
        manifest_path = manifest_root / f"{stem}.manifest.json"
        directory.mkdir(parents=False, exist_ok=False)
        (directory / SENTINEL_NAME).write_text(
            json.dumps({"run_id": run_id, "store_id": store_id}), encoding="utf-8"
        )

        manifest: dict[str, Any] = {
            "run_id": run_id,
            "store_id": store_id,
            "state": StoreState.CREATING.value,
            "bulk_directory": str(directory.resolve()),
            "bulk_root": str(bulk_root.resolve()),
            "layout": {
                "num_temperatures": layout.num_temperatures,
                "num_positions": layout.num_positions,
                "num_maps": layout.num_maps,
                "num_buckets": layout.num_buckets,
                "dtype": layout.dtype,
                "slab_bytes": layout.slab_bytes,
                "total_bytes": layout.total_bytes,
            },
            "preflight": preflight or {},
            "process": _process_identity(),
            "checksum_enabled": bool(checksum),
            "written_rows": [[0] * layout.num_maps
                             for _ in range(layout.num_temperatures)],
            "reservation": None,
            "slabs": {},
            "error": None,
        }

        store = cls(
            manifest_path=manifest_path,
            directory=directory,
            root=bulk_root,
            layout=layout,
            run_id=run_id,
            store_id=store_id,
            manifest=manifest,
        )
        store._write_manifest()
        store._allocate()
        store._set_state(StoreState.COLLECTING)
        return store

    def _allocate(self) -> None:
        """Preallocate every slab at its exact final size.

        ``posix_fallocate`` reserves physical blocks where the platform and
        filesystem support it. ``truncate`` establishes only the logical length
        and may leave a sparse file, so on those systems the free-space
        preflight remains the only protection and ``ENOSPC`` can still surface
        mid-write. The manifest records which of the two happened rather than
        implying a guarantee that was not obtained.
        """

        reservation = "fallocate"
        for temperature in range(self.layout.num_temperatures):
            for index in range(self.layout.num_maps):
                path = self._slab_path(temperature, index)
                with open(path, "wb") as handle:
                    try:
                        os.posix_fallocate(handle.fileno(), 0, self.layout.slab_bytes)
                    except (AttributeError, OSError):
                        handle.truncate(self.layout.slab_bytes)
                        reservation = "truncate"
        self._manifest["reservation"] = reservation
        self._write_manifest()

    def _slab_path(self, temperature: int, map_index: int) -> Path:
        return self.directory / f"sketch_T{temperature:02d}_M{map_index:02d}.f32"

    # -- manifest ------------------------------------------------------------

    def _write_manifest(self) -> None:
        descriptor = self.manifest_path.with_suffix(".tmp")
        descriptor.write_text(
            json.dumps(self._manifest, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        os.replace(descriptor, self.manifest_path)

    def _set_state(self, state: StoreState, *, error: str | None = None) -> None:
        self._manifest["state"] = state.value
        if error is not None:
            self._manifest["error"] = error
        self._write_manifest()

    @property
    def state(self) -> StoreState:
        return StoreState(self._manifest["state"])

    @property
    def manifest(self) -> dict[str, Any]:
        return dict(self._manifest)

    # -- writing -------------------------------------------------------------

    def _slab(self, temperature: int, map_index: int) -> np.memmap:
        key = (temperature, map_index)
        handle = self._slabs.get(key)
        if handle is None:
            handle = np.memmap(
                self._slab_path(temperature, map_index),
                dtype=self.layout.dtype,
                mode="r+",
                shape=(self.layout.num_positions, self.layout.num_buckets),
            )
            self._slabs[key] = handle
        return handle

    def write_row(
        self, temperature_index: int, position: int, rows: np.ndarray
    ) -> None:
        """Store one position's ``[M, K]`` sketch rows.

        Written **by index**, never appended, so a row's location is structural
        rather than a consequence of call order. The rows arrive already
        quantized to float32 by the projection path; nothing is re-quantized
        here.
        """

        if self.state is not StoreState.COLLECTING:
            raise RuntimeError(
                f"Cannot write to a store in state {self.state.value!r}; only a "
                "collecting store accepts rows."
            )
        values = np.asarray(rows)
        expected = (self.layout.num_maps, self.layout.num_buckets)
        if values.shape != expected:
            raise ValueError(
                f"Expected a {expected} block for one position; got {values.shape}."
            )
        if not 0 <= position < self.layout.num_positions:
            raise ValueError(
                f"position {position} is outside [0, {self.layout.num_positions})."
            )
        if not 0 <= temperature_index < self.layout.num_temperatures:
            raise ValueError(
                f"temperature_index {temperature_index} is outside "
                f"[0, {self.layout.num_temperatures})."
            )

        for index in range(self.layout.num_maps):
            self._slab(temperature_index, index)[position] = values[index]
            self._written[temperature_index, index] += 1

        self._since_flush += 1
        if self._since_flush >= DEFAULT_FLUSH_EVERY:
            self.flush()

    def flush(self) -> None:
        """Push buffered rows to disk and record progress in the manifest."""

        for handle in self._slabs.values():
            handle.flush()
        self._manifest["written_rows"] = self._written.tolist()
        self._write_manifest()
        self._since_flush = 0

    # -- sealing -------------------------------------------------------------

    def seal(self) -> dict[str, Any]:
        """Close collection and prove the store is complete.

        Every slab must be exactly its declared size and must have received
        exactly ``D`` rows. Only after that does the store become readable --
        which is what lets every reader skip the question of partial writes
        entirely.
        """

        if self.state is not StoreState.COLLECTING:
            raise RuntimeError(
                f"Cannot seal a store in state {self.state.value!r}."
            )
        for handle in self._slabs.values():
            handle.flush()
        self._slabs.clear()

        from llm_behavior_lab.analysis.alignment_metrics import sha256_of_file

        slabs: dict[str, Any] = {}
        for temperature in range(self.layout.num_temperatures):
            for index in range(self.layout.num_maps):
                path = self._slab_path(temperature, index)
                size = path.stat().st_size
                if size != self.layout.slab_bytes:
                    self.fail_collection(
                        f"{path.name} is {size} bytes, expected "
                        f"{self.layout.slab_bytes}."
                    )
                    raise ValueError(
                        f"{path} is {size} bytes but the layout requires "
                        f"{self.layout.slab_bytes}; the store is incomplete."
                    )
                written = int(self._written[temperature, index])
                if written != self.layout.num_positions:
                    self.fail_collection(
                        f"{path.name} received {written} of "
                        f"{self.layout.num_positions} rows."
                    )
                    raise ValueError(
                        f"{path.name} received {written} rows but the layout "
                        f"declares {self.layout.num_positions}; the store is "
                        "incomplete and will not be read."
                    )
                entry: dict[str, Any] = {"bytes": size, "rows": written}
                if self._manifest["checksum_enabled"]:
                    entry["sha256"] = sha256_of_file(path)
                slabs[path.name] = entry

        self._manifest["slabs"] = slabs
        self._manifest["written_rows"] = self._written.tolist()
        self._set_state(StoreState.COLLECTED)
        return dict(slabs)

    # -- reading -------------------------------------------------------------

    def validate(self) -> None:
        """Re-check a sealed store's slabs before consuming them."""

        if self.state not in _PRECIOUS:
            raise RuntimeError(
                f"A store in state {self.state.value!r} is not readable; only a "
                "sealed store is."
            )
        from llm_behavior_lab.analysis.alignment_metrics import sha256_of_file

        for name, entry in self._manifest["slabs"].items():
            path = self.directory / name
            if not path.is_file():
                raise ValueError(f"Sealed slab {path} is missing.")
            size = path.stat().st_size
            if size != entry["bytes"]:
                raise ValueError(
                    f"{path} is {size} bytes but was sealed at {entry['bytes']}."
                )
            if "sha256" in entry and sha256_of_file(path) != entry["sha256"]:
                raise ValueError(
                    f"{path} no longer matches the digest recorded when it was "
                    "sealed; the slab has been modified or corrupted."
                )

    def iter_slabs(self) -> Iterator[tuple[int, int, np.ndarray]]:
        """Yield ``(temperature_index, map_index, [D, K] float32)`` one at a time.

        One slab is resident at a time and is released before the next is
        opened, which is what keeps finalization linear in ``D`` rather than in
        the whole store.
        """

        if self.state not in _PRECIOUS:
            raise RuntimeError(
                f"A store in state {self.state.value!r} is not readable."
            )
        for temperature in range(self.layout.num_temperatures):
            for index in range(self.layout.num_maps):
                handle = np.memmap(
                    self._slab_path(temperature, index),
                    dtype=self.layout.dtype,
                    mode="r",
                    shape=(self.layout.num_positions, self.layout.num_buckets),
                )
                try:
                    yield temperature, index, handle
                finally:
                    del handle

    # -- lifecycle transitions ----------------------------------------------

    def begin_finalization(self) -> None:
        if self.state is not StoreState.COLLECTED:
            raise RuntimeError(
                f"Cannot finalize from state {self.state.value!r}; the store "
                "must be sealed first."
            )
        self._set_state(StoreState.FINALIZING)

    def fail_collection(self, error: str) -> None:
        """Collection failed: the rows are incomplete, so remove them.

        The manifest survives with the error. Nothing irreplaceable is lost,
        because a partially written store cannot be finalized anyway.
        """

        self._close_handles()
        self._set_state(StoreState.FAILED_COLLECTION, error=error)
        self._remove_bulk()

    def mark_recoverable(self, error: str) -> None:
        """Finalization or publication failed: **keep** the rows.

        This is the state the whole design exists to make possible. Collection
        succeeded, so the store holds hours of measurement that can be finalized
        again once the failure is understood.
        """

        self._close_handles()
        self._set_state(StoreState.RECOVERABLE, error=error)

    def discard(self) -> None:
        """Remove the bulk data after its results are safely published.

        Refuses unless the store has been marked published, because "the results
        exist somewhere readable" is the only thing that makes deleting the rows
        safe.
        """

        if self.state is not StoreState.PUBLISHED:
            raise RuntimeError(
                f"Refusing to discard a store in state {self.state.value!r}. "
                "Only a published run's rows are safe to delete; mark the store "
                "published once the record and its metrics have been re-read."
            )
        self._close_handles()
        self._remove_bulk()
        self._set_state(StoreState.DELETED)

    def mark_published(self) -> None:
        self._set_state(StoreState.PUBLISHED)

    def compact_manifest(self) -> dict[str, Any]:
        """A self-contained summary, free of machine-specific absolute paths.

        The working manifest names a bulk directory that may sit on node-local
        scratch, and after a successful run that directory is gone. Embedding
        *this* in the published record instead keeps a completed run portable:
        it can be copied to another machine and still say exactly what was
        collected, how much of it there was, and how it was reserved -- without
        depending on a path that no longer resolves anywhere.

        Failed and recoverable stores keep their full manifest under the
        temporary root, where the absolute path is still the useful thing.
        """

        layout = self._manifest["layout"]
        return {
            "store_id": self.store_id,
            "run_id": self.run_id,
            "state": self._manifest["state"],
            "reservation": self._manifest["reservation"],
            "checksum_enabled": self._manifest["checksum_enabled"],
            "layout": dict(layout),
            "num_slabs": len(self._manifest["slabs"]),
            "slab_digests": {
                name: entry.get("sha256")
                for name, entry in self._manifest["slabs"].items()
            },
            "preflight": {
                key: value
                for key, value in self._manifest.get("preflight", {}).items()
                if key != "free_bytes"
            },
            "manifest_filename": self.manifest_path.name,
        }

    def _close_handles(self) -> None:
        for handle in self._slabs.values():
            try:
                handle.flush()
            except (ValueError, OSError):
                pass
        self._slabs.clear()

    def _remove_bulk(self) -> None:
        remove_store_directory(
            self.directory, root=self.root, run_id=self.run_id,
            store_id=self.store_id,
        )

    # -- context management --------------------------------------------------

    def __enter__(self) -> "TemporarySketchStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        """Never leave a store in an ambiguous state, whatever went wrong.

        ``KeyboardInterrupt`` and ``SystemExit`` are handled exactly like any
        other failure: they arrive during collection just as often as an error
        does, and a store abandoned mid-write must not be left looking sealed.
        """

        if exc_type is None:
            self._close_handles()
            return False
        message = f"{exc_type.__name__}: {exc}"
        if self.state is StoreState.COLLECTING or self.state is StoreState.CREATING:
            self.fail_collection(message)
        elif self.state in {StoreState.COLLECTED, StoreState.FINALIZING}:
            self.mark_recoverable(message)
        else:
            self._close_handles()
        return False


def open_store(manifest_path: str | Path) -> TemporarySketchStore:
    """Reopen a store from its manifest, for resuming a finalization."""

    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    layout = SketchStoreLayout(
        num_temperatures=manifest["layout"]["num_temperatures"],
        num_positions=manifest["layout"]["num_positions"],
        num_maps=manifest["layout"]["num_maps"],
        num_buckets=manifest["layout"]["num_buckets"],
    )
    return TemporarySketchStore(
        manifest_path=path,
        directory=Path(manifest["bulk_directory"]),
        root=Path(manifest["bulk_root"]),
        layout=layout,
        run_id=manifest["run_id"],
        store_id=manifest["store_id"],
        manifest=manifest,
    )


class InMemoryRowSink:
    """Collect rows into one ``[N_T, D, M, K]`` array, the historical behaviour.

    This is what ``per_position`` mode and the tests use. It exists beside
    :class:`TemporarySketchStore` so that the measurement loop has exactly one
    way to emit a row and no knowledge of which destination it is talking to --
    the loop must not learn about storage modes, run directories or manifests.

    Rank-conditional on the way out, matching the established record layout: at
    ``M = 1`` the array is ``[N_T, D, K]`` with no length-one map axis to
    squeeze, because that is the shape every existing consumer and the frozen
    fixture depend on.
    """

    def __init__(self, layout: SketchStoreLayout) -> None:
        self.layout = layout
        self._rows = np.zeros(
            (layout.num_temperatures, layout.num_positions,
             layout.num_maps, layout.num_buckets),
            dtype=layout.dtype,
        )
        self._written = np.zeros(
            (layout.num_temperatures, layout.num_maps), dtype=np.int64
        )
        self._sealed = False

    def write_row(
        self, temperature_index: int, position: int, rows: np.ndarray
    ) -> None:
        if self._sealed:
            raise RuntimeError("This sink has been sealed and accepts no more rows.")
        values = np.asarray(rows)
        expected = (self.layout.num_maps, self.layout.num_buckets)
        if values.shape != expected:
            raise ValueError(
                f"Expected a {expected} block for one position; got {values.shape}."
            )
        self._rows[temperature_index, position] = values
        self._written[temperature_index] += 1

    def flush(self) -> None:  # pragma: no cover - nothing is buffered
        return None

    def seal(self) -> dict[str, Any]:
        expected = self.layout.num_positions
        if not np.all(self._written == expected):
            raise ValueError(
                f"Expected {expected} rows for every temperature and map; got "
                f"{self._written.tolist()}."
            )
        self._sealed = True
        return {}

    @property
    def temperature_sketches(self) -> np.ndarray:
        """``[N_T, D, K]`` at one map, ``[N_T, D, M, K]`` above it."""

        if self.layout.num_maps == 1:
            return self._rows[:, :, 0, :]
        return self._rows

    def iter_slabs(self) -> Iterator[tuple[int, int, np.ndarray]]:
        for temperature in range(self.layout.num_temperatures):
            for index in range(self.layout.num_maps):
                yield temperature, index, self._rows[temperature, :, index, :]
