"""Independent production CountSketch maps, with map 0 still the historical one.

One map estimates an inner product without error bars. Several independent maps
of the *same* gradient give a spread to read the projection error off, which is
the whole reason the replica bank exists.

That only works if two things hold, and both are pinned here:

* **map 0 does not move.** Adding replicas must not disturb the established
  observable, so map 0 is checked against the same frozen reference the
  single-map implementation is checked against -- at ``map_count = 1`` and again
  at ``map_count = 4``;
* **the replicas are genuinely independent, of one gradient.** Different maps,
  same gradient tuple: distinct tables, distinct projections, no recomputation,
  and no ``autograd`` call anywhere in the sketcher.

The reference lives in ``tests/assets/countsketch_golden_map_v1.npz`` and must
never be regenerated. See the header of ``tests/assets/make_countsketch_golden.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from llm_behavior_lab.evaluation import position_gradients as pg
from llm_behavior_lab.evaluation.position_gradients import (
    _SKETCH_BUCKET_DEVICE_DTYPE,
    _SKETCH_SIGN_DEVICE_DTYPE,
    _GradientSketcher,
    _production_sketch_map_seed,
    production_sketch_tables,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN = REPO_ROOT / "tests" / "assets" / "countsketch_golden_map_v1.npz"


@pytest.fixture(scope="module")
def golden():
    """The frozen pre-replica reference: map values and the float64 projection."""

    with np.load(GOLDEN, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def _sizes(golden) -> tuple[int, ...]:
    return tuple(int(value) for value in golden["tensor_sizes"])


def _gradients(golden) -> list[torch.Tensor]:
    return [
        torch.arange(size, dtype=torch.float64) * 0.5 - 1.25 for size in _sizes(golden)
    ]


def _sketcher(golden, map_count: int = 1) -> _GradientSketcher:
    return _GradientSketcher(
        [torch.zeros(size) for size in _sizes(golden)],
        dimension=int(golden["dimension"]),
        seed=int(golden["seed"]),
        map_count=map_count,
    )


# -- map 0 is untouched, at every map count --------------------------------------


@pytest.mark.parametrize("map_count", [1, 4])
def test_map_zero_equals_the_frozen_reference(golden, map_count) -> None:
    """The established map, whether or not replicas were built beside it."""

    sketcher = _sketcher(golden, map_count)
    buckets = np.concatenate(
        [b.cpu().to(torch.int64).numpy() for b in sketcher.buckets]
    )
    signs = np.concatenate([s.cpu().to(torch.float64).numpy() for s in sketcher.signs])

    assert np.array_equal(buckets, golden["buckets"])
    assert np.array_equal(signs, golden["signs"])


@pytest.mark.parametrize("map_count", [1, 4])
def test_map_zero_projection_is_bit_identical_to_the_frozen_reference(
    golden, map_count
) -> None:
    """``torch.equal``, not ``allclose``: no tolerance is being spent here."""

    sketch = _sketcher(golden, map_count).project(_gradients(golden))

    assert sketch.dtype is torch.float64
    assert torch.equal(sketch, torch.from_numpy(golden["sketch"]))


def test_the_historical_attributes_stay_flat_per_tensor_lists(golden) -> None:
    """``buckets``/``signs`` are read directly by existing tests and callers.

    They must stay one entry per parameter tensor for map 0 -- not a nested
    ``[map][tensor]`` structure -- or every existing reader breaks silently.
    """

    sketcher = _sketcher(golden, 4)

    assert len(sketcher.buckets) == len(_sizes(golden))
    assert len(sketcher.signs) == len(_sizes(golden))
    assert sketcher.buckets[0].dtype is _SKETCH_BUCKET_DEVICE_DTYPE
    assert sketcher.signs[0].dtype is _SKETCH_SIGN_DEVICE_DTYPE
    # The same objects as map 0's bank, not copies of it.
    assert sketcher.buckets is sketcher._map_buckets[0]
    assert sketcher.signs is sketcher._map_signs[0]


def test_the_public_state_is_unchanged_by_replicas(golden) -> None:
    """``seed`` keeps its historical meaning: it is the base seed, map 0's seed."""

    single = _sketcher(golden, 1)
    many = _sketcher(golden, 4)

    for sketcher in (single, many):
        assert sketcher.dimension == int(golden["dimension"])
        assert sketcher.seed == int(golden["seed"])
        assert sketcher.tensor_sizes == list(_sizes(golden))
    assert single.map_count == 1
    assert many.map_count == 4


def test_numpy_map_still_exports_map_zero(golden) -> None:
    """The public export is the historical map in its historical dtypes."""

    buckets, signs = _sketcher(golden, 4).numpy_map()

    assert buckets.dtype == np.int64
    assert signs.dtype == np.float64
    assert np.array_equal(buckets, golden["buckets"])
    assert np.array_equal(signs, golden["signs"])


# -- the replicas ----------------------------------------------------------------


def test_project_maps_shape_and_dtype(golden) -> None:
    sketch = _sketcher(golden, 4).project_maps(_gradients(golden))

    assert sketch.shape == (4, int(golden["dimension"]))
    assert sketch.dtype is torch.float64


def test_a_single_map_still_gets_a_map_axis(golden) -> None:
    """``[1, K]``, never a squeezed ``[K]``: the shape says how many maps ran."""

    assert _sketcher(golden, 1).project_maps(_gradients(golden)).shape == (
        1,
        int(golden["dimension"]),
    )


@pytest.mark.parametrize("map_count", [1, 4])
def test_row_zero_is_exactly_the_historical_projection(golden, map_count) -> None:
    sketcher = _sketcher(golden, map_count)
    grads = _gradients(golden)

    assert torch.equal(sketcher.project_maps(grads)[0], sketcher.project(grads))


def test_every_replica_differs_from_map_zero_and_from_the_others(golden) -> None:
    """Independence has to be visible, not merely intended."""

    sketcher = _sketcher(golden, 4)
    rows = sketcher.project_maps(_gradients(golden))

    assert len({tuple(row.tolist()) for row in rows}) == 4
    for index in range(1, 4):
        assert not torch.equal(sketcher._map_buckets[index][0], sketcher.buckets[0])


def test_every_replica_stores_compact_device_tables(golden) -> None:
    """The 5-bytes-per-parameter-per-map saving must apply to all of them.

    It is what makes M > 1 affordable at all; a replica that quietly kept wide
    tables would cost 16 bytes per parameter and defeat the point.
    """

    sketcher = _sketcher(golden, 4)

    for buckets, signs in zip(sketcher._map_buckets, sketcher._map_signs):
        assert all(table.dtype is _SKETCH_BUCKET_DEVICE_DTYPE for table in buckets)
        assert all(table.dtype is _SKETCH_SIGN_DEVICE_DTYPE for table in signs)


def test_all_maps_project_the_same_gradient_objects(golden) -> None:
    """One gradient set, M projections -- the replica design's whole premise.

    A gradient consumed once and re-derived per map would confound projection
    variance with recomputation variance, which is exactly the quantity the
    replicas exist to isolate. Asserted by identity, not by value: the tensors
    handed to each map must be the same objects.
    """

    sketcher = _sketcher(golden, 4)
    seen: list[list[int]] = []
    original = sketcher._project_tables

    def recording(grads, buckets, signs):
        seen.append([id(tensor) for tensor in grads])
        return original(grads, buckets, signs)

    sketcher._project_tables = recording
    sketcher.project_maps(_gradients(golden))

    assert len(seen) == 4
    assert all(row == seen[0] for row in seen)


def test_a_one_shot_gradient_iterable_reaches_every_map(golden) -> None:
    """A generator would otherwise be drained by the first map, silently.

    The later maps would then project nothing and return all-zero sketches with
    no error at all -- the same failure mode the parameter list already had.
    """

    sketcher = _sketcher(golden, 4)
    grads = _gradients(golden)

    from_generator = sketcher.project_maps(tensor for tensor in grads)

    assert torch.equal(from_generator, sketcher.project_maps(grads))
    assert not torch.equal(from_generator[3], torch.zeros_like(from_generator[3]))


def test_the_sketcher_never_calls_autograd(golden) -> None:
    """Projection is arithmetic over gradients someone else already computed."""

    sketcher = _sketcher(golden, 4)
    calls = []
    original = torch.autograd.grad

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    torch.autograd.grad = counting
    try:
        sketcher.project_maps(_gradients(golden))
        sketcher.project(_gradients(golden))
    finally:
        torch.autograd.grad = original

    assert calls == []


def test_a_generator_of_parameters_builds_every_map(golden) -> None:
    """``parameters`` is consumed once per map; it must be materialized first."""

    sizes = (7, 5, 11, 3)
    sketcher = _GradientSketcher(
        (torch.zeros(size) for size in sizes),
        dimension=8,
        seed=20240917,
        map_count=4,
    )

    assert sketcher.tensor_sizes == list(sizes)
    assert len(sketcher._map_buckets) == 4
    assert len(sketcher._map_signs) == 4
    for buckets, signs in zip(sketcher._map_buckets, sketcher._map_signs):
        assert len(buckets) == len(sizes)
        assert len(signs) == len(sizes)


# -- the seed schedule -----------------------------------------------------------


def test_map_zero_uses_the_base_seed_unchanged() -> None:
    """The compatibility hinge: m = 0 must be the identity on the seed."""

    assert _production_sketch_map_seed(20240917, 0) == 20240917


def test_map_seeds_follow_the_declared_arithmetic() -> None:
    base = 20240917
    for index in range(5):
        assert (
            _production_sketch_map_seed(base, index)
            == base + pg._SKETCH_MAP_SEED_STRIDE * index
        )


def test_map_zero_tables_match_the_historical_construction() -> None:
    """Map 0 is literally ``production_sketch_tables`` at the base seed."""

    sizes = (7, 5, 11, 3)
    historical = production_sketch_tables(sizes, 8, 20240917)
    sketcher = _GradientSketcher(
        [torch.zeros(size) for size in sizes], dimension=8, seed=20240917, map_count=3
    )

    for (buckets, signs), device_buckets, device_signs in zip(
        historical, sketcher.buckets, sketcher.signs
    ):
        assert torch.equal(buckets, device_buckets.to(torch.int64))
        assert torch.equal(signs, device_signs.to(torch.float64))


def test_no_two_map_tensor_seeds_collide() -> None:
    """A shared tensor seed would make two maps share that tensor's map.

    They would then be correlated exactly where the replica design assumes
    independence, and the spread across maps would understate the real error.
    Checked over an envelope well past anything planned: 8 maps, 2,000 tensors.
    """

    base = 20240917
    seeds = {
        _production_sketch_map_seed(base, m) + 1000003 * i
        for m in range(8)
        for i in range(2000)
    }

    assert len(seeds) == 8 * 2000


# -- validation ------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0, -1, 2.5, True, "4", None])
def test_map_count_must_be_a_positive_integer(bad) -> None:
    with pytest.raises(ValueError, match="map_count"):
        _GradientSketcher([torch.zeros(4)], dimension=8, seed=1, map_count=bad)


def test_a_negative_map_index_is_refused() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _production_sketch_map_seed(20240917, -1)


def test_a_seed_that_would_overflow_the_generator_is_refused() -> None:
    """Silent wrap-around would land the map on an unrelated RNG stream."""

    with pytest.raises(ValueError, match="outside the range"):
        _production_sketch_map_seed(2**64 - 1, 1)


def test_a_per_tensor_seed_that_would_overflow_is_refused() -> None:
    """The map seed can be in range while a later tensor's seed is not."""

    base = 2**64 - 1 - pg._SKETCH_MAP_SEED_STRIDE
    # The last map's own seed still fits ...
    assert _production_sketch_map_seed(base, 1) <= pg._MAX_GENERATOR_SEED
    # ... but its per-tensor seeds run past the end.
    with pytest.raises(ValueError, match="per-tensor seed"):
        _GradientSketcher(
            [torch.zeros(4), torch.zeros(4)], dimension=8, seed=base, map_count=2
        )


# -- the public surface did not move ---------------------------------------------


def test_the_module_exports_are_unchanged() -> None:
    """Replica machinery is an implementation choice, not a contract."""

    assert pg.__all__ == [
        "CANONICAL_GRADIENT_TEMPERATURE",
        "GRADIENT_DEFINITION",
        "GRADIENT_TEMPERATURES",
        "PositionGradientResult",
        "DEFAULT_SANITY_POSITIONS",
        "DEFAULT_SKETCH_DIMENSION",
        "production_sketch_map",
        "production_sketch_tables",
        "compute_position_gradient_norms",
        "evenly_spaced_indices",
        "masked_evaluation_logits",
        "single_position_losses",
    ]


def test_the_default_map_count_is_one() -> None:
    """Nothing gains replicas by accident; the runner is untouched this stage."""

    assert _GradientSketcher([torch.zeros(4)], dimension=8, seed=1).map_count == 1
