"""The measurement path carries a map count without paying for it.

Stage 8b2-a put a bank of independent CountSketch maps inside the sketcher. This
module is about the layer above: what
:func:`compute_position_gradient_norms` does with it.

Three properties carry the whole substage.

* **A single map is untouched.** ``M = 1`` produces the arrays it always did, at
  the ranks it always did, through the historical ``project`` call. The default
  and an explicit ``sketch_maps=1`` are the same measurement, bit for bit.
* **Replicas cost no backward passes.** ``torch.autograd.grad`` is called
  ``positions x temperatures`` times whatever ``M`` is. That is the governing
  constraint of the whole replica design -- maps are extra *projections* of one
  gradient, never extra gradients -- and it is asserted with a counter rather
  than inferred from a runtime.
* **The map count is stated, not deduced.** ``sketch_protocol["map_count"]`` is
  written at every ``M`` including one, so no reader has to guess from an array's
  rank.

The record schema, the runner and every consumer are deliberately untouched here:
the production runner still measures at the default ``M = 1``.
"""

from __future__ import annotations

import pytest
import torch

from llm_behavior_lab.evaluation.init_distribution import build_evaluation_positions
from llm_behavior_lab.evaluation.position_gradients import (
    CANONICAL_GRADIENT_TEMPERATURE,
    _GradientSketcher,
    compute_position_gradient_norms,
)

VOCAB = 12
BLOCK = 4
WINDOWS = 2
D = WINDOWS * BLOCK
K = 6
TOKENS = [(index * 5 + 2) % VOCAB for index in range(64)]
GRID = (0.60, CANONICAL_GRADIENT_TEMPERATURE)

#: Every key the protocol carried before replicas existed. None may disappear or
#: change meaning; artifacts and readers already refer to them by these names.
LEGACY_PROTOCOL_KEYS = {
    "dimension",
    "seed",
    "temperature",
    "projection",
    "preserves",
    "parameter_count",
}


class TinyModel(torch.nn.Module):
    """Small enough that a full measurement runs in milliseconds."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.embedding = torch.nn.Embedding(VOCAB, 8)
        self.output = torch.nn.Linear(8, VOCAB, bias=False)
        with torch.no_grad():
            self.embedding.weight.copy_(
                torch.randn(VOCAB, 8, generator=generator, dtype=torch.float64)
            )
            self.output.weight.copy_(
                torch.randn(VOCAB, 8, generator=generator, dtype=torch.float64)
            )
        self.double()

    def forward(self, input_ids: torch.Tensor):
        return type("Output", (), {"logits": self.output(self.embedding(input_ids))})()


def _positions():
    return build_evaluation_positions(TOKENS, block_size=BLOCK, num_windows=WINDOWS)


def _measure(**overrides):
    settings = {
        "vocab_size": VOCAB,
        "temperatures": GRID,
        "gradient_sketch": True,
        "sketch_dimension": K,
    }
    settings.update(overrides)
    return compute_position_gradient_norms(TinyModel(), _positions(), **settings)


# -- a single map is exactly what it was -----------------------------------------


def test_the_default_and_an_explicit_single_map_are_bitwise_identical() -> None:
    """Naming the default must not change the measurement."""

    default = _measure()
    explicit = _measure(sketch_maps=1)

    assert torch.equal(
        default.temperature_gradient_sketches, explicit.temperature_gradient_sketches
    )
    assert torch.equal(default.gradient_sketches, explicit.gradient_sketches)
    assert torch.equal(
        default.temperature_gradient_norms, explicit.temperature_gradient_norms
    )


def test_a_single_map_keeps_the_historical_ranks() -> None:
    """No length-one replica axis: ``[N_T, D, K]`` and ``[D, K]``, as always."""

    result = _measure(sketch_maps=1)

    assert result.temperature_gradient_sketches.shape == (len(GRID), D, K)
    assert result.gradient_sketches.shape == (D, K)


def test_the_storage_dtype_is_unchanged() -> None:
    """float32 storage; the projection's float64 accumulation is internal."""

    for maps in (1, 4):
        result = _measure(sketch_maps=maps)
        assert result.temperature_gradient_sketches.dtype is torch.float32
        assert result.gradient_sketches.dtype is torch.float32


# -- replicas ---------------------------------------------------------------------


@pytest.mark.parametrize("maps", [2, 4])
def test_replica_ranks(maps) -> None:
    result = _measure(sketch_maps=maps)

    assert result.temperature_gradient_sketches.shape == (len(GRID), D, maps, K)
    assert result.gradient_sketches.shape == (D, maps, K)


@pytest.mark.parametrize("maps", [1, 2, 4])
def test_the_canonical_field_is_the_canonical_slice(maps) -> None:
    """One measured field read twice, never two fields that happen to agree."""

    result = _measure(sketch_maps=maps)

    assert torch.equal(
        result.gradient_sketches,
        result.temperature_gradient_sketches[result.canonical_index],
    )
    assert result.temperatures[result.canonical_index] == CANONICAL_GRADIENT_TEMPERATURE


def test_a_single_map_uses_the_historical_projection_path(monkeypatch) -> None:
    """``project``, not ``project_maps(...)[0]``.

    Both give the same numbers, but routing the established observable through
    the replica path would put a stack-and-index between it and its own code.
    """

    calls = {"project": 0, "project_maps": 0}
    original = _GradientSketcher.project

    def counting_project(self, grads):
        calls["project"] += 1
        return original(self, grads)

    def counting_project_maps(self, grads):
        calls["project_maps"] += 1
        raise AssertionError("M = 1 must not touch project_maps.")

    monkeypatch.setattr(_GradientSketcher, "project", counting_project)
    monkeypatch.setattr(_GradientSketcher, "project_maps", counting_project_maps)

    _measure(sketch_maps=1)

    assert calls["project"] == D * len(GRID)
    assert calls["project_maps"] == 0


def test_replicas_use_one_project_maps_call_per_gradient_tuple(monkeypatch) -> None:
    """One call per (position, temperature), each projecting every map once."""

    calls = {"project": 0, "project_maps": 0}
    original_maps = _GradientSketcher.project_maps
    original_project = _GradientSketcher.project

    def counting_project(self, grads):
        calls["project"] += 1
        return original_project(self, grads)

    def counting_project_maps(self, grads):
        calls["project_maps"] += 1
        return original_maps(self, grads)

    monkeypatch.setattr(_GradientSketcher, "project", counting_project)
    monkeypatch.setattr(_GradientSketcher, "project_maps", counting_project_maps)

    _measure(sketch_maps=4)

    assert calls["project_maps"] == D * len(GRID)
    assert calls["project"] == 0


# -- the governing constraint ------------------------------------------------------


@pytest.mark.parametrize("maps", [1, 2, 4])
def test_the_backward_count_does_not_depend_on_the_map_count(maps, monkeypatch) -> None:
    """``positions x temperatures``, whatever ``M`` is.

    Counted directly. A runtime comparison would not distinguish "no extra
    backward passes" from "extra backward passes that happened to be cheap",
    and the difference is the entire justification for in-process replicas over
    multi-process recomputation.
    """

    calls = []
    original = torch.autograd.grad

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", counting)
    _measure(sketch_maps=maps)

    assert len(calls) == D * len(GRID)


# -- result compatibility ----------------------------------------------------------


@pytest.mark.parametrize("maps", [1, 4])
def test_sketch_map_is_always_map_zero_in_historical_dtypes(maps) -> None:
    """The fidelity export does not gain a map axis because replicas exist."""

    import numpy as np

    result = _measure(sketch_maps=maps)
    buckets, signs = result.sketch_map

    assert buckets.dtype == np.int64
    assert signs.dtype == np.float64
    assert buckets.shape == signs.shape
    assert buckets.shape[0] == sum(result.sketch_tensor_sizes)

    reference = _GradientSketcher(
        [torch.zeros(size) for size in result.sketch_tensor_sizes],
        dimension=K,
        seed=result.sketch_protocol["base_seed"],
        map_count=maps,
    ).numpy_map()
    assert np.array_equal(buckets, reference[0])
    assert np.array_equal(signs, reference[1])


@pytest.mark.parametrize("maps", [1, 4])
def test_the_non_sketch_observables_are_unaffected(maps) -> None:
    """Norms, losses and identities do not depend on how many maps were built."""

    baseline = _measure(sketch_maps=1)
    result = _measure(sketch_maps=maps)

    assert torch.equal(result.temperature_gradient_norms, baseline.temperature_gradient_norms)
    assert torch.equal(result.gradient_norms, baseline.gradient_norms)
    assert torch.equal(result.target_ids, baseline.target_ids)
    assert torch.equal(result.greedy_ids, baseline.greedy_ids)
    assert torch.equal(result.position_indices, baseline.position_indices)
    assert result.parameter_count == baseline.parameter_count


# -- protocol metadata --------------------------------------------------------------


@pytest.mark.parametrize("maps", [1, 4])
def test_every_legacy_protocol_key_survives_unchanged(maps) -> None:
    """Additive means additive: nothing removed, renamed or reinterpreted."""

    protocol = _measure(sketch_maps=maps).sketch_protocol

    assert LEGACY_PROTOCOL_KEYS <= set(protocol)
    assert protocol["dimension"] == K
    assert protocol["seed"] == 20240917
    assert protocol["temperature"] == CANONICAL_GRADIENT_TEMPERATURE
    assert protocol["projection"] == "count_sketch_signed_feature_hashing"
    assert protocol["preserves"] == "inner_products_in_expectation"
    assert protocol["parameter_count"] > 0


@pytest.mark.parametrize("maps", [1, 2, 4])
def test_the_new_metadata_is_written_at_every_map_count(maps) -> None:
    """Including one -- that is the provenance gap this closes."""

    protocol = _measure(sketch_maps=maps).sketch_protocol

    assert protocol["map_count"] == maps
    assert protocol["base_seed"] == protocol["seed"]
    assert len(protocol["map_seeds"]) == maps
    assert protocol["seed_derivation"] == "base_seed + 1000000007 * map_index"
    assert protocol["seed_derivation_version"] == 1


@pytest.mark.parametrize("maps", [1, 4])
def test_map_seeds_are_ordered_and_match_the_constructed_bank(maps) -> None:
    """Read off the bank the sketcher built, not a duplicated formula.

    A second copy of the derivation is exactly how recorded seeds and real seeds
    drift apart, so this compares against a sketcher constructed the same way.
    """

    result = _measure(sketch_maps=maps)
    bank = _GradientSketcher(
        [torch.zeros(size) for size in result.sketch_tensor_sizes],
        dimension=K,
        seed=result.sketch_protocol["base_seed"],
        map_count=maps,
    )

    assert result.sketch_protocol["map_seeds"] == [int(s) for s in bank._map_seeds]
    assert result.sketch_protocol["map_seeds"][0] == result.sketch_protocol["base_seed"]
    assert result.sketch_protocol["map_seeds"] == sorted(
        result.sketch_protocol["map_seeds"]
    )


def test_the_protocol_is_json_safe() -> None:
    """It is persisted as JSON metadata; a tensor or numpy scalar would break it."""

    import json

    protocol = _measure(sketch_maps=4).sketch_protocol

    assert json.loads(json.dumps(protocol)) == protocol


def test_the_metadata_payload_carries_the_protocol() -> None:
    """``as_metadata`` is what actually reaches a record's gradient_analysis."""

    payload = _measure(sketch_maps=4).as_metadata()

    assert payload["gradient_sketch"]["map_count"] == 4
    assert payload["gradient_sketch"]["seed"] == 20240917


# -- validation -----------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0, -1, 2.5, True, "4", None])
def test_an_invalid_map_count_is_refused(bad) -> None:
    with pytest.raises(ValueError, match="sketch_maps"):
        _measure(sketch_maps=bad)


def test_requesting_replicas_without_the_sketch_is_refused() -> None:
    """Refused, not ignored.

    Silently measuring one map after being asked for four would leave a record
    claiming a replica budget it never had, and the mistake would surface only
    as a missing axis much further downstream.
    """

    with pytest.raises(ValueError, match="gradient_sketch is off"):
        _measure(gradient_sketch=False, sketch_maps=4)


def test_the_sketch_may_still_be_disabled_at_the_default() -> None:
    """Turning sketching off with an untouched map count stays legal."""

    result = _measure(gradient_sketch=False)

    assert result.gradient_sketches is None
    assert result.temperature_gradient_sketches is None
    assert result.sketch_protocol is None
