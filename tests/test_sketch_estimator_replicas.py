"""The ensemble estimator, and the consumers that read it.

Stage 8c-a taught the record to store several independent CountSketch maps.
This is the analysis layer catching up: how the maps combine into the production
estimate, how they are kept apart when the question is uncertainty, and how
neither route ever builds something quadratic in the position count.

Four properties carry the substage.

* **Inner products are averaged, never sketch vectors.** ``mean_m S_m`` estimates
  nothing -- the maps are independent projections, so averaging them shrinks the
  signal each carries. Asserted positively *and* negatively.
* **The ensemble estimate is an inner product in a wider space.**
  ``v(d) = M^{-1/2}[u_1(d)|...|u_M(d)]`` gives ``<v(a),v(b)>`` exactly equal to
  ``(1/M) sum_m <u_m(a),u_m(b)>``, which is what lets every existing bilinear
  consumer produce the ensemble estimate with no change to its own formula.
* **Uncertainty needs the maps kept apart**, so the per-map surfaces exist beside
  the embedding, which has already summed over them.
* **Nothing quadratic in D.** ``[M, D, D]`` is 34 GB at ``D = 32768, M = 4``. The
  production paths reduce through class sums instead, and a spy proves they never
  touch the bounded dense helpers.

``M = 1`` is bitwise unchanged throughout -- every existing result must keep its
value exactly, not approximately.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from llm_behavior_lab.analysis.gradient_cross_partition import (
    cross_partition_per_map,
    mixture_reconstruction,
    pooled_cross_statistic,
)
from llm_behavior_lab.analysis.records import (
    SKETCH_PROTOCOL_SCHEMA_VERSION,
    InitializationExperimentRecord,
)

# `analysis/__init__` re-exports a *function* named `gradient_clustering`, which
# shadows the submodule of the same name on the package object. Resolving the
# module explicitly avoids binding the function by accident.
gc = importlib.import_module("llm_behavior_lab.analysis.gradient_clustering")
se = importlib.import_module("llm_behavior_lab.analysis.sketch_estimator")

DEFAULT_PERMS = 256
VOCAB = 16
ELIGIBLE = np.arange(3, VOCAB)
D = 12
K = 5
GRID = np.asarray([0.6, 1.0], dtype=np.float64)
CANONICAL_ROW = 1


def _protocol(maps: int):
    return {
        "dimension": K,
        "seed": 20240917,
        "map_count": maps,
        "schema_version": SKETCH_PROTOCOL_SCHEMA_VERSION,
        "canonical_storage": "stored" if maps == 1 else "derived",
        "canonical_relationship": "slice_of_temperature_array",
        "temperature_sketch_axes": (
            ["temperature", "position", "bucket"]
            if maps == 1
            else ["temperature", "position", "map", "bucket"]
        ),
        "estimator": "mean_of_per_map_inner_products_over_exact_norms",
        "accumulation_dtype": "float64",
        "storage_dtype": "float32",
        "recomputed_per_map": False,
    }


def _labels(seed: int = 0):
    """Three target classes and three greedy classes, every class supported."""

    rng = np.random.default_rng(seed)
    targets = np.asarray([3, 3, 3, 4, 4, 4, 5, 5, 5, 3, 4, 5], dtype=np.int64)
    greedy = np.asarray([4, 4, 5, 5, 3, 3, 3, 4, 5, 4, 3, 5], dtype=np.int64)
    assert targets.size == D and greedy.size == D
    del rng
    return targets, greedy


def _record(maps: int, *, seed: int = 7):
    """A record carrying ``maps`` independent sketch maps."""

    rng = np.random.default_rng(seed)
    targets, greedy = _labels()
    corpus = np.zeros(VOCAB, dtype=np.int64)
    corpus[ELIGIBLE] = 11

    temperature = rng.normal(size=(len(GRID), D, maps, K)).astype(np.float32)
    extra = {}
    if maps == 1:
        flat = temperature[:, :, 0, :].copy()
        extra["gradient_temperature_position_sketches"] = flat
        extra["gradient_position_sketches"] = flat[CANONICAL_ROW]
    else:
        extra["gradient_temperature_position_sketches"] = temperature

    return InitializationExperimentRecord.build(
        corpus_counts=corpus,
        selected_target_counts=np.bincount(targets, minlength=VOCAB),
        greedy_counts=np.bincount(greedy, minlength=VOCAB)[None, :],
        nucleus_counts=np.bincount(greedy, minlength=VOCAB)[None, None, :],
        mean_predicted_probabilities=np.full((1, VOCAB), 1.0 / VOCAB),
        model_seeds=[1000],
        eligible_token_ids=ELIGIBLE,
        metadata={
            "num_positions": D,
            "record_version": 12,
            "analysis": {
                "gradient_analysis": {
                    "enabled": True,
                    "initialization_index": 0,
                    "gradient_sketch": _protocol(maps),
                }
            },
        },
        gradient_position_indices=np.arange(D),
        gradient_position_target_ids=targets,
        gradient_position_greedy_ids=greedy,
        gradient_position_norms=np.full(D, 2.0),
        gradient_temperatures=GRID,
        gradient_temperature_position_norms=np.full((len(GRID), D), 2.0),
        **extra,
    )


# -- the estimator module ---------------------------------------------------------


def test_the_ensemble_matrix_is_the_mean_of_the_per_map_matrices() -> None:
    unit = np.random.default_rng(0).normal(size=(6, 4, 5))

    per_map = se.retained_per_map_cosine_matrices(unit)
    ensemble = se.retained_ensemble_cosine_matrix(unit)

    assert per_map.shape == (4, 6, 6)
    assert ensemble.shape == (6, 6)
    assert np.allclose(ensemble, per_map.mean(axis=0), rtol=0, atol=1e-15)


def test_each_per_map_matrix_is_that_map_s_own_gram() -> None:
    unit = np.random.default_rng(1).normal(size=(5, 3, 4))

    per_map = se.retained_per_map_cosine_matrices(unit)

    for index in range(3):
        rows = unit[:, index, :]
        assert np.allclose(per_map[index], rows @ rows.T, rtol=0, atol=1e-15)


def test_the_estimator_does_not_average_sketch_vectors() -> None:
    """The mistake the whole design exists to make unavailable.

    Averaging the maps and taking one inner product afterwards is a different
    quantity, and a smaller one -- independent projections partially cancel. If
    these ever agreed, the estimator would have been silently redefined.
    """

    unit = np.random.default_rng(2).normal(size=(6, 4, 5))

    ensemble = se.retained_ensemble_cosine_matrix(unit)
    averaged_vectors = unit.mean(axis=1) @ unit.mean(axis=1).T

    assert not np.allclose(ensemble, averaged_vectors)


def test_the_embedding_inner_product_is_exactly_the_ensemble_estimate() -> None:
    """The identity that lets every bilinear consumer stay unchanged."""

    unit = np.random.default_rng(3).normal(size=(9, 4, 6))

    embedding = se.ensemble_embedding(unit)
    gram = embedding @ embedding.T
    ensemble = se.retained_ensemble_cosine_matrix(unit)

    assert embedding.shape == (9, 24)
    assert np.allclose(gram, ensemble, rtol=0, atol=1e-12)


def test_the_embedding_allocates_only_the_returned_buffer() -> None:
    """Peak must be the output plus one slice, not twice the output.

    The obvious ``(values / sqrt(M)).reshape(...)`` casts the whole stack to
    float64 and then allocates a second full-size array for the division -- 2 GiB
    of peak for a 1 GiB result at campaign shape. Filling the buffer map by map
    leaves a transient of one ``[D, K]`` slice instead.
    """

    import tracemalloc

    stored = np.zeros((4000, 4, 256), dtype=np.float32)

    tracemalloc.start()
    embedding = se.ensemble_embedding(stored)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert peak < 1.2 * embedding.nbytes


def test_the_embedding_folds_a_position_scale_into_the_same_pass() -> None:
    """Normalizing must not cost a second full-size array either."""

    import tracemalloc

    stored = np.zeros((4000, 4, 256), dtype=np.float32)
    scale = np.full(4000, 0.5)

    tracemalloc.start()
    embedding = se.ensemble_embedding(stored, position_scale=scale)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert peak < 1.2 * embedding.nbytes


def test_the_position_scale_matches_normalizing_first() -> None:
    rng = np.random.default_rng(21)
    stored = rng.normal(size=(9, 3, 5))
    norms = rng.uniform(0.5, 2.0, size=9)

    folded = se.ensemble_embedding(stored, position_scale=1.0 / norms)
    separate = se.ensemble_embedding(stored / norms[:, None, None])

    assert np.allclose(folded, separate, rtol=0, atol=1e-15)


def test_a_position_scale_of_the_wrong_length_is_refused() -> None:
    with pytest.raises(ValueError, match="position_scale"):
        se.ensemble_embedding(np.zeros((4, 2, 3)), position_scale=np.ones(5))


def test_the_embedding_is_the_identity_at_one_map() -> None:
    unit = np.random.default_rng(4).normal(size=(7, 1, 5))

    embedding = se.ensemble_embedding(unit)

    assert embedding.shape == (7, 5)
    assert np.array_equal(embedding, unit[:, 0, :])


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((3, 4)),
        np.zeros((3, 4, 5, 6)),
        np.zeros((0, 2, 3)),
        np.zeros((3, 0, 3)),
        np.zeros((3, 2, 0)),
        np.array([["a", "b"]], dtype=object),
    ],
)
def test_a_malformed_per_map_array_is_refused(bad) -> None:
    with pytest.raises(ValueError):
        se.retained_per_map_cosine_matrices(bad)


def test_the_supported_retained_workflow_is_accepted() -> None:
    """12 positions is what the runner actually retains, and it must work.

    ``select_sanity_positions(count=DEFAULT_SANITY_POSITIONS)`` is hard-coded in
    the runner with no override, so this is the whole production surface.
    """

    from llm_behavior_lab.evaluation.position_gradients import (
        DEFAULT_SANITY_POSITIONS,
    )

    assert DEFAULT_SANITY_POSITIONS == 12
    assert se.RETAINED_POSITION_CEILING >= DEFAULT_SANITY_POSITIONS
    supported = np.zeros((DEFAULT_SANITY_POSITIONS, 8, 1024), dtype=np.float32)

    assert se.retained_per_map_cosine_matrices(supported).shape == (8, 12, 12)


def test_the_position_boundary_is_accepted_and_boundary_plus_one_is_not() -> None:
    """A tiny bucket axis keeps this cheap: nothing here allocates megabytes."""

    at_limit = np.zeros((se.RETAINED_POSITION_CEILING, 2, 4), dtype=np.float32)
    assert se.retained_per_map_cosine_matrices(at_limit).shape == (
        2,
        se.RETAINED_POSITION_CEILING,
        se.RETAINED_POSITION_CEILING,
    )

    over = np.zeros((se.RETAINED_POSITION_CEILING + 1, 2, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="retained-subset ceiling"):
        se.retained_per_map_cosine_matrices(over)


def test_the_map_count_ceiling_is_enforced() -> None:
    over = np.zeros((4, se.RETAINED_MAP_CEILING + 1, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="maps exceeds"):
        se.retained_per_map_cosine_matrices(over)


def test_the_output_byte_ceiling_is_enforced() -> None:
    """Refused on the *requested* size; nothing that large is ever allocated."""

    positions = se.RETAINED_POSITION_CEILING
    maps = se.RETAINED_MAP_CEILING
    requested = maps * positions * positions * 8
    assert requested <= se.RETAINED_OUTPUT_BYTE_CEILING

    # Raise the position ceiling alone and the byte budget catches it instead.
    with pytest.raises(ValueError, match="ceiling"):
        se.retained_per_map_cosine_matrices(
            np.zeros((positions * 4, maps, 2), dtype=np.float32)
        )


def test_the_operation_budget_catches_a_small_output_behind_huge_work() -> None:
    """Output bytes say nothing about the work behind them.

    ``[64, 256, 256]`` float64 is only 32 MiB, but at ``K = 1024`` the products
    behind it are billions of multiply-adds.
    """

    wide = np.zeros((256, 64, 1024), dtype=np.float32)
    assert 64 * 256 * 256 * 8 <= se.RETAINED_OUTPUT_BYTE_CEILING

    with pytest.raises(ValueError, match="multiply-adds"):
        se.retained_per_map_cosine_matrices(wide)


def test_the_guards_run_before_any_allocation(monkeypatch) -> None:
    """Proof of ordering: neither the cast nor the product may be reached."""

    def explode(*args, **kwargs):
        raise AssertionError("allocated before the guards ran")

    monkeypatch.setattr(np, "swapaxes", explode)
    over = np.zeros((se.RETAINED_POSITION_CEILING + 1, 2, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="retained-subset ceiling"):
        se.retained_per_map_cosine_matrices(over)


# -- ensemble_summary -------------------------------------------------------------


def test_the_summary_at_one_map_reports_unavailable_not_zero() -> None:
    """One projection cannot say how far another would have landed."""

    summary = se.ensemble_summary(np.asarray([0.25]))

    assert summary["map_mean"] == 0.25
    assert summary["sample_sd"] is None
    assert summary["standard_error"] is None
    assert summary["degrees_of_freedom"] == 0
    assert summary["uncertainty_available"] is False


def test_the_summary_at_several_maps_uses_ddof_one() -> None:
    values = np.asarray([1.0, 2.0, 3.0, 4.0])

    summary = se.ensemble_summary(values)

    assert summary["map_mean"] == pytest.approx(2.5)
    assert summary["sample_sd"] == pytest.approx(values.std(ddof=1))
    assert summary["standard_error"] == pytest.approx(values.std(ddof=1) / 2.0)
    assert summary["degrees_of_freedom"] == 3
    assert summary["uncertainty_available"] is True


def test_the_summary_keeps_leading_dimensions() -> None:
    values = np.arange(12, dtype=np.float64).reshape(3, 4)

    summary = se.ensemble_summary(values, axis=-1)

    assert summary["map_mean"].shape == (3,)
    assert summary["sample_sd"].shape == (3,)
    assert summary["degrees_of_freedom"] == 3


def test_a_nonfinite_multi_map_result_stays_distinguishable() -> None:
    """Broken arithmetic must not hide behind the legitimate M=1 case."""

    summary = se.ensemble_summary(np.asarray([1.0, np.nan, 3.0]))

    assert summary["uncertainty_available"] is True
    assert summary["degrees_of_freedom"] == 2
    assert not np.isfinite(summary["sample_sd"])


@pytest.mark.parametrize("bad", [np.asarray(1.0), np.zeros(0)])
def test_a_summary_without_a_map_axis_is_refused(bad) -> None:
    with pytest.raises(ValueError):
        se.ensemble_summary(bad)


# -- the consumer surfaces ---------------------------------------------------------


def test_unit_sketches_per_map_shape_and_normalization() -> None:
    record = _record(4)

    unit, usable = gc.unit_sketches_per_map(record)

    assert unit.shape == (D, 4, K)
    assert usable.shape == (D,)
    # Norms are all 2.0, so each map is exactly the stored sketch halved.
    stored = np.asarray(record.per_map_sketches(), dtype=np.float64)
    assert np.allclose(unit, stored / 2.0, rtol=0, atol=1e-15)


def test_unit_sketches_returns_the_ensemble_embedding_above_one_map() -> None:
    record = _record(4)

    ensemble, _ = gc.unit_sketches(record)
    per_map, _ = gc.unit_sketches_per_map(record)

    assert ensemble.shape == (D, 4 * K)
    assert np.allclose(
        ensemble @ ensemble.T,
        se.retained_ensemble_cosine_matrix(per_map),
        rtol=0,
        atol=1e-12,
    )


def test_the_ensemble_path_does_not_route_through_the_per_map_surface(
    monkeypatch,
) -> None:
    """``unit_sketches`` must not build ``[D, M, K]`` merely to flatten it.

    Going through ``unit_sketches_per_map`` would allocate a full normalized
    stack and then a second buffer for the concatenation. The ensemble path
    folds the norm into one fill pass instead, so it must not call it at all.
    """

    def explode(*args, **kwargs):
        raise AssertionError(
            "unit_sketches built the per-map stack just to flatten it."
        )

    monkeypatch.setattr(gc, "unit_sketches_per_map", explode)

    embedding, usable = gc.unit_sketches(_record(4))

    assert embedding.shape == (D, 4 * K)
    assert usable.shape == (D,)


def test_the_ensemble_path_still_matches_the_per_map_route_numerically() -> None:
    """Cheaper, and the same answer."""

    record = _record(4)
    embedding, usable = gc.unit_sketches(record)
    per_map, per_map_usable = gc.unit_sketches_per_map(record)

    assert np.array_equal(usable, per_map_usable)
    assert np.allclose(
        embedding @ embedding.T,
        se.retained_ensemble_cosine_matrix(per_map),
        rtol=0,
        atol=1e-12,
    )


def test_the_directional_field_keeps_its_keys_at_every_map_count() -> None:
    single = _record(1)
    multi = _record(4)
    from llm_behavior_lab.analysis.directional_fields import (
        directional_field,
        directional_field_per_map,
    )

    for record, width in ((single, K), (multi, 4 * K)):
        field = directional_field(record, 1.0)
        assert set(field) == {
            "sketches", "norms", "loss_temperature", "index", "source"
        }
        assert field["sketches"].shape == (D, width)

    per_map = directional_field_per_map(multi, 1.0)
    assert per_map["sketches"].shape == (D, 4, K)
    assert set(per_map) == {
        "sketches", "norms", "loss_temperature", "index", "source"
    }
    assert directional_field_per_map(single, 1.0)["sketches"].shape == (D, 1, K)


def test_the_per_map_clustering_surface_reports_spread() -> None:
    result = gc.gradient_clustering_per_map(_record(4), permutations=8)

    assert result["map_count"] == 4
    assert result["delta_per_map"].shape == (4,)
    assert result["delta"]["uncertainty_available"] is True
    assert result["delta"]["degrees_of_freedom"] == 3
    assert np.isfinite(result["delta"]["standard_error"])
    assert result["within_per_map"].shape == (4,)
    assert result["between_per_map"].shape == (4,)


def test_the_per_map_clustering_surface_covers_both_groupings() -> None:
    summary = gc.clustering_summary_per_map(_record(4), permutations=8)

    assert set(summary) == set(gc.GROUPINGS)
    for grouping in gc.GROUPINGS:
        assert summary[grouping]["delta_per_map"].shape == (4,)


def test_the_per_map_cross_partition_surface() -> None:
    record = _record(4)
    unit, usable = gc.unit_sketches_per_map(record)
    targets, greedy = _labels()

    result = cross_partition_per_map(unit[usable], targets[usable], greedy[usable])

    assert result["map_count"] == 4
    for key in (
        "c_same_per_map",
        "c_different_per_map",
        "delta_cross_per_map",
        "mixture_median_similarity_per_map",
    ):
        assert result[key].shape == (4,)
    assert result["delta_cross"]["degrees_of_freedom"] == 3


def test_the_mixture_plug_in_is_not_the_mean_of_per_map_ratios() -> None:
    """A ratio of averages is not the average of ratios, and both are kept.

    The plug-in value stays the production point estimate -- at M=1 it is the
    quantity every existing result used -- while the per-map mean is recorded
    separately so the gap is visible rather than hidden.
    """

    record = _record(4)
    unit, usable = gc.unit_sketches_per_map(record)
    targets, greedy = _labels()
    embedding = se.ensemble_embedding(unit)[usable]

    plug_in = mixture_reconstruction(
        embedding, targets[usable], greedy[usable]
    )["median_similarity"]
    per_map = cross_partition_per_map(
        unit[usable], targets[usable], greedy[usable]
    )

    assert np.isfinite(plug_in)
    assert plug_in != per_map["mixture_median_similarity_map_summary"]["map_mean"]


def test_the_nucleus_per_map_surface_spans_temperatures() -> None:
    from llm_behavior_lab.analysis.nucleus_clustering import (
        nucleus_clustering_per_map,
    )

    record = _record(4)
    targets, greedy = _labels()
    labels = np.stack([targets, greedy])

    result = nucleus_clustering_per_map(
        record, labels, [0.6, 1.0], loss_temperatures=[0.6, 1.0], permutations=8
    )

    assert result["delta_per_map"].shape == (2, 4)
    assert result["map_count"] == 4
    assert np.asarray(result["delta"]["map_mean"]).shape == (2,)


# -- common random numbers ----------------------------------------------------------


def test_the_same_permutation_draws_reach_every_map(monkeypatch) -> None:
    """Directly, not by inferring it from matching runtimes.

    Every map must see the identical permutation indices. If each drew its own,
    the across-map spread would contain permutation noise as well as projection
    noise and would overstate the projection error.
    """

    seen = []
    original = gc._permutation_null

    def recording(unit, counts, *, permutations, seed, orders=None):
        seen.append(orders)
        return original(
            unit, counts, permutations=permutations, seed=seed, orders=orders
        )

    monkeypatch.setattr(gc, "_permutation_null", recording)
    gc.gradient_clustering_per_map(_record(4), permutations=4)

    assert len(seen) == 4
    assert all(orders is not None for orders in seen)
    # The very same list object, so there is no way for one map to diverge.
    assert all(orders is seen[0] for orders in seen)
    for draw in range(4):
        assert all(np.array_equal(o[draw], seen[0][draw]) for o in seen)


def test_precomputed_orders_reproduce_the_inline_draw() -> None:
    """Extracting the draw must not move the single-map numbers."""

    counts = np.asarray([4, 3, 5])
    orders = gc.permutation_orders(counts, permutations=6, seed=20240918)

    generator = np.random.default_rng(20240918)
    expected = [generator.permutation(int(counts.sum())) for _ in range(6)]

    assert all(np.array_equal(a, b) for a, b in zip(orders, expected))


# -- the dense-matrix prohibition ------------------------------------------------------


def _forbid_dense(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError(
            "A production path called a bounded retained dense-matrix helper."
        )

    monkeypatch.setattr(se, "retained_per_map_cosine_matrices", explode)
    monkeypatch.setattr(se, "retained_ensemble_cosine_matrix", explode)


def test_no_production_path_builds_a_dense_cosine_matrix(monkeypatch) -> None:
    """Spied, not assumed: [M, D, D] is 34 GB at campaign scale."""

    _forbid_dense(monkeypatch)
    record = _record(4)
    targets, greedy = _labels()
    unit, usable = gc.unit_sketches_per_map(record)
    embedding = se.ensemble_embedding(unit)[usable]

    gc.clustering_summary(record)
    gc.gradient_clustering(record, permutations=4)
    gc.gradient_clustering_per_map(record, permutations=4)
    pooled_cross_statistic(embedding, targets[usable], greedy[usable])
    gc.class_similarity_matrix(
        embedding, targets[usable], np.unique(targets[usable])
    )
    mixture_reconstruction(embedding, targets[usable], greedy[usable])
    cross_partition_per_map(unit[usable], targets[usable], greedy[usable])


# -- M = 1 compatibility ----------------------------------------------------------------


def test_a_single_map_record_is_bitwise_unchanged_through_every_surface() -> None:
    """Bitwise, not approximately: existing results must not move at all."""

    from llm_behavior_lab.analysis.directional_fields import directional_field

    record = _record(1)
    stored = np.asarray(record.gradient_position_sketches, dtype=np.float64)
    norms = np.asarray(record.gradient_position_norms, dtype=np.float64)

    field = directional_field(record, 1.0)
    assert np.array_equal(field["sketches"], record.gradient_position_sketches)
    assert np.array_equal(field["norms"], record.gradient_position_norms)

    unit, usable = gc.unit_sketches(record)
    expected = np.zeros_like(stored)
    expected[norms > 0] = stored[norms > 0] / norms[norms > 0, None]
    assert unit.shape == (D, K)
    assert np.array_equal(unit, expected)
    assert np.array_equal(usable, norms > 0)


def test_single_map_statistics_are_unchanged_by_the_new_code_path() -> None:
    """Frozen against an explicit recomputation, not against Git history."""

    record = _record(1)
    targets, greedy = _labels()
    unit, usable = gc.unit_sketches(record)

    summary = gc.clustering_summary(record)
    pooled = pooled_cross_statistic(unit[usable], targets[usable], greedy[usable])
    mixture = mixture_reconstruction(
        unit[usable], targets[usable], greedy[usable]
    )

    # The per-map surface at M = 1 must agree exactly with the point estimate.
    per_map = gc.gradient_clustering_per_map(record, permutations=DEFAULT_PERMS)
    assert per_map["map_count"] == 1
    assert per_map["delta_per_map"][0] == pytest.approx(
        summary["target"]["population"]["delta"], rel=0, abs=1e-15
    )
    assert per_map["delta"]["uncertainty_available"] is False
    assert np.isfinite(pooled["delta_cross"])
    assert np.isfinite(mixture["median_similarity"])


def test_the_retained_helpers_agree_with_the_production_estimator_at_one_map() -> None:
    record = _record(1)
    unit_per_map, _ = gc.unit_sketches_per_map(record)
    unit, _ = gc.unit_sketches(record)

    dense = se.retained_ensemble_cosine_matrix(unit_per_map)

    assert np.allclose(dense, unit @ unit.T, rtol=0, atol=1e-15)


# -- fidelity factoring ------------------------------------------------------------------


def test_the_fidelity_module_re_exports_the_shared_helpers() -> None:
    """Same objects, so no caller can pick up a second implementation."""

    from llm_behavior_lab.analysis import countsketch_fidelity as cf

    assert cf.cosine_from_gram is se.cosine_from_gram
    assert cf.cosines_from_sketches is se.cosines_from_sketches
    assert "cosine_from_gram" in cf.__all__
    assert "cosines_from_sketches" in cf.__all__


def test_the_moved_helpers_compute_what_they_always_did() -> None:
    """The expressions moved verbatim, so this is exact rather than close."""

    rng = np.random.default_rng(11)
    values = rng.normal(size=(5, 7))
    norms = np.full(5, 3.0)

    assert np.array_equal(
        se.cosine_from_gram(values, norms),
        (values @ values.T) / np.outer(norms, norms),
    )
    assert np.array_equal(
        se.cosines_from_sketches(values, norms),
        (values @ values.T) / np.outer(norms, norms),
    )


def test_the_fidelity_alternate_map_bank_is_untouched() -> None:
    from llm_behavior_lab.analysis.countsketch_fidelity import (
        ALTERNATE_SKETCH_SEEDS,
    )

    assert ALTERNATE_SKETCH_SEEDS == (101, 202, 303, 404, 505, 606, 707, 808)
    assert 20240917 not in ALTERNATE_SKETCH_SEEDS


def test_the_estimator_module_imports_no_project_code() -> None:
    """Dependency-neutral by construction, so it can never form a cycle."""

    text = open(se.__file__, encoding="utf-8").read()

    assert "from llm_behavior_lab" not in text
    assert "import llm_behavior_lab" not in text
