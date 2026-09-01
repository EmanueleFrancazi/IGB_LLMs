"""The factorized estimators must be the established ones, one epoch later.

Two obligations run through this file, and they are different in kind.

**Exactness where nothing changed.** The block reduction must tile its
selection exactly and sum each block in its own order, asserted bitwise against
an independent reference rather than approximately.

**Tolerance where something did.** ``blockwise_class_sum/v1`` accumulates a class
sum pairwise where the v12 route accumulates it sequentially. The statistic, the
draws, the estimator and the permutation sequence are identical; only the float64
summation order moves. Every comparison against the legacy route is therefore
made under the declared ``rtol = 1e-12, atol = 1e-14`` **and** an exact match of
the NaN pattern -- because a statistic that is unavailable in one route and
merely small in the other is a defect, not a rounding difference, and a bare
``allclose`` would wave it through.

The correctness cases below are not generic array tests. Each one pins a
property the scientific reading of the figures depends on: that ``Q`` is the
measured self term rather than a class count, that a singleton has no within-class
estimate rather than a zero, that antipodal structure comes out negative, and
that maps are averaged as inner products rather than as vectors.
"""

from __future__ import annotations

import numpy as np
import pytest

from llm_behavior_lab.analysis import alignment_estimators as est
from llm_behavior_lab.analysis.gradient_clustering import (
    _class_sums,
    _permutation_null,
    _pooled_from_blocks,
    class_similarity_matrix,
    permutation_orders as legacy_permutation_orders,
)

RTOL = 1e-12
ATOL = 1e-14


def assert_matching(actual, expected, *, what: str) -> None:
    """Declared tolerance on the finite entries, exact match on the NaN pattern."""

    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    assert actual.shape == expected.shape, f"{what}: shape"
    finite = np.isfinite(expected)
    assert np.array_equal(finite, np.isfinite(actual)), (
        f"{what}: availability pattern differs; a value that is unavailable in "
        "one route and finite in the other is a defect, not a rounding "
        "difference."
    )
    np.testing.assert_allclose(
        actual[finite], expected[finite], rtol=RTOL, atol=ATOL, err_msg=what
    )


def _rows(count: int, buckets: int, *, seed: int = 0, scale: float = 1.0) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return generator.standard_normal((count, buckets)) * scale


# -- block_factors: the blocks tile the selection exactly --------------------


def _explicit_block_factors(rows, index, counts):
    """Every block summed by explicit indexing, written independently."""

    sums, selves, lo = [], [], 0
    for size in counts:
        members = index[lo : lo + int(size)]
        sums.append(rows[members].sum(axis=0))
        selves.append(float((rows[members] ** 2).sum()))
        lo += int(size)
    return sums, selves


@pytest.mark.parametrize(
    "counts",
    [
        np.array([3, 3, 3, 3], dtype=np.int64),
        np.array([1, 1, 1, 1, 1, 1], dtype=np.int64),    # all singletons
        np.array([5, 2, 2, 9, 1], dtype=np.int64),       # ragged
        np.array([7], dtype=np.int64),                   # a single block
        np.array([2, 2, 5, 5, 5, 1, 1], dtype=np.int64),
    ],
)
def test_blocks_tile_the_selection_and_sum_independently(counts) -> None:
    """Each block must consume exactly its own rows, in order.

    An off-by-one in the running offset would silently mix two classes together
    and still produce a plausible, finite, wrong number -- which is why this is
    asserted bitwise against explicit indexing rather than by a shape check.
    """

    total = int(counts.sum())
    rows = _rows(total, 6, seed=1)
    index = np.random.default_rng(31).permutation(total)
    row_squared = np.einsum("ij,ij->i", rows, rows)

    sums, selves = est.block_factors(rows, row_squared, index, counts)
    expected_sums, expected_selves = _explicit_block_factors(rows, index, counts)
    for block in range(counts.size):
        assert np.array_equal(sums[block], expected_sums[block]), block
        assert selves[block] == pytest.approx(expected_selves[block], rel=1e-12)


def test_indexing_inside_the_reduction_equals_gathering_first() -> None:
    """The pass-saving is an implementation detail; the values are not.

    Gathering the rows into a reordered copy and reducing that must give bitwise
    the same sums, or the optimization has changed the arithmetic.
    """

    counts = np.array([4, 1, 3, 2], dtype=np.int64)
    total = int(counts.sum())
    rows = _rows(total, 5, seed=32)
    index = np.random.default_rng(33).permutation(total)
    row_squared = np.einsum("ij,ij->i", rows, rows)

    fused, _ = est.block_factors(rows, row_squared, index, counts)
    gathered = rows[index]
    lo = 0
    for block, size in enumerate(counts):
        assert np.array_equal(fused[block], gathered[lo : lo + size].sum(axis=0))
        lo += size


def test_block_factors_write_into_supplied_buffers() -> None:
    counts = np.array([2, 2, 3], dtype=np.int64)
    rows = _rows(7, 4, seed=3)
    index = np.arange(7)
    row_squared = np.einsum("ij,ij->i", rows, rows)
    sums_out = np.empty((3, 4))
    self_out = np.empty(3)
    sums, selves = est.block_factors(
        rows, row_squared, index, counts, sums_out=sums_out, self_out=self_out
    )
    assert sums is sums_out and selves is self_out
    expected_sums, _ = _explicit_block_factors(rows, index, counts)
    for block in range(3):
        assert np.array_equal(sums_out[block], expected_sums[block])


def test_blocks_must_tile_the_selection_exactly() -> None:
    rows = _rows(5, 2)
    with pytest.raises(ValueError, match="exactly tile"):
        est.block_factors(
            rows, np.einsum("ij,ij->i", rows, rows), np.arange(5),
            np.array([2, 2], dtype=np.int64),
        )


def test_wrongly_shaped_buffers_are_refused() -> None:
    rows = _rows(4, 3)
    row_squared = np.einsum("ij,ij->i", rows, rows)
    counts = np.array([2, 2], dtype=np.int64)
    with pytest.raises(ValueError, match="sums_out must have shape"):
        est.block_factors(
            rows, row_squared, np.arange(4), counts, sums_out=np.empty((2, 2))
        )
    with pytest.raises(ValueError, match="self_out must have shape"):
        est.block_factors(
            rows, row_squared, np.arange(4), counts, self_out=np.empty(3)
        )


# -- class factors -----------------------------------------------------------


def test_factors_match_the_legacy_class_sums() -> None:
    generator = np.random.default_rng(5)
    labels = generator.integers(0, 6, size=40)
    classes = np.unique(labels)
    unit = _rows(40, 8, seed=6)

    legacy_sums, legacy_counts, legacy_self = _class_sums(unit, labels, classes)
    factors = est.class_factors(unit, labels, classes)

    assert np.array_equal(factors.counts, legacy_counts)
    assert np.array_equal(factors.classes, classes)
    assert_matching(factors.sums, legacy_sums, what="class sums")
    assert_matching(factors.self_squared, legacy_self, what="self squared")


def test_rows_outside_the_class_set_are_excluded() -> None:
    """The display subset reuses this, so it must not quietly absorb the rest of
    the population into whatever class happens to be nearest."""

    unit = _rows(6, 3, seed=7)
    labels = np.array([1, 2, 99, 1, 2, 77], dtype=np.int64)
    factors = est.class_factors(unit, labels, np.array([1, 2], dtype=np.int64))
    assert np.array_equal(factors.counts, np.array([2, 2]))
    assert factors.num_positions == 4
    assert_matching(factors.sums[0], unit[0] + unit[3], what="class 1 sum")


def test_the_self_term_is_measured_not_the_class_count() -> None:
    """Rows are sketches over exact norms, so their lengths scatter around 1.

    Planting rows whose squared norms are nowhere near 1 makes a mistaken
    ``n_a`` subtraction fail loudly instead of hiding inside a plausible number.
    """

    unit = np.array([[3.0, 0.0], [0.0, 4.0], [5.0, 0.0]])
    labels = np.array([1, 1, 1], dtype=np.int64)
    factors = est.class_factors(unit, labels, np.array([1], dtype=np.int64))
    assert factors.self_squared[0] == pytest.approx(9.0 + 16.0 + 25.0)
    assert factors.self_squared[0] != factors.counts[0]


def test_classes_may_not_repeat() -> None:
    with pytest.raises(ValueError, match="must not repeat"):
        est.class_factors(
            _rows(3, 2), np.array([1, 1, 2]), np.array([1, 1, 2])
        )


def test_label_and_row_counts_must_agree() -> None:
    with pytest.raises(ValueError, match="labels has"):
        est.class_factors(_rows(4, 2), np.array([1, 2]), np.array([1, 2]))


# -- pooled statistics against explicit pair enumeration ---------------------


def _explicit_pooled(unit: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Within and between by enumerating every ordered pair.

    The definition the factorized form is supposed to be equal to, written the
    slow obvious way so the two cannot be wrong together.
    """

    gram = unit @ unit.T
    count = unit.shape[0]
    within_num = within_den = between_num = between_den = 0.0
    for a in range(count):
        for b in range(count):
            if a == b:
                continue
            if labels[a] == labels[b]:
                within_num += gram[a, b]
                within_den += 1
            else:
                between_num += gram[a, b]
                between_den += 1
    return (
        within_num / within_den if within_den else float("nan"),
        between_num / between_den if between_den else float("nan"),
    )


def test_pooled_factors_equal_explicit_row_pair_enumeration() -> None:
    generator = np.random.default_rng(11)
    labels = generator.integers(0, 4, size=24)
    classes = np.unique(labels)
    unit = _rows(24, 5, seed=12)

    factors = est.class_factors(unit, labels, classes)
    within, between = est.pooled_from_factors(
        factors.sums, factors.counts, factors.self_squared
    )
    expected_within, expected_between = _explicit_pooled(unit, labels)
    assert within == pytest.approx(expected_within, rel=1e-12)
    assert between == pytest.approx(expected_between, rel=1e-12)


def test_pooled_factors_hold_when_row_norms_are_far_from_one() -> None:
    """The same equality, on rows whose lengths make an ``n_a`` subtraction
    visibly wrong."""

    unit = _rows(20, 4, seed=13, scale=7.5)
    labels = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 3] * 2, dtype=np.int64)
    factors = est.class_factors(unit, labels, np.unique(labels))
    within, between = est.pooled_from_factors(
        factors.sums, factors.counts, factors.self_squared
    )
    expected_within, expected_between = _explicit_pooled(unit, labels)
    assert within == pytest.approx(expected_within, rel=1e-12)
    assert between == pytest.approx(expected_between, rel=1e-12)


def test_pooled_matches_the_legacy_implementation() -> None:
    generator = np.random.default_rng(14)
    labels = generator.integers(0, 5, size=30)
    classes = np.unique(labels)
    unit = _rows(30, 6, seed=15)
    legacy = _pooled_from_blocks(*_class_sums(unit, labels, classes))
    factors = est.class_factors(unit, labels, classes)
    current = est.pooled_from_factors(
        factors.sums, factors.counts, factors.self_squared
    )
    assert_matching(np.array(current), np.array(legacy), what="pooled")


# -- geometric correctness ---------------------------------------------------


def test_identical_rows_give_within_one() -> None:
    unit = np.tile(np.array([[0.6, 0.8]]), (4, 1))
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    factors = est.class_factors(unit, labels, np.unique(labels))
    within, between = est.pooled_from_factors(
        factors.sums, factors.counts, factors.self_squared
    )
    assert within == pytest.approx(1.0)
    assert between == pytest.approx(1.0)


def test_balanced_antipodal_classes_give_a_negative_within() -> None:
    """The diagonal-excluded within must be able to go negative.

    A class of ``u`` and ``-u`` has exactly one distinct pair and it is
    perfectly opposed. Reporting ``+1`` here -- which is what a self-similarity
    convention or an ``n_a`` subtraction would produce -- would invert the
    scientific reading of the figure.
    """

    direction = np.array([0.0, 1.0])
    unit = np.stack([direction, -direction, direction, -direction])
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    factors = est.class_factors(unit, labels, np.unique(labels))
    within, _ = est.pooled_from_factors(
        factors.sums, factors.counts, factors.self_squared
    )
    assert within == pytest.approx(-1.0)


def test_orthogonal_classes_give_zero_between() -> None:
    unit = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    factors = est.class_factors(unit, labels, np.unique(labels))
    within, between = est.pooled_from_factors(
        factors.sums, factors.counts, factors.self_squared
    )
    assert within == pytest.approx(1.0)
    assert between == pytest.approx(0.0)


def test_an_all_singleton_grouping_has_no_within_estimate() -> None:
    """Unavailable, not zero. A singleton has no distinct pair to average."""

    unit = _rows(5, 3, seed=16)
    labels = np.arange(5, dtype=np.int64)
    factors = est.class_factors(unit, labels, labels)
    within, between = est.pooled_from_factors(
        factors.sums, factors.counts, factors.self_squared
    )
    assert np.isnan(within)
    assert np.isfinite(between)


def test_a_singleton_still_forms_between_pairs() -> None:
    """Dropping it would silently redefine 'between' as 'between non-singleton
    classes', which moves with the condition being compared."""

    unit = _rows(5, 3, seed=17)
    labels = np.array([0, 0, 0, 0, 1], dtype=np.int64)
    factors = est.class_factors(unit, labels, np.unique(labels))
    _, between = est.pooled_from_factors(
        factors.sums, factors.counts, factors.self_squared
    )
    assert np.isfinite(between)
    assert factors.num_positions == 5


def test_sparse_and_unequal_class_sizes_match_explicit_pairs() -> None:
    labels = np.array([0] * 12 + [1, 1, 1] + [2, 2] + [3], dtype=np.int64)
    unit = _rows(labels.size, 5, seed=18)
    factors = est.class_factors(unit, labels, np.unique(labels))
    within, between = est.pooled_from_factors(
        factors.sums, factors.counts, factors.self_squared
    )
    expected_within, expected_between = _explicit_pooled(unit, labels)
    assert within == pytest.approx(expected_within, rel=1e-12)
    assert between == pytest.approx(expected_between, rel=1e-12)


# -- ensemble semantics ------------------------------------------------------


def _ensemble_reference(per_map: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Within/between from the concatenated embedding, the v12 route."""

    maps = per_map.shape[1]
    scale = 1.0 if maps == 1 else 1.0 / np.sqrt(float(maps))
    embedded = np.concatenate(
        [per_map[:, index, :] * scale for index in range(maps)], axis=1
    )
    classes = np.unique(labels)
    return _pooled_from_blocks(*_class_sums(embedded, labels, classes))


@pytest.mark.parametrize("maps", [1, 4])
def test_the_ensemble_is_the_arithmetic_mean_over_maps(maps) -> None:
    """Inner products are averaged, never sketch vectors.

    Averaging the sketches and taking one inner product afterwards is a
    different and wrong quantity -- it shrinks the very signal each independent
    map carries.
    """

    generator = np.random.default_rng(19)
    labels = generator.integers(0, 4, size=28)
    classes = np.unique(labels)
    per_map = generator.standard_normal((28, maps, 7))

    withins, betweens = [], []
    for index in range(maps):
        factors = est.class_factors(per_map[:, index, :], labels, classes)
        within, between = est.pooled_from_factors(
            factors.sums, factors.counts, factors.self_squared
        )
        withins.append(within)
        betweens.append(between)

    expected_within, expected_between = _ensemble_reference(per_map, labels)
    assert float(np.mean(withins)) == pytest.approx(expected_within, rel=1e-12)
    assert float(np.mean(betweens)) == pytest.approx(expected_between, rel=1e-12)


# -- the display matrix ------------------------------------------------------


def test_the_similarity_matrix_matches_the_legacy_one() -> None:
    generator = np.random.default_rng(21)
    labels = generator.integers(0, 5, size=26)
    classes = np.unique(labels)
    unit = _rows(26, 6, seed=22)

    legacy = class_similarity_matrix(unit, labels, classes)
    current = est.class_similarity_from_factors(
        est.class_factors(unit, labels, classes)
    )

    assert_matching(current["matrix"], legacy["matrix"], what="display matrix")
    assert_matching(current["within"], legacy["within"], what="within by class")
    assert np.array_equal(current["counts"], legacy["counts"])
    assert current["num_within_measurable"] == legacy["num_within_measurable"]


def test_the_diagonal_of_a_singleton_class_is_unavailable() -> None:
    unit = _rows(4, 3, seed=23)
    labels = np.array([0, 0, 0, 1], dtype=np.int64)
    result = est.class_similarity_from_factors(
        est.class_factors(unit, labels, np.unique(labels))
    )
    assert np.isfinite(result["matrix"][0, 0])
    assert np.isnan(result["matrix"][1, 1])
    assert np.isfinite(result["matrix"][0, 1])


# -- the permutation null ----------------------------------------------------


def test_the_draw_sequence_is_the_legacy_sequence() -> None:
    """The epoch changes accumulation, never which permutations are drawn."""

    counts = np.array([4, 3, 2, 1], dtype=np.int64)
    for current, legacy in zip(
        est.permutation_orders(counts, permutations=16, seed=99),
        legacy_permutation_orders(counts, permutations=16, seed=99),
    ):
        assert np.array_equal(current, legacy)


def test_the_null_matches_the_legacy_null_within_tolerance() -> None:
    generator = np.random.default_rng(24)
    labels = generator.integers(0, 5, size=48)
    classes, counts = np.unique(labels, return_counts=True)
    unit = _rows(48, 9, seed=25)

    legacy = _permutation_null(unit, counts, permutations=64, seed=31)
    current = est.permutation_null(
        unit, counts, permutations=64, seed=31, draw_batch=16
    )

    assert current["permutations"] == legacy["permutations"]
    assert current["seed"] == legacy["seed"]
    for key in (
        "within_mean", "between_mean", "delta_mean",
        "delta_std", "delta_low", "delta_high",
    ):
        assert_matching(
            np.array(current[key]), np.array(legacy[key]), what=f"null {key}"
        )


@pytest.mark.parametrize("draw_batch", [1, 3, 16, 64, 128])
def test_the_draw_batch_is_a_memory_knob_only(draw_batch) -> None:
    counts = np.array([5, 4, 3, 2, 1], dtype=np.int64)
    unit = _rows(15, 4, seed=26)
    reference = est.permutation_null(
        unit, counts, permutations=32, seed=7, draw_batch=32
    )
    assert est.permutation_null(
        unit, counts, permutations=32, seed=7, draw_batch=draw_batch
    ) == reference


def test_the_null_holds_no_full_size_scratch_buffer(monkeypatch) -> None:
    """Indexing inside the reduction removes the ``[D, K]`` gather entirely.

    A reintroduced buffer would be 256 MiB at experiment scale and would not
    show up in any value, so it is trapped by allocation rather than by result.
    """

    counts = np.array([4, 4, 2], dtype=np.int64)
    unit = _rows(10, 5, seed=27)
    guard = _AllocationGuard(monkeypatch, max_elements=counts.size * 5 * 2)
    est.permutation_null(unit, counts, permutations=8, seed=8)
    assert guard.largest <= counts.size * 5 * 2


def test_an_all_singleton_null_reports_unavailable_within() -> None:
    unit = _rows(8, 3, seed=28)
    counts = np.ones(8, dtype=np.int64)
    result = est.permutation_null(unit, counts, permutations=8, seed=9)
    assert np.isnan(result["within_mean"])
    assert np.isnan(result["delta_mean"])
    assert np.isfinite(result["between_mean"])


def test_the_null_refuses_a_count_vector_that_does_not_cover_the_rows() -> None:
    with pytest.raises(ValueError, match="counts sum to"):
        est.permutation_null(
            _rows(6, 2), np.array([2, 2], dtype=np.int64), permutations=4, seed=1
        )


@pytest.mark.parametrize("bad", [0, -1])
def test_the_null_refuses_a_non_positive_draw_count(bad) -> None:
    with pytest.raises(ValueError, match="permutations must be positive"):
        est.permutation_null(
            _rows(4, 2), np.array([2, 2], dtype=np.int64), permutations=bad, seed=1
        )


# -- the epoch is declared, not implicit -------------------------------------


def test_the_conventions_are_named() -> None:
    """These strings land in record metadata, so a stored result always says
    which accumulation produced it."""

    assert est.ESTIMATOR_CONVENTION == "countsketch_pairwise_mean_q_self/v1"
    assert est.REDUCTION_CONVENTION == "blockwise_class_sum/v1"
    assert est.NUMERICAL_EPOCH == "alignment_reduction_epoch/v1"


# -- cross-partition factors -------------------------------------------------
#
# Target and greedy partition the SAME positions, so every position sits in one
# cell of the contingency and every occupied cell -- not just the diagonal --
# must subtract the positions it shares. These pin that against the established
# row-based route, and pin that the sparse contingency is not an approximation
# of the dense one but the same numbers without the 8 GB.


def _cross_inputs(size: int = 36, buckets: int = 7, seed: int = 41):
    generator = np.random.default_rng(seed)
    targets = generator.integers(0, 5, size=size)
    greedy = generator.integers(0, 5, size=size)
    unit = _rows(size, buckets, seed=seed + 1)
    classes = np.union1d(np.unique(targets), np.unique(greedy))
    return unit, targets, greedy, classes


def test_pooled_cross_matches_the_legacy_route() -> None:
    from llm_behavior_lab.analysis.gradient_cross_partition import (
        pooled_cross_statistic,
    )

    unit, targets, greedy, classes = _cross_inputs()
    legacy = pooled_cross_statistic(unit, targets, greedy)
    current = est.pooled_cross_from_factors(
        est.cross_factors(unit, targets, greedy, classes)
    )

    assert np.array_equal(current["classes"], legacy["classes"])
    assert current["num_same_pairs"] == legacy["num_same_pairs"]
    assert current["num_different_pairs"] == legacy["num_different_pairs"]
    assert (
        current["num_true_positive_positions"]
        == legacy["num_true_positive_positions"]
    )
    for key in ("c_same", "c_different", "delta_cross"):
        assert_matching(
            np.array(current[key]), np.array(legacy[key]), what=f"cross {key}"
        )


def test_the_cross_matrix_matches_the_legacy_route() -> None:
    from llm_behavior_lab.analysis.gradient_cross_partition import (
        cross_partition_matrix,
    )

    unit, targets, greedy, classes = _cross_inputs()
    shown = classes[:4]
    slots = np.searchsorted(classes, shown)

    legacy = cross_partition_matrix(unit, targets, greedy, shown, shown)
    current = est.cross_matrix_from_factors(
        est.cross_factors(unit, targets, greedy, classes), slots, slots
    )

    assert_matching(current["matrix"], legacy["matrix"], what="cross matrix")
    assert np.array_equal(current["contingency"], legacy["contingency"])
    assert np.array_equal(current["target_counts"], legacy["target_counts"])
    assert np.array_equal(current["greedy_counts"], legacy["greedy_counts"])
    assert current["num_usable_cells"] == legacy["num_usable_cells"]


def test_the_off_diagonal_correction_is_applied_everywhere() -> None:
    """At initialization the two labels differ almost everywhere, so the
    off-diagonal cells are the ones actually being read. A correction applied
    only to the diagonal would leave every one of them including positions
    compared against themselves."""

    unit = _rows(6, 4, seed=44)
    targets = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    greedy = np.array([1, 1, 1, 0, 0, 0], dtype=np.int64)
    classes = np.array([0, 1], dtype=np.int64)
    factors = est.cross_factors(unit, targets, greedy, classes)

    # Every position sits in an off-diagonal cell, so both off-diagonal cells
    # carry a non-zero self term and both diagonal cells are empty.
    diagonal = factors.cell_target_index == factors.cell_greedy_index
    assert not diagonal.any()
    assert factors.cell_self_squared.sum() == pytest.approx(
        factors.total_self_squared
    )


def test_the_contingency_is_sparse_not_dense() -> None:
    """A dense table over a subword vocabulary is about 8 GB and almost entirely
    zero. Only occupied cells may be materialized."""

    unit = _rows(12, 3, seed=45)
    targets = np.arange(12, dtype=np.int64) % 3
    greedy = np.arange(12, dtype=np.int64) % 3
    classes = np.arange(4000, dtype=np.int64)
    factors = est.cross_factors(unit, targets, greedy, classes)
    assert factors.cell_counts.size <= 12
    assert int(factors.cell_counts.sum()) == 12


def test_every_position_lands_in_exactly_one_cell() -> None:
    unit, targets, greedy, classes = _cross_inputs()
    factors = est.cross_factors(unit, targets, greedy, classes)
    assert int(factors.cell_counts.sum()) == targets.size
    assert factors.total_self_squared == pytest.approx(
        float(np.einsum("ij,ij->i", unit, unit).sum())
    )


def test_cross_factors_refuse_mismatched_label_vectors() -> None:
    with pytest.raises(ValueError, match="same positions"):
        est.cross_factors(
            _rows(4, 2), np.array([0, 1, 0, 1]), np.array([0, 1]), np.array([0, 1])
        )


# -- the sparse contingency is a scaling property, not an implementation note --
#
# The v12 route builds a dense [C, C] contingency (`gradient_cross_partition.py`
# `_contingency`, reached from `pooled_cross_statistic`). Over a subword
# vocabulary that is gigabytes of almost entirely zero, and it is the most
# likely cause of the minutes-per-arm cross-partition cost. The v13 path lists
# only occupied cells instead. That is a guarantee worth a trap rather than a
# comment: a future refactor could reintroduce a dense table and every test
# above would still pass, because the numbers would be right.


class _AllocationGuard:
    """Refuse any single array allocation above a ceiling.

    Patched over the constructors the dense route would have to reach for. It
    watches the *requested shape*, so it fires before the memory is committed
    rather than after the machine has started swapping.
    """

    def __init__(self, monkeypatch, *, max_elements: int) -> None:
        self.max_elements = max_elements
        self.largest = 0
        for name in ("zeros", "empty", "full", "ones"):
            original = getattr(np, name)
            monkeypatch.setattr(np, name, self._wrap(original))

    def _wrap(self, original):
        def guarded(shape, *args, **kwargs):
            try:
                size = int(np.prod(shape)) if np.ndim(shape) else int(shape)
            except (TypeError, ValueError):
                size = 0
            self.largest = max(self.largest, size)
            if size > self.max_elements:
                raise AssertionError(
                    f"Allocated an array of {size} elements, above the "
                    f"{self.max_elements} ceiling. A dense [C, C] contingency "
                    "has been reintroduced."
                )
            return original(shape, *args, **kwargs)

        return guarded


def test_the_cross_path_never_allocates_a_dense_contingency(monkeypatch) -> None:
    """32000 classes, 64 positions: a dense table would be 1.024e9 cells.

    The ceiling is set far above everything the sparse path legitimately needs
    (the largest is the [C, K] class-sum block) and far below the dense table, so
    it separates the two without being sensitive to incidental allocations.
    """

    vocabulary = 32000
    buckets = 4
    positions = 64
    generator = np.random.default_rng(51)
    unit = _rows(positions, buckets, seed=52)
    targets = generator.integers(0, 40, size=positions)
    greedy = generator.integers(0, 40, size=positions)
    classes = np.arange(vocabulary, dtype=np.int64)

    ceiling = 8 * vocabulary * buckets
    guard = _AllocationGuard(monkeypatch, max_elements=ceiling)

    factors = est.cross_factors(unit, targets, greedy, classes)
    pooled = est.pooled_cross_from_factors(factors)

    assert guard.largest <= ceiling
    assert guard.largest < vocabulary * vocabulary
    assert np.isfinite(pooled["c_same"]) or np.isnan(pooled["c_same"])


def test_the_occupied_cell_count_bounds_the_contingency_storage() -> None:
    """Storage scales with occupied cells, never with the class set.

    At most ``D`` cells can be occupied, because every position lies in exactly
    one. Growing the class set a thousandfold must not grow the contingency.
    """

    unit = _rows(48, 5, seed=53)
    generator = np.random.default_rng(54)
    targets = generator.integers(0, 6, size=48)
    greedy = generator.integers(0, 6, size=48)

    small = est.cross_factors(unit, targets, greedy, np.arange(32, dtype=np.int64))
    large = est.cross_factors(unit, targets, greedy, np.arange(32000, dtype=np.int64))

    assert small.cell_counts.size == large.cell_counts.size
    assert large.cell_counts.size <= 48
    assert_matching(
        large.cell_self_squared, small.cell_self_squared, what="cell self terms"
    )


def test_widening_the_class_set_does_not_change_the_pooled_result() -> None:
    """Unrepresented classes contribute nothing, so padding the vocabulary must
    be invisible in the numbers as well as in the storage."""

    unit = _rows(40, 6, seed=55)
    generator = np.random.default_rng(56)
    targets = generator.integers(0, 7, size=40)
    greedy = generator.integers(0, 7, size=40)

    narrow = est.pooled_cross_from_factors(
        est.cross_factors(unit, targets, greedy, np.arange(8, dtype=np.int64))
    )
    wide = est.pooled_cross_from_factors(
        est.cross_factors(unit, targets, greedy, np.arange(5000, dtype=np.int64))
    )
    for key in ("c_same", "c_different", "delta_cross"):
        assert_matching(np.array(wide[key]), np.array(narrow[key]), what=key)
    assert wide["num_same_pairs"] == narrow["num_same_pairs"]
    assert wide["num_different_pairs"] == narrow["num_different_pairs"]


def test_the_v12_dense_route_is_left_exactly_as_it_was() -> None:
    """The scaling problem is fixed forward, in v13, not by editing v12.

    Existing records were analysed with the dense implementation and their
    published numbers came from it; rewriting it would silently change what a
    re-analysis of those records produces.
    """

    import inspect

    from llm_behavior_lab.analysis import gradient_cross_partition as legacy

    source = inspect.getsource(legacy._contingency)
    assert "np.zeros((target_classes.size, greedy_classes.size)" in source
    assert "np.zeros((target_classes.size, greedy_classes.size)" not in inspect.getsource(
        est.cross_factors
    )
