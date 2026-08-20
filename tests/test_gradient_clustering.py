"""Tests for directional clustering of per-position gradients.

Two things need to be true and are tested separately.

The **estimator** must be right: within- and between-class mean cosines are
computed from class sums rather than by forming pairs, which is an exact
algebraic identity but an easy one to get subtly wrong. Every such quantity is
checked against a brute-force double loop over explicit pairs.

The **finding** must be detectable: on synthetic sketches planted with known
structure, clustering must be reported when it is there, absent when it is not,
and negative when subgroups genuinely oppose. A diagnostic that only ever says
"structure" would be useless, so the null and negative cases are tested as
carefully as the positive one.

These tests are NumPy-only. The count-sketch projection itself needs PyTorch and
is covered in ``tests/test_gradient_sketch.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from llm_behavior_lab.analysis import InitializationExperimentRecord
from llm_behavior_lab.analysis.gradient_clustering import (
    class_similarity_matrix,
    clustering_summary,
    gradient_clustering,
    unit_sketches,
)

VOCAB = 12
ELIGIBLE = np.arange(2, VOCAB)
K = 64


def _unit(rows: np.ndarray) -> np.ndarray:
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def _record(
    sketches: np.ndarray,
    targets: np.ndarray,
    greedy: np.ndarray,
    norms: np.ndarray | None = None,
):
    """A record carrying gradient sketches and the two label vectors."""

    positions = targets.size
    corpus = np.zeros(VOCAB, dtype=np.int64)
    corpus[ELIGIBLE] = 11
    corpus[targets] = np.maximum(corpus[targets], 1)
    return InitializationExperimentRecord.build(
        corpus_counts=corpus,
        selected_target_counts=np.bincount(targets, minlength=VOCAB),
        greedy_counts=np.bincount(greedy, minlength=VOCAB)[None, :],
        nucleus_counts=np.bincount(greedy, minlength=VOCAB)[None, None, :],
        mean_predicted_probabilities=np.full((1, VOCAB), 1.0 / VOCAB),
        model_seeds=[1000],
        eligible_token_ids=ELIGIBLE,
        metadata={
            "num_positions": positions,
            "analysis": {
                "num_positions": positions,
                "gradient_analysis": {
                    "enabled": True,
                    "initialization_index": 0,
                    "covers_all_positions": True,
                    "gradient_sketch": {"dimension": K, "seed": 1},
                },
            },
        },
        gradient_position_indices=np.arange(positions),
        gradient_position_target_ids=targets,
        gradient_position_greedy_ids=greedy,
        # The exact norms are the divisor. Unit norms make u_d equal the
        # sketch, which keeps the planted geometry easy to reason about.
        gradient_position_norms=(
            np.ones(positions) if norms is None else np.asarray(norms, dtype=float)
        ),
        gradient_position_sketches=sketches,
    )


def _clustered(num_classes: int, per_class: int, alignment: float, seed: int = 5):
    """Sketches whose within-class alignment is controlled by ``alignment``.

    Each class gets a fixed centre; every member is that centre mixed with fresh
    noise. ``alignment = 0`` gives pure noise and no structure at all.
    """

    generator = np.random.default_rng(seed)
    centres = generator.normal(size=(num_classes, K))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    rows, labels = [], []
    for index in range(num_classes):
        noise = generator.normal(size=(per_class, K))
        noise /= np.linalg.norm(noise, axis=1, keepdims=True)
        rows.append(alignment * centres[index] + (1.0 - alignment) * noise)
        labels.extend([ELIGIBLE[index]] * per_class)
    return _unit(np.vstack(rows)), np.asarray(labels, dtype=np.int64)


# -- the estimator, against brute force --------------------------------------


def _brute_force(unit: np.ndarray, labels: np.ndarray, classes: np.ndarray):
    """Mean cosines from an explicit double loop over pairs."""

    size = classes.size
    matrix = np.full((size, size), np.nan)
    for i, first in enumerate(classes):
        for j, second in enumerate(classes):
            left = np.flatnonzero(labels == first)
            right = np.flatnonzero(labels == second)
            values = [
                float(unit[a] @ unit[b])
                for a in left
                for b in right
                if not (i == j and a == b)
            ]
            if values:
                matrix[i, j] = float(np.mean(values))
    return matrix


def test_the_similarity_matrix_matches_an_explicit_pair_computation() -> None:
    """The class-sum identity must reproduce brute force exactly."""

    unit, labels = _clustered(num_classes=4, per_class=6, alignment=0.6)
    classes = np.unique(labels)

    result = class_similarity_matrix(unit, labels, classes)
    expected = _brute_force(unit, labels, classes)

    assert np.allclose(result["matrix"], expected, rtol=0, atol=1e-12, equal_nan=True)


def test_the_diagonal_excludes_self_comparisons() -> None:
    """A diagonal fixed to 1 would hide the entire effect being measured."""

    unit, labels = _clustered(num_classes=3, per_class=5, alignment=0.5)
    classes = np.unique(labels)

    result = class_similarity_matrix(unit, labels, classes)
    diagonal = np.diag(result["matrix"])

    assert np.all(np.isfinite(diagonal))
    assert np.all(diagonal < 1.0 - 1e-9)
    for index, token in enumerate(classes):
        members = np.flatnonzero(labels == token)
        pairs = [
            float(unit[a] @ unit[b]) for a in members for b in members if a != b
        ]
        assert diagonal[index] == pytest.approx(float(np.mean(pairs)), abs=1e-12)


def test_a_singleton_class_has_no_within_class_value() -> None:
    """One member admits no distinct pair; that is undefined, not zero."""

    unit, labels = _clustered(num_classes=3, per_class=1, alignment=0.9)
    classes = np.unique(labels)

    result = class_similarity_matrix(unit, labels, classes)

    assert np.all(np.isnan(np.diag(result["matrix"])))
    assert result["num_within_measurable"] == 0


# -- the finding: positive, null and negative --------------------------------


def test_planted_clustering_is_detected() -> None:
    unit, labels = _clustered(num_classes=5, per_class=12, alignment=0.8)
    record = _record(unit, labels, labels)

    result = gradient_clustering(record, grouping="target")

    assert result["population"]["within"] > result["population"]["between"]
    assert result["population"]["delta"] > 0.2
    # And the permutation reference must not reproduce it.
    assert abs(result["null"]["delta_mean"]) < 0.05


def test_unstructured_gradients_show_no_clustering() -> None:
    """The null result must be reported as a null, not as weak structure."""

    unit, labels = _clustered(num_classes=5, per_class=12, alignment=0.0)
    record = _record(unit, labels, labels)

    result = gradient_clustering(record, grouping="target")

    assert abs(result["population"]["delta"]) < 0.05
    # Within noise of the null. Deliberately not "inside the 95% interval": with
    # no planted structure the observed delta is itself a draw from that null, so
    # it falls outside a 95% interval about one time in twenty and such an
    # assertion would flake by construction.
    null = result["null"]
    assert abs(result["population"]["delta"] - null["delta_mean"]) < 4.0 * null["delta_std"]


def test_opposing_subgroups_give_a_negative_delta() -> None:
    """Anti-alignment is a different finding from absence and must look it."""

    generator = np.random.default_rng(11)
    axis = generator.normal(size=K)
    axis /= np.linalg.norm(axis)
    # Two classes whose members straddle one axis: within-class pairs disagree,
    # so within-class similarity falls below the between-class level.
    rows, labels = [], []
    for index in range(2):
        half = np.tile(axis, (6, 1)) * np.array([1, -1] * 3)[:, None]
        rows.append(half + 0.05 * generator.normal(size=(6, K)))
        labels.extend([ELIGIBLE[index]] * 6)
    unit = _unit(np.vstack(rows))
    record = _record(unit, np.asarray(labels), np.asarray(labels))

    result = gradient_clustering(record, grouping="target")

    assert result["population"]["within"] < 0.0
    assert result["population"]["delta"] < 0.0


def test_the_two_groupings_are_computed_independently() -> None:
    """Target and greedy labels answer different questions and can disagree.

    The greedy labels here cut across the planted target structure by taking one
    member from each target class in turn. That does not merely erase the
    signal, it inverts it: a greedy class of eight has only 4 of its 28 internal
    pairs sharing a target, while a pair of greedy classes has 16 of 64, so
    between-class similarity ends up *above* within-class and the delta comes
    out negative. Keeping both groupings separate is what makes that visible
    instead of being averaged away.
    """

    unit, targets = _clustered(num_classes=4, per_class=8, alignment=0.85)
    greedy = np.asarray(
        [ELIGIBLE[index % 4] for index in range(targets.size)], dtype=np.int64
    )
    record = _record(unit, targets, greedy)

    summary = clustering_summary(record)
    target_delta = summary["target"]["population"]["delta"]
    greedy_delta = summary["greedy"]["population"]["delta"]

    assert target_delta > 0.5
    assert greedy_delta < 0.0
    # The two groupings must not be reporting the same number.
    assert target_delta - greedy_delta > 0.5


# -- housekeeping ------------------------------------------------------------


def test_the_analysis_is_deterministic() -> None:
    unit, labels = _clustered(num_classes=4, per_class=7, alignment=0.5)
    record = _record(unit, labels, labels)

    first = gradient_clustering(record, grouping="target")
    second = gradient_clustering(record, grouping="target")

    assert np.array_equal(first["display"]["matrix"], second["display"]["matrix"], equal_nan=True)
    assert first["population"]["delta"] == second["population"]["delta"]
    assert first["null"] == second["null"]


def test_the_permutation_null_preserves_class_sizes() -> None:
    unit, labels = _clustered(num_classes=4, per_class=9, alignment=0.7)
    record = _record(unit, labels, labels)

    result = gradient_clustering(record, grouping="target")

    assert result["null"]["permutations"] == 256
    assert result["null"]["delta_low"] <= result["null"]["delta_mean"]
    assert result["null"]["delta_mean"] <= result["null"]["delta_high"]


def test_gradient_magnitude_divides_out() -> None:
    """Direction only: scaling a gradient must not change any similarity.

    Scaling ``g_d`` scales its sketch by the same factor, and the divisor is the
    exact norm, so the two move together and every cosine is unchanged. That is
    the invariance the implemented definition actually has -- scaling the sketch
    *alone* would change the answer, and should.
    """

    unit, labels = _clustered(num_classes=3, per_class=6, alignment=0.7)
    factors = np.linspace(0.1, 50.0, unit.shape[0])

    plain = gradient_clustering(_record(unit, labels, labels), grouping="target")
    rescaled = gradient_clustering(
        _record(unit * factors[:, None], labels, labels, norms=factors),
        grouping="target",
    )

    assert np.allclose(plain["display"]["matrix"], rescaled["display"]["matrix"], atol=1e-10, equal_nan=True)


def test_the_diagonal_subtracts_measured_self_norms_not_the_class_count() -> None:
    """The rows are not unit length, so ``n_i`` is the wrong self term.

    Sketch rows are ``sketch(g_d) / ||g_d||``; their lengths scatter around 1
    rather than equalling it. Planted far from 1 here on purpose, so an
    implementation subtracting ``n_i`` instead of ``sum_d ||u_d||^2`` gives a
    visibly different diagonal and this test fails.
    """

    from llm_behavior_lab.analysis.gradient_clustering import _class_sums

    generator = np.random.default_rng(23)
    rows = _unit(generator.normal(size=(8, K)))
    rows = rows * np.array([0.3, 0.4, 0.5, 0.6, 2.0, 2.5, 3.0, 3.5])[:, None]
    labels = np.array([ELIGIBLE[0]] * 4 + [ELIGIBLE[1]] * 4, dtype=np.int64)
    classes = np.unique(labels)

    result = class_similarity_matrix(rows, labels, classes)

    for index, token in enumerate(classes):
        members = np.flatnonzero(labels == token)
        expected = float(np.mean([
            rows[a] @ rows[b] for a in members for b in members if a != b
        ]))
        assert result["matrix"][index, index] == pytest.approx(expected, abs=1e-12)

    # The wrong estimator must be genuinely different, or this has no teeth.
    sums, counts, self_squared = _class_sums(rows, labels, classes)
    wrong = (np.einsum("ij,ij->i", sums, sums) - counts) / (counts * (counts - 1))
    assert not np.allclose(self_squared, counts)
    assert not np.allclose(wrong, np.diag(result["matrix"]), atol=1e-6)


def test_the_pooled_summary_agrees_with_the_matrix_route() -> None:
    """The O(C K) pooled identity must equal the O(C^2 K) matrix computation."""

    from llm_behavior_lab.analysis.gradient_clustering import (
        _class_sums,
        _pooled_from_blocks,
        _summary_from_matrix,
    )

    unit, labels = _clustered(num_classes=5, per_class=7, alignment=0.6)
    classes = np.unique(labels)

    matrix_route = _summary_from_matrix(class_similarity_matrix(unit, labels, classes))
    sums, counts, self_squared = _class_sums(unit, labels, classes)
    within, between = _pooled_from_blocks(sums, counts, self_squared)

    assert within == pytest.approx(matrix_route["within"], abs=1e-12)
    assert between == pytest.approx(matrix_route["between"], abs=1e-12)


def test_min_support_and_display_limits_are_reported_not_silent() -> None:
    unit, labels = _clustered(num_classes=6, per_class=4, alignment=0.5)
    record = _record(unit, labels, labels)

    limited = gradient_clustering(record, grouping="target", display_classes=3)

    assert limited["display"]["classes"].size == 3
    assert limited["min_support"] == 2
    assert limited["population"]["num_classes"] > 3


def test_a_record_without_sketches_is_refused() -> None:
    unit, labels = _clustered(num_classes=3, per_class=4, alignment=0.5)
    record = _record(unit, labels, labels)
    stripped = InitializationExperimentRecord.build(
        corpus_counts=record.corpus_counts,
        selected_target_counts=record.selected_target_counts,
        greedy_counts=record.greedy_counts,
        nucleus_counts=record.nucleus_counts,
        mean_predicted_probabilities=record.mean_predicted_probabilities,
        model_seeds=record.model_seeds,
        eligible_token_ids=record.eligible_token_ids,
        metadata=record.metadata,
        gradient_position_indices=record.gradient_position_indices,
        gradient_position_target_ids=record.gradient_position_target_ids,
        gradient_position_greedy_ids=record.gradient_position_greedy_ids,
        gradient_position_norms=record.gradient_position_norms,
    )

    assert stripped.gradient_position_sketches is None
    with pytest.raises(ValueError, match="no per-position gradient sketches"):
        unit_sketches(stripped)


def test_a_record_carrying_sketches_round_trips(tmp_path) -> None:
    """Version 10 persistence, and old records stay loadable without it."""

    from llm_behavior_lab.analysis import load_record

    unit, labels = _clustered(num_classes=3, per_class=5, alignment=0.6)
    record = _record(unit, labels, labels)
    record.save(tmp_path)

    reloaded = load_record(tmp_path)
    assert reloaded.gradient_position_sketches is not None
    assert reloaded.gradient_position_sketches.shape == (labels.size, K)
    # float32 storage: the sketch's own 1/sqrt(K) error dwarfs the rounding.
    assert np.allclose(reloaded.gradient_position_sketches, unit, atol=1e-6)


# -- the figure, on data whose expected pattern is known ---------------------


def test_the_clustering_figure_renders_and_shows_the_planted_diagonal(tmp_path):
    """A warm diagonal must actually appear when clustering is planted."""

    matplotlib = pytest.importorskip("matplotlib")
    from llm_behavior_lab.analysis.figures import plot_gradient_directional_clustering

    unit, labels = _clustered(num_classes=6, per_class=10, alignment=0.85)
    record = _record(unit, labels, labels)

    written = plot_gradient_directional_clustering(record, tmp_path, display_classes=6)

    assert len(written) == 1
    assert written[0].stat().st_size > 0
    assert written[0].name.startswith("figure20_")

    # The pattern the figure is supposed to reveal, asserted on the numbers it draws.
    result = gradient_clustering(record, grouping="target", display_classes=6)
    diagonal = np.diag(result["display"]["matrix"])
    off = result["display"]["matrix"][~np.eye(6, dtype=bool)]
    assert np.nanmin(diagonal) > np.nanmax(off)


def test_the_clustering_figure_renders_when_there_is_no_structure(tmp_path):
    """A null must render as a null rather than failing or inventing a pattern."""

    matplotlib = pytest.importorskip("matplotlib")
    from llm_behavior_lab.analysis.figures import plot_gradient_directional_clustering

    unit, labels = _clustered(num_classes=5, per_class=8, alignment=0.0)
    record = _record(unit, labels, labels)

    written = plot_gradient_directional_clustering(record, tmp_path, display_classes=5)

    assert len(written) == 1
    result = gradient_clustering(record, grouping="target", display_classes=5)
    assert abs(result["population"]["delta"]) < 0.05


# -- population versus display, the correction this phase makes ---------------


def test_the_pooled_statistic_ignores_the_display_subset() -> None:
    """Display selection must never move the population number.

    An earlier version applied one argument to both, so the reported effect
    silently shrank to whatever fitted on the axes. This is the regression guard.
    """

    unit, labels = _clustered(num_classes=8, per_class=6, alignment=0.7)
    record = _record(unit, labels, labels)

    full = gradient_clustering(record, grouping="target", display_classes=None)
    tiny = gradient_clustering(record, grouping="target", display_classes=3)

    assert full["population"]["delta"] == tiny["population"]["delta"]
    assert full["population"]["num_classes"] == tiny["population"]["num_classes"] == 8
    assert full["null"]["delta_mean"] == tiny["null"]["delta_mean"]
    # ... while the drawn matrix really did shrink.
    assert tiny["display"]["matrix"].shape == (3, 3)
    assert full["display"]["matrix"].shape == (8, 8)


def test_the_display_subset_takes_the_most_supported_classes() -> None:
    """Deterministic selection, documented in the returned description."""

    sizes = [2, 3, 4, 9, 10, 11]
    rows, labels = [], []
    generator = np.random.default_rng(3)
    for index, size in enumerate(sizes):
        rows.append(_unit(generator.normal(size=(size, K))))
        labels.extend([ELIGIBLE[index]] * size)
    record = _record(np.vstack(rows), np.asarray(labels), np.asarray(labels))

    result = gradient_clustering(record, grouping="target", display_classes=3)

    assert sorted(result["display"]["classes"]) == sorted(ELIGIBLE[3:6])
    assert "3 most supported of 6 qualifying classes" in result["display"]["selection"]
    assert result["population"]["num_classes"] == 6


def test_both_groupings_share_the_gradients_and_differ_only_in_labels() -> None:
    """Same sketches, same norms, same sketch realization; only labels change."""

    unit, targets = _clustered(num_classes=4, per_class=8, alignment=0.8)
    greedy = np.asarray(
        [ELIGIBLE[index % 4] for index in range(targets.size)], dtype=np.int64
    )
    record = _record(unit, targets, greedy)

    summary = clustering_summary(record)

    # Identical inputs: the position count and sketch width cannot differ.
    assert summary["target"]["num_positions"] == summary["greedy"]["num_positions"]
    assert summary["target"]["sketch_dimension"] == summary["greedy"]["sketch_dimension"]
    # But the labels genuinely differ, so the answers do too.
    assert not np.array_equal(
        record.gradient_position_target_ids, record.gradient_position_greedy_ids
    )
    assert summary["target"]["population"]["delta"] != summary["greedy"]["population"]["delta"]


def test_sketch_rows_stay_aligned_with_both_label_arrays() -> None:
    """Row d must be the same position across sketch, norm, target and greedy."""

    unit, targets = _clustered(num_classes=4, per_class=5, alignment=0.6)
    greedy = np.roll(targets, 3)
    record = _record(unit, targets, greedy)

    positions = targets.size
    assert record.gradient_position_sketches.shape[0] == positions
    assert record.gradient_position_norms.shape[0] == positions
    assert record.gradient_position_target_ids.shape[0] == positions
    assert record.gradient_position_greedy_ids.shape[0] == positions
    assert np.array_equal(record.gradient_position_indices, np.arange(positions))
    # And the scaled rows keep that order.
    scaled, usable = unit_sketches(record)
    assert scaled.shape[0] == positions and usable.all()


# -- display labels ----------------------------------------------------------


def test_invisible_and_special_tokens_get_readable_labels() -> None:
    """No axis tick may render blank, and identity must survive decoding."""

    matplotlib = pytest.importorskip("matplotlib")
    from llm_behavior_lab.analysis.figures import _token_axis_label

    unit, labels = _clustered(num_classes=4, per_class=4, alignment=0.5)
    record = _record(unit, labels, labels)
    # A vocabulary whose awkward entries are exactly the ones that render blank
    # or break an axis if passed through untouched.
    vocabulary = [f"t{index}" for index in range(VOCAB)]
    for token_id, text in ((2, "\n"), (3, " "), (4, "<0x0A>"), (5, ""), (6, "\t")):
        vocabulary[token_id] = text
    record.metadata["tokens"] = vocabulary

    for token_id in (2, 3, 4, 5, 6):
        label = _token_axis_label(record, token_id)
        assert label.strip(), f"token {token_id} rendered blank"
        assert "\n" not in label, "a raw newline would break the axis"

    # Decoding is presentational: it must not change which class is which.
    result = gradient_clustering(record, grouping="target", display_classes=4)
    assert result["display"]["classes"].dtype.kind in "iu"
    assert set(result["display"]["classes"]).issubset(set(labels))


def test_a_record_without_a_vocabulary_falls_back_to_ids() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    from llm_behavior_lab.analysis.figures import _token_axis_label

    unit, labels = _clustered(num_classes=3, per_class=4, alignment=0.5)
    record = _record(unit, labels, labels)
    record.metadata.pop("tokens", None)

    assert _token_axis_label(record, 7) == "7"


# -- the two heatmaps must be directly comparable ----------------------------


def test_both_heatmaps_share_one_colour_normalization(tmp_path) -> None:
    """Independently rescaled panels would make any grouping look structured."""

    matplotlib = pytest.importorskip("matplotlib")
    from llm_behavior_lab.analysis import figures as module

    unit, targets = _clustered(num_classes=5, per_class=6, alignment=0.8)
    greedy = np.asarray(
        [ELIGIBLE[index % 5] for index in range(targets.size)], dtype=np.int64
    )
    record = _record(unit, targets, greedy)

    held = {}
    original = module.save_figure

    def spy(figure, *args, **kwargs):
        held["figure"] = figure
        return original(figure, *args, **kwargs)

    module.save_figure = spy
    try:
        module.plot_gradient_directional_clustering(record, tmp_path, display_classes=5)
    finally:
        module.save_figure = original

    images = [
        image
        for axes in held["figure"].axes
        for image in axes.get_images()
    ]
    assert len(images) == 2
    first, second = images
    assert first.get_clim() == second.get_clim()
    assert first.get_cmap().name == second.get_cmap().name
    # Zero-centred.
    low, high = first.get_clim()
    assert low == pytest.approx(-high)


# -- figure routing ----------------------------------------------------------


def test_figures_route_into_their_category_folders(tmp_path) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    from llm_behavior_lab.analysis.figures import figure_category

    assert figure_category("figure0_sampling_adequacy") == "sanity_checks"
    assert figure_category("figure20_gradient_directional_clustering") == "main"
    assert figure_category("figure1_ranked") == "main"
    assert figure_category("figure17_initial") == "diagnostics"
    assert figure_category("figure21_correction") == "diagnostics"
    # An unknown stem stays visible rather than being dropped.
    assert figure_category("supplementary_t1_gradient") == "diagnostics"


def test_save_figure_creates_the_category_subdirectory(tmp_path) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    from llm_behavior_lab.analysis import figures as module

    unit, labels = _clustered(num_classes=4, per_class=5, alignment=0.7)
    record = _record(unit, labels, labels)

    written = module.plot_gradient_directional_clustering(
        record, tmp_path, display_classes=4
    )

    assert written[0].parent.name == "main"
    assert written[0].parent.parent == tmp_path
    assert written[0].stat().st_size > 0


# -- the all-position pooled definition --------------------------------------


def _brute_force_pooled(unit, labels):
    """Pooled within/between by explicit enumeration over every ordered pair."""

    within, between = [], []
    for a in range(len(labels)):
        for b in range(len(labels)):
            if a == b:
                continue
            value = float(unit[a] @ unit[b])
            (within if labels[a] == labels[b] else between).append(value)
    return (
        float(np.mean(within)) if within else float("nan"),
        float(np.mean(between)) if between else float("nan"),
    )


def test_the_pooled_statistic_matches_brute_force_over_all_positions() -> None:
    """Including singleton classes, which have no within-pair but do have between."""

    generator = np.random.default_rng(31)
    rows = _unit(generator.normal(size=(11, K)))
    # Two real classes plus three singletons.
    labels = np.array(
        [ELIGIBLE[0]] * 4 + [ELIGIBLE[1]] * 4 + [ELIGIBLE[2], ELIGIBLE[3], ELIGIBLE[4]],
        dtype=np.int64,
    )
    record = _record(rows, labels, labels)

    result = gradient_clustering(record, grouping="target")
    # Brute force from the values the record actually holds, not from the
    # pre-save array. Sketches are persisted as float32 on purpose -- the
    # projection's own 1/sqrt(K) error dwarfs float32 rounding -- so comparing
    # against the original float64 rows measures that downcast, roughly 2e-9 in
    # the pooled statistic, rather than the formula. Feeding both paths the same
    # inputs keeps this assertion exact instead of loosening it to hide a
    # difference that has nothing to do with what is being tested.
    stored, usable = unit_sketches(record)
    assert usable.all()
    within, between = _brute_force_pooled(stored, labels)

    assert result["population"]["within"] == pytest.approx(within, abs=1e-12)
    assert result["population"]["between"] == pytest.approx(between, abs=1e-12)
    assert result["population"]["delta"] == pytest.approx(within - between, abs=1e-12)


def test_singletons_contribute_between_pairs_but_no_within_pairs() -> None:
    """The whole reason the population is all positions rather than n >= 2."""

    generator = np.random.default_rng(32)
    rows = _unit(generator.normal(size=(6, K)))
    labels = np.array(
        [ELIGIBLE[0]] * 3 + [ELIGIBLE[1], ELIGIBLE[2], ELIGIBLE[3]], dtype=np.int64
    )
    record = _record(rows, labels, labels)

    population = gradient_clustering(record, grouping="target")["population"]

    assert population["num_positions"] == 6
    assert population["num_represented"] == 4
    assert population["num_classes"] == 1          # only one class reaches n >= 2
    # Within pairs come from the triple alone: 3 * 2 = 6 ordered pairs.
    assert population["num_within_pairs"] == 6
    # Between pairs use every position: 36 - (9 + 1 + 1 + 1) = 24.
    assert population["num_between_pairs"] == 24


def test_the_permutation_null_keeps_the_full_class_count_vector() -> None:
    """Singletons must be permuted too, or the null answers a different question."""

    generator = np.random.default_rng(33)
    rows = _unit(generator.normal(size=(9, K)))
    labels = np.array(
        [ELIGIBLE[0]] * 4 + [ELIGIBLE[1]] * 3 + [ELIGIBLE[2], ELIGIBLE[3]],
        dtype=np.int64,
    )
    record = _record(rows, labels, labels)

    result = gradient_clustering(record, grouping="target", permutations=32)

    # The null's within-pair count is fixed by the class sizes it preserves.
    assert result["null"]["permutations"] == 32
    assert result["population"]["num_within_pairs"] == 4 * 3 + 3 * 2
    assert np.isfinite(result["null"]["delta_mean"])


# -- final figure-20 wording and labels ---------------------------------------


def test_the_annotation_says_all_positions_not_n_ge_2() -> None:
    """The pooled statistic spans every position; the wording must not imply
    it is restricted to the qualifying or the displayed classes."""

    matplotlib = pytest.importorskip("matplotlib")
    from llm_behavior_lab.analysis.figures import _population_annotation

    unit, labels = _clustered(num_classes=4, per_class=5, alignment=0.6)
    result = gradient_clustering(_record(unit, labels, labels), grouping="target")

    text = _population_annotation(result)

    assert "All positions" in text
    assert f"D = {result['population']['num_positions']:,}" in text
    assert "n>=2" not in text and "n >= 2" not in text
    assert "delta" in text and "Permutation null 95%" in text


def test_the_subtitle_states_the_display_restriction_separately(tmp_path) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    from llm_behavior_lab.analysis import figures as module

    unit, labels = _clustered(num_classes=5, per_class=5, alignment=0.6)
    record = _record(unit, labels, labels)

    held = {}
    original = module.save_figure

    def spy(figure, *args, **kwargs):
        held["figure"] = figure
        return original(figure, *args, **kwargs)

    module.save_figure = spy
    try:
        module.plot_gradient_directional_clustering(record, tmp_path, display_classes=3)
    finally:
        module.save_figure = original

    blurb = " ".join(text.get_text() for text in held["figure"].texts)
    assert "heatmap:" in blurb
    assert "n >=" in blurb


@pytest.mark.parametrize(
    "text, expected",
    [
        ("▁The", "<sp>The"),
        ("▁reg", "<sp>reg"),
        ("Ġthe", "<sp>the"),
        ("<0x0A>", "<0x0A>"),
        ("\n", "<newline>"),
        ("\t", "<tab>"),
        (" ", "<space>"),
        ("", "<empty>"),
        ("abc", "abc"),
        ("半", "\\u534a"),
    ],
)
def test_token_display_formatting(text, expected) -> None:
    """SentencePiece reads semantically; anything else non-ASCII escapes."""

    matplotlib = pytest.importorskip("matplotlib")
    from llm_behavior_lab.analysis.figures import _token_axis_label

    unit, labels = _clustered(num_classes=3, per_class=4, alignment=0.5)
    record = _record(unit, labels, labels)
    vocabulary = [f"t{index}" for index in range(VOCAB)]
    vocabulary[4] = text
    record.metadata["tokens"] = vocabulary

    label = _token_axis_label(record, 4)

    assert label == expected
    assert label.isascii(), "labels must not depend on local font coverage"


def test_display_formatting_does_not_alter_token_ids() -> None:
    matplotlib = pytest.importorskip("matplotlib")

    unit, labels = _clustered(num_classes=4, per_class=5, alignment=0.6)
    record = _record(unit, labels, labels)
    record.metadata["tokens"] = ["▁x"] * VOCAB

    result = gradient_clustering(record, grouping="target", display_classes=4)

    assert set(result["display"]["classes"]).issubset(set(labels.tolist()))
    assert result["display"]["classes"].dtype.kind in "iu"
