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


def _record(sketches: np.ndarray, targets: np.ndarray, greedy: np.ndarray):
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
        gradient_position_norms=np.ones(positions),
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

    assert result["observed"]["within"] > result["observed"]["between"]
    assert result["observed"]["delta"] > 0.2
    # And the permutation reference must not reproduce it.
    assert abs(result["permuted"]["delta"]) < 0.05


def test_unstructured_gradients_show_no_clustering() -> None:
    """The null result must be reported as a null, not as weak structure."""

    unit, labels = _clustered(num_classes=5, per_class=12, alignment=0.0)
    record = _record(unit, labels, labels)

    result = gradient_clustering(record, grouping="target")

    assert abs(result["observed"]["delta"]) < 0.05
    assert abs(result["observed"]["delta"] - result["permuted"]["delta"]) < 0.08


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

    assert result["observed"]["within"] < 0.0
    assert result["observed"]["delta"] < 0.0


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
    target_delta = summary["target"]["observed"]["delta"]
    greedy_delta = summary["greedy"]["observed"]["delta"]

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

    assert np.array_equal(first["matrix"], second["matrix"], equal_nan=True)
    assert first["observed"] == second["observed"]
    assert first["permuted"] == second["permuted"]


def test_the_permutation_null_preserves_class_sizes() -> None:
    unit, labels = _clustered(num_classes=4, per_class=9, alignment=0.7)
    record = _record(unit, labels, labels)

    result = gradient_clustering(record, grouping="target")

    assert result["permuted"]["num_within_pairs"] == result["observed"]["num_within_pairs"]
    assert result["permuted"]["num_classes"] == result["observed"]["num_classes"]


def test_sketches_are_normalized_before_comparison() -> None:
    """Direction only: magnitude must not influence any similarity."""

    unit, labels = _clustered(num_classes=3, per_class=6, alignment=0.7)
    scaled = unit * np.linspace(0.1, 50.0, unit.shape[0])[:, None]

    plain = gradient_clustering(_record(unit, labels, labels), grouping="target")
    rescaled = gradient_clustering(_record(scaled, labels, labels), grouping="target")

    assert np.allclose(plain["matrix"], rescaled["matrix"], atol=1e-10, equal_nan=True)


def test_min_support_and_display_limits_are_reported_not_silent() -> None:
    unit, labels = _clustered(num_classes=6, per_class=4, alignment=0.5)
    record = _record(unit, labels, labels)

    limited = gradient_clustering(record, grouping="target", max_classes=3)

    assert limited["classes"].size == 3
    assert limited["min_support"] == 2
    assert limited["observed"]["num_classes"] == 3


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
    result = gradient_clustering(record, grouping="target", max_classes=6)
    diagonal = np.diag(result["matrix"])
    off = result["matrix"][~np.eye(6, dtype=bool)]
    assert np.nanmin(diagonal) > np.nanmax(off)


def test_the_clustering_figure_renders_when_there_is_no_structure(tmp_path):
    """A null must render as a null rather than failing or inventing a pattern."""

    matplotlib = pytest.importorskip("matplotlib")
    from llm_behavior_lab.analysis.figures import plot_gradient_directional_clustering

    unit, labels = _clustered(num_classes=5, per_class=8, alignment=0.0)
    record = _record(unit, labels, labels)

    written = plot_gradient_directional_clustering(record, tmp_path, display_classes=5)

    assert len(written) == 1
    result = gradient_clustering(record, grouping="target", max_classes=5)
    assert abs(result["observed"]["delta"]) < 0.05
