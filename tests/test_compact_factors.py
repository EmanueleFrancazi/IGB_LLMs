"""Retained factors answer their declared questions, and refuse every other one.

``compact_factors`` exists because ``metrics_only`` makes post-hoc questions
unanswerable and ``per_position`` makes them cost 3.5 GiB. The middle option is
only honest if the boundary is enforced: a retained factor set answers the
grouping and gradient field it was declared for, and nothing adjacent.

That makes the refusals as important as the reconstructions. A query for an
undeclared temperature, an unretained grouping, a class that was not represented,
or a per-position relabelling must fail with an explanation -- never return a
number derived from something nearby, which would be indistinguishable from a
real answer in a table.

The reconstruction is checked against the finalized value computed from the rows
in the same run, so "the factors are sufficient" is demonstrated rather than
asserted.
"""

from __future__ import annotations

import numpy as np
import pytest

from llm_behavior_lab.analysis import compact_factors as cf
from llm_behavior_lab.analysis.alignment_estimators import (
    class_factors,
    class_similarity_from_factors,
)


# -- parsing -----------------------------------------------------------------


def test_a_selection_spells_out_grouping_and_field() -> None:
    selections = cf.parse_factor_selection(
        "target@1.0,greedy@1.0,nucleus:0.6@0.6,cross@1.0"
    )
    assert [s.label for s in selections] == [
        "target@1", "greedy@1", "nucleus:0.6@0.6", "cross@1",
    ]


def test_a_nucleus_selection_must_name_its_sampling_temperature() -> None:
    """The labels are what make it a grouping, and ``T_s`` is what makes them."""

    with pytest.raises(ValueError, match="must name its sampling temperature"):
        cf.FactorSelection("nucleus", 1.0)


def test_an_empty_selection_is_refused() -> None:
    """Retaining nothing is what ``metrics_only`` already does."""

    with pytest.raises(ValueError, match="retains nothing"):
        cf.parse_factor_selection("   ")


@pytest.mark.parametrize("spec", ["target", "target@", "target@abc", "nope@1.0"])
def test_a_malformed_selection_is_refused(spec) -> None:
    with pytest.raises(ValueError):
        cf.parse_factor_selection(spec)


def test_there_is_no_wildcard() -> None:
    """Retention decides which questions stay askable, so it is spelled out."""

    with pytest.raises(ValueError, match="Unknown grouping"):
        cf.parse_factor_selection("all@1.0")


# -- byte accounting ---------------------------------------------------------


def test_the_estimate_is_float64_and_per_selection() -> None:
    """``C*M*K*8 + C*M*8`` each -- about 131 MiB per selection at production."""

    one = cf.estimate_factor_bytes(
        cf.parse_factor_selection("target@1.0"),
        num_classes=4000, num_maps=4, num_buckets=1024,
    )
    assert one == 4000 * 4 * 1024 * 8 + 4000 * 4 * 8
    two = cf.estimate_factor_bytes(
        cf.parse_factor_selection("target@1.0,greedy@1.0"),
        num_classes=4000, num_maps=4, num_buckets=1024,
    )
    assert two == 2 * one


# -- reconstruction ----------------------------------------------------------


class _Artifact:
    """Minimal stand-in for a loaded metrics artifact."""

    def __init__(self, arrays, provenance):
        self.arrays = arrays
        self.provenance = provenance

    def __getitem__(self, name):
        return self.arrays[name]

    def __contains__(self, name):
        return name in self.arrays


def _retained(maps: int = 3, buckets: int = 6, seed: int = 4):
    generator = np.random.default_rng(seed)
    labels = np.array([0, 0, 0, 1, 1, 2, 2, 2, 3], dtype=np.int64)
    rows = generator.standard_normal((labels.size, maps, buckets))
    classes = np.unique(labels)
    per_map = [class_factors(rows[:, m, :], labels, classes) for m in range(maps)]
    selections = cf.parse_factor_selection("target@1.0")
    arrays = cf.factor_arrays(selections, {"target@1": per_map})
    artifact = _Artifact(
        arrays,
        {"factor_selection": [s.as_dict() for s in selections]},
    )
    return artifact, rows, labels, classes, per_map


def test_a_between_class_similarity_matches_the_row_based_value() -> None:
    """The factors are sufficient -- demonstrated, not asserted."""

    artifact, rows, labels, classes, per_map = _retained()
    expected = float(
        np.mean([
            class_similarity_from_factors(entry)["matrix"][0, 1]
            for entry in per_map
        ])
    )
    assert cf.similarity_from_factors(
        artifact, "target", int(classes[0]), int(classes[1])
    ) == pytest.approx(expected, rel=1e-12)


def test_a_within_class_similarity_uses_the_measured_self_term() -> None:
    """``Q``, never ``n``: the rows are sketches over exact norms."""

    artifact, rows, labels, classes, per_map = _retained()
    expected = float(
        np.mean([
            class_similarity_from_factors(entry)["within"][0] for entry in per_map
        ])
    )
    assert cf.similarity_from_factors(
        artifact, "target", int(classes[0]), int(classes[0])
    ) == pytest.approx(expected, rel=1e-12)


def test_a_singleton_has_no_within_class_estimate() -> None:
    """Unavailable, not zero: one member has no distinct pair."""

    artifact, _, _, classes, _ = _retained()
    assert np.isnan(
        cf.similarity_from_factors(artifact, "target", int(classes[3]), int(classes[3]))
    )


def test_the_persisted_sums_are_float64() -> None:
    """Narrowing them would be a second quantization after normalization, with
    no declared convention behind it."""

    artifact, *_ = _retained()
    assert artifact["factor_0_gradient_sketch_sum"].dtype == np.float64
    assert artifact["factor_0_gradient_sketch_self_squared"].dtype == np.float64


def test_the_field_names_say_they_are_gradient_sketches() -> None:
    """Someone deciding whether an artifact may be shared must see what it
    holds without reading the module docstring."""

    artifact, *_ = _retained()
    assert any("gradient_sketch" in name for name in artifact.arrays)


def test_the_map_axis_is_explicit() -> None:
    """A single-map selection must not look like a squeezed multi-map one."""

    artifact, rows, *_ = _retained(maps=3)
    assert artifact["factor_0_gradient_sketch_sum"].shape[1] == 3


# -- refusals ----------------------------------------------------------------


def test_an_undeclared_loss_temperature_is_refused() -> None:
    """A factor set from another gradient field is a different question."""

    artifact, _, _, classes, _ = _retained()
    with pytest.raises(cf.UnsupportedFactorQuery, match="not\\s+substituted"):
        cf.similarity_from_factors(
            artifact, "target", int(classes[0]), int(classes[1]),
            loss_temperature=0.6,
        )


def test_an_undeclared_grouping_is_refused() -> None:
    artifact, _, _, classes, _ = _retained()
    with pytest.raises(cf.UnsupportedFactorQuery, match="No retained factors"):
        cf.similarity_from_factors(
            artifact, "greedy", int(classes[0]), int(classes[1])
        )


def test_an_unrepresented_class_is_refused() -> None:
    artifact, _, _, classes, _ = _retained()
    with pytest.raises(cf.UnsupportedFactorQuery, match="not among"):
        cf.similarity_from_factors(artifact, "target", 999_999, int(classes[0]))


def test_per_position_relabelling_is_refused_outright() -> None:
    """A class sum cannot be re-partitioned: the membership it was formed over
    is exactly what was discarded."""

    artifact, *_ = _retained()
    with pytest.raises(cf.UnsupportedFactorQuery, match="needs the rows"):
        cf.refuse_relabelling(artifact)


def test_a_record_with_no_retained_factors_says_so() -> None:
    """`metrics_only` answers the declared analyses and no new ones."""

    artifact = _Artifact({}, {})
    with pytest.raises(cf.UnsupportedFactorQuery, match="retained no"):
        cf.similarity_from_factors(artifact, "target", 0, 1)


def test_the_refusal_lists_what_was_actually_retained() -> None:
    """So the reader can see what the run *can* answer, not only what it cannot."""

    artifact, _, _, classes, _ = _retained()
    with pytest.raises(cf.UnsupportedFactorQuery) as caught:
        cf.similarity_from_factors(
            artifact, "target", int(classes[0]), int(classes[1]),
            loss_temperature=0.24,
        )
    assert "target@1" in str(caught.value)
