"""Additive uncertainty in the derived artifacts, and safe absence of it.

The point estimates in an artifact are complete at every map count. What was
missing is the spread *behind* them: without it, map uncertainty is computed in
memory and then discarded at exactly the boundary where the scientific output
gets written.

Two rules govern how absence is represented, and they differ by medium:

* a Python summary says ``None`` and JSON says ``null`` -- unambiguous, and a
  reader must branch on it;
* an **NPZ** says float64 ``NaN``, never ``None``. A ``None`` inside an NPZ makes
  that entry object-dtype, and the artifact loaders call ``np.load`` with
  NumPy's default ``allow_pickle=False``, so such an archive would not load at
  all. That is a hard failure, not a style preference, and it is asserted here.

``uncertainty_available`` is the availability signal. ``NaN`` alone is not: at
``M > 1`` a non-finite value means a genuinely invalid computation, and
collapsing the two would hide it behind the legitimate single-map case.
"""

from __future__ import annotations

import numpy as np
import pytest

from llm_behavior_lab.analysis.nucleus_clustering_artifact import (
    summary_arrays,
    uncertainty_arrays,
    uncertainty_defaults,
)
from llm_behavior_lab.analysis.sketch_estimator import ensemble_summary


# -- the discriminators -----------------------------------------------------------


def test_a_single_map_reports_unavailable_rather_than_zero() -> None:
    fields = uncertainty_arrays(1)

    assert int(fields["map_count"]) == 1
    assert int(fields["degrees_of_freedom"]) == 0
    assert bool(fields["uncertainty_available"]) is False


@pytest.mark.parametrize("maps", [2, 4, 8])
def test_several_maps_report_available_with_m_minus_one_degrees(maps) -> None:
    fields = uncertainty_arrays(maps)

    assert int(fields["map_count"]) == maps
    assert int(fields["degrees_of_freedom"]) == maps - 1
    assert bool(fields["uncertainty_available"]) is True


def test_historical_artifacts_default_to_single_map() -> None:
    """An artifact written before replicas came from exactly one map."""

    assert uncertainty_defaults() == {
        "map_count": 1,
        "degrees_of_freedom": 0,
        "uncertainty_available": False,
    }


def test_the_discriminator_arrays_are_never_object_dtype() -> None:
    for maps in (1, 4):
        for value in uncertainty_arrays(maps).values():
            assert value.dtype != object


# -- NaN, not None, inside an NPZ ---------------------------------------------------


def test_unavailable_spread_becomes_nan_in_the_archive() -> None:
    single = summary_arrays("delta", ensemble_summary(np.asarray([0.25])))

    assert single["delta_sd"].dtype == np.float64
    assert single["delta_se"].dtype == np.float64
    assert np.isnan(single["delta_sd"])
    assert np.isnan(single["delta_se"])


def test_available_spread_is_finite_and_matches_the_formulas() -> None:
    """``ddof = 1`` and ``sd / sqrt(M)``, on a fixed example."""

    values = np.asarray([1.0, 2.0, 3.0, 4.0])
    fields = summary_arrays("delta", ensemble_summary(values))

    assert float(fields["delta_sd"]) == pytest.approx(values.std(ddof=1))
    assert float(fields["delta_se"]) == pytest.approx(values.std(ddof=1) / 2.0)
    assert np.isfinite(fields["delta_sd"])


def test_an_npz_of_these_fields_loads_without_pickle(tmp_path) -> None:
    """The property that makes NaN mandatory rather than tidy.

    A ``None`` here would be object-dtype and ``np.load`` would refuse the whole
    archive -- which is how the artifact loaders read it.
    """

    payload = {}
    payload.update(uncertainty_arrays(1))
    payload.update(summary_arrays("delta", ensemble_summary(np.asarray([0.25]))))
    payload.update(
        summary_arrays("within", ensemble_summary(np.asarray([1.0, 2.0, 3.0])))
    )
    path = tmp_path / "artifact.npz"
    np.savez_compressed(path, **payload)

    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            assert data[key].dtype != object, key
        assert np.isnan(data["delta_sd"])
        assert np.isfinite(data["within_sd"])
        assert bool(data["uncertainty_available"]) is False


def test_a_none_in_an_npz_would_break_the_loader(tmp_path) -> None:
    """The negative control, so the rule is not merely asserted.

    This is why ``summary_arrays`` converts rather than storing the summary's
    ``None`` directly.
    """

    path = tmp_path / "broken.npz"
    np.savez_compressed(path, delta_sd=np.asarray(None))

    with pytest.raises(ValueError, match="[Oo]bject arrays"):
        with np.load(path, allow_pickle=False) as data:
            data["delta_sd"]


def test_nan_is_not_the_availability_signal() -> None:
    """At M > 1 a non-finite value is a broken computation, not absence.

    Both cases put NaN in the array; only ``uncertainty_available`` tells them
    apart, which is why a reader must consult it rather than test the value.
    """

    unavailable = ensemble_summary(np.asarray([0.25]))
    invalid = ensemble_summary(np.asarray([1.0, np.nan, 3.0]))

    assert unavailable["uncertainty_available"] is False
    assert invalid["uncertainty_available"] is True
    assert np.isnan(summary_arrays("d", unavailable)["d_sd"])
    assert np.isnan(summary_arrays("d", invalid)["d_sd"])
    assert invalid["degrees_of_freedom"] == 2


# -- cross-partition: independent oracles -------------------------------------------


def _cross_fixture():
    """An asymmetric fixture: the three statistics must be distinguishable."""

    import importlib

    gcm = importlib.import_module("llm_behavior_lab.analysis.gradient_clustering")
    # Bare module name: pytest puts the test directory on sys.path.
    shared = importlib.import_module("test_figure_uncertainty_artists")
    _labels, _record = shared._labels, shared._record

    record = _record(4)
    rows, usable = gcm.unit_sketches_per_map(record)
    targets, greedy = _labels()
    return record, rows[usable], targets[usable], greedy[usable]


def test_the_three_cross_statistics_are_distinguishable() -> None:
    """A symmetric fixture would let a copied field pass; this one cannot."""

    from llm_behavior_lab.analysis.gradient_cross_partition import (
        cross_partition_per_map,
    )

    _, rows, targets, greedy = _cross_fixture()
    result = cross_partition_per_map(rows, targets, greedy)

    same = result["c_same_per_map"]
    different = result["c_different_per_map"]
    delta = result["delta_cross_per_map"]

    assert not np.allclose(same, different)
    assert not np.allclose(same, delta)
    assert not np.allclose(different, delta)


def test_delta_cross_per_map_is_same_minus_different() -> None:
    from llm_behavior_lab.analysis.gradient_cross_partition import (
        cross_partition_per_map,
    )

    _, rows, targets, greedy = _cross_fixture()
    result = cross_partition_per_map(rows, targets, greedy)

    assert np.allclose(
        result["delta_cross_per_map"],
        result["c_same_per_map"] - result["c_different_per_map"],
        rtol=0, atol=1e-15,
    )


def test_cross_sd_and_se_follow_the_formulas() -> None:
    from llm_behavior_lab.analysis.gradient_cross_partition import (
        cross_partition_per_map,
    )

    _, rows, targets, greedy = _cross_fixture()
    result = cross_partition_per_map(rows, targets, greedy)

    for name in ("c_same", "c_different", "delta_cross"):
        values = result[f"{name}_per_map"]
        assert result[name]["sample_sd"] == pytest.approx(values.std(ddof=1))
        assert result[name]["standard_error"] == pytest.approx(
            values.std(ddof=1) / np.sqrt(values.size)
        )
        assert result[name]["degrees_of_freedom"] == values.size - 1


def test_the_mixture_plug_in_differs_from_the_mean_of_per_map_ratios() -> None:
    """A ratio of averages is not the average of ratios, and both are kept."""

    import importlib

    from llm_behavior_lab.analysis.gradient_cross_partition import (
        cross_partition_per_map,
        mixture_reconstruction,
    )
    from llm_behavior_lab.analysis.sketch_estimator import ensemble_embedding

    _, rows, targets, greedy = _cross_fixture()
    plug_in = mixture_reconstruction(
        ensemble_embedding(rows), targets, greedy
    )["median_similarity"]
    summary = cross_partition_per_map(rows, targets, greedy)[
        "mixture_median_similarity_map_summary"
    ]

    assert np.isfinite(plug_in)
    assert plug_in != summary["map_mean"]
    # SD and SE differ by sqrt(M), so one cannot be mistaken for the other.
    assert summary["sample_sd"] == pytest.approx(
        summary["standard_error"] * np.sqrt(4.0)
    )
