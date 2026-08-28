"""Uncertainty reaches the figures, and only where it is available.

Structural, not pixel snapshots: these read the Matplotlib artists, the plotted
data and the annotation text directly, so a bar of the wrong magnitude or a
missing centre fails rather than shifting a rendered image by a few pixels.

The three figures carry uncertainty differently, deliberately.

* **Figure 20** plots no linear statistics -- it is two heatmaps and a text
  annotation -- so its uncertainty goes into that annotation. No third panel, no
  new axes, no error-bar artists.
* **Figure 22** has a Δ trajectory, so it gets ±SE bars per point.
* **Figure 24** panel (b) has bars for the three pooled statistics, so those get
  ±SE. Its mixture panel gets **SD**, not SE: the centre there is a *ratio* of
  ensemble-averaged components, and the band says how far the per-map ratios
  scatter -- a projection diagnostic, not an uncertainty interval for the
  plug-in value.

At ``M = 1`` every figure renders its historical form. Absent spread means
unavailable, never zero, so no zero-width bar is ever drawn.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from llm_behavior_lab.analysis.records import (  # noqa: E402
    SKETCH_PROTOCOL_SCHEMA_VERSION,
    InitializationExperimentRecord,
)

figures = importlib.import_module("llm_behavior_lab.analysis.figures")
gc = importlib.import_module("llm_behavior_lab.analysis.gradient_clustering")

VOCAB = 16
ELIGIBLE = np.arange(3, VOCAB)
D = 12
K = 5
GRID = np.asarray([0.6, 1.0], dtype=np.float64)
CANONICAL_ROW = 1


def _labels():
    """Asymmetric on purpose: within, between and delta must differ."""

    targets = np.asarray([3, 3, 3, 4, 4, 4, 5, 5, 5, 3, 4, 5], dtype=np.int64)
    greedy = np.asarray([4, 4, 5, 5, 3, 3, 3, 4, 5, 4, 3, 5], dtype=np.int64)
    return targets, greedy


def _record(maps: int, *, seed: int = 5):
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
    protocol = {
        "dimension": K, "seed": 20240917, "map_count": maps,
        "schema_version": SKETCH_PROTOCOL_SCHEMA_VERSION,
        "canonical_storage": "stored" if maps == 1 else "derived",
        "canonical_relationship": "slice_of_temperature_array",
        "temperature_sketch_axes": (
            ["temperature", "position", "bucket"] if maps == 1
            else ["temperature", "position", "map", "bucket"]
        ),
        "estimator": "mean_of_per_map_inner_products_over_exact_norms",
        "accumulation_dtype": "float64", "storage_dtype": "float32",
        "recomputed_per_map": False,
    }
    return InitializationExperimentRecord.build(
        corpus_counts=corpus,
        selected_target_counts=np.bincount(targets, minlength=VOCAB),
        greedy_counts=np.bincount(greedy, minlength=VOCAB)[None, :],
        nucleus_counts=np.bincount(greedy, minlength=VOCAB)[None, None, :],
        mean_predicted_probabilities=np.full((1, VOCAB), 1.0 / VOCAB),
        model_seeds=[1000],
        eligible_token_ids=ELIGIBLE,
        metadata={
            "num_positions": D, "record_version": 12,
            "analysis": {"gradient_analysis": {
                "enabled": True, "initialization_index": 0,
                "gradient_sketch": protocol,
            }},
        },
        gradient_position_indices=np.arange(D),
        gradient_position_target_ids=targets,
        gradient_position_greedy_ids=greedy,
        gradient_position_norms=np.full(D, 2.0),
        gradient_temperatures=GRID,
        gradient_temperature_position_norms=np.full((len(GRID), D), 2.0),
        **extra,
    )



def _capture(monkeypatch):
    """Grab the figure before ``save_figure`` writes and closes it."""

    captured = {}
    original = figures.save_figure

    def spy(figure, directory, stem, *args, **kwargs):
        captured["figure"] = figure
        captured["axes"] = list(figure.get_axes())
        captured["stem"] = stem
        return original(figure, directory, stem, *args, **kwargs)

    monkeypatch.setattr(figures, "save_figure", spy)
    return captured


# -- Figure 20: annotation only ------------------------------------------------

#: The single-map annotation exactly as committed, for the frozen ``_record(1)``
#: fixture above. Pinned as a literal rather than rebuilt from the result,
#: because "the historical text" is a claim about *bytes*: a format change and a
#: numerics change are both regressions here, and rebuilding the expected string
#: from the same values would only ever catch the first.
#:
#: If this fails, decide which moved before touching it. The wording is the
#: reference every existing figure 20 was produced against; the numbers are the
#: clustering result on a seeded fixture.
HISTORICAL_SINGLE_MAP_ANNOTATION = (
    "All positions (D = 12)\n"
    "delta = +0.19590\n"
    "within +0.09911 | between -0.09679\n"
    "Permutation null 95%: [-0.12923, +0.11731]  (M = 8)"
)


def _multi_map_spread(record, *, permutations=8):
    """The spread block figure 20 is handed at ``M > 1``."""

    per_map = gc.gradient_clustering_per_map(record, permutations=permutations)
    return {
        name: {**per_map[name], "map_count": per_map["map_count"]}
        for name in ("within", "between", "delta")
    }


def test_the_single_map_annotation_is_the_historical_text() -> None:
    """Byte for byte. No 'unavailable', no NaN, no zero spread.

    The single-map branch deliberately keeps ``(M = n)`` for the permutation
    count. Disambiguating it was a change to the *multi-map* branch only,
    because that is the only branch where a second meaning of ``M`` can appear
    beside it -- and rewording this one would invalidate the comparison every
    already-produced figure rests on.
    """

    result = gc.gradient_clustering(_record(1), permutations=8)
    text = figures._population_annotation(result)

    assert text == HISTORICAL_SINGLE_MAP_ANNOTATION
    assert "+/-" not in text
    assert "df=" not in text
    assert "maps M=" not in text
    assert "permutations = " not in text
    assert text.startswith("All positions (D = ")
    assert text.count("\n") == 3


def test_the_multi_map_annotation_carries_centres_and_standard_errors() -> None:
    record = _record(4)
    result = gc.gradient_clustering(record, permutations=8)
    spread = _multi_map_spread(record)

    text = figures._population_annotation(result, spread)

    # Centres are the ensemble values, unchanged.
    assert f"delta = {result['population']['delta']:+.5f}" in text
    assert f"within {result['population']['within']:+.5f}" in text
    assert f"between {result['population']['between']:+.5f}" in text
    # Each carries its own standard error.
    for name in ("delta", "within", "between"):
        assert f"+/- {spread[name]['standard_error']:.5f}" in text
    assert "maps M=4, df=3" in text
    assert "standard error across maps" in text


def test_the_multi_map_annotation_names_the_permutation_count_explicitly() -> None:
    """The two counts in this block must not both be spelled ``M``.

    At ``M > 1`` the annotation carries a permutation count and a CountSketch
    map count two lines apart. Writing both as ``M`` put two meanings on one
    symbol, against the project's own convention -- ``D`` positions, ``N_T``
    loss temperatures, ``K`` sketch width, ``M`` maps -- and the reader has no
    way to tell which is which from the text.
    """

    record = _record(4)
    result = gc.gradient_clustering(record, permutations=8)
    text = figures._population_annotation(result, _multi_map_spread(record))

    # The permutation count, named.
    assert "(permutations = 8)" in text
    # The map count and its degrees of freedom, still separate and still there.
    assert "maps M=4" in text
    assert "df=3" in text
    # And the ambiguous label is gone from this branch.
    assert "(M = 8)" not in text
    assert "(M = " not in text


def test_the_relabel_changed_no_number_in_the_multi_map_annotation() -> None:
    """Only the label moved. Every quantity beside it is what it was.

    Asserted against independently computed values rather than a copy of the
    string, so a relabel that also perturbed rounding, ordering or a statistic
    would fail here rather than pass on the label alone.
    """

    record = _record(4)
    result = gc.gradient_clustering(record, permutations=8)
    per_map = gc.gradient_clustering_per_map(record, permutations=8)
    spread = _multi_map_spread(record)
    text = figures._population_annotation(result, spread)

    population, null = result["population"], result["null"]

    # Centres, at the historical .5f precision.
    assert f"All positions (D = {population['num_positions']:,})" in text
    assert f"delta = {population['delta']:+.5f}" in text
    assert f"within {population['within']:+.5f}" in text
    assert f"between {population['between']:+.5f}" in text
    # Standard errors, each SD/sqrt(M) of its own statistic.
    for name in ("delta", "within", "between"):
        assert spread[name]["standard_error"] == pytest.approx(
            spread[name]["sample_sd"] / np.sqrt(4.0)
        )
        assert f"+/- {spread[name]['standard_error']:.5f}" in text
    # Permutation null bounds and count.
    assert (
        f"Permutation null 95%: [{null['delta_low']:+.5f}, {null['delta_high']:+.5f}]"
        in text
    )
    assert null["permutations"] == 8
    # Map count and degrees of freedom.
    assert per_map["map_count"] == 4
    assert spread["delta"]["degrees_of_freedom"] == 3
    # Line count is unchanged: four lines at M = 1, five at M > 1.
    assert text.count("\n") == 4


def test_only_figure_twenty_uses_the_population_annotation() -> None:
    """So the relabel cannot have leaked into another figure.

    Figures 22, 23 and 24 build their own text. A single production call site
    is what makes "this changed figure 20 and nothing else" checkable rather
    than merely intended.
    """

    import inspect

    source = inspect.getsource(figures)
    calls = [
        line.strip()
        for line in source.splitlines()
        if "_population_annotation(" in line and not line.strip().startswith("def ")
    ]

    assert len(calls) == 1, calls


def test_figure_twenty_gains_no_axes_or_error_bars(tmp_path, monkeypatch) -> None:
    """Option (b): the annotation carries it, the layout does not move."""

    for maps in (1, 4):
        captured = _capture(monkeypatch)
        figures.plot_gradient_directional_clustering(_record(maps), tmp_path)
        axes = captured["axes"]
        # Two heatmap panels plus the shared colorbar axis -- as before.
        assert len(axes) == 3, maps
        for axis in axes:
            assert not axis.containers, "figure 20 must draw no error-bar artists"


# -- Figure 24: SE on the pooled panel, SD on the mixture panel -----------------


def test_figure_twenty_four_draws_standard_error_bars_above_one_map(tmp_path, monkeypatch) -> None:
    import matplotlib.pyplot as plt

    from llm_behavior_lab.analysis.gradient_cross_partition import (
        cross_partition_per_map,
    )

    record = _record(4)
    rows, usable = gc.unit_sketches_per_map(record)
    targets, greedy = _labels()
    expected = cross_partition_per_map(rows[usable], targets[usable], greedy[usable])

    captured = _capture(monkeypatch)
    figures.plot_cross_partition_geometry(record, tmp_path)
    axes = captured["axes"]
    bars = [
        container
        for axis in axes
        for container in axis.containers
        if hasattr(container, "has_yerr") and container.has_yerr
    ]
    assert bars, "expected at least one error-bar container"

    se = [
        expected["c_same"]["standard_error"],
        expected["c_different"]["standard_error"],
        expected["delta_cross"]["standard_error"],
    ]
    assert all(np.isfinite(se))
    assert expected["delta_cross"]["degrees_of_freedom"] == 3
    plt.close("all")


def test_figure_twenty_four_has_no_uncertainty_bars_at_one_map(tmp_path, monkeypatch) -> None:
    import matplotlib.pyplot as plt

    captured = _capture(monkeypatch)
    figures.plot_cross_partition_geometry(_record(1), tmp_path)
    axes = captured["axes"]
    labels = [
        text.get_text()
        for axis in axes
        for text in ([axis.get_legend()] if axis.get_legend() else [])
        for text in axis.get_legend().get_texts()
    ]
    assert not any("SE across" in label for label in labels)
    assert not any("per-map ratio spread" in label for label in labels)
    plt.close("all")


def test_the_mixture_panel_uses_sd_and_never_se(tmp_path, monkeypatch) -> None:
    """Numerically distinguishable, so the wrong one cannot pass by accident."""

    import matplotlib.pyplot as plt

    from llm_behavior_lab.analysis.gradient_cross_partition import (
        cross_partition_per_map,
    )

    record = _record(4)
    rows, usable = gc.unit_sketches_per_map(record)
    targets, greedy = _labels()
    summary = cross_partition_per_map(
        rows[usable], targets[usable], greedy[usable]
    )["mixture_median_similarity_map_summary"]
    sd = summary["sample_sd"]
    se = summary["standard_error"]
    # SD and SE differ by sqrt(M) = 2, so the assertion below is decisive.
    assert sd == pytest.approx(se * 2.0)

    captured = _capture(monkeypatch)
    figures.plot_cross_partition_geometry(record, tmp_path)
    axes = captured["axes"]
    heights = []
    for axis in axes:
        for patch in axis.patches:
            if type(patch).__name__ == "Rectangle":
                heights.append(float(patch.get_height()))
            elif type(patch).__name__ == "Polygon":
                ys = patch.get_xy()[:, 1]
                if ys.size:
                    heights.append(float(ys.max() - ys.min()))
    assert any(
        height == pytest.approx(2.0 * sd, rel=1e-6) for height in heights
    ), f"expected a band of full height 2*SD={2 * sd}; got {heights}"
    assert not any(
        height == pytest.approx(2.0 * se, rel=1e-6) for height in heights
    ), "the mixture band must be SD, not SE"

    labels = [
        text.get_text()
        for axis in axes
        if axis.get_legend()
        for text in axis.get_legend().get_texts()
    ]
    assert any("per-map ratio spread" in label for label in labels)
    assert any("diagnostic" in label for label in labels)
    plt.close("all")


def test_figure_twenty_three_is_untouched() -> None:
    """The fidelity figure and its call path are not part of this stage."""

    import inspect

    source = inspect.getsource(figures.plot_countsketch_fidelity)

    assert "sketch_map_count" not in source
    assert "standard_error" not in source
    assert "per_map" not in source


# -- Figure 22: SE bars on the trajectory ------------------------------------------


def _nucleus_result(maps, temperatures=(0.6, 1.0)):
    """A loader-shaped sweep result, with or without the additive spread."""

    count = len(temperatures)
    result = {
        "sampling_temperatures": np.asarray(temperatures),
        "loss_temperatures": np.full(count, 1.0),
        "temperatures": tuple(float(v) for v in temperatures),
        "pair_metadata": "explicit",
        "pairing": "T_g pinned",
        "by_temperature": [
            {
                "temperature": float(t), "sampling_temperature": float(t),
                "loss_temperature": 1.0, "grouping": f"nucleus T_s={t:g}",
                "population": {"delta": 0.10 + 0.01 * i,
                               "within": 0.30, "between": 0.20},
                "null": {"delta_mean": 0.0, "delta_low": -0.05, "delta_high": 0.05},
                "support": {
                    "num_represented": 3, "num_qualifying": 3, "num_singletons": 0,
                    "singleton_fraction": 0.0, "positions_in_qualifying": D,
                    "fraction_positions_in_qualifying": 1.0, "largest_class": 4,
                    "median_class": 4, "num_within_pairs": 12,
                    "num_between_pairs": 96,
                },
                "display": {
                    "classes": np.asarray([3, 4, 5]),
                    "matrix": np.zeros((3, 3)), "counts": np.asarray([4, 4, 4]),
                },
                "num_positions": D, "num_positions_excluded": 0,
                "sketch_dimension": K,
            }
            for i, t in enumerate(temperatures)
        ],
        "references": {
            g: {"population": {"delta": 0.12},
                "null": {"delta_low": -0.04, "delta_high": 0.04}}
            for g in ("target", "greedy")
        },
        "references_by_loss_temperature": {},
        "nucleus_labels": np.zeros((count, D), dtype=np.int64),
        "min_support": 2, "permutations": 8, "permutation_seed": 20240918,
        "map_count": maps,
        "degrees_of_freedom": max(maps - 1, 0),
        "uncertainty_available": maps > 1,
    }
    if maps > 1:
        result["uncertainty"] = {
            "delta_se": np.asarray([0.011, 0.022][:count]),
        }
    else:
        result["uncertainty"] = {"delta_se": None}
    return result


def test_figure_twenty_two_draws_one_se_bar_per_point(tmp_path, monkeypatch) -> None:
    """Magnitudes checked against the stored values, not merely presence."""

    captured = _capture(monkeypatch)
    expected = np.asarray([0.011, 0.022])
    figures.plot_nucleus_temperature_clustering(
        _nucleus_result(4), tmp_path
    )
    trajectory = captured["axes"][0]
    bars = [c for c in trajectory.containers if getattr(c, "has_yerr", False)]

    assert len(bars) == 1
    segments = bars[0].lines[2][0].get_segments()
    assert len(segments) == len(expected)
    for segment, value in zip(segments, expected):
        assert float(segment[:, 1].ptp()) == pytest.approx(2.0 * value, rel=1e-6)


def test_figure_twenty_two_draws_no_bars_for_historical_artifacts(
    tmp_path, monkeypatch
) -> None:
    """Absent spread means unavailable, so nothing is drawn -- not a zero bar."""

    captured = _capture(monkeypatch)
    figures.plot_nucleus_temperature_clustering(_nucleus_result(1), tmp_path)
    trajectory = captured["axes"][0]

    assert not [c for c in trajectory.containers if getattr(c, "has_yerr", False)]
