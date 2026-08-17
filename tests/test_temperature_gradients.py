"""Tests for temperature-conditioned gradients and the six-panel figure 10.

Temperature enters the **loss** here, not a rescaling of a finished result:

    ell_T(d) = -log softmax(z_d / T)[y_d]
    d ell_T / d z_i = (p_T(i) - 1[i = y]) / T

so the gradient genuinely changes with ``T`` through both the explicit ``1/T``
and the sharpening of ``p_T``. What does *not* change is the greedy decision,
because softmax is strictly increasing. That asymmetry is the experiment: the
guessing bias is pinned while the learning signal varies, so anything that moves
across panels moves for gradient reasons alone.

The limiting behaviour is checked on analytic logits where the answer is known
in closed form, and deliberately not on any expectation about the token-level
correlation, which is an empirical result rather than a property to assert.
"""

from __future__ import annotations

import numpy as np
import pytest

from llm_behavior_lab.analysis import (
    InitializationExperimentRecord,
    load_record,
    temperature_gradient_summary,
    temperature_gradient_table,
)

matplotlib = pytest.importorskip("matplotlib")

from llm_behavior_lab.analysis.figures import (  # noqa: E402
    generate_all_figures,
    plot_temperature_gradient_vs_guess_bias,
)

VOCAB_SIZE = 12
ELIGIBLE = np.arange(2, VOCAB_SIZE)
NUM_POSITIONS = 24
TEMPERATURES = (0.12, 0.24, 0.36, 0.48, 0.60, 1.00, 1.20)
PANELS = tuple(value for value in TEMPERATURES if value != 1.00)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def _logit_gradient_norm(logits: np.ndarray, target: int, temperature: float) -> float:
    """``|| d ell_T / dz ||`` in closed form: ``(p_T - onehot) / T``."""

    residual = _softmax(logits / temperature)
    residual[target] -= 1.0
    return float(np.linalg.norm(residual / temperature))


def _record(*, with_temperatures: bool = True, seed: int = 4):
    rng = np.random.default_rng(seed)
    targets = rng.choice(ELIGIBLE, size=NUM_POSITIONS)
    greedy = rng.choice(ELIGIBLE[:3], size=NUM_POSITIONS)
    corpus = np.zeros(VOCAB_SIZE, dtype=np.int64)
    corpus[ELIGIBLE] = rng.integers(4, 50, size=ELIGIBLE.size)
    corpus[targets] = np.maximum(corpus[targets], 1)

    # Norms that genuinely differ per temperature, so a table that ignored the
    # temperature index would be caught.
    base = rng.lognormal(0.0, 0.4, size=NUM_POSITIONS)
    per_temperature = np.stack([base / value for value in TEMPERATURES])
    canonical = TEMPERATURES.index(1.00)

    arrays = {
        "gradient_position_indices": np.arange(NUM_POSITIONS),
        "gradient_position_target_ids": targets,
        "gradient_position_greedy_ids": greedy,
        "gradient_position_norms": per_temperature[canonical],
    }
    metadata = {
        "tokens": [f"t{index}" for index in range(VOCAB_SIZE)],
        "analysis": {
            "num_positions": NUM_POSITIONS,
            "gradient_analysis": {
                "enabled": True,
                "softmax_support": "eligible",
                "initialization_index": 0,
                "model_seed": 1000,
                "input_condition": "real",
                "covers_all_positions": True,
                "parameter_count": 8_585_856,
            },
        },
    }
    if with_temperatures:
        arrays["gradient_temperatures"] = np.asarray(TEMPERATURES)
        arrays["gradient_temperature_position_norms"] = per_temperature
        metadata["analysis"]["gradient_analysis"]["temperatures"] = list(TEMPERATURES)

    return InitializationExperimentRecord.build(
        corpus_counts=corpus,
        selected_target_counts=np.bincount(targets, minlength=VOCAB_SIZE),
        greedy_counts=np.bincount(greedy, minlength=VOCAB_SIZE)[None, :],
        nucleus_counts=np.bincount(greedy, minlength=VOCAB_SIZE)[None, None, :],
        mean_predicted_probabilities=np.full((1, VOCAB_SIZE), 1.0 / VOCAB_SIZE),
        model_seeds=[1000],
        eligible_token_ids=ELIGIBLE,
        metadata=metadata,
        **arrays,
    )


# -- analytic limiting behaviour ---------------------------------------------


def test_a_correct_greedy_prediction_loses_its_gradient_as_temperature_falls() -> None:
    """When the target already wins, low T drives the gradient to zero.

    The explicit 1/T grows without bound, but the probability error decays
    exponentially in 1/T and wins. Asserting the limit rather than a rate keeps
    this a statement about the mathematics, not about any model.
    """

    logits = np.array([3.0, 1.0, 0.5, 0.0])
    target = 0  # the argmax: greedy is correct here

    norms = [_logit_gradient_norm(logits, target, t) for t in (1.0, 0.5, 0.25, 0.12, 0.05)]

    assert all(later < earlier for earlier, later in zip(norms, norms[1:]))
    assert norms[-1] < 1e-6


def test_an_incorrect_greedy_prediction_grows_like_one_over_temperature() -> None:
    """When the target loses, low T makes the gradient grow as O(1/T).

    The probability error saturates near one, so only the explicit 1/T remains,
    and the norm scales inversely with temperature.
    """

    logits = np.array([3.0, 1.0, 0.5, 0.0])
    target = 3  # far from the argmax: greedy is wrong here

    for temperature in (0.12, 0.06, 0.03):
        norm = _logit_gradient_norm(logits, target, temperature)
        expected = np.sqrt(2.0) / temperature  # |p - onehot| -> sqrt(2)
        assert norm == pytest.approx(expected, rel=1e-3), temperature

    halved = _logit_gradient_norm(logits, target, 0.06)
    full = _logit_gradient_norm(logits, target, 0.12)
    assert halved / full == pytest.approx(2.0, rel=1e-3)


def test_greedy_identity_is_untouched_by_the_loss_temperature() -> None:
    """The decision is invariant even where the gradient changes by orders."""

    logits = np.array([3.0, 1.0, 0.5, 0.0])

    for temperature in (0.01, 0.12, 1.0, 1.2, 100.0):
        assert _softmax(logits / temperature).argmax() == logits.argmax()


# -- what must stay fixed across temperature ---------------------------------


def test_positions_targets_and_greedy_ids_are_shared_by_every_temperature() -> None:
    record = _record()
    reference = temperature_gradient_table(record, 1.00)

    for temperature in TEMPERATURES:
        table = temperature_gradient_table(record, temperature)
        assert np.array_equal(
            table["target_occurrence_count"], reference["target_occurrence_count"]
        )
        assert np.array_equal(table["greedy_guess_count"], reference["greedy_guess_count"])
        assert np.array_equal(
            table["greedy_guess_fraction"], reference["greedy_guess_fraction"]
        )
        assert np.array_equal(table["corpus_fraction"], reference["corpus_fraction"])


def test_only_the_gradient_column_moves_with_temperature() -> None:
    record = _record()
    canonical = temperature_gradient_table(record, 1.00)["mean_gradient_norm"]
    cold = temperature_gradient_table(record, 0.12)["mean_gradient_norm"]

    measured = ~np.isnan(canonical)
    assert not np.allclose(canonical[measured], cold[measured])


def test_the_canonical_row_reproduces_the_legacy_gradient_observable() -> None:
    """T = 1 is one slice of the generalized path, not a second implementation."""

    from llm_behavior_lab.analysis import gradient_guess_table

    record = _record()
    legacy = gradient_guess_table(record)["mean_gradient_norm"]
    generalized = temperature_gradient_table(record, 1.00)["mean_gradient_norm"]

    assert np.array_equal(
        record.gradient_temperature_position_norms[record.gradient_temperature_index(1.00)],
        record.gradient_position_norms,
    )
    measured = ~np.isnan(legacy)
    assert np.array_equal(legacy[measured], generalized[measured])


def test_a_drifted_canonical_row_is_rejected() -> None:
    record = _record()
    broken = np.array(record.gradient_temperature_position_norms)
    broken[record.gradient_temperature_index(1.00)] += 0.5

    with pytest.raises(ValueError, match="canonical gradient temperature row"):
        InitializationExperimentRecord.build(
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
            gradient_temperatures=record.gradient_temperatures,
            gradient_temperature_position_norms=broken,
        )


def test_token_aggregation_is_hand_computable() -> None:
    """G_i(T) is the mean over exactly the positions whose target is i."""

    record = _record()
    targets = np.asarray(record.gradient_position_target_ids)
    for temperature in (0.12, 1.00):
        index = record.gradient_temperature_index(temperature)
        norms = np.asarray(record.gradient_temperature_position_norms[index])
        table = temperature_gradient_table(record, temperature)
        for token in np.unique(targets):
            expected = norms[targets == token].mean()
            assert table["mean_gradient_norm"][token] == pytest.approx(expected)


def test_the_summary_shares_one_occurrence_distribution() -> None:
    """n(i) counts targets, so reporting it per temperature would mislead."""

    summary = temperature_gradient_summary(_record())

    assert "target_occurrence_count" in summary
    assert all("target_occurrence_count" not in row for row in summary["rows"])
    assert summary["greedy_is_temperature_invariant"] is True
    assert {row["temperature"] for row in summary["rows"]} == set(TEMPERATURES)
    for row in summary["rows"]:
        assert len(row["strata"]) == 7
        assert len(row["sensitivity_by_min_occurrences"]) == 6
        assert row["num_tokens"] == summary["num_tokens"]


# -- persistence --------------------------------------------------------------


def test_temperature_gradient_norms_survive_a_round_trip(tmp_path) -> None:
    record = _record()
    record.save(tmp_path)
    reloaded = load_record(tmp_path)

    assert reloaded.has_temperature_gradient_analysis is True
    assert reloaded.gradient_temperature_grid == TEMPERATURES
    assert reloaded.gradient_temperature_position_norms.shape == (
        len(TEMPERATURES),
        NUM_POSITIONS,
    )
    assert np.allclose(
        reloaded.gradient_temperature_position_norms,
        record.gradient_temperature_position_norms,
    )


def test_a_record_with_only_canonical_gradients_still_loads(tmp_path) -> None:
    record = _record(with_temperatures=False)
    record.save(tmp_path)
    reloaded = load_record(tmp_path)

    assert reloaded.has_position_gradients is True
    assert reloaded.has_temperature_gradient_analysis is False
    assert reloaded.gradient_temperature_grid == ()
    with pytest.raises(ValueError, match="no temperature-conditioned gradient"):
        temperature_gradient_summary(reloaded)


def test_temperature_gradient_arrays_are_all_or_nothing() -> None:
    record = _record()
    with pytest.raises(ValueError, match="all-or-nothing"):
        InitializationExperimentRecord.build(
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
            gradient_temperatures=record.gradient_temperatures,
        )


# -- figure 10 ----------------------------------------------------------------


def _panels_from(record, directory, **kwargs):
    from matplotlib.figure import Figure

    created = []
    original = Figure.subplots

    def capture(self, *args, **inner):
        axes = original(self, *args, **inner)
        created.append(axes)
        return axes

    Figure.subplots = capture
    try:
        written = plot_temperature_gradient_vs_guess_bias(record, directory, **kwargs)
    finally:
        Figure.subplots = original
    grid = created[0]
    return [axes for row in grid for axes in row if axes.get_visible()], written


def test_figure_ten_has_six_panels(tmp_path) -> None:
    panels, written = _panels_from(_record(), tmp_path)

    assert [path.name for path in written] == [
        "figure10_temperature_gradient_vs_initial_guess_bias.svg"
    ]
    assert written[0].is_file() and written[0].stat().st_size > 0
    assert len(panels) == len(PANELS) == 6
    assert [axes.get_title().split()[2] for axes in panels] == [
        f"{value:g}" for value in PANELS
    ]


def test_every_panel_plots_the_same_guess_values(tmp_path) -> None:
    """q(i) is temperature-invariant, so the y coordinates must be identical."""

    panels, _ = _panels_from(_record(), tmp_path)
    reference = panels[0].collections[0].get_offsets()[:, 1]

    for axes in panels[1:]:
        assert np.array_equal(axes.collections[0].get_offsets()[:, 1], reference)
    assert (reference == 0.0).any()  # zero-guess tokens are retained


def test_every_panel_shares_one_y_transform_and_range(tmp_path) -> None:
    """q(i) is identical across temperature, so its axis must be too."""

    panels, _ = _panels_from(_record(), tmp_path)

    assert len({axes.get_ylim() for axes in panels}) == 1
    assert len({axes.get_yscale() for axes in panels}) == 1
    assert len({tuple(axes.get_yticks()) for axes in panels}) == 1
    for axes in panels:
        assert axes.get_ylim()[0] <= 0.0  # q = 0 is on the axis, not at the frame

    # A shared y axis blanks the inner panels' labels by design, so the drawn
    # text lives on the left column; the ticks themselves are shared above.
    labels = [label.get_text() for label in panels[0].get_yticklabels()]
    assert labels[0] == "0"
    assert all(label.endswith("/D") for label in labels[1:])


def test_the_symlog_representation_remains_selectable(tmp_path) -> None:
    panels, _ = _panels_from(_record(), tmp_path, y_transform="symlog")

    assert all(axes.get_yscale() == "symlog" for axes in panels)


def test_one_shared_colour_normalization_and_one_colorbar(tmp_path) -> None:
    from matplotlib.colors import LogNorm

    panels, _ = _panels_from(_record(), tmp_path)
    norms = [axes.collections[0].norm for axes in panels]

    assert all(isinstance(norm, LogNorm) for norm in norms)
    assert len({(norm.vmin, norm.vmax) for norm in norms}) == 1
    assert len({axes.collections[0].cmap.name for axes in panels}) == 1
    assert panels[0].collections[0].cmap.name == "viridis"


def test_exactly_one_colorbar_is_drawn(tmp_path) -> None:
    """Six colorbars would imply six colour scales; there is only one."""

    from matplotlib.figure import Figure

    created = []
    original = Figure.colorbar

    def capture(self, *args, **kwargs):
        created.append(True)
        return original(self, *args, **kwargs)

    Figure.colorbar = capture
    try:
        plot_temperature_gradient_vs_guess_bias(_record(), tmp_path)
    finally:
        Figure.colorbar = original

    assert len(created) == 1


def test_each_panel_gets_limits_from_its_own_gradient_distribution(tmp_path) -> None:
    """G(i, T) sits at a different place for every T, so the limits follow it.

    Shared limits leave every cloud in a sliver of its panel, which is a display
    failure rather than a scientific one. The transformation stays the same in
    all six; only the limits differ.
    """

    record = _record()
    panels, _ = _panels_from(record, tmp_path)

    limits = [axes.get_xlim() for axes in panels]
    assert len(set(limits)) == len(panels)  # genuinely per panel
    assert all(axes.get_xscale() == "log" for axes in panels)

    for axes, temperature in zip(panels, PANELS):
        table = temperature_gradient_table(record, temperature)
        values = table["mean_gradient_norm"][table["target_occurrence_count"] > 0]
        finite = values[np.isfinite(values) & (values > 0)]
        low, high = axes.get_xlim()
        # Derived from this temperature's own data, and covering all of it.
        assert low < float(finite.min()) <= float(finite.max()) < high
        assert f"{low:.3g}" in axes.get_title()


def test_no_point_falls_outside_its_panel_limits(tmp_path) -> None:
    """Nothing is clipped: every plotted marker lies inside the axes."""

    panels, _ = _panels_from(_record(), tmp_path)

    for axes in panels:
        offsets = axes.collections[0].get_offsets()
        low, high = axes.get_xlim()
        bottom, top = axes.get_ylim()
        assert offsets[:, 0].min() >= low and offsets[:, 0].max() <= high
        assert offsets[:, 1].min() >= bottom and offsets[:, 1].max() <= top


def test_shared_x_limits_remain_available(tmp_path) -> None:
    panels, _ = _panels_from(_record(), tmp_path, x_limits="shared")

    assert len({axes.get_xlim() for axes in panels}) == 1


def test_the_count_transform_maps_zero_exactly_and_is_monotonic(tmp_path) -> None:
    """log10(1 + k) sends q = 0 to exactly 0 and reorders nothing.

    A logarithmic axis would delete the zero-guess tokens, which are most of
    them; this keeps every one and separates "never guessed" from "guessed once"
    by a visible 0.30 of height.
    """

    record = _record()
    panels, _ = _panels_from(record, tmp_path)
    table = temperature_gradient_table(record, PANELS[0])
    plotted = table["target_occurrence_count"] > 0
    counts = table["greedy_guess_count"][plotted]

    drawn = panels[0].collections[0].get_offsets()[:, 1]
    expected = np.log10(1.0 + counts)
    assert np.allclose(drawn, expected)
    assert np.all(drawn[counts == 0] == 0.0)
    order = np.argsort(counts)
    assert np.all(np.diff(drawn[order]) >= 0.0)
    # Never-guessed tokens are a real population here, not a rounding artefact.
    assert (counts == 0).sum() > 0


def test_the_original_guess_fractions_are_untouched_by_the_display(tmp_path) -> None:
    """The transform is display only: q(i) in the data is unchanged."""

    record = _record()
    before = temperature_gradient_table(record, 1.00)["greedy_guess_fraction"].copy()
    _panels_from(record, tmp_path)
    after = temperature_gradient_table(record, 1.00)["greedy_guess_fraction"]

    assert np.array_equal(before, after)
    counts = record.greedy_counts[0]
    assert np.array_equal(
        after, counts.astype(np.float64) / float(record.gradient_position_norms.shape[0])
    )


def test_a_record_without_temperature_gradients_refuses_figure_ten(tmp_path) -> None:
    with pytest.raises(ValueError, match="no temperature-conditioned gradient"):
        plot_temperature_gradient_vs_guess_bias(_record(with_temperatures=False), tmp_path)


def test_the_figure_set_prefers_the_temperature_version(tmp_path) -> None:
    """With the grid present figure 10 is drawn; without it, the supplement."""

    modern = {path.name for path in generate_all_figures(_record(), tmp_path / "new")}
    legacy = {
        path.name
        for path in generate_all_figures(_record(with_temperatures=False), tmp_path / "old")
    }

    assert "figure10_temperature_gradient_vs_initial_guess_bias.svg" in modern
    assert "supplementary_t1_gradient_vs_initial_guess_bias.svg" not in modern
    assert "supplementary_t1_gradient_vs_initial_guess_bias.svg" in legacy
    assert "figure10_temperature_gradient_vs_initial_guess_bias.svg" not in legacy


def test_the_renderer_maps_figure_ten_to_the_temperature_version() -> None:
    import importlib.util
    import sys
    from pathlib import Path

    from llm_behavior_lab.analysis import figures as figure_module

    path = Path(__file__).resolve().parents[1] / "scripts" / "render_record_figures.py"
    spec = importlib.util.spec_from_file_location("render_record_figures", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    assert module.FIGURES["figure10"] == "plot_temperature_gradient_vs_guess_bias"
    assert module.CONDITIONAL_FIGURES["figure10"] == "has_temperature_gradient_analysis"
    assert callable(getattr(figure_module, module.FIGURES["figure10"]))
    assert callable(getattr(figure_module, module.FIGURES["supplementary-gradient"]))
