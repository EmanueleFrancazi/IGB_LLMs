"""Tests for figure 8 and the statistics printed beside it.

Two properties matter more than the drawing itself and are pinned hardest here.

Tokens that were never greedily guessed must survive. ``q_i = 0`` is a measured
outcome, and both the figure's axis choice and the primary correlation have to
keep those tokens rather than quietly dropping them onto a logarithmic axis.

The rank correlation must be tie-corrected. Most tokens sit in one enormous tie
block at ``q_i = 0``, so an ordinal ranking would invent an order inside it and
report a coefficient that is partly an artefact of the sort.
"""

from __future__ import annotations

import numpy as np
import pytest

from llm_behavior_lab.analysis import (
    InitializationExperimentRecord,
    gradient_guess_correlations,
    gradient_observable_summary,
    spearman_rho,
)

matplotlib = pytest.importorskip("matplotlib")

from llm_behavior_lab.analysis.figures import (  # noqa: E402
    generate_all_figures,
    plot_gradient_vs_guess_bias,
)

VOCAB_SIZE = 60
NUM_POSITIONS = 512


def _record(*, with_gradients: bool = True, seed: int = 5, num_positions: int = NUM_POSITIONS):
    """A record shaped like the real one: heavy-tailed corpus, mostly-zero q."""

    rng = np.random.default_rng(seed)
    eligible = np.arange(3, VOCAB_SIZE)
    weights = rng.dirichlet(np.ones(eligible.size) * 0.3)

    # Targets are drawn from a realized corpus sequence, not independently from
    # the weights, because that is how the experiment works: an evaluation target
    # is a position *in* the split. Every target token therefore has corpus mass
    # by construction, and the fixture cannot accidentally produce a token with
    # n(i) > 0 and p(i) = 0, which the real record can never contain either.
    sequence = rng.choice(eligible, size=20_000, p=weights)
    corpus = np.bincount(sequence, minlength=VOCAB_SIZE).astype(np.int64)
    targets = rng.choice(sequence, size=num_positions)
    # Greedy collapses onto a handful of tokens at initialization, so most
    # tokens end with q(i) = 0 exactly as in the real experiment.
    favoured = eligible[:4]
    greedy = rng.choice(favoured, size=num_positions)
    norms = rng.lognormal(mean=0.0, sigma=0.5, size=num_positions)

    selected = np.bincount(targets, minlength=VOCAB_SIZE)
    greedy_counts = np.bincount(greedy, minlength=VOCAB_SIZE)
    metadata = {
        "tokens": [f"tok{index}" for index in range(VOCAB_SIZE)],
        "analysis": {"num_positions": num_positions},
    }
    gradients = {}
    if with_gradients:
        metadata["analysis"]["gradient_analysis"] = {
            "enabled": True,
            "softmax_support": "eligible",
            "initialization_index": 0,
            "model_seed": 1000,
            "input_condition": "real",
            "covers_all_positions": True,
            "parameter_count": 8_585_856,
        }
        gradients = {
            "gradient_position_indices": np.arange(num_positions),
            "gradient_position_target_ids": targets,
            "gradient_position_greedy_ids": greedy,
            "gradient_position_norms": norms,
        }

    return InitializationExperimentRecord.build(
        corpus_counts=corpus,
        selected_target_counts=selected,
        greedy_counts=greedy_counts[None, :],
        nucleus_counts=greedy_counts[None, None, :],
        mean_predicted_probabilities=np.full((1, VOCAB_SIZE), 1.0 / VOCAB_SIZE),
        model_seeds=[1000],
        eligible_token_ids=eligible,
        metadata=metadata,
        **gradients,
    )


# -- the rank correlation ----------------------------------------------------


def test_spearman_is_exact_on_a_hand_computable_case() -> None:
    """Monotone data gives exactly +/-1, regardless of spacing."""

    assert spearman_rho(np.array([1.0, 2, 3, 4]), np.array([10.0, 200, 3000, 40000])) == 1.0
    assert spearman_rho(np.array([1.0, 2, 3, 4]), np.array([4.0, 3, 2, 1])) == -1.0


def test_spearman_uses_average_ranks_for_ties() -> None:
    """The tie correction, checked against a value computed by hand.

    x = [1, 1, 2, 3] has average ranks [1.5, 1.5, 3, 4]; y = [5, 6, 7, 8] has
    ranks [1, 2, 3, 4]. Pearson on those ranks is 0.9486832980505138.
    """

    rho = spearman_rho(np.array([1.0, 1.0, 2.0, 3.0]), np.array([5.0, 6.0, 7.0, 8.0]))
    assert rho == pytest.approx(0.9486832980505138, rel=1e-12)


def test_spearman_is_undefined_rather_than_zero_for_constant_input() -> None:
    """A constant vector has no ranking, so the coefficient does not exist."""

    assert np.isnan(spearman_rho(np.ones(6), np.arange(6.0)))
    assert np.isnan(spearman_rho(np.array([1.0]), np.array([2.0])))


def test_a_massive_tie_block_does_not_manufacture_correlation() -> None:
    """Zero-guess tokens are the majority; ties among them must stay unordered.

    With y constant on all but one point, ordinal ranking would impose the x
    order inside the tie block. The tie-corrected coefficient must not.
    """

    x = np.arange(100.0)
    y = np.zeros(100)
    y[-1] = 1.0
    assert abs(spearman_rho(x, y)) < 0.2


# -- distribution reporting --------------------------------------------------


def test_the_summary_reports_every_distribution_needed_for_a_scale_choice() -> None:
    summary = gradient_observable_summary(_record())

    for key in ("mean_gradient_norm", "greedy_guess_fraction", "positive_corpus_fraction",
                "target_occurrence_count"):
        assert summary[key]["count"] > 0
        assert summary[key]["p00"] <= summary[key]["p50"] <= summary[key]["p100"]
    assert summary["num_plotted_tokens"] > 0
    assert summary["occurrence_histogram"]["1"] >= 0
    assert 0.0 <= summary["zero_guess_fraction"] <= 1.0
    assert summary["gradient_dynamic_range"] >= 1.0


def test_zero_guess_tokens_are_counted_not_dropped() -> None:
    """The count of never-guessed tokens is reported and is non-trivial here."""

    record = _record()
    summary = gradient_observable_summary(record)
    correlations = gradient_guess_correlations(record)

    assert summary["zero_guess_tokens"] > 0
    # Every token with a target is used, whether or not it was ever guessed.
    assert correlations["num_tokens"] == summary["num_plotted_tokens"]
    assert correlations["includes_zero_guess_tokens"] is True
    assert correlations["weighted"] is False


def test_every_plotted_token_has_corpus_mass() -> None:
    """A selected target always occurs in the split, so p(i) > 0 for all marks."""

    summary = gradient_observable_summary(_record())

    assert summary["nonpositive_corpus_tokens"] == 0
    assert summary["positive_corpus_fraction"]["count"] == summary["num_plotted_tokens"]
    assert summary["positive_corpus_fraction"]["p00"] > 0.0


def test_a_token_without_corpus_mass_is_reported_not_silently_dropped(tmp_path) -> None:
    """The colour scale is logarithmic, so p(i) = 0 cannot be placed on it.

    This cannot arise from the experiment, where every target is a position in
    the split, but the figure must not quietly lose a mark if it ever did: the
    omission is counted and stated on the figure.
    """

    record = _record()
    # Strip one plotted token's corpus mass, keeping its gradient measurement.
    victim = int(record.gradient_position_target_ids[0])
    record.corpus_counts[victim] = 0

    summary = gradient_observable_summary(record)
    assert summary["nonpositive_corpus_tokens"] == 1

    from matplotlib.figure import Figure

    created = []
    original = Figure.subplots

    def capture(self, *args, **kwargs):
        axes = original(self, *args, **kwargs)
        created.append(axes)
        return axes

    Figure.subplots = capture
    try:
        plot_gradient_vs_guess_bias(record, tmp_path)
    finally:
        Figure.subplots = original

    marks = created[0].collections[0].get_offsets().shape[0]
    assert marks == summary["num_plotted_tokens"] - 1
    assert any("no corpus mass" in text.get_text() for text in created[0].texts)


def test_correlations_cover_the_three_requested_pairs_and_strata() -> None:
    correlations = gradient_guess_correlations(_record())

    for key in ("spearman_gradient_vs_guess", "spearman_gradient_vs_corpus",
                "spearman_guess_vs_corpus"):
        value = correlations[key]
        assert np.isnan(value) or -1.0 <= value <= 1.0
    assert sum(s["num_tokens"] for s in correlations["strata"]) == correlations["num_tokens"]
    assert correlations["sensitivity_by_min_occurrences"][0]["num_tokens"] == correlations["num_tokens"]


def test_the_primary_correlation_matches_a_direct_computation() -> None:
    """The reported rho is exactly Spearman over the plotted tokens."""

    from llm_behavior_lab.analysis import gradient_guess_table

    record = _record()
    table = gradient_guess_table(record)
    plotted = table["target_occurrence_count"] > 0
    expected = spearman_rho(
        table["mean_gradient_norm"][plotted], table["greedy_guess_fraction"][plotted]
    )

    assert gradient_guess_correlations(record)["spearman_gradient_vs_guess"] == expected


# -- the figure --------------------------------------------------------------


def test_figure_eight_is_written_as_svg(tmp_path) -> None:
    written = plot_gradient_vs_guess_bias(_record(), tmp_path)

    assert [path.name for path in written] == ["figure8_gradient_vs_initial_guess_bias.svg"]
    assert written[0].is_file() and written[0].stat().st_size > 0
    assert "svg" in written[0].read_text(encoding="utf-8")[:2000].lower()


def test_the_y_axis_keeps_zero_guess_tokens_visible(tmp_path) -> None:
    """The axis must reach q = 0 and be symlog, not log.

    A plain logarithmic y axis would silently delete every never-guessed token,
    which is most of them and the point of the figure.
    """

    from matplotlib.figure import Figure

    record = _record()
    created: list[Figure] = []
    original = Figure.subplots

    def capture(self, *args, **kwargs):
        axes = original(self, *args, **kwargs)
        created.append(axes)
        return axes

    Figure.subplots = capture
    try:
        plot_gradient_vs_guess_bias(record, tmp_path)
    finally:
        Figure.subplots = original

    axes = created[0]
    assert axes.get_yscale() == "symlog"
    assert axes.get_ylim()[0] <= 0.0
    summary = gradient_observable_summary(record)
    assert summary["zero_guess_tokens"] > 0  # there really were points at zero


def test_the_gradient_axis_follows_the_observed_range(tmp_path) -> None:
    """Log once the norms span a decade, linear otherwise, overridable."""

    from matplotlib.figure import Figure

    def scale_for(record, **kwargs) -> str:
        created = []
        original = Figure.subplots

        def capture(self, *args, **inner):
            axes = original(self, *args, **inner)
            created.append(axes)
            return axes

        Figure.subplots = capture
        try:
            plot_gradient_vs_guess_bias(record, tmp_path, **kwargs)
        finally:
            Figure.subplots = original
        return created[0].get_xscale()

    record = _record()
    narrow = _record()
    # Compress the norms into a fraction of a decade; the axis should follow.
    narrow.gradient_position_norms[:] = 1.0 + 0.01 * np.arange(narrow.gradient_position_norms.size)

    assert scale_for(narrow) == "linear"
    assert scale_for(narrow, x_scale="log") == "log"
    assert scale_for(record, x_scale="linear") == "linear"


def test_the_colour_scale_is_logarithmic_and_unclipped(tmp_path) -> None:
    """Heavy-tailed p(i) needs a log norm spanning the real data, not a clip."""

    from matplotlib.colors import LogNorm
    from matplotlib.figure import Figure

    record = _record()
    created = []
    original = Figure.subplots

    def capture(self, *args, **kwargs):
        axes = original(self, *args, **kwargs)
        created.append(axes)
        return axes

    Figure.subplots = capture
    try:
        plot_gradient_vs_guess_bias(record, tmp_path)
    finally:
        Figure.subplots = original

    collection = created[0].collections[0]
    assert isinstance(collection.norm, LogNorm)
    corpus = record.corpus_fractions[
        np.asarray(record.gradient_position_target_ids)
    ]
    assert collection.norm.vmin == pytest.approx(corpus.min(), rel=1e-9)
    assert collection.norm.vmax == pytest.approx(corpus.max(), rel=1e-9)
    assert collection.cmap.name == "viridis"


def test_every_plotted_token_appears_as_a_mark(tmp_path) -> None:
    """No token is dropped between the table and the canvas."""

    from matplotlib.figure import Figure

    record = _record()
    created = []
    original = Figure.subplots

    def capture(self, *args, **kwargs):
        axes = original(self, *args, **kwargs)
        created.append(axes)
        return axes

    Figure.subplots = capture
    try:
        plot_gradient_vs_guess_bias(record, tmp_path)
    finally:
        Figure.subplots = original

    offsets = created[0].collections[0].get_offsets()
    assert offsets.shape[0] == gradient_observable_summary(record)["num_plotted_tokens"]
    assert (offsets[:, 1] == 0.0).any()  # zero-guess tokens really are drawn


def test_annotations_stay_a_small_deterministic_set(tmp_path) -> None:
    """At thousands of marks, labelling broadly would destroy the figure."""

    from matplotlib.figure import Figure

    created = []
    original = Figure.subplots

    def capture(self, *args, **kwargs):
        axes = original(self, *args, **kwargs)
        created.append(axes)
        return axes

    Figure.subplots = capture
    try:
        plot_gradient_vs_guess_bias(_record(), tmp_path, num_labels=4)
    finally:
        Figure.subplots = original

    # The one-guess reference note plus at most four token labels.
    assert len(created[0].texts) <= 6


# -- integration -------------------------------------------------------------


def test_the_figure_set_includes_figure_eight_only_with_gradient_data(tmp_path) -> None:
    with_gradients = generate_all_figures(_record(), tmp_path / "with")
    without = generate_all_figures(_record(with_gradients=False), tmp_path / "without")

    names_with = {path.name for path in with_gradients}
    names_without = {path.name for path in without}

    assert "figure8_gradient_vs_initial_guess_bias.svg" in names_with
    assert "figure8_gradient_vs_initial_guess_bias.svg" not in names_without
    # Runs without gradient data are otherwise completely unchanged.
    assert names_with - {"figure8_gradient_vs_initial_guess_bias.svg"} == names_without


def test_a_record_without_gradients_refuses_to_draw_figure_eight(tmp_path) -> None:
    with pytest.raises(ValueError, match="no per-position gradient analysis"):
        plot_gradient_vs_guess_bias(_record(with_gradients=False), tmp_path)


def test_an_invalid_axis_choice_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="x_scale"):
        plot_gradient_vs_guess_bias(_record(), tmp_path, x_scale="sqrt")


# -- the record-only entry point --------------------------------------------


def _render_script():
    """Import the record-only renderer without running it."""

    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "render_record_figures.py"
    spec = importlib.util.spec_from_file_location("render_record_figures", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_renderer_selects_real_figure_functions() -> None:
    """Every selectable name resolves, so --only cannot silently do nothing."""

    from llm_behavior_lab.analysis import figures as figure_module

    module = _render_script()
    assert "figure8" in module.FIGURES
    for name, function_name in module.FIGURES.items():
        assert callable(getattr(figure_module, function_name)), name


def test_the_renderer_needs_no_torch() -> None:
    """Re-analysis must not require the framework that produced the record."""

    import ast
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "render_record_figures.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    assert "torch" not in imported


def test_the_renderer_reports_the_statistics(capsys) -> None:
    """The printed report carries the numbers the scale choices rest on."""

    module = _render_script()
    statistics = module.report_gradient_statistics(_record())
    printed = capsys.readouterr().out

    assert "rho(G, q)" in printed and "<- primary" in printed
    assert "kept, not discarded" in printed
    assert set(statistics) == {"distributions", "correlations"}
