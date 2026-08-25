"""Tests for figure generation.

Figures are checked for the properties a test can actually establish: they are
produced without a display, they land under deterministic names, and they draw
the data they claim to. Their visual quality is not something a unit test can
judge, so nothing here pretends to.

``matplotlib`` is optional, so the whole module is skipped when it is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("matplotlib", reason="figures require the optional [analysis] extra")

from llm_behavior_lab.analysis import InitializationExperimentRecord  # noqa: E402
from llm_behavior_lab.analysis.figures import (  # noqa: E402
    FIGURE_FORMATS,
    generate_all_figures,
    plot_ranked_frequency_profiles,
    plot_sampling_adequacy,
    plot_token_identity_scatter,
    plot_token_wise_mismatch,
)

VOCAB_SIZE = 8
NUM_INITIALIZATIONS = 4
NUM_REPLICATES = 3
NUM_POSITIONS = 64


def _record() -> InitializationExperimentRecord:
    """A small record with enough structure for every figure."""

    rng = np.random.default_rng(11)
    corpus_probabilities = rng.dirichlet(np.ones(VOCAB_SIZE))
    return InitializationExperimentRecord.build(
        corpus_counts=rng.multinomial(2000, corpus_probabilities),
        selected_target_counts=rng.multinomial(NUM_POSITIONS, corpus_probabilities),
        greedy_counts=np.stack(
            [
                rng.multinomial(NUM_POSITIONS, rng.dirichlet(np.ones(VOCAB_SIZE) * 0.3))
                for _ in range(NUM_INITIALIZATIONS)
            ]
        ),
        nucleus_counts=np.stack(
            [
                np.stack(
                    [
                        rng.multinomial(NUM_POSITIONS, rng.dirichlet(np.ones(VOCAB_SIZE) * 0.3))
                        for _ in range(NUM_REPLICATES)
                    ]
                )
                for _ in range(NUM_INITIALIZATIONS)
            ]
        ),
        mean_predicted_probabilities=rng.dirichlet(np.ones(VOCAB_SIZE), size=NUM_INITIALIZATIONS),
        model_seeds=np.arange(NUM_INITIALIZATIONS),
        metadata={
            "num_positions": NUM_POSITIONS,
            "tokens": [chr(ord("a") + index) for index in range(VOCAB_SIZE)],
        },
    )


@pytest.mark.parametrize(
    "plot_function, stem",
    [
        (plot_sampling_adequacy, "figure0_sampling_adequacy"),
        (plot_ranked_frequency_profiles, "figure1_ranked_frequency_profiles"),
        (plot_token_wise_mismatch, "figure2_token_wise_mismatch"),
        (plot_token_identity_scatter, "figure3_token_identity_scatter"),
    ],
)
def test_each_figure_saves_non_empty_files_headlessly(plot_function, stem, tmp_path) -> None:
    """No display is available in CI, so drawing must never require one."""

    written = plot_function(_record(), tmp_path)

    assert [path.name for path in written] == [f"{stem}.{suffix}" for suffix in FIGURE_FORMATS]
    for path in written:
        assert path.is_file()
        assert path.stat().st_size > 0


def test_generate_all_figures_writes_the_full_set(tmp_path) -> None:
    """One call produces the required figures plus the complementary one."""

    written = generate_all_figures(_record(), tmp_path)

    stems = sorted({path.stem for path in written})
    assert stems == [
        "figure0_sampling_adequacy",
        "figure1_ranked_frequency_profiles",
        "figure2_token_wise_mismatch",
        "figure3_token_identity_scatter",
    ]
    assert len(written) == len(stems) * len(FIGURE_FORMATS)


def test_the_output_directory_is_created_when_missing(tmp_path) -> None:
    """The caller should not have to prepare the destination."""

    destination = tmp_path / "figures" / "initialization"

    generate_all_figures(_record(), destination)

    assert destination.is_dir()


def test_rerunning_overwrites_rather_than_accumulating(tmp_path) -> None:
    """Deterministic filenames keep a run directory from filling with variants."""

    generate_all_figures(_record(), tmp_path)
    first = sorted(path.name for path in tmp_path.rglob("*.svg"))

    generate_all_figures(_record(), tmp_path)

    assert sorted(path.name for path in tmp_path.rglob("*.svg")) == first


def test_figures_do_not_touch_global_pyplot_state() -> None:
    """The object-oriented API keeps notebooks and tests from interfering.

    A leaked ``pyplot`` figure would accumulate across calls and eventually warn
    or exhaust memory in a long notebook session.
    """

    pyplot = pytest.importorskip("matplotlib.pyplot")
    before = len(pyplot.get_fignums())

    generate_all_figures(_record(), pytest.importorskip("tempfile").mkdtemp())

    assert len(pyplot.get_fignums()) == before


def test_a_single_initialization_still_plots() -> None:
    """SEM is undefined for one sample; the figure must degrade, not crash."""

    rng = np.random.default_rng(3)
    record = InitializationExperimentRecord.build(
        corpus_counts=np.array([10, 5, 3, 2]),
        selected_target_counts=np.array([4, 2, 1, 1]),
        greedy_counts=np.array([[4, 2, 1, 1]]),
        nucleus_counts=np.array([[[4, 2, 1, 1]]]),
        mean_predicted_probabilities=rng.dirichlet(np.ones(4), size=1),
        model_seeds=np.array([1]),
        metadata={"num_positions": 8, "tokens": list("abcd")},
    )

    written = generate_all_figures(record, pytest.importorskip("tempfile").mkdtemp())

    assert all(path.stat().st_size > 0 for path in written)


# --- Large-vocabulary rendering ------------------------------------------

from llm_behavior_lab.analysis.figures import (  # noqa: E402
    LARGE_VOCAB_THRESHOLD,
    escape_token_label,
)

LARGE_VOCAB = 32000


def _large_record(num_positions: int = 4096) -> InitializationExperimentRecord:
    """A subword-sized record with structural tokens and a long zero tail."""

    rng = np.random.default_rng(5)
    eligible = np.arange(3, LARGE_VOCAB)
    weights = 1.0 / np.arange(1, eligible.size + 1) ** 1.1
    weights /= weights.sum()

    corpus = np.zeros(LARGE_VOCAB, dtype=np.int64)
    corpus[eligible] = rng.multinomial(200_000, weights)
    selected = np.zeros(LARGE_VOCAB, dtype=np.int64)
    selected[eligible] = rng.multinomial(num_positions, weights)

    peaked = 1.0 / np.arange(1, eligible.size + 1) ** 2.0
    peaked /= peaked.sum()
    greedy = np.zeros((2, LARGE_VOCAB), dtype=np.int64)
    nucleus = np.zeros((2, 2, LARGE_VOCAB), dtype=np.int64)
    for index in range(2):
        greedy[index, eligible] = rng.multinomial(num_positions, peaked)
        nucleus[index][:, eligible] = rng.multinomial(num_positions, peaked, size=2)

    return InitializationExperimentRecord.build(
        corpus_counts=corpus,
        selected_target_counts=selected,
        greedy_counts=greedy,
        nucleus_counts=nucleus,
        mean_predicted_probabilities=np.tile(corpus / corpus.sum(), (2, 1)),
        model_seeds=np.array([1, 2]),
        eligible_token_ids=eligible,
        metadata={
            "num_positions": num_positions,
            "tokens": [f"▁piece{index}" for index in range(LARGE_VOCAB)],
        },
    )


def test_all_figures_render_at_a_subword_vocabulary(tmp_path) -> None:
    """32k tokens, long zero tails, and wide rank ranges must not break plotting."""

    written = generate_all_figures(_large_record(), tmp_path)

    assert len(written) == 4 * len(FIGURE_FORMATS)
    for path in written:
        assert path.stat().st_size > 0


def test_large_vocabulary_output_stays_a_reasonable_size(tmp_path) -> None:
    """Vector output must not blow up once curves have tens of thousands of points.

    Rasterizing only the dense curves keeps axes and text as vector while
    preventing a multi-megabyte SVG.
    """

    generate_all_figures(_large_record(), tmp_path)

    for path in tmp_path.rglob("*.svg"):
        assert path.stat().st_size < 2_000_000, f"{path.name} is too large"


def test_the_large_vocabulary_threshold_is_what_switches_rendering() -> None:
    """The regime is chosen by eligible support, not by the full vocabulary."""

    assert _large_record().eligible_vocab_size > LARGE_VOCAB_THRESHOLD
    assert _record().eligible_vocab_size <= LARGE_VOCAB_THRESHOLD


def test_a_zero_tail_does_not_prevent_rendering(tmp_path) -> None:
    """Most of a subword vocabulary is never guessed; a log axis must cope."""

    record = _large_record(num_positions=64)
    guessed = (record.greedy_counts[0] > 0).sum()

    written = generate_all_figures(record, tmp_path)

    assert guessed < record.eligible_vocab_size / 100
    assert all(path.stat().st_size > 0 for path in written)


def test_token_labels_escape_whitespace_and_control_characters() -> None:
    """Two different tokens must never render identically."""

    assert escape_token_label("\n") != escape_token_label(" ")
    assert "\\n" in escape_token_label("\n")
    assert escape_token_label("▁the").strip("'\"") == "▁the"


def test_token_labels_are_truncated_rather_than_overflowing() -> None:
    """A long byte-fallback piece must not take over the figure."""

    label = escape_token_label("a" * 200)

    assert len(label) <= 16
    assert "…" in label


# --- SVG-only artifacts ---------------------------------------------------


def test_only_svg_artifacts_are_written(tmp_path) -> None:
    """One artifact per figure, and it is vector.

    The dense curves of a subword figure are rasterized *inside* the SVG, so a
    companion PNG added a second file without adding a second format.
    """

    generate_all_figures(_record(), tmp_path)

    # Figures are routed into category subdirectories, so recurse.
    written = sorted(path.name for path in tmp_path.rglob("*.svg"))
    assert written == [
        "figure0_sampling_adequacy.svg",
        "figure1_ranked_frequency_profiles.svg",
        "figure2_token_wise_mismatch.svg",
        "figure3_token_identity_scatter.svg",
    ]


def test_no_png_is_produced(tmp_path) -> None:
    """Asserted separately, because it is the property that regressed."""

    generate_all_figures(_record(), tmp_path)

    assert list(tmp_path.glob("*.png")) == []
    assert len(list(tmp_path.rglob("*.svg"))) == 4


def test_the_declared_format_list_is_svg_only() -> None:
    """Every caller derives its expectations from this constant."""

    assert FIGURE_FORMATS == ("svg",)


def test_a_subword_figure_set_is_also_svg_only(tmp_path) -> None:
    """The large-vocabulary path shares the export code, so it must agree."""

    generate_all_figures(_large_record(), tmp_path)

    assert list(tmp_path.glob("*.png")) == []
    assert len(list(tmp_path.rglob("*.svg"))) == 4


# --- Uniform null and input-structure figures -----------------------------

from llm_behavior_lab.analysis.figures import (  # noqa: E402
    plot_input_structure_profiles,
)
from llm_behavior_lab.analysis.nulls import simulate_uniform_null  # noqa: E402


def _record_with_extras(vocab_size: int = 12, num_positions: int = 64):
    """A record carrying both the uniform null and all three input conditions."""

    rng = np.random.default_rng(21)
    eligible = np.arange(1, vocab_size)
    weights = rng.dirichlet(np.ones(eligible.size))
    corpus = np.zeros(vocab_size, dtype=np.int64)
    corpus[eligible] = rng.multinomial(5000, weights)
    selected = np.zeros(vocab_size, dtype=np.int64)
    selected[eligible] = rng.multinomial(num_positions, weights)

    initializations = 3
    greedy = np.zeros((initializations, vocab_size), dtype=np.int64)
    nucleus = np.zeros((initializations, 1, vocab_size), dtype=np.int64)
    conditions = {name: np.zeros((initializations, vocab_size), dtype=np.int64)
                  for name in ("shuffled", "gaussian")}
    condition_nucleus = {name: np.zeros((initializations, 1, vocab_size), dtype=np.int64)
                         for name in ("shuffled", "gaussian")}
    for index in range(initializations):
        greedy[index, eligible] = rng.multinomial(num_positions, rng.dirichlet(np.ones(eligible.size)))
        nucleus[index][:, eligible] = rng.multinomial(
            num_positions, rng.dirichlet(np.ones(eligible.size)), size=1
        )
        for name in conditions:
            conditions[name][index, eligible] = rng.multinomial(
                num_positions, rng.dirichlet(np.ones(eligible.size))
            )
            condition_nucleus[name][index][:, eligible] = rng.multinomial(
                num_positions, rng.dirichlet(np.ones(eligible.size)), size=1
            )

    null = simulate_uniform_null(
        eligible_vocab_size=eligible.size, num_draws=num_positions, num_replicates=16, seed=5
    )
    return InitializationExperimentRecord.build(
        corpus_counts=corpus,
        selected_target_counts=selected,
        greedy_counts=greedy,
        nucleus_counts=nucleus,
        mean_predicted_probabilities=np.tile(corpus / corpus.sum(), (initializations, 1)),
        model_seeds=np.arange(initializations),
        eligible_token_ids=eligible,
        condition_greedy_counts=conditions,
        condition_nucleus_counts=condition_nucleus,
        uniform_null={
            "ranked_mean": null.ranked_mean,
            "ranked_low": null.ranked_low,
            "ranked_high": null.ranked_high,
        },
        metadata={
            "num_positions": num_positions,
            "tokens": [f"t{index}" for index in range(vocab_size)],
            "uniform_null": null.as_dict(),
        },
    )


def test_figure_one_draws_the_uniform_null(tmp_path) -> None:
    """The null belongs on the ranked-concentration figure."""

    written = plot_ranked_frequency_profiles(_record_with_extras(), tmp_path)

    assert [path.name for path in written] == ["figure1_ranked_frequency_profiles.svg"]
    assert written[0].stat().st_size > 0


def test_figure_one_labels_the_null_interval_as_monte_carlo(tmp_path) -> None:
    """It must not be readable as a SEM across model initializations.

    Checked in the rendered SVG text, which is where a reader actually sees it.
    """

    plot_ranked_frequency_profiles(_record_with_extras(), tmp_path)

    content = (
        tmp_path / "main" / "figure1_ranked_frequency_profiles.svg"
    ).read_text(encoding="utf-8")
    assert "Monte" in content and "Carlo" in content
    assert "SEM" in content


def test_figure_one_still_renders_without_a_null(tmp_path) -> None:
    """The null is optional; older records must keep plotting."""

    written = plot_ranked_frequency_profiles(_record(), tmp_path)

    assert written[0].stat().st_size > 0


def test_figure_four_compares_the_input_conditions(tmp_path) -> None:
    """Two panels, one per policy, three conditions each."""

    written = plot_input_structure_profiles(_record_with_extras(), tmp_path)

    assert [path.name for path in written] == ["figure4_input_structure_profiles.svg"]
    content = written[0].read_text(encoding="utf-8")
    for label in ("real", "shuffled", "Gaussian", "greedy", "nucleus"):
        assert label in content


def test_figure_four_requires_input_conditions() -> None:
    """A record without them must say so rather than draw a misleading figure."""

    with pytest.raises(ValueError, match="no input-structure conditions"):
        plot_input_structure_profiles(_record(), pytest.importorskip("tempfile").mkdtemp())


def test_a_sweep_run_writes_the_three_temperature_figures(tmp_path) -> None:
    """The former three-panel figure is now three standalone SVGs."""

    from llm_behavior_lab.analysis.figures import (
        plot_temperature_ranked_distances,
        plot_temperature_ranked_profiles,
        plot_temperature_support_and_agreement,
    )
    from llm_behavior_lab.analysis.nulls import simulate_uniform_null

    base = _record_with_extras()
    eligible = base.eligible_token_ids
    temperatures = [0.12, 0.6, 1.2]
    rng = np.random.default_rng(31)
    sweeps = {c: np.zeros((base.num_initializations, len(temperatures), base.vocab_size), dtype=np.int64)
              for c in ("real", "shuffled", "gaussian")}
    agreement = {c: np.zeros((base.num_initializations, len(temperatures))) for c in sweeps}
    for condition in sweeps:
        for s_ in range(base.num_initializations):
            for t_ in range(len(temperatures)):
                sweeps[condition][s_, t_, eligible] = rng.multinomial(
                    64, rng.dirichlet(np.ones(eligible.size))
                )
    null = simulate_uniform_null(
        eligible_vocab_size=eligible.size, num_draws=64, num_replicates=8, seed=3
    )
    metadata = dict(base.metadata)
    metadata["analysis"] = {"sampling": {"temperature": 0.6, "top_p": 0.9},
                            "temperature_sweep": {"enabled": True, "temperatures": temperatures}}
    record = InitializationExperimentRecord.build(
        corpus_counts=base.corpus_counts, selected_target_counts=base.selected_target_counts,
        greedy_counts=base.greedy_counts, nucleus_counts=base.nucleus_counts,
        mean_predicted_probabilities=base.mean_predicted_probabilities,
        model_seeds=base.model_seeds, eligible_token_ids=eligible,
        condition_greedy_counts=base.condition_greedy_counts,
        condition_nucleus_counts=base.condition_nucleus_counts,
        uniform_null={"ranked_mean": null.ranked_mean, "ranked_low": null.ranked_low,
                      "ranked_high": null.ranked_high},
        sweep_counts_by_condition=sweeps, sweep_agreement_by_condition=agreement,
        metadata=metadata,
    )

    written = generate_all_figures(record, tmp_path)

    assert sorted(path.name for path in written) == [
        "figure0_sampling_adequacy.svg",
        "figure1_ranked_frequency_profiles.svg",
        "figure2_token_wise_mismatch.svg",
        "figure3_token_identity_scatter.svg",
        "figure4_input_structure_profiles.svg",
        "figure5_temperature_ranked_profiles.svg",
        "figure6_temperature_ranked_distances.svg",
        "figure7_temperature_support_and_greedy_agreement.svg",
    ]
    assert list(tmp_path.glob("*.png")) == []
    # Each renders standalone too.
    for plot in (plot_temperature_ranked_profiles, plot_temperature_ranked_distances,
                 plot_temperature_support_and_agreement):
        assert plot(record, tmp_path)[0].stat().st_size > 0


def test_a_full_run_writes_exactly_five_svgs(tmp_path) -> None:
    """The expected artifact set for the current protocol."""

    written = generate_all_figures(_record_with_extras(), tmp_path)

    assert sorted(path.name for path in written) == [
        "figure0_sampling_adequacy.svg",
        "figure1_ranked_frequency_profiles.svg",
        "figure2_token_wise_mismatch.svg",
        "figure3_token_identity_scatter.svg",
        "figure4_input_structure_profiles.svg",
    ]
    assert list(tmp_path.glob("*.png")) == []


def test_a_run_without_input_conditions_writes_four(tmp_path) -> None:
    """Figure 4 appears only when there is something to compare."""

    written = generate_all_figures(_record(), tmp_path)

    assert len(written) == 4
    assert not (tmp_path / "main" / "figure4_input_structure_profiles.svg").exists()


def test_the_new_figures_render_at_a_subword_vocabulary(tmp_path) -> None:
    """32k tokens, a long null zero tail, and six condition curves."""

    written = generate_all_figures(_record_with_extras(vocab_size=4000, num_positions=2048), tmp_path)

    assert len(written) == 5
    for path in written:
        assert path.stat().st_size > 0
        assert path.suffix == ".svg"


# -- temperature colour identity ---------------------------------------------
#
# Colour is bound to the temperature value rather than to a curve's position in
# the list. Spreading a colormap across ``len(temperatures)`` gave the same
# temperature a different colour whenever the number of curves differed -- the
# six-temperature nucleus sweep of figure 5 against the seven-temperature
# confidence grid of figures 11 and 14 -- so a reader could not carry a colour
# from one figure to the next.


def _viridis_at(count: int) -> list:
    """The historical ordinal mapping, recomputed independently of the module."""

    from matplotlib import cm

    return [cm.viridis(value) for value in np.linspace(0.05, 0.92, count)]


def test_the_canonical_grid_keeps_the_colours_the_seven_curve_figures_had() -> None:
    """Figures 11 and 14 must be pixel-identical to their previous rendering."""

    from llm_behavior_lab.analysis.figures import (
        _CANONICAL_TEMPERATURES,
        _temperature_colours,
    )

    assert list(_CANONICAL_TEMPERATURES) == [0.12, 0.24, 0.36, 0.48, 0.60, 1.00, 1.20]
    assert _temperature_colours(_CANONICAL_TEMPERATURES) == _viridis_at(7)


def test_a_shared_temperature_gets_one_colour_whatever_the_curve_count() -> None:
    """The defect this mapping exists to fix, stated as an equality.

    Figure 5 draws six temperatures and figures 11/14 draw seven. Under the old
    ordinal mapping only the two endpoints agreed; every interior temperature
    moved.
    """

    from llm_behavior_lab.analysis.figures import _temperature_colours

    seven = [0.12, 0.24, 0.36, 0.48, 0.60, 1.00, 1.20]
    six = [0.12, 0.24, 0.36, 0.48, 0.60, 1.20]

    palette = dict(zip(seven, _temperature_colours(seven)))
    assert _temperature_colours(six) == [palette[value] for value in six]

    # The old mapping genuinely disagreed, so this test can fail.
    assert _temperature_colours(six) != _viridis_at(6)


def test_a_non_canonical_grid_falls_back_to_the_ordinal_mapping() -> None:
    """Historical and custom grids keep exactly their previous appearance."""

    from llm_behavior_lab.analysis.figures import _temperature_colours

    for temperatures in ([0.5, 0.9], [0.1, 0.2, 0.3, 0.4], [0.7]):
        assert _temperature_colours(temperatures) == _viridis_at(len(temperatures))

    # A canonical grid with one extra value is not the canonical grid.
    extended = [0.12, 0.24, 0.36, 0.48, 0.60, 1.00, 1.20, 1.50]
    assert _temperature_colours(extended) == _viridis_at(8)


def test_a_near_miss_temperature_is_not_snapped_onto_a_canonical_key() -> None:
    """Lookup is exact: no tolerance contract exists for temperature keys.

    Snapping would silently claim two different temperatures are the same
    condition, which is a scientific statement a colour helper must not make.
    """

    from llm_behavior_lab.analysis.figures import _temperature_colours

    assert _temperature_colours([0.1200001, 0.24]) == _viridis_at(2)


# -- figures 23 and 24: the sanity check and the cross-partition diagnostic ---

from llm_behavior_lab.analysis.figures import (  # noqa: E402
    figure_category,
    plot_countsketch_fidelity,
    plot_cross_partition_geometry,
    save_figure,
)

SKETCH_K = 32


def _sketch_record(num_positions: int = 120, vocab: int = 20):
    """A record carrying sketches and both label vectors, as figure 24 needs."""

    generator = np.random.default_rng(24)
    rows = generator.normal(size=(num_positions, SKETCH_K))
    rows /= np.linalg.norm(rows, axis=1, keepdims=True)
    eligible = np.arange(2, vocab)
    targets = generator.choice(eligible, size=num_positions)
    greedy = generator.choice(eligible, size=num_positions)
    corpus = np.zeros(vocab, dtype=np.int64)
    corpus[eligible] = 7
    return InitializationExperimentRecord.build(
        corpus_counts=corpus,
        selected_target_counts=np.bincount(targets, minlength=vocab),
        greedy_counts=np.bincount(greedy, minlength=vocab)[None, :],
        nucleus_counts=np.bincount(greedy, minlength=vocab)[None, None, :],
        mean_predicted_probabilities=np.full((1, vocab), 1.0 / vocab),
        model_seeds=[1000],
        eligible_token_ids=eligible,
        metadata={
            "num_positions": num_positions,
            "analysis": {
                "num_positions": num_positions,
                "gradient_analysis": {
                    "enabled": True,
                    "initialization_index": 0,
                    "covers_all_positions": True,
                    "gradient_sketch": {"dimension": SKETCH_K, "seed": 1},
                },
            },
        },
        gradient_position_indices=np.arange(num_positions),
        gradient_position_target_ids=targets.astype(np.int64),
        gradient_position_greedy_ids=greedy.astype(np.int64),
        gradient_position_norms=np.ones(num_positions),
        gradient_position_sketches=rows,
    )


def _fidelity_report(num_gradients: int = 12):
    """A report shaped like the real artifact: matrices, not pair vectors.

    ``exact_cosines`` and ``production_cosines`` are square over the selected
    positions, and the production diagonal carries ``||S(g)||^2 / ||g||^2``
    rather than 1 -- that is what makes the projected-space comparator derivable
    without any sketch vectors, so the fixture has to reproduce it rather than
    plant ones on the diagonal.
    """

    generator = np.random.default_rng(23)
    exact_vectors = generator.normal(size=(num_gradients, 64))
    exact_vectors /= np.linalg.norm(exact_vectors, axis=1, keepdims=True)
    exact = exact_vectors @ exact_vectors.T

    # A sketch of the same vectors: inner products preserved in expectation,
    # divided by the exact (unit) norms, so the diagonal drifts off 1.
    sketch = exact_vectors + generator.normal(scale=0.05, size=(num_gradients, 64))
    production = sketch @ sketch.T

    return {
        "num_gradients": num_gradients,
        "gradient_dtype": "float32",
        "accumulation_dtype": "float64",
        "production_dimension": 512,
        "production_seed": 20240501,
        "map_semantics": "production",
        "exact_cosines": exact,
        "production_cosines": production,
        "sensitivity": [
            {"dimension": 128, "mean_absolute_error": 0.081, "rmse": 0.101},
            {"dimension": 256, "mean_absolute_error": 0.057, "rmse": 0.072},
            {"dimension": 512, "mean_absolute_error": 0.041, "rmse": 0.051},
            {"dimension": 1024, "mean_absolute_error": 0.029, "rmse": 0.036},
        ],
    }


def test_figure_twenty_three_stays_in_sanity_checks(tmp_path) -> None:
    """A methodological check must never be filed with the results."""

    written = plot_countsketch_fidelity(_fidelity_report(), tmp_path)

    assert len(written) == 1
    assert written[0].parent.name == "sanity_checks"
    assert written[0].suffix == ".svg"
    assert written[0].stat().st_size > 0
    assert figure_category(written[0].stem) == "sanity_checks"


def test_figure_twenty_three_does_not_clip_out_of_range_estimates(tmp_path) -> None:
    """The production estimator is not a cosine, and the figure must show that.

    A pair pushed past 1 is exactly the case the panel exists to reveal, so it
    has to survive into the drawn data rather than being silently pulled back
    to the bound.
    """

    report = _fidelity_report()
    report["production_cosines"] = np.asarray(report["production_cosines"]).copy()
    report["production_cosines"][0, 1] = 1.4
    report["production_cosines"][1, 0] = 1.4

    held = {}
    original = save_figure

    def capture(figure, *args, **kwargs):
        held["axes"] = figure.axes
        return original(figure, *args, **kwargs)

    import llm_behavior_lab.analysis.figures as module

    module.save_figure = capture
    try:
        module.plot_countsketch_fidelity(report, tmp_path)
    finally:
        module.save_figure = original

    drawn = [
        value
        for axes in held["axes"]
        for collection in axes.collections
        for value in np.asarray(collection.get_offsets())[:, 1]
    ]
    assert any(np.isclose(value, 1.4) for value in drawn)


def test_figure_twenty_four_renders_into_diagnostics(tmp_path) -> None:
    written = plot_cross_partition_geometry(
        _sketch_record(), tmp_path, display_classes=12
    )

    assert len(written) == 1
    assert written[0].parent.name == "diagnostics"
    assert written[0].stat().st_size > 0


def test_figure_twenty_four_selects_tokens_by_support_not_by_similarity() -> None:
    """Ranking cells by their value would choose the conclusion in advance.

    Built so the two orderings disagree: one token pair is given a strongly
    aligned cross similarity while holding little support, so a similarity-ranked
    selection would show it and a support-ranked selection must not.
    """

    from llm_behavior_lab.analysis.gradient_cross_partition import (
        cross_partition_matrix,
    )

    record = _sketch_record(num_positions=120, vocab=20)
    rows = np.asarray(record.gradient_position_sketches)
    targets = np.asarray(record.gradient_position_target_ids)
    greedy = np.asarray(record.gradient_position_greedy_ids)

    classes = np.union1d(np.unique(targets), np.unique(greedy))
    score = np.minimum(
        np.bincount(targets, minlength=classes.max() + 1)[classes],
        np.bincount(greedy, minlength=classes.max() + 1)[classes],
    )
    eligible = np.flatnonzero(score >= 2)
    by_support = set(
        classes[eligible[np.argsort(-score[eligible], kind="stable")][:6]].tolist()
    )

    matrix = cross_partition_matrix(rows, targets, greedy, classes, classes)["matrix"]
    diagonal = np.diagonal(matrix)
    finite = np.flatnonzero(np.isfinite(diagonal))
    by_similarity = set(
        classes[finite[np.argsort(-diagonal[finite])[:6]]].tolist()
    )

    assert by_support != by_similarity


# -- layout regressions for the two newest figures ---------------------------
#
# Structural rather than pixel-perfect: no baseline images, since the repository
# has no such infrastructure and brittle rasters would fail on every matplotlib
# bump. These catch the failures that actually happened -- a colourbar label
# landing on a neighbouring axis's label, and figure-level header/footer text
# drawn on top of the axes -- by checking reserved bands and artist counts.

from llm_behavior_lab.analysis.figures import (  # noqa: E402
    plot_nucleus_temperature_clustering,
)


def _held_figure(function, *args, **kwargs):
    """Render through the real path but keep the Figure for inspection."""

    import llm_behavior_lab.analysis.figures as module

    held = {}
    original = module.save_figure

    def capture(figure, *inner_args, **inner_kwargs):
        held["figure"] = figure
        return original(figure, *inner_args, **inner_kwargs)

    module.save_figure = capture
    try:
        written = function(*args, **kwargs)
    finally:
        module.save_figure = original
    return held["figure"], written


def _nucleus_result(record, temperatures=(0.12, 0.24, 0.60, 1.20)):
    from llm_behavior_lab.analysis.nucleus_clustering import nucleus_clustering

    greedy = np.asarray(record.gradient_position_greedy_ids)
    generator = np.random.default_rng(22)
    pool = np.unique(greedy)
    labels = np.stack([
        np.where(
            generator.random(greedy.size) < min(1.0, value / 1.4),
            generator.choice(pool, size=greedy.size),
            greedy,
        )
        for value in temperatures
    ])
    # T_g pinned to the canonical field, which is the only one this record
    # measured -- and the design figure 22 actually represents.
    return nucleus_clustering(
        record, labels, list(temperatures),
        loss_temperatures=[1.0] * len(temperatures),
        permutations=8, display_classes=10,
    )


def _data_axes(figure):
    """Axes carrying data, i.e. everything that is not a colourbar."""

    return [axes for axes in figure.axes if axes.get_label() != "<colorbar>"]


def _colourbar_axes(figure):
    return [axes for axes in figure.axes if axes.get_label() == "<colorbar>"]


def test_figure_twenty_two_has_its_four_panels_and_one_shared_colourbar(tmp_path) -> None:
    """One colourbar, because the two heatmaps share a scale on purpose."""

    record = _sketch_record()
    figure, written = _held_figure(
        plot_nucleus_temperature_clustering,
        _nucleus_result(record), tmp_path, record=record, display_classes=10,
    )

    # Four panels plus the twin axis on panel (b).
    assert len(_data_axes(figure)) == 5
    assert len(_colourbar_axes(figure)) == 1
    assert written[0].parent.name == "main"
    assert figure_category(written[0].stem) == "main"


def test_figure_twenty_four_has_its_three_panels_and_one_colourbar(tmp_path) -> None:
    figure, written = _held_figure(
        plot_cross_partition_geometry, _sketch_record(), tmp_path, display_classes=10
    )

    assert len(_data_axes(figure)) == 3
    assert len(_colourbar_axes(figure)) == 1
    assert written[0].parent.name == "diagnostics"
    assert figure_category(written[0].stem) == "diagnostics"


@pytest.mark.parametrize("which", ["figure22", "figure24"])
def test_the_header_and_footer_bands_are_reserved_not_overdrawn(which, tmp_path) -> None:
    """Figure-level text is invisible to automatic layout, so it gets a band.

    Relying on tight_layout here is what produced the collisions this checks
    for: every axes must stay clear of the top and bottom strips the header and
    footer are drawn into.
    """

    record = _sketch_record()
    if which == "figure22":
        figure, _ = _held_figure(
            plot_nucleus_temperature_clustering,
            _nucleus_result(record), tmp_path, record=record, display_classes=10,
        )
    else:
        figure, _ = _held_figure(
            plot_cross_partition_geometry, record, tmp_path, display_classes=10
        )

    # Only figure 24 carries a summary footer; figure 22 spends that space on a
    # second row of panels. The band is checked where it exists.
    footers = [text for text in figure.texts if text.get_position()[1] < 0.10]
    if which == "figure24":
        assert footers, "figure 24 should carry a figure-level summary footer"
    if footers:
        footer_top = max(text.get_position()[1] for text in footers)
        for axes in _data_axes(figure) + _colourbar_axes(figure):
            assert axes.get_position().y0 > footer_top, (
                f"{axes.get_title() or axes.get_label()} reaches into the footer"
            )

    for axes in _data_axes(figure) + _colourbar_axes(figure):
        assert axes.get_position().y1 < 0.94, "an axes reaches into the header band"


def test_the_two_heatmaps_do_not_overlap_the_colourbar(tmp_path) -> None:
    """The colourbar label used to be drawn over a neighbouring y-label."""

    record = _sketch_record()
    figure, _ = _held_figure(
        plot_nucleus_temperature_clustering,
        _nucleus_result(record), tmp_path, record=record, display_classes=10,
    )

    bar = _colourbar_axes(figure)[0].get_position()
    for axes in _data_axes(figure):
        box = axes.get_position()
        overlaps = box.x1 > bar.x0 and box.x0 < bar.x1 and box.y1 > bar.y0 and box.y0 < bar.y1
        assert not overlaps


def test_rendering_does_not_mutate_the_analysis_it_was_given(tmp_path) -> None:
    """A figure reads results; it must never write back into them."""

    record = _sketch_record()
    result = _nucleus_result(record)
    before = {
        "delta": [e["population"]["delta"] for e in result["by_temperature"]],
        "null": [e["null"]["delta_low"] for e in result["by_temperature"]],
        "classes": [e["display"]["classes"].copy() for e in result["by_temperature"]],
        "matrix": [e["display"]["matrix"].copy() for e in result["by_temperature"]],
        "sketches": np.asarray(record.gradient_position_sketches).copy(),
    }

    plot_nucleus_temperature_clustering(
        result, tmp_path, record=record, display_classes=10
    )
    plot_cross_partition_geometry(record, tmp_path, display_classes=10)

    for index, entry in enumerate(result["by_temperature"]):
        assert entry["population"]["delta"] == before["delta"][index]
        assert entry["null"]["delta_low"] == before["null"][index]
        assert np.array_equal(entry["display"]["classes"], before["classes"][index])
        assert np.allclose(
            entry["display"]["matrix"], before["matrix"][index], equal_nan=True
        )
    assert np.array_equal(
        np.asarray(record.gradient_position_sketches), before["sketches"]
    )


def test_the_heatmap_axes_keep_one_tick_label_per_displayed_class(tmp_path) -> None:
    """Thinning labels would silently misalign a reader's row lookup."""

    record = _sketch_record()
    result = _nucleus_result(record)
    figure, _ = _held_figure(
        plot_nucleus_temperature_clustering,
        result, tmp_path, record=record, display_classes=10,
    )

    heatmaps = [axes for axes in _data_axes(figure) if axes.images]
    assert len(heatmaps) == 2
    for axes, entry in zip(heatmaps, (result["by_temperature"][0],
                                      result["by_temperature"][-1])):
        expected = min(10, entry["display"]["classes"].size)
        assert len(axes.get_xticklabels()) == expected
        assert len(axes.get_yticklabels()) == expected


def test_figure_twenty_two_keeps_its_panel_c_title_between_the_two_rows(tmp_path) -> None:
    """The row label sits in the gap, touching neither row.

    It is figure-level text placed at a fixed height, so a change to the grid
    ratios can slide a panel underneath it without anything raising.
    """

    record = _sketch_record()
    figure, _ = _held_figure(
        plot_nucleus_temperature_clustering,
        _nucleus_result(record), tmp_path, record=record, display_classes=10,
    )

    row_label = [
        text for text in figure.texts
        if text.get_position()[1] < 0.6 and "coldest and hottest" in text.get_text()
    ]
    assert len(row_label) == 1
    height = row_label[0].get_position()[1]

    boxes = [axes.get_position() for axes in _data_axes(figure)]
    top_row = [box for box in boxes if box.y0 > height]
    bottom_row = [box for box in boxes if box.y1 < height]

    assert len(top_row) == 3          # trajectory, support, and support's twin
    assert len(bottom_row) == 2       # the two heatmaps
    assert min(box.y0 for box in top_row) - height > 0.01
    assert height - max(box.y1 for box in bottom_row) > 0.01


def test_figure_twenty_two_names_both_temperatures(tmp_path) -> None:
    """A bare "temperature" is ambiguous once T_s and T_g can differ.

    The x-axis carries T_s, so it says so, and the subtitle states what T_g was
    doing -- otherwise a reader cannot tell a controlled sweep from a matched one.
    """

    record = _sketch_record()
    result = _nucleus_result(record)
    result["loss_temperatures"] = np.ones(len(result["by_temperature"]))

    figure, _ = _held_figure(
        plot_nucleus_temperature_clustering,
        result, tmp_path, record=record, display_classes=10,
    )

    labels = [
        axes.get_xlabel() for axes in _data_axes(figure) if axes.get_xlabel()
    ]
    assert labels, "the temperature panels should label their x-axis"
    for label in labels:
        assert "T_s" in label

    subtitle = " ".join(text.get_text() for text in figure.texts)
    # The condition is stated in the title, not buried in a subtitle clause.
    assert "fixed gradient field" in subtitle
    assert "$T_g$ = 1" in subtitle
    assert "$T_s$" in subtitle


def test_figure_twenty_two_states_a_matched_sweep_as_matched(tmp_path) -> None:
    record = _sketch_record()
    # A varying-T_g result must carry its per-loss reference axis, so this is
    # built through the paired helper rather than by relabelling a pinned one.
    result = _paired_result(
        record, list(_nucleus_result(record)["sampling_temperatures"])
    )

    figure, _ = _held_figure(
        plot_nucleus_temperature_clustering,
        result, tmp_path, record=record, display_classes=10,
    )

    subtitle = " ".join(text.get_text() for text in figure.texts)
    assert "matched gradient field" in subtitle
    assert "$T_g$ = $T_s$" in subtitle


def test_figure_twenty_two_falls_back_for_a_pre_split_result(tmp_path) -> None:
    """A result carrying only the old key still renders, at T_g = 1."""

    record = _sketch_record()
    result = _nucleus_result(record)
    sampling = list(result["sampling_temperatures"])
    for key in ("sampling_temperatures", "loss_temperatures", "pairing"):
        result.pop(key, None)
    result["temperatures"] = tuple(sampling)

    figure, _ = _held_figure(
        plot_nucleus_temperature_clustering,
        result, tmp_path, record=record, display_classes=10,
    )

    subtitle = " ".join(text.get_text() for text in figure.texts)
    assert "fixed gradient field" in subtitle
    assert "$T_g$ = 1" in subtitle


# -- the two figure-22 conditions --------------------------------------------
#
# Control and matched are different experiments sharing one figure family. What
# has to hold is that they stay distinguishable and that each is read against
# the gradient field it was actually computed in.


def _paired_result(record, loss):
    """A figure-22 result with an explicit T_g per point and per-T_g references."""

    result = _nucleus_result(record)
    sampling = np.asarray(result["sampling_temperatures"], dtype=float)
    loss = np.asarray(loss, dtype=float)
    result["loss_temperatures"] = loss
    for entry, value in zip(result["by_temperature"], loss):
        entry["loss_temperature"] = float(value)
    # A distinct reference per T_g, so reading the wrong one is visible.
    result["references_by_loss_temperature"] = {
        float(value): {
            grouping: {
                "population": {"delta": base + 0.1 * index},
                "null": {"delta_low": -1e-4, "delta_high": 1e-4},
            }
            for grouping, base in (("target", 0.13), ("greedy", 0.11))
        }
        for index, value in enumerate(sorted(set(loss.tolist())))
    }
    return result


def _reference_series(figure, label_fragment):
    """The y-data of the reference artist whose legend label matches."""

    for axes in _data_axes(figure):
        for line in axes.get_lines():
            if label_fragment in (line.get_label() or ""):
                return np.asarray(line.get_ydata(), dtype=float)
    return None


def test_a_pinned_loss_temperature_draws_flat_reference_lines(tmp_path) -> None:
    """The control's established behaviour, unchanged."""

    record = _sketch_record()
    result = _paired_result(record, [1.0] * 4)

    figure, _ = _held_figure(
        plot_nucleus_temperature_clustering,
        result, tmp_path, record=record, display_classes=10,
    )

    # axhline artists carry two-point y-data at a constant height. Matched on
    # the exact reference labels, since the trajectory itself is also labelled
    # "... grouping" and is deliberately not flat.
    flat = [
        np.asarray(line.get_ydata(), dtype=float)
        for axes in _data_axes(figure)
        for line in axes.get_lines()
        if (line.get_label() or "") in ("target grouping", "greedy grouping")
    ]
    assert len(flat) == 2
    for series in flat:
        assert np.ptp(series) == 0.0


def test_a_varying_loss_temperature_draws_a_reference_per_pair(tmp_path) -> None:
    """The bug this phase exists to fix.

    With T_g varying, a flat line taken from one T_g compares most of the sweep
    against a geometry it was never computed in. Each point must be read against
    the reference measured at its own T_g.
    """

    record = _sketch_record()
    sampling = list(_nucleus_result(record)["sampling_temperatures"])
    result = _paired_result(record, sampling)          # matched: T_g = T_s

    figure, _ = _held_figure(
        plot_nucleus_temperature_clustering,
        result, tmp_path, record=record, display_classes=10,
    )

    for grouping, base in (("target", 0.13), ("greedy", 0.11)):
        series = _reference_series(figure, f"{grouping} grouping")
        assert series is not None, f"no {grouping} reference drawn"
        assert series.size == len(sampling)
        # Distinct per T_g, and in the order the pairs were requested.
        assert np.ptp(series) > 0.0
        expected = [base + 0.1 * index for index in range(len(sampling))]
        assert np.allclose(series, expected)


def test_the_matched_variant_writes_its_own_stem(tmp_path) -> None:
    """Both conditions coexist instead of overwriting one another."""

    record = _sketch_record()
    sampling = list(_nucleus_result(record)["sampling_temperatures"])

    control = plot_nucleus_temperature_clustering(
        _paired_result(record, [1.0] * len(sampling)), tmp_path,
        record=record, display_classes=10,
    )
    matched = plot_nucleus_temperature_clustering(
        _paired_result(record, sampling), tmp_path,
        record=record, display_classes=10, stem_suffix="_matched_TsTg",
    )

    assert control[0].name == "figure22_nucleus_temperature_clustering.svg"
    assert matched[0].name == (
        "figure22_nucleus_temperature_clustering_matched_TsTg.svg"
    )
    # Same family, so both belong in the same category directory.
    assert control[0].parent.name == matched[0].parent.name == "main"
    assert control[0] != matched[0]


def test_a_varying_sweep_refuses_a_missing_reference_temperature(tmp_path) -> None:
    """Snapping to the nearest measured T_g would be the wrong failure mode.

    A reference measured in a different gradient field describes different
    geometry, so a plotted T_g with no reference of its own must stop the figure
    rather than borrow the closest one.
    """

    record = _sketch_record()
    sampling = list(_nucleus_result(record)["sampling_temperatures"])
    result = _paired_result(record, sampling)

    # Drop one temperature from the reference axis, leaving the pairs intact.
    missing = sorted(result["references_by_loss_temperature"])[1]
    del result["references_by_loss_temperature"][missing]

    with pytest.raises(ValueError, match="No .* reference was measured"):
        plot_nucleus_temperature_clustering(
            result, tmp_path, record=record, display_classes=10
        )


def test_a_varying_sweep_refuses_a_result_with_no_reference_axis(tmp_path) -> None:
    """The scalar reference cannot stand in for six different gradient fields."""

    record = _sketch_record()
    sampling = list(_nucleus_result(record)["sampling_temperatures"])
    result = _paired_result(record, sampling)
    result["references_by_loss_temperature"] = {}

    with pytest.raises(ValueError, match="no per-loss reference axis"):
        plot_nucleus_temperature_clustering(
            result, tmp_path, record=record, display_classes=10
        )


def test_a_pinned_sweep_still_needs_no_reference_axis(tmp_path) -> None:
    """Historical compatibility: the control draws from the scalar as before."""

    record = _sketch_record()
    result = _paired_result(record, [1.0] * 4)
    result["references_by_loss_temperature"] = {}

    written = plot_nucleus_temperature_clustering(
        result, tmp_path, record=record, display_classes=10
    )

    assert written[0].name == "figure22_nucleus_temperature_clustering.svg"
