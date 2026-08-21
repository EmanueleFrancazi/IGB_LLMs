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
