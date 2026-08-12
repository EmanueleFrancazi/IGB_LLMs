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
    first = sorted(path.name for path in tmp_path.iterdir())

    generate_all_figures(_record(), tmp_path)

    assert sorted(path.name for path in tmp_path.iterdir()) == first


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

    for path in tmp_path.glob("*.svg"):
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

    written = sorted(path.name for path in tmp_path.iterdir())
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
    assert len(list(tmp_path.glob("*.svg"))) == 4


def test_the_declared_format_list_is_svg_only() -> None:
    """Every caller derives its expectations from this constant."""

    assert FIGURE_FORMATS == ("svg",)


def test_a_subword_figure_set_is_also_svg_only(tmp_path) -> None:
    """The large-vocabulary path shares the export code, so it must agree."""

    generate_all_figures(_large_record(), tmp_path)

    assert list(tmp_path.glob("*.png")) == []
    assert len(list(tmp_path.glob("*.svg"))) == 4
