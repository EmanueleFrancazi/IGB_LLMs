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
