"""Tests for the raw predictive-probability diagnostics and figures 8 and 9.

The statistic that is easiest to get wrong is the order of two operations, so
that is what most of these pin. Ranking must happen **within each position
first**, and only the ranked values may be averaged across positions. Ranking an
average instead answers the question figure 1 already answers, and on many inputs
the two produce similar-looking curves -- which is exactly why it needs a test
that can tell them apart rather than a plausibility check.

The fixture is built so the correct answer is known in closed form: every
position uses a permutation of one fixed probability vector, so the rank-first
profile is that vector sorted, while the average-first profile is uniform.
"""

from __future__ import annotations

import numpy as np
import pytest

from llm_behavior_lab.analysis import (
    InitializationExperimentRecord,
    load_record,
    max_probability_summary,
    predictive_probability_summary,
    ranked_probability_profile,
    target_probability_summary,
)

matplotlib = pytest.importorskip("matplotlib")

from llm_behavior_lab.analysis.figures import (  # noqa: E402
    generate_all_figures,
    plot_max_predictive_probability,
    plot_ranked_predictive_probabilities,
)

VOCAB_SIZE = 8
ELIGIBLE = np.arange(2, VOCAB_SIZE)  # K = 6
K = ELIGIBLE.size
NUM_POSITIONS = 12
NUM_INITS = 3

#: One fixed descending probability vector, permuted differently at every
#: position. Rank-first therefore recovers it exactly; average-first cannot.
BASE_PROFILE = np.array([0.5, 0.2, 0.15, 0.1, 0.04, 0.01])


def _permuted_probabilities(seed: int = 3) -> np.ndarray:
    """``[D, K]`` probabilities: a cyclic shift of BASE_PROFILE at each position.

    Cyclic shifts rather than random permutations, and ``D`` a multiple of ``K``,
    so every token takes every value exactly ``D / K`` times. The average-first
    profile is then *exactly* uniform rather than approximately so, which is what
    lets the rank-order test assert an equality instead of a tolerance.
    """

    del seed  # deterministic by construction; kept for call-site symmetry
    return np.stack([np.roll(BASE_PROFILE, shift) for shift in range(NUM_POSITIONS)])


def _record(
    *,
    with_probabilities: bool = True,
    num_inits: int = NUM_INITS,
    seed: int = 3,
):
    """A record whose ranked profile is known in closed form."""

    rng = np.random.default_rng(seed)
    probabilities = _permuted_probabilities(seed)
    greedy_local = probabilities.argmax(axis=1)
    greedy_ids = ELIGIBLE[greedy_local]
    target_local = rng.integers(0, K, size=NUM_POSITIONS)
    target_ids = ELIGIBLE[target_local]

    corpus = np.zeros(VOCAB_SIZE, dtype=np.int64)
    corpus[ELIGIBLE] = rng.integers(3, 30, size=K)
    greedy_counts = np.stack(
        [np.bincount(greedy_ids, minlength=VOCAB_SIZE) for _ in range(num_inits)]
    )

    metadata = {"analysis": {"num_positions": NUM_POSITIONS}}
    arrays = {}
    if with_probabilities:
        metadata["analysis"]["predictive_probabilities"] = {
            "enabled": True,
            "from_raw_logits": True,
            "support": "eligible",
            "temperature": 1.0,
            "before_top_p": True,
            "before_sampling": True,
        }
        ranked = np.tile(np.sort(BASE_PROFILE)[::-1], (num_inits, 1))
        maxima = np.tile(probabilities.max(axis=1), (num_inits, 1))
        targets = np.tile(probabilities[np.arange(NUM_POSITIONS), target_local], (num_inits, 1))
        arrays = {
            "predictive_ranked_probabilities": ranked,
            "predictive_max_probabilities": maxima,
            "predictive_target_probabilities": targets,
            "predictive_target_losses": -np.log(targets),
        }

    return InitializationExperimentRecord.build(
        corpus_counts=corpus,
        selected_target_counts=np.bincount(target_ids, minlength=VOCAB_SIZE),
        greedy_counts=greedy_counts,
        nucleus_counts=greedy_counts[:, None, :],
        mean_predicted_probabilities=np.tile(
            np.bincount(ELIGIBLE, weights=BASE_PROFILE, minlength=VOCAB_SIZE), (num_inits, 1)
        ),
        model_seeds=np.arange(num_inits) + 1000,
        eligible_token_ids=ELIGIBLE,
        metadata=metadata,
        **arrays,
    )


# -- rank-before-average, the property the whole analysis rests on -----------


def test_rank_first_then_average_is_not_average_then_rank() -> None:
    """The two orders give provably different answers on this fixture.

    Every position holds a permutation of the same vector, so ranking within
    each position and averaging recovers that vector exactly, while averaging
    the per-token probabilities first gives a flat 1/K at every token and would
    rank to a constant profile. Any implementation that confuses the two fails
    here by a wide margin rather than subtly.
    """

    probabilities = _permuted_probabilities()

    rank_first = np.sort(probabilities, axis=1)[:, ::-1].mean(axis=0)
    average_first = np.sort(probabilities.mean(axis=0))[::-1]

    assert np.allclose(rank_first, np.sort(BASE_PROFILE)[::-1])
    assert np.allclose(average_first, np.full(K, 1.0 / K))
    assert not np.allclose(rank_first, average_first)

    profile = ranked_probability_profile(_record())
    assert np.allclose(profile["mean"], rank_first)


def test_the_profile_is_non_increasing() -> None:
    profile = ranked_probability_profile(_record())

    assert np.all(np.diff(profile["mean"]) <= 1e-12)
    for row in profile["profiles"]:
        assert np.all(np.diff(row) <= 1e-12)


def test_the_profile_sums_to_one() -> None:
    profile = ranked_probability_profile(_record())

    assert profile["mean"].sum() == pytest.approx(1.0, abs=1e-9)
    assert np.allclose(profile["profiles"].sum(axis=1), 1.0, atol=1e-9)


def test_rank_one_equals_the_mean_stored_maximum() -> None:
    """The invariant tying the profile to the per-position statistics."""

    record = _record()
    profile = ranked_probability_profile(record)
    stored = np.asarray(record.predictive_max_probabilities)

    assert np.allclose(profile["profiles"][:, 0], stored.mean(axis=1), rtol=1e-9)
    assert profile["mean"][0] == pytest.approx(BASE_PROFILE.max())


def test_a_broken_invariant_is_rejected_by_the_record() -> None:
    """Validation refuses a profile that disagrees with its own maxima."""

    with pytest.raises(ValueError, match="Rank 1 of each ranked profile"):
        record = _record()
        InitializationExperimentRecord.build(
            corpus_counts=record.corpus_counts,
            selected_target_counts=record.selected_target_counts,
            greedy_counts=record.greedy_counts,
            nucleus_counts=record.nucleus_counts,
            mean_predicted_probabilities=record.mean_predicted_probabilities,
            model_seeds=record.model_seeds,
            eligible_token_ids=record.eligible_token_ids,
            metadata=record.metadata,
            predictive_ranked_probabilities=record.predictive_ranked_probabilities,
            predictive_max_probabilities=record.predictive_max_probabilities * 0.5,
            predictive_target_probabilities=record.predictive_target_probabilities,
            predictive_target_losses=record.predictive_target_losses,
        )


def test_an_unsorted_profile_is_rejected() -> None:
    record = _record()
    broken = np.array(record.predictive_ranked_probabilities)
    broken[:, [0, 1]] = broken[:, [1, 0]]

    with pytest.raises(ValueError, match="non-increasing"):
        InitializationExperimentRecord.build(
            corpus_counts=record.corpus_counts,
            selected_target_counts=record.selected_target_counts,
            greedy_counts=record.greedy_counts,
            nucleus_counts=record.nucleus_counts,
            mean_predicted_probabilities=record.mean_predicted_probabilities,
            model_seeds=record.model_seeds,
            eligible_token_ids=record.eligible_token_ids,
            metadata=record.metadata,
            predictive_ranked_probabilities=broken,
            predictive_max_probabilities=record.predictive_max_probabilities,
            predictive_target_probabilities=record.predictive_target_probabilities,
            predictive_target_losses=record.predictive_target_losses,
        )


# -- per-position statistics -------------------------------------------------


def test_maximum_probabilities_lie_in_the_unit_interval() -> None:
    values = np.asarray(_record().predictive_max_probabilities)

    assert values.min() >= 0.0 and values.max() <= 1.0
    assert np.allclose(values, BASE_PROFILE.max())


def test_the_loss_is_the_negative_log_of_the_target_probability() -> None:
    record = _record()

    assert np.allclose(
        record.predictive_target_losses,
        -np.log(record.predictive_target_probabilities),
        rtol=1e-12,
    )


def test_the_summary_reports_everything_the_figures_need() -> None:
    summary = predictive_probability_summary(_record())

    assert summary["eligible_vocab_size"] == K
    assert summary["uniform_probability"] == pytest.approx(1.0 / K)
    assert summary["ranked_profile_at_rank"]["1"] == pytest.approx(BASE_PROFILE.max())
    assert summary["rank1_over_uniform"] == pytest.approx(0.5 * K)
    assert summary["profile_dynamic_range"] == pytest.approx(0.5 / 0.01)
    for key in ("p00", "p50", "p100", "mean"):
        assert key in summary["max_probability"]["pooled"]
    assert summary["target_probability"]["uniform_loss"] == pytest.approx(np.log(K))


def test_target_and_maximum_summaries_answer_different_questions() -> None:
    """p_max is confidence in the preferred token; p_target is mass on the truth."""

    record = _record()
    maximum = max_probability_summary(record)
    target = target_probability_summary(record)

    assert maximum["pooled"]["mean"] == pytest.approx(BASE_PROFILE.max())
    assert target["probability"]["mean"] < maximum["pooled"]["mean"]


# -- persistence -------------------------------------------------------------


def test_no_full_probability_tensor_is_persisted(tmp_path) -> None:
    """Only sufficient statistics. The [I, D, K] tensor must never be stored."""

    record = _record()
    record.save(tmp_path)

    with np.load(tmp_path / "initialization_distribution.npz") as archive:
        for key in archive.files:
            assert archive[key].ndim <= 3, key
            if key.startswith("predictive_"):
                expected = (
                    (NUM_INITS, K) if key.endswith("ranked_probabilities")
                    else (NUM_INITS, NUM_POSITIONS)
                )
                assert archive[key].shape == expected, key


def test_the_new_fields_survive_a_round_trip(tmp_path) -> None:
    record = _record()
    record.save(tmp_path)
    reloaded = load_record(tmp_path)

    assert reloaded.has_predictive_probability_analysis is True
    for name in (
        "predictive_ranked_probabilities",
        "predictive_max_probabilities",
        "predictive_target_probabilities",
        "predictive_target_losses",
    ):
        assert np.allclose(getattr(reloaded, name), getattr(record, name))
    assert reloaded.predictive_probability_protocol["temperature"] == 1.0
    assert reloaded.predictive_probability_protocol["support"] == "eligible"


def test_a_historical_record_without_probabilities_still_loads(tmp_path) -> None:
    record = _record(with_probabilities=False)
    record.save(tmp_path)
    reloaded = load_record(tmp_path)

    assert reloaded.has_predictive_probability_analysis is False
    assert reloaded.predictive_ranked_probabilities is None
    assert reloaded.predictive_probability_protocol == {}
    with pytest.raises(ValueError, match="no raw predictive-probability analysis"):
        ranked_probability_profile(reloaded)


def test_probability_arrays_are_all_or_nothing() -> None:
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
            predictive_max_probabilities=record.predictive_max_probabilities,
        )


# -- figures -----------------------------------------------------------------


def _axes_from(function, record, directory, **kwargs):
    """Draw, capturing the axes so the scale choices can be inspected."""

    from matplotlib.figure import Figure

    created = []
    original = Figure.subplots

    def capture(self, *args, **inner):
        axes = original(self, *args, **inner)
        created.append(axes)
        return axes

    Figure.subplots = capture
    try:
        written = function(record, directory, **kwargs)
    finally:
        Figure.subplots = original
    return created[0], written


def test_figure_eight_renders_from_a_record(tmp_path) -> None:
    axes, written = _axes_from(plot_ranked_predictive_probabilities, _record(), tmp_path)

    assert [path.name for path in written] == ["figure8_ranked_predictive_probabilities.svg"]
    assert written[0].is_file() and written[0].stat().st_size > 0
    # The uniform reference must be present and labelled.
    assert any("1/K" in str(text) for text in axes.get_legend().get_texts())


def test_figure_nine_renders_from_a_record(tmp_path) -> None:
    axes, written = _axes_from(plot_max_predictive_probability, _record(), tmp_path)

    assert [path.name for path in written] == [
        "figure9_max_predictive_probability_distribution.svg"
    ]
    assert written[0].is_file() and written[0].stat().st_size > 0
    assert axes.get_ylim() == (0.0, 1.0)
    labels = [str(text) for text in axes.get_legend().get_texts()]
    assert any("per initialization" in label for label in labels)
    assert any("pooled" in label for label in labels)


def test_figure_scales_follow_the_data_and_can_be_pinned(tmp_path) -> None:
    """Auto follows the dynamic range; an explicit choice always wins."""

    record = _record()  # spans 0.5 down to 0.01, i.e. well over a decade
    axes, _ = _axes_from(plot_ranked_predictive_probabilities, record, tmp_path)
    assert axes.get_yscale() == "log"

    axes, _ = _axes_from(
        plot_ranked_predictive_probabilities, record, tmp_path, y_scale="linear"
    )
    assert axes.get_yscale() == "linear"

    # p_max is constant on this fixture, so its range is a single value.
    axes, _ = _axes_from(plot_max_predictive_probability, record, tmp_path)
    assert axes.get_xscale() == "linear"
    axes, _ = _axes_from(plot_max_predictive_probability, record, tmp_path, x_scale="log")
    assert axes.get_xscale() == "log"


def test_the_figure_set_includes_the_new_figures_only_with_probability_data(tmp_path) -> None:
    with_data = {path.name for path in generate_all_figures(_record(), tmp_path / "with")}
    without = {
        path.name
        for path in generate_all_figures(_record(with_probabilities=False), tmp_path / "without")
    }

    new = {
        "figure8_ranked_predictive_probabilities.svg",
        "figure9_max_predictive_probability_distribution.svg",
    }
    assert new <= with_data
    assert not (new & without)
    assert with_data - new == without


def test_records_without_probabilities_refuse_the_new_figures(tmp_path) -> None:
    record = _record(with_probabilities=False)

    for function in (plot_ranked_predictive_probabilities, plot_max_predictive_probability):
        with pytest.raises(ValueError, match="no raw predictive-probability analysis"):
            function(record, tmp_path)


def test_a_single_initialization_still_renders(tmp_path) -> None:
    """No band to draw, but the figure must not fall over."""

    record = _record(num_inits=1)

    plot_ranked_predictive_probabilities(record, tmp_path)
    plot_max_predictive_probability(record, tmp_path)


# -- record-only rendering ---------------------------------------------------


def test_the_renderer_maps_all_three_recent_figures() -> None:
    import importlib.util
    import sys
    from pathlib import Path

    from llm_behavior_lab.analysis import figures as figure_module

    path = Path(__file__).resolve().parents[1] / "scripts" / "render_record_figures.py"
    spec = importlib.util.spec_from_file_location("render_record_figures", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    assert module.FIGURES["figure8"] == "plot_ranked_predictive_probabilities"
    assert module.FIGURES["figure9"] == "plot_max_predictive_probability"
    assert module.FIGURES["figure10"] == "plot_gradient_vs_guess_bias"
    for name, function_name in module.FIGURES.items():
        assert callable(getattr(figure_module, function_name)), name
    assert module.CONDITIONAL_FIGURES["figure8"] == "has_predictive_probability_analysis"
    assert module.CONDITIONAL_FIGURES["figure10"] == "has_position_gradients"
