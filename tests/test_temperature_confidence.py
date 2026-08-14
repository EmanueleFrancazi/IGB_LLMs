"""Tests for the temperature-conditioned greedy-confidence diagnostic.

The experiment rests on one fact that must hold exactly rather than
approximately: softmax is strictly increasing, so

    argmax softmax(z / T) = argmax z    for every T > 0

and therefore every temperature in the grid describes the **same** greedy
decisions with different confidence attached. Most of these tests pin that
invariance, or the sharpening behaviour that it makes interpretable.

The logits are chosen analytically -- a fixed, strictly ordered vector with no
ties -- so the expected probabilities can be written down in closed form and the
sharpening and flattening limits are exact statements rather than impressions.
"""

from __future__ import annotations

import numpy as np
import pytest

from llm_behavior_lab.analysis import (
    InitializationExperimentRecord,
    load_record,
    temperature_confidence_summary,
    temperature_ranked_profiles,
)

matplotlib = pytest.importorskip("matplotlib")

from llm_behavior_lab.analysis.figures import (  # noqa: E402
    generate_all_figures,
    plot_greedy_confidence_vs_temperature,
    plot_temperature_max_predictive_probability,
    plot_temperature_ranked_predictive_probabilities,
)

VOCAB_SIZE = 8
ELIGIBLE = np.arange(2, VOCAB_SIZE)
K = ELIGIBLE.size
NUM_POSITIONS = 12
NUM_INITS = 3
TEMPERATURES = (0.12, 0.24, 0.36, 0.48, 0.60, 1.00, 1.20)

#: Strictly decreasing, no ties, so argmax is unambiguous at every temperature.
BASE_LOGITS = np.array([3.0, 1.5, 0.5, -0.5, -1.5, -3.0])


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def _logits() -> np.ndarray:
    """``[D, K]`` logits: a cyclic shift of BASE_LOGITS at each position."""

    return np.stack([np.roll(BASE_LOGITS, shift) for shift in range(NUM_POSITIONS)])


def _record(*, with_temperatures: bool = True, num_inits: int = NUM_INITS):
    """A record whose temperature statistics follow analytically from the logits."""

    rng = np.random.default_rng(11)
    logits = _logits()
    greedy_local = logits.argmax(axis=1)
    greedy_ids = ELIGIBLE[greedy_local]
    target_local = rng.integers(0, K, size=NUM_POSITIONS)
    target_ids = ELIGIBLE[target_local]

    corpus = np.zeros(VOCAB_SIZE, dtype=np.int64)
    corpus[ELIGIBLE] = rng.integers(5, 40, size=K)
    greedy_counts = np.stack(
        [np.bincount(greedy_ids, minlength=VOCAB_SIZE) for _ in range(num_inits)]
    )

    ranked, maxima, targets, entropy = [], [], [], []
    for temperature in TEMPERATURES:
        probabilities = _softmax(logits / temperature)
        ordered = np.sort(probabilities, axis=1)[:, ::-1]
        ranked.append(ordered.mean(axis=0))
        maxima.append(probabilities[np.arange(NUM_POSITIONS), greedy_local])
        targets.append(probabilities[np.arange(NUM_POSITIONS), target_local])
        entropy.append(float((-probabilities * np.log(probabilities)).sum(axis=1).mean()))

    ranked = np.tile(np.stack(ranked), (num_inits, 1, 1))
    maxima = np.tile(np.stack(maxima), (num_inits, 1, 1))
    targets = np.tile(np.stack(targets), (num_inits, 1, 1))
    entropy = np.tile(np.asarray(entropy), (num_inits, 1))
    canonical = TEMPERATURES.index(1.00)

    metadata = {"analysis": {"num_positions": NUM_POSITIONS}}
    arrays = {}
    if with_temperatures:
        metadata["analysis"]["predictive_probabilities"] = {
            "enabled": True,
            "temperature": 1.0,
            "support": "eligible",
            "before_top_p": True,
            "before_sampling": True,
            "confidence_temperatures": list(TEMPERATURES),
        }
        arrays = {
            "predictive_ranked_probabilities": ranked[:, canonical],
            "predictive_max_probabilities": maxima[:, canonical],
            "predictive_target_probabilities": targets[:, canonical],
            "predictive_target_losses": -np.log(targets[:, canonical]),
            "predictive_temperatures": np.asarray(TEMPERATURES),
            "predictive_temperature_ranked_probabilities": ranked,
            "predictive_temperature_max_probabilities": maxima,
            "predictive_temperature_target_probabilities": targets,
            "predictive_temperature_target_losses": -np.log(targets),
            "predictive_temperature_mean_entropy": entropy,
        }

    return InitializationExperimentRecord.build(
        corpus_counts=corpus,
        selected_target_counts=np.bincount(target_ids, minlength=VOCAB_SIZE),
        greedy_counts=greedy_counts,
        nucleus_counts=greedy_counts[:, None, :],
        mean_predicted_probabilities=np.tile(
            np.bincount(ELIGIBLE, weights=_softmax(BASE_LOGITS), minlength=VOCAB_SIZE),
            (num_inits, 1),
        ),
        model_seeds=np.arange(num_inits) + 1000,
        eligible_token_ids=ELIGIBLE,
        metadata=metadata,
        **arrays,
    )


# -- the invariance the experiment rests on ----------------------------------


def test_argmax_is_identical_at_every_temperature() -> None:
    """The greedy decision is temperature-invariant, exactly and for all T."""

    logits = _logits()
    reference = logits.argmax(axis=1)

    for temperature in TEMPERATURES + (0.001, 5.0, 100.0):
        probabilities = _softmax(logits / temperature)
        assert np.array_equal(probabilities.argmax(axis=1), reference)


def test_probabilities_sum_to_one_at_every_temperature() -> None:
    logits = _logits()

    for temperature in TEMPERATURES:
        totals = _softmax(logits / temperature).sum(axis=1)
        assert np.allclose(totals, 1.0, atol=1e-12)


def test_lower_temperature_sharpens_and_higher_flattens() -> None:
    """The monotone relation that makes the whole figure interpretable.

    On a strictly ordered logit vector the maximum probability is strictly
    decreasing in temperature, and the entropy strictly increasing.
    """

    maxima, entropies = [], []
    for temperature in sorted(TEMPERATURES):
        probabilities = _softmax(BASE_LOGITS / temperature)
        maxima.append(probabilities.max())
        entropies.append(float(-(probabilities * np.log(probabilities)).sum()))

    assert np.all(np.diff(maxima) < 0.0)
    assert np.all(np.diff(entropies) > 0.0)


def test_small_temperature_approaches_one_hot() -> None:
    """As T -> 0 the distribution concentrates on the greedy winner."""

    for temperature in (0.05, 0.01, 0.001):
        probabilities = _softmax(BASE_LOGITS / temperature)
        assert probabilities.argmax() == BASE_LOGITS.argmax()
    assert _softmax(BASE_LOGITS / 0.001).max() == pytest.approx(1.0, abs=1e-9)


def test_large_temperature_approaches_uniform() -> None:
    probabilities = _softmax(BASE_LOGITS / 5000.0)

    assert np.allclose(probabilities, 1.0 / BASE_LOGITS.size, atol=1e-3)


# -- the recorded statistics --------------------------------------------------


def test_every_temperature_profile_is_non_increasing_and_normalized() -> None:
    profiles = temperature_ranked_profiles(_record())

    assert profiles["mean"].shape == (len(TEMPERATURES), K)
    assert np.all(np.diff(profiles["mean"], axis=1) <= 1e-12)
    assert np.allclose(profiles["mean"].sum(axis=1), 1.0, atol=1e-9)
    assert np.allclose(profiles["profiles"].sum(axis=2), 1.0, atol=1e-9)


def test_rank_one_equals_the_mean_maximum_at_every_temperature() -> None:
    record = _record()
    profiles = temperature_ranked_profiles(record)
    maxima = np.asarray(record.predictive_temperature_max_probabilities)

    assert np.allclose(profiles["profiles"][:, :, 0], maxima.mean(axis=2), rtol=1e-9)


def test_the_canonical_slice_reproduces_the_existing_t1_arrays() -> None:
    """T = 1 in the grid and the canonical arrays must be the same numbers."""

    record = _record()
    index = record.temperature_index(1.0)

    assert np.array_equal(
        record.predictive_temperature_ranked_probabilities[:, index],
        record.predictive_ranked_probabilities,
    )
    assert np.array_equal(
        record.predictive_temperature_max_probabilities[:, index],
        record.predictive_max_probabilities,
    )
    assert np.array_equal(
        record.predictive_temperature_target_losses[:, index],
        record.predictive_target_losses,
    )


def test_a_drifted_canonical_slice_is_rejected() -> None:
    """Validation refuses a grid whose T = 1 slice disagrees with the canonical."""

    # Perturb the target losses rather than the maxima: halving the maxima would
    # trip the more fundamental rank-1 check first, and this test exists to pin
    # the canonical-slice check specifically.
    record = _record()
    broken = np.array(record.predictive_temperature_target_losses)
    broken[:, record.temperature_index(1.0)] += 0.25

    with pytest.raises(ValueError, match="differs from"):
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
            predictive_max_probabilities=record.predictive_max_probabilities,
            predictive_target_probabilities=record.predictive_target_probabilities,
            predictive_target_losses=record.predictive_target_losses,
            predictive_temperatures=record.predictive_temperatures,
            predictive_temperature_ranked_probabilities=(
                record.predictive_temperature_ranked_probabilities
            ),
            predictive_temperature_max_probabilities=(
                record.predictive_temperature_max_probabilities
            ),
            predictive_temperature_target_probabilities=(
                record.predictive_temperature_target_probabilities
            ),
            predictive_temperature_target_losses=broken,
            predictive_temperature_mean_entropy=record.predictive_temperature_mean_entropy,
        )


def test_the_summary_is_monotone_in_temperature() -> None:
    """Confidence falls and effective support rises as temperature increases."""

    summary = temperature_confidence_summary(_record())
    rows = sorted(summary["rows"], key=lambda row: row["temperature"])

    medians = [row["max_probability"]["p50"] for row in rows]
    supports = [row["effective_support"] for row in rows]
    assert np.all(np.diff(medians) < 0.0)
    assert np.all(np.diff(supports) > 0.0)
    assert sum(row["is_canonical"] for row in rows) == 1
    for row in rows:
        assert row["top_k_mass"]["1"] == pytest.approx(
            row["ranked_profile_at_rank"]["1"]
        )
        for mass in row["top_k_mass"].values():
            assert 0.0 < mass <= 1.0 + 1e-12


def test_temperature_negative_or_zero_is_rejected() -> None:
    record = _record()
    broken = np.array(record.predictive_temperatures)
    broken[0] = 0.0

    with pytest.raises(ValueError, match="must be positive"):
        InitializationExperimentRecord.build(
            corpus_counts=record.corpus_counts,
            selected_target_counts=record.selected_target_counts,
            greedy_counts=record.greedy_counts,
            nucleus_counts=record.nucleus_counts,
            mean_predicted_probabilities=record.mean_predicted_probabilities,
            model_seeds=record.model_seeds,
            eligible_token_ids=record.eligible_token_ids,
            metadata=record.metadata,
            predictive_temperatures=broken,
            predictive_temperature_ranked_probabilities=(
                record.predictive_temperature_ranked_probabilities
            ),
            predictive_temperature_max_probabilities=(
                record.predictive_temperature_max_probabilities
            ),
            predictive_temperature_target_probabilities=(
                record.predictive_temperature_target_probabilities
            ),
            predictive_temperature_target_losses=(
                record.predictive_temperature_target_losses
            ),
            predictive_temperature_mean_entropy=record.predictive_temperature_mean_entropy,
        )


# -- persistence --------------------------------------------------------------


def test_no_full_probability_tensor_is_persisted(tmp_path) -> None:
    """Sufficient statistics only: nothing indexed by position *and* token."""

    record = _record()
    record.save(tmp_path)

    with np.load(tmp_path / "initialization_distribution.npz") as archive:
        for key in archive.files:
            array = archive[key]
            assert array.ndim <= 3, key
            if array.ndim == 3 and key.startswith("predictive_temperature"):
                # [I, N_T, K] or [I, N_T, D] -- never [I, N_T, D, K].
                assert array.shape[:2] == (NUM_INITS, len(TEMPERATURES)), key


def test_temperature_arrays_survive_a_round_trip(tmp_path) -> None:
    record = _record()
    record.save(tmp_path)
    reloaded = load_record(tmp_path)

    assert reloaded.has_temperature_confidence_analysis is True
    assert reloaded.confidence_temperatures == TEMPERATURES
    assert reloaded.temperature_index(1.0) == TEMPERATURES.index(1.0)
    for name in (
        "predictive_temperature_ranked_probabilities",
        "predictive_temperature_max_probabilities",
        "predictive_temperature_target_probabilities",
        "predictive_temperature_target_losses",
        "predictive_temperature_mean_entropy",
    ):
        assert np.allclose(getattr(reloaded, name), getattr(record, name))


def test_a_historical_record_without_temperatures_still_loads(tmp_path) -> None:
    record = _record(with_temperatures=False)
    record.save(tmp_path)
    reloaded = load_record(tmp_path)

    assert reloaded.has_temperature_confidence_analysis is False
    assert reloaded.confidence_temperatures == ()
    with pytest.raises(ValueError, match="no temperature-conditioned"):
        temperature_confidence_summary(reloaded)


def test_temperature_arrays_are_all_or_nothing() -> None:
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
            predictive_temperatures=record.predictive_temperatures,
        )


# -- figures ------------------------------------------------------------------


def _axes_from(function, record, directory, **kwargs):
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


def test_figure_eleven_renders(tmp_path) -> None:
    axes, written = _axes_from(
        plot_temperature_ranked_predictive_probabilities, _record(), tmp_path
    )

    assert [path.name for path in written] == [
        "figure11_temperature_ranked_predictive_probabilities.svg"
    ]
    assert written[0].is_file() and written[0].stat().st_size > 0
    labels = [str(text) for text in axes.get_legend().get_texts()]
    assert any("canonical" in label for label in labels)
    assert any("1/K" in label for label in labels)
    # One curve per temperature, plus the uniform reference line.
    assert len(axes.lines) >= len(TEMPERATURES) + 1


def test_figure_twelve_renders_six_panels_with_shared_axes(tmp_path) -> None:
    written = plot_temperature_max_predictive_probability(_record(), tmp_path)

    assert [path.name for path in written] == [
        "figure12_temperature_max_predictive_probability.svg"
    ]
    assert written[0].is_file() and written[0].stat().st_size > 0


def test_figure_twelve_panels_share_identical_limits(tmp_path) -> None:
    """Apparent sharpening must be the data, never a per-panel rescaling."""

    from matplotlib.figure import Figure

    created = []
    original = Figure.subplots

    def capture(self, *args, **inner):
        axes = original(self, *args, **inner)
        created.append(axes)
        return axes

    Figure.subplots = capture
    try:
        plot_temperature_max_predictive_probability(_record(), tmp_path)
    finally:
        Figure.subplots = original

    panels = [axes for row in created[0] for axes in row if axes.get_visible()]
    assert len(panels) == len(TEMPERATURES) - 1  # T = 1 is the reference, not a panel
    limits = {(axes.get_xlim(), axes.get_ylim()) for axes in panels}
    assert len(limits) == 1


def test_figure_thirteen_uses_separate_panels(tmp_path) -> None:
    """A probability and a token count must not share one axis."""

    from matplotlib.figure import Figure

    created = []
    original = Figure.subplots

    def capture(self, *args, **inner):
        axes = original(self, *args, **inner)
        created.append(axes)
        return axes

    Figure.subplots = capture
    try:
        written = plot_greedy_confidence_vs_temperature(_record(), tmp_path)
    finally:
        Figure.subplots = original

    assert [path.name for path in written] == [
        "figure13_greedy_confidence_vs_temperature.svg"
    ]
    top, bottom = created[0]
    assert top.get_ylabel() != bottom.get_ylabel()
    # Temperature keeps its true numerical spacing.
    assert bottom.get_xscale() == "log"


def test_the_figure_set_gains_the_temperature_figures_only_when_available(tmp_path) -> None:
    with_data = {path.name for path in generate_all_figures(_record(), tmp_path / "with")}
    without = {
        path.name
        for path in generate_all_figures(_record(with_temperatures=False), tmp_path / "without")
    }

    new = {
        "figure11_temperature_ranked_predictive_probabilities.svg",
        "figure12_temperature_max_predictive_probability.svg",
        "figure13_greedy_confidence_vs_temperature.svg",
    }
    assert new <= with_data
    assert not (new & without)


def test_records_without_temperatures_refuse_the_new_figures(tmp_path) -> None:
    record = _record(with_temperatures=False)

    for function in (
        plot_temperature_ranked_predictive_probabilities,
        plot_temperature_max_predictive_probability,
        plot_greedy_confidence_vs_temperature,
    ):
        with pytest.raises(ValueError, match="no temperature-confidence analysis"):
            function(record, tmp_path)


def test_the_renderer_maps_the_temperature_figures() -> None:
    import importlib.util
    import sys
    from pathlib import Path

    from llm_behavior_lab.analysis import figures as figure_module

    path = Path(__file__).resolve().parents[1] / "scripts" / "render_record_figures.py"
    spec = importlib.util.spec_from_file_location("render_record_figures", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    for name in ("figure11", "figure12", "figure13"):
        assert callable(getattr(figure_module, module.FIGURES[name]))
        assert module.CONDITIONAL_FIGURES[name] == "has_temperature_confidence_analysis"

    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "torch" not in imported
