"""Tests for figure 14: average at fixed token identity first, then rank.

Figure 11 and figure 14 differ only in the order of two operations, and on real
data they can look similar, so the fixture is built to separate them *exactly*
rather than plausibly. Every position holds a cyclic shift of one fixed
probability vector, with the number of positions a multiple of the support size.
Then:

* ranking inside each position and averaging recovers that vector exactly;
* averaging at fixed identity gives a perfectly uniform vector, which ranks flat.

That is the textbook Case A -- concentrated individual predictions, no persistent
identity preference -- and any implementation that confuses the two orders fails
here by the whole distance between a peaked vector and a flat one.
"""

from __future__ import annotations

import numpy as np
import pytest

from llm_behavior_lab.analysis import (
    InitializationExperimentRecord,
    cumulative_order_comparison,
    load_record,
    ranked_mean_token_probabilities,
    temperature_ranked_profiles,
)

matplotlib = pytest.importorskip("matplotlib")

from llm_behavior_lab.analysis.figures import (  # noqa: E402
    generate_all_figures,
    plot_ranked_mean_token_probabilities,
)

VOCAB_SIZE = 8
ELIGIBLE = np.arange(2, VOCAB_SIZE)
K = ELIGIBLE.size
NUM_POSITIONS = 12
NUM_INITS = 3
TEMPERATURES = (0.12, 0.24, 0.36, 0.48, 0.60, 1.00, 1.20)
BASE = np.array([0.5, 0.2, 0.15, 0.1, 0.04, 0.01])


def _cyclic() -> np.ndarray:
    """``[D, K]``: a cyclic shift of BASE at each position, D a multiple of K."""

    return np.stack([np.roll(BASE, shift) for shift in range(NUM_POSITIONS)])


def _record(*, with_tokens: bool = True, num_inits: int = NUM_INITS):
    rng = np.random.default_rng(9)
    probabilities = _cyclic()
    greedy = ELIGIBLE[probabilities.argmax(axis=1)]
    targets = ELIGIBLE[rng.integers(0, K, size=NUM_POSITIONS)]

    corpus = np.zeros(VOCAB_SIZE, dtype=np.int64)
    corpus[ELIGIBLE] = rng.integers(3, 30, size=K)
    greedy_counts = np.stack(
        [np.bincount(greedy, minlength=VOCAB_SIZE) for _ in range(num_inits)]
    )

    # Identity-preserving means over the full vocabulary, exactly uniform on the
    # eligible support and zero on the structural tokens.
    identity = np.zeros(VOCAB_SIZE)
    identity[ELIGIBLE] = probabilities.mean(axis=0)
    mean_tokens = np.tile(identity, (num_inits, len(TEMPERATURES), 1))
    ranked = np.tile(np.sort(BASE)[::-1], (num_inits, len(TEMPERATURES), 1))
    maxima = np.tile(probabilities.max(axis=1), (num_inits, len(TEMPERATURES), 1))
    targets_p = np.tile(
        probabilities[np.arange(NUM_POSITIONS), rng.integers(0, K, NUM_POSITIONS)],
        (num_inits, len(TEMPERATURES), 1),
    )
    canonical = TEMPERATURES.index(1.00)

    arrays = {
        "predictive_ranked_probabilities": ranked[:, canonical],
        "predictive_max_probabilities": maxima[:, canonical],
        "predictive_target_probabilities": targets_p[:, canonical],
        "predictive_target_losses": -np.log(targets_p[:, canonical]),
        "predictive_temperatures": np.asarray(TEMPERATURES),
        "predictive_temperature_ranked_probabilities": ranked,
        "predictive_temperature_max_probabilities": maxima,
        "predictive_temperature_target_probabilities": targets_p,
        "predictive_temperature_target_losses": -np.log(targets_p),
        "predictive_temperature_mean_entropy": np.ones((num_inits, len(TEMPERATURES))),
    }
    if with_tokens:
        arrays["predictive_temperature_mean_token_probabilities"] = mean_tokens

    return InitializationExperimentRecord.build(
        corpus_counts=corpus,
        selected_target_counts=np.bincount(targets, minlength=VOCAB_SIZE),
        greedy_counts=greedy_counts,
        nucleus_counts=greedy_counts[:, None, :],
        # The canonical row of the grid generalizes this field, so it must match.
        mean_predicted_probabilities=np.tile(identity, (num_inits, 1)),
        model_seeds=np.arange(num_inits) + 1000,
        eligible_token_ids=ELIGIBLE,
        metadata={"analysis": {"num_positions": NUM_POSITIONS}},
        **arrays,
    )


# -- the order of operations -------------------------------------------------


def test_average_then_rank_differs_provably_from_rank_then_average() -> None:
    """On this fixture the two orders are a peaked vector and a flat one."""

    probabilities = _cyclic()

    rank_first = np.sort(probabilities, axis=1)[:, ::-1].mean(axis=0)
    identity_first = np.sort(probabilities.mean(axis=0))[::-1]

    assert np.allclose(rank_first, np.sort(BASE)[::-1])
    assert np.allclose(identity_first, np.full(K, 1.0 / K))
    assert not np.allclose(rank_first, identity_first)

    record = _record()
    assert np.allclose(ranked_mean_token_probabilities(record)["mean"][0], identity_first)
    assert np.allclose(temperature_ranked_profiles(record)["mean"][0], rank_first)


def test_ranking_happens_per_initialization_before_summarizing() -> None:
    """Averaging across initializations first would blur identity structure.

    Two initializations that each favour one token, but different tokens, have
    peaked per-initialization profiles and a flat average of their probabilities.
    Ranking first must preserve the peak.
    """

    first = np.zeros(K)
    first[0] = 1.0
    second = np.zeros(K)
    second[-1] = 1.0
    stacked = np.stack([first, second])

    per_initialization = np.sort(stacked, axis=1)[:, ::-1].mean(axis=0)
    averaged_first = np.sort(stacked.mean(axis=0))[::-1]

    assert per_initialization[0] == pytest.approx(1.0)
    assert averaged_first[0] == pytest.approx(0.5)


def test_the_profile_is_non_increasing_and_normalized() -> None:
    profile = ranked_mean_token_probabilities(_record())

    assert profile["mean"].shape == (len(TEMPERATURES), K)
    assert np.all(np.diff(profile["mean"], axis=1) <= 1e-12)
    assert np.allclose(profile["mean"].sum(axis=1), 1.0, atol=1e-6)
    assert np.all(profile["profiles"] >= 0.0)
    assert np.all(np.isfinite(profile["profiles"]))


def test_rank_one_is_the_largest_mean_token_probability() -> None:
    record = _record()
    profile = ranked_mean_token_probabilities(record)
    values = np.asarray(record.predictive_temperature_mean_token_probabilities)

    for index in range(len(TEMPERATURES)):
        assert profile["profiles"][0, index, 0] == pytest.approx(
            values[0, index, ELIGIBLE].max()
        )


def test_the_canonical_row_generalizes_the_existing_mean_probabilities() -> None:
    """T = 1 is the field the experiment already had, not a second version."""

    record = _record()
    index = record.temperature_index(1.0)

    assert np.allclose(
        record.predictive_temperature_mean_token_probabilities[:, index],
        record.mean_predicted_probabilities,
        rtol=1e-6,
        atol=1e-9,
    )


def test_a_drifted_canonical_row_is_rejected() -> None:
    record = _record()
    broken = np.array(record.predictive_temperature_mean_token_probabilities)
    broken[:, record.temperature_index(1.0)] = np.roll(
        broken[:, record.temperature_index(1.0)], 1, axis=1
    )

    with pytest.raises(ValueError, match="mean_predicted_probabilities"):
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
            predictive_temperature_mean_token_probabilities=broken,
        )


# -- the cumulative comparison -----------------------------------------------


def test_rank_first_captures_at_least_as_much_top_k_mass() -> None:
    """Delta >= 0: letting the top-k identities vary cannot capture less."""

    comparison = cumulative_order_comparison(_record())

    for row in comparison["rows"]:
        for depth, values in row["depths"].items():
            assert values["delta"] >= -1e-12, (row["temperature"], depth)
            assert values["rank_then_average"] >= values["average_then_rank"] - 1e-12


def test_the_comparison_is_hand_computable_on_the_fixture() -> None:
    """A is BASE sorted, B is uniform, so the depths follow in closed form."""

    comparison = cumulative_order_comparison(_record())
    row = comparison["rows"][0]

    assert row["depths"]["1"]["rank_then_average"] == pytest.approx(0.5)
    assert row["depths"]["1"]["average_then_rank"] == pytest.approx(1.0 / K)
    assert row["depths"]["1"]["delta"] == pytest.approx(0.5 - 1.0 / K)


# -- persistence -------------------------------------------------------------


def test_no_full_probability_tensor_is_persisted(tmp_path) -> None:
    """Sufficient statistics only: [I, N_T, V], never [I, N_T, D, V]."""

    record = _record()
    record.save(tmp_path)

    with np.load(tmp_path / "initialization_distribution.npz") as archive:
        values = archive["predictive_temperature_mean_token_probabilities"]
        assert values.shape == (NUM_INITS, len(TEMPERATURES), VOCAB_SIZE)
        for key in archive.files:
            assert archive[key].ndim <= 3, key


def test_mean_token_probabilities_survive_a_round_trip(tmp_path) -> None:
    record = _record()
    record.save(tmp_path)
    reloaded = load_record(tmp_path)

    assert reloaded.has_mean_token_probabilities is True
    assert np.allclose(
        reloaded.predictive_temperature_mean_token_probabilities,
        record.predictive_temperature_mean_token_probabilities,
    )


def test_a_historical_record_without_them_still_loads(tmp_path) -> None:
    record = _record(with_tokens=False)
    record.save(tmp_path)
    reloaded = load_record(tmp_path)

    assert reloaded.has_mean_token_probabilities is False
    with pytest.raises(ValueError, match="no identity-preserving"):
        ranked_mean_token_probabilities(reloaded)


# -- the figure --------------------------------------------------------------


def test_figure_fourteen_renders(tmp_path) -> None:
    written = plot_ranked_mean_token_probabilities(_record(), tmp_path)

    assert [path.name for path in written] == ["figure14_ranked_mean_token_probabilities.svg"]
    assert written[0].is_file() and written[0].stat().st_size > 0


def test_figure_fourteen_appears_only_with_the_statistic(tmp_path) -> None:
    with_data = {path.name for path in generate_all_figures(_record(), tmp_path / "with")}
    without = {
        path.name
        for path in generate_all_figures(_record(with_tokens=False), tmp_path / "without")
    }

    name = "figure14_ranked_mean_token_probabilities.svg"
    assert name in with_data
    assert name not in without
    # Figure 11 is unchanged and still produced in both.
    assert "figure11_temperature_ranked_predictive_probabilities.svg" in without


def test_a_record_without_the_statistic_refuses_the_figure(tmp_path) -> None:
    with pytest.raises(ValueError, match="no identity-preserving"):
        plot_ranked_mean_token_probabilities(_record(with_tokens=False), tmp_path)


def test_the_renderer_maps_figure_fourteen() -> None:
    import importlib.util
    import sys
    from pathlib import Path

    from llm_behavior_lab.analysis import figures as figure_module

    path = Path(__file__).resolve().parents[1] / "scripts" / "render_record_figures.py"
    spec = importlib.util.spec_from_file_location("render_record_figures", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    assert module.FIGURES["figure14"] == "plot_ranked_mean_token_probabilities"
    assert module.CONDITIONAL_FIGURES["figure14"] == "has_mean_token_probabilities"
    assert callable(getattr(figure_module, module.FIGURES["figure14"]))
