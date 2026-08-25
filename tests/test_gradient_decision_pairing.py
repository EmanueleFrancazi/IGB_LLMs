"""Tests for figures 15 and 16: the two paired gradient diagnostics.

Figure 10 puts a deterministic decision on the y axis, figure 15 a single
realized stochastic decision, and figure 16 the continuous predictive mass the
decision would have been drawn from. The three share an x axis exactly -- the
same ``G_i(T)``, on the same initialization, over the same positions -- so the
tests here are mostly about that pairing being real rather than approximate.

Neither figure adds a measurement. Figure 15 reads the nucleus draws the sweep
already realized (one per position, ``R = 1``, no resampling) and figure 16
reads the mean token probabilities figure 14 already persists.
"""

from __future__ import annotations

import numpy as np
import pytest

from llm_behavior_lab.analysis import (
    InitializationExperimentRecord,
    mean_probability_gradient_table,
    nucleus_gradient_table,
    temperature_gradient_table,
)
from llm_behavior_lab.analysis.nulls import simulate_uniform_null

matplotlib = pytest.importorskip("matplotlib")

from llm_behavior_lab.analysis.figures import (  # noqa: E402
    generate_all_figures,
    plot_gradient_vs_mean_probability,
    plot_gradient_vs_nucleus_guess_bias,
)


VOCAB = 40
ELIGIBLE = np.arange(2, VOCAB)
K = ELIGIBLE.size
D = 96
INITS = 3
GRAD_T = (0.12, 0.24, 0.36, 0.48, 0.60, 1.00, 1.20)
SWEEP_T = [0.12, 0.24, 0.36, 0.48, 0.60, 1.20]
PANELS = [value for value in GRAD_T if value != 1.0]


def full_record(*, covers_all=True, with_sweep=True, with_mean_tokens=True,
                num_replicates=1, sweep_grid=None, confidence_grid=None):
    """A record carrying gradients, the sweep, and the mean token statistic.

    ``sweep_grid`` and ``confidence_grid`` default to grids that contain every
    panel temperature. They are overridable because the three axes -- gradient,
    nucleus sweep, and the fixed confidence grid -- are configured independently
    in a real run, so a record can legitimately carry a gradient temperature
    that one of the other two never covered.
    """

    sweep_t = list(SWEEP_T if sweep_grid is None else sweep_grid)
    confidence_t = tuple(GRAD_T if confidence_grid is None else confidence_grid)
    rng = np.random.default_rng(11)
    targets = rng.choice(ELIGIBLE, size=D)
    greedy = rng.choice(ELIGIBLE[:6], size=D)

    corpus = np.zeros(VOCAB, dtype=np.int64)
    corpus[ELIGIBLE] = rng.integers(5, 400, size=K)
    corpus[targets] = np.maximum(corpus[targets], 1)

    base = rng.lognormal(0.0, 0.5, size=D)
    per_temperature = np.stack([base / value for value in GRAD_T])

    greedy_counts = np.stack(
        [np.bincount(rng.choice(ELIGIBLE, size=D), minlength=VOCAB) for _ in range(INITS)]
    )
    nucleus_counts = np.repeat(greedy_counts[:, None, :], num_replicates, axis=1)

    arrays = {}
    metadata = {
        "tokens": [f"t{index}" for index in range(VOCAB)],
        "num_positions": D,
        "analysis": {
            "num_positions": D,
            "gradient_analysis": {
                "enabled": True,
                "softmax_support": "eligible",
                "initialization_index": 0,
                "model_seed": 1000,
                "input_condition": "real",
                "covers_all_positions": covers_all,
                "parameter_count": 8_585_856,
                "temperatures": list(GRAD_T),
            },
        },
    }

    if with_sweep:
        # Realized draws: exactly one sampled token per position per temperature,
        # so each row is a multinomial over D draws and sums to D.
        # 'real' only: the input-structure conditions would additionally need
        # their own greedy/nucleus counts, and figure 15 reads the real
        # condition regardless.
        sweeps = {}
        agreement = {}
        for condition in ("real",):
            counts = np.zeros((INITS, len(sweep_t), VOCAB), dtype=np.int64)
            for s_ in range(INITS):
                for t_ in range(len(sweep_t)):
                    counts[s_, t_, ELIGIBLE] = rng.multinomial(
                        D, rng.dirichlet(np.ones(K)))
            sweeps[condition] = counts
            agreement[condition] = rng.random((INITS, len(sweep_t)))
        metadata["analysis"]["temperature_sweep"] = {
            "enabled": True, "temperatures": sweep_t}
        metadata["analysis"]["sampling"] = {"temperature": 0.6, "top_p": 0.9}
    else:
        sweeps, agreement = {}, {}

    if with_mean_tokens:
        mean_tokens = np.zeros((INITS, len(confidence_t), VOCAB))
        for s_ in range(INITS):
            for t_ in range(len(confidence_t)):
                row = rng.dirichlet(np.ones(K))
                mean_tokens[s_, t_, ELIGIBLE] = row
        canonical = confidence_t.index(1.00)
        ranked = np.sort(mean_tokens[:, :, ELIGIBLE], axis=-1)[:, :, ::-1]
        # The record requires rank 1 to equal the mean stored maximum for each
        # (initialization, temperature): a flat p_max vector at that value makes
        # the fixture satisfy the invariant exactly rather than approximately.
        maxima = np.repeat(ranked[:, :, :1], D, axis=2)
        targets_p = np.repeat(ranked[:, :, 1:2], D, axis=2)
        arrays.update(
            predictive_ranked_probabilities=ranked[:, canonical],
            predictive_max_probabilities=maxima[:, canonical],
            predictive_target_probabilities=targets_p[:, canonical],
            predictive_target_losses=-np.log(targets_p[:, canonical]),
            predictive_temperatures=np.asarray(confidence_t),
            predictive_temperature_ranked_probabilities=ranked,
            predictive_temperature_max_probabilities=maxima,
            predictive_temperature_target_probabilities=targets_p,
            predictive_temperature_target_losses=-np.log(targets_p),
            predictive_temperature_mean_entropy=np.ones((INITS, len(confidence_t))),
            predictive_temperature_mean_token_probabilities=mean_tokens,
        )
        mean_predicted = mean_tokens[:, canonical, :]
    else:
        mean_predicted = np.tile(corpus / corpus.sum(), (INITS, 1))

    null = simulate_uniform_null(
        eligible_vocab_size=K, num_draws=D, num_replicates=8, seed=3)

    return InitializationExperimentRecord.build(
        corpus_counts=corpus,
        selected_target_counts=np.bincount(targets, minlength=VOCAB),
        greedy_counts=greedy_counts,
        nucleus_counts=nucleus_counts,
        mean_predicted_probabilities=mean_predicted,
        model_seeds=np.arange(INITS) + 1000,
        eligible_token_ids=ELIGIBLE,
        sweep_counts_by_condition=sweeps,
        sweep_agreement_by_condition=agreement,
        uniform_null={"ranked_mean": null.ranked_mean,
                      "ranked_low": null.ranked_low,
                      "ranked_high": null.ranked_high},
        metadata=metadata,
        gradient_position_indices=np.arange(D),
        gradient_position_target_ids=targets,
        gradient_position_greedy_ids=greedy,
        gradient_position_norms=per_temperature[GRAD_T.index(1.00)],
        gradient_temperatures=np.asarray(GRAD_T),
        gradient_temperature_position_norms=per_temperature,
        **arrays,
    )

# -- figure 15: the realized nucleus decision --------------------------------


@pytest.mark.parametrize("temperature", PANELS)
def test_realized_nucleus_counts_are_one_draw_per_position(temperature) -> None:
    """R = 1: every position contributes exactly one sampled token."""

    table = nucleus_gradient_table(full_record(), temperature)
    counts = table["nucleus_guess_count"]

    assert counts.dtype.kind == "i"
    assert (counts >= 0).all()
    assert counts.shape == (VOCAB,)
    assert int(counts.sum()) == D
    assert table["nucleus_num_positions"] == D
    assert table["nucleus_guess_fraction"].sum() == pytest.approx(1.0)


def test_nucleus_counts_are_the_persisted_draws_not_a_resample() -> None:
    """The figure reads realized decisions; it must not invent new ones."""

    record = full_record()
    row = record.sweep_counts("real")[record.gradient_initialization_index,
                                      SWEEP_T.index(0.36)]

    assert np.array_equal(
        nucleus_gradient_table(record, 0.36)["nucleus_guess_count"], row
    )


def test_figure_15_x_axis_is_bitwise_figure_10s() -> None:
    """The pairing is the point: same gradient statistic, not a recomputation."""

    record = full_record()
    for temperature in PANELS:
        expected = temperature_gradient_table(record, temperature)["mean_gradient_norm"]
        actual = nucleus_gradient_table(record, temperature)["mean_gradient_norm"]
        measured = ~np.isnan(expected)
        assert np.array_equal(actual[measured], expected[measured])
        assert np.array_equal(np.isnan(actual), np.isnan(expected))


def test_a_gradient_temperature_the_sweep_never_sampled_is_refused() -> None:
    """The sweep grid is configured independently of the gradient grid.

    T = 1 sits in the gradient grid and not in the sweep grid, so asking for it
    must name the problem rather than silently pair mismatched temperatures.
    """

    with pytest.raises(KeyError, match="nucleus sweep grid"):
        nucleus_gradient_table(full_record(), 1.00)


def test_figure_15_needs_a_sweep() -> None:
    with pytest.raises(ValueError, match="no nucleus temperature sweep"):
        nucleus_gradient_table(full_record(with_sweep=False), 0.36)


# -- figure 16: the continuous predictive mass -------------------------------


@pytest.mark.parametrize("temperature", PANELS)
def test_mean_predictive_probabilities_are_a_distribution(temperature) -> None:
    values = mean_probability_gradient_table(
        full_record(), temperature)["mean_predictive_probability"]

    assert np.isfinite(values).all()
    assert (values >= 0).all()
    assert values.sum() == pytest.approx(1.0)


def test_figure_16_reuses_the_figure_14_statistic_exactly() -> None:
    """No second copy of the same numbers, and no re-derivation."""

    record = full_record()
    for temperature in PANELS:
        index = record.temperature_index(temperature)
        expected = np.asarray(
            record.predictive_temperature_mean_token_probabilities[
                record.gradient_initialization_index, index]
        )
        actual = mean_probability_gradient_table(
            record, temperature)["mean_predictive_probability"]
        assert np.array_equal(actual, expected)


def test_figure_16_is_not_averaged_across_initializations() -> None:
    """Averaging would break the pairing with initialization-0 gradients."""

    record = full_record()
    averaged = np.asarray(
        record.predictive_temperature_mean_token_probabilities
    ).mean(axis=0)[record.temperature_index(0.36)]
    actual = mean_probability_gradient_table(
        record, 0.36)["mean_predictive_probability"]

    assert not np.allclose(actual, averaged)


def test_figure_16_x_axis_is_bitwise_figure_10s() -> None:
    record = full_record()
    for temperature in PANELS:
        expected = temperature_gradient_table(record, temperature)["mean_gradient_norm"]
        actual = mean_probability_gradient_table(
            record, temperature)["mean_gradient_norm"]
        measured = ~np.isnan(expected)
        assert np.array_equal(actual[measured], expected[measured])


def test_figure_16_needs_the_mean_token_statistic() -> None:
    with pytest.raises(ValueError, match="mean token probabilities"):
        mean_probability_gradient_table(full_record(with_mean_tokens=False), 0.36)


# -- pairing guard rail ------------------------------------------------------


@pytest.mark.parametrize(
    "builder", [nucleus_gradient_table, mean_probability_gradient_table]
)
def test_a_subset_gradient_run_cannot_be_paired(builder) -> None:
    """Both y quantities span every position; G_i may span only a subset.

    Pairing them anyway would put a subset quantity on one axis and a
    whole-experiment quantity on the other without saying so.
    """

    with pytest.raises(ValueError, match="subset"):
        builder(full_record(covers_all=False), 0.36)


# -- rendering ---------------------------------------------------------------


def test_both_figures_render(tmp_path) -> None:
    record = full_record()

    written = plot_gradient_vs_nucleus_guess_bias(record, tmp_path)
    written += plot_gradient_vs_mean_probability(record, tmp_path)

    assert len(written) == 2
    for path in written:
        assert path.stat().st_size > 0
        assert path.suffix == ".svg"


def test_the_six_panels_match_figure_10s_temperatures(tmp_path) -> None:
    from llm_behavior_lab.analysis.figures import _gradient_panel_temperatures

    assert _gradient_panel_temperatures(full_record(), None) == [
        0.12, 0.24, 0.36, 0.48, 0.60, 1.20
    ]


def test_a_record_without_the_prerequisites_skips_them_cleanly(tmp_path) -> None:
    """Old records must still render figures 0-14 and simply omit 15 and 16."""

    written = generate_all_figures(
        full_record(with_sweep=False, with_mean_tokens=False), tmp_path
    )

    names = [path.name for path in written]
    assert written
    assert not any("figure15" in name or "figure16" in name for name in names)


def test_a_subset_gradient_record_still_renders_the_rest(tmp_path) -> None:
    written = generate_all_figures(full_record(covers_all=False), tmp_path)

    names = [path.name for path in written]
    assert any("figure10" in name for name in names)
    assert not any("figure15" in name or "figure16" in name for name in names)


def test_a_gradient_temperature_the_sweep_missed_omits_only_figure_15(tmp_path) -> None:
    """The three temperature axes are configured independently.

    A gradient temperature the nucleus sweep never sampled leaves figure 15
    undefined, exactly as R != 1 or a subset run does. The measurement is still
    valid, so the rest of the set -- including figure 16, whose own axis does
    cover the temperature -- must still be drawn rather than the whole run
    failing after the record has been written.
    """

    record = full_record(sweep_grid=[value for value in SWEEP_T if value != 0.12])

    written = generate_all_figures(record, tmp_path)

    names = [path.name for path in written]
    assert any("figure10" in name for name in names)
    assert not any("figure15" in name for name in names)
    assert any("figure16" in name for name in names)


def test_a_gradient_temperature_off_the_confidence_grid_omits_only_figure_16(
    tmp_path,
) -> None:
    """The mirror case: figure 16 reads the fixed confidence grid.

    That grid is a module constant with no command-line flag, so a gradient
    temperature can miss it however the run was configured. Figure 15, whose
    axis does cover the temperature, must still be drawn.
    """

    record = full_record(
        confidence_grid=tuple(0.13 if value == 0.12 else value for value in GRAD_T)
    )

    written = generate_all_figures(record, tmp_path)

    names = [path.name for path in written]
    assert any("figure10" in name for name in names)
    assert any("figure15" in name for name in names)
    assert not any("figure16" in name for name in names)


# -- R = 1 is part of the definition ----------------------------------------


def test_figure_15_requires_one_replicate() -> None:
    """R != 1 leaves "the realized decision at position d" undefined.

    Averaging, pooling, or picking one replicate would each answer a different
    question, so the builder refuses rather than choosing for the reader.
    """

    record = full_record(num_replicates=3)
    assert record.num_replicates == 3

    with pytest.raises(ValueError, match="only for R = 1"):
        nucleus_gradient_table(record, 0.36)


def test_the_r_guard_is_runtime_not_merely_a_test_assumption() -> None:
    """The single-replicate record must still work, so the guard is real."""

    assert full_record().num_replicates == 1
    assert nucleus_gradient_table(full_record(), 0.36)["nucleus_guess_count"].sum() == D


def test_an_r_greater_than_one_record_still_renders_everything_else(tmp_path) -> None:
    """Figure 15 drops out; figures 10 and 16 are unaffected by R."""

    written = generate_all_figures(full_record(num_replicates=3), tmp_path)
    names = [path.name for path in written]

    assert any("figure10" in name for name in names)
    assert any("figure16" in name for name in names)
    assert not any("figure15" in name for name in names)


# -- figure 16 axis policy ---------------------------------------------------


def test_figure_16_uses_one_transformation_for_every_panel() -> None:
    """Six panels of the same quantity must not mix log and linear axes."""

    from llm_behavior_lab.analysis.figures import _common_probability_axis

    record = full_record()
    plotted = temperature_gradient_table(record, 0.12)["target_occurrence_count"] > 0
    panels = [
        np.asarray(
            mean_probability_gradient_table(record, temperature)[
                "mean_predictive_probability"
            ]
        )[plotted]
        for temperature in PANELS
    ]

    # One bool for the whole figure, not one per panel.
    assert isinstance(_common_probability_axis(panels, "auto"), bool)
    assert _common_probability_axis(panels, "log") is True
    assert _common_probability_axis(panels, "linear") is False


def test_an_exact_zero_forces_a_linear_axis() -> None:
    """pbar can underflow to zero at low T; a log axis would delete the token."""

    from llm_behavior_lab.analysis.figures import _common_probability_axis

    spread = [np.array([1e-6, 1e-1])]
    assert _common_probability_axis(spread, "auto") is True

    with_zero = [np.array([0.0, 1e-6, 1e-1])]
    assert _common_probability_axis(with_zero, "auto") is False
