"""Tests for the initial-gradient measurements.

The fixture is built from an **explicit probability matrix**. Every position's
full probability vector is written down, so ``A_i``, ``S_i`` and their splits can
be checked against their mathematical definitions

    A_i = (1/D) sum_{y_d=i} (1 - s_{d,i}) / T
    S_i = (1/D) sum_{y_d!=i}     s_{d,i}  / T

by summing over that matrix directly, rather than only checking that quantities
derived from the same sufficient statistics agree with each other. That
distinction matters here: the module derives ``A`` and ``S`` from persisted
per-position target probabilities and mean token probabilities, so the identity
``A_i - S_i = (f_i - pbar_i)/T`` holds by construction and cannot detect a wrong
conditioning set. Summing the explicit matrix can.

Values are chosen so an indexing mistake is visible: token IDs, target counts and
guess counts are all distinct, no two classes share a target count, and the
structural tokens carry exactly zero mass.
"""

from __future__ import annotations

import numpy as np
import pytest

from llm_behavior_lab.analysis import InitializationExperimentRecord
from llm_behavior_lab.analysis.gradients import temperature_gradient_table
from llm_behavior_lab.analysis.initial_gradients import (
    class_frequency_table,
    confidence_split,
    correctness_table,
    gradient_correctness_split,
    logit_correction,
)

VOCAB = 10
ELIGIBLE = np.arange(2, VOCAB)          # 8 eligible tokens, ids 2..9
K = ELIGIBLE.size
D = 20
TEMPERATURES = (0.12, 0.24, 0.36, 0.48, 0.60, 1.00, 1.20)
CANONICAL = TEMPERATURES.index(1.00)
INITS = 3


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def _logits(seed: int) -> np.ndarray:
    """``[D, K]`` logits with no ties, so argmax is unambiguous."""

    rng = np.random.default_rng(seed)
    values = rng.normal(0.0, 1.5, size=(D, K))
    # Break any accidental tie: distinct columns get distinct nudges.
    return values + np.arange(K) * 1e-6


#: Initialization 0 is the one the gradients are measured on; the others carry
#: deliberately different logits so a test can catch an implementation that
#: averaged across initializations or read the wrong row.
LOGITS = [_logits(seed) for seed in (11, 22, 33)]


def probabilities(initialization: int, temperature: float) -> np.ndarray:
    """``[D, VOCAB]`` explicit probabilities, zero on the structural tokens."""

    eligible = _softmax(LOGITS[initialization] / temperature)
    full = np.zeros((D, VOCAB), dtype=np.float64)
    full[:, ELIGIBLE] = eligible
    return full


def _targets_and_greedy() -> tuple[np.ndarray, np.ndarray]:
    """Targets with a deliberate mix of correct and failed assignments."""

    greedy = ELIGIBLE[LOGITS[0].argmax(axis=1)]
    targets = greedy.copy()
    # Positions 0..12 are failures, 13..19 are true positives. That leaves both
    # conditional means defined for several classes and neither group empty.
    rng = np.random.default_rng(7)
    for position in range(13):
        alternatives = ELIGIBLE[ELIGIBLE != greedy[position]]
        targets[position] = alternatives[rng.integers(0, alternatives.size)]
    return targets, greedy


TARGETS, GREEDY = _targets_and_greedy()


def _record():
    """A v9 record whose predictive arrays come from the explicit matrices."""

    corpus = np.zeros(VOCAB, dtype=np.int64)
    corpus[ELIGIBLE] = np.arange(3, 3 + K) * 7          # all distinct
    corpus[TARGETS] = np.maximum(corpus[TARGETS], 1)

    ranked = np.zeros((INITS, len(TEMPERATURES), K))
    maxima = np.zeros((INITS, len(TEMPERATURES), D))
    target_probability = np.zeros((INITS, len(TEMPERATURES), D))
    mean_tokens = np.zeros((INITS, len(TEMPERATURES), VOCAB))
    for initialization in range(INITS):
        for index, temperature in enumerate(TEMPERATURES):
            matrix = probabilities(initialization, temperature)
            eligible_only = matrix[:, ELIGIBLE]
            ranked[initialization, index] = np.sort(eligible_only, axis=1)[:, ::-1].mean(axis=0)
            maxima[initialization, index] = eligible_only.max(axis=1)
            target_probability[initialization, index] = matrix[np.arange(D), TARGETS]
            mean_tokens[initialization, index] = matrix.mean(axis=0)

    # Gradient norms that differ per position and per temperature, so a table
    # reading the wrong row is caught.
    base = np.linspace(1.0, 3.0, D) + 0.1 * np.arange(D) % 0.7
    per_temperature = np.stack([base / value for value in TEMPERATURES])

    return InitializationExperimentRecord.build(
        corpus_counts=corpus,
        selected_target_counts=np.bincount(TARGETS, minlength=VOCAB),
        greedy_counts=np.stack(
            [np.bincount(GREEDY, minlength=VOCAB) for _ in range(INITS)]
        ),
        nucleus_counts=np.stack(
            [np.bincount(GREEDY, minlength=VOCAB) for _ in range(INITS)]
        )[:, None, :],
        mean_predicted_probabilities=mean_tokens[:, CANONICAL, :],
        model_seeds=np.arange(INITS) + 1000,
        eligible_token_ids=ELIGIBLE,
        metadata={
            "tokens": [f"t{index}" for index in range(VOCAB)],
            "num_positions": D,
            "analysis": {
                "num_positions": D,
                "gradient_analysis": {
                    "enabled": True,
                    "initialization_index": 0,
                    "model_seed": 1000,
                    "input_condition": "real",
                    "softmax_support": "eligible",
                    "covers_all_positions": True,
                    "parameter_count": 1234,
                    "temperatures": list(TEMPERATURES),
                },
            },
        },
        gradient_position_indices=np.arange(D),
        gradient_position_target_ids=TARGETS,
        gradient_position_greedy_ids=GREEDY,
        gradient_position_norms=per_temperature[CANONICAL],
        gradient_temperatures=np.asarray(TEMPERATURES),
        gradient_temperature_position_norms=per_temperature,
        predictive_ranked_probabilities=ranked[:, CANONICAL],
        predictive_max_probabilities=maxima[:, CANONICAL],
        predictive_target_probabilities=target_probability[:, CANONICAL],
        predictive_target_losses=-np.log(target_probability[:, CANONICAL]),
        predictive_temperatures=np.asarray(TEMPERATURES),
        predictive_temperature_ranked_probabilities=ranked,
        predictive_temperature_max_probabilities=maxima,
        predictive_temperature_target_probabilities=target_probability,
        predictive_temperature_target_losses=-np.log(target_probability),
        predictive_temperature_mean_entropy=np.ones((INITS, len(TEMPERATURES))),
        predictive_temperature_mean_token_probabilities=mean_tokens,
    )


# -- frequencies and the two biases ------------------------------------------


def test_target_and_guess_frequencies_are_distributions() -> None:
    freq = class_frequency_table(_record(), 1.0)

    assert freq["target_count"].sum() == D
    assert freq["guess_count"].sum() == D
    assert freq["target_fraction"].sum() == pytest.approx(1.0, abs=1e-15)
    assert freq["guess_fraction"].sum() == pytest.approx(1.0, abs=1e-15)
    # Structural tokens are never targets or guesses.
    assert freq["target_count"][:2].sum() == 0
    assert freq["guess_count"][:2].sum() == 0


def test_hard_bias_sums_to_zero_and_matches_its_definition() -> None:
    freq = class_frequency_table(_record(), 1.0)

    assert freq["hard_bias"].sum() == pytest.approx(0.0, abs=1e-15)
    assert np.allclose(
        freq["hard_bias"], freq["guess_fraction"] - freq["target_fraction"], atol=0
    )


def test_soft_bias_uses_mean_probability_not_greedy_frequency() -> None:
    """b_soft is a mass average; b_hard is a top-1 count. They must differ."""

    record = _record()
    freq = class_frequency_table(record, 1.0)
    expected = probabilities(0, 1.0).mean(axis=0) - freq["target_fraction"]

    assert np.allclose(freq["soft_bias"], expected, rtol=0, atol=1e-15)
    assert freq["soft_bias"].sum() == pytest.approx(0.0, abs=1e-12)
    # The whole point of keeping both: on real data they are not the same thing.
    assert not np.allclose(freq["soft_bias"], freq["hard_bias"])


def test_the_gradient_initialization_row_is_used_not_an_average() -> None:
    """Reading row 0 by accident, or averaging, must be detectable."""

    record = _record()
    freq = class_frequency_table(record, 1.0)

    averaged = np.mean([probabilities(i, 1.0).mean(axis=0) for i in range(INITS)], axis=0)
    assert not np.allclose(freq["mean_probability"], averaged)
    assert np.allclose(freq["mean_probability"], probabilities(0, 1.0).mean(axis=0))


# -- correctness -------------------------------------------------------------


def test_true_positive_plus_false_negative_equals_support() -> None:
    record = _record()
    freq = class_frequency_table(record, 1.0)
    corr = correctness_table(record, 1.0)

    assert np.array_equal(
        corr["true_positive"] + corr["false_negative"], freq["target_count"]
    )
    assert corr["correct_positions"] == int((TARGETS == GREEDY).sum())
    assert corr["correct_positions"] == 7          # positions 13..19


def test_recall_and_precision_match_their_definitions() -> None:
    record = _record()
    corr = correctness_table(record, 1.0)

    for token in ELIGIBLE:
        support = int((TARGETS == token).sum())
        guesses = int((GREEDY == token).sum())
        hits = int(((TARGETS == token) & (GREEDY == token)).sum())
        if support:
            assert corr["recall"][token] == pytest.approx(hits / support)
        if guesses:
            assert corr["precision"][token] == pytest.approx(hits / guesses)


def test_undefined_recall_and_precision_stay_undefined() -> None:
    """A class never targeted has no recall; never guessed, no precision."""

    record = _record()
    corr = correctness_table(record, 1.0)
    freq = class_frequency_table(record, 1.0)

    unrepresented = freq["target_count"] == 0
    unguessed = freq["guess_count"] == 0
    assert unrepresented.any() and unguessed.any()
    assert np.all(np.isnan(corr["recall"][unrepresented]))
    assert np.all(np.isnan(corr["precision"][unguessed]))
    # Never a silent zero.
    assert not np.any(corr["recall"][unrepresented] == 0.0)


def test_the_independence_null_is_the_marginal_preserving_one() -> None:
    """R_indep = q_i and P_indep = f_i, not 1/K."""

    record = _record()
    freq = class_frequency_table(record, 1.0)
    corr = correctness_table(record, 1.0)

    assert np.array_equal(corr["recall_independent"], freq["guess_fraction"])
    assert np.array_equal(corr["precision_independent"], freq["target_fraction"])
    assert np.allclose(
        corr["delta_recall"], corr["recall"] - freq["guess_fraction"], equal_nan=True
    )
    assert corr["independence_accuracy"] == pytest.approx(
        float((freq["target_fraction"] * freq["guess_fraction"]).sum())
    )


def test_excess_true_positive_mass_factors_two_ways() -> None:
    """X_i = f_i * DeltaR_i = q_i * DeltaP_i, exactly."""

    record = _record()
    freq = class_frequency_table(record, 1.0)
    corr = correctness_table(record, 1.0)

    excess = corr["excess_true_positive"]
    through_recall = freq["target_fraction"] * corr["delta_recall"]
    through_precision = freq["guess_fraction"] * corr["delta_precision"]

    recall_defined = np.isfinite(through_recall)
    precision_defined = np.isfinite(through_precision)
    assert recall_defined.sum() > 1 and precision_defined.sum() > 1
    assert np.allclose(excess[recall_defined], through_recall[recall_defined], atol=1e-15)
    assert np.allclose(
        excess[precision_defined], through_precision[precision_defined], atol=1e-15
    )
    both = recall_defined & precision_defined
    assert np.allclose(through_recall[both], through_precision[both], atol=1e-15)


# -- gradient split ----------------------------------------------------------


def test_gradient_intensity_matches_direct_conditional_means() -> None:
    record = _record()
    split = gradient_correctness_split(record, 1.0)
    norms = np.asarray(record.gradient_temperature_position_norms[CANONICAL])

    for token in ELIGIBLE:
        hit = (TARGETS == token) & (GREEDY == token)
        miss = (TARGETS == token) & (GREEDY != token)
        if hit.any():
            assert split["intensity_true_positive"][token] == pytest.approx(
                norms[hit].mean()
            )
        else:
            assert np.isnan(split["intensity_true_positive"][token])
        if miss.any():
            assert split["intensity_false_negative"][token] == pytest.approx(
                norms[miss].mean()
            )
        else:
            assert np.isnan(split["intensity_false_negative"][token])


def test_existing_class_gradient_norm_is_reconstructed_from_the_split() -> None:
    """G_i = [TP_i G_i^TP + FN_i G_i^FN] / n_i, tying the split to figure 10."""

    record = _record()
    freq = class_frequency_table(record, 1.0)
    corr = correctness_table(record, 1.0)
    split = gradient_correctness_split(record, 1.0)
    reference = temperature_gradient_table(record, 1.0)["mean_gradient_norm"]

    represented = freq["target_count"] > 0
    intensity_tp = np.nan_to_num(split["intensity_true_positive"], nan=0.0)
    intensity_fn = np.nan_to_num(split["intensity_false_negative"], nan=0.0)
    rebuilt = (
        corr["true_positive"] * intensity_tp + corr["false_negative"] * intensity_fn
    )[represented] / freq["target_count"][represented]

    assert np.allclose(rebuilt, reference[represented], rtol=0, atol=1e-12)


def test_norm_mass_is_a_sum_not_a_mean() -> None:
    """Mass is frequency times intensity, and totals the whole norm sum."""

    record = _record()
    split = gradient_correctness_split(record, 1.0)
    norms = np.asarray(record.gradient_temperature_position_norms[CANONICAL])

    total = split["total_mass_true_positive"] + split["total_mass_false_negative"]
    assert total == pytest.approx(norms.sum() / D)

    correct = TARGETS == GREEDY
    assert split["total_mass_true_positive"] == pytest.approx(norms[correct].sum() / D)
    assert split["total_mass_false_negative"] == pytest.approx(norms[~correct].sum() / D)


# -- the A / S decomposition, against its definition --------------------------


def _direct_attraction_and_suppression(temperature: float):
    """A_i and S_i summed straight from the explicit probability matrix."""

    matrix = probabilities(0, temperature)
    attraction = np.zeros(VOCAB)
    suppression = np.zeros(VOCAB)
    attraction_tp = np.zeros(VOCAB)
    suppression_fp = np.zeros(VOCAB)
    for position in range(D):
        target = TARGETS[position]
        winner = GREEDY[position]
        for token in range(VOCAB):
            mass = matrix[position, token]
            if token == target:
                attraction[token] += 1.0 - mass
                if winner == target:
                    attraction_tp[token] += 1.0 - mass
            else:
                suppression[token] += mass
                if winner == token:
                    suppression_fp[token] += mass
    scale = D * temperature
    return (
        attraction / scale,
        suppression / scale,
        attraction_tp / scale,
        suppression_fp / scale,
    )


@pytest.mark.parametrize("temperature", [0.12, 0.60, 1.00, 1.20])
def test_attraction_and_suppression_match_their_definitions(temperature) -> None:
    """Summed from the explicit matrix, not re-derived from the same statistics."""

    record = _record()
    logit = logit_correction(record, temperature)
    attraction, suppression, attraction_tp, suppression_fp = (
        _direct_attraction_and_suppression(temperature)
    )

    assert np.allclose(logit["attraction"], attraction, rtol=0, atol=1e-14)
    assert np.allclose(logit["suppression"], suppression, rtol=0, atol=1e-14)
    assert np.allclose(
        logit["attraction_true_positive"], attraction_tp, rtol=0, atol=1e-14
    )
    assert np.allclose(
        logit["attraction_false_negative"], attraction - attraction_tp, rtol=0, atol=1e-14
    )
    assert np.allclose(
        logit["suppression_false_positive"], suppression_fp, rtol=0, atol=1e-14
    )
    assert np.allclose(
        logit["suppression_other"], suppression - suppression_fp, rtol=0, atol=1e-14
    )


def test_the_two_decompositions_are_exact_partitions() -> None:
    record = _record()
    logit = logit_correction(record, 1.0)

    assert np.allclose(
        logit["attraction_true_positive"] + logit["attraction_false_negative"],
        logit["attraction"],
        rtol=0,
        atol=1e-15,
    )
    assert np.allclose(
        logit["suppression_false_positive"] + logit["suppression_other"],
        logit["suppression"],
        rtol=0,
        atol=1e-15,
    )


@pytest.mark.parametrize("temperature", [0.24, 1.00])
def test_the_sign_convention_of_the_net_correction(temperature) -> None:
    """S - A is the mean logit gradient; A - S is the descent correction."""

    record = _record()
    freq = class_frequency_table(record, temperature)
    logit = logit_correction(record, temperature)
    soft_bias = freq["soft_bias"]

    assert np.allclose(
        logit["suppression"] - logit["attraction"], soft_bias / temperature, atol=1e-14
    )
    assert np.allclose(
        logit["attraction"] - logit["suppression"], -soft_bias / temperature, atol=1e-14
    )
    assert np.allclose(logit["net_correction"], -soft_bias / temperature, atol=1e-14)


def test_correction_fractions_keep_undefined_denominators_undefined() -> None:
    record = _record()
    logit = logit_correction(record, 1.0)

    zero_attraction = logit["attraction"] == 0
    assert zero_attraction.any()            # the structural tokens
    assert np.all(np.isnan(logit["failed_attraction_fraction"][zero_attraction]))


# -- confidence --------------------------------------------------------------


def test_confidence_split_matches_the_explicit_matrix() -> None:
    record = _record()
    conf = confidence_split(record, 1.0)
    matrix = probabilities(0, 1.0)

    assert np.allclose(
        conf["position_confidence"], matrix[np.arange(D), TARGETS], atol=1e-15
    )
    assert np.allclose(
        conf["position_winner"], matrix[np.arange(D), GREEDY], atol=1e-15
    )
    # The margin is positive exactly on failures, by construction.
    failures = ~conf["position_correct"]
    margin = conf["position_winner"] - conf["position_confidence"]
    assert np.all(margin[failures] > 0)
    assert np.allclose(margin[~failures], 0.0, atol=1e-15)


def test_conditional_confidence_means_stay_undefined_when_empty() -> None:
    record = _record()
    conf = confidence_split(record, 1.0)
    corr = correctness_table(record, 1.0)

    no_hits = corr["true_positive"] == 0
    assert no_hits.any()
    assert np.all(np.isnan(conf["confidence_true_positive"][no_hits]))


# -- alignment ---------------------------------------------------------------


def test_a_subset_gradient_run_is_refused() -> None:
    """Predictive arrays span every position; gradients may not."""

    record = _record()
    truncated = InitializationExperimentRecord.build(
        corpus_counts=record.corpus_counts,
        selected_target_counts=record.selected_target_counts,
        greedy_counts=record.greedy_counts,
        nucleus_counts=record.nucleus_counts,
        mean_predicted_probabilities=record.mean_predicted_probabilities,
        model_seeds=record.model_seeds,
        eligible_token_ids=record.eligible_token_ids,
        metadata=record.metadata,
        gradient_position_indices=np.arange(0, D, 2),
        gradient_position_target_ids=TARGETS[::2],
        gradient_position_greedy_ids=GREEDY[::2],
        gradient_position_norms=np.asarray(
            record.gradient_temperature_position_norms[CANONICAL]
        )[::2],
        gradient_temperatures=np.asarray(TEMPERATURES),
        gradient_temperature_position_norms=np.asarray(
            record.gradient_temperature_position_norms
        )[:, ::2],
        predictive_temperatures=np.asarray(TEMPERATURES),
        predictive_ranked_probabilities=record.predictive_ranked_probabilities,
        predictive_max_probabilities=record.predictive_max_probabilities,
        predictive_target_probabilities=record.predictive_target_probabilities,
        predictive_target_losses=record.predictive_target_losses,
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
        predictive_temperature_mean_token_probabilities=(
            record.predictive_temperature_mean_token_probabilities
        ),
    )

    with pytest.raises(ValueError, match="complete evaluation grid"):
        class_frequency_table(truncated, 1.0)


def test_canonical_token_alignment_is_preserved() -> None:
    """Every table is indexed by token ID over the full vocabulary."""

    record = _record()
    for table in (
        class_frequency_table(record, 1.0),
        correctness_table(record, 1.0),
        gradient_correctness_split(record, 1.0),
        logit_correction(record, 1.0),
    ):
        for key, value in table.items():
            if isinstance(value, np.ndarray) and value.ndim == 1 and value.size > D:
                assert value.size == VOCAB, key


# -- the vector-split accessor (no torch: reads persisted metadata only) ------


def test_the_vector_split_is_absent_from_a_record_that_never_measured_it() -> None:
    """Old records must stay valid; absence is None, never a fabricated zero."""

    from llm_behavior_lab.analysis import gradient_vector_split

    assert gradient_vector_split(_record()) is None


def test_the_vector_split_is_returned_when_the_block_is_present() -> None:
    from llm_behavior_lab.analysis import gradient_vector_split

    record = _record()
    metadata = {
        **record.metadata,
        "analysis": {
            **record.metadata["analysis"],
            "gradient_analysis": {
                **record.metadata["analysis"]["gradient_analysis"],
                "vector_split": {
                    "temperature": 1.0,
                    "norm_correct": 2.0,
                    "norm_wrong": 3.0,
                    "norm_total": 4.0,
                    "dot": 1.5,
                    "cosine": 0.25,
                    "num_correct": 7,
                    "num_wrong": D - 7,
                },
            },
        },
    }
    carried = InitializationExperimentRecord.build(
        corpus_counts=record.corpus_counts,
        selected_target_counts=record.selected_target_counts,
        greedy_counts=record.greedy_counts,
        nucleus_counts=record.nucleus_counts,
        mean_predicted_probabilities=record.mean_predicted_probabilities,
        model_seeds=record.model_seeds,
        eligible_token_ids=record.eligible_token_ids,
        metadata=metadata,
        gradient_position_indices=record.gradient_position_indices,
        gradient_position_target_ids=record.gradient_position_target_ids,
        gradient_position_greedy_ids=record.gradient_position_greedy_ids,
        gradient_position_norms=record.gradient_position_norms,
        gradient_temperatures=record.gradient_temperatures,
        gradient_temperature_position_norms=record.gradient_temperature_position_norms,
    )

    split = gradient_vector_split(carried)
    assert split is not None
    assert split["num_correct"] + split["num_wrong"] == D
    assert split["temperature"] == 1.0
    # A copy, so a caller cannot mutate the record's metadata through it.
    split["norm_correct"] = -1.0
    assert gradient_vector_split(carried)["norm_correct"] == 2.0


def test_a_record_without_gradients_has_no_vector_split() -> None:
    from llm_behavior_lab.analysis import gradient_vector_split

    plain = InitializationExperimentRecord.build(
        corpus_counts=np.bincount(TARGETS, minlength=VOCAB),
        selected_target_counts=np.bincount(TARGETS, minlength=VOCAB),
        greedy_counts=np.bincount(GREEDY, minlength=VOCAB)[None, :],
        nucleus_counts=np.bincount(GREEDY, minlength=VOCAB)[None, None, :],
        mean_predicted_probabilities=np.full((1, VOCAB), 1.0 / VOCAB),
        model_seeds=[1000],
        eligible_token_ids=ELIGIBLE,
        metadata={},
    )

    assert gradient_vector_split(plain) is None
