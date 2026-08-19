"""Initial-gradient measurements: from guessing bias to corrective force.

Everything here is derived from an already-persisted record. No model is built,
no forward or backward pass is run, and no random number is drawn. The module
exists to answer one chain of questions about the *initialized* network:

    initial bias -> initial correctness -> confidence/error
                 -> gradient magnitude -> corrective force

Three quantities that are easy to conflate are kept apart throughout:

``f_i``
    empirical target frequency over the evaluated positions. This is the
    frequency that enters the exact cross-entropy identity.
``p_i`` (corpus)
    frequency over the whole analysis split, the broader imbalance story.
``q_i``
    greedy guess frequency, a *decision* statistic.

and correspondingly two different notions of "bias":

``b_i^hard = q_i - f_i``
    realized argmax bias.
``b_i^soft(T) = pbar_i(T) - f_i``
    probability-mass bias, the quantity that actually appears in the mean logit
    error. Their relationship is **empirical, not an identity**: one is a top-1
    count, the other a mass average, and nothing forces them to agree.

Every quantity is aligned to canonical token IDs and paired to the *gradient
analysis's own initialization*, never to row 0 by assumption and never to a
quantity averaged over initializations.

Undefined stays undefined. A class that is never a target has no recall, one
that is never guessed has no precision, and an empty conditional group has no
mean; those entries are NaN, never a silent zero. Every returned table reports
how many classes contribute to each statistic.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "class_frequency_table",
    "confidence_split",
    "correctness_table",
    "gradient_correctness_split",
    "logit_correction",
]


def _positions(record: Any, temperature: float) -> dict[str, np.ndarray]:
    """Per-position arrays for the gradient initialization, at one temperature.

    The gradient arrays are indexed by ``gradient_position_indices``, the flat
    ``window * block_size + offset``; the predictive per-position arrays are
    filled in evaluation order. They describe the same positions only when the
    gradient analysis covered all of them, which is checked here rather than
    assumed -- pairing a subset against a whole-experiment vector would be the
    silent mismatch this whole module is built to avoid.
    """

    if not record.has_temperature_gradient_analysis:
        raise ValueError("This record carries no temperature-conditioned gradients.")
    if not record.has_temperature_confidence_analysis:
        raise ValueError("This record carries no temperature-confidence analysis.")

    initialization = record.gradient_initialization_index
    gradient_index = record.gradient_temperature_index(temperature)
    confidence_index = record.temperature_index(temperature)

    indices = np.asarray(record.gradient_position_indices, dtype=np.int64)
    num_positions = int(record.metadata.get("analysis", {}).get("num_positions", 0))
    if not np.array_equal(indices, np.arange(indices.size)) or indices.size != num_positions:
        raise ValueError(
            "The gradient positions are not the complete evaluation grid in order, "
            "so they cannot be paired position-by-position with the predictive "
            "per-position arrays."
        )

    return {
        "targets": np.asarray(record.gradient_position_target_ids, dtype=np.int64),
        "greedy": np.asarray(record.gradient_position_greedy_ids, dtype=np.int64),
        "norms": np.asarray(
            record.gradient_temperature_position_norms[gradient_index], dtype=np.float64
        ),
        "target_probability": np.asarray(
            record.predictive_temperature_target_probabilities[
                initialization, confidence_index
            ],
            dtype=np.float64,
        ),
        "winner_probability": np.asarray(
            record.predictive_temperature_max_probabilities[
                initialization, confidence_index
            ],
            dtype=np.float64,
        ),
        "mean_token_probability": np.asarray(
            record.predictive_temperature_mean_token_probabilities[
                initialization, confidence_index
            ],
            dtype=np.float64,
        ),
    }


def _conditional_mean(
    values: np.ndarray, keys: np.ndarray, mask: np.ndarray, size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-class mean of ``values`` over a subset, NaN where the subset is empty."""

    counts = np.bincount(keys[mask], minlength=size).astype(np.int64)
    sums = np.bincount(keys[mask], weights=values[mask], minlength=size)
    means = np.full(size, np.nan, dtype=np.float64)
    measured = counts > 0
    means[measured] = sums[measured] / counts[measured]
    return means, counts


def class_frequency_table(record: Any, temperature: float = 1.0) -> dict[str, Any]:
    """Target, guess and probability frequencies, and the two biases.

    ``b_hard`` and ``b_soft`` are returned side by side precisely so a reader
    cannot substitute one for the other: only ``b_soft`` appears in the mean
    logit error, while ``b_hard`` describes realized decisions.
    """

    data = _positions(record, temperature)
    size = record.vocab_size
    total = float(data["targets"].size)

    target_counts = np.bincount(data["targets"], minlength=size).astype(np.int64)
    guess_counts = np.bincount(data["greedy"], minlength=size).astype(np.int64)
    target_fraction = target_counts / total
    guess_fraction = guess_counts / total
    mean_probability = data["mean_token_probability"]

    return {
        "temperature": float(temperature),
        "num_positions": int(total),
        "token_id": np.arange(size),
        "target_count": target_counts,
        "guess_count": guess_counts,
        "target_fraction": target_fraction,
        "guess_fraction": guess_fraction,
        "corpus_fraction": np.asarray(record.corpus_fractions, dtype=np.float64),
        "mean_probability": mean_probability,
        "hard_bias": guess_fraction - target_fraction,
        "soft_bias": mean_probability - target_fraction,
        "corpus_bias": guess_fraction - np.asarray(record.corpus_fractions, dtype=np.float64),
        "initialization_index": record.gradient_initialization_index,
    }


def correctness_table(record: Any, temperature: float = 1.0) -> dict[str, Any]:
    """Greedy TP/FN/FP, recall, precision, and the independence corrections.

    The reference is the **marginal-preserving independence null**, not ``1/K``.
    If targets and greedy predictions were independent while keeping both
    observed marginals, then ``P(greedy=i | y=i) = q_i`` and
    ``P(y=i | greedy=i) = f_i``. So ``q_i`` is the recall a class earns purely by
    being guessed often, and ``f_i`` the precision it earns purely by being
    frequent. Subtracting them is what separates recognition from mechanics:
    high recall for a heavily over-guessed class is not evidence of anything.

    Differences are used rather than ratios because a lift ratio is unstable for
    the rare classes that dominate a subword vocabulary.
    """

    data = _positions(record, temperature)
    size = record.vocab_size
    total = float(data["targets"].size)
    targets, greedy = data["targets"], data["greedy"]

    correct = targets == greedy
    target_counts = np.bincount(targets, minlength=size).astype(np.int64)
    guess_counts = np.bincount(greedy, minlength=size).astype(np.int64)
    true_positive = np.bincount(targets[correct], minlength=size).astype(np.int64)
    false_negative = target_counts - true_positive
    false_positive = guess_counts - true_positive

    target_fraction = target_counts / total
    guess_fraction = guess_counts / total

    recall = np.full(size, np.nan, dtype=np.float64)
    represented = target_counts > 0
    recall[represented] = true_positive[represented] / target_counts[represented]

    precision = np.full(size, np.nan, dtype=np.float64)
    guessed = guess_counts > 0
    precision[guessed] = true_positive[guessed] / guess_counts[guessed]

    return {
        "temperature": float(temperature),
        "num_positions": int(total),
        "correct_positions": int(correct.sum()),
        "micro_accuracy": float(correct.mean()),
        "independence_accuracy": float((target_fraction * guess_fraction).sum()),
        "true_positive": true_positive,
        "false_negative": false_negative,
        "false_positive": false_positive,
        "recall": recall,
        "precision": precision,
        "recall_independent": guess_fraction,
        "precision_independent": target_fraction,
        "delta_recall": recall - guess_fraction,
        "delta_precision": precision - target_fraction,
        "excess_true_positive": true_positive / total - target_fraction * guess_fraction,
        "num_represented": int(represented.sum()),
        "num_guessed": int(guessed.sum()),
        "num_true_positive_classes": int((true_positive > 0).sum()),
        "num_false_positive_classes": int((false_positive > 0).sum()),
    }


def gradient_correctness_split(record: Any, temperature: float) -> dict[str, Any]:
    """Split ``G_i`` into correctly- and incorrectly-assigned targets.

    Two different questions are kept apart:

    *intensity* -- ``G_i^TP`` and ``G_i^FN``, the mean per-example gradient norm
    within each group;

    *norm mass* -- ``M_i^TP`` and ``M_i^FN``, the group's summed norm divided by
    ``D``, i.e. frequency times mean intensity.

    Norm mass is **not** an SGD update magnitude: ``sum_d ||g_d||`` is not
    ``||sum_d g_d||``, and the two differ by however much the individual
    gradients cancel. It answers "how much gradient-norm does this group
    generate", not "how far does the optimizer move".
    """

    data = _positions(record, temperature)
    size = record.vocab_size
    total = float(data["targets"].size)
    targets, greedy, norms = data["targets"], data["greedy"], data["norms"]

    correct = targets == greedy
    intensity_tp, count_tp = _conditional_mean(norms, targets, correct, size)
    intensity_fn, count_fn = _conditional_mean(norms, targets, ~correct, size)

    mass_tp = np.bincount(targets[correct], weights=norms[correct], minlength=size) / total
    mass_fn = np.bincount(targets[~correct], weights=norms[~correct], minlength=size) / total

    denominator = mass_tp + mass_fn
    failed_fraction = np.full(size, np.nan, dtype=np.float64)
    positive = denominator > 0
    failed_fraction[positive] = mass_fn[positive] / denominator[positive]

    return {
        "temperature": float(temperature),
        "intensity_true_positive": intensity_tp,
        "intensity_false_negative": intensity_fn,
        "mass_true_positive": mass_tp,
        "mass_false_negative": mass_fn,
        "failed_mass_fraction": failed_fraction,
        "count_true_positive": count_tp,
        "count_false_negative": count_fn,
        "total_mass_true_positive": float(mass_tp.sum()),
        "total_mass_false_negative": float(mass_fn.sum()),
        "num_intensity_true_positive": int(np.isfinite(intensity_tp).sum()),
        "num_intensity_false_negative": int(np.isfinite(intensity_fn).sum()),
    }


def confidence_split(record: Any, temperature: float) -> dict[str, Any]:
    """Target confidence, wrong-winner confidence, and one margin.

    ``c_d`` is the probability the model gives the true next token and ``w_d``
    the probability it gives the token greedy actually selected. On a failed
    position the margin ``m_d = w_d - c_d`` is positive by construction; only
    this one margin is defined, to avoid a family of near-duplicate statistics.
    """

    data = _positions(record, temperature)
    size = record.vocab_size
    targets, greedy = data["targets"], data["greedy"]
    confidence, winner = data["target_probability"], data["winner_probability"]

    correct = targets == greedy
    confidence_tp, count_tp = _conditional_mean(confidence, targets, correct, size)
    confidence_fn, count_fn = _conditional_mean(confidence, targets, ~correct, size)
    winner_fn, _ = _conditional_mean(winner, targets, ~correct, size)
    margin_fn, _ = _conditional_mean(winner - confidence, targets, ~correct, size)

    return {
        "temperature": float(temperature),
        "confidence_true_positive": confidence_tp,
        "confidence_false_negative": confidence_fn,
        "winner_false_negative": winner_fn,
        "margin_false_negative": margin_fn,
        "count_true_positive": count_tp,
        "count_false_negative": count_fn,
        "position_confidence": confidence,
        "position_winner": winner,
        "position_correct": correct,
    }


def logit_correction(record: Any, temperature: float) -> dict[str, Any]:
    """Exact class-wise cross-entropy forces at the output layer.

    ``A_i`` is the upward pull on class ``i`` accumulated where it *is* the
    target, ``S_i`` the downward pressure accumulated where it is not::

        A_i = (1/D) sum_{y_d=i} (1 - s_{d,i}) / T
        S_i = (1/D) sum_{y_d!=i}      s_{d,i} / T

    Sign convention, stated once: ``S_i - A_i`` is the mean **logit-gradient
    component**, and ``A_i - S_i`` the **gradient-descent correction**. The
    exact identity is ``A_i - S_i = (f_i - pbar_i(T)) / T = -b_i^soft(T) / T``.

    Both are computed from persisted sufficient statistics: ``sum_{y_d=i}
    s_{d,i}`` is the target-probability sum over positions of class ``i``, and
    ``sum_d s_{d,i}`` is ``D * pbar_i``. The identity is therefore satisfied by
    construction here rather than being an independent check -- see the module
    tests, which validate the *inputs* (``sum_i pbar_i = 1``, the canonical row
    agreeing with ``mean_predicted_probabilities``) instead of re-deriving it.

    The splits answer where each force comes from. ``A_i^TP``/``A_i^FN`` divide
    the upward pull by whether the target was already ranked first.
    ``S_i^FP`` isolates the suppression generated at positions where ``i`` was
    the *incorrect winner*, the rest being ``S_i^other``.

    Note the asymmetry: TP/FN partitions positions by target class, so
    ``sum_i (TP_i + FN_i) = D``; FP/other is a non-target view in which one
    position contributes to many classes. Cross-class totals of ``A`` and ``S``
    are therefore not comparable partitions of the same examples.
    """

    data = _positions(record, temperature)
    size = record.vocab_size
    total = float(data["targets"].size)
    scale = total * float(temperature)
    targets, greedy = data["targets"], data["greedy"]
    confidence, winner = data["target_probability"], data["winner_probability"]
    mean_probability = data["mean_token_probability"]

    correct = targets == greedy
    target_counts = np.bincount(targets, minlength=size).astype(np.int64)
    confidence_sum = np.bincount(targets, weights=confidence, minlength=size)

    attraction = (target_counts - confidence_sum) / scale
    suppression = (total * mean_probability - confidence_sum) / scale

    attraction_tp = (
        np.bincount(targets[correct], minlength=size)
        - np.bincount(targets[correct], weights=confidence[correct], minlength=size)
    ) / scale
    attraction_fn = attraction - attraction_tp

    # At a position where greedy chose i and the target was not i, s_{d,i} is
    # exactly the winner probability already persisted.
    false_positive = ~correct
    suppression_fp = (
        np.bincount(
            greedy[false_positive], weights=winner[false_positive], minlength=size
        )
        / scale
    )
    suppression_other = suppression - suppression_fp

    def _fraction(part: np.ndarray, whole: np.ndarray) -> np.ndarray:
        out = np.full(size, np.nan, dtype=np.float64)
        nonzero = whole != 0
        out[nonzero] = part[nonzero] / whole[nonzero]
        return out

    return {
        "temperature": float(temperature),
        "attraction": attraction,
        "suppression": suppression,
        "attraction_true_positive": attraction_tp,
        "attraction_false_negative": attraction_fn,
        "suppression_false_positive": suppression_fp,
        "suppression_other": suppression_other,
        "failed_attraction_fraction": _fraction(attraction_fn, attraction),
        "false_positive_suppression_fraction": _fraction(suppression_fp, suppression),
        "net_correction": attraction - suppression,
        "soft_bias": mean_probability - target_counts / total,
    }
