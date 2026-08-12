"""Aggregation for the initialization-distribution experiment.

Two orderings appear in this module and must not be confused.

*Concentration* questions ignore token identity: each distribution is sorted
independently, and rank ``r`` of one curve has nothing to do with rank ``r`` of
another. :func:`ranked_profile` serves those.

*Mismatch* questions keep token identity: the difference is taken between the
same token ID in both distributions, and only the resulting gaps are sorted.
:func:`token_wise_absolute_gaps` enforces that order. Sorting first and
subtracting afterwards would answer a different, much weaker question, so the
two paths are separate functions rather than one function with a flag.

Variability is reported as the standard error across **independent model
initializations**. Stochastic sampling replicates are averaged inside an
initialization before that happens, so replicate noise is never mistaken for
initialization noise; :func:`within_initialization_sampling_spread` reports it
on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "MeanWithError",
    "PolicySummary",
    "ZeroFrequency",
    "corpus_observed_zero_guess_counts",
    "effective_support",
    "effective_supports",
    "eligible_view",
    "js_divergence",
    "mean_with_sem",
    "persistent_absolute_gaps",
    "ranked_profile",
    "ranked_profiles",
    "sampling_adequacy",
    "shannon_entropy",
    "summarize_policy",
    "token_wise_absolute_gaps",
    "policy_zero_frequency",
    "pooled_nucleus_zero_frequency",
    "support_summary",
    "top_two_gap",
    "total_variation_distance",
    "within_initialization_sampling_spread",
    "zero_guess_counts",
]

_EPS = 1e-12


def eligible_view(record: Any, array: np.ndarray) -> np.ndarray:
    """Restrict a token-indexed array to the predictive support.

    Structural tokens carry no corpus mass and can never be guessed, so leaving
    them in would pad every ranked profile with a tail of zeros and inflate the
    apparent vocabulary. Restricting keeps the statistics on the support the
    model is actually scored over. Canonical IDs are untouched; this is a view
    for computing, not a renumbering.
    """

    return np.asarray(array)[..., record.eligible_mask]


@dataclass(frozen=True)
class MeanWithError:
    """A mean profile and its standard error across initializations."""

    mean: np.ndarray
    sem: np.ndarray
    num_samples: int


def ranked_profile(vector: np.ndarray) -> np.ndarray:
    """Sort one distribution into descending order.

    Token identity is discarded: the result describes concentration only.
    """

    values = np.asarray(vector, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("ranked_profile expects a one-dimensional vector.")
    return np.sort(values)[::-1]


def ranked_profiles(matrix: np.ndarray) -> np.ndarray:
    """Sort each row of ``[S, V]`` independently into descending order."""

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("ranked_profiles expects a two-dimensional array.")
    return np.sort(values, axis=1)[:, ::-1]


def mean_with_sem(matrix: np.ndarray) -> MeanWithError:
    """Average over the leading axis and report the standard error.

    ``SEM = s / sqrt(N)`` with the sample standard deviation (``ddof=1``). A
    single initialization has no estimable spread, so the error is reported as
    zero rather than ``nan``; callers should show the sample count alongside.
    """

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("mean_with_sem expects a two-dimensional array.")
    num_samples = values.shape[0]
    mean = values.mean(axis=0)
    if num_samples < 2:
        return MeanWithError(mean=mean, sem=np.zeros_like(mean), num_samples=num_samples)
    sem = values.std(axis=0, ddof=1) / np.sqrt(num_samples)
    return MeanWithError(mean=mean, sem=sem, num_samples=num_samples)


def token_wise_absolute_gaps(guess_fractions: np.ndarray, corpus_fractions: np.ndarray) -> np.ndarray:
    """Absolute per-token mismatch ``|q[s,i] - p[i]|``, computed before ranking.

    Args:
        guess_fractions: ``[S, V]`` guess fractions, one row per initialization.
        corpus_fractions: ``[V]`` empirical corpus fractions.

    Returns:
        ``[S, V]`` gaps still indexed by token ID. Rank them afterwards with
        :func:`ranked_profiles` if a rank profile is wanted.
    """

    guesses = np.asarray(guess_fractions, dtype=np.float64)
    corpus = np.asarray(corpus_fractions, dtype=np.float64)
    if guesses.ndim != 2:
        raise ValueError("guess_fractions must be two-dimensional [initializations, vocab].")
    if corpus.ndim != 1 or corpus.shape[0] != guesses.shape[1]:
        raise ValueError("corpus_fractions must be one-dimensional and share the vocabulary axis.")
    return np.abs(guesses - corpus[None, :])


def persistent_absolute_gaps(
    guess_fractions: np.ndarray,
    corpus_fractions: np.ndarray,
) -> np.ndarray:
    """Mismatch of the initialization-averaged guess distribution.

    ``|E_s[q[s,i]] - p[i]|``, still indexed by token ID.

    Read together with the mean of :func:`token_wise_absolute_gaps`, this
    separates two very different situations. When the typical per-initialization
    gap is large but this one is small, initializations disagree about *which*
    tokens are over-selected and the excess averages away. When both are large,
    the same tokens are favoured every time, which is a systematic property of
    the architecture and initialization scheme rather than of one draw.
    """

    guesses = np.asarray(guess_fractions, dtype=np.float64)
    corpus = np.asarray(corpus_fractions, dtype=np.float64)
    if guesses.ndim != 2:
        raise ValueError("guess_fractions must be two-dimensional [initializations, vocab].")
    if corpus.ndim != 1 or corpus.shape[0] != guesses.shape[1]:
        raise ValueError("corpus_fractions must be one-dimensional and share the vocabulary axis.")
    return np.abs(guesses.mean(axis=0) - corpus)


def total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    """``TV(p, q) = 0.5 * sum_i |p_i - q_i|``, bounded in ``[0, 1]``."""

    left = np.asarray(p, dtype=np.float64)
    right = np.asarray(q, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("total_variation_distance requires matching shapes.")
    return float(0.5 * np.abs(left - right).sum())


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    p_safe = np.clip(p, _EPS, None)
    q_safe = np.clip(q, _EPS, None)
    return float((p_safe * (np.log(p_safe) - np.log(q_safe))).sum())


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in nats; bounded above by ``ln 2``."""

    left = np.asarray(p, dtype=np.float64)
    right = np.asarray(q, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("js_divergence requires matching shapes.")
    midpoint = 0.5 * (left + right)
    return 0.5 * _kl(left, midpoint) + 0.5 * _kl(right, midpoint)


def shannon_entropy(p: np.ndarray) -> float:
    """Shannon entropy in nats."""

    values = np.clip(np.asarray(p, dtype=np.float64), _EPS, None)
    return float(-(values * np.log(values)).sum())


def effective_support(p: np.ndarray) -> float:
    """``exp(H(p))``: the size of a uniform distribution with the same entropy.

    Easier to read than entropy itself, because it is expressed in tokens and is
    directly comparable with the vocabulary size.
    """

    return float(np.exp(shannon_entropy(p)))


def zero_guess_counts(guess_fractions: np.ndarray) -> np.ndarray:
    """Number of valid tokens never selected, one entry per initialization.

    Strongly dependent on how many positions were evaluated: with fewer
    positions than vocabulary entries, most tokens are unreachable regardless of
    the model. Always report this next to the evaluated-position count.
    """

    guesses = np.asarray(guess_fractions, dtype=np.float64)
    if guesses.ndim != 2:
        raise ValueError("zero_guess_counts expects [initializations, vocab].")
    return (guesses <= 0.0).sum(axis=1)


def top_two_gap(vector: np.ndarray) -> float:
    """``p_(1) - p_(2)``: how far the most frequent entry leads the second."""

    ranked = ranked_profile(vector)
    if ranked.shape[0] < 2:
        raise ValueError("top_two_gap needs at least two entries.")
    return float(ranked[0] - ranked[1])


def within_initialization_sampling_spread(record: Any) -> dict[str, float]:
    """Quantify stochastic-sampling noise *inside* one initialization.

    For every initialization the replicate-wise distance to the corpus
    distribution is computed, and the spread of those replicate values is
    averaged over initializations. Comparing it with the between-initialization
    spread of the same quantity shows which source of randomness dominates.
    """

    replicate_fractions = record.nucleus_replicate_fractions
    corpus = record.corpus_fractions
    per_replicate = np.array(
        [
            [total_variation_distance(corpus, replicate) for replicate in initialization]
            for initialization in replicate_fractions
        ]
    )
    within_std = (
        per_replicate.std(axis=1, ddof=1) if per_replicate.shape[1] > 1 else np.zeros(len(per_replicate))
    )
    initialization_means = per_replicate.mean(axis=1)
    between_std = (
        float(initialization_means.std(ddof=1)) if initialization_means.shape[0] > 1 else 0.0
    )
    return {
        "mean_within_initialization_std_tv": float(np.mean(within_std)),
        "between_initialization_std_tv": between_std,
        "mean_tv_to_corpus": float(initialization_means.mean()),
        "num_replicates": int(per_replicate.shape[1]),
        "num_initializations": int(per_replicate.shape[0]),
    }


@dataclass(frozen=True)
class ZeroFrequency:
    """How much of the eligible vocabulary a policy never selected.

    Comparable across policies only when both are measured over the **same
    number of draws**, which is what :func:`policy_zero_frequency` guarantees.
    """

    counts: np.ndarray
    fractions: np.ndarray
    draws_per_measurement: int
    eligible_vocab_size: int


def _zero_counts(counts: np.ndarray) -> np.ndarray:
    """Count zero entries along the last axis."""

    return (np.asarray(counts) == 0).sum(axis=-1)


def policy_zero_frequency(record: Any, policy: str) -> ZeroFrequency:
    """Zero-frequency support for one policy, measured at ``N`` draws.

    Greedy makes exactly one selection per evaluation position, so it has ``N``
    draws. A single nucleus replicate also has ``N`` draws, but the ``R``
    replicates *pooled* have ``R*N``, and more draws mean more chances to reach
    a rare token. Pooling therefore flatters nucleus coverage and makes the two
    policies incomparable -- badly so at a 32k vocabulary, where zero-frequency
    counts are dominated by how many draws were available.

    The nucleus value returned here is the **per-replicate** one: each replicate
    is scored on its own ``N`` draws and the counts are averaged within the
    initialization. Both policies are then measured at the same sample size.
    :func:`pooled_nucleus_zero_frequency` keeps the pooled figure available as a
    separately named coverage diagnostic.

    Returns:
        One :class:`ZeroFrequency` with per-initialization counts and fractions.
    """

    eligible_size = record.eligible_vocab_size
    draws = int(record.metadata.get("num_positions", 0))

    if policy == "greedy":
        counts = _zero_counts(eligible_view(record, record.greedy_counts)).astype(np.float64)
    elif policy == "nucleus":
        # [S, R] -> mean over replicates -> [S]. Averaging counts, not counting
        # zeros of an average, is the whole correction.
        per_replicate = _zero_counts(eligible_view(record, record.nucleus_counts))
        counts = per_replicate.astype(np.float64).mean(axis=1)
    else:
        raise ValueError(f"Unknown policy {policy!r}; expected 'greedy' or 'nucleus'.")

    return ZeroFrequency(
        counts=counts,
        fractions=counts / eligible_size,
        draws_per_measurement=draws,
        eligible_vocab_size=eligible_size,
    )


def pooled_nucleus_zero_frequency(record: Any) -> ZeroFrequency:
    """Eligible tokens never selected by **any** nucleus replicate.

    A coverage diagnostic over ``R*N`` pooled draws. Useful for asking what the
    policy can reach given more attempts, and **not** comparable with the greedy
    figure, which has ``N``. Never display the two side by side; use
    :func:`policy_zero_frequency` for the headline comparison.
    """

    pooled = eligible_view(record, record.nucleus_counts).sum(axis=1)
    counts = _zero_counts(pooled).astype(np.float64)
    return ZeroFrequency(
        counts=counts,
        fractions=counts / record.eligible_vocab_size,
        draws_per_measurement=int(record.metadata.get("num_positions", 0))
        * record.num_replicates,
        eligible_vocab_size=record.eligible_vocab_size,
    )


def corpus_observed_zero_guess_counts(record: Any, policy: str) -> np.ndarray:
    """Eligible tokens that occur in the corpus but were never guessed.

    A secondary diagnostic. The headline zero-frequency statistic is taken over
    the whole eligible support, because it describes the breadth of the model's
    guess distribution; this one asks the narrower question of how much of the
    text the model never proposes. With a 32k vocabulary the two differ a lot,
    since most of the vocabulary never appears in a modest corpus.
    """

    observed = eligible_view(record, record.corpus_counts) > 0
    if policy == "greedy":
        guessed = eligible_view(record, record.greedy_counts) > 0
    elif policy == "nucleus":
        guessed = eligible_view(record, record.nucleus_counts).sum(axis=1) > 0
    else:
        raise ValueError(f"Unknown policy {policy!r}; expected 'greedy' or 'nucleus'.")
    return (observed[None, :] & ~guessed).sum(axis=1).astype(np.float64)


def effective_supports(record: Any, policy: str) -> np.ndarray:
    """Per-initialization effective support ``exp(H(q))`` of the guess frequencies."""

    guesses = eligible_view(record, record.policy_fractions(policy))
    return np.array([effective_support(row) for row in guesses])


def support_summary(record: Any) -> dict[str, Any]:
    """The vocabulary sizes and effective supports, kept distinct.

    Four different questions, four different numbers:

    * ``vocab_size`` -- how many tokens the tokenizer defines;
    * ``eligible_vocab_size`` -- how many a model may be scored on;
    * ``corpus_observed_support`` -- how many actually occur in the split;
    * ``corpus_effective_support`` -- how broadly the corpus mass is spread.

    The last is not a count of anything. ``exp(H)`` is the size of a uniform
    distribution with the same entropy, so it can be far below the observed
    support when a few tokens dominate.
    """

    corpus = eligible_view(record, record.corpus_fractions)
    selected = eligible_view(record, record.selected_target_fractions)
    return {
        "vocab_size": record.vocab_size,
        "eligible_vocab_size": record.eligible_vocab_size,
        "excluded_special_token_ids": record.special_token_ids.tolist(),
        "corpus_observed_support": record.corpus_observed_support,
        "corpus_effective_support": effective_support(corpus),
        "selected_effective_support": effective_support(selected),
        "selected_observed_support": int(
            (eligible_view(record, record.selected_target_counts) > 0).sum()
        ),
    }


@dataclass(frozen=True)
class PolicySummary:
    """Scalar summary of one guessing policy across initializations.

    Every scalar is computed per initialization and only then averaged, so each
    reported error is the spread across initializations.

    ``zero_frequency_*`` is always the per-replicate comparable figure for
    nucleus, measured at the same draw count as greedy. The pooled figure lives
    in ``pooled_zero_frequency_count_mean`` under a name that cannot be mistaken
    for it.
    """

    policy: str
    num_initializations: int
    num_positions: int
    eligible_vocab_size: int
    total_variation_mean: float
    total_variation_sem: float
    js_divergence_mean: float
    entropy_mean: float
    effective_support_mean: float
    effective_support_sem: float
    zero_frequency_count_mean: float
    zero_frequency_count_sem: float
    zero_frequency_fraction_mean: float
    zero_frequency_fraction_sem: float
    zero_frequency_draws: int
    pooled_zero_frequency_count_mean: float | None
    pooled_zero_frequency_draws: int | None
    corpus_observed_zero_guess_mean: float
    top_two_gap_mean: float
    top_two_gap_sem: float
    persistent_gap_max: float
    typical_gap_max_mean: float

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping."""

        return {field: getattr(self, field) for field in self.__dataclass_fields__}


def _mean_sem(values: np.ndarray) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    if array.shape[0] < 2:
        return mean, 0.0
    return mean, float(array.std(ddof=1) / np.sqrt(array.shape[0]))


def summarize_policy(record: Any, policy: str) -> PolicySummary:
    """Compute the scalar summary for one guessing policy.

    All statistics are taken over the eligible predictive support, so structural
    tokens neither pad the vocabulary nor count as tokens the model failed to
    guess.
    """

    guesses = eligible_view(record, record.policy_fractions(policy))
    corpus = eligible_view(record, record.corpus_fractions)

    tv = np.array([total_variation_distance(corpus, row) for row in guesses])
    js = np.array([js_divergence(corpus, row) for row in guesses])
    entropy = np.array([shannon_entropy(row) for row in guesses])
    support = effective_supports(record, policy)
    gaps = np.array([top_two_gap(row) for row in guesses])

    zero = policy_zero_frequency(record, policy)
    pooled = pooled_nucleus_zero_frequency(record) if policy == "nucleus" else None

    typical_gap_max = ranked_profiles(token_wise_absolute_gaps(guesses, corpus))[:, 0]
    persistent = persistent_absolute_gaps(guesses, corpus)

    tv_mean, tv_sem = _mean_sem(tv)
    support_mean, support_sem = _mean_sem(support)
    zero_count_mean, zero_count_sem = _mean_sem(zero.counts)
    zero_fraction_mean, zero_fraction_sem = _mean_sem(zero.fractions)
    gap_mean, gap_sem = _mean_sem(gaps)

    return PolicySummary(
        policy=policy,
        num_initializations=record.num_initializations,
        num_positions=int(record.metadata.get("num_positions", 0)),
        eligible_vocab_size=record.eligible_vocab_size,
        total_variation_mean=tv_mean,
        total_variation_sem=tv_sem,
        js_divergence_mean=float(js.mean()),
        entropy_mean=float(entropy.mean()),
        effective_support_mean=support_mean,
        effective_support_sem=support_sem,
        zero_frequency_count_mean=zero_count_mean,
        zero_frequency_count_sem=zero_count_sem,
        zero_frequency_fraction_mean=zero_fraction_mean,
        zero_frequency_fraction_sem=zero_fraction_sem,
        zero_frequency_draws=zero.draws_per_measurement,
        pooled_zero_frequency_count_mean=(
            None if pooled is None else float(pooled.counts.mean())
        ),
        pooled_zero_frequency_draws=(
            None if pooled is None else pooled.draws_per_measurement
        ),
        corpus_observed_zero_guess_mean=float(
            corpus_observed_zero_guess_counts(record, policy).mean()
        ),
        top_two_gap_mean=gap_mean,
        top_two_gap_sem=gap_sem,
        persistent_gap_max=float(persistent.max()),
        typical_gap_max_mean=float(typical_gap_max.mean()),
    )


def sampling_adequacy(record: Any) -> dict[str, Any]:
    """Judge whether the analyzed positions represent the whole split.

    A statement about the *corpus sample*, not about the model. A large distance
    here means the evaluation-position count is too small (or too unevenly
    placed) for any model comparison built on it to be trusted, and no number of
    extra initializations repairs that.

    Changing the tokenizer invalidates any previous verdict: a subword
    vocabulary changes the token count, the support, and the frequency
    structure all at once, so adequacy has to be re-established from scratch.
    """

    corpus = eligible_view(record, record.corpus_fractions)
    selected = eligible_view(record, record.selected_target_fractions)
    selected_counts = eligible_view(record, record.selected_target_counts)
    represented = int((selected_counts > 0).sum())
    summary = support_summary(record)
    return {
        "total_variation_distance": total_variation_distance(corpus, selected),
        "js_divergence": js_divergence(corpus, selected),
        "num_positions": int(selected_counts.sum()),
        "vocab_size": summary["vocab_size"],
        "eligible_vocab_size": summary["eligible_vocab_size"],
        "corpus_observed_support": summary["corpus_observed_support"],
        "corpus_effective_support": summary["corpus_effective_support"],
        "selected_effective_support": summary["selected_effective_support"],
        "corpus_tokens": int(eligible_view(record, record.corpus_counts).sum()),
        "tokens_represented_in_selection": represented,
        "fraction_of_vocabulary_represented": represented / summary["eligible_vocab_size"],
        "corpus_mass_covered_by_selection": float(corpus[selected_counts > 0].sum()),
    }
