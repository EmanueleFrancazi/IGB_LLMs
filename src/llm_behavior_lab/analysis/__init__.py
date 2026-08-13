"""Reusable analysis of persisted experiment results.

This package is the read side of the project: experiments write records, and
everything here consumes them. It depends only on NumPy, so a finished
experiment can be re-analyzed without PyTorch, without a GPU, and without
rerunning any model.

:mod:`llm_behavior_lab.analysis.figures` additionally needs ``matplotlib`` and is
therefore **not** imported here. Import it explicitly once the optional
dependency is installed::

    python3 -m pip install -e ".[analysis]"
    from llm_behavior_lab.analysis.figures import generate_all_figures

The split between the three modules is the one that keeps this extensible:
:mod:`records` owns the persisted format, :mod:`aggregation` owns every
statistic, and :mod:`figures` only draws what aggregation has already computed.
A new measurement is added to ``aggregation`` and used by a figure; it never
appears in a figure alone.
"""

from llm_behavior_lab.analysis.aggregation import (
    CONDITION_PAIRS,
    MeanWithError,
    PolicySummary,
    ZeroFrequency,
    corpus_observed_zero_guess_counts,
    effective_support,
    effective_supports,
    eligible_view,
    js_divergence,
    mean_with_sem,
    paired_concentration_differences,
    paired_condition_distances,
    persistent_absolute_gaps,
    policy_zero_frequency,
    pooled_nucleus_zero_frequency,
    ranked_profile,
    ranked_profiles,
    sampling_adequacy,
    shannon_entropy,
    summarize_policy,
    support_summary,
    token_wise_absolute_gaps,
    top_two_gap,
    total_variation_distance,
    within_initialization_sampling_spread,
    zero_guess_counts,
)
from llm_behavior_lab.analysis.nulls import (
    DEFAULT_NULL_REPLICATES,
    UniformNullSummary,
    expected_occupancy_counts,
    expected_zero_frequency_count,
    expected_zero_frequency_fraction,
    simulate_uniform_null,
)
from llm_behavior_lab.analysis.transition import (
    TRANSITION_METRICS,
    ranked_distance_to_greedy,
    ranked_distance_to_uniform,
    sweep_condition_summary,
    sweep_summary,
)
from llm_behavior_lab.analysis.records import (
    RECORD_VERSION,
    InitializationExperimentRecord,
    load_record,
)

__all__ = [
    "CONDITION_PAIRS",
    "TRANSITION_METRICS",
    "ranked_distance_to_greedy",
    "ranked_distance_to_uniform",
    "sweep_condition_summary",
    "sweep_summary",
    "DEFAULT_NULL_REPLICATES",
    "InitializationExperimentRecord",
    "UniformNullSummary",
    "expected_occupancy_counts",
    "expected_zero_frequency_count",
    "expected_zero_frequency_fraction",
    "paired_concentration_differences",
    "paired_condition_distances",
    "simulate_uniform_null",
    "MeanWithError",
    "PolicySummary",
    "RECORD_VERSION",
    "ZeroFrequency",
    "corpus_observed_zero_guess_counts",
    "effective_support",
    "effective_supports",
    "eligible_view",
    "js_divergence",
    "load_record",
    "mean_with_sem",
    "persistent_absolute_gaps",
    "policy_zero_frequency",
    "pooled_nucleus_zero_frequency",
    "ranked_profile",
    "ranked_profiles",
    "sampling_adequacy",
    "shannon_entropy",
    "summarize_policy",
    "support_summary",
    "token_wise_absolute_gaps",
    "top_two_gap",
    "total_variation_distance",
    "within_initialization_sampling_spread",
    "zero_guess_counts",
]
