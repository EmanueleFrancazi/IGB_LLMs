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
    ranked_profile_with_error,
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
from llm_behavior_lab.analysis.gradients import (
    TokenGradientSummary,
    gradient_guess_correlations,
    gradient_guess_table,
    gradient_observable_summary,
    gradient_vector_split,
    mean_probability_gradient_table,
    nucleus_gradient_table,
    spearman_rho,
    temperature_gradient_summary,
    temperature_gradient_table,
    token_gradient_norms,
)
from llm_behavior_lab.analysis.gradient_clustering import (
    GROUPINGS,
    class_similarity_matrix,
    clustering_summary,
    gradient_clustering,
    has_gradient_sketches,
    unit_sketches,
)
from llm_behavior_lab.analysis.gradient_cross_partition import (
    contingency_summary,
    cross_identity_null,
    cross_partition_matrix,
    mixture_reconstruction,
    pooled_cross_statistic,
)
from llm_behavior_lab.analysis.scale_comparison import greedy_scale_comparison
from llm_behavior_lab.analysis.predictive import (
    PROBABILITY_QUANTILES,
    REPORTED_RANKS,
    CUMULATIVE_DEPTHS,
    TEMPERATURE_RANKS,
    TOP_K_MASSES,
    cumulative_order_comparison,
    ranked_mean_token_probabilities,
    max_probability_summary,
    predictive_probability_summary,
    ranked_probability_profile,
    target_probability_summary,
    temperature_confidence_summary,
    temperature_ranked_profiles,
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
    "TokenGradientSummary",
    "UniformNullSummary",
    "PROBABILITY_QUANTILES",
    "REPORTED_RANKS",
    "max_probability_summary",
    "predictive_probability_summary",
    "ranked_probability_profile",
    "target_probability_summary",
    "CUMULATIVE_DEPTHS",
    "TEMPERATURE_RANKS",
    "TOP_K_MASSES",
    "cumulative_order_comparison",
    "greedy_scale_comparison",
    "ranked_mean_token_probabilities",
    "temperature_confidence_summary",
    "temperature_ranked_profiles",
    "gradient_guess_correlations",
    "gradient_guess_table",
    "gradient_observable_summary",
    "gradient_vector_split",
    "GROUPINGS",
    "contingency_summary",
    "cross_identity_null",
    "cross_partition_matrix",
    "mixture_reconstruction",
    "pooled_cross_statistic",
    "class_similarity_matrix",
    "clustering_summary",
    "gradient_clustering",
    "has_gradient_sketches",
    "unit_sketches",
    "mean_probability_gradient_table",
    "nucleus_gradient_table",
    "spearman_rho",
    "temperature_gradient_summary",
    "temperature_gradient_table",
    "token_gradient_norms",
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
    "ranked_profile_with_error",
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
