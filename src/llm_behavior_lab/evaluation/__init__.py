"""Evaluation and analysis utilities for model-output behavior."""

from llm_behavior_lab.evaluation.gradient_norms import (
    GradientNormResult,
    GradientTrendFit,
    LayerGradientNorm,
    compute_per_layer_gradient_norms,
    fit_log_gradient_trend,
    gradient_norm_result_to_dict,
    save_gradient_norm_result,
)
from llm_behavior_lab.evaluation.output_stats import (
    OutputDistributionSummary,
    entropy_from_probabilities,
    logits_to_probabilities,
    summarize_output_distribution,
    top1_probability_values,
    top1_token_ids,
    topk_probability_mass,
)
from llm_behavior_lab.evaluation.token_frequency import (
    TokenFrequencySummary,
    TokenProbabilityGap,
    average_predicted_probabilities,
    empirical_token_counts,
    empirical_token_frequencies,
    js_divergence,
    kl_divergence,
    top_probability_gaps,
    top_token_frequencies,
)
from llm_behavior_lab.evaluation.untrained_analysis import (
    TopKPositionExample,
    UntrainedAnalysisResult,
    analyze_untrained_outputs,
    collect_topk_examples,
    summarize_top1_predictions,
)

__all__ = [
    "GradientNormResult",
    "GradientTrendFit",
    "LayerGradientNorm",
    "OutputDistributionSummary",
    "TokenFrequencySummary",
    "TokenProbabilityGap",
    "TopKPositionExample",
    "UntrainedAnalysisResult",
    "average_predicted_probabilities",
    "analyze_untrained_outputs",
    "collect_topk_examples",
    "compute_per_layer_gradient_norms",
    "empirical_token_counts",
    "empirical_token_frequencies",
    "entropy_from_probabilities",
    "fit_log_gradient_trend",
    "gradient_norm_result_to_dict",
    "js_divergence",
    "kl_divergence",
    "logits_to_probabilities",
    "save_gradient_norm_result",
    "summarize_output_distribution",
    "summarize_top1_predictions",
    "top1_probability_values",
    "top1_token_ids",
    "top_probability_gaps",
    "top_token_frequencies",
    "topk_probability_mass",
]
