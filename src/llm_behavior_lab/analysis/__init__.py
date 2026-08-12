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
    MeanWithError,
    PolicySummary,
    effective_support,
    js_divergence,
    mean_with_sem,
    persistent_absolute_gaps,
    ranked_profile,
    ranked_profiles,
    sampling_adequacy,
    shannon_entropy,
    summarize_policy,
    token_wise_absolute_gaps,
    top_two_gap,
    total_variation_distance,
    within_initialization_sampling_spread,
    zero_guess_counts,
)
from llm_behavior_lab.analysis.records import (
    RECORD_VERSION,
    InitializationExperimentRecord,
    load_record,
)

__all__ = [
    "InitializationExperimentRecord",
    "MeanWithError",
    "PolicySummary",
    "RECORD_VERSION",
    "effective_support",
    "js_divergence",
    "load_record",
    "mean_with_sem",
    "persistent_absolute_gaps",
    "ranked_profile",
    "ranked_profiles",
    "sampling_adequacy",
    "shannon_entropy",
    "summarize_policy",
    "token_wise_absolute_gaps",
    "top_two_gap",
    "total_variation_distance",
    "within_initialization_sampling_spread",
    "zero_guess_counts",
]
