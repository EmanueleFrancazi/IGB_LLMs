"""Console report for an initialization-distribution run.

The experiment runner used to carry this reporting inline, which left one
function owning corpus preparation, measurement orchestration, record assembly,
persistence *and* presentation. Presentation lives here instead, one function
per section of the report, so each block of the scientific summary can be read
and checked on its own.

Every function only formats what it is given, or reads the finished record.
None of them compute a new scientific quantity, and none of them write a file:
the summaries come from the same analysis functions the runner already used,
and persistence stays with the runner. The wording, ordering and number
formatting are the run's observable output and are reproduced exactly.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .aggregation import (
    paired_concentration_differences,
    paired_condition_distances,
    pooled_nucleus_zero_frequency,
    support_summary,
)
from .records import InitializationExperimentRecord

__all__ = [
    "report_gradient_norms",
    "report_input_structure",
    "report_policy_summaries",
    "report_predictive_probabilities",
    "report_runtime",
    "report_sampling_adequacy",
    "report_sampling_variability",
    "report_support",
    "report_temperature_sweep",
    "report_uniform_null",
]


def report_support(record: InitializationExperimentRecord) -> None:
    """Four different questions about vocabulary size, four numbers."""

    support = support_summary(record)
    print("\nVocabulary and support (four different questions, four numbers):")
    print(f"  V full                : {support['vocab_size']:,}")
    print(f"  V eligible            : {support['eligible_vocab_size']:,}")
    print(f"  V corpus-observed     : {support['corpus_observed_support']:,}")
    print(f"  corpus effective N_eff: {support['corpus_effective_support']:,.2f}")


def report_sampling_adequacy(adequacy: Mapping[str, Any]) -> None:
    """How well the selected positions stand in for the whole split."""

    print("\nEmpirical sampling adequacy of the evaluation positions:")
    print(f"  TV(split, selected targets): {adequacy['total_variation_distance']:.6f}")
    print(f"  JS(split, selected targets): {adequacy['js_divergence']:.6f}")
    print(f"  selected effective N_eff   : {adequacy['selected_effective_support']:,.2f}")
    print(
        f"  eligible vocabulary represented in targets: "
        f"{adequacy['tokens_represented_in_selection']:,}/{adequacy['eligible_vocab_size']:,}"
    )


def report_policy_summaries(
    record: InitializationExperimentRecord, summaries: Mapping[str, Any]
) -> None:
    """Greedy and nucleus side by side, plus the pooled nucleus diagnostic."""

    for policy, summary in summaries.items():
        print(
            f"\n{policy} guessing policy "
            f"(mean ± SEM across {summary.num_initializations} initializations):"
        )
        print(f"  TV(corpus, guesses)  : {summary.total_variation_mean:.6f} ± {summary.total_variation_sem:.6f}")
        print(
            f"  effective support    : {summary.effective_support_mean:,.3f}"
            f" ± {summary.effective_support_sem:,.3f} tokens"
        )
        print(
            f"  zero-frequency tokens: {summary.zero_frequency_count_mean:,.2f}"
            f" ± {summary.zero_frequency_count_sem:,.2f}"
            f"  ({summary.zero_frequency_fraction_mean:.2%})"
            f"  over {summary.zero_frequency_draws:,} draws"
            + ("  [per replicate]" if policy == "nucleus" else "")
        )
        print(f"  q(1) - q(2)          : {summary.top_two_gap_mean:.6f} ± {summary.top_two_gap_sem:.6f}")
        print(f"  max typical |q-p|    : {summary.typical_gap_max_mean:.6f}")
        print(f"  max persistent       : {summary.persistent_gap_max:.6f}")
        print(f"  corpus tokens never guessed: {summary.corpus_observed_zero_guess_mean:,.1f}")

    pooled = pooled_nucleus_zero_frequency(record)
    print(
        f"\nNucleus pooled coverage diagnostic (NOT comparable with greedy): "
        f"{pooled.counts.mean():,.1f} tokens unreached over "
        f"{pooled.draws_per_measurement:,} pooled draws"
    )


def report_runtime(elapsed: float, peak_rss_mib: float, parameter_count: int) -> None:
    """Cost of the run, not a scientific result."""

    print(f"\nRuntime: {elapsed:.1f}s | peak RSS: {peak_rss_mib:,.0f} MiB")
    print(f"Model parameters: {parameter_count:,}")


def report_sampling_variability(sampling_spread: Mapping[str, Any]) -> None:
    """At R=1 the within-initialization spread is not estimable; say so."""

    print("\nStochastic sampling variability (nucleus policy):")
    if sampling_spread.get("estimated"):
        print(
            f"  mean within-initialization std of TV: "
            f"{sampling_spread['mean_within_initialization_std_tv']:.6f}"
        )
        print(
            f"  between-initialization std of TV: "
            f"{sampling_spread['between_initialization_std_tv']:.6f}"
        )
    else:
        print(f"  within-initialization stochastic variability: {sampling_spread['reason']}")
        print(
            "  the nucleus SEM across initializations therefore describes the combined "
            "initialization + one fixed sampling realization"
        )


def report_uniform_null(null_summary: Any) -> None:
    """The finite-D uniform categorical reference."""

    described = null_summary.as_dict()
    print(
        f"\nUniform categorical output null "
        f"(K={described['eligible_vocab_size']:,}, D={described['num_draws']:,}, "
        f"M={described['monte_carlo_replicates']} Monte Carlo replicates):"
    )
    print(
        f"  zero-frequency: {described['zero_frequency_count_mean']:,.1f} "
        f"({described['zero_frequency_fraction_mean']:.2%}); "
        f"analytic K(1-1/K)^D = {described['analytic_zero_frequency_count']:,.1f}"
    )
    print(
        f"  effective support: {described['effective_support_mean']:,.1f} "
        f"[{described['effective_support_mc_low']:,.1f}, "
        f"{described['effective_support_mc_high']:,.1f}] Monte Carlo interval"
    )
    print(f"  top1-top2 gap: {described['top_two_gap_mean']:.6f}")


def report_input_structure(record: InitializationExperimentRecord) -> None:
    """Paired real / shuffled / Gaussian comparison, within each initialization."""

    print("\nInput-structure comparison (paired within each initialization):")
    for policy in ("greedy", "nucleus"):
        distances = paired_condition_distances(record, policy)
        differences = paired_concentration_differences(record, policy)
        print(f"  {policy}:")
        for name, values in distances["pairs"].items():
            label = name.replace("tv_", "").replace("_vs_", " vs ")
            print(f"    same-token TV {label}: {values['mean']:.6f} ± {values['sem']:.6f}")
        for pair, measures in differences["pairs"].items():
            support = measures["effective_support"]
            zero = measures["zero_frequency_fraction"]
            print(
                f"    delta {pair}: N_eff {support['mean']:+,.2f} ± {support['sem']:,.2f}, "
                f"zero-frac {zero['mean']:+.4f} ± {zero['sem']:.4f}"
            )


def report_temperature_sweep(
    summary: Mapping[str, Any], sweep_temperatures: Sequence[float]
) -> None:
    """The crossover from the greedy regime toward the finite-D uniform null.

    The caller owns persisting ``summary``; this only prints it.
    """

    print("\nTemperature transition (real input, mean across initializations):")
    print(
        f"  {'T':>6} {'N_eff':>10} {'/null':>7} {'zero%':>7} {'TV(corp)':>9} "
        f"{'TVrk_greedy':>12} {'TVrk_unif':>10} {'agree':>7}"
    )
    metrics = summary["conditions"]["real"]["metrics"]
    for index, temperature in enumerate(sweep_temperatures):
        print(
            f"  {temperature:>6g} {metrics['effective_support']['mean'][index]:>10,.1f}"
            f" {metrics['effective_support_over_null']['mean'][index]:>7.3f}"
            f" {metrics['zero_frequency_fraction']['mean'][index]:>6.1%}"
            f" {metrics['tv_to_corpus']['mean'][index]:>9.4f}"
            f" {metrics['tv_rank_to_greedy']['mean'][index]:>12.4f}"
            f" {metrics['tv_rank_to_uniform']['mean'][index]:>10.4f}"
            f" {metrics['agreement_with_greedy']['mean'][index]:>7.3f}"
        )
    others = [name for name in summary["conditions"] if name != "real"]
    if others:
        print("  N_eff / N_eff(null) by input condition:")
        for name in ("real", *others):
            ratios = summary["conditions"][name]["metrics"]["effective_support_over_null"]["mean"]
            print(f"    {name:<9} " + "  ".join(f"{value:.3f}" for value in ratios))


def report_predictive_probabilities(record: InitializationExperimentRecord) -> None:
    """The raw T = 1 predictive vectors, before top-p and before sampling."""

    from .predictive import predictive_probability_summary

    predictive = predictive_probability_summary(record)
    uniform = predictive["uniform_probability"]
    print(
        "\nRaw predictive distribution (T = 1, eligible support, before top-p "
        "and before any sampling decision):"
    )
    print(
        f"  ranked WITHIN each position, then averaged over positions "
        f"(not figure 1's across-position ranking)"
    )
    ranks = predictive["ranked_profile_at_rank"]
    print("  " + "  ".join(f"P({rank})={value:.4g}" for rank, value in ranks.items()))
    print(
        f"  uniform 1/K = {uniform:.4g};  rank 1 is "
        f"{predictive['rank1_over_uniform']:.2f}x uniform;  "
        f"profile dynamic range {predictive['profile_dynamic_range']:.4g}"
    )
    pooled = predictive["max_probability"]["pooled"]
    print(
        f"  p_max: min {pooled['p00']:.4g}, p05 {pooled['p05']:.4g}, "
        f"p25 {pooled['p25']:.4g}, median {pooled['p50']:.4g}, "
        f"p75 {pooled['p75']:.4g}, p95 {pooled['p95']:.4g}, "
        f"p99 {pooled['p99']:.4g}, max {pooled['p100']:.4g}"
    )
    print(
        f"  p_max mean {pooled['mean']:.4g} "
        f"({predictive['max_probability']['mean_over_uniform']:.2f}x uniform); "
        "greedy always takes the top token, which is not the same as that "
        "token carrying much mass"
    )
    target = predictive["target_probability"]
    print(
        f"  p_target mean {target['probability']['mean']:.4g}, "
        f"loss mean {target['loss']['mean']:.4f} "
        f"(uniform log K = {target['uniform_loss']:.4f})"
    )


def report_gradient_norms(
    record: InitializationExperimentRecord,
    gradient_result: Any,
    gradient_metadata: Mapping[str, Any],
    num_positions: int,
) -> None:
    """Exact per-position parameter-gradient norms at the canonical temperature."""

    from .gradients import gradient_guess_table

    table = gradient_guess_table(record)
    measured = table["target_occurrence_count"] > 0
    norms = table["mean_gradient_norm"][measured]
    print(
        f"\nPer-position parameter-gradient norms "
        f"(initialization {gradient_metadata['initialization_index']}, "
        f"seed {gradient_metadata['model_seed']}, real input, "
        f"{gradient_metadata['softmax_support']} support):"
    )
    print(
        f"  positions differentiated : {table['num_positions']:,}"
        f"  ({'all' if table['covers_all_positions'] else 'SUBSET of'} "
        f"{num_positions:,})"
    )
    print(f"  parameters in the norm   : {gradient_metadata['parameter_count']:,}")
    print(f"  mean single-position loss: {gradient_metadata['mean_loss']:.6f}")
    print(
        f"  G_i over {int(measured.sum()):,} tokens with targets: "
        f"min {norms.min():.6g}, median {np.median(norms):.6g}, max {norms.max():.6g}"
    )
    print(
        f"  measurement time         : {gradient_result.seconds:,.1f}s "
        f"({table['num_positions'] / max(gradient_result.seconds, 1e-9):,.1f} positions/s)"
    )
