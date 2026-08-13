"""Redraw figures, and print gradient statistics, from a persisted record.

The experiment runner draws figures at the end of a run, which is the wrong tool
once the run is over: regenerating one plot should not require a model, a GPU, or
recomputing anything. This entry point reads an already-written record directory
and nothing else.

It deliberately imports **no PyTorch**. Records carry complete per-token vectors,
so re-analysis needs NumPy, and drawing needs matplotlib; neither needs the model
that produced them. That is also why this script cannot accidentally rerun an
experiment -- it has no way to.

Examples::

    # every figure the record supports
    python3 scripts/render_record_figures.py outputs/<run>/analyses

    # figure 8 alone, leaving the others untouched
    python3 scripts/render_record_figures.py outputs/<run>/analyses --only figure8

    # the distributions and correlations, without drawing anything
    python3 scripts/render_record_figures.py outputs/<run>/analyses --stats-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Allow running from the repository root before editable installation.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from llm_behavior_lab.analysis import (  # noqa: E402
    gradient_guess_correlations,
    gradient_observable_summary,
    load_record,
    predictive_probability_summary,
)

#: Selectable figures. ``all`` runs the record's full figure set.
FIGURES = {
    "figure0": "plot_sampling_adequacy",
    "figure1": "plot_ranked_frequency_profiles",
    "figure2": "plot_token_wise_mismatch",
    "figure3": "plot_token_identity_scatter",
    "figure4": "plot_input_structure_profiles",
    "figure5": "plot_temperature_ranked_profiles",
    "figure6": "plot_temperature_ranked_distances",
    "figure7": "plot_temperature_support_and_agreement",
    "figure8": "plot_ranked_predictive_probabilities",
    "figure9": "plot_max_predictive_probability",
    "figure10": "plot_gradient_vs_guess_bias",
}

#: Figures that exist only when the record carries the analysis behind them.
CONDITIONAL_FIGURES = {
    "figure8": "has_predictive_probability_analysis",
    "figure9": "has_predictive_probability_analysis",
    "figure10": "has_position_gradients",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "record_dir",
        type=Path,
        help="Directory holding <name>.npz and <name>.json, normally <run>/analyses.",
    )
    parser.add_argument(
        "--name",
        default="initialization_distribution",
        help="Record stem inside that directory.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Where to write. Defaults to <record_dir>/../figures, the run's own.",
    )
    parser.add_argument(
        "--only",
        choices=sorted(FIGURES) + ["all"],
        default="all",
        help="Draw a single figure instead of the record's whole set.",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Print the gradient distributions and correlations, draw nothing.",
    )
    parser.add_argument(
        "--stats-json",
        type=Path,
        default=None,
        help="Also write those statistics to this JSON file.",
    )
    parser.add_argument(
        "--scale",
        choices=["auto", "log", "linear"],
        default="auto",
        help=(
            "Probability/gradient axis scale for whichever of figures 8, 9 or 10 "
            "is drawn. 'auto' follows the observed dynamic range."
        ),
    )
    return parser.parse_args()


def _print_quantiles(title: str, summary: dict[str, Any]) -> None:
    """Print one distribution's percentiles in a fixed, scannable order."""

    if not summary.get("count"):
        print(f"  {title}: no values")
        return
    ordered = [key for key in summary if key.startswith("p")]
    print(f"  {title}  (n = {summary['count']:,}, mean = {summary['mean']:.6g})")
    print("    " + "  ".join(f"{key}={summary[key]:.6g}" for key in ordered))


def report_gradient_statistics(record: Any) -> dict[str, Any]:
    """Print, and return, everything needed to choose scales for figure 10."""

    summary = gradient_observable_summary(record)
    correlations = gradient_guess_correlations(record)

    print("\n== Gradient observable, distributions ==")
    print(
        f"  positions D = {summary['num_positions']:,}"
        f"  ({'all' if summary['covers_all_positions'] else 'SUBSET'});"
        f"  initialization {summary['initialization_index']}"
    )
    print(
        f"  tokens with n(i) > 0: {summary['num_plotted_tokens']:,}"
        f"  of eligible {summary['eligible_vocab_size']:,}"
    )
    _print_quantiles("G(i)  mean gradient norm", summary["mean_gradient_norm"])
    print(f"    dynamic range max/min = {summary['gradient_dynamic_range']:.6g}")
    _print_quantiles("q(i)  greedy guess fraction", summary["greedy_guess_fraction"])
    print(
        f"    q(i) == 0 for {summary['zero_guess_tokens']:,} tokens"
        f"  ({summary['zero_guess_fraction']:.2%} of plotted) -- kept, not discarded"
    )
    print(
        f"    greedy mass on plotted tokens = {summary['greedy_mass_on_plotted_tokens']:.4f}"
        "  (the remainder sits on tokens that are never a target)"
    )
    _print_quantiles("p(i)  positive corpus fraction", summary["positive_corpus_fraction"])
    if summary["nonpositive_corpus_tokens"]:
        print(f"    WARNING: {summary['nonpositive_corpus_tokens']:,} plotted tokens have p(i) <= 0")
    _print_quantiles("n(i)  target occurrences", summary["target_occurrence_count"])
    print("    occurrences histogram: " + ", ".join(
        f"{key}={value:,}" for key, value in summary["occurrence_histogram"].items()
    ))

    print("\n== Rank correlations (Spearman, tie-corrected) ==")
    print(f"  tokens used = {correlations['num_tokens']:,};"
          f"  zero-guess tokens included = {correlations['includes_zero_guess_tokens']};"
          f"  weighted = {correlations['weighted']}")
    print(f"  rho(G, q) = {correlations['spearman_gradient_vs_guess']:+.6f}   <- primary")
    print(f"  rho(G, p) = {correlations['spearman_gradient_vs_corpus']:+.6f}")
    print(f"  rho(q, p) = {correlations['spearman_guess_vs_corpus']:+.6f}")

    print("\n  G(i) precision varies with n(i); these are diagnostics, not the primary value.")
    print(f"  {'n(i) band':>12} {'tokens':>8} {'rho(G, q)':>12}")
    for stratum in correlations["strata"]:
        high = stratum["max_occurrences"]
        band = f"{stratum['min_occurrences']}+" if high is None else (
            str(stratum["min_occurrences"]) if high == stratum["min_occurrences"]
            else f"{stratum['min_occurrences']}-{high}"
        )
        print(f"  {band:>12} {stratum['num_tokens']:>8,} {stratum['spearman_gradient_vs_guess']:>12.6f}")
    print(f"\n  {'n(i) >=':>12} {'tokens':>8} {'rho(G, q)':>12}")
    for item in correlations["sensitivity_by_min_occurrences"]:
        print(
            f"  {item['min_occurrences']:>12} {item['num_tokens']:>8,}"
            f" {item['spearman_gradient_vs_guess']:>12.6f}"
        )

    return {"distributions": summary, "correlations": correlations}


def report_predictive_statistics(record: Any) -> dict[str, Any]:
    """Print, and return, the raw predictive-probability diagnostics.

    These describe the distribution *before* any sampling policy: raw logits,
    eligible support, softmax at T = 1, no top-p truncation, no greedy or nucleus
    decision. They are not the nucleus distribution and must not be read as it.
    """

    summary = predictive_probability_summary(record)
    uniform = summary["uniform_probability"]

    print("\n== Raw predictive distribution (T = 1, eligible support, pre-sampling) ==")
    protocol = summary["protocol"]
    if protocol:
        print(
            f"  temperature {protocol.get('temperature')},"
            f" support {protocol.get('support')},"
            f" before top-p: {protocol.get('before_top_p')},"
            f" before sampling: {protocol.get('before_sampling')}"
        )
    print(
        f"  D = {summary['num_positions']:,},  I = {summary['num_initializations']},"
        f"  eligible K = {summary['eligible_vocab_size']:,},  uniform 1/K = {uniform:.6g}"
    )

    print("\n  Ranked profile: probabilities ranked WITHIN each position, then averaged.")
    print("  (Distinct from figure 1, which ranks guess frequencies accumulated ACROSS positions.)")
    print(f"  {'rank':>8} {'Pbar(r)':>14} {'x uniform':>12}")
    for rank, value in summary["ranked_profile_at_rank"].items():
        print(f"  {rank:>8} {value:>14.6g} {value / uniform:>12.3f}")
    print(f"    profile dynamic range max/min = {summary['profile_dynamic_range']:.6g}")
    print(f"    ranks with exactly zero mean probability: {summary['num_zero_ranks']:,}")

    maximum = summary["max_probability"]
    print("\n  p_max(d): probability of the token greedy selects")
    _print_quantiles("p_max", maximum["pooled"])
    print(
        f"    median / uniform = {maximum['median_over_uniform']:.3f}x,"
        f"  mean / uniform = {maximum['mean_over_uniform']:.3f}x"
    )
    print(
        "    per-initialization means: "
        + ", ".join(f"{value:.4g}" for value in maximum["per_initialization_mean"])
    )

    target = summary["target_probability"]
    print("\n  p_target(d) and -log p_target(d): persisted for the gradient analysis")
    _print_quantiles("p_target", target["probability"])
    _print_quantiles("loss", target["loss"])
    print(f"    uniform loss log(K) = {target['uniform_loss']:.6g}")

    return summary


def main() -> None:
    """Report the available statistics and redraw the requested figures."""

    args = parse_args()
    record = load_record(args.record_dir, name=args.name)

    print(f"Record: {args.record_dir}")
    print(f"  vocabulary {record.vocab_size:,}, initializations {record.num_initializations}")
    print(f"  temperature sweep: {record.has_temperature_sweep}")
    print(f"  predictive probabilities: {record.has_predictive_probability_analysis}")
    print(f"  position gradients: {record.has_position_gradients}")

    statistics: dict[str, Any] = {}
    if record.has_predictive_probability_analysis:
        statistics["predictive_probabilities"] = report_predictive_statistics(record)
    if record.has_position_gradients:
        statistics["gradients"] = report_gradient_statistics(record)
    if statistics:
        if args.stats_json is not None:
            args.stats_json.parent.mkdir(parents=True, exist_ok=True)
            args.stats_json.write_text(
                json.dumps(statistics, indent=2, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )
            print(f"\nStatistics written: {args.stats_json}")

    if args.stats_only:
        return

    figures_dir = args.figures_dir or (args.record_dir.parent / "figures")
    from llm_behavior_lab.analysis import figures as figure_module

    if args.only == "all":
        written = figure_module.generate_all_figures(record, figures_dir)
    else:
        required = CONDITIONAL_FIGURES.get(args.only)
        if required is not None and not getattr(record, required):
            raise SystemExit(
                f"This record does not carry the analysis behind {args.only} "
                f"({required} is false), so it cannot be drawn from it."
            )
        function = getattr(figure_module, FIGURES[args.only])
        # Each of the three newer figures exposes exactly one scale knob, under
        # the name its own axis uses.
        scale_argument = {"figure8": "y_scale", "figure9": "x_scale", "figure10": "x_scale"}
        keyword = scale_argument.get(args.only)
        written = (
            function(record, figures_dir, **{keyword: args.scale})
            if keyword is not None
            else function(record, figures_dir)
        )

    print()
    for path in written:
        print(f"Figure: {path}")


if __name__ == "__main__":
    main()
