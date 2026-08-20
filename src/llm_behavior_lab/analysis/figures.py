"""Figures for the initialization-distribution experiment.

Plotting is kept separate from measurement and from aggregation: everything here
consumes an already-persisted record and writes files. Nothing computes a
statistic that :mod:`llm_behavior_lab.analysis.aggregation` does not already
define, so a figure can never disagree with the numbers reported beside it.

Every figure works on the **eligible predictive support**. Structural tokens of a
pretrained vocabulary carry no corpus mass and can never be guessed, so including
them would pad each ranked profile with a tail of zeros.

Figures are built through the object-oriented ``Figure`` API with an explicit
Agg canvas rather than ``pyplot``. There is no global figure state, no window is
ever opened, and the functions behave identically in a notebook, in a test, and
on a headless machine.

Two vocabulary regimes are supported. A character vocabulary of tens of tokens
reads best as discrete steps on a linear rank axis. A subword vocabulary of tens
of thousands needs a logarithmic rank axis -- otherwise the head, which is the
interesting part, occupies a pixel -- plain lines instead of step edges, and
rasterized scatter marks to keep vector output a sane size.

``matplotlib`` is an optional dependency; install it with
``python3 -m pip install -e ".[analysis]"``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from llm_behavior_lab.analysis.aggregation import (
    corpus_observed_zero_guess_counts,
    effective_support,
    ranked_profile_with_error,
    paired_condition_distances,
    effective_supports,
    eligible_view,
    mean_with_sem,
    persistent_absolute_gaps,
    policy_zero_frequency,
    ranked_profile,
    ranked_profiles,
    sampling_adequacy,
    support_summary,
    token_wise_absolute_gaps,
    top_two_gap,
)

__all__ = [
    "FIGURE_FORMATS",
    "LARGE_VOCAB_THRESHOLD",
    "CONDITION_STYLES",
    "NULL_STYLE",
    "POLICY_STYLES",
    "escape_token_label",
    "generate_all_figures",
    "plot_gradient_vs_guess_bias",
    "plot_gradient_vs_mean_probability",
    "plot_gradient_vs_nucleus_guess_bias",
    "plot_correction_provenance",
    "plot_gradient_directional_clustering",
    "plot_initial_gradient_split",
    "plot_initial_logit_correction",
    "plot_initial_performance_and_bias",
    "plot_greedy_confidence_vs_temperature",
    "plot_max_predictive_probability",
    "plot_ranked_mean_token_probabilities",
    "plot_ranked_predictive_probabilities",
    "plot_temperature_gradient_vs_guess_bias",
    "plot_temperature_max_predictive_probability",
    "plot_temperature_ranked_predictive_probabilities",
    "plot_ranked_frequency_profiles",
    "plot_sampling_adequacy",
    "plot_input_structure_profiles",
    "plot_temperature_ranked_distances",
    "plot_temperature_ranked_profiles",
    "plot_temperature_support_and_agreement",
    "plot_token_identity_scatter",
    "plot_token_wise_mismatch",
    "save_figure",
]

#: One artifact per figure. SVG alone: it stays sharp at any zoom, carries the
#: text as text, and -- because the dense curves of a 32k-token figure are
#: rasterized *inside* the SVG -- costs about the same as the PNG it replaces.
#: A second raster file per figure was duplication, not a second format.
FIGURE_FORMATS = ("svg",)

#: Above this eligible-support size the figures switch to large-vocabulary
#: rendering: logarithmic rank axis, plain lines, rasterized scatter.
LARGE_VOCAB_THRESHOLD = 2000

#: Stable visual identities for the already-established temperature conditions:
#: private, and a presentation key only.
#:
#: This defines **no** temperature grid. ``CONFIDENCE_TEMPERATURES`` and
#: ``GRADIENT_TEMPERATURES`` in :mod:`llm_behavior_lab.evaluation` remain the
#: authoritative definitions for what an experiment measures; this tuple only
#: says which colour each already-measured condition is drawn in, so a reader
#: can carry a temperature's colour from one figure to the next.
#:
#: Deliberately a literal copy rather than an import: the analysis layer stays
#: free of :mod:`llm_behavior_lab.evaluation`, so a finished record remains
#: re-analyzable with NumPy and matplotlib alone. A temperature absent from this
#: tuple simply falls back to the ordinal mapping; it is never rejected.
_CANONICAL_TEMPERATURES = (0.12, 0.24, 0.36, 0.48, 0.60, 1.00, 1.20)

#: One colour per policy, reused across every figure so curves stay comparable.
POLICY_STYLES = {
    "corpus": {"color": "#222222", "label": "corpus empirical"},
    "greedy": {"color": "#1f77b4", "label": "greedy guesses"},
    "nucleus": {"color": "#d62728", "label": "nucleus guesses"},
    "selected": {"color": "#2ca02c", "label": "selected-position targets"},
}

#: The uniform null is deliberately styled unlike any policy: grey, dashed, and
#: with a hatched band. Its envelope is a **Monte Carlo** interval over null
#: realizations, not a SEM across model initializations, and the two must never
#: be mistaken for one another on the same axes.
NULL_STYLE = {"color": "#7f7f7f", "label": "uniform D-draw null"}

#: One colour per input condition, shared by both panels of figure 4.
CONDITION_STYLES = {
    "real": {"color": "#1f77b4", "label": "real corpus"},
    "shuffled": {"color": "#ff7f0e", "label": "shuffled tokens"},
    "gaussian": {"color": "#9467bd", "label": "Gaussian embeddings"},
}

_ANNOTATION_BOX = {"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#999999"}


def escape_token_label(token: str, *, max_length: int = 12) -> str:
    """Render one token so whitespace and control characters stay visible.

    A subword vocabulary contains newlines, tabs, byte-fallback pieces, and the
    marker some tokenizers use for a leading space. Printed raw, several of them
    are invisible or actively break the layout, and two different tokens can look
    identical on a plot.
    """

    text = str(token)
    if len(text) > max_length:
        text = text[: max_length - 1] + "…"
    return repr(text)


def _format_count(value: float) -> str:
    """Format a token count readably at any vocabulary size."""

    return f"{value:,.0f}" if abs(value) >= 1000 else f"{value:.0f}"


def _new_figure(width: float = 8.0, height: float = 5.0) -> Figure:
    """Create a canvas-backed figure that never needs a display."""

    figure = Figure(figsize=(width, height), dpi=150)
    FigureCanvasAgg(figure)
    return figure


def _positive(values: np.ndarray) -> np.ndarray:
    """Mask non-positive entries so they vanish on a logarithmic axis.

    A large vocabulary produces long zero tails -- most tokens are never guessed
    -- and a logarithmic axis cannot show them. Masking makes the curve stop
    where the data stops instead of plunging to the axis floor.
    """

    array = np.asarray(values, dtype=np.float64)
    return np.where(array > 0.0, array, np.nan)


def _is_large(record: Any) -> bool:
    """Whether large-vocabulary rendering applies."""

    return record.eligible_vocab_size > LARGE_VOCAB_THRESHOLD


def _profile_kwargs(large: bool) -> dict[str, Any]:
    """Line style for a ranked profile, by vocabulary regime.

    Curves over tens of thousands of ranks are rasterized: kept as vector paths
    they make an SVG tens of times larger for detail no reader can resolve. Axes,
    text, and legends stay vector, so the figure remains publication-quality.
    """

    if large:
        return {"rasterized": True}
    return {"drawstyle": "steps-mid"}


def _configure_rank_axis(axes: Any, record: Any, label: str) -> None:
    """Apply the rank-axis scaling appropriate to the vocabulary size."""

    axes.set_yscale("log")
    axes.set_xlabel(label)
    if _is_large(record):
        axes.set_xscale("log")
    axes.grid(True, which="both", alpha=0.25)


def save_figure(
    figure: Figure,
    directory: str | Path,
    stem: str,
    *,
    formats: Sequence[str] = FIGURE_FORMATS,
) -> list[Path]:
    """Write one figure under deterministic filenames.

    Args:
        figure: Figure to write.
        directory: Destination directory, created when missing.
        stem: Filename without extension. Deterministic, so reruns overwrite
            rather than accumulate.
        formats: Extensions to emit.

    Returns:
        The written paths, in the order requested.
    """

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for extension in formats:
        path = directory / f"{stem}.{extension}"
        figure.savefig(path, format=extension, bbox_inches="tight")
        written.append(path)
    return written


def plot_sampling_adequacy(record: Any, directory: str | Path) -> list[Path]:
    """Figure 0 -- are the analyzed positions representative of the split?

    Compares the ranked token-frequency profile of the whole analysis split with
    the ranked profile of the next-token targets at the analyzed positions. The
    two curves answer a question about the *corpus sample only*; no model is
    involved. When they separate visibly, the evaluation-position count is too
    small for a model comparison built on it, and adding initializations will not
    help.

    A change of tokenizer invalidates any earlier verdict, so this figure has to
    be re-read after switching from characters to subwords.
    """

    adequacy = sampling_adequacy(record)
    corpus = ranked_profile(eligible_view(record, record.corpus_fractions))
    selected = ranked_profile(eligible_view(record, record.selected_target_fractions))
    ranks = np.arange(1, record.eligible_vocab_size + 1)
    style = _profile_kwargs(_is_large(record))

    figure = _new_figure(width=8.5, height=5.5)
    axes = figure.subplots()
    axes.plot(
        ranks,
        _positive(corpus),
        color=POLICY_STYLES["corpus"]["color"],
        linewidth=2.0,
        label=f"whole split ({_format_count(adequacy['corpus_tokens'])} tokens)",
        **style,
    )
    axes.plot(
        ranks,
        _positive(selected),
        color=POLICY_STYLES["selected"]["color"],
        linewidth=1.6,
        linestyle="--",
        label=f"selected positions ({_format_count(adequacy['num_positions'])} targets)",
        **style,
    )
    _configure_rank_axis(axes, record, "token frequency rank")
    axes.set_ylabel("token fraction")
    axes.set_title("Empirical sampling adequacy of the evaluation positions")
    axes.legend(loc="upper right", fontsize=8, frameon=True)

    annotation = "\n".join(
        [
            f"V full / eligible: {_format_count(adequacy['vocab_size'])}"
            f" / {_format_count(adequacy['eligible_vocab_size'])}",
            f"V corpus-observed: {_format_count(adequacy['corpus_observed_support'])}",
            f"N_eff corpus: {adequacy['corpus_effective_support']:,.1f}",
            f"N_eff selected: {adequacy['selected_effective_support']:,.1f}",
            f"TV(split, selected) = {adequacy['total_variation_distance']:.4f}",
            f"JS(split, selected) = {adequacy['js_divergence']:.4f}",
            f"positions = {_format_count(adequacy['num_positions'])}",
        ]
    )
    axes.text(
        0.02,
        0.03,
        annotation,
        transform=axes.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox=_ANNOTATION_BOX,
    )
    return save_figure(figure, directory, "figure0_sampling_adequacy")


def plot_ranked_frequency_profiles(record: Any, directory: str | Path) -> list[Path]:
    """Figure 1 -- concentration profiles, token identity discarded.

    Each distribution is ranked independently, so rank ``r`` of the guess curve
    and rank ``r`` of the corpus curve are generally different tokens. The
    comparison is therefore about *how concentrated* the distributions are, not
    about which tokens agree. The corpus curve carries no error band because it
    is a single fixed distribution, not a sample over initializations.

    The inset reports effective support and zero-frequency support side by side
    because they answer different questions: effective support is how broadly
    the mass is spread, zero frequency is how much of the vocabulary went
    untouched in a fixed number of draws. The nucleus zero-frequency value is
    the per-replicate one, measured at the same draw count as greedy.
    """

    corpus_fractions = eligible_view(record, record.corpus_fractions)
    ranks = np.arange(1, record.eligible_vocab_size + 1)
    style = _profile_kwargs(_is_large(record))

    figure = _new_figure(width=9.0, height=6.0)
    axes = figure.subplots()
    axes.plot(
        ranks,
        _positive(ranked_profile(corpus_fractions)),
        color=POLICY_STYLES["corpus"]["color"],
        linewidth=2.0,
        label="corpus empirical (fixed)",
        **style,
    )

    if record.has_uniform_null:
        null_mean = record.uniform_null["ranked_mean"]
        axes.plot(
            ranks[: null_mean.shape[0]],
            _positive(null_mean),
            color=NULL_STYLE["color"],
            linewidth=1.4,
            linestyle="--",
            label="uniform D-draw null (mean)",
            **style,
        )
        if "ranked_low" in record.uniform_null and "ranked_high" in record.uniform_null:
            axes.fill_between(
                ranks[: null_mean.shape[0]],
                _positive(record.uniform_null["ranked_low"]),
                _positive(record.uniform_null["ranked_high"]),
                color=NULL_STYLE["color"],
                alpha=0.30,
                linewidth=0.0,
                hatch="///",
                edgecolor=NULL_STYLE["color"],
                label="null Monte Carlo interval (not a SEM)",
                **({"step": "mid"} if not _is_large(record) else {"rasterized": True}),
            )

    effective_lines = [f"  corpus {effective_support(corpus_fractions):,.1f}"]
    gap_lines = [f"  corpus {top_two_gap(corpus_fractions):.4f}"]
    zero_lines: list[str] = []

    for policy in ("greedy", "nucleus"):
        guesses = eligible_view(record, record.policy_fractions(policy))
        profile = ranked_profile_with_error(guesses)
        colour = POLICY_STYLES[policy]["color"]
        axes.plot(
            ranks,
            _positive(profile.mean),
            color=colour,
            linewidth=1.6,
            label=f"{POLICY_STYLES[policy]['label']} (mean of {profile.num_samples} inits)",
            **style,
        )
        axes.fill_between(
            ranks,
            _positive(profile.mean - profile.sem),
            _positive(profile.mean + profile.sem),
            color=colour,
            alpha=0.25,
            linewidth=0.0,
            label=f"{POLICY_STYLES[policy]['label']} ± SEM",
            **({"step": "mid"} if not _is_large(record) else {"rasterized": True}),
        )

        support = mean_with_sem(effective_supports(record, policy)[:, None])
        effective_lines.append(
            f"  {policy} {support.mean[0]:,.1f} ± {support.sem[0]:,.1f}"
        )
        gaps = mean_with_sem(np.array([[top_two_gap(row)] for row in guesses]))
        gap_lines.append(f"  {policy} {gaps.mean[0]:.4f} ± {gaps.sem[0]:.4f}")

        zero = policy_zero_frequency(record, policy)
        counts = mean_with_sem(zero.counts[:, None])
        fractions = mean_with_sem(zero.fractions[:, None])
        qualifier = " (per replicate)" if policy == "nucleus" else ""
        zero_lines.append(
            f"  {policy}{qualifier} {_format_count(counts.mean[0])}"
            f" ± {_format_count(counts.sem[0])}"
            f"  ({fractions.mean[0]:.1%} ± {fractions.sem[0]:.1%})"
        )

    null_summary = record.metadata.get("uniform_null", {})
    if null_summary:
        effective_lines.append(
            f"  null {null_summary.get('effective_support_mean', float('nan')):,.1f}"
            " (MC)"
        )
        gap_lines.append(f"  null {null_summary.get('top_two_gap_mean', float('nan')):.4f}")
        zero_lines.append(
            f"  null {_format_count(null_summary.get('zero_frequency_count_mean', 0))}"
            f"  ({null_summary.get('zero_frequency_fraction_mean', 0):.1%})"
        )

    annotation = "\n".join(
        ["effective support e^H:"]
        + effective_lines
        + ["top1-top2 gap:"]
        + gap_lines
        + [f"zero-frequency support (of {_format_count(record.eligible_vocab_size)} eligible,"
           f" {_format_count(record.metadata.get('num_positions', 0))} draws):"]
        + zero_lines
        + [
            f"eligible V = {_format_count(record.eligible_vocab_size)}"
            f" | N = {_format_count(record.metadata.get('num_positions', 0))}"
            f" | S = {record.num_initializations}"
        ]
    )

    _configure_rank_axis(axes, record, "frequency rank")
    axes.set_ylabel("fraction (corpus tokens or selected guesses)")
    axes.set_title("Ranked concentration: corpus frequencies vs. guess frequencies")
    axes.legend(loc="upper right", fontsize=8, frameon=True)
    axes.text(
        0.02,
        0.03,
        annotation,
        transform=axes.transAxes,
        fontsize=7.5,
        va="bottom",
        ha="left",
        bbox=_ANNOTATION_BOX,
    )
    return save_figure(figure, directory, "figure1_ranked_frequency_profiles")


def plot_token_wise_mismatch(record: Any, directory: str | Path) -> list[Path]:
    """Figure 2 -- same-token mismatch, ranked only after differencing.

    Solid curves are the mean over initializations of the per-initialization
    ranked gap profile: the *typical* mismatch one initialization shows. Dashed
    curves rank ``|E_s[q] - p|`` instead: the mismatch that survives averaging
    over initializations. A large gap between the two means initializations
    disagree about which tokens they over-select; curves that sit together mean
    the bias is systematic.
    """

    corpus_fractions = eligible_view(record, record.corpus_fractions)
    ranks = np.arange(1, record.eligible_vocab_size + 1)
    style = _profile_kwargs(_is_large(record))

    figure = _new_figure(width=9.0, height=6.0)
    axes = figure.subplots()

    annotation_lines: list[str] = []
    for policy in ("greedy", "nucleus"):
        guesses = eligible_view(record, record.policy_fractions(policy))
        colour = POLICY_STYLES[policy]["color"]

        typical = mean_with_sem(ranked_profiles(token_wise_absolute_gaps(guesses, corpus_fractions)))
        persistent = ranked_profile(persistent_absolute_gaps(guesses, corpus_fractions))

        axes.plot(
            ranks,
            _positive(typical.mean),
            color=colour,
            linewidth=1.8,
            label=f"{POLICY_STYLES[policy]['label']}: typical |q-p| (mean ± SEM)",
            **style,
        )
        axes.fill_between(
            ranks,
            _positive(typical.mean - typical.sem),
            _positive(typical.mean + typical.sem),
            color=colour,
            alpha=0.25,
            linewidth=0.0,
            **({"step": "mid"} if not _is_large(record) else {"rasterized": True}),
        )
        axes.plot(
            ranks,
            _positive(persistent),
            color=colour,
            linewidth=1.4,
            linestyle="--",
            label=f"{POLICY_STYLES[policy]['label']}: persistent |mean(q)-p|",
            **style,
        )
        annotation_lines.append(
            f"{policy}: max typical = {typical.mean[0]:.4f}, max persistent = {persistent[0]:.4f}"
        )

    annotation_lines.append(
        f"eligible V = {_format_count(record.eligible_vocab_size)}"
        f" | N = {_format_count(record.metadata.get('num_positions', 0))}"
        f" | S = {record.num_initializations}"
    )

    _configure_rank_axis(axes, record, "rank of the same-token absolute gap")
    axes.set_ylabel("|guess fraction - corpus fraction|")
    axes.set_title("Token-wise mismatch: typical per initialization vs. persistent")
    axes.legend(loc="upper right", fontsize=8, frameon=True)
    axes.text(
        0.02,
        0.03,
        "\n".join(annotation_lines),
        transform=axes.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox=_ANNOTATION_BOX,
    )
    return save_figure(figure, directory, "figure2_token_wise_mismatch")


def plot_token_identity_scatter(
    record: Any,
    directory: str | Path,
    *,
    num_labels: int = 6,
) -> list[Path]:
    """Complementary figure -- which specific tokens are over- or under-selected.

    Both ranked figures discard token identity at some point. This one keeps it
    throughout: each marker is one eligible token, positioned by its corpus
    fraction and its initialization-averaged guess fraction. Markers on the
    identity line are selected as often as they occur; markers above it are
    over-selected.

    At a subword vocabulary this is tens of thousands of marks, so they are drawn
    small, semi-transparent, and rasterized -- the density itself is the
    information. Only the largest deviations are labelled; labelling thousands of
    tokens would produce an unreadable figure and an enormous file.
    """

    corpus = eligible_view(record, record.corpus_fractions)
    eligible_ids = record.eligible_token_ids
    tokens = record.tokens
    large = _is_large(record)

    figure = _new_figure(width=7.5, height=6.5)
    axes = figure.subplots()

    mean_guesses = {
        policy: eligible_view(record, record.policy_fractions(policy)).mean(axis=0)
        for policy in ("greedy", "nucleus")
    }
    positive_values = [corpus[corpus > 0]] + [
        values[values > 0] for values in mean_guesses.values() if np.any(values > 0)
    ]
    lower = min(float(values.min()) for values in positive_values) * 0.5
    upper = max(float(values.max()) for values in positive_values) * 2.0
    axes.plot([lower, upper], [lower, upper], color="#888888", linewidth=1.0, linestyle=":", label="y = x")

    for policy in ("greedy", "nucleus"):
        mean_guess = mean_guesses[policy]
        colour = POLICY_STYLES[policy]["color"]
        axes.scatter(
            _positive(corpus),
            _positive(mean_guess),
            s=4 if large else 22,
            alpha=0.35 if large else 0.7,
            color=colour,
            edgecolors="none",
            rasterized=large,
            label=POLICY_STYLES[policy]["label"],
        )
        if tokens and num_labels > 0:
            deviation = np.abs(mean_guess - corpus)
            for index in np.argsort(deviation)[::-1][:num_labels]:
                if corpus[index] <= 0 or mean_guess[index] <= 0:
                    continue
                token_id = int(eligible_ids[index])
                axes.annotate(
                    escape_token_label(tokens[token_id]),
                    (corpus[index], mean_guess[index]),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=7,
                    color=colour,
                )

    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xlabel("corpus token fraction  p(i)")
    axes.set_ylabel("mean guess fraction  mean_s q(s, i)")
    axes.set_title("Token identity: corpus frequency vs. mean guess frequency")
    axes.legend(loc="lower right", fontsize=8, frameon=True)
    axes.grid(True, which="both", alpha=0.25)

    never_guessed = {
        policy: int(corpus_observed_zero_guess_counts(record, policy).mean())
        for policy in ("greedy", "nucleus")
    }
    axes.text(
        0.02,
        0.97,
        "\n".join(
            [
                f"eligible V = {_format_count(record.eligible_vocab_size)}",
                f"corpus-observed = {_format_count(support_summary(record)['corpus_observed_support'])}",
                "corpus tokens never guessed:",
                f"  greedy {_format_count(never_guessed['greedy'])}"
                f" | nucleus {_format_count(never_guessed['nucleus'])}",
            ]
        ),
        transform=axes.transAxes,
        fontsize=7.5,
        va="top",
        ha="left",
        bbox=_ANNOTATION_BOX,
    )
    return save_figure(figure, directory, "figure3_token_identity_scatter")


def plot_input_structure_profiles(record: Any, directory: str | Path) -> list[Path]:
    """Figure 4 -- does the guess distribution depend on input structure?

    One initialized model, three inputs: the real corpus windows, the same token
    multiset with its ordering destroyed, and synthetic Gaussian vectors at the
    embedding boundary. Any difference between the curves is attributable to the
    input, because the weights are identical.

    Two panels, because greedy and nucleus respond differently and overlaying
    six curves would be unreadable. Each condition is ranked independently
    within an initialization and then aggregated rank by rank, exactly as in
    figure 1, so **token identity is discarded here too**. Curves that coincide
    mean the *shape* is insensitive to input structure; they say nothing about
    whether the same tokens are chosen. The paired same-token distances printed
    with the run answer that question.

    Interpretation of the contrasts:

    * real vs shuffled -- approximately isolates sequential ordering;
    * shuffled vs Gaussian -- what discrete token identity adds once ordering
      is already gone;
    * real vs Gaussian -- the broadest contrast, and **not** a clean test of
      correlation on its own.
    """

    if not record.has_input_structure:
        raise ValueError(
            "This record carries no input-structure conditions; run the experiment with "
            "input_structure enabled to produce figure 4."
        )

    ranks = np.arange(1, record.eligible_vocab_size + 1)
    style = _profile_kwargs(_is_large(record))
    conditions = record.available_conditions

    figure = _new_figure(width=11.0, height=5.5)
    panels = figure.subplots(1, 2, sharey=True)

    for axes, policy in zip(panels, ("greedy", "nucleus")):
        for condition in conditions:
            fractions = eligible_view(
                record, record.condition_policy_fractions(condition, policy)
            )
            profile = ranked_profile_with_error(fractions)
            colour = CONDITION_STYLES[condition]["color"]
            axes.plot(
                ranks,
                _positive(profile.mean),
                color=colour,
                linewidth=1.6,
                label=f"{CONDITION_STYLES[condition]['label']} (mean of {profile.num_samples})",
                **style,
            )
            axes.fill_between(
                ranks,
                _positive(profile.mean - profile.sem),
                _positive(profile.mean + profile.sem),
                color=colour,
                alpha=0.25,
                linewidth=0.0,
                **({"step": "mid"} if not _is_large(record) else {"rasterized": True}),
            )
        _configure_rank_axis(axes, record, "frequency rank")
        axes.set_title(f"{policy} guesses")
        axes.legend(loc="upper right", fontsize=7.5, frameon=True)

        distances = paired_condition_distances(record, policy)
        lines = [
            f"{name.replace('tv_', '').replace('_vs_', ' vs ')}: "
            f"{values['mean']:.4f} ± {values['sem']:.4f}"
            for name, values in distances.get("pairs", {}).items()
        ]
        if lines:
            axes.text(
                0.02,
                0.03,
                "same-token TV between conditions\n(± SEM across initializations)\n"
                + "\n".join(f"  {line}" for line in lines),
                transform=axes.transAxes,
                fontsize=7,
                va="bottom",
                ha="left",
                bbox=_ANNOTATION_BOX,
            )

    panels[0].set_ylabel("selected-guess fraction")
    figure.suptitle(
        "Input structure: ranked guess concentration under real, shuffled, and Gaussian input"
    )
    return save_figure(figure, directory, "figure4_input_structure_profiles")



def _sweep_context(record: Any):
    """Shared setup for the three temperature figures."""

    from llm_behavior_lab.analysis.transition import sweep_summary

    if not record.has_temperature_sweep:
        raise ValueError(
            "This record carries no temperature sweep; run the experiment with "
            "temperature_sweep enabled to produce the temperature figures."
        )
    summary = sweep_summary(record)
    return summary, list(summary["temperatures"])


def _ordinal_temperature_colours(count: int) -> list:
    """Colours spread by position within ``count`` curves.

    The historical mapping, kept as the fallback for any temperature sequence
    that is not the canonical grid.
    """

    from matplotlib import cm

    return [cm.viridis(value) for value in np.linspace(0.05, 0.92, count)]


def _canonical_temperature_palette() -> dict:
    """Bind each canonical temperature to a fixed colour.

    Built from ``linspace(0.05, 0.92, 7)`` over the canonical grid, so the seven
    colours are exactly the ones the ordinal mapping already produced for the
    seven-temperature figures. Figures 11 and 14 are therefore unchanged.
    """

    return dict(
        zip(
            _CANONICAL_TEMPERATURES,
            _ordinal_temperature_colours(len(_CANONICAL_TEMPERATURES)),
        )
    )


def _temperature_colours(temperatures: Sequence[float]) -> list:
    """Perceptually ordered colours, keyed by temperature *value*.

    Sequential and colour-blind safe; deliberately not a rainbow, where hue
    ordering carries no perceptual ordering.

    Colour is bound to the temperature itself rather than to its position in the
    list. Spreading a colormap across ``len(temperatures)`` gives the same
    temperature a different colour whenever the number of curves differs -- the
    six-temperature nucleus sweep of figure 5 against the seven-temperature
    confidence grid of figures 11 and 14 -- so a reader cannot carry a colour
    from one figure to the next. Keying by value fixes each temperature's
    identity across the whole family.

    The lookup is **exact**, not tolerant: canonical temperatures reach here as
    the module constants they were defined as, or as parsed literals of the same
    values, so they compare equal. A near-miss must fall back visibly rather
    than be snapped onto a key it does not equal.

    A sequence carrying any non-canonical temperature -- a historical record, or
    a custom grid -- falls back to the ordinal mapping for that whole sequence,
    preserving exactly the previous appearance.
    """

    values = [float(temperature) for temperature in temperatures]
    palette = _canonical_temperature_palette()
    if all(value in palette for value in values):
        return [palette[value] for value in values]
    return _ordinal_temperature_colours(len(values))


def plot_temperature_ranked_profiles(record: Any, directory: str | Path) -> list[Path]:
    """Figure 5 -- ranked guess profiles from the greedy anchor to the null.

    Each curve is ``mean_s( sort(q_s) )``: **rank within each initialization,
    then average corresponding ranks**, the same convention as figure 1. Token
    identity is discarded, so rank ``r`` is generally a different token on every
    curve.

    Greedy is the ``T = 0`` anchor and the finite-``D`` uniform categorical null
    is the high-temperature reference; the sweep curves sit between them.
    """

    _summary, temperatures = _sweep_context(record)
    ranks = np.arange(1, record.eligible_vocab_size + 1)
    style = _profile_kwargs(_is_large(record))

    figure = _new_figure(width=9.0, height=6.0)
    axes = figure.subplots()

    greedy = ranked_profile_with_error(eligible_view(record, record.greedy_fractions))
    axes.plot(
        ranks, _positive(greedy.mean), color="#000000", linewidth=2.4,
        label="greedy  (T = 0 anchor)", zorder=5, **style,
    )
    sweep = eligible_view(record, _sweep_fractions(record, "real"))
    for index, (temperature, colour) in enumerate(
        zip(temperatures, _temperature_colours(temperatures))
    ):
        profile = ranked_profile_with_error(sweep[:, index, :])
        axes.plot(
            ranks, _positive(profile.mean), color=colour, linewidth=1.5,
            label=f"T = {temperature:g}", zorder=3, **style,
        )
    if record.has_uniform_null:
        axes.plot(
            ranks, _positive(record.uniform_null["ranked_mean"]),
            color=NULL_STYLE["color"], linewidth=2.2, linestyle="--",
            label="uniform D-draw null", zorder=4, **style,
        )

    _configure_rank_axis(axes, record, "frequency rank")
    axes.set_ylabel("mean ranked selected-guess fraction")
    axes.set_title("Ranked guess concentration across temperature (real input)")
    legend = axes.legend(
        loc="upper right", fontsize=8.5, frameon=True, title="increasing temperature",
    )
    legend.get_title().set_fontsize(8)
    axes.text(
        0.01, 0.01,
        "ranked within each initialization, then averaged rank by rank\n"
        f"{record.num_initializations} initializations, "
        f"N = {_format_count(record.metadata.get('num_positions', 0))}",
        transform=axes.transAxes, fontsize=7.5, va="bottom", ha="left", bbox=_ANNOTATION_BOX,
    )
    return save_figure(figure, directory, "figure5_temperature_ranked_profiles")


def plot_temperature_ranked_distances(record: Any, directory: str | Path) -> list[Path]:
    """Figure 6 -- ranked-profile distance to each anchor, against temperature.

    Both curves are total variation between **independently ranked** profiles:
    the sample against greedy, and the sample against the uniform null. Ranking
    first discards token identity, so these are *shape* distances and must not
    be read as the same-token total variation of figure 2.

    Their crossing is the readable signature of the transition: departure from
    the greedy regime and approach toward the finite-``D`` null.
    """

    summary, temperatures = _sweep_context(record)
    metrics = summary["conditions"]["real"]["metrics"]

    figure = _new_figure(width=8.0, height=5.5)
    axes = figure.subplots()
    for name, colour, marker, label in (
        ("tv_rank_to_greedy", "#000000", "o", "distance from greedy (T = 0)"),
        ("tv_rank_to_uniform", NULL_STYLE["color"], "s", "distance from uniform D-draw null"),
    ):
        axes.errorbar(
            temperatures, np.array(metrics[name]["mean"]), yerr=np.array(metrics[name]["sem"]),
            color=colour, marker=marker, markersize=5, linewidth=1.8, capsize=3, label=label,
        )

    axes.set_xlabel("nucleus temperature  T")
    axes.set_ylabel("ranked-profile total variation")
    axes.set_title(
        "Departure from greedy and approach toward the uniform null\n"
        "(ranked-profile distance; token identity discarded)",
        fontsize=11,
    )
    axes.grid(True, alpha=0.25)
    axes.legend(loc="center right", fontsize=9, frameon=True)
    axes.text(
        0.01, 0.01,
        f"error bars: SEM across {summary['conditions']['real']['num_initializations']} "
        f"initializations | top_p = {summary.get('top_p')} fixed",
        transform=axes.transAxes, fontsize=7.5, va="bottom", ha="left", bbox=_ANNOTATION_BOX,
    )
    return save_figure(figure, directory, "figure6_temperature_ranked_distances")


def plot_temperature_support_and_agreement(record: Any, directory: str | Path) -> list[Path]:
    """Figure 7 -- normalized effective support, and agreement with greedy.

    Two different quantities on two axes, deliberately not conflated:

    * **left** -- ``N_eff(T) / N_eff(null)`` per input condition. 1.0 means the
      guess distribution is as broad as pure chance at the same draw count;
    * **right** -- the fraction of positions where the sampled token *is* the
      greedy argmax. That one preserves token identity, unlike everything on the
      left axis.

    Real and shuffled input often track each other almost exactly. They are
    drawn with different line styles, markers, and widths, and in an order that
    keeps the earlier curve visible, so near-perfect overlap reads as overlap
    rather than as a missing curve.
    """

    summary, temperatures = _sweep_context(record)

    figure = _new_figure(width=8.5, height=5.5)
    axes = figure.subplots()
    # Widest and drawn first, so a later coincident curve sits visibly on top.
    widths = {"real": 3.4, "shuffled": 2.0, "gaussian": 1.6}
    dashes = {"real": (None, None), "shuffled": (5, 2), "gaussian": (1, 1.5)}
    markers = {"real": "o", "shuffled": "^", "gaussian": "D"}
    for condition in summary["conditions"]:
        metrics = summary["conditions"][condition]["metrics"]
        line = axes.errorbar(
            temperatures,
            np.array(metrics["effective_support_over_null"]["mean"]),
            yerr=np.array(metrics["effective_support_over_null"]["sem"]),
            color=CONDITION_STYLES[condition]["color"],
            marker=markers[condition], markersize=5,
            markerfacecolor="none" if condition != "real" else None,
            linewidth=widths[condition], capsize=2, alpha=0.9,
            label=f"{condition}:  N_eff / N_eff(null)",
        )
        if dashes[condition][0] is not None:
            line[0].set_dashes(dashes[condition])

    axes.axhline(1.0, color=NULL_STYLE["color"], linestyle=":", linewidth=1.2)
    axes.annotate(
        "N_eff / N_eff(null) = 1  (as broad as the null)",
        xy=(temperatures[0], 1.0), xytext=(4, 4), textcoords="offset points",
        fontsize=7.5, color=NULL_STYLE["color"],
    )
    axes.set_xlabel("nucleus temperature  T")
    axes.set_ylabel("effective support relative to the uniform null  (left axis)")
    axes.grid(True, alpha=0.25)

    agreement = axes.twinx()
    agreement.plot(
        temperatures,
        np.array(summary["conditions"]["real"]["metrics"]["agreement_with_greedy"]["mean"]),
        color="#b8860b", marker="s", markersize=5, linewidth=1.8, linestyle="-.",
        label="real:  fraction of positions matching greedy argmax",
    )
    agreement.set_ylabel(
        "fraction of positions matching greedy argmax  (right axis)", color="#b8860b"
    )
    agreement.tick_params(axis="y", labelcolor="#b8860b")
    agreement.set_ylim(-0.02, 1.02)

    handles, labels = axes.get_legend_handles_labels()
    extra_handles, extra_labels = agreement.get_legend_handles_labels()
    axes.legend(
        handles + extra_handles, labels + extra_labels,
        loc="center left", fontsize=8, frameon=True,
    )
    axes.set_title(
        "Guess breadth relative to the null, and identity agreement with greedy", fontsize=11
    )
    return save_figure(figure, directory, "figure7_temperature_support_and_greedy_agreement")


def _sweep_fractions(record: Any, condition: str) -> np.ndarray:
    """``[S, T, V]`` sweep fractions, normalized per temperature."""

    counts = record.sweep_counts(condition).astype(np.float64)
    return counts / counts.sum(axis=-1, keepdims=True)


def _probability_scale(values: np.ndarray, requested: str) -> bool:
    """Whether to use a logarithmic probability axis.

    ``auto`` follows the observed dynamic range: logarithmic once the values
    span at least a decade, linear otherwise, because a logarithmic axis over a
    fraction of a decade magnifies noise and flattens real structure. Whichever
    is used is written into the axis label, so a reader never has to infer it.
    """

    if requested == "log":
        return True
    if requested == "linear":
        return False
    positive = values[values > 0]
    return positive.size > 0 and float(positive.max() / positive.min()) >= 10.0


def plot_ranked_predictive_probabilities(
    record: Any,
    directory: str | Path,
    *,
    y_scale: str = "auto",
) -> list[Path]:
    """Figure 8 -- how concentrated is one individual next-token prediction?

    Each position's raw ``T = 1`` probabilities over the eligible support are
    ranked **within that position**, and only then averaged at equal rank across
    positions. The curve is that mean profile, averaged again over
    initializations, against a uniform ``1/K`` reference.

    **This is not figure 1.** Figure 1 ranks token guess *frequencies accumulated
    across positions*, describing the aggregate distribution of what the model
    picks. This figure ranks *probabilities within each single prediction*,
    describing the shape of one predictive vector. A model can be nearly flat at
    every individual position and still produce a sharply peaked aggregate, or
    the reverse, so neither figure implies the other.

    The distribution shown is upstream of both sampling policies: no temperature
    scaling, no top-p truncation, no greedy or nucleus decision.
    """

    from llm_behavior_lab.analysis.predictive import (
        predictive_probability_summary,
        ranked_probability_profile,
    )

    if not record.has_predictive_probability_analysis:
        raise ValueError(
            "This record carries no raw predictive-probability analysis, so "
            "figure 8 has nothing to draw."
        )
    if y_scale not in ("auto", "log", "linear"):
        raise ValueError(f"y_scale must be 'auto', 'log', or 'linear'; got {y_scale!r}.")

    profile = ranked_probability_profile(record)
    summary = predictive_probability_summary(record)
    ranks = profile["ranks"]
    mean = profile["mean"]
    uniform = record.uniform_probability
    large = _is_large(record)

    figure = _new_figure(width=8.0, height=5.5)
    axes = figure.subplots()

    # Two nested bands rather than twelve curves: the full spread across
    # initializations, and the much narrower uncertainty on their mean.
    if profile["profiles"].shape[0] > 1:
        axes.fill_between(
            ranks,
            _positive(profile["low"]),
            _positive(profile["high"]),
            color="#1f77b4",
            alpha=0.15,
            linewidth=0,
            rasterized=large,
            label="initialization min-max",
        )
        axes.fill_between(
            ranks,
            _positive(mean - profile["sem"]),
            _positive(mean + profile["sem"]),
            color="#1f77b4",
            alpha=0.35,
            linewidth=0,
            rasterized=large,
            label="mean ± SEM across initializations",
        )
    axes.plot(
        ranks,
        _positive(mean),
        color="#1f77b4",
        linewidth=1.4,
        label="mean ranked predictive probability",
        **_profile_kwargs(large),
    )
    axes.axhline(
        uniform,
        color=NULL_STYLE["color"],
        linestyle="--",
        linewidth=1.1,
        label=f"uniform categorical 1/K = {uniform:.3g}",
    )

    if _probability_scale(mean, y_scale):
        axes.set_yscale("log")
        scale_note = "log scale"
    else:
        scale_note = "linear scale"
    if large:
        axes.set_xscale("log")
    axes.set_xlabel("within-position probability rank  r  (ranked inside each position)")
    axes.set_ylabel(f"mean predictive probability  Pbar(r)  [{scale_note}]")
    # Padded: on a linear axis matplotlib puts a shared "1e-5" exponent above the
    # y axis, exactly where an unpadded title sits.
    axes.set_title(
        "Ranked predictive probabilities within a single prediction (raw, T = 1)",
        pad=14,
    )
    axes.grid(True, which="both", alpha=0.25)
    axes.legend(loc="upper right", fontsize=8, frameon=True)

    axes.text(
        0.02,
        0.05,
        "\n".join(
            [
                f"D = {_format_count(summary['num_positions'])} positions,"
                f"  I = {summary['num_initializations']} initializations",
                f"eligible K = {_format_count(summary['eligible_vocab_size'])}",
                f"mean rank-1 probability = {mean[0]:.4g}",
                f"uniform 1/K = {uniform:.4g}"
                f"   (rank 1 is {summary['rank1_over_uniform']:.1f}x uniform)",
            ]
        ),
        transform=axes.transAxes,
        fontsize=7.5,
        va="bottom",
        ha="left",
        bbox=_ANNOTATION_BOX,
    )
    return save_figure(figure, directory, "figure8_ranked_predictive_probabilities")


def plot_max_predictive_probability(
    record: Any,
    directory: str | Path,
    *,
    x_scale: str = "auto",
) -> list[Path]:
    """Figure 9 -- how much probability does the greedy-winning token actually get?

    Greedy decoding always selects the top-ranked token. That says nothing about
    how much mass the token carries, and this figure separates the two: it shows
    the empirical distribution of ``p_max[d]``, the probability of the selected
    token, over the evaluated positions.

    Drawn as an ECDF rather than a histogram. The question is "how large is
    ``p_max`` at a typical position", which is a quantile question, and an ECDF
    answers it without a bin width to choose.

    The values are **not silently pooled**. Each initialization contributes its
    own curve, and the pooled curve is drawn over them, because ``I * D`` values
    are not one independent sample: every position within an initialization
    shares one set of weights.
    """

    from llm_behavior_lab.analysis.predictive import (
        max_probability_summary,
        predictive_probability_summary,
    )

    if not record.has_predictive_probability_analysis:
        raise ValueError(
            "This record carries no raw predictive-probability analysis, so "
            "figure 9 has nothing to draw."
        )
    if x_scale not in ("auto", "log", "linear"):
        raise ValueError(f"x_scale must be 'auto', 'log', or 'linear'; got {x_scale!r}.")

    values = np.asarray(record.predictive_max_probabilities, dtype=np.float64)
    summary = predictive_probability_summary(record)
    maximum = max_probability_summary(record)
    uniform = record.uniform_probability

    figure = _new_figure(width=8.0, height=5.5)
    axes = figure.subplots()

    # Evaluated on a fixed quantile grid: exact where it matters and a few
    # hundred vertices per curve instead of tens of thousands.
    levels = np.linspace(0.0, 1.0, 501)
    for index in range(values.shape[0]):
        axes.plot(
            np.quantile(values[index], levels),
            levels,
            color="#1f77b4",
            alpha=0.35,
            linewidth=0.8,
            label="per initialization" if index == 0 else None,
        )
    axes.plot(
        np.quantile(values.reshape(-1), levels),
        levels,
        color="#08306b",
        linewidth=1.8,
        label="pooled over all initializations",
    )
    axes.axvline(
        uniform,
        color=NULL_STYLE["color"],
        linestyle="--",
        linewidth=1.1,
        label=f"uniform 1/K = {uniform:.3g}",
    )
    median = maximum["pooled"]["p50"]
    axes.axvline(median, color="#d62728", linestyle=":", linewidth=1.1, label=f"median = {median:.3g}")

    if _probability_scale(values.reshape(-1), x_scale):
        axes.set_xscale("log")
        scale_note = "log scale"
    else:
        scale_note = "linear scale"
    axes.set_xlabel(
        f"probability of the greedy-selected token  p_max(d)  [{scale_note}]"
    )
    axes.set_ylabel("empirical CDF over evaluation positions")
    axes.set_title("How much probability does the greedy winner carry? (raw, T = 1)")
    axes.set_ylim(0.0, 1.0)
    axes.grid(True, which="both", alpha=0.25)
    axes.legend(loc="lower right", fontsize=8, frameon=True)

    pooled = maximum["pooled"]
    axes.text(
        0.02,
        0.97,
        "\n".join(
            [
                f"D = {_format_count(summary['num_positions'])},"
                f"  I = {summary['num_initializations']},"
                f"  K = {_format_count(summary['eligible_vocab_size'])}",
                f"mean {pooled['mean']:.4g}   median {pooled['p50']:.4g}"
                f"   max {pooled['p100']:.4g}",
                f"p05 {pooled['p05']:.3g}  p25 {pooled['p25']:.3g}"
                f"  p75 {pooled['p75']:.3g}  p95 {pooled['p95']:.3g}",
                f"median / uniform = {maximum['median_over_uniform']:.1f}x",
            ]
        ),
        transform=axes.transAxes,
        fontsize=7.5,
        va="top",
        ha="left",
        bbox=_ANNOTATION_BOX,
    )
    return save_figure(figure, directory, "figure9_max_predictive_probability_distribution")


def plot_gradient_vs_guess_bias(
    record: Any,
    directory: str | Path,
    *,
    x_scale: str = "auto",
    num_labels: int = 4,
) -> list[Path]:
    """Supplementary single-temperature scatter, the canonical ``T = 1`` slice.

    Superseded as figure 10 by the six-panel temperature version, and kept for
    the case where only the canonical observable is wanted.

    One marker per token with ``n_i > 0``: its exact mean full-parameter gradient
    norm ``G_i`` against the fraction ``q_i`` of the *same* evaluation positions
    at which greedy argmax selected it, coloured by the whole-split corpus
    fraction ``p_i``.

    Three scale decisions are made from the data rather than by habit.

    The **y axis is symmetric-log with the linear region ending at one guess**,
    ``1/D``. Most tokens are never greedily guessed, and ``q_i = 0`` is a measured
    outcome, not missing data; a plain logarithmic axis would delete exactly the
    tokens the figure exists to show. Below ``1/D`` the only attainable value is
    zero, so the linear region is precisely the gap between "never guessed" and
    "guessed once", and a reference line marks it.

    The **x axis follows the observed dynamic range**: logarithmic once ``G_i``
    spans at least a decade, linear otherwise, because a log axis over a
    half-decade spreads noise and hides structure. The choice is recorded in the
    axis label so a reader never has to guess which one they are looking at.

    The **colour is logarithmic** over positive ``p_i`` with a perceptually
    uniform map. Corpus frequency is heavy-tailed; a linear map would collapse
    every token but the few most common into one shade. Nothing is clipped.
    """

    from llm_behavior_lab.analysis.gradients import (
        gradient_guess_correlations,
        gradient_guess_table,
        gradient_observable_summary,
    )

    if not record.has_position_gradients:
        raise ValueError(
            "This record carries no per-position gradient analysis, so the "
            "supplementary T = 1 gradient scatter has nothing to draw."
        )
    if x_scale not in ("auto", "log", "linear"):
        raise ValueError(f"x_scale must be 'auto', 'log', or 'linear'; got {x_scale!r}.")

    table = gradient_guess_table(record)
    summary = gradient_observable_summary(record)
    correlations = gradient_guess_correlations(record)

    plotted = table["target_occurrence_count"] > 0
    norms = table["mean_gradient_norm"][plotted]
    guesses = table["greedy_guess_fraction"][plotted]
    corpus = table["corpus_fraction"][plotted]
    token_ids = table["token_id"][plotted]

    # Every plotted token occurs as a target, so it occurs in the split and has
    # positive corpus mass. Guarded rather than assumed: a token that somehow had
    # none cannot be placed on a logarithmic colour scale, and dropping it
    # silently would misstate the token count.
    coloured = corpus > 0
    dropped = int((~coloured).sum())

    from matplotlib.colors import LogNorm

    figure = _new_figure(width=8.0, height=6.0)
    axes = figure.subplots()

    one_guess = 1.0 / float(table["num_positions"])
    marks = axes.scatter(
        norms[coloured],
        guesses[coloured],
        c=corpus[coloured],
        s=7,
        alpha=0.55,
        cmap="viridis",
        norm=LogNorm(vmin=float(corpus[coloured].min()), vmax=float(corpus[coloured].max())),
        edgecolors="none",
        rasterized=True,
    )
    colourbar = figure.colorbar(marks, ax=axes, pad=0.02)
    colourbar.set_label("Whole-corpus empirical token frequency  p(i)", fontsize=9)

    axes.axhline(
        one_guess,
        color="#888888",
        linewidth=0.9,
        linestyle=":",
        zorder=0,
    )
    axes.annotate(
        f"one guess = 1/D = {one_guess:.2g}\nbelow: never guessed (q = 0)",
        (0.985, one_guess),
        xycoords=("axes fraction", "data"),
        textcoords="offset points",
        xytext=(0, 4),
        fontsize=7,
        color="#666666",
        ha="right",
        va="bottom",
    )

    positive_norms = norms[norms > 0]
    use_log_x = x_scale == "log" or (
        x_scale == "auto"
        and positive_norms.size > 0
        and float(positive_norms.max() / positive_norms.min()) >= 10.0
    )
    if use_log_x:
        axes.set_xscale("log")
    # Linear below one guess keeps q = 0 on the axis; logarithmic above it keeps
    # the four decades of guess frequency legible.
    axes.set_yscale("symlog", linthresh=one_guess, linscale=0.6)
    axes.set_ylim(bottom=0.0)

    scale_note = "log scale" if use_log_x else "linear scale"
    axes.set_xlabel(
        f"mean single-position parameter-gradient norm  G(i)  [{scale_note}]"
    )
    axes.set_ylabel("greedy guess fraction  q(i)   [symlog below 1/D]")
    axes.set_title("Gradient magnitude vs. initial guessing bias, by token")
    axes.grid(True, which="both", alpha=0.22)

    tokens = record.tokens
    if tokens and num_labels > 0:
        # A small deterministic set of extremes, not a label per token: at a few
        # thousand marks, labelling broadly destroys the figure it annotates.
        candidates: list[tuple[str, int]] = [
            ("largest q", int(np.argmax(guesses))),
            ("largest G", int(np.argmax(norms))),
            ("smallest G", int(np.argmin(norms))),
            ("largest p", int(np.argmax(corpus))),
        ]
        seen: set[int] = set()
        for label, index in candidates[:num_labels]:
            if index in seen or not coloured[index]:
                continue
            seen.add(index)
            axes.annotate(
                f"{escape_token_label(tokens[int(token_ids[index])])} ({label})",
                (norms[index], guesses[index]),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=7,
                color="#222222",
            )

    rho = correlations["spearman_gradient_vs_guess"]
    lines = [
        f"initialization {summary['initialization_index']}"
        f" / seed {record.gradient_analysis.get('model_seed', 'n/a')}",
        f"D = {_format_count(summary['num_positions'])} positions"
        f"{'' if summary['covers_all_positions'] else ' (SUBSET)'}",
        f"tokens plotted (n(i) > 0) = {_format_count(summary['num_plotted_tokens'])}",
        f"never guessed: {_format_count(summary['zero_guess_tokens'])}"
        f" ({summary['zero_guess_fraction']:.1%})",
        f"Spearman rho(G, q) = {rho:.4f}",
    ]
    if dropped:
        lines.append(f"omitted, no corpus mass: {_format_count(dropped)}")
    axes.text(
        0.02,
        0.97,
        "\n".join(lines),
        transform=axes.transAxes,
        fontsize=7.5,
        va="top",
        ha="left",
        bbox=_ANNOTATION_BOX,
    )
    return save_figure(
        figure, directory, "supplementary_t1_gradient_vs_initial_guess_bias"
    )


def generate_all_figures(record: Any, directory: str | Path) -> list[Path]:
    """Write every figure for one record and return the paths in figure order."""

    written: list[Path] = []
    written.extend(plot_sampling_adequacy(record, directory))
    written.extend(plot_ranked_frequency_profiles(record, directory))
    written.extend(plot_token_wise_mismatch(record, directory))
    written.extend(plot_token_identity_scatter(record, directory))
    if record.has_input_structure:
        written.extend(plot_input_structure_profiles(record, directory))
    if record.has_temperature_sweep:
        written.extend(plot_temperature_ranked_profiles(record, directory))
        written.extend(plot_temperature_ranked_distances(record, directory))
        written.extend(plot_temperature_support_and_agreement(record, directory))
    if record.has_predictive_probability_analysis:
        written.extend(plot_ranked_predictive_probabilities(record, directory))
        written.extend(plot_max_predictive_probability(record, directory))
    if record.has_temperature_gradient_analysis:
        written.extend(plot_temperature_gradient_vs_guess_bias(record, directory))
    elif record.has_position_gradients:
        # Older records carry only the canonical slice; draw the supplementary
        # single-temperature scatter rather than nothing.
        written.extend(plot_gradient_vs_guess_bias(record, directory))
    if record.has_temperature_confidence_analysis:
        written.extend(plot_temperature_ranked_predictive_probabilities(record, directory))
        written.extend(plot_temperature_max_predictive_probability(record, directory))
        written.extend(plot_greedy_confidence_vs_temperature(record, directory))
    if record.has_mean_token_probabilities:
        written.extend(plot_ranked_mean_token_probabilities(record, directory))
    # Figures 15 and 16 pair whole-experiment statistics with the gradient
    # table, so they exist only when the gradient analysis covered every
    # position. A subset run draws figures 0-14 and simply omits these two.
    if record.has_temperature_gradient_analysis and _covers_all_gradient_positions(record):
        # R != 1 leaves figure 15 undefined; the rest of the set still renders.
        if record.has_temperature_sweep and record.num_replicates == 1:
            written.extend(plot_gradient_vs_nucleus_guess_bias(record, directory))
        if record.has_mean_token_probabilities:
            written.extend(plot_gradient_vs_mean_probability(record, directory))

    # Figures 17-21. They read the gradient table against whole-experiment
    # quantities, so they need the same paired position set figures 15 and 16
    # require, and figure 20 additionally needs the per-position sketches.
    if _supports_initial_gradient_figures(record):
        written.extend(plot_initial_performance_and_bias(record, directory))
        written.extend(plot_initial_gradient_split(record, directory))
        written.extend(plot_initial_logit_correction(record, directory))
        written.extend(plot_correction_provenance(record, directory))
        if _supports_gradient_clustering(record):
            written.extend(plot_gradient_directional_clustering(record, directory))
    return written


def _supports_initial_gradient_figures(record: Any) -> bool:
    """Whether figures 17-19 and 21 have everything they read."""

    return (
        record.has_temperature_gradient_analysis
        and record.has_temperature_confidence_analysis
        and _covers_all_gradient_positions(record)
    )


def _supports_gradient_clustering(record: Any) -> bool:
    """Whether figure 20 has the per-position gradient sketches it needs."""

    from llm_behavior_lab.analysis.gradient_clustering import has_gradient_sketches

    return has_gradient_sketches(record) and record.gradient_position_norms is not None


def _covers_all_gradient_positions(record: Any) -> bool:
    """Whether the gradient analysis spanned every evaluation position."""

    from llm_behavior_lab.analysis.gradients import _covers_all_positions

    try:
        return _covers_all_positions(record)
    except ValueError:
        return False


def plot_temperature_ranked_predictive_probabilities(
    record: Any,
    directory: str | Path,
    *,
    y_scale: str = "auto",
) -> list[Path]:
    """Figure 11 -- ranked predictive profiles across the diagnostic temperatures.

    One curve per temperature, showing how lowering ``T`` concentrates mass into
    the leading ranks. Every curve describes the **same greedy decisions**:
    softmax is strictly increasing, so ``argmax softmax(z/T) = argmax z`` for
    every positive ``T``. Only the confidence attached to those decisions moves.

    That is what separates this from the nucleus sweep. The sweep *samples* from
    the transformed distribution, so its selections really do change with
    temperature; here nothing is sampled and nothing is truncated.

    Uncertainty is drawn only for the canonical ``T = 1`` reference. Seven
    overlapping bands would be an unreadable forest, and the initialization
    spread is far smaller than the separation between temperatures -- so one band
    establishes the scale and the rest stay legible.
    """

    from llm_behavior_lab.analysis.predictive import temperature_ranked_profiles

    if not record.has_temperature_confidence_analysis:
        raise ValueError(
            "This record carries no temperature-confidence analysis, so figure 11 "
            "has nothing to draw."
        )
    if y_scale not in ("auto", "log", "linear"):
        raise ValueError(f"y_scale must be 'auto', 'log', or 'linear'; got {y_scale!r}.")

    profiles = temperature_ranked_profiles(record)
    temperatures = profiles["temperatures"]
    ranks = profiles["ranks"]
    uniform = record.uniform_probability
    large = _is_large(record)
    colours = _temperature_colours(temperatures)

    figure = _new_figure(width=8.5, height=5.8)
    axes = figure.subplots()

    for index, temperature in enumerate(temperatures):
        canonical = float(temperature) == 1.0
        mean = profiles["mean"][index]
        if canonical:
            axes.fill_between(
                ranks,
                _positive(mean - profiles["sem"][index]),
                _positive(mean + profiles["sem"][index]),
                color="#333333",
                alpha=0.30,
                linewidth=0,
                rasterized=large,
                label="T = 1 mean ± SEM",
            )
        axes.plot(
            ranks,
            _positive(mean),
            color="#111111" if canonical else colours[index],
            linewidth=2.0 if canonical else 1.2,
            linestyle="-" if canonical else "--",
            label=f"T = {temperature:g}" + ("  (canonical)" if canonical else ""),
            **_profile_kwargs(large),
        )
    axes.axhline(
        uniform,
        color=NULL_STYLE["color"],
        linestyle=":",
        linewidth=1.1,
        label=f"uniform 1/K = {uniform:.3g}",
    )

    if _probability_scale(profiles["mean"], y_scale):
        axes.set_yscale("log")
        scale_note = "log scale"
    else:
        scale_note = "linear scale"
    if large:
        axes.set_xscale("log")
    axes.set_xlabel("within-position probability rank  r  (ranked inside each position)")
    axes.set_ylabel(f"mean predictive probability  Pbar_T(r)  [{scale_note}]")
    axes.set_title("Confidence geometry vs. temperature, at fixed greedy decisions")
    axes.grid(True, which="both", alpha=0.22)
    axes.legend(loc="upper right", fontsize=7.5, frameon=True, ncol=2)
    axes.text(
        0.02,
        0.05,
        "argmax softmax(z/T) = argmax z for every T > 0:\n"
        "every curve describes the SAME greedy decisions",
        transform=axes.transAxes,
        fontsize=7.5,
        va="bottom",
        ha="left",
        bbox=_ANNOTATION_BOX,
    )
    return save_figure(figure, directory, "figure11_temperature_ranked_predictive_probabilities")


def plot_temperature_max_predictive_probability(
    record: Any,
    directory: str | Path,
    *,
    panel_temperatures: Sequence[float] | None = None,
    x_scale: str = "auto",
) -> list[Path]:
    """Figure 12 -- the greedy winner's confidence, one ECDF panel per temperature.

    The same representation as figure 9, repeated across the six non-canonical
    confidence-grid temperatures so the panels are directly comparable. These
    come from the temperature-confidence grid, not from the nucleus sweep:
    nothing here is sampled or truncated. The canonical ``T = 1`` median is
    marked in each panel as a fixed reference. Axis limits are shared across
    panels, so apparent sharpening is the data rather than a rescaling.

    Kept as an ECDF: the question is "how large is the greedy winner's
    probability at a typical position", which is a quantile question.
    """

    from llm_behavior_lab.analysis.predictive import temperature_confidence_summary

    if not record.has_temperature_confidence_analysis:
        raise ValueError(
            "This record carries no temperature-confidence analysis, so figure 12 "
            "has nothing to draw."
        )
    if x_scale not in ("auto", "log", "linear"):
        raise ValueError(f"x_scale must be 'auto', 'log', or 'linear'; got {x_scale!r}.")

    summary = temperature_confidence_summary(record)
    grid = list(record.confidence_temperatures)
    chosen = (
        [value for value in grid if value != 1.0]
        if panel_temperatures is None
        else [float(value) for value in panel_temperatures]
    )
    if not chosen:
        raise ValueError("No panel temperatures are available in this record.")

    maxima = np.asarray(record.predictive_temperature_max_probabilities, dtype=np.float64)
    uniform = record.uniform_probability
    canonical_median = None
    if 1.0 in grid:
        canonical_median = summary["rows"][grid.index(1.0)]["max_probability"]["p50"]

    columns = 3
    rows = int(np.ceil(len(chosen) / columns))
    figure = _new_figure(width=4.1 * columns, height=3.4 * rows)
    panels = figure.subplots(rows, columns, squeeze=False, sharex=True, sharey=True)
    levels = np.linspace(0.0, 1.0, 401)

    # Shared limits so a panel cannot look sharper merely by being rescaled.
    selected = [maxima[:, grid.index(value), :] for value in chosen]
    # The uniform reference is included in the shared range even when no panel's
    # data reaches it: it is the scientific floor every panel is read against,
    # and excluding it would leave "how far above chance" unanswerable by eye.
    lower = min([float(values.min()) for values in selected] + [uniform])
    upper = max(float(values.max()) for values in selected)
    use_log = _probability_scale(np.concatenate([v.reshape(-1) for v in selected]), x_scale)

    for panel_index, temperature in enumerate(chosen):
        axes = panels[panel_index // columns][panel_index % columns]
        values = maxima[:, grid.index(temperature), :]
        for initialization in range(values.shape[0]):
            axes.plot(
                np.quantile(values[initialization], levels),
                levels,
                color="#1f77b4",
                alpha=0.30,
                linewidth=0.7,
            )
        axes.plot(
            np.quantile(values.reshape(-1), levels),
            levels,
            color="#08306b",
            linewidth=1.6,
        )
        median = float(np.median(values))
        axes.axvline(median, color="#d62728", linestyle=":", linewidth=1.0)
        if lower <= uniform <= upper:
            axes.axvline(uniform, color=NULL_STYLE["color"], linestyle="--", linewidth=0.9)
        if canonical_median is not None and lower <= canonical_median <= upper:
            axes.axvline(canonical_median, color="#111111", linestyle="-.", linewidth=0.9)

        row = summary["rows"][grid.index(temperature)]["max_probability"]
        axes.set_title(f"T = {temperature:g}", fontsize=10)
        axes.text(
            0.03,
            0.97,
            f"median {row['p50']:.3g}\np95 {row['p95']:.3g}\np99 {row['p99']:.3g}\n"
            f"median/uniform {row['p50'] / uniform:.1f}x",
            transform=axes.transAxes,
            fontsize=7,
            va="top",
            ha="left",
            bbox=_ANNOTATION_BOX,
        )
        axes.grid(True, which="both", alpha=0.22)
        if use_log:
            axes.set_xscale("log")
        axes.set_xlim(lower, upper)
        axes.set_ylim(0.0, 1.0)

    for panel_index in range(len(chosen), rows * columns):
        panels[panel_index // columns][panel_index % columns].set_visible(False)
    for column in range(columns):
        panels[rows - 1][column].set_xlabel(
            f"p_max(d)  [{'log' if use_log else 'linear'} scale]"
        )
    for row_index in range(rows):
        panels[row_index][0].set_ylabel("empirical CDF")

    figure.suptitle(
        "Greedy-winner confidence across temperature (identical greedy decisions)",
        fontsize=12,
    )
    figure.text(
        0.5,
        0.005,
        "thin blue: per initialization   dark blue: pooled   red dotted: panel median   "
        "black dash-dot: T = 1 median   grey dashed: uniform 1/K",
        ha="center",
        fontsize=7.5,
    )
    return save_figure(figure, directory, "figure12_temperature_max_predictive_probability")


def plot_greedy_confidence_vs_temperature(record: Any, directory: str | Path) -> list[Path]:
    """Figure 13 -- how fast confidence rises as temperature falls.

    Two stacked panels rather than one: ``p_max`` is a probability and the
    effective support is a token count, and forcing them onto a shared axis to
    save space would misrepresent both.

    Temperature is drawn on a logarithmic axis at its true numerical spacing --
    the grid is not uniformly spaced, and plotting it as if it were would distort
    the shape of the very trend the figure exists to show.
    """

    from llm_behavior_lab.analysis.predictive import temperature_confidence_summary

    if not record.has_temperature_confidence_analysis:
        raise ValueError(
            "This record carries no temperature-confidence analysis, so figure 13 "
            "has nothing to draw."
        )

    summary = temperature_confidence_summary(record)
    temperatures = np.asarray([row["temperature"] for row in summary["rows"]])
    order = np.argsort(temperatures)
    temperatures = temperatures[order]
    rows = [summary["rows"][index] for index in order]
    uniform = record.uniform_probability

    figure = _new_figure(width=7.5, height=6.8)
    top, bottom = figure.subplots(2, 1, sharex=True)

    for key, label, style in (
        ("p50", "median p_max", {"color": "#08306b", "marker": "o"}),
        ("mean", "mean p_max", {"color": "#1f77b4", "marker": "s"}),
        ("p95", "p95 p_max", {"color": "#6baed6", "marker": "^"}),
    ):
        top.plot(
            temperatures,
            [row["max_probability"][key] for row in rows],
            linewidth=1.4,
            markersize=4,
            label=label,
            **style,
        )
    top.axhline(
        uniform,
        color=NULL_STYLE["color"],
        linestyle="--",
        linewidth=1.0,
        label=f"uniform 1/K = {uniform:.3g}",
    )
    top.set_yscale("log")
    top.set_ylabel("probability of the greedy token")
    top.grid(True, which="both", alpha=0.22)
    top.legend(fontsize=8, loc="upper right")
    top.set_title("Confidence of fixed greedy decisions vs. temperature")

    bottom.plot(
        temperatures,
        [row["effective_support"] for row in rows],
        color="#2ca02c",
        marker="o",
        markersize=4,
        linewidth=1.4,
        label="effective support  exp(mean H)",
    )
    bottom.axhline(
        record.eligible_vocab_size,
        color=NULL_STYLE["color"],
        linestyle="--",
        linewidth=1.0,
        label=f"eligible K = {_format_count(record.eligible_vocab_size)}",
    )
    bottom.set_yscale("log")
    bottom.set_xscale("log")
    bottom.set_xlabel("softmax temperature T  (log axis, true spacing)")
    bottom.set_ylabel("effective support (tokens)")
    bottom.grid(True, which="both", alpha=0.22)
    bottom.legend(fontsize=8, loc="upper left")

    for axes in (top, bottom):
        axes.axvline(1.0, color="#111111", linestyle="-.", linewidth=0.9, alpha=0.7)
    # Labelled on the lower panel: the upper one's legend already occupies the
    # corner an annotation there would land in.
    bottom.annotate(
        "T = 1 reference",
        (1.0, bottom.get_ylim()[0]),
        textcoords="offset points",
        xytext=(5, 10),
        fontsize=7.5,
        color="#111111",
    )
    return save_figure(figure, directory, "figure13_greedy_confidence_vs_temperature")


def plot_temperature_gradient_vs_guess_bias(
    record: Any,
    directory: str | Path,
    *,
    panel_temperatures: Sequence[float] | None = None,
    x_limits: str = "per_panel",
    y_transform: str = "count_log",
) -> list[Path]:
    """Figure 10 -- gradient magnitude vs. a *fixed* guessing bias, across temperature.

    One panel per temperature, each a token scatter of ``G_i(T)`` against
    ``q_i``, coloured by the whole-corpus fraction ``p_i``.

    The design is controlled by an exact invariance. Softmax is strictly
    increasing, so ``argmax softmax(z/T) = argmax z`` and the greedy guess
    fraction ``q_i`` is *identical in every panel*. So is ``n_i``, which counts
    targets, and so is ``p_i``. Only the gradient moves, because temperature is
    inside the loss: ``ell_T(d) = -log softmax(z_d/T)[y_d]``, whose logit
    gradient carries both an explicit ``1/T`` and a sharpening ``p_T``. Any
    change visible across panels therefore comes entirely from the gradient side.

    The y axis is shared and identical by construction. It keeps the
    symmetric-log treatment of the single-temperature figure, with the linear
    region ending at one guess, so tokens that were never greedily guessed --
    a measured outcome, and most of them -- stay on the axis instead of being
    deleted by a logarithmic scale.

    The x axis is the decision that has to be made from the data. ``G_i(T)``
    can move by orders of magnitude across the grid, so shared limits are used
    only when every panel stays readable within them; otherwise each panel gets
    its own limits and **says so in its title**, because a silent rescaling would
    invent a comparison the figure cannot support.
    """

    from matplotlib.colors import LogNorm

    from llm_behavior_lab.analysis.gradients import (
        temperature_gradient_summary,
        temperature_gradient_table,
    )

    if not record.has_temperature_gradient_analysis:
        raise ValueError(
            "This record carries no temperature-conditioned gradient analysis, so "
            "figure 10 has nothing to draw."
        )
    if x_limits not in ("shared", "per_panel"):
        raise ValueError(f"x_limits must be 'shared' or 'per_panel'; got {x_limits!r}.")

    grid = list(record.gradient_temperature_grid)
    chosen = (
        [value for value in grid if value != 1.0]
        if panel_temperatures is None
        else [float(value) for value in panel_temperatures]
    )
    if not chosen:
        raise ValueError("No panel temperatures are available in this record.")

    summary = temperature_gradient_summary(record)
    tables = [temperature_gradient_table(record, value) for value in chosen]
    plotted = tables[0]["target_occurrence_count"] > 0
    guesses = tables[0]["greedy_guess_fraction"][plotted]
    corpus = tables[0]["corpus_fraction"][plotted]
    coloured = corpus > 0
    dropped = int((~coloured).sum())
    one_guess = 1.0 / float(tables[0]["num_positions"])

    norms = [table["mean_gradient_norm"][plotted] for table in tables]

    # -- y: a count-aware, zero-preserving display transform -----------------
    #
    # q(i) = k(i) / D with k an integer guess count, and the distribution is
    # strongly zero-inflated. A logarithmic axis deletes the zeros outright, and
    # symlog keeps them but still spends most of the height on the few tokens
    # with large k. Plotting log10(1 + k) instead maps q = 0 to exactly 0, puts a
    # full 0.30 of height between "never guessed" and "guessed once", expands the
    # low counts where nearly all tokens live, and compresses the tail.
    #
    # It is monotone and exactly invertible, so it reorders nothing and discards
    # nothing. The axis still *reads* as q(i): the ticks are labelled k/D.
    counts = np.asarray(tables[0]["greedy_guess_count"][plotted], dtype=np.float64)
    if y_transform not in ("count_log", "symlog"):
        raise ValueError(
            f"y_transform must be 'count_log' or 'symlog'; got {y_transform!r}."
        )
    use_count_log = y_transform == "count_log"
    y_values = np.log10(1.0 + counts) if use_count_log else guesses

    # -- x: limits from each temperature's own distribution -------------------
    #
    # G(i, T) sits at a different location for every T -- the explicit 1/T alone
    # moves the median by an order of magnitude across the grid -- so one shared
    # range leaves every cloud in a sliver of its panel. Each panel therefore
    # gets limits derived from its own finite values, padded slightly, covering
    # the full range so nothing is clipped.
    #
    # The cost is that horizontal position is no longer comparable between
    # panels, which the figure states rather than leaves to be inferred.
    finite = [values[np.isfinite(values) & (values > 0)] for values in norms]
    if any(values.size == 0 for values in finite):
        raise ValueError("Every panel needs at least one positive gradient norm.")
    if x_limits == "shared":
        lower = min(float(values.min()) for values in finite)
        upper = max(float(values.max()) for values in finite)
        panel_limits = [(lower / 1.15, upper * 1.15)] * len(chosen)
    else:
        panel_limits = [
            (float(values.min()) / 1.15, float(values.max()) * 1.15) for values in finite
        ]
    share_x = x_limits == "shared"

    columns = 3
    rows = int(np.ceil(len(chosen) / columns))
    figure = _new_figure(width=4.8 * columns, height=4.1 * rows)
    panels = figure.subplots(rows, columns, squeeze=False, sharey=True)
    norm = LogNorm(
        vmin=float(corpus[coloured].min()), vmax=float(corpus[coloured].max())
    )
    marks = None

    for panel_index, temperature in enumerate(chosen):
        axes = panels[panel_index // columns][panel_index % columns]
        values = norms[panel_index]
        marks = axes.scatter(
            values[coloured],
            y_values[coloured],
            c=corpus[coloured],
            s=9,
            alpha=0.45,
            cmap="viridis",
            norm=norm,
            edgecolors="none",
            rasterized=True,
        )
        axes.set_xscale("log")
        axes.set_xlim(*panel_limits[panel_index])
        _configure_gradient_panel_x_axis(axes, *panel_limits[panel_index])
        if use_count_log:
            # A hairline above zero, so the never-guessed population reads as a
            # population rather than as the axis frame.
            axes.axhline(0.0, color="#bbbbbb", linewidth=0.7, linestyle="-", zorder=0)
        else:
            axes.axhline(one_guess, color="#888888", linewidth=0.8, linestyle=":", zorder=0)
            axes.set_yscale("symlog", linthresh=one_guess, linscale=0.6)
            axes.set_ylim(bottom=0.0)
        axes.grid(True, which="both", alpha=0.20)

        row = summary["rows"][grid.index(temperature)]
        rho = row["spearman_gradient_vs_guess"]
        low, high = panel_limits[panel_index]
        axes.set_title(
            f"T = {temperature:g}\nG in [{low:.3g}, {high:.3g}]",
            fontsize=9.5,
        )
        axes.text(
            0.03,
            0.97,
            f"rho(G, q) = {rho:.4f}\nmedian G {row['token_gradient_norm']['p50']:.3g}",
            transform=axes.transAxes,
            fontsize=7,
            va="top",
            ha="left",
            bbox=_ANNOTATION_BOX,
        )

    if use_count_log:
        # Ticks at meaningful guess counts, labelled as the fraction they are.
        ladder = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
        largest = int(counts.max())
        selected = [value for value in ladder if value <= largest]
        # Add the true maximum only when it is far enough from the last ladder
        # tick to be legible; otherwise the two labels overprint each other.
        if largest and largest not in selected:
            gap = np.log10(1.0 + largest) - np.log10(1.0 + selected[-1])
            if gap > 0.25:
                selected.append(largest)
        top = float(np.log10(1.0 + max(largest, 1)))
        for row_index in range(rows):
            for column in range(columns):
                axes = panels[row_index][column]
                axes.set_yticks(np.log10(1.0 + np.asarray(selected, dtype=np.float64)))
                axes.set_yticklabels(
                    ["0"] + [f"{value}/D" for value in selected[1:]], fontsize=7.5
                )
                axes.set_ylim(-0.045 * top, top * 1.06)

    for panel_index in range(len(chosen), rows * columns):
        panels[panel_index // columns][panel_index % columns].set_visible(False)
    # Row 2's two-line titles otherwise sit on row 1's tick labels.
    figure.subplots_adjust(hspace=0.34)
    figure.supxlabel("mean gradient norm  G(i, T)   [log scale]", fontsize=10)
    figure.supylabel(
        "greedy guess fraction  q(i) = k/D"
        + (
            f"   [axis: log10(1 + k), D = {int(tables[0]['num_positions']):,}]"
            if use_count_log
            else "   [symlog below 1/D]"
        ),
        fontsize=10,
    )

    # One colorbar for the whole figure: the colour means the same thing in every
    # panel, and six of them would imply six different scales.
    colourbar = figure.colorbar(marks, ax=panels, pad=0.02, fraction=0.03)
    colourbar.set_label("Whole-corpus empirical token frequency  p(i)", fontsize=9)

    note = (
        "q(i), n(i) and p(i) are identical in every panel: argmax softmax(z/T) = argmax z, "
        "so only the gradient moves"
    )
    if not share_x:
        note += (
            "   |   x limits are per-panel: horizontal position is NOT comparable "
            "between panels"
        )
    if dropped:
        note += f"   |   omitted, no corpus mass: {_format_count(dropped)}"
    figure.suptitle(
        "Gradient magnitude vs. fixed greedy guessing bias, across loss temperature",
        fontsize=12,
    )
    # Directly under the title, where there is free space: at the foot it
    # collides with the shared x label.
    figure.text(0.5, 0.945, note, ha="center", fontsize=7.5, color="#444444")
    return save_figure(
        figure, directory, "figure10_temperature_gradient_vs_initial_guess_bias"
    )


def plot_ranked_mean_token_probabilities(
    record: Any,
    directory: str | Path,
    *,
    y_scale: str = "auto",
) -> list[Path]:
    """Figure 14 -- average at fixed token identity first, then rank.

    The complement of figure 11, and the comparison between the two is the point.

    Figure 11 ranks probabilities **inside each prediction** and then averages at
    equal rank. It asks: at a typical next-token prediction, how concentrated is
    the distribution? Token identity is discarded before averaging, so a model
    that is sharply peaked on a *different* token at every position still gives a
    steep curve.

    Figure 14 averages **at fixed token identity** across positions and ranks
    afterwards. It asks: do the *same* tokens systematically receive elevated
    probability across many different inputs?

    Reading the pair:

    * figure 11 steep, figure 14 comparatively flat -- individual predictions are
      concentrated, but on tokens that change with the input;
    * both steep -- individual predictions are concentrated **and** particular
      token identities are persistently favoured.

    That second case is what a persistent initialization guessing bias looks
    like, which is why the two curves are kept apart rather than merged.

    Ranking happens per initialization before initializations are summarized;
    averaging probabilities across initializations first would blur away
    precisely the identity structure being measured.
    """

    from llm_behavior_lab.analysis.predictive import ranked_mean_token_probabilities

    if not record.has_mean_token_probabilities:
        raise ValueError(
            "This record carries no identity-preserving mean token probabilities, "
            "so figure 14 has nothing to draw."
        )
    if y_scale not in ("auto", "log", "linear"):
        raise ValueError(f"y_scale must be 'auto', 'log', or 'linear'; got {y_scale!r}.")

    profile = ranked_mean_token_probabilities(record)
    temperatures = profile["temperatures"]
    ranks = profile["ranks"]
    uniform = record.uniform_probability
    large = _is_large(record)
    colours = _temperature_colours(temperatures)

    figure = _new_figure(width=8.5, height=5.8)
    axes = figure.subplots()

    for index, temperature in enumerate(temperatures):
        canonical = float(temperature) == 1.0
        mean = profile["mean"][index]
        if canonical:
            axes.fill_between(
                ranks,
                _positive(mean - profile["sem"][index]),
                _positive(mean + profile["sem"][index]),
                color="#333333",
                alpha=0.30,
                linewidth=0,
                rasterized=large,
                label="T = 1 mean ± SEM",
            )
        axes.plot(
            ranks,
            _positive(mean),
            color="#111111" if canonical else colours[index],
            linewidth=2.0 if canonical else 1.2,
            linestyle="-" if canonical else "--",
            label=f"T = {temperature:g}" + ("  (canonical)" if canonical else ""),
            **_profile_kwargs(large),
        )
    axes.axhline(
        uniform,
        color=NULL_STYLE["color"],
        linestyle=":",
        linewidth=1.1,
        label=f"uniform 1/K = {uniform:.3g}",
    )

    # Chosen from this figure's own range: identity-preserving averaging washes
    # out position-specific peaks, so the curve is generally flatter than figure
    # 11's and need not want the same scale.
    if _probability_scale(profile["mean"], y_scale):
        axes.set_yscale("log")
        scale_note = "log scale"
    else:
        scale_note = "linear scale"
    if large:
        axes.set_xscale("log")
    axes.set_xlabel("rank of token after averaging at fixed identity")
    axes.set_ylabel(f"mean token probability  Pbar_T(i)  [{scale_note}]")
    axes.set_title(
        "Persistent token-identity probability bias (average first, then rank)",
        pad=14,
    )
    axes.grid(True, which="both", alpha=0.22)
    axes.legend(loc="upper right", fontsize=7.5, frameon=True, ncol=2)
    axes.text(
        0.02,
        0.05,
        "averaged at FIXED token identity, then ranked\n"
        "figure 11 ranks within each prediction first: steep there with a flat\n"
        "curve here means concentrated predictions on input-dependent tokens",
        transform=axes.transAxes,
        fontsize=7.5,
        va="bottom",
        ha="left",
        bbox=_ANNOTATION_BOX,
    )
    return save_figure(figure, directory, "figure14_ranked_mean_token_probabilities")


def _common_probability_axis(panels: list[np.ndarray], requested: str) -> bool:
    """One axis transformation for every panel of a temperature comparison.

    Decided once from the pooled values, never per panel: the panels of figure 16
    show the *same physical quantity* at different temperatures, and letting one
    panel go logarithmic while another stayed linear would make a change of axis
    look like a change in the data.

    ``auto`` requires the pooled values to be **strictly positive** before
    choosing a logarithmic axis. ``_probability_scale`` drops non-positive
    entries before measuring the range, which is right for a ranked profile whose
    tail is padding but wrong here: ``pbar_i(T)`` can underflow to exactly zero at
    low temperature, and that is a measured outcome about a token the model gives
    no mass to. A logarithmic axis would delete precisely those tokens.
    """

    if requested == "log":
        return True
    if requested == "linear":
        return False
    pooled = np.concatenate([np.asarray(values, dtype=np.float64) for values in panels])
    if pooled.size == 0 or not np.all(pooled > 0.0):
        return False
    return float(pooled.max() / pooled.min()) >= 10.0


def _configure_gradient_panel_x_axis(axes: Any, low: float, high: float) -> None:
    """Shared x-tick convention for the three gradient scatter figures.

    Presentation only: the scale stays logarithmic, the data are untouched, and
    the per-panel limits passed in are exactly the ones the panel already uses.

    The default log ticker is the problem these panels have. Per-panel ``G``
    limits routinely span well under a decade, and in that regime matplotlib
    subdivides the axis finely and labels the minor ticks, so a panel ends up
    with ``10^1 1.2x10^1 1.4x10^1 ...`` running into its neighbour's labels.
    Restricting ``subs`` does not help: the sub-decade fallback ignores it.

    So the ticks are chosen explicitly from a 1-2-5 decade ladder clipped to the
    panel's own range, capped at four labels, and written as plain significant
    figures -- over a range like 0.33 to 18, ``0.5  1  2  5  10`` is simply
    easier to read than scientific notation. If the ladder lands fewer than two
    ticks inside a very narrow range, three geometrically spaced values are used
    instead, so no panel is ever left unlabelled.
    """

    from matplotlib.ticker import FixedFormatter, FixedLocator

    if not (np.isfinite(low) and np.isfinite(high)) or low <= 0.0 or high <= low:
        return

    decades = range(int(np.floor(np.log10(low))), int(np.ceil(np.log10(high))) + 1)
    ladder = [
        base * 10.0**decade for decade in decades for base in (1.0, 2.0, 5.0)
    ]
    ticks = [value for value in ladder if low <= value <= high]

    if len(ticks) < 2:
        ticks = list(np.geomspace(low, high, 3))
    elif len(ticks) > 4:
        step = int(np.ceil(len(ticks) / 4))
        ticks = ticks[::step]

    def label(value: float) -> str:
        if value >= 1000 or value < 0.001:
            return f"{value:.0e}".replace("e-0", "e-").replace("e+0", "e")
        text = f"{value:.3g}"
        return text

    axes.xaxis.set_major_locator(FixedLocator(ticks))
    axes.xaxis.set_major_formatter(FixedFormatter([label(value) for value in ticks]))
    axes.xaxis.set_minor_locator(FixedLocator([]))
    axes.tick_params(axis="x", which="both", labelsize=7.5)


def _gradient_panel_temperatures(
    record: Any, panel_temperatures: Sequence[float] | None
) -> list[float]:
    """The six displayed gradient temperatures: the grid without the canonical.

    Figure 10's convention, shared by its two siblings so the three are read as
    one controlled sequence.
    """

    grid = list(record.gradient_temperature_grid)
    chosen = (
        [value for value in grid if value != 1.0]
        if panel_temperatures is None
        else [float(value) for value in panel_temperatures]
    )
    if not chosen:
        raise ValueError("No panel temperatures are available in this record.")
    return chosen


def _plot_gradient_against(
    record: Any,
    directory: str | Path,
    *,
    tables: list[dict[str, Any]],
    chosen: list[float],
    y_key: str,
    count_key: str | None,
    count_total_key: str | None,
    y_label: str,
    title: str,
    note: str,
    stem: str,
    y_scale: str,
    panel_statistic,
) -> list[Path]:
    """Draw one six-panel gradient scatter, figure 10's geometry reused.

    Figures 15 and 16 differ from figure 10 only in the y quantity and in the
    axis treatment that quantity needs. Everything else -- per-panel x limits
    from each temperature's own finite values, the shared ``LogNorm`` colour over
    ``p_i``, the single figure-wide colorbar, the token set, the panel grid -- is
    the same, because the three are meant to be compared.

    Figure 10 itself is not routed through this helper: it stays exactly as it
    was written and validated.
    """

    from matplotlib.colors import LogNorm

    plotted = tables[0]["target_occurrence_count"] > 0
    corpus = tables[0]["corpus_fraction"][plotted]
    coloured = corpus > 0
    dropped = int((~coloured).sum())

    norms = [table["mean_gradient_norm"][plotted] for table in tables]
    finite = [values[np.isfinite(values) & (values > 0)] for values in norms]
    if any(values.size == 0 for values in finite):
        raise ValueError("Every panel needs at least one positive gradient norm.")
    panel_limits = [
        (float(values.min()) / 1.15, float(values.max()) * 1.15) for values in finite
    ]

    raw = [np.asarray(table[y_key], dtype=np.float64)[plotted] for table in tables]
    if count_key is not None:
        # Figure 10's zero-preserving count transform, reused verbatim so the
        # three figures' y axes are read the same way. A logarithmic probability
        # axis would delete every token the decoder never emitted -- a measured
        # outcome, and most of them.
        counts = [
            np.asarray(table[count_key], dtype=np.float64)[plotted] for table in tables
        ]
        y_values = [np.log10(1.0 + values) for values in counts]
        use_log_y = False
    else:
        y_values = raw
        use_log_y = _common_probability_axis(y_values, y_scale)

    columns = 3
    rows = int(np.ceil(len(chosen) / columns))
    figure = _new_figure(width=4.8 * columns, height=4.1 * rows)
    panels = figure.subplots(rows, columns, squeeze=False, sharey=True)
    norm = LogNorm(vmin=float(corpus[coloured].min()), vmax=float(corpus[coloured].max()))
    marks = None

    for panel_index, temperature in enumerate(chosen):
        axes = panels[panel_index // columns][panel_index % columns]
        marks = axes.scatter(
            norms[panel_index][coloured],
            y_values[panel_index][coloured],
            c=corpus[coloured],
            s=9,
            alpha=0.45,
            cmap="viridis",
            norm=norm,
            edgecolors="none",
            rasterized=True,
        )
        axes.set_xscale("log")
        axes.set_xlim(*panel_limits[panel_index])
        _configure_gradient_panel_x_axis(axes, *panel_limits[panel_index])
        if count_key is not None:
            # A hairline above zero, so the never-emitted population reads as a
            # population rather than as the axis frame.
            axes.axhline(0.0, color="#bbbbbb", linewidth=0.7, linestyle="-", zorder=0)
        elif use_log_y:
            axes.set_yscale("log")
        axes.grid(True, which="both", alpha=0.20)

        low, high = panel_limits[panel_index]
        axes.set_title(f"T = {temperature:g}\nG in [{low:.3g}, {high:.3g}]", fontsize=9.5)
        axes.text(
            0.03,
            0.97,
            panel_statistic(tables[panel_index], y_values[panel_index]),
            transform=axes.transAxes,
            fontsize=7,
            va="top",
            ha="left",
            bbox=_ANNOTATION_BOX,
        )

    if count_key is not None:
        # Ticks at meaningful counts, labelled as the fraction they are, exactly
        # as figure 10 labels its own count axis.
        ladder = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
        largest = int(max(float(values.max()) for values in counts))
        selected = [value for value in ladder if value <= largest]
        if largest and largest not in selected:
            gap = np.log10(1.0 + largest) - np.log10(1.0 + selected[-1])
            if gap > 0.25:
                selected.append(largest)
        top = float(np.log10(1.0 + max(largest, 1)))
        for row_index in range(rows):
            for column in range(columns):
                axes = panels[row_index][column]
                axes.set_yticks(np.log10(1.0 + np.asarray(selected, dtype=np.float64)))
                axes.set_yticklabels(
                    ["0"] + [f"{value}/D" for value in selected[1:]], fontsize=7.5
                )
                axes.set_ylim(-0.045 * top, top * 1.06)

    for panel_index in range(len(chosen), rows * columns):
        panels[panel_index // columns][panel_index % columns].set_visible(False)

    # Row 2's two-line titles otherwise sit on row 1's tick labels.
    figure.subplots_adjust(hspace=0.34)
    figure.supxlabel("mean gradient norm  G(i, T)   [log scale]", fontsize=10)
    if count_key is not None:
        figure.supylabel(
            f"{y_label}   [axis: log10(1 + k), D = {int(tables[0][count_total_key]):,}]",
            fontsize=10,
        )
    else:
        figure.supylabel(
            f"{y_label}   [{'log' if use_log_y else 'linear'} scale]", fontsize=10
        )
    colourbar = figure.colorbar(marks, ax=panels, pad=0.02, fraction=0.03)
    colourbar.set_label("Whole-corpus empirical token frequency  p(i)", fontsize=9)

    full_note = note + (
        "   |   x limits are per-panel: horizontal position is NOT comparable "
        "between panels"
    )
    if dropped:
        full_note += f"   |   omitted, no corpus mass: {_format_count(dropped)}"
    figure.suptitle(title, fontsize=12)
    figure.text(0.5, 0.945, full_note, ha="center", fontsize=7.5, color="#444444")
    return save_figure(figure, directory, stem)


def plot_gradient_vs_nucleus_guess_bias(
    record: Any,
    directory: str | Path,
    *,
    panel_temperatures: Sequence[float] | None = None,
    y_scale: str = "auto",
) -> list[Path]:
    """Figure 15 -- gradient magnitude vs. the *realized* nucleus guess fraction.

    Figure 10's stochastic sibling. Figure 10 asks the question of a
    deterministic decoder: are tokens the model would *greedily* emit associated
    with different gradient magnitudes when they occur as targets? This asks it
    of the decoder actually used to sample, at the same temperatures, on the same
    initialization, over the same positions.

    The y quantity is the fraction of positions at which the single realized
    nucleus draw produced token ``i``. There is exactly one draw per position per
    temperature -- ``R`` remains 1 -- and those draws are the ones the experiment
    already made under the recorded ``top_p`` and sampling seed. Nothing is
    resampled to draw this figure.

    Unlike figure 10's ``q_i``, this y quantity **does** move with temperature:
    sampling is not argmax, so raising ``T`` genuinely changes which tokens are
    emitted. Both axes therefore vary across panels, which is the difference
    between the two figures rather than an inconsistency.
    """

    from llm_behavior_lab.analysis.gradients import nucleus_gradient_table

    chosen = _gradient_panel_temperatures(record, panel_temperatures)
    tables = [nucleus_gradient_table(record, value) for value in chosen]

    def statistic(table: dict[str, Any], values: np.ndarray) -> str:
        emitted = int((values > 0).sum())
        return (
            f"tokens ever emitted: {_format_count(emitted)}\n"
            f"D = {_format_count(table['nucleus_num_positions'])} draws"
        )

    return _plot_gradient_against(
        record,
        directory,
        tables=tables,
        chosen=chosen,
        y_key="nucleus_guess_fraction",
        count_key="nucleus_guess_count",
        count_total_key="nucleus_num_positions",
        y_label="realized nucleus guess fraction  q_nuc(i, T) = k/D",
        title=(
            "Gradient magnitude vs. realized nucleus guessing bias, "
            "across loss temperature"
        ),
        note=(
            "y is the REALIZED one-sample-per-position nucleus fraction at each T, "
            "not the expected nucleus distribution"
        ),
        stem="figure15_temperature_gradient_vs_nucleus_guess_bias",
        y_scale=y_scale,
        panel_statistic=statistic,
    )


def plot_gradient_vs_mean_probability(
    record: Any,
    directory: str | Path,
    *,
    panel_temperatures: Sequence[float] | None = None,
    y_scale: str = "auto",
) -> list[Path]:
    """Figure 16 -- gradient magnitude vs. mean predictive probability by token.

    The continuous end of the sequence. Figures 10 and 15 both put a *decision*
    on the y axis -- one deterministic, one sampled. This one removes the
    decision entirely and uses the predictive mass the decision would have been
    made from: ``pbar_i(T)``, the mean over positions of ``softmax(z/T)_i`` at
    fixed token identity, before top-p and before sampling.

    Reading the three together separates a gradient's relationship with what the
    model *emits* from its relationship with what the model *prefers*. A token
    can carry appreciable mean probability without ever winning a draw, and those
    tokens are exactly the ones the two decision figures cannot place.

    The statistic is the one figure 14 already persists, so this figure adds no
    measurement and no storage.
    """

    from llm_behavior_lab.analysis.gradients import mean_probability_gradient_table

    chosen = _gradient_panel_temperatures(record, panel_temperatures)
    tables = [mean_probability_gradient_table(record, value) for value in chosen]

    def statistic(table: dict[str, Any], values: np.ndarray) -> str:
        return (
            f"median pbar {float(np.median(values)):.3g}\n"
            f"max pbar {float(values.max()):.3g}"
        )

    return _plot_gradient_against(
        record,
        directory,
        tables=tables,
        chosen=chosen,
        y_key="mean_predictive_probability",
        count_key=None,
        count_total_key=None,
        y_label="mean predictive probability  pbar(i, T)",
        title=(
            "Gradient magnitude vs. mean predictive probability by token, "
            "across loss temperature"
        ),
        note=(
            "y is the mean softmax(z/T) at fixed token identity, before top-p and "
            "before sampling"
        ),
        stem="figure16_temperature_gradient_vs_mean_probability",
        y_scale=y_scale,
        panel_statistic=statistic,
    )


def _initial_scatter(axes: Any, x, y, *, size_by=None, colour=None, cmap="viridis"):
    """One class-wise scatter with support encoded by marker size."""

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    sizes = 6.0
    if size_by is not None:
        support = np.asarray(size_by, dtype=np.float64)[keep]
        sizes = 3.0 + 14.0 * np.log1p(support) / max(np.log1p(support).max(), 1e-12)
    marks = axes.scatter(
        x[keep], y[keep], s=sizes,
        c=None if colour is None else np.asarray(colour, dtype=np.float64)[keep],
        cmap=None if colour is None else cmap,
        color=None if colour is not None else "#1f77b4",
        alpha=0.45, edgecolors="none", rasterized=True,
    )
    return marks, int(keep.sum())


def _symlog_x(axes: Any, values) -> None:
    """Symmetric-log x axis sized to the data, so zero stays on the axis."""

    values = np.asarray(values, dtype=np.float64)
    finite = np.abs(values[np.isfinite(values) & (values != 0)])
    axes.set_xscale("symlog", linthresh=float(finite.min()) if finite.size else 1e-6)
    axes.axvline(0.0, color="#999999", linewidth=0.8, linestyle=":", zorder=0)


def plot_initial_performance_and_bias(record: Any, directory: str | Path) -> list[Path]:
    """Figure 17 -- initial guessing bias against initial correctness.

    The null is the **marginal-preserving independence** expectation, not
    ``1/K``. Keeping both observed marginals, ``P(greedy=i | y=i) = q_i``, so a
    class earns recall simply by being guessed often; ``DeltaR_i = R_i - q_i`` is
    what remains after that mechanical part is removed. Panel (d) is the bridge
    to the cross-entropy error: only ``b_soft`` enters the mean logit gradient,
    and its relation to the realized argmax bias ``b_hard`` is empirical.
    """

    from llm_behavior_lab.analysis.initial_gradients import (
        class_frequency_table,
        correctness_table,
    )

    freq = class_frequency_table(record, 1.0)
    corr = correctness_table(record, 1.0)
    n = freq["target_count"]
    represented = n > 0

    figure = _new_figure(width=11.0, height=8.6)
    panels = figure.subplots(2, 2)

    a = panels[0][0]
    _initial_scatter(a, freq["target_fraction"][represented], corr["recall"][represented],
                     size_by=n[represented])
    order = np.argsort(freq["target_fraction"][represented])
    a.plot(freq["target_fraction"][represented][order],
           freq["guess_fraction"][represented][order],
           color="#d62728", linewidth=0.9, linestyle="--",
           label="independence expectation  R = q(i)")
    a.set_xscale("log")
    a.set_xlabel("target frequency  f(i)   [log]")
    a.set_ylabel("recall  R(i)")
    a.set_title("(a) recall vs target frequency", fontsize=10)
    a.legend(loc="upper left", fontsize=7.5, frameon=True)

    b = panels[0][1]
    _initial_scatter(b, freq["hard_bias"][represented], corr["delta_recall"][represented],
                     size_by=n[represented])
    _symlog_x(b, freq["hard_bias"][represented])
    b.axhline(0.0, color="#999999", linewidth=0.8, linestyle=":", zorder=0)
    b.set_xlabel("hard bias  b_hard(i) = q(i) - f(i)   [symlog]")
    b.set_ylabel("DeltaR(i) = R(i) - q(i)")
    b.set_title("(b) recognition beyond the independence null", fontsize=10)

    c = panels[1][0]
    guessed = freq["guess_count"] > 0
    _initial_scatter(c, freq["hard_bias"][guessed], corr["delta_precision"][guessed],
                     size_by=freq["guess_count"][guessed])
    _symlog_x(c, freq["hard_bias"][guessed])
    c.axhline(0.0, color="#999999", linewidth=0.8, linestyle=":", zorder=0)
    c.set_xlabel("hard bias  b_hard(i)   [symlog]")
    c.set_ylabel("DeltaP(i) = P(i) - f(i)")
    c.set_title("(c) precision beyond the independence null", fontsize=10)

    d = panels[1][1]
    _initial_scatter(d, freq["hard_bias"], freq["soft_bias"], size_by=n)
    limit = float(np.nanmax(np.abs(np.concatenate([freq["hard_bias"], freq["soft_bias"]]))))
    d.plot([-limit, limit], [-limit, limit], color="#d62728", linewidth=0.9,
           linestyle="--", label="b_soft = b_hard")
    _symlog_x(d, freq["hard_bias"])
    d.set_yscale("symlog", linthresh=1e-6)
    d.axhline(0.0, color="#999999", linewidth=0.8, linestyle=":", zorder=0)
    d.set_xlabel("hard bias  b_hard(i)   [symlog]")
    d.set_ylabel("soft bias  b_soft(i) = pbar(i) - f(i)   [symlog]")
    d.set_title("(d) realized argmax bias vs probability-mass bias", fontsize=10)
    d.legend(loc="upper left", fontsize=7.5, frameon=True)

    for axes in (a, b, c, d):
        axes.grid(True, which="both", alpha=0.20)
        axes.tick_params(labelsize=8)

    figure.suptitle(
        "Initial guessing bias and initial correctness  (alpha = 1, T = 1, "
        f"initialization {freq['initialization_index']})", fontsize=12)
    figure.text(0.5, 0.945,
                f"marker size ~ support n(i)   |   {corr['num_represented']:,} classes "
                f"represented, {corr['num_guessed']:,} guessed, "
                f"{corr['num_true_positive_classes']} with TP > 0",
                ha="center", fontsize=7.5, color="#444444")
    figure.subplots_adjust(hspace=0.30, wspace=0.24)
    return save_figure(figure, directory, "figure17_initial_performance_and_bias")


def plot_initial_gradient_split(record: Any, directory: str | Path) -> list[Path]:
    """Figure 18 -- per-example gradient intensity for failed assignments.

    At initialization almost every position is a failed assignment, so the
    TP side of the split is reported as a count rather than drawn as a
    distribution: ``G^TP`` exists for a single class and cannot support a
    comparison. What the data do support is how the *failed* intensity
    ``G_i^FN`` varies with support, bias and target confidence.

    Intensity is a mean per-example norm; the norm mass ``M`` printed in the
    annotation is a sum over the group and is **not** an SGD update magnitude.
    """

    from llm_behavior_lab.analysis.initial_gradients import (
        class_frequency_table,
        confidence_split,
        gradient_correctness_split,
    )

    freq = class_frequency_table(record, 1.0)
    grad = gradient_correctness_split(record, 1.0)
    conf = confidence_split(record, 1.0)
    n = freq["target_count"]
    defined = np.isfinite(grad["intensity_false_negative"])

    figure = _new_figure(width=13.5, height=4.4)
    panels = figure.subplots(1, 3)

    a = panels[0]
    _initial_scatter(a, n[defined], grad["intensity_false_negative"][defined],
                     size_by=n[defined])
    tp = np.isfinite(grad["intensity_true_positive"])
    if tp.any():
        a.scatter(n[tp], grad["intensity_true_positive"][tp], s=40, marker="*",
                  color="#d62728", zorder=5, label="the single TP class")
        a.legend(loc="upper right", fontsize=7.5, frameon=True)
    a.set_xscale("log")
    a.set_xlabel("target support  n(i)   [log]")
    a.set_ylabel("G_FN(i)  mean gradient norm, failed targets")
    a.set_title("(a) failed-target intensity vs support", fontsize=10)

    b = panels[1]
    _initial_scatter(b, freq["hard_bias"][defined],
                     grad["intensity_false_negative"][defined], size_by=n[defined])
    _symlog_x(b, freq["hard_bias"][defined])
    b.set_xlabel("hard bias  b_hard(i)   [symlog]")
    b.set_ylabel("G_FN(i)")
    b.set_title("(b) failed-target intensity vs hard bias", fontsize=10)

    c = panels[2]
    both = defined & np.isfinite(conf["confidence_false_negative"])
    _initial_scatter(c, conf["confidence_false_negative"][both],
                     grad["intensity_false_negative"][both], size_by=n[both])
    c.set_xscale("log")
    c.set_xlabel("C_FN(i)  mean target confidence on failures   [log]")
    c.set_ylabel("G_FN(i)")
    c.set_title("(c) intensity vs target confidence", fontsize=10)

    for axes in panels:
        axes.grid(True, which="both", alpha=0.20)
        axes.tick_params(labelsize=8)

    total = grad["total_mass_true_positive"] + grad["total_mass_false_negative"]
    figure.suptitle(
        "Gradient intensity of failed assignments at initialization "
        "(alpha = 1, T = 1)", fontsize=12)
    figure.text(0.5, 0.905,
                f"G_TP defined for {grad['num_intensity_true_positive']} class, "
                f"G_FN for {grad['num_intensity_false_negative']:,}   |   "
                f"norm mass M_FN / (M_TP + M_FN) = "
                f"{grad['total_mass_false_negative']/total:.6f}   "
                "(norm mass, not an SGD update magnitude)",
                ha="center", fontsize=7.5, color="#444444")
    figure.subplots_adjust(wspace=0.28, top=0.80)
    return save_figure(figure, directory, "figure18_initial_gradient_split")


def plot_initial_logit_correction(record: Any, directory: str | Path) -> list[Path]:
    """Figure 19 -- where the initial cross-entropy correction comes from.

    ``A_i`` is the upward pull accumulated where ``i`` is the target and ``S_i``
    the downward pressure accumulated where it is not. Their difference is the
    exact mean logit correction, so this figure asks the *provenance* question
    instead: which positions generate each force.

    The identity ``A_i - S_i = -b_soft(i)/T`` is not drawn -- it is algebra, and
    is enforced as a numerical invariant in the tests. Panel (c) instead relates
    the net correction to the **hard** bias, which is not an identity and shows
    how far the realized argmax bias predicts the actual corrective force.
    """

    from llm_behavior_lab.analysis.initial_gradients import (
        class_frequency_table,
        logit_correction,
    )

    freq = class_frequency_table(record, 1.0)
    logit = logit_correction(record, 1.0)
    n = freq["target_count"]

    figure = _new_figure(width=13.5, height=4.4)
    panels = figure.subplots(1, 3)

    a = panels[0]
    both = (logit["attraction"] > 0) & (logit["suppression"] > 0)
    _initial_scatter(a, logit["attraction"][both], logit["suppression"][both],
                     size_by=n[both])
    lo = float(min(logit["attraction"][both].min(), logit["suppression"][both].min()))
    hi = float(max(logit["attraction"][both].max(), logit["suppression"][both].max()))
    a.plot([lo, hi], [lo, hi], color="#d62728", linewidth=0.9, linestyle="--",
           label="A = S  (no net force)")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("target attraction  A(i)   [log]")
    a.set_ylabel("non-target suppression  S(i)   [log]")
    a.set_title("(a) attraction against suppression", fontsize=10)
    a.legend(loc="upper left", fontsize=7.5, frameon=True)

    b = panels[1]
    fp = np.isfinite(logit["false_positive_suppression_fraction"]) & (freq["guess_count"] > 0)
    _initial_scatter(b, freq["hard_bias"][fp],
                     logit["false_positive_suppression_fraction"][fp],
                     size_by=freq["guess_count"][fp])
    _symlog_x(b, freq["hard_bias"][fp])
    b.set_xlabel("hard bias  b_hard(i)   [symlog]")
    b.set_ylabel("S_FP(i) / S(i)")
    b.set_title("(b) suppression generated by false-positive wins", fontsize=10)

    c = panels[2]
    _initial_scatter(c, freq["hard_bias"], logit["net_correction"], size_by=n)
    _symlog_x(c, freq["hard_bias"])
    c.set_yscale("symlog", linthresh=1e-9)
    c.axhline(0.0, color="#999999", linewidth=0.8, linestyle=":", zorder=0)
    c.set_xlabel("hard bias  b_hard(i)   [symlog]")
    c.set_ylabel("net correction  A(i) - S(i)   [symlog]")
    c.set_title("(c) net descent correction vs hard bias", fontsize=10)

    for axes in panels:
        axes.grid(True, which="both", alpha=0.20)
        axes.tick_params(labelsize=8)

    figure.suptitle(
        "Class-wise cross-entropy correction at initialization (alpha = 1, T = 1)",
        fontsize=12)
    figure.text(0.5, 0.905,
                "A - S is the gradient-DESCENT correction; S - A is the mean logit "
                "gradient component.   TP/FN partitions positions by target; "
                "FP/other is a non-target view.",
                ha="center", fontsize=7.5, color="#444444")
    figure.subplots_adjust(wspace=0.28, top=0.80)
    return save_figure(figure, directory, "figure19_initial_logit_correction")


def plot_gradient_directional_clustering(
    record: Any,
    directory: str | Path,
    *,
    display_classes: int = 40,
    min_support: int = 2,
) -> list[Path]:
    """Figure 20 -- do gradients cluster by token subgroup?

    The heatmap is the figure. Cell ``(i, j)`` is the mean cosine between the
    gradients of class ``i`` and class ``j``, and the **diagonal is a
    measurement**: the mean cosine over distinct pairs *within* a class, not the
    trivial 1 a self-similarity convention would put there. Clustering, if it
    exists, is a visibly warmer diagonal against a near-zero field.

    The colormap diverges about zero because the sign is the finding. Cosines
    near zero mean gradients are near-orthogonal -- the default expectation in
    high dimension -- while systematic negative values would mean subgroups
    actively oppose one another, which is a different claim from "no structure"
    and must not be allowed to look like it.

    Only the most frequent classes are drawn, deterministically and with the
    count stated. That is a readability limit on the *display*: the analysis
    itself runs over every class meeting ``min_support``, and the summary panel
    reports those pooled numbers rather than the drawn subset's.

    The design is deliberately not initialization-specific. At initialization
    almost every position is a failure and the greedy grouping may be nearly
    degenerate; the same three panels become more informative as training
    separates the subgroups, which is what this diagnostic is for.
    """

    from llm_behavior_lab.analysis.gradient_clustering import gradient_clustering

    target = gradient_clustering(
        record, grouping="target", min_support=min_support, max_classes=display_classes
    )
    greedy = gradient_clustering(
        record, grouping="greedy", min_support=min_support, max_classes=display_classes
    )

    figure = _new_figure(width=14.0, height=5.2)
    # Generous horizontal spacing: the heatmap's colourbar label and the next
    # panel's y label are both long and otherwise collide.
    grid = figure.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.0], wspace=0.60)
    heat, bars, coherence = (figure.add_subplot(grid[0, index]) for index in range(3))

    matrix = target["matrix"]
    finite = matrix[np.isfinite(matrix)]
    extent = float(np.abs(finite).max()) if finite.size else 1.0
    image = heat.imshow(
        matrix, cmap="RdBu_r", vmin=-extent, vmax=extent, interpolation="nearest"
    )
    heat.set_title(
        f"(a) mean cosine, target grouping\n{target['classes'].size} most frequent classes",
        fontsize=10,
    )
    heat.set_xlabel("token class")
    heat.set_ylabel("token class")
    heat.tick_params(labelsize=6)
    # Token strings would be unreadable at this count; ranks are honest labels.
    step = max(1, target["classes"].size // 8)
    ticks = np.arange(0, target["classes"].size, step)
    heat.set_xticks(ticks)
    heat.set_yticks(ticks)
    heat.set_xticklabels([str(int(target["classes"][i])) for i in ticks], rotation=90)
    heat.set_yticklabels([str(int(target["classes"][i])) for i in ticks])
    colourbar = figure.colorbar(image, ax=heat, fraction=0.046, pad=0.03)
    colourbar.set_label("mean cosine", fontsize=8)
    colourbar.ax.tick_params(labelsize=7)

    # The observed delta of each grouping against its own permutation null, drawn
    # as an interval rather than a single shuffled value: one permutation says
    # nothing about how much a delta of this size varies by chance.
    entries = [("target", target, "#1f77b4"), ("greedy", greedy, "#2ca02c")]
    for index, (name, result, colour) in enumerate(entries):
        null = result["null"]
        bars.bar(index, result["observed"]["delta"], width=0.55, color=colour,
                 label="observed delta" if index == 0 else None)
        bars.errorbar(
            index, null["delta_mean"],
            yerr=[[null["delta_mean"] - null["delta_low"]],
                  [null["delta_high"] - null["delta_mean"]]],
            fmt="o", color="#333333", markersize=4, capsize=5, linewidth=1.2,
            label="permutation null, 2.5-97.5%" if index == 0 else None,
        )
    bars.axhline(0.0, color="#333333", linewidth=0.8)
    bars.set_xticks(range(len(entries)))
    bars.set_xticklabels([name for name, _, _ in entries], fontsize=9)
    bars.set_ylabel("delta = within - between")
    bars.set_title("(b) clustering effect against its permutation null", fontsize=10)
    bars.grid(True, axis="y", alpha=0.20)
    bars.legend(loc="upper right", fontsize=7, frameon=True)
    lines = []
    for name, result, _ in entries:
        null = result["null"]
        lines.append(
            f"{name:<7}obs {result['observed']['delta']:+.5f}  "
            f"null {null['delta_mean']:+.5f} "
            f"[{null['delta_low']:+.5f}, {null['delta_high']:+.5f}]"
        )
    bars.text(
        0.02, 0.02, "\n".join(lines) + f"\nM = {target['null']['permutations']}",
        transform=bars.transAxes, fontsize=6.5, va="bottom", ha="left",
        bbox=_ANNOTATION_BOX, family="monospace",
    )

    within = target["within_by_class"]
    measurable = np.isfinite(within)
    coherence.scatter(
        target["counts"][measurable], within[measurable],
        s=8, alpha=0.5, color="#1f77b4", edgecolors="none", rasterized=True,
    )
    coherence.axhline(
        target["observed"]["between"], color="#d62728", linewidth=1.0, linestyle="--",
        label="pooled between-class",
    )
    coherence.axhline(0.0, color="#999999", linewidth=0.8, linestyle=":")
    coherence.set_xscale("log")
    coherence.set_xlabel("class support  n(i)   [log]")
    coherence.set_ylabel("within-class mean cosine")
    coherence.set_title("(c) per-token coherence vs support", fontsize=10)
    coherence.legend(loc="upper right", fontsize=7.5, frameon=True)
    coherence.grid(True, which="both", alpha=0.20)

    figure.suptitle(
        "Directional clustering of per-position gradients (T = 1, sketch "
        f"K = {target['sketch_dimension']})",
        fontsize=12,
    )
    figure.text(
        0.5, 0.925,
        f"analysis over all classes with n >= {min_support}: "
        f"{target['observed']['num_classes']:,} target, "
        f"{greedy['observed']['num_classes']:,} greedy   |   "
        f"diagonal is within-class over DISTINCT pairs, not self-similarity   |   "
        "cosines are sketch estimates: reliable pooled, noisy per pair",
        ha="center", fontsize=7.5, color="#444444",
    )
    figure.subplots_adjust(top=0.82)
    return save_figure(figure, directory, "figure20_gradient_directional_clustering")


def plot_correction_provenance(record: Any, directory: str | Path) -> list[Path]:
    """Figure 21 -- where does the corrective signal come from?

    Figure 19 already relates ``A`` to ``S``, shows the suppression side's
    provenance through ``S_FP / S``, and plots the net correction. What it never
    shows is the **target side**: ``A = A_TP + A_FN`` is computed and then only
    reported numerically. That is the gap this fills, and it is the reason this
    is one figure rather than a dashboard.

    Panel (a) is the provenance itself. Panel (b) puts the two sides beside each
    other, so the question "is the correction driven by missed targets or by
    false-positive wins" is answered on one axis pair.

    At initialization ``A_FN / A`` is close to 1 almost everywhere, because
    almost nothing is correct. That is not a defect of the figure: it is the
    measurement, and the same panels become informative as ``TP`` grows during
    training.
    """

    from llm_behavior_lab.analysis.initial_gradients import (
        class_frequency_table,
        logit_correction,
    )

    freq = class_frequency_table(record, 1.0)
    logit = logit_correction(record, 1.0)
    support = freq["target_count"]

    figure = _new_figure(width=11.0, height=4.6)
    panels = figure.subplots(1, 2)

    a = panels[0]
    represented = support > 0
    _initial_scatter(
        a, freq["soft_bias"][represented],
        logit["failed_attraction_fraction"][represented], size_by=support[represented],
    )
    _symlog_x(a, freq["soft_bias"][represented])
    a.axhline(1.0, color="#d62728", linewidth=0.9, linestyle="--",
              label="all attraction from missed targets")
    a.set_ylim(-0.02, 1.05)
    a.set_xlabel("soft bias  b_soft(i) = pbar(i) - f(i)   [symlog]")
    a.set_ylabel("A_FN(i) / A(i)")
    a.set_title("(a) target-side provenance: attraction from failures", fontsize=10)
    a.legend(loc="lower left", fontsize=7.5, frameon=True)

    b = panels[1]
    both = represented & (freq["guess_count"] > 0)
    _initial_scatter(
        b, logit["failed_attraction_fraction"][both],
        logit["false_positive_suppression_fraction"][both], size_by=support[both],
    )
    b.set_xlabel("A_FN(i) / A(i)   target side")
    b.set_ylabel("S_FP(i) / S(i)   suppression side")
    b.set_title("(b) the two provenances against each other", fontsize=10)

    for axes in panels:
        axes.grid(True, which="both", alpha=0.20)
        axes.tick_params(labelsize=8)

    total_attraction = float(logit["attraction"].sum())
    figure.suptitle(
        "Provenance of the initial cross-entropy correction (T = 1)", fontsize=12
    )
    figure.text(
        0.5, 0.90,
        f"A = A_TP + A_FN with A_TP / A = "
        f"{float(logit['attraction_true_positive'].sum()) / total_attraction:.6f}   |   "
        "TP/FN partitions positions by target; FP/other is a non-target view",
        ha="center", fontsize=7.5, color="#444444",
    )
    figure.subplots_adjust(top=0.80, wspace=0.26)
    return save_figure(figure, directory, "figure21_correction_provenance")
