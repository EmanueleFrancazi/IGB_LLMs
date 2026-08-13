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
    effective_support,
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


def _temperature_colours(count: int) -> list:
    """Perceptually ordered colours so the sweep reads as a progression.

    Sequential and colour-blind safe; deliberately not a rainbow, where hue
    ordering carries no perceptual ordering.
    """

    from matplotlib import cm

    return [cm.viridis(value) for value in np.linspace(0.05, 0.92, count)]


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
        zip(temperatures, _temperature_colours(len(temperatures)))
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
    return written
