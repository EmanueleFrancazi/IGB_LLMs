"""Figures for the initialization-distribution experiment.

Plotting is kept separate from measurement and from aggregation: everything here
consumes an already-persisted record and writes files. Nothing computes a
statistic that :mod:`llm_behavior_lab.analysis.aggregation` does not already
define, so a figure can never disagree with the numbers reported beside it.

Figures are built through the object-oriented ``Figure`` API with an explicit
Agg canvas rather than ``pyplot``. There is no global figure state, no window is
ever opened, and the functions behave identically in a notebook, in a test, and
on a headless machine.

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
    mean_with_sem,
    persistent_absolute_gaps,
    ranked_profile,
    ranked_profiles,
    sampling_adequacy,
    token_wise_absolute_gaps,
    top_two_gap,
    zero_guess_counts,
)

__all__ = [
    "FIGURE_FORMATS",
    "POLICY_STYLES",
    "generate_all_figures",
    "plot_ranked_frequency_profiles",
    "plot_sampling_adequacy",
    "plot_token_identity_scatter",
    "plot_token_wise_mismatch",
    "save_figure",
]

#: Raster for quick viewing, vector for publication. Two formats, no more.
FIGURE_FORMATS = ("png", "svg")

#: One colour per policy, reused across every figure so curves stay comparable.
POLICY_STYLES = {
    "corpus": {"color": "#222222", "label": "corpus empirical"},
    "greedy": {"color": "#1f77b4", "label": "greedy guesses"},
    "nucleus": {"color": "#d62728", "label": "nucleus guesses"},
    "selected": {"color": "#2ca02c", "label": "selected-position targets"},
}

_ANNOTATION_BOX = {"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#999999"}


def _new_figure(width: float = 8.0, height: float = 5.0) -> Figure:
    """Create a canvas-backed figure that never needs a display."""

    figure = Figure(figsize=(width, height), dpi=150)
    FigureCanvasAgg(figure)
    return figure


def _positive(values: np.ndarray) -> np.ndarray:
    """Mask non-positive entries so they vanish on a logarithmic axis."""

    array = np.asarray(values, dtype=np.float64)
    return np.where(array > 0.0, array, np.nan)


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
    """

    adequacy = sampling_adequacy(record)
    corpus = ranked_profile(record.corpus_fractions)
    selected = ranked_profile(record.selected_target_fractions)
    ranks = np.arange(1, record.vocab_size + 1)

    figure = _new_figure()
    axes = figure.subplots()
    axes.step(
        ranks,
        _positive(corpus),
        where="mid",
        color=POLICY_STYLES["corpus"]["color"],
        linewidth=2.0,
        label=f"whole split ({adequacy['corpus_tokens']} tokens)",
    )
    axes.step(
        ranks,
        _positive(selected),
        where="mid",
        color=POLICY_STYLES["selected"]["color"],
        linewidth=1.6,
        linestyle="--",
        label=f"selected positions ({adequacy['num_positions']} targets)",
    )
    axes.set_yscale("log")
    axes.set_xlabel("token frequency rank")
    axes.set_ylabel("token fraction")
    axes.set_title("Empirical sampling adequacy of the evaluation positions")
    axes.legend(loc="upper right", frameon=True)
    axes.grid(True, which="both", alpha=0.25)

    annotation = "\n".join(
        [
            f"TV(split, selected) = {adequacy['total_variation_distance']:.4f}",
            f"JS(split, selected) = {adequacy['js_divergence']:.4f}",
            f"evaluated positions = {adequacy['num_positions']}",
            f"vocabulary size = {adequacy['vocab_size']}",
            f"vocabulary seen in targets = {adequacy['tokens_represented_in_selection']}"
            f" ({adequacy['fraction_of_vocabulary_represented']:.1%})",
            f"split mass covered = {adequacy['corpus_mass_covered_by_selection']:.4f}",
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
    """

    corpus_fractions = record.corpus_fractions
    ranks = np.arange(1, record.vocab_size + 1)

    figure = _new_figure(width=8.5, height=5.5)
    axes = figure.subplots()
    axes.step(
        ranks,
        _positive(ranked_profile(corpus_fractions)),
        where="mid",
        color=POLICY_STYLES["corpus"]["color"],
        linewidth=2.0,
        label="corpus empirical (fixed)",
    )

    annotation_lines = [f"p(1) - p(2) = {top_two_gap(corpus_fractions):.4f}"]
    for policy in ("greedy", "nucleus"):
        guesses = record.policy_fractions(policy)
        profile = mean_with_sem(ranked_profiles(guesses))
        style = POLICY_STYLES[policy]
        axes.step(
            ranks,
            _positive(profile.mean),
            where="mid",
            color=style["color"],
            linewidth=1.6,
            label=f"{style['label']} (mean of {profile.num_samples} inits)",
        )
        axes.fill_between(
            ranks,
            _positive(profile.mean - profile.sem),
            _positive(profile.mean + profile.sem),
            step="mid",
            color=style["color"],
            alpha=0.25,
            linewidth=0.0,
            label=f"{style['label']} ± SEM",
        )
        gaps = np.array([top_two_gap(row) for row in guesses])
        zeros = zero_guess_counts(guesses).astype(np.float64)
        annotation_lines.append(
            f"{policy}: q(1)-q(2) = {gaps.mean():.4f} ± {_sem(gaps):.4f}"
        )
        annotation_lines.append(
            f"{policy}: zero-guess tokens = {zeros.mean():.1f} ± {_sem(zeros):.1f}"
            f" of {record.vocab_size}"
        )

    annotation_lines.append(f"evaluated positions = {record.metadata.get('num_positions', 'n/a')}")
    annotation_lines.append(f"initializations = {record.num_initializations}")

    axes.set_yscale("log")
    axes.set_xlabel("frequency rank")
    axes.set_ylabel("fraction (corpus tokens or selected guesses)")
    axes.set_title("Ranked concentration: corpus frequencies vs. guess frequencies")
    axes.legend(loc="upper right", fontsize=8, frameon=True)
    axes.grid(True, which="both", alpha=0.25)
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

    corpus_fractions = record.corpus_fractions
    ranks = np.arange(1, record.vocab_size + 1)

    figure = _new_figure(width=8.5, height=5.5)
    axes = figure.subplots()

    annotation_lines: list[str] = []
    for policy in ("greedy", "nucleus"):
        guesses = record.policy_fractions(policy)
        style = POLICY_STYLES[policy]

        typical = mean_with_sem(ranked_profiles(token_wise_absolute_gaps(guesses, corpus_fractions)))
        persistent = ranked_profile(persistent_absolute_gaps(guesses, corpus_fractions))

        axes.step(
            ranks,
            _positive(typical.mean),
            where="mid",
            color=style["color"],
            linewidth=1.8,
            label=f"{style['label']}: typical |q-p| (mean ± SEM)",
        )
        axes.fill_between(
            ranks,
            _positive(typical.mean - typical.sem),
            _positive(typical.mean + typical.sem),
            step="mid",
            color=style["color"],
            alpha=0.25,
            linewidth=0.0,
        )
        axes.step(
            ranks,
            _positive(persistent),
            where="mid",
            color=style["color"],
            linewidth=1.4,
            linestyle="--",
            label=f"{style['label']}: persistent |mean(q)-p|",
        )
        annotation_lines.append(
            f"{policy}: max typical = {typical.mean[0]:.4f}, max persistent = {persistent[0]:.4f}"
        )

    annotation_lines.append(f"initializations = {record.num_initializations}")
    annotation_lines.append(f"evaluated positions = {record.metadata.get('num_positions', 'n/a')}")

    axes.set_yscale("log")
    axes.set_xlabel("rank of the same-token absolute gap")
    axes.set_ylabel("|guess fraction - corpus fraction|")
    axes.set_title("Token-wise mismatch: typical per initialization vs. persistent")
    axes.legend(loc="upper right", fontsize=8, frameon=True)
    axes.grid(True, which="both", alpha=0.25)
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
    throughout: each marker is one token ID, positioned by its corpus fraction
    and its initialization-averaged guess fraction. Markers on the identity line
    are selected as often as they occur; markers above it are over-selected.
    Only the largest deviations are labelled, so the plot stays readable.
    """

    corpus = record.corpus_fractions
    tokens = record.tokens

    figure = _new_figure(width=7.5, height=6.5)
    axes = figure.subplots()

    mean_guesses = {
        policy: record.policy_fractions(policy).mean(axis=0) for policy in ("greedy", "nucleus")
    }
    positive_values = [corpus[corpus > 0]] + [
        values[values > 0] for values in mean_guesses.values() if np.any(values > 0)
    ]
    lower = min(float(values.min()) for values in positive_values) * 0.5
    upper = max(float(values.max()) for values in positive_values) * 2.0
    axes.plot([lower, upper], [lower, upper], color="#888888", linewidth=1.0, linestyle=":", label="y = x")

    for policy in ("greedy", "nucleus"):
        mean_guess = mean_guesses[policy]
        style = POLICY_STYLES[policy]
        axes.scatter(
            _positive(corpus),
            _positive(mean_guess),
            s=22,
            alpha=0.7,
            color=style["color"],
            edgecolors="none",
            label=style["label"],
        )
        if tokens and num_labels > 0:
            deviation = np.abs(mean_guess - corpus)
            for token_id in np.argsort(deviation)[::-1][:num_labels]:
                if corpus[token_id] <= 0 or mean_guess[token_id] <= 0:
                    continue
                axes.annotate(
                    repr(tokens[token_id]),
                    (corpus[token_id], mean_guess[token_id]),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=7,
                    color=style["color"],
                )

    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xlabel("corpus token fraction  p(i)")
    axes.set_ylabel("mean guess fraction  mean_s q(s, i)")
    axes.set_title("Token identity: corpus frequency vs. mean guess frequency")
    axes.legend(loc="lower right", fontsize=8, frameon=True)
    axes.grid(True, which="both", alpha=0.25)
    return save_figure(figure, directory, "figure3_token_identity_scatter")


def _sem(values: np.ndarray) -> float:
    """Standard error across initializations; zero when only one is present."""

    array = np.asarray(values, dtype=np.float64)
    if array.shape[0] < 2:
        return 0.0
    return float(array.std(ddof=1) / np.sqrt(array.shape[0]))


def generate_all_figures(record: Any, directory: str | Path) -> list[Path]:
    """Write every figure for one record and return the paths in figure order."""

    written: list[Path] = []
    written.extend(plot_sampling_adequacy(record, directory))
    written.extend(plot_ranked_frequency_profiles(record, directory))
    written.extend(plot_token_wise_mismatch(record, directory))
    written.extend(plot_token_identity_scatter(record, directory))
    return written
