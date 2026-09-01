"""Retaining per-class gradient-sketch factors, and answering only what they support.

``metrics_only`` keeps the finalized numbers and nothing else, which makes every
post-hoc question unanswerable. ``compact_factors`` is the middle option: it
retains ``T_{m,a}`` and ``Q_{m,a}`` for an **explicitly declared** set of
groupings, so similarities between those groups can be recomputed later without
the rows.

Three things this module is careful about, because each has a way of going quietly
wrong.

**These are gradient sketches.** Summed per class, but sketches all the same. The
field names say so -- ``factor_0_gradient_sketch_sum`` rather than anything more
abstract -- because someone deciding whether an artifact may be shared needs to
see what it holds without reading this docstring.

**float64, not float32.** ``T`` is accumulated in float64 and persisted in
float64. Narrowing it on the way out would be a second quantization, applied
after normalization, with no declared convention or tolerance behind it -- a
silent numerical change dressed as a storage saving.

**A retained factor answers exactly one question.** A selection is a
``(grouping, loss temperature)`` pair, and a query outside the declared set is
refused rather than approximated from a neighbour. Per-position relabelling is
refused outright: the rows are gone, and a class sum cannot be re-partitioned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

__all__ = [
    "FactorSelection",
    "UnsupportedFactorQuery",
    "estimate_factor_bytes",
    "factor_arrays",
    "parse_factor_selection",
    "supported_selections",
    "similarity_from_factors",
]

#: Groupings a selection may name. ``cross`` is excluded deliberately: the
#: cross-partition statistics need target factors, greedy factors, the sparse
#: contingency and a per-cell ``Q``, which is a different retention shape and is
#: declared as its own selection kind below.
GROUPINGS = ("target", "greedy", "nucleus", "cross")


class UnsupportedFactorQuery(RuntimeError):
    """A query the retained factors cannot answer.

    Raised rather than answered approximately. The whole point of declaring a
    selection before the run is that afterwards the honest answer to anything
    outside it is "that was not retained" -- not a number derived from a
    grouping or a gradient field that happens to be nearby.
    """


@dataclass(frozen=True)
class FactorSelection:
    """One declared retention: a grouping measured in one gradient field."""

    grouping: str
    loss_temperature: float
    #: Only for ``nucleus``: which sampling temperature labelled the positions.
    sampling_temperature: float | None = None

    def __post_init__(self) -> None:
        if self.grouping not in GROUPINGS:
            raise ValueError(
                f"Unknown grouping {self.grouping!r}; expected one of "
                f"{', '.join(GROUPINGS)}."
            )
        if self.grouping == "nucleus" and self.sampling_temperature is None:
            raise ValueError(
                "A nucleus selection must name its sampling temperature: the "
                "labels are what make it a grouping, and T_s is what makes them."
            )
        if self.loss_temperature <= 0:
            raise ValueError(
                f"Loss temperature must be positive; got {self.loss_temperature}."
            )

    @property
    def label(self) -> str:
        if self.grouping == "nucleus":
            return f"nucleus:{self.sampling_temperature:g}@{self.loss_temperature:g}"
        return f"{self.grouping}@{self.loss_temperature:g}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "grouping": self.grouping,
            "loss_temperature": float(self.loss_temperature),
            "sampling_temperature": (
                None if self.sampling_temperature is None
                else float(self.sampling_temperature)
            ),
            "label": self.label,
        }


def parse_factor_selection(spec: str) -> list[FactorSelection]:
    """Parse ``target@1.0,greedy@1.0,nucleus:0.6@0.6,cross@1.0``.

    Explicit by construction: there is no wildcard and no "all". Retaining
    factors is a storage decision with a scientific consequence -- it decides
    which questions stay askable -- so it is spelled out rather than defaulted.
    """

    selections: list[FactorSelection] = []
    for raw in str(spec).split(","):
        entry = raw.strip()
        if not entry:
            continue
        if "@" not in entry:
            raise ValueError(
                f"Malformed factor selection {entry!r}: expected "
                "'<grouping>@<T_g>', e.g. 'target@1.0'."
            )
        head, _, loss = entry.partition("@")
        sampling = None
        if ":" in head:
            head, _, sampling_text = head.partition(":")
            try:
                sampling = float(sampling_text)
            except ValueError:
                raise ValueError(
                    f"Malformed sampling temperature in {entry!r}."
                ) from None
        try:
            loss_value = float(loss)
        except ValueError:
            raise ValueError(
                f"Malformed loss temperature in {entry!r}."
            ) from None
        selections.append(
            FactorSelection(head.strip(), loss_value, sampling)
        )
    if not selections:
        raise ValueError(
            "An empty factor selection retains nothing, which is what "
            "metrics_only already does. Name the groupings, or use that mode."
        )
    return selections


def estimate_factor_bytes(
    selections: Sequence[FactorSelection],
    *,
    num_classes: int,
    num_maps: int,
    num_buckets: int,
) -> int:
    """Bytes the declared selection will occupy.

    ``sum_i (C_i * M * K * 8 + C_i * M * 8)`` -- float64 throughout. At
    ``C = 4000, M = 4, K = 1024`` that is about 131 MiB **per selection**, which
    is why the limit is checked before measurement rather than discovered after.
    """

    per_selection = num_classes * num_maps * num_buckets * 8 + num_classes * num_maps * 8
    return int(per_selection * len(selections))


def factor_arrays(
    selections: Sequence[FactorSelection],
    factors_by_selection: dict[str, list[Any]],
    *,
    cross_by_selection: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Lay the retained factors out as archive arrays.

    Names carry ``gradient_sketch`` on purpose. A ``cross`` selection is
    self-contained -- both groupings' sums, both class sets, the sparse
    contingency and the per-cell ``Q`` -- because contingency counts alone
    cannot reconstruct a cross-partition similarity.
    """

    out: dict[str, np.ndarray] = {}
    for index, selection in enumerate(selections):
        if selection.grouping == "cross":
            # Cross retention is a different shape and lives in its own map, so
            # it is looked up there rather than in the per-grouping factors --
            # keying it off the latter silently dropped every cross selection.
            cross = (cross_by_selection or {}).get(selection.label)
            if cross is None:
                continue
            out[f"factor_{index}_cross_target_classes"] = cross["target_classes"]
            out[f"factor_{index}_cross_target_counts"] = cross["target_counts"]
            out[f"factor_{index}_cross_greedy_classes"] = cross["greedy_classes"]
            out[f"factor_{index}_cross_greedy_counts"] = cross["greedy_counts"]
            out[f"factor_{index}_cross_target_gradient_sketch_sum"] = np.stack(
                cross["target_sums"], axis=1
            ).astype(np.float64)
            out[f"factor_{index}_cross_greedy_gradient_sketch_sum"] = np.stack(
                cross["greedy_sums"], axis=1
            ).astype(np.float64)
            out[f"factor_{index}_cross_contingency_targets"] = cross["cell_targets"]
            out[f"factor_{index}_cross_contingency_greedy"] = cross["cell_greedy"]
            out[f"factor_{index}_cross_contingency_counts"] = cross["cell_counts"]
            out[f"factor_{index}_cross_cell_self_squared"] = np.stack(
                cross["cell_self_squared"], axis=1
            ).astype(np.float64)
            continue

        entries = factors_by_selection.get(selection.label)
        if entries is None:
            continue
        first = entries[0]
        out[f"factor_{index}_classes"] = np.asarray(first.classes, dtype=np.int64)
        out[f"factor_{index}_counts"] = np.asarray(first.counts, dtype=np.int64)
        # [C, M, K] and [C, M]: the map axis is explicit so a single-map
        # selection is not mistaken for a squeezed multi-map one.
        out[f"factor_{index}_gradient_sketch_sum"] = np.stack(
            [np.asarray(entry.sums, dtype=np.float64) for entry in entries], axis=1
        )
        out[f"factor_{index}_gradient_sketch_self_squared"] = np.stack(
            [np.asarray(entry.self_squared, dtype=np.float64) for entry in entries],
            axis=1,
        )
    return out


def supported_selections(metrics: Any) -> list[FactorSelection]:
    """What this artifact actually retained, read from its own provenance."""

    declared = metrics.provenance.get("factor_selection") or []
    return [
        FactorSelection(
            entry["grouping"],
            entry["loss_temperature"],
            entry.get("sampling_temperature"),
        )
        for entry in declared
    ]


def _resolve(metrics: Any, query: FactorSelection) -> int:
    """Index of the retained selection answering ``query``, or refuse."""

    available = supported_selections(metrics)
    if not available:
        raise UnsupportedFactorQuery(
            "This record retained no gradient-sketch factors. Only the "
            "finalized metrics are available, and they answer the analyses "
            "declared before the run -- not new ones."
        )
    for index, selection in enumerate(available):
        if (
            selection.grouping == query.grouping
            and abs(selection.loss_temperature - query.loss_temperature) <= 1e-6
            and (
                selection.sampling_temperature is None
                or query.sampling_temperature is None
                or abs(
                    selection.sampling_temperature - query.sampling_temperature
                ) <= 1e-6
            )
        ):
            return index
    listed = ", ".join(selection.label for selection in available)
    raise UnsupportedFactorQuery(
        f"No retained factors answer {query.label}. This record retained "
        f"{listed}. A factor set from a different grouping or a different "
        "gradient field describes a different question, so it is not "
        "substituted."
    )


def similarity_from_factors(
    metrics: Any,
    grouping: str,
    class_a: int,
    class_b: int,
    *,
    loss_temperature: float = 1.0,
    sampling_temperature: float | None = None,
) -> float:
    """Estimated mean similarity between two classes, from retained factors.

    Reconstructs exactly what the finalized statistics were built from::

        A_ab = (1/M) sum_m  T_{m,a} . T_{m,b} / (n_a n_b)          a != b
        A_aa = (1/M) sum_m (||T_{m,a}||^2 - Q_{m,a}) / (n_a(n_a-1)) a == b

    ``nan`` for a within-class query on a singleton: one member has no distinct
    pair, which is an unavailable statistic rather than a zero.
    """

    query = FactorSelection(grouping, loss_temperature, sampling_temperature)
    index = _resolve(metrics, query)
    classes = np.asarray(metrics[f"factor_{index}_classes"])
    counts = np.asarray(metrics[f"factor_{index}_counts"])
    sums = np.asarray(metrics[f"factor_{index}_gradient_sketch_sum"])
    self_squared = np.asarray(
        metrics[f"factor_{index}_gradient_sketch_self_squared"]
    )

    slots = []
    for token in (class_a, class_b):
        found = np.flatnonzero(classes == int(token))
        if found.size != 1:
            raise UnsupportedFactorQuery(
                f"Class {token} is not among the {classes.size} retained for "
                f"{query.label}. Only classes that were represented when the "
                "factors were formed can be compared."
            )
        slots.append(int(found[0]))

    maps = sums.shape[1]
    if slots[0] == slots[1]:
        size = int(counts[slots[0]])
        if size < 2:
            return float("nan")
        per_map = [
            (float(sums[slots[0], m] @ sums[slots[0], m])
             - float(self_squared[slots[0], m]))
            / (size * (size - 1))
            for m in range(maps)
        ]
        return float(np.mean(per_map))

    pairs = int(counts[slots[0]]) * int(counts[slots[1]])
    if pairs <= 0:
        return float("nan")
    per_map = [
        float(sums[slots[0], m] @ sums[slots[1], m]) / pairs for m in range(maps)
    ]
    return float(np.mean(per_map))


def refuse_relabelling(metrics: Any) -> None:
    """Per-position relabelling is not answerable from factors, ever.

    A class sum is the sum over a fixed membership. Moving one position between
    classes changes both sums, and the information needed to do that -- which
    row belonged where -- is exactly what was discarded.
    """

    raise UnsupportedFactorQuery(
        "Per-position relabelling needs the rows, which compact_factors does "
        "not retain. A class sum cannot be re-partitioned: the membership it "
        "was formed over is gone. Re-run with --sketch-storage per_position if "
        "arbitrary regrouping has to stay possible."
    )
