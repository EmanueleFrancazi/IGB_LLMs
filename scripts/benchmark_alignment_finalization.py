#!/usr/bin/env python3
"""Cost of the exact label-permutation null at production scale.

The finalization design replaces a persisted ``[N_T, D, M, K]`` sketch tensor
with metrics computed while the rows are still available. Every other part of
that reduction is linear in ``D`` and cheap. One part is not: the label
permutation null draws 256 permutations of ``D`` rows per grouping per map, and
each draw touches the whole ``[D, K]`` block. Whether the lifecycle is viable at
all is therefore a question about *this loop*, and it has to be measured before
anything is integrated -- an unmeasured estimate is not evidence.

What is measured is the **exact** null, not an approximation of it. The loop here
calls the same :func:`_pooled_from_blocks` the production estimator uses and
consumes the same :func:`permutation_orders` draw sequence, and
:func:`check_numerical_equivalence` asserts bitwise equality against the
established :func:`_permutation_null` before any timing is reported. A benchmark
of a subtly different computation would answer the wrong question.

Vocabulary, fixed once so the extrapolation cannot double-count ``M``:

* a **job** is one ``(grouping, T_g)`` null -- 256 draws;
* a **map-job** is one ``(job, map)`` pair -- 256 draws over a single ``[D, K]``
  block;
* total map-jobs = jobs x M.

Timings are reported **per draw** and **per map-job**, and the total is
``jobs x M x seconds_per_map_job``.

Class-size profiles are generated here rather than read from a campaign record,
so this runs on a fresh checkout with no data. That matters for more than
convenience: block structure drives ``add.reduceat``, and a profile of a few
thousand uneven classes behaves differently from 32768 singletons. Four regimes
are covered -- the Zipf-like target grouping, the near-degenerate greedy grouping
seen at initialization, the fragmented nucleus grouping at high ``T_s``, and the
high-cardinality bound where every class is a singleton.

NumPy only. No model, no CUDA, no network.
"""

from __future__ import annotations

import argparse
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_behavior_lab.analysis import alignment_estimators as _estimators  # noqa: E402
from llm_behavior_lab.analysis.gradient_clustering import (  # noqa: E402
    _permutation_null,
    _pooled_from_blocks,
    permutation_orders,
)

#: Draws in the production null. Not a knob: it is what the established
#: `DEFAULT_PERMUTATIONS` uses, and changing it would change the null itself.
PRODUCTION_DRAWS = 256

#: Row-permutation null jobs the declared analysis contract requires, at the
#: production temperature grid. Two target/greedy at the canonical T_g, six
#: nucleus-control points at the canonical T_g, six matched points (one at each
#: non-canonical T_g), and target+greedy references at each of those six.
#: The cross-partition identity null is computed from group factors, not rows,
#: and is deliberately absent.
PRODUCTION_JOBS = 26

PROFILES = ("target", "greedy", "nucleus", "singleton")


def class_counts(profile: str, positions: int, *, seed: int = 0) -> np.ndarray:
    """Class sizes summing to ``positions``, for one grouping regime.

    Only the *sizes* matter here. The null permutes rows into blocks of the
    observed sizes, so the cost depends on how many blocks there are and how
    uneven they are, never on which token a block belongs to.
    """

    generator = np.random.default_rng(seed)

    if profile == "singleton":
        # The high-cardinality bound: C = D. No within-class pair exists, so the
        # statistic is unavailable -- but the reduction still runs, and this is
        # the worst case for both the class-sum allocation and `reduceat`.
        return np.ones(positions, dtype=np.int64)

    if profile == "greedy":
        # Near-degenerate, as the greedy grouping is at initialization: a handful
        # of tokens absorb most positions and a light tail carries the rest.
        heads = 5
        head_mass = int(round(0.80 * positions))
        head = np.full(heads, head_mass // heads, dtype=np.int64)
        head[0] += head_mass - int(head.sum())
        tail_total = positions - int(head.sum())
        tail = np.ones(max(tail_total // 4, 1), dtype=np.int64)
        labels = generator.integers(0, tail.size, size=tail_total)
        tail = np.bincount(labels, minlength=tail.size).astype(np.int64)
        counts = np.concatenate([head, tail[tail > 0]])
        return counts

    # `target` and `nucleus` are both Zipf-like over a subword vocabulary; the
    # nucleus grouping at a high sampling temperature is the flatter of the two,
    # so it fragments into more, smaller classes.
    exponent = 1.35 if profile == "target" else 1.08
    vocabulary = 32000
    weights = 1.0 / np.arange(1, vocabulary + 1, dtype=np.float64) ** exponent
    weights /= weights.sum()
    labels = generator.choice(vocabulary, size=positions, p=weights)
    counts = np.bincount(labels)
    return counts[counts > 0].astype(np.int64)


def normalized_rows(slab: np.ndarray, norms: np.ndarray) -> np.ndarray:
    """``[D, K]`` float64 rows, sketches divided by the exact gradient norms.

    A **division**, matching `gradient_clustering.unit_sketches`, never a
    multiplication by a precomputed reciprocal: the two do not round alike, and
    the established numbers come from the division.
    """

    unit = np.asarray(slab, dtype=np.float64)
    return unit / np.asarray(norms, dtype=np.float64)[:, None]


#: Reduction strategies for the per-draw class sums.
#:
#: ``reduceat`` is the established production reduction and is **bitwise** the
#: one :func:`_permutation_null` performs. ``blockwise`` sums each block with
#: ``ndarray.sum(axis=0)``, which NumPy evaluates pairwise rather than
#: sequentially: the statistic, the draws and the estimator are identical, but
#: the float64 accumulation order inside a class sum is not, so the last bits of
#: the result move. It is offered here to be *measured*, never adopted silently.
REDUCTIONS = ("reduceat", "blockwise")


def permutation_null_buffered(
    unit: np.ndarray,
    counts: np.ndarray,
    *,
    permutations: int,
    seed: int,
    draw_batch: int,
    gather: np.ndarray | None = None,
    reduction: str = "reduceat",
) -> dict[str, Any]:
    """The production null, with a reused gather buffer and batched draws.

    Two departures from :func:`_permutation_null` at the default ``reduction``,
    both chosen so the *values* cannot move:

    * the shuffled block is written into a caller-owned buffer through
      ``np.take(..., out=)`` instead of allocating ``unit[order]`` afresh each
      draw. A gather copies; it does not compute, so the buffer holds exactly
      the same bytes the temporary would have held.
    * the permutation draws are pulled from the generator in bounded batches
      rather than all at once. They come from the same generator in the same
      order, so the sequence is identical -- it is merely not all resident.

    :func:`check_numerical_equivalence` asserts both claims bitwise rather than
    trusting this docstring.

    ``reduction="blockwise"`` does not run a private copy of the production
    reduction -- it **calls** ``alignment_estimators.permutation_null``, so what
    this script reports is what production runs. A benchmark of a lookalike would
    go stale the first time the real one was tuned, and would do so silently.
    That arm deliberately breaks the bitwise claim above and exists so its cost
    and its effect can be reported.
    """

    if reduction == "blockwise":
        return _estimators.permutation_null(
            unit, counts, permutations=permutations, seed=seed,
            draw_batch=draw_batch,
        )

    edges = np.concatenate([[0], np.cumsum(counts)])
    starts = edges[:-1]
    row_squared = np.einsum("ij,ij->i", unit, unit)
    if gather is None:
        gather = np.empty_like(unit)

    deltas = np.empty(permutations, dtype=np.float64)
    withins = np.empty(permutations, dtype=np.float64)
    betweens = np.empty(permutations, dtype=np.float64)

    generator = np.random.default_rng(seed)
    total = int(np.sum(counts))
    drawn = 0
    while drawn < permutations:
        batch = min(draw_batch, permutations - drawn)
        orders = [generator.permutation(total) for _ in range(batch)]
        for order in orders:
            np.take(unit, order, axis=0, out=gather)
            sums = np.add.reduceat(gather, starts, axis=0)
            self_squared = np.add.reduceat(row_squared[order], starts)
            within, between = _pooled_from_blocks(sums, counts, self_squared)
            withins[drawn] = within
            betweens[drawn] = between
            deltas[drawn] = within - between
            drawn += 1
        del orders

    low, high = np.percentile(deltas, [2.5, 97.5])
    return {
        "permutations": int(permutations),
        "seed": int(seed),
        "within_mean": float(withins.mean()),
        "between_mean": float(betweens.mean()),
        "delta_mean": float(deltas.mean()),
        "delta_std": float(deltas.std(ddof=1)) if permutations > 1 else float("nan"),
        "delta_low": float(low),
        "delta_high": float(high),
    }


def check_numerical_equivalence(*, seed: int = 11) -> dict[str, Any]:
    """Bitwise equality against the established null, before any timing.

    Small enough to run in a fraction of a second and large enough to exercise
    uneven classes, singletons and a batch boundary that does not divide the
    draw count.
    """

    generator = np.random.default_rng(seed)
    counts = np.array([5, 3, 1, 4, 1, 2], dtype=np.int64)
    positions = int(counts.sum())
    buckets = 7
    slab = generator.standard_normal((positions, buckets)).astype(np.float32)
    norms = generator.uniform(0.5, 2.0, size=positions)
    unit = normalized_rows(slab, norms)

    reference = _permutation_null(unit, counts, permutations=32, seed=seed)
    buffered = permutation_null_buffered(
        unit, counts, permutations=32, seed=seed, draw_batch=5
    )

    mismatched = sorted(
        key for key in reference if reference[key] != buffered[key]
    )
    orders_match = all(
        np.array_equal(left, right)
        for left, right in zip(
            permutation_orders(counts, permutations=32, seed=seed),
            _batched_orders(positions, permutations=32, seed=seed, batch=5),
        )
    )
    return {
        "identical": not mismatched and orders_match,
        "mismatched_keys": mismatched,
        "draw_sequence_identical": orders_match,
        "delta_mean": reference["delta_mean"],
    }


def _batched_orders(
    positions: int, *, permutations: int, seed: int, batch: int
) -> list[np.ndarray]:
    """The draw sequence as the batched loop consumes it."""

    generator = np.random.default_rng(seed)
    orders: list[np.ndarray] = []
    while len(orders) < permutations:
        take = min(batch, permutations - len(orders))
        orders.extend(generator.permutation(positions) for _ in range(take))
    return orders


def _rss_kib() -> int:
    """Peak resident set size so far, in KiB (Linux ``ru_maxrss`` units)."""

    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _live_rss_kib() -> int:
    """Current resident set size in KiB, or 0 where ``/proc`` is unavailable."""

    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except OSError:
        pass
    return 0


def compare_reductions(
    *, positions: int, buckets: int, draws: int, seed: int, profile: str = "target"
) -> dict[str, Any]:
    """Effect of the ``blockwise`` reduction on the **reported** statistics.

    The intermediate class sums necessarily differ -- pairwise and sequential
    summation of the same float64 values are not the same computation. What
    matters for a review decision is not that intermediate but what the null
    actually publishes: ``delta_mean``, ``delta_std`` and the 2.5/97.5
    percentiles. Those are what this reports, as absolute and relative
    differences against the established reduction.
    """

    counts = class_counts(profile, positions, seed=seed)
    generator = np.random.default_rng(seed + 1)
    slab = generator.standard_normal((positions, buckets)).astype(np.float32)
    norms = generator.uniform(0.5, 2.0, size=positions)
    unit = normalized_rows(slab, norms)
    del slab
    gather = np.empty_like(unit)

    started = time.perf_counter()
    exact = permutation_null_buffered(
        unit, counts, permutations=draws, seed=seed, draw_batch=32,
        gather=gather, reduction="reduceat",
    )
    exact_seconds = time.perf_counter() - started

    started = time.perf_counter()
    fast = permutation_null_buffered(
        unit, counts, permutations=draws, seed=seed, draw_batch=32,
        gather=gather, reduction="blockwise",
    )
    fast_seconds = time.perf_counter() - started

    reported = ("within_mean", "between_mean", "delta_mean", "delta_std",
                "delta_low", "delta_high")
    differences = {
        key: {
            "reduceat": exact[key],
            "blockwise": fast[key],
            "absolute": abs(exact[key] - fast[key]),
            "relative": (
                abs(exact[key] - fast[key]) / abs(exact[key])
                if exact[key] else float("nan")
            ),
        }
        for key in reported
    }
    return {
        "profile": profile,
        "draws": draws,
        "reduceat_seconds": exact_seconds,
        "blockwise_seconds": fast_seconds,
        "speedup": exact_seconds / fast_seconds,
        "differences": differences,
        "max_relative": max(
            entry["relative"] for entry in differences.values()
            if np.isfinite(entry["relative"])
        ),
    }


def time_profile(
    profile: str,
    *,
    positions: int,
    buckets: int,
    timed_draws: int,
    draw_batch: int,
    seed: int,
    reduction: str = "reduceat",
) -> dict[str, Any]:
    """Time one map-job's inner loop and report its accounted buffers."""

    counts = class_counts(profile, positions, seed=seed)
    generator = np.random.default_rng(seed + 1)
    slab = generator.standard_normal((positions, buckets)).astype(np.float32)
    norms = generator.uniform(0.5, 2.0, size=positions)

    baseline_peak = _rss_kib()
    baseline_live = _live_rss_kib()

    started = time.perf_counter()
    unit = normalized_rows(slab, norms)
    normalize_seconds = time.perf_counter() - started
    # The float32 slab is not needed again once the rows are normalized. Dropping
    # it here is what the finalizer does, and leaving it alive would overstate the
    # accounted workspace by 128 MiB at production scale.
    del slab

    # Only the legacy arm needs a reordered copy of the rows. The production
    # reduction indexes them in place, which is precisely the buffer this
    # accounting must not keep claiming.
    gather = None if reduction == "blockwise" else np.empty_like(unit)

    started = time.perf_counter()
    permutation_null_buffered(
        unit, counts, permutations=timed_draws, seed=seed, draw_batch=draw_batch,
        gather=gather, reduction=reduction,
    )
    timed_seconds = time.perf_counter() - started

    peak_increment = _rss_kib() - baseline_peak
    live_increment = _live_rss_kib() - baseline_live

    per_draw = timed_seconds / timed_draws
    per_map_job = normalize_seconds + per_draw * PRODUCTION_DRAWS
    class_sum_bytes = int(counts.size) * buckets * 8

    accounted = {
        "normalized_rows_f64": positions * buckets * 8,
        "class_sums_f64": class_sum_bytes,
        "row_squared_f64": positions * 8,
        "order_batch_i64": draw_batch * positions * 8,
    }
    if gather is not None:
        accounted["gather_buffer_f64"] = positions * buckets * 8

    del unit, gather

    return {
        "profile": profile,
        "num_classes": int(counts.size),
        "largest_class": int(counts.max()),
        "num_singletons": int((counts == 1).sum()),
        "normalize_seconds": normalize_seconds,
        "timed_draws": timed_draws,
        "seconds_per_draw": per_draw,
        "seconds_per_map_job": per_map_job,
        "rss_peak_increment_bytes": peak_increment * 1024,
        "rss_live_increment_bytes": live_increment * 1024,
        "accounted_bytes": accounted,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the exact label-permutation null at production scale and "
            "extrapolate the finalization cost."
        )
    )
    parser.add_argument("--positions", type=int, default=32768, help="D")
    parser.add_argument("--buckets", type=int, default=1024, help="K")
    parser.add_argument("--maps", type=int, default=4, help="M")
    parser.add_argument(
        "--jobs", type=int, default=PRODUCTION_JOBS,
        help="Row-permutation null jobs the analysis contract requires.",
    )
    parser.add_argument(
        "--timed-draws", type=int, default=16,
        help=(
            "Draws actually executed per profile. The report extrapolates to "
            f"the production {PRODUCTION_DRAWS}; timing all of them for every "
            "profile would cost more than the measurement is worth."
        ),
    )
    parser.add_argument("--draw-batch", type=int, default=32)
    parser.add_argument(
        "--profiles", default=",".join(PROFILES),
        help="Comma-separated class-size regimes to measure.",
    )
    parser.add_argument("--seed", type=int, default=20240918)
    parser.add_argument(
        "--reduction", choices=REDUCTIONS, default="reduceat",
        help=(
            "Per-draw class-sum reduction. 'reduceat' is bitwise the production "
            "one. 'blockwise' is faster but changes the float64 accumulation "
            "order inside a class sum, so it must not be adopted without review."
        ),
    )
    parser.add_argument(
        "--compare-reductions", action="store_true",
        help=(
            "Also run both reductions over the full production draw count and "
            "report the effect on the reported null statistics."
        ),
    )
    parser.add_argument(
        "--collection-seconds", type=float, default=2861.92,
        help=(
            "Measured gradient-collection wall time to judge the budget "
            "against. The default is the value recorded in "
            "docs/experiment_logbook/supporting_material/metadata.json."
        ),
    )
    parser.add_argument(
        "--budget-fraction", type=float, default=0.10,
        help=(
            "Stretch target as a fraction of collection wall time. Reported per "
            "run; not a gate."
        ),
    )
    parser.add_argument(
        "--ceiling-seconds", type=float, default=2958.0,
        help=(
            "Hard performance-regression ceiling for the complete null, in "
            "seconds. The default is the reviewed baseline for K=1024/M=4 on "
            "the host this was established on (about 49 minutes); remeasure "
            "before relying on it elsewhere."
        ),
    )
    parser.add_argument(
        "--memory-limit-bytes", type=int, default=1024 ** 3,
        help="Accounted finalization workspace limit.",
    )
    return parser.parse_args(argv)


def _mib(value: float) -> str:
    return f"{value / 1024 ** 2:,.0f} MiB"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print("CountSketch alignment-finalization benchmark")
    print(f"  D = {args.positions:,}   K = {args.buckets:,}   M = {args.maps}")
    print(f"  jobs = {args.jobs}   map-jobs = {args.jobs * args.maps}"
          f"   draws per null = {PRODUCTION_DRAWS}")
    print()

    equivalence = check_numerical_equivalence()
    status = "identical" if equivalence["identical"] else "DIFFERENT"
    print(f"Numerical equivalence vs. _permutation_null: {status}")
    print(f"  draw sequence identical: {equivalence['draw_sequence_identical']}")
    if not equivalence["identical"]:
        print(f"  mismatched keys: {equivalence['mismatched_keys']}")
        print("\nRefusing to report timings for a computation that is not the "
              "production null.")
        return 2
    print()

    requested = [name.strip() for name in args.profiles.split(",") if name.strip()]
    unknown = sorted(set(requested) - set(PROFILES))
    if unknown:
        raise SystemExit(f"Unknown profile(s): {', '.join(unknown)}")

    if args.reduction != "reduceat":
        print(f"Reduction: {args.reduction} -- NOT bitwise the production "
              "reduction. Reported as a measured option, not an adopted one.\n")

    results = [
        time_profile(
            profile,
            positions=args.positions,
            buckets=args.buckets,
            timed_draws=args.timed_draws,
            draw_batch=args.draw_batch,
            seed=args.seed,
            reduction=args.reduction,
        )
        for profile in requested
    ]

    header = (
        f"{'profile':<10} {'classes':>9} {'largest':>8} {'s/draw':>9} "
        f"{'s/map-job':>10} {'RSS+':>10} {'accounted':>10}"
    )
    print(header)
    print("-" * len(header))
    for entry in results:
        accounted = sum(entry["accounted_bytes"].values())
        print(
            f"{entry['profile']:<10} {entry['num_classes']:>9,} "
            f"{entry['largest_class']:>8,} {entry['seconds_per_draw']:>9.4f} "
            f"{entry['seconds_per_map_job']:>10.2f} "
            f"{_mib(entry['rss_peak_increment_bytes']):>10} "
            f"{_mib(accounted):>10}"
        )
    print()

    stretch = args.collection_seconds * args.budget_fraction
    map_jobs = args.jobs * args.maps
    print(f"Collection wall time (measured): {args.collection_seconds:,.0f} s")
    print(f"Stretch target at {args.budget_fraction:.0%}: {stretch:,.0f} s "
          "(reported, not a gate)")
    print(f"Regression ceiling: {args.ceiling_seconds:,.0f} s "
          f"({args.ceiling_seconds / 60:.0f} min) -- this is the hard limit")
    print()

    worst = max(results, key=lambda entry: entry["seconds_per_map_job"])
    representative = [
        entry for entry in results if entry["profile"] != "singleton"
    ] or results
    typical = sum(
        entry["seconds_per_map_job"] for entry in representative
    ) / len(representative)

    for label, per_map_job in (
        ("mean of representative profiles", typical),
        (f"worst profile ({worst['profile']})", worst["seconds_per_map_job"]),
    ):
        total = per_map_job * map_jobs
        share = total / args.collection_seconds
        verdict = "WITHIN" if total <= args.ceiling_seconds else "OVER CEILING"
        stretch_note = "meets stretch" if total <= stretch else "above stretch"
        print(
            f"  {label:<34} {per_map_job:8.2f} s/map-job "
            f"-> {total:9,.0f} s total = {share:6.1%} of collection "
            f"[{verdict}; {stretch_note}]"
        )
    print()

    over_memory = [
        entry for entry in results
        if sum(entry["accounted_bytes"].values()) > args.memory_limit_bytes
    ]
    if over_memory:
        print(f"Accounted workspace exceeds {_mib(args.memory_limit_bytes)} for: "
              + ", ".join(entry["profile"] for entry in over_memory))
    else:
        print(f"Accounted workspace within {_mib(args.memory_limit_bytes)} "
              "for every profile.")

    if args.compare_reductions:
        print()
        comparison = compare_reductions(
            positions=args.positions, buckets=args.buckets,
            draws=PRODUCTION_DRAWS, seed=args.seed,
        )
        print(f"Reduction comparison over the full {PRODUCTION_DRAWS} draws "
              f"({comparison['profile']} profile):")
        print(f"  reduceat  {comparison['reduceat_seconds']:8.1f} s"
              f"   blockwise {comparison['blockwise_seconds']:8.1f} s"
              f"   speedup {comparison['speedup']:.1f}x")
        print(f"  {'statistic':<14} {'reduceat':>18} {'blockwise':>18} "
              f"{'abs':>10} {'rel':>10}")
        for key, entry in comparison["differences"].items():
            print(f"  {key:<14} {entry['reduceat']:18.14f} "
                  f"{entry['blockwise']:18.14f} {entry['absolute']:10.2e} "
                  f"{entry['relative']:10.2e}")
        print(f"  worst relative difference on a reported statistic: "
              f"{comparison['max_relative']:.2e}")

    total = typical * map_jobs
    exceeded = total > args.ceiling_seconds
    if exceeded:
        print()
        print("REGRESSION CEILING EXCEEDED: the null costs "
              f"{total:,.0f} s against a {args.ceiling_seconds:,.0f} s ceiling.")
        print("Do not absorb this by reducing draws, dropping outputs or "
              "changing the estimator; report it.")
    elif total > stretch:
        print()
        print(f"Within the ceiling; above the {args.budget_fraction:.0%} stretch "
              "target, which is reported rather than enforced.")
    return 1 if exceeded else 0


if __name__ == "__main__":
    raise SystemExit(main())
