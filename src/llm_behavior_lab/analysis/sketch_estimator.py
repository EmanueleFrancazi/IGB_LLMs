"""The CountSketch estimator arithmetic, shared and dependency-free.

NumPy only, and **no project imports at all** -- deliberately. Both the
production directional analyses and the offline fidelity methodology need the
same estimator, and neither should have to import the other to get it. A lower
layer reaching up into ``countsketch_fidelity`` for arithmetic would be a cycle
waiting to happen.

The estimator, once, so it cannot be restated differently in two places::

    cos(a, b) ~ (1/M) sum_m <S_m(a), S_m(b)> / (||a|| ||b||)
              = (1/M) sum_m <u_m(a), u_m(b)>,     u_m(d) = S_m(g_d) / ||g_d||

Two properties of that definition drive everything here.

**Inner products are averaged, never sketch vectors.** ``mean_m S_m`` is not an
estimator of anything: the maps are independent projections, so averaging them
shrinks the very signal each one carries. :func:`ensemble_cosine_matrix`
averages *matrices of inner products*, and :func:`ensemble_embedding` reaches
the same answer by a route that keeps the arithmetic bilinear.

**The ensemble estimate is an inner product in a wider space.** Writing
``v(d) = M^{-1/2} [u_1(d) | ... | u_M(d)]`` gives ``<v(a), v(b)>`` exactly equal
to the ensemble estimate above. That identity is what lets every existing
bilinear consumer -- class sums, pooled within/between, permutation nulls,
cross-partition cells -- produce the ensemble estimate with no change to its own
formula. What it cannot do is preserve the *individual* map estimates, which is
why :func:`ensemble_summary` and the per-map surfaces exist beside it.

The ``retained_*`` helpers build dense ``[m, m]`` matrices and are for the
**bounded** exact-gradient fidelity subset only. Never the full-D production
path: at ``D = 32768`` and ``M = 4`` a dense float64 stack is 34 GB, so the
production statistics are computed from class sums instead and stay linear in
``D``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "RETAINED_POSITION_CEILING",
    "cosine_from_gram",
    "cosines_from_sketches",
    "ensemble_cosine_matrix",
    "ensemble_embedding",
    "ensemble_summary",
    "retained_ensemble_cosine_matrix",
    "retained_per_map_cosine_matrices",
]

#: Largest position count the dense ``retained_*`` helpers will accept.
#:
#: The supported workflow retains **12** positions: the runner hard-codes
#: ``DEFAULT_SANITY_POSITIONS`` at ``select_sanity_positions(count=...)`` with no
#: CLI or config override, and no test uses more. 256 is over twenty times that,
#: so nothing that works today is refused -- while the dense output it permits
#: stays small enough to be uninteresting.
#:
#: An earlier draft used 4096, derived only from "1 GiB of output at eight maps".
#: That was too permissive on three counts: a gibibyte is not a small
#: diagnostic allocation, the figure ignored BLAS workspace and the operand
#: arrays beside the output, and it said nothing at all about a caller passing
#: more than eight maps. Position count alone is the wrong single lever, which is
#: why the three budgets below sit beside it.
RETAINED_POSITION_CEILING = 256

#: Largest map count the dense helpers will accept. Eight is the alternate-seed
#: bank; 64 leaves generous room without letting the map axis alone drive the
#: allocation.
RETAINED_MAP_CEILING = 64

#: Largest dense output the helpers will allocate, in bytes. At the ceilings
#: above this binds first: ``64 x 256^2 x 8`` is 32 MiB, comfortably inside it.
RETAINED_OUTPUT_BYTE_CEILING = 64 * 1024 * 1024

#: Multiply-accumulate budget for the Gram products, ``M * m^2 * K``. Output
#: bytes say nothing about the *work*: a narrow output can still sit behind an
#: enormous matrix multiplication when the bucket axis is wide.
RETAINED_OPERATION_CEILING = 2**30


def cosine_from_gram(gradients: np.ndarray, norms: np.ndarray) -> np.ndarray:
    """Full-gradient cosine matrix, accumulated in float64.

    "Exact" here means unprojected, not infinitely precise: the stored vectors
    are float32 and the products below are float64.
    """

    values = np.asarray(gradients, dtype=np.float64)
    norms = np.asarray(norms, dtype=np.float64)
    return (values @ values.T) / np.outer(norms, norms)


def cosines_from_sketches(sketches: np.ndarray, norms: np.ndarray) -> np.ndarray:
    """The production estimator on already-projected sketches.

    Exact norms in the denominator, as everywhere else, so the result estimates
    ``cos(g_a, g_b)`` without bias and is not confined to ``[-1, 1]``.
    """

    sketches = np.asarray(sketches, dtype=np.float64)
    norms = np.asarray(norms, dtype=np.float64)
    return (sketches @ sketches.T) / np.outer(norms, norms)


def _validated_unit_per_map(
    unit_per_map: Any, *, name: str, cast: bool = True
) -> np.ndarray:
    """Check a ``[position, map, bucket]`` array, optionally casting to float64.

    ``cast=False`` returns the array in its stored dtype. Callers that fill a
    buffer map by map want that: casting the whole stack to float64 up front
    would materialize a second full-size copy, which is precisely the peak this
    module is written to avoid.
    """

    values = np.asarray(unit_per_map)
    if values.ndim != 3:
        raise ValueError(
            f"{name} must be [positions, maps, buckets]; got {values.ndim} "
            f"dimensions with shape {values.shape}."
        )
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"{name} must be numeric; got dtype {values.dtype}.")
    positions, maps, buckets = values.shape
    if positions == 0 or maps == 0 or buckets == 0:
        raise ValueError(
            f"{name} has an empty axis (shape {values.shape}); a cosine over no "
            "positions, no maps or no buckets is undefined."
        )
    return values.astype(np.float64, copy=False) if cast else values


def _require_bounded(shape: tuple[int, int, int], itemsize: int) -> None:
    """Refuse a dense matrix that is not the bounded fidelity subset.

    **Runs before anything is allocated and before any matrix multiplication.**
    Four budgets, because no single one of them is sufficient: positions bound
    the shape, maps bound the stack, output bytes bound the result, and the
    operation count bounds the *work* -- a narrow output can still hide an
    enormous product when the bucket axis is wide.
    """

    positions, maps, buckets = shape
    if positions > RETAINED_POSITION_CEILING:
        raise ValueError(
            f"{positions} positions exceeds the retained-subset ceiling of "
            f"{RETAINED_POSITION_CEILING}. These helpers build dense "
            "[maps, positions, positions] matrices and exist for the bounded "
            "exact-gradient fidelity subset; the production directional "
            "statistics are computed from class sums and stay linear in the "
            "position count. Reaching here with a full position set is a bug, "
            "not a configuration to raise."
        )
    if maps > RETAINED_MAP_CEILING:
        raise ValueError(
            f"{maps} maps exceeds the retained-subset ceiling of "
            f"{RETAINED_MAP_CEILING}."
        )
    output_bytes = maps * positions * positions * itemsize
    if output_bytes > RETAINED_OUTPUT_BYTE_CEILING:
        raise ValueError(
            f"The dense output would be {output_bytes} bytes, above the "
            f"{RETAINED_OUTPUT_BYTE_CEILING}-byte ceiling for this bounded "
            "diagnostic."
        )
    operations = maps * positions * positions * buckets
    if operations > RETAINED_OPERATION_CEILING:
        raise ValueError(
            f"The Gram products would take about {operations} multiply-adds, "
            f"above the {RETAINED_OPERATION_CEILING} ceiling. The output may be "
            "small, but the work behind it is not."
        )


def retained_per_map_cosine_matrices(unit_per_map: Any) -> np.ndarray:
    """``[M, m, m]`` per-map Gram matrices of normalized sketches.

    One matrix per map: entry ``(a, b)`` of map ``m`` is
    ``<u_m(a), u_m(b)>``, that map's own estimate of ``cos(g_a, g_b)``.

    **Bounded subset only.** See :data:`RETAINED_POSITION_CEILING`; the
    production path never calls this, and a test spies on it to prove so.

    Args:
        unit_per_map: ``[positions, maps, buckets]`` normalized sketches.

    Returns:
        ``[maps, positions, positions]`` float64.
    """

    raw = _validated_unit_per_map(unit_per_map, name="unit_per_map", cast=False)
    # Guards first, on the *requested* allocation, before the float64 cast and
    # before the product. Checking afterwards would be checking a thing that has
    # already been built.
    _require_bounded(raw.shape, np.dtype(np.float64).itemsize)
    values = raw.astype(np.float64, copy=False)
    # [position, map, bucket] -> [map, position, bucket], then a Gram per map.
    by_map = np.swapaxes(values, 0, 1)
    return by_map @ np.swapaxes(by_map, 1, 2)


def ensemble_cosine_matrix(per_map_matrices: Any) -> np.ndarray:
    """Mean of per-map cosine matrices -- the ensemble estimate.

    The arithmetic mean over the map axis, which is the estimator's definition.
    Averaging the *sketch vectors* and taking one inner product afterwards is a
    different and wrong quantity; that is the mistake this function exists to
    make unavailable.

    Args:
        per_map_matrices: ``[maps, positions, positions]``.

    Returns:
        ``[positions, positions]`` float64.
    """

    values = np.asarray(per_map_matrices, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(
            "per_map_matrices must be [maps, positions, positions]; got "
            f"{values.ndim} dimensions."
        )
    if values.shape[0] == 0:
        raise ValueError("per_map_matrices must contain at least one map.")
    return values.mean(axis=0)


def retained_ensemble_cosine_matrix(unit_per_map: Any) -> np.ndarray:
    """``[m, m]`` ensemble cosine matrix for the bounded retained subset."""

    return ensemble_cosine_matrix(retained_per_map_cosine_matrices(unit_per_map))


def ensemble_embedding(
    unit_per_map: Any, *, position_scale: Any | None = None
) -> np.ndarray:
    """``[D, M*K]`` concatenation scaled by ``1/sqrt(M)``.

    The identity that makes this useful::

        <v(a), v(b)> = (1/M) sum_m <u_m(a), u_m(b)>

    exactly -- so any consumer whose statistic is a bilinear form in the rows
    produces the ensemble estimate by reading this instead of a single map's
    sketches, with no change to its own formula. Class sums, pooled
    within/between, permutation nulls and cross-partition cells are all of that
    kind.

    What it does **not** carry is the individual map estimates: the sum has
    already been taken. Uncertainty comes from the per-map surfaces and
    :func:`ensemble_summary`, never from this array.

    At ``M = 1`` the scale factor is exactly ``1.0``, so the values are the
    historical ones unchanged -- but callers that must guarantee bitwise identity
    should branch on ``M == 1`` and skip this rather than rely on that.

    **One buffer, filled map by map.** The obvious implementation --
    ``(values / sqrt(M)).reshape(...)`` -- casts the whole ``[D, M, K]`` stack to
    float64 and then allocates a second full-size array for the division, so its
    peak is *twice* the returned buffer: 2 GiB for a 1 GiB result at
    ``D = 32768, M = 4, K = 1024``. Writing each map's block straight into the
    output leaves a transient of one ``[D, K]`` slice instead.

    Args:
        unit_per_map: ``[D, M, K]``. Read only; never modified.
        position_scale: Optional ``[D]`` per-position multiplier, folded into the
            same pass. Passing ``1/||g_d||`` here is what lets a caller normalize
            and concatenate without building a normalized ``[D, M, K]`` first.

    Returns:
        ``[D, M*K]`` float64.
    """

    values = _validated_unit_per_map(unit_per_map, name="unit_per_map", cast=False)
    positions, maps, buckets = values.shape

    factor = 1.0 if maps == 1 else 1.0 / np.sqrt(float(maps))
    if position_scale is None:
        column = None if maps == 1 else np.float64(factor)
    else:
        scale = np.asarray(position_scale, dtype=np.float64)
        if scale.shape != (positions,):
            raise ValueError(
                f"position_scale must have shape ({positions},); got {scale.shape}."
            )
        column = (scale * factor)[:, None]

    out = np.empty((positions, maps * buckets), dtype=np.float64)
    for index in range(maps):
        block = values[:, index, :]
        start = index * buckets
        if column is None:
            out[:, start : start + buckets] = block
        else:
            np.multiply(block, column, out=out[:, start : start + buckets])
    return out


def ensemble_summary(theta_per_map: Any, *, axis: int = -1) -> dict[str, Any]:
    """Summarize a statistic measured once per map.

    Args:
        theta_per_map: Per-map values. Scalars give a 1-D array over maps;
            leading dimensions are kept, so a ``[T, M]`` input summarizes ``T``
            statistics at once.
        axis: Which axis is the map axis. Defaults to the last.

    Returns:
        ``map_mean``, ``sample_sd``, ``standard_error``, ``degrees_of_freedom``
        and ``uncertainty_available``. Sample SD uses ``ddof = 1`` and the
        standard error is ``sample_sd / sqrt(M)``.

        At ``M = 1`` spread is **unavailable, not zero**: ``sample_sd`` and
        ``standard_error`` are ``None``, ``degrees_of_freedom`` is 0 and
        ``uncertainty_available`` is ``False``. One projection cannot say how far
        another would have landed, and reporting 0.0 would assert precisely the
        certainty the replica design exists to measure.

        A non-finite value at ``M > 1`` is left non-finite and
        ``uncertainty_available`` stays ``True``: that is a broken computation,
        and collapsing it into the "unavailable" case would hide it behind the
        legitimate single-map one.

    Raises:
        ValueError: If the map axis is empty or the input is not numeric.
    """

    values = np.asarray(theta_per_map)
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError(
            f"theta_per_map must be numeric; got dtype {values.dtype}."
        )
    if values.ndim == 0:
        raise ValueError(
            "theta_per_map must have a map axis; a bare scalar does not say how "
            "many maps produced it."
        )
    maps = values.shape[axis]
    if maps == 0:
        raise ValueError("theta_per_map has an empty map axis.")

    values = values.astype(np.float64, copy=False)
    mean = values.mean(axis=axis)
    scalar = np.ndim(mean) == 0

    if maps == 1:
        return {
            "map_mean": float(mean) if scalar else mean,
            "sample_sd": None,
            "standard_error": None,
            "degrees_of_freedom": 0,
            "uncertainty_available": False,
        }

    sample_sd = values.std(axis=axis, ddof=1)
    standard_error = sample_sd / np.sqrt(float(maps))
    return {
        "map_mean": float(mean) if scalar else mean,
        "sample_sd": float(sample_sd) if scalar else sample_sd,
        "standard_error": float(standard_error) if scalar else standard_error,
        "degrees_of_freedom": maps - 1,
        "uncertainty_available": True,
    }
