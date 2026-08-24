"""Paired sampling and loss temperatures for a directional measurement.

Two different temperatures appear in this work and the code used one word for
both. They are not the same quantity:

* the **sampling temperature** ``T_s`` shapes the distribution a grouping label
  is drawn from, ``h_d ~ nucleus(softmax(z_d / T_s), top_p)``; and
* the **loss temperature** ``T_g`` shapes the objective the gradient comes from,
  ``L_d(T_g) = -log softmax(z_d / T_g)[y_d]``, whose target ``y_d`` is always the
  true corpus token.

The sampled token is a *grouping label*. It is never the supervised target: a
measurement that trained on its own sample would be a different experiment.

Pairs are elementwise, ``(T_s[i], T_g[i])``, not a Cartesian product. A sweep is
therefore an explicit list of the measurements wanted, which keeps a controlled
design expressible -- figure 22 varies ``T_s`` with ``T_g`` pinned at 1 -- while
the matched design ``T_g = T_s`` is just the default. Product semantics would
make the controlled design impossible to state without post-hoc filtering.

Omitting the loss temperatures means ``T_g = T_s`` elementwise. That default
applies to **new** requests only. Historical records predate this interface and
must keep their original reading; see
:func:`~llm_behavior_lab.analysis.nucleus_clustering_artifact.load_nucleus_clustering_artifact`.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

__all__ = [
    "describe_pairing",
    "temperature_pairs",
    "validate_pair_aligned",
]


def _validated(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"Every value in {name} must be finite.")
    bad = array[array <= 0.0]
    if bad.size:
        raise ValueError(
            f"Every value in {name} must be positive; got {bad[0]:g}. Softmax at "
            "T = 0 is undefined, and greedy is already the T -> 0 limit rather "
            "than a temperature that can be requested here."
        )
    return array


def temperature_pairs(
    sampling_temperatures: Sequence[float] | np.ndarray,
    loss_temperatures: Sequence[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Validate a requested sweep and group it for reuse.

    Args:
        sampling_temperatures: ``T_s`` per requested measurement.
        loss_temperatures: ``T_g`` per requested measurement, elementwise. When
            omitted, defaults to a copy of ``sampling_temperatures``, i.e.
            ``T_g = T_s``.

    Returns:
        ``sampling_temperatures`` and ``loss_temperatures`` as equal-length
        float arrays; ``num_pairs``; the ``unique_sampling`` and ``unique_loss``
        values with the ``sampling_index`` / ``loss_index`` that map each pair
        onto them; and ``loss_defaulted``, recording whether ``T_g`` was
        requested or inherited.

    Raises:
        ValueError: On empty input, on lengths that differ, or on any
            non-positive or non-finite temperature. Lengths are never broadcast:
            a mismatch means the caller has a different sweep in mind than the
            one it wrote down, and guessing which would silently measure
            something nobody asked for.
    """

    sampling = _validated(sampling_temperatures, "sampling_temperatures")
    defaulted = loss_temperatures is None
    loss = (
        sampling.copy()
        if defaulted
        else _validated(loss_temperatures, "loss_temperatures")
    )
    if loss.shape != sampling.shape:
        raise ValueError(
            f"sampling_temperatures holds {sampling.size} value(s) and "
            f"loss_temperatures holds {loss.size}. They are paired elementwise, "
            "so they must be the same length; omit loss_temperatures for "
            "T_g = T_s."
        )

    unique_sampling, sampling_index = np.unique(sampling, return_inverse=True)
    unique_loss, loss_index = np.unique(loss, return_inverse=True)
    return {
        "sampling_temperatures": sampling,
        "loss_temperatures": loss,
        "num_pairs": int(sampling.size),
        "unique_sampling": unique_sampling,
        "unique_loss": unique_loss,
        "sampling_index": sampling_index.astype(np.int64),
        "loss_index": loss_index.astype(np.int64),
        "loss_defaulted": bool(defaulted),
        "matched": bool(np.array_equal(sampling, loss)),
    }


def validate_pair_aligned(
    sampling_temperatures: np.ndarray,
    loss_temperatures: np.ndarray,
    **per_pair_arrays: np.ndarray,
) -> int:
    """Check that every per-pair array shares the temperature-pair dimension.

    A result array whose leading dimension has drifted from the temperature
    arrays would silently plot one measurement against another's temperature, so
    a reader validates this before believing anything it loaded.

    Returns:
        The number of pairs.
    """

    sampling = np.asarray(sampling_temperatures).reshape(-1)
    loss = np.asarray(loss_temperatures).reshape(-1)
    if sampling.size != loss.size:
        raise ValueError(
            f"The artifact stores {sampling.size} sampling temperature(s) and "
            f"{loss.size} loss temperature(s); they must be paired elementwise."
        )
    if sampling.size == 0:
        raise ValueError("The artifact stores no temperature pairs.")
    for name, values in per_pair_arrays.items():
        array = np.asarray(values)
        if array.shape[0] != sampling.size:
            raise ValueError(
                f"{name!r} has leading dimension {array.shape[0]} but the "
                f"artifact stores {sampling.size} temperature pair(s)."
            )
    return int(sampling.size)


def describe_pairing(
    sampling_temperatures: np.ndarray, loss_temperatures: np.ndarray
) -> str:
    """A short phrase stating how ``T_g`` relates to ``T_s`` in this sweep.

    Written for a caption, where the reader needs to know which temperature the
    x-axis carries and what the other one was doing. The three cases a sweep
    actually falls into are named rather than printing a list of pairs.
    """

    sampling = np.asarray(sampling_temperatures, dtype=np.float64).reshape(-1)
    loss = np.asarray(loss_temperatures, dtype=np.float64).reshape(-1)
    if np.array_equal(sampling, loss):
        return "T_g = T_s (matched)"
    unique_loss = np.unique(loss)
    if unique_loss.size == 1:
        return f"T_g = {unique_loss[0]:g} fixed"
    return "T_g varies independently of T_s"
