"""Selecting the gradient directions measured at one loss temperature.

A directional analysis needs two things per position: the count sketch of the
gradient, and the **exact** full-parameter norm to divide it by. Both exist per
loss temperature once a record carries the temperature-resolved sketch field;
before that, they existed only at the canonical ``T = 1``.

The whole point of this module is to make the missing case loud. A record
measured at ``T_g = 1`` alone contains no information whatever about the
direction of a ``T_g = 0.12`` gradient, and there is no way to derive one from
the other -- temperature enters the loss, so ``g(T)`` is the gradient of a
different objective and not a rescaling of ``g(1)``. Substituting the canonical
field would therefore not be an approximation; it would silently answer a
different question. So an unmeasured temperature raises, and nothing here
interpolates between measured ones.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from llm_behavior_lab.analysis.records import (
    CANONICAL_TEMPERATURE,
    TEMPERATURE_MATCH_TOLERANCE,
)

__all__ = [
    "TEMPERATURE_MATCH_TOLERANCE",
    "available_loss_temperatures",
    "directional_field",
    "loss_temperature_index",
]

# TEMPERATURE_MATCH_TOLERANCE is re-exported, not defined here. The record layer
# owns it now, because `records.per_map_sketches` resolves temperatures too and
# this module already imports from `records` -- owning it here and importing it
# there would be a cycle. The value and behaviour are unchanged, and the two
# existing importers (`nucleus_clustering`, `figures`) keep working untouched.


def available_loss_temperatures(record: Any) -> tuple[float, ...]:
    """Loss temperatures this record measured gradient directions at."""

    if getattr(record, "has_temperature_gradient_sketches", False):
        return tuple(float(value) for value in record.gradient_temperatures)
    if getattr(record, "has_gradient_position_sketches", False):
        # Directions at the canonical temperature only.
        return (float(CANONICAL_TEMPERATURE),)
    return ()


def loss_temperature_index(record: Any, loss_temperature: float) -> int:
    """Row of the temperature axis holding ``T_g``, or raise.

    Raises:
        ValueError: When the record measured no directional field at this
            temperature. The message lists what *was* measured, because the
            usual cause is a sweep requesting a temperature the measurement run
            was never asked for.
    """

    requested = float(loss_temperature)
    measured = available_loss_temperatures(record)
    if not measured:
        raise ValueError(
            "This record carries no gradient direction information at all, so "
            f"no field exists for loss temperature T_g = {requested:g}."
        )

    values = np.asarray(measured, dtype=np.float64)
    close = np.flatnonzero(np.abs(values - requested) <= TEMPERATURE_MATCH_TOLERANCE)
    if close.size == 0:
        listed = ", ".join(f"{value:g}" for value in measured)
        raise ValueError(
            f"No measured gradient-direction field exists for loss temperature "
            f"T_g = {requested:g}. Measured: {listed}. Directions at one "
            "temperature say nothing about another -- temperature enters the "
            "loss, so g(T) is the gradient of a different objective -- so this "
            "cannot fall back to the canonical field or interpolate."
        )
    if close.size > 1:
        raise ValueError(
            f"Loss temperature T_g = {requested:g} matches {close.size} measured "
            "temperatures, which cannot be resolved unambiguously."
        )
    return int(close[0])


def directional_field(record: Any, loss_temperature: float) -> dict[str, Any]:
    """The sketches and exact norms measured at one loss temperature.

    The norms returned are always the **exact** full-parameter norms measured
    alongside the sketches, never the sketch norms: the production estimator
    divides a sketched inner product by an exact denominator, and swapping in a
    projected one would change the observable.

    Returns:
        ``sketches`` ``[D, K]``, ``norms`` ``[D]``, the resolved
        ``loss_temperature``, its ``index`` on the temperature axis, and
        ``source`` -- ``"temperature_resolved"`` or ``"canonical"``, so a caller
        can report which array it actually read.
    """

    index = loss_temperature_index(record, loss_temperature)
    if getattr(record, "has_temperature_gradient_sketches", False):
        return {
            "sketches": np.asarray(record.gradient_temperature_position_sketches[index]),
            "norms": np.asarray(record.gradient_temperature_position_norms[index]),
            "loss_temperature": float(record.gradient_temperatures[index]),
            "index": index,
            "source": "temperature_resolved",
        }
    # Canonical-only record: the index resolved above can only be the canonical
    # entry, since that is the single temperature such a record reports.
    return {
        "sketches": np.asarray(record.gradient_position_sketches),
        "norms": np.asarray(record.gradient_position_norms),
        "loss_temperature": float(CANONICAL_TEMPERATURE),
        "index": 0,
        "source": "canonical",
    }
