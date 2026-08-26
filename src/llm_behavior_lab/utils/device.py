"""Device selection helpers."""

from __future__ import annotations

import os
from typing import Any

import torch


def get_device(preferred: str = "auto") -> torch.device:
    """Return a PyTorch device from a user-friendly string.

    Args:
        preferred: One of ``"auto"``, ``"cpu"``, ``"cuda"``, ``"mps"``, or a
            valid PyTorch device string such as ``"cuda:0"``.

    Returns:
        Selected ``torch.device``.

    Raises:
        ValueError: If the requested device type is unavailable or invalid.
    """

    requested = preferred.strip().lower()

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cpu":
        return torch.device("cpu")

    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available.")
        return torch.device(requested)

    if requested == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise ValueError("MPS was requested but is not available.")
        return torch.device("mps")

    try:
        return torch.device(requested)
    except RuntimeError as exc:
        raise ValueError(f"Invalid device string: {preferred!r}") from exc


def describe_device(device: torch.device | str) -> dict[str, Any]:
    """Describe which physical accelerator a run actually used.

    ``str(device)`` alone cannot answer that when a campaign is split across two
    GPUs, because it is ambiguous in two directions at once. Two processes
    launched as ``cuda:0`` and ``cuda:1`` differ in the string; two processes
    each launched as ``cuda`` under ``CUDA_VISIBLE_DEVICES=0`` and ``=1`` record
    the *same* string while running on different hardware. And two identical
    cards report the same product name. Only the logical index, the name and the
    raw environment value together separate all three cases.

    Returns:
        A JSON-serializable mapping:

        ``device``
            The canonical device string, matching what the run already records.
        ``type``
            ``"cuda"``, ``"cpu"``, ``"mps"``, and so on.
        ``index``
            The index as *requested*, or ``None`` when the caller said just
            ``"cuda"`` and left the choice to the current context.
        ``resolved_index``
            The logical index actually in use, filled in from
            ``torch.cuda.current_device()`` when the request omitted one.
            ``None`` off CUDA.
        ``name``
            ``torch.cuda.get_device_name`` for that index; ``None`` off CUDA.
        ``cuda_visible_devices``
            The raw environment value. ``None`` means the variable was **not
            set**; ``""`` means it was set to the empty string, which hides every
            GPU. The two are different situations and are recorded differently.

    Note:
        Off CUDA no CUDA API is called at all -- not ``current_device`` and not
        ``get_device_name`` -- so this is safe on a CPU-only machine and on a
        host whose driver is too old to initialize.
    """

    device = torch.device(device)
    description: dict[str, Any] = {
        "device": str(device),
        "type": device.type,
        "index": device.index,
        "resolved_index": None,
        "name": None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }

    if device.type != "cuda":
        return description

    resolved_index = (
        torch.cuda.current_device() if device.index is None else device.index
    )
    description["resolved_index"] = int(resolved_index)
    description["name"] = torch.cuda.get_device_name(resolved_index)
    return description
