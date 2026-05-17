"""Device selection helpers."""

from __future__ import annotations

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
