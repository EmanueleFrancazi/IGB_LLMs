"""Small serialization helpers shared by Phase 6 persistence modules."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def utc_now_iso() -> str:
    """Return a timezone-aware UTC timestamp in ISO 8601 format."""

    return datetime.now(timezone.utc).isoformat()


def to_jsonable(value: Any) -> Any:
    """Convert common scientific-Python values to JSON-compatible objects."""

    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Value of type {type(value).__name__} is not JSON serializable.")


def atomic_write_text(path: str | Path, content: str) -> Path:
    """Write text through a temporary file and atomically replace the target."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return destination


def atomic_write_json(path: str | Path, payload: Any, *, indent: int = 2) -> Path:
    """Serialize JSON-compatible data through an atomic text write."""

    return atomic_write_text(
        path,
        json.dumps(to_jsonable(payload), indent=indent, sort_keys=True) + "\n",
    )


def atomic_write_yaml(path: str | Path, payload: Any) -> Path:
    """Serialize configuration data through an atomic YAML write."""

    return atomic_write_text(
        path,
        yaml.safe_dump(to_jsonable(payload), sort_keys=False),
    )
