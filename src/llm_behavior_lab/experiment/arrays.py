"""Storage for array-valued experiment diagnostics."""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


@dataclass(frozen=True)
class ArrayArtifact:
    """Reference to one saved NumPy archive."""

    metric_name: str
    step: int
    stage: str
    split: str | None
    path: Path
    relative_path: str
    keys: tuple[str, ...]


def _safe_component(value: str) -> str:
    """Convert a user-facing identifier into a filesystem-safe component."""

    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    normalized = normalized.strip("._")
    if not normalized:
        raise ValueError("Artifact identifiers must contain at least one safe character.")
    return normalized


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class ArrayMetricStore:
    """Save and reload named array diagnostics as compressed ``.npz`` files."""

    def __init__(self, root_dir: str | Path, *, run_dir: str | Path | None = None) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = None if run_dir is None else Path(run_dir)

    def save(
        self,
        *,
        metric_name: str,
        step: int,
        arrays: Mapping[str, Any],
        stage: str,
        split: str | None = None,
        checkpoint_id: str | None = None,
        overwrite: bool = False,
    ) -> ArrayArtifact:
        """Save one compressed array artifact and return its reference."""

        if step < 0:
            raise ValueError("step must be non-negative.")
        if not arrays:
            raise ValueError("arrays must be a non-empty mapping.")

        safe_metric = _safe_component(metric_name)
        safe_stage = _safe_component(stage)
        safe_split = "all" if split is None else _safe_component(split)
        checkpoint_suffix = "" if checkpoint_id is None else f"__{_safe_component(checkpoint_id)}"
        filename = (
            f"{safe_metric}__{safe_stage}__{safe_split}__step_{step:06d}"
            f"{checkpoint_suffix}.npz"
        )
        destination = self.root_dir / filename
        if destination.exists() and not overwrite:
            raise FileExistsError(f"Array artifact already exists: {destination}")

        normalized_arrays = {str(name): _to_numpy(value) for name, value in arrays.items()}
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.stem}.",
            suffix=".npz",
            dir=destination.parent,
        )
        os.close(file_descriptor)
        temporary_path = Path(temporary_name)
        try:
            np.savez_compressed(temporary_path, **normalized_arrays)
            os.replace(temporary_path, destination)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

        if self.run_dir is not None:
            try:
                relative_path = str(destination.relative_to(self.run_dir))
            except ValueError:
                relative_path = str(destination)
        else:
            relative_path = str(destination)

        return ArrayArtifact(
            metric_name=metric_name,
            step=step,
            stage=stage,
            split=split,
            path=destination,
            relative_path=relative_path,
            keys=tuple(normalized_arrays.keys()),
        )

    @staticmethod
    def load(path: str | Path) -> dict[str, np.ndarray]:
        """Load a compressed array artifact into independent NumPy arrays."""

        artifact_path = Path(path)
        if not artifact_path.exists():
            raise FileNotFoundError(f"Array artifact does not exist: {artifact_path}")
        with np.load(artifact_path, allow_pickle=False) as archive:
            return {name: archive[name].copy() for name in archive.files}
