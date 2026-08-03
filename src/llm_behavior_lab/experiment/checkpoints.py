"""Local model checkpoint save/load and discovery utilities."""

from __future__ import annotations

import inspect
import json
import os
import random
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from llm_behavior_lab.experiment.serialization import atomic_write_json, to_jsonable, utc_now_iso

CHECKPOINT_FORMAT_VERSION = 1
_CHECKPOINT_PATTERN = re.compile(r"^checkpoint_step_(\d+)\.pt$")


@dataclass(frozen=True)
class CheckpointInfo:
    """Filesystem and step metadata for a saved checkpoint."""

    path: Path
    step: int
    checkpoint_id: str


def capture_rng_state() -> dict[str, Any]:
    """Capture CPU/Python/NumPy RNG state without initializing CUDA."""

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_initialized():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore RNG state fields present in a checkpoint."""

    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_initialized():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _torch_load(path: Path, *, map_location: str | torch.device | None) -> dict[str, Any]:
    """Load trusted local checkpoints across PyTorch versions."""

    kwargs: dict[str, Any] = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    payload = torch.load(path, **kwargs)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint payload must be a dictionary: {path}")
    return payload


class CheckpointManager:
    """Save, discover, validate, and restore local checkpoints."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        *,
        filename_width: int = 6,
        save_latest_pointer: bool = True,
    ) -> None:
        if filename_width <= 0:
            raise ValueError("filename_width must be positive.")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.filename_width = filename_width
        self.save_latest_pointer = save_latest_pointer
        self.latest_pointer_path = self.checkpoint_dir / "latest.json"

    def checkpoint_path(self, step: int) -> Path:
        if step < 0:
            raise ValueError("step must be non-negative.")
        return self.checkpoint_dir / f"checkpoint_step_{step:0{self.filename_width}d}.pt"

    def save(
        self,
        *,
        model: nn.Module,
        step: int,
        model_config: Mapping[str, Any] | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        additional_state: Mapping[str, Any] | None = None,
        include_rng_state: bool = True,
        overwrite: bool = False,
    ) -> CheckpointInfo:
        """Save a checkpoint through a temporary file and update ``latest.json``."""

        destination = self.checkpoint_path(step)
        if destination.exists() and not overwrite:
            raise FileExistsError(f"Checkpoint already exists: {destination}")

        payload: dict[str, Any] = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "created_at": utc_now_iso(),
            "global_step": int(step),
            "model_state_dict": model.state_dict(),
            "model_config": None if model_config is None else to_jsonable(dict(model_config)),
            "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
            "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
            "rng_state": capture_rng_state() if include_rng_state else None,
            "additional_state": {} if additional_state is None else dict(additional_state),
        }

        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.stem}.",
            suffix=".tmp",
            dir=destination.parent,
        )
        os.close(file_descriptor)
        temporary_path = Path(temporary_name)
        try:
            torch.save(payload, temporary_path)
            os.replace(temporary_path, destination)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

        info = CheckpointInfo(
            path=destination,
            step=step,
            checkpoint_id=destination.stem,
        )
        if self.save_latest_pointer:
            atomic_write_json(
                self.latest_pointer_path,
                {
                    "checkpoint_id": info.checkpoint_id,
                    "filename": destination.name,
                    "global_step": step,
                    "updated_at": utc_now_iso(),
                },
            )
        return info

    def list_checkpoints(self) -> list[CheckpointInfo]:
        """Return available checkpoints sorted by global step."""

        checkpoints: list[CheckpointInfo] = []
        for path in self.checkpoint_dir.glob("checkpoint_step_*.pt"):
            match = _CHECKPOINT_PATTERN.match(path.name)
            if match is None:
                continue
            step = int(match.group(1))
            checkpoints.append(
                CheckpointInfo(path=path, step=step, checkpoint_id=path.stem)
            )
        return sorted(checkpoints, key=lambda item: item.step)

    def latest_checkpoint(self) -> CheckpointInfo | None:
        """Return the latest checkpoint, preferring a valid latest pointer."""

        if self.latest_pointer_path.exists():
            try:
                pointer = json.loads(self.latest_pointer_path.read_text(encoding="utf-8"))
                filename = pointer["filename"]
                step = int(pointer["global_step"])
                path = self.checkpoint_dir / filename
                if path.exists():
                    return CheckpointInfo(path=path, step=step, checkpoint_id=path.stem)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                pass

        checkpoints = self.list_checkpoints()
        return checkpoints[-1] if checkpoints else None

    @staticmethod
    def validate_payload(payload: Mapping[str, Any], *, path: Path | None = None) -> None:
        """Validate required checkpoint fields and supported format version."""

        location = "checkpoint" if path is None else str(path)
        required_fields = {"format_version", "global_step", "model_state_dict"}
        missing = sorted(required_fields.difference(payload.keys()))
        if missing:
            raise ValueError(f"Malformed checkpoint {location}; missing fields: {missing}")
        if int(payload["format_version"]) != CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported checkpoint format version in {location}: "
                f"{payload['format_version']} (expected {CHECKPOINT_FORMAT_VERSION})."
            )
        if int(payload["global_step"]) < 0:
            raise ValueError(f"Malformed checkpoint {location}; global_step must be non-negative.")
        if not isinstance(payload["model_state_dict"], Mapping):
            raise ValueError(f"Malformed checkpoint {location}; model_state_dict must be a mapping.")

    def load_payload(
        self,
        checkpoint: str | Path | CheckpointInfo | None = None,
        *,
        map_location: str | torch.device | None = "cpu",
    ) -> dict[str, Any]:
        """Load and validate a selected or latest checkpoint payload."""

        if checkpoint is None:
            info = self.latest_checkpoint()
            if info is None:
                raise FileNotFoundError(f"No checkpoints found in {self.checkpoint_dir}")
            path = info.path
        elif isinstance(checkpoint, CheckpointInfo):
            path = checkpoint.path
        else:
            path = Path(checkpoint)
            if not path.is_absolute():
                path = self.checkpoint_dir / path

        if not path.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        payload = _torch_load(path, map_location=map_location)
        self.validate_payload(payload, path=path)
        return payload

    def restore(
        self,
        *,
        model: nn.Module,
        checkpoint: str | Path | CheckpointInfo | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        map_location: str | torch.device | None = "cpu",
        strict: bool = True,
        restore_rng: bool = False,
    ) -> dict[str, Any]:
        """Load a checkpoint into model and optional optimizer/scheduler objects."""

        payload = self.load_payload(checkpoint, map_location=map_location)
        model.load_state_dict(payload["model_state_dict"], strict=strict)

        optimizer_state = payload.get("optimizer_state_dict")
        if optimizer is not None and optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)

        scheduler_state = payload.get("scheduler_state_dict")
        if scheduler is not None and scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)

        rng_state = payload.get("rng_state")
        if restore_rng and rng_state is not None:
            restore_rng_state(rng_state)
        return payload
