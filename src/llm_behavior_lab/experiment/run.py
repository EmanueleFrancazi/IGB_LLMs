"""Experiment-run directory creation, metadata, and config snapshots."""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from llm_behavior_lab.experiment.arrays import ArrayMetricStore
from llm_behavior_lab.experiment.checkpoints import CheckpointManager
from llm_behavior_lab.experiment.config import ExperimentSettings
from llm_behavior_lab.experiment.metrics import MetricLogger
from llm_behavior_lab.experiment.serialization import atomic_write_json, atomic_write_yaml, utc_now_iso


def _safe_component(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._")
    if not normalized:
        raise ValueError("Experiment and run identifiers must contain a safe character.")
    return normalized


def generate_run_id() -> str:
    """Generate a collision-resistant, sortable run identifier."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}_{uuid.uuid4().hex[:8]}"


def get_git_commit(repo_root: str | Path | None) -> str | None:
    """Return the current Git commit without making Git a hard dependency."""

    if repo_root is None:
        return None
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(repo_root),
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    commit = completed.stdout.strip()
    return commit or None


def collect_environment_info() -> dict[str, Any]:
    """Collect lightweight package and platform information."""

    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "process_id": os.getpid(),
    }


@dataclass(frozen=True)
class RunPaths:
    """Predictable paths inside one experiment run."""

    run_dir: Path
    metadata_path: Path
    config_dir: Path
    metrics_dir: Path
    array_metrics_dir: Path
    checkpoints_dir: Path
    analyses_dir: Path
    logs_dir: Path


class ExperimentRun:
    """Own the directory and persistence interfaces for one immutable run."""

    def __init__(self, *, experiment_name: str, run_id: str, settings: ExperimentSettings, paths: RunPaths) -> None:
        self.experiment_name = experiment_name
        self.run_id = run_id
        self.settings = settings
        self.paths = paths
        self.training_metrics = MetricLogger(
            paths.metrics_dir / settings.logging.training_metrics_filename,
            flush_every_records=settings.logging.flush_every_records,
        )
        self.evaluation_metrics = MetricLogger(
            paths.metrics_dir / settings.logging.evaluation_metrics_filename,
            flush_every_records=settings.logging.flush_every_records,
        )
        self.array_metrics = ArrayMetricStore(paths.array_metrics_dir, run_dir=paths.run_dir)
        self.checkpoints = CheckpointManager(
            paths.checkpoints_dir,
            filename_width=settings.checkpointing.filename_width,
            save_latest_pointer=settings.checkpointing.save_latest_pointer,
        )

    @classmethod
    def create(
        cls,
        settings: ExperimentSettings,
        *,
        run_id: str | None = None,
        repo_root: str | Path | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ExperimentRun":
        """Create a new run directory and immutable metadata file."""

        settings.validate()
        experiment_name = _safe_component(settings.name)
        resolved_run_id = _safe_component(run_id or generate_run_id())
        run_dir = settings.output_dir / experiment_name / resolved_run_id
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise FileExistsError(
                f"Experiment run already exists and will not be overwritten: {run_dir}"
            ) from exc

        paths = RunPaths(
            run_dir=run_dir,
            metadata_path=run_dir / "metadata.json",
            config_dir=run_dir / "config",
            metrics_dir=run_dir / "metrics",
            array_metrics_dir=run_dir / "metrics" / "array_metrics",
            checkpoints_dir=run_dir / "checkpoints",
            analyses_dir=run_dir / "analyses",
            logs_dir=run_dir / "logs",
        )
        for directory in (
            paths.config_dir,
            paths.metrics_dir,
            paths.array_metrics_dir,
            paths.checkpoints_dir,
            paths.analyses_dir,
            paths.logs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=False)

        base_metadata: dict[str, Any] = {
            "experiment_name": experiment_name,
            "run_id": resolved_run_id,
            "created_at": utc_now_iso(),
            "phase": settings.phase,
            "seed": settings.seed,
            "notes": settings.notes,
            "tags": list(settings.tags),
            "git_commit": get_git_commit(repo_root),
            "environment": collect_environment_info(),
        }
        if metadata is not None:
            overlapping = sorted(set(base_metadata).intersection(metadata))
            if overlapping:
                raise ValueError(
                    "Additional metadata cannot replace immutable fields: " + ", ".join(overlapping)
                )
            base_metadata.update(dict(metadata))
        atomic_write_json(paths.metadata_path, base_metadata)
        return cls(
            experiment_name=experiment_name,
            run_id=resolved_run_id,
            settings=settings,
            paths=paths,
        )

    def snapshot_config(self, filename: str, config: Mapping[str, Any]) -> Path:
        """Save one YAML config snapshot without overwriting an existing file."""

        if Path(filename).name != filename:
            raise ValueError("Config snapshot filename must not contain directories.")
        if not filename.endswith((".yaml", ".yml")):
            raise ValueError("Config snapshot filename must end in .yaml or .yml.")
        destination = self.paths.config_dir / filename
        if destination.exists():
            raise FileExistsError(f"Config snapshot already exists: {destination}")
        return atomic_write_yaml(destination, dict(config))

    def snapshot_configs(
        self,
        *,
        model_config: Mapping[str, Any],
        data_config: Mapping[str, Any],
        experiment_config: Mapping[str, Any],
    ) -> dict[str, Path]:
        """Save the three standard Phase 6 config snapshots."""

        return {
            "model": self.snapshot_config("model_config.yaml", model_config),
            "data": self.snapshot_config("data_config.yaml", data_config),
            "experiment": self.snapshot_config("experiment_config.yaml", experiment_config),
        }

    def save_analysis_json(self, filename: str, payload: Any, *, overwrite: bool = False) -> Path:
        """Save a structured analysis artifact under ``analyses/``."""

        if Path(filename).name != filename:
            raise ValueError("Analysis filename must not contain directories.")
        if not filename.endswith(".json"):
            raise ValueError("Analysis filename must end in .json.")
        destination = self.paths.analyses_dir / filename
        if destination.exists() and not overwrite:
            raise FileExistsError(f"Analysis artifact already exists: {destination}")
        return atomic_write_json(destination, payload)
