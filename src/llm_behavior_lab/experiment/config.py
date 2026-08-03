"""Configuration helpers for experiment persistence.

Phase 6 keeps experiment configuration intentionally lightweight. YAML files
are loaded by scripts as dictionaries, then validated here before run creation.
The validation is separate from training logic so later phases can reuse the
same settings for logging intervals and checkpoint cadence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class LoggingSettings:
    """Settings that control local metric-file behavior and future cadence."""

    log_every_steps: int = 10
    evaluate_every_steps: int = 100
    flush_every_records: int = 1
    training_metrics_filename: str = "training_metrics.jsonl"
    evaluation_metrics_filename: str = "evaluation_metrics.jsonl"

    def validate(self) -> None:
        if self.log_every_steps <= 0:
            raise ValueError("logging.log_every_steps must be positive.")
        if self.evaluate_every_steps <= 0:
            raise ValueError("logging.evaluate_every_steps must be positive.")
        if self.flush_every_records <= 0:
            raise ValueError("logging.flush_every_records must be positive.")
        for field_name, value in (
            ("training_metrics_filename", self.training_metrics_filename),
            ("evaluation_metrics_filename", self.evaluation_metrics_filename),
        ):
            if not value.strip():
                raise ValueError(f"logging.{field_name} must be non-empty.")
            if Path(value).name != value:
                raise ValueError(f"logging.{field_name} must be a filename, not a path.")


@dataclass(frozen=True)
class CheckpointSettings:
    """Settings used by current smoke checks and future training loops."""

    enabled: bool = True
    save_every_steps: int = 500
    save_latest_pointer: bool = True
    filename_width: int = 6

    def validate(self) -> None:
        if self.save_every_steps <= 0:
            raise ValueError("checkpointing.save_every_steps must be positive.")
        if self.filename_width <= 0:
            raise ValueError("checkpointing.filename_width must be positive.")


@dataclass(frozen=True)
class ExperimentSettings:
    """Validated experiment-run settings loaded from YAML."""

    name: str
    output_dir: Path = Path("outputs")
    phase: str = "phase6"
    seed: int = 1234
    notes: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)
    logging: LoggingSettings = field(default_factory=LoggingSettings)
    checkpointing: CheckpointSettings = field(default_factory=CheckpointSettings)

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("experiment.name must be non-empty.")
        if not self.phase.strip():
            raise ValueError("experiment.phase must be non-empty.")
        if self.seed < 0:
            raise ValueError("experiment.seed must be non-negative.")
        if not str(self.output_dir).strip():
            raise ValueError("experiment.output_dir must be non-empty.")
        if any(not tag.strip() for tag in self.tags):
            raise ValueError("experiment.tags cannot contain empty values.")
        self.logging.validate()
        self.checkpointing.validate()


def experiment_settings_from_config(config: Mapping[str, Any]) -> ExperimentSettings:
    """Build and validate ``ExperimentSettings`` from a config dictionary.

    Expected structure::

        experiment:
          name: untrained_baseline
          output_dir: outputs
          phase: phase6
          seed: 1234
          notes: null
          tags: [baseline, initialization]
        logging:
          flush_every_records: 1
        checkpointing:
          enabled: true
          save_latest_pointer: true
          filename_width: 6
    """

    experiment_section = config.get("experiment")
    if not isinstance(experiment_section, Mapping):
        raise KeyError("Experiment config must contain a dictionary section named 'experiment'.")

    logging_section = config.get("logging", {})
    checkpoint_section = config.get("checkpointing", {})
    if not isinstance(logging_section, Mapping):
        raise TypeError("Config field 'logging' must be a dictionary.")
    if not isinstance(checkpoint_section, Mapping):
        raise TypeError("Config field 'checkpointing' must be a dictionary.")

    raw_tags = experiment_section.get("tags", ())
    if raw_tags is None:
        tags: tuple[str, ...] = ()
    elif isinstance(raw_tags, (list, tuple)):
        tags = tuple(str(tag) for tag in raw_tags)
    else:
        raise TypeError("experiment.tags must be a list or tuple when provided.")

    settings = ExperimentSettings(
        name=str(experiment_section.get("name", "")).strip(),
        output_dir=Path(str(experiment_section.get("output_dir", "outputs"))),
        phase=str(experiment_section.get("phase", "phase6")),
        seed=int(experiment_section.get("seed", 1234)),
        notes=None if experiment_section.get("notes") is None else str(experiment_section["notes"]),
        tags=tags,
        logging=LoggingSettings(
            log_every_steps=int(logging_section.get("log_every_steps", 10)),
            evaluate_every_steps=int(logging_section.get("evaluate_every_steps", 100)),
            flush_every_records=int(logging_section.get("flush_every_records", 1)),
            training_metrics_filename=str(
                logging_section.get("training_metrics_filename", "training_metrics.jsonl")
            ),
            evaluation_metrics_filename=str(
                logging_section.get("evaluation_metrics_filename", "evaluation_metrics.jsonl")
            ),
        ),
        checkpointing=CheckpointSettings(
            enabled=bool(checkpoint_section.get("enabled", True)),
            save_every_steps=int(checkpoint_section.get("save_every_steps", 500)),
            save_latest_pointer=bool(checkpoint_section.get("save_latest_pointer", True)),
            filename_width=int(checkpoint_section.get("filename_width", 6)),
        ),
    )
    settings.validate()
    return settings
