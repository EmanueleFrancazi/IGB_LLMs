"""Append-only JSON Lines metric logging.

JSON Lines is used because every record is independently readable, appends do
not require loading prior history, and the files remain easy to inspect with
standard command-line or Python tools.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from llm_behavior_lab.experiment.serialization import to_jsonable, utc_now_iso


class MetricLogger:
    """Append and read step-indexed scalar metric records."""

    def __init__(self, path: str | Path, *, flush_every_records: int = 1) -> None:
        if flush_every_records <= 0:
            raise ValueError("flush_every_records must be positive.")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self.flush_every_records = flush_every_records
        self._records_since_sync = 0

    def log(
        self,
        *,
        step: int,
        metrics: Mapping[str, Any],
        stage: str,
        split: str | None = None,
        checkpoint_id: str | None = None,
        artifacts: Mapping[str, str] | None = None,
        timestamp: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append one metric record and return its normalized representation."""

        if step < 0:
            raise ValueError("step must be non-negative.")
        if not stage.strip():
            raise ValueError("stage must be non-empty.")
        if not isinstance(metrics, Mapping) or not metrics:
            raise ValueError("metrics must be a non-empty mapping.")

        normalized_metrics: dict[str, Any] = {}
        for name, value in metrics.items():
            if not str(name).strip():
                raise ValueError("Metric names must be non-empty.")
            normalized = to_jsonable(value)
            if isinstance(normalized, (list, dict)):
                raise TypeError(
                    f"Metric {name!r} is not scalar. Save arrays separately and log an artifact reference."
                )
            normalized_metrics[str(name)] = normalized

        record: dict[str, Any] = {
            "timestamp": timestamp or utc_now_iso(),
            "step": int(step),
            "stage": stage,
            "split": split,
            "checkpoint_id": checkpoint_id,
            "metrics": normalized_metrics,
            "artifacts": {} if artifacts is None else dict(artifacts),
        }
        if extra is not None:
            record["extra"] = to_jsonable(dict(extra))

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            self._records_since_sync += 1
            if self._records_since_sync >= self.flush_every_records:
                os.fsync(handle.fileno())
                self._records_since_sync = 0
        return record

    def read(self) -> list[dict[str, Any]]:
        """Read every valid record from the metric file."""

        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Malformed JSONL record in {self.path} at line {line_number}."
                    ) from exc
                if not isinstance(record, dict):
                    raise ValueError(
                        f"Metric record in {self.path} at line {line_number} is not an object."
                    )
                records.append(record)
        return records

    def iter_records(self) -> Iterable[dict[str, Any]]:
        """Yield records lazily without loading the full metric history."""

        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Malformed JSONL record in {self.path} at line {line_number}."
                    ) from exc
                if not isinstance(record, dict):
                    raise ValueError(
                        f"Metric record in {self.path} at line {line_number} is not an object."
                    )
                yield record
