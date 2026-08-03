"""Tests for Phase 6 experiment logging and checkpoint infrastructure."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from llm_behavior_lab.experiment import (
    ArrayMetricStore,
    CheckpointManager,
    ExperimentRun,
    ExperimentSettings,
    MetricLogger,
)
from llm_behavior_lab.models import build_model


def _settings(tmp_path) -> ExperimentSettings:
    return ExperimentSettings(
        name="unit_test",
        output_dir=tmp_path,
        phase="phase6_test",
        seed=1234,
    )


def _tiny_model():
    return build_model(
        "llama_tiny",
        vocab_size=32,
        dim=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        multiple_of=16,
        max_batch_size=2,
        max_seq_len=8,
    ).to("cpu")


def test_run_creation_is_collision_safe_and_snapshots_configs(tmp_path) -> None:
    """Run directories should be predictable and never silently overwritten."""

    settings = _settings(tmp_path)
    run = ExperimentRun.create(
        settings,
        run_id="fixed_run",
        metadata={"device": "cpu"},
    )
    snapshots = run.snapshot_configs(
        model_config={"model": {"name": "test"}},
        data_config={"dataset": {"name": "tiny"}},
        experiment_config={"experiment": {"name": "unit_test"}},
    )

    assert run.paths.metadata_path.exists()
    assert all(path.exists() for path in snapshots.values())
    metadata = json.loads(run.paths.metadata_path.read_text(encoding="utf-8"))
    assert metadata["run_id"] == "fixed_run"
    assert metadata["device"] == "cpu"

    with pytest.raises(FileExistsError, match="will not be overwritten"):
        ExperimentRun.create(settings, run_id="fixed_run")


def test_metric_logger_appends_and_reads_records(tmp_path) -> None:
    """JSONL logging should append without rewriting previous records."""

    logger = MetricLogger(tmp_path / "metrics.jsonl")
    first = logger.log(step=0, stage="initialization", split="train", metrics={"loss": 2.0})
    second = logger.log(
        step=1,
        stage="training",
        split="train",
        checkpoint_id="checkpoint_step_000001",
        metrics={"loss": 1.5, "learning_rate": 1e-3},
    )

    records = logger.read()
    assert records == [first, second]
    assert len((tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_metric_logger_rejects_embedded_arrays(tmp_path) -> None:
    """Array-valued metrics should be stored separately rather than in JSONL."""

    logger = MetricLogger(tmp_path / "metrics.jsonl")
    with pytest.raises(TypeError, match="Save arrays separately"):
        logger.log(step=0, stage="test", metrics={"vector": [1, 2, 3]})


def test_array_metric_store_round_trip(tmp_path) -> None:
    """NumPy archives should preserve named arrays and reject silent overwrite."""

    store = ArrayMetricStore(tmp_path / "arrays", run_dir=tmp_path)
    artifact = store.save(
        metric_name="gradient_norms",
        step=3,
        stage="evaluation",
        split="val",
        arrays={"layers": torch.arange(3), "values": np.array([1.0, 0.5, 0.25])},
    )
    loaded = store.load(artifact.path)

    assert artifact.relative_path.startswith("arrays/")
    assert np.array_equal(loaded["layers"], np.arange(3))
    assert np.allclose(loaded["values"], np.array([1.0, 0.5, 0.25]))
    with pytest.raises(FileExistsError):
        store.save(
            metric_name="gradient_norms",
            step=3,
            stage="evaluation",
            split="val",
            arrays={"values": [0.0]},
        )


def test_checkpoint_round_trip_and_latest_discovery(tmp_path) -> None:
    """Model, optimizer, scheduler, and additional state should round-trip."""

    manager = CheckpointManager(tmp_path / "checkpoints")
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

    input_ids = torch.randint(0, 32, (1, 8), device="cpu")
    targets = torch.randint(0, 32, (1, 8), device="cpu")
    loss = model(input_ids=input_ids, targets=targets).loss
    assert loss is not None
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)

    first = manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=2,
        model_config={"model": {"name": "llama_tiny"}},
        additional_state={"epoch": 1},
    )
    second = manager.save(model=model, step=5)

    assert [info.step for info in manager.list_checkpoints()] == [2, 5]
    assert manager.latest_checkpoint() == second

    restored_model = _tiny_model()
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=5e-4)
    restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, step_size=2)
    payload = manager.restore(
        model=restored_model,
        checkpoint=first,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        map_location="cpu",
    )

    assert payload["global_step"] == 2
    assert payload["additional_state"] == {"epoch": 1}
    for name, value in model.state_dict().items():
        assert torch.equal(value, restored_model.state_dict()[name])
    assert restored_optimizer.state_dict()["param_groups"][0]["lr"] == optimizer.state_dict()["param_groups"][0]["lr"]
    assert restored_scheduler.state_dict()["last_epoch"] == scheduler.state_dict()["last_epoch"]


def test_checkpoint_load_uses_map_location_cpu(tmp_path) -> None:
    """Checkpoint loading should support a portable CPU map location."""

    manager = CheckpointManager(tmp_path / "checkpoints")
    model = _tiny_model()
    info = manager.save(model=model, step=0)
    payload = manager.load_payload(info, map_location=torch.device("cpu"))

    assert payload["global_step"] == 0
    assert all(tensor.device.type == "cpu" for tensor in payload["model_state_dict"].values())


def test_checkpoint_errors_are_informative(tmp_path) -> None:
    """Missing and malformed checkpoints should fail with clear messages."""

    manager = CheckpointManager(tmp_path / "checkpoints")
    with pytest.raises(FileNotFoundError, match="No checkpoints found"):
        manager.load_payload()

    malformed = manager.checkpoint_dir / "checkpoint_step_000001.pt"
    torch.save({"global_step": 1}, malformed)
    with pytest.raises(ValueError, match="missing fields"):
        manager.load_payload(malformed)


def test_analysis_json_accepts_existing_dataclass_style_payload(tmp_path) -> None:
    """Experiment runs should save nested analysis dictionaries cleanly."""

    run = ExperimentRun.create(_settings(tmp_path), run_id="analysis")
    path = run.save_analysis_json(
        "untrained_analysis.json",
        {
            "output_summary": {"mean_entropy": 3.2},
            "top1_concentration": 0.25,
            "layer_values": torch.tensor([1.0, 0.5]),
        },
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["layer_values"] == [1.0, 0.5]


def test_settings_can_be_overridden_without_mutation(tmp_path) -> None:
    """Dataclass replacement should support script-level output overrides."""

    settings = _settings(tmp_path)
    changed = replace(settings, output_dir=tmp_path / "other", seed=99)
    assert settings.output_dir == tmp_path
    assert changed.output_dir == tmp_path / "other"
    assert changed.seed == 99
