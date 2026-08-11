"""Tests that the migrated scripts still behave as they did before.

These drive the real command-line entry points rather than the library, because
the point of the migration is that user-facing behavior is unchanged. Each
script is exercised on the tracked tiny fixture with everything written to a
pytest temporary directory.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
TINY_DATA_CONFIG = REPO_ROOT / "configs" / "data" / "tiny_text.yaml"
TINY_MODEL_CONFIG = REPO_ROOT / "configs" / "model" / "tiny_llama.yaml"
EXPERIMENT_CONFIG = REPO_ROOT / "configs" / "experiment" / "untrained_baseline.yaml"


def load_script(name: str) -> ModuleType:
    """Import a script by path without triggering its ``__main__`` guard."""

    spec = importlib.util.spec_from_file_location(f"{name}_entry", SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_script(name: str, argv: list[str]) -> None:
    """Run a script's entry point with a controlled command line."""

    module = load_script(name)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sys, "argv", [f"{name}.py", *argv])
        module.main()


BASE_ARGS = [
    "--data-config", str(TINY_DATA_CONFIG),
    "--model-config", str(TINY_MODEL_CONFIG),
]


def test_data_pipeline_check_still_runs_on_the_tiny_fixture(capsys) -> None:
    """The Phase 3 workflow must be unchanged from a user's perspective."""

    run_script("check_data_pipeline", BASE_ARGS)
    out = capsys.readouterr().out

    assert "Phase 3 data pipeline check completed successfully." in out
    assert "Train input shape: (4, 16)" in out
    assert "Validation input shape: (4, 16)" in out
    assert "Logits shape: (4, 16, 256)" in out
    assert "data/raw/tiny_corpus.txt" in out
    # New, additive: the route the dataset came from.
    assert "Dataset resolved via: repo_fixture" in out


def test_inference_check_still_runs_on_the_tiny_fixture(capsys) -> None:
    """The Phase 4 workflow must be unchanged from a user's perspective."""

    run_script("run_inference", BASE_ARGS)
    out = capsys.readouterr().out

    assert "Phase 4 inference check completed successfully." in out
    assert "Tokenizer vocab size:" in out
    assert "Decoded generated text:" in out
    assert "Dataset resolved via: repo_fixture" in out


def test_untrained_analysis_still_runs_on_the_tiny_fixture(capsys) -> None:
    """The Phase 5 workflow must be unchanged from a user's perspective."""

    run_script("analyze_untrained_model", [*BASE_ARGS, "--num-batches", "1"])
    out = capsys.readouterr().out

    assert "Phase 5 untrained-model analysis completed successfully." in out
    assert "Mean output entropy:" in out
    assert "KL(predicted || empirical):" in out
    assert "Dataset resolved via: repo_fixture" in out


def test_offline_flag_does_not_disturb_the_local_workflow(capsys) -> None:
    """The tiny fixture never needs the network, so --offline changes nothing."""

    run_script("check_data_pipeline", [*BASE_ARGS, "--offline"])
    out = capsys.readouterr().out

    assert "Phase 3 data pipeline check completed successfully." in out
    assert "Dataset resolved via: repo_fixture" in out


def test_persisted_run_keeps_dataset_path_and_adds_provenance(tmp_path) -> None:
    """Existing metadata must survive; richer provenance is added beside it."""

    import json

    run_script(
        "analyze_untrained_model",
        [
            *BASE_ARGS,
            "--experiment-config", str(EXPERIMENT_CONFIG),
            "--num-batches", "1",
            "--persist-run",
            "--output-dir", str(tmp_path),
            "--run-id", "migration_check",
        ],
    )

    metadata = json.loads(
        (tmp_path / "untrained_baseline" / "migration_check" / "metadata.json").read_text(
            encoding="utf-8"
        )
    )

    # Unchanged contract relied on by existing tooling.
    assert metadata["dataset_path"].endswith("data/raw/tiny_corpus.txt")

    # Additive provenance.
    assert metadata["dataset"]["name"] == "tiny_local_text"
    assert metadata["dataset"]["source"] == "local_text"
    assert metadata["dataset"]["route"] == "repo_fixture"
    assert metadata["dataset"]["path"] == metadata["dataset_path"]


def test_scripts_do_not_branch_on_dataset_source() -> None:
    """Dataset knowledge must live in the resolver, not in the scripts."""

    for name in ("check_data_pipeline", "run_inference", "analyze_untrained_model"):
        source = (SCRIPTS / f"{name}.py").read_text(encoding="utf-8")
        assert "huggingface" not in source.lower()
        assert "source_type" not in source
        assert "load_dataset" not in source


def test_prepare_dataset_resolves_the_tiny_fixture(capsys) -> None:
    """The preparation command works on local data without any network."""

    run_script("prepare_dataset", ["--data-config", str(TINY_DATA_CONFIG)])
    out = capsys.readouterr().out

    assert "Dataset preparation completed successfully." in out
    assert "Resolved via: repo_fixture" in out


def test_prepare_dataset_reports_what_is_missing_when_offline(tmp_path, capsys) -> None:
    """Offline preparation must explain what it checked and how to proceed."""

    config = REPO_ROOT / "configs" / "data" / "wikitext2.yaml"

    with pytest.raises(SystemExit) as excinfo:
        run_script(
            "prepare_dataset",
            ["--data-config", str(config), "--offline", "--data-root", str(tmp_path)],
        )

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "Acquisition: disabled" in captured.out
    assert "is not available" in captured.err
    assert "offline mode is enabled" in captured.err
    assert "prepare_dataset.py" in captured.err


def test_data_pipeline_check_can_skip_the_model(capsys) -> None:
    """Dataset-only checking is useful when the model config is irrelevant."""

    run_script("check_data_pipeline", [*BASE_ARGS, "--skip-model-check"])
    out = capsys.readouterr().out

    assert "Phase 3 data pipeline check completed successfully." in out
    assert "Model check: skipped" in out
    assert "Logits shape:" not in out


def test_external_dataset_configs_are_valid() -> None:
    """The shipped external configs must parse into a usable identity."""

    import yaml

    from llm_behavior_lab.data import HUGGINGFACE_SOURCE, DatasetConfig

    for name in ("wikitext2", "tinystories"):
        path = REPO_ROOT / "configs" / "data" / f"{name}.yaml"
        dataset = DatasetConfig.from_config(yaml.safe_load(path.read_text(encoding="utf-8")))
        assert dataset.source == HUGGINGFACE_SOURCE
        assert dataset.repo_id
        assert dataset.max_characters is not None, "external configs must stay bounded"
