"""End-to-end tests for the persisted initialization-analysis workflow.

``scripts/analyze_untrained_model.py --persist-run`` is the integration point
where the Phase 5 analysis meets the Phase 6 persistence interfaces. The script
entry point is driven directly so the wiring between analysis results, array
artifacts, scalar metric records, configuration snapshots, and the step-zero
checkpoint is verified as a whole instead of being reimplemented here.

Every write goes to a pytest temporary directory. The data configuration is
loaded from the tracked tiny-corpus config and pinned to CPU, so the workflow
never selects an accelerator, never reaches the network, and never leaves
artifacts inside the repository.
"""

from __future__ import annotations

import importlib.util
import json
import math
import random
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import torch
import yaml

from llm_behavior_lab.experiment import ArrayMetricStore, CheckpointManager
from llm_behavior_lab.models import build_model_from_config

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "analyze_untrained_model.py"
DATA_CONFIG_PATH = REPO_ROOT / "configs" / "data" / "tiny_text.yaml"
MODEL_CONFIG_PATH = REPO_ROOT / "configs" / "model" / "tiny_llama.yaml"
EXPERIMENT_CONFIG_PATH = REPO_ROOT / "configs" / "experiment" / "untrained_baseline.yaml"

RUN_ID = "integration_test_run"
NUM_BATCHES = 1
TOP_K = 3
MAX_EXAMPLES = 1
GRAD_NORM_EPS = 1e-12
CHECKPOINT_ID = "checkpoint_step_000000"


def _load_script_module() -> ModuleType:
    """Import the analysis script by path without triggering its ``__main__`` guard."""

    spec = importlib.util.spec_from_file_location("analyze_untrained_model_entry", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load the analysis script: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cpu_data_config(destination_dir: Path) -> Path:
    """Copy the tracked tiny-corpus data config and pin the runtime device to CPU."""

    config = yaml.safe_load(DATA_CONFIG_PATH.read_text(encoding="utf-8"))
    config["runtime"]["device"] = "cpu"
    destination = destination_dir / "tiny_text_cpu.yaml"
    destination.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return destination


def _script_argv(*, data_config: Path, output_root: Path, run_id: str) -> list[str]:
    """Build the command line for one persisted analysis run."""

    return [
        "analyze_untrained_model.py",
        "--data-config", str(data_config),
        "--model-config", str(MODEL_CONFIG_PATH),
        "--experiment-config", str(EXPERIMENT_CONFIG_PATH),
        "--split", "train",
        "--num-batches", str(NUM_BATCHES),
        "--top-k", str(TOP_K),
        "--max-examples", str(MAX_EXAMPLES),
        "--compute-grad-norms",
        "--grad-norm-num-batches", "1",
        "--grad-norm-eps", repr(GRAD_NORM_EPS),
        "--persist-run",
        "--run-id", run_id,
        "--output-dir", str(output_root),
    ]


def _capture_global_rng_state() -> dict[str, Any]:
    """Capture global RNG state without initializing CUDA.

    These helpers are intentionally local rather than imported from
    ``llm_behavior_lab.experiment``: the fixture's isolation guarantee should not
    depend on the same code the suite exercises.
    """

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    # ``is_initialized`` never creates a CUDA context, unlike ``is_available``.
    if torch.cuda.is_initialized():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_global_rng_state(state: dict[str, Any]) -> None:
    """Restore every RNG stream captured by :func:`_capture_global_rng_state`."""

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_initialized():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _invoke_workflow(*, data_config: Path, output_root: Path, run_id: str) -> None:
    """Run the analysis script entry point with a controlled command line."""

    module = _load_script_module()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sys, "argv", _script_argv(
            data_config=data_config,
            output_root=output_root,
            run_id=run_id,
        ))
        module.main()


@pytest.fixture(scope="module")
def persisted_run(tmp_path_factory):
    """Execute the persisted workflow once and expose the resulting locations.

    The workflow calls the project's seeding helper, which reseeds the global
    Python, NumPy, and PyTorch generators. The pre-test state is captured before
    the run and restored in ``finally`` so neither a failed setup nor a failed
    test can leak seeded state into the rest of the session.
    """

    rng_state = _capture_global_rng_state()
    try:
        workspace = tmp_path_factory.mktemp("persisted_analysis")
        output_root = workspace / "outputs"
        data_config = _cpu_data_config(workspace)

        repository_outputs = REPO_ROOT / "outputs"
        outputs_existed_before = repository_outputs.exists()

        _invoke_workflow(data_config=data_config, output_root=output_root, run_id=RUN_ID)

        yield {
            "data_config": data_config,
            "output_root": output_root,
            "run_dir": output_root / "untrained_baseline" / RUN_ID,
            "outputs_existed_before": outputs_existed_before,
        }
    finally:
        _restore_global_rng_state(rng_state)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read every record from a JSON Lines metric file."""

    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_metadata(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))


def test_persisted_run_creates_expected_directory_structure(persisted_run) -> None:
    """The run directory should follow the documented Phase 6 layout."""

    run_dir = persisted_run["run_dir"]

    assert run_dir.is_dir()
    for relative in (
        "metadata.json",
        "config/model_config.yaml",
        "config/data_config.yaml",
        "config/experiment_config.yaml",
        "metrics/evaluation_metrics.jsonl",
        "metrics/array_metrics",
        "checkpoints",
        "checkpoints/latest.json",
        f"checkpoints/{CHECKPOINT_ID}.pt",
        "analyses/untrained_analysis.json",
        "analyses/gradient_norm_analysis.json",
        "logs",
    ):
        assert (run_dir / relative).exists(), f"missing run artifact: {relative}"

    # The training log is created empty because no training step runs here.
    assert (run_dir / "metrics" / "training_metrics.jsonl").read_text(encoding="utf-8") == ""


def test_persisted_run_metadata_records_reproducibility_context(persisted_run) -> None:
    """Metadata should capture the experiment, device, model, and analysis settings."""

    metadata = _load_metadata(persisted_run["run_dir"])
    experiment_config = yaml.safe_load(EXPERIMENT_CONFIG_PATH.read_text(encoding="utf-8"))

    assert metadata["experiment_name"] == "untrained_baseline"
    assert metadata["run_id"] == RUN_ID
    assert metadata["phase"] == experiment_config["experiment"]["phase"]
    assert metadata["seed"] == experiment_config["experiment"]["seed"]
    assert metadata["tags"] == experiment_config["experiment"]["tags"]
    assert metadata["experiment_type"] == "untrained_model_analysis"

    # The workflow must stay on CPU for this test.
    assert metadata["device"] == "cpu"

    model_config = yaml.safe_load(MODEL_CONFIG_PATH.read_text(encoding="utf-8"))
    reference_model = build_model_from_config(model_config).to("cpu")
    assert metadata["model_name"] == model_config["model"]["name"]
    assert metadata["model_parameter_count"] == reference_model.count_parameters()

    assert metadata["dataset_path"].endswith("data/raw/tiny_corpus.txt")
    assert metadata["tokenizer"]["type"] == "char"
    assert metadata["tokenizer"]["vocab_size"] > 0

    assert metadata["analysis"] == {
        "split": "train",
        "num_batches": NUM_BATCHES,
        "top_k": TOP_K,
        "max_examples": MAX_EXAMPLES,
        "compute_grad_norms": True,
    }

    environment = metadata["environment"]
    for key in ("python_version", "platform", "torch_version", "numpy_version", "process_id"):
        assert key in environment
    # Git may be unavailable; when present the commit must be a full SHA.
    assert metadata["git_commit"] is None or len(metadata["git_commit"]) == 40


def test_persisted_run_snapshots_the_configs_it_used(persisted_run) -> None:
    """Config snapshots should reproduce the exact inputs of the run."""

    config_dir = persisted_run["run_dir"] / "config"

    saved_model = yaml.safe_load((config_dir / "model_config.yaml").read_text(encoding="utf-8"))
    saved_data = yaml.safe_load((config_dir / "data_config.yaml").read_text(encoding="utf-8"))
    saved_experiment = yaml.safe_load(
        (config_dir / "experiment_config.yaml").read_text(encoding="utf-8")
    )

    assert saved_model == yaml.safe_load(MODEL_CONFIG_PATH.read_text(encoding="utf-8"))
    assert saved_experiment == yaml.safe_load(EXPERIMENT_CONFIG_PATH.read_text(encoding="utf-8"))
    assert saved_data == yaml.safe_load(
        persisted_run["data_config"].read_text(encoding="utf-8")
    )
    # The snapshot must record the tracked dataset selection and the pinned device.
    tracked_data = yaml.safe_load(DATA_CONFIG_PATH.read_text(encoding="utf-8"))
    assert saved_data["dataset"] == tracked_data["dataset"]
    assert saved_data["batching"] == tracked_data["batching"]
    assert saved_data["runtime"]["device"] == "cpu"


def test_persisted_run_logs_scalar_evaluation_metrics(persisted_run) -> None:
    """Both evaluation records should be scalar-only and reference existing artifacts."""

    run_dir = persisted_run["run_dir"]
    records = _read_jsonl(run_dir / "metrics" / "evaluation_metrics.jsonl")

    assert len(records) == 2
    analysis_record, gradient_record = records

    for record in records:
        assert record["step"] == 0
        assert record["split"] == "train"
        assert record["checkpoint_id"] == CHECKPOINT_ID
        assert record["timestamp"]
        for name, value in record["metrics"].items():
            assert not isinstance(value, (list, dict, str)), f"metric {name} is not scalar"
        for artifact_path in record["artifacts"].values():
            assert (run_dir / artifact_path).exists()

    assert analysis_record["stage"] == "initialization"
    assert set(analysis_record["metrics"]) == {
        "mean_output_entropy",
        "min_output_entropy",
        "max_output_entropy",
        "mean_top1_probability",
        "mean_topk_probability_mass",
        "top1_concentration",
        "kl_predicted_to_empirical",
        "js_predicted_empirical",
        "analyzed_positions",
    }

    metadata = _load_metadata(run_dir)
    vocab_size = metadata["tokenizer"]["vocab_size"]
    metrics = analysis_record["metrics"]
    assert metrics["analyzed_positions"] > 0
    assert 0.0 <= metrics["mean_output_entropy"] <= math.log(vocab_size) + 1e-6
    assert metrics["min_output_entropy"] <= metrics["mean_output_entropy"]
    assert metrics["mean_output_entropy"] <= metrics["max_output_entropy"]
    assert 0.0 < metrics["mean_top1_probability"] <= 1.0
    assert metrics["mean_top1_probability"] <= metrics["mean_topk_probability_mass"] <= 1.0
    assert 0.0 < metrics["top1_concentration"] <= 1.0
    assert metrics["kl_predicted_to_empirical"] >= 0.0
    assert metrics["js_predicted_empirical"] >= 0.0

    assert gradient_record["stage"] == "initialization_gradient_diagnostic"
    assert set(gradient_record["metrics"]) == {
        "gradient_diagnostic_mean_loss",
        "gradient_log_slope",
        "gradient_log_intercept",
        "gradient_slope_standard_error",
        "gradient_r_squared",
    }
    assert gradient_record["metrics"]["gradient_diagnostic_mean_loss"] > 0.0
    # The tiny model has two decoder blocks, so the fit has no residual degrees
    # of freedom and the standard error is intentionally unavailable.
    assert gradient_record["metrics"]["gradient_slope_standard_error"] is None


def test_persisted_run_saves_output_distribution_arrays(persisted_run) -> None:
    """Distribution arrays should be complete, aligned, and consistent with the metrics."""

    run_dir = persisted_run["run_dir"]
    records = _read_jsonl(run_dir / "metrics" / "evaluation_metrics.jsonl")
    artifact_path = run_dir / records[0]["artifacts"]["output_distributions"]

    assert artifact_path.name.startswith("untrained_output_distributions__initialization__train__step_000000")
    arrays = ArrayMetricStore.load(artifact_path)

    assert set(arrays) == {
        "mean_predicted_probabilities",
        "empirical_token_frequencies",
        "probability_frequency_gaps",
        "top1_assignment_counts",
    }

    vocab_size = _load_metadata(run_dir)["tokenizer"]["vocab_size"]
    for name, array in arrays.items():
        assert array.shape == (vocab_size,), f"{name} has shape {array.shape}"

    assert math.isclose(float(arrays["mean_predicted_probabilities"].sum()), 1.0, rel_tol=1e-5)
    assert math.isclose(float(arrays["empirical_token_frequencies"].sum()), 1.0, rel_tol=1e-5)
    np.testing.assert_allclose(
        arrays["probability_frequency_gaps"],
        arrays["mean_predicted_probabilities"] - arrays["empirical_token_frequencies"],
        rtol=1e-6,
        atol=1e-8,
    )

    analyzed_positions = records[0]["metrics"]["analyzed_positions"]
    assert int(arrays["top1_assignment_counts"].sum()) == analyzed_positions


def test_persisted_run_saves_per_layer_gradient_arrays(persisted_run) -> None:
    """Gradient arrays should cover every decoder block and match the log transform."""

    run_dir = persisted_run["run_dir"]
    records = _read_jsonl(run_dir / "metrics" / "evaluation_metrics.jsonl")
    artifact_path = run_dir / records[1]["artifacts"]["per_layer_gradient_norms"]

    assert CHECKPOINT_ID in artifact_path.name
    arrays = ArrayMetricStore.load(artifact_path)
    assert set(arrays) == {"layer_indices", "squared_l2_norms", "log_squared_l2_norms"}

    n_layers = int(
        yaml.safe_load(MODEL_CONFIG_PATH.read_text(encoding="utf-8"))["model"]["params"]["n_layers"]
    )
    assert arrays["layer_indices"].tolist() == list(range(n_layers))
    for name in ("squared_l2_norms", "log_squared_l2_norms"):
        assert arrays[name].shape == (n_layers,)
        assert np.all(np.isfinite(arrays[name]))
    assert np.all(arrays["squared_l2_norms"] > 0.0)
    # GRAD_NORM_EPS is the value this test passes on the command line, not an
    # argparse default, so the log transform is checked against a known input.
    np.testing.assert_allclose(
        arrays["log_squared_l2_norms"],
        np.log(arrays["squared_l2_norms"] + GRAD_NORM_EPS),
        rtol=1e-9,
        atol=1e-9,
    )


def test_persisted_run_saves_structured_analysis_artifacts(persisted_run) -> None:
    """Analysis JSON files should serialize the full nested analysis results."""

    run_dir = persisted_run["run_dir"]
    records = _read_jsonl(run_dir / "metrics" / "evaluation_metrics.jsonl")

    analysis = json.loads(
        (run_dir / "analyses" / "untrained_analysis.json").read_text(encoding="utf-8")
    )
    assert analysis["checkpoint_id"] == CHECKPOINT_ID
    assert analysis["array_artifact"] == records[0]["artifacts"]["output_distributions"]
    assert analysis["output_summary"]["num_positions"] == records[0]["metrics"]["analyzed_positions"]
    assert len(analysis["topk_examples"]) == MAX_EXAMPLES
    for example in analysis["topk_examples"]:
        assert len(example["predictions"]) == TOP_K
        for prediction in example["predictions"]:
            assert set(prediction) == {"token_id", "token", "probability"}

    gradients = json.loads(
        (run_dir / "analyses" / "gradient_norm_analysis.json").read_text(encoding="utf-8")
    )
    assert gradients["definition"] == "squared_l2_norm_of_gradient_wrt_decoder_block_output"
    assert gradients["num_batches"] == 1
    assert gradients["metadata"]["checkpoint_id"] == CHECKPOINT_ID
    assert gradients["metadata"]["array_artifact"] == records[1]["artifacts"]["per_layer_gradient_norms"]
    assert len(gradients["per_layer"]) == len(gradients["layer_indices"])
    assert gradients["trend_fit"]["slope_standard_error"] is None


def test_persisted_run_checkpoint_restores_into_a_compatible_model(persisted_run) -> None:
    """The step-zero checkpoint should restore exactly into a freshly built model."""

    run_dir = persisted_run["run_dir"]
    model_config = yaml.safe_load(MODEL_CONFIG_PATH.read_text(encoding="utf-8"))

    latest_pointer = json.loads((run_dir / "checkpoints" / "latest.json").read_text(encoding="utf-8"))
    assert latest_pointer["checkpoint_id"] == CHECKPOINT_ID
    assert latest_pointer["global_step"] == 0

    manager = CheckpointManager(run_dir / "checkpoints")
    latest = manager.latest_checkpoint()
    assert latest is not None
    assert latest.step == 0
    assert latest.path.name == f"{CHECKPOINT_ID}.pt"

    restored_model = build_model_from_config(model_config).to("cpu")
    payload = manager.restore(model=restored_model, checkpoint=latest, map_location="cpu")

    assert payload["format_version"] == 1
    assert payload["global_step"] == 0
    assert payload["model_config"] == model_config
    assert payload["optimizer_state_dict"] is None
    assert payload["rng_state"] is not None
    assert payload["additional_state"]["stage"] == "initialization"
    assert payload["additional_state"]["analysis_split"] == "train"
    assert (
        payload["additional_state"]["tokenizer_vocab_size"]
        == _load_metadata(run_dir)["tokenizer"]["vocab_size"]
    )

    # The saved parameter set must match a model built independently from the
    # snapshot config, and every tensor must survive the round trip unchanged.
    saved_state = payload["model_state_dict"]
    restored_state = restored_model.state_dict()
    assert saved_state.keys() == restored_state.keys()
    for name, tensor in saved_state.items():
        assert torch.equal(tensor.cpu(), restored_state[name].cpu()), f"parameter mismatch: {name}"


def test_persisted_run_refuses_to_overwrite_an_existing_run(persisted_run) -> None:
    """Reusing an explicit run ID must fail instead of overwriting saved results."""

    with pytest.raises(FileExistsError, match="will not be overwritten"):
        _invoke_workflow(
            data_config=persisted_run["data_config"],
            output_root=persisted_run["output_root"],
            run_id=RUN_ID,
        )


def test_persisted_run_leaves_no_artifacts_in_the_repository(persisted_run) -> None:
    """Persisted runs must not write into the working tree, including legacy paths."""

    repository_outputs = REPO_ROOT / "outputs"
    assert repository_outputs.exists() == persisted_run["outputs_existed_before"]
    assert not (repository_outputs / "phase5_gradient_norms").exists()
