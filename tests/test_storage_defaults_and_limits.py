"""The production default, and the provenance of every operational limit.

Two claims that were being made without evidence.

**"``metrics_only`` is the production default."** Every other storage test names
its mode explicitly, so the default path -- the one an ordinary run takes -- was
never exercised. A default that is only asserted in a comment is not a default
anyone can rely on, and the failure mode is silent: a run would keep 3.5 GiB of
rows and look entirely normal.

**"The resolved limits are recorded."** A limit that gates a run and then
vanishes leaves a record that cannot say what admitted it. That matters most for
the two caps a reader would want to check: what ceiling a ``compact_factors``
selection was measured against, and what cap let a ``per_position`` run keep its
rows.

The configuration model these pin, which is deliberate rather than accidental:

* the storage **mode** is protocol configuration -- it may come from the
  experiment config, be overridden on the command line, or fall to the
  production default;
* the resource **limits** are operational controls and live on the command line
  only;
* whichever way they were resolved, the values are persisted for provenance.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

yaml = pytest.importorskip("yaml")
pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "run_initialization_distribution_experiment.py"


def _runner():
    spec = importlib.util.spec_from_file_location("defaults_runner", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _configs(directory: Path) -> tuple[Path, Path]:
    """A data config and an experiment config that name **no** storage mode.

    The experiment config is copied and its ``gradient_analysis`` block left
    without ``sketch_storage``, so the run has nothing to fall back on but the
    production default.
    """

    data = yaml.safe_load(
        (REPO / "configs" / "data" / "tiny_text.yaml").read_text(encoding="utf-8")
    )
    data.setdefault("runtime", {})["device"] = "cpu"
    data_path = directory / "data.yaml"
    data_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    experiment = yaml.safe_load(
        (
            REPO / "configs" / "experiment" / "initialization_distribution.yaml"
        ).read_text(encoding="utf-8")
    )
    gradients = experiment.setdefault("gradient_analysis", {})
    gradients.pop("sketch_storage", None)
    assert "sketch_storage" not in gradients
    experiment_path = directory / "experiment.yaml"
    experiment_path.write_text(yaml.safe_dump(experiment), encoding="utf-8")
    return data_path, experiment_path


def _run(directory: Path, run_id: str, extra: tuple[str, ...] = ()) -> Path:
    module = _runner()
    data, experiment = _configs(directory)
    argv = [
        "run_initialization_distribution_experiment.py",
        "--data-config", str(data),
        "--experiment-config", str(experiment),
        "--model-config", str(REPO / "configs" / "model" / "tiny_llama.yaml"),
        "--num-initializations", "1", "--num-windows", "6", "--block-size", "8",
        "--num-replicates", "1", "--no-uniform-null", "--no-input-structure",
        "--gradient-analysis", "--gradient-sketch",
        "--sketch-maps", "2", "--sketch-dimension", "8",
        "--gradient-temperatures", "0.6", "1.0",
        "--no-figures", "--offline",
        "--output-dir", str(directory / "out"), "--run-id", run_id,
        *extra,
    ]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sys, "argv", argv)
        module.main()
    return directory / "out" / "initialization_distribution" / run_id / "analyses"


def _block(analyses: Path) -> dict:
    payload = json.loads(
        (analyses / "initialization_distribution.json").read_text(encoding="utf-8")
    )
    return payload["analysis"]["gradient_analysis"]["gradient_alignment"]


# -- the production default, exercised rather than asserted -------------------


@pytest.fixture(scope="module")
def default_run(tmp_path_factory) -> Path:
    """A run with **no** ``--sketch-storage`` and no config entry for it."""

    return _run(tmp_path_factory.mktemp("default"), "default")


def test_omitting_the_flag_yields_a_metrics_only_record(default_run) -> None:
    from llm_behavior_lab.analysis.alignment_metrics import load_finalized_record

    record = load_finalized_record(default_run)
    assert record.record_version == 13
    assert record.storage_mode == "metrics_only"


def test_the_default_record_is_physically_metrics_only(default_run) -> None:
    """Checked on the archive's own key list.

    An accessor can report a field missing for a dozen reasons; only the key
    list says the bytes are not there -- which is the whole point of the default.
    """

    with np.load(
        default_run / "initialization_distribution.npz", allow_pickle=False
    ) as archive:
        keys = list(archive.files)

    assert "gradient_position_sketches" not in keys
    assert "gradient_temperature_position_sketches" not in keys
    assert [key for key in keys if "sketch" in key] == []
    assert [key for key in keys if key.startswith("factor")] == []


def test_the_default_record_still_carries_its_finalized_metrics(default_run) -> None:
    """Dropping the rows is only acceptable because the results survive."""

    from llm_behavior_lab.analysis.alignment_metrics import load_finalized_record

    record = load_finalized_record(default_run)
    assert record.has_finalized_gradient_metrics
    assert not record.has_gradient_position_sketches
    assert not record.has_gradient_sketch_factors

    metrics = record.alignment_metrics
    for name in ("reference_target_delta", "nucleus_delta", "cross_delta_cross"):
        assert name in metrics, name


# -- the configuration model -------------------------------------------------


def _protocol(experiment_config: dict, argv: tuple[str, ...] = ()) -> dict:
    """Resolve the protocol as `main` does, from a config plus a command line.

    ``parse_args`` reads ``sys.argv`` rather than taking a list, so the command
    line is supplied the same way the runner receives one.
    """

    module = _runner()
    full = [
        "run_initialization_distribution_experiment.py",
        "--data-config", "unused", "--model-config", "unused",
        *argv,
    ]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sys, "argv", full)
        args = module.parse_args()
    return module._resolve_protocol(experiment_config, args)


def test_the_mode_may_come_from_the_experiment_config() -> None:
    """Storage mode is protocol configuration, so a config may set it."""

    protocol = _protocol({"gradient_analysis": {"sketch_storage": "per_position"}})
    assert protocol["sketch_storage"] == "per_position"


def test_the_command_line_overrides_the_config() -> None:
    protocol = _protocol(
        {"gradient_analysis": {"sketch_storage": "per_position"}},
        ("--sketch-storage", "metrics_only"),
    )
    assert protocol["sketch_storage"] == "metrics_only"


def test_the_production_default_applies_when_neither_names_a_mode() -> None:
    assert _protocol({})["sketch_storage"] == "metrics_only"
    assert _protocol({"gradient_analysis": {}})["sketch_storage"] == "metrics_only"


def test_limits_are_command_line_only() -> None:
    """Resource limits are operational controls, not protocol.

    A config entry for one must not silently take effect -- an operator reading
    the command line has to be able to see every ceiling that applied.
    """

    protocol = _protocol({
        "gradient_analysis": {
            "sketch_storage_max_bytes": 7,
            "sketch_factors_max_bytes": 7,
            "gradient_temp_max_bytes": 7,
        }
    })
    assert protocol["sketch_storage_max_bytes"] == 512 * 1024 * 1024
    assert protocol["sketch_factors_max_bytes"] == 512 * 1024 * 1024
    assert protocol["gradient_temp_max_bytes"] == 8 * 1024 ** 3


# -- provenance of the resolved limits ---------------------------------------


def test_the_resolved_caps_are_persisted(default_run) -> None:
    """One authoritative block, so a record can say what admitted it."""

    limits = _block(default_run)["storage_limits"]
    assert limits["sketch_storage_max_bytes"] == 512 * 1024 * 1024
    assert limits["sketch_factors_max_bytes"] == 512 * 1024 * 1024


def test_the_temp_byte_limit_is_not_duplicated(default_run) -> None:
    """It already has an owner in ``temporary_store.preflight``.

    Two copies of one fact is one too many: they drift, and then neither can be
    trusted.
    """

    block = _block(default_run)
    assert "gradient_temp_max_bytes" not in block["storage_limits"]
    assert block["temporary_store"]["preflight"]["max_bytes"] == 8 * 1024 ** 3


def test_the_temp_directory_is_recorded_as_policy_not_as_a_path(
    default_run,
) -> None:
    """A completed record must stay portable.

    The absolute location may be node-local scratch that no longer resolves
    anywhere; it belongs in the live manifest, where it is operationally
    necessary, not in the published artifact.
    """

    limits = _block(default_run)["storage_limits"]
    assert limits["temp_dir_policy"] == "output_root_gradient_tmp"
    assert limits["temp_dir_overridden"] is False

    blob = json.dumps(_block(default_run))
    assert "/tmp/" not in blob
    assert "_gradient_tmp/" not in blob


def test_an_explicit_override_is_recorded_as_such(tmp_path) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    analyses = _run(
        tmp_path, "override", extra=("--gradient-temp-dir", str(scratch))
    )
    limits = _block(analyses)["storage_limits"]
    assert limits["temp_dir_policy"] == "explicit_override"
    assert limits["temp_dir_overridden"] is True
    assert str(scratch) not in json.dumps(_block(analyses))


# -- round trip, both modes that own a cap ------------------------------------


def test_a_compact_factors_record_round_trips_its_cap(tmp_path) -> None:
    analyses = _run(
        tmp_path, "cf",
        extra=(
            "--sketch-storage", "compact_factors",
            "--sketch-factors", "target@1.0,greedy@1.0",
            "--sketch-factors-max-bytes", "33554432",
        ),
    )
    from llm_behavior_lab.analysis.alignment_metrics import load_finalized_record

    record = load_finalized_record(analyses)
    assert record.storage_mode == "compact_factors"
    assert record.has_gradient_sketch_factors
    assert _block(analyses)["storage_limits"]["sketch_factors_max_bytes"] == 33554432


def test_a_per_position_record_round_trips_its_cap(tmp_path) -> None:
    analyses = _run(
        tmp_path, "pp",
        extra=(
            "--sketch-storage", "per_position",
            "--sketch-storage-max-bytes", "16777216",
        ),
    )
    from llm_behavior_lab.analysis.alignment_metrics import load_finalized_record

    record = load_finalized_record(analyses)
    assert record.storage_mode == "per_position"
    assert record.has_gradient_position_sketches
    assert _block(analyses)["storage_limits"]["sketch_storage_max_bytes"] == 16777216
