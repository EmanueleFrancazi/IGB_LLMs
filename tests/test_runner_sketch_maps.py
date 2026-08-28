"""``--sketch-maps`` on the experiment runner: resolution, and refusal *first*.

This is the option that makes a production ``M > 1`` measurement reachable at
all. Everything beneath it -- the map bank, the record schema, the estimator,
the artifacts, the figures -- already exists; what is new here is the one place a
person can ask for more than one map, and therefore the one place an
incompatible request can still be cheap to refuse.

Two properties are worth more than the rest.

**Omitted and ``--sketch-maps 1`` must be the same run.** Not "almost the same":
the historical single-map measurement is the reference every existing record and
the frozen CountSketch fixture were produced against, and an option that quietly
perturbed it would invalidate the comparison it exists to enable.

**An impossible combination must fail before the work, not during it.** The
estimator already refuses ``M > 1`` with the sketch off, and the Stage 8b2-c
writer guard already refuses the fidelity reconstruction above one map. Both
would fire -- after the dataset, the tokenizer, the model, and in the second case
the entire gradient measurement. At campaign scale that is hours spent arriving
at a guaranteed failure. So the tests below do not merely assert that a bad
combination raises; they make the dataset, tokenizer, model and measurement
entry points *explode if reached*, and assert the refusal beat all four.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_initialization_distribution_experiment.py"
DATA_CONFIG = REPO_ROOT / "configs" / "data" / "tiny_text.yaml"
MODEL_CONFIG = REPO_ROOT / "configs" / "model" / "tiny_llama.yaml"
EXPERIMENT_CONFIG = REPO_ROOT / "configs" / "experiment" / "initialization_distribution.yaml"


def _runner():
    """Import the runner the way every other script test imports a script."""

    spec = importlib.util.spec_from_file_location(
        "initialization_distribution_script", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _args(argv=()):
    """Parse through the runner's *real* parser, so a default cannot drift.

    Hand-building the namespace is what rotted the older helpers: a new option
    turned unrelated tests into ``AttributeError``. Parsing gives exactly the
    object ``_resolve_protocol`` is handed in production.
    """

    module = _runner()
    saved = sys.argv
    try:
        sys.argv = ["run_initialization_distribution_experiment.py", *argv]
        return module.parse_args()
    finally:
        sys.argv = saved


# -- the flag itself ----------------------------------------------------------


def test_the_flag_defaults_to_none_so_omission_is_distinguishable() -> None:
    """``None``, not 1, on the parser.

    Only ``is None`` separates "not asked for" from "asked for explicitly", and
    a parser default of 1 would make ``gradient_analysis.sketch_maps`` dead
    configuration that could never take effect -- the exact defect
    ``--sketch-dimension`` had before it was fixed.
    """

    assert _args().sketch_maps is None
    assert _args(["--sketch-maps", "1"]).sketch_maps == 1


def test_the_resolved_default_is_a_single_map() -> None:
    assert _runner()._resolve_protocol({}, _args())["sketch_maps"] == 1


def test_omitting_the_flag_and_asking_for_one_map_are_the_same_run() -> None:
    """The whole protocol, not just the map count.

    If introducing this option moved anything else, the historical measurement
    would no longer be reproducible from a command line that does not mention
    maps -- and every existing record was produced by one.
    """

    module = _runner()

    omitted = module._resolve_protocol({}, _args())
    explicit = module._resolve_protocol({}, _args(["--sketch-maps", "1"]))

    assert omitted == explicit


# -- three-level resolution ---------------------------------------------------


def test_the_experiment_config_can_request_replicas() -> None:
    config = {"gradient_analysis": {"sketch": True, "sketch_maps": 4}}

    assert _runner()._resolve_protocol(config, _args())["sketch_maps"] == 4


def test_the_command_line_overrides_the_config() -> None:
    config = {"gradient_analysis": {"sketch": True, "sketch_maps": 4}}

    protocol = _runner()._resolve_protocol(config, _args(["--sketch-maps", "2"]))

    assert protocol["sketch_maps"] == 2


def test_an_absent_flag_falls_through_rather_than_overriding_with_none() -> None:
    """The ``is None`` branch, stated directly.

    A truthiness-based chain would behave identically here and differently on
    ``--sketch-maps 0``; the pair of tests separates them.
    """

    module = _runner()
    config = {"gradient_analysis": {"sketch": True, "sketch_maps": 3}}
    namespace = argparse.Namespace(**{**vars(_args()), "sketch_maps": None})

    assert module._resolve_protocol(config, namespace)["sketch_maps"] == 3


def test_a_namespace_predating_the_option_still_resolves() -> None:
    """Read defensively, like ``sketch_dimension`` and ``initialization_scale``.

    An older caller that builds its own namespace should get the historical
    single map, not an ``AttributeError`` about an option it never had.
    """

    module = _runner()
    fields = dict(vars(_args()))
    del fields["sketch_maps"]

    assert module._resolve_protocol({}, argparse.Namespace(**fields))["sketch_maps"] == 1


# -- values that are not map counts -------------------------------------------


@pytest.mark.parametrize("value", [0, -1, 2.5, True, "4", "", object()])
def test_an_unusable_map_count_is_refused_with_the_option_named(value) -> None:
    """Including ``True``: ``True == 1`` in Python, so a boolean would otherwise
    pass as a single map and read as if a yes/no answer had been taken as a
    count. And ``"4"``: the *string* is not an integer even though it parses to
    one, so a config that quoted its value is reported rather than guessed at.
    """

    module = _runner()
    namespace = argparse.Namespace(**{**vars(_args()), "sketch_maps": value})

    with pytest.raises(ValueError, match=r"--sketch-maps must be an integer"):
        module._resolve_protocol({}, namespace)


@pytest.mark.parametrize("value", [0, -1, 2.5, True])
def test_an_unusable_map_count_in_the_config_is_refused_too(value) -> None:
    """The config is not a trusted path around the validator."""

    module = _runner()
    config = {"gradient_analysis": {"sketch": True, "sketch_maps": value}}

    with pytest.raises(ValueError, match=r"--sketch-maps must be an integer"):
        module._resolve_protocol(config, _args())


def test_the_validation_rule_is_the_measurement_s_own() -> None:
    """Not a second opinion about what a legal map count is.

    A CLI that accepted a value the estimator refuses fails hours into a run; a
    CLI that refuses one the estimator accepts cannot reach its own
    measurement. Sharing the function removes both.
    """

    from llm_behavior_lab.evaluation import position_gradients

    assert _runner()._validated_map_count is position_gradients._validated_map_count


# -- the count is never inferred ----------------------------------------------


def test_the_sketch_width_does_not_influence_the_map_count() -> None:
    """``K`` and ``M`` are independent axes and must stay that way.

    ``K`` is the width of one projection; ``M`` is how many projections there
    are. Nothing may read one from the other -- nor from an array's rank, which
    is why the record resolves the count from metadata alone.
    """

    module = _runner()

    protocol = module._resolve_protocol(
        {}, _args(["--sketch-dimension", "4096", "--gradient-sketch"])
    )

    assert protocol["sketch_dimension"] == 4096
    assert protocol["sketch_maps"] == 1


def test_the_width_resolution_is_untouched_by_the_new_option() -> None:
    """``--sketch-dimension`` keeps its own three levels and its own refusal."""

    module = _runner()
    config = {"gradient_analysis": {"sketch_dimension": 128}}

    assert module._resolve_protocol({}, _args())["sketch_dimension"] == 512
    assert module._resolve_protocol(config, _args())["sketch_dimension"] == 128
    assert (
        module._resolve_protocol(config, _args(["--sketch-dimension", "32"]))[
            "sketch_dimension"
        ]
        == 32
    )
    with pytest.raises(ValueError, match="positive number of buckets"):
        module._resolve_protocol({}, _args(["--sketch-dimension", "0"]))


# -- incompatible combinations ------------------------------------------------


def test_replicas_without_the_sketch_are_refused() -> None:
    module = _runner()

    with pytest.raises(ValueError, match="gradient sketch is off"):
        module._resolve_protocol({}, _args(["--sketch-maps", "4"]))


def test_a_single_map_without_the_sketch_stays_legal() -> None:
    """The default configuration of every historical run."""

    protocol = _runner()._resolve_protocol({}, _args())

    assert protocol["gradient_sketch"] is False
    assert protocol["sketch_maps"] == 1


def test_the_sketch_may_be_enabled_from_the_config_side() -> None:
    """The refusal reads the resolved setting, not the flag."""

    module = _runner()
    config = {"gradient_analysis": {"sketch": True}}

    assert module._resolve_protocol(config, _args(["--sketch-maps", "4"]))["sketch_maps"] == 4


def test_replicas_with_the_fidelity_sanity_check_are_refused() -> None:
    """And the message must say all four things a reader needs.

    The refusal is narrow, and stating it loosely would be worse than not
    stating it: production replicas *are* supported, it is this diagnostic's
    map-0 two-dimensional reconstruction that is not, the separate offline
    alternate-map methodology is unaffected, and the way forward is M = 1.
    """

    module = _runner()

    with pytest.raises(ValueError) as error:
        module._resolve_protocol(
            {},
            _args(
                [
                    "--sketch-maps", "4",
                    "--gradient-sketch",
                    "--countsketch-fidelity-sanity",
                ]
            ),
        )

    message = str(error.value)
    assert "--countsketch-fidelity-sanity" in message
    assert "map-0" in message
    assert "offline alternate-map" in message
    assert "--sketch-maps 1" in message


def test_the_fidelity_sanity_check_still_runs_at_one_map() -> None:
    """The diagnostic is not disabled; it is scoped."""

    protocol = _runner()._resolve_protocol(
        {}, _args(["--gradient-sketch", "--countsketch-fidelity-sanity"])
    )

    assert protocol["countsketch_fidelity_sanity"] is True
    assert protocol["sketch_maps"] == 1


def test_the_stage_8b2c_writer_guard_is_still_in_place() -> None:
    """Defence in depth: the early refusal does not replace the late one.

    The CLI check makes the failure cheap. The writer guard makes it
    *unavoidable* -- including for any caller that reaches the fidelity writer
    without passing through argument resolution at all.
    """

    source = SCRIPT.read_text(encoding="utf-8")

    assert "def _write_countsketch_fidelity" in source
    assert "fidelity_map_count = _measured_map_count(gradient_result)" in source
    assert "if fidelity_map_count != 1:" in source


# -- refusal happens before any expensive work --------------------------------


class _Detonator:
    """Raises if called at all, naming what should never have been reached."""

    def __init__(self, what: str) -> None:
        self.what = what

    def __call__(self, *args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError(
            f"{self.what} was reached; the refusal was not early enough."
        )


def _run_expecting_refusal(monkeypatch, argv, tmp_path):
    """Drive ``main()`` with every expensive entry point mined."""

    module = _runner()
    for name in (
        "resolve_dataset",
        "build_tokenizer",
        "build_model_from_config",
        "build_evaluation_positions",
        "compute_position_gradient_norms",
    ):
        monkeypatch.setattr(module, name, _Detonator(name))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_initialization_distribution_experiment.py",
            "--data-config", str(DATA_CONFIG),
            "--model-config", str(MODEL_CONFIG),
            "--experiment-config", str(EXPERIMENT_CONFIG),
            "--output-dir", str(tmp_path),
            "--offline",
            *argv,
        ],
    )
    with pytest.raises(ValueError) as error:
        module.main()
    return str(error.value)


def test_a_zero_map_count_fails_before_anything_is_loaded(monkeypatch, tmp_path) -> None:
    message = _run_expecting_refusal(monkeypatch, ["--sketch-maps", "0"], tmp_path)

    assert "--sketch-maps" in message


def test_replicas_without_the_sketch_fail_before_anything_is_loaded(
    monkeypatch, tmp_path
) -> None:
    message = _run_expecting_refusal(
        monkeypatch, ["--gradient-analysis", "--sketch-maps", "4"], tmp_path
    )

    assert "gradient sketch is off" in message


def test_replicas_with_fidelity_sanity_fail_before_the_measurement(
    monkeypatch, tmp_path
) -> None:
    """The expensive one. Without this hoist the refusal costs a full run."""

    message = _run_expecting_refusal(
        monkeypatch,
        [
            "--gradient-analysis",
            "--gradient-sketch",
            "--sketch-maps", "4",
            "--countsketch-fidelity-sanity",
        ],
        tmp_path,
    )

    assert "--countsketch-fidelity-sanity" in message


def test_nothing_was_written_by_a_refused_run(monkeypatch, tmp_path) -> None:
    """A refusal must not leave a half-created run directory behind."""

    _run_expecting_refusal(
        monkeypatch, ["--gradient-analysis", "--sketch-maps", "4"], tmp_path
    )

    assert list(tmp_path.iterdir()) == []
