"""The gradient benchmark must time the configuration it claims to time.

Two ways it could mislead, both cheap to prevent and expensive to discover from
a number that merely looks plausible:

* running with the count sketch off while a campaign runs it on, so the device
  map tables are never allocated and the memory columns understate the real
  peak;
* running a wide output head through a narrow tokenizer, so logits are truncated
  before the loss and the output-layer gradient -- the dominant cost -- is a
  fraction of the real one.

Everything here is CPU-only and needs no network.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from llm_behavior_lab.evaluation.position_gradients import (
    DEFAULT_SKETCH_DIMENSION,
    compute_position_gradient_norms,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _benchmark_module():
    """Import the benchmark the way the other script tests import scripts."""

    path = REPO_ROOT / "scripts" / "benchmark_position_gradients.py"
    spec = importlib.util.spec_from_file_location("benchmark_position_gradients", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _args(module, argv=()):
    """Parse through the script's *real* parser, so defaults cannot drift."""

    saved = sys.argv
    try:
        sys.argv = ["benchmark_position_gradients.py", *argv]
        return module.parse_args()
    finally:
        sys.argv = saved


class _Tokenizer:
    """Minimal stand-in exposing only what the vocabulary guard reads."""

    def __init__(self, vocab_size: int, kind: str = "character", identifier=None):
        self.vocab_size = vocab_size
        self._kind = kind
        self._identifier = identifier

    def describe(self):
        return {"type": self._kind, "identifier": self._identifier}


# -- sketch resolution --------------------------------------------------------


def test_the_default_invocation_leaves_sketching_disabled() -> None:
    """Backward compatibility: every historical invocation still means the same."""

    module = _benchmark_module()

    resolved = module._resolve_sketch(_args(module), {})

    assert resolved["enabled"] is False
    assert resolved["dimension"] == DEFAULT_SKETCH_DIMENSION
    assert resolved["maps"] == 1
    assert resolved["seed"] == module.DEFAULT_SKETCH_SEED


def test_the_production_seed_default_is_read_from_the_production_signature() -> None:
    """Not copied: a benchmark that hard-coded it could drift from the estimator."""

    import inspect

    module = _benchmark_module()
    production = inspect.signature(compute_position_gradient_norms)

    assert module.DEFAULT_SKETCH_SEED == production.parameters["sketch_seed"].default


def test_enabling_the_sketch_forwards_the_requested_settings() -> None:
    module = _benchmark_module()

    resolved = module._resolve_sketch(
        _args(module, ["--gradient-sketch", "--sketch-dimension", "64", "--sketch-seed", "7"]),
        {},
    )

    assert resolved == {"enabled": True, "dimension": 64, "maps": 1, "seed": 7}


def test_the_experiment_config_can_enable_and_size_the_sketch() -> None:
    """The runner's own vocabulary, not a second one invented for the benchmark."""

    module = _benchmark_module()
    config = {"gradient_analysis": {"sketch": True, "sketch_dimension": 128}}

    resolved = module._resolve_sketch(_args(module), config)

    assert resolved["enabled"] is True
    assert resolved["dimension"] == 128


def test_the_command_line_width_overrides_the_experiment_config() -> None:
    module = _benchmark_module()
    config = {"gradient_analysis": {"sketch": True, "sketch_dimension": 128}}

    resolved = module._resolve_sketch(
        _args(module, ["--sketch-dimension", "32"]), config
    )

    assert resolved["dimension"] == 32
    # The config still enables it; the flag only overrides the width.
    assert resolved["enabled"] is True


def test_more_than_one_map_is_refused_rather_than_faked() -> None:
    """M > 1 is selected but unimplemented; timing one map under that label lies."""

    module = _benchmark_module()

    with pytest.raises(ValueError, match="--sketch-maps must be 1"):
        module._resolve_sketch(_args(module, ["--sketch-maps", "4"]), {})


def test_a_non_positive_width_is_refused() -> None:
    module = _benchmark_module()

    with pytest.raises(ValueError, match="must be positive"):
        module._resolve_sketch(_args(module, ["--sketch-dimension", "0"]), {})


# -- the sketch path is really exercised --------------------------------------


def _tiny_positions(block_size=4, windows=2):
    from llm_behavior_lab.evaluation.init_distribution import build_evaluation_positions

    corpus = [(index * 5 + 1) % 16 for index in range(80)]
    return build_evaluation_positions(
        corpus, block_size=block_size, num_windows=windows
    )


def _tiny_model(vocab_size=16):
    from llm_behavior_lab.models import build_model
    from llm_behavior_lab.utils import seed_everything

    seed_everything(1000)
    return build_model(
        "llama_tiny",
        vocab_size=vocab_size,
        dim=8,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        multiple_of=4,
        max_batch_size=2,
        max_seq_len=8,
    )


def test_the_requested_width_reaches_the_production_sketch() -> None:
    """A real call, not a mock: the shapes come back from the estimator itself."""

    model = _tiny_model()
    positions = _tiny_positions()
    temperatures = (0.5, 1.0)

    result = compute_position_gradient_norms(
        model,
        positions,
        vocab_size=16,
        eligible_token_ids=None,
        num_windows=1,
        temperatures=temperatures,
        gradient_sketch=True,
        sketch_dimension=16,
        sketch_seed=4242,
    )

    assert result.temperature_gradient_sketches.shape == (2, 4, 16)
    assert result.gradient_sketches.shape == (4, 16)
    assert result.sketch_protocol["dimension"] == 16
    assert result.sketch_protocol["seed"] == 4242


def test_one_map_covers_every_parameter_exactly_once() -> None:
    """M = 1 means one bucket and one sign per parameter, and no tensor missed.

    This is what the benchmark's device-memory arithmetic rests on: map tables
    are ``M x 16 bytes x P``, so a map that covered a different number of entries
    would invalidate both the measured peak and the M > 1 projection.
    """

    model = _tiny_model()
    positions = _tiny_positions()

    result = compute_position_gradient_norms(
        model,
        positions,
        vocab_size=16,
        eligible_token_ids=None,
        num_windows=1,
        temperatures=(1.0,),
        gradient_sketch=True,
        sketch_dimension=16,
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    buckets, signs = result.sketch_map

    assert len(result.sketch_tensor_sizes) == len(trainable)
    assert sum(result.sketch_tensor_sizes) == result.parameter_count
    assert buckets.shape == (result.parameter_count,)
    assert signs.shape == (result.parameter_count,)


def test_the_benchmark_forwards_its_resolved_sketch_settings(monkeypatch) -> None:
    """The resolved values must reach the estimator, not just the header."""

    module = _benchmark_module()
    captured = {}

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return compute_position_gradient_norms(*args, **kwargs)

    monkeypatch.setattr(module, "compute_position_gradient_norms", spy)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_position_gradients.py",
            "--window-counts", "1",
            "--block-size", "4",
            "--temperatures", "1.0",
            "--offline",
            "--allow-narrow-vocabulary",
            "--gradient-sketch",
            "--sketch-dimension", "16",
            "--sketch-seed", "99",
        ],
    )
    module.main()

    assert captured["gradient_sketch"] is True
    assert captured["sketch_dimension"] == 16
    assert captured["sketch_seed"] == 99
    # The optional vector-split diagnostic stays off.
    assert captured.get("vector_split", False) is False


def test_the_no_sketch_path_still_runs_and_asks_for_no_sketch(monkeypatch) -> None:
    """The historical behaviour, unchanged, is still the default."""

    module = _benchmark_module()
    captured = {}

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return compute_position_gradient_norms(*args, **kwargs)

    monkeypatch.setattr(module, "compute_position_gradient_norms", spy)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_position_gradients.py",
            "--window-counts", "1",
            "--block-size", "4",
            "--temperatures", "1.0",
            "--offline",
            "--allow-narrow-vocabulary",
        ],
    )
    module.main()

    assert captured["gradient_sketch"] is False


# -- the vocabulary guard -----------------------------------------------------


def test_a_tokenizer_wider_than_the_model_is_refused() -> None:
    module = _benchmark_module()

    with pytest.raises(ValueError, match="exceeds model vocab size"):
        module._check_vocabulary_match(_Tokenizer(512), {"vocab_size": 256})


def test_a_pretrained_tokenizer_must_match_exactly() -> None:
    """The runner's rule, for the runner's reason."""

    module = _benchmark_module()
    tokenizer = _Tokenizer(31_997, kind="pretrained", identifier="mistralai/Mistral-7B-v0.1")

    with pytest.raises(ValueError, match="requires an exact match"):
        module._check_vocabulary_match(tokenizer, {"vocab_size": 32_000})


def test_a_matching_vocabulary_proceeds_without_a_warning() -> None:
    module = _benchmark_module()
    tokenizer = _Tokenizer(32_000, kind="pretrained", identifier="mistralai/Mistral-7B-v0.1")

    assert module._check_vocabulary_match(tokenizer, {"vocab_size": 32_000}) is None


def test_a_narrower_corpus_vocabulary_is_refused_by_default() -> None:
    """The trap the runner's own rule does not catch, now fatal here.

    A character tokenizer against a 32000-row head passes both runner
    conditions -- a corpus-derived vocabulary is allowed to be narrower -- while
    truncating the output-layer gradient to half a percent of the model. A
    benchmark number gets used as a cost estimate, so this must not merely warn.
    """

    module = _benchmark_module()

    with pytest.raises(ValueError, match="--allow-narrow-vocabulary"):
        module._check_vocabulary_match(_Tokenizer(147), {"vocab_size": 32_000})


def test_narrowing_can_be_opted_into_and_is_then_labelled() -> None:
    """Permitted only when asked for, and never silently."""

    module = _benchmark_module()

    warning = module._check_vocabulary_match(
        _Tokenizer(147), {"vocab_size": 32_000}, allow_narrow=True
    )

    assert warning is not None
    assert "--allow-narrow-vocabulary is in effect" in warning
    assert "147" in warning and "32,000" in warning
    assert "NOT representative of the full output head" in warning
    assert "must not be used as a production cost estimate" in warning


def test_the_scripts_legacy_default_pairing_must_now_acknowledge_itself() -> None:
    """Defaults pair a tiny corpus with a 256-vocabulary model: opt-in required."""

    module = _benchmark_module()

    with pytest.raises(ValueError, match="--allow-narrow-vocabulary"):
        module._check_vocabulary_match(_Tokenizer(39), {"vocab_size": 256})

    assert module._check_vocabulary_match(
        _Tokenizer(39), {"vocab_size": 256}, allow_narrow=True
    ) is not None
