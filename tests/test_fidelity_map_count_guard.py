"""The fidelity sanity check refuses production replicas, and says why.

``_write_countsketch_fidelity`` reconstructs the production sketch from retained
exact gradients to prove the map, the flattening order and the position alignment
all agree. That reconstruction is **map-0 and strictly two-dimensional**: one
``(buckets, signs)`` pair against a ``[positions, K]`` block. A multi-map
measurement produces ``[positions, M, K]``, and every line of it would be wrong.

So it refuses. The refusal has to happen *before* the reconstruction and before
anything is written, or a rejected run could still leave a half-built artifact
that a later reader would take at face value.

Two things this guard is emphatically **not** about:

* it does not touch the valid ``M = 1`` path, which is what production runs today;
* it does not restrict the *offline alternate-map methodology* in
  :mod:`llm_behavior_lab.analysis.countsketch_fidelity`, which re-projects
  retained gradients through independent maps to estimate how far a reported
  error would move. That remains valid and is a different question entirely.

The guard is unreachable through today's CLI -- the runner has no
``--sketch-maps`` -- and exists so it is already in place when Stage 8c makes
``M > 1`` reachable.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def _runner():
    """Load the runner by path; it is a script, not an installed module."""

    path = REPO_ROOT / "scripts" / "run_initialization_distribution_experiment.py"
    spec = importlib.util.spec_from_file_location("_fidelity_guard_runner", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _result(map_count, *, include_key=True):
    """A gradient result whose protocol is the only thing the guard reads.

    Everything after the guard is left deliberately unusable -- ``None`` where
    real arrays would be -- so a test that reaches the reconstruction fails
    loudly rather than quietly exercising it.
    """

    protocol = {"dimension": 4, "seed": 20240917}
    if include_key:
        protocol["map_count"] = map_count
    return SimpleNamespace(sketch_protocol=protocol, exact_gradients=None)


def test_a_multi_map_result_is_no_longer_refused(tmp_path) -> None:
    """The M = 1 restriction has been deliberately lifted.

    It existed because the check re-projected retained gradients through map 0
    and assumed two-dimensional ``[positions, K]`` sketches, which is not what a
    multi-map measurement produces. The check now reads the sketches every
    production map actually produced, captured live during the measurement, so
    there is no reconstruction left to be map-0 about -- and validating map 0
    alone would have validated an instrument the figures do not use.

    Asserted as the *absence* of the old refusal: reaching the null-object's
    missing attribute proves the guard let the result through, exactly as the
    single-map case has always been checked.
    """

    module = _runner()
    run = SimpleNamespace(paths=SimpleNamespace(run_dir=tmp_path))
    with pytest.raises(AttributeError):
        module._write_countsketch_fidelity(run, _result(4), {})


def test_the_multi_map_path_reports_its_map_count(tmp_path) -> None:
    """The count comes from the protocol, never inferred from an array's rank."""

    module = _runner()
    assert module._measured_map_count(_result(4)) == 4
    assert module._measured_map_count(_result(1)) == 1


@pytest.mark.parametrize(
    "result",
    [
        pytest.param(_result(1), id="explicit-map-count-1"),
        pytest.param(_result(1, include_key=False), id="legacy-no-map-count-key"),
    ],
)
def test_a_single_map_result_passes_the_guard(result, tmp_path) -> None:
    """Both the explicit and the historical shapes get through it.

    A record written before the map count existed was single-map by
    construction, so a missing key is read as one rather than refused -- older
    in-memory results and tests must keep working unchanged.

    The guard is all that is under test here: execution then continues into the
    reconstruction and fails on the deliberately absent gradients, which is
    exactly how we know the guard let it past.
    """

    runner = _runner()

    with pytest.raises(AttributeError):
        runner._write_countsketch_fidelity(
            SimpleNamespace(paths=SimpleNamespace(run_dir=tmp_path)),
            result,
            {"sketch_dimension": 4},
        )


def test_a_missing_protocol_entirely_is_read_as_one_map(tmp_path) -> None:
    """``sketch_protocol`` is ``None`` when sketching was off."""

    runner = _runner()

    with pytest.raises(AttributeError):
        runner._write_countsketch_fidelity(
            SimpleNamespace(paths=SimpleNamespace(run_dir=tmp_path)),
            SimpleNamespace(sketch_protocol=None, exact_gradients=None),
            {"sketch_dimension": 4},
        )


def test_the_runner_no_longer_refuses_the_flag_with_replicas() -> None:
    """``--countsketch-fidelity-sanity`` and ``--sketch-maps N`` now compose.

    The pairing used to be rejected during argument resolution. Keeping that
    rejection after the check became map-aware would have left the flag refusing
    the only configuration it is now most useful in.
    """

    source = (
        REPO_ROOT / "scripts" / "run_initialization_distribution_experiment.py"
    ).read_text(encoding="utf-8")
    assert "--countsketch-fidelity-sanity" in source
    assert "cannot be combined with" not in source


def test_the_offline_alternate_map_bank_is_untouched() -> None:
    """The methodology the error message points at still works as it did."""

    from llm_behavior_lab.analysis.countsketch_fidelity import ALTERNATE_SKETCH_SEEDS

    assert ALTERNATE_SKETCH_SEEDS == (101, 202, 303, 404, 505, 606, 707, 808)
    assert len(set(ALTERNATE_SKETCH_SEEDS)) == 8
    assert 20240917 not in ALTERNATE_SKETCH_SEEDS
