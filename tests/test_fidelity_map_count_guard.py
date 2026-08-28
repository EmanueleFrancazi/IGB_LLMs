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


def test_an_explicit_multi_map_result_is_refused(tmp_path) -> None:
    runner = _runner()

    with pytest.raises(ValueError, match="map_count=4"):
        runner._write_countsketch_fidelity(
            SimpleNamespace(paths=SimpleNamespace(run_dir=tmp_path)),
            _result(4),
            {"sketch_dimension": 4},
        )


def test_the_refusal_happens_before_any_reconstruction(tmp_path) -> None:
    """``exact_gradients`` is ``None``.

    If the guard ran late, the reconstruction would raise ``AttributeError`` on
    it instead. Getting a ``ValueError`` naming the map count is what proves the
    ordering.
    """

    runner = _runner()

    with pytest.raises(ValueError) as caught:
        runner._write_countsketch_fidelity(
            SimpleNamespace(paths=SimpleNamespace(run_dir=tmp_path)),
            _result(2),
            {"sketch_dimension": 4},
        )

    assert "map_count=2" in str(caught.value)
    assert not isinstance(caught.value, AttributeError)


def test_the_refusal_leaves_no_partial_artifact(tmp_path) -> None:
    """A rejected run must not leave anything a later reader could trust."""

    runner = _runner()
    before = set(tmp_path.rglob("*"))

    with pytest.raises(ValueError):
        runner._write_countsketch_fidelity(
            SimpleNamespace(paths=SimpleNamespace(run_dir=tmp_path)),
            _result(4),
            {"sketch_dimension": 4},
        )

    assert set(tmp_path.rglob("*")) == before
    assert list(tmp_path.rglob("countsketch_fidelity.npz")) == []


def test_the_error_separates_replicas_from_the_offline_methodology(tmp_path) -> None:
    """The two are easy to confuse and the message must not let them be."""

    runner = _runner()

    with pytest.raises(ValueError) as caught:
        runner._write_countsketch_fidelity(
            SimpleNamespace(paths=SimpleNamespace(run_dir=tmp_path)),
            _result(4),
            {"sketch_dimension": 4},
        )

    message = str(caught.value)
    assert "alternate-map" in message
    assert "remains valid" in message
    assert "map-0" in message


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


def test_the_runner_now_exposes_the_flag_and_refuses_the_combination() -> None:
    """The tripwire this replaced was a *dated* one, and it has now fired.

    It asserted ``--sketch-maps`` was absent from the runner, so that ``M > 1``
    could not become reachable before the consumers could read it. They can now,
    the flag exists, and that assertion has done its job.

    What must not weaken is the reason it existed. The guard below stays the
    last line of defence, and the runner gains an *earlier* one: the two
    together mean the combination is refused at argument resolution, before a
    dataset is loaded, and refused again at the writer for any caller that
    never passed through argument resolution at all.
    """

    source = (
        REPO_ROOT / "scripts" / "run_initialization_distribution_experiment.py"
    ).read_text(encoding="utf-8")

    assert '"--sketch-maps",' in source
    # The early refusal.
    assert "def _validate_sketch_map_request" in source
    assert "--countsketch-fidelity-sanity" in source
    # The writer guard, unchanged in its role.
    assert "def _write_countsketch_fidelity" in source
    assert "if fidelity_map_count != 1:" in source


def test_the_offline_alternate_map_bank_is_untouched() -> None:
    """The methodology the error message points at still works as it did."""

    from llm_behavior_lab.analysis.countsketch_fidelity import ALTERNATE_SKETCH_SEEDS

    assert ALTERNATE_SKETCH_SEEDS == (101, 202, 303, 404, 505, 606, 707, 808)
    assert len(set(ALTERNATE_SKETCH_SEEDS)) == 8
    assert 20240917 not in ALTERNATE_SKETCH_SEEDS
