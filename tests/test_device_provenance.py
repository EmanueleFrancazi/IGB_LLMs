"""Which physical accelerator a run used must be recoverable from its metadata.

Near-term campaigns run one complete model replica per GPU on a two-device host,
so ``str(device)`` is no longer enough to identify a run's hardware. It is
ambiguous in two directions at once:

* two processes launched as ``cuda:0`` and ``cuda:1`` differ in the string, but
* two processes each launched as plain ``cuda``, one under
  ``CUDA_VISIBLE_DEVICES=0`` and one under ``=1``, record the *same* string while
  running on different cards,
* and two identical cards report the same product name either way.

Only the logical index, the device name and the raw environment value together
separate all three. Every test here mocks the CUDA layer: none requires a
working driver, and the suite still runs on a CPU-only host.
"""

from __future__ import annotations

import pytest
import torch

from llm_behavior_lab.utils import describe_device

CUDA_VISIBLE = "CUDA_VISIBLE_DEVICES"


@pytest.fixture
def mocked_cuda(monkeypatch):
    """Pretend two identical cards are present, without touching a driver."""

    def factory(*, current_index: int = 0, name: str = "Mock GPU 10GB"):
        monkeypatch.setattr(torch.cuda, "current_device", lambda: current_index)
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: name)

    return factory


# -- CPU and other non-CUDA devices -------------------------------------------


def test_cpu_provenance_is_unambiguous_and_complete(monkeypatch) -> None:
    monkeypatch.delenv(CUDA_VISIBLE, raising=False)

    description = describe_device("cpu")

    assert description == {
        "device": "cpu",
        "type": "cpu",
        "index": None,
        "resolved_index": None,
        "name": None,
        "cuda_visible_devices": None,
    }


def test_the_cpu_path_calls_no_cuda_api(monkeypatch) -> None:
    """A CPU-only or old-driver host must not be made to initialize CUDA."""

    def forbidden(*args, **kwargs):
        raise AssertionError("describe_device touched a CUDA API on a CPU device")

    monkeypatch.setattr(torch.cuda, "current_device", forbidden)
    monkeypatch.setattr(torch.cuda, "get_device_name", forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)

    assert describe_device("cpu")["type"] == "cpu"
    assert describe_device(torch.device("cpu"))["name"] is None


def test_a_non_cuda_accelerator_is_recorded_without_cuda_fields(monkeypatch) -> None:
    monkeypatch.delenv(CUDA_VISIBLE, raising=False)

    description = describe_device("mps")

    assert description["type"] == "mps"
    assert description["resolved_index"] is None
    assert description["name"] is None


# -- explicit and implicit logical index --------------------------------------


def test_an_explicit_logical_index_is_recorded_as_requested(mocked_cuda, monkeypatch) -> None:
    monkeypatch.delenv(CUDA_VISIBLE, raising=False)
    mocked_cuda(current_index=0)

    zero = describe_device("cuda:0")
    one = describe_device("cuda:1")

    assert zero["index"] == 0 and zero["resolved_index"] == 0
    # Requested index wins over the current device; no silent substitution.
    assert one["index"] == 1 and one["resolved_index"] == 1
    assert zero["device"] == "cuda:0" and one["device"] == "cuda:1"


def test_an_implicit_index_is_resolved_from_the_current_device(mocked_cuda, monkeypatch) -> None:
    """Plain ``cuda`` records which device it actually landed on."""

    monkeypatch.delenv(CUDA_VISIBLE, raising=False)
    mocked_cuda(current_index=1)

    description = describe_device("cuda")

    assert description["device"] == "cuda"
    assert description["index"] is None
    assert description["resolved_index"] == 1
    assert description["name"] == "Mock GPU 10GB"


# -- CUDA_VISIBLE_DEVICES -----------------------------------------------------


@pytest.mark.parametrize("value", ["0", "1", "0,1", ""])
def test_the_raw_environment_value_is_recorded_verbatim(
    mocked_cuda, monkeypatch, value
) -> None:
    monkeypatch.setenv(CUDA_VISIBLE, value)
    mocked_cuda()

    assert describe_device("cuda:0")["cuda_visible_devices"] == value


def test_an_absent_variable_is_distinguishable_from_an_empty_one(
    mocked_cuda, monkeypatch
) -> None:
    """``None`` means unset; ``""`` means set-but-empty, which hides every GPU."""

    mocked_cuda()

    monkeypatch.delenv(CUDA_VISIBLE, raising=False)
    absent = describe_device("cuda:0")["cuda_visible_devices"]
    monkeypatch.setenv(CUDA_VISIBLE, "")
    empty = describe_device("cuda:0")["cuda_visible_devices"]

    assert absent is None
    assert empty == ""
    assert absent != empty


# -- the three cases the campaign has to tell apart ---------------------------


def test_two_devices_on_one_host_are_distinguishable(mocked_cuda, monkeypatch) -> None:
    """All GPUs visible, processes pinned by logical index."""

    monkeypatch.setenv(CUDA_VISIBLE, "0,1")
    mocked_cuda(current_index=0)

    first = describe_device("cuda:0")
    second = describe_device("cuda:1")

    assert first != second
    assert (first["resolved_index"], second["resolved_index"]) == (0, 1)


def test_masked_processes_with_identical_strings_are_distinguishable(
    mocked_cuda, monkeypatch
) -> None:
    """The hard case: same device string, same card name, different hardware.

    Two processes each see one GPU as logical device 0, so ``str(device)`` and
    ``get_device_name`` agree completely. Only the environment value separates
    them, which is why it is recorded raw rather than parsed.
    """

    mocked_cuda(current_index=0)

    monkeypatch.setenv(CUDA_VISIBLE, "0")
    on_first = describe_device("cuda")
    monkeypatch.setenv(CUDA_VISIBLE, "1")
    on_second = describe_device("cuda")

    assert on_first["device"] == on_second["device"] == "cuda"
    assert on_first["resolved_index"] == on_second["resolved_index"] == 0
    assert on_first["name"] == on_second["name"]
    assert on_first["cuda_visible_devices"] != on_second["cuda_visible_devices"]
    assert on_first != on_second


def test_identical_card_names_do_not_collapse_the_record(mocked_cuda, monkeypatch) -> None:
    """Two of the same model of GPU must still yield different provenance."""

    monkeypatch.setenv(CUDA_VISIBLE, "0,1")
    mocked_cuda(name="NVIDIA GeForce RTX 3080")

    first = describe_device("cuda:0")
    second = describe_device("cuda:1")

    assert first["name"] == second["name"]
    assert first != second


# -- agreement with what the runner already stores ----------------------------


def test_the_description_agrees_with_the_canonical_device_string(
    mocked_cuda, monkeypatch
) -> None:
    """The added block must not contradict the run's existing ``device`` key."""

    monkeypatch.delenv(CUDA_VISIBLE, raising=False)
    mocked_cuda()

    for requested in ("cpu", "cuda", "cuda:1"):
        assert describe_device(requested)["device"] == str(torch.device(requested))


def test_the_description_is_json_serializable(mocked_cuda, monkeypatch) -> None:
    """It is persisted into metadata.json, so it has to survive the round trip."""

    import json

    monkeypatch.setenv(CUDA_VISIBLE, "1")
    mocked_cuda()

    for requested in ("cpu", "cuda:0"):
        description = describe_device(requested)
        assert json.loads(json.dumps(description)) == description
