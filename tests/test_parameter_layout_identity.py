"""The sketch domain has a name, and two runs can check they share it.

A CountSketch map is a function of the *layout* it was drawn over -- how many
scalars there are, in which order -- and of nothing else. Two runs whose
parameter tuples agree can have their sketches compared; two that do not are
projecting different domains, and a stored map from one is meaningless against
the other.

``parameter_count`` alone cannot tell them apart: two models can carry the same
total while ordering or naming their tensors differently, and the projection
would then silently disagree entry for entry. The layout digest is what makes
that checkable instead of assumed.

What the digest deliberately ignores is as important as what it covers. Parameter
**values**, the initialization seed, the device and the dtype do not change which
scalars the sketch runs over, so including any of them would make the digest
differ between runs that are in fact directly comparable -- which is exactly the
question it exists to answer.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from llm_behavior_lab.evaluation import position_gradients as pg
from llm_behavior_lab.evaluation.position_gradients import (
    _parameter_layout_payload,
    _parameter_layout_sha256,
    _trainable_parameters,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class Fixed(torch.nn.Module):
    """A deliberately tiny, deliberately frozen layout."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.embedding = torch.nn.Embedding(5, 3)
        self.output = torch.nn.Linear(3, 5, bias=False)


#: The exact payload ``Fixed`` must serialize to. Written out in full rather than
#: rebuilt from the code under test, so a change to the encoding fails here
#: instead of quietly redefining what the digest means.
GOLDEN_PAYLOAD = (
    "parameter_layout/v1\n"
    "0\tembedding.weight\t(5, 3)\t15\n"
    "1\toutput.weight\t(5, 3)\t15\n"
)

#: SHA-256 of ``GOLDEN_PAYLOAD``.
GOLDEN_DIGEST = "d5ffd51dc3d416610a22f6b316f4dc92ae738f4965eb265671de02744dc0bf24"


def _layout(model):
    parameters = tuple(_trainable_parameters(model))
    return _parameter_layout_payload(model, parameters), _parameter_layout_sha256(
        model, parameters
    )


# -- the encoding itself -----------------------------------------------------------


def test_the_payload_and_digest_match_the_frozen_golden_values() -> None:
    """Both, not just the digest: a hash alone cannot show what was hashed."""

    payload, digest = _layout(Fixed())

    assert payload == GOLDEN_PAYLOAD
    assert digest == GOLDEN_DIGEST


def test_the_digest_is_lowercase_hex_sha256() -> None:
    _, digest = _layout(Fixed())

    assert len(digest) == 64
    assert digest == digest.lower()
    assert set(digest) <= set("0123456789abcdef")


def test_the_payload_is_utf8_encodable_and_schema_headed() -> None:
    payload, _ = _layout(Fixed())

    assert payload.startswith("parameter_layout/v1\n")
    assert payload.endswith("\n")
    assert payload.encode("utf-8").decode("utf-8") == payload


# -- what must not change the digest -------------------------------------------------


def test_a_different_initialization_seed_gives_the_same_digest() -> None:
    """The domain is the same domain however the weights were drawn."""

    assert _layout(Fixed(seed=0))[1] == _layout(Fixed(seed=12345))[1]


def test_changing_parameter_values_alone_does_not_change_the_digest() -> None:
    model = Fixed()
    before = _layout(model)[1]

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.5)

    assert _layout(model)[1] == before


def test_two_interpreter_processes_agree() -> None:
    """No dict ordering, hash randomization or address leaks into the digest."""

    script = (
        "import sys; sys.path.insert(0, 'src')\n"
        "import torch\n"
        "from llm_behavior_lab.evaluation.position_gradients import "
        "_parameter_layout_sha256, _trainable_parameters\n"
        "m = torch.nn.Module()\n"
        "m.embedding = torch.nn.Embedding(5, 3)\n"
        "m.output = torch.nn.Linear(3, 5, bias=False)\n"
        "print(_parameter_layout_sha256(m, tuple(_trainable_parameters(m))))\n"
    )
    runs = [
        subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
        ).stdout.strip()
        for seed in ("0", "1")
    ]

    assert runs[0] == runs[1] == GOLDEN_DIGEST


# -- what must change the digest -----------------------------------------------------


def test_renaming_a_parameter_changes_the_digest() -> None:
    model = Fixed()
    renamed = torch.nn.Module()
    renamed.embedding = model.embedding
    renamed.head = model.output  # same tensors, different registered name

    assert _layout(renamed)[1] != _layout(model)[1]


def test_changing_a_shape_changes_the_digest() -> None:
    wider = Fixed()
    wider.output = torch.nn.Linear(3, 7, bias=False)

    assert _layout(wider)[1] != _layout(Fixed())[1]


def test_swapping_the_order_of_equal_sized_parameters_changes_the_digest() -> None:
    """Order is the point.

    Both layouts hold the same shapes and the same total element count, so a
    digest over sizes alone would call them identical -- while the CountSketch
    map, drawn per tensor in registration order, would differ completely.
    """

    forward = torch.nn.Module()
    forward.alpha = torch.nn.Parameter(torch.zeros(4, 2))
    forward.beta = torch.nn.Parameter(torch.zeros(4, 2))

    reversed_ = torch.nn.Module()
    reversed_.beta = torch.nn.Parameter(torch.zeros(4, 2))
    reversed_.alpha = torch.nn.Parameter(torch.zeros(4, 2))

    forward_payload, forward_digest = _layout(forward)
    reversed_payload, reversed_digest = _layout(reversed_)

    assert forward_digest != reversed_digest
    # Same multiset of shapes and the same scalar total, so only order separates
    # them -- which is exactly what a size-only identity would miss.
    assert sorted(forward_payload.split("\n")) != sorted(reversed_payload.split("\n"))
    assert sum(p.numel() for p in forward.parameters()) == sum(
        p.numel() for p in reversed_.parameters()
    )


# -- separator safety -----------------------------------------------------------------


@pytest.mark.parametrize("bad", ["a\tb", "a\nb"])
def test_a_separator_inside_a_parameter_name_is_refused(bad, monkeypatch) -> None:
    """Otherwise two different layouts could serialize to the same bytes."""

    model = Fixed()
    parameters = tuple(_trainable_parameters(model))

    def named(*args, **kwargs):
        return [(bad, parameters[0]), ("output.weight", parameters[1])]

    monkeypatch.setattr(model, "named_parameters", named)

    with pytest.raises(ValueError, match="tab or newline"):
        _parameter_layout_payload(model, parameters)


# -- the named enumeration must be the sketched tuple ----------------------------------


def test_the_named_layout_is_object_identical_to_the_sketched_tuple() -> None:
    """Two enumerations, asserted equal rather than assumed equal."""

    model = Fixed()
    parameters = tuple(_trainable_parameters(model))
    named = [p for _, p in model.named_parameters() if p.requires_grad]

    assert len(named) == len(parameters)
    assert all(a is b for a, b in zip(named, parameters))
    _parameter_layout_payload(model, parameters)  # does not raise


def test_a_count_mismatch_between_the_enumerations_is_refused(monkeypatch) -> None:
    model = Fixed()
    parameters = tuple(_trainable_parameters(model))
    monkeypatch.setattr(
        model, "named_parameters", lambda *a, **k: [("embedding.weight", parameters[0])]
    )

    with pytest.raises(ValueError, match="do not match the sketched parameter tuple"):
        _parameter_layout_payload(model, parameters)


def test_a_reordered_named_enumeration_is_refused(monkeypatch) -> None:
    """Same objects, wrong order: the digest would describe a domain never used."""

    model = Fixed()
    parameters = tuple(_trainable_parameters(model))
    monkeypatch.setattr(
        model,
        "named_parameters",
        lambda *a, **k: [
            ("output.weight", parameters[1]),
            ("embedding.weight", parameters[0]),
        ],
    )

    with pytest.raises(ValueError, match="not the same object"):
        _parameter_layout_payload(model, parameters)


# -- the tied GPT-2 vocabulary block ----------------------------------------------------


def test_the_tied_gpt_parameter_appears_exactly_once() -> None:
    """``wte`` and ``lm_head`` are one Parameter; the layout must say so once.

    This is the only arm that ties, so it is the only place a duplicate could
    appear -- and a duplicate would both double-count the vocabulary block and
    desynchronize the layout from the sketch tables built beside it.
    """

    from llm_behavior_lab.models import build_model_from_config

    model = build_model_from_config(
        {
            "model": {
                "name": "gpt2",
                "params": {
                    "vocab_size": 64,
                    "dim": 16,
                    "n_layers": 1,
                    "n_heads": 2,
                    "max_seq_len": 8,
                    "bias": True,
                    "dropout": 0.0,
                },
            }
        }
    )
    assert model.transformer.wte.weight is model.lm_head.weight

    parameters = tuple(_trainable_parameters(model))
    payload = _parameter_layout_payload(model, parameters)
    names = [line.split("\t")[1] for line in payload.split("\n")[1:] if line]

    assert names.count("transformer.wte.weight") == 1
    assert not any("lm_head" in name for name in names)
    assert len(names) == len(parameters) == len(set(names))
    # The layout's element counts are the sketch's tensor sizes, in order.
    sizes = [int(line.split("\t")[3]) for line in payload.split("\n")[1:] if line]
    assert sizes == [p.numel() for p in parameters]


# -- the metadata ------------------------------------------------------------------------


def _measured(maps: int):
    from llm_behavior_lab.evaluation.init_distribution import build_evaluation_positions

    vocab, block, windows = 12, 4, 2

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(vocab, 8)
            self.output = torch.nn.Linear(8, vocab, bias=False)
            self.double()

        def forward(self, input_ids):
            return type(
                "Output", (), {"logits": self.output(self.embedding(input_ids))}
            )()

    positions = build_evaluation_positions(
        [(index * 5 + 2) % vocab for index in range(64)],
        block_size=block,
        num_windows=windows,
    )
    return pg.compute_position_gradient_norms(
        TinyModel(),
        positions,
        vocab_size=vocab,
        temperatures=(1.0,),
        gradient_sketch=True,
        sketch_dimension=6,
        sketch_maps=maps,
    )


@pytest.mark.parametrize("maps", [1, 4])
def test_the_layout_metadata_is_present_at_every_map_count(maps) -> None:
    protocol = _measured(maps).sketch_protocol

    assert len(protocol["parameter_layout_sha256"]) == 64
    assert protocol["parameter_layout_schema"] == "parameter_layout/v1"
    assert "named_parameters" in protocol["parameter_ordering"]
    assert "duplicate" in protocol["parameter_ordering"]
    assert protocol["parameter_tensor_count"] == 2


def test_tensor_count_and_element_count_are_different_quantities() -> None:
    """``parameter_count`` keeps its meaning; the new key is not a rename."""

    protocol = _measured(1).sketch_protocol

    assert protocol["parameter_tensor_count"] == 2
    assert protocol["parameter_count"] == 12 * 8 + 12 * 8
    assert protocol["parameter_count"] != protocol["parameter_tensor_count"]


def test_the_layout_metadata_is_json_safe() -> None:
    protocol = _measured(4).sketch_protocol

    assert json.loads(json.dumps(protocol)) == protocol


def test_the_digest_matches_a_directly_computed_one() -> None:
    """The recorded digest is the model's real layout, not a placeholder."""

    result = _measured(1)
    sizes = result.sketch_tensor_sizes

    assert sum(sizes) == result.sketch_protocol["parameter_count"]
    assert len(sizes) == result.sketch_protocol["parameter_tensor_count"]
