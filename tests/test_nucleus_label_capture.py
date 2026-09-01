"""Nucleus labels captured in-run must still pass the histogram gate.

The experiment used to keep only ``[vocab]``-sized counters, so grouping
gradients by the sampled token meant recovering the labels afterwards: a forward
pass over every evaluation position, once per sampling temperature, at roughly
five hours per production arm. Capturing them during the run removes that
entirely, because the draw has already happened -- the labels are simply kept
instead of only tallied.

That makes the histogram gate *expected* to pass, which is exactly why it still
runs. "Correct by construction" is a claim about the code, and the gate is a
check on the data; the whole reason the gate exists is that a reconstruction
which does not reproduce the recorded counts is a reconstruction of some other
model. Replacing a check with an argument is how that protection is lost.

The capture also sidesteps a real hazard the post-hoc route has to manage: the
logits are the draw's other input, and on accelerator kernels a matrix
multiplication can return bit-different values at different batch shapes, moving
a token that sits on a truncation boundary. Labels taken from the original draw
cannot disagree with the counts that draw produced, whatever the batch size was.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from llm_behavior_lab.analysis.nucleus_clustering import histogram_gate
from llm_behavior_lab.evaluation.init_distribution import (
    NucleusSamplingSettings,
    build_evaluation_positions,
    measure_initialization,
)

VOCAB = 12
BLOCK = 4
WINDOWS = 4
TOKENS = [(index * 5 + 2) % VOCAB for index in range(80)]
SWEEP = (0.12, 0.6, 1.2)


class TinyModel(torch.nn.Module):
    def __init__(self, seed: int = 5) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.embed = torch.nn.Embedding(VOCAB, 8)
        self.out = torch.nn.Linear(8, VOCAB, bias=False)
        with torch.no_grad():
            self.embed.weight.copy_(torch.randn(VOCAB, 8, generator=generator))
            self.out.weight.copy_(torch.randn(VOCAB, 8, generator=generator))
        self.double()

    def forward(self, input_ids):
        return type("Output", (), {"logits": self.out(self.embed(input_ids))})()


def _measure(*, collect: bool, batch: int = 2):
    positions = build_evaluation_positions(
        TOKENS, block_size=BLOCK, num_windows=WINDOWS
    )
    return measure_initialization(
        TinyModel(), positions,
        model_seed=11, vocab_size=VOCAB,
        sampling=NucleusSamplingSettings(temperature=0.6, top_p=0.9, seed=3),
        forward_batch_size=batch,
        sweep_temperatures=SWEEP,
        collect_sweep_labels=collect,
    ), positions


def test_labels_are_off_by_default() -> None:
    """A run that does not need them pays nothing for them."""

    measurement, _ = _measure(collect=False)
    assert measurement.sweep_labels is None
    assert measurement.sweep_counts is not None


def test_captured_labels_pass_the_exact_histogram_gate() -> None:
    """Integer equality, per temperature. Not "close", not "mostly"."""

    measurement, positions = _measure(collect=True)
    labels = measurement.sweep_labels.numpy()

    histogram_gate(
        labels,
        measurement.sweep_counts.numpy(),
        vocab_size=VOCAB,
        temperatures=SWEEP,
    )


def test_the_gate_still_rejects_labels_that_do_not_match() -> None:
    """The gate must have teeth, or passing it means nothing.

    A single moved label is enough: the counts are exact integers and a
    reconstruction that drifted at all is describing a different model.
    """

    measurement, _ = _measure(collect=True)
    labels = measurement.sweep_labels.numpy().copy()
    labels[0, 0] = (labels[0, 0] + 1) % VOCAB

    with pytest.raises(ValueError):
        histogram_gate(
            labels,
            measurement.sweep_counts.numpy(),
            vocab_size=VOCAB,
            temperatures=SWEEP,
        )


def test_one_label_per_position_per_temperature() -> None:
    measurement, positions = _measure(collect=True)
    assert measurement.sweep_labels.shape == (
        len(SWEEP), positions.num_positions
    )
    assert measurement.sweep_labels.dtype == torch.int32


def test_every_label_is_a_real_token() -> None:
    measurement, _ = _measure(collect=True)
    labels = measurement.sweep_labels.numpy()
    assert labels.min() >= 0
    assert labels.max() < VOCAB


def test_the_labels_are_the_draw_that_produced_the_counts() -> None:
    """Not merely consistent in aggregate: the histogram of the kept labels is
    the recorded histogram, token for token."""

    measurement, _ = _measure(collect=True)
    labels = measurement.sweep_labels.numpy()
    counts = measurement.sweep_counts.numpy()
    for index in range(len(SWEEP)):
        rebuilt = np.bincount(labels[index], minlength=VOCAB)
        assert np.array_equal(rebuilt, counts[index])


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_capture_is_invariant_to_the_forward_batch_size(batch) -> None:
    """The hazard the post-hoc reconstruction has to manage does not arise here.

    Batching changes when logits are discarded, not what they are, and the
    uniforms are indexed by position -- so labels taken from the original draw
    agree across batch shapes on CPU, and cannot disagree with their own counts
    on any device.
    """

    reference, _ = _measure(collect=True, batch=2)
    measurement, _ = _measure(collect=True, batch=batch)
    assert np.array_equal(
        measurement.sweep_labels.numpy(), reference.sweep_labels.numpy()
    )


def test_capturing_labels_does_not_change_the_counts() -> None:
    """Turning the capture on must not perturb the measurement it observes."""

    without, _ = _measure(collect=False)
    with_labels, _ = _measure(collect=True)

    assert np.array_equal(
        without.sweep_counts.numpy(), with_labels.sweep_counts.numpy()
    )
    assert np.array_equal(
        without.greedy_counts.numpy(), with_labels.greedy_counts.numpy()
    )
    assert np.array_equal(
        without.nucleus_counts.numpy(), with_labels.nucleus_counts.numpy()
    )


def test_capturing_labels_does_not_shift_any_random_stream() -> None:
    """Enabling an analysis must not move the sampling draws.

    The model is built *before* the state is captured, deliberately: constructing
    the layers draws from the global generator, and including that would test
    ``torch.nn.Embedding`` rather than anything this change touches.
    """

    model = TinyModel()
    positions = build_evaluation_positions(
        TOKENS, block_size=BLOCK, num_windows=WINDOWS
    )
    sampling = NucleusSamplingSettings(temperature=0.6, top_p=0.9, seed=3)

    state = torch.get_rng_state()
    measure_initialization(
        model, positions, model_seed=11, vocab_size=VOCAB, sampling=sampling,
        forward_batch_size=2, sweep_temperatures=SWEEP, collect_sweep_labels=True,
    )
    assert torch.equal(torch.get_rng_state(), state)


def test_the_labels_cost_a_predictable_amount() -> None:
    """786 KiB at experiment scale, against ~5 h of forward replay per arm."""

    measurement, positions = _measure(collect=True)
    assert (
        measurement.sweep_labels.numpy().nbytes
        == len(SWEEP) * positions.num_positions * 4
    )
