"""Tests for the initialization-distribution measurement core.

The scientific claim the experiment rests on is that only the initialization
changes between runs. Most of these tests exist to hold that claim up: the
evaluation positions must be reproducible and independent of any seed, and the
sampling stream must be separable from the initialization stream.
"""

from __future__ import annotations

import torch

import pytest

from llm_behavior_lab.evaluation.init_distribution import (
    NucleusSamplingSettings,
    build_evaluation_positions,
    compute_evaluation_logits,
    deterministic_window_starts,
    measure_from_logits,
    measure_initialization,
)
from llm_behavior_lab.utils import seed_everything

VOCAB_SIZE = 6
TOKENS = [index % VOCAB_SIZE for index in range(200)]


class _StubModel:
    """Minimal stand-in exposing the interface the measurement needs.

    Its logits are a deterministic function of the tokens it is shown, so a
    chunked forward pass is genuinely compared against an unchunked one rather
    than against a constant.
    """

    def __init__(self, *, output_size: int) -> None:
        self.output_size = output_size
        self.eval_calls = 0
        self.forward_calls = 0

    def eval(self) -> "_StubModel":
        self.eval_calls += 1
        return self

    def __call__(self, *, input_ids: torch.Tensor):
        self.forward_calls += 1
        tokens = input_ids.unsqueeze(-1).float()
        offsets = torch.arange(self.output_size, dtype=torch.float32)
        return type("Output", (), {"logits": torch.cos(tokens * 0.7 + offsets * 1.3)})()


def _settings(**overrides) -> NucleusSamplingSettings:
    fields = {"temperature": 0.6, "top_p": 0.9, "seed": 555, "num_replicates": 3}
    fields.update(overrides)
    return NucleusSamplingSettings(**fields)


# -- deterministic evaluation positions ----------------------------------


def test_window_starts_span_the_split_evenly() -> None:
    """The first and last legal offsets are always included."""

    starts = deterministic_window_starts(101, block_size=10, num_windows=5)

    assert starts[0] == 0
    assert starts[-1] == 101 - 10 - 1
    assert starts == sorted(starts)
    assert len(set(starts)) == 5


def test_window_starts_are_identical_across_calls() -> None:
    """Reproducibility is what lets separate runs be compared at all."""

    first = deterministic_window_starts(500, block_size=16, num_windows=32)
    second = deterministic_window_starts(500, block_size=16, num_windows=32)

    assert first == second


def test_window_starts_ignore_the_global_random_state() -> None:
    """The central control: changing the model seed must not move the positions.

    If selection consumed the global RNG, every initialization would silently be
    evaluated on different text and the comparison would be meaningless.
    """

    seed_everything(1)
    first = deterministic_window_starts(500, block_size=16, num_windows=16)
    seed_everything(999)
    second = deterministic_window_starts(500, block_size=16, num_windows=16)

    assert first == second


def test_requesting_more_windows_than_available_is_rejected() -> None:
    """Duplicated windows would inflate the position count without new data."""

    with pytest.raises(ValueError, match="distinct start offsets"):
        deterministic_window_starts(20, block_size=10, num_windows=50)


def test_a_split_shorter_than_one_window_is_rejected() -> None:
    """The error must name the requirement rather than fail on a slice."""

    with pytest.raises(ValueError, match="cannot supply a window"):
        deterministic_window_starts(5, block_size=10, num_windows=1)


def test_targets_are_the_inputs_shifted_by_one_position() -> None:
    """Next-token targets define what the guesses are compared against."""

    positions = build_evaluation_positions(TOKENS, block_size=4, num_windows=3)

    assert positions.input_ids.shape == (3, 4)
    assert positions.target_ids.shape == (3, 4)
    for window in range(3):
        start = positions.starts[window]
        assert positions.input_ids[window].tolist() == TOKENS[start : start + 4]
        assert positions.target_ids[window].tolist() == TOKENS[start + 1 : start + 5]


def test_position_count_is_windows_times_block_size() -> None:
    """The quantity every zero-guess and adequacy statement depends on."""

    positions = build_evaluation_positions(TOKENS, block_size=8, num_windows=5)

    assert positions.num_windows == 5
    assert positions.num_positions == 40
    assert positions.as_dict()["num_positions"] == 40


def test_positions_are_reused_unchanged_for_every_initialization() -> None:
    """Rebuilding after reseeding must yield byte-identical windows."""

    seed_everything(11)
    first = build_evaluation_positions(TOKENS, block_size=8, num_windows=6)
    seed_everything(22)
    second = build_evaluation_positions(TOKENS, block_size=8, num_windows=6)

    assert first.starts == second.starts
    assert torch.equal(first.input_ids, second.input_ids)
    assert torch.equal(first.target_ids, second.target_ids)


# -- logits and measurement ----------------------------------------------


def test_evaluation_logits_are_restricted_to_the_valid_support() -> None:
    """The model may be wider than the tokenizer; the comparison may not be."""

    positions = build_evaluation_positions(TOKENS, block_size=4, num_windows=3)
    model = _StubModel(output_size=10)

    logits = compute_evaluation_logits(model, positions, vocab_size=VOCAB_SIZE)

    assert logits.shape == (3, 4, VOCAB_SIZE)


def test_forward_chunking_does_not_change_the_logits() -> None:
    """Chunking is a memory decision and must stay invisible in the result."""

    positions = build_evaluation_positions(TOKENS, block_size=4, num_windows=8)
    model = _StubModel(output_size=VOCAB_SIZE)

    whole = compute_evaluation_logits(
        model, positions, vocab_size=VOCAB_SIZE, forward_batch_size=8
    )
    chunked = compute_evaluation_logits(
        model, positions, vocab_size=VOCAB_SIZE, forward_batch_size=3
    )

    assert torch.equal(whole, chunked)
    assert model.forward_calls == 1 + 3


def test_measurement_produces_aligned_per_token_vectors() -> None:
    """Every vector spans the full vocabulary so records align by token ID."""

    logits = torch.randn(4, 5, VOCAB_SIZE)

    measurement = measure_from_logits(
        model_seed=1000, logits=logits, vocab_size=VOCAB_SIZE, sampling=_settings()
    )

    assert measurement.greedy_counts.shape == (VOCAB_SIZE,)
    assert measurement.nucleus_counts.shape == (3, VOCAB_SIZE)
    assert measurement.mean_predicted_probabilities.shape == (VOCAB_SIZE,)
    assert measurement.num_positions == 20
    assert int(measurement.greedy_counts.sum()) == 20
    assert [int(row.sum()) for row in measurement.nucleus_counts] == [20, 20, 20]


def test_mean_predicted_probabilities_form_a_distribution() -> None:
    """Kept distinct from guess frequencies, but still a valid distribution."""

    logits = torch.randn(3, 4, VOCAB_SIZE)

    measurement = measure_from_logits(
        model_seed=7, logits=logits, vocab_size=VOCAB_SIZE, sampling=_settings()
    )

    assert float(measurement.mean_predicted_probabilities.sum()) == pytest.approx(1.0, abs=1e-5)


def test_mean_predicted_probabilities_differ_from_guess_frequencies() -> None:
    """The two must never be treated as interchangeable.

    With a near-deterministic distribution, greedy places all its mass on one
    token while the mean predictive vector keeps some elsewhere.
    """

    logits = torch.tensor([[[5.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]).repeat(1, 8, 1)

    measurement = measure_from_logits(
        model_seed=3, logits=logits, vocab_size=VOCAB_SIZE, sampling=_settings()
    )
    greedy_fractions = measurement.greedy_counts.float() / measurement.greedy_counts.sum()

    assert greedy_fractions[0] == pytest.approx(1.0)
    assert float(measurement.mean_predicted_probabilities[0]) < 1.0
    assert float(measurement.mean_predicted_probabilities[1]) > 0.0


def test_replicates_within_one_initialization_are_not_identical() -> None:
    """Otherwise averaging replicates would remove no sampling noise at all."""

    logits = torch.randn(6, 8, VOCAB_SIZE)

    measurement = measure_from_logits(
        model_seed=1000, logits=logits, vocab_size=VOCAB_SIZE, sampling=_settings(num_replicates=4)
    )

    rows = {tuple(row.tolist()) for row in measurement.nucleus_counts}
    assert len(rows) > 1


def test_measurement_is_reproducible_for_the_same_seeds() -> None:
    """Same logits, same model seed, same sampling settings, same result."""

    logits = torch.randn(4, 6, VOCAB_SIZE)

    first = measure_from_logits(
        model_seed=1000, logits=logits, vocab_size=VOCAB_SIZE, sampling=_settings()
    )
    second = measure_from_logits(
        model_seed=1000, logits=logits, vocab_size=VOCAB_SIZE, sampling=_settings()
    )

    assert torch.equal(first.nucleus_counts, second.nucleus_counts)
    assert torch.equal(first.greedy_counts, second.greedy_counts)


def test_sampling_stream_is_separated_per_initialization() -> None:
    """Two initializations must not share one stochastic draw sequence.

    The logits are held fixed here so the only thing that can differ is the
    sampling stream, which is exactly what the seed mixing is meant to change.
    """

    logits = torch.randn(8, 8, VOCAB_SIZE)

    first = measure_from_logits(
        model_seed=1000, logits=logits, vocab_size=VOCAB_SIZE, sampling=_settings()
    )
    second = measure_from_logits(
        model_seed=1001, logits=logits, vocab_size=VOCAB_SIZE, sampling=_settings()
    )

    assert not torch.equal(first.nucleus_counts, second.nucleus_counts)
    # Greedy is deterministic given the logits, so it must be unaffected.
    assert torch.equal(first.greedy_counts, second.greedy_counts)


def test_sampling_seed_changes_the_draws_without_touching_greedy() -> None:
    """Sampling randomness is a separate knob from initialization randomness."""

    logits = torch.randn(8, 8, VOCAB_SIZE)

    first = measure_from_logits(
        model_seed=1000, logits=logits, vocab_size=VOCAB_SIZE, sampling=_settings(seed=1)
    )
    second = measure_from_logits(
        model_seed=1000, logits=logits, vocab_size=VOCAB_SIZE, sampling=_settings(seed=99999)
    )

    assert not torch.equal(first.nucleus_counts, second.nucleus_counts)
    assert torch.equal(first.greedy_counts, second.greedy_counts)


def test_unrestricted_logits_are_rejected() -> None:
    """Measuring on a wider vocabulary would silently break every comparison."""

    with pytest.raises(ValueError, match="does not match vocab_size"):
        measure_from_logits(
            model_seed=1,
            logits=torch.randn(2, 3, VOCAB_SIZE + 4),
            vocab_size=VOCAB_SIZE,
            sampling=_settings(),
        )


def test_sampling_settings_validate_their_parameters() -> None:
    """Recorded policy values must be usable ones."""

    with pytest.raises(ValueError, match="temperature"):
        _settings(temperature=0.0).validate()
    with pytest.raises(ValueError, match="top_p"):
        _settings(top_p=1.5).validate()
    with pytest.raises(ValueError, match="num_replicates"):
        _settings(num_replicates=0).validate()


def test_sampling_settings_are_recorded_for_persistence() -> None:
    """A figure must never be readable without the policy behind it."""

    described = _settings().as_dict()

    assert described["method"] == "temperature_top_p"
    assert described["temperature"] == 0.6
    assert described["top_p"] == 0.9
    assert described["sampling_seed"] == 555
    assert described["num_replicates"] == 3


# -- streaming measurement and its memory invariant -----------------------
#
# A realistic subword vocabulary makes an all-position logits tensor
# prohibitive: 32768 positions over 32000 tokens is about 3.9 GiB in float32.
# measure_initialization therefore folds each batch into per-token counters and
# discards it. These tests pin both halves of that claim -- the result is
# unchanged, and no per-position accumulator exists.


def test_streamed_measurement_equals_the_all_at_once_reference() -> None:
    """Batching must change when logits are freed, not what they are.

    Both paths consume the same pre-drawn uniforms, so this is exact equality
    rather than a statistical comparison.
    """

    positions = build_evaluation_positions(TOKENS, block_size=4, num_windows=8)
    model = _StubModel(output_size=VOCAB_SIZE)
    settings = _settings()

    streamed = measure_initialization(
        model,
        positions,
        model_seed=1000,
        vocab_size=VOCAB_SIZE,
        sampling=settings,
        forward_batch_size=3,
    )
    reference = measure_from_logits(
        model_seed=1000,
        logits=compute_evaluation_logits(model, positions, vocab_size=VOCAB_SIZE),
        vocab_size=VOCAB_SIZE,
        sampling=settings,
    )

    assert torch.equal(streamed.greedy_counts, reference.greedy_counts)
    assert torch.equal(streamed.nucleus_counts, reference.nucleus_counts)
    assert torch.allclose(
        streamed.mean_predicted_probabilities,
        reference.mean_predicted_probabilities,
        atol=1e-6,
    )
    assert streamed.num_positions == reference.num_positions


def test_the_batch_size_does_not_change_the_result() -> None:
    """Memory is a free parameter; the measurement is not."""

    positions = build_evaluation_positions(TOKENS, block_size=4, num_windows=12)
    model = _StubModel(output_size=VOCAB_SIZE)

    results = [
        measure_initialization(
            model,
            positions,
            model_seed=7,
            vocab_size=VOCAB_SIZE,
            sampling=_settings(),
            forward_batch_size=size,
        )
        for size in (1, 5, 12)
    ]

    for other in results[1:]:
        assert torch.equal(results[0].greedy_counts, other.greedy_counts)
        assert torch.equal(results[0].nucleus_counts, other.nucleus_counts)


def test_no_accumulator_is_indexed_by_evaluation_position() -> None:
    """Only ``[vocab]``-sized state may survive a batch.

    Asserted on the returned tensors, which are the only things the function
    keeps: anything proportional to the position count would show up here.
    """

    positions = build_evaluation_positions(TOKENS, block_size=8, num_windows=16)
    settings = _settings(num_replicates=3)

    measurement = measure_initialization(
        _StubModel(output_size=VOCAB_SIZE),
        positions,
        model_seed=1,
        vocab_size=VOCAB_SIZE,
        sampling=settings,
        forward_batch_size=4,
    )

    assert positions.num_positions == 128
    assert measurement.greedy_counts.shape == (VOCAB_SIZE,)
    assert measurement.nucleus_counts.shape == (3, VOCAB_SIZE)
    assert measurement.mean_predicted_probabilities.shape == (VOCAB_SIZE,)


def test_the_streaming_path_never_concatenates_logits() -> None:
    """The reference implementation may materialize everything; the experiment may not.

    Checked on the parsed source of ``measure_initialization`` alone, so the
    module may still offer ``compute_evaluation_logits`` for small cases.
    """

    import ast
    import inspect

    from llm_behavior_lab.evaluation import init_distribution

    tree = ast.parse(inspect.getsource(init_distribution.measure_initialization))
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert "cat" not in calls
    assert "stack" not in calls


def test_transient_logits_are_bounded_by_the_batch_size() -> None:
    """The model is only ever asked for one batch of windows at a time."""

    positions = build_evaluation_positions(TOKENS, block_size=4, num_windows=10)

    class _RecordingModel(_StubModel):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.batch_shapes: list[tuple[int, ...]] = []

        def __call__(self, *, input_ids):
            self.batch_shapes.append(tuple(input_ids.shape))
            return super().__call__(input_ids=input_ids)

    model = _RecordingModel(output_size=VOCAB_SIZE)
    measure_initialization(
        model,
        positions,
        model_seed=1,
        vocab_size=VOCAB_SIZE,
        sampling=_settings(),
        forward_batch_size=3,
    )

    assert max(shape[0] for shape in model.batch_shapes) == 3
    assert sum(shape[0] for shape in model.batch_shapes) == positions.num_windows


# -- the predictive support ----------------------------------------------


def test_excluded_tokens_are_never_guessed() -> None:
    """Structural IDs leave the support for greedy, nucleus, and probabilities alike."""

    positions = build_evaluation_positions(TOKENS, block_size=4, num_windows=6)
    eligible = [2, 3, 4, 5]

    measurement = measure_initialization(
        _StubModel(output_size=VOCAB_SIZE),
        positions,
        model_seed=1,
        vocab_size=VOCAB_SIZE,
        sampling=_settings(),
        eligible_token_ids=eligible,
    )

    excluded = [0, 1]
    assert measurement.greedy_counts[excluded].sum() == 0
    assert measurement.nucleus_counts[:, excluded].sum() == 0
    assert float(measurement.mean_predicted_probabilities[excluded].sum()) == pytest.approx(0.0)


def test_excluded_tokens_keep_their_canonical_positions() -> None:
    """Vectors stay full length; exclusion is a mask, never a renumbering."""

    positions = build_evaluation_positions(TOKENS, block_size=4, num_windows=6)

    measurement = measure_initialization(
        _StubModel(output_size=VOCAB_SIZE),
        positions,
        model_seed=1,
        vocab_size=VOCAB_SIZE,
        sampling=_settings(),
        eligible_token_ids=[2, 3, 4, 5],
    )

    assert measurement.greedy_counts.shape == (VOCAB_SIZE,)
    assert int(measurement.greedy_counts.sum()) == positions.num_positions


def test_probability_mass_is_renormalized_over_the_support() -> None:
    """Masked logits give the excluded tokens exactly zero, not a small residue."""

    positions = build_evaluation_positions(TOKENS, block_size=4, num_windows=4)

    measurement = measure_initialization(
        _StubModel(output_size=VOCAB_SIZE),
        positions,
        model_seed=1,
        vocab_size=VOCAB_SIZE,
        sampling=_settings(),
        eligible_token_ids=[1, 2, 3],
    )

    assert float(measurement.mean_predicted_probabilities.sum()) == pytest.approx(1.0, abs=1e-5)
    assert float(measurement.mean_predicted_probabilities[0]) == 0.0
