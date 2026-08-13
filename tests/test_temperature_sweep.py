"""Tests for the multi-temperature nucleus sweep.

The sweep is *additional* analysis: the canonical policy stays the scalar
``sampling.temperature`` and figures 0-4 keep their meanings. Most of these tests
defend that boundary, plus the three properties the integration rests on --
one forward pass serves every temperature, one uniform draw serves every
temperature, and a sweep value at the canonical temperature is byte-identical to
the canonical result.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from llm_behavior_lab.analysis import (
    InitializationExperimentRecord,
    ranked_distance_to_greedy,
    ranked_distance_to_uniform,
    simulate_uniform_null,
    sweep_condition_summary,
    sweep_summary,
)
from llm_behavior_lab.analysis.transition import TRANSITION_METRICS
from llm_behavior_lab.evaluation.guessing import (
    greedy_guess_ids,
    nucleus_guess_ids,
    nucleus_guess_ids_by_temperature,
)
from llm_behavior_lab.evaluation.init_distribution import (
    NucleusSamplingSettings,
    build_evaluation_positions,
    measure_initialization,
)
from llm_behavior_lab.models.llama.config import LlamaConfig
from llm_behavior_lab.models.llama.model import LlamaForCausalLM
from llm_behavior_lab.utils import seed_everything

REPO_ROOT = Path(__file__).resolve().parents[1]
SWEEP = (0.12, 0.24, 0.36, 0.48, 0.60, 1.20)
CANONICAL = 0.6
VOCAB_SIZE = 32
TOKENS = [(index * 7) % VOCAB_SIZE for index in range(400)]


def _settings(**overrides) -> NucleusSamplingSettings:
    fields = {
        "temperature": CANONICAL,
        "top_p": 0.9,
        "seed": 555,
        "num_replicates": 1,
        "common_random_numbers": True,
    }
    fields.update(overrides)
    return NucleusSamplingSettings(**fields)


def _model() -> LlamaForCausalLM:
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=VOCAB_SIZE, dim=32, n_layers=1, n_heads=2, n_kv_heads=1,
            multiple_of=8, norm_eps=1e-5, max_batch_size=8, max_seq_len=32,
        )
    )


# -- sampling: one sort, one draw, many temperatures ----------------------


def test_the_sweep_matches_the_canonical_sampler_exactly() -> None:
    """The single most important invariant of this integration.

    The sweep contains the canonical temperature, so with the same logits, the
    same ``top_p``, and the same uniforms it must reproduce the canonical draw
    **exactly** -- not approximately. Both paths share one implementation, so the
    equality is by construction rather than by numerical luck.
    """

    logits = torch.randn(256, 48)
    uniforms = torch.rand(256, generator=torch.Generator().manual_seed(3))

    canonical = nucleus_guess_ids(logits, temperature=CANONICAL, top_p=0.9, uniforms=uniforms)
    swept = nucleus_guess_ids_by_temperature(
        logits, temperatures=SWEEP, top_p=0.9, uniforms=uniforms
    )[CANONICAL]

    assert torch.equal(canonical, swept)


def test_every_temperature_uses_the_same_uniform_draws() -> None:
    """Common random numbers extended to temperature.

    Only the distribution being sampled changes; the stochastic realization is
    held fixed, so a difference between temperatures cannot be sampling noise.
    """

    logits = torch.randn(64, 24)
    uniforms = torch.rand(64, generator=torch.Generator().manual_seed(5))

    first = nucleus_guess_ids_by_temperature(logits, temperatures=SWEEP, top_p=0.9, uniforms=uniforms)
    second = nucleus_guess_ids_by_temperature(logits, temperatures=SWEEP, top_p=0.9, uniforms=uniforms)

    for temperature in SWEEP:
        assert torch.equal(first[temperature], second[temperature])


def test_temperature_does_not_reorder_the_logits() -> None:
    """Why one sort can serve every temperature: T > 0 is a positive scale."""

    logits = torch.randn(32, 40)
    order = torch.argsort(logits, dim=-1, descending=True)

    for temperature in SWEEP:
        assert torch.equal(torch.argsort(logits / temperature, dim=-1, descending=True), order)


def test_the_shared_sort_reproduces_a_per_temperature_sort() -> None:
    """The optimization must not change numerical semantics.

    Sorting the temperature-scaled probabilities directly is the obvious
    implementation; gathering by a single logit ordering must agree with it.
    """

    logits = torch.randn(128, 32)
    uniforms = torch.rand(128, generator=torch.Generator().manual_seed(7))
    swept = nucleus_guess_ids_by_temperature(logits, temperatures=SWEEP, top_p=0.9, uniforms=uniforms)

    for temperature in SWEEP:
        reference = nucleus_guess_ids(
            logits, temperature=temperature, top_p=0.9, uniforms=uniforms
        )
        assert torch.equal(swept[temperature], reference)


def test_a_lower_temperature_agrees_with_greedy_more_often() -> None:
    """A sanity check on direction, not an assertion of monotonicity.

    Compares only the two extremes of the sweep, which is a property of the
    sampler rather than a claim about the transition's shape.
    """

    logits = torch.randn(512, 32)
    uniforms = torch.rand(512, generator=torch.Generator().manual_seed(11))
    greedy = greedy_guess_ids(logits)
    swept = nucleus_guess_ids_by_temperature(logits, temperatures=SWEEP, top_p=0.9, uniforms=uniforms)

    coldest = (swept[SWEEP[0]] == greedy).float().mean()
    hottest = (swept[SWEEP[-1]] == greedy).float().mean()

    assert coldest > hottest


def test_zero_and_negative_temperatures_are_rejected() -> None:
    """Greedy is the T=0 anchor and is computed by argmax, never by softmax."""

    logits = torch.randn(8, 16)
    uniforms = torch.rand(8)

    for bad in (0.0, -0.5):
        with pytest.raises(ValueError, match="positive"):
            nucleus_guess_ids_by_temperature(
                logits, temperatures=(bad,), top_p=0.9, uniforms=uniforms
            )


def test_duplicate_temperatures_are_collapsed() -> None:
    """The same temperature twice is one distribution, not two."""

    logits = torch.randn(16, 16)
    uniforms = torch.rand(16)

    result = nucleus_guess_ids_by_temperature(
        logits, temperatures=(0.6, 0.6, 0.3), top_p=0.9, uniforms=uniforms
    )

    assert sorted(result) == [0.3, 0.6]


def test_an_empty_sweep_is_rejected() -> None:
    """An empty sweep is a configuration mistake, not a no-op."""

    with pytest.raises(ValueError, match="must not be empty"):
        nucleus_guess_ids_by_temperature(
            torch.randn(4, 8), temperatures=(), top_p=0.9, uniforms=torch.rand(4)
        )


# -- streaming: one forward pass for the whole sweep ----------------------


class _CountingModel(LlamaForCausalLM):
    """Records how many forward passes the measurement actually performs."""

    def __init__(self) -> None:
        super().__init__(
            LlamaConfig(
                vocab_size=VOCAB_SIZE, dim=32, n_layers=1, n_heads=2, n_kv_heads=1,
                multiple_of=8, norm_eps=1e-5, max_batch_size=8, max_seq_len=32,
            )
        )
        self.forward_calls = 0

    def forward(self, *args, **kwargs):  # type: ignore[override]
        self.forward_calls += 1
        return super().forward(*args, **kwargs)


def test_the_sweep_adds_no_model_forward_passes() -> None:
    """A hard requirement: temperatures must not multiply the expensive part."""

    seed_everything(2)
    positions = build_evaluation_positions(TOKENS, block_size=8, num_windows=6)

    without = _CountingModel()
    measure_initialization(
        without, positions, model_seed=2, vocab_size=VOCAB_SIZE,
        sampling=_settings(), forward_batch_size=2,
    )
    with_sweep = _CountingModel()
    measure_initialization(
        with_sweep, positions, model_seed=2, vocab_size=VOCAB_SIZE,
        sampling=_settings(), forward_batch_size=2, sweep_temperatures=SWEEP,
    )

    assert without.forward_calls == 3
    assert with_sweep.forward_calls == without.forward_calls


def test_the_sweep_does_not_disturb_the_canonical_measurement() -> None:
    """Enabling the sweep must leave greedy and canonical nucleus untouched."""

    seed_everything(3)
    model = _model()
    positions = build_evaluation_positions(TOKENS, block_size=8, num_windows=4)
    common = {"model_seed": 3, "vocab_size": VOCAB_SIZE, "sampling": _settings()}

    plain = measure_initialization(model, positions, **common)
    swept = measure_initialization(model, positions, sweep_temperatures=SWEEP, **common)

    assert torch.equal(plain.greedy_counts, swept.greedy_counts)
    assert torch.equal(plain.nucleus_counts, swept.nucleus_counts)


def test_the_streamed_sweep_reproduces_the_canonical_counts_at_that_temperature() -> None:
    """End-to-end version of the exact-equality invariant."""

    seed_everything(4)
    model = _model()
    positions = build_evaluation_positions(TOKENS, block_size=8, num_windows=5)

    measurement = measure_initialization(
        model, positions, model_seed=4, vocab_size=VOCAB_SIZE,
        sampling=_settings(), sweep_temperatures=SWEEP,
    )

    index = measurement.sweep_temperatures.index(CANONICAL)
    assert torch.equal(measurement.sweep_counts[index], measurement.nucleus_counts[0])


def test_sweep_counters_have_one_row_per_temperature() -> None:
    """Shape contract for the persisted arrays."""

    seed_everything(5)
    model = _model()
    positions = build_evaluation_positions(TOKENS, block_size=8, num_windows=4)

    measurement = measure_initialization(
        model, positions, model_seed=5, vocab_size=VOCAB_SIZE,
        sampling=_settings(), sweep_temperatures=SWEEP,
    )

    assert measurement.sweep_counts.shape == (len(SWEEP), VOCAB_SIZE)
    assert measurement.sweep_agreement.shape == (len(SWEEP),)
    assert measurement.sweep_temperatures == SWEEP
    for row in measurement.sweep_counts:
        assert int(row.sum()) == positions.num_positions


def test_the_sweep_result_is_independent_of_the_forward_batch_size() -> None:
    """Streaming is a memory decision; the measurement is not."""

    seed_everything(6)
    model = _model()
    positions = build_evaluation_positions(TOKENS, block_size=8, num_windows=6)

    results = [
        measure_initialization(
            model, positions, model_seed=6, vocab_size=VOCAB_SIZE,
            sampling=_settings(), forward_batch_size=size, sweep_temperatures=SWEEP,
        )
        for size in (1, 3, 6)
    ]

    for other in results[1:]:
        assert torch.equal(results[0].sweep_counts, other.sweep_counts)
        assert torch.allclose(results[0].sweep_agreement, other.sweep_agreement)


def test_the_sweep_respects_the_eligible_support() -> None:
    """Structural tokens stay unreachable at every temperature."""

    seed_everything(7)
    model = _model()
    positions = build_evaluation_positions(TOKENS, block_size=8, num_windows=4)

    measurement = measure_initialization(
        model, positions, model_seed=7, vocab_size=VOCAB_SIZE, sampling=_settings(),
        eligible_token_ids=list(range(3, VOCAB_SIZE)), sweep_temperatures=SWEEP,
    )

    assert int(measurement.sweep_counts[:, :3].sum()) == 0


def test_greedy_agreement_is_a_fraction_of_evaluated_positions() -> None:
    """It is the identity-preserving diagnostic, so it must be a real fraction."""

    seed_everything(8)
    model = _model()
    positions = build_evaluation_positions(TOKENS, block_size=8, num_windows=4)

    agreement = measure_initialization(
        model, positions, model_seed=8, vocab_size=VOCAB_SIZE,
        sampling=_settings(), sweep_temperatures=SWEEP,
    ).sweep_agreement

    assert torch.all(agreement >= 0.0) and torch.all(agreement <= 1.0)


def test_no_sweep_leaves_the_measurement_unchanged() -> None:
    """Historical behaviour when the sweep is absent."""

    seed_everything(9)
    model = _model()
    positions = build_evaluation_positions(TOKENS, block_size=8, num_windows=4)

    measurement = measure_initialization(
        model, positions, model_seed=9, vocab_size=VOCAB_SIZE, sampling=_settings()
    )

    assert measurement.sweep_counts is None
    assert measurement.sweep_agreement is None
    assert measurement.sweep_temperatures == ()


# -- configuration --------------------------------------------------------


def _resolve_protocol():
    """Load the script's protocol resolver without importing the whole script."""

    path = REPO_ROOT / "scripts" / "run_initialization_distribution_experiment.py"
    spec = importlib.util.spec_from_file_location("initialization_distribution_script", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module._resolve_protocol, module._resolve_temperatures


def _args(**overrides) -> argparse.Namespace:
    fields = {
        "num_initializations": None, "num_windows": None, "block_size": None,
        "num_replicates": None, "split": None, "forward_batch_size": None,
        "temperatures": None, "no_temperature_sweep": False,
        "no_uniform_null": False, "no_input_structure": False,
        "gradient_analysis": False, "no_gradient_analysis": False,
        "gradient_windows": None,
    }
    fields.update(overrides)
    return argparse.Namespace(**fields)


def test_a_config_without_a_sweep_block_disables_it() -> None:
    """Historical configs must run exactly as before."""

    resolve, _ = _resolve_protocol()

    protocol = resolve({"sampling": {"temperature": 0.6}}, _args())

    assert protocol["temperature_sweep_enabled"] is False
    assert protocol["sweep_temperatures"] == ()
    assert protocol["temperature"] == 0.6


def test_the_sweep_can_be_disabled_in_the_config() -> None:
    """Enabled is an explicit switch, not an inference from the list."""

    resolve, _ = _resolve_protocol()

    protocol = resolve(
        {"temperature_sweep": {"enabled": False, "temperatures": [0.3]}}, _args()
    )

    assert protocol["temperature_sweep_enabled"] is False


def test_the_shipped_config_enables_the_documented_sweep() -> None:
    """The first sweep is the one recorded in the experiment log."""

    import yaml

    config = yaml.safe_load(
        (REPO_ROOT / "configs" / "experiment" / "initialization_distribution.yaml").read_text(
            encoding="utf-8"
        )
    )
    resolve, _ = _resolve_protocol()

    protocol = resolve(config, _args())

    assert protocol["temperature_sweep_enabled"] is True
    assert protocol["sweep_temperatures"] == SWEEP
    # The canonical policy is untouched by the sweep.
    assert protocol["temperature"] == CANONICAL
    assert protocol["top_p"] == 0.9
    assert protocol["num_replicates"] == 1


def test_the_command_line_overrides_the_configured_sweep() -> None:
    """And enables it even when the config left it off."""

    resolve, _ = _resolve_protocol()

    protocol = resolve(
        {"temperature_sweep": {"enabled": False, "temperatures": [0.3]}},
        _args(temperatures=[0.2, 0.9]),
    )

    assert protocol["temperature_sweep_enabled"] is True
    assert protocol["sweep_temperatures"] == (0.2, 0.9)


def test_the_command_line_can_disable_the_sweep() -> None:
    """Even when the config asks for it."""

    import yaml

    config = yaml.safe_load(
        (REPO_ROOT / "configs" / "experiment" / "initialization_distribution.yaml").read_text(
            encoding="utf-8"
        )
    )
    resolve, _ = _resolve_protocol()

    protocol = resolve(config, _args(no_temperature_sweep=True))

    assert protocol["temperature_sweep_enabled"] is False


def test_temperature_validation_rejects_non_positive_values() -> None:
    """Zero is not a valid stochastic temperature; greedy covers T=0."""

    _, resolve_temperatures = _resolve_protocol()

    for bad in ([0.0], [-1.0], [0.3, 0.0]):
        with pytest.raises(ValueError, match="strictly positive"):
            resolve_temperatures(bad)


def test_temperature_validation_rejects_non_numeric_values() -> None:
    """A typo in YAML must fail loudly rather than silently drop a temperature."""

    _, resolve_temperatures = _resolve_protocol()

    with pytest.raises(ValueError, match="non-numeric"):
        resolve_temperatures(["warm"])


def test_temperatures_are_deduplicated_in_the_given_order() -> None:
    """User order is preserved so a config reads back as written."""

    _, resolve_temperatures = _resolve_protocol()

    assert resolve_temperatures([0.6, 0.12, 0.6, 0.24]) == (0.6, 0.12, 0.24)


def test_a_single_temperature_sweep_is_valid() -> None:
    """Useful for a quick check, and must not be special-cased away."""

    _, resolve_temperatures = _resolve_protocol()

    assert resolve_temperatures([0.35]) == (0.35,)


def test_the_sweep_need_not_contain_the_canonical_temperature() -> None:
    """It is additional analysis, not a redefinition of the canonical policy."""

    resolve, _ = _resolve_protocol()

    protocol = resolve(
        {"sampling": {"temperature": 0.6},
         "temperature_sweep": {"enabled": True, "temperatures": [0.2, 1.0]}},
        _args(),
    )

    assert CANONICAL not in protocol["sweep_temperatures"]
    assert protocol["temperature"] == CANONICAL


# -- transition metrics ---------------------------------------------------


def _sweep_record(num_initializations: int = 3) -> InitializationExperimentRecord:
    """A small record carrying a sweep, a null, and all three conditions."""

    vocab_size, draws = 40, 200
    eligible = np.arange(2, vocab_size)
    rng = np.random.default_rng(17)
    weights = rng.dirichlet(np.ones(eligible.size))
    corpus = np.zeros(vocab_size, dtype=np.int64)
    corpus[eligible] = rng.multinomial(5000, weights)
    selected = np.zeros(vocab_size, dtype=np.int64)
    selected[eligible] = rng.multinomial(draws, weights)

    greedy = np.zeros((num_initializations, vocab_size), dtype=np.int64)
    nucleus = np.zeros((num_initializations, 1, vocab_size), dtype=np.int64)
    conditions = {c: np.zeros((num_initializations, vocab_size), dtype=np.int64)
                  for c in ("shuffled", "gaussian")}
    condition_nucleus = {c: np.zeros((num_initializations, 1, vocab_size), dtype=np.int64)
                         for c in ("shuffled", "gaussian")}
    sweeps = {c: np.zeros((num_initializations, len(SWEEP), vocab_size), dtype=np.int64)
              for c in ("real", "shuffled", "gaussian")}
    agreement = {c: np.zeros((num_initializations, len(SWEEP))) for c in sweeps}
    for index in range(num_initializations):
        greedy[index, eligible] = rng.multinomial(draws, rng.dirichlet(np.ones(eligible.size) * 0.2))
        nucleus[index][:, eligible] = rng.multinomial(draws, weights, size=1)
        for name in conditions:
            conditions[name][index, eligible] = rng.multinomial(draws, weights)
            condition_nucleus[name][index][:, eligible] = rng.multinomial(draws, weights, size=1)
        for name in sweeps:
            for position, temperature in enumerate(SWEEP):
                sweeps[name][index, position, eligible] = rng.multinomial(draws, weights)
                agreement[name][index, position] = max(0.0, 0.9 - 0.7 * temperature)

    # The canonical temperature is NOT an independent draw. In a real run the
    # sweep samples the same logits with the same uniforms as the canonical
    # policy, so that slot holds bitwise exactly the canonical nucleus counts --
    # that is the invariant the figure-one/figure-five agreement test exists to
    # check. Drawing it independently above would make the fixture violate the
    # very property under test, so it is overwritten here. Every other
    # temperature stays independently synthetic.
    canonical_index = SWEEP.index(CANONICAL)
    sweeps["real"][:, canonical_index, :] = nucleus[:, 0, :]
    for name, counts in condition_nucleus.items():
        sweeps[name][:, canonical_index, :] = counts[:, 0, :]

    null = simulate_uniform_null(
        eligible_vocab_size=eligible.size, num_draws=draws, num_replicates=16, seed=2
    )
    return InitializationExperimentRecord.build(
        corpus_counts=corpus, selected_target_counts=selected,
        greedy_counts=greedy, nucleus_counts=nucleus,
        mean_predicted_probabilities=np.tile(corpus / corpus.sum(), (num_initializations, 1)),
        model_seeds=np.arange(num_initializations), eligible_token_ids=eligible,
        condition_greedy_counts=conditions, condition_nucleus_counts=condition_nucleus,
        uniform_null={"ranked_mean": null.ranked_mean, "ranked_low": null.ranked_low,
                      "ranked_high": null.ranked_high},
        sweep_counts_by_condition=sweeps, sweep_agreement_by_condition=agreement,
        metadata={"num_positions": draws, "tokens": [f"t{i}" for i in range(vocab_size)],
                  "uniform_null": null.as_dict(),
                  "analysis": {"sampling": {"temperature": CANONICAL, "top_p": 0.9},
                               "temperature_sweep": {"enabled": True, "temperatures": list(SWEEP)}}},
    )


def test_ranked_distance_to_greedy_is_zero_for_the_same_shape() -> None:
    """It compares concentration, not token identity.

    A distribution that is the greedy one with its token labels permuted has an
    identical ranked profile and therefore distance zero. That is the intended
    meaning and the reason the name says ``rank``.
    """

    greedy = np.array([0.6, 0.3, 0.1, 0.0])
    permuted = np.array([0.1, 0.6, 0.0, 0.3])

    assert ranked_distance_to_greedy(permuted, greedy) == pytest.approx(0.0)


def test_ranked_distance_to_greedy_is_hand_computable() -> None:
    """Half the summed absolute difference of the two ranked profiles."""

    greedy = np.array([1.0, 0.0, 0.0, 0.0])
    spread = np.array([0.25, 0.25, 0.25, 0.25])

    assert ranked_distance_to_greedy(spread, greedy) == pytest.approx(0.75)


def test_ranked_distance_to_uniform_is_hand_computable() -> None:
    """Against an already-ranked null profile."""

    null_ranked = np.array([0.4, 0.3, 0.2, 0.1])

    assert ranked_distance_to_uniform(np.array([0.4, 0.3, 0.2, 0.1]), null_ranked) == pytest.approx(0.0)
    assert ranked_distance_to_uniform(np.array([1.0, 0.0, 0.0, 0.0]), null_ranked) == pytest.approx(0.6)


def test_ranked_distances_reject_a_length_mismatch() -> None:
    """Both profiles must span the eligible support."""

    with pytest.raises(ValueError, match="eligible support"):
        ranked_distance_to_uniform(np.array([0.5, 0.5]), np.array([0.4, 0.3, 0.3]))


def test_the_condition_summary_covers_every_temperature_and_metric() -> None:
    """Array alignment: one value per temperature, for each metric."""

    summary = sweep_condition_summary(_sweep_record(), "real")

    assert summary["available"] is True
    assert summary["temperatures"] == list(SWEEP)
    for name in TRANSITION_METRICS:
        assert len(summary["metrics"][name]["mean"]) == len(SWEEP)
        assert len(summary["metrics"][name]["sem"]) == len(SWEEP)


def test_the_summary_reports_agreement_from_the_record() -> None:
    """Agreement is accumulated during streaming, not recomputed here."""

    summary = sweep_condition_summary(_sweep_record(), "real")

    observed = summary["metrics"]["agreement_with_greedy"]["mean"]
    expected = [max(0.0, 0.9 - 0.7 * temperature) for temperature in SWEEP]
    assert observed == pytest.approx(expected)


def test_effective_support_is_normalized_by_the_null() -> None:
    """The ratio is the readable form: 1.0 means as broad as pure chance.

    The denominator is the effective support of the *ranked null profile*, which
    is a finite-D occupancy reference rather than a latent model distribution.
    """

    from llm_behavior_lab.analysis import effective_support

    record = _sweep_record()
    summary = sweep_condition_summary(record, "real")
    null_support = effective_support(record.uniform_null["ranked_mean"])

    ratios = summary["metrics"]["effective_support_over_null"]["mean"]
    supports = summary["metrics"]["effective_support"]["mean"]

    assert null_support > 0
    for support, ratio in zip(supports, ratios):
        assert ratio == pytest.approx(support / null_support)


def test_the_summary_spans_every_input_condition() -> None:
    """So the transition can be compared across input structure."""

    summary = sweep_summary(_sweep_record())

    assert summary["available"] is True
    assert set(summary["conditions"]) == {"real", "shuffled", "gaussian"}
    assert summary["canonical_temperature"] == CANONICAL
    assert summary["top_p"] == 0.9


def test_sem_is_taken_across_initializations() -> None:
    """One initialization has no estimable spread; several do."""

    single = sweep_condition_summary(_sweep_record(num_initializations=1), "real")
    several = sweep_condition_summary(_sweep_record(num_initializations=4), "real")

    assert all(value == 0.0 for value in single["metrics"]["effective_support"]["sem"])
    assert any(value > 0.0 for value in several["metrics"]["effective_support"]["sem"])


def test_a_record_without_a_sweep_reports_it_as_unavailable() -> None:
    """Historical records must load and simply say there is nothing to show."""

    record = _sweep_record()
    bare = InitializationExperimentRecord.build(
        corpus_counts=record.corpus_counts,
        selected_target_counts=record.selected_target_counts,
        greedy_counts=record.greedy_counts,
        nucleus_counts=record.nucleus_counts,
        mean_predicted_probabilities=record.mean_predicted_probabilities,
        model_seeds=record.model_seeds,
        eligible_token_ids=record.eligible_token_ids,
        metadata={"num_positions": 200},
    )

    assert bare.has_temperature_sweep is False
    assert sweep_summary(bare) == {"available": False, "conditions": {}}


def test_sweep_arrays_round_trip(tmp_path) -> None:
    """Counts and agreement must survive persistence for a later replot."""

    from llm_behavior_lab.analysis import load_record

    record = _sweep_record()
    record.save(tmp_path)

    reloaded = load_record(tmp_path)

    assert reloaded.has_temperature_sweep
    assert reloaded.sweep_temperatures == SWEEP
    assert np.array_equal(reloaded.sweep_counts("gaussian"), record.sweep_counts("gaussian"))
    assert np.allclose(reloaded.sweep_agreement("real"), record.sweep_agreement("real"))


def test_the_sweep_shares_one_sort_in_the_implementation() -> None:
    """Documented optimization, asserted on the parsed source.

    One ``argsort`` for the whole sweep; a sort inside the temperature loop
    would silently restore the per-temperature cost.
    """

    from llm_behavior_lab.evaluation import guessing

    tree = ast.parse(Path(guessing.__file__).read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "nucleus_guess_ids_by_temperature"
    )
    loops = [node for node in ast.walk(function) if isinstance(node, ast.For)]
    sorts_in_loops = [
        node for loop in loops for node in ast.walk(loop)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"sort", "argsort"}
    ]

    assert not sorts_in_loops


# -- ranked-profile aggregation order -------------------------------------
#
# Figure 1 and figure 5 panel A must answer the same question. Figure 5 once
# averaged by token identity across initializations and ranked afterwards, which
# is a different quantity and visibly flattened the high-temperature curves.

from llm_behavior_lab.analysis import ranked_profile_with_error  # noqa: E402
from llm_behavior_lab.analysis.aggregation import ranked_profile, ranked_profiles  # noqa: E402


def test_ranking_and_averaging_do_not_commute() -> None:
    """The whole reason the convention has to be stated explicitly.

    Two initializations place their spike on *different* tokens. Ranking first
    preserves the spike in the profile; averaging by identity first splits it
    across two tokens and flattens it.
    """

    guesses = np.array([[0.8, 0.2, 0.0], [0.0, 0.2, 0.8]])

    rank_first = ranked_profile_with_error(guesses).mean
    average_first = ranked_profile(guesses.mean(axis=0))

    assert rank_first.tolist() == pytest.approx([0.8, 0.2, 0.0])
    assert average_first.tolist() == pytest.approx([0.4, 0.4, 0.2])
    assert not np.allclose(rank_first, average_first)


def test_the_helper_ranks_within_each_realization_first() -> None:
    """``mean_s(sort(q_s))``, the authoritative convention."""

    guesses = np.array([[0.1, 0.9], [0.7, 0.3]])

    profile = ranked_profile_with_error(guesses)

    assert profile.mean.tolist() == pytest.approx([0.8, 0.2])
    assert np.allclose(profile.mean, ranked_profiles(guesses).mean(axis=0))


def test_the_helper_reports_sem_across_realizations() -> None:
    """SEM is taken rank by rank, after ranking."""

    guesses = np.array([[0.9, 0.1], [0.7, 0.3]])

    profile = ranked_profile_with_error(guesses)

    assert profile.num_samples == 2
    # std(ddof=1) of [0.9, 0.7] is 0.1*sqrt(2); dividing by sqrt(2) leaves 0.1.
    assert profile.sem.tolist() == pytest.approx([0.1, 0.1])


def test_figure_one_and_figure_five_agree_at_the_canonical_temperature() -> None:
    """The strongest check that the two figures now plot the same quantity.

    The sweep entry at the canonical temperature holds the same counts as the
    canonical nucleus policy, so their ranked profiles must match exactly -- mean
    and SEM alike -- once both use rank-first aggregation.
    """

    from llm_behavior_lab.analysis import eligible_view

    record = _sweep_record(num_initializations=4)
    index = list(record.sweep_temperatures).index(CANONICAL)

    # Figure 1 builds its nucleus curve from the canonical counts.
    canonical = eligible_view(record, record.nucleus_fractions)
    figure_one = ranked_profile_with_error(canonical)

    # Figure 5 panel A builds its T=0.6 curve from the sweep counts.
    sweep_counts = record.sweep_counts("real").astype(np.float64)
    sweep_fractions = sweep_counts / sweep_counts.sum(axis=-1, keepdims=True)
    figure_five = ranked_profile_with_error(eligible_view(record, sweep_fractions)[:, index, :])

    assert np.array_equal(
        record.sweep_counts("real")[:, index, :], record.nucleus_counts[:, 0, :]
    )
    assert np.allclose(figure_one.mean, figure_five.mean, rtol=0, atol=0)
    assert np.allclose(figure_one.sem, figure_five.sem, rtol=0, atol=0)


def test_the_uniform_null_profile_is_already_rank_first() -> None:
    """The null averages ranked realizations, so re-ranking it is a no-op.

    Each Monte Carlo replicate is ranked before the mean is taken, exactly the
    convention the figures use, so the stored profile is already non-increasing.
    """

    from llm_behavior_lab.analysis import simulate_uniform_null

    null = simulate_uniform_null(
        eligible_vocab_size=64, num_draws=200, num_replicates=32, seed=5
    )

    assert np.all(np.diff(null.ranked_mean) <= 1e-15)
    assert np.allclose(ranked_profile(null.ranked_mean), null.ranked_mean)


def test_the_transition_metrics_were_already_rank_first() -> None:
    """They compare per-initialization profiles, so the bug never reached them.

    Recomputing a metric from per-initialization rows must reproduce what the
    summary reports.
    """

    from llm_behavior_lab.analysis import eligible_view

    record = _sweep_record(num_initializations=3)
    summary = sweep_condition_summary(record, "real")

    counts = record.sweep_counts("real").astype(np.float64)
    fractions = eligible_view(record, counts / counts.sum(axis=-1, keepdims=True))
    greedy = eligible_view(record, record.greedy_fractions)
    expected = float(
        np.mean([
            ranked_distance_to_greedy(fractions[s, 0], greedy[s])
            for s in range(record.num_initializations)
        ])
    )

    assert summary["metrics"]["tv_rank_to_greedy"]["mean"][0] == pytest.approx(expected)
