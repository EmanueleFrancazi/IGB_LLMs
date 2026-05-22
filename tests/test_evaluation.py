"""Tests for Phase 5 baseline-analysis utilities."""

import math

import torch

from llm_behavior_lab.data import CharTokenizer
from llm_behavior_lab.evaluation import (
    analyze_untrained_outputs,
    average_predicted_probabilities,
    empirical_token_counts,
    empirical_token_frequencies,
    entropy_from_probabilities,
    js_divergence,
    kl_divergence,
    logits_to_probabilities,
    summarize_output_distribution,
    top_probability_gaps,
    top1_token_ids,
    topk_probability_mass,
)


def test_logits_to_probabilities_and_entropy() -> None:
    """Probability conversion should normalize logits and support entropy."""

    logits = torch.zeros(2, 3, 4)
    probabilities = logits_to_probabilities(logits)
    entropy = entropy_from_probabilities(probabilities)

    assert probabilities.shape == (2, 3, 4)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(2, 3))
    assert torch.allclose(entropy, torch.full((2, 3), math.log(4)), atol=1e-6)


def test_output_summary_shapes_and_concentration() -> None:
    """Output summaries should report shape and concentration statistics."""

    logits = torch.tensor([[[3.0, 1.0, 0.0], [0.0, 2.0, 1.0]]])
    probabilities = logits_to_probabilities(logits)
    summary = summarize_output_distribution(logits, probabilities, top_k=2)

    assert summary.logits_shape == (1, 2, 3)
    assert summary.probabilities_shape == (1, 2, 3)
    assert summary.num_positions == 2
    assert summary.mean_entropy > 0
    assert summary.mean_top1_probability > 0.5
    assert torch.equal(top1_token_ids(probabilities), torch.tensor([[0, 1]]))
    assert torch.all(topk_probability_mass(probabilities, k=2) <= 1.0)


def test_empirical_frequencies_and_probability_gaps() -> None:
    """Token-frequency helpers should compare predicted and empirical distributions."""

    tokenizer = CharTokenizer.from_text("abca")
    token_ids = tokenizer.encode("abca")
    counts = empirical_token_counts(token_ids, vocab_size=tokenizer.vocab_size)
    frequencies = empirical_token_frequencies(token_ids, vocab_size=tokenizer.vocab_size)
    predicted = torch.tensor([0.2, 0.3, 0.5])

    assert counts.tolist() == [2, 1, 1]
    assert torch.allclose(frequencies, torch.tensor([0.5, 0.25, 0.25]))

    positive_gaps = top_probability_gaps(predicted, frequencies, tokenizer, k=1, largest=True)
    negative_gaps = top_probability_gaps(predicted, frequencies, tokenizer, k=1, largest=False)

    assert positive_gaps[0].token == "c"
    assert negative_gaps[0].token == "a"
    assert kl_divergence(predicted, frequencies) >= 0.0
    assert js_divergence(predicted, frequencies) >= 0.0


def test_analyze_untrained_outputs_returns_expected_sections() -> None:
    """End-to-end analysis should return output, frequency, gap, and top-k sections."""

    tokenizer = CharTokenizer.from_text("abcabc")
    input_ids = torch.tensor([[0, 1, 2], [1, 2, 0]])
    logits = torch.randn(2, 3, 8)
    empirical_ids = tokenizer.encode("abcabc")

    result = analyze_untrained_outputs(
        logits=logits,
        input_ids=input_ids,
        empirical_token_ids=empirical_ids,
        tokenizer=tokenizer,
        top_k=2,
        max_examples=2,
    )

    assert result.output_summary.logits_shape == (2, 3, 8)
    assert result.output_summary.probabilities_shape == (2, 3, tokenizer.vocab_size)
    assert len(result.empirical_top_tokens) == 2
    assert len(result.top1_predicted_tokens) <= 2
    assert len(result.positive_probability_gaps) == 2
    assert len(result.negative_probability_gaps) == 2
    assert len(result.topk_examples) == 2
    assert 0.0 <= result.top1_concentration <= 1.0


def test_average_predicted_probabilities_shape() -> None:
    """Averaging over batch and sequence should leave one value per token."""

    probabilities = torch.full((2, 4, 5), 0.2)
    averaged = average_predicted_probabilities(probabilities)

    assert averaged.shape == (5,)
    assert torch.allclose(averaged, torch.full((5,), 0.2))
