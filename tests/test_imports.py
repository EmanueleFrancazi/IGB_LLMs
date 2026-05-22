"""Basic import tests for the Phase 5 package skeleton."""

from llm_behavior_lab import __version__
from llm_behavior_lab.data import CharTokenizer, CausalLMBatcher
from llm_behavior_lab.evaluation import analyze_untrained_outputs, logits_to_probabilities
from llm_behavior_lab.inference import generate_text, prepare_prompt_tensor
from llm_behavior_lab.models import list_models
from llm_behavior_lab.utils import get_device, seed_everything


def test_package_imports() -> None:
    """The package should expose core modules without import errors."""

    assert isinstance(__version__, str)
    assert callable(list_models)
    assert callable(get_device)
    assert callable(seed_everything)
    assert CharTokenizer is not None
    assert CausalLMBatcher is not None
    assert callable(prepare_prompt_tensor)
    assert callable(generate_text)
    assert callable(logits_to_probabilities)
    assert callable(analyze_untrained_outputs)
