"""Basic import tests for the Phase 1 package skeleton."""

from llm_behavior_lab import __version__
from llm_behavior_lab.models import list_models
from llm_behavior_lab.utils import get_device, seed_everything


def test_package_imports() -> None:
    """The package should expose core modules without import errors."""

    assert isinstance(__version__, str)
    assert callable(list_models)
    assert callable(get_device)
    assert callable(seed_everything)
