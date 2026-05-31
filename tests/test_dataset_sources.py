"""Tests for config-driven dataset-source loading."""

from __future__ import annotations

import sys
import types

import pytest

from llm_behavior_lab.data import (
    DatasetSourceError,
    concatenate_text_examples,
    load_huggingface_text_from_config,
    load_text_dataset_from_config,
)


def test_load_local_text_dataset_from_config(tmp_path) -> None:
    """Local source configs should load text without external dependencies."""

    data_path = tmp_path / "tiny.txt"
    data_path.write_text("hello local dataset", encoding="utf-8")
    config = {
        "dataset": {
            "source_type": "local_text",
            "path": "tiny.txt",
        }
    }

    loaded = load_text_dataset_from_config(config, repo_root=tmp_path)

    assert loaded.source_type == "local_text"
    assert loaded.text == "hello local dataset"
    assert loaded.num_examples == 1


def test_concatenate_text_examples_respects_limits() -> None:
    """Concatenation should skip empty rows and respect limits."""

    rows = [
        {"text": "alpha"},
        {"text": ""},
        {"text": "beta"},
        {"text": "gamma"},
    ]

    text, num_examples = concatenate_text_examples(
        rows,
        text_field="text",
        max_examples=4,
        max_characters=9,
        document_separator="|",
    )

    assert text == "alpha|beta"
    assert num_examples == 2


def test_huggingface_loader_can_be_mocked(monkeypatch) -> None:
    """The Hugging Face source path should be testable without network access."""

    calls = {}

    def fake_load_dataset(name, subset=None, **kwargs):
        calls["name"] = name
        calls["subset"] = subset
        calls["kwargs"] = kwargs
        return [
            {"text": "first example"},
            {"text": "second example"},
            {"text": "third example"},
        ]

    fake_module = types.SimpleNamespace(load_dataset=fake_load_dataset)
    monkeypatch.setitem(sys.modules, "datasets", fake_module)

    loaded = load_huggingface_text_from_config(
        {
            "source_type": "huggingface",
            "name": "fake/dataset",
            "subset": "fake-subset",
            "split": "train[:2]",
            "text_field": "text",
            "max_examples": 2,
            "streaming": False,
        }
    )

    assert calls["name"] == "fake/dataset"
    assert calls["subset"] == "fake-subset"
    assert calls["kwargs"] == {"split": "train[:2]", "streaming": False}
    assert loaded.source_type == "huggingface"
    assert loaded.source_name == "fake/dataset"
    assert loaded.text == "first example\n\nsecond example"
    assert loaded.num_examples == 2


def test_huggingface_dependency_preflight_reports_numpy_scipy_mismatch(monkeypatch) -> None:
    """A known NumPy 2 plus old SciPy mismatch should produce an actionable error."""

    from llm_behavior_lab.data import sources

    def fake_installed_version(package_name: str) -> str | None:
        versions = {"numpy": "2.2.6", "scipy": "1.8.0"}
        return versions.get(package_name)

    monkeypatch.setattr(sources, "_installed_version", fake_installed_version)

    with pytest.raises(DatasetSourceError, match="NumPy 2.2.6 together with SciPy 1.8.0"):
        sources._check_huggingface_dependency_environment()


def test_huggingface_loader_wraps_numpy_binary_errors(monkeypatch) -> None:
    """Binary compatibility failures from datasets should be wrapped with guidance."""

    from llm_behavior_lab.data import sources

    def fake_load_dataset(*args, **kwargs):
        raise ImportError("numpy.core.multiarray failed to import")

    fake_module = types.SimpleNamespace(load_dataset=fake_load_dataset)
    monkeypatch.setitem(sys.modules, "datasets", fake_module)
    monkeypatch.setattr(sources, "_check_huggingface_dependency_environment", lambda: None)

    with pytest.raises(DatasetSourceError, match="NumPy/SciPy binary"):
        load_huggingface_text_from_config(
            {
                "source_type": "huggingface",
                "name": "fake/dataset",
                "split": "train",
                "text_field": "text",
                "max_examples": 1,
            }
        )
