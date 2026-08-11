"""Tests for Hugging Face acquisition, driven entirely by a fake loader.

No test may reach the network, and the optional ``datasets`` package may or may
not be installed. Both concerns are handled by patching the single function
through which the loader is imported, so these tests exercise the acquisition
path without any external service and without depending on the environment.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from llm_behavior_lab.data import (
    HUGGINGFACE_SOURCE,
    ROUTE_ACQUIRED,
    ROUTE_SOURCE_CACHE,
    TEXT_FILENAME,
    DatasetAcquisitionError,
    DatasetConfig,
    DatasetNotAvailableError,
    ResolutionPolicy,
    hf_cache_directory,
    materialize,
    prepared_directory,
    resolve_dataset,
    write_prepared,
)
from llm_behavior_lab.data import huggingface
from llm_behavior_lab.data.huggingface import (
    check_dependency_environment,
    concatenate_text_examples,
    describe_source,
)

# Captured before the suite-wide fixture replaces it, so the real import
# behavior can still be exercised deliberately.
REAL_IMPORT_LOAD_DATASET = huggingface._import_load_dataset

ROWS = [{"text": "first document"}, {"text": ""}, {"text": "second document"}]


def _dataset(**overrides) -> DatasetConfig:
    fields = {
        "name": "wikitext2",
        "source": HUGGINGFACE_SOURCE,
        "repo_id": "Salesforce/wikitext",
        "subset": "wikitext-2-raw-v1",
        "split": "train[:10]",
    }
    fields.update(overrides)
    return DatasetConfig(**fields)


def install_fake_loader(monkeypatch, rows=None, fail_offline: bool = False) -> list[dict]:
    """Replace the loader import with a fake and record every call."""

    calls: list[dict] = []

    def load_dataset(repo_id, subset=None, **kwargs):
        offline = os.environ.get("HF_HUB_OFFLINE") == "1"
        calls.append(
            {"repo_id": repo_id, "subset": subset, "kwargs": kwargs, "offline": offline}
        )
        if fail_offline and offline:
            raise RuntimeError("nothing cached")
        return list(ROWS if rows is None else rows)

    monkeypatch.setattr(huggingface, "_import_load_dataset", lambda: load_dataset)
    monkeypatch.setattr(huggingface, "datasets_available", lambda: True)
    return calls


@pytest.fixture
def fake_datasets(monkeypatch):
    """A fake loader that always succeeds, recording how it was called."""

    return install_fake_loader(monkeypatch)


def test_datasets_is_imported_lazily() -> None:
    """Importing the project must never import the optional dependency."""

    source = Path(huggingface.__file__).read_text(encoding="utf-8")
    import_line = "from datasets import load_dataset"

    assert import_line in source
    for line in source.splitlines():
        if import_line in line:
            assert line.startswith("        "), "loader import must stay inside a function"


def test_missing_dependency_produces_an_install_hint(monkeypatch, tmp_path) -> None:
    """Asking for an external dataset without the package must say what to install."""

    monkeypatch.setattr(huggingface, "_import_load_dataset", REAL_IMPORT_LOAD_DATASET)
    monkeypatch.setitem(__import__("sys").modules, "datasets", None)
    with pytest.raises(DatasetAcquisitionError, match=r'pip install -e ".\[hf\]"'):
        materialize(
            _dataset(),
            tmp_path / "dest",
            cache_dir=tmp_path / "hf",
            allow_network=True,
        )


def test_materialize_writes_a_prepared_dataset(fake_datasets, tmp_path) -> None:
    """Acquisition produces a prepared directory that later runs can reuse."""

    destination = prepared_directory(tmp_path, "wikitext2")

    manifest = materialize(
        _dataset(),
        destination,
        cache_dir=tmp_path / "hf",
        allow_network=True,
        prepared_by="unit test",
    )

    assert (destination / TEXT_FILENAME).read_text(encoding="utf-8") == (
        "first document\n\nsecond document"
    )
    assert manifest["num_documents"] == 2
    assert manifest["prepared_by"] == "unit test"
    assert fake_datasets[0]["repo_id"] == "Salesforce/wikitext"
    assert fake_datasets[0]["subset"] == "wikitext-2-raw-v1"
    assert fake_datasets[0]["kwargs"]["split"] == "train[:10]"
    assert fake_datasets[0]["kwargs"]["cache_dir"] == str(tmp_path / "hf")


def test_revision_is_passed_through_when_pinned(fake_datasets, tmp_path) -> None:
    """A pinned revision must reach the loader so runs stay reproducible."""

    materialize(
        _dataset(revision="abc123"),
        prepared_directory(tmp_path, "wikitext2"),
        cache_dir=tmp_path / "hf",
        allow_network=True,
    )

    assert fake_datasets[0]["kwargs"]["revision"] == "abc123"


def test_acquisition_is_announced_before_the_network_is_used(
    fake_datasets, tmp_path
) -> None:
    """The user must see what is being fetched, from where, and to where."""

    messages: list[str] = []

    materialize(
        _dataset(),
        prepared_directory(tmp_path, "wikitext2"),
        cache_dir=tmp_path / "hf",
        allow_network=True,
        reporter=messages.append,
    )

    announcement = messages[0]
    assert "Acquiring dataset 'wikitext2'" in announcement
    assert "Salesforce/wikitext" in announcement
    assert str(tmp_path / "hf") in announcement
    assert str(prepared_directory(tmp_path, "wikitext2")) in announcement
    assert "--offline" in announcement
    # The announcement is emitted before load_dataset runs.
    assert messages.index(announcement) == 0


def test_cache_reuse_sets_hub_offline_and_never_announces_a_download(
    fake_datasets, tmp_path
) -> None:
    """Reading a cached copy must be incapable of reaching the network."""

    messages: list[str] = []

    materialize(
        _dataset(),
        prepared_directory(tmp_path, "wikitext2"),
        cache_dir=tmp_path / "hf",
        allow_network=False,
        reporter=messages.append,
    )

    assert fake_datasets[0]["offline"] is True
    assert not any("Acquiring" in message for message in messages)


def test_hub_offline_is_restored_after_a_cache_read(fake_datasets, tmp_path, monkeypatch) -> None:
    """The environment must not be left modified for the rest of the process."""

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)

    materialize(
        _dataset(),
        prepared_directory(tmp_path, "wikitext2"),
        cache_dir=tmp_path / "hf",
        allow_network=False,
    )

    assert "HF_HUB_OFFLINE" not in os.environ


def test_resolution_uses_the_cache_when_downloads_are_forbidden(
    fake_datasets, tmp_path
) -> None:
    """A cached copy satisfies no-download and offline modes."""

    policy = ResolutionPolicy(data_root=tmp_path, offline=True)

    resolved = resolve_dataset(_dataset(), policy, reporter=None)

    assert resolved.route == ROUTE_SOURCE_CACHE
    assert fake_datasets[0]["offline"] is True
    assert resolved.read_text().startswith("first document")


def test_resolution_acquires_when_permitted_and_cache_is_empty(tmp_path, monkeypatch) -> None:
    """With no cached copy and permission granted, acquisition runs."""

    calls = install_fake_loader(monkeypatch, fail_offline=True)

    policy = ResolutionPolicy(data_root=tmp_path, allow_download=True)
    resolved = resolve_dataset(_dataset(), policy, reporter=None)

    assert [call["offline"] for call in calls] == [True, False]
    assert resolved.route == ROUTE_ACQUIRED
    assert resolved.read_text().startswith("first document")


def test_no_download_refuses_to_acquire_when_cache_is_empty(tmp_path, monkeypatch) -> None:
    """Forbidding downloads must stop before any networked call."""

    def always_fails(repo_id, subset=None, **kwargs):
        calls.append(os.environ.get("HF_HUB_OFFLINE") == "1")
        raise RuntimeError("nothing cached")

    calls: list[bool] = []
    monkeypatch.setattr(huggingface, "_import_load_dataset", lambda: always_fails)
    monkeypatch.setattr(huggingface, "datasets_available", lambda: True)

    with pytest.raises(DatasetNotAvailableError) as excinfo:
        resolve_dataset(_dataset(), ResolutionPolicy(data_root=tmp_path), reporter=None)

    assert calls == [True]  # only the offline cache probe ran
    message = str(excinfo.value)
    assert "downloads are not permitted" in message
    assert "prepare_dataset.py" in message
    assert "--download" not in message


def test_offline_never_permits_acquisition(tmp_path, monkeypatch) -> None:
    """Offline forbids the network even when downloads are allowed."""

    calls = install_fake_loader(monkeypatch, fail_offline=True)

    policy = ResolutionPolicy(data_root=tmp_path, allow_download=True, offline=True)
    with pytest.raises(DatasetNotAvailableError, match="offline mode is enabled"):
        resolve_dataset(_dataset(), policy, reporter=None)

    assert [call["offline"] for call in calls] == [True]


def test_force_refresh_replaces_existing_prepared_data(tmp_path, monkeypatch) -> None:
    """A refresh must not be shadowed by the prepared copy it means to replace."""

    dataset = _dataset()
    destination = prepared_directory(tmp_path, dataset.name)
    write_prepared(destination, "stale corpus", dataset)

    install_fake_loader(monkeypatch, rows=[{"text": "fresh corpus"}], fail_offline=True)

    policy = ResolutionPolicy(data_root=tmp_path, allow_download=True, force_refresh=True)
    resolved = resolve_dataset(dataset, policy, reporter=None)

    assert resolved.route == ROUTE_ACQUIRED
    assert resolved.read_text() == "fresh corpus"


def test_incomplete_prepared_directory_is_replaced_not_fatal(tmp_path, monkeypatch) -> None:
    """An interrupted attempt must not block a later successful preparation."""

    dataset = _dataset()
    destination = prepared_directory(tmp_path, dataset.name)
    destination.mkdir(parents=True)
    (destination / TEXT_FILENAME).write_text("half written", encoding="utf-8")

    install_fake_loader(monkeypatch, rows=[{"text": "complete corpus"}])

    resolved = resolve_dataset(
        dataset, ResolutionPolicy(data_root=tmp_path, allow_download=True), reporter=None
    )

    assert resolved.read_text() == "complete corpus"


def test_missing_dependency_is_reported_among_checked_locations(tmp_path, monkeypatch) -> None:
    """Without the optional package the error should say so plainly."""

    monkeypatch.setattr(huggingface, "datasets_available", lambda: False)
    with pytest.raises(DatasetNotAvailableError) as excinfo:
        resolve_dataset(
            _dataset(),
            ResolutionPolicy(data_root=tmp_path, allow_download=True),
            reporter=None,
        )

    assert "'datasets' package" in str(excinfo.value)


def test_local_text_never_touches_huggingface(tmp_path, monkeypatch) -> None:
    """The tiny fixture path must not import or call the optional dependency."""

    def explode():
        raise AssertionError("the loader must not be imported for local_text")

    monkeypatch.setattr(huggingface, "_import_load_dataset", explode)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("tiny", encoding="utf-8")

    resolved = resolve_dataset(
        DatasetConfig(name="tiny_local_text", path=str(corpus)),
        ResolutionPolicy(data_root=tmp_path, allow_download=True),
        reporter=None,
    )

    assert resolved.read_text() == "tiny"


@pytest.mark.parametrize(
    ("kwargs", "expected", "documents"),
    [
        ({}, "first document\n\nsecond document", 2),
        ({"max_examples": 1}, "first document", 1),
        ({"max_characters": 5}, "first", 1),
        ({"document_separator": " | "}, "first document | second document", 2),
    ],
)
def test_concatenation_honours_limits(kwargs, expected: str, documents: int) -> None:
    """Limits must bound the corpus while iterating, not afterwards."""

    text, used = concatenate_text_examples(
        ROWS, text_field="text", **{"document_separator": "\n\n", **kwargs}
    )

    assert text == expected
    assert used == documents


def test_missing_text_field_is_actionable() -> None:
    """A wrong field name should name the setting to change."""

    with pytest.raises(DatasetAcquisitionError, match="dataset.text_field"):
        concatenate_text_examples([{"content": "x"}], text_field="text")


def test_empty_result_is_rejected() -> None:
    """Silently analysing an empty corpus would be worse than failing."""

    with pytest.raises(DatasetAcquisitionError, match="empty corpus"):
        concatenate_text_examples([{"text": "   "}], text_field="text")


def test_loader_failures_are_wrapped_with_the_source(fake_datasets, tmp_path, monkeypatch) -> None:
    """A loader error should identify which dataset failed."""

    def failing(*args, **kwargs):
        raise RuntimeError("upstream exploded")

    monkeypatch.setattr(huggingface, "_import_load_dataset", lambda: failing)

    with pytest.raises(DatasetAcquisitionError) as excinfo:
        materialize(
            _dataset(),
            prepared_directory(tmp_path, "wikitext2"),
            cache_dir=tmp_path / "hf",
            allow_network=True,
        )

    message = str(excinfo.value)
    assert "Salesforce/wikitext" in message
    assert "upstream exploded" in message


def test_dependency_preflight_flags_incompatible_numpy_and_scipy(monkeypatch) -> None:
    """A known-bad combination should be reported as an environment problem."""

    versions = {"numpy": "2.2.6", "scipy": "1.8.0"}
    monkeypatch.setattr(
        "llm_behavior_lab.data.huggingface._installed_version", versions.get
    )

    with pytest.raises(DatasetAcquisitionError, match="incompatible with NumPy 2"):
        check_dependency_environment()


def test_dependency_preflight_accepts_supported_combinations(monkeypatch) -> None:
    """Ordinary environments must not be rejected."""

    for versions in ({"numpy": "1.26.4", "scipy": "1.8.0"}, {"numpy": "2.2.6", "scipy": "1.14.0"}):
        monkeypatch.setattr(
            "llm_behavior_lab.data.huggingface._installed_version", versions.get
        )
        check_dependency_environment()


def test_cache_directory_lives_under_the_data_root(tmp_path) -> None:
    """All downloaded bytes stay under one configurable root."""

    assert hf_cache_directory(tmp_path) == tmp_path / "hf"


def test_source_description_is_readable() -> None:
    """Messages should identify the dataset the way a user configured it."""

    described = describe_source(_dataset(revision="abc"))

    assert "Salesforce/wikitext" in described
    assert "subset=wikitext-2-raw-v1" in described
    assert "split=train[:10]" in described
    assert "revision=abc" in described


def test_streaming_is_forwarded_to_the_loader(monkeypatch, tmp_path) -> None:
    """Lazy iteration must be requested when the config asks for it."""

    calls = install_fake_loader(monkeypatch)

    materialize(
        _dataset(streaming=True),
        prepared_directory(tmp_path, "wikitext2"),
        cache_dir=tmp_path / "hf",
        allow_network=True,
    )

    assert calls[0]["kwargs"]["streaming"] is True


def test_streaming_is_absent_unless_requested(fake_datasets, tmp_path) -> None:
    """The default must remain ordinary non-streaming loading."""

    materialize(
        _dataset(),
        prepared_directory(tmp_path, "wikitext2"),
        cache_dir=tmp_path / "hf",
        allow_network=True,
    )

    assert "streaming" not in fake_datasets[0]["kwargs"]


def test_streaming_changes_prepared_identity(tmp_path) -> None:
    """A corpus prepared by streaming is not interchangeable with one that was not."""

    from llm_behavior_lab.data.prepared import describe_mismatch

    manifest = {"name": "wikitext2", **{f: getattr(_dataset(), f) for f in ("source",
        "repo_id", "subset", "split", "revision", "text_field", "max_examples",
        "max_characters", "document_separator", "streaming")}}

    assert describe_mismatch(manifest, _dataset()) is None
    assert "streaming" in describe_mismatch(manifest, _dataset(streaming=True))


def test_force_refresh_bypasses_an_available_cache(tmp_path, monkeypatch) -> None:
    """A refresh must reach the source even when a cached copy would satisfy it."""

    calls = install_fake_loader(monkeypatch)  # offline probe would succeed
    dataset = _dataset()
    write_prepared(prepared_directory(tmp_path, dataset.name), "stale", dataset)

    policy = ResolutionPolicy(data_root=tmp_path, allow_download=True, force_refresh=True)
    resolved = resolve_dataset(dataset, policy, reporter=None)

    assert resolved.route == ROUTE_ACQUIRED
    # Only the networked call ran; the cache-only probe was skipped entirely.
    assert [call["offline"] for call in calls] == [False]


def test_force_refresh_requests_a_fresh_download(tmp_path, monkeypatch) -> None:
    """The loader must be told not to reuse its own cache."""

    calls = install_fake_loader(monkeypatch)

    policy = ResolutionPolicy(data_root=tmp_path, allow_download=True, force_refresh=True)
    resolve_dataset(_dataset(), policy, reporter=None)

    assert calls[0]["kwargs"]["download_mode"] == "force_redownload"


def test_ordinary_acquisition_does_not_force_a_download(tmp_path, monkeypatch) -> None:
    """Normal runs must keep reusing the loader's cache."""

    calls = install_fake_loader(monkeypatch, fail_offline=True)

    resolve_dataset(
        _dataset(), ResolutionPolicy(data_root=tmp_path, allow_download=True), reporter=None
    )

    assert "download_mode" not in calls[-1]["kwargs"]


def test_failed_refresh_leaves_the_previous_prepared_copy(tmp_path, monkeypatch) -> None:
    """A refresh that cannot reach the source must not destroy usable data."""

    dataset = _dataset()
    destination = prepared_directory(tmp_path, dataset.name)
    write_prepared(destination, "previous corpus", dataset)

    def failing(*args, **kwargs):
        raise RuntimeError("source unreachable")

    monkeypatch.setattr(huggingface, "_import_load_dataset", lambda: failing)
    monkeypatch.setattr(huggingface, "datasets_available", lambda: True)

    policy = ResolutionPolicy(data_root=tmp_path, allow_download=True, force_refresh=True)
    with pytest.raises(DatasetAcquisitionError):
        resolve_dataset(dataset, policy, reporter=None)

    assert (destination / TEXT_FILENAME).read_text(encoding="utf-8") == "previous corpus"


def test_manifest_records_what_the_loader_returned(monkeypatch, tmp_path) -> None:
    """The manifest must say what came back, not only what was asked for."""

    class FakeInfo:
        version = "1.0.0"
        download_size = 4321
        dataset_size = 8765

    class FakeRows(list):
        info = FakeInfo()
        _fingerprint = "abc123fingerprint"
        config_name = "wikitext-2-raw-v1"
        split = "train"

    monkeypatch.setattr(
        huggingface, "_import_load_dataset", lambda: lambda *a, **k: FakeRows(ROWS)
    )
    monkeypatch.setattr(huggingface, "datasets_available", lambda: True)

    manifest = materialize(
        _dataset(),
        prepared_directory(tmp_path, "wikitext2"),
        cache_dir=tmp_path / "hf",
        allow_network=True,
    )

    resolved = manifest["resolved"]
    assert resolved["fingerprint"] == "abc123fingerprint"
    assert resolved["version"] == "1.0.0"
    assert resolved["config_name"] == "wikitext-2-raw-v1"
    assert resolved["download_size"] == 4321
    assert resolved["dataset_size"] == 8765


def test_resolved_fields_are_none_when_the_loader_exposes_nothing(
    fake_datasets, tmp_path
) -> None:
    """A minimal loader must not break preparation."""

    manifest = materialize(
        _dataset(),
        prepared_directory(tmp_path, "wikitext2"),
        cache_dir=tmp_path / "hf",
        allow_network=True,
    )

    assert manifest["resolved"]["fingerprint"] is None
    assert manifest["resolved"]["version"] is None


def test_resolved_details_never_trigger_a_staleness_mismatch(
    fake_datasets, tmp_path
) -> None:
    """Two preparations may differ upstream without the copy looking stale."""

    from llm_behavior_lab.data.prepared import describe_mismatch

    manifest = materialize(
        _dataset(),
        prepared_directory(tmp_path, "wikitext2"),
        cache_dir=tmp_path / "hf",
        allow_network=True,
    )
    manifest["resolved"]["fingerprint"] = "something-completely-different"

    assert describe_mismatch(manifest, _dataset()) is None


def test_no_hub_lookup_is_performed_for_provenance(fake_datasets, tmp_path) -> None:
    """Recording provenance must not add a network operation."""

    import sys

    assert "huggingface_hub" not in Path(huggingface.__file__).read_text(encoding="utf-8")
    materialize(
        _dataset(),
        prepared_directory(tmp_path, "wikitext2"),
        cache_dir=tmp_path / "hf",
        allow_network=True,
    )
    assert len(fake_datasets) == 1  # exactly one loader call, no extra lookup
