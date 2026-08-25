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
    """Recording provenance must not add a network operation.

    The module does reference huggingface_hub, but only to set offline mode.
    What must not appear is a Hub API query for the resolved revision.
    """

    source = Path(huggingface.__file__).read_text(encoding="utf-8")
    for networked in ("HfApi", "dataset_info(", "list_repo_", "hf_hub_download"):
        assert networked not in source

    materialize(
        _dataset(),
        prepared_directory(tmp_path, "wikitext2"),
        cache_dir=tmp_path / "hf",
        allow_network=True,
    )
    assert len(fake_datasets) == 1  # exactly one loader call, no extra lookup


def test_acquisition_does_not_enable_offline_mode(fake_datasets, tmp_path) -> None:
    """Fetching must leave offline mode off."""

    materialize(
        _dataset(),
        prepared_directory(tmp_path, "wikitext2"),
        cache_dir=tmp_path / "hf",
        allow_network=True,
    )

    assert fake_datasets[0]["offline"] is False


import importlib
import sys
import types

from llm_behavior_lab.data.huggingface import _LOADER_LOCK, _offline_mode


def _fake_datasets_config(monkeypatch, **attributes):
    """Install a fake datasets.config exposing only the given attributes."""

    module = types.ModuleType("datasets.config")
    for name, value in attributes.items():
        setattr(module, name, value)
    monkeypatch.setitem(sys.modules, "datasets.config", module)
    return module


def _fake_hub_constants(monkeypatch, **attributes):
    """Install a fake huggingface_hub.constants exposing only the given attributes.

    ``huggingface_hub`` arrives with the optional ``hf`` extra, so the default
    suite must not require it. The package and its ``constants`` submodule are
    both placed in ``sys.modules`` so that the production ``import
    huggingface_hub.constants`` resolves to this fake instead of the real
    library, whether or not the extra happens to be installed.
    """

    package = types.ModuleType("huggingface_hub")
    module = types.ModuleType("huggingface_hub.constants")
    for name, value in attributes.items():
        setattr(module, name, value)
    package.constants = module
    monkeypatch.setitem(sys.modules, "huggingface_hub", package)
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", module)
    return module


def test_offline_mode_sets_every_flag_the_version_exposes(monkeypatch) -> None:
    """Newer Datasets carries both names; both must be enabled."""

    hub_constants = _fake_hub_constants(monkeypatch, HF_HUB_OFFLINE=False)

    config = _fake_datasets_config(
        monkeypatch, HF_HUB_OFFLINE=False, HF_DATASETS_OFFLINE=False
    )

    with _offline_mode():
        assert hub_constants.HF_HUB_OFFLINE is True
        assert config.HF_HUB_OFFLINE is True
        assert config.HF_DATASETS_OFFLINE is True
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["HF_DATASETS_OFFLINE"] == "1"

    assert hub_constants.HF_HUB_OFFLINE is False
    assert config.HF_HUB_OFFLINE is False
    assert config.HF_DATASETS_OFFLINE is False


def test_offline_mode_supports_older_datasets_without_the_hub_name(monkeypatch) -> None:
    """Datasets 2.19 exposes only HF_DATASETS_OFFLINE; it must still be enabled."""

    hub_constants = _fake_hub_constants(monkeypatch, HF_HUB_OFFLINE=False)

    config = _fake_datasets_config(monkeypatch, HF_DATASETS_OFFLINE=False)

    with _offline_mode():
        assert config.HF_DATASETS_OFFLINE is True
        assert hub_constants.HF_HUB_OFFLINE is True

    assert config.HF_DATASETS_OFFLINE is False
    # The absent attribute must not be invented on a version that lacks it.
    assert not hasattr(config, "HF_HUB_OFFLINE")


def test_offline_mode_creates_no_attributes_when_datasets_is_absent(monkeypatch) -> None:
    """A missing datasets.config must not stop offline mode working."""

    hub_constants = _fake_hub_constants(monkeypatch, HF_HUB_OFFLINE=False)

    monkeypatch.delitem(sys.modules, "datasets.config", raising=False)

    with _offline_mode():
        assert hub_constants.HF_HUB_OFFLINE is True

    assert hub_constants.HF_HUB_OFFLINE is False


def test_offline_mode_preserves_a_pre_existing_offline_setting(monkeypatch) -> None:
    """A process already offline must stay offline afterwards."""

    hub_constants = _fake_hub_constants(monkeypatch, HF_HUB_OFFLINE=True)

    config = _fake_datasets_config(monkeypatch, HF_DATASETS_OFFLINE=True)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    with _offline_mode():
        assert hub_constants.HF_HUB_OFFLINE is True

    assert hub_constants.HF_HUB_OFFLINE is True
    assert config.HF_DATASETS_OFFLINE is True
    assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_offline_state_is_restored_after_an_exception(monkeypatch) -> None:
    """A failure inside the context must not leave the process offline."""

    hub_constants = _fake_hub_constants(monkeypatch, HF_HUB_OFFLINE=False)

    config = _fake_datasets_config(monkeypatch, HF_DATASETS_OFFLINE=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)

    with pytest.raises(RuntimeError):
        with _offline_mode():
            raise RuntimeError("loading failed")

    assert hub_constants.HF_HUB_OFFLINE is False
    assert config.HF_DATASETS_OFFLINE is False
    assert "HF_HUB_OFFLINE" not in os.environ
    assert "HF_DATASETS_OFFLINE" not in os.environ


def test_a_broken_hub_install_is_not_treated_as_an_absent_one(
    monkeypatch, tmp_path
) -> None:
    """Only the optional package's own absence may be tolerated.

    An installed ``huggingface_hub`` whose import fails because something it
    needs is missing is a real fault, not a package the caller declined to
    install, so the original error must reach the caller unchanged.
    """

    package = tmp_path / "huggingface_hub"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "constants.py").write_text(
        "import a_dependency_that_is_not_installed\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(tmp_path)
    monkeypatch.delitem(sys.modules, "huggingface_hub", raising=False)
    monkeypatch.delitem(sys.modules, "huggingface_hub.constants", raising=False)
    importlib.invalidate_caches()

    with pytest.raises(ModuleNotFoundError) as failure:
        with _offline_mode():
            pass

    assert failure.value.name == "a_dependency_that_is_not_installed"


def test_loader_calls_are_serialized(monkeypatch, tmp_path) -> None:
    """Global offline state is only safe if one loader call runs at a time."""

    observed = {}

    def load_dataset(repo_id, subset=None, **kwargs):
        observed["locked"] = _LOADER_LOCK.locked()
        return list(ROWS)

    monkeypatch.setattr(huggingface, "_import_load_dataset", lambda: load_dataset)
    monkeypatch.setattr(huggingface, "datasets_available", lambda: True)

    materialize(
        _dataset(),
        prepared_directory(tmp_path, "wikitext2"),
        cache_dir=tmp_path / "hf",
        allow_network=True,
    )

    assert observed["locked"] is True
    assert not _LOADER_LOCK.locked(), "the lock must be released afterwards"


@pytest.mark.parametrize("limit", [1, 2, 3, 5, 99, 100, 101, 102, 200, 201, 205, 1000, 5000])
def test_prepared_text_never_exceeds_the_character_budget(limit: int) -> None:
    """max_characters bounds the corpus that is written, separators included."""

    rows = [{"text": "a" * 100} for _ in range(10)]

    text, _ = concatenate_text_examples(
        rows, text_field="text", max_characters=limit, document_separator="\n\n"
    )

    assert len(text) <= limit


def test_wikitext_style_budget_is_met_exactly() -> None:
    """The live WikiText case: many documents whose content reaches the budget."""

    rows = [{"text": "x" * 500} for _ in range(1000)]

    text, documents = concatenate_text_examples(
        rows, text_field="text", max_characters=200000, document_separator="\n\n"
    )

    assert len(text) == 200000
    # Separators are inside the budget, so content is now slightly below it.
    assert len(text.replace("\n\n", "")) == 200000 - 2 * (documents - 1)


def test_budget_is_consumed_exactly_at_a_separator_boundary() -> None:
    """Two 100-character documents plus one separator is exactly 202."""

    rows = [{"text": "a" * 100}, {"text": "b" * 100}, {"text": "c" * 100}]

    text, documents = concatenate_text_examples(
        rows, text_field="text", max_characters=202, document_separator="\n\n"
    )

    assert len(text) == 202
    assert documents == 2
    assert text == "a" * 100 + "\n\n" + "b" * 100


def test_a_budget_too_small_for_a_separator_stops_cleanly() -> None:
    """No trailing separator, and no partial separator, when the budget runs out."""

    rows = [{"text": "a" * 100}, {"text": "b" * 100}]

    text, documents = concatenate_text_examples(
        rows, text_field="text", max_characters=101, document_separator="\n\n"
    )

    assert text == "a" * 100
    assert documents == 1
    assert not text.endswith("\n\n")


def test_single_document_behaviour_is_unchanged() -> None:
    """A lone document is still truncated to the raw budget, with no overhead."""

    text, documents = concatenate_text_examples(
        [{"text": "abcdef"}], text_field="text", max_characters=3, document_separator="\n\n"
    )

    assert (text, documents) == ("abc", 1)


def test_max_examples_still_bounds_documents(monkeypatch) -> None:
    """Charging separators must not disturb the example limit."""

    rows = [{"text": "a" * 10} for _ in range(50)]

    text, documents = concatenate_text_examples(
        rows, text_field="text", max_examples=3, document_separator="|"
    )

    assert documents == 3
    assert text == "|".join(["a" * 10] * 3)


def test_manifest_num_characters_respects_the_configured_limit(monkeypatch, tmp_path) -> None:
    """The recorded corpus size must be comparable to the configured budget."""

    rows = [{"text": "a" * 100} for _ in range(20)]
    monkeypatch.setattr(
        huggingface, "_import_load_dataset", lambda: lambda *a, **k: list(rows)
    )
    monkeypatch.setattr(huggingface, "datasets_available", lambda: True)

    manifest = materialize(
        _dataset(max_characters=500),
        prepared_directory(tmp_path, "wikitext2"),
        cache_dir=tmp_path / "hf",
        allow_network=True,
    )

    assert manifest["num_characters"] <= manifest["max_characters"]
    assert (
        len((prepared_directory(tmp_path, "wikitext2") / TEXT_FILENAME).read_text("utf-8"))
        == manifest["num_characters"]
    )


# --- Streaming resource release ------------------------------------------
#
# The upstream shutdown abort these tests exist for is a native failure that no
# unit test can provoke. What is testable is the contract that prevents it:
# streamed datasets are released explicitly once nothing needs them, and
# ordinary loads are left alone. Each test patches the module's own collection
# seam rather than touching interpreter-wide garbage collection.


def record_releases(monkeypatch, *, on_release=None) -> list[str]:
    """Replace the collection seam with a recorder."""

    events: list[str] = []

    def release() -> None:
        events.append("released")
        if on_release is not None:
            on_release()

    monkeypatch.setattr(huggingface, "_release_stream", release)
    return events


def test_streaming_materialization_releases_the_stream(monkeypatch, tmp_path) -> None:
    """A streamed dataset must not be left for interpreter shutdown to free."""

    install_fake_loader(monkeypatch)
    releases = record_releases(monkeypatch)

    manifest = materialize(
        _dataset(streaming=True),
        prepared_directory(tmp_path, "wikitext2"),
        cache_dir=tmp_path / "hf",
        allow_network=True,
    )

    assert releases == ["released"]
    assert manifest["num_documents"] == 2


def test_non_streaming_materialization_does_not_force_collection(
    monkeypatch, tmp_path
) -> None:
    """Ordinary loads keep their existing behavior; collection is not free."""

    install_fake_loader(monkeypatch)
    releases = record_releases(monkeypatch)

    materialize(
        _dataset(),
        prepared_directory(tmp_path, "wikitext2"),
        cache_dir=tmp_path / "hf",
        allow_network=True,
    )

    assert releases == []


def test_the_stream_is_released_before_the_corpus_is_written(
    monkeypatch, tmp_path
) -> None:
    """Release happens once the dataset is no longer needed, not at the end."""

    destination = prepared_directory(tmp_path, "wikitext2")
    install_fake_loader(monkeypatch)
    published: list[bool] = []
    record_releases(monkeypatch, on_release=lambda: published.append(destination.exists()))

    materialize(
        _dataset(streaming=True),
        destination,
        cache_dir=tmp_path / "hf",
        allow_network=True,
    )

    assert published == [False]
    assert destination.exists()


def test_the_stream_is_released_when_concatenation_fails(monkeypatch, tmp_path) -> None:
    """A partially consumed stream must be released on the failure path too."""

    install_fake_loader(monkeypatch, rows=[{"wrong_field": "no text here"}])
    releases = record_releases(monkeypatch)

    with pytest.raises(DatasetAcquisitionError, match="text_field"):
        materialize(
            _dataset(streaming=True),
            prepared_directory(tmp_path, "wikitext2"),
            cache_dir=tmp_path / "hf",
            allow_network=True,
        )

    assert releases == ["released"]


def test_the_stream_is_released_when_provenance_extraction_fails(
    monkeypatch, tmp_path
) -> None:
    """Failure before consumption must still release the dataset."""

    class HostileRows(list):
        @property
        def info(self):
            raise RuntimeError("upstream metadata is unavailable")

    monkeypatch.setattr(
        huggingface, "_import_load_dataset", lambda: lambda *a, **k: HostileRows(ROWS)
    )
    monkeypatch.setattr(huggingface, "datasets_available", lambda: True)
    releases = record_releases(monkeypatch)

    with pytest.raises(RuntimeError, match="upstream metadata"):
        materialize(
            _dataset(streaming=True),
            prepared_directory(tmp_path, "wikitext2"),
            cache_dir=tmp_path / "hf",
            allow_network=True,
        )

    assert releases == ["released"]


def test_streaming_release_leaves_corpus_and_provenance_unchanged(
    monkeypatch, tmp_path
) -> None:
    """Releasing the stream must not cost any recorded information."""

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
    destination = prepared_directory(tmp_path, "wikitext2")

    manifest = materialize(
        _dataset(streaming=True),
        destination,
        cache_dir=tmp_path / "hf",
        allow_network=True,
    )

    assert (destination / TEXT_FILENAME).read_text(encoding="utf-8") == (
        "first document\n\nsecond document"
    )
    assert manifest["num_documents"] == 2
    assert manifest["resolved"]["fingerprint"] == "abc123fingerprint"
    assert manifest["resolved"]["download_size"] == 4321


def test_release_uses_the_module_seam_rather_than_global_collection() -> None:
    """The workaround must stay patchable and scoped to this module."""

    source = Path(huggingface.__file__).read_text(encoding="utf-8")

    assert source.count("gc.collect()") == 1
    assert "_release_stream()" in source
