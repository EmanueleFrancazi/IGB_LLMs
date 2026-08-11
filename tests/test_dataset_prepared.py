"""Tests for prepared dataset directories, manifests, and their reuse.

All preparation happens under pytest temporary directories. Nothing here
touches the external data root, the repository, or the network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_behavior_lab.data import (
    HUGGINGFACE_SOURCE,
    MANIFEST_FILENAME,
    ROUTE_DATA_ROOT_PREPARED,
    ROUTE_REPO_PREPARED,
    TEXT_FILENAME,
    DatasetConfig,
    DatasetIntegrityError,
    DatasetNotAvailableError,
    ResolutionPolicy,
    acquisition_destination,
    load_prepared,
    prepared_directory,
    resolve_dataset,
    write_prepared,
)
from llm_behavior_lab.data.prepared import PARTIAL_SUFFIX, text_digest


def _external_dataset(**overrides) -> DatasetConfig:
    """Build an external dataset identity, with optional field overrides."""

    fields = {
        "name": "wikitext2",
        "source": HUGGINGFACE_SOURCE,
        "repo_id": "Salesforce/wikitext",
        "subset": "wikitext-2-raw-v1",
        "split": "train[:10]",
        "max_characters": 1000,
    }
    fields.update(overrides)
    return DatasetConfig(**fields)


def test_write_prepared_creates_text_and_manifest(tmp_path) -> None:
    """A prepared directory holds exactly the corpus and its manifest."""

    dataset = _external_dataset()
    directory = prepared_directory(tmp_path, dataset.name)

    manifest = write_prepared(directory, "corpus text", dataset, num_documents=3)

    assert (directory / TEXT_FILENAME).read_text(encoding="utf-8") == "corpus text"
    assert sorted(p.name for p in directory.iterdir()) == [MANIFEST_FILENAME, TEXT_FILENAME]
    on_disk = json.loads((directory / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert on_disk == manifest
    assert manifest["name"] == "wikitext2"
    assert manifest["num_characters"] == len("corpus text")
    assert manifest["num_documents"] == 3
    assert manifest["sha256"] == text_digest("corpus text")


def test_manifest_records_identity_but_no_paths(tmp_path) -> None:
    """Prepared directories must stay relocatable between machines."""

    dataset = _external_dataset()
    directory = prepared_directory(tmp_path, dataset.name)

    manifest = write_prepared(directory, "corpus", dataset)

    assert manifest["split"] == "train[:10]"
    assert manifest["subset"] == "wikitext-2-raw-v1"
    assert manifest["max_characters"] == 1000
    # No filesystem paths: identifiers such as "Salesforce/wikitext" are fine,
    # but nothing may be rooted at a location on this machine.
    serialized = json.dumps(manifest)
    assert str(tmp_path) not in serialized
    assert not any(
        isinstance(value, str) and value.startswith(("/", "~"))
        for value in manifest.values()
    )


def test_prepared_data_is_reused(tmp_path) -> None:
    """A matching prepared copy resolves without any acquisition."""

    dataset = _external_dataset()
    directory = prepared_directory(tmp_path, dataset.name)
    write_prepared(directory, "prepared corpus", dataset)

    resolved = resolve_dataset(dataset, ResolutionPolicy(data_root=tmp_path))

    assert resolved.route == ROUTE_DATA_ROOT_PREPARED
    assert resolved.read_text() == "prepared corpus"
    assert resolved.details["manifest"]["name"] == "wikitext2"


def test_prepared_data_is_reused_when_downloads_are_forbidden(tmp_path) -> None:
    """Forbidding downloads restricts fetching, not reuse of existing data."""

    dataset = _external_dataset()
    write_prepared(prepared_directory(tmp_path, dataset.name), "corpus", dataset)

    policy = ResolutionPolicy(data_root=tmp_path, allow_download=False, offline=True)
    assert resolve_dataset(dataset, policy).read_text() == "corpus"


def test_repository_local_prepared_copy_is_preferred(tmp_path) -> None:
    """A repository-local copy is read before the external root."""

    dataset = _external_dataset()
    repo_root = tmp_path / "repo"
    write_prepared(prepared_directory(repo_root / "data", dataset.name), "repo copy", dataset)
    write_prepared(prepared_directory(tmp_path / "root", dataset.name), "root copy", dataset)

    resolved = resolve_dataset(
        dataset, ResolutionPolicy(data_root=tmp_path / "root"), repo_root=repo_root
    )

    assert resolved.route == ROUTE_REPO_PREPARED
    assert resolved.read_text() == "repo copy"


def test_new_data_is_destined_for_the_external_root(tmp_path) -> None:
    """Repository-local data may be read, but never becomes the write target."""

    dataset = _external_dataset()
    destination = acquisition_destination(dataset, ResolutionPolicy(data_root=tmp_path))

    assert destination == tmp_path / "prepared" / "wikitext2"


def test_incomplete_preparation_is_skipped_not_trusted(tmp_path) -> None:
    """Text without a manifest is an interrupted preparation."""

    dataset = _external_dataset()
    directory = prepared_directory(tmp_path, dataset.name)
    directory.mkdir(parents=True)
    (directory / TEXT_FILENAME).write_text("half written", encoding="utf-8")

    assert load_prepared(directory, dataset) is None

    with pytest.raises(DatasetNotAvailableError, match="incomplete preparation"):
        resolve_dataset(dataset, ResolutionPolicy(data_root=tmp_path))


def test_unreadable_manifest_is_treated_as_incomplete(tmp_path) -> None:
    """A corrupt manifest must not be trusted."""

    dataset = _external_dataset()
    directory = prepared_directory(tmp_path, dataset.name)
    write_prepared(directory, "corpus", dataset)
    (directory / MANIFEST_FILENAME).write_text("{not json", encoding="utf-8")

    assert load_prepared(directory, dataset) is None


def test_missing_text_file_is_treated_as_incomplete(tmp_path) -> None:
    """A manifest without its corpus is not usable data."""

    dataset = _external_dataset()
    directory = prepared_directory(tmp_path, dataset.name)
    write_prepared(directory, "corpus", dataset)
    (directory / TEXT_FILENAME).unlink()

    assert load_prepared(directory, dataset) is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("split", "train[:99]"),
        ("subset", "wikitext-103-raw-v1"),
        ("max_characters", 5000),
        ("revision", "pinned-abc"),
        ("text_field", "content"),
    ],
)
def test_stale_prepared_data_is_reported_not_reused(tmp_path, field: str, value) -> None:
    """Changing the configuration must not silently reuse the old corpus."""

    prepared_with = _external_dataset()
    directory = prepared_directory(tmp_path, prepared_with.name)
    write_prepared(directory, "old corpus", prepared_with)

    requested = _external_dataset(**{field: value})

    with pytest.raises(DatasetIntegrityError) as excinfo:
        resolve_dataset(
            requested, ResolutionPolicy(data_root=tmp_path, allow_download=True)
        )

    message = str(excinfo.value)
    assert field in message
    assert "--force-refresh" in message


def test_verification_detects_changed_text(tmp_path) -> None:
    """Opt-in verification catches a corpus edited after preparation."""

    dataset = _external_dataset(verify=True)
    directory = prepared_directory(tmp_path, dataset.name)
    write_prepared(directory, "original", dataset)
    (directory / TEXT_FILENAME).write_text("tampered", encoding="utf-8")

    with pytest.raises(DatasetIntegrityError, match="has changed since it was prepared"):
        resolve_dataset(dataset, ResolutionPolicy(data_root=tmp_path))


def test_verification_is_off_by_default(tmp_path) -> None:
    """Digest checking is opt-in so ordinary loads stay cheap."""

    dataset = _external_dataset()
    directory = prepared_directory(tmp_path, dataset.name)
    write_prepared(directory, "original", dataset)
    (directory / TEXT_FILENAME).write_text("tampered", encoding="utf-8")

    assert resolve_dataset(dataset, ResolutionPolicy(data_root=tmp_path)).read_text() == "tampered"


def test_existing_prepared_data_is_not_overwritten_by_default(tmp_path) -> None:
    """Preparation must not silently discard an existing corpus."""

    dataset = _external_dataset()
    directory = prepared_directory(tmp_path, dataset.name)
    write_prepared(directory, "first", dataset)

    with pytest.raises(FileExistsError, match="already exists"):
        write_prepared(directory, "second", dataset)

    assert (directory / TEXT_FILENAME).read_text(encoding="utf-8") == "first"


def test_overwrite_replaces_prepared_data_atomically(tmp_path) -> None:
    """An explicit refresh replaces both files together."""

    dataset = _external_dataset()
    directory = prepared_directory(tmp_path, dataset.name)
    write_prepared(directory, "first", dataset)

    write_prepared(directory, "second", dataset, overwrite=True)

    assert (directory / TEXT_FILENAME).read_text(encoding="utf-8") == "second"
    manifest = json.loads((directory / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["sha256"] == text_digest("second")


def test_failed_preparation_leaves_nothing_publishable(tmp_path, monkeypatch) -> None:
    """An exception mid-write must not produce a directory that looks complete."""

    dataset = _external_dataset()
    directory = prepared_directory(tmp_path, dataset.name)

    import llm_behavior_lab.data.prepared as prepared_module

    real_replace = prepared_module.os.replace

    def failing_replace(src, dst):
        raise OSError("simulated interruption")

    monkeypatch.setattr(prepared_module.os, "replace", failing_replace)
    with pytest.raises(OSError, match="simulated interruption"):
        write_prepared(directory, "corpus", dataset)
    monkeypatch.setattr(prepared_module.os, "replace", real_replace)

    assert not directory.exists()
    leftovers = [p for p in directory.parent.iterdir() if p.name.endswith(PARTIAL_SUFFIX)]
    assert leftovers == []
    with pytest.raises(DatasetNotAvailableError):
        resolve_dataset(dataset, ResolutionPolicy(data_root=tmp_path))


def test_partial_directories_are_ignored_and_left_alone(tmp_path, monkeypatch) -> None:
    """Another process's staging directory must never resolve, nor be destroyed."""

    dataset = _external_dataset()
    directory = prepared_directory(tmp_path, dataset.name)
    directory.parent.mkdir(parents=True)
    other_process = directory.parent / f".{dataset.name}.abc123{PARTIAL_SUFFIX}"
    other_process.mkdir()
    (other_process / TEXT_FILENAME).write_text("in flight elsewhere", encoding="utf-8")

    with pytest.raises(DatasetNotAvailableError):
        resolve_dataset(dataset, ResolutionPolicy(data_root=tmp_path), reporter=None)

    # Preparing our own copy must not touch it.
    write_prepared(directory, "our corpus", dataset)

    assert other_process.exists()
    assert (other_process / TEXT_FILENAME).read_text(encoding="utf-8") == "in flight elsewhere"


def test_failed_preparation_removes_only_its_own_staging(tmp_path, monkeypatch) -> None:
    """Cleanup on failure is scoped to the staging this call created."""

    dataset = _external_dataset()
    directory = prepared_directory(tmp_path, dataset.name)
    directory.parent.mkdir(parents=True)
    foreign = directory.parent / f".{dataset.name}.other{PARTIAL_SUFFIX}"
    foreign.mkdir()

    import llm_behavior_lab.data.prepared as prepared_module

    monkeypatch.setattr(
        prepared_module.os, "replace", lambda *a: (_ for _ in ()).throw(OSError("boom"))
    )
    with pytest.raises(OSError):
        write_prepared(directory, "corpus", dataset)

    assert foreign.exists()
    leftovers = [p for p in directory.parent.iterdir() if p.name.endswith(PARTIAL_SUFFIX)]
    assert leftovers == [foreign]


def test_unavailable_message_lists_every_prepared_location(tmp_path) -> None:
    """The error must name the repository and external locations that were tried."""

    dataset = _external_dataset()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    with pytest.raises(DatasetNotAvailableError) as excinfo:
        resolve_dataset(
            dataset, ResolutionPolicy(data_root=tmp_path / "root"), repo_root=repo_root
        )

    message = str(excinfo.value)
    assert ROUTE_REPO_PREPARED in message
    assert ROUTE_DATA_ROOT_PREPARED in message
    assert str(tmp_path / "root" / "prepared" / "wikitext2") in message


def test_local_text_never_consults_prepared_directories(tmp_path) -> None:
    """The tiny fixture workflow must not depend on the data root at all."""

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("tiny", encoding="utf-8")
    dataset = DatasetConfig(name="tiny_local_text", path=str(corpus))

    resolved = resolve_dataset(
        dataset, ResolutionPolicy(data_root=Path("/nonexistent/data/root"))
    )

    assert resolved.read_text() == "tiny"


def test_integrity_remedy_matches_the_policy_in_effect(tmp_path) -> None:
    """Advice must not name a command the current policy forbids."""

    prepared_with = _external_dataset()
    write_prepared(prepared_directory(tmp_path, prepared_with.name), "old", prepared_with)
    requested = _external_dataset(split="train[:99]")

    with pytest.raises(DatasetIntegrityError) as offline:
        resolve_dataset(requested, ResolutionPolicy(data_root=tmp_path, offline=True))
    message = str(offline.value)
    assert "--offline" in message
    assert "Re-prepare it with --force-refresh" not in message
    assert "remove the prepared directory by hand" in message

    with pytest.raises(DatasetIntegrityError) as no_download:
        resolve_dataset(requested, ResolutionPolicy(data_root=tmp_path))
    assert "--no-download" in str(no_download.value)

    with pytest.raises(DatasetIntegrityError) as permitted:
        resolve_dataset(
            requested, ResolutionPolicy(data_root=tmp_path, allow_download=True)
        )
    assert "--force-refresh" in str(permitted.value)
