"""Tests for central dataset resolution.

Everything here is offline. Datasets are ordinary text files created under
pytest temporary directories, and no test consults the external data root or
the network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_behavior_lab.data import (
    HUGGINGFACE_SOURCE,
    LOCAL_TEXT_SOURCE,
    ROUTE_LOCAL_PATH,
    ROUTE_REPOSITORY_PATH,
    DatasetConfig,
    DatasetNotAvailableError,
    DatasetPathNotFoundError,
    ResolutionPolicy,
    resolve_dataset,
    resolve_repo_path,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACKED_FIXTURE = "data/raw/tiny_corpus.txt"


def _fixture_tree(tmp_path: Path, text: str = "hello dataset") -> Path:
    """Create a repository-like tree containing the tracked fixture path."""

    target = tmp_path / TRACKED_FIXTURE
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def test_tracked_fixture_resolves_from_a_relative_path(tmp_path) -> None:
    """The default local workflow resolves against the repository root."""

    _fixture_tree(tmp_path, "tiny corpus text")
    dataset = DatasetConfig(name="tiny_local_text", path=TRACKED_FIXTURE)

    resolved = resolve_dataset(dataset, ResolutionPolicy(), repo_root=tmp_path)

    assert resolved.path == tmp_path / TRACKED_FIXTURE
    assert resolved.route == ROUTE_REPOSITORY_PATH
    assert resolved.source == LOCAL_TEXT_SOURCE
    assert resolved.read_text() == "tiny corpus text"


def test_real_repository_fixture_resolves() -> None:
    """The dataset shipped with the repository must resolve as-is."""

    dataset = DatasetConfig(name="tiny_local_text", path=TRACKED_FIXTURE)

    resolved = resolve_dataset(dataset, ResolutionPolicy(), repo_root=REPO_ROOT)

    assert resolved.path == REPO_ROOT / TRACKED_FIXTURE
    assert resolved.route == ROUTE_REPOSITORY_PATH
    assert len(resolved.read_text()) > 0


def test_path_outside_the_repository_is_reported_as_local(tmp_path) -> None:
    """A file outside the checkout is a machine-local path, not a repository one."""

    external = tmp_path / "elsewhere" / "corpus.txt"
    external.parent.mkdir(parents=True)
    external.write_text("external text", encoding="utf-8")
    dataset = DatasetConfig(name="external", path=str(external))

    resolved = resolve_dataset(dataset, ResolutionPolicy(), repo_root=REPO_ROOT)

    assert resolved.path == external
    assert resolved.route == ROUTE_LOCAL_PATH
    assert resolved.read_text() == "external text"


def test_resolution_does_not_require_a_repository_root(tmp_path, monkeypatch) -> None:
    """Relative paths still work when no repository root is supplied."""

    _fixture_tree(tmp_path, "relative text")
    monkeypatch.chdir(tmp_path)
    dataset = DatasetConfig(name="tiny_local_text", path=TRACKED_FIXTURE)

    resolved = resolve_dataset(dataset, ResolutionPolicy())

    assert resolved.read_text() == "relative text"
    assert resolved.route == ROUTE_LOCAL_PATH


def test_missing_local_file_names_the_path(tmp_path) -> None:
    """A local dataset with a bad path must say which file is missing."""

    dataset = DatasetConfig(name="tiny_local_text", path="data/raw/absent.txt")

    with pytest.raises(DatasetPathNotFoundError) as excinfo:
        resolve_dataset(dataset, ResolutionPolicy(), repo_root=tmp_path)

    message = str(excinfo.value)
    assert "tiny_local_text" in message
    assert str(tmp_path / "data/raw/absent.txt") in message
    assert "dataset.path" in message


def test_missing_local_file_never_falls_through_to_acquisition(tmp_path) -> None:
    """Permitting downloads must not turn a typo into a download attempt."""

    dataset = DatasetConfig(name="tiny_local_text", path="data/raw/absent.txt")
    policy = ResolutionPolicy(allow_download=True)

    with pytest.raises(DatasetPathNotFoundError):
        resolve_dataset(dataset, policy, repo_root=tmp_path)


def test_a_directory_is_not_accepted_as_a_dataset(tmp_path) -> None:
    """Only regular files count as resolved data."""

    (tmp_path / "data" / "raw").mkdir(parents=True)
    dataset = DatasetConfig(name="tiny_local_text", path="data/raw")

    with pytest.raises(DatasetPathNotFoundError):
        resolve_dataset(dataset, ResolutionPolicy(), repo_root=tmp_path)


def test_external_dataset_without_local_data_is_unavailable(tmp_path) -> None:
    """An external dataset with nothing on disk reports what was checked."""

    dataset = DatasetConfig(
        name="wikitext2", source=HUGGINGFACE_SOURCE, repo_id="Salesforce/wikitext"
    )

    with pytest.raises(DatasetNotAvailableError) as excinfo:
        resolve_dataset(dataset, ResolutionPolicy(), repo_root=tmp_path)

    message = str(excinfo.value)
    assert "wikitext2" in message
    assert "Checked:" in message
    assert "configured path: not set" in message


def test_unavailable_message_names_the_blocking_policy(tmp_path) -> None:
    """The user must learn why acquisition did not happen."""

    dataset = DatasetConfig(
        name="wikitext2", source=HUGGINGFACE_SOURCE, repo_id="Salesforce/wikitext"
    )

    with pytest.raises(DatasetNotAvailableError, match="offline mode is enabled"):
        resolve_dataset(dataset, ResolutionPolicy(offline=True), repo_root=tmp_path)

    with pytest.raises(DatasetNotAvailableError, match="downloads are not permitted"):
        resolve_dataset(dataset, ResolutionPolicy(), repo_root=tmp_path)


def test_external_dataset_can_use_a_configured_path(tmp_path) -> None:
    """Pointing an external dataset at a prepared file must work offline."""

    prepared = tmp_path / "prepared.txt"
    prepared.write_text("already prepared", encoding="utf-8")
    dataset = DatasetConfig(
        name="wikitext2",
        source=HUGGINGFACE_SOURCE,
        repo_id="Salesforce/wikitext",
        path=str(prepared),
    )

    resolved = resolve_dataset(dataset, ResolutionPolicy(offline=True), repo_root=tmp_path)

    assert resolved.read_text() == "already prepared"
    assert resolved.source == HUGGINGFACE_SOURCE


def test_external_dataset_with_a_missing_configured_path_lists_it(tmp_path) -> None:
    """A configured but absent path should appear among the checked locations."""

    dataset = DatasetConfig(
        name="wikitext2",
        source=HUGGINGFACE_SOURCE,
        repo_id="Salesforce/wikitext",
        path=str(tmp_path / "absent.txt"),
    )

    with pytest.raises(DatasetNotAvailableError) as excinfo:
        resolve_dataset(dataset, ResolutionPolicy(), repo_root=tmp_path)

    assert "(missing)" in str(excinfo.value)


def test_provenance_describes_the_resolution(tmp_path) -> None:
    """Experiment metadata needs name, source, route, and path."""

    _fixture_tree(tmp_path)
    dataset = DatasetConfig(name="tiny_local_text", path=TRACKED_FIXTURE)

    provenance = resolve_dataset(
        dataset, ResolutionPolicy(), repo_root=tmp_path
    ).provenance()

    assert provenance["name"] == "tiny_local_text"
    assert provenance["source"] == LOCAL_TEXT_SOURCE
    assert provenance["route"] == ROUTE_REPOSITORY_PATH
    assert provenance["path"] == str(tmp_path / TRACKED_FIXTURE)


def test_resolution_never_creates_anything(tmp_path) -> None:
    """Resolving must not write to the repository or the data root."""

    _fixture_tree(tmp_path)
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))

    resolve_dataset(
        DatasetConfig(name="tiny_local_text", path=TRACKED_FIXTURE),
        ResolutionPolicy(),
        repo_root=tmp_path,
    )

    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")) == before


def test_resolve_repo_path_handles_relative_and_absolute(tmp_path) -> None:
    """The shared path helper replaces the copies currently living in scripts."""

    assert resolve_repo_path("data/raw/x.txt", tmp_path) == tmp_path / "data/raw/x.txt"
    assert resolve_repo_path(tmp_path / "abs.txt", tmp_path) == tmp_path / "abs.txt"
    assert resolve_repo_path("relative.txt") == Path("relative.txt")
    assert "~" not in str(resolve_repo_path("~/corpus.txt", tmp_path))


def test_repository_route_does_not_imply_the_file_is_tracked(tmp_path) -> None:
    """The route describes location only.

    An ignored or untracked file inside the checkout still reports
    ``repository_path``. Deciding trackedness would need a Git lookup on every
    dataset load, and the route deliberately makes no reproducibility claim.
    """

    untracked = tmp_path / "data" / "prepared" / "scratch.txt"
    untracked.parent.mkdir(parents=True)
    untracked.write_text("not tracked by git", encoding="utf-8")
    dataset = DatasetConfig(name="scratch", path=str(untracked))

    resolved = resolve_dataset(dataset, ResolutionPolicy(), repo_root=tmp_path)

    assert resolved.route == ROUTE_REPOSITORY_PATH
