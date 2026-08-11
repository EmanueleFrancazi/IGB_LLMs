"""Tests that dataset contents cannot reach Git.

The whole point of the data root is that real corpora stay out of the
repository. These tests pin that boundary by asking Git directly, so a future
change to the ignore rules cannot quietly start tracking downloaded data.

The tiny fixture is the deliberate exception: it is small, committed, and what
makes a fresh clone runnable with no downloads.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or not (REPO_ROOT / ".git").exists(),
    reason="requires a Git checkout",
)


def is_ignored(path: str) -> bool:
    """Ask Git whether it would ignore ``path``."""

    result = subprocess.run(
        ["git", "check-ignore", "-q", "--no-index", path],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    return result.returncode == 0


@pytest.mark.parametrize(
    "path",
    [
        "data/prepared/wikitext2/text.txt",
        "data/prepared/wikitext2/manifest.json",
        "data/prepared/.wikitext2.abc.partial/text.txt",
        "data/hf/datasets--Salesforce--wikitext/blob",
        "data/raw/downloaded_corpus.txt",
        "outputs/untrained_baseline/run/metadata.json",
    ],
)
def test_dataset_and_run_contents_are_ignored(path: str) -> None:
    """Prepared corpora, caches, and experiment output must never be tracked."""

    assert is_ignored(path), f"{path} would be committed"


@pytest.mark.parametrize("path", ["data/raw/tiny_corpus.txt", "data/README.md"])
def test_deliberately_tracked_files_are_not_ignored(path: str) -> None:
    """The offline fixture and the data guide must remain in the repository."""

    assert not is_ignored(path)


def test_tiny_fixture_is_actually_tracked() -> None:
    """A fresh clone must contain the corpus every default workflow uses."""

    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "data/raw/tiny_corpus.txt"],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    assert result.returncode == 0


def test_no_dataset_contents_are_tracked_beyond_the_fixture() -> None:
    """Only the fixture and documentation may live under data/ in Git."""

    result = subprocess.run(
        ["git", "ls-files", "data/"], cwd=REPO_ROOT, capture_output=True, text=True
    )
    tracked = sorted(line for line in result.stdout.split() if line)

    assert tracked == ["data/README.md", "data/raw/tiny_corpus.txt"]
