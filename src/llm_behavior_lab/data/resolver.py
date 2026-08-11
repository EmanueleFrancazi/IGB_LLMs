"""Central dataset resolution.

Every part of the project asks for a dataset through :func:`resolve_dataset`.
Keeping one resolution path means scripts never branch on dataset source, and
the rules for where data may come from live in a single readable place.

Resolution walks an ordered list of candidate locations and returns the first
one that holds usable data. When nothing is found, the error names every
location that was tried so the message is actionable from a terminal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_behavior_lab.data.config import (
    LOCAL_TEXT_SOURCE,
    DatasetConfig,
    ResolutionPolicy,
    resolve_data_root,
)
from llm_behavior_lab.data.errors import (
    DatasetNotAvailableError,
    DatasetPathNotFoundError,
)
from llm_behavior_lab.data.prepared import load_prepared, prepared_directory
from llm_behavior_lab.data.text_dataset import load_text_file

ROUTE_EXPLICIT_PATH = "explicit_path"
ROUTE_REPO_FIXTURE = "repo_fixture"
ROUTE_REPO_PREPARED = "repo_prepared"
ROUTE_DATA_ROOT_PREPARED = "data_root_prepared"


@dataclass(frozen=True)
class ResolvedDataset:
    """A dataset that is present on disk and ready to read.

    Attributes:
        name: Dataset identifier from the configuration.
        source: Dataset source kind the configuration requested.
        path: Text file holding the dataset contents.
        route: Which candidate location supplied the data. Useful in logs and
            in experiment provenance.
        details: Extra provenance recorded by the route that resolved the
            dataset. Empty for plain local files.
    """

    name: str
    source: str
    path: Path
    route: str
    details: dict[str, Any] = field(default_factory=dict)

    def read_text(self) -> str:
        """Read the dataset contents.

        Uses the same loader as the rest of the project, so encoding and
        empty-file behavior are unchanged.
        """

        return load_text_file(self.path)

    def provenance(self) -> dict[str, Any]:
        """Summarize where the data came from, for experiment metadata."""

        record: dict[str, Any] = {
            "name": self.name,
            "source": self.source,
            "route": self.route,
            "path": str(self.path),
        }
        record.update(self.details)
        return record


def resolve_repo_path(value: str | Path, repo_root: str | Path | None = None) -> Path:
    """Resolve a possibly relative path against the repository root.

    Args:
        value: Path from configuration or the command line.
        repo_root: Repository root used for relative values. When ``None``,
            relative values are left relative to the current directory.

    Returns:
        The resolved path, with a leading ``~`` expanded.
    """

    path = Path(value).expanduser()
    if path.is_absolute() or repo_root is None:
        return path
    return Path(repo_root) / path


def resolve_dataset(
    dataset: DatasetConfig,
    policy: ResolutionPolicy,
    *,
    repo_root: str | Path | None = None,
) -> ResolvedDataset:
    """Locate the text file for a configured dataset.

    Candidate locations are tried in order:

    1. the configured ``dataset.path``
    2. a prepared copy inside the repository, under ``data/prepared/<name>/``
    3. a prepared copy under the external data root

    Acquisition is added as the subsystem grows. A ``local_text`` dataset always
    resolves at step 1 and never consults the data root or the network.

    Args:
        dataset: Validated dataset identity.
        policy: Runtime policy for this invocation.
        repo_root: Repository root used to resolve relative configured paths.

    Returns:
        A :class:`ResolvedDataset` pointing at readable data.

    Raises:
        DatasetPathNotFoundError: If a local dataset names a missing file.
        DatasetIntegrityError: If a prepared copy exists but was built for a
            different dataset identity.
        DatasetNotAvailableError: If no candidate location holds the dataset.
    """

    checked: list[str] = []

    if dataset.path:
        candidate = resolve_repo_path(dataset.path, repo_root)
        if candidate.is_file():
            return ResolvedDataset(
                name=dataset.name,
                source=dataset.source,
                path=candidate,
                route=_path_route(candidate, repo_root),
            )
        if dataset.source == LOCAL_TEXT_SOURCE:
            raise DatasetPathNotFoundError(
                f"Dataset {dataset.name!r} points at a file that does not exist: "
                f"{candidate}\n"
                "Check dataset.path in the data config, or point it at an existing file."
            )
        checked.append(f"configured path: {candidate} (missing)")
    else:
        checked.append("configured path: not set")

    for route, directory in prepared_candidates(dataset, policy, repo_root=repo_root):
        found = load_prepared(directory, dataset, verify=dataset.verify)
        if found is not None:
            text_path, manifest = found
            return ResolvedDataset(
                name=dataset.name,
                source=dataset.source,
                path=text_path,
                route=route,
                details={"prepared_dir": str(directory), "manifest": manifest},
            )
        checked.append(f"{route}: {directory} ({_absence_reason(directory)})")

    raise DatasetNotAvailableError(unavailable_message(dataset, policy, checked))


def prepared_candidates(
    dataset: DatasetConfig,
    policy: ResolutionPolicy,
    *,
    repo_root: str | Path | None = None,
) -> list[tuple[str, Path]]:
    """List the prepared-dataset directories consulted for ``dataset``, in order.

    The repository-local directory is read when it exists but is never the
    default destination for new data; that is always the external data root.
    """

    candidates: list[tuple[str, Path]] = []
    if repo_root is not None:
        candidates.append(
            (ROUTE_REPO_PREPARED, prepared_directory(Path(repo_root) / "data", dataset.name))
        )
    candidates.append(
        (
            ROUTE_DATA_ROOT_PREPARED,
            prepared_directory(resolve_data_root(policy.data_root), dataset.name),
        )
    )
    return candidates


def acquisition_destination(
    dataset: DatasetConfig,
    policy: ResolutionPolicy,
) -> Path:
    """Return where newly acquired data is written: always the external root."""

    return prepared_directory(resolve_data_root(policy.data_root), dataset.name)


def _absence_reason(directory: Path) -> str:
    """Explain why a prepared directory did not supply data."""

    if not directory.exists():
        return "not prepared"
    return "incomplete preparation"


def _path_route(path: Path, repo_root: str | Path | None) -> str:
    """Classify a resolved path as a tracked fixture or an explicit override."""

    if repo_root is None:
        return ROUTE_EXPLICIT_PATH
    root = Path(repo_root).expanduser()
    try:
        path.relative_to(root)
    except ValueError:
        return ROUTE_EXPLICIT_PATH
    return ROUTE_REPO_FIXTURE


def unavailable_message(
    dataset: DatasetConfig,
    policy: ResolutionPolicy,
    checked: list[str],
    *,
    remedy: str | None = None,
) -> str:
    """Build the message shown when a dataset cannot be resolved.

    The message names the dataset, every location that was tried, and why
    acquisition did not happen, so a user can act on it without reading code.
    """

    lines = [f"Dataset {dataset.name!r} is not available.", "", "Checked:"]
    lines.extend(f"  - {entry}" for entry in checked)
    lines.append("")

    if policy.offline:
        lines.append("Acquisition was skipped because offline mode is enabled.")
    elif not policy.allow_download:
        lines.append("Acquisition was skipped because downloads are not permitted.")
    else:
        lines.append("Acquisition is permitted but did not produce the dataset.")

    if remedy:
        lines.extend(["", remedy])
    return "\n".join(lines)
