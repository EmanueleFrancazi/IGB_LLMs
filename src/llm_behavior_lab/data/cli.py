"""Command-line surface for dataset resolution.

Every script that loads a dataset offers the same three controls. Defining them
once keeps the flags, their defaults, and their meanings identical across
scripts, and keeps the scripts themselves thin.

Scripts opt in to acquire-on-miss: a missing external dataset is obtained
automatically, announced first, unless the user disables it. The library
default remains conservative, so this convenience exists only where a human is
watching the output.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_behavior_lab.data.config import DATA_ROOT_ENV_VAR, ResolutionPolicy


def add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the shared dataset-resolution options to ``parser``."""

    group = parser.add_argument_group("dataset resolution")
    group.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "Root for prepared and cached datasets. Defaults to "
            f"${DATA_ROOT_ENV_VAR}, then a platform cache directory outside the "
            "repository."
        ),
    )
    group.add_argument(
        "--no-download",
        action="store_true",
        help=(
            "Reuse datasets already available locally, but never obtain a "
            "missing one."
        ),
    )
    group.add_argument(
        "--offline",
        action="store_true",
        help="Forbid all network access. Implies --no-download.",
    )
    group.add_argument(
        "--force-refresh",
        action="store_true",
        help="Re-acquire an external dataset, replacing any prepared copy.",
    )


def resolution_policy_from_args(args: argparse.Namespace) -> ResolutionPolicy:
    """Build the runtime policy for one script invocation.

    Scripts permit acquisition by default; ``--no-download`` and ``--offline``
    withdraw that permission, and ``--offline`` additionally forbids reaching
    the network to reuse a cached copy.
    """

    offline = bool(getattr(args, "offline", False))
    no_download = bool(getattr(args, "no_download", False))
    return ResolutionPolicy(
        data_root=getattr(args, "data_root", None),
        offline=offline,
        allow_download=not (offline or no_download),
        force_refresh=bool(getattr(args, "force_refresh", False)),
    )
