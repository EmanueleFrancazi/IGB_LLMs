"""Tests for the shared dataset command-line surface.

Scripts opt in to acquire-on-miss, unlike the conservative library default.
These tests pin that difference and the meaning of each flag, because the
combination decides whether a workflow may reach the network.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from llm_behavior_lab.data import (
    add_dataset_arguments,
    resolution_policy_from_args,
)


def _parse(*argv: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_dataset_arguments(parser)
    return parser.parse_args(list(argv))


def test_scripts_permit_acquisition_by_default() -> None:
    """A user running a script gets acquire-on-miss without extra flags."""

    policy = resolution_policy_from_args(_parse())

    assert policy.allow_download is True
    assert policy.offline is False
    assert policy.may_acquire is True


def test_library_default_stays_conservative() -> None:
    """The script default must not have leaked into the library default."""

    from llm_behavior_lab.data import ResolutionPolicy

    assert ResolutionPolicy().may_acquire is False


def test_no_download_forbids_acquisition_but_not_reuse() -> None:
    """--no-download withdraws permission to fetch, nothing more."""

    policy = resolution_policy_from_args(_parse("--no-download"))

    assert policy.allow_download is False
    assert policy.offline is False
    assert policy.may_acquire is False


def test_offline_implies_no_download() -> None:
    """--offline forbids the network, so acquisition is impossible."""

    policy = resolution_policy_from_args(_parse("--offline"))

    assert policy.offline is True
    assert policy.allow_download is False
    assert policy.may_acquire is False


def test_offline_wins_over_an_explicit_download_request() -> None:
    """Combining the flags must resolve to the safer behavior."""

    policy = resolution_policy_from_args(_parse("--offline", "--no-download"))

    assert policy.may_acquire is False


def test_data_root_is_passed_through(tmp_path) -> None:
    """--data-root selects where prepared and cached data live."""

    policy = resolution_policy_from_args(_parse("--data-root", str(tmp_path)))

    assert policy.data_root == Path(tmp_path)


def test_force_refresh_requires_acquisition_to_be_possible() -> None:
    """Refreshing while offline is contradictory and must be rejected."""

    with pytest.raises(Exception):
        resolution_policy_from_args(_parse("--force-refresh", "--offline"))


def test_force_refresh_is_accepted_with_the_default_script_policy() -> None:
    """The ordinary refresh invocation must be constructible."""

    policy = resolution_policy_from_args(_parse("--force-refresh"))

    assert policy.force_refresh is True
    assert policy.may_acquire is True
