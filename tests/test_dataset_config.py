"""Tests for dataset configuration, resolution policy, and data-root selection.

These are vocabulary tests only. Nothing here resolves, prepares, or acquires a
dataset. No test touches the network, and no test writes outside a pytest
temporary directory.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import yaml

from llm_behavior_lab.data import (
    DATA_ROOT_ENV_VAR,
    HUGGINGFACE_SOURCE,
    LOCAL_TEXT_SOURCE,
    DatasetConfig,
    DatasetConfigError,
    DatasetError,
    ResolutionPolicy,
    UnsupportedDatasetSourceError,
    resolve_data_root,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TINY_TEXT_CONFIG_PATH = REPO_ROOT / "configs" / "data" / "tiny_text.yaml"


def _local_config(**overrides) -> dict:
    """Build a minimal valid data config, with optional dataset overrides."""

    dataset = {"name": "unit_test", "path": "data/raw/tiny_corpus.txt"}
    dataset.update(overrides)
    return {"dataset": dataset}


def test_tracked_tiny_text_config_still_parses() -> None:
    """The shipped tiny-corpus config must keep working without modification."""

    config = yaml.safe_load(TINY_TEXT_CONFIG_PATH.read_text(encoding="utf-8"))
    dataset = DatasetConfig.from_config(config)

    assert dataset.name == "tiny_local_text"
    assert dataset.source == LOCAL_TEXT_SOURCE
    assert dataset.path == "data/raw/tiny_corpus.txt"
    # val_fraction belongs to the splitting step, not to dataset identity.
    assert "val_fraction" in config["dataset"]


def test_source_defaults_to_local_text() -> None:
    """A config without an explicit source describes a local text dataset."""

    assert DatasetConfig.from_config(_local_config()).source == LOCAL_TEXT_SOURCE


def test_optional_fields_have_documented_defaults() -> None:
    """Unset optional fields should not require callers to special-case them."""

    dataset = DatasetConfig.from_config(_local_config())

    assert dataset.split == "train"
    assert dataset.text_field == "text"
    assert dataset.document_separator == "\n\n"
    assert dataset.repo_id is None
    assert dataset.subset is None
    assert dataset.revision is None
    assert dataset.max_examples is None
    assert dataset.max_characters is None
    assert dataset.verify is False


def test_huggingface_fields_round_trip() -> None:
    """External dataset identity should be preserved exactly as configured."""

    dataset = DatasetConfig.from_config(
        {
            "dataset": {
                "name": "wikitext2_raw_train1k",
                "source": HUGGINGFACE_SOURCE,
                "repo_id": "Salesforce/wikitext",
                "subset": "wikitext-2-raw-v1",
                "split": "train[:1000]",
                "revision": "abc123",
                "max_examples": 1000,
                "max_characters": 200000,
                "verify": True,
            }
        }
    )

    assert dataset.repo_id == "Salesforce/wikitext"
    assert dataset.subset == "wikitext-2-raw-v1"
    assert dataset.split == "train[:1000]"
    assert dataset.revision == "abc123"
    assert dataset.max_examples == 1000
    assert dataset.max_characters == 200000
    assert dataset.verify is True


def test_missing_dataset_section_is_rejected() -> None:
    """A data config without a dataset section cannot describe a dataset."""

    with pytest.raises(DatasetConfigError, match="section named 'dataset'"):
        DatasetConfig.from_config({"batching": {"batch_size": 4}})


def test_dataset_section_must_be_a_mapping() -> None:
    """A scalar dataset section is a configuration mistake, not a name."""

    with pytest.raises(DatasetConfigError, match="section named 'dataset'"):
        DatasetConfig.from_config({"dataset": "tiny_local_text"})


def test_name_is_required() -> None:
    """Every dataset needs a stable identifier."""

    with pytest.raises(DatasetConfigError, match="dataset.name is required"):
        DatasetConfig.from_config({"dataset": {"path": "data/raw/tiny_corpus.txt"}})


@pytest.mark.parametrize("name", ["", "  ", "../escape", "with/slash", ".hidden", "-leading"])
def test_unsafe_names_are_rejected(name: str) -> None:
    """Names become directory components, so traversal must be impossible."""

    with pytest.raises(DatasetConfigError, match="dataset.name"):
        DatasetConfig(name=name, path="data/raw/tiny_corpus.txt")


@pytest.mark.parametrize("name", ["tiny_local_text", "wikitext2-raw", "v1.0", "a"])
def test_safe_names_are_accepted(name: str) -> None:
    """Ordinary identifiers should not be rejected."""

    assert DatasetConfig(name=name, path="data/raw/tiny_corpus.txt").name == name


def test_unsupported_source_is_rejected_with_the_supported_list() -> None:
    """An unknown source should tell the user what is actually available."""

    with pytest.raises(UnsupportedDatasetSourceError) as excinfo:
        DatasetConfig.from_config(_local_config(source="s3"))

    message = str(excinfo.value)
    assert LOCAL_TEXT_SOURCE in message
    assert HUGGINGFACE_SOURCE in message


def test_local_text_requires_a_path() -> None:
    """A local dataset without a path cannot be located later."""

    with pytest.raises(DatasetConfigError, match="dataset.path is required"):
        DatasetConfig.from_config({"dataset": {"name": "unit_test"}})


def test_huggingface_requires_a_repo_id() -> None:
    """An external dataset without an identifier cannot be obtained later."""

    with pytest.raises(DatasetConfigError, match="dataset.repo_id is required"):
        DatasetConfig.from_config(
            {"dataset": {"name": "unit_test", "source": HUGGINGFACE_SOURCE}}
        )


@pytest.mark.parametrize("field_name", ["max_examples", "max_characters"])
@pytest.mark.parametrize("value", [0, -1])
def test_limits_must_be_positive_when_provided(field_name: str, value: int) -> None:
    """A non-positive cap would silently produce an empty corpus."""

    with pytest.raises(DatasetConfigError, match=f"dataset.{field_name} must be positive"):
        DatasetConfig.from_config(_local_config(**{field_name: value}))


@pytest.mark.parametrize("field_name", ["split", "text_field"])
def test_required_strings_must_be_non_empty(field_name: str) -> None:
    """Blank values are configuration mistakes rather than defaults."""

    with pytest.raises(DatasetConfigError, match=f"dataset.{field_name} must be non-empty"):
        DatasetConfig.from_config(_local_config(**{field_name: "  "}))


def test_dataset_config_is_immutable() -> None:
    """Dataset identity must not change after validation."""

    dataset = DatasetConfig.from_config(_local_config())
    with pytest.raises(dataclasses.FrozenInstanceError):
        dataset.name = "renamed"  # type: ignore[misc]


def test_policy_defaults_cannot_reach_the_network() -> None:
    """A policy built without arguments must never authorize acquisition."""

    policy = ResolutionPolicy()

    assert policy.offline is False
    assert policy.allow_download is False
    assert policy.force_refresh is False
    assert policy.data_root is None
    assert policy.may_acquire is False


@pytest.mark.parametrize(
    ("offline", "allow_download", "expected"),
    [
        (False, False, False),
        (False, True, True),
        (True, False, False),
        (True, True, False),
    ],
)
def test_may_acquire_truth_table(offline: bool, allow_download: bool, expected: bool) -> None:
    """Offline always wins: it forbids acquisition even when downloads are allowed."""

    policy = ResolutionPolicy(offline=offline, allow_download=allow_download)
    assert policy.may_acquire is expected


def test_no_download_still_permits_reusing_local_data() -> None:
    """Forbidding downloads restricts fetching, not reading what is already there."""

    policy = ResolutionPolicy(allow_download=False)

    assert policy.may_acquire is False
    assert policy.offline is False


def test_force_refresh_requires_acquisition_to_be_permitted() -> None:
    """Refreshing without permission to acquire is a contradictory request."""

    with pytest.raises(DatasetConfigError, match="force_refresh requires acquisition"):
        ResolutionPolicy(force_refresh=True)

    with pytest.raises(DatasetConfigError, match="force_refresh requires acquisition"):
        ResolutionPolicy(force_refresh=True, allow_download=True, offline=True)


def test_force_refresh_is_accepted_when_acquisition_is_permitted() -> None:
    """The combination a user actually wants must be constructible."""

    policy = ResolutionPolicy(force_refresh=True, allow_download=True)

    assert policy.may_acquire is True
    assert policy.force_refresh is True


def test_explicit_data_root_wins(tmp_path, monkeypatch) -> None:
    """An explicit argument takes precedence over the environment."""

    monkeypatch.setenv(DATA_ROOT_ENV_VAR, str(tmp_path / "from_env"))

    assert resolve_data_root(tmp_path / "explicit") == tmp_path / "explicit"


def test_environment_variable_is_used_when_no_argument_is_given(tmp_path, monkeypatch) -> None:
    """The project environment variable is the cluster-friendly override."""

    monkeypatch.setenv(DATA_ROOT_ENV_VAR, str(tmp_path / "from_env"))

    assert resolve_data_root() == tmp_path / "from_env"


def test_xdg_cache_home_is_used_before_the_home_default(tmp_path, monkeypatch) -> None:
    """A configured XDG cache directory should be respected."""

    monkeypatch.delenv(DATA_ROOT_ENV_VAR, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))

    assert resolve_data_root() == tmp_path / "xdg" / "llm-behavior-lab"


def test_home_cache_is_the_final_default(tmp_path, monkeypatch) -> None:
    """The default must be outside the repository."""

    monkeypatch.delenv(DATA_ROOT_ENV_VAR, raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))

    resolved = resolve_data_root()

    assert resolved == tmp_path / "home" / ".cache" / "llm-behavior-lab"
    assert REPO_ROOT not in resolved.parents


def test_blank_environment_values_fall_through(tmp_path, monkeypatch) -> None:
    """An empty variable should not select an empty path."""

    monkeypatch.setenv(DATA_ROOT_ENV_VAR, "   ")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))

    assert resolve_data_root() == tmp_path / "xdg" / "llm-behavior-lab"


def test_user_home_shorthand_is_expanded(monkeypatch) -> None:
    """``~`` in a configured root should not become a literal directory name."""

    monkeypatch.setenv(DATA_ROOT_ENV_VAR, "~/datasets")

    resolved = resolve_data_root()

    assert "~" not in str(resolved)
    assert resolved.is_absolute()


def test_resolve_data_root_does_not_create_anything(tmp_path, monkeypatch) -> None:
    """Selecting a root must stay a pure computation."""

    monkeypatch.setenv(DATA_ROOT_ENV_VAR, str(tmp_path / "unused"))

    resolved = resolve_data_root()

    assert not resolved.exists()


def test_every_dataset_error_shares_one_base() -> None:
    """Command-line callers should be able to catch the whole layer at once."""

    assert issubclass(DatasetConfigError, DatasetError)
    assert issubclass(UnsupportedDatasetSourceError, DatasetConfigError)

    with pytest.raises(DatasetError):
        DatasetConfig.from_config(_local_config(source="s3"))
