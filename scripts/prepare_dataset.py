"""Prepare a dataset ahead of time.

Experiment workflows already obtain a missing dataset on their own. This script
exists for the cases where doing that inline is inconvenient: staging data on a
login node before submitting a batch job, warming a shared cache, or refreshing
a corpus after changing its configuration.

It resolves the dataset exactly as the experiment scripts do, so whatever it
prepares is what they will subsequently find. With --no-download or --offline it
reports what is already available without obtaining anything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

# Allow running from the repository root before editable installation.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from llm_behavior_lab.data import (  # noqa: E402
    DatasetConfig,
    DatasetError,
    add_dataset_arguments,
    resolution_policy_from_args,
    resolve_data_root,
    resolve_dataset,
)


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load a YAML config file."""

    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise TypeError(f"Expected config dictionary, got {type(config)!r}")

    return config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Prepare a dataset so later runs find it locally."
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=REPO_ROOT / "configs" / "data" / "tiny_text.yaml",
        help="Path to the data YAML config naming the dataset to prepare.",
    )
    add_dataset_arguments(parser)
    return parser.parse_args()


def main() -> None:
    """Resolve the configured dataset, obtaining it when permitted."""

    args = parse_args()
    data_config = load_yaml_config(args.data_config)
    dataset = DatasetConfig.from_config(data_config)
    policy = resolution_policy_from_args(args)

    print(f"Dataset: {dataset.name}")
    print(f"Source: {dataset.source}")
    print(f"Data root: {resolve_data_root(policy.data_root)}")
    print(
        "Acquisition: "
        + ("permitted" if policy.may_acquire else "disabled")
        + f" (offline={policy.offline}, allow_download={policy.allow_download})"
    )

    try:
        resolved = resolve_dataset(dataset, policy, repo_root=REPO_ROOT)
    except DatasetError as error:
        print(f"\n{error}", file=sys.stderr)
        raise SystemExit(1) from error

    print("\nDataset preparation completed successfully.")
    print(f"Resolved via: {resolved.route}")
    print(f"Path: {resolved.path}")
    manifest = resolved.details.get("manifest")
    if manifest:
        print(f"Characters: {manifest['num_characters']}")
        print(f"Documents: {manifest['num_documents']}")
        print(f"Prepared at: {manifest['created_at']}")


if __name__ == "__main__":
    main()
