"""Loading YAML configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load a YAML config file.

    Both checks matter for a config the user typed on the command line: a
    missing file and a file holding something other than a mapping are ordinary
    mistakes, and each is worth naming rather than surfacing later as an
    attribute error deep inside a run.
    """

    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise TypeError(f"Expected config dictionary, got {type(config)!r}")

    return config
