"""Suite-wide safety net keeping the default test run offline.

Two hazards exist once the dataset layer can acquire data. A test could reach
the network, and a test could read or write the developer's real data root.
Both are prevented here rather than left to the discipline of individual tests.

The optional ``datasets`` package may or may not be installed, so neither
outcome can be assumed. Blocking the single seam through which it is imported
makes the guarantee independent of the environment.

A test that needs to exercise acquisition supplies its own fake by patching
``llm_behavior_lab.data.huggingface._import_load_dataset``.
"""

from __future__ import annotations

import pytest

from llm_behavior_lab.data import huggingface
from llm_behavior_lab.data.config import DATA_ROOT_ENV_VAR
from llm_behavior_lab.data.errors import DatasetAcquisitionError


@pytest.fixture(autouse=True)
def offline_dataset_environment(tmp_path_factory, monkeypatch):
    """Point the data root at a temporary directory and disable acquisition."""

    monkeypatch.setenv(DATA_ROOT_ENV_VAR, str(tmp_path_factory.mktemp("data_root")))
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)

    def blocked_loader():
        raise DatasetAcquisitionError(
            "The test suite runs offline and dataset acquisition is disabled. "
            "Patch llm_behavior_lab.data.huggingface._import_load_dataset to "
            "supply a fake loader."
        )

    monkeypatch.setattr(huggingface, "_import_load_dataset", blocked_loader)
