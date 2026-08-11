"""Errors raised by the dataset layer.

A single hierarchy keeps command-line handling simple: catching ``DatasetError``
catches every failure this layer can produce, while the individual types let a
caller distinguish a configuration mistake from genuinely missing data.

Only the configuration errors are raised at this stage. The remaining types are
defined together so the vocabulary stays stable as resolution, prepared data,
and acquisition are added.
"""

from __future__ import annotations


class DatasetError(RuntimeError):
    """Base class for every dataset-layer failure."""


class DatasetConfigError(DatasetError):
    """Raised when dataset configuration or resolution policy is invalid."""


class UnsupportedDatasetSourceError(DatasetConfigError):
    """Raised when a configuration requests an unknown dataset source."""


class DatasetPathNotFoundError(DatasetError):
    """Raised when an explicitly configured dataset path does not exist."""


class DatasetNotAvailableError(DatasetError):
    """Raised when a dataset is absent and acquisition is not permitted."""


class DatasetIntegrityError(DatasetError):
    """Raised when prepared data is incomplete or disagrees with its config."""


class DatasetAcquisitionError(DatasetError):
    """Raised when obtaining a dataset from an external source fails."""
