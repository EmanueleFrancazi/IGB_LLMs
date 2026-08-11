"""Data utilities for text-only language-modeling experiments."""

from llm_behavior_lab.data.config import (
    DATA_ROOT_ENV_VAR,
    HUGGINGFACE_SOURCE,
    LOCAL_TEXT_SOURCE,
    SUPPORTED_SOURCES,
    DatasetConfig,
    ResolutionPolicy,
    resolve_data_root,
)
from llm_behavior_lab.data.dataloader import CausalLMBatch, CausalLMBatcher
from llm_behavior_lab.data.errors import (
    DatasetAcquisitionError,
    DatasetConfigError,
    DatasetError,
    DatasetIntegrityError,
    DatasetNotAvailableError,
    DatasetPathNotFoundError,
    UnsupportedDatasetSourceError,
)
from llm_behavior_lab.data.resolver import (
    ROUTE_EXPLICIT_PATH,
    ROUTE_REPO_FIXTURE,
    ResolvedDataset,
    resolve_dataset,
    resolve_repo_path,
)
from llm_behavior_lab.data.text_dataset import TokenSplits, load_text_file, split_token_ids
from llm_behavior_lab.data.tokenizer import CharTokenizer

__all__ = [
    "CausalLMBatch",
    "CausalLMBatcher",
    "CharTokenizer",
    "DATA_ROOT_ENV_VAR",
    "DatasetAcquisitionError",
    "DatasetConfig",
    "DatasetConfigError",
    "DatasetError",
    "DatasetIntegrityError",
    "DatasetNotAvailableError",
    "DatasetPathNotFoundError",
    "HUGGINGFACE_SOURCE",
    "LOCAL_TEXT_SOURCE",
    "ROUTE_EXPLICIT_PATH",
    "ROUTE_REPO_FIXTURE",
    "ResolutionPolicy",
    "ResolvedDataset",
    "SUPPORTED_SOURCES",
    "TokenSplits",
    "UnsupportedDatasetSourceError",
    "load_text_file",
    "resolve_data_root",
    "resolve_dataset",
    "resolve_repo_path",
    "split_token_ids",
]
