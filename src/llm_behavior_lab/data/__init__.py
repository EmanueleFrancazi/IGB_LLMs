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
from llm_behavior_lab.data.huggingface import (
    hf_cache_directory,
    materialize,
)
from llm_behavior_lab.data.prepared import (
    MANIFEST_FILENAME,
    TEXT_FILENAME,
    clean_partial_directories,
    load_prepared,
    prepared_directory,
    write_prepared,
)
from llm_behavior_lab.data.resolver import (
    ROUTE_ACQUIRED,
    ROUTE_DATA_ROOT_PREPARED,
    ROUTE_EXPLICIT_PATH,
    ROUTE_REPO_FIXTURE,
    ROUTE_REPO_PREPARED,
    ROUTE_SOURCE_CACHE,
    ResolvedDataset,
    acquisition_destination,
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
    "MANIFEST_FILENAME",
    "ROUTE_ACQUIRED",
    "ROUTE_DATA_ROOT_PREPARED",
    "ROUTE_EXPLICIT_PATH",
    "ROUTE_REPO_FIXTURE",
    "ROUTE_REPO_PREPARED",
    "ROUTE_SOURCE_CACHE",
    "ResolutionPolicy",
    "ResolvedDataset",
    "SUPPORTED_SOURCES",
    "TEXT_FILENAME",
    "TokenSplits",
    "UnsupportedDatasetSourceError",
    "acquisition_destination",
    "clean_partial_directories",
    "hf_cache_directory",
    "load_prepared",
    "load_text_file",
    "materialize",
    "prepared_directory",
    "resolve_data_root",
    "resolve_dataset",
    "resolve_repo_path",
    "split_token_ids",
    "write_prepared",
]
