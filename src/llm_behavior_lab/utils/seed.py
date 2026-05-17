"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed common random-number generators.

    Args:
        seed: Integer seed used for Python, NumPy, PyTorch, and CUDA.
        deterministic: If true, request deterministic PyTorch algorithms where
            possible. This can make debugging easier, but may reduce speed or
            raise errors for unsupported operations.
    """

    if seed < 0:
        raise ValueError("seed must be non-negative.")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
