"""Regenerate the frozen CountSketch reference. Run deliberately, never in CI.

The fixture pins the production map construction and the float64 CPU projection
as they behaved at the source commit below, *before* the device tables were
narrowed to int32/int8. Its whole value is being independent of the code it
checks, so it must be generated from the production source exactly as committed.

That is enforced here rather than asked for in prose: this script refuses to run
unless ``src/llm_behavior_lab/evaluation/position_gradients.py`` in the working
tree is **byte-identical to its blob at SOURCE_COMMIT**. The generator and the
fixture may themselves be untracked -- they are new files -- but the production
source may not differ by a single byte.

    python3 tests/assets/make_countsketch_golden.py

Regenerating requires deleting the existing fixture first, which is deliberate:
an accidental rerun cannot silently replace the reference with output from the
implementation it is supposed to police.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

#: The commit whose behaviour this fixture freezes.
SOURCE_COMMIT = "4df44246d6dff573e12ec40c8c62ce0c079bc008"
#: Repository-relative path of the production source being frozen.
SOURCE_PATH = "src/llm_behavior_lab/evaluation/position_gradients.py"

#: Bumped if the fixture's contents or meaning ever change.
FIXTURE_VERSION = 1

#: Deliberately tiny and irregular, so a bucket/sign ordering error cannot hide.
TENSOR_SIZES = (7, 5, 11, 3)
DIMENSION = 8
SEED = 20240917
OUTPUT = Path(__file__).resolve().parent / "countsketch_golden_map_v1.npz"


def _git(*args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        check=True,
        capture_output=True,
    ).stdout


def verify_source_matches_commit() -> str:
    """Refuse unless the production source equals its blob at ``SOURCE_COMMIT``.

    Returns:
        The SHA-256 of that blob, recorded in the fixture as provenance.

    Raises:
        SystemExit: If the commit is missing or the working source differs.
    """

    try:
        _git("rev-parse", "--verify", f"{SOURCE_COMMIT}^{{commit}}")
    except subprocess.CalledProcessError:
        raise SystemExit(f"REFUSING: source commit {SOURCE_COMMIT} is not in this repository.")

    committed = _git("cat-file", "blob", f"{SOURCE_COMMIT}:{SOURCE_PATH}")
    working = (REPO_ROOT / SOURCE_PATH).read_bytes()
    if committed != working:
        raise SystemExit(
            f"REFUSING: {SOURCE_PATH} differs from its blob at {SOURCE_COMMIT}.\n"
            "The reference must be generated from the committed implementation, "
            "not from the implementation it is meant to police. Restore the file "
            "to that commit, generate, then reapply your changes."
        )
    return hashlib.sha256(committed).hexdigest()


def synthetic_gradients() -> list[torch.Tensor]:
    """Fixed, RNG-free gradients so the test reconstructs them exactly."""

    return [
        torch.arange(size, dtype=torch.float64) * 0.5 - 1.25
        for size in TENSOR_SIZES
    ]


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit(
            f"REFUSING: {OUTPUT.name} already exists. Delete it deliberately to "
            "regenerate; an accidental rerun must not replace the reference."
        )

    source_sha256 = verify_source_matches_commit()

    from llm_behavior_lab.evaluation.position_gradients import (
        _GradientSketcher,
        production_sketch_tables,
    )

    tables = production_sketch_tables(TENSOR_SIZES, DIMENSION, SEED)
    buckets = np.concatenate([b.numpy() for b, _ in tables])
    signs = np.concatenate([s.numpy() for _, s in tables])

    parameters = [torch.zeros(size) for size in TENSOR_SIZES]
    sketcher = _GradientSketcher(parameters, dimension=DIMENSION, seed=SEED)
    sketch = sketcher.project(synthetic_gradients()).numpy()

    np.savez_compressed(
        OUTPUT,
        fixture_version=np.asarray(FIXTURE_VERSION, dtype=np.int64),
        source_commit=np.asarray(SOURCE_COMMIT),
        source_path=np.asarray(SOURCE_PATH),
        source_sha256=np.asarray(source_sha256),
        tensor_sizes=np.asarray(TENSOR_SIZES, dtype=np.int64),
        dimension=np.asarray(DIMENSION, dtype=np.int64),
        seed=np.asarray(SEED, dtype=np.int64),
        buckets=buckets,
        signs=signs,
        sketch=sketch,
    )

    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    print(f"source commit : {SOURCE_COMMIT}")
    print(f"source path   : {SOURCE_PATH}")
    print(f"source sha256 : {source_sha256}")
    print(f"wrote         : {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
    print(f"fixture sha256: {digest}")


if __name__ == "__main__":
    main()
