"""Run the initialization-distribution experiment at three initialization scales.

The scientific control is one number, ``alpha``, multiplying every audited
zero-centred random weight:

.. code-block:: text

    alpha = 1.0   sigma per group        variance per group
    alpha = 0.5   sigma / 2              variance / 4
    alpha = 0.25  sigma / 4              variance / 16

**There is no single architecture-wide ``sigma_w``.** The token embedding is
``normal_(0, 1)`` while every linear is ``kaiming_uniform_(a=sqrt(5))`` with a
fan-in dependent scale, so ``alpha`` is the analogue of an intervention on one
``sigma_w``, not a rescaling of one value. Standard deviations are therefore
reported per parameter group, never as one number.

Everything else is held identical across the three conditions: dataset,
tokenizer and its pinned revision, eligible vocabulary, architecture, seed list,
evaluation positions, input controls and their seeds, greedy and nucleus
definitions, sampling seed, common random numbers, temperature grids, top-p, the
uniform null, and the whole gradient protocol. Each condition is a complete,
self-contained run in its own subdirectory, so nothing overwrites anything.

Each scale is launched as a **separate process** running the established
experiment script unchanged. That is deliberate: the scientific logic is not
duplicated or re-implemented here, and no state can leak between conditions.

This driver only orchestrates and records the pairing. It writes a manifest
proving which quantities were held fixed and which single knob moved.

Example::

    python3 scripts/run_initialization_scale_experiment.py \\
        --data-config configs/data/wikitext2_subword.yaml \\
        --model-config configs/model/tiny_llama_32k.yaml \\
        --offline --num-initializations 1 --num-windows 4 --gradient-analysis
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "run_initialization_distribution_experiment.py"

#: The three conditions, with the directory name each writes into. The label is
#: filesystem-safe and unmistakable: "0p25" cannot be misread as "0.25" truncated.
SCALES: tuple[tuple[float, str], ...] = (
    (1.0, "scale_1"),
    (0.5, "scale_0p5"),
    (0.25, "scale_0p25"),
)


def parse_args() -> argparse.Namespace:
    """Parse arguments, passing everything unrecognized to the runner."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "initialization_scale_experiment",
        help="Parent directory holding one subdirectory per scale.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Parent run identifier. Defaults to a timestamp.",
    )
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=None,
        metavar="ALPHA",
        help="Override the scale grid. The default is 1.0 0.5 0.25.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands and write the manifest without running anything.",
    )
    return parser.parse_known_args()[0], parser.parse_known_args()[1]


def _label(alpha: float) -> str:
    for value, name in SCALES:
        if value == alpha:
            return name
    return "scale_" + f"{alpha:g}".replace(".", "p")


def main() -> None:
    """Run every scale condition and write the parent manifest."""

    args, passthrough = parse_args()
    scales = (
        [(float(value), _label(float(value))) for value in args.scales]
        if args.scales
        else list(SCALES)
    )
    if any(alpha <= 0.0 for alpha, _ in scales):
        raise ValueError("Every initialization scale must be positive.")

    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    parent = args.output_root / run_id
    parent.mkdir(parents=True, exist_ok=True)

    print(f"Initialization-scale experiment: {parent}")
    print(f"Scales: {', '.join(f'{alpha:g}' for alpha, _ in scales)}")
    print(
        "Held identical across scales: dataset, tokenizer and revision, eligible "
        "vocabulary, architecture, seeds, evaluation positions, input controls, "
        "greedy and nucleus definitions, sampling seed, common random numbers, "
        "temperature grids, top-p, uniform null, and the whole gradient protocol."
    )
    print(f"Runner arguments shared by every scale: {' '.join(passthrough) or '(none)'}")

    conditions: list[dict[str, Any]] = []
    for alpha, label in scales:
        directory = parent / label
        command = [
            sys.executable,
            str(RUNNER),
            "--initialization-scale",
            repr(alpha),
            "--output-dir",
            str(directory),
            *passthrough,
        ]
        print(f"\n=== alpha = {alpha:g}  ->  {directory} ===")
        print("  " + " ".join(command))
        started = time.perf_counter()
        returncode = 0
        if not args.dry_run:
            completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
            returncode = completed.returncode
        seconds = time.perf_counter() - started
        conditions.append(
            {
                "alpha": alpha,
                "variance_factor": alpha**2,
                "is_no_op": alpha == 1.0,
                "label": label,
                "output_dir": str(directory),
                "command": command,
                "returncode": returncode,
                "seconds": round(seconds, 3),
            }
        )
        if returncode != 0:
            print(f"  FAILED with exit code {returncode}; stopping.")
            break

    manifest = {
        "experiment_type": "initialization_scale",
        "run_id": run_id,
        "scales": [alpha for alpha, _ in scales],
        "shared_runner_arguments": passthrough,
        "runner": str(RUNNER.relative_to(REPO_ROOT)),
        "pairing": {
            "same_underlying_random_draw": True,
            "how": (
                "Each condition builds the model from the same seed with the same "
                "architecture and then multiplies the audited random tensors by "
                "alpha. The draw, and therefore every sign and direction, is "
                "shared; only the magnitude differs."
            ),
            "alpha_1_is_literal_no_op": True,
            "scaled_parameter_classes": [
                "token embeddings",
                "attention query/key/value/output projections",
                "feed-forward projections",
                "output head",
            ],
            "unscaled_parameter_classes": [
                "RMSNorm gains (deterministic, initialized to one)",
            ],
            "bias_parameters": (
                "none exist: every nn.Linear is constructed bias=False, so the "
                "zero-bias condition is identical to every historical experiment"
            ),
            "has_single_sigma_w": False,
            "sigma_w_note": (
                "The embedding is normal_(0,1) and every linear is "
                "kaiming_uniform_(a=sqrt(5)) with a fan-in dependent scale, so no "
                "single sigma_w describes the architecture. alpha is the analogue "
                "of a sigma_w intervention; standard deviations are reported per "
                "parameter group."
            ),
        },
        "deferred": {
            "layerwise_gradient_stability": (
                "Not implemented in this phase. The model has two transformer "
                "blocks, which is too shallow for a depth-propagation claim."
            ),
            "cross_scale_figures": (
                "Not implemented in this phase. Three complete figure sets are "
                "produced first; comparative figures are designed after inspection."
            ),
        },
        "conditions": conditions,
    }
    manifest_path = parent / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"\nManifest: {manifest_path}")
    for condition in conditions:
        state = "ok" if condition["returncode"] == 0 else f"FAILED {condition['returncode']}"
        print(f"  alpha {condition['alpha']:<6g} {condition['label']:<12} {state}")

    if any(condition["returncode"] != 0 for condition in conditions):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
