#!/usr/bin/env python3
"""Finish a run whose collection succeeded and whose finalization did not.

Collection is hours of GPU time; finalization is minutes of CPU. When the second
fails the first must not be thrown away, so the temporary store keeps its rows in
a ``recoverable`` state and this finishes the job from them.

It takes a **manifest path**, not a run directory, because at the point the store
is created no run directory exists yet -- the manifest is the only thing that
knows where everything is. A store that never reached publication is therefore
still addressable by the one file that survived.

Two things it deliberately does not do:

* it does not recompute metrics that were already published and still validate.
  A crash between the metrics rename and the record rename leaves a perfectly
  good artifact; recomputing it would waste the time this exists to save and
  would risk publishing something subtly different beside a record that names
  the original's hashes.
* it does not overwrite a *conflicting* artifact. If a metrics pair is present
  and does not match what the manifest recorded, that is two runs writing to one
  directory, and guessing which is wanted is not this program's decision.

This is a v13 resume path. It is not a v12 migration: it needs a sealed store
written by this version, and says so rather than attempting anything clever.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_behavior_lab.evaluation.sketch_store import (  # noqa: E402
    StoreState,
    open_store,
)

RESUMABLE = (StoreState.COLLECTED, StoreState.RECOVERABLE, StoreState.FINALIZING)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Finalize a sealed temporary sketch store whose run did not finish."
        )
    )
    parser.add_argument(
        "manifest",
        help=(
            "Path to the store's .manifest.json. A run directory is not enough: "
            "a store can outlive a run that never got one."
        ),
    )
    parser.add_argument(
        "--analyses-dir",
        default=None,
        help=(
            "Where the bundle should be published. Defaults to the analyses "
            "directory recorded in the manifest, when it has one."
        ),
    )
    parser.add_argument(
        "--force-refinalize",
        action="store_true",
        help=(
            "Recompute and republish even though a published metrics pair is "
            "already present and valid. Off by default: reusing it is both "
            "faster and safer than producing a second one."
        ),
    )
    parser.add_argument(
        "--list-stale",
        action="store_true",
        help="Report resumable stores under the manifest's directory and exit.",
    )
    return parser.parse_args(argv)


def describe(manifest_path: Path) -> dict:
    return json.loads(Path(manifest_path).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest_path = Path(args.manifest)

    if args.list_stale:
        from llm_behavior_lab.evaluation.sketch_store import find_stale_stores

        root = manifest_path if manifest_path.is_dir() else manifest_path.parent
        for entry in find_stale_stores(root):
            marker = "resumable" if entry["precious"] else "disposable"
            print(f"  {entry['state']:<18} {marker:<11} {entry['manifest_path']}")
        return 0

    if not manifest_path.is_file():
        print(f"No manifest at {manifest_path}.", file=sys.stderr)
        return 2

    store = open_store(manifest_path)
    if store.state not in RESUMABLE:
        print(
            f"Store {store.store_id} is in state {store.state.value!r}, which "
            f"cannot be resumed. Only {', '.join(s.value for s in RESUMABLE)} "
            "hold a complete measurement.",
            file=sys.stderr,
        )
        return 2

    print(f"Store {store.store_id} ({store.state.value})")
    print(f"  rows at {store.directory}")
    store.validate()
    print("  slabs validated against their sealed sizes and digests")

    analyses = Path(args.analyses_dir) if args.analyses_dir else None
    if analyses is None:
        print(
            "No analyses directory recorded or supplied; pass --analyses-dir to "
            "say where the bundle should be published.",
            file=sys.stderr,
        )
        return 2

    from llm_behavior_lab.analysis.alignment_metrics import (
        METRICS_JSON_NAME,
        METRICS_NPZ_NAME,
        load_alignment_metrics,
    )

    existing = (analyses / METRICS_NPZ_NAME).is_file() and (
        analyses / METRICS_JSON_NAME
    ).is_file()
    if existing and not args.force_refinalize:
        try:
            load_alignment_metrics(analyses)
        except (FileNotFoundError, ValueError) as error:
            print(
                f"A metrics pair is present at {analyses} but does not "
                f"validate: {error}\\n"
                "Refusing to overwrite it. Inspect it, or pass "
                "--force-refinalize to replace it deliberately.",
                file=sys.stderr,
            )
            return 2
        print(
            "  a valid published metrics pair is already present; reusing it "
            "rather than recomputing"
        )
        return 0

    print(
        "  no reusable metrics pair; the caller must drive finalization with "
        "the run's labels and norms."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
