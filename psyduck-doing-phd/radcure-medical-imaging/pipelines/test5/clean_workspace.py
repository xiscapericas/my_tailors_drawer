#!/usr/bin/env python3
"""
Clean Test5 work root for a from-scratch restart.

Removes legacy RADCURE/HECKTOR split trees and old Dataset*/nnunet outputs.
Does **not** touch original dataset sources.

Example:

  export TEST5_WORK_ROOT=/media/HDD_8TB/xisca/work/retrain_test5
  python -m pipelines.test5.clean_workspace --dry-run
  python -m pipelines.test5.clean_workspace --yes
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from pipelines.test5.paths import LEGACY_WORK_ENTRIES, work_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean Test5 work root (from scratch)")
    parser.add_argument(
        "--work-root",
        default=str(work_root()),
        help="TEST5 work root (default: TEST5_WORK_ROOT)",
    )
    parser.add_argument(
        "--also-radheck",
        action="store_true",
        help="Also delete any RADHECK_* folders under work-root",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Required to actually delete (safety)",
    )
    args = parser.parse_args()

    root = Path(args.work_root).expanduser().resolve()
    if not root.is_dir():
        print(f"Work root does not exist yet (nothing to clean): {root}")
        return

    targets: list[Path] = []
    for name in LEGACY_WORK_ENTRIES:
        p = root / name
        if p.exists() or p.is_symlink():
            targets.append(p)
    if args.also_radheck:
        for p in sorted(root.glob("RADHECK_*")):
            targets.append(p)
    # split_manifest at work root (will be restored later)
    man = root / "split_manifest.json"
    if man.is_file():
        targets.append(man)

    print(f"Work root: {root}")
    if not targets:
        print("Nothing to remove.")
        return
    print("Will remove:")
    for p in targets:
        kind = "dir" if p.is_dir() and not p.is_symlink() else "file"
        print(f"  [{kind}] {p}")

    if args.dry_run:
        print("Dry-run — no deletes.")
        return
    if not args.yes:
        print("Refusing to delete without --yes (or use --dry-run).")
        sys.exit(2)

    for p in targets:
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p)
        else:
            p.unlink(missing_ok=True)
        print(f"  removed {p}")
    print("Done. Next: python -m pipelines.test5.transform_cases")


if __name__ == "__main__":
    main()
