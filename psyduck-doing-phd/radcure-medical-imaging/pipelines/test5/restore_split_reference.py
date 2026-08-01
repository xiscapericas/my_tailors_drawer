#!/usr/bin/env python3
"""
Restore Test1 Dataset650 ``split_manifest.json`` after the original folder was deleted.

Writes the recovered manifest into the Test5 reference / output Dataset650 folder
and prints how splits will be reconstructed (Dataset366 + HECKTOR lists).

Example:

  export TEST5_WORK_ROOT=/media/HDD_8TB/xisca/work/retrain_test5
  export TEST5_REFERENCE_DATASET650=${TEST5_WORK_ROOT}/Dataset650_TotalSegmentator

  python -m pipelines.test5.restore_split_reference
  python -m pipelines.test5.build_dataset650 --link hardlink
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

BUNDLED_MANIFEST = (
    _REPO_ROOT / "experiments" / "artifacts" / "test1_dataset650_split_manifest.json"
)

DEFAULT_TARGET = (
    "/media/HDD_8TB/xisca/work/retrain_test5/Dataset650_TotalSegmentator"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Restore recovered Test1 split_manifest.json for Test5 build"
    )
    parser.add_argument(
        "--target",
        default=os.getenv("TEST5_REFERENCE_DATASET650", DEFAULT_TARGET),
        help="Dataset650 folder that should receive split_manifest.json",
    )
    parser.add_argument(
        "--manifest",
        default=os.getenv("TEST5_SPLIT_MANIFEST", str(BUNDLED_MANIFEST)),
        help="Source manifest JSON (default: bundled Test1 recovery)",
    )
    parser.add_argument(
        "--radcure-dataset366",
        default=os.getenv(
            "TEST5_RADCURE_DATASET366",
            os.getenv(
                "RADCURE_DATASET",
                "/media/HDD_8TB/xisca/work/nnunet_retrain_radcure366/Dataset366_TotalSegmentator",
            ),
        ),
        help="Dataset366 path used to reconstruct RADCURE Tr/Va/Ts stems",
    )
    args = parser.parse_args()

    src = Path(args.manifest).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Manifest not found: {src}")

    target = Path(args.target).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    dst = target / "split_manifest.json"

    with open(src, encoding="utf-8") as f:
        manifest = json.load(f)

    # Point dataset_folder at the new location; keep radcure_dataset overridable
    manifest["dataset_folder"] = str(target)
    if args.radcure_dataset366:
        rad = Path(args.radcure_dataset366).expanduser()
        manifest["radcure_dataset"] = str(rad)
        if not rad.is_dir():
            print(
                f"WARNING: Dataset366 not found at {rad}\n"
                "  build_dataset650 needs it to reconstruct RADCURE Tr/Va/Ts stems\n"
                "  when imagesTr/Va/Ts are empty on the reference Dataset650."
            )

    shutil.copy2(src, dst)  # keep original file copy first
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print("=" * 70)
    print("Restored split_manifest.json")
    print(f"  from: {src}")
    print(f"  to:   {dst}")
    print(f"  radcure_dataset (Dataset366): {manifest.get('radcure_dataset')}")
    print(f"  expected counts: {manifest.get('split_counts_after_dedupe')}")
    print("=" * 70)
    print("Next:")
    print("  export TEST5_REFERENCE_DATASET650=" + str(target))
    print("  export TEST5_SPLIT_MANIFEST=" + str(dst))
    print("  python -m pipelines.test5.build_dataset650 --link hardlink")


if __name__ == "__main__":
    main()
