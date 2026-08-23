#!/usr/bin/env python3
"""List Dataset650 cases missing a CT or PET nnUNet channel (fingerprint crash)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nnunet_training.prepare_dataset import find_incomplete_nnunet_channel_cases


def _report(folder: Path, split: str) -> int:
    images = folder / f"images{split}"
    if not images.is_dir():
        print(f"  images{split}: (missing folder)")
        return 0
    n0000 = len(list(images.glob("*_0000.nii.gz")))
    n0001 = len(list(images.glob("*_0001.nii.gz")))
    missing = find_incomplete_nnunet_channel_cases(str(images), required_channels=(0, 1))
    print(f"  images{split}: CT={n0000} PET={n0001} incomplete={len(missing)}")
    for stem, miss in missing[:40]:
        print(f"    {stem}: missing " + ", ".join(f"_{c:04d}" for c in miss))
    if len(missing) > 40:
        print(f"    … and {len(missing) - 40} more")
    return len(missing)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find imagesTr/Ts/Va cases without both _0000 and _0001."
    )
    parser.add_argument(
        "--dataset-folder",
        default=os.getenv("DATASET_FOLDER", ""),
        help="Dataset650 folder (default: DATASET_FOLDER)",
    )
    args = parser.parse_args()
    if not args.dataset_folder.strip():
        raise SystemExit("Set DATASET_FOLDER or pass --dataset-folder")
    folder = Path(args.dataset_folder).expanduser().resolve()
    if not folder.is_dir():
        raise SystemExit(f"Not a directory: {folder}")
    print(f"Checking {folder}")
    n = 0
    for split in ("Tr", "Va", "Ts"):
        n += _report(folder, split)
    if n:
        print(
            "\nFix: python -m pipelines.test8_0.build_dataset --only-missing-pet\n"
            "Then re-run: python train_nnunet.py --step plan"
        )
        raise SystemExit(1)
    print("All splits have matching CT (_0000) and PET (_0001) counts per case.")


if __name__ == "__main__":
    main()
