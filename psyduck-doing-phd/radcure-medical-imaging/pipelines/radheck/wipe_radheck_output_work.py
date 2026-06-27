#!/usr/bin/env python3
"""
Wipe combined RADHECK nnUNet dataset output (radheck_output_work) for a clean rebuild.

Removes Dataset*_TotalSegmentator folders under the output work directory only.
Does NOT touch:
  - HECKTOR processed cases (…/HECKTOR2025…/unzipped/…/output/)
  - Dataset366 / Dataset152
  - nnUNet retrain folder (unless --also-wipe-retrain)

Typical full reset on the server:

  # 1) Preview
  python -m pipelines.radheck.wipe_radheck_output_work \\
      --output-work /media/HDD_8TB/xisca/work/nnunet_radheck_test_1 \\
      --also-wipe-retrain /media/HDD_8TB/xisca/work/nnunet_radheck_test_1_retrain \\
      --dry-run

  # 2) Wipe
  python -m pipelines.radheck.wipe_radheck_output_work \\
      --output-work /media/HDD_8TB/xisca/work/nnunet_radheck_test_1 \\
      --also-wipe-retrain /media/HDD_8TB/xisca/work/nnunet_radheck_test_1_retrain \\
      --yes

  # 3) Rebuild dataset (processed HECKTOR already on disk)
  python -m pipelines.radheck.build_nnunet_dataset \\
      --config pipelines/radheck/radheck_server_paths.json \\
      --skip-download --skip-process \\
      --hecktor-cases-root /media/HDD_8TB/xisca/dataset/hecktor/HECKTOR2025_task1_training/unzipped/task1

  # 4) Verify + retrain (see printed next steps)
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import List, Tuple


def _dataset_dirs(output_work: Path) -> List[Path]:
    out: List[Path] = []
    if not output_work.is_dir():
        return out
    for name in sorted(os.listdir(output_work)):
        if name.startswith(".") or not name.startswith("Dataset"):
            continue
        if "_TotalSegmentator" not in name:
            continue
        p = output_work / name
        if p.is_dir() or p.is_symlink():
            out.append(p)
    return out


def _retrain_artifacts(retrain_path: Path) -> List[Tuple[Path, str]]:
    items: List[Tuple[Path, str]] = []
    if not retrain_path.is_dir():
        return items
    for sub, desc in (
        ("nnUNet_preprocessed", "preprocessed cache"),
        ("nnUNet_results", "trained models"),
        ("logs", "logs"),
    ):
        p = retrain_path / sub
        if p.exists():
            items.append((p, desc))
    for name in os.listdir(retrain_path):
        if name.startswith("Dataset") and "_TotalSegmentator" in name:
            p = retrain_path / name
            if p.exists():
                kind = "symlink" if p.is_symlink() else "nnUNet_raw copy"
                items.append((p, kind))
    return items


def _remove(path: Path, dry_run: bool) -> None:
    if not path.exists():
        return
    if dry_run:
        kind = "symlink" if path.is_symlink() else "dir" if path.is_dir() else "file"
        print(f"  [dry-run] would remove {kind}: {path}")
        return
    if path.is_symlink():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    print(f"  removed: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Wipe radheck_output_work dataset folder(s) for a clean rebuild"
    )
    parser.add_argument(
        "--output-work",
        required=True,
        help="radheck_output_work (e.g. .../nnunet_radheck_test_1)",
    )
    parser.add_argument(
        "--also-wipe-retrain",
        default="",
        help="Also remove nnUNet preprocessed/results/logs under this retrain path",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    output_work = Path(args.output_work).resolve()
    retrain = Path(args.also_wipe_retrain).resolve() if args.also_wipe_retrain.strip() else None

    targets: List[Tuple[Path, str]] = []
    for d in _dataset_dirs(output_work):
        targets.append((d, "combined nnUNet dataset (rebuild with build_nnunet_dataset.py)"))

    if retrain:
        targets.extend(_retrain_artifacts(retrain))

    print("=" * 70)
    print("Wipe RADHECK output for clean rebuild")
    print("=" * 70)
    print(f"Output work: {output_work}")
    if retrain:
        print(f"Retrain path: {retrain}")
    print()

    if not targets:
        print("Nothing to remove (no Dataset*_TotalSegmentator under output work).")
    else:
        print("Will remove:")
        for path, desc in targets:
            print(f"  - {path}  ({desc})")

    print("\nNOT removed:")
    print("  - HECKTOR case folders with output/image and output/labels")
    print("  - Dataset366 (RADCURE source) and Dataset152 (HECKTOR test)")
    print("  - TotalSegmentatorRetrain / per-case processing outputs")

    if not targets:
        return 0

    if not args.dry_run and not args.yes:
        reply = input("\nProceed? (yes/no): ").strip().lower()
        if reply != "yes":
            print("Aborted.")
            return 1

    for path, _ in targets:
        _remove(path, args.dry_run)

    print("\nDone.")
    if args.dry_run:
        print("Re-run with --yes (no --dry-run) to apply.")
        return 0

    print("\n--- Next: rebuild dataset ---")
    print(
        "python -m pipelines.radheck.build_nnunet_dataset \\\n"
        "  --config pipelines/radheck/radheck_server_paths.json \\\n"
        "  --skip-download --skip-process \\\n"
        "  --hecktor-cases-root /media/HDD_8TB/xisca/dataset/hecktor/HECKTOR2025_task1_training/unzipped/task1"
    )
    print("\n--- Then verify ---")
    print(
        "python -m pipelines.radheck.verify_radheck_no_leak \\\n"
        "  --combined-dataset <new DatasetXXX under output work> \\\n"
        "  --hecktor-test-dataset /media/HDD_8TB/xisca/work/nnunet_hecktor_test1/Dataset152_TotalSegmentator \\\n"
        "  --radcure-dataset /media/HDD_8TB/xisca/work/nnunet_retrain_radcure366/Dataset366_TotalSegmentator"
    )
    print("\n--- Then retrain ---")
    print("export DATASET_FOLDER=<new dataset path>")
    print("export NNUNET_RETRAIN_PATH=/media/HDD_8TB/xisca/work/nnunet_radheck_test_1_retrain")
    print("python train_nnunet.py --step prepare --link-raw")
    print("python train_nnunet.py --step plan")
    print("python train_nnunet.py --step train")
    return 0


if __name__ == "__main__":
    sys.exit(main())
