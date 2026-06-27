#!/usr/bin/env python3
"""
Remove nnUNet training / prediction artifacts without touching processed case data.

Safe to delete (default):
  - {NNUNET_RETRAIN_PATH}/nnUNet_preprocessed/
  - {NNUNET_RETRAIN_PATH}/nnUNet_results/
  - {NNUNET_RETRAIN_PATH}/logs/  (or LOG_DIR if set)
  - Symlink at {NNUNET_RETRAIN_PATH}/DatasetXXX_TotalSegmentator (link only, not target)
  - {DATASET_FOLDER}/labelsTs_predicted/
  - {DATASET_FOLDER}/labelsTs_dice_and_viz/

NOT deleted (processed inputs preserved):
  - HECKTOR case folders (…/output/image, …/output/labels)
  - RADCURE TotalSegmentatorRetrain case folders
  - Source combined dataset imagesTr/Va/Ts (unless --remove-combined-dataset)

Run from repo root:
    python -m pipelines.radheck.cleanup_retrain_artifacts \\
        --nnunet-retrain-path /path/to/nnunet_radheck_test_1_retrain \\
        --dataset-folder /path/to/Dataset650_TotalSegmentator \\
        --hecktor-test-dataset /path/to/Dataset152_TotalSegmentator

Add --dry-run to preview. Add --yes to skip confirmation.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional


def _paths_to_remove(
    nnunet_retrain_path: str,
    dataset_folder: Optional[str],
    hecktor_test_dataset: Optional[str],
    dataset_name: Optional[str],
    log_dir: Optional[str],
) -> List[tuple]:
    """Return list of (path, description) to remove."""
    out: List[tuple] = []
    retrain = Path(nnunet_retrain_path)

    for sub, desc in (
        ("nnUNet_preprocessed", "nnUNet preprocessed cache"),
        ("nnUNet_results", "trained models and predictions"),
    ):
        p = retrain / sub
        if p.exists():
            out.append((str(p), desc))

    logs = Path(log_dir) if log_dir else retrain / "logs"
    if logs.exists():
        out.append((str(logs), "training/prediction logs"))

    if dataset_name:
        raw_link = retrain / dataset_name
        if raw_link.is_symlink():
            out.append((str(raw_link), "nnUNet_raw symlink (target dataset kept)"))
        elif raw_link.is_dir():
            out.append(
                (
                    str(raw_link),
                    "nnUNet_raw dataset COPY (use --keep-raw-copy to skip deleting this tree)",
                )
            )

    for folder in (dataset_folder, hecktor_test_dataset):
        if not folder:
            continue
        base = Path(folder)
        for sub, desc in (
            ("labelsTs_predicted", "test predictions"),
            ("labelsTs_dice_and_viz", "evaluation PDFs and Dice CSVs"),
        ):
            p = base / sub
            if p.exists():
                out.append((str(p), f"{desc} under {base.name}"))

    return out


def _remove_path(path: str, dry_run: bool) -> None:
    p = Path(path)
    if not p.exists():
        return
    if dry_run:
        kind = "symlink" if p.is_symlink() else "dir" if p.is_dir() else "file"
        print(f"  [dry-run] would remove {kind}: {path}")
        return
    if p.is_symlink():
        p.unlink()
    elif p.is_dir():
        shutil.rmtree(p)
    else:
        p.unlink()
    print(f"  removed: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clean nnUNet retrain artifacts; keep processed HECKTOR/RADCURE case outputs"
    )
    parser.add_argument("--nnunet-retrain-path", required=True)
    parser.add_argument(
        "--dataset-folder",
        default="",
        help="Combined dataset (e.g. Dataset650) — removes labelsTs_predicted / dice viz only",
    )
    parser.add_argument(
        "--hecktor-test-dataset",
        default="",
        help="Dataset152 — removes prediction/eval outputs only",
    )
    parser.add_argument(
        "--dataset-name",
        default="",
        help="Dataset folder name under nnUNet_retrain (default: inferred from --dataset-folder)",
    )
    parser.add_argument("--log-dir", default="", help="Override LOG_DIR (default: {retrain}/logs)")
    parser.add_argument(
        "--keep-raw-copy",
        action="store_true",
        help="Do not delete a real directory copy under nnUNet_retrain (only symlinks)",
    )
    parser.add_argument(
        "--remove-combined-dataset",
        action="store_true",
        help="Also delete the entire --dataset-folder tree (rebuild from build_radheck after)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    dataset_folder = args.dataset_folder.strip() or None
    hecktor_test = args.hecktor_test_dataset.strip() or None
    dataset_name = args.dataset_name.strip()
    if not dataset_name and dataset_folder:
        dataset_name = Path(dataset_folder).name

    targets = _paths_to_remove(
        args.nnunet_retrain_path,
        dataset_folder,
        hecktor_test,
        dataset_name or None,
        args.log_dir.strip() or None,
    )

    if args.keep_raw_copy:
        targets = [
            (p, d)
            for p, d in targets
            if not (
                dataset_name
                and p == str(Path(args.nnunet_retrain_path) / dataset_name)
                and os.path.isdir(p)
                and not os.path.islink(p)
            )
        ]

    if args.remove_combined_dataset and dataset_folder:
        targets.append((dataset_folder, "full combined nnUNet dataset (rebuild required)"))

    print("=" * 70)
    print("Cleanup nnUNet retrain artifacts (processed cases NOT touched)")
    print("=" * 70)
    if not targets:
        print("Nothing to remove.")
        return 0

    print("Will remove:")
    for path, desc in targets:
        print(f"  - {path}  ({desc})")

    print("\nPreserved (not listed above):")
    print("  - HECKTOR per-case output/image and output/labels under cases root")
    print("  - RADCURE TotalSegmentatorRetrain/{case}/output/")
    if not args.remove_combined_dataset and dataset_folder:
        print(f"  - {dataset_folder}/imagesTr|Va|Ts (unless you dedupe or rebuild separately)")

    if not args.dry_run and not args.yes:
        reply = input("\nProceed? (yes/no): ").strip().lower()
        if reply != "yes":
            print("Aborted.")
            return 1

    for path, _desc in targets:
        _remove_path(path, args.dry_run)

    print("\nDone.")
    if args.dry_run:
        print("Re-run without --dry-run to apply.")
    else:
        print("Next: dedupe splits, rebuild if needed, then train_nnunet.py --step prepare --link-raw")
    return 0


if __name__ == "__main__":
    sys.exit(main())
