#!/usr/bin/env python3
"""
Check train/val vs test disjointness for the combined RADHECK nnUNet dataset.

Verifies:
  1) Within Dataset650: imagesTr, imagesVa, imagesTs have no overlapping case stems.
  2) Dataset650 train+val vs RADCURE test (imagesTs) — should be empty overlap.
  3) Dataset650 train+val vs HECKTOR test (Dataset152 imagesTs) — should be empty overlap.
  4) split_manifest.json HECKTOR train/val lists vs excluded test cases (if manifest present).

Run from repo root:
    python research_notebooks/retrain_radheck/verify_radheck_no_leak.py \\
        --combined-dataset /path/to/Dataset650_TotalSegmentator \\
        --hecktor-test-dataset /path/to/Dataset152_TotalSegmentator

Optional:
    --radcure-dataset /path/to/Dataset366_TotalSegmentator  (compare 650 Ts vs 366 Ts)
    --hecktor-cases-root /path/to/training/cases  (re-run exclusion logic vs manifest)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]


def _load_build_helpers():
    build_path = _SCRIPT_DIR / "build_radheck_nnunet_dataset.py"
    spec = importlib.util.spec_from_file_location("build_radheck_nnunet_dataset", build_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {build_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def nnunet_stems_from_images_dir(images_dir: str) -> Set[str]:
    """Case stems from an images* folder (handles _0000 suffix)."""
    out: Set[str] = set()
    if not os.path.isdir(images_dir):
        return out
    for name in os.listdir(images_dir):
        if not name.endswith(".nii.gz"):
            continue
        if name.endswith("_0000.nii.gz"):
            out.add(name.replace("_0000.nii.gz", ""))
        else:
            out.add(name.replace(".nii.gz", ""))
    return out


def split_stems(combined_dataset: str) -> Dict[str, Set[str]]:
    base = Path(combined_dataset)
    return {
        "Tr": nnunet_stems_from_images_dir(str(base / "imagesTr")),
        "Va": nnunet_stems_from_images_dir(str(base / "imagesVa")),
        "Ts": nnunet_stems_from_images_dir(str(base / "imagesTs")),
    }


def _report_overlap(label: str, a: Set[str], b: Set[str], limit: int = 30) -> List[str]:
    overlap = sorted(a & b)
    if overlap:
        print(f"  FAIL {label}: {len(overlap)} overlapping stem(s)")
        for stem in overlap[:limit]:
            print(f"       - {stem}")
        if len(overlap) > limit:
            print(f"       ... and {len(overlap) - limit} more")
    else:
        print(f"  OK   {label}: no overlap ({len(a)} vs {len(b)} stems checked)")
    return overlap


def hecktor_stems_from_manifest(
    manifest: dict,
    build_mod,
) -> Tuple[Set[str], Set[str], Set[str]]:
    """Return (train_stems, val_stems, excluded_folder_ids) from manifest + cases_root."""
    cases_root = manifest.get("hecktor_cases_root") or ""
    train_ids = manifest.get("hecktor_train_cases") or []
    val_ids = manifest.get("hecktor_val_cases") or []
    excluded = set(manifest.get("hecktor_excluded_case_folders") or [])

    train_stems: Set[str] = set()
    val_stems: Set[str] = set()
    if cases_root and os.path.isdir(cases_root):
        for cid in train_ids:
            train_stems.add(build_mod.hecktor_processed_nnunet_base(cases_root, cid))
        for cid in val_ids:
            val_stems.add(build_mod.hecktor_processed_nnunet_base(cases_root, cid))
    return train_stems, val_stems, excluded


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify RADHECK train/val vs test disjointness")
    parser.add_argument(
        "--combined-dataset",
        required=True,
        help="Path to Dataset650_TotalSegmentator (combined train/val/Ts)",
    )
    parser.add_argument(
        "--hecktor-test-dataset",
        required=True,
        help="Path to Dataset152_TotalSegmentator (HECKTOR held-out test)",
    )
    parser.add_argument(
        "--radcure-dataset",
        default="",
        help="Optional Dataset366 path to confirm Ts matches source RADCURE test",
    )
    parser.add_argument(
        "--hecktor-cases-root",
        default="",
        help="Optional HECKTOR training cases root to re-run exclusion check",
    )
    args = parser.parse_args()

    build_mod = _load_build_helpers()
    combined = os.path.abspath(args.combined_dataset)
    hecktor_test = os.path.abspath(args.hecktor_test_dataset)

    if not os.path.isdir(combined):
        print(f"ERROR: combined dataset not found: {combined}")
        return 1
    if not os.path.isdir(hecktor_test):
        print(f"ERROR: HECKTOR test dataset not found: {hecktor_test}")
        return 1

    stems = split_stems(combined)
    train_val = stems["Tr"] | stems["Va"]
    hecktor_test_stems = build_mod.collect_nnunet_case_basenames_from_test_folder(hecktor_test)

    print("=" * 70)
    print("RADHECK leak check")
    print("=" * 70)
    print(f"Combined dataset:     {combined}")
    print(f"HECKTOR test dataset: {hecktor_test}")
    print(f"Counts — Tr: {len(stems['Tr'])}, Va: {len(stems['Va'])}, Ts: {len(stems['Ts'])}")
    print(f"Train+val stems: {len(train_val)} | RADCURE Ts stems: {len(stems['Ts'])} | HECKTOR test stems: {len(hecktor_test_stems)}")
    print()

    failures: List[str] = []

    print("1) Internal splits (Dataset650)")
    failures.extend(_report_overlap("Tr ∩ Va", stems["Tr"], stems["Va"]))
    failures.extend(_report_overlap("Tr ∩ Ts", stems["Tr"], stems["Ts"]))
    failures.extend(_report_overlap("Va ∩ Ts", stems["Va"], stems["Ts"]))
    print()

    print("2) Training data vs RADCURE test (imagesTs in Dataset650)")
    failures.extend(_report_overlap(" (Tr ∪ Va) ∩ Ts", train_val, stems["Ts"]))
    print()

    print("3) Training data vs HECKTOR test (Dataset152)")
    failures.extend(_report_overlap(" (Tr ∪ Va) ∩ Dataset152", train_val, hecktor_test_stems))
    print()

    manifest_path = os.path.join(combined, "split_manifest.json")
    if os.path.isfile(manifest_path):
        print("4) split_manifest.json")
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        h_tr, h_va, excluded_folders = hecktor_stems_from_manifest(manifest, build_mod)
        print(f"  HECKTOR train cases (folders): {len(manifest.get('hecktor_train_cases') or [])}")
        print(f"  HECKTOR val cases (folders):   {len(manifest.get('hecktor_val_cases') or [])}")
        print(f"  Excluded from train/val:       {len(excluded_folders)}")
        if excluded_folders:
            print(f"  Excluded folder sample: {sorted(excluded_folders)[:10]}")

        if h_tr or h_va:
            failures.extend(_report_overlap("manifest HECKTOR Tr stems ∩ Dataset152", h_tr, hecktor_test_stems))
            failures.extend(_report_overlap("manifest HECKTOR Va stems ∩ Dataset152", h_va, hecktor_test_stems))

        cases_root = args.hecktor_cases_root.strip() or (manifest.get("hecktor_cases_root") or "")
        test_path_cfg = manifest.get("hecktor_test_dataset_excluded_from_train_val") or hecktor_test
        if cases_root and os.path.isdir(cases_root) and os.path.isdir(test_path_cfg):
            print()
            print("5) Re-run build exclusion logic (training zip vs configured test folder)")
            all_ids = build_mod.list_processed_hecktor_case_ids(cases_root)
            test_basenames = build_mod.collect_nnunet_case_basenames_from_test_folder(test_path_cfg)
            kept, excluded = build_mod.filter_hecktor_cases_not_in_test_set(
                cases_root, all_ids, test_basenames
            )
            kept_stems = {
                build_mod.hecktor_processed_nnunet_base(cases_root, cid) for cid in kept
            }
            leak_recheck = kept_stems & test_basenames
            if leak_recheck:
                print(f"  FAIL re-check: {len(leak_recheck)} kept training case(s) still in test basenames")
                for s in sorted(leak_recheck)[:20]:
                    print(f"       - {s}")
                failures.extend(list(leak_recheck))
            else:
                print(
                    f"  OK   re-check: {len(excluded)} excluded, {len(kept)} kept; "
                    f"none of kept stems appear in test folder"
                )
            # Stems in 650 train/val that are HECKTOR but should not be in 152
            hecktor_in_combined_tr = train_val & (h_tr | h_va) if (h_tr or h_va) else set()
            if hecktor_in_combined_tr:
                failures.extend(
                    _report_overlap("combined Tr∪Va HECKTOR stems ∩ Dataset152", hecktor_in_combined_tr, hecktor_test_stems)
                )
    else:
        print("4) split_manifest.json not found — skip manifest checks")

    if args.radcure_dataset:
        rad = os.path.abspath(args.radcure_dataset)
        if os.path.isdir(rad):
            print()
            print("6) RADCURE source Dataset366 Ts vs Dataset650 Ts")
            ts366 = nnunet_stems_from_images_dir(os.path.join(rad, "imagesTs"))
            failures.extend(_report_overlap("366 Ts vs 650 Ts (should match)", ts366, stems["Ts"]))
            only_366 = ts366 - stems["Ts"]
            only_650 = stems["Ts"] - ts366
            if only_366:
                print(f"  NOTE: in 366 Ts but not 650 Ts: {len(only_366)}")
            if only_650:
                print(f"  NOTE: in 650 Ts but not 366 Ts: {len(only_650)}")
        else:
            print(f"WARNING: --radcure-dataset not found: {rad}")

    print()
    print("=" * 70)
    if failures:
        print(f"RESULT: LEAK DETECTED — {len(set(failures))} overlapping case stem(s). Review lists above.")
        return 1
    print("RESULT: OK — train/val stems are disjoint from both test pools (by nnUNet filename stem).")
    print()
    print("Notes:")
    print("  - nnUNet trains on imagesTr only (fold CV); imagesVa is extra held-out, not used by default trainer.")
    print("  - This check uses filename stems; it cannot detect same patient under different names (RADCURE vs HECKTOR).")
    print("  - If build ran with --no-exclude-hecktor-test or missing Dataset152 path, HECKTOR leak is possible.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
