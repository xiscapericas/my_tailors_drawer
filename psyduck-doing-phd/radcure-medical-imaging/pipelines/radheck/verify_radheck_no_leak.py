#!/usr/bin/env python3
"""
Check train/val vs test disjointness for the combined RADHECK nnUNet dataset.

Verifies:
  1) Within Dataset650: imagesTr, imagesVa, imagesTs have no overlapping case stems.
  2) RADCURE train+val vs RADCURE test (650 imagesTs) — disjoint.
  3) HECKTOR train+val (manifest stems) vs Dataset152 test — disjoint.
  4) split_manifest.json exclusion lists vs Dataset152.

Important: RADCURE uses case_0014 (4 digits), HECKTOR uses case_014 (3 digits). A naive
(Tr∪Va)∩152 check can false-alarm on shared strings; section 3 uses manifest HECKTOR IDs only.

Run from repo root:
    python -m pipelines.radheck.verify_radheck_no_leak \\
        --combined-dataset /path/to/Dataset650_TotalSegmentator \\
        --hecktor-test-dataset /path/to/Dataset152_TotalSegmentator \\
        --radcure-dataset /path/to/Dataset366_TotalSegmentator
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Set

from pipelines.radheck import build_nnunet_dataset as build_mod
from pipelines.radheck.nnunet_split_utils import all_radcure_stems, list_stems_in_split


def _load_build_helpers():
    return build_mod


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


def hecktor_manifest_stems(manifest: dict, build_mod, cases_root: str) -> Set[str]:
    stems: Set[str] = set()
    if not cases_root or not os.path.isdir(cases_root):
        return stems
    for cid in (manifest.get("hecktor_train_cases") or []) + (
        manifest.get("hecktor_val_cases") or []
    ):
        stems.add(build_mod.hecktor_processed_nnunet_base(cases_root, cid))
    return stems


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify RADHECK train/val vs test disjointness")
    parser.add_argument("--combined-dataset", required=True)
    parser.add_argument("--hecktor-test-dataset", required=True)
    parser.add_argument(
        "--radcure-dataset",
        default="",
        help="Dataset366 — used to separate RADCURE vs HECKTOR stems in combined train/val",
    )
    parser.add_argument("--hecktor-cases-root", default="")
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

    tr_stems = list_stems_in_split(combined, "Tr")
    va_stems = list_stems_in_split(combined, "Va")
    ts_stems = list_stems_in_split(combined, "Ts")
    train_val = tr_stems | va_stems
    hecktor_test_stems = build_mod.collect_nnunet_case_basenames_from_test_folder(hecktor_test)

    radcure_all: Set[str] = set()
    radcure_train_val: Set[str] = set()
    if args.radcure_dataset.strip() and os.path.isdir(args.radcure_dataset):
        rad = os.path.abspath(args.radcure_dataset)
        radcure_all = all_radcure_stems(rad)
        radcure_train_val = list_stems_in_split(rad, "Tr") | list_stems_in_split(rad, "Va")

    radcure_in_combined_tr_va = train_val & radcure_all if radcure_all else train_val
    hecktor_in_combined_tr_va = train_val - radcure_all if radcure_all else set()

    print("=" * 70)
    print("RADHECK leak check")
    print("=" * 70)
    print(f"Combined dataset:     {combined}")
    print(f"HECKTOR test dataset: {hecktor_test}")
    print(f"Counts — Tr: {len(tr_stems)}, Va: {len(va_stems)}, Ts: {len(ts_stems)}")
    print(f"HECKTOR test stems (152): {len(hecktor_test_stems)}")
    if radcure_all:
        print(
            f"In Tr∪Va — RADCURE stems: {len(radcure_in_combined_tr_va)}, "
            f"HECKTOR-only stems: {len(hecktor_in_combined_tr_va)}"
        )
    print()

    failures: List[str] = []

    print("1) Internal splits (Dataset650)")
    failures.extend(_report_overlap("Tr ∩ Va", tr_stems, va_stems))
    failures.extend(_report_overlap("Tr ∩ Ts", tr_stems, ts_stems))
    failures.extend(_report_overlap("Va ∩ Ts", va_stems, ts_stems))
    print()

    print("2) RADCURE train+val vs RADCURE test (650 imagesTs)")
    rad_check = radcure_in_combined_tr_va if radcure_all else train_val
    failures.extend(_report_overlap("RADCURE (Tr∪Va) ∩ Ts", rad_check, ts_stems))
    print()

    print("3) HECKTOR train+val vs HECKTOR test (Dataset152)")
    manifest_path = os.path.join(combined, "split_manifest.json")
    h_manifest_stems: Set[str] = set()
    if os.path.isfile(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        cases_root = args.hecktor_cases_root.strip() or manifest.get("hecktor_cases_root") or ""
        h_manifest_stems = hecktor_manifest_stems(manifest, build_mod, cases_root)
        h_tr_only = set()
        h_va_only = set()
        if cases_root and os.path.isdir(cases_root):
            for cid in manifest.get("hecktor_train_cases") or []:
                h_tr_only.add(build_mod.hecktor_processed_nnunet_base(cases_root, cid))
            for cid in manifest.get("hecktor_val_cases") or []:
                h_va_only.add(build_mod.hecktor_processed_nnunet_base(cases_root, cid))
        h_train_val = h_tr_only | h_va_only
        failures.extend(_report_overlap("HECKTOR manifest (Tr∪Va) ∩ 152", h_train_val, hecktor_test_stems))
        excluded = manifest.get("hecktor_excluded_case_folders") or []
        print(f"  Manifest excluded HECKTOR folders: {len(excluded)}")
        if len(excluded) == 0 and h_train_val & hecktor_test_stems:
            print("  WARNING: hecktor_excluded_case_folders is empty but HECKTOR leak detected.")
    elif hecktor_in_combined_tr_va:
        failures.extend(
            _report_overlap("HECKTOR-only (Tr∪Va) ∩ 152", hecktor_in_combined_tr_va, hecktor_test_stems)
        )
    else:
        print("  SKIP (no manifest / radcure-dataset): using naive Tr∪Va ∩ 152")
        failures.extend(_report_overlap("naive (Tr∪Va) ∩ 152", train_val, hecktor_test_stems))
    print()

    naive_152 = train_val & hecktor_test_stems
    hecktor_real = (h_manifest_stems & hecktor_test_stems) if h_manifest_stems else naive_152
    false_pos = naive_152 - hecktor_real
    if false_pos:
        print("4) Stem name collisions (RADCURE vs HECKTOR — NOT counted as leak)")
        print(
            f"  {len(false_pos)} stem(s) in Tr∪Va match Dataset152 names but are not "
            "HECKTOR manifest train/val cases (likely RADCURE case_00XX vs HECKTOR case_0XX):"
        )
        for s in sorted(false_pos)[:20]:
            in_rad = s in radcure_all if radcure_all else "?"
            print(f"       - {s}  (in Dataset366: {in_rad})")
        if len(false_pos) > 20:
            print(f"       ... and {len(false_pos) - 20} more")
        print()

    if args.radcure_dataset.strip() and os.path.isdir(args.radcure_dataset):
        print("5) RADCURE source Dataset366 Ts vs Dataset650 Ts")
        ts366 = list_stems_in_split(args.radcure_dataset, "Ts")
        failures.extend(_report_overlap("366 Ts vs 650 Ts (should match)", ts366, ts_stems))

    print()
    print("=" * 70)
    if failures:
        print(f"RESULT: LEAK DETECTED — {len(set(failures))} overlapping case stem(s).")
        print("If section 3 failed: python -m pipelines.radheck.remove_hecktor_test_leak")
        return 1
    if naive_152 and false_pos == naive_152:
        print("RESULT: OK — overlaps with Dataset152 are filename collisions only (not HECKTOR leak).")
    else:
        print("RESULT: OK — train/val disjoint from RADCURE and HECKTOR test pools.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
