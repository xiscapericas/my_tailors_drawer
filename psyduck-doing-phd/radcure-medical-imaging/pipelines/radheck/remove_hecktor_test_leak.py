#!/usr/bin/env python3
"""
Remove HECKTOR held-out test cases from Dataset650 train/val (imagesTr / imagesVa).

Use when verify_radheck_no_leak reports overlap between train+val and Dataset152.
Only removes stems tied to HECKTOR case folders in split_manifest.json (not RADCURE
stems that happen to share the same case_00X string).

Run from repo root:
    python -m pipelines.radheck.remove_hecktor_test_leak \\
        --combined-dataset /path/to/Dataset650_TotalSegmentator \\
        --hecktor-test-dataset /path/to/Dataset152_TotalSegmentator \\
        --dry-run

Then re-run verify_radheck_no_leak and refresh dataset.json (train_nnunet prepare).
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Set

from pipelines.radheck import build_nnunet_dataset as build_mod
from pipelines.radheck.nnunet_split_utils import (
    audit_split_overlaps,
    list_stems_in_split,
    print_audit,
    remove_stems_from_splits,
)


def _load_build_helpers():
    return build_mod


def hecktor_train_val_stems(manifest: dict, build_mod, cases_root: str) -> Set[str]:
    stems: Set[str] = set()
    for cid in (manifest.get("hecktor_train_cases") or []) + (
        manifest.get("hecktor_val_cases") or []
    ):
        stems.add(build_mod.hecktor_processed_nnunet_base(cases_root, cid))
    return stems


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove HECKTOR test cases from combined dataset train/val"
    )
    parser.add_argument("--combined-dataset", required=True)
    parser.add_argument("--hecktor-test-dataset", required=True)
    parser.add_argument(
        "--hecktor-cases-root",
        default="",
        help="Override manifest hecktor_cases_root if needed",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    combined = Path(args.combined_dataset).resolve()
    hecktor_test = Path(args.hecktor_test_dataset).resolve()
    manifest_path = combined / "split_manifest.json"
    if not manifest_path.is_file():
        print(f"ERROR: missing {manifest_path}")
        return 1

    build_mod = _load_build_helpers()
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    cases_root = args.hecktor_cases_root.strip() or manifest.get("hecktor_cases_root") or ""
    if not cases_root or not os.path.isdir(cases_root):
        print(f"ERROR: HECKTOR cases root not found: {cases_root!r}")
        return 1

    test_stems = build_mod.collect_nnunet_case_basenames_from_test_folder(str(hecktor_test))
    h_stems = hecktor_train_val_stems(manifest, build_mod, cases_root)
    train_val = list_stems_in_split(str(combined), "Tr") | list_stems_in_split(str(combined), "Va")

    # HECKTOR train/val cases that also appear in Dataset152 test
    hecktor_leak_stems = h_stems & test_stems
    # Stems in Tr/Va that are both HECKTOR (per manifest) and in test
    to_remove = hecktor_leak_stems & train_val

    # Naive stem overlap (may include RADCURE false positives)
    naive_overlap = train_val & test_stems

    print("=" * 70)
    print("Remove HECKTOR test leak from train/val")
    print("=" * 70)
    print(f"Combined:     {combined}")
    print(f"HECKTOR test: {hecktor_test}")
    print(f"Test stems (152): {len(test_stems)}")
    print(f"HECKTOR train+val stems (manifest): {len(h_stems)}")
    print(f"Naive (Tr∪Va) ∩ 152: {len(naive_overlap)} stem(s)")
    print(f"HECKTOR-only leak to remove: {len(to_remove)} stem(s)")

    if naive_overlap - to_remove:
        print(
            f"\nNote: {len(naive_overlap - to_remove)} stem(s) overlap by name only "
            "(likely RADCURE case_00XX vs HECKTOR — not removed):"
        )
        for s in sorted(naive_overlap - to_remove)[:15]:
            print(f"  - {s}")
        if len(naive_overlap - to_remove) > 15:
            print(f"  ... and {len(naive_overlap - to_remove) - 15} more")

    if not to_remove:
        print("\nNothing to remove (no HECKTOR manifest stems in Dataset152 test).")
        if naive_overlap:
            print(
                "If verify still fails, overlap may be RADCURE/HECKTOR filename collision — "
                "see updated verify_radheck_no_leak.py."
            )
        return 0

    print("\nRemoving from Tr/Va:")
    for s in sorted(to_remove)[:20]:
        print(f"  - {s}")
    if len(to_remove) > 20:
        print(f"  ... and {len(to_remove) - 20} more")

    removed = remove_stems_from_splits(
        str(combined), to_remove, split_suffixes=("Tr", "Va"), dry_run=args.dry_run
    )
    print(f"\n{'Would remove' if args.dry_run else 'Removed'} {len(removed)} stem(s) from Tr/Va.")

    if not args.dry_run:
        report = {
            "removed_stems": sorted(removed),
            "hecktor_test_dataset": str(hecktor_test),
            "naive_overlap_count": len(naive_overlap),
            "hecktor_leak_removed_count": len(removed),
        }
        out_path = combined / "hecktor_test_leak_removal.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Wrote {out_path}")
        print("\nAfter dedupe audit:")
        print_audit(audit_split_overlaps(str(combined)))
        print("\nRe-run: verify_radheck_no_leak.py")
        print("Update numTraining: python train_nnunet.py --step prepare --link-raw")

    return 0


if __name__ == "__main__":
    sys.exit(main())
