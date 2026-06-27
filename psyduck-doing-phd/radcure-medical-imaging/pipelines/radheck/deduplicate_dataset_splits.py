#!/usr/bin/env python3
"""
Audit and deduplicate nnUNet split folders (imagesTr / imagesVa / imagesTs).

Default priority when deduplicating: Ts > Tr > Va (test cases win).

Examples:
    # Audit only
    python -m pipelines.radheck.deduplicate_dataset_splits \\
        --dataset /path/to/Dataset366_TotalSegmentator --audit-only

    # Fix RADCURE source, then rebuild Dataset650
    python -m pipelines.radheck.deduplicate_dataset_splits \\
        --dataset /path/to/Dataset366_TotalSegmentator

    # Fix combined dataset in place
    python -m pipelines.radheck.deduplicate_dataset_splits \\
        --dataset /path/to/Dataset650_TotalSegmentator
"""

from __future__ import annotations

import argparse

from pipelines.radheck.nnunet_split_utils import (
    audit_split_overlaps,
    deduplicate_dataset_splits,
    print_audit,
    print_dedupe_report,
    write_dedupe_report_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit/deduplicate nnUNet Tr/Va/Ts splits")
    parser.add_argument("--dataset", required=True, help="DatasetXXX_TotalSegmentator folder")
    parser.add_argument("--audit-only", action="store_true", help="Report overlaps only")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be removed")
    parser.add_argument(
        "--report-json",
        default="",
        help="Write dedupe report JSON (default: {dataset}/dedupe_splits_report.json)",
    )
    args = parser.parse_args()

    dataset = Path(args.dataset).resolve()
    if not dataset.is_dir():
        print(f"ERROR: not a directory: {dataset}")
        return 1

    audit = audit_split_overlaps(str(dataset))
    print("=" * 70)
    print("Split audit")
    print("=" * 70)
    print_audit(audit)

    if args.audit_only:
        overlaps = audit["overlaps"]
        if any(overlaps[k] for k in overlaps):
            return 1
        return 0

    print()
    print("=" * 70)
    report = deduplicate_dataset_splits(str(dataset), dry_run=args.dry_run)
    print_dedupe_report(report)

    if not args.dry_run:
        report_path = args.report_json or str(dataset / "dedupe_splits_report.json")
        write_dedupe_report_json(report, report_path)
        print(f"Wrote {report_path}")

        audit2 = audit_split_overlaps(str(dataset))
        print()
        print("After dedupe:")
        print_audit(audit2)
        if any(audit2["overlaps"][k] for k in audit2["overlaps"]):
            print("ERROR: overlaps remain after dedupe")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
