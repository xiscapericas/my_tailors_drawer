"""
Helpers to audit and enforce disjoint nnUNet splits (imagesTr / imagesVa / imagesTs).

Priority when a case stem appears in multiple splits (default): Ts > Tr > Va
  — test set is preserved; duplicates are removed from train/val.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

DEFAULT_SPLIT_PRIORITY: Tuple[str, ...] = ("Ts", "Tr", "Va")


def stem_from_image_filename(filename: str) -> str:
    """``case_0014_0000.nii.gz`` -> ``case_0014``."""
    if filename.endswith("_0000.nii.gz"):
        return filename.replace("_0000.nii.gz", "")
    if filename.endswith(".nii.gz"):
        return filename.replace(".nii.gz", "")
    return filename


def label_filename_for_stem(stem: str) -> str:
    return f"{stem}.nii.gz"


def image_filename_for_stem(stem: str) -> str:
    return f"{stem}_0000.nii.gz"


def list_stems_in_split(dataset_folder: str, split_suffix: str) -> Set[str]:
    """Case stems with paired image+label in images{split}/labels{split}."""
    img_dir = os.path.join(dataset_folder, f"images{split_suffix}")
    lbl_dir = os.path.join(dataset_folder, f"labels{split_suffix}")
    if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
        return set()
    stems: Set[str] = set()
    for name in os.listdir(img_dir):
        if not name.endswith(".nii.gz"):
            continue
        stem = stem_from_image_filename(name)
        lbl = os.path.join(lbl_dir, label_filename_for_stem(stem))
        if os.path.isfile(os.path.join(img_dir, name)) and os.path.isfile(lbl):
            stems.add(stem)
    return stems


def audit_split_overlaps(
    dataset_folder: str,
    split_suffixes: Sequence[str] = DEFAULT_SPLIT_PRIORITY,
) -> Dict[str, object]:
    """Return stem counts and pairwise overlaps for each split."""
    stems_by_split = {s: list_stems_in_split(dataset_folder, s) for s in split_suffixes}
    overlaps: Dict[str, List[str]] = {}
    splits = list(split_suffixes)
    for i, a in enumerate(splits):
        for b in splits[i + 1 :]:
            key = f"{a}∩{b}"
            overlaps[key] = sorted(stems_by_split[a] & stems_by_split[b])
    return {
        "dataset_folder": dataset_folder,
        "counts": {s: len(stems_by_split[s]) for s in splits},
        "stems_by_split": {s: sorted(stems_by_split[s]) for s in splits},
        "overlaps": overlaps,
    }


@dataclass
class DedupeReport:
    dataset_folder: str
    removed: Dict[str, List[str]] = field(default_factory=dict)
    kept: Dict[str, List[str]] = field(default_factory=dict)
    dry_run: bool = False

    def total_removed(self) -> int:
        return sum(len(v) for v in self.removed.values())


def _priority_index(split: str, priority: Sequence[str]) -> int:
    try:
        return priority.index(split)
    except ValueError:
        return len(priority)


def deduplicate_dataset_splits(
    dataset_folder: str,
    *,
    priority: Sequence[str] = DEFAULT_SPLIT_PRIORITY,
    dry_run: bool = False,
) -> DedupeReport:
    """
    Ensure each case stem appears in at most one split.

    For duplicates, keep the stem in the highest-priority split (lowest index in ``priority``)
    and delete image/label files from the other splits.
    """
    report = DedupeReport(dataset_folder=dataset_folder, dry_run=dry_run)
    stems_by_split = {s: list_stems_in_split(dataset_folder, s) for s in priority}

    stem_to_splits: Dict[str, List[str]] = {}
    for split, stems in stems_by_split.items():
        for stem in stems:
            stem_to_splits.setdefault(stem, []).append(split)

    for split in priority:
        report.kept[split] = []
        report.removed[split] = []

    for stem, splits_present in sorted(stem_to_splits.items()):
        if len(splits_present) == 1:
            report.kept[splits_present[0]].append(stem)
            continue
        keep_split = min(splits_present, key=lambda s: _priority_index(s, priority))
        report.kept[keep_split].append(stem)
        for split in splits_present:
            if split == keep_split:
                continue
            report.removed[split].append(stem)
            img_path = os.path.join(
                dataset_folder, f"images{split}", image_filename_for_stem(stem)
            )
            lbl_path = os.path.join(
                dataset_folder, f"labels{split}", label_filename_for_stem(stem)
            )
            if not dry_run:
                for path in (img_path, lbl_path):
                    if os.path.isfile(path):
                        os.remove(path)

    return report


def remove_stems_from_splits(
    dataset_folder: str,
    stems: Set[str],
    *,
    split_suffixes: Sequence[str] = ("Tr", "Va"),
    dry_run: bool = False,
) -> List[str]:
    """Remove case stems from given splits (images + labels). Returns stems actually removed."""
    removed: List[str] = []
    for stem in sorted(stems):
        found = False
        for split in split_suffixes:
            img_path = os.path.join(
                dataset_folder, f"images{split}", image_filename_for_stem(stem)
            )
            lbl_path = os.path.join(
                dataset_folder, f"labels{split}", label_filename_for_stem(stem)
            )
            for path in (img_path, lbl_path):
                if os.path.isfile(path):
                    found = True
                    if not dry_run:
                        os.remove(path)
        if found:
            removed.append(stem)
    return removed


def all_radcure_stems(radcure_dataset: str) -> Set[str]:
    """Union of Tr/Va/Ts stems in a RADCURE nnUNet dataset folder."""
    out: Set[str] = set()
    for split in ("Tr", "Va", "Ts"):
        out |= list_stems_in_split(radcure_dataset, split)
    return out


def print_audit(audit: Dict[str, object]) -> None:
    print(f"Dataset: {audit['dataset_folder']}")
    counts = audit["counts"]
    print(f"  Counts — Tr: {counts.get('Tr', 0)}, Va: {counts.get('Va', 0)}, Ts: {counts.get('Ts', 0)}")
    overlaps = audit["overlaps"]
    any_overlap = False
    for key, stems in overlaps.items():
        if stems:
            any_overlap = True
            print(f"  OVERLAP {key}: {len(stems)} case(s)")
            for s in stems[:15]:
                print(f"    - {s}")
            if len(stems) > 15:
                print(f"    ... and {len(stems) - 15} more")
    if not any_overlap:
        print("  No overlaps between Tr / Va / Ts.")


def print_dedupe_report(report: DedupeReport) -> None:
    mode = "DRY RUN" if report.dry_run else "APPLIED"
    print(f"Deduplicate splits ({mode}): {report.dataset_folder}")
    for split in DEFAULT_SPLIT_PRIORITY:
        n_removed = len(report.removed.get(split, []))
        n_kept = len(report.kept.get(split, []))
        if n_removed:
            print(f"  Removed {n_removed} duplicate(s) from {split} (kept elsewhere)")
            for stem in report.removed[split][:10]:
                print(f"    - {stem}")
            if n_removed > 10:
                print(f"    ... and {n_removed - 10} more")
        print(f"  {split}: {n_kept} unique case(s) after dedupe")
    print(f"  Total files removed from splits: {report.total_removed()}")


def write_dedupe_report_json(report: DedupeReport, path: str) -> None:
    payload = {
        "dataset_folder": report.dataset_folder,
        "dry_run": report.dry_run,
        "priority": list(DEFAULT_SPLIT_PRIORITY),
        "removed": report.removed,
        "kept_counts": {k: len(v) for k, v in report.kept.items()},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
