#!/usr/bin/env python3
"""
Test5 Phase 3 — build Dataset650 from improved-preprocess relabels.

Uses the **same Tr/Va/Ts membership as Test4** (reference Dataset650
``split_manifest.json`` / ``images{Tr,Va,Ts}``), then **drops stems** that
failed anatomy QC in Phase 2 (or have no relabeled output because of QC).

Reference should be Test4's Dataset650 when available (identical splits to
Test3/Test4). Env: ``TEST5_REFERENCE_DATASET650``.

Example:

  export TEST5_WORK_ROOT=/media/.../work/retrain_test5
  export TEST5_REFERENCE_DATASET650=/media/.../work/retrain_test4/Dataset650_TotalSegmentator
  export ORGAN_DICTIONARY_PATH=${TEST5_WORK_ROOT}/radcure_dictionary_test5.json

  python -m pipelines.test5.build_dataset650 --dry-run
  python -m pipelines.test5.build_dataset650
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from image_processor.conventions import HECKTOR, RADCURE, get_nnunet_case_number
from pipelines.radheck.build_nnunet_dataset import copy_processed_hecktor_case, write_dataset_json
from pipelines.radheck.nnunet_split_utils import audit_split_overlaps, print_audit, stem_from_image_filename
from pipelines.test4.build_dataset650 import (
    _build_hecktor_stem_map,
    _copy_radcure_relabel,
    _list_split_images,
    _load_reference_manifest,
    _radcure_case_id_from_stem,
)


def _hecktor_stem(case_id: str) -> str:
    return f"case_{get_nnunet_case_number(case_id, HECKTOR)}"


def _load_qc_discarded_case_ids(work_root: Path) -> Set[str]:
    """Case IDs discarded by anatomy QC (from JSONL and/or CSV)."""
    discarded: Set[str] = set()
    jsonl = work_root / "logs" / "anatomy_qc" / "anatomy_qc_decisions.jsonl"
    if jsonl.is_file():
        with open(jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("decision") == "discard" or row.get("keep") is False:
                    cid = row.get("case_id")
                    if cid:
                        discarded.add(str(cid))
    csv_path = work_root / "anatomy_qc_discarded.csv"
    if csv_path.is_file():
        import csv

        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cid = row.get("case_id")
                if cid:
                    discarded.add(str(cid))
    return discarded


def _discarded_stems(discarded_case_ids: Set[str], hecktor_by_stem: Dict[str, str]) -> Set[str]:
    """Map discarded case IDs to nnUNet stems present in the reference splits."""
    stems: Set[str] = set()
    hecktor_id_to_stem = {v: k for k, v in hecktor_by_stem.items()}
    for case_id in discarded_case_ids:
        if case_id.startswith("RADCURE-"):
            num = case_id.replace("RADCURE-", "")
            stems.add(f"case_{num}")
        elif case_id in hecktor_id_to_stem:
            stems.add(hecktor_id_to_stem[case_id])
        else:
            # HECKTOR id may not be in train/val map (e.g. only in Ts via images)
            stems.add(_hecktor_stem(case_id))
    return stems


def build_dataset650(
    work_root: Path,
    reference_dataset650: Path,
    organ_dictionary_path: Path,
    dataset_id: str = "650",
    dry_run: bool = False,
) -> Path:
    manifest = _load_reference_manifest(reference_dataset650)
    hecktor_by_stem = _build_hecktor_stem_map(manifest)
    discarded_ids = _load_qc_discarded_case_ids(work_root)
    discarded_stems = _discarded_stems(discarded_ids, hecktor_by_stem)

    radcure_retrain = work_root / "TotalSegmentatorRetrain"
    hecktor_root = work_root / "hecktor"
    dataset_name = f"Dataset{dataset_id}_TotalSegmentator"
    dataset_folder = work_root / dataset_name

    if dataset_folder.is_dir() and not dry_run:
        print(f"Removing existing {dataset_folder}")
        shutil.rmtree(dataset_folder)

    counts = {"Tr": 0, "Va": 0, "Ts": 0}
    skipped_qc: List[str] = []
    missing: List[str] = []

    print(f"QC discarded case IDs: {len(discarded_ids)}")
    if discarded_ids:
        print("  e.g.", sorted(discarded_ids)[:8])

    for split in ("Tr", "Va", "Ts"):
        ref_images = _list_split_images(reference_dataset650, split)
        print(f"\n{split}: {len(ref_images)} reference cases")
        if dry_run:
            n_keep = sum(
                1
                for img in ref_images
                if stem_from_image_filename(img) not in discarded_stems
            )
            counts[split] = n_keep
            print(f"  dry-run keep≈{n_keep} (excluding QC discards)")
            continue

        dst_img = dataset_folder / f"images{split}"
        dst_lbl = dataset_folder / f"labels{split}"
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)

        for img_file in ref_images:
            stem = stem_from_image_filename(img_file)
            if stem in discarded_stems:
                skipped_qc.append(f"{split}/{stem}")
                continue
            try:
                if stem in hecktor_by_stem:
                    copy_processed_hecktor_case(
                        str(hecktor_root),
                        hecktor_by_stem[stem],
                        str(dst_img),
                        str(dst_lbl),
                    )
                else:
                    case_id = _radcure_case_id_from_stem(stem)
                    _copy_radcure_relabel(
                        radcure_retrain,
                        case_id,
                        str(dst_img),
                        str(dst_lbl),
                    )
                counts[split] += 1
            except FileNotFoundError as exc:
                # Relabel may have QC-discarded without writing JSONL keep lines only
                missing.append(f"{split}/{stem}: {exc}")

    if missing:
        raise RuntimeError(
            f"{len(missing)} case(s) missing Test5 relabeled output "
            f"(not listed as QC discard). First 10:\n"
            + "\n".join(missing[:10])
        )

    if dry_run:
        print("\nDry run — no files written.")
        print(f"Would skip QC: {len(skipped_qc)}")
        return dataset_folder

    print(f"\nSkipped (QC): {len(skipped_qc)}")
    audit = audit_split_overlaps(str(dataset_folder))
    if any(audit["overlaps"][k] for k in audit["overlaps"]):
        print("WARNING: split overlaps detected:")
        print_audit(audit)

    n_tr = len(list((dataset_folder / "imagesTr").glob("*.nii.gz")))
    write_dataset_json(
        str(dataset_folder),
        dataset_name,
        str(organ_dictionary_path),
        num_training=n_tr,
    )

    out_manifest = {
        **manifest,
        "test5_work_root": str(work_root),
        "reference_dataset650": str(reference_dataset650),
        "organ_dictionary_path": str(organ_dictionary_path),
        "dataset_folder": str(dataset_folder),
        "dataset_id": dataset_id,
        "split_source": "Test4/Test3 reference split_manifest + images{{Tr,Va,Ts}}",
        "label_source": (
            "Test5 Phase 2: improved bg + anatomy QC + separate GTVp/GTVn"
        ),
        "anatomy_qc_discarded_case_ids": sorted(discarded_ids),
        "anatomy_qc_skipped_stems": skipped_qc,
        "counts_built": counts,
        "background_mode": "improved",
    }
    man_path = dataset_folder / "split_manifest.json"
    with open(man_path, "w") as f:
        json.dump(out_manifest, f, indent=2)
    print(f"\nWrote {man_path}")
    print(f"Counts: {counts}")
    print(f"Done: {dataset_folder}")
    return dataset_folder


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Test5 Phase 3: build Dataset650 from Test4 splits, "
            "excluding anatomy-QC discards"
        )
    )
    parser.add_argument(
        "--work-root",
        default=os.getenv("TEST5_WORK_ROOT", "work/retrain_test5"),
        help="Phase 2 output root (env: TEST5_WORK_ROOT)",
    )
    parser.add_argument(
        "--reference-dataset650",
        default=os.getenv(
            "TEST5_REFERENCE_DATASET650",
            os.getenv("TEST4_REFERENCE_DATASET650", os.getenv("RADHECK_DATASET", "")),
        ),
        help=(
            "Reference Dataset650 with Test4/Test3 splits "
            "(prefer Test4 Dataset650; env: TEST5_REFERENCE_DATASET650)"
        ),
    )
    parser.add_argument(
        "--organ-dictionary-path",
        default=os.getenv("ORGAN_DICTIONARY_PATH", ""),
        help="Default: {work-root}/radcure_dictionary_test5.json",
    )
    parser.add_argument("--dataset-id", default="650")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    work_root = Path(args.work_root).resolve()
    if not args.reference_dataset650:
        print(
            "ERROR: set --reference-dataset650 or TEST5_REFERENCE_DATASET650 "
            "(prefer Test4 Dataset650 for identical Tr/Va/Ts)"
        )
        sys.exit(1)

    reference = Path(args.reference_dataset650).resolve()
    organ_dict = Path(
        args.organ_dictionary_path or (work_root / "radcure_dictionary_test5.json")
    ).resolve()

    if not reference.is_dir():
        raise FileNotFoundError(f"Reference dataset not found: {reference}")
    if not organ_dict.is_file():
        raise FileNotFoundError(f"Organ dictionary not found: {organ_dict}")

    print("=" * 70)
    print("Test5 Phase 3 — build Dataset650 (Test4 splits − QC discards)")
    print(f"Work root:     {work_root}")
    print(f"Reference:     {reference}")
    print(f"Organ dict:    {organ_dict}")
    print("=" * 70)

    build_dataset650(
        work_root=work_root,
        reference_dataset650=reference,
        organ_dictionary_path=organ_dict,
        dataset_id=str(args.dataset_id),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
