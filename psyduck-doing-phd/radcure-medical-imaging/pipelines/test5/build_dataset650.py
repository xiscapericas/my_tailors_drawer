#!/usr/bin/env python3
"""
Test5 Phase 3 — build Dataset650 from improved-preprocess relabels.

Uses the **same Tr/Va/Ts membership** as Test2/Test3 (reference Dataset650),
drops anatomy-QC discards, and copies labels with this preference:

1. Test5 Phase 2 relabel (improved bg) — preferred
2. Test4 Phase 2 relabel (separate GTV, old bg)
3. Test4 Dataset650 (if present)
4. Reference Dataset650 (Test2/Test3 labels) — last resort

Test4/Test5 often only have ~11 HECKTOR cases in ``work/*/hecktor``; the
reference split expects many more. Fallbacks keep split membership intact;
``split_manifest.json`` records ``copy_source_counts``.

Example:

  export TEST5_WORK_ROOT=/media/.../work/retrain_test5
  export TEST5_REFERENCE_DATASET650=/media/.../nnunet_radheck_test_1/Dataset650_TotalSegmentator
  export TEST4_WORK_ROOT=/media/.../work/retrain_test4
  export ORGAN_DICTIONARY_PATH=${TEST5_WORK_ROOT}/radcure_dictionary_test5.json

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

from image_processor.conventions import HECKTOR, get_nnunet_case_number
from pipelines.radheck.build_nnunet_dataset import copy_processed_hecktor_case, write_dataset_json
from pipelines.radheck.nnunet_split_utils import audit_split_overlaps, print_audit, stem_from_image_filename
from pipelines.test4.build_dataset650 import (
    _build_hecktor_stem_map,
    _copy_radcure_relabel,
    _list_split_images,
    _radcure_case_id_from_stem,
)


def _hecktor_stem(case_id: str) -> str:
    return f"case_{get_nnunet_case_number(case_id, HECKTOR)}"


def _resolve_reference_dataset650(reference_dataset650: Path) -> Tuple[Path, dict]:
    """Load split_manifest; fall back to Test2/Test3 path if needed."""
    candidates: List[Path] = [reference_dataset650]
    env_radheck = os.getenv("RADHECK_DATASET", "").strip()
    if env_radheck:
        candidates.append(Path(env_radheck))
    candidates.append(
        Path("/media/HDD_8TB/xisca/work/nnunet_radheck_test_1/Dataset650_TotalSegmentator")
    )

    tried = []
    for cand in candidates:
        if not cand:
            continue
        cand = cand.resolve() if cand.exists() else cand
        man = cand / "split_manifest.json"
        tried.append(str(cand))
        if man.is_file():
            with open(man) as f:
                manifest = json.load(f)
            if reference_dataset650.exists() and cand != reference_dataset650.resolve():
                print(
                    f"NOTE: {reference_dataset650}/split_manifest.json missing.\n"
                    f"      Using reference with manifest: {cand}"
                )
            return cand, manifest

    raise FileNotFoundError(
        "split_manifest.json not found.\n"
        f"  Tried:\n    - "
        + "\n    - ".join(tried)
        + "\n\n"
        "Fix:\n"
        "  export TEST5_REFERENCE_DATASET650="
        "/media/HDD_8TB/xisca/work/nnunet_radheck_test_1/Dataset650_TotalSegmentator\n"
    )


def _load_qc_discarded_case_ids(work_root: Path) -> Set[str]:
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
    stems: Set[str] = set()
    hecktor_id_to_stem = {v: k for k, v in hecktor_by_stem.items()}
    for case_id in discarded_case_ids:
        if case_id.startswith("RADCURE-"):
            stems.add(f"case_{case_id.replace('RADCURE-', '')}")
        elif case_id in hecktor_id_to_stem:
            stems.add(hecktor_id_to_stem[case_id])
        else:
            stems.add(_hecktor_stem(case_id))
    return stems


def _copy_stem_from_dataset650(
    dataset_folder: Path,
    split: str,
    stem: str,
    dst_images: str,
    dst_labels: str,
) -> None:
    img_name = f"{stem}_0000.nii.gz"
    lbl_name = f"{stem}.nii.gz"
    src_img = dataset_folder / f"images{split}" / img_name
    src_lbl = dataset_folder / f"labels{split}" / lbl_name
    if not src_img.is_file():
        matches = sorted((dataset_folder / f"images{split}").glob(f"{stem}*.nii.gz"))
        if not matches:
            raise FileNotFoundError(f"Missing {src_img}")
        src_img = matches[0]
        img_name = src_img.name
    if not src_lbl.is_file():
        raise FileNotFoundError(f"Missing {src_lbl}")
    os.makedirs(dst_images, exist_ok=True)
    os.makedirs(dst_labels, exist_ok=True)
    shutil.copy2(src_img, os.path.join(dst_images, img_name))
    shutil.copy2(src_lbl, os.path.join(dst_labels, lbl_name))


def _try_copy_case(
    *,
    split: str,
    stem: str,
    hecktor_by_stem: Dict[str, str],
    test5_radcure: Path,
    test5_hecktor: Path,
    test4_radcure: Optional[Path],
    test4_hecktor: Optional[Path],
    test4_dataset650: Optional[Path],
    reference_dataset650: Path,
    dst_img: str,
    dst_lbl: str,
    allow_reference_fallback: bool,
) -> str:
    is_hecktor = stem in hecktor_by_stem
    errors: List[str] = []

    if is_hecktor:
        cid = hecktor_by_stem[stem]
        try:
            copy_processed_hecktor_case(str(test5_hecktor), cid, dst_img, dst_lbl)
            return "test5_relabel"
        except (FileNotFoundError, OSError) as exc:
            errors.append(f"test5:{exc}")
        if test4_hecktor is not None:
            try:
                copy_processed_hecktor_case(str(test4_hecktor), cid, dst_img, dst_lbl)
                return "test4_relabel"
            except (FileNotFoundError, OSError) as exc:
                errors.append(f"test4_relabel:{exc}")
    else:
        case_id = _radcure_case_id_from_stem(stem)
        try:
            _copy_radcure_relabel(test5_radcure, case_id, dst_img, dst_lbl)
            return "test5_relabel"
        except (FileNotFoundError, OSError) as exc:
            errors.append(f"test5:{exc}")
        if test4_radcure is not None:
            try:
                _copy_radcure_relabel(test4_radcure, case_id, dst_img, dst_lbl)
                return "test4_relabel"
            except (FileNotFoundError, OSError) as exc:
                errors.append(f"test4_relabel:{exc}")

    if test4_dataset650 is not None and test4_dataset650.is_dir():
        try:
            _copy_stem_from_dataset650(test4_dataset650, split, stem, dst_img, dst_lbl)
            return "test4_dataset650"
        except (FileNotFoundError, OSError) as exc:
            errors.append(f"test4_ds:{exc}")

    if allow_reference_fallback:
        try:
            _copy_stem_from_dataset650(
                reference_dataset650, split, stem, dst_img, dst_lbl
            )
            return "reference_dataset650"
        except (FileNotFoundError, OSError) as exc:
            errors.append(f"reference:{exc}")

    raise FileNotFoundError(" | ".join(errors) if errors else f"no source for {stem}")


def build_dataset650(
    work_root: Path,
    reference_dataset650: Path,
    organ_dictionary_path: Path,
    dataset_id: str = "650",
    dry_run: bool = False,
    hecktor_output_root: Optional[Path] = None,
    skip_missing: bool = False,
    test4_work_root: Optional[Path] = None,
    allow_reference_fallback: bool = True,
) -> Path:
    reference_dataset650, manifest = _resolve_reference_dataset650(reference_dataset650)
    hecktor_by_stem = _build_hecktor_stem_map(manifest)
    discarded_ids = _load_qc_discarded_case_ids(work_root)
    discarded_stems = _discarded_stems(discarded_ids, hecktor_by_stem)

    radcure_retrain = work_root / "TotalSegmentatorRetrain"
    hecktor_root = Path(hecktor_output_root) if hecktor_output_root else (work_root / "hecktor")
    dataset_name = f"Dataset{dataset_id}_TotalSegmentator"
    dataset_folder = work_root / dataset_name

    if test4_work_root is None:
        env_t4 = os.getenv("TEST4_WORK_ROOT", "").strip()
        test4_work_root = Path(env_t4) if env_t4 else Path(
            "/media/HDD_8TB/xisca/work/retrain_test4"
        )
    test4_work_root = test4_work_root.resolve() if test4_work_root.exists() else None
    test4_hecktor = (test4_work_root / "hecktor") if test4_work_root else None
    test4_radcure = (
        (test4_work_root / "TotalSegmentatorRetrain") if test4_work_root else None
    )
    test4_dataset650 = None
    if test4_work_root is not None:
        cand = test4_work_root / "Dataset650_TotalSegmentator"
        if cand.is_dir() and (cand / "imagesTr").is_dir():
            test4_dataset650 = cand

    n_hecktor_ready = sum(
        1
        for p in hecktor_root.glob("*/output/image")
        if p.is_dir() and any(p.glob("*.nii.gz"))
    )
    n_radcure_ready = sum(
        1
        for p in radcure_retrain.glob("*/output/image")
        if p.is_dir() and any(p.glob("*.nii.gz"))
    )
    print(f"Test5 relabel outputs: RADCURE={n_radcure_ready}  HECKTOR={n_hecktor_ready}")
    print(f"  RADCURE root: {radcure_retrain}")
    print(f"  HECKTOR root: {hecktor_root}")
    if test4_work_root:
        print(f"Test4 fallback root: {test4_work_root}")
        print(f"  Test4 Dataset650: {test4_dataset650 or '(not found)'}")
    if n_hecktor_ready < 50:
        print(
            "\nNOTE: Only a small HECKTOR relabel set (Test4 matches this).\n"
            "  Missing HECKTOR stems will be filled from Test4 Dataset650\n"
            "  or the reference Dataset650. Only Test5-relabeled cases get\n"
            "  the improved background.\n"
        )

    if dataset_folder.is_dir() and not dry_run:
        print(f"Removing existing {dataset_folder}")
        shutil.rmtree(dataset_folder)

    counts = {"Tr": 0, "Va": 0, "Ts": 0}
    source_counts: Dict[str, int] = {}
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
                src_tag = _try_copy_case(
                    split=split,
                    stem=stem,
                    hecktor_by_stem=hecktor_by_stem,
                    test5_radcure=radcure_retrain,
                    test5_hecktor=hecktor_root,
                    test4_radcure=test4_radcure,
                    test4_hecktor=test4_hecktor,
                    test4_dataset650=test4_dataset650,
                    reference_dataset650=reference_dataset650,
                    dst_img=str(dst_img),
                    dst_lbl=str(dst_lbl),
                    allow_reference_fallback=allow_reference_fallback,
                )
                source_counts[src_tag] = source_counts.get(src_tag, 0) + 1
                counts[split] += 1
            except (FileNotFoundError, OSError) as exc:
                missing.append(f"{split}/{stem}: {exc}")

    print("\nCopy sources:", dict(sorted(source_counts.items())))

    if missing and not skip_missing:
        raise RuntimeError(
            f"{len(missing)} case(s) missing from Test5, Test4, and reference.\n"
            f"First 10:\n"
            + "\n".join(missing[:10])
            + "\n\nUse --skip-missing to drop them."
        )

    if missing and skip_missing:
        print(f"\nWARNING: --skip-missing: dropping {len(missing)} stems")

    if dry_run:
        print("\nDry run — no files written.")
        print(f"Would skip QC: {len(skipped_qc)}")
        return dataset_folder

    print(f"\nSkipped (QC): {len(skipped_qc)}")
    if missing and skip_missing:
        print(f"Skipped (missing): {len(missing)}")
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
        "split_source": "Test2/Test3 reference split_manifest + images{{Tr,Va,Ts}}",
        "label_source": (
            "Prefer Test5 improved relabel; fallback Test4 relabel / "
            "Test4 Dataset650 / reference Dataset650"
        ),
        "hecktor_output_root": str(hecktor_root),
        "test4_work_root": str(test4_work_root) if test4_work_root else None,
        "copy_source_counts": source_counts,
        "anatomy_qc_discarded_case_ids": sorted(discarded_ids),
        "anatomy_qc_skipped_stems": skipped_qc,
        "missing_skipped": missing if skip_missing else [],
        "counts_built": counts,
        "background_mode": "improved_where_test5_relabel_exists",
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
            "Test5 Phase 3: build Dataset650 (Test2/3 splits − QC), "
            "with Test4/reference fallbacks for missing HECKTOR relabels"
        )
    )
    parser.add_argument(
        "--work-root",
        default=os.getenv("TEST5_WORK_ROOT", "work/retrain_test5"),
    )
    parser.add_argument(
        "--reference-dataset650",
        default=os.getenv(
            "TEST5_REFERENCE_DATASET650",
            os.getenv("TEST4_REFERENCE_DATASET650", os.getenv("RADHECK_DATASET", "")),
        ),
    )
    parser.add_argument(
        "--organ-dictionary-path",
        default=os.getenv("ORGAN_DICTIONARY_PATH", ""),
    )
    parser.add_argument(
        "--test4-work-root",
        default=os.getenv("TEST4_WORK_ROOT", "/media/HDD_8TB/xisca/work/retrain_test4"),
        help="Fallback for missing Test5 relabels (env: TEST4_WORK_ROOT)",
    )
    parser.add_argument("--dataset-id", default="650")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--hecktor-output-root",
        default=os.getenv("TEST5_HECKTOR_OUTPUT_ROOT", ""),
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Drop stems missing from all fallbacks",
    )
    parser.add_argument(
        "--no-reference-fallback",
        action="store_true",
        help="Do not copy from Test2/Test3 Dataset650 labels",
    )
    args = parser.parse_args()

    work_root = Path(args.work_root).resolve()
    if not args.reference_dataset650:
        print("ERROR: set TEST5_REFERENCE_DATASET650 (Test2/Test3 Dataset650)")
        sys.exit(1)

    reference = Path(args.reference_dataset650).resolve()
    organ_dict = Path(
        args.organ_dictionary_path or (work_root / "radcure_dictionary_test5.json")
    ).resolve()
    if not reference.is_dir():
        raise FileNotFoundError(f"Reference dataset not found: {reference}")
    if not organ_dict.is_file():
        raise FileNotFoundError(f"Organ dictionary not found: {organ_dict}")

    hecktor_out = (
        Path(args.hecktor_output_root).resolve() if args.hecktor_output_root else None
    )
    t4 = Path(args.test4_work_root) if args.test4_work_root else None

    print("=" * 70)
    print("Test5 Phase 3 — build Dataset650")
    print(f"Work root:     {work_root}")
    print(f"Reference:     {reference}")
    print(f"Organ dict:    {organ_dict}")
    print(f"Test4 root:    {t4}")
    print("=" * 70)

    build_dataset650(
        work_root=work_root,
        reference_dataset650=reference,
        organ_dictionary_path=organ_dict,
        dataset_id=str(args.dataset_id),
        dry_run=args.dry_run,
        hecktor_output_root=hecktor_out,
        skip_missing=bool(args.skip_missing),
        test4_work_root=t4,
        allow_reference_fallback=not bool(args.no_reference_fallback),
    )


if __name__ == "__main__":
    main()
