#!/usr/bin/env python3
"""
Test4 Phase 3 — build Dataset650 from relabeled Phase 2 outputs.

Reuses **exact Tr/Va/Ts membership** from Test3 via ``split_manifest.json`` on the
reference Dataset650. Copies new labels (separate GTVp/GTVn) from:

  {work-root}/TotalSegmentatorRetrain/RADCURE-XXXX/output/
  {work-root}/hecktor/{HECKTOR-ID}/output/

Example:

  export TEST4_WORK_ROOT=/media/.../work/retrain_test4
  export TEST4_REFERENCE_DATASET650=/media/.../nnunet_radheck_test_1/Dataset650_TotalSegmentator
  export ORGAN_DICTIONARY_PATH=/media/.../work/retrain_test4/radcure_dictionary_test4.json

  python -m pipelines.test4.build_dataset650
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


def _hecktor_stem(case_id: str) -> str:
    return f"case_{get_nnunet_case_number(case_id, HECKTOR)}"


def _radcure_case_id_from_stem(stem: str) -> str:
    num = stem.replace("case_", "")
    return f"RADCURE-{num}"


def _load_reference_manifest(reference_dataset650: Path) -> dict:
    path = reference_dataset650 / "split_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"split_manifest.json not found in {reference_dataset650}. "
            "Point --reference-dataset650 at Test2/Test3 Dataset650."
        )
    with open(path) as f:
        return json.load(f)


def _build_hecktor_stem_map(manifest: dict) -> Dict[str, str]:
    """Map nnUNet stem (case_014) → HECKTOR folder id (CHUM-001)."""
    mapping: Dict[str, str] = {}
    for key in ("hecktor_train_cases", "hecktor_val_cases"):
        for case_id in manifest.get(key) or []:
            mapping[_hecktor_stem(case_id)] = case_id
    return mapping


def _list_split_images(dataset_folder: Path, split: str) -> List[str]:
    img_dir = dataset_folder / f"images{split}"
    if not img_dir.is_dir():
        return []
    return sorted(f for f in os.listdir(img_dir) if f.endswith(".nii.gz"))


def _copy_radcure_relabel(
    radcure_retrain: Path,
    case_id: str,
    dst_images: str,
    dst_labels: str,
) -> None:
    out_i = radcure_retrain / case_id / "output" / "image"
    out_l = radcure_retrain / case_id / "output" / "labels"
    if not out_i.is_dir() or not out_l.is_dir():
        raise FileNotFoundError(f"Missing relabeled output for {case_id}: {out_i}")
    imgs = [f for f in os.listdir(out_i) if f.endswith(".nii.gz")]
    lbls = [f for f in os.listdir(out_l) if f.endswith(".nii.gz")]
    if not imgs or not lbls:
        raise FileNotFoundError(f"No nifti in {out_i} / {out_l}")
    src_img = out_i / imgs[0]
    src_lbl = out_l / lbls[0]
    if imgs[0].endswith("_0000.nii.gz"):
        base = imgs[0].replace("_0000.nii.gz", "")
    else:
        base = imgs[0].replace(".nii.gz", "")
    dst_img = os.path.join(dst_images, imgs[0])
    dst_lbl = os.path.join(dst_labels, f"{base}.nii.gz")
    os.makedirs(dst_images, exist_ok=True)
    os.makedirs(dst_labels, exist_ok=True)
    shutil.copy2(src_img, dst_img)
    shutil.copy2(src_lbl, dst_lbl)


def build_dataset650(
    work_root: Path,
    reference_dataset650: Path,
    organ_dictionary_path: Path,
    dataset_id: str = "650",
    dry_run: bool = False,
) -> Path:
    manifest = _load_reference_manifest(reference_dataset650)
    hecktor_by_stem = _build_hecktor_stem_map(manifest)

    radcure_retrain = work_root / "TotalSegmentatorRetrain"
    hecktor_root = work_root / "hecktor"
    dataset_name = f"Dataset{dataset_id}_TotalSegmentator"
    dataset_folder = work_root / dataset_name

    if dataset_folder.is_dir() and not dry_run:
        print(f"Removing existing {dataset_folder}")
        shutil.rmtree(dataset_folder)

    counts = {"Tr": 0, "Va": 0, "Ts": 0}
    missing: List[str] = []

    for split in ("Tr", "Va", "Ts"):
        ref_images = _list_split_images(reference_dataset650, split)
        print(f"\n{split}: {len(ref_images)} cases (from reference Dataset650)")
        if dry_run:
            counts[split] = len(ref_images)
            continue

        dst_img = dataset_folder / f"images{split}"
        dst_lbl = dataset_folder / f"labels{split}"
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)

        for img_file in ref_images:
            stem = stem_from_image_filename(img_file)
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
                missing.append(f"{split}/{stem}: {exc}")

    if missing:
        raise RuntimeError(
            f"{len(missing)} case(s) missing Test4 relabeled output. First 10:\n"
            + "\n".join(missing[:10])
        )

    if dry_run:
        print("\nDry run — no files written.")
        return dataset_folder

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
        "test4_work_root": str(work_root),
        "reference_dataset650": str(reference_dataset650),
        "organ_dictionary_path": str(organ_dictionary_path),
        "dataset_folder": str(dataset_folder),
        "dataset_id": dataset_id,
        "split_source": "reference split_manifest.json (Test3)",
        "label_source": "Test4 Phase 2 relabel (separate GTVp/GTVn)",
        "counts_built": counts,
    }
    man_path = dataset_folder / "split_manifest.json"
    with open(man_path, "w") as f:
        json.dump(out_manifest, f, indent=2)
    print(f"\nWrote {man_path}")
    print(f"Done: {dataset_folder}")
    return dataset_folder


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test4 Phase 3: build Dataset650 with Test3 splits + Test4 labels"
    )
    parser.add_argument(
        "--work-root",
        default=os.getenv("TEST4_WORK_ROOT", "work/retrain_test4"),
        help="Phase 2 output root (env: TEST4_WORK_ROOT)",
    )
    parser.add_argument(
        "--reference-dataset650",
        default=os.getenv("TEST4_REFERENCE_DATASET650", os.getenv("RADHECK_DATASET", "")),
        help="Test3 Dataset650 with split_manifest.json (env: TEST4_REFERENCE_DATASET650)",
    )
    parser.add_argument(
        "--organ-dictionary-path",
        default=os.getenv(
            "ORGAN_DICTIONARY_PATH",
            "",
        ),
        help="Test4 dict with GTVp+GTVn (default: {work-root}/radcure_dictionary_test4.json)",
    )
    parser.add_argument("--dataset-id", default="650")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    work_root = Path(args.work_root).resolve()
    if not args.reference_dataset650:
        print("ERROR: set --reference-dataset650 or TEST4_REFERENCE_DATASET650 / RADHECK_DATASET")
        sys.exit(1)

    reference = Path(args.reference_dataset650).resolve()
    organ_dict = Path(
        args.organ_dictionary_path or (work_root / "radcure_dictionary_test4.json")
    ).resolve()

    if not reference.is_dir():
        raise FileNotFoundError(f"Reference dataset not found: {reference}")
    if not organ_dict.is_file():
        raise FileNotFoundError(f"Organ dictionary not found: {organ_dict}")

    print("=" * 70)
    print("Test4 Phase 3 — build Dataset650")
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
