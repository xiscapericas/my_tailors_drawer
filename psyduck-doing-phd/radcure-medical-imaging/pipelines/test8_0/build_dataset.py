#!/usr/bin/env python3
"""
Test 8.0 — HECKTOR-only Dataset650 with CT (_0000) + PET (_0001).

Keeps Test5 Tr/Va/Ts membership for cohort==hecktor. Hardlinks Test5 CT and
labels; writes PET aligned to the original CT grid and the Test5 slice crop.
Does not re-run TotalSegmentator. Does not write into the Test5 Dataset650.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from image_processor.conventions import HECKTOR, get_hecktor_paths, get_nnunet_case_number
from image_processor.io.pet_align import build_pet_nnunet_channel
from pipelines.test5.build_dataset650 import _hecktor_unique_stem, _link_or_copy
from pipelines.test5.paths import default_hecktor_sources
from pipelines.test8_0.paths import (
    pin_test8_0_env,
    resolve_test5_cases_root,
    resolve_test5_organ_dictionary,
    test5_dataset650,
    test5_work_root,
    work_root,
    write_test8_0_env_sh,
)


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def hecktor_rows_from_case_map(case_map: dict) -> Dict[str, dict]:
    """stem → row for Test5 HECKTOR cases only."""
    out = {}
    for stem, row in case_map.items():
        if not isinstance(row, dict):
            continue
        if row.get("cohort") == "hecktor":
            out[stem] = row
    return out


def index_test5_ct_channels(src650: Path) -> Dict[str, Tuple[str, Path]]:
    """stem → (split, path) for every ``*_0000.nii.gz`` in Test5 imagesTr/Va/Ts."""
    found: Dict[str, Tuple[str, Path]] = {}
    for split in ("Tr", "Va", "Ts"):
        folder = src650 / f"images{split}"
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*_0000.nii.gz")):
            stem = path.name[: -len("_0000.nii.gz")]
            found[stem] = (split, path)
    return found


def candidate_stems(stem: str, case_id: str) -> List[str]:
    """Filenames Test5 may have used for one HECKTOR case."""
    out: List[str] = []
    for s in (
        stem,
        _hecktor_unique_stem(case_id) if case_id else "",
        f"case_{get_nnunet_case_number(case_id, HECKTOR)}" if case_id else "",
    ):
        if s and s not in out:
            out.append(s)
    return out


def find_radheck_output_pair(cases_root: Path, case_id: str) -> Tuple[Path, Path]:
    """Test5 transformed CT + labels under ``RADHECK_*/cases/{id}/output/``."""
    case_dir = cases_root / case_id
    out_i = case_dir / "output" / "image"
    out_l = case_dir / "output" / "labels"
    num = get_nnunet_case_number(case_id, HECKTOR)
    fname = f"case_{num}_0000.nii.gz"
    img = out_i / fname
    lbl = out_l / fname
    if img.is_file() and lbl.is_file():
        return img, lbl
    imgs = sorted(out_i.glob("*.nii.gz")) if out_i.is_dir() else []
    lbls = sorted(out_l.glob("*.nii.gz")) if out_l.is_dir() else []
    if imgs and lbls:
        return imgs[0], lbls[0]
    raise FileNotFoundError(
        f"No Test5 transform output for {case_id} under {case_dir}/output/"
    )


def resolve_test5_ct(
    src650: Path,
    *,
    stem: str,
    split: str,
    case_id: str,
    index: Dict[str, Tuple[str, Path]],
    cases_root: Path | None = None,
) -> Tuple[str, str, Path, Path, str]:
    """
    Locate Test5 processed CT + label.

    Order: Dataset650 (any stem variant) → ``RADHECK_*/cases/{id}/output/``.
    Destination stem/split stay the case_map values (do not drop map cases).
    PET is never taken from RADHECK cases (no ``__PT`` there).

    Returns (dest_stem, dest_split, ct_path, label_path, ct_source).
    """
    dest_stem = stem
    dest_split = split
    for cand in candidate_stems(stem, case_id):
        if cand in index:
            _found_split, ct_path = index[cand]
            lbl = src650 / f"labels{_found_split}" / f"{cand}.nii.gz"
            if lbl.is_file():
                return dest_stem, dest_split, ct_path, lbl, "dataset650"
        ct = src650 / f"images{split}" / f"{cand}_0000.nii.gz"
        lbl = src650 / f"labels{split}" / f"{cand}.nii.gz"
        if ct.is_file() and lbl.is_file():
            return dest_stem, dest_split, ct, lbl, "dataset650"
    if cases_root is not None:
        img, lbl = find_radheck_output_pair(cases_root, case_id)
        return dest_stem, dest_split, img, lbl, "radheck_cases_output"
    raise FileNotFoundError(
        f"No Test5 CT/label for {case_id} in Dataset650 or RADHECK cases/output "
        f"(map stem={stem} split={split})"
    )


def resolve_hecktor_case_dir(case_id: str, sources: List[Path]) -> Path:
    for src in sources:
        cand = src / case_id
        if cand.is_dir():
            return cand
    raise FileNotFoundError(
        f"HECKTOR case folder not found for {case_id} under:\n"
        + "\n".join(f"  {s}" for s in sources)
    )


def _replace_dir(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _count_split_stems(dataset_folder: Path, split: str) -> int:
    folder = dataset_folder / f"images{split}"
    if not folder.is_dir():
        return 0
    return len(list(folder.glob("*_0000.nii.gz")))


def finalize_dataset_json(dataset_folder: Path, organ_dictionary_path: Path) -> int:
    """Write dataset.json from files already in Dataset650 (repair after a failed finish)."""
    n_tr = _count_split_stems(dataset_folder, "Tr")
    img_tr = dataset_folder / "imagesTr"
    n_pet = len(list(img_tr.glob("*_0001.nii.gz"))) if img_tr.is_dir() else 0
    if n_tr == 0:
        raise FileNotFoundError(
            f"No *_0000.nii.gz in {dataset_folder}/imagesTr — run the full build first."
        )
    if n_pet != n_tr:
        raise ValueError(
            f"imagesTr CT={n_tr} PET={n_pet} (must match). "
            "Some cases lack _0001. Run: python -m pipelines.test8_0.verify_channels "
            "then python -m pipelines.test8_0.build_dataset --only-missing-pet"
        )
    _write_dataset_json(dataset_folder, organ_dictionary_path, num_training=n_tr)
    print(
        f"Finalize: Tr={n_tr} Va={_count_split_stems(dataset_folder, 'Va')} "
        f"Ts={_count_split_stems(dataset_folder, 'Ts')} PET_Tr={n_pet}"
    )
    return n_tr


def _write_dataset_json(
    dataset_folder: Path,
    organ_dictionary_path: Path,
    num_training: int,
) -> None:
    with open(organ_dictionary_path, encoding="utf-8") as f:
        labels = json.load(f)
    payload = {
        "channel_names": {"0": "CT", "1": "PET"},
        "labels": labels,
        "numTraining": num_training,
        "file_ending": ".nii.gz",
        "dataset_name": "Dataset650_TotalSegmentator",
        "description": (
            "Test 8.0: Test5 HECKTOR-only split, CT + PET (SUV) channels, "
            "separate GTVp/GTVn, improved anatomical background"
        ),
        "reference": "Test5 Dataset650 membership; HECKTOR 2025 Task 1 PET",
        "licence": "Of course I checked! I'm not lazy",
        "converted_by": "Xisca Pe",
    }
    path = dataset_folder / "dataset.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"Wrote {path}")


def _place_split_dirs(dst650: Path) -> None:
    for name in (
        "imagesTr",
        "labelsTr",
        "imagesVa",
        "labelsVa",
        "imagesTs",
        "labelsTs",
    ):
        (dst650 / name).mkdir(parents=True, exist_ok=True)


def build_one_case(
    *,
    stem: str,
    row: dict,
    src650: Path,
    dst650: Path,
    sources: List[Path],
    cases_root: Path | None,
    link_mode: str,
    dry_run: bool,
    index: Dict[str, Tuple[str, Path]],
    skip_if_pet_exists: bool = False,
) -> dict:
    split = row.get("split") or "Tr"
    case_id = row["case_id"]
    dest_stem, dest_split, src_img, src_lbl, ct_source = resolve_test5_ct(
        src650,
        stem=stem,
        split=split,
        case_id=case_id,
        index=index,
        cases_root=cases_root,
    )

    dst_img0 = dst650 / f"images{dest_split}" / f"{dest_stem}_0000.nii.gz"
    dst_img1 = dst650 / f"images{dest_split}" / f"{dest_stem}_0001.nii.gz"
    dst_lbl = dst650 / f"labels{dest_split}" / f"{dest_stem}.nii.gz"
    if skip_if_pet_exists and dst_img0.is_file() and dst_img1.is_file() and dst_lbl.is_file():
        return {
            "stem": dest_stem,
            "map_stem": stem,
            "case_id": case_id,
            "split": dest_split,
            "ct_source": ct_source,
            "skipped": "pet_exists",
        }

    # PET always from original HECKTOR zip (RADHECK cases/ did not copy __PT)
    pet_dir = resolve_hecktor_case_dir(case_id, sources)
    paths = get_hecktor_paths(str(pet_dir), case_id)
    if not os.path.isfile(paths["path_pet"]):
        raise FileNotFoundError(
            f"Original PET missing for {case_id}: {paths['path_pet']}"
        )
    if not os.path.isfile(paths["path_ct"]):
        raise FileNotFoundError(
            f"Original CT missing for PET resample ({case_id}): {paths['path_ct']}"
        )

    expected = tuple(int(x) for x in nib.load(str(src_img)).shape)
    info = {
        "stem": dest_stem,
        "map_stem": stem,
        "case_id": case_id,
        "split": dest_split,
        "ct_source": ct_source,
        "pet_source": str(paths["path_pet"]),
        "expected_shape": expected,
    }
    if dry_run:
        info["dry_run"] = True
        return info

    _link_or_copy(src_img, dst_img0, link_mode)
    _link_or_copy(src_lbl, dst_lbl, link_mode)
    pet_info = build_pet_nnunet_channel(
        paths["path_pet"],
        paths["path_ct"],
        paths["path_mask"],
        dst_img1,
        expected_shape=expected,
        reference_crop_ct=src_img,
    )
    info.update(pet_info)
    return info


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test 8.0: HECKTOR-only Test5 split + PET channel"
    )
    parser.add_argument("--work-root", default=str(work_root()))
    parser.add_argument("--dataset650", default=str(test5_dataset650()))
    parser.add_argument(
        "--test5-work-root",
        default=os.getenv("TEST5_WORK_ROOT", ""),
        help="Test5 work root (organ dict).",
    )
    parser.add_argument("--organ-dictionary", default="")
    parser.add_argument(
        "--cases-root",
        default=os.getenv("TEST5_RADHECK_CASES", ""),
        help="Test5 RADHECK_*/cases (transform output). PET still comes from HECKTOR zip.",
    )
    parser.add_argument(
        "--link",
        choices=("hardlink", "symlink", "copy"),
        default=os.getenv("TEST8_0_DATASET_LINK_MODE", "hardlink"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-cases", type=int, default=0, help="0 = all")
    parser.add_argument(
        "--write-env-only",
        action="store_true",
        help="Only write TEST8_0_ENV.sh under the work root, then exit.",
    )
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Write dataset.json from existing Dataset650 images (no rebuild).",
    )
    parser.add_argument(
        "--only-missing-pet",
        action="store_true",
        help="Skip cases that already have images{split}/{stem}_0001.nii.gz.",
    )
    args = parser.parse_args()

    work = Path(args.work_root).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    write_test8_0_env_sh(work)
    if args.write_env_only:
        return
    if args.finalize_only:
        dst650 = work / "Dataset650_TotalSegmentator"
        organ_dst = work / "organ_dictionary_test5.json"
        if not organ_dst.is_file():
            test5_work = (
                Path(args.test5_work_root).expanduser().resolve()
                if args.test5_work_root.strip()
                else test5_work_root().resolve()
            )
            organ_src = resolve_test5_organ_dictionary(
                test5_work if test5_work.is_dir() else work,
                explicit=args.organ_dictionary,
            )
            shutil.copy2(organ_src, organ_dst)
        n_tr = finalize_dataset_json(dst650, organ_dst)
        write_test8_0_env_sh(work, organ=organ_dst)
        print(f"dataset.json ready (numTraining={n_tr})")
        return
    src650 = Path(args.dataset650).expanduser().resolve()
    if not src650.is_dir():
        raise FileNotFoundError(
            f"Test5 Dataset650 not found: {src650}\n"
            "Set TEST8_0_DATASET650 / finish Test5 build_datasets first."
        )
    map_path = src650 / "case_map.json"
    if not map_path.is_file():
        raise FileNotFoundError(f"Need Test5 case_map.json: {map_path}")

    test5_work = (
        Path(args.test5_work_root).expanduser().resolve()
        if args.test5_work_root.strip()
        else test5_work_root().resolve()
    )
    case_map = _load_json(map_path)
    hecktor = hecktor_rows_from_case_map(case_map)
    if not hecktor:
        raise RuntimeError(f"No cohort=hecktor rows in {map_path}")

    index = index_test5_ct_channels(src650)
    n_disk = len(list((src650 / "imagesTr").glob("*_0000.nii.gz"))) if (src650 / "imagesTr").is_dir() else 0
    n_disk_ts = len(list((src650 / "imagesTs").glob("*_0000.nii.gz"))) if (src650 / "imagesTs").is_dir() else 0

    sources = default_hecktor_sources()
    if args.cases_root.strip():
        cases_root = Path(args.cases_root).expanduser().resolve()
    else:
        cases_root = resolve_test5_cases_root(test5_work)
    items = sorted(hecktor.items(), key=lambda kv: (kv[1].get("split", ""), kv[0]))
    if args.max_cases and args.max_cases > 0:
        items = items[: args.max_cases]

    print("=" * 70)
    print("Test 8.0 — HECKTOR-only Test5 split + PET channel")
    print(f"Test5 Dataset650: {src650}")
    print(f"Test5 RADHECK cases: {cases_root}")
    print(f"HECKTOR rows in case_map.json: {len(hecktor)}")
    print(f"Test5 images on disk: Tr={n_disk} Ts={n_disk_ts} (all cohorts)")
    print(f"Indexed Dataset650 CT channels: {len(index)}")
    print(f"Original HECKTOR (PET) sources: {sources}")
    print(f"Dest work: {work}")
    print("=" * 70)
    print(
        "CT/labels: Dataset650 if present, else RADHECK_*/cases/*/output/. "
        "PET: original {id}__PT.nii.gz (not copied into RADHECK cases). "
        "Split membership from case_map.json."
    )
    print(
        "imagesVa is expected to be empty (same as Test5). nnUNet validation "
        "is fold 0 inside imagesTr via splits_final.json, not a Va folder."
    )

    dst650 = work / "Dataset650_TotalSegmentator"
    if not args.dry_run:
        if dst650.is_symlink():
            dst650.unlink()
        dst650.mkdir(parents=True, exist_ok=True)
        _place_split_dirs(dst650)

    built: List[dict] = []
    failed: List[Tuple[str, str]] = []
    for stem, row in items:
        try:
            built.append(
                build_one_case(
                    stem=stem,
                    row=row,
                    src650=src650,
                    dst650=dst650,
                    sources=sources,
                    cases_root=cases_root,
                    link_mode=args.link,
                    dry_run=args.dry_run,
                    index=index,
                    skip_if_pet_exists=args.only_missing_pet,
                )
            )
            print(f"  ok {row.get('split')} {stem} {row.get('case_id')}")
        except Exception as e:
            failed.append((stem, str(e)))
            print(f"  FAIL {stem}: {e}")

    print(f"\nBuilt={len(built)} failed={len(failed)}")

    if not built and not args.dry_run:
        raise RuntimeError(
            "No HECKTOR cases built. Need case_map HECKTOR rows plus "
            "Dataset650 and/or RADHECK_*/cases/*/output/."
        )

    n_tr = sum(1 for b in built if b.get("split") == "Tr")
    n_ts = sum(1 for b in built if b.get("split") == "Ts")
    n_va = sum(1 for b in built if b.get("split") == "Va")

    if not args.dry_run and built:
        organ_src = resolve_test5_organ_dictionary(
            test5_work if test5_work.is_dir() else work,
            explicit=args.organ_dictionary,
        )
        organ_dst = work / "organ_dictionary_test5.json"
        if organ_dst.exists() or organ_dst.is_symlink():
            organ_dst.unlink()
        shutil.copy2(organ_src, organ_dst)

        filtered_map = {}
        for b in built:
            key = b["stem"]
            src_row = hecktor.get(b.get("map_stem") or key) or hecktor.get(key) or {}
            filtered_map[key] = {
                **src_row,
                "stem": key,
                "case_id": b["case_id"],
                "cohort": "hecktor",
                "split": b["split"],
            }
        ts_map = {k: v for k, v in filtered_map.items() if v.get("split") == "Ts"}
        with open(dst650 / "case_map.json", "w") as f:
            json.dump(filtered_map, f, indent=2)
            f.write("\n")
        with open(dst650 / "ts_case_map.json", "w") as f:
            json.dump(ts_map, f, indent=2)
            f.write("\n")
        _write_dataset_json(dst650, organ_dst, num_training=n_tr)

        os.environ["ORGAN_DICTIONARY_PATH"] = str(organ_dst)
        pin_test8_0_env(work)
        write_test8_0_env_sh(work, organ=organ_dst)

    if failed:
        raise RuntimeError(
            f"{len(failed)} HECKTOR case(s) failed (missing PET or shape mismatch). "
            f"dataset.json was still written for the {len(built)} successful case(s).\n"
            + "\n".join(f"  {s}: {m}" for s, m in failed[:20])
        )

    status = {
        "dataset650": str(dst650),
        "dataset650_source": str(src650),
        "hecktor_only": True,
        "n_hecktor_in_test5_map": len(hecktor),
        "n_built": len(built),
        "radheck_cases_root": str(cases_root),
        "n_tr": n_tr,
        "n_va": n_va,
        "n_ts": n_ts,
        "dry_run": args.dry_run,
        "channels": {"0": "CT", "1": "PET"},
        "reprocess_totalsegmentator": False,
        "reuse_test5_split_membership": True,
        "imagesVa_note": (
            "Empty by design (Test5). nnUNet val = fold 0 of imagesTr "
            "(splits_final.json), not imagesVa."
        ),
    }
    with open(work / "STATUS.json", "w") as f:
        json.dump(status, f, indent=2)
        f.write("\n")
    print(f"Wrote {work / 'STATUS.json'}")
    print(f"Tr={n_tr} Va={n_va} Ts={n_ts}")


if __name__ == "__main__":
    main()
