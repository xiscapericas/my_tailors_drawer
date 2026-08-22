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

from image_processor.conventions import get_hecktor_paths
from image_processor.io.pet_align import build_pet_nnunet_channel
from pipelines.test5.build_dataset650 import _link_or_copy
from pipelines.test5.paths import default_hecktor_sources
from pipelines.test8_0.paths import (
    pin_test8_0_env,
    resolve_test5_organ_dictionary,
    test5_dataset650,
    test5_work_root,
    work_root,
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
    link_mode: str,
    dry_run: bool,
) -> dict:
    split = row.get("split") or "Tr"
    case_id = row["case_id"]
    src_img = src650 / f"images{split}" / f"{stem}_0000.nii.gz"
    src_lbl = src650 / f"labels{split}" / f"{stem}.nii.gz"
    if not src_img.is_file():
        raise FileNotFoundError(f"Test5 CT missing: {src_img}")
    if not src_lbl.is_file():
        raise FileNotFoundError(f"Test5 label missing: {src_lbl}")

    dst_img0 = dst650 / f"images{split}" / f"{stem}_0000.nii.gz"
    dst_img1 = dst650 / f"images{split}" / f"{stem}_0001.nii.gz"
    dst_lbl = dst650 / f"labels{split}" / f"{stem}.nii.gz"

    case_dir = resolve_hecktor_case_dir(case_id, sources)
    paths = get_hecktor_paths(str(case_dir), case_id)
    if not os.path.isfile(paths["path_pet"]):
        raise FileNotFoundError(
            f"PET missing for {case_id} (split={split} stem={stem}): {paths['path_pet']}"
        )

    expected = tuple(int(x) for x in nib.load(str(src_img)).shape)
    info = {
        "stem": stem,
        "case_id": case_id,
        "split": split,
        "source_case_dir": str(case_dir),
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
        "--link",
        choices=("hardlink", "symlink", "copy"),
        default=os.getenv("TEST8_0_DATASET_LINK_MODE", "hardlink"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-cases", type=int, default=0, help="0 = all")
    args = parser.parse_args()

    work = Path(args.work_root).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
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

    sources = default_hecktor_sources()
    items = sorted(hecktor.items(), key=lambda kv: (kv[1].get("split", ""), kv[0]))
    if args.max_cases and args.max_cases > 0:
        items = items[: args.max_cases]

    print("=" * 70)
    print("Test 8.0 — HECKTOR-only Test5 split + PET channel")
    print(f"Test5 Dataset650: {src650}")
    print(f"HECKTOR cases in map: {len(hecktor)}  this run: {len(items)}")
    print(f"HECKTOR sources: {sources}")
    print(f"Dest work: {work}")
    print("=" * 70)

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
                    link_mode=args.link,
                    dry_run=args.dry_run,
                )
            )
            print(f"  ok {row.get('split')} {stem} {row.get('case_id')}")
        except Exception as e:
            failed.append((stem, str(e)))
            print(f"  FAIL {stem}: {e}")

    if failed:
        raise RuntimeError(
            f"{len(failed)} HECKTOR case(s) failed (missing PET or shape mismatch). "
            "Fix sources or drop those ids from the Test5 map.\n"
            + "\n".join(f"  {s}: {m}" for s, m in failed[:20])
        )

    n_tr = sum(1 for _, r in items if r.get("split") == "Tr")
    n_ts = sum(1 for _, r in items if r.get("split") == "Ts")
    n_va = sum(1 for _, r in items if r.get("split") == "Va")

    if not args.dry_run:
        organ_src = resolve_test5_organ_dictionary(
            test5_work if test5_work.is_dir() else work,
            explicit=args.organ_dictionary,
        )
        organ_dst = work / "organ_dictionary_test5.json"
        if organ_dst.exists() or organ_dst.is_symlink():
            organ_dst.unlink()
        shutil.copy2(organ_src, organ_dst)

        filtered_map = {stem: hecktor[stem] for stem, _ in items}
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

        env_sh = work / "TEST8_0_ENV.sh"
        env_sh.write_text(
            "\n".join(
                [
                    f"export TEST8_0_WORK_ROOT={work}",
                    f"export DATASET_FOLDER={dst650}",
                    f"export NNUNET_RETRAIN_PATH={work / 'nnunet_retrain'}",
                    f"export ORGAN_DICTIONARY_PATH={organ_dst}",
                    "export DATASET_ID=650",
                    "export NNUNET_TRAINER=nnUNetTrainer_700epochs_NoMirroring",
                    "export NNUNET_CONFIGURATION=3d_fullres",
                    "export NNUNET_FOLD=0",
                    "export NNUNET_USE_LOCAL_PREPROCESS=1",
                    "export nnUNet_compile=false",
                    "unset NNUNET_PREPROCESSED_PATH",
                    "",
                ]
            )
        )
        print(f"Wrote {env_sh}")

    status = {
        "dataset650": str(dst650),
        "dataset650_source": str(src650),
        "hecktor_only": True,
        "n_hecktor_in_test5_map": len(hecktor),
        "n_built": len(built),
        "n_tr": n_tr,
        "n_va": n_va,
        "n_ts": n_ts,
        "dry_run": args.dry_run,
        "channels": {"0": "CT", "1": "PET"},
        "reprocess_totalsegmentator": False,
        "reuse_test5_split_membership": True,
    }
    with open(work / "STATUS.json", "w") as f:
        json.dump(status, f, indent=2)
        f.write("\n")
    print(f"Wrote {work / 'STATUS.json'}")
    print(f"Tr={n_tr} Va={n_va} Ts={n_ts}")


if __name__ == "__main__":
    main()
