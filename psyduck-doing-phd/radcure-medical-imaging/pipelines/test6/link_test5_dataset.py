#!/usr/bin/env python3
"""
Test6 — link Test5 Dataset650 (no reprocess) into the Test6 work root.

Creates a *clean* Dataset650 under TEST6 that only exposes training/eval
inputs (images*/labels* + maps). Does **not** symlink the whole Test5
folder (avoids pulling Test5 ``labelsTs_predicted`` / dice viz into Test6,
and avoids writing Test6 outputs back into Test5).

Organ dictionary is copied from Test5
``…/retrain_test5/organ_dictionary_test5.json`` (never the legacy
``RadcureComplete/radcure_dictionary.json``).

Example:

  export TEST6_WORK_ROOT=/media/HDD_8TB/xisca/work/retrain_test6_stunet
  export TEST6_DATASET650=/media/HDD_8TB/xisca/work/retrain_test5/Dataset650_TotalSegmentator

  python -m pipelines.test6.link_test5_dataset
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from pipelines.test6.paths import (
    DEFAULT_TEST5_ORGAN_DICTIONARY,
    resolve_test5_organ_dictionary,
    test5_dataset650,
    work_root,
)

# Only these Dataset650 entries are linked/copied into Test6.
_LINK_DIRS = (
    "imagesTr",
    "labelsTr",
    "imagesVa",
    "labelsVa",
    "imagesTs",
    "labelsTs",
)
_COPY_FILES = (
    "dataset.json",
    "case_map.json",
    "ts_case_map.json",
    "splits_final.json",
)


def _replace_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _symlink_dir(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    _replace_path(dst)
    os.symlink(src.resolve(), dst)
    print(f"  link  {dst.name}/ → {src.resolve()}")


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    _replace_path(dst)
    shutil.copy2(src, dst)
    print(f"  copy  {dst.name} ← {src}")


def _link_clean_dataset650(src650: Path, dst650: Path) -> dict:
    """
    Build Test6 Dataset650 as a real directory with selective links/copies.

    If ``dst650`` was previously a full symlink into Test5, replace it.
    """
    if dst650.is_symlink() or (dst650.exists() and not dst650.is_dir()):
        print(f"  removing whole-folder symlink/file: {dst650}")
        dst650.unlink()
    elif dst650.is_dir():
        # Drop any leftover eval dirs from an older full-folder link layout
        for junk in (
            "labelsTs_predicted",
            "labelsTs_dice_and_viz",
            "predictions",
            "pred_Ts",
        ):
            p = dst650 / junk
            if p.exists() or p.is_symlink():
                _replace_path(p)
                print(f"  removed leftover {junk}")

    dst650.mkdir(parents=True, exist_ok=True)

    linked = []
    for name in _LINK_DIRS:
        src = src650 / name
        if src.is_dir():
            _symlink_dir(src, dst650 / name)
            linked.append(name)
        elif name in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
            raise FileNotFoundError(f"Required folder missing in Test5: {src}")

    copied = []
    for name in _COPY_FILES:
        src = src650 / name
        if src.is_file():
            _copy_file(src, dst650 / name)
            copied.append(name)

    # Surface unexpected Test5 clutter (informational)
    skip = set(_LINK_DIRS) | set(_COPY_FILES) | {".DS_Store"}
    extras = sorted(
        p.name
        for p in src650.iterdir()
        if p.name not in skip and not p.name.startswith(".")
    )
    if extras:
        print(
            "  NOTE: not linking Test5 extras (eval/artifacts stay in Test5): "
            + ", ".join(extras)
        )

    return {"linked_dirs": linked, "copied_files": copied, "skipped_extras": extras}


def _ensure_organ_dictionary(
    work: Path, test5_work: Path, src650: Path, explicit: str
) -> Path:
    """Copy Test5 organ_dictionary_test5.json into Test6 work (never radcure_dictionary)."""
    del src650
    os.environ["TEST5_WORK_ROOT"] = str(test5_work)

    organ_dst = work / "organ_dictionary_test5.json"
    if organ_dst.is_symlink() or organ_dst.exists():
        try:
            organ_dst.unlink()
            print(f"  removed previous {organ_dst.name}")
        except OSError as e:
            raise RuntimeError(
                f"Cannot remove broken organ dict at {organ_dst}: {e}\n"
                f"  Run: rm -f {organ_dst}"
            ) from e

    # Prefer the Test5 work-root file (what Test5 actually trained with)
    preferred = test5_work / "organ_dictionary_test5.json"
    if explicit.strip():
        organ_src = Path(explicit).expanduser().resolve()
    elif preferred.is_file():
        organ_src = preferred.resolve()
    else:
        organ_src = resolve_test5_organ_dictionary(
            test5_work, explicit=str(Path(DEFAULT_TEST5_ORGAN_DICTIONARY))
        )

    if not organ_src.is_file():
        raise FileNotFoundError(
            f"Test5 organ dictionary not found.\n"
            f"Expected: {preferred}\n"
            f"(or {DEFAULT_TEST5_ORGAN_DICTIONARY})\n"
            "Do not use …/RadcureComplete/radcure_dictionary.json"
        )

    shutil.copy2(organ_src, organ_dst)
    print(f"  {organ_dst.name} ← copy {organ_src}")

    with open(organ_dst) as f:
        labels = json.load(f)
    for need in ("GTVp", "GTVn"):
        if need not in labels:
            raise RuntimeError(f"{organ_dst} missing {need}")
    print(f"  labels: GTVp={labels['GTVp']} GTVn={labels['GTVn']} (n={len(labels)})")
    os.environ["ORGAN_DICTIONARY_PATH"] = str(organ_dst)
    return organ_dst


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test6: clean-link Test5 Dataset650 into TEST6_WORK_ROOT"
    )
    parser.add_argument("--work-root", default=str(work_root()))
    parser.add_argument("--dataset650", default=str(test5_dataset650()))
    parser.add_argument(
        "--test5-work-root",
        default=os.getenv("TEST5_WORK_ROOT", ""),
        help="Test5 work root (for RADHECK_*/organ dict). Default: parent of Dataset650.",
    )
    parser.add_argument(
        "--organ-dictionary",
        default="",
        help="Optional explicit organ dict. Default: Test5 RADHECK_*/organ_dictionary_test5.json",
    )
    args = parser.parse_args()

    work = Path(args.work_root).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    src650 = Path(args.dataset650).expanduser().resolve()
    if not src650.is_dir():
        raise FileNotFoundError(
            f"Test5 Dataset650 not found: {src650}\n"
            "Set TEST6_DATASET650 or finish Test5 build_datasets first."
        )
    if not (src650 / "imagesTr").is_dir() or not (src650 / "imagesTs").is_dir():
        raise FileNotFoundError(
            f"{src650} must contain imagesTr/ and imagesTs/ (Test5 unified split)."
        )

    test5_work = (
        Path(args.test5_work_root).expanduser().resolve()
        if args.test5_work_root.strip()
        else src650.parent
    )

    n_tr = len(list((src650 / "imagesTr").glob("*_0000.nii.gz")))
    n_ts = len(list((src650 / "imagesTs").glob("*_0000.nii.gz")))
    print("=" * 70)
    print("Test6 — clean-link Test5 Dataset650 (no TotalSegmentator reprocess)")
    print(f"Source Dataset650: {src650}")
    print(f"  imagesTr: {n_tr}")
    print(f"  imagesTs: {n_ts}")
    print(f"Test5 work:        {test5_work}")
    print(f"Dest:              {work}")
    print("=" * 70)

    dst650 = work / "Dataset650_TotalSegmentator"
    layout = _link_clean_dataset650(src650, dst650)
    organ_dst = _ensure_organ_dictionary(
        work, test5_work, src650, args.organ_dictionary
    )

    status = {
        "dataset650": str(dst650),
        "dataset650_source": str(src650),
        "test5_work_root": str(test5_work),
        "organ_dictionary": str(organ_dst),
        "layout": "clean_selective_links",
        "n_imagesTr": n_tr,
        "n_imagesTs": n_ts,
        "reuse_test5": True,
        "reprocess_totalsegmentator": False,
        **layout,
        "case_map": str(dst650 / "case_map.json")
        if (dst650 / "case_map.json").is_file()
        else None,
        "ts_case_map": str(dst650 / "ts_case_map.json")
        if (dst650 / "ts_case_map.json").is_file()
        else None,
    }
    with open(work / "STATUS.json", "w") as f:
        json.dump(status, f, indent=2)
        f.write("\n")
    print(f"Wrote {work / 'STATUS.json'}")
    print("\nNext: python -m pipelines.test6.train_finetune --step prepare")


if __name__ == "__main__":
    main()
