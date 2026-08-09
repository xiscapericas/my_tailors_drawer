#!/usr/bin/env python3
"""
Test6 — link Test5 Dataset650 (no reprocess) into the Test6 work root.

Reuses the same Tr / Ts (unified RADCURE + HECKTOR test) and organ dictionary
from Test5. Does **not** copy NIfTIs by default (symlink).

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
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from pipelines.test6.paths import test5_dataset650, work_root

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CANONICAL_ORGAN = (
    _REPO_ROOT / "image_processor" / "resources" / "organ_dictionary_hn_canonical.json"
)


def _symlink_or_replace(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    os.symlink(src.resolve(), dst)
    print(f"  {dst} → {src.resolve()}")


def _ensure_organ_dictionary(work: Path, src650: Path, explicit: str) -> Path:
    """
    Place organ_dictionary_test5.json under Test6 work root.

    Prefer an existing Test5 / repo file; otherwise generate the same H&N
    canonical map Test5 uses (GTVp=91, GTVn=92).
    """
    organ_dst = work / "organ_dictionary_test5.json"
    candidates: list[Path] = []
    if explicit.strip():
        candidates.append(Path(explicit).expanduser())
    candidates.extend(
        [
            src650.parent / "organ_dictionary_test5.json",
            Path(os.getenv("TEST5_WORK_ROOT", src650.parent)).expanduser()
            / "organ_dictionary_test5.json",
            src650 / "organ_dictionary_test5.json",
            _CANONICAL_ORGAN,
        ]
    )

    organ_src = None
    for cand in candidates:
        try:
            c = cand.resolve()
        except OSError:
            continue
        if c.is_file():
            organ_src = c
            break

    if organ_dst.exists() or organ_dst.is_symlink():
        organ_dst.unlink()

    if organ_src is not None:
        if organ_src.resolve() == _CANONICAL_ORGAN.resolve():
            shutil.copy2(organ_src, organ_dst)
            print(f"  {organ_dst} ← copy {_CANONICAL_ORGAN.name}")
        else:
            os.symlink(organ_src.resolve(), organ_dst)
            print(f"  {organ_dst} → {organ_src.resolve()}")
    else:
        # Server checkout may lack resources/*.json — build like Test5 Phase 2
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        from image_processor.utils.organ_dictionary import OrganDictionary

        OrganDictionary.from_hn_canonical(
            str(organ_dst), separate_gtvp_gtvn=True, save=True
        )
        print(
            f"  {organ_dst} ← generated H&N canonical "
            f"(repo template missing: {_CANONICAL_ORGAN})"
        )

    with open(organ_dst) as f:
        labels = json.load(f)
    for need in ("GTVp", "GTVn"):
        if need not in labels:
            raise RuntimeError(f"{organ_dst} missing {need}")
    print(f"  labels: GTVp={labels['GTVp']} GTVn={labels['GTVn']} (n={len(labels)})")
    os.environ["ORGAN_DICTIONARY_PATH"] = str(organ_dst.resolve())
    return organ_dst


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test6: symlink Test5 Dataset650 into TEST6_WORK_ROOT"
    )
    parser.add_argument("--work-root", default=str(work_root()))
    parser.add_argument("--dataset650", default=str(test5_dataset650()))
    parser.add_argument(
        "--organ-dictionary",
        default=os.getenv("ORGAN_DICTIONARY_PATH", ""),
        help="Optional; default: Test5 work root / organ_dictionary_test5.json",
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

    n_tr = len(list((src650 / "imagesTr").glob("*_0000.nii.gz")))
    n_ts = len(list((src650 / "imagesTs").glob("*_0000.nii.gz")))
    print("=" * 70)
    print("Test6 — link Test5 Dataset650 (no TotalSegmentator reprocess)")
    print(f"Source:  {src650}")
    print(f"  imagesTr: {n_tr}")
    print(f"  imagesTs: {n_ts}")
    print(f"Dest:    {work}")
    print("=" * 70)

    dst650 = work / "Dataset650_TotalSegmentator"
    _symlink_or_replace(src650, dst650)

    organ_dst = _ensure_organ_dictionary(work, src650, args.organ_dictionary)

    # Status pointer
    status = {
        "dataset650": str(dst650.resolve()),
        "dataset650_source": str(src650),
        "organ_dictionary": str(organ_dst.resolve()),
        "n_imagesTr": n_tr,
        "n_imagesTs": n_ts,
        "reuse_test5": True,
        "reprocess_totalsegmentator": False,
        "case_map": str(src650 / "case_map.json")
        if (src650 / "case_map.json").is_file()
        else None,
        "ts_case_map": str(src650 / "ts_case_map.json")
        if (src650 / "ts_case_map.json").is_file()
        else None,
    }
    with open(work / "STATUS.json", "w") as f:
        json.dump(status, f, indent=2)
        f.write("\n")
    print(f"Wrote {work / 'STATUS.json'}")
    print("\nNext: python -m pipelines.test6.train_finetune --step prepare")


if __name__ == "__main__":
    main()
