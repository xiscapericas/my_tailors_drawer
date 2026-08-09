#!/usr/bin/env python3
"""
Test6 — link Test5 Dataset650 (no reprocess) into the Test6 work root.

Creates a *clean* Dataset650 under TEST6 that only exposes training/eval
inputs (images*/labels* + maps). Does **not** symlink the whole Test5
folder (avoids pulling Test5 ``labelsTs_predicted`` / dice viz into Test6,
and avoids writing Test6 outputs back into Test5).

Organ dictionary is taken from Test5 ``RADHECK_*`` (e.g. RADHECK_1047) or
``RADHECK_CURRENT``, not from a stale ``ORGAN_DICTIONARY_PATH``.

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

from pipelines.test5.paths import resolve_radheck
from pipelines.test6.paths import test5_dataset650, work_root

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CANONICAL_ORGAN = (
    _REPO_ROOT / "image_processor" / "resources" / "organ_dictionary_hn_canonical.json"
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


def _safe_resolve(path: Path) -> Path | None:
    try:
        return path.expanduser().resolve()
    except (OSError, RuntimeError):
        return None


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


def _organ_candidates(test5_work: Path, src650: Path, explicit: str) -> list[Path]:
    out: list[Path] = []
    if explicit.strip():
        out.append(Path(explicit).expanduser())

    # Prefer RADHECK_* (RADHECK_CURRENT → RADHECK_1047, …)
    try:
        radheck = resolve_radheck(test5_work)
        out.append(radheck / "organ_dictionary_test5.json")
        print(f"  Test5 RADHECK: {radheck}")
    except FileNotFoundError as e:
        print(f"  WARNING: {e}")

    out.extend(
        [
            test5_work / "organ_dictionary_test5.json",
            src650.parent / "organ_dictionary_test5.json",
            src650 / "organ_dictionary_test5.json",
            _CANONICAL_ORGAN,
        ]
    )
    # Also scan any RADHECK_* for a dict (newest first already via resolve; add rest)
    for rad in sorted(test5_work.glob("RADHECK_*"), reverse=True):
        if rad.is_dir() and rad.name != "RADHECK_CURRENT":
            out.append(rad / "organ_dictionary_test5.json")
    return out


def _ensure_organ_dictionary(
    work: Path, test5_work: Path, src650: Path, explicit: str
) -> Path:
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

    dst_abs = str(organ_dst)
    organ_src = None
    for cand in _organ_candidates(test5_work, src650, explicit):
        cand_exp = cand.expanduser()
        if str(cand_exp) == dst_abs:
            continue
        resolved = _safe_resolve(cand_exp)
        if resolved is None or str(resolved) == dst_abs:
            continue
        if resolved.is_file():
            organ_src = resolved
            break

    if organ_src is not None and organ_src.is_file():
        shutil.copy2(organ_src, organ_dst)
        print(f"  {organ_dst.name} ← copy {organ_src}")
    else:
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        from image_processor.utils.organ_dictionary import OrganDictionary

        OrganDictionary.from_hn_canonical(
            str(organ_dst), separate_gtvp_gtvn=True, save=True
        )
        print(f"  {organ_dst.name} ← generated H&N canonical (GTVp/GTVn)")

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
