#!/usr/bin/env python3
"""
Test7 — link Test5 Dataset650 (no reprocess) into the Test7 work root.

Same clean selective-link layout as Test6 (Tr/Va/Ts membership identical).
Does **not** symlink the whole Test5 folder (avoids mixing Test5 predictions).

Example:

  export TEST7_WORK_ROOT=/media/HDD_8TB/xisca/work/retrain_test7_prob
  export TEST7_DATASET650=/media/HDD_8TB/xisca/work/retrain_test5/Dataset650_TotalSegmentator

  python -m pipelines.test7.link_dataset
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

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipelines.test7.paths import (
    DEFAULT_TEST5_ORGAN_DICTIONARY,
    resolve_test5_organ_dictionary,
    test5_dataset650,
    test5_work_root,
    work_root,
)

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
    if dst650.is_symlink() or (dst650.exists() and not dst650.is_dir()):
        print(f"  removing whole-folder symlink/file: {dst650}")
        dst650.unlink()
    elif dst650.is_dir():
        for junk in (
            "labelsTs_predicted",
            "labelsTs_dice_and_viz",
            "labelsTs_probabilities",
            "labelsTs_probability_viz",
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
    work: Path, test5_work: Path, explicit: str
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
            f"(or {DEFAULT_TEST5_ORGAN_DICTIONARY})"
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
        description="Test7: clean-link Test5 Dataset650 into TEST7_WORK_ROOT"
    )
    parser.add_argument("--work-root", default=str(work_root()))
    parser.add_argument("--dataset650", default=str(test5_dataset650()))
    parser.add_argument(
        "--test5-work-root",
        default=os.getenv("TEST5_WORK_ROOT", ""),
        help="Test5 work root (organ dict). Default: parent of Dataset650.",
    )
    parser.add_argument("--organ-dictionary", default="", help="Optional explicit organ dict")
    args = parser.parse_args()

    work = Path(args.work_root).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    src650 = Path(args.dataset650).expanduser().resolve()
    if not src650.is_dir():
        raise FileNotFoundError(
            f"Test5 Dataset650 not found: {src650}\n"
            "Set TEST7_DATASET650 or finish Test5 build_datasets first."
        )
    if not (src650 / "imagesTr").is_dir() or not (src650 / "imagesTs").is_dir():
        raise FileNotFoundError(
            f"{src650} must contain imagesTr/ and imagesTs/ (Test5/Test6 unified split)."
        )

    test5_work = (
        Path(args.test5_work_root).expanduser().resolve()
        if args.test5_work_root.strip()
        else src650.parent
    )
    # Prefer explicit TEST5_WORK_ROOT when set via paths helper default
    if not args.test5_work_root.strip():
        tw = test5_work_root()
        if (tw / "organ_dictionary_test5.json").is_file() or tw.is_dir():
            test5_work = tw.resolve()

    n_tr = len(list((src650 / "imagesTr").glob("*_0000.nii.gz")))
    n_ts = len(list((src650 / "imagesTs").glob("*_0000.nii.gz")))
    print("=" * 70)
    print("Test7 — clean-link Test5 Dataset650 (no TotalSegmentator reprocess)")
    print(f"Source Dataset650: {src650}")
    print(f"  imagesTr: {n_tr}")
    print(f"  imagesTs: {n_ts}")
    print(f"Test5 work:        {test5_work}")
    print(f"Dest:              {work}")
    print("=" * 70)

    dst650 = work / "Dataset650_TotalSegmentator"
    layout = _link_clean_dataset650(src650, dst650)
    organ_dst = _ensure_organ_dictionary(work, test5_work, args.organ_dictionary)

    status = {
        "dataset650": str(dst650),
        "dataset650_source": str(src650),
        "test5_work_root": str(test5_work),
        "organ_dictionary": str(organ_dst),
        "layout": "clean_selective_links",
        "n_imagesTr": n_tr,
        "n_imagesTs": n_ts,
        "reuse_test5": True,
        "reuse_test6_splits": True,
        "reprocess_totalsegmentator": False,
        "retrain": False,
        **layout,
    }
    with open(work / "STATUS.json", "w") as f:
        json.dump(status, f, indent=2)
        f.write("\n")
    print(f"Wrote {work / 'STATUS.json'}")

    env_sh = work / "TEST7_ENV.sh"
    test5_retrain = os.getenv(
        "RETRAIN_RADHECK_TEST5",
        str(test5_work / "nnunet_retrain"),
    )
    env_sh.write_text(
        f"""# Source before Test7 predict / curves / viz
# Clears common leftovers so they cannot override Test7 paths.

unset NNUNET_PREPROCESSED_PATH NNUNET_EVAL_OUTPUT_DIR HECKTOR_EVAL_OUTPUT_DIR 2>/dev/null || true

export TEST7_WORK_ROOT={work}
export TEST5_WORK_ROOT={test5_work}
export TEST7_DATASET650={src650}
export RETRAIN_RADHECK_TEST5={test5_retrain}
export RETRAIN_RADHECK_TEST7={work}/nnunet_retrain
export RADHECK_DATASET_TEST7={dst650}
export ORGAN_DICTIONARY_PATH={organ_dst}
export DATASET_FOLDER={dst650}
export DATASET_ID=650
export NNUNET_TRAINER=nnUNetTrainer_700epochs_NoMirroring
export NNUNET_CONFIGURATION=3d_fullres
export NNUNET_DISABLE_TTA=true
export nnUNet_compile=false
export CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-1}}
# Prefer the known-good nnUNet source tree over a broken site-packages copy
export NNUNET_PATH=${{NNUNET_PATH:-/media/HDD_8TB/xisca/envs/nnUNet}}
export PYTHONPATH=${{NNUNET_PATH}}${{PYTHONPATH:+:$PYTHONPATH}}

echo "Test7 env pinned: WORK=$TEST7_WORK_ROOT MODEL_RETRAIN=$RETRAIN_RADHECK_TEST5 NNUNET_PATH=$NNUNET_PATH"
"""
    )
    print(f"Wrote {env_sh}")
    print("\nNext:")
    print(f"  source {env_sh}")
    print("  python -m pipelines.test7.predict_probabilities")


if __name__ == "__main__":
    main()
