#!/usr/bin/env python3
"""
Test6 — prepare / plan / fine-tune STU-Net on Dataset650 (Test5 labels + splits).

Uses STU-Net nnUNetv2 ``run_finetuning_stunet.py`` with ``STUNetTrainer_*_ft``
and TotalSegmentator-pretrained weights. Seg head is re-initialized for our
GTVp/GTVn (+ H&N) label set; encoder/decoder weights transfer.

Example:

  source ${TEST6_WORK_ROOT}/TEST6_ENV.sh   # from setup_stunet
  export CUDA_VISIBLE_DEVICES=1

  python -m pipelines.test6.train_finetune --step prepare
  python -m pipelines.test6.train_finetune --step plan
  python -m pipelines.test6.train_finetune --step train
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

_REPO_ROOT = Path(__file__).resolve().parents[2]

from pipelines.radheck.build_nnunet_dataset import write_dataset_json
from pipelines.test6.paths import (
    TRAINER_FT,
    WEIGHT_FILENAMES,
    check_numpy_blosc2,
    nnunet_cmd,
    nnunet_v2_root,
    python_env_with_stunet,
    stunet_clone,
    variant as default_variant,
    work_root,
)


def _ensure_env(work: Path) -> dict:
    """Configure nnUNet_* paths under TEST6 work root."""
    retrain = Path(
        os.getenv("NNUNET_RETRAIN_PATH", str(work / "nnunet_retrain"))
    ).expanduser()
    retrain.mkdir(parents=True, exist_ok=True)
    raw = retrain  # same layout as Test5 train_nnunet config
    preproc = retrain / "nnUNet_preprocessed"
    results = retrain / "nnUNet_results"
    preproc.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    os.environ["nnUNet_raw"] = str(raw)
    os.environ["nnUNet_preprocessed"] = str(preproc)
    os.environ["nnUNet_results"] = str(results)
    os.environ.setdefault("nnUNet_compile", "false")
    return {
        "retrain": retrain,
        "raw": raw,
        "preprocessed": preproc,
        "results": results,
    }


def _dataset_folder(work: Path) -> Path:
    """
    Prefer ``${TEST6_WORK_ROOT}/Dataset650_TotalSegmentator``.

    A leftover ``DATASET_FOLDER`` from Test1/2/… in the shell or ``.env`` must
    not win — that caused plan to look under ``nnunet_radheck_test_1``.
    """
    default = (work / "Dataset650_TotalSegmentator").expanduser().resolve()
    if default.is_dir():
        stale = os.getenv("DATASET_FOLDER", "").strip()
        if stale and Path(stale).expanduser().resolve() != default:
            print(
                f"NOTE: ignoring stale DATASET_FOLDER={stale}\n"
                f"      using Test6 dataset {default}"
            )
        os.environ["DATASET_FOLDER"] = str(default)
        return default

    env = os.getenv("DATASET_FOLDER", "").strip()
    if env:
        folder = Path(env).expanduser().resolve()
        if folder.is_dir():
            print(
                f"WARNING: {default} missing; falling back to DATASET_FOLDER={folder}"
            )
            os.environ["DATASET_FOLDER"] = str(folder)
            return folder

    raise FileNotFoundError(
        f"Test6 Dataset650 missing: {default}\n"
        "Run: python -m pipelines.test6.link_test5_dataset\n"
        f"(stale DATASET_FOLDER was {os.getenv('DATASET_FOLDER', '')!r})"
    )


def _link_raw(work: Path, dataset: Path, raw: Path) -> Path:
    target = raw / dataset.name
    if target.is_symlink() or target.exists():
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()
    os.symlink(dataset.resolve(), target)
    print(f"Linked raw dataset: {target} → {dataset.resolve()}")
    # nnUNetv2 also looks under nnUNet_raw/DatasetXXX — our raw root IS retrain
    return target


def _pretrained_weights(work: Path, variant: str) -> Path:
    env = os.getenv("TEST6_PRETRAINED_WEIGHTS", "").strip()
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return p
    p = work / "weights" / WEIGHT_FILENAMES[variant]
    if not p.is_file():
        raise FileNotFoundError(
            f"Missing pretrained weights: {p}\n"
            "Run: python -m pipelines.test6.setup_stunet"
        )
    return p


def _finetune_script(stunet: Path) -> Path:
    script = (
        stunet
        / "nnUNet-2.2"
        / "nnunetv2"
        / "run"
        / "run_finetuning_stunet.py"
    )
    if not script.is_file():
        raise FileNotFoundError(
            f"Missing {script}. Re-run setup_stunet / re-clone STU-Net."
        )
    return script


def _copy_splits_final(dataset: Path, preprocessed: Path, dataset_name: str) -> None:
    """Reuse Test5 fold assignment when available (same Tr pool)."""
    src_candidates = [
        Path(os.getenv("TEST5_WORK_ROOT", ""))
        / "nnunet_retrain"
        / "nnUNet_preprocessed"
        / dataset_name
        / "splits_final.json",
        dataset / "splits_final.json",
    ]
    # Also allow pointing at Test5 preprocessed explicitly
    env_ref = os.getenv("TEST6_SPLITS_REFERENCE", "").strip()
    if env_ref:
        src_candidates.insert(0, Path(env_ref) / dataset_name / "splits_final.json")
        src_candidates.insert(0, Path(env_ref) / "splits_final.json")

    dest_dir = preprocessed / dataset_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "splits_final.json"
    if dest.is_file():
        print(f"splits_final.json already present: {dest}")
        return
    for src in src_candidates:
        if src and src.is_file():
            shutil.copy2(src, dest)
            print(f"Copied splits_final.json from {src}")
            return
    print(
        "NOTE: no Test5 splits_final.json found — nnUNet will create one on train "
        "(same imagesTr pool as Test5)."
    )


def _organ_dictionary_path(work: Path) -> Path:
    env = os.getenv("ORGAN_DICTIONARY_PATH", "").strip()
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return p
    for cand in (
        work / "organ_dictionary_test5.json",
        Path(os.getenv("TEST5_WORK_ROOT", "")) / "organ_dictionary_test5.json",
        _REPO_ROOT / "image_processor" / "resources" / "organ_dictionary_hn_canonical.json",
    ):
        if cand and cand.is_file():
            return cand
    raise FileNotFoundError(
        "No organ dictionary found. Set ORGAN_DICTIONARY_PATH or run "
        "python -m pipelines.test6.link_test5_dataset"
    )


def _refresh_dataset_json(work: Path, dataset: Path) -> None:
    """
    Ensure dataset.json labels include GTVp/GTVn (91/92).

    Test5 NIfTIs are fine; an outdated dataset.json listing only 0–90 makes
    nnUNet --verify_dataset_integrity fail on label 92 (GTVn).
    """
    organ = _organ_dictionary_path(work)
    with open(organ) as f:
        labels = json.load(f)
    for need in ("GTVp", "GTVn"):
        if need not in labels:
            raise RuntimeError(f"{organ} missing {need} — refuse to plan")
    n_tr = len(list((dataset / "imagesTr").glob("*_0000.nii.gz")))
    write_dataset_json(
        str(dataset),
        dataset.name,
        str(organ),
        num_training=n_tr,
    )
    print(
        f"Refreshed dataset.json labels from {organ} "
        f"(GTVp={labels['GTVp']}, GTVn={labels['GTVn']}, numTraining={n_tr})"
    )


def step_prepare(work: Path) -> None:
    paths = _ensure_env(work)
    dataset = _dataset_folder(work)
    _link_raw(work, dataset, paths["raw"])
    _refresh_dataset_json(work, dataset)
    n_tr = len(list((dataset / "imagesTr").glob("*_0000.nii.gz")))
    n_ts = len(list((dataset / "imagesTs").glob("*_0000.nii.gz")))
    print(f"Prepare OK — Tr={n_tr} Ts={n_ts} (Test5 unified Dataset650)")
    print(f"  nnUNet_raw={os.environ['nnUNet_raw']}")


def step_plan(work: Path, dataset_id: str = "650") -> None:
    paths = _ensure_env(work)
    dataset = _dataset_folder(work)
    _link_raw(work, dataset, paths["raw"])
    _refresh_dataset_json(work, dataset)
    check_numpy_blosc2()
    print(f"Using Python: {sys.executable}")
    cmd, env = nnunet_cmd(
        "nnUNetv2_plan_and_preprocess",
        "-d",
        str(dataset_id),
        "-c",
        "3d_fullres",
        "--verify_dataset_integrity",
        work=work,
    )
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, env=env)
    _copy_splits_final(dataset, paths["preprocessed"], dataset.name)
    print("Plan/preprocess complete.")


def step_train(
    work: Path,
    *,
    variant: str,
    fold: int,
    dataset_id: str = "650",
) -> None:
    paths = _ensure_env(work)
    stunet = stunet_clone(work)
    weights = _pretrained_weights(work, variant)
    trainer = TRAINER_FT[variant]
    script = _finetune_script(stunet)

    nnunet_v2 = nnunet_v2_root(work)
    if str(nnunet_v2) not in sys.path:
        sys.path.insert(0, str(nnunet_v2))

    dataset = _dataset_folder(work)
    _copy_splits_final(dataset, paths["preprocessed"], dataset.name)

    # run_finetuning_stunet.py uses same CLI as nnUNetv2_train + pretrained patch
    cmd = [
        sys.executable,
        str(script),
        str(dataset_id),
        "3d_fullres",
        str(fold),
        "-tr",
        trainer,
        "-pretrained_weights",
        str(weights),
    ]
    print("=" * 70)
    print("Test6 — STU-Net fine-tune")
    print(f"  python:   {sys.executable}")
    print(f"  variant:  {variant} → {trainer}")
    print(f"  weights:  {weights}")
    print(f"  fold:     {fold}")
    print(f"  dataset:  {dataset_id}")
    print(f"  results:  {paths['results']}")
    print("=" * 70)
    print("+", " ".join(cmd))
    env = python_env_with_stunet(work)
    subprocess.check_call(cmd, cwd=str(script.parent), env=env)
    print("\nTraining finished.")
    print("Next: python -m pipelines.test6.evaluate")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test6: STU-Net fine-tune")
    parser.add_argument("--work-root", default=str(work_root()))
    parser.add_argument(
        "--step",
        choices=("prepare", "plan", "train", "all"),
        default="all",
    )
    parser.add_argument(
        "--variant",
        default=os.getenv("TEST6_STU_VARIANT", default_variant()),
        choices=sorted(TRAINER_FT),
    )
    parser.add_argument("--fold", type=int, default=int(os.getenv("NNUNET_FOLD", "0")))
    parser.add_argument("--dataset-id", default=os.getenv("DATASET_ID", "650"))
    args = parser.parse_args()

    work = Path(args.work_root).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)

    steps = (
        ["prepare", "plan", "train"]
        if args.step == "all"
        else [args.step]
    )
    for step in steps:
        if step == "prepare":
            step_prepare(work)
        elif step == "plan":
            step_plan(work, dataset_id=str(args.dataset_id))
        elif step == "train":
            step_train(
                work,
                variant=str(args.variant),
                fold=int(args.fold),
                dataset_id=str(args.dataset_id),
            )


if __name__ == "__main__":
    main()
