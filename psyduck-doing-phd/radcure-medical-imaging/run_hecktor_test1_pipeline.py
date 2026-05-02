#!/usr/bin/env python3
"""
End-to-end pipeline: download HECKTOR test1 from S3, process with image_processor,
build nnUNet test dataset, run prediction and evaluation (Dice + viz).

All paths are configurable via environment variables or script defaults.

Usage:
    # Use defaults (see CONFIG below)
    python run_hecktor_test1_pipeline.py

    # Override via env
    export HECKTOR_S3_URI=s3://my-bucket/HECKTOR/test1.zip
    export HECKTOR_DOWNLOAD_DIR=/path/to/hecktor/test1
    export HECKTOR_UNZIPPED_DIR=/path/to/hecktor/test1/unzipped
    export NNUNET_WORK_DIR=/path/to/work/nnunet_hecktor_test1
    export DATASET_ID=152
    python run_hecktor_test1_pipeline.py

    # Use existing unzipped cases (skip download and unzip, run process + build dataset)
    export HECKTOR_CASES_ROOT=/path/to/hecktor/test1/unzipped/test1
    export NNUNET_WORK_DIR=/path/to/work/nnunet_hecktor_test1
    python run_hecktor_test1_pipeline.py

    # Only run prediction (dataset and model already exist)
    export DATASET_FOLDER=/path/to/work/nnunet_hecktor_test1/Dataset152_TotalSegmentator
    export NNUNET_WORK_DIR=/path/to/work/nnunet_hecktor_test1
    python run_hecktor_test1_pipeline.py --predict-only

    # Predict using a different dataset's trained model (e.g. Dataset366 model on Dataset152 images)
    export NNUNET_RETRAIN_PATH=/path/to/nnunet_retrain_radcure366
    export DATASET_ID=366
    export DATASET_FOLDER=/path/to/nnunet_hecktor_test1/Dataset152_TotalSegmentator
    export NNUNET_WORK_DIR=/path/to/nnunet_hecktor_test1
    python run_hecktor_test1_pipeline.py --predict-only
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from urllib.parse import urlparse

# Try to load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Set short TMPDIR early to avoid "AF_UNIX path too long" in multiprocessing (must be before any spawn)
_tmp = os.environ.get("TMPDIR") or os.environ.get("TEMP") or ""
if not _tmp or len(os.path.abspath(_tmp)) > 60:
    os.environ["TMPDIR"] = "/tmp"
    os.environ["TEMP"] = "/tmp"

# -----------------------------------------------------------------------------
# Configurable paths (env overrides)
# -----------------------------------------------------------------------------
DEFAULT_S3_URI = "s3://xisca-lab/HECKTOR/test1.zip"
DEFAULT_DOWNLOAD_DIR = "/media/HDD_8TB/xisca/dataset/hecktor/test1"
DEFAULT_UNZIPPED_DIR = "/media/HDD_8TB/xisca/dataset/hecktor/test1/unzipped"
DEFAULT_WORK_DIR = "/media/HDD_8TB/xisca/work/nnunet_hecktor_test1"
DEFAULT_DATASET_ID = "152"
DEFAULT_ORGAN_DICTIONARY_PATH = None  # will use env ORGAN_DICTIONARY_PATH or MAIN_PATH/radcure_dictionary.json

CONFIG = {
    "s3_uri": os.getenv("HECKTOR_S3_URI", DEFAULT_S3_URI),
    "download_dir": os.getenv("HECKTOR_DOWNLOAD_DIR", DEFAULT_DOWNLOAD_DIR),
    "unzipped_dir": os.getenv("HECKTOR_UNZIPPED_DIR", DEFAULT_UNZIPPED_DIR),
    "work_dir": os.getenv("NNUNET_WORK_DIR", DEFAULT_WORK_DIR),
    "dataset_id": os.getenv("HECKTOR_DATASET_ID", os.getenv("DATASET_ID", DEFAULT_DATASET_ID)),
    "organ_dictionary_path": os.getenv("ORGAN_DICTIONARY_PATH") or (os.path.join(os.getenv("MAIN_PATH", ""), "radcure_dictionary.json") if os.getenv("MAIN_PATH") else None),
    "nnunet_path": os.getenv("NNUNET_PATH", "/path/to/nnUNet"),
    "skip_download": os.getenv("HECKTOR_SKIP_DOWNLOAD", "").lower() in ("1", "true", "yes"),
    "skip_process": os.getenv("HECKTOR_SKIP_PROCESS", "").lower() in ("1", "true", "yes"),
    "skip_predict": os.getenv("HECKTOR_SKIP_PREDICT", "").lower() in ("1", "true", "yes"),
    "cases_root": os.getenv("HECKTOR_CASES_ROOT", "").strip() or None,
}


def parse_s3_uri(uri: str):
    """Parse s3://bucket/key into (bucket, key)."""
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Invalid S3 URI: {uri}")
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    return bucket, key


def download_from_s3(s3_uri: str, local_path: str, region: str = "eu-west-1") -> str:
    """Download a single file from S3 to local_path."""
    import boto3
    bucket, key = parse_s3_uri(s3_uri)
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    print(f"Downloading s3://{bucket}/{key} -> {local_path}")
    client = boto3.client("s3", region_name=region)
    client.download_file(bucket, key, local_path)
    print(f"Downloaded: {local_path}")
    return local_path


def detect_hecktor_cases_root(unpacked_parent_dir: str, max_depth: int = 4) -> str:
    """
    Find the directory whose immediate children are HECKTOR case folders.

    Handles archives that unpack to e.g. ``unzipped/Task 1/CHUM-001/`` (CT + mask at case level)
    or ``unzipped/CHUM-001/`` directly. Skips ``__MACOSX``. Uses BFS so a wrapper folder like
    ``Task 1`` is found even when ``__MACOSX`` is also at the top level.
    """
    from collections import deque

    from image_processor.conventions import get_hecktor_paths

    def dir_contains_hecktor_case_folders(directory: str) -> bool:
        try:
            names = os.listdir(directory)
        except OSError:
            return False
        for name in names:
            if name.startswith(".") or name == "__MACOSX":
                continue
            case_folder = os.path.join(directory, name)
            if not os.path.isdir(case_folder):
                continue
            paths = get_hecktor_paths(case_folder, name)
            if os.path.isfile(paths["path_ct"]) and os.path.isfile(paths["path_mask"]):
                return True
        return False

    start = os.path.abspath(unpacked_parent_dir)
    if not os.path.isdir(start):
        raise FileNotFoundError(f"Not a directory: {start}")
    q = deque([(start, 0)])
    seen = {os.path.realpath(start)}
    while q:
        d, depth = q.popleft()
        if dir_contains_hecktor_case_folders(d):
            return d
        if depth >= max_depth:
            continue
        try:
            names = os.listdir(d)
        except OSError:
            continue
        for name in sorted(names):
            if name.startswith(".") or name == "__MACOSX":
                continue
            sub = os.path.join(d, name)
            if not os.path.isdir(sub):
                continue
            r = os.path.realpath(sub)
            if r in seen:
                continue
            seen.add(r)
            q.append((sub, depth + 1))
    raise FileNotFoundError(
        f"No HECKTOR case folders under {unpacked_parent_dir} "
        f"(expected each case dir to contain {{id}}__CT.nii.gz and {{id}}.nii.gz; searched up to depth {max_depth})."
    )


def unzip_and_detect_cases_root(zip_path: str, unzipped_dir: str) -> str:
    """Unzip and return cases_root for HECKTOR (handles ``Task 1/`` and similar wrappers)."""
    from image_processor.io.file_handler import FileHandler

    FileHandler.unzip_file(zip_path, unzipped_dir)
    return detect_hecktor_cases_root(unzipped_dir)


def build_nnunet_dataset_from_processed(
    cases_root: str,
    dataset_folder: str,
    dataset_id: str,
    organ_dictionary_path: str,
) -> None:
    """
    Scan cases under cases_root (each case has output/image and output/labels),
    copy into dataset_folder/imagesTs and dataset_folder/labelsTs.
    Images: case_XXX_0000.nii.gz. Labels: copy to case_XXX.nii.gz (nnUNet expects no _0000 for labels).
    """
    images_ts = os.path.join(dataset_folder, "imagesTs")
    labels_ts = os.path.join(dataset_folder, "labelsTs")
    os.makedirs(images_ts, exist_ok=True)
    os.makedirs(labels_ts, exist_ok=True)
    case_ids = []
    for name in sorted(os.listdir(cases_root)):
        if name.startswith("."):
            continue
        case_folder = os.path.join(cases_root, name)
        if not os.path.isdir(case_folder):
            continue
        out_image_dir = os.path.join(case_folder, "output", "image")
        out_labels_dir = os.path.join(case_folder, "output", "labels")
        if not os.path.isdir(out_image_dir) or not os.path.isdir(out_labels_dir):
            continue
        # Expect one image file case_*_0000.nii.gz and one label file case_*_0000.nii.gz
        imgs = [f for f in os.listdir(out_image_dir) if f.endswith(".nii.gz")]
        lbls = [f for f in os.listdir(out_labels_dir) if f.endswith(".nii.gz")]
        if not imgs or not lbls:
            continue
        src_img = os.path.join(out_image_dir, imgs[0])
        src_lbl = os.path.join(out_labels_dir, lbls[0])
        # nnUNet: imagesTs case_XXX_0000.nii.gz, labelsTs case_XXX.nii.gz
        if imgs[0].endswith("_0000.nii.gz"):
            base = imgs[0].replace("_0000.nii.gz", "")
        else:
            base = imgs[0].replace(".nii.gz", "")
        dst_img = os.path.join(images_ts, imgs[0])
        dst_lbl = os.path.join(labels_ts, f"{base}.nii.gz")
        shutil.copy2(src_img, dst_img)
        # Label file from processor is case_XXX_0000.nii.gz -> save as case_XXX.nii.gz
        if lbls[0].endswith("_0000.nii.gz"):
            # copy without _0000
            shutil.copy2(src_lbl, dst_lbl)
        else:
            shutil.copy2(src_lbl, dst_lbl)
        case_ids.append(base)
    if not case_ids:
        # Diagnose: list what's under cases_root so user can see why nothing was found
        try:
            entries = [e for e in os.listdir(cases_root) if not e.startswith(".")]
            subdirs = [e for e in entries if os.path.isdir(os.path.join(cases_root, e))]
            has_output = []
            no_output = []
            for d in subdirs[:20]:
                out_img = os.path.join(cases_root, d, "output", "image")
                out_lbl = os.path.join(cases_root, d, "output", "labels")
                if os.path.isdir(out_img) and os.path.isdir(out_lbl):
                    has_output.append(d)
                else:
                    no_output.append(d)
            if len(subdirs) > 20:
                no_output.append(f"... and {len(subdirs) - 20} more")
            msg = (
                f"No processed cases found under {cases_root}\n"
                f"Expected each case folder to contain output/image/ and output/labels/ with .nii.gz files.\n"
                f"Subdirs found: {len(subdirs)}. With output: {len(has_output)}. Without output: {no_output[:5]}"
            )
        except Exception as e:
            msg = f"No processed cases found under {cases_root} (expected output/image and output/labels per case). List dir failed: {e}"
        raise FileNotFoundError(msg)
    print(f"Built nnUNet test dataset with {len(case_ids)} cases in {dataset_folder}")
    # Write dataset.json (minimal for inference)
    if organ_dictionary_path and os.path.isfile(organ_dictionary_path):
        with open(organ_dictionary_path, "r") as f:
            labels = json.load(f)
    else:
        labels = {"background": 0, "GTVp": 1}
    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": labels,
        "numTraining": 0,
        "file_ending": ".nii.gz",
        "dataset_name": f"Dataset{dataset_id}_TotalSegmentator",
        "description": "HECKTOR test1 (test set only)",
        "reference": "",
        "licence": "",
    }
    dataset_json_path = os.path.join(dataset_folder, "dataset.json")
    with open(dataset_json_path, "w") as f:
        json.dump(dataset_json, f, indent=2)
    print(f"Wrote {dataset_json_path}")


def main():
    parser = argparse.ArgumentParser(description="HECKTOR test1: download, process, nnUNet predict and evaluate")
    parser.add_argument("--skip-download", action="store_true", help="Skip S3 download (use existing zip)")
    parser.add_argument("--skip-process", action="store_true", help="Skip image_processor (use existing unzipped output)")
    parser.add_argument("--skip-predict", action="store_true", help="Skip nnUNet predict and eval")
    parser.add_argument("--skip-eval-viz", action="store_true", help="Run predict but skip evaluation/visualization")
    parser.add_argument("--predict-only", action="store_true", help="Only run nnUNet prediction (and eval/viz). Requires DATASET_FOLDER and NNUNET_WORK_DIR.")
    args = parser.parse_args()
    cfg = CONFIG.copy()
    if args.skip_download:
        cfg["skip_download"] = True
    if args.skip_process:
        cfg["skip_process"] = True
    if args.skip_predict:
        cfg["skip_predict"] = True

    s3_uri = cfg["s3_uri"]
    download_dir = cfg["download_dir"]
    unzipped_dir = cfg["unzipped_dir"]
    work_dir = cfg["work_dir"]
    dataset_id = str(cfg["dataset_id"]).strip()
    organ_dict_path = cfg["organ_dictionary_path"]
    nnunet_path = cfg["nnunet_path"]

    # ----- Predict-only mode: skip to Step 5 -----
    if getattr(args, "predict_only", False):
        dataset_folder = os.getenv("DATASET_FOLDER")
        if not dataset_folder or not os.path.isdir(dataset_folder):
            raise FileNotFoundError(
                "With --predict-only you must set DATASET_FOLDER to the dataset folder (e.g. .../Dataset152_TotalSegmentator) containing imagesTs and labelsTs."
            )
        work_dir = os.getenv("NNUNET_WORK_DIR") or os.path.dirname(dataset_folder)
        if not os.path.isdir(os.path.join(dataset_folder, "imagesTs")):
            raise FileNotFoundError(f"Dataset folder must contain imagesTs: {dataset_folder}")
        print("=" * 70)
        print("HECKTOR test1 pipeline (predict only)")
        print("=" * 70)
        print(f"Dataset folder:   {dataset_folder}")
        print(f"Work dir:         {work_dir}")
        print("=" * 70)
        # Jump to Step 5
        cfg["skip_predict"] = False
    else:
        print("=" * 70)
        print("HECKTOR test1 pipeline")
        print("=" * 70)
        print(f"S3 URI:           {s3_uri}")
        print(f"Download dir:     {download_dir}")
        print(f"Unzipped dir:     {unzipped_dir}")
        print(f"Work dir:         {work_dir}")
        print(f"Dataset ID:       {dataset_id}")
        print(f"Organ dictionary: {organ_dict_path}")
        print("=" * 70)

        # ----- Step 1 & 2: Download / Unzip, or use HECKTOR_CASES_ROOT -----
        if cfg["cases_root"]:
            # Use existing directory as cases root: skip download and unzip, run processing
            cases_root = cfg["cases_root"]
            if not os.path.isdir(cases_root):
                raise FileNotFoundError(f"HECKTOR_CASES_ROOT is set but directory not found: {cases_root}")
            print(f"Using HECKTOR_CASES_ROOT as cases root (skip download and unzip): {cases_root}")
        else:
            # ----- Step 1: Download from S3 -----
            zip_path = os.path.join(download_dir, os.path.basename(parse_s3_uri(s3_uri)[1]) or "test1.zip")
            if not cfg["skip_download"]:
                os.makedirs(download_dir, exist_ok=True)
                download_from_s3(s3_uri, zip_path)
            else:
                if not os.path.isfile(zip_path):
                    raise FileNotFoundError(f"Skip download requested but zip not found: {zip_path}")
                print(f"Using existing zip: {zip_path}")

            # ----- Step 2: Unzip -----
            if not cfg["skip_process"]:
                os.makedirs(unzipped_dir, exist_ok=True)
                cases_root = unzip_and_detect_cases_root(zip_path, unzipped_dir)
                print(f"Cases root: {cases_root}")
            else:
                if not os.path.isdir(unzipped_dir):
                    raise FileNotFoundError(f"Skip process requested but unzipped dir not found: {unzipped_dir}")
                try:
                    cases_root = detect_hecktor_cases_root(unzipped_dir)
                except FileNotFoundError:
                    # Already-processed tree: e.g. one folder whose children have output/
                    cases_root = unzipped_dir
                    entries = [e for e in os.listdir(cases_root) if not e.startswith(".")]
                    if len(entries) == 1:
                        subdir = os.path.join(cases_root, entries[0])
                        if os.path.isdir(subdir):
                            for c in os.listdir(subdir):
                                if os.path.isdir(os.path.join(subdir, c, "output")):
                                    cases_root = subdir
                                    break
                print(f"Using existing unzipped dir as cases root: {cases_root}")

        # ----- Step 3: Run image_processor (HECKTOR) -----
        if not cfg["skip_process"] or cfg["cases_root"]:
            from image_processor import CaseProcessor, HECKTOR
            # CaseProcessor expects main_path, aws_bucket, aws_folder (not used for HECKTOR but required)
            processor = CaseProcessor(
                main_path=work_dir,
                aws_bucket_name="dummy",
                aws_folder="dummy/",
                convention=HECKTOR,
                cases_root=cases_root,
                organ_dictionary_path=organ_dict_path,
                slice_expansion=5,
            )
            processor.process_multiple_cases(case_ids=None)
            # Quick check: if no case has output/, processing found no cases or failed
            if cfg["cases_root"]:
                any_output = any(
                    os.path.isdir(os.path.join(cases_root, d, "output", "image"))
                    for d in os.listdir(cases_root)
                    if not d.startswith(".") and os.path.isdir(os.path.join(cases_root, d))
                )
                if not any_output:
                    from image_processor.conventions import get_hecktor_paths
                    sample = [d for d in os.listdir(cases_root) if not d.startswith(".") and os.path.isdir(os.path.join(cases_root, d))][:3]
                    hints = []
                    for d in sample:
                        p = get_hecktor_paths(os.path.join(cases_root, d), d)
                        hints.append(f"  {d}: CT exists={os.path.isfile(p['path_ct'])}, mask exists={os.path.isfile(p['path_mask'])}")
                    raise FileNotFoundError(
                        "Processing ran but no case produced output. Each case folder under HECKTOR_CASES_ROOT must contain "
                        f"{'{case_id}__CT.nii.gz'} and {'{case_id}.nii.gz'}. Sample:\n" + "\n".join(hints)
                    )

        # ----- Step 4: Build nnUNet dataset folder (imagesTs, labelsTs) -----
        dataset_name = f"Dataset{dataset_id}_TotalSegmentator"
        dataset_folder = os.path.join(work_dir, dataset_name)
        build_nnunet_dataset_from_processed(
            cases_root=cases_root,
            dataset_folder=dataset_folder,
            dataset_id=dataset_id,
            organ_dictionary_path=organ_dict_path or "",
        )

    # When --predict-only, dataset_folder and work_dir were set above; dataset_id from folder name
    if getattr(args, "predict_only", False):
        import re
        match = re.search(r"Dataset(\d+)_TotalSegmentator", os.path.basename(dataset_folder))
        dataset_id = match.group(1) if match else dataset_id

    # ----- Step 5: Run nnUNet predict and evaluation -----
    if not cfg["skip_predict"]:
        # Use existing NNUNET_RETRAIN_PATH if set (e.g. path to trained model); otherwise use work_dir
        if "NNUNET_RETRAIN_PATH" not in os.environ:
            os.environ["NNUNET_RETRAIN_PATH"] = work_dir
        os.environ["DATASET_FOLDER"] = dataset_folder
        log_dir = os.path.join(work_dir, "logs")
        os.environ["LOG_DIR"] = log_dir
        os.makedirs(log_dir, exist_ok=True)
        if organ_dict_path:
            os.environ["ORGAN_DICTIONARY_PATH"] = organ_dict_path
        if "DATASET_ID" not in os.environ:
            os.environ["DATASET_ID"] = dataset_id
        if nnunet_path and nnunet_path != "/path/to/nnUNet":
            os.environ["NNUNET_PATH"] = nnunet_path
        # Reload config after setting env so TrainingConfig sees them
        from nnunet_training.config import TrainingConfig
        from nnunet_training.predict_and_evaluate import (
            add_nnunet_to_path,
            predict_on_test_set,
            evaluation_visualization,
        )
        config = TrainingConfig()
        add_nnunet_to_path(config.nnunet_path)
        config.setup_nnunet_environment()
        # Dataset ID from the *folder* (prediction data) vs *config* (model to use)
        import re
        match = re.search(r"Dataset(\d+)_TotalSegmentator", os.path.basename(dataset_folder))
        dataset_id_from_folder = match.group(1) if match else None
        using_external_model = (
            dataset_id_from_folder is not None
            and str(config.dataset_id) != str(dataset_id_from_folder)
        )
        if using_external_model:
            print(f"Using trained model from dataset ID {config.dataset_id} to predict on dataset folder (ID {dataset_id_from_folder}).")
        # Model folder: where the trained weights/plans live (Dataset{config.dataset_id}_...)
        model_dataset_name = f"Dataset{config.dataset_id}_TotalSegmentator"
        results_dataset_dir = os.path.join(config.main_retrain_path, "nnUNet_results", model_dataset_name)
        model_output_dir = os.path.join(
            results_dataset_dir,
            f"{config.trainer}__nnUNetPlans__{config.configuration}",
        )
        if using_external_model:
            # Don't copy dataset.json/plans from prediction dataset; model folder must already have them
            if not os.path.isfile(os.path.join(model_output_dir, "plans.json")):
                raise FileNotFoundError(
                    f"Model folder must contain plans.json when using a different dataset's model. "
                    f"Not found in: {model_output_dir}"
                )
            if not os.path.isfile(os.path.join(model_output_dir, "dataset.json")):
                raise FileNotFoundError(
                    f"Model folder must contain dataset.json when using a different dataset's model. "
                    f"Not found in: {model_output_dir}"
                )
            print(f"Model folder OK: {model_output_dir}")
        elif os.path.isfile(os.path.join(dataset_folder, "dataset.json")):
            dataset_json_src = os.path.join(dataset_folder, "dataset.json")
            copied = False
            if os.path.isdir(model_output_dir):
                dataset_json_dst = os.path.join(model_output_dir, "dataset.json")
                shutil.copy2(dataset_json_src, dataset_json_dst)
                print(f"Copied dataset.json to trained model folder: {dataset_json_dst}")
                copied = True
            # Also copy to any other trainer/plans folders under this dataset (in case name differs)
            if os.path.isdir(results_dataset_dir):
                for sub in os.listdir(results_dataset_dir):
                    if "__nnUNetPlans__" in sub and os.path.isdir(os.path.join(results_dataset_dir, sub)):
                        dst = os.path.join(results_dataset_dir, sub, "dataset.json")
                        if not os.path.isfile(dst):
                            shutil.copy2(dataset_json_src, dst)
                            print(f"Copied dataset.json to {dst}")
                            copied = True
            if not copied:
                # No trainer folder had dataset.json; create expected folder and copy so prediction can find it
                os.makedirs(model_output_dir, exist_ok=True)
                shutil.copy2(dataset_json_src, os.path.join(model_output_dir, "dataset.json"))
                print(f"Created {model_output_dir} and copied dataset.json (ensure trained weights are in this folder).")
        else:
            raise FileNotFoundError(
                f"dataset.json not found at {os.path.join(dataset_folder, 'dataset.json')}. "
                "Ensure the dataset folder contains dataset.json (e.g. from Step 4 or from a previous run)."
            )
        if not using_external_model:
            # nnUNet prediction also needs plans.json in the trained model folder; copy from preprocessed if missing
            preprocessed_dataset_dir = os.path.join(
                config.main_retrain_path, "nnUNet_preprocessed", config.dataset_name
            )
            plans_src = None
            for candidate in (
                f"nnUNetPlans_{config.configuration}.json",
                "nnUNetPlans.json",
            ):
                p = os.path.join(preprocessed_dataset_dir, candidate)
                if os.path.isfile(p):
                    plans_src = p
                    break
            if not plans_src and os.path.isdir(preprocessed_dataset_dir):
                for f in os.listdir(preprocessed_dataset_dir):
                    if f.startswith("nnUNetPlans") and f.endswith(".json"):
                        plans_src = os.path.join(preprocessed_dataset_dir, f)
                        break
            if plans_src:
                trainer_folders = []
                if os.path.isdir(model_output_dir):
                    trainer_folders.append(model_output_dir)
                if os.path.isdir(results_dataset_dir):
                    for sub in os.listdir(results_dataset_dir):
                        if "__nnUNetPlans__" in sub and os.path.isdir(os.path.join(results_dataset_dir, sub)):
                            tdir = os.path.join(results_dataset_dir, sub)
                            if tdir not in trainer_folders:
                                trainer_folders.append(tdir)
                for tdir in trainer_folders:
                    plans_dst = os.path.join(tdir, "plans.json")
                    if not os.path.isfile(plans_dst):
                        shutil.copy2(plans_src, plans_dst)
                        print(f"Copied plans to trained model folder: {plans_dst}")
            else:
                if not os.path.isfile(os.path.join(model_output_dir, "plans.json")):
                    raise FileNotFoundError(
                        f"plans.json not found in {model_output_dir} and no plans file found in "
                        f"{preprocessed_dataset_dir}. Run nnUNet planning/preprocessing first "
                        "(e.g. train_nnunet.py --step plan) or ensure the trained model folder contains plans.json."
                    )
        print("\nStep 5a: Running nnUNet prediction...")
        predict_on_test_set(config)
        if not getattr(args, "skip_eval_viz", False):
            print("\nStep 5b: Running evaluation and visualization (labelsTs_dice_and_viz)...")
            evaluation_visualization(config)
        print(f"\nPredictions: {os.path.join(dataset_folder, 'labelsTs_predicted')}")
        print(f"Dice/viz:    {os.path.join(dataset_folder, 'labelsTs_dice_and_viz')}")
    else:
        print("Skipping prediction (--skip-predict). Dataset ready at:", dataset_folder)

    print("\n" + "=" * 70)
    print("Pipeline finished.")
    print("=" * 70)


if __name__ == "__main__":
    main()
