#!/usr/bin/env python3
"""
Build combined RADCURE + HECKTOR nnUNet dataset for retraining.

All server-specific paths live in a JSON file that is **not** committed to GitHub
(see README): copy ``radheck_server_paths.example.json`` to ``radheck_server_paths.json``
next to this script, or set ``RADHECK_SERVER_CONFIG`` to that file's absolute path.

Steps (unchanged logically):
  1) Download HECKTOR training zip from S3 (URI from config).
  2) Unzip and run image_processor (HECKTOR).
  3) Exclude cases already in the HECKTOR test nnUNet folder (Dataset152, etc.).
  4) 80/20 train/val on remaining HECKTOR cases.
  5) Merge with RADCURE-366 into RADHECK_OUTPUT_WORK/DatasetXXX_TotalSegmentator.

Before any S3 download, the script checks that ``numpy``, ``nibabel``, ``boto3``, and
``image_processor`` import successfully (same venv as ``pip install -r requirements.txt``).
If the training zip already exists under ``radheck_download_dir``, the download step is skipped.

Run from repository root:
    python research_notebooks/retrain_radheck/build_radheck_nnunet_dataset.py

Optional environment overrides (same names as JSON keys, UPPER_SNAKE for env):
    RADHECK_SERVER_CONFIG, RADHECK_S3_URI, RADHECK_DOWNLOAD_DIR, RADHECK_UNZIPPED_DIR,
    RADHECK_RADCURE_DATASET, RADHECK_OUTPUT_WORK, RADHECK_HECKTOR_TEST_DATASET,
    RADHECK_DATASET_ID, RADHECK_HECKTOR_TRAIN_FRAC, RADHECK_SPLIT_SEED,
    ORGAN_DICTIONARY_PATH, MAIN_PATH

AWS S3 downloads use the same credentials as the rest of the project: variables in the
repository-root ``.env`` (see ``env.example``): ``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``,
and ``AWS_REGION``. The script reads ``<repo>/.env`` with a small built-in parser (no
``python-dotenv`` required) and also uses ``python-dotenv`` when installed. ``~/.aws/credentials``
is still used if static keys are not in ``.env``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# Repo root (…/radcure-medical-imaging)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_SERVER_CONFIG_PATH = _SCRIPT_DIR / "radheck_server_paths.json"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

def _read_repo_dotenv_into_environ(dotenv_path: Path) -> None:
    """
    Parse repo-root .env into os.environ (no python-dotenv required).

    Handles UTF-8 BOM, optional ``export `` prefix, and ``KEY=value`` / quoted values.
    Lines in the file override existing environment variables for those keys so a real
    .env wins over empty placeholders exported in the parent shell.
    """
    if not dotenv_path.is_file():
        return
    try:
        text = dotenv_path.read_text(encoding="utf-8-sig")
    except OSError:
        return
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ[key] = value


def _load_project_dotenv() -> None:
    """Load repo-root .env first (manual parse), then optional python-dotenv for cwd extras."""
    repo_env = _REPO_ROOT / ".env"
    _read_repo_dotenv_into_environ(repo_env)
    try:
        from dotenv import load_dotenv

        if repo_env.is_file():
            load_dotenv(repo_env, override=True)
        load_dotenv(override=False)
    except ImportError:
        pass
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if region and not os.environ.get("AWS_DEFAULT_REGION"):
        os.environ["AWS_DEFAULT_REGION"] = region


_load_project_dotenv()


def _require_aws_credentials_for_s3_download() -> None:
    """Fail fast if neither static env keys nor boto3 default chain has credentials."""
    access_key = (os.getenv("AWS_ACCESS_KEY_ID") or "").strip()
    secret_key = (os.getenv("AWS_SECRET_ACCESS_KEY") or "").strip()
    if access_key and secret_key:
        return
    try:
        import boto3

        if boto3.Session().get_credentials() is not None:
            return
    except Exception:
        pass
    repo_env = _REPO_ROOT / ".env"
    env_example = _REPO_ROOT / "env.example"
    hint = (
        f"Repo .env path: {repo_env} (exists={repo_env.is_file()}). "
        f"After load: AWS_ACCESS_KEY_ID non-empty={bool(access_key)}, "
        f"AWS_SECRET_ACCESS_KEY non-empty={bool(secret_key)}.\n"
        "If .env exists but keys are still empty, check for typos, quotes, or placeholders; "
        "ensure variable names are exactly AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.\n"
        "If the parent shell exported empty AWS_* variables, they are overridden by values from .env now.\n"
    )
    raise RuntimeError(
        "S3 download requires AWS credentials. Add to the repository root .env file (see env.example):\n"
        "  AWS_ACCESS_KEY_ID=...\n"
        "  AWS_SECRET_ACCESS_KEY=...\n"
        "  AWS_REGION=eu-west-1\n"
        f"{hint}"
        f"Copy from: {env_example}\n"
        "Alternatively configure ~/.aws/credentials or an IAM role."
    )


def _verify_radheck_dependencies() -> None:
    """
    Import packages required before unzip/processing (image_processor loads numpy on import).

    Run this before S3 download so missing venv packages fail fast without re-downloading.
    """
    required = [
        ("numpy", "numpy"),
        ("nibabel", "nibabel"),
        ("boto3", "boto3"),
    ]
    missing: List[str] = []
    for pip_name, import_name in required:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pip_name)
    if missing:
        req_file = _REPO_ROOT / "requirements.txt"
        raise RuntimeError(
            "Missing Python package(s): "
            + ", ".join(missing)
            + f". Install the project environment, e.g.:\n  python -m pip install -r {req_file}\n"
            "Use the same interpreter you use to run this script."
        )
    try:
        from image_processor import CaseProcessor, HECKTOR  # noqa: F401
    except ImportError as e:
        req_file = _REPO_ROOT / "requirements.txt"
        raise RuntimeError(
            f"image_processor dependencies failed to import: {e}\n"
            f"Install full stack: python -m pip install -r {req_file}"
        ) from e


# Short TMPDIR for multiprocessing (same idea as run_hecktor_test1_pipeline)
_tmp = os.environ.get("TMPDIR") or os.environ.get("TEMP") or ""
if not _tmp or len(os.path.abspath(_tmp)) > 60:
    os.environ["TMPDIR"] = "/tmp"
    os.environ["TEMP"] = "/tmp"


def _resolve_config_path(cli_path: str) -> Path:
    if cli_path.strip():
        return Path(cli_path).expanduser().resolve()
    env_path = os.getenv("RADHECK_SERVER_CONFIG", "").strip()
    if env_path:
        return Path(env_path).expanduser().resolve()
    return _DEFAULT_SERVER_CONFIG_PATH


def load_radheck_server_config(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        example = _SCRIPT_DIR / "radheck_server_paths.example.json"
        raise FileNotFoundError(
            f"Server config not found: {path}\n\n"
            "Create a JSON file on the server (not in Git) with your paths. For example:\n"
            f"  cp {example} {_DEFAULT_SERVER_CONFIG_PATH}\n"
            "then edit the values. Or set RADHECK_SERVER_CONFIG to your JSON path.\n"
            "See research_notebooks/retrain_radheck/README.md"
        )
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Server config must be a JSON object at the top level.")
    return data


def _pick_str(
    cfg: Dict[str, Any],
    json_key: str,
    env_key: str,
    cli_val: Optional[str] = None,
    *,
    required: bool = True,
) -> str:
    if cli_val is not None and str(cli_val).strip():
        return str(cli_val).strip()
    ev = os.getenv(env_key)
    if ev is not None and str(ev).strip():
        return str(ev).strip()
    raw = cfg.get(json_key)
    if raw is not None and str(raw).strip():
        return str(raw).strip()
    if not required:
        return ""
    raise ValueError(f"Missing required setting {json_key!r} in server config (or set {env_key}).")


def collect_nnunet_case_basenames_from_test_folder(nnunet_dataset_dir: str) -> Set[str]:
    """
    Case identifiers already present in a nnUNet test split (imagesTs + labelsTs stems).

    Stems match nnUNet convention: image `CASE_0000.nii.gz` -> CASE; label `CASE.nii.gz` -> CASE.
    """
    out: Set[str] = set()
    images_ts = os.path.join(nnunet_dataset_dir, "imagesTs")
    labels_ts = os.path.join(nnunet_dataset_dir, "labelsTs")
    if os.path.isdir(images_ts):
        for f in os.listdir(images_ts):
            if not f.endswith(".nii.gz"):
                continue
            if f.endswith("_0000.nii.gz"):
                out.add(f.replace("_0000.nii.gz", ""))
            else:
                out.add(f.replace(".nii.gz", ""))
    if os.path.isdir(labels_ts):
        for f in os.listdir(labels_ts):
            if not f.endswith(".nii.gz"):
                continue
            stem = f.replace(".nii.gz", "")
            if stem.endswith("_0000"):
                stem = stem[: -len("_0000")]
            out.add(stem)
    return out


def hecktor_processed_nnunet_base(cases_root: str, case_folder_name: str) -> str:
    """Stem used for nnUNet filenames after processing (from first image in output/image)."""
    out_i = os.path.join(cases_root, case_folder_name, "output", "image")
    if not os.path.isdir(out_i):
        return case_folder_name
    imgs = [f for f in os.listdir(out_i) if f.endswith(".nii.gz")]
    if not imgs:
        return case_folder_name
    name = imgs[0]
    if name.endswith("_0000.nii.gz"):
        return name.replace("_0000.nii.gz", "")
    return name.replace(".nii.gz", "")


def filter_hecktor_cases_not_in_test_set(
    cases_root: str,
    case_ids: Sequence[str],
    test_basenames: Set[str],
) -> Tuple[List[str], List[str]]:
    """
    Keep only cases whose folder name and processed nnUNet base are not in test_basenames.

    Returns (kept_ids, excluded_ids).
    """
    kept: List[str] = []
    excluded: List[str] = []
    test_lower = {x.lower() for x in test_basenames}
    for cid in case_ids:
        base = hecktor_processed_nnunet_base(cases_root, cid)
        if cid in test_basenames or base in test_basenames:
            excluded.append(cid)
            continue
        if cid.lower() in test_lower or base.lower() in test_lower:
            excluded.append(cid)
            continue
        kept.append(cid)
    return kept, excluded


def _list_nifti_pairs(images_dir: str, labels_dir: str) -> List[Tuple[str, str]]:
    """Return list of (image_filename, label_filename) for nnUNet-style naming."""
    if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
        return []
    pairs = []
    for img in sorted(os.listdir(images_dir)):
        if not img.endswith(".nii.gz"):
            continue
        if img.endswith("_0000.nii.gz"):
            base = img.replace("_0000.nii.gz", "")
            lbl = f"{base}.nii.gz"
        else:
            base = img.replace(".nii.gz", "")
            lbl = f"{base}.nii.gz"
        ip = os.path.join(images_dir, img)
        lp = os.path.join(labels_dir, lbl)
        if os.path.isfile(ip) and os.path.isfile(lp):
            pairs.append((img, lbl))
    return pairs


def count_radcure_cases(radcure_dataset: str) -> Dict[str, int]:
    counts = {}
    for split in ("Tr", "Va", "Ts"):
        img_dir = os.path.join(radcure_dataset, f"images{split}")
        lbl_dir = os.path.join(radcure_dataset, f"labels{split}")
        counts[split] = len(_list_nifti_pairs(img_dir, lbl_dir))
    counts["total"] = counts["Tr"] + counts["Va"] + counts["Ts"]
    return counts


def copy_radcure_split(
    radcure_dataset: str,
    split_suffix: str,
    dst_images: str,
    dst_labels: str,
    use_symlink: bool,
) -> int:
    """Copy or symlink all cases from images{split}/labels{split} into dst_*."""
    src_img = os.path.join(radcure_dataset, f"images{split_suffix}")
    src_lbl = os.path.join(radcure_dataset, f"labels{split_suffix}")
    pairs = _list_nifti_pairs(src_img, src_lbl)
    os.makedirs(dst_images, exist_ok=True)
    os.makedirs(dst_labels, exist_ok=True)
    for img_name, lbl_name in pairs:
        s_i = os.path.join(src_img, img_name)
        s_l = os.path.join(src_lbl, lbl_name)
        d_i = os.path.join(dst_images, img_name)
        d_l = os.path.join(dst_labels, lbl_name)
        if use_symlink:
            if os.path.lexists(d_i):
                os.remove(d_i)
            if os.path.lexists(d_l):
                os.remove(d_l)
            os.symlink(os.path.abspath(s_i), d_i)
            os.symlink(os.path.abspath(s_l), d_l)
        else:
            shutil.copy2(s_i, d_i)
            shutil.copy2(s_l, d_l)
    return len(pairs)


def list_processed_hecktor_case_ids(cases_root: str) -> List[str]:
    ids = []
    for name in sorted(os.listdir(cases_root)):
        if name.startswith("."):
            continue
        case_folder = os.path.join(cases_root, name)
        if not os.path.isdir(case_folder):
            continue
        out_i = os.path.join(case_folder, "output", "image")
        out_l = os.path.join(case_folder, "output", "labels")
        if not os.path.isdir(out_i) or not os.path.isdir(out_l):
            continue
        imgs = [f for f in os.listdir(out_i) if f.endswith(".nii.gz")]
        lbls = [f for f in os.listdir(out_l) if f.endswith(".nii.gz")]
        if imgs and lbls:
            ids.append(name)
    return ids


def copy_processed_hecktor_case(
    cases_root: str,
    case_id: str,
    dst_images: str,
    dst_labels: str,
) -> None:
    out_i = os.path.join(cases_root, case_id, "output", "image")
    out_l = os.path.join(cases_root, case_id, "output", "labels")
    imgs = [f for f in os.listdir(out_i) if f.endswith(".nii.gz")]
    lbls = [f for f in os.listdir(out_l) if f.endswith(".nii.gz")]
    if not imgs or not lbls:
        raise FileNotFoundError(f"No nifti in {out_i} / {out_l}")
    src_img = os.path.join(out_i, imgs[0])
    src_lbl = os.path.join(out_l, lbls[0])
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


def split_train_val(
    case_ids: Sequence[str],
    train_fraction: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    ids = list(case_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_train = int(round(len(ids) * train_fraction))
    n_train = max(1, min(n_train, len(ids) - 1)) if len(ids) >= 2 else len(ids)
    if len(ids) == 1:
        return ids, []
    return ids[:n_train], ids[n_train:]


def write_dataset_json(
    dataset_folder: str,
    dataset_name: str,
    organ_dictionary_path: str,
    num_training: int,
) -> None:
    if organ_dictionary_path and os.path.isfile(organ_dictionary_path):
        with open(organ_dictionary_path, "r") as f:
            labels = json.load(f)
    else:
        labels = {"background": 0, "GTVp": 1}
    payload = {
        "channel_names": {"0": "CT"},
        "labels": labels,
        "numTraining": num_training,
        "file_ending": ".nii.gz",
        "dataset_name": dataset_name,
        "description": "Combined RADCURE-366 splits + HECKTOR training (excl. test1 set), 80/20 train/val",
        "reference": "",
        "licence": "",
    }
    path = os.path.join(dataset_folder, "dataset.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build combined RADCURE + HECKTOR nnUNet dataset")
    parser.add_argument(
        "--config",
        default="",
        help="Path to radheck_server_paths.json (default: RADHECK_SERVER_CONFIG env or file next to this script).",
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-process", action="store_true")
    parser.add_argument(
        "--hecktor-cases-root",
        default="",
        help="If set, skip download/unzip and use this as HECKTOR cases root (must contain processed output/).",
    )
    parser.add_argument("--link-radcure", action="store_true", help="Symlink RADCURE files instead of copying")
    parser.add_argument("--train-frac", type=float, default=None, help="Override train fraction from server config")
    parser.add_argument("--seed", type=int, default=None, help="Override split seed from server config")
    parser.add_argument(
        "--no-exclude-hecktor-test",
        action="store_true",
        help="Do not remove cases that already appear in the HECKTOR test dataset folder (default: exclude them).",
    )
    parser.add_argument(
        "--hecktor-test-dataset",
        default="",
        help="Override path to nnUNet HECKTOR test folder (imagesTs/labelsTs).",
    )
    args = parser.parse_args()

    config_path = _resolve_config_path(args.config)
    cfg = load_radheck_server_config(config_path)
    print(f"Using server config: {config_path}")

    s3_uri = _pick_str(cfg, "radheck_s3_uri", "RADHECK_S3_URI")
    download_dir = _pick_str(cfg, "radheck_download_dir", "RADHECK_DOWNLOAD_DIR")
    unzipped_dir = _pick_str(cfg, "radheck_unzipped_dir", "RADHECK_UNZIPPED_DIR")
    radcure_dataset = _pick_str(cfg, "radheck_radcure_dataset", "RADHECK_RADCURE_DATASET")
    output_work = _pick_str(cfg, "radheck_output_work", "RADHECK_OUTPUT_WORK")
    hecktor_cli = args.hecktor_test_dataset.strip() or None
    hecktor_test_dataset = _pick_str(
        cfg,
        "radheck_hecktor_test_dataset",
        "RADHECK_HECKTOR_TEST_DATASET",
        cli_val=hecktor_cli,
    )

    if args.train_frac is not None:
        train_frac = float(args.train_frac)
    else:
        ev = os.getenv("RADHECK_HECKTOR_TRAIN_FRAC")
        if ev is not None and str(ev).strip():
            train_frac = float(ev)
        elif cfg.get("radheck_hecktor_train_frac") is not None:
            train_frac = float(cfg["radheck_hecktor_train_frac"])
        else:
            train_frac = 0.8

    if args.seed is not None:
        seed = int(args.seed)
    else:
        ev = os.getenv("RADHECK_SPLIT_SEED")
        if ev is not None and str(ev).strip():
            seed = int(ev)
        elif cfg.get("radheck_split_seed") is not None:
            seed = int(cfg["radheck_split_seed"])
        else:
            seed = 42

    organ_dict = os.getenv("ORGAN_DICTIONARY_PATH", "").strip()
    if not organ_dict:
        organ_dict = (cfg.get("organ_dictionary_path") or "").strip()
    if not organ_dict:
        main_path = (cfg.get("main_path") or "").strip() or os.getenv("MAIN_PATH", "").strip()
        if main_path:
            organ_dict = os.path.join(main_path, "radcure_dictionary.json")

    if not os.path.isdir(radcure_dataset):
        raise FileNotFoundError(f"RADCURE dataset not found: {radcure_dataset}")

    _verify_radheck_dependencies()
    print("Dependency check OK (numpy, nibabel, boto3, image_processor).")

    from run_hecktor_test1_pipeline import (  # noqa: E402
        detect_hecktor_cases_root,
        download_from_s3,
        parse_s3_uri,
        unzip_and_detect_cases_root,
    )

    cases_root = args.hecktor_cases_root.strip() if args.hecktor_cases_root else ""
    if cases_root:
        if not os.path.isdir(cases_root):
            raise FileNotFoundError(f"--hecktor-cases-root not found: {cases_root}")
        print(f"Using existing HECKTOR cases root: {cases_root}")
    else:
        os.makedirs(download_dir, exist_ok=True)
        zip_name = os.path.basename(parse_s3_uri(s3_uri)[1]) or "HECKTOR2025 Task 1 Training.zip"
        zip_path = os.path.join(download_dir, zip_name)
        if not args.skip_download:
            if os.path.isfile(zip_path):
                print(f"Zip already present, skipping download: {zip_path}")
            else:
                _require_aws_credentials_for_s3_download()
                download_from_s3(s3_uri, zip_path)
        else:
            if not os.path.isfile(zip_path):
                raise FileNotFoundError(f"--skip-download but zip missing: {zip_path}")
            print(f"Using existing zip: {zip_path}")
        if not args.skip_process:
            os.makedirs(unzipped_dir, exist_ok=True)
            cases_root = unzip_and_detect_cases_root(zip_path, unzipped_dir)
            print(f"Unzipped; cases root: {cases_root}")
        else:
            try:
                cases_root = detect_hecktor_cases_root(unzipped_dir)
            except FileNotFoundError:
                cases_root = unzipped_dir
                print(
                    f"Warning: could not find HECKTOR case folders ({{id}}__CT / {{id}}.nii.gz) under {unzipped_dir}; "
                    f"using as-is. Set --hecktor-cases-root if needed."
                )
            print(f"--skip-process: HECKTOR cases root: {cases_root}")

    if not args.skip_process:
        from image_processor import CaseProcessor, HECKTOR  # noqa: E402

        processor = CaseProcessor(
            main_path=output_work,
            aws_bucket_name="dummy",
            aws_folder="dummy/",
            convention=HECKTOR,
            cases_root=cases_root,
            organ_dictionary_path=organ_dict or "",
            slice_expansion=5,
        )
        processor.process_multiple_cases(case_ids=None)

    hecktor_ids_all = list_processed_hecktor_case_ids(cases_root)
    if not hecktor_ids_all:
        raise FileNotFoundError(
            f"No processed HECKTOR cases under {cases_root}. "
            "Expected each case folder to contain output/image and output/labels with .nii.gz files."
        )
    print(f"Found {len(hecktor_ids_all)} processed HECKTOR cases (full training zip).")

    excluded_ids: List[str] = []
    test_basenames: Set[str] = set()
    if args.no_exclude_hecktor_test:
        hecktor_ids = list(hecktor_ids_all)
        print("--no-exclude-hecktor-test: not filtering against held-out test folder.")
    elif not os.path.isdir(hecktor_test_dataset):
        print(
            f"Warning: HECKTOR test dataset folder not found ({hecktor_test_dataset}); "
            "cannot exclude overlapping cases. Set RADHECK_HECKTOR_TEST_DATASET or use --no-exclude-hecktor-test."
        )
        hecktor_ids = list(hecktor_ids_all)
    else:
        test_basenames = collect_nnunet_case_basenames_from_test_folder(hecktor_test_dataset)
        hecktor_ids, excluded_ids = filter_hecktor_cases_not_in_test_set(
            cases_root, hecktor_ids_all, test_basenames
        )
        print(
            f"Excluding {len(excluded_ids)} case(s) already in test set ({hecktor_test_dataset}); "
            f"{len(hecktor_ids)} HECKTOR case(s) for train/val split."
        )
        if excluded_ids:
            print(f"  Excluded folder names: {excluded_ids[:20]}{' ...' if len(excluded_ids) > 20 else ''}")
    if not hecktor_ids:
        raise RuntimeError(
            "After excluding cases present in the HECKTOR test nnUNet folder, no HECKTOR cases remain for "
            "train/val. Check paths, naming, or use --no-exclude-hecktor-test for debugging."
        )

    rad_counts = count_radcure_cases(radcure_dataset)
    print(
        f"RADCURE case counts — Tr: {rad_counts['Tr']}, Va: {rad_counts['Va']}, Ts: {rad_counts['Ts']}, total: {rad_counts['total']}"
    )

    n_hecktor = len(hecktor_ids)
    total_cases = rad_counts["total"] + n_hecktor
    dataset_num_ev = os.getenv("RADHECK_DATASET_ID", "").strip()
    cfg_id = cfg.get("radheck_dataset_id")
    if dataset_num_ev:
        dataset_num = dataset_num_ev
    elif cfg_id is not None and str(cfg_id).strip() != "":
        dataset_num = str(cfg_id).strip()
    else:
        dataset_num = str(total_cases)
    dataset_name = f"Dataset{dataset_num}_TotalSegmentator"
    dataset_folder = os.path.join(output_work, dataset_name)

    if os.path.isdir(dataset_folder):
        print(f"Removing existing output folder: {dataset_folder}")
        shutil.rmtree(dataset_folder)
    os.makedirs(dataset_folder, exist_ok=True)

    tr_img = os.path.join(dataset_folder, "imagesTr")
    tr_lbl = os.path.join(dataset_folder, "labelsTr")
    va_img = os.path.join(dataset_folder, "imagesVa")
    va_lbl = os.path.join(dataset_folder, "labelsVa")
    ts_img = os.path.join(dataset_folder, "imagesTs")
    ts_lbl = os.path.join(dataset_folder, "labelsTs")

    n_r_tr = copy_radcure_split(radcure_dataset, "Tr", tr_img, tr_lbl, args.link_radcure)
    n_r_va = copy_radcure_split(radcure_dataset, "Va", va_img, va_lbl, args.link_radcure)
    n_r_ts = copy_radcure_split(radcure_dataset, "Ts", ts_img, ts_lbl, args.link_radcure)
    print(f"Copied RADCURE — Tr: {n_r_tr}, Va: {n_r_va}, Ts: {n_r_ts} (symlink={args.link_radcure})")

    h_train, h_val = split_train_val(hecktor_ids, train_frac, seed)
    for cid in h_train:
        copy_processed_hecktor_case(cases_root, cid, tr_img, tr_lbl)
    for cid in h_val:
        copy_processed_hecktor_case(cases_root, cid, va_img, va_lbl)
    print(f"HECKTOR (after exclusion) — train: {len(h_train)}, val: {len(h_val)} (frac={train_frac}, seed={seed})")

    n_tr_files = len([f for f in os.listdir(tr_img) if f.endswith(".nii.gz")])
    write_dataset_json(
        dataset_folder,
        dataset_name,
        organ_dict or "",
        num_training=n_tr_files,
    )

    manifest = {
        "server_config_path": str(config_path),
        "radcure_dataset": radcure_dataset,
        "radcure_counts": rad_counts,
        "hecktor_s3": s3_uri,
        "hecktor_cases_root": cases_root,
        "hecktor_test_dataset_excluded_from_train_val": hecktor_test_dataset,
        "hecktor_test_case_basenames": sorted(test_basenames),
        "hecktor_processed_before_exclude": len(hecktor_ids_all),
        "hecktor_excluded_case_folders": excluded_ids,
        "hecktor_train_cases": h_train,
        "hecktor_val_cases": h_val,
        "hecktor_train_frac": train_frac,
        "hecktor_split_seed": seed,
        "dataset_folder": dataset_folder,
        "dataset_id": dataset_num,
        "note": "RADCURE test in imagesTs/labelsTs; HECKTOR held-out test is Dataset152 (not copied here).",
    }
    man_path = os.path.join(dataset_folder, "split_manifest.json")
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {man_path}")
    print("\nDone.")
    print(f"Combined dataset: {dataset_folder}")


if __name__ == "__main__":
    main()
