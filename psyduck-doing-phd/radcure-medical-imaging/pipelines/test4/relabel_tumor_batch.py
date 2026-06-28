#!/usr/bin/env python3
"""
Test4 Phase 2 — batch relabel all cases using existing TotalSegmentator outputs.

Reuses ``total_segmentator_output/`` (organs unchanged) and rebuilds nnUNet
``output/image`` + ``output/labels`` with ``tumor_label_mode=separate`` (GTVp + GTVn).

Outputs (under --work-root, default work/retrain_test4):
  TotalSegmentatorRetrain/{RADCURE-XXXX}/output/   ← RADCURE nnUNet pairs
  hecktor/{HECKTOR-ID}/output/                     ← HECKTOR nnUNet pairs

Phase 3 (later): build Dataset650 from these folders + same splits as Test3.

Example (server):

  cd /path/to/radcure-medical-imaging
  source .venv/bin/activate
  set -a && source .env && set +a

  export CUDA_VISIBLE_DEVICES=0
  export TEST4_WORK_ROOT=/media/HDD_8TB/xisca/work/retrain_test4
  export TEST4_RADCURE_SOURCE_MAIN_PATH=/media/HDD_8TB/xisca/dataset/RadcureComplete
  export TEST4_HECKTOR_SOURCE_CASES_ROOT=/media/HDD_8TB/xisca/dataset/hecktor/.../unzipped/task1

  python -m pipelines.test4.relabel_tumor_batch

  # Preview counts only:
  python -m pipelines.test4.relabel_tumor_batch --dry-run

  # RADCURE only, first 5 cases:
  python -m pipelines.test4.relabel_tumor_batch --skip-hecktor --max-cases 5
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from image_processor import (
    CaseProcessor,
    HECKTOR,
    RADCURE,
    TUMOR_LABEL_MODE_SEPARATE,
)
from image_processor.conventions import get_hecktor_paths, get_nnunet_case_number


def _list_radcure_cases_with_ts(source_main_path: str) -> List[str]:
    retrain = Path(source_main_path) / "TotalSegmentatorRetrain"
    if not retrain.is_dir():
        raise FileNotFoundError(f"RADCURE source not found: {retrain}")
    cases = []
    for name in sorted(os.listdir(retrain)):
        if not name.startswith("RADCURE-"):
            continue
        case_dir = retrain / name
        if not case_dir.is_dir():
            continue
        if (case_dir / "total_segmentator_output").is_dir():
            cases.append(name)
    return cases


def _list_hecktor_cases_with_ts(source_cases_root: str) -> List[str]:
    root = Path(source_cases_root)
    if not root.is_dir():
        raise FileNotFoundError(f"HECKTOR source not found: {root}")
    cases = []
    for name in sorted(os.listdir(root)):
        if name.startswith("."):
            continue
        case_dir = root / name
        if not case_dir.is_dir():
            continue
        if not (case_dir / "total_segmentator_output").is_dir():
            continue
        paths = get_hecktor_paths(str(case_dir), name)
        if os.path.isfile(paths["path_ct"]) and os.path.isfile(paths["path_mask"]):
            cases.append(name)
    return cases


def _output_pair_exists(case_folder: str, case_id: str, convention: str) -> bool:
    num = get_nnunet_case_number(case_id, convention)
    fname = f"case_{num}_0000.nii.gz"
    out = Path(case_folder) / "output"
    return (
        (out / "image" / fname).is_file()
        and (out / "labels" / fname).is_file()
    )


def _run_batch(
    label: str,
    case_ids: List[str],
    source_root: Path,
    dest_root: Path,
    processor: CaseProcessor,
    convention: str,
    force: bool,
    dry_run: bool,
    log_ok: Optional[Path],
    log_fail: Optional[Path],
    max_cases: Optional[int],
) -> Tuple[int, int, int]:
    ok = skipped = failed = 0
    subset = case_ids[:max_cases] if max_cases else case_ids
    print(f"\n{label}: {len(subset)} case(s) to process (of {len(case_ids)} with TS)")

    for i, case_id in enumerate(subset, 1):
        source_case = source_root / case_id
        dest_case = dest_root / case_id
        print(f"\n[{i}/{len(subset)}] {case_id}")
        print(f"  source: {source_case}")
        print(f"  dest:   {dest_case}")

        if _output_pair_exists(str(dest_case), case_id, convention) and not force:
            print("  ○ skip — output already exists (use --force to overwrite)")
            skipped += 1
            continue

        if dry_run:
            print("  (dry-run — would relabel)")
            ok += 1
            continue

        try:
            processor.relabel_from_existing_total_segmentator(
                case_id=case_id,
                source_case_folder=str(source_case),
                dest_case_folder=str(dest_case),
                write_pdf=False,
            )
            if log_ok:
                with open(log_ok, "a") as f:
                    f.write(case_id + "\n")
            print("  ✓ done")
            ok += 1
        except Exception as exc:
            msg = f"{case_id}: {exc}"
            print(f"  ✗ {exc}")
            if log_fail:
                with open(log_fail, "a") as f:
                    f.write(msg + "\n")
            failed += 1

    return ok, skipped, failed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test4 Phase 2: relabel all cases (reuse TotalSegmentator, separate GTVp/GTVn)"
    )
    parser.add_argument(
        "--work-root",
        default=os.getenv("TEST4_WORK_ROOT", "work/retrain_test4"),
        help="Output root (env: TEST4_WORK_ROOT)",
    )
    parser.add_argument(
        "--radcure-source-main-path",
        default=os.getenv("TEST4_RADCURE_SOURCE_MAIN_PATH", os.getenv("MAIN_PATH", "")),
        help="Existing RADCURE MAIN_PATH with TotalSegmentatorRetrain/ (env: TEST4_RADCURE_SOURCE_MAIN_PATH)",
    )
    parser.add_argument(
        "--hecktor-source-cases-root",
        default=os.getenv("TEST4_HECKTOR_SOURCE_CASES_ROOT", os.getenv("HECKTOR_CASES_ROOT", "")),
        help="Existing processed HECKTOR cases root (env: TEST4_HECKTOR_SOURCE_CASES_ROOT)",
    )
    parser.add_argument(
        "--organ-dictionary-path",
        default="",
        help="Defaults to {work-root}/radcure_dictionary_test4.json",
    )
    parser.add_argument("--skip-radcure", action="store_true")
    parser.add_argument("--skip-hecktor", action="store_true")
    parser.add_argument("--force", action="store_true", help="Overwrite existing dest output/")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-cases", type=int, default=None, help="Limit cases per cohort (testing)")
    args = parser.parse_args()

    work_root = Path(args.work_root).resolve()
    work_root.mkdir(parents=True, exist_ok=True)

    organ_dict = args.organ_dictionary_path or str(work_root / "radcure_dictionary_test4.json")
    log_ok = work_root / "relabel_ok.txt"
    log_fail = work_root / "relabel_failed.txt"

    print("=" * 70)
    print("Test4 Phase 2 — relabel from existing TotalSegmentator")
    print("=" * 70)
    print(f"Started:  {datetime.now().isoformat(timespec='seconds')}")
    print(f"Work root: {work_root}")
    print(f"Organ dict: {organ_dict}")
    print(f"Tumor mode: {TUMOR_LABEL_MODE_SEPARATE}")
    print("=" * 70)

    totals = {"ok": 0, "skipped": 0, "failed": 0}

    if not args.skip_radcure:
        if not args.radcure_source_main_path:
            print("ERROR: set --radcure-source-main-path or TEST4_RADCURE_SOURCE_MAIN_PATH / MAIN_PATH")
            sys.exit(1)
        radcure_cases = _list_radcure_cases_with_ts(args.radcure_source_main_path)
        dest_radcure = work_root / "TotalSegmentatorRetrain"
        dest_radcure.mkdir(parents=True, exist_ok=True)
        processor_r = CaseProcessor(
            main_path=str(work_root),
            aws_bucket_name="dummy",
            aws_folder="dummy/",
            organ_dictionary_path=organ_dict,
            convention=RADCURE,
            tumor_label_mode=TUMOR_LABEL_MODE_SEPARATE,
        )
        ok, sk, fl = _run_batch(
            "RADCURE",
            radcure_cases,
            Path(args.radcure_source_main_path) / "TotalSegmentatorRetrain",
            dest_radcure,
            processor_r,
            RADCURE,
            args.force,
            args.dry_run,
            None if args.dry_run else log_ok,
            None if args.dry_run else log_fail,
            args.max_cases,
        )
        totals["ok"] += ok
        totals["skipped"] += sk
        totals["failed"] += fl

    if not args.skip_hecktor:
        if not args.hecktor_source_cases_root:
            print("ERROR: set --hecktor-source-cases-root or TEST4_HECKTOR_SOURCE_CASES_ROOT")
            sys.exit(1)
        hecktor_cases = _list_hecktor_cases_with_ts(args.hecktor_source_cases_root)
        dest_hecktor = work_root / "hecktor"
        dest_hecktor.mkdir(parents=True, exist_ok=True)
        processor_h = CaseProcessor(
            main_path=str(work_root),
            aws_bucket_name="dummy",
            aws_folder="dummy/",
            organ_dictionary_path=organ_dict,
            convention=HECKTOR,
            cases_root=str(dest_hecktor),
            tumor_label_mode=TUMOR_LABEL_MODE_SEPARATE,
        )
        ok, sk, fl = _run_batch(
            "HECKTOR",
            hecktor_cases,
            Path(args.hecktor_source_cases_root),
            dest_hecktor,
            processor_h,
            HECKTOR,
            args.force,
            args.dry_run,
            None if args.dry_run else log_ok,
            None if args.dry_run else log_fail,
            args.max_cases,
        )
        totals["ok"] += ok
        totals["skipped"] += sk
        totals["failed"] += fl

    print("\n" + "=" * 70)
    print("Summary")
    print(f"  Relabeled: {totals['ok']}")
    print(f"  Skipped:   {totals['skipped']}")
    print(f"  Failed:    {totals['failed']}")
    if not args.dry_run:
        print(f"  Logs:      {log_ok}, {log_fail}")
    print("\nNext (Phase 3): build Dataset650 under work/retrain_test4 using same splits as Test3.")
    print("=" * 70)

    if totals["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
