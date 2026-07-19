#!/usr/bin/env python3
"""
Test5 Phase 2 — relabel cases with improved background + anatomy QC.

Same as Test4 (reuse ``total_segmentator_output/``, ``tumor_label_mode=separate``)
plus:

- Canonical H&N organ dictionary (case-independent indices)
- ``background_mode=improved`` (FOV body mask + L/R symmetry + Z continuity)
- Anatomy QC threshold (default 0.70): discard non-human / wrong FOV

Outputs under ``--work-root`` (default ``work/retrain_test5``):

  TotalSegmentatorRetrain/{RADCURE-XXXX}/output/
  hecktor/{HECKTOR-ID}/output/
  logs/anatomy_qc/anatomy_qc_decisions.jsonl
  anatomy_qc_discarded.csv
  radcure_dictionary_test5.json

Example:

  export TEST5_WORK_ROOT=/media/.../work/retrain_test5
  export TEST5_RADCURE_SOURCE_MAIN_PATH=.../RadcureComplete
  export TEST5_HECKTOR_SOURCE_CASES_ROOT=.../hecktor/.../cases

  python -m pipelines.test5.relabel_tumor_batch --dry-run
  python -m pipelines.test5.relabel_tumor_batch
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from image_processor import (
    AnatomyQCRejected,
    CaseProcessor,
    HECKTOR,
    RADCURE,
    TUMOR_LABEL_MODE_SEPARATE,
)
from image_processor.conventions import get_hecktor_paths, get_nnunet_case_number
from image_processor.core.mask_generator import MaskGenerator
from image_processor.utils.anatomy_qc import append_qc_log, write_discard_summary_csv
from image_processor.utils.organ_dictionary import OrganDictionary

DEFAULT_ANATOMY_QC_THRESHOLD = 0.70
CANONICAL_DICT_TEMPLATE = (
    _REPO_ROOT
    / "image_processor"
    / "resources"
    / "organ_dictionary_hn_canonical.json"
)


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


def _ensure_canonical_organ_dict(path: Path) -> Path:
    """Load or create case-independent GTVp/GTVn organ dictionary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        print(f"Using existing organ dictionary: {path}")
        return path
    if CANONICAL_DICT_TEMPLATE.is_file():
        shutil.copy2(CANONICAL_DICT_TEMPLATE, path)
        print(f"Seeded organ dictionary from template → {path}")
        return path
    OrganDictionary.from_hn_canonical(
        str(path),
        separate_gtvp_gtvn=True,
        save=True,
    )
    print(f"Built canonical organ dictionary → {path}")
    return path


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
    qc_log: Optional[Path],
    discard_records: list,
    max_cases: Optional[int],
) -> Tuple[int, int, int, int]:
    ok = skipped = failed = discarded = 0
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
            print("  (dry-run — would relabel + QC)")
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
            if qc_log and processor.anatomy_qc_threshold is not None:
                append_qc_log(
                    str(qc_log),
                    case_id=case_id,
                    convention=convention,
                    record={
                        "decision": "keep",
                        "keep": True,
                        "threshold": processor.anatomy_qc_threshold,
                    },
                )
            print("  ✓ done")
            ok += 1
        except AnatomyQCRejected as exc:
            discarded += 1
            record = {
                "case_id": case_id,
                "convention": convention,
                **exc.record,
            }
            discard_records.append(record)
            if qc_log:
                append_qc_log(
                    str(qc_log),
                    case_id=case_id,
                    convention=convention,
                    record=exc.record,
                )
            print(f"  ⊗ QC discard — {exc}")
        except Exception as exc:
            msg = f"{case_id}: {exc}"
            print(f"  ✗ {exc}")
            if log_fail:
                with open(log_fail, "a") as f:
                    f.write(msg + "\n")
            failed += 1

    return ok, skipped, failed, discarded


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Test5 Phase 2: relabel with improved bg + anatomy QC "
            "(reuse TotalSegmentator, separate GTVp/GTVn)"
        )
    )
    parser.add_argument(
        "--work-root",
        default=os.getenv("TEST5_WORK_ROOT", "work/retrain_test5"),
        help="Output root (env: TEST5_WORK_ROOT)",
    )
    parser.add_argument(
        "--radcure-source-main-path",
        default=os.getenv(
            "TEST5_RADCURE_SOURCE_MAIN_PATH",
            os.getenv("TEST4_RADCURE_SOURCE_MAIN_PATH", os.getenv("MAIN_PATH", "")),
        ),
        help="RADCURE MAIN_PATH with TotalSegmentatorRetrain/",
    )
    parser.add_argument(
        "--hecktor-source-cases-root",
        default=os.getenv(
            "TEST5_HECKTOR_SOURCE_CASES_ROOT",
            os.getenv("TEST4_HECKTOR_SOURCE_CASES_ROOT", os.getenv("HECKTOR_CASES_ROOT", "")),
        ),
        help="Processed HECKTOR cases root with total_segmentator_output/",
    )
    parser.add_argument(
        "--organ-dictionary-path",
        default="",
        help="Defaults to {work-root}/radcure_dictionary_test5.json",
    )
    parser.add_argument(
        "--anatomy-qc-threshold",
        type=float,
        default=float(os.getenv("TEST5_ANATOMY_QC_THRESHOLD", DEFAULT_ANATOMY_QC_THRESHOLD)),
        help=f"Anatomy QC keep threshold (default {DEFAULT_ANATOMY_QC_THRESHOLD})",
    )
    parser.add_argument(
        "--skip-anatomy-qc",
        action="store_true",
        help="Disable anatomy QC (not recommended for Test5)",
    )
    parser.add_argument("--skip-radcure", action="store_true")
    parser.add_argument("--skip-hecktor", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-cases", type=int, default=None)
    args = parser.parse_args()

    work_root = Path(args.work_root).resolve()
    work_root.mkdir(parents=True, exist_ok=True)

    organ_dict = Path(
        args.organ_dictionary_path or (work_root / "radcure_dictionary_test5.json")
    )
    if not args.dry_run:
        _ensure_canonical_organ_dict(organ_dict)

    qc_threshold = None if args.skip_anatomy_qc else float(args.anatomy_qc_threshold)
    log_ok = work_root / "relabel_ok.txt"
    log_fail = work_root / "relabel_failed.txt"
    qc_dir = work_root / "logs" / "anatomy_qc"
    qc_log = qc_dir / "anatomy_qc_decisions.jsonl"
    discard_csv = work_root / "anatomy_qc_discarded.csv"
    discard_records: list = []

    print("=" * 70)
    print("Test5 Phase 2 — improved preprocess relabel")
    print("=" * 70)
    print(f"Started:     {datetime.now().isoformat(timespec='seconds')}")
    print(f"Work root:   {work_root}")
    print(f"Organ dict:  {organ_dict}")
    print(f"Tumor mode:  {TUMOR_LABEL_MODE_SEPARATE}")
    print(f"Background:  {MaskGenerator.BACKGROUND_MODE_IMPROVED}")
    print(f"Anatomy QC:  {qc_threshold if qc_threshold is not None else 'disabled'}")
    print("=" * 70)

    totals = {"ok": 0, "skipped": 0, "failed": 0, "discarded": 0}

    def _processor(convention: str, cases_root: Optional[str] = None) -> CaseProcessor:
        kwargs = dict(
            main_path=str(work_root),
            aws_bucket_name="dummy",
            aws_folder="dummy/",
            organ_dictionary_path=str(organ_dict),
            convention=convention,
            tumor_label_mode=TUMOR_LABEL_MODE_SEPARATE,
            background_mode=MaskGenerator.BACKGROUND_MODE_IMPROVED,
            anatomy_qc_threshold=qc_threshold,
        )
        if cases_root is not None:
            kwargs["cases_root"] = cases_root
        return CaseProcessor(**kwargs)

    if not args.skip_radcure:
        if not args.radcure_source_main_path:
            print("ERROR: set --radcure-source-main-path or TEST5_RADCURE_SOURCE_MAIN_PATH")
            sys.exit(1)
        radcure_cases = _list_radcure_cases_with_ts(args.radcure_source_main_path)
        dest_radcure = work_root / "TotalSegmentatorRetrain"
        dest_radcure.mkdir(parents=True, exist_ok=True)
        ok, sk, fl, dc = _run_batch(
            "RADCURE",
            radcure_cases,
            Path(args.radcure_source_main_path) / "TotalSegmentatorRetrain",
            dest_radcure,
            _processor(RADCURE),
            RADCURE,
            args.force,
            args.dry_run,
            None if args.dry_run else log_ok,
            None if args.dry_run else log_fail,
            None if args.dry_run else qc_log,
            discard_records,
            args.max_cases,
        )
        totals["ok"] += ok
        totals["skipped"] += sk
        totals["failed"] += fl
        totals["discarded"] += dc

    if not args.skip_hecktor:
        if not args.hecktor_source_cases_root:
            print("ERROR: set --hecktor-source-cases-root or TEST5_HECKTOR_SOURCE_CASES_ROOT")
            sys.exit(1)
        hecktor_cases = _list_hecktor_cases_with_ts(args.hecktor_source_cases_root)
        dest_hecktor = work_root / "hecktor"
        dest_hecktor.mkdir(parents=True, exist_ok=True)
        ok, sk, fl, dc = _run_batch(
            "HECKTOR",
            hecktor_cases,
            Path(args.hecktor_source_cases_root),
            dest_hecktor,
            _processor(HECKTOR, cases_root=str(dest_hecktor)),
            HECKTOR,
            args.force,
            args.dry_run,
            None if args.dry_run else log_ok,
            None if args.dry_run else log_fail,
            None if args.dry_run else qc_log,
            discard_records,
            args.max_cases,
        )
        totals["ok"] += ok
        totals["skipped"] += sk
        totals["failed"] += fl
        totals["discarded"] += dc

    if discard_records and not args.dry_run:
        write_discard_summary_csv(discard_records, str(discard_csv))
        print(f"Wrote discard summary: {discard_csv}")

    print("\n" + "=" * 70)
    print("Summary")
    print(f"  Relabeled:  {totals['ok']}")
    print(f"  Skipped:    {totals['skipped']}")
    print(f"  QC discard: {totals['discarded']}")
    print(f"  Failed:     {totals['failed']}")
    if not args.dry_run:
        print(f"  Logs:       {log_ok}, {log_fail}")
        print(f"  QC log:     {qc_log}")
    print("\nNext: python -m pipelines.test5.build_dataset650")
    print("=" * 70)

    if totals["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
