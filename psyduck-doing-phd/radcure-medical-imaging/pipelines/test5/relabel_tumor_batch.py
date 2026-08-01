#!/usr/bin/env python3
"""
Test5 Phase 2 — transform original sources → improved bg + anatomy QC labels.

Reuses ``total_segmentator_output/`` when present (RADCURE / processed HECKTOR).
If HECKTOR cases are raw CT+mask only (no TS), runs full ``process_case`` into
the work root with intermediates cleaned up to limit disk/RAM.

Defaults (server):

  RADCURE: /media/HDD_8TB/xisca/dataset/RadcureComplete/TotalSegmentatorRetrain
  HECKTOR: /media/HDD_8TB/xisca/dataset/hecktor/test1/unzipped/test1

Example:

  export TEST5_WORK_ROOT=/media/HDD_8TB/xisca/work/retrain_test5
  export TEST5_RADCURE_SOURCE=/media/HDD_8TB/xisca/dataset/RadcureComplete/TotalSegmentatorRetrain
  export TEST5_HECKTOR_SOURCE_CASES_ROOT=/media/HDD_8TB/xisca/dataset/hecktor/test1/unzipped/test1

  python -m pipelines.test5.relabel_tumor_batch --dry-run
  python -m pipelines.test5.relabel_tumor_batch
"""

from __future__ import annotations

import argparse
import gc
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
from pipelines.hecktor.test_pipeline import hecktor_case_processor_memory_kwargs

# Soft keep threshold: hard-fail still catches empty/non-human FOVs.
DEFAULT_ANATOMY_QC_THRESHOLD = 0.50
CANONICAL_DICT_TEMPLATE = (
    _REPO_ROOT
    / "image_processor"
    / "resources"
    / "organ_dictionary_hn_canonical.json"
)

DEFAULT_RADCURE_SOURCE = (
    "/media/HDD_8TB/xisca/dataset/RadcureComplete/TotalSegmentatorRetrain"
)
DEFAULT_HECKTOR_SOURCE = (
    "/media/HDD_8TB/xisca/work/retrain_test5/hecktor"
)


def _resolve_radcure_retrain(source_path: str) -> Path:
    """
    Accept either MAIN_PATH (…/RadcureComplete) or …/TotalSegmentatorRetrain.
    """
    p = Path(source_path).expanduser().resolve()
    if not p.is_dir():
        raise FileNotFoundError(f"RADCURE source not found: {p}")
    nested = p / "TotalSegmentatorRetrain"
    if nested.is_dir():
        return nested
    # Direct TotalSegmentatorRetrain (or a folder of RADCURE-* cases)
    sample = next(
        (
            d
            for d in sorted(p.iterdir())
            if d.is_dir() and d.name.startswith("RADCURE-")
        ),
        None,
    )
    if sample is not None:
        return p
    raise FileNotFoundError(
        f"No TotalSegmentatorRetrain/ or RADCURE-* cases under {p}"
    )


def _list_radcure_cases_with_ts(retrain: Path) -> List[str]:
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


def _hecktor_raw_ready(case_dir: Path, case_id: str) -> bool:
    paths = get_hecktor_paths(str(case_dir), case_id)
    return os.path.isfile(paths["path_ct"]) and os.path.isfile(paths["path_mask"])


def _list_hecktor_cases(source_cases_root: Path) -> List[Tuple[str, bool]]:
    """
    Return (case_id, has_total_segmentator_output).

    Includes raw HECKTOR cases (CT+mask only) so they can be fully processed.
    """
    if not source_cases_root.is_dir():
        raise FileNotFoundError(f"HECKTOR source not found: {source_cases_root}")
    cases: List[Tuple[str, bool]] = []
    for name in sorted(os.listdir(source_cases_root)):
        if name.startswith("."):
            continue
        case_dir = source_cases_root / name
        if not case_dir.is_dir():
            continue
        if not _hecktor_raw_ready(case_dir, name):
            continue
        has_ts = (case_dir / "total_segmentator_output").is_dir()
        cases.append((name, has_ts))
    return cases


def _output_pair_exists(case_folder: str, case_id: str, convention: str) -> bool:
    num = get_nnunet_case_number(case_id, convention)
    fname = f"case_{num}_0000.nii.gz"
    out = Path(case_folder) / "output"
    return (out / "image" / fname).is_file() and (out / "labels" / fname).is_file()


def _ensure_canonical_organ_dict(path: Path) -> Path:
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


def _link_or_copy_file(src: Path, dst: Path) -> None:
    """Prefer hardlink (same filesystem) to avoid duplicating large NIfTIs."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _stage_hecktor_inputs(source_case: Path, dest_case: Path, case_id: str) -> None:
    """Place CT+mask under dest without writing into the original dataset tree."""
    dest_case.mkdir(parents=True, exist_ok=True)
    src_paths = get_hecktor_paths(str(source_case), case_id)
    dst_paths = get_hecktor_paths(str(dest_case), case_id)
    for key in ("path_ct", "path_mask"):
        src = Path(src_paths[key])
        dst = Path(dst_paths[key])
        if dst.is_file():
            continue
        _link_or_copy_file(src, dst)


def _free_memory(processor: Optional[CaseProcessor] = None) -> None:
    gc.collect()
    if processor is not None:
        processor._maybe_empty_cuda_cache()
    else:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _run_radcure_batch(
    case_ids: List[str],
    source_root: Path,
    dest_root: Path,
    processor: CaseProcessor,
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
    print(f"\nRADCURE: {len(subset)} case(s) to process (of {len(case_ids)} with TS)")

    for i, case_id in enumerate(subset, 1):
        source_case = source_root / case_id
        dest_case = dest_root / case_id
        print(f"\n[{i}/{len(subset)}] {case_id}")
        print(f"  source: {source_case}")
        print(f"  dest:   {dest_case}")

        if _output_pair_exists(str(dest_case), case_id, RADCURE) and not force:
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
                    convention=RADCURE,
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
                "convention": RADCURE,
                **exc.record,
            }
            discard_records.append(record)
            if qc_log:
                append_qc_log(
                    str(qc_log),
                    case_id=case_id,
                    convention=RADCURE,
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
        finally:
            _free_memory(processor)

    return ok, skipped, failed, discarded


def _run_hecktor_batch(
    cases: List[Tuple[str, bool]],
    source_root: Path,
    dest_root: Path,
    processor_relabel: CaseProcessor,
    processor_full: CaseProcessor,
    force: bool,
    dry_run: bool,
    log_ok: Optional[Path],
    log_fail: Optional[Path],
    qc_log: Optional[Path],
    discard_records: list,
    max_cases: Optional[int],
) -> Tuple[int, int, int, int]:
    ok = skipped = failed = discarded = 0
    subset = cases[:max_cases] if max_cases else cases
    n_ts = sum(1 for _, has_ts in subset if has_ts)
    n_raw = len(subset) - n_ts
    print(
        f"\nHECKTOR: {len(subset)} case(s) "
        f"(relabel TS={n_ts}, full process raw={n_raw})"
    )

    for i, (case_id, has_ts) in enumerate(subset, 1):
        source_case = source_root / case_id
        dest_case = dest_root / case_id
        mode = "relabel" if has_ts else "full_process"
        print(f"\n[{i}/{len(subset)}] {case_id} ({mode})")
        print(f"  source: {source_case}")
        print(f"  dest:   {dest_case}")

        if _output_pair_exists(str(dest_case), case_id, HECKTOR) and not force:
            print("  ○ skip — output already exists (use --force to overwrite)")
            skipped += 1
            continue

        if dry_run:
            print(f"  (dry-run — would {mode} + QC)")
            ok += 1
            continue

        try:
            if has_ts:
                processor_relabel.relabel_from_existing_total_segmentator(
                    case_id=case_id,
                    source_case_folder=str(source_case),
                    dest_case_folder=str(dest_case),
                    write_pdf=False,
                )
            else:
                # Never write TotalSegmentator into the original dataset tree.
                _stage_hecktor_inputs(source_case, dest_case, case_id)
                result = processor_full.process_case(case_id)
                if result.get("status") == "skipped" and force:
                    # Force: remove outputs and re-run
                    out = dest_case / "output"
                    if out.is_dir():
                        shutil.rmtree(out)
                    result = processor_full.process_case(case_id)
                # QC for full process (not wired inside process_case)
                if processor_full.anatomy_qc_threshold is not None:
                    from image_processor.utils.image_processing import ImageProcessor
                    import nibabel as nib
                    import numpy as np

                    paths = get_hecktor_paths(str(dest_case), case_id)
                    ct = nib.load(paths["path_ct"]).get_fdata().astype(np.float32)
                    tumor = nib.load(paths["path_mask"]).get_fdata().astype(np.int32)
                    non_zero = ImageProcessor.get_non_zero_slices(tumor)
                    z = tumor.shape[2]
                    if not non_zero:
                        slices = list(range(z))
                    else:
                        start = max(
                            int(min(non_zero)) - processor_full.slice_expansion, 0
                        )
                        end = min(
                            int(max(non_zero)) + processor_full.slice_expansion, z - 1
                        )
                        slices = list(range(start, end + 1))
                    processor_full._maybe_reject_anatomy_qc(
                        case_id, ct, tumor, slices
                    )

            if log_ok:
                with open(log_ok, "a") as f:
                    f.write(case_id + "\n")
            active = processor_relabel if has_ts else processor_full
            if qc_log and active.anatomy_qc_threshold is not None:
                append_qc_log(
                    str(qc_log),
                    case_id=case_id,
                    convention=HECKTOR,
                    record={
                        "decision": "keep",
                        "keep": True,
                        "threshold": active.anatomy_qc_threshold,
                        "mode": mode,
                    },
                )
            print("  ✓ done")
            ok += 1
        except AnatomyQCRejected as exc:
            discarded += 1
            record = {
                "case_id": case_id,
                "convention": HECKTOR,
                **exc.record,
            }
            discard_records.append(record)
            if qc_log:
                append_qc_log(
                    str(qc_log),
                    case_id=case_id,
                    convention=HECKTOR,
                    record=exc.record,
                )
            # Drop heavy outputs for discarded cases to free disk
            out = dest_case / "output"
            if out.is_dir():
                shutil.rmtree(out, ignore_errors=True)
            print(f"  ⊗ QC discard — {exc}")
        except Exception as exc:
            msg = f"{case_id}: {exc}"
            print(f"  ✗ {exc}")
            if log_fail:
                with open(log_fail, "a") as f:
                    f.write(msg + "\n")
            failed += 1
        finally:
            _free_memory(processor_full if not has_ts else processor_relabel)

    return ok, skipped, failed, discarded


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Test5 Phase 2: transform sources with improved bg + anatomy QC "
            "(relabel TS when present; full process raw HECKTOR)"
        )
    )
    parser.add_argument(
        "--work-root",
        default=os.getenv("TEST5_WORK_ROOT", "work/retrain_test5"),
        help="Output root (env: TEST5_WORK_ROOT)",
    )
    parser.add_argument(
        "--radcure-source",
        "--radcure-source-main-path",
        dest="radcure_source",
        default=os.getenv(
            "TEST5_RADCURE_SOURCE",
            os.getenv(
                "TEST5_RADCURE_SOURCE_MAIN_PATH",
                os.getenv("TEST4_RADCURE_SOURCE_MAIN_PATH", DEFAULT_RADCURE_SOURCE),
            ),
        ),
        help=(
            "RADCURE TotalSegmentatorRetrain/ or its parent MAIN_PATH "
            f"(default: {DEFAULT_RADCURE_SOURCE})"
        ),
    )
    parser.add_argument(
        "--hecktor-source-cases-root",
        default=os.getenv(
            "TEST5_HECKTOR_SOURCE_CASES_ROOT",
            os.getenv("TEST4_HECKTOR_SOURCE_CASES_ROOT", DEFAULT_HECKTOR_SOURCE),
        ),
        help=f"HECKTOR cases root (default: {DEFAULT_HECKTOR_SOURCE})",
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

    # Prefer low memory for any full HECKTOR TotalSegmentator runs
    os.environ.setdefault("HECKTOR_CLEANUP_INTERMEDIATES", "1")
    os.environ.setdefault("HECKTOR_TS_NR_THR_SAVING", "1")
    mem_kwargs = hecktor_case_processor_memory_kwargs()

    print("=" * 70)
    print("Test5 Phase 2 — transform original sources (memory-aware)")
    print("=" * 70)
    print(f"Started:     {datetime.now().isoformat(timespec='seconds')}")
    print(f"Work root:   {work_root}")
    print(f"Organ dict:  {organ_dict}")
    print(f"Tumor mode:  {TUMOR_LABEL_MODE_SEPARATE}")
    print(f"Background:  {MaskGenerator.BACKGROUND_MODE_IMPROVED}")
    print(f"Anatomy QC:  {qc_threshold if qc_threshold is not None else 'disabled'}")
    print(f"HECKTOR mem: {mem_kwargs}")
    print("=" * 70)

    totals = {"ok": 0, "skipped": 0, "failed": 0, "discarded": 0}

    def _processor(
        convention: str,
        cases_root: Optional[str] = None,
        *,
        full_hecktor: bool = False,
    ) -> CaseProcessor:
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
        if full_hecktor:
            kwargs.update(mem_kwargs)
        return CaseProcessor(**kwargs)

    if not args.skip_radcure:
        if not args.radcure_source:
            print("ERROR: set --radcure-source or TEST5_RADCURE_SOURCE")
            sys.exit(1)
        radcure_retrain = _resolve_radcure_retrain(args.radcure_source)
        print(f"RADCURE source (resolved): {radcure_retrain}")
        radcure_cases = _list_radcure_cases_with_ts(radcure_retrain)
        dest_radcure = work_root / "TotalSegmentatorRetrain"
        dest_radcure.mkdir(parents=True, exist_ok=True)
        ok, sk, fl, dc = _run_radcure_batch(
            radcure_cases,
            radcure_retrain,
            dest_radcure,
            _processor(RADCURE),
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
        hecktor_source = Path(args.hecktor_source_cases_root).expanduser().resolve()
        print(f"HECKTOR source: {hecktor_source}")
        hecktor_cases = _list_hecktor_cases(hecktor_source)
        dest_hecktor = work_root / "hecktor"
        dest_hecktor.mkdir(parents=True, exist_ok=True)
        ok, sk, fl, dc = _run_hecktor_batch(
            hecktor_cases,
            hecktor_source,
            dest_hecktor,
            _processor(HECKTOR, cases_root=str(dest_hecktor)),
            _processor(HECKTOR, cases_root=str(dest_hecktor), full_hecktor=True),
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
    print(f"  Transformed: {totals['ok']}")
    print(f"  Skipped:     {totals['skipped']}")
    print(f"  QC discard:  {totals['discarded']}")
    print(f"  Failed:      {totals['failed']}")
    if not args.dry_run:
        print(f"  Logs:        {log_ok}, {log_fail}")
        print(f"  QC log:      {qc_log}")
    print("\nNext: python -m pipelines.test5.build_dataset650 --link")
    print("=" * 70)

    if totals["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
