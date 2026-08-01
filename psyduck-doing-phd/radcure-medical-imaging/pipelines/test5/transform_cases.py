#!/usr/bin/env python3
"""
Test5 Phase 2 — transform RADCURE + HECKTOR into one unified RADHECK_{N}/cases/ tree.

- Improved background + separate GTVp/GTVn + canonical organ dictionary
- **No anatomy QC** (keep all transformable cases)
- RADCURE + HECKTOR live side-by-side under ``cases/`` (no separate trees)
- HECKTOR: scan train/val source **and** held-out test1 (both needed for 650 + 152)

Example:

  export TEST5_WORK_ROOT=/media/HDD_8TB/xisca/work/retrain_test5
  export TEST5_RADCURE_SOURCE=/media/HDD_8TB/xisca/dataset/RadcureComplete/TotalSegmentatorRetrain
  # Optional: colon-separated. Defaults = training task1 + test1.
  # export TEST5_HECKTOR_SOURCES=/path/to/task1:/path/to/test1

  python -m pipelines.test5.transform_cases --dry-run
  python -m pipelines.test5.transform_cases
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from image_processor import (
    CaseProcessor,
    HECKTOR,
    RADCURE,
    TUMOR_LABEL_MODE_SEPARATE,
)
from image_processor.conventions import get_hecktor_paths, get_nnunet_case_number
from image_processor.core.mask_generator import MaskGenerator
from image_processor.utils.organ_dictionary import OrganDictionary
from pipelines.hecktor.test_pipeline import hecktor_case_processor_memory_kwargs
from pipelines.test5.paths import (
    BUNDLED_SPLIT_MANIFEST,
    DEFAULT_RADCURE_SOURCE,
    cases_root as cases_root_of,
    default_hecktor_sources,
    radheck_dir,
    work_root,
)

CANONICAL_DICT_TEMPLATE = (
    _REPO_ROOT
    / "image_processor"
    / "resources"
    / "organ_dictionary_hn_canonical.json"
)


def _resolve_radcure_retrain(source_path: str) -> Path:
    p = Path(source_path).expanduser().resolve()
    if not p.is_dir():
        raise FileNotFoundError(f"RADCURE source not found: {p}")
    nested = p / "TotalSegmentatorRetrain"
    if nested.is_dir():
        return nested
    sample = next(
        (d for d in sorted(p.iterdir()) if d.is_dir() and d.name.startswith("RADCURE-")),
        None,
    )
    if sample is not None:
        return p
    raise FileNotFoundError(f"No TotalSegmentatorRetrain/ or RADCURE-* under {p}")


def _list_radcure_with_ts(retrain: Path) -> List[str]:
    out = []
    for name in sorted(os.listdir(retrain)):
        if not name.startswith("RADCURE-"):
            continue
        case_dir = retrain / name
        if case_dir.is_dir() and (case_dir / "total_segmentator_output").is_dir():
            out.append(name)
    return out


def _hecktor_raw_ready(case_dir: Path, case_id: str) -> bool:
    paths = get_hecktor_paths(str(case_dir), case_id)
    return os.path.isfile(paths["path_ct"]) and os.path.isfile(paths["path_mask"])


def _list_hecktor(source: Path) -> List[Tuple[str, bool]]:
    """Return [(case_id, has_total_segmentator_output), ...]."""
    if not source.is_dir():
        return []
    out: List[Tuple[str, bool]] = []
    for name in sorted(os.listdir(source)):
        if name.startswith("."):
            continue
        case_dir = source / name
        if not case_dir.is_dir():
            continue
        if not _hecktor_raw_ready(case_dir, name):
            continue
        has_ts = (case_dir / "total_segmentator_output").is_dir()
        out.append((name, has_ts))
    return out


def _merge_hecktor_sources(
    sources: List[Path],
) -> Tuple[Dict[str, Tuple[Path, bool]], List[str]]:
    """
    Dedupe case IDs across HECKTOR roots (first source wins).

    Returns (case_id → (src_dir, has_ts), warnings).
    """
    merged: Dict[str, Tuple[Path, bool]] = {}
    warnings: List[str] = []
    for src in sources:
        if not src.is_dir():
            warnings.append(f"HECKTOR source missing (skipped): {src}")
            continue
        listed = _list_hecktor(src)
        print(f"  HECKTOR source {src}: {len(listed)} raw-ready case(s)")
        for case_id, has_ts in listed:
            if case_id in merged:
                continue
            merged[case_id] = (src / case_id, has_ts)
    return merged, warnings


def _output_ok(case_folder: Path, case_id: str, convention: str) -> bool:
    num = get_nnunet_case_number(case_id, convention)
    fname = f"case_{num}_0000.nii.gz"
    out = case_folder / "output"
    return (out / "image" / fname).is_file() and (out / "labels" / fname).is_file()


def _ensure_organ_dict(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        return path
    if CANONICAL_DICT_TEMPLATE.is_file():
        shutil.copy2(CANONICAL_DICT_TEMPLATE, path)
    else:
        OrganDictionary.from_hn_canonical(
            str(path), separate_gtvp_gtvn=True, save=True
        )
    return path


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _stage_hecktor(src_case: Path, dest_case: Path, case_id: str) -> None:
    dest_case.mkdir(parents=True, exist_ok=True)
    sp = get_hecktor_paths(str(src_case), case_id)
    dp = get_hecktor_paths(str(dest_case), case_id)
    for key in ("path_ct", "path_mask"):
        s, d = Path(sp[key]), Path(dp[key])
        if not d.is_file():
            _link_or_copy(s, d)


def _free_memory(processor: Optional[CaseProcessor] = None) -> None:
    gc.collect()
    if processor is not None:
        processor._maybe_empty_cuda_cache()


def _write_status(radheck: Path, payload: dict) -> None:
    path = radheck / "STATUS.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"Wrote {path}")


def _set_current_pointer(work: Path, radheck: Path) -> None:
    pointer = work / "RADHECK_CURRENT"
    if pointer.is_symlink() or pointer.exists():
        pointer.unlink()
    try:
        pointer.symlink_to(radheck.name)
    except OSError:
        pointer.write_text(str(radheck) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test5: transform RADCURE+HECKTOR into unified RADHECK_{N}/cases/"
    )
    parser.add_argument("--work-root", default=str(work_root()))
    parser.add_argument(
        "--radcure-source",
        default=os.getenv("TEST5_RADCURE_SOURCE", DEFAULT_RADCURE_SOURCE),
    )
    parser.add_argument(
        "--hecktor-source",
        action="append",
        default=None,
        help="HECKTOR cases root (repeatable). Default: train task1 + test1.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--skip-radcure", action="store_true")
    parser.add_argument("--skip-hecktor", action="store_true")
    args = parser.parse_args()

    work = Path(args.work_root).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)

    radcure_retrain = (
        None if args.skip_radcure else _resolve_radcure_retrain(args.radcure_source)
    )
    if args.skip_hecktor:
        hecktor_sources: List[Path] = []
    elif args.hecktor_source:
        hecktor_sources = [Path(p).expanduser().resolve() for p in args.hecktor_source]
    else:
        hecktor_sources = [p.expanduser().resolve() for p in default_hecktor_sources()]

    radcure_ids = [] if radcure_retrain is None else _list_radcure_with_ts(radcure_retrain)
    hecktor_merged: Dict[str, Tuple[Path, bool]] = {}
    hecktor_warnings: List[str] = []
    if hecktor_sources:
        print("Scanning HECKTOR sources:")
        hecktor_merged, hecktor_warnings = _merge_hecktor_sources(hecktor_sources)
        for w in hecktor_warnings:
            print(f"  WARNING: {w}")

    n_planned = len(radcure_ids) + len(hecktor_merged)
    if n_planned == 0:
        print("ERROR: no cases found under the given sources")
        sys.exit(1)

    radheck = radheck_dir(work, n_planned)
    dest_cases = cases_root_of(radheck)
    dest_cases.mkdir(parents=True, exist_ok=True)
    organ_dict = _ensure_organ_dict(radheck / "organ_dictionary_test5.json")
    # Convenience copy at work root for ORGAN_DICTIONARY_PATH
    work_dict = work / "organ_dictionary_test5.json"
    if not args.dry_run and organ_dict.is_file():
        if work_dict.exists() or work_dict.is_symlink():
            work_dict.unlink()
        try:
            os.link(organ_dict, work_dict)
        except OSError:
            shutil.copy2(organ_dict, work_dict)

    man_dst = work / "split_manifest.json"
    if BUNDLED_SPLIT_MANIFEST.is_file() and not man_dst.is_file():
        shutil.copy2(BUNDLED_SPLIT_MANIFEST, man_dst)
        print(f"Restored split_manifest.json → {man_dst}")

    os.environ.setdefault("HECKTOR_CLEANUP_INTERMEDIATES", "1")
    os.environ.setdefault("HECKTOR_TS_NR_THR_SAVING", "1")
    mem_kwargs = hecktor_case_processor_memory_kwargs()

    print("=" * 70)
    print("Test5 Phase 2 — unified RADHECK transform (no anatomy QC)")
    print("=" * 70)
    print(f"Started:     {datetime.now().isoformat(timespec='seconds')}")
    print(f"Work root:   {work}")
    print(f"RADHECK dir: {radheck}")
    print(f"Cases root:  {dest_cases}")
    print(
        f"Planned N:   {n_planned} "
        f"(RADCURE={len(radcure_ids)}, HECKTOR={len(hecktor_merged)})"
    )
    print(f"Organ dict:  {organ_dict}")
    print(f"Background:  {MaskGenerator.BACKGROUND_MODE_IMPROVED}")
    print(f"Tumor mode:  {TUMOR_LABEL_MODE_SEPARATE}")
    print("Anatomy QC:  disabled (keep all cases)")
    if len(hecktor_merged) < 200 and not args.skip_hecktor:
        print(
            "\nWARNING: HECKTOR case count looks low for Dataset650 train/val.\n"
            "  Manifest expects ~227 train + ~57 val from the training zip,\n"
            "  plus held-out test from test1.\n"
            f"  Sources used: {hecktor_sources}\n"
            "  Set TEST5_HECKTOR_TRAIN_SOURCE if task1 training is elsewhere.\n"
        )
    print("=" * 70)

    def _proc(convention: str, *, full_hecktor: bool = False) -> CaseProcessor:
        kwargs = dict(
            main_path=str(work),
            aws_bucket_name="dummy",
            aws_folder="dummy/",
            organ_dictionary_path=str(organ_dict),
            convention=convention,
            tumor_label_mode=TUMOR_LABEL_MODE_SEPARATE,
            background_mode=MaskGenerator.BACKGROUND_MODE_IMPROVED,
            anatomy_qc_threshold=None,
            cases_root=str(dest_cases),
        )
        if full_hecktor:
            kwargs.update(mem_kwargs)
        return CaseProcessor(**kwargs)

    totals = {"ok": 0, "skipped": 0, "failed": 0}
    log_ok = radheck / "transform_ok.txt"
    log_fail = radheck / "transform_failed.txt"
    if not args.dry_run:
        log_ok.write_text("")
        log_fail.write_text("")

    # --- RADCURE ---
    if radcure_ids:
        subset = radcure_ids[: args.max_cases] if args.max_cases else radcure_ids
        proc = _proc(RADCURE)
        print(f"\nRADCURE: {len(subset)} case(s)")
        for i, case_id in enumerate(subset, 1):
            src = radcure_retrain / case_id
            dst = dest_cases / case_id
            print(f"\n[{i}/{len(subset)}] {case_id}")
            if _output_ok(dst, case_id, RADCURE) and not args.force:
                print("  ○ skip — output exists")
                totals["skipped"] += 1
                continue
            if args.dry_run:
                print("  (dry-run)")
                totals["ok"] += 1
                continue
            try:
                proc.relabel_from_existing_total_segmentator(
                    case_id=case_id,
                    source_case_folder=str(src),
                    dest_case_folder=str(dst),
                    write_pdf=False,
                )
                with open(log_ok, "a") as f:
                    f.write(case_id + "\n")
                totals["ok"] += 1
                print("  ✓")
            except Exception as exc:
                with open(log_fail, "a") as f:
                    f.write(f"{case_id}: {exc}\n")
                totals["failed"] += 1
                print(f"  ✗ {exc}")
            finally:
                _free_memory(proc)

    # --- HECKTOR ---
    if hecktor_merged:
        items = list(hecktor_merged.items())
        if args.max_cases:
            items = items[: args.max_cases]
        proc_relabel = _proc(HECKTOR)
        proc_full = _proc(HECKTOR, full_hecktor=True)
        print(f"\nHECKTOR: {len(items)} case(s)")
        for i, (case_id, (src, has_ts)) in enumerate(items, 1):
            dst = dest_cases / case_id
            mode = "relabel" if has_ts else "full_process"
            print(f"\n[{i}/{len(items)}] {case_id} ({mode})")
            if _output_ok(dst, case_id, HECKTOR) and not args.force:
                print("  ○ skip — output exists")
                totals["skipped"] += 1
                continue
            if args.dry_run:
                print("  (dry-run)")
                totals["ok"] += 1
                continue
            try:
                if has_ts:
                    proc_relabel.relabel_from_existing_total_segmentator(
                        case_id=case_id,
                        source_case_folder=str(src),
                        dest_case_folder=str(dst),
                        write_pdf=False,
                    )
                else:
                    _stage_hecktor(src, dst, case_id)
                    result = proc_full.process_case(case_id)
                    if result.get("status") == "skipped" and args.force:
                        out = dst / "output"
                        if out.is_dir():
                            shutil.rmtree(out)
                        proc_full.process_case(case_id)
                with open(log_ok, "a") as f:
                    f.write(case_id + "\n")
                totals["ok"] += 1
                print("  ✓")
            except Exception as exc:
                with open(log_fail, "a") as f:
                    f.write(f"{case_id}: {exc}\n")
                totals["failed"] += 1
                print(f"  ✗ {exc}")
            finally:
                _free_memory(proc_full if not has_ts else proc_relabel)

    n_ready = sum(
        1
        for p in dest_cases.iterdir()
        if p.is_dir()
        and (p / "output" / "image").is_dir()
        and any((p / "output" / "image").glob("*.nii.gz"))
    )
    status = {
        "updated": datetime.now().isoformat(timespec="seconds"),
        "work_root": str(work),
        "radheck_dir": str(radheck),
        "cases_root": str(dest_cases),
        "n_planned": n_planned,
        "n_ready_outputs": n_ready,
        "radcure_source": str(radcure_retrain) if radcure_retrain else None,
        "hecktor_sources": [str(p) for p in hecktor_sources],
        "tumor_label_mode": TUMOR_LABEL_MODE_SEPARATE,
        "background_mode": MaskGenerator.BACKGROUND_MODE_IMPROVED,
        "anatomy_qc": False,
        "totals": totals,
        "next": [
            "python -m pipelines.test5.build_datasets --dry-run",
            "python -m pipelines.test5.build_datasets --link hardlink",
        ],
    }
    if not args.dry_run:
        _write_status(radheck, status)
        _set_current_pointer(work, radheck)

    print("\n" + "=" * 70)
    print("Summary")
    print(f"  Transformed: {totals['ok']}")
    print(f"  Skipped:     {totals['skipped']}")
    print(f"  Failed:      {totals['failed']}")
    print(f"  Ready outs:  {n_ready}")
    print(f"  Folder:      {radheck}")
    print("\nNext: python -m pipelines.test5.build_datasets")
    print("=" * 70)
    if totals["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
