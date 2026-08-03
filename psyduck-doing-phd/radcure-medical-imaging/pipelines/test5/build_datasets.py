#!/usr/bin/env python3
"""
Test5 Phase 3 — build Dataset650 + Dataset152 from unified RADHECK_{N}/cases/.

  Dataset650 — Test1 Tr/Va/Ts manifesto (RADCURE + HECKTOR train/val)
  Dataset152 — held-out HECKTOR test (manifest hecktor_excluded_case_folders)

Example:

  export TEST5_WORK_ROOT=/media/HDD_8TB/xisca/work/retrain_test5
  python -m pipelines.test5.build_datasets --dry-run
  python -m pipelines.test5.build_datasets --link hardlink
"""

from __future__ import annotations

import argparse
import os
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

from pipelines.hecktor.test_pipeline import (
    build_nnunet_dataset_from_processed,
    load_hecktor_test_case_allowlist,
)
from pipelines.test5.build_dataset650 import build_dataset650
from pipelines.test5.paths import (
    BUNDLED_SPLIT_MANIFEST,
    DEFAULT_RADCURE_DATASET366,
    DEFAULT_WORK_ROOT,
    resolve_cases_root,
    resolve_radheck,
    work_root as default_work_root,
)


def _resolve_organ_dict(work: Path, explicit: str) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    try:
        candidates.append(resolve_radheck(work) / "organ_dictionary_test5.json")
    except FileNotFoundError:
        pass
    candidates.extend(
        [
            work / "organ_dictionary_test5.json",
            work / "radcure_dictionary_test5.json",
        ]
    )
    for p in candidates:
        if p.is_file():
            return p.resolve()
    raise FileNotFoundError(
        "Organ dictionary not found. Run transform_cases first "
        "(writes organ_dictionary_test5.json under RADHECK_*)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test5: build Dataset650 + Dataset152 from RADHECK_{N}/cases"
    )
    parser.add_argument("--work-root", default=str(default_work_root()))
    parser.add_argument("--cases-root", default=os.getenv("TEST5_CASES_ROOT", ""))
    parser.add_argument(
        "--reference-dataset650",
        default=os.getenv(
            "TEST5_REFERENCE_DATASET650",
            f"{DEFAULT_WORK_ROOT}/Dataset650_TotalSegmentator",
        ),
    )
    parser.add_argument(
        "--split-manifest",
        default=os.getenv("TEST5_SPLIT_MANIFEST", ""),
    )
    parser.add_argument(
        "--radcure-dataset366",
        default=os.getenv(
            "TEST5_RADCURE_DATASET366",
            os.getenv("RADCURE_DATASET", DEFAULT_RADCURE_DATASET366),
        ),
    )
    parser.add_argument(
        "--organ-dictionary-path",
        default=os.getenv("ORGAN_DICTIONARY_PATH", ""),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--skip-650", action="store_true")
    parser.add_argument("--skip-152", action="store_true")
    parser.add_argument(
        "--no-reference-fallback",
        action="store_true",
        help="Do not fall back to old Dataset650 labels",
    )
    parser.add_argument(
        "--link",
        choices=("hardlink", "symlink", "copy"),
        default=os.getenv("TEST5_DATASET_LINK_MODE", "hardlink"),
    )
    parser.add_argument(
        "--train-all-except-ts",
        dest="train_all_except_ts",
        action="store_true",
        default=True,
        help="Keep fixed 74 Ts; train on all other ready cases (default)",
    )
    parser.add_argument(
        "--manifest-splits",
        dest="train_all_except_ts",
        action="store_false",
        help="Use original Test1 Tr≈361 / Va≈71 / Ts≈74",
    )
    parser.add_argument(
        "--no-hecktor-allowlist",
        action="store_true",
        help="Include every HECKTOR folder with output/ in Dataset152 (unsafe)",
    )
    args = parser.parse_args()

    work = Path(args.work_root).expanduser().resolve()
    cases_raw = (args.cases_root or "").strip()
    cases_arg = Path(cases_raw).expanduser() if cases_raw else None
    radheck, cases = resolve_cases_root(work, cases=cases_arg)
    organ = _resolve_organ_dict(work, args.organ_dictionary_path)

    man_path = (args.split_manifest or "").strip()
    if not man_path:
        work_man = work / "split_manifest.json"
        man_path = str(work_man if work_man.is_file() else BUNDLED_SPLIT_MANIFEST)

    reference = Path(args.reference_dataset650).expanduser()
    reference.mkdir(parents=True, exist_ok=True)
    reference = reference.resolve()
    rad366 = (
        Path(args.radcure_dataset366).expanduser() if args.radcure_dataset366 else None
    )

    print("=" * 70)
    print("Test5 Phase 3 — build Dataset650 + Dataset152")
    print(f"Work root:   {work}")
    print(f"RADHECK:     {radheck}")
    print(f"Cases:       {cases}")
    print(f"Organ dict:  {organ}")
    print(f"Manifest:    {man_path}")
    print(f"Link mode:   {args.link}")
    print(
        "Train mode:  "
        + (
            "all except fixed Ts"
            if args.train_all_except_ts
            else "manifest Tr/Va/Ts"
        )
    )
    print("=" * 70)

    if not args.skip_650:
        print("\n--- Dataset650 ---")
        build_dataset650(
            work_root=work,
            reference_dataset650=reference,
            organ_dictionary_path=organ,
            dataset_id="650",
            dry_run=args.dry_run,
            skip_missing=bool(args.skip_missing),
            allow_reference_fallback=not bool(args.no_reference_fallback),
            link_mode=str(args.link),
            split_manifest=Path(man_path),
            radcure_dataset366=rad366,
            cases_root=cases,
            train_all_except_ts=bool(args.train_all_except_ts),
        )

    if not args.skip_152:
        print("\n--- Dataset152 (HECKTOR held-out test) ---")
        ds152 = work / "Dataset152_TotalSegmentator"
        allowlist = None
        if not args.no_hecktor_allowlist:
            allowlist = load_hecktor_test_case_allowlist(manifest_path=man_path)
            if not allowlist:
                print(
                    "WARNING: empty HECKTOR test allowlist — "
                    "Dataset152 would be empty. Check split_manifest."
                )
        if args.dry_run:
            n = 0
            for name in sorted(os.listdir(cases)):
                if allowlist is not None and name not in allowlist:
                    continue
                out = cases / name / "output" / "image"
                if out.is_dir() and any(out.glob("*.nii.gz")):
                    n += 1
            print(f"Dry-run: would copy ≈{n} HECKTOR test case(s) → {ds152}")
        else:
            build_nnunet_dataset_from_processed(
                cases_root=str(cases),
                dataset_folder=str(ds152),
                dataset_id="152",
                organ_dictionary_path=str(organ),
                case_allowlist=allowlist,
            )
            print(f"Done: {ds152}")

    print("\nNext: train with NNUNET_TRAINER=nnUNetTrainer_700epochs_NoMirroring")
    print("  See pipelines/radheck/Retrain-Radheck-Test5.md")


if __name__ == "__main__":
    main()
