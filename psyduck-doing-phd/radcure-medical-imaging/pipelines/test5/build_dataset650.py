#!/usr/bin/env python3
"""
Test5 Phase 3 — build Dataset650 from improved-preprocess transforms.

Uses the **same Tr/Va/Ts membership** as Test1 (recovered ``split_manifest.json``).
If Dataset650 ``images*`` are missing, reconstructs stems from Dataset366 +
HECKTOR train/val lists (Ts > Tr > Va dedupe).

Preferred Phase 2 layout (from-scratch):

  ``${TEST5_WORK_ROOT}/RADHECK_{N}/cases/{case_id}/output/…``

Legacy layout still accepted: separate ``TotalSegmentatorRetrain/`` + ``hecktor/``.

Anatomy QC is **off** for the from-scratch path (no discards unless old QC logs exist).

Copy mode default is **hardlink** (falls back to copy cross-device) to save disk.

Example:

  export TEST5_WORK_ROOT=/media/HDD_8TB/xisca/work/retrain_test5
  export ORGAN_DICTIONARY_PATH=${TEST5_WORK_ROOT}/organ_dictionary_test5.json

  python -m pipelines.test5.build_datasets --link hardlink
  # or Dataset650 only:
  python -m pipelines.test5.build_dataset650 --link hardlink
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from image_processor.conventions import HECKTOR, RADCURE, get_nnunet_case_number
from pipelines.radheck.build_nnunet_dataset import write_dataset_json
from pipelines.radheck.nnunet_split_utils import (
    audit_split_overlaps,
    list_stems_in_split,
    print_audit,
    stem_from_image_filename,
)
from pipelines.test4.build_dataset650 import (
    _list_split_images,
    _radcure_case_id_from_stem,
)
from pipelines.test5.paths import (
    BUNDLED_SPLIT_MANIFEST,
    DEFAULT_RADCURE_DATASET366,
    DEFAULT_WORK_ROOT,
    resolve_cases_root,
    resolve_radheck,
)

DEFAULT_REFERENCE_DATASET650 = (
    "/media/HDD_8TB/xisca/work/retrain_test5/Dataset650_TotalSegmentator"
)


def _case_id_to_stem(case_id: str) -> str:
    if case_id.startswith("RADCURE-"):
        return f"case_{get_nnunet_case_number(case_id, RADCURE)}"
    return f"case_{get_nnunet_case_number(case_id, HECKTOR)}"


def _hecktor_ts_stem(case_id: str) -> str:
    """
    Unique stem for HECKTOR held-out test in Dataset650 imagesTs.

    Avoids collisions with RADCURE ``case_XXXX`` and between centers that share
    a numeric suffix (CHUM-017 vs HGJ-017).
    """
    return f"case_hek_{case_id.replace('-', '_')}"


def _cohort_for_case(case_id: str) -> str:
    return "radcure" if case_id.startswith("RADCURE-") else "hecktor"


def _list_ready_case_ids(cases_root: Path) -> List[str]:
    out: List[str] = []
    if not cases_root.is_dir():
        return out
    for p in sorted(cases_root.iterdir()):
        if not p.is_dir() or p.name.startswith("."):
            continue
        img = p / "output" / "image"
        if img.is_dir() and any(img.glob("*.nii.gz")):
            out.append(p.name)
    return out


def _max_train_membership(
    *,
    cases_root: Path,
    manifest: dict,
    ts_image_names: List[str],
    include_hecktor_test_in_ts: bool = True,
) -> Tuple[Dict[str, List[Tuple[str, str]]], Dict[str, int], Dict[str, dict]]:
    """
    Keep manifesto RADCURE Ts; put every other ready case into Tr; Va empty.

    When ``include_hecktor_test_in_ts`` (default), HECKTOR held-out test folders
    also go into Dataset650 ``imagesTs`` (unique ``case_hek_*`` stems) so one
    evaluate covers RADCURE + HECKTOR.

    Returns ({split: [(stem, case_id), ...]}, stats, ts_case_map).
    """
    ts_stems = {stem_from_image_filename(n) for n in ts_image_names}
    excluded = set(str(x) for x in (manifest.get("hecktor_excluded_case_folders") or []))

    tr_pairs: List[Tuple[str, str]] = []
    ts_pairs: List[Tuple[str, str]] = []
    ts_case_map: Dict[str, dict] = {}
    stem_owner: Dict[str, str] = {}
    stats = {
        "ready": 0,
        "to_tr": 0,
        "to_ts_radcure": 0,
        "to_ts_hecktor": 0,
        "to_ts": 0,
        "stem_collision_skipped": 0,
        "stem_blocked_by_ts": 0,
        "hecktor_test_missing_output": 0,
    }

    ready = set(_list_ready_case_ids(cases_root))

    for case_id in sorted(ready):
        stats["ready"] += 1
        stem = _case_id_to_stem(case_id)

        if case_id in excluded:
            if not include_hecktor_test_in_ts:
                continue
            hstem = _hecktor_ts_stem(case_id)
            if hstem in stem_owner:
                stats["stem_collision_skipped"] += 1
                print(
                    f"  WARNING: stem {hstem} already used by {stem_owner[hstem]}; "
                    f"skipping {case_id}"
                )
                continue
            stem_owner[hstem] = case_id
            ts_pairs.append((hstem, case_id))
            ts_case_map[hstem] = {
                "case_id": case_id,
                "cohort": "hecktor",
                "role": "held_out_test",
            }
            stats["to_ts_hecktor"] += 1
            stats["to_ts"] += 1
            continue

        if case_id.startswith("RADCURE-") and stem in ts_stems:
            ts_pairs.append((stem, case_id))
            ts_case_map[stem] = {
                "case_id": case_id,
                "cohort": "radcure",
                "role": "manifest_ts",
            }
            stats["to_ts_radcure"] += 1
            stats["to_ts"] += 1
            continue

        # Do not put HECKTOR (or extra RADCURE) into Tr if stem is reserved for Ts
        if stem in ts_stems:
            stats["stem_blocked_by_ts"] += 1
            continue

        if stem in stem_owner:
            stats["stem_collision_skipped"] += 1
            print(
                f"  WARNING: stem {stem} already used by {stem_owner[stem]}; "
                f"skipping {case_id}"
            )
            continue

        stem_owner[stem] = case_id
        tr_pairs.append((stem, case_id))
        stats["to_tr"] += 1

    # Ensure manifesto RADCURE Ts stems appear even if listing order differed
    have_ts = {s for s, _ in ts_pairs}
    for img in ts_image_names:
        stem = stem_from_image_filename(img)
        if stem in have_ts:
            continue
        case_id = _radcure_case_id_from_stem(stem)
        ts_pairs.append((stem, case_id))
        ts_case_map[stem] = {
            "case_id": case_id,
            "cohort": "radcure",
            "role": "manifest_ts",
        }
        stats["to_ts_radcure"] += 1
        stats["to_ts"] += 1

    if include_hecktor_test_in_ts:
        for case_id in sorted(excluded):
            if case_id in ready:
                continue
            stats["hecktor_test_missing_output"] += 1

    return (
        {
            "Tr": sorted(tr_pairs, key=lambda x: x[0]),
            "Va": [],
            "Ts": sorted(ts_pairs, key=lambda x: x[0]),
        },
        stats,
        ts_case_map,
    )


def _place_from_case_id(
    cases_root: Path,
    case_id: str,
    dst_images: str,
    dst_labels: str,
    mode: str,
    target_stem: Optional[str] = None,
) -> None:
    """Place image/label from unified case folder output/, optionally renaming stem."""
    out_i = cases_root / case_id / "output" / "image"
    out_l = cases_root / case_id / "output" / "labels"
    if not out_i.is_dir() or not out_l.is_dir():
        raise FileNotFoundError(f"Missing output for {case_id}")
    imgs = [f for f in os.listdir(out_i) if f.endswith(".nii.gz")]
    lbls = [f for f in os.listdir(out_l) if f.endswith(".nii.gz")]
    if not imgs or not lbls:
        raise FileNotFoundError(f"No nifti in {out_i} / {out_l}")
    src_img = out_i / imgs[0]
    src_lbl = out_l / lbls[0]
    if target_stem:
        img_name = f"{target_stem}_0000.nii.gz"
        lbl_name = f"{target_stem}.nii.gz"
    elif imgs[0].endswith("_0000.nii.gz"):
        base = imgs[0].replace("_0000.nii.gz", "")
        img_name = imgs[0]
        lbl_name = f"{base}.nii.gz"
    else:
        img_name = imgs[0]
        lbl_name = lbls[0]
    _place_pair(src_img, src_lbl, dst_images, dst_labels, img_name, lbl_name, mode)


def _hecktor_stem(case_id: str) -> str:
    return f"case_{get_nnunet_case_number(case_id, HECKTOR)}"


def _hecktor_stem_maps_by_split(manifest: dict) -> Dict[str, Dict[str, str]]:
    """
    Per-split stem → HECKTOR folder id.

    Train and val can share numeric suffixes (CHUM-017 vs HGJ-017 → case_017);
    a single global map would overwrite and pick the wrong center.
    """
    maps: Dict[str, Dict[str, str]] = {"Tr": {}, "Va": {}, "Ts": {}}
    for cid in manifest.get("hecktor_train_cases") or []:
        maps["Tr"][_hecktor_stem(str(cid))] = str(cid)
    for cid in manifest.get("hecktor_val_cases") or []:
        maps["Va"][_hecktor_stem(str(cid))] = str(cid)
    return maps


def _all_hecktor_stems(maps: Dict[str, Dict[str, str]]) -> Set[str]:
    stems: Set[str] = set()
    for m in maps.values():
        stems.update(m.keys())
    return stems


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    """
    Place ``src`` at ``dst``.

    mode:
      - hardlink (default): os.link, else copy2
      - symlink: os.symlink
      - copy: shutil.copy2
    """
    dst = Path(dst)
    src = Path(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "symlink":
        os.symlink(os.path.abspath(src), dst)
        return
    # hardlink
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _load_manifest(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _resolve_manifest(
    reference_dataset650: Path,
    split_manifest: Optional[Path] = None,
) -> Tuple[Path, dict, str]:
    """
    Return (reference_folder, manifest, manifest_path_used).

    Search order:
      1) explicit --split-manifest / TEST5_SPLIT_MANIFEST
      2) {reference}/split_manifest.json
      3) bundled Test1 recovery artifact
    """
    tried: List[str] = []

    if split_manifest is not None:
        sm = Path(split_manifest).expanduser()
        tried.append(str(sm))
        if sm.is_file():
            return reference_dataset650, _load_manifest(sm), str(sm)

    env_sm = os.getenv("TEST5_SPLIT_MANIFEST", "").strip()
    if env_sm:
        sm = Path(env_sm).expanduser()
        tried.append(str(sm))
        if sm.is_file():
            return reference_dataset650, _load_manifest(sm), str(sm)

    ref_man = reference_dataset650 / "split_manifest.json"
    tried.append(str(ref_man))
    if ref_man.is_file():
        return reference_dataset650, _load_manifest(ref_man), str(ref_man)

    tried.append(str(BUNDLED_SPLIT_MANIFEST))
    if BUNDLED_SPLIT_MANIFEST.is_file():
        print(
            "NOTE: no split_manifest next to reference Dataset650.\n"
            f"      Using bundled Test1 recovery: {BUNDLED_SPLIT_MANIFEST}"
        )
        return (
            reference_dataset650,
            _load_manifest(BUNDLED_SPLIT_MANIFEST),
            str(BUNDLED_SPLIT_MANIFEST),
        )

    raise FileNotFoundError(
        "split_manifest.json not found.\n"
        f"  Tried:\n    - "
        + "\n    - ".join(tried)
        + "\n\n"
        "Fix:\n"
        "  python -m pipelines.test5.restore_split_reference\n"
        "  # or copy experiments/artifacts/test1_dataset650_split_manifest.json\n"
    )


def _reference_has_split_images(reference_dataset650: Path) -> bool:
    for split in ("Tr", "Va", "Ts"):
        if _list_split_images(reference_dataset650, split):
            return True
    return False


def _reconstruct_split_stems(
    manifest: dict,
    radcure_dataset366: Optional[Path] = None,
) -> Dict[str, List[str]]:
    """
    Rebuild Tr/Va/Ts stems when Dataset650 images* were deleted.

    RADCURE membership comes from Dataset366; HECKTOR from manifest lists;
    then apply Ts > Tr > Va dedupe (same priority as Test1).
    """
    rad_path = radcure_dataset366
    if rad_path is None:
        env = os.getenv("TEST5_RADCURE_DATASET366", "").strip() or os.getenv(
            "RADCURE_DATASET", ""
        ).strip()
        if env:
            rad_path = Path(env)
        elif manifest.get("radcure_dataset"):
            rad_path = Path(manifest["radcure_dataset"])
        else:
            rad_path = Path(DEFAULT_RADCURE_DATASET366)
    rad_path = Path(rad_path).expanduser()
    if not rad_path.is_dir():
        raise FileNotFoundError(
            "Cannot reconstruct splits: Dataset366 not found at "
            f"{rad_path}. Set TEST5_RADCURE_DATASET366."
        )

    tr = set(list_stems_in_split(str(rad_path), "Tr"))
    va = set(list_stems_in_split(str(rad_path), "Va"))
    ts = set(list_stems_in_split(str(rad_path), "Ts"))
    print(
        f"Dataset366 stems: Tr={len(tr)} Va={len(va)} Ts={len(ts)}  ({rad_path})"
    )

    for cid in manifest.get("hecktor_train_cases") or []:
        tr.add(_hecktor_stem(str(cid)))
    for cid in manifest.get("hecktor_val_cases") or []:
        va.add(_hecktor_stem(str(cid)))

    # Ts > Tr > Va
    tr -= ts
    va -= ts
    va -= tr

    stems = {
        "Tr": sorted(tr),
        "Va": sorted(va),
        "Ts": sorted(ts),
    }
    expected = manifest.get("split_counts_after_dedupe") or {}
    print(
        "Reconstructed split stems:",
        {k: len(v) for k, v in stems.items()},
        "| expected:",
        expected,
    )
    for k, exp in expected.items():
        got = len(stems.get(k, []))
        if int(exp) != got:
            print(
                f"  WARNING: {k} count {got} != expected {exp} "
                "(check Dataset366 path / manifest HECKTOR lists)"
            )
    return stems


def _split_image_names(
    reference_dataset650: Path,
    manifest: dict,
    radcure_dataset366: Optional[Path] = None,
) -> Tuple[Dict[str, List[str]], str]:
    """
    Return ({split: [image filenames]}, source_tag).

    Prefer existing imagesTr/Va/Ts on the reference Dataset650; otherwise
    reconstruct from Dataset366 + recovered Test1 manifest.
    """
    if _reference_has_split_images(reference_dataset650):
        out = {
            s: _list_split_images(reference_dataset650, s) for s in ("Tr", "Va", "Ts")
        }
        return out, "reference_images"

    stems = _reconstruct_split_stems(manifest, radcure_dataset366=radcure_dataset366)
    out = {
        s: [f"{stem}_0000.nii.gz" for stem in stems[s]] for s in ("Tr", "Va", "Ts")
    }
    return out, "reconstructed_from_dataset366_and_manifest"


def _load_qc_discarded_case_ids(work_root: Path) -> Set[str]:
    discarded: Set[str] = set()
    jsonl = work_root / "logs" / "anatomy_qc" / "anatomy_qc_decisions.jsonl"
    if jsonl.is_file():
        with open(jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("decision") == "discard" or row.get("keep") is False:
                    cid = row.get("case_id")
                    if cid:
                        discarded.add(str(cid))
    csv_path = work_root / "anatomy_qc_discarded.csv"
    if csv_path.is_file():
        import csv

        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cid = row.get("case_id")
                if cid:
                    discarded.add(str(cid))
    return discarded


def _discarded_stems(discarded_case_ids: Set[str], hecktor_by_stem: Dict[str, str]) -> Set[str]:
    stems: Set[str] = set()
    hecktor_id_to_stem = {v: k for k, v in hecktor_by_stem.items()}
    for case_id in discarded_case_ids:
        if case_id.startswith("RADCURE-"):
            stems.add(f"case_{case_id.replace('RADCURE-', '')}")
        elif case_id in hecktor_id_to_stem:
            stems.add(hecktor_id_to_stem[case_id])
        else:
            stems.add(_hecktor_stem(case_id))
    return stems


def _place_pair(
    src_img: Path,
    src_lbl: Path,
    dst_images: str,
    dst_labels: str,
    img_name: str,
    lbl_name: str,
    mode: str,
) -> None:
    _link_or_copy(src_img, Path(dst_images) / img_name, mode)
    _link_or_copy(src_lbl, Path(dst_labels) / lbl_name, mode)


def _place_from_dataset650(
    dataset_folder: Path,
    split: str,
    stem: str,
    dst_images: str,
    dst_labels: str,
    mode: str,
) -> None:
    img_name = f"{stem}_0000.nii.gz"
    lbl_name = f"{stem}.nii.gz"
    src_img = dataset_folder / f"images{split}" / img_name
    src_lbl = dataset_folder / f"labels{split}" / lbl_name
    if not src_img.is_file():
        matches = sorted((dataset_folder / f"images{split}").glob(f"{stem}*.nii.gz"))
        if not matches:
            raise FileNotFoundError(f"Missing {src_img}")
        src_img = matches[0]
        img_name = src_img.name
    if not src_lbl.is_file():
        raise FileNotFoundError(f"Missing {src_lbl}")
    _place_pair(src_img, src_lbl, dst_images, dst_labels, img_name, lbl_name, mode)


def _place_radcure_relabel(
    radcure_retrain: Path,
    case_id: str,
    dst_images: str,
    dst_labels: str,
    mode: str,
) -> None:
    out_i = radcure_retrain / case_id / "output" / "image"
    out_l = radcure_retrain / case_id / "output" / "labels"
    if not out_i.is_dir() or not out_l.is_dir():
        raise FileNotFoundError(f"Missing relabeled output for {case_id}: {out_i}")
    imgs = [f for f in os.listdir(out_i) if f.endswith(".nii.gz")]
    lbls = [f for f in os.listdir(out_l) if f.endswith(".nii.gz")]
    if not imgs or not lbls:
        raise FileNotFoundError(f"No nifti in {out_i} / {out_l}")
    src_img = out_i / imgs[0]
    src_lbl = out_l / lbls[0]
    if imgs[0].endswith("_0000.nii.gz"):
        base = imgs[0].replace("_0000.nii.gz", "")
    else:
        base = imgs[0].replace(".nii.gz", "")
    _place_pair(
        src_img,
        src_lbl,
        dst_images,
        dst_labels,
        imgs[0],
        f"{base}.nii.gz",
        mode,
    )


def _place_hecktor_relabel(
    cases_root: Path,
    case_id: str,
    dst_images: str,
    dst_labels: str,
    mode: str,
) -> None:
    out_i = cases_root / case_id / "output" / "image"
    out_l = cases_root / case_id / "output" / "labels"
    if not out_i.is_dir() or not out_l.is_dir():
        raise FileNotFoundError(f"Missing HECKTOR output for {case_id}")
    imgs = [f for f in os.listdir(out_i) if f.endswith(".nii.gz")]
    lbls = [f for f in os.listdir(out_l) if f.endswith(".nii.gz")]
    if not imgs or not lbls:
        raise FileNotFoundError(f"No nifti in {out_i} / {out_l}")
    src_img = out_i / imgs[0]
    src_lbl = out_l / lbls[0]
    if imgs[0].endswith("_0000.nii.gz"):
        base = imgs[0].replace("_0000.nii.gz", "")
        img_name = imgs[0]
        lbl_name = f"{base}.nii.gz"
    else:
        img_name = imgs[0]
        lbl_name = lbls[0]
    _place_pair(src_img, src_lbl, dst_images, dst_labels, img_name, lbl_name, mode)


def _try_place_case(
    *,
    split: str,
    stem: str,
    hecktor_by_split: Dict[str, Dict[str, str]],
    test5_radcure: Path,
    test5_hecktor: Path,
    test5_cases_root: Optional[Path],
    test4_radcure: Optional[Path],
    test4_hecktor: Optional[Path],
    test4_dataset650: Optional[Path],
    reference_dataset650: Path,
    dst_img: str,
    dst_lbl: str,
    allow_reference_fallback: bool,
    link_mode: str,
) -> str:
    hecktor_cid = (hecktor_by_split.get(split) or {}).get(stem)
    errors: List[str] = []

    if hecktor_cid is not None:
        if test5_cases_root is not None:
            try:
                _place_hecktor_relabel(
                    test5_cases_root, hecktor_cid, dst_img, dst_lbl, link_mode
                )
                return "test5_radheck"
            except (FileNotFoundError, OSError) as exc:
                errors.append(f"test5_radheck:{exc}")
        try:
            _place_hecktor_relabel(
                test5_hecktor, hecktor_cid, dst_img, dst_lbl, link_mode
            )
            return "test5_relabel"
        except (FileNotFoundError, OSError) as exc:
            errors.append(f"test5:{exc}")
        if test4_hecktor is not None:
            try:
                _place_hecktor_relabel(
                    test4_hecktor, hecktor_cid, dst_img, dst_lbl, link_mode
                )
                return "test4_relabel"
            except (FileNotFoundError, OSError) as exc:
                errors.append(f"test4_relabel:{exc}")
    else:
        case_id = _radcure_case_id_from_stem(stem)
        if test5_cases_root is not None:
            try:
                _place_radcure_relabel(
                    test5_cases_root, case_id, dst_img, dst_lbl, link_mode
                )
                return "test5_radheck"
            except (FileNotFoundError, OSError) as exc:
                errors.append(f"test5_radheck:{exc}")
        try:
            _place_radcure_relabel(test5_radcure, case_id, dst_img, dst_lbl, link_mode)
            return "test5_relabel"
        except (FileNotFoundError, OSError) as exc:
            errors.append(f"test5:{exc}")
        if test4_radcure is not None:
            try:
                _place_radcure_relabel(
                    test4_radcure, case_id, dst_img, dst_lbl, link_mode
                )
                return "test4_relabel"
            except (FileNotFoundError, OSError) as exc:
                errors.append(f"test4_relabel:{exc}")

    if test4_dataset650 is not None and test4_dataset650.is_dir():
        try:
            _place_from_dataset650(
                test4_dataset650, split, stem, dst_img, dst_lbl, link_mode
            )
            return "test4_dataset650"
        except (FileNotFoundError, OSError) as exc:
            errors.append(f"test4_ds:{exc}")

    if allow_reference_fallback:
        try:
            _place_from_dataset650(
                reference_dataset650, split, stem, dst_img, dst_lbl, link_mode
            )
            return "reference_dataset650"
        except (FileNotFoundError, OSError) as exc:
            errors.append(f"reference:{exc}")

    raise FileNotFoundError(" | ".join(errors) if errors else f"no source for {stem}")


def build_dataset650(
    work_root: Path,
    reference_dataset650: Path,
    organ_dictionary_path: Path,
    dataset_id: str = "650",
    dry_run: bool = False,
    hecktor_output_root: Optional[Path] = None,
    skip_missing: bool = False,
    test4_work_root: Optional[Path] = None,
    allow_reference_fallback: bool = True,
    link_mode: str = "hardlink",
    split_manifest: Optional[Path] = None,
    radcure_dataset366: Optional[Path] = None,
    cases_root: Optional[Path] = None,
    train_all_except_ts: bool = True,
    include_hecktor_test_in_ts: bool = True,
    ts_only: bool = False,
) -> Path:
    reference_dataset650, manifest, manifest_used = _resolve_manifest(
        reference_dataset650, split_manifest=split_manifest
    )
    print(f"Split manifest: {manifest_used}")
    hecktor_by_split = _hecktor_stem_maps_by_split(manifest)
    hecktor_by_stem = {**hecktor_by_split["Tr"], **hecktor_by_split["Va"]}
    discarded_ids = _load_qc_discarded_case_ids(work_root)
    discarded_stems = _discarded_stems(discarded_ids, hecktor_by_stem)

    radcure_retrain = work_root / "TotalSegmentatorRetrain"
    hecktor_root = (
        Path(hecktor_output_root) if hecktor_output_root else (work_root / "hecktor")
    )
    unified_cases: Optional[Path] = None
    if cases_root is not None:
        unified_cases = Path(cases_root).expanduser().resolve()
    else:
        try:
            _, unified_cases = resolve_cases_root(work_root)
        except FileNotFoundError:
            unified_cases = None

    if train_all_except_ts and unified_cases is None:
        raise FileNotFoundError(
            "--train-all-except-ts requires unified RADHECK_{N}/cases/. "
            "Run transform_cases first, or pass --cases-root."
        )

    dataset_name = f"Dataset{dataset_id}_TotalSegmentator"
    dataset_folder = work_root / dataset_name

    if test4_work_root is not None and not test4_work_root.exists():
        print(f"NOTE: Test4 work root missing (skipped): {test4_work_root}")
        test4_work_root = None
    test4_hecktor = (test4_work_root / "hecktor") if test4_work_root else None
    test4_radcure = (
        (test4_work_root / "TotalSegmentatorRetrain") if test4_work_root else None
    )
    test4_dataset650 = None
    if test4_work_root is not None:
        cand = test4_work_root / "Dataset650_TotalSegmentator"
        if cand.is_dir() and (cand / "imagesTr").is_dir():
            test4_dataset650 = cand

    if unified_cases is not None and unified_cases.is_dir():
        n_unified = sum(
            1
            for p in unified_cases.glob("*/output/image")
            if p.is_dir() and any(p.glob("*.nii.gz"))
        )
        print(f"Test5 unified cases: {n_unified} ready under {unified_cases}")
        n_hecktor_ready = n_unified
        n_radcure_ready = n_unified
    else:
        n_hecktor_ready = sum(
            1
            for p in hecktor_root.glob("*/output/image")
            if p.is_dir() and any(p.glob("*.nii.gz"))
        )
        n_radcure_ready = sum(
            1
            for p in radcure_retrain.glob("*/output/image")
            if p.is_dir() and any(p.glob("*.nii.gz"))
        )
        print(f"Test5 outputs: RADCURE={n_radcure_ready}  HECKTOR={n_hecktor_ready}")
        print(f"  RADCURE root: {radcure_retrain}")
        print(f"  HECKTOR root: {hecktor_root}")
    print(f"  Link mode:    {link_mode}")
    print(
        "  Train mode:   "
        + (
            "ALL except fixed Ts"
            if train_all_except_ts
            else "manifest Tr/Va/Ts"
        )
    )
    print(
        "  Test set:     "
        + (
            "RADCURE Ts + HECKTOR held-out (unified imagesTs)"
            if include_hecktor_test_in_ts
            else "RADCURE Ts only (HECKTOR → Dataset152)"
        )
    )
    if ts_only:
        print("  Mode:         refresh imagesTs/labelsTs only (keep Tr)")
    if test4_work_root:
        print(f"Test4 fallback root: {test4_work_root}")
        print(f"  Test4 Dataset650: {test4_dataset650 or '(not found)'}")

    # Resolve membership before possibly wiping dataset_folder (== reference).
    split_images, split_source = _split_image_names(
        reference_dataset650,
        manifest,
        radcure_dataset366=radcure_dataset366,
    )
    print(f"Split stem source: {split_source}")

    max_train_pairs: Optional[Dict[str, List[Tuple[str, str]]]] = None
    max_train_stats: Optional[Dict[str, int]] = None
    ts_case_map: Dict[str, dict] = {}
    if train_all_except_ts:
        assert unified_cases is not None
        max_train_pairs, max_train_stats, ts_case_map = _max_train_membership(
            cases_root=unified_cases,
            manifest=manifest,
            ts_image_names=split_images.get("Ts", []),
            include_hecktor_test_in_ts=include_hecktor_test_in_ts,
        )
        split_source = "train_all_except_fixed_ts_unified_test"
        print("Max-train membership:", max_train_stats)
        print(
            "  Planned:",
            {k: len(v) for k, v in max_train_pairs.items()},
            "| fixed RADCURE Ts from manifesto:",
            len(split_images.get("Ts", [])),
        )

    ref_has_files = _reference_has_split_images(reference_dataset650)
    if allow_reference_fallback and (
        not ref_has_files
        or reference_dataset650.resolve() == dataset_folder.resolve()
    ):
        print(
            "NOTE: disabling reference Dataset650 label fallback "
            "(empty or same as output folder)."
        )
        allow_reference_fallback = False

    if n_hecktor_ready < 50 and allow_reference_fallback:
        print(
            "\nNOTE: Few HECKTOR Test5 outputs yet.\n"
            "  Missing HECKTOR stems will use reference Dataset650 labels\n"
            "  (not improved bg) until Phase 2 finishes more cases.\n"
        )

    if dataset_folder.is_dir() and not dry_run:
        if ts_only:
            if not (dataset_folder / "imagesTr").is_dir():
                raise FileNotFoundError(
                    f"--ts-only requires existing Dataset650 with imagesTr: {dataset_folder}"
                )
            for split_name in ("imagesTs", "labelsTs"):
                p = dataset_folder / split_name
                if p.is_dir():
                    shutil.rmtree(p)
            print(f"Cleared imagesTs/labelsTs under {dataset_folder} (kept Tr)")
        else:
            print(f"Removing existing {dataset_folder}")
            shutil.rmtree(dataset_folder)

    counts = {"Tr": 0, "Va": 0, "Ts": 0}
    source_counts: Dict[str, int] = {}
    skipped_qc: List[str] = []
    missing: List[str] = []

    print(f"QC discarded case IDs: {len(discarded_ids)}")
    if discarded_ids:
        print("  e.g.", sorted(discarded_ids)[:8])

    if train_all_except_ts:
        assert max_train_pairs is not None and unified_cases is not None
        splits_to_build = ("Ts",) if ts_only else ("Tr", "Va", "Ts")
        for split in splits_to_build:
            pairs = max_train_pairs.get(split, [])
            print(f"\n{split}: {len(pairs)} cases")
            if dry_run:
                counts[split] = len(pairs)
                continue
            dst_img = dataset_folder / f"images{split}"
            dst_lbl = dataset_folder / f"labels{split}"
            dst_img.mkdir(parents=True, exist_ok=True)
            dst_lbl.mkdir(parents=True, exist_ok=True)
            for stem, case_id in pairs:
                if stem in discarded_stems:
                    skipped_qc.append(f"{split}/{stem}")
                    continue
                try:
                    _place_from_case_id(
                        unified_cases,
                        case_id,
                        str(dst_img),
                        str(dst_lbl),
                        link_mode,
                        target_stem=stem,
                    )
                    source_counts["test5_radheck"] = (
                        source_counts.get("test5_radheck", 0) + 1
                    )
                    counts[split] += 1
                except (FileNotFoundError, OSError) as exc:
                    missing.append(f"{split}/{stem}/{case_id}: {exc}")
        if ts_only and not dry_run and (dataset_folder / "imagesTr").is_dir():
            counts["Tr"] = len(list((dataset_folder / "imagesTr").glob("*.nii.gz")))
            counts["Va"] = (
                len(list((dataset_folder / "imagesVa").glob("*.nii.gz")))
                if (dataset_folder / "imagesVa").is_dir()
                else 0
            )
    else:
        for split in ("Tr", "Va", "Ts"):
            ref_images = split_images.get(split, [])
            print(f"\n{split}: {len(ref_images)} reference cases")
            if dry_run:
                n_keep = sum(
                    1
                    for img in ref_images
                    if stem_from_image_filename(img) not in discarded_stems
                )
                counts[split] = n_keep
                print(f"  dry-run keep≈{n_keep} (excluding QC discards)")
                continue

            dst_img = dataset_folder / f"images{split}"
            dst_lbl = dataset_folder / f"labels{split}"
            dst_img.mkdir(parents=True, exist_ok=True)
            dst_lbl.mkdir(parents=True, exist_ok=True)

            for img_file in ref_images:
                stem = stem_from_image_filename(img_file)
                if stem in discarded_stems:
                    skipped_qc.append(f"{split}/{stem}")
                    continue
                try:
                    src_tag = _try_place_case(
                        split=split,
                        stem=stem,
                        hecktor_by_split=hecktor_by_split,
                        test5_radcure=radcure_retrain,
                        test5_hecktor=hecktor_root,
                        test5_cases_root=unified_cases,
                        test4_radcure=test4_radcure,
                        test4_hecktor=test4_hecktor,
                        test4_dataset650=test4_dataset650,
                        reference_dataset650=reference_dataset650,
                        dst_img=str(dst_img),
                        dst_lbl=str(dst_lbl),
                        allow_reference_fallback=allow_reference_fallback,
                        link_mode=link_mode,
                    )
                    source_counts[src_tag] = source_counts.get(src_tag, 0) + 1
                    counts[split] += 1
                except (FileNotFoundError, OSError) as exc:
                    missing.append(f"{split}/{stem}: {exc}")

    print("\nCopy sources:", dict(sorted(source_counts.items())))

    if missing and not skip_missing:
        hek_stems = _all_hecktor_stems(hecktor_by_split)
        n_hek = n_rad = 0
        for m in missing:
            left = m.split(":", 1)[0]
            stem = left.split("/", 1)[-1] if "/" in left else left
            # max-train format: split/stem/case_id
            parts = left.split("/")
            stem = parts[1] if len(parts) >= 2 else stem
            if stem in hek_stems or (
                len(parts) >= 3 and not parts[2].startswith("RADCURE-")
            ):
                n_hek += 1
            else:
                n_rad += 1
        n_manifest_hek = len(hecktor_by_split["Tr"]) + len(hecktor_by_split["Va"])
        raise RuntimeError(
            f"{len(missing)} case(s) missing from Test5 / optional Test4 / reference.\n"
            f"  HECKTOR missing: {n_hek}  |  RADCURE missing: {n_rad}\n"
            f"  Test5 HECKTOR outputs ready: {n_hecktor_ready}  "
            f"(manifest train+val ≈ {n_manifest_hek})\n"
            "\n"
            "This usually means Phase 2 has not transformed all HECKTOR train/val cases yet.\n"
            "Options:\n"
            "  1) Finish Phase 2 with BOTH HECKTOR sources (training task1 + test1):\n"
            f"       export TEST5_HECKTOR_TRAIN_SOURCE="
            f"{manifest.get('hecktor_cases_root', '.../HECKTOR2025_task1_training/unzipped/task1')}\n"
            "       export TEST5_HECKTOR_TEST_SOURCE="
            ".../hecktor/test1/unzipped/test1\n"
            "       python -m pipelines.test5.transform_cases\n"
            "  2) Build with what you have (drops missing stems):\n"
            "       python -m pipelines.test5.build_datasets --link hardlink --skip-missing\n"
            "\nFirst 10:\n"
            + "\n".join(missing[:10])
        )

    if missing and skip_missing:
        print(f"\nWARNING: --skip-missing: dropping {len(missing)} stems")

    if dry_run:
        print("\nDry run — no files written.")
        print(f"Would skip QC: {len(skipped_qc)}")
        if max_train_stats:
            print(f"Max-train stats: {max_train_stats}")
        return dataset_folder

    print(f"\nSkipped (QC): {len(skipped_qc)}")
    if missing and skip_missing:
        print(f"Skipped (missing): {len(missing)}")
    audit = audit_split_overlaps(str(dataset_folder))
    if any(audit["overlaps"][k] for k in audit["overlaps"]):
        print("WARNING: split overlaps detected:")
        print_audit(audit)

    n_tr = len(list((dataset_folder / "imagesTr").glob("*.nii.gz")))
    write_dataset_json(
        str(dataset_folder),
        dataset_name,
        str(organ_dictionary_path),
        num_training=n_tr,
    )

    out_manifest = {
        **manifest,
        "test5_work_root": str(work_root),
        "reference_dataset650": str(reference_dataset650),
        "organ_dictionary_path": str(organ_dictionary_path),
        "dataset_folder": str(dataset_folder),
        "dataset_id": dataset_id,
        "link_mode": link_mode,
        "split_source": split_source,
        "split_manifest_used": manifest_used,
        "train_all_except_ts": train_all_except_ts,
        "include_hecktor_test_in_ts": include_hecktor_test_in_ts,
        "ts_only_refresh": ts_only,
        "max_train_stats": max_train_stats,
        "ts_case_map": ts_case_map,
        "label_source": (
            "Prefer Test5 improved transform; optional Test4 if present; "
            "else reference Dataset650"
        ),
        "hecktor_output_root": str(hecktor_root),
        "unified_cases_root": str(unified_cases) if unified_cases else None,
        "test4_work_root": str(test4_work_root) if test4_work_root else None,
        "copy_source_counts": source_counts,
        "anatomy_qc_discarded_case_ids": sorted(discarded_ids),
        "anatomy_qc_skipped_stems": skipped_qc,
        "missing_skipped": missing if skip_missing else [],
        "counts_built": counts,
        "background_mode": "improved_where_test5_relabel_exists",
        "anatomy_qc": False,
        "fixed_ts_note": (
            "Ts = Test1 RADCURE 74 + HECKTOR held-out (case_hek_* stems); "
            "Tr = all other ready RADHECK cases. Single evaluate on imagesTs."
            if train_all_except_ts and include_hecktor_test_in_ts
            else (
                "Ts kept identical to Test1 manifesto for RADCURE comparison; "
                "Tr expanded; HECKTOR test left for Dataset152."
                if train_all_except_ts
                else None
            )
        ),
    }
    man_path = dataset_folder / "split_manifest.json"
    with open(man_path, "w") as f:
        json.dump(out_manifest, f, indent=2)
    print(f"\nWrote {man_path}")
    if ts_case_map and not dry_run:
        map_path = dataset_folder / "ts_case_map.json"
        with open(map_path, "w") as f:
            json.dump(ts_case_map, f, indent=2)
            f.write("\n")
        print(f"Wrote {map_path} ({len(ts_case_map)} test cases)")
    print(f"Counts: {counts}")
    print(f"Done: {dataset_folder}")
    return dataset_folder

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Test5 Phase 3: build Dataset650 (Test2/3 splits − QC). "
            "Default --link hardlinks files to save disk."
        )
    )
    parser.add_argument(
        "--work-root",
        default=os.getenv("TEST5_WORK_ROOT", DEFAULT_WORK_ROOT),
    )
    parser.add_argument(
        "--cases-root",
        default=os.getenv("TEST5_CASES_ROOT", ""),
        help="Unified RADHECK_{N}/cases (auto-detected via RADHECK_CURRENT if empty)",
    )
    parser.add_argument(
        "--reference-dataset650",
        default=os.getenv(
            "TEST5_REFERENCE_DATASET650",
            DEFAULT_REFERENCE_DATASET650,
        ),
    )
    parser.add_argument(
        "--split-manifest",
        default=os.getenv("TEST5_SPLIT_MANIFEST", ""),
        help="Recovered Test1 split_manifest.json (default: bundled artifact)",
    )
    parser.add_argument(
        "--radcure-dataset366",
        default=os.getenv(
            "TEST5_RADCURE_DATASET366",
            os.getenv("RADCURE_DATASET", DEFAULT_RADCURE_DATASET366),
        ),
        help="Dataset366 used to reconstruct RADCURE Tr/Va/Ts when images* missing",
    )
    parser.add_argument(
        "--organ-dictionary-path",
        default=os.getenv("ORGAN_DICTIONARY_PATH", ""),
    )
    parser.add_argument(
        "--test4-work-root",
        default=os.getenv("TEST4_WORK_ROOT", ""),
        help="Optional fallback if still on disk (env: TEST4_WORK_ROOT); empty = skip",
    )
    parser.add_argument("--dataset-id", default="650")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--hecktor-output-root",
        default=os.getenv("TEST5_HECKTOR_OUTPUT_ROOT", ""),
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Drop stems missing from all available sources",
    )
    parser.add_argument(
        "--no-reference-fallback",
        action="store_true",
        help="Do not use Test2/Test3 Dataset650 labels",
    )
    parser.add_argument(
        "--link",
        choices=("hardlink", "symlink", "copy"),
        default=os.getenv("TEST5_DATASET_LINK_MODE", "hardlink"),
        help="How to place NIfTIs into Dataset650 (default: hardlink)",
    )
    parser.add_argument(
        "--train-all-except-ts",
        dest="train_all_except_ts",
        action="store_true",
        default=True,
        help=(
            "Default: keep manifesto Ts (74); put all other ready cases into Tr; "
            "Va empty"
        ),
    )
    parser.add_argument(
        "--manifest-splits",
        dest="train_all_except_ts",
        action="store_false",
        help="Use original Test1 Tr/Va/Ts membership (Tr≈361) instead of max-train",
    )
    parser.add_argument(
        "--include-hecktor-test-in-ts",
        dest="include_hecktor_test_in_ts",
        action="store_true",
        default=True,
        help="Put HECKTOR held-out test into Dataset650 imagesTs (default)",
    )
    parser.add_argument(
        "--hecktor-test-to-152",
        dest="include_hecktor_test_in_ts",
        action="store_false",
        help="Legacy: leave HECKTOR test out of 650 (use Dataset152 instead)",
    )
    parser.add_argument(
        "--ts-only",
        action="store_true",
        help="Only refresh imagesTs/labelsTs (keep existing imagesTr)",
    )
    args = parser.parse_args()

    work_root = Path(args.work_root).resolve()
    if not args.reference_dataset650:
        print("ERROR: set TEST5_REFERENCE_DATASET650")
        sys.exit(1)

    reference = Path(args.reference_dataset650).expanduser()
    reference.mkdir(parents=True, exist_ok=True)
    reference = reference.resolve()
    organ_candidates = []
    if args.organ_dictionary_path:
        organ_candidates.append(Path(args.organ_dictionary_path))
    organ_candidates.extend(
        [
            work_root / "organ_dictionary_test5.json",
            work_root / "radcure_dictionary_test5.json",
        ]
    )
    try:
        radheck_guess = resolve_radheck(work_root)
        organ_candidates.insert(1, radheck_guess / "organ_dictionary_test5.json")
    except FileNotFoundError:
        pass
    organ_dict = next((p.resolve() for p in organ_candidates if p.is_file()), None)
    if organ_dict is None:
        raise FileNotFoundError(
            "Organ dictionary not found. Expected organ_dictionary_test5.json "
            "under work root or RADHECK_* after transform_cases."
        )

    hecktor_out = (
        Path(args.hecktor_output_root).resolve() if args.hecktor_output_root else None
    )
    t4_raw = (args.test4_work_root or "").strip()
    t4 = Path(t4_raw) if t4_raw else None
    sm_raw = (args.split_manifest or "").strip()
    sm = Path(sm_raw) if sm_raw else None
    rad366 = Path(args.radcure_dataset366).expanduser() if args.radcure_dataset366 else None
    cases_raw = (args.cases_root or "").strip()
    cases = Path(cases_raw).expanduser() if cases_raw else None

    print("=" * 70)
    print("Test5 Phase 3 — build Dataset650")
    print(f"Work root:     {work_root}")
    print(f"Reference:     {reference}")
    print(f"Organ dict:    {organ_dict}")
    print(f"Cases root:    {cases or '(auto RADHECK_*/cases)'}")
    print(f"Split manif.:  {sm or '(auto/bundled)'}")
    print(f"Dataset366:    {rad366}")
    print(f"Test4 root:    {t4 or '(none)'}")
    print(f"Link mode:     {args.link}")
    print(
        f"Train mode:    "
        f"{'all except fixed Ts' if args.train_all_except_ts else 'manifest splits'}"
    )
    print(
        f"Test set:      "
        f"{'unified RADCURE+HECKTOR Ts' if args.include_hecktor_test_in_ts else 'RADCURE Ts only'}"
    )
    if args.ts_only:
        print("Refresh:       Ts only")
    print("=" * 70)

    build_dataset650(
        work_root=work_root,
        reference_dataset650=reference,
        organ_dictionary_path=organ_dict,
        dataset_id=str(args.dataset_id),
        dry_run=args.dry_run,
        hecktor_output_root=hecktor_out,
        skip_missing=bool(args.skip_missing),
        test4_work_root=t4,
        allow_reference_fallback=not bool(args.no_reference_fallback),
        link_mode=str(args.link),
        split_manifest=sm,
        radcure_dataset366=rad366,
        cases_root=cases,
        train_all_except_ts=bool(args.train_all_except_ts),
        include_hecktor_test_in_ts=bool(args.include_hecktor_test_in_ts),
        ts_only=bool(args.ts_only),
    )


if __name__ == "__main__":
    main()
