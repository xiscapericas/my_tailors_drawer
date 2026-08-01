"""Test5 path defaults and workspace layout helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_WORK_ROOT = "/media/HDD_8TB/xisca/work/retrain_test5"
DEFAULT_RADCURE_SOURCE = (
    "/media/HDD_8TB/xisca/dataset/RadcureComplete/TotalSegmentatorRetrain"
)
# Held-out HECKTOR test (Dataset152) — what the user listed as "test1"
DEFAULT_HECKTOR_TEST_SOURCE = (
    "/media/HDD_8TB/xisca/dataset/hecktor/test1/unzipped/test1"
)
# HECKTOR train/val pool used by Test1 Dataset650 (manifest hecktor_cases_root)
DEFAULT_HECKTOR_TRAIN_SOURCE = (
    "/media/HDD_8TB/xisca/dataset/hecktor/HECKTOR2025_task1_training/unzipped/task1"
)
# Back-compat alias (often pointed at test1 only — incomplete for Dataset650)
DEFAULT_HECKTOR_SOURCE = DEFAULT_HECKTOR_TEST_SOURCE

DEFAULT_RADCURE_DATASET366 = (
    "/media/HDD_8TB/xisca/work/nnunet_retrain_radcure366/Dataset366_TotalSegmentator"
)
BUNDLED_SPLIT_MANIFEST = (
    _REPO_ROOT / "experiments" / "artifacts" / "test1_dataset650_split_manifest.json"
)

# Legacy clutter under TEST5_WORK_ROOT to remove on clean restart
LEGACY_WORK_ENTRIES = (
    "TotalSegmentatorRetrain",
    "hecktor",
    "Dataset650_TotalSegmentator",
    "Dataset152_TotalSegmentator",
    "nnunet_retrain",
    "logs",
    "relabel_ok.txt",
    "relabel_failed.txt",
    "anatomy_qc_discarded.csv",
    "radcure_dictionary_test5.json",
    "RADHECK_CURRENT",
)


def work_root() -> Path:
    return Path(os.getenv("TEST5_WORK_ROOT", DEFAULT_WORK_ROOT)).expanduser()


def radheck_dir(work: Path, n_cases: int) -> Path:
    """Unified cohort folder: RADHECK_{n}/cases/…"""
    return work / f"RADHECK_{n_cases}"


def cases_root(radheck: Path) -> Path:
    return radheck / "cases"


def default_hecktor_sources() -> List[Path]:
    """
    HECKTOR roots to scan (train/val + held-out test).

    Env:
      TEST5_HECKTOR_SOURCES — colon-separated list (preferred)
      else TEST5_HECKTOR_TRAIN_SOURCE + TEST5_HECKTOR_TEST_SOURCE /
           TEST5_HECKTOR_SOURCE / TEST5_HECKTOR_SOURCE_CASES_ROOT
    """
    multi = os.getenv("TEST5_HECKTOR_SOURCES", "").strip()
    if multi:
        return [Path(p).expanduser() for p in multi.split(":") if p.strip()]

    out: List[Path] = []
    train = os.getenv("TEST5_HECKTOR_TRAIN_SOURCE", DEFAULT_HECKTOR_TRAIN_SOURCE).strip()
    test = (
        os.getenv("TEST5_HECKTOR_TEST_SOURCE", "").strip()
        or os.getenv("TEST5_HECKTOR_SOURCE", "").strip()
        or os.getenv("TEST5_HECKTOR_SOURCE_CASES_ROOT", "").strip()
        or DEFAULT_HECKTOR_TEST_SOURCE
    )
    for raw in (train, test):
        if not raw:
            continue
        p = Path(raw).expanduser()
        if p not in out:
            out.append(p)
    return out


def resolve_radheck(work: Path, explicit: Optional[Path] = None) -> Path:
    """
    Find the active RADHECK_* folder under work root.

    Order: explicit path / TEST5_RADHECK_DIR / RADHECK_CURRENT / newest RADHECK_* .
    """
    if explicit is not None:
        p = Path(explicit).expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(f"RADHECK dir not found: {p}")
        return p

    env = os.getenv("TEST5_RADHECK_DIR", "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(f"TEST5_RADHECK_DIR not found: {p}")
        return p

    pointer = work / "RADHECK_CURRENT"
    if pointer.is_symlink():
        target = pointer.resolve()
        if target.is_dir():
            return target
    if pointer.is_file():
        target = Path(pointer.read_text(encoding="utf-8").strip()).expanduser()
        if not target.is_absolute():
            target = (work / target).resolve()
        if target.is_dir():
            return target

    candidates = [p for p in work.glob("RADHECK_*") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(
            f"No RADHECK_* under {work}. Run: python -m pipelines.test5.transform_cases"
        )
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0].resolve()


def resolve_cases_root(
    work: Path,
    *,
    radheck: Optional[Path] = None,
    cases: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Return (radheck_dir, cases_root)."""
    if cases is not None:
        c = Path(cases).expanduser().resolve()
        if not c.is_dir():
            raise FileNotFoundError(f"cases root not found: {c}")
        return c.parent, c
    r = resolve_radheck(work, explicit=radheck)
    c = cases_root(r)
    if not c.is_dir():
        raise FileNotFoundError(f"Missing cases/ under {r}")
    return r, c
