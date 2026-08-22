"""Test 8.0 path defaults — CT+PET nnUNet on HECKTOR-only Test5 split."""

from __future__ import annotations

import json
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_WORK_ROOT = "/media/HDD_8TB/xisca/work/retrain_test8_0"
DEFAULT_TEST5_WORK_ROOT = "/media/HDD_8TB/xisca/work/retrain_test5"
DEFAULT_DATASET650 = f"{DEFAULT_TEST5_WORK_ROOT}/Dataset650_TotalSegmentator"
DEFAULT_TEST5_ORGAN_DICTIONARY = (
    f"{DEFAULT_TEST5_WORK_ROOT}/organ_dictionary_test5.json"
)


def work_root() -> Path:
    return Path(os.getenv("TEST8_0_WORK_ROOT", DEFAULT_WORK_ROOT)).expanduser()


def test5_work_root() -> Path:
    return Path(os.getenv("TEST5_WORK_ROOT", DEFAULT_TEST5_WORK_ROOT)).expanduser()


def test5_dataset650() -> Path:
    return Path(
        os.getenv(
            "TEST8_0_DATASET650",
            os.getenv("TEST5_REFERENCE_DATASET650", DEFAULT_DATASET650),
        )
    ).expanduser()


def resolve_test5_cases_root(test5_work: Path | None = None) -> Path:
    """Test5 ``RADHECK_{N}/cases`` (CT+labels in ``output/``; PET was not copied)."""
    explicit = os.getenv("TEST5_RADHECK_CASES", "").strip()
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if p.is_dir():
            return p
    root = (test5_work or test5_work_root()).expanduser().resolve()
    pointer = root / "RADHECK_CURRENT"
    if pointer.exists():
        try:
            cases = pointer.resolve() / "cases"
            if cases.is_dir():
                return cases
        except (OSError, RuntimeError):
            pass
    for rad in sorted(root.glob("RADHECK_*"), reverse=True):
        if rad.name == "RADHECK_CURRENT":
            continue
        cases = rad / "cases"
        if cases.is_dir():
            return cases
    raise FileNotFoundError(
        f"No RADHECK_*/cases under {root}. Set TEST5_RADHECK_CASES or TEST5_WORK_ROOT."
    )


def organ_dictionary_has_gtvp_gtvn(path: Path) -> bool:
    try:
        with open(path) as f:
            labels = json.load(f)
        return "GTVp" in labels and "GTVn" in labels
    except (OSError, json.JSONDecodeError, TypeError):
        return False


def resolve_test5_organ_dictionary(
    work: Path | None = None,
    *,
    explicit: str = "",
) -> Path:
    root = (work or work_root()).expanduser().resolve()
    test5 = test5_work_root()
    candidates: list[Path] = []
    if explicit.strip():
        candidates.append(Path(explicit).expanduser())
    candidates.extend(
        [
            root / "organ_dictionary_test5.json",
            test5 / "organ_dictionary_test5.json",
            Path(DEFAULT_TEST5_ORGAN_DICTIONARY),
        ]
    )
    if test5.is_dir():
        pointer = test5 / "RADHECK_CURRENT"
        if pointer.exists():
            try:
                candidates.append(pointer.resolve() / "organ_dictionary_test5.json")
            except (OSError, RuntimeError):
                pass
        for rad in sorted(test5.glob("RADHECK_*"), reverse=True):
            if rad.is_dir() and rad.name != "RADHECK_CURRENT":
                candidates.append(rad / "organ_dictionary_test5.json")

    seen: set[str] = set()
    for cand in candidates:
        try:
            p = cand.expanduser().resolve()
        except (OSError, RuntimeError):
            continue
        if str(p) in seen:
            continue
        seen.add(str(p))
        if p.is_file() and organ_dictionary_has_gtvp_gtvn(p):
            return p
    raise FileNotFoundError(
        "Need Test5 organ_dictionary_test5.json with GTVp/GTVn.\n"
        f"Expected e.g. {DEFAULT_TEST5_ORGAN_DICTIONARY}"
    )


def pin_test8_0_env(work: Path | None = None) -> dict:
    root = (work or work_root()).expanduser().resolve()
    retrain = (root / "nnunet_retrain").resolve()
    retrain.mkdir(parents=True, exist_ok=True)
    preproc = retrain / "nnUNet_preprocessed"
    results = retrain / "nnUNet_results"
    preproc.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    os.environ["NNUNET_RETRAIN_PATH"] = str(retrain)
    os.environ["nnUNet_raw"] = str(retrain)
    os.environ["nnUNet_preprocessed"] = str(preproc)
    os.environ["nnUNet_results"] = str(results)
    os.environ.setdefault("nnUNet_compile", "false")
    os.environ["DATASET_ID"] = "650"
    os.environ["NNUNET_USE_LOCAL_PREPROCESS"] = "1"
    os.environ.pop("NNUNET_PREPROCESSED_PATH", None)

    dataset = (root / "Dataset650_TotalSegmentator").resolve()
    if dataset.is_dir():
        os.environ["DATASET_FOLDER"] = str(dataset)

    try:
        organ = resolve_test5_organ_dictionary(root)
        local = root / "organ_dictionary_test5.json"
        if organ.resolve() != local.resolve() and organ.is_file():
            import shutil

            if local.is_symlink() or local.exists():
                local.unlink()
            shutil.copy2(organ, local)
            organ = local.resolve()
        os.environ["ORGAN_DICTIONARY_PATH"] = str(organ)
    except FileNotFoundError:
        organ = None

    return {
        "work": root,
        "retrain": retrain,
        "preprocessed": preproc,
        "results": results,
        "dataset": dataset,
        "organ": organ,
    }
