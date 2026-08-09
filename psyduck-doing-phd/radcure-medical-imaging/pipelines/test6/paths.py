"""Test6 path defaults — STU-Net fine-tune on Test5 Dataset650."""

from __future__ import annotations

import json
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_WORK_ROOT = "/media/HDD_8TB/xisca/work/retrain_test6_stunet"
DEFAULT_TEST5_WORK_ROOT = "/media/HDD_8TB/xisca/work/retrain_test5"
DEFAULT_DATASET650 = f"{DEFAULT_TEST5_WORK_ROOT}/Dataset650_TotalSegmentator"
# Test5 GTVp/GTVn dict — never use dataset/RadcureComplete/radcure_dictionary.json
DEFAULT_TEST5_ORGAN_DICTIONARY = (
    f"{DEFAULT_TEST5_WORK_ROOT}/organ_dictionary_test5.json"
)
DEFAULT_STUNET_REPO = "https://github.com/uni-medical/STU-Net.git"

# Google Drive file ids for TotalSegmentator-pretrained checkpoints (ep4k)
WEIGHT_DRIVE_IDS = {
    "small": "1HReH6dDrEuXgHPrsw7OrHSjvEUF3f4mv",
    "base": "1BHCp1Ort-OaVFwaZmvsG4qHiKiPeNb4h",
    "large": "1KA1eXWWf_xAoJg5KHYrxTmfiz7wxGhHS",
    "huge": "1Qrq7oGPJ7ileFHWOAxwpeWdaB6hySptU",
}

WEIGHT_FILENAMES = {
    "small": "small_ep4k.model",
    "base": "base_ep4k.model",
    "large": "large_ep4k.model",
    "huge": "huge_ep4k.model",
}

TRAINER_FT = {
    "small": "STUNetTrainer_small_ft",
    "base": "STUNetTrainer_base_ft",
    "large": "STUNetTrainer_large_ft",
    "huge": "STUNetTrainer_huge_ft",
}


def work_root() -> Path:
    return Path(os.getenv("TEST6_WORK_ROOT", DEFAULT_WORK_ROOT)).expanduser()


def _warn_stale(name: str, stale: str, using: Path) -> None:
    if not stale:
        return
    try:
        if Path(stale).expanduser().resolve() == using.resolve():
            return
    except (OSError, RuntimeError):
        pass
    print(f"NOTE: ignoring stale {name}={stale}\n      using Test6 {using}")


def organ_dictionary_has_gtvp_gtvn(path: Path) -> bool:
    try:
        with open(path) as f:
            labels = json.load(f)
        return "GTVp" in labels and "GTVn" in labels
    except (OSError, json.JSONDecodeError, TypeError):
        return False


def is_legacy_radcure_dictionary(path: Path) -> bool:
    """True for old merged-tumour dict (no separate GTVn)."""
    name = path.name.lower()
    return name == "radcure_dictionary.json" or (
        "radcurecomplete" in str(path).lower() and "test5" not in name
    )


def resolve_test5_organ_dictionary(
    work: Path | None = None,
    *,
    explicit: str = "",
) -> Path:
    """
    Organ dictionary for Test6: Test5 ``organ_dictionary_test5.json`` only.

    Never ``…/RadcureComplete/radcure_dictionary.json`` (stale ORGAN_DICTIONARY_PATH).
    """
    root = (work or work_root()).expanduser().resolve()
    test5 = Path(
        os.getenv("TEST5_WORK_ROOT", DEFAULT_TEST5_WORK_ROOT)
    ).expanduser()

    candidates: list[Path] = []
    if explicit.strip():
        candidates.append(Path(explicit).expanduser())
    candidates.extend(
        [
            root / "organ_dictionary_test5.json",
            Path(os.getenv("TEST6_ORGAN_DICTIONARY", DEFAULT_TEST5_ORGAN_DICTIONARY)),
            test5 / "organ_dictionary_test5.json",
            Path(DEFAULT_TEST5_ORGAN_DICTIONARY),
        ]
    )
    # RADHECK_1047 / RADHECK_CURRENT, etc.
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
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        if is_legacy_radcure_dictionary(p):
            print(f"NOTE: skipping legacy organ dict {p}")
            continue
        if p.is_file() and organ_dictionary_has_gtvp_gtvn(p):
            return p

    raise FileNotFoundError(
        "Need Test5 organ_dictionary_test5.json with GTVp/GTVn.\n"
        f"Expected e.g. {DEFAULT_TEST5_ORGAN_DICTIONARY}\n"
        "Do not use …/RadcureComplete/radcure_dictionary.json.\n"
        "Run: python -m pipelines.test6.link_test5_dataset"
    )


def pin_test6_env(work: Path | None = None) -> dict:
    """
    Force nnUNet / dataset / organ paths under TEST6 work root.

    Shell / ``.env`` leftovers from Test1–3 (``NNUNET_RETRAIN_PATH``,
    ``ORGAN_DICTIONARY_PATH`` → radcure_dictionary.json, etc.) must not win.
    """
    root = (work or work_root()).expanduser().resolve()
    retrain = (root / "nnunet_retrain").resolve()
    retrain.mkdir(parents=True, exist_ok=True)
    preproc = retrain / "nnUNet_preprocessed"
    results = retrain / "nnUNet_results"
    preproc.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    _warn_stale("NNUNET_RETRAIN_PATH", os.getenv("NNUNET_RETRAIN_PATH", ""), retrain)
    os.environ["NNUNET_RETRAIN_PATH"] = str(retrain)
    os.environ["nnUNet_raw"] = str(retrain)
    os.environ["nnUNet_preprocessed"] = str(preproc)
    os.environ["nnUNet_results"] = str(results)
    os.environ.setdefault("nnUNet_compile", "false")
    os.environ["DATASET_ID"] = "650"

    dataset = (root / "Dataset650_TotalSegmentator").resolve()
    if dataset.is_dir():
        _warn_stale("DATASET_FOLDER", os.getenv("DATASET_FOLDER", ""), dataset)
        os.environ["DATASET_FOLDER"] = str(dataset)

    # Drop legacy ORGAN_DICTIONARY_PATH before anything else reads it
    env_org = os.getenv("ORGAN_DICTIONARY_PATH", "").strip()
    if env_org and is_legacy_radcure_dictionary(Path(env_org)):
        print(
            f"NOTE: ignoring legacy ORGAN_DICTIONARY_PATH={env_org}\n"
            f"      use {DEFAULT_TEST5_ORGAN_DICTIONARY}"
        )
        os.environ.pop("ORGAN_DICTIONARY_PATH", None)

    organ_path = None
    try:
        organ_path = resolve_test5_organ_dictionary(root)
        # Ensure a copy under Test6 work for a stable path
        local = root / "organ_dictionary_test5.json"
        if organ_path.resolve() != local.resolve():
            import shutil

            if local.is_symlink() or local.exists():
                local.unlink()
            shutil.copy2(organ_path, local)
            organ_path = local.resolve()
            print(f"NOTE: copied organ dict → {organ_path}")
        os.environ["ORGAN_DICTIONARY_PATH"] = str(organ_path)
    except FileNotFoundError as e:
        print(f"WARNING: {e}")

    return {
        "work": root,
        "retrain": retrain,
        "raw": retrain,
        "preprocessed": preproc,
        "results": results,
        "dataset": dataset,
        "organ": organ_path,
    }


def test5_dataset650() -> Path:
    return Path(
        os.getenv(
            "TEST6_DATASET650",
            os.getenv("TEST5_REFERENCE_DATASET650", DEFAULT_DATASET650),
        )
    ).expanduser()


def stunet_clone(work: Path | None = None) -> Path:
    root = work or work_root()
    return Path(os.getenv("TEST6_STUNET_CLONE", str(root / "STU-Net"))).expanduser()


def variant() -> str:
    v = os.getenv("TEST6_STU_VARIANT", "small").strip().lower()
    if v not in WEIGHT_DRIVE_IDS:
        raise ValueError(
            f"TEST6_STU_VARIANT must be one of {sorted(WEIGHT_DRIVE_IDS)}, got {v!r}"
        )
    return v


def nnunet_v2_root(work: Path | None = None) -> Path:
    env = os.getenv("TEST6_NNUNET_V2", "").strip()
    if env:
        return Path(env).expanduser()
    return stunet_clone(work) / "nnUNet-2.2"


# Console-script name → (module, entry_function) from nnUNetv2 pyproject.toml
_NNUNET_ENTRYPOINTS = {
    "nnUNetv2_plan_and_preprocess": (
        "nnunetv2.experiment_planning.plan_and_preprocess_entrypoints",
        "plan_and_preprocess_entry",
    ),
    "nnUNetv2_predict": (
        "nnunetv2.inference.predict_from_raw_data",
        "predict_entry_point",
    ),
    "nnUNetv2_train": ("nnunetv2.run.run_training", "run_training_entry"),
}


def python_env_with_stunet(work: Path | None = None) -> dict:
    """Env for subprocesses: prefer STU-Net nnUNet-2.2 on PYTHONPATH."""
    import sys

    env = os.environ.copy()
    root = nnunet_v2_root(work)
    parts = [str(root)] if root.is_dir() else []
    existing = env.get("PYTHONPATH", "")
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    env["TEST6_PYTHON"] = sys.executable
    return env


def nnunet_cmd(name: str, *cli_args: str, work: Path | None = None) -> tuple[list[str], dict]:
    """
    Build argv + env for an nnUNetv2 CLI using *this* Python, not PATH.

    Bare ``nnUNetv2_*`` on PATH often resolves to ``~/.local`` (another Python)
    and triggers ``numpy.dtype size changed`` / blosc2 ABI errors.
    """
    import sys

    env = python_env_with_stunet(work)
    here = Path(sys.executable).resolve().parent / name
    if here.is_file() and os.access(here, os.X_OK):
        return [str(here), *cli_args], env

    entry = _NNUNET_ENTRYPOINTS.get(name)
    if entry is None:
        raise FileNotFoundError(
            f"nnUNet CLI {name!r} not next to {sys.executable} "
            f"and no entry-point fallback.\n"
            "Use the project .venv and: python -m pipelines.test6.setup_stunet"
        )
    mod, func = entry
    argv = [name, *cli_args]
    code = (
        f"import sys; from {mod} import {func}; "
        f"sys.argv = {argv!r}; {func}()"
    )
    return [sys.executable, "-c", code], env


def check_numpy_blosc2() -> None:
    """Fail early with a clear fix if numpy/blosc2 are ABI-incompatible."""
    import sys

    try:
        import numpy  # noqa: F401
        import blosc2  # noqa: F401
    except ValueError as e:
        if "numpy.dtype size changed" in str(e) or "binary incompatibility" in str(e):
            raise RuntimeError(
                f"numpy/blosc2 binary mismatch under {sys.executable}.\n"
                "Fix in the *same* env you use for Test6:\n"
                "  pip install --force-reinstall --no-cache-dir 'numpy' 'blosc2'\n"
                "Then confirm: which python; which nnUNetv2_plan_and_preprocess\n"
                "(both should be under your .venv, not ~/.local)."
            ) from e
        raise
    except ImportError:
        # blosc2 may only be needed when acvl_utils loads; plan will surface it
        pass
