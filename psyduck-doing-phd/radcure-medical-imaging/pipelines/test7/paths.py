"""Test7 path defaults — probability inference on Test5 model / Dataset650."""

from __future__ import annotations

import json
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_WORK_ROOT = "/media/HDD_8TB/xisca/work/retrain_test7_prob"
DEFAULT_TEST5_WORK_ROOT = "/media/HDD_8TB/xisca/work/retrain_test5"
DEFAULT_DATASET650 = f"{DEFAULT_TEST5_WORK_ROOT}/Dataset650_TotalSegmentator"
DEFAULT_TEST5_RETRAIN = f"{DEFAULT_TEST5_WORK_ROOT}/nnunet_retrain"
DEFAULT_TEST5_ORGAN_DICTIONARY = (
    f"{DEFAULT_TEST5_WORK_ROOT}/organ_dictionary_test5.json"
)

# Same trainer / checkpoint family as Test5 (not Test6 STU-Net)
DEFAULT_TRAINER = "nnUNetTrainer_700epochs_NoMirroring"
DEFAULT_CONFIGURATION = "3d_fullres"
DEFAULT_DATASET_ID = "650"

# Non-organ labels excluded from "competing region" discovery
_EXCLUDE_COMPETING = frozenset(
    {
        "background",
        "anatomical_region",
        "other-tissue",
        "GTVp",
        "GTVn",
    }
)


def work_root() -> Path:
    return Path(os.getenv("TEST7_WORK_ROOT", DEFAULT_WORK_ROOT)).expanduser()


def test5_work_root() -> Path:
    return Path(
        os.getenv("TEST5_WORK_ROOT", DEFAULT_TEST5_WORK_ROOT)
    ).expanduser()


def test5_dataset650() -> Path:
    return Path(
        os.getenv(
            "TEST7_DATASET650",
            os.getenv("TEST5_REFERENCE_DATASET650", DEFAULT_DATASET650),
        )
    ).expanduser()


def test5_retrain_path() -> Path:
    env = os.getenv("RETRAIN_RADHECK_TEST5", "").strip()
    if env:
        return Path(env).expanduser()
    env2 = os.getenv("TEST5_NNUNET_RETRAIN", "").strip()
    if env2:
        return Path(env2).expanduser()
    return Path(DEFAULT_TEST5_RETRAIN).expanduser()


def trainer_name() -> str:
    return os.getenv("NNUNET_TRAINER", DEFAULT_TRAINER)


def organ_dictionary_has_gtvp_gtvn(path: Path) -> bool:
    try:
        with open(path) as f:
            labels = json.load(f)
        return "GTVp" in labels and "GTVn" in labels
    except (OSError, json.JSONDecodeError, TypeError):
        return False


def is_legacy_radcure_dictionary(path: Path) -> bool:
    name = path.name.lower()
    return name == "radcure_dictionary.json" or (
        "radcurecomplete" in str(path).lower() and "test5" not in name
    )


def resolve_test5_organ_dictionary(
    work: Path | None = None,
    *,
    explicit: str = "",
) -> Path:
    """Organ dictionary: Test5 organ_dictionary_test5.json with GTVp/GTVn."""
    root = (work or work_root()).expanduser().resolve()
    test5 = test5_work_root()

    candidates: list[Path] = []
    if explicit.strip():
        candidates.append(Path(explicit).expanduser())
    candidates.extend(
        [
            root / "organ_dictionary_test5.json",
            Path(
                os.getenv(
                    "TEST7_ORGAN_DICTIONARY",
                    DEFAULT_TEST5_ORGAN_DICTIONARY,
                )
            ),
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
        "Run: python -m pipelines.test7.link_dataset"
    )


def _warn_stale(name: str, stale: str, using: Path) -> None:
    if not stale:
        return
    try:
        if Path(stale).expanduser().resolve() == using.resolve():
            return
    except (OSError, RuntimeError):
        pass
    print(f"NOTE: ignoring stale {name}={stale}\n      using Test7 {using}")


def pin_test7_env(work: Path | None = None) -> dict:
    """
    Pin dataset + organ under TEST7; point nnUNet_results at Test5 weights.

    No plan/preprocess/train — inference only.
    """
    root = (work or work_root()).expanduser().resolve()
    retrain = (root / "nnunet_retrain").resolve()
    retrain.mkdir(parents=True, exist_ok=True)
    logs = retrain / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    test5_retrain = test5_retrain_path().resolve()
    test5_results = test5_retrain / "nnUNet_results"
    if not test5_results.is_dir():
        raise FileNotFoundError(
            f"Test5 nnUNet_results not found: {test5_results}\n"
            "Set RETRAIN_RADHECK_TEST5 to the Test5 nnunet_retrain root.\n"
            "Test7 reuses Test5 weights (no retrain)."
        )

    _warn_stale("NNUNET_RETRAIN_PATH", os.getenv("NNUNET_RETRAIN_PATH", ""), retrain)
    os.environ["NNUNET_RETRAIN_PATH"] = str(retrain)
    # Predict needs model under nnUNet_results → Test5
    os.environ["nnUNet_results"] = str(test5_results)
    # raw/preprocessed unused for Test7 predict, but set local stubs to avoid stale env
    os.environ["nnUNet_raw"] = str(retrain)
    local_preproc = retrain / "nnUNet_preprocessed"
    local_preproc.mkdir(parents=True, exist_ok=True)
    os.environ["nnUNet_preprocessed"] = str(local_preproc)
    os.environ.setdefault("nnUNet_compile", "false")
    os.environ["DATASET_ID"] = DEFAULT_DATASET_ID
    os.environ["NNUNET_TRAINER"] = trainer_name()
    os.environ["NNUNET_CONFIGURATION"] = DEFAULT_CONFIGURATION
    os.environ.setdefault("NNUNET_DISABLE_TTA", "true")
    os.environ["LOG_DIR"] = str(logs)

    # Prefer a complete nnUNet source tree (site-packages can be incomplete)
    nnunet_src = os.getenv("NNUNET_PATH", "").strip()
    if not nnunet_src or nnunet_src == "/path/to/nnUNet":
        default_src = Path("/media/HDD_8TB/xisca/envs/nnUNet")
        if (default_src / "nnunetv2").is_dir():
            nnunet_src = str(default_src)
            os.environ["NNUNET_PATH"] = nnunet_src
    if nnunet_src:
        src = Path(nnunet_src).expanduser()
        root = src if (src / "nnunetv2").is_dir() else (
            src.parent if src.name == "nnunetv2" else None
        )
        if root is not None and root.is_dir():
            root_s = str(root.resolve())
            existing = os.environ.get("PYTHONPATH", "")
            if not existing.startswith(root_s):
                os.environ["PYTHONPATH"] = (
                    root_s if not existing else root_s + os.pathsep + existing
                )
            print(f"NOTE: using NNUNET_PATH for imports → {root_s}")

    dataset = (root / "Dataset650_TotalSegmentator").resolve()
    if dataset.is_dir():
        _warn_stale("DATASET_FOLDER", os.getenv("DATASET_FOLDER", ""), dataset)
        os.environ["DATASET_FOLDER"] = str(dataset)

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

    pred_root = root / "predictions"
    pred_root.mkdir(parents=True, exist_ok=True)
    os.environ["NNUNET_EVAL_OUTPUT_DIR"] = str(pred_root)

    return {
        "work": root,
        "retrain": retrain,
        "test5_retrain": test5_retrain,
        "results": test5_results,
        "dataset": dataset,
        "organ": organ_path,
        "predictions": pred_root,
        "logs": logs,
    }


def predictions_dir(work: Path | None = None) -> Path:
    root = work or work_root()
    return Path(
        os.getenv(
            "TEST7_PRED_DIR",
            str(Path(root) / "predictions" / "labelsTs_predicted"),
        )
    ).expanduser()


def probabilities_dir(work: Path | None = None) -> Path:
    root = work or work_root()
    return Path(
        os.getenv(
            "TEST7_PROB_DIR",
            str(Path(root) / "predictions" / "labelsTs_probabilities"),
        )
    ).expanduser()


def curves_dir(work: Path | None = None) -> Path:
    root = work or work_root()
    return Path(
        os.getenv(
            "TEST7_CURVES_DIR",
            str(Path(root) / "region_tumor_probabilities_vs_dice_curves"),
        )
    ).expanduser()


def probability_viz_dir(work: Path | None = None) -> Path:
    root = work or work_root()
    return Path(
        os.getenv(
            "TEST7_PROB_VIZ_DIR",
            str(Path(root) / "predictions" / "labelsTs_probability_viz"),
        )
    ).expanduser()


def load_organ_dict(path: Path | None = None) -> dict[str, int]:
    p = path or Path(os.environ.get("ORGAN_DICTIONARY_PATH", ""))
    if not p or not Path(p).is_file():
        p = resolve_test5_organ_dictionary()
    with open(p) as f:
        return json.load(f)


def competing_region_names(organ_dict: dict[str, int]) -> list[str]:
    return sorted(
        name
        for name in organ_dict
        if name not in _EXCLUDE_COMPETING and isinstance(organ_dict[name], int)
    )


# Console-script name → (module, entry_function) from nnUNetv2
_NNUNET_ENTRYPOINTS = {
    "nnUNetv2_predict": (
        "nnunetv2.inference.predict_from_raw_data",
        "predict_entry_point",
    ),
}


def check_numpy_blosc2() -> None:
    """Fail early if PATH/venv numpy and blosc2 are ABI-incompatible."""
    import sys

    try:
        import numpy  # noqa: F401
        import blosc2  # noqa: F401
    except ValueError as e:
        if "numpy.dtype size changed" in str(e) or "binary incompatibility" in str(e):
            raise RuntimeError(
                f"numpy/blosc2 binary mismatch under {sys.executable}.\n"
                "Fix in the *same* env you use for Test7:\n"
                "  which python; which nnUNetv2_predict\n"
                "  # both must be under the project .venv, not ~/.local\n"
                "  pip install --force-reinstall --no-cache-dir 'numpy' 'blosc2'\n"
                "Then re-run: python -m pipelines.test7.predict_probabilities"
            ) from e
        raise
    except ImportError:
        pass


def nnunet_cmd(name: str, *cli_args: str) -> tuple[list[str], dict]:
    """
    Build argv + env for an nnUNetv2 CLI using *this* Python, not PATH.

    Bare ``nnUNetv2_predict`` on PATH often resolves to ``~/.local`` (Python 3.9)
    and triggers ``numpy.dtype size changed`` / blosc2 ABI errors.

    If ``NNUNET_PATH`` points at a source tree containing ``nnunetv2/``, prepend
    it to ``PYTHONPATH`` so a broken site-packages copy cannot win.
    """
    import sys

    env = os.environ.copy()
    nnunet_src = os.getenv("NNUNET_PATH", "").strip()
    if nnunet_src and nnunet_src != "/path/to/nnUNet":
        src = Path(nnunet_src).expanduser()
        # Accept either …/nnUNet (repo root) or …/nnUNet/nnunetv2
        if (src / "nnunetv2").is_dir():
            root = str(src.resolve())
        elif src.name == "nnunetv2" and src.is_dir():
            root = str(src.parent.resolve())
        else:
            root = ""
        if root:
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = (
                root if not existing else root + os.pathsep + existing
            )
            print(f"NOTE: PYTHONPATH prepend NNUNET_PATH root → {root}")

    here = Path(sys.executable).resolve().parent / name
    if here.is_file() and os.access(here, os.X_OK):
        return [str(here), *cli_args], env

    entry = _NNUNET_ENTRYPOINTS.get(name)
    if entry is None:
        raise FileNotFoundError(
            f"nnUNet CLI {name!r} not next to {sys.executable} "
            f"and no entry-point fallback.\n"
            "Use the project .venv (not ~/.local)."
        )
    mod, func = entry
    argv = [name, *cli_args]
    code = (
        f"import sys; from {mod} import {func}; "
        f"sys.argv = {argv!r}; {func}()"
    )
    return [sys.executable, "-c", code], env


def expected_model_dir(test5_retrain: Path | None = None) -> Path:
    """Expected Test5 fold-0 model folder under nnUNet_results."""
    root = (test5_retrain or test5_retrain_path()).resolve()
    return (
        root
        / "nnUNet_results"
        / "Dataset650_TotalSegmentator"
        / f"{DEFAULT_TRAINER}__nnUNetPlans__{DEFAULT_CONFIGURATION}"
        / "fold_0"
    )
