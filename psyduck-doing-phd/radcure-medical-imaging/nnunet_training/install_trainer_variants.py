"""
Copy custom nnUNet trainer variants from this repo into the nnUNet package.

nnUNet discovers trainer classes under nnunetv2/training/nnUNetTrainer/.
When nnUNet is installed with pip, that path is in site-packages — not NNUNET_PATH.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import shutil
import sys
from pathlib import Path

_VARIANTS_DIR = Path(__file__).resolve().parent / "trainer_variants"
_RELATIVE_TRAINER_VARIANTS = Path("training/nnUNetTrainer/variants/training_length")


def _set_placeholder_nnunet_paths() -> None:
    """Avoid nnUNet path warnings when this module only installs trainer files."""
    for key in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        os.environ.setdefault(key, "/tmp/nnunet_trainer_install_only")


def get_nnunet_package_root() -> Path:
    """
    Resolve the installed ``nnunetv2`` package directory.

    ``nnunetv2.__file__`` can be ``None`` for some installs/namespace layouts;
    fall back to ``importlib.util.find_spec``.
    """
    import nnunetv2

    if getattr(nnunetv2, "__file__", None):
        return Path(nnunetv2.__file__).resolve().parent

    spec = importlib.util.find_spec("nnunetv2")
    if spec is not None:
        if spec.submodule_search_locations:
            return Path(list(spec.submodule_search_locations)[0]).resolve()
        if spec.origin and spec.origin != "namespace":
            return Path(spec.origin).resolve().parent

    raise RuntimeError(
        "Could not locate the nnunetv2 package directory "
        f"(nnunetv2.__file__={getattr(nnunetv2, '__file__', None)!r}).\n"
        "Install nnUNet into this venv, e.g.:\n"
        "  pip install nnunetv2\n"
        "or set NNUNET_PATH to a source checkout that contains nnunetv2/,\n"
        "then: python -m nnunet_training.install_trainer_variants"
    )


def get_trainer_variants_install_dirs(nnunet_path: str | None = None) -> list[Path]:
    """
    Return trainer variant install folders.

    Always includes the active nnunetv2 import location (pip/venv) when resolvable.
    Also includes NNUNET_PATH when set and different (editable/source installs).
    """
    dirs: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        resolved = path.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        dirs.append(path)

    try:
        add(get_nnunet_package_root() / _RELATIVE_TRAINER_VARIANTS)
    except RuntimeError as e:
        print(f"NOTE: {e}")

    candidate = nnunet_path or os.getenv("NNUNET_PATH")
    if candidate and candidate != "/path/to/nnUNet":
        add(Path(candidate).resolve() / "nnunetv2" / _RELATIVE_TRAINER_VARIANTS)

    return dirs


def _ensure_package_inits(dest_dir: Path) -> None:
    """
    nnUNet's ``recursive_find_python_class`` walks packages via pkgutil.
    Missing ``__init__.py`` under variants/ can hide trainers from discovery.
    """
    # dest_dir = …/nnUNetTrainer/variants/training_length
    for folder in (dest_dir, dest_dir.parent):
        init = folder / "__init__.py"
        if folder.is_dir() and not init.exists():
            init.write_text("# auto-created for nnUNet trainer discovery\n")
            print(f"✓ Created {init}")


def _trainer_file_path(trainer_name: str) -> Path | None:
    try:
        root = get_nnunet_package_root()
    except RuntimeError:
        return None
    path = root / _RELATIVE_TRAINER_VARIANTS / f"{trainer_name}.py"
    return path if path.is_file() else None


def _import_trainer_via_file(trainer_name: str) -> tuple[bool, str]:
    """Load trainer class from its .py file (same path nnUNet install uses)."""
    path = _trainer_file_path(trainer_name)
    if path is None:
        return False, f"file not found under {_RELATIVE_TRAINER_VARIANTS}"
    try:
        spec = importlib.util.spec_from_file_location(
            f"_radcure_trainer_{trainer_name}", path
        )
        if spec is None or spec.loader is None:
            return False, f"could not build import spec for {path}"
        mod = importlib.util.module_from_spec(spec)
        # Isolate from sys.modules cache so re-verify after install works
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        if not hasattr(mod, trainer_name):
            return False, f"{path.name} has no attribute {trainer_name}"
        return True, f"ok ({path})"
    except Exception as e:
        return False, f"import failed for {path}: {type(e).__name__}: {e}"


def _import_trainer_via_package(trainer_name: str) -> tuple[bool, str]:
    mod_name = (
        "nnunetv2.training.nnUNetTrainer.variants.training_length." + trainer_name
    )
    try:
        # Drop cached miss / stale module
        for key in list(sys.modules):
            if key == mod_name or key.startswith(mod_name + "."):
                del sys.modules[key]
        mod = importlib.import_module(mod_name)
        if not hasattr(mod, trainer_name):
            return False, f"{mod_name} imported but has no {trainer_name}"
        return True, f"ok ({mod_name})"
    except Exception as e:
        return False, f"package import failed: {type(e).__name__}: {e}"


def trainer_is_available(trainer_name: str, *, verbose: bool = False) -> bool:
    """
    True if nnUNet can discover the trainer (or we can import it the same way).

    Order: package import → file import → nnUNet recursive_find_python_class.
    """
    ok, msg = _import_trainer_via_package(trainer_name)
    if ok:
        if verbose:
            print(f"✓ trainer discoverable via package: {msg}")
        return True
    if verbose:
        print(f"NOTE: {msg}")

    ok, msg = _import_trainer_via_file(trainer_name)
    if ok:
        if verbose:
            print(f"✓ trainer loadable via file: {msg}")
        # Still try recursive_find — needed for nnUNet CLI; warn if it fails
        if not _trainer_found_by_nnunet(trainer_name) and verbose:
            print(
                "WARNING: file imports OK but nnUNet recursive_find did not see it yet.\n"
                "  Ensure variants/ and training_length/ have __init__.py "
                "(install_trainer_variants now creates them)."
            )
        return True
    if verbose:
        print(f"NOTE: {msg}")

    found = _trainer_found_by_nnunet(trainer_name)
    if verbose and not found:
        print("NOTE: nnUNet recursive_find_python_class did not find the trainer")
    return found


def _trainer_found_by_nnunet(trainer_name: str) -> bool:
    try:
        from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
    except ImportError:
        return False

    try:
        trainer_root = get_nnunet_package_root() / "training" / "nnUNetTrainer"
    except RuntimeError:
        return False
    if not trainer_root.is_dir():
        return False
    return (
        recursive_find_python_class(
            folder=str(trainer_root),
            class_name=trainer_name,
            current_module="nnunetv2.training.nnUNetTrainer",
        )
        is not None
    )


def install_trainer_variants(nnunet_path: str | None = None) -> list[str]:
    """
    Copy all *.py files from trainer_variants/ into nnUNet training_length folder(s).

    Returns list of installed filenames (from the last successful install target).
    """
    src_dir = _VARIANTS_DIR
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Trainer variants folder not found: {src_dir}")

    sources = [
        src for src in sorted(src_dir.glob("*.py")) if not src.name.startswith("_")
    ]
    if not sources:
        print(f"No trainer variants found in {src_dir}")
        return []

    installed: list[str] = []
    dest_dirs = get_trainer_variants_install_dirs(nnunet_path)
    if not dest_dirs:
        raise FileNotFoundError("Could not resolve nnUNet trainer install directory.")

    for dest_dir in dest_dirs:
        dest_dir.mkdir(parents=True, exist_ok=True)
        _ensure_package_inits(dest_dir)
        for src in sources:
            dest = dest_dir / src.name
            shutil.copy2(src, dest)
            if src.name not in installed:
                installed.append(src.name)
            print(f"✓ Installed {src.name} -> {dest}")

    if not installed:
        raise FileNotFoundError(
            "No nnUNet trainer install directory found.\n"
            f"Tried: {', '.join(str(d) for d in dest_dirs)}"
        )

    return installed


def ensure_trainer_installed(trainer_name: str, nnunet_path: str | None = None) -> None:
    """Install repo trainer variants if trainer_name is not yet discoverable."""
    if trainer_is_available(trainer_name, verbose=True):
        print(f"✓ nnUNet trainer already available: {trainer_name}")
        return

    print(f"Installing custom trainer variant for: {trainer_name}")
    try:
        install_trainer_variants(nnunet_path)
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Cannot install trainer {trainer_name}: {e}\n"
            "Diagnose:\n"
            "  python -c \"import nnunetv2,sys; print(nnunetv2, getattr(nnunetv2,'__file__',None), sys.executable)\"\n"
            "  ls .venv/bin/nnUNetv2_predict\n"
            "Install nnUNet into this .venv (not ~/.local), or set NNUNET_PATH to a\n"
            "source tree containing nnunetv2/, then:\n"
            "  python -m nnunet_training.install_trainer_variants"
        ) from e

    # Invalidate import caches so package import sees new files / __init__.py
    importlib.invalidate_caches()

    if not trainer_is_available(trainer_name, verbose=True):
        try:
            package_root = get_nnunet_package_root()
            expected = package_root / _RELATIVE_TRAINER_VARIANTS / f"{trainer_name}.py"
        except RuntimeError:
            expected = "(nnunetv2 package root unresolved)"
        raise RuntimeError(
            f"Trainer {trainer_name} is still not discoverable after install.\n"
            f"Expected file: {expected}\n"
            "Debug:\n"
            f"  ls -la $(python -c \"from nnunet_training.install_trainer_variants import get_nnunet_package_root as g; print(g())\")/training/nnUNetTrainer/variants/training_length/\n"
            "  python -c \"from nnunet_training.install_trainer_variants import trainer_is_available; print(trainer_is_available('nnUNetTrainer_700epochs_NoMirroring', verbose=True))\""
        )

    if not _trainer_found_by_nnunet(trainer_name):
        print(
            "WARNING: trainer imports, but nnUNet recursive_find still misses it.\n"
            "  Predict may fail with 'could not find trainer'. Check __init__.py under variants/."
        )
    else:
        print(f"✓ nnUNet trainer ready: {trainer_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Install custom nnUNet trainer variants from this repository."
    )
    parser.add_argument(
        "--nnunet-path",
        default=os.getenv("NNUNET_PATH"),
        help="Optional extra nnUNet source tree (also installs into active nnunetv2 package)",
    )
    parser.add_argument(
        "--verify",
        default="nnUNetTrainer_700epochs_NoMirroring",
        help="Trainer class name to verify after install (default: 700-epoch variant)",
    )
    args = parser.parse_args()

    _set_placeholder_nnunet_paths()
    install_trainer_variants(args.nnunet_path)
    importlib.invalidate_caches()
    if args.verify:
        ensure_trainer_installed(args.verify, args.nnunet_path)


if __name__ == "__main__":
    main()
