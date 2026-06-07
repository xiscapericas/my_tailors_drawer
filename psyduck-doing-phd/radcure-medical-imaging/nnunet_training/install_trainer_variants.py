"""
Copy custom nnUNet trainer variants from this repo into the nnUNet package.

nnUNet discovers trainer classes under nnunetv2/training/nnUNetTrainer/.
When nnUNet is installed with pip, that path is in site-packages — not NNUNET_PATH.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

_VARIANTS_DIR = Path(__file__).resolve().parent / "trainer_variants"
_RELATIVE_TRAINER_VARIANTS = Path("training/nnUNetTrainer/variants/training_length")


def _set_placeholder_nnunet_paths() -> None:
    """Avoid nnUNet path warnings when this module only installs trainer files."""
    for key in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        os.environ.setdefault(key, "/tmp/nnunet_trainer_install_only")


def get_nnunet_package_root() -> Path:
    import nnunetv2

    return Path(nnunetv2.__file__).resolve().parent


def get_trainer_variants_install_dirs(nnunet_path: str | None = None) -> list[Path]:
    """
    Return trainer variant install folders.

    Always includes the active nnunetv2 import location (pip/venv).
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

    add(get_nnunet_package_root() / _RELATIVE_TRAINER_VARIANTS)

    candidate = nnunet_path or os.getenv("NNUNET_PATH")
    if candidate and candidate != "/path/to/nnUNet":
        add(Path(candidate).resolve() / "nnunetv2" / _RELATIVE_TRAINER_VARIANTS)

    return dirs


def trainer_is_available(trainer_name: str) -> bool:
    from nnunetv2.utilities.find_class_by_name import recursive_find_python_class

    trainer_root = get_nnunet_package_root() / "training" / "nnUNetTrainer"
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
        if not dest_dir.is_dir():
            print(f"⚠️  Skipping missing folder: {dest_dir}")
            continue

        dest_dir.mkdir(parents=True, exist_ok=True)
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
    if trainer_is_available(trainer_name):
        print(f"✓ nnUNet trainer already available: {trainer_name}")
        return

    print(f"Installing custom trainer variant for: {trainer_name}")
    install_trainer_variants(nnunet_path)

    if not trainer_is_available(trainer_name):
        package_root = get_nnunet_package_root()
        raise RuntimeError(
            f"Trainer {trainer_name} is still not discoverable after install.\n"
            f"Expected it under: {package_root / _RELATIVE_TRAINER_VARIANTS}\n"
            "Run manually: python -m nnunet_training.install_trainer_variants"
        )

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
    if args.verify:
        ensure_trainer_installed(args.verify, args.nnunet_path)


if __name__ == "__main__":
    main()
