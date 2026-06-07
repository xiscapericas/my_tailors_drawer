"""
Copy custom nnUNet trainer variants from this repo into the nnUNet installation.

nnUNet discovers trainer classes under nnunetv2/training/nnUNetTrainer/.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

_VARIANTS_DIR = Path(__file__).resolve().parent / "trainer_variants"
_NNUNET_VARIANTS_SUBDIR = (
    "nnunetv2/training/nnUNetTrainer/variants/training_length"
)


def install_trainer_variants(nnunet_path: str) -> list[str]:
    """
    Copy all *.py files from trainer_variants/ into the nnUNet training_length folder.

    Returns list of installed filenames.
    """
    if not nnunet_path:
        raise ValueError("nnunet_path is required")

    src_dir = _VARIANTS_DIR
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Trainer variants folder not found: {src_dir}")

    dest_dir = Path(nnunet_path) / _NNUNET_VARIANTS_SUBDIR
    if not dest_dir.is_dir():
        raise FileNotFoundError(
            f"nnUNet training_length folder not found: {dest_dir}\n"
            f"Check NNUNET_PATH ({nnunet_path})."
        )

    installed = []
    for src in sorted(src_dir.glob("*.py")):
        if src.name.startswith("_"):
            continue
        dest = dest_dir / src.name
        shutil.copy2(src, dest)
        installed.append(src.name)
        print(f"✓ Installed {src.name} -> {dest}")

    if not installed:
        print(f"No trainer variants found in {src_dir}")
    return installed


def main():
    parser = argparse.ArgumentParser(
        description="Install custom nnUNet trainer variants from this repository."
    )
    parser.add_argument(
        "--nnunet-path",
        default=os.getenv("NNUNET_PATH"),
        help="Path to nnUNet installation (default: NNUNET_PATH env var)",
    )
    args = parser.parse_args()
    if not args.nnunet_path:
        raise SystemExit("Set NNUNET_PATH or pass --nnunet-path")

    install_trainer_variants(args.nnunet_path)


if __name__ == "__main__":
    main()
