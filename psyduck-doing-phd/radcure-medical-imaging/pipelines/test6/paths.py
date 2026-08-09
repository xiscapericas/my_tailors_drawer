"""Test6 path defaults — STU-Net fine-tune on Test5 Dataset650."""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_WORK_ROOT = "/media/HDD_8TB/xisca/work/retrain_test6_stunet"
DEFAULT_TEST5_WORK_ROOT = "/media/HDD_8TB/xisca/work/retrain_test5"
DEFAULT_DATASET650 = f"{DEFAULT_TEST5_WORK_ROOT}/Dataset650_TotalSegmentator"
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
