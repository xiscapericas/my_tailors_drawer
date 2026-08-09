#!/usr/bin/env python3
"""
Test6 — install STU-Net (nnUNetv2) + download pretrained weights on the server.

Does **not** re-run TotalSegmentator. Uses the bundled nnUNet-2.2 from STU-Net
(editable install) so STUNetTrainer_*_ft is available.

Example:

  export TEST6_WORK_ROOT=/media/HDD_8TB/xisca/work/retrain_test6_stunet
  export TEST6_STU_VARIANT=small   # small|base|large|huge

  python -m pipelines.test6.setup_stunet
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from pipelines.test6.paths import (
    DEFAULT_STUNET_REPO,
    WEIGHT_DRIVE_IDS,
    WEIGHT_FILENAMES,
    stunet_clone,
    variant as default_variant,
    work_root,
)


def _run(cmd: list[str], **kwargs) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, **kwargs)


def _clone_stunet(dest: Path, repo: str) -> None:
    if (dest / ".git").is_dir() or (dest / "nnUNet-2.2").is_dir():
        print(f"STU-Net already present: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--depth", "1", repo, str(dest)])


def _install_nnunet_v2(stunet: Path) -> Path:
    nnunet_v2 = stunet / "nnUNet-2.2"
    if not nnunet_v2.is_dir():
        raise FileNotFoundError(
            f"Missing {nnunet_v2}. Re-clone STU-Net or check the repo layout."
        )
    _run([sys.executable, "-m", "pip", "install", "-e", str(nnunet_v2)])
    _run([sys.executable, "-m", "pip", "install", "torchinfo", "gdown"])
    return nnunet_v2


def _download_weights(work: Path, variant: str) -> Path:
    weights_dir = work / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    fname = WEIGHT_FILENAMES[variant]
    dest = weights_dir / fname
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        print(f"Weights already present: {dest}")
        return dest

    file_id = WEIGHT_DRIVE_IDS[variant]
    tmp = weights_dir / f"{variant}_download.bin"
    print(f"Downloading STU-Net-{variant} weights from Google Drive ({file_id})…")
    try:
        import gdown
    except ImportError:
        _run([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown

    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, str(tmp), quiet=False)
    # Upstream downloads may be the raw .model (zip-like) or a wrapper zip
    if tmp.suffix == ".bin" or tmp.name.endswith("_download.bin"):
        # Detect zip-of-checkpoint vs raw checkpoint
        import zipfile

        if zipfile.is_zipfile(tmp):
            with zipfile.ZipFile(tmp) as zf:
                members = [m for m in zf.namelist() if m.endswith(".model")]
                if members:
                    zf.extract(members[0], weights_dir)
                    extracted = weights_dir / members[0]
                    if extracted.resolve() != dest.resolve():
                        if dest.exists():
                            dest.unlink()
                        extracted.rename(dest)
                    tmp.unlink(missing_ok=True)
                    print(f"Extracted {dest}")
                    return dest
        # Raw checkpoint file (PyTorch files are also zip-format)
        if dest.exists():
            dest.unlink()
        tmp.rename(dest)
    print(f"Saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def _write_env_hint(work: Path, stunet: Path, weights: Path, variant: str) -> None:
    hint = work / "TEST6_ENV.sh"
    nnunet_v2 = stunet / "nnUNet-2.2"
    text = f"""# Source before Test6 train/eval
export TEST6_WORK_ROOT={work}
export TEST6_STU_VARIANT={variant}
export TEST6_STUNET_CLONE={stunet}
export TEST6_PRETRAINED_WEIGHTS={weights}
export TEST6_NNUNET_V2={nnunet_v2}
export NNUNET_RETRAIN_PATH={work}/nnunet_retrain
export DATASET_FOLDER={work}/Dataset650_TotalSegmentator
export DATASET_ID=650
export ORGAN_DICTIONARY_PATH={work}/organ_dictionary_test5.json
export nnUNet_compile=false
"""
    hint.write_text(text)
    print(f"Wrote {hint}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test6: install STU-Net + weights")
    parser.add_argument("--work-root", default=str(work_root()))
    parser.add_argument(
        "--variant",
        default=os.getenv("TEST6_STU_VARIANT", default_variant()),
        choices=sorted(WEIGHT_DRIVE_IDS),
    )
    parser.add_argument("--repo", default=os.getenv("TEST6_STUNET_REPO", DEFAULT_STUNET_REPO))
    parser.add_argument("--skip-weights", action="store_true")
    parser.add_argument("--skip-pip", action="store_true")
    args = parser.parse_args()

    work = Path(args.work_root).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    stunet = stunet_clone(work)

    print("=" * 70)
    print("Test6 — STU-Net setup")
    print(f"Work root: {work}")
    print(f"Variant:   {args.variant}")
    print(f"Clone:     {stunet}")
    print("=" * 70)

    _clone_stunet(stunet, args.repo)
    if not args.skip_pip:
        _install_nnunet_v2(stunet)
    else:
        print("Skipping pip install (--skip-pip)")

    weights = work / "weights" / WEIGHT_FILENAMES[args.variant]
    if not args.skip_weights:
        weights = _download_weights(work, args.variant)
    elif not weights.is_file():
        print(f"WARNING: --skip-weights but missing {weights}")

    _write_env_hint(work, stunet, weights, args.variant)
    print("\nNext:")
    print("  python -m pipelines.test6.link_test5_dataset")
    print("  python -m pipelines.test6.train_finetune --step prepare")
    print("  python -m pipelines.test6.train_finetune --step plan")
    print("  python -m pipelines.test6.train_finetune --step train")


if __name__ == "__main__":
    main()
