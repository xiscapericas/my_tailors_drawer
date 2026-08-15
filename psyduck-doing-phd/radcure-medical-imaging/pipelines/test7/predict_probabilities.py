#!/usr/bin/env python3
"""
Test7 — predict hard masks + per-class softmax probabilities (Test5 model).

Uses Test5 ``nnUNetTrainer_700epochs_NoMirroring`` weights via
``nnUNet_results`` → Test5 retrain. Writes under TEST7_WORK_ROOT/predictions/:

  labelsTs_predicted/       hard argmax NIfTI (nnUNet default)
  labelsTs_probabilities/   slim cropped float16 ``*.slim.npz`` (raw .npz deleted)

Example:

  source …/TEST7_ENV.sh   # optional
  python -m pipelines.test7.predict_probabilities
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

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipelines.test7.paths import (
    DEFAULT_CONFIGURATION,
    DEFAULT_DATASET_ID,
    pin_test7_env,
    predictions_dir,
    probabilities_dir,
    trainer_name,
    work_root,
)


def _relocate_npz(pred_dir: Path, prob_dir: Path) -> int:
    """
    nnUNetv2 writes ``{case}.npz`` next to ``{case}.nii.gz``.
    Move/copy them into ``labelsTs_probabilities/`` for a clear layout.
    """
    prob_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    for npz in pred_dir.glob("*.npz"):
        dest = prob_dir / npz.name
        if dest.exists():
            dest.unlink()
        shutil.move(str(npz), str(dest))
        moved += 1
    return moved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test7: nnUNet predict with -save_probabilities (Test5 model)"
    )
    parser.add_argument("--work-root", default=str(work_root()))
    parser.add_argument("--fold", type=int, default=int(os.getenv("NNUNET_FOLD", "0")))
    parser.add_argument("--dataset-id", default=os.getenv("DATASET_ID", DEFAULT_DATASET_ID))
    parser.add_argument(
        "--trainer",
        default=os.getenv("NNUNET_TRAINER", trainer_name()),
    )
    parser.add_argument(
        "--configuration",
        default=os.getenv("NNUNET_CONFIGURATION", DEFAULT_CONFIGURATION),
    )
    parser.add_argument(
        "--keep-npz-in-pred",
        action="store_true",
        help="Leave .npz beside hard masks (default: move to labelsTs_probabilities)",
    )
    parser.add_argument(
        "--skip-slim",
        action="store_true",
        help="Do not convert raw .npz → .slim.npz after predict",
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep raw .npz after slim conversion (default: delete)",
    )
    parser.add_argument("--slim-margin", type=int, default=8)
    parser.add_argument("--slim-dilate", type=int, default=2)
    parser.add_argument("--slim-top-k", type=int, default=5)
    args = parser.parse_args()

    work = Path(args.work_root).expanduser().resolve()
    paths = pin_test7_env(work)
    dataset = paths["dataset"]
    if not dataset.is_dir():
        raise FileNotFoundError(
            f"Dataset650 missing: {dataset}\n"
            "Run: python -m pipelines.test7.link_dataset"
        )
    images_ts = dataset / "imagesTs"
    if not images_ts.is_dir():
        raise FileNotFoundError(f"imagesTs not found: {images_ts}")

    pred_dir = predictions_dir(work)
    prob_dir = probabilities_dir(work)
    pred_dir.mkdir(parents=True, exist_ok=True)
    prob_dir.mkdir(parents=True, exist_ok=True)

    disable_tta = os.getenv("NNUNET_DISABLE_TTA", "true").lower() in (
        "1",
        "true",
        "yes",
    )

    cmd = [
        "nnUNetv2_predict",
        "-i",
        str(images_ts),
        "-o",
        str(pred_dir),
        "-d",
        str(args.dataset_id),
        "-c",
        args.configuration,
        "-tr",
        args.trainer,
        "-f",
        str(args.fold),
        "-save_probabilities",
    ]
    if disable_tta:
        cmd.append("--disable_tta")

    log_file = paths["logs"] / f"prediction_probs_d{args.dataset_id}.log"
    print("=" * 70)
    print("Test7 — predict with probabilities (Test5 model)")
    print(f"  nnUNet_results: {os.environ.get('nnUNet_results')}")
    print(f"  trainer:        {args.trainer}")
    print(f"  input:          {images_ts}")
    print(f"  hard masks:     {pred_dir}")
    print(f"  probabilities:  {prob_dir}")
    print(f"  log:            {log_file}")
    print("=" * 70)
    print("+", " ".join(cmd))

    with open(log_file, "w") as log:
        result = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"Prediction failed (exit {result.returncode}). See {log_file}"
        )

    if not args.keep_npz_in_pred:
        n = _relocate_npz(pred_dir, prob_dir)
        print(f"Moved {n} .npz file(s) → {prob_dir}")
    else:
        # Still copy for the dedicated probabilities folder
        prob_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for npz in pred_dir.glob("*.npz"):
            dest = prob_dir / npz.name
            shutil.copy2(npz, dest)
            n += 1
        print(f"Copied {n} .npz file(s) → {prob_dir}")

    if not args.skip_slim:
        from pipelines.test7.slim_probabilities import slim_all

        print("\nSlimming raw .npz → cropped float16 .slim.npz …")
        slim_all(
            work,
            dilate_iter=args.slim_dilate,
            margin=args.slim_margin,
            top_k=args.slim_top_k,
            keep_raw=args.keep_raw,
        )
    else:
        print("\nSkipped slim step (--skip-slim). Raw .npz left in place.")
        print("  Run later: python -m pipelines.test7.slim_probabilities")

    print("\nNext:")
    print("  python -m pipelines.test7.region_tumor_probabilities_vs_dice_curves")
    print("  python -m pipelines.test7.probability_visualisation")


if __name__ == "__main__":
    main()
