#!/usr/bin/env python3
"""
Test6 — predict + Dice on Dataset650 imagesTs (unified RADCURE + HECKTOR test).

Uses the fine-tuned STU-Net checkpoint; metrics via the same evaluation helpers
as Test5 (cohort split from ts_case_map.json when present).

Example:

  source ${TEST6_WORK_ROOT}/TEST6_ENV.sh
  python -m pipelines.test6.evaluate
  python -m pipelines.test6.evaluate --viz
"""

from __future__ import annotations

import argparse
import os
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

from pipelines.test6.paths import (
    TRAINER_FT,
    nnunet_cmd,
    variant as default_variant,
    work_root,
)


def _ensure_env(work: Path) -> None:
    retrain = Path(
        os.getenv("NNUNET_RETRAIN_PATH", str(work / "nnunet_retrain"))
    ).expanduser()
    os.environ["nnUNet_raw"] = str(retrain)
    os.environ["nnUNet_preprocessed"] = str(retrain / "nnUNet_preprocessed")
    os.environ["nnUNet_results"] = str(retrain / "nnUNet_results")
    os.environ.setdefault("nnUNet_compile", "false")
    os.environ["NNUNET_RETRAIN_PATH"] = str(retrain)
    dataset = (work / "Dataset650_TotalSegmentator").resolve()
    if dataset.is_dir():
        stale = os.getenv("DATASET_FOLDER", "").strip()
        if stale and Path(stale).expanduser().resolve() != dataset:
            print(
                f"NOTE: ignoring stale DATASET_FOLDER={stale}\n"
                f"      using Test6 dataset {dataset}"
            )
        os.environ["DATASET_FOLDER"] = str(dataset)
    else:
        os.environ["DATASET_FOLDER"] = str(
            Path(os.getenv("DATASET_FOLDER", str(dataset))).expanduser()
        )
    os.environ["DATASET_ID"] = os.getenv("DATASET_ID", "650")
    organ = work / "organ_dictionary_test5.json"
    if organ.is_file():
        os.environ["ORGAN_DICTIONARY_PATH"] = str(organ.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description="Test6: STU-Net evaluate")
    parser.add_argument("--work-root", default=str(work_root()))
    parser.add_argument(
        "--variant",
        default=os.getenv("TEST6_STU_VARIANT", default_variant()),
        choices=sorted(TRAINER_FT),
    )
    parser.add_argument("--fold", type=int, default=int(os.getenv("NNUNET_FOLD", "0")))
    parser.add_argument("--dataset-id", default=os.getenv("DATASET_ID", "650"))
    parser.add_argument(
        "--skip-predict",
        action="store_true",
        help="Only run Dice (predictions already exist)",
    )
    parser.add_argument("--viz", action="store_true", help="Also run evaluation_visualization")
    args = parser.parse_args()

    work = Path(args.work_root).expanduser().resolve()
    _ensure_env(work)
    trainer = TRAINER_FT[args.variant]

    dataset = Path(os.environ["DATASET_FOLDER"])
    images_ts = dataset / "imagesTs"
    # Keep predictions under Test6 work (never write into Test5 Dataset650)
    pred_dir = Path(
        os.getenv(
            "TEST6_PRED_DIR",
            str(work / "predictions" / "labelsTs_predicted"),
        )
    ).expanduser()
    pred_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_predict:
        print(f"Using Python: {sys.executable}")
        cmd, env = nnunet_cmd(
            "nnUNetv2_predict",
            "-i",
            str(images_ts),
            "-o",
            str(pred_dir),
            "-d",
            str(args.dataset_id),
            "-c",
            "3d_fullres",
            "-f",
            str(args.fold),
            "-tr",
            trainer,
            "--disable_tta",
            work=work,
        )
        print("+", " ".join(cmd))
        subprocess.check_call(cmd, env=env)
    else:
        print(f"Skipping predict — using {pred_dir}")

    # Keep logs + dice viz under Test6 (not inside Test5 Dataset650)
    os.environ["LOG_DIR"] = str(
        Path(os.environ["NNUNET_RETRAIN_PATH"]) / "logs"
    )
    Path(os.environ["LOG_DIR"]).mkdir(parents=True, exist_ok=True)
    os.environ["NNUNET_EVAL_OUTPUT_DIR"] = str(work / "predictions")

    from nnunet_training.config import TrainingConfig
    from nnunet_training.predict_and_evaluate import (
        evaluate_predictions,
        evaluation_visualization,
    )

    config = TrainingConfig()
    print("\nEvaluating Dice (overall + cohort if ts_case_map.json present)…")
    evaluate_predictions(config, str(pred_dir))

    if args.viz:
        print("\nEvaluation visualization…")
        evaluation_visualization(config)

    print(f"\nDone. Logs: {os.environ['LOG_DIR']}")
    print(f"Predictions / viz: {work / 'predictions'}")


if __name__ == "__main__":
    main()
