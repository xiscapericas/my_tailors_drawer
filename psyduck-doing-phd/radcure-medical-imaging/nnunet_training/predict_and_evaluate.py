"""
Predict on test set and evaluate model performance.

This module handles:
- Running predictions on test set
- Computing evaluation metrics (Dice, Surface Dice)
"""

import os
import json
import subprocess
import sys
from pathlib import Path
from functools import partial
from typing import Dict, List, Optional
import numpy as np
import nibabel as nib
import pandas as pd

# Try to load from .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from nnunet_training.config import TrainingConfig
from image_processor.io.slicer_prediction_export import (
    export_test_set_predictions_for_slicer,
)
from image_processor.evaluation.dsc_agg import (
    binary_confusion_counts,
    dsc_agg_from_totals,
)

from tqdm import tqdm
from p_tqdm import p_map
from surface_distance import compute_surface_distances, compute_surface_dice_at_tolerance

# Import visualization and evaluation modules
try:
    from image_processor.visualization import MedicalImageVisualizer
    from image_processor.evaluation import SegmentationEvaluator
except ImportError as e:
    print(f"Warning: Could not import visualization/evaluation modules: {e}")
    MedicalImageVisualizer = None
    SegmentationEvaluator = None


def add_nnunet_to_path(nnunet_path: str):
    """Add nnUNet to Python path."""
    if nnunet_path not in sys.path:
        sys.path.append(nnunet_path)


def get_predictions_output_dir(config: TrainingConfig) -> str:
    """Directory for nnUNet prediction masks (labelsTs_predicted)."""
    if config.eval_output_dir:
        return os.path.join(config.eval_output_dir, "labelsTs_predicted")
    return os.path.join(config.dataset_folder, "labelsTs_predicted")


def get_slicer_predictions_output_dir(config: TrainingConfig) -> str:
    """Multiclass prediction NIfTIs aligned to CT/PET for 3D Slicer."""
    return os.path.join(get_eval_viz_output_dir(config), "predictions_nifti")


def export_predictions_for_slicer(config: TrainingConfig, pred_dir: str) -> str:
    """Copy nnUNet masks onto the CT grid; integers match dataset.json labels."""
    paths = config.get_dataset_paths()
    dest = Path(get_slicer_predictions_output_dir(config))
    dataset_json = Path(config.dataset_folder) / "dataset.json"
    n = export_test_set_predictions_for_slicer(
        images_ts=paths["imagesTs"],
        pred_dir=pred_dir,
        dest_dir=dest,
        dataset_json_path=dataset_json,
    )
    print(f"✓ 3D Slicer prediction NIfTIs: {n} case(s) → {dest}")
    print(f"  Label IDs: {dest / 'dataset_labels.json'} (from {dataset_json})")
    return str(dest)


def get_eval_viz_output_dir(config: TrainingConfig) -> str:
    """Directory for Dice scores and comparison PDFs (labelsTs_dice_and_viz)."""
    if config.eval_output_dir:
        return os.path.join(config.eval_output_dir, "labelsTs_dice_and_viz")
    return os.path.join(config.dataset_folder, "labelsTs_dice_and_viz")


def predict_on_test_set(
    config: TrainingConfig,
    log_file: str = None,
    save_probabilities: bool = False,
):
    """
    Run predictions on test set.

    Parameters
    ----------
    config : TrainingConfig
        Configuration object
    log_file : str, optional
        Path to log file. If None, uses default.
    save_probabilities : bool
        If True, pass ``--save_probabilities`` so nnUNetv2 also writes
        per-class softmax ``.npz`` next to hard masks (used by Test7).
        Can also be enabled with env ``NNUNET_SAVE_PROBABILITIES=1``.
    """
    if log_file is None:
        log_file = os.path.join(config.log_dir, f'prediction_d{config.dataset_id}.log')

    env_probs = os.getenv("NNUNET_SAVE_PROBABILITIES", "").lower() in (
        "1",
        "true",
        "yes",
    )
    save_probabilities = bool(save_probabilities or env_probs)

    paths = config.get_dataset_paths()
    input_dir = paths['imagesTs']
    output_dir = get_predictions_output_dir(config)

    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Test images folder not found: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Running predictions on test set...")
    print(f"  Input: {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Dataset ID: {config.dataset_id}")
    print(f"  Configuration: {config.configuration}")
    print(f"  Trainer: {config.trainer}")
    print(f"  Fold: {config.prediction_fold}")
    print(f"  TTA: {'enabled' if not config.disable_tta else 'disabled'}")
    print(f"  Save probabilities: {save_probabilities}")
    print(f"  Log file: {log_file}")

    cmd = [
        'nnUNetv2_predict',
        '-i', input_dir,
        '-o', output_dir,
        '-d', str(config.dataset_id),
        '-c', config.configuration,
        '-tr', config.trainer,
        '-f', str(config.prediction_fold)
    ]

    if config.disable_tta:
        cmd.append('--disable_tta')
    if save_probabilities:
        # nnUNetv2 CLI uses long-form --save_probabilities (not -save_probabilities)
        cmd.append('--save_probabilities')

    print(f"Running: {' '.join(cmd)}")

    with open(log_file, 'w') as log:
        result = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True
        )

    if result.returncode == 0:
        print(f"✓ Predictions completed successfully")
        print(f"  Predictions saved to: {output_dir}")
        print(f"  Log saved to: {log_file}")
    else:
        print(f"✗ Prediction failed (exit code: {result.returncode})")
        print(f"  Check log file: {log_file}")
        raise RuntimeError(f"Prediction failed. Check {log_file} for details.")

    return output_dir


def dice_score_2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Binary Dice score.
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary mask
    y_pred : np.ndarray
        Predicted binary mask
    
    Returns
    -------
    float
        Dice score
    """
    intersect = np.sum(y_true * y_pred)
    denominator = np.sum(y_true) + np.sum(y_pred)
    f1 = (2 * intersect) / (denominator + 1e-6)
    return f1


def calc_metrics(
    subject: str,
    gt_dir: Path,
    pred_dir: Path,
    class_map: Dict[int, str],
    surface_distance_tolerance: float = 3.0
) -> Dict:
    """
    Calculate metrics for a single subject.
    
    Parameters
    ----------
    subject : str
        Subject ID (filename without extension)
    gt_dir : Path
        Path to ground truth labels directory
    pred_dir : Path
        Path to predicted labels directory
    class_map : Dict[int, str]
        Mapping from label index to organ name
    surface_distance_tolerance : float
        Tolerance for surface dice calculation
    
    Returns
    -------
    Dict
        Dictionary with metrics for each organ
    """
    gt_all = nib.load(gt_dir / f"{subject}.nii.gz").get_fdata()
    pred_all = nib.load(pred_dir / f"{subject}.nii.gz").get_fdata()
    
    r = {"subject": subject}
    for idx, roi_name in class_map.items():
        gt = gt_all == idx
        pred = pred_all == idx
        
        if gt.max() > 0 and pred.max() == 0:
            # Ground truth exists but prediction is empty
            r[f"dice-{roi_name}"] = 0.0
            r[f"surface_dice_3-{roi_name}"] = 0.0
        elif gt.max() > 0:
            # Both exist, calculate metrics
            r[f"dice-{roi_name}"] = dice_score_2(gt, pred)
            sd = compute_surface_distances(gt, pred, [1.5, 1.5, 1.5])
            r[f"surface_dice_3-{roi_name}"] = compute_surface_dice_at_tolerance(
                sd, 
                surface_distance_tolerance
            )
        else:
            # Ground truth doesn't exist (organ not in image)
            r[f"dice-{roi_name}"] = np.nan
            r[f"surface_dice_3-{roi_name}"] = np.nan

        # HECKTOR DSCagg counts: same binary masks as Dice (label == class index).
        # Count every subject, including GT-empty (per-patient Dice is NaN).
        tp, fp, fn = binary_confusion_counts(pred, gt)
        r[f"tp-{roi_name}"] = tp
        r[f"fp-{roi_name}"] = fp
        r[f"fn-{roi_name}"] = fn
    
    return r


def _gtvp_label_index(config: TrainingConfig) -> Optional[int]:
    """GTVp integer from the organ dictionary / dataset labels (not assumed)."""
    raw = config.labels.get("GTVp")
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        return int(raw[0]) if raw else None
    return int(raw)


def _print_gtvp_test_summary(res_df: pd.DataFrame, gtvp_name: str = "GTVp") -> None:
    """Mean/std of per-patient Dice plus HECKTOR DSCagg (cohort voxel totals)."""
    dice_col = f"dice-{gtvp_name}"
    tp_key, fp_key, fn_key = f"tp-{gtvp_name}", f"fp-{gtvp_name}", f"fn-{gtvp_name}"
    if dice_col not in res_df.columns or tp_key not in res_df.columns:
        print("⚠️  GTVp columns missing; skip DSCagg summary")
        return

    n_subjects = len(res_df)
    patient_dice = res_df[dice_col].dropna()
    mean_dice = float(patient_dice.mean()) if len(patient_dice) else float("nan")
    std_dice = float(patient_dice.std(ddof=1)) if len(patient_dice) > 1 else float("nan")

    # Reset cohort totals to zero, then sum every test subject (including GT-empty).
    tp_total = np.int64(0)
    fp_total = np.int64(0)
    fn_total = np.int64(0)
    tp_total = tp_total + np.int64(np.nansum(res_df[tp_key].to_numpy(dtype=np.int64)))
    fp_total = fp_total + np.int64(np.nansum(res_df[fp_key].to_numpy(dtype=np.int64)))
    fn_total = fn_total + np.int64(np.nansum(res_df[fn_key].to_numpy(dtype=np.int64)))
    dsc_agg = dsc_agg_from_totals(tp_total, fp_total, fn_total)

    print("\n---------------- Test segmentation results ----------------")
    print(f"Number of subjects: {n_subjects}")
    print(f"Mean GTVp Dice: {mean_dice:.4f}")
    print(f"Std GTVp Dice: {std_dice:.4f}")
    print(f"GTVp DSCagg: {dsc_agg:.4f}")
    print("-----------------------------------------------------------")
    print(
        f"  (DSCagg from TP={tp_total} FP={fp_total} FN={fn_total}; "
        f"mean/std over n={len(patient_dice)} cases with GTVp in GT)"
    )

    if "cohort" in res_df.columns:
        for cohort in sorted(res_df["cohort"].dropna().unique()):
            sub = res_df[res_df["cohort"] == cohort]
            d = sub[dice_col].dropna()
            tp = np.int64(np.nansum(sub[tp_key].to_numpy(dtype=np.int64)))
            fp = np.int64(np.nansum(sub[fp_key].to_numpy(dtype=np.int64)))
            fn = np.int64(np.nansum(sub[fn_key].to_numpy(dtype=np.int64)))
            std_c = float(d.std(ddof=1)) if len(d) > 1 else float("nan")
            mean_c = float(d.mean()) if len(d) else float("nan")
            print(
                f"  cohort={cohort}: n={len(sub)} "
                f"mean={mean_c:.4f} std={std_c:.4f} "
                f"DSCagg={dsc_agg_from_totals(tp, fp, fn):.4f}"
            )


def evaluate_predictions(config: TrainingConfig, pred_dir: str):
    """
    Evaluate predictions against ground truth.
    
    Parameters
    ----------
    config : TrainingConfig
        Configuration object
    pred_dir : str
        Path to predictions directory
    """
    paths = config.get_dataset_paths()
    gt_dir = Path(paths['labelsTs'])
    pred_path = Path(pred_dir)
    
    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground truth labels folder not found: {gt_dir}")
    
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions folder not found: {pred_path}")
    
    # Get subjects
    subjects = [x.stem.split(".")[0] for x in gt_dir.glob("*.nii.gz")]
    
    if len(subjects) == 0:
        raise ValueError(f"No ground truth files found in {gt_dir}")
    
    print(f"\nEvaluating {len(subjects)} subjects...")
    print(f"  Ground truth: {gt_dir}")
    print(f"  Predictions: {pred_path}")
    print(f"  Number of CPUs: {config.evaluation_num_cpus}")
    
    # Create class map from organ dictionary (invert: {name: idx} -> {idx: name})
    class_map = {idx: name for name, idx in config.labels.items()}
    
    # Calculate metrics for all subjects
    print("Computing metrics...")
    res = p_map(
        partial(
            calc_metrics, 
            gt_dir=gt_dir, 
            pred_dir=pred_path, 
            class_map=class_map,
            surface_distance_tolerance=config.surface_distance_tolerance
        ),
        subjects,
        num_cpus=config.evaluation_num_cpus,
        disable=False
    )
    
    # Convert to DataFrame
    res_df = pd.DataFrame(res)

    # Optional cohort / original-id columns (Test5: ts_case_map.json / case_map.json)
    map_path = Path(config.dataset_folder) / "ts_case_map.json"
    if not map_path.is_file():
        map_path = Path(config.dataset_folder) / "case_map.json"
    if map_path.is_file() and "subject" in res_df.columns:
        try:
            with open(map_path, encoding="utf-8") as f:
                ts_map = json.load(f)
            res_df["cohort"] = res_df["subject"].map(
                lambda s: (ts_map.get(s) or {}).get("cohort", "unknown")
            )
            res_df["case_id"] = res_df["subject"].map(
                lambda s: (ts_map.get(s) or {}).get("case_id")
            )
            res_df["center"] = res_df["subject"].map(
                lambda s: (ts_map.get(s) or {}).get("center")
            )
        except Exception as exc:
            print(f"⚠️  Could not load case map for cohort split: {exc}")
    
    # Print results
    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)
    
    for metric in ["dice", "surface_dice_3"]:
        print(f"\n{metric.upper()} Scores (all test cases):")
        res_all_rois = []
        for roi_name in class_map.values():
            col_name = f"{metric}-{roi_name}"
            if col_name in res_df.columns:
                row_wo_nan = res_df[col_name].dropna()
                if len(row_wo_nan) > 0:
                    mean_score = row_wo_nan.mean()
                    res_all_rois.append(mean_score)
                    print(f"  {roi_name:30s}: {mean_score:.4f} (n={len(row_wo_nan)})")
        
        if res_all_rois:
            overall_mean = np.array(res_all_rois).mean()
            print(f"  {'Overall Mean':30s}: {overall_mean:.4f}")

        if "cohort" in res_df.columns:
            for cohort in sorted(res_df["cohort"].dropna().unique()):
                sub = res_df[res_df["cohort"] == cohort]
                print(f"\n{metric.upper()} — cohort={cohort} (n_subjects={len(sub)}):")
                for roi_name in class_map.values():
                    col_name = f"{metric}-{roi_name}"
                    if col_name not in sub.columns:
                        continue
                    row_wo_nan = sub[col_name].dropna()
                    if len(row_wo_nan) > 0:
                        print(
                            f"  {roi_name:30s}: {row_wo_nan.mean():.4f} "
                            f"(n={len(row_wo_nan)})"
                        )
    
    # Save results to CSV
    results_file = os.path.join(config.log_dir, f'evaluation_d{config.dataset_id}.csv')
    res_df.to_csv(results_file, index=False)
    print(f"\n✓ Detailed results saved to: {results_file}")

    gtvp_idx = _gtvp_label_index(config)
    if gtvp_idx is None:
        print("⚠️  Organ dictionary has no GTVp; skip HECKTOR DSCagg summary")
    else:
        print(f"GTVp label index (from organ dictionary): {gtvp_idx}")
        _print_gtvp_test_summary(res_df, gtvp_name="GTVp")

    try:
        export_predictions_for_slicer(config, str(pred_path))
    except Exception as exc:
        print(f"⚠️  3D Slicer NIfTI export skipped: {exc}")
    
    return res_df


def main():
    """Main function for prediction and evaluation."""
    print("=" * 70)
    print("nnUNet Prediction and Evaluation")
    print("=" * 70)
    
    # Load configuration
    config = TrainingConfig()
    
    # Add nnUNet to path
    add_nnunet_to_path(config.nnunet_path)
    
    # Setup environment
    config.setup_nnunet_environment()
    
    # Run predictions
    print("\nStep 1: Running predictions on test set...")
    pred_dir = predict_on_test_set(config)
    
    # Evaluate predictions
    print("\nStep 2: Evaluating predictions...")
    results_df = evaluate_predictions(config, pred_dir)
    
    print("\n" + "=" * 70)
    print("Evaluation complete!")
    print("=" * 70)


def main_eval_only():
    """Score existing ``labelsTs_predicted`` NIfTIs (no nnUNet predict)."""
    print("=" * 70)
    print("nnUNet Evaluation (existing predictions)")
    print("=" * 70)

    config = TrainingConfig()
    add_nnunet_to_path(config.nnunet_path)
    config.setup_nnunet_environment()

    pred_dir = get_predictions_output_dir(config)
    print(f"Using predictions: {pred_dir}")
    evaluate_predictions(config, pred_dir)

    print("\n" + "=" * 70)
    print("Evaluation complete!")
    print("=" * 70)


def evaluation_visualization(config: TrainingConfig):
    """
    Generate Dice score calculations and visualization comparisons for all test cases.
    
    This function:
    1. Calculates Dice scores slice-by-slice per organ for each test case
    2. Creates visualization PDFs comparing GT vs predicted masks
    3. Exports full multiclass prediction NIfTIs aligned to CT/PET (3D Slicer)
    4. Saves all results in labelsTs_dice_and_viz folder
    
    Parameters
    ----------
    config : TrainingConfig
        Configuration object
    """
    if MedicalImageVisualizer is None or SegmentationEvaluator is None:
        raise ImportError(
            "Visualization and evaluation modules not available. "
            "Ensure image_processor is properly installed."
        )
    
    paths = config.get_dataset_paths()
    labels_ts_dir = Path(paths['labelsTs'])
    images_ts_dir = Path(paths['imagesTs'])
    pred_dir = Path(get_predictions_output_dir(config))
    output_dir = Path(get_eval_viz_output_dir(config))
    
    # Validate paths
    if not labels_ts_dir.exists():
        raise FileNotFoundError(f"Ground truth labels folder not found: {labels_ts_dir}")
    
    if not pred_dir.exists():
        raise FileNotFoundError(
            f"Predictions folder not found: {pred_dir}. "
            "Please run prediction step first."
        )
    
    if not images_ts_dir.exists():
        raise FileNotFoundError(f"Test images folder not found: {images_ts_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    dice_output_dir = output_dir / 'dice_scores'
    surface_dice_output_dir = output_dir / 'surface_dice_scores'
    viz_output_dir = output_dir / 'visualizations'
    dice_output_dir.mkdir(exist_ok=True)
    surface_dice_output_dir.mkdir(exist_ok=True)
    viz_output_dir.mkdir(exist_ok=True)
    slicer_pred_dir = Path(get_slicer_predictions_output_dir(config))
    slicer_pred_dir.mkdir(exist_ok=True)
    try:
        export_predictions_for_slicer(config, str(pred_dir))
    except Exception as exc:
        print(f"⚠️  3D Slicer NIfTI export skipped: {exc}")
    
    # Get all test cases
    test_cases = [x.stem.split(".")[0] for x in labels_ts_dir.glob("*.nii.gz")]
    
    if len(test_cases) == 0:
        raise ValueError(f"No test cases found in {labels_ts_dir}")
    
    print(f"\nGenerating evaluation visualizations for {len(test_cases)} test cases...")
    print(f"  Ground truth: {labels_ts_dir}")
    print(f"  Predictions: {pred_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Organ dictionary: {config.organ_dictionary_path}")
    print(f"  Surface DICE tolerance: {getattr(config, 'surface_distance_tolerance', 3.0)} mm")
    
    # Initialize evaluator and visualizer
    evaluator = SegmentationEvaluator()
    visualizer = MedicalImageVisualizer()
    
    # Process each test case
    successful = 0
    failed = []
    
    # Get spacing and tolerance for Surface DICE
    # Default spacing (1.0, 1.0, 1.0) - should be adjusted based on actual image spacing
    spacing_mm = getattr(config, 'voxel_spacing_mm', (1.0, 1.0, 1.0))
    tolerance_mm = getattr(config, 'surface_distance_tolerance', 3.0)
    
    for case_id in tqdm(test_cases, desc="Processing cases"):
        try:
            # Calculate Dice scores
            dice_csv_path = dice_output_dir / f"{case_id}_dice_scores.csv"
            dice_df, dice_metrics = evaluator.calculate_dice_slice_by_slice(
                case_id=case_id,
                labels_folder=str(labels_ts_dir),
                predicted_labels_folder=str(pred_dir),
                organ_dictionary_path=config.organ_dictionary_path,
                axis=2,  # axial slices
                save_csv_path=str(dice_csv_path)
            )
            
            # Calculate Surface Dice scores
            surface_dice_csv_path = surface_dice_output_dir / f"{case_id}_surface_dice_scores.csv"
            try:
                surface_dice_df, surface_dice_metrics = evaluator.calculate_surface_dice_slice_by_slice(
                    case_id=case_id,
                    labels_folder=str(labels_ts_dir),
                    predicted_labels_folder=str(pred_dir),
                    organ_dictionary_path=config.organ_dictionary_path,
                    axis=2,  # axial slices
                    spacing_mm=spacing_mm,
                    tolerance_mm=tolerance_mm,
                    save_csv_path=str(surface_dice_csv_path)
                )
            except ImportError as e:
                print(f"\n⚠️  Surface DICE calculation skipped for {case_id}: {e}")
                surface_dice_df = None
                surface_dice_metrics = None
            except Exception as e:
                print(f"\n⚠️  Error calculating Surface DICE for {case_id}: {e}")
                surface_dice_df = None
                surface_dice_metrics = None
            
            # Create visualization PDF
            viz_pdf_path = viz_output_dir / f"{case_id}_comparison.pdf"
            visualizer.visualize_prediction_comparison(
                case_id=case_id,
                images_folder=str(images_ts_dir),
                labels_folder=str(labels_ts_dir),
                predicted_labels_folder=str(pred_dir),
                organ_dictionary_path=config.organ_dictionary_path,
                axis=2,  # axial slices
                save_pdf_path=str(viz_pdf_path),
                show=False,  # Don't display, just save
                slice_indices=None  # All slices
            )
            
            successful += 1
            
        except Exception as e:
            print(f"\n⚠️  Error processing {case_id}: {e}")
            failed.append(case_id)
            continue
    
    # Print summary
    print("\n" + "=" * 70)
    print("Evaluation Visualization Summary")
    print("=" * 70)
    print(f"Successfully processed: {successful}/{len(test_cases)} cases")
    if failed:
        print(f"Failed cases: {len(failed)}")
        for case_id in failed:
            print(f"  - {case_id}")
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Dice scores: {dice_output_dir}")
    print(f"  - Surface Dice scores: {surface_dice_output_dir}")
    print(f"  - Visualizations: {viz_output_dir}")
    print(f"  - Slicer prediction NIfTIs: {slicer_pred_dir}")
    print("=" * 70)
    
    return output_dir


def main_visualization():
    """Main function for evaluation visualization."""
    print("=" * 70)
    print("nnUNet Evaluation Visualization")
    print("=" * 70)
    
    # Load configuration
    config = TrainingConfig()
    
    # Run evaluation visualization
    output_dir = evaluation_visualization(config)
    
    print("\n" + "=" * 70)
    print("Evaluation visualization complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()

