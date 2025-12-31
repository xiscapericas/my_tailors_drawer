"""
Predict on test set and evaluate model performance.

This module handles:
- Running predictions on test set
- Computing evaluation metrics (Dice, Surface Dice)
"""

import os
import subprocess
import sys
from pathlib import Path
from functools import partial
from typing import Dict, List
import numpy as np
import nibabel as nib
import pandas as pd
from nnunet_training.config import TrainingConfig

try:
    from tqdm import tqdm
    from p_tqdm import p_map
    from surface_distance import compute_surface_distances, compute_surface_dice_at_tolerance
except ImportError as e:
    print(f"Warning: Required packages not installed: {e}")
    print("Install with: pip install p_tqdm git+https://github.com/google-deepmind/surface-distance.git")
    raise


def add_nnunet_to_path(nnunet_path: str):
    """Add nnUNet to Python path."""
    if nnunet_path not in sys.path:
        sys.path.append(nnunet_path)


def predict_on_test_set(config: TrainingConfig, log_file: str = None):
    """
    Run predictions on test set.
    
    Parameters
    ----------
    config : TrainingConfig
        Configuration object
    log_file : str, optional
        Path to log file. If None, uses default.
    """
    if log_file is None:
        log_file = os.path.join(config.log_dir, f'prediction_d{config.dataset_id}.log')
    
    paths = config.get_dataset_paths()
    input_dir = paths['imagesTs']
    output_dir = os.path.join(config.dataset_folder, 'labelsTs_predicted')
    
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
    
    return r


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
    
    # Print results
    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)
    
    for metric in ["dice", "surface_dice_3"]:
        print(f"\n{metric.upper()} Scores:")
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
    
    # Save results to CSV
    results_file = os.path.join(config.log_dir, f'evaluation_d{config.dataset_id}.csv')
    res_df.to_csv(results_file, index=False)
    print(f"\n✓ Detailed results saved to: {results_file}")
    
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


if __name__ == '__main__':
    main()

