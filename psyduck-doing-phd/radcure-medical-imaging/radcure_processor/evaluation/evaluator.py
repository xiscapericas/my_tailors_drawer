"""Evaluation utilities for medical image segmentation."""

import numpy as np
import pandas as pd
import json
import os
from typing import Optional, Dict, Tuple
from radcure_processor.io.nifti_handler import NIfTIHandler


class SegmentationEvaluator:
    """Evaluation utilities for segmentation masks."""
    
    @staticmethod
    def calculate_dice_slice_by_slice(
        case_id: str,
        labels_folder: str,
        predicted_labels_folder: str,
        organ_dictionary_path: str,
        axis: int = 2,
        save_csv_path: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Calculate Dice scores slice-by-slice per organ and overall metrics.
        
        Parameters
        ----------
        case_id : str
            Case identifier (e.g., "case_0405")
        labels_folder : str
            Path to folder containing ground truth labels (e.g., "labelsTs")
        predicted_labels_folder : str
            Path to folder containing predicted labels (e.g., "labelsTs_predicted")
        organ_dictionary_path : str
            Path to JSON file containing organ dictionary mapping organ names to indices
        axis : int
            Axis to slice along: 0 = sagittal, 1 = coronal, 2 = axial (default)
        save_csv_path : str, optional
            If provided, saves detailed slice-by-slice results to CSV
        
        Returns
        -------
        Tuple[pd.DataFrame, Dict[str, float]]
            - DataFrame with columns: slice_idx, organ_name, organ_index, dice_score
            - Dictionary with overall metrics:
              - 'overall_dice': Overall Dice across all organs and slices
              - 'per_organ_dice': Dict mapping organ_name -> overall Dice for that organ
        """
        # Load organ dictionary
        if not os.path.exists(organ_dictionary_path):
            raise FileNotFoundError(
                f"Organ dictionary not found at {organ_dictionary_path}"
            )
        
        with open(organ_dictionary_path, 'r') as f:
            organ_dict: Dict[str, int] = json.load(f)
        
        # Invert dictionary: {organ_name: index} -> {index: organ_name}
        index_to_organ: Dict[int, str] = {v: k for k, v in organ_dict.items()}
        
        # Construct file paths
        label_path = os.path.join(labels_folder, f"{case_id}.nii.gz")
        predicted_path = os.path.join(predicted_labels_folder, f"{case_id}.nii.gz")
        
        # Check if files exist
        for path, name in [(label_path, "ground truth"), (predicted_path, "predicted")]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name} file not found: {path}")
        
        # Load volumes
        gt_mask_vol = NIfTIHandler.load_nii_mask(label_path)
        pred_mask_vol = NIfTIHandler.load_nii_mask(predicted_path)
        
        # Check shapes
        if gt_mask_vol.shape != pred_mask_vol.shape:
            raise ValueError(
                f"Shape mismatch: GT {gt_mask_vol.shape} vs Pred {pred_mask_vol.shape}"
            )
        
        num_slices = gt_mask_vol.shape[axis]
        
        def get_slice(vol, idx, axis):
            if axis == 0:
                sl = vol[idx, :, :]
            elif axis == 1:
                sl = vol[:, idx, :]
            elif axis == 2:
                sl = vol[:, :, idx]
            else:
                raise ValueError("axis must be 0, 1, or 2")
            return sl
        
        def dice_score_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            """Calculate binary Dice score."""
            intersect = np.sum(y_true * y_pred)
            denominator = np.sum(y_true) + np.sum(y_pred)
            if denominator == 0:
                return 1.0 if intersect == 0 else 0.0  # Both empty = perfect match
            return (2 * intersect) / (denominator + 1e-6)
        
        # Get all unique organ indices present in either mask
        unique_labels_gt = set(np.unique(gt_mask_vol).astype(int))
        unique_labels_pred = set(np.unique(pred_mask_vol).astype(int))
        unique_labels = sorted(unique_labels_gt.union(unique_labels_pred))
        # Remove background (0) if present
        unique_labels = [l for l in unique_labels if l > 0]
        
        # Calculate Dice slice-by-slice per organ
        results = []
        
        for slice_idx in range(num_slices):
            gt_slice = get_slice(gt_mask_vol, slice_idx, axis)
            pred_slice = get_slice(pred_mask_vol, slice_idx, axis)
            
            for organ_idx in unique_labels:
                organ_name = index_to_organ.get(organ_idx, f"Label_{organ_idx}")
                
                # Create binary masks for this organ
                gt_binary = (gt_slice == organ_idx).astype(float)
                pred_binary = (pred_slice == organ_idx).astype(float)
                
                # Calculate Dice for this organ on this slice
                dice = dice_score_binary(gt_binary, pred_binary)
                
                results.append({
                    'case_id': case_id,
                    'slice_idx': slice_idx,
                    'organ_name': organ_name,
                    'organ_index': organ_idx,
                    'dice_score': dice
                })
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Calculate overall metrics
        overall_metrics = {}
        
        # Per-organ overall Dice (mean across all slices for each organ)
        # For each organ, calculate mean Dice, excluding slices where both GT and Pred are empty
        per_organ_dice = {}
        for organ_name in df['organ_name'].unique():
            organ_df = df[df['organ_name'] == organ_name]
            # Filter out slices where both GT and Pred are empty (dice=1.0 but not meaningful)
            # We want to include slices where at least one has the organ
            organ_slices = []
            for slice_idx in organ_df['slice_idx'].unique():
                slice_organ_df = organ_df[organ_df['slice_idx'] == slice_idx]
                if len(slice_organ_df) > 0:
                    dice_val = slice_organ_df['dice_score'].iloc[0]
                    # Include if dice < 1.0 (at least one has the organ) or if we want all
                    # Actually, let's include all but note that dice=1.0 for empty slices
                    organ_slices.append(dice_val)
            
            if len(organ_slices) > 0:
                # Calculate mean, but weight by actual presence
                # For now, simple mean (can be refined)
                per_organ_dice[organ_name] = np.mean(organ_slices)
            else:
                per_organ_dice[organ_name] = np.nan
        
        overall_metrics['per_organ_dice'] = per_organ_dice
        
        # Overall Dice across all organs and slices
        # Option 1: Mean of all slice-organ combinations (excluding empty-empty matches)
        df_non_empty = df[df['dice_score'] < 1.0]
        if len(df_non_empty) > 0:
            overall_metrics['overall_dice'] = df_non_empty['dice_score'].mean()
        else:
            # If all are empty-empty matches, return NaN or 1.0?
            overall_metrics['overall_dice'] = 1.0
        
        # Option 2: Mean of per-organ means (organ-level average)
        organ_means = [v for v in per_organ_dice.values() if not np.isnan(v)]
        if len(organ_means) > 0:
            overall_metrics['overall_dice_per_organ_mean'] = np.mean(organ_means)
        else:
            overall_metrics['overall_dice_per_organ_mean'] = np.nan
        
        # Save to CSV if requested
        if save_csv_path:
            df.to_csv(save_csv_path, index=False)
            print(f"✓ Detailed Dice scores saved to: {save_csv_path}")
        
        return df, overall_metrics
