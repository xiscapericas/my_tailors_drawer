"""Evaluation utilities for medical image segmentation."""

import numpy as np
import pandas as pd
import json
import os
from typing import Optional, Dict, Tuple, Union
from radcure_processor.io.nifti_handler import NIfTIHandler

# Try to import surface_distance for Surface DICE calculation
try:
    from surface_distance import compute_surface_distances, compute_surface_dice_at_tolerance
    SURFACE_DICE_AVAILABLE = True
except ImportError:
    SURFACE_DICE_AVAILABLE = False
    print("Warning: surface_distance package not available. Surface DICE calculation will not work.")
    print("Install with: pip install git+https://github.com/google-deepmind/surface-distance.git")


class SegmentationEvaluator:
    """Evaluation utilities for segmentation masks."""
    
    @staticmethod
    def calculate_dice(
        gt_mask: Union[np.ndarray, str],
        pred_mask: Union[np.ndarray, str],
        organ_dictionary_path: Optional[str] = None,
        spacing_mm: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> Dict[str, float]:
        """
        Calculate DICE score per organ for entire 3D volume.
        
        This function calculates DICE scores for each organ present in the masks.
        It works on the entire 3D volume (not slice-by-slice).
        
        Parameters
        ----------
        gt_mask : np.ndarray or str
            Ground truth mask as numpy array or path to NIfTI file
        pred_mask : np.ndarray or str
            Predicted mask as numpy array or path to NIfTI file
        organ_dictionary_path : str, optional
            Path to JSON file containing organ dictionary mapping organ names to indices.
            If provided, results will include organ names. If None, results use label indices.
        spacing_mm : tuple of float, optional
            Voxel spacing in mm (x, y, z). Default is (1.0, 1.0, 1.0).
            Only used for Surface DICE calculation.
        
        Returns
        -------
        Dict[str, float]
            Dictionary mapping organ_name (or "Label_{index}") -> DICE score.
            Also includes 'overall_dice' key with mean DICE across all organs.
        """
        # Load masks if paths are provided
        if isinstance(gt_mask, str):
            gt_mask_vol = NIfTIHandler.load_nii_mask(gt_mask)
        else:
            gt_mask_vol = gt_mask.copy()
        
        if isinstance(pred_mask, str):
            pred_mask_vol = NIfTIHandler.load_nii_mask(pred_mask)
        else:
            pred_mask_vol = pred_mask.copy()
        
        # Check shapes
        if gt_mask_vol.shape != pred_mask_vol.shape:
            raise ValueError(
                f"Shape mismatch: GT {gt_mask_vol.shape} vs Pred {pred_mask_vol.shape}"
            )
        
        # Load organ dictionary if provided
        index_to_organ: Dict[int, str] = {}
        if organ_dictionary_path and os.path.exists(organ_dictionary_path):
            with open(organ_dictionary_path, 'r') as f:
                organ_dict: Dict[str, int] = json.load(f)
            index_to_organ = {v: k for k, v in organ_dict.items()}
        
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
        
        # Calculate DICE per organ
        dice_scores = {}
        
        for organ_idx in unique_labels:
            organ_name = index_to_organ.get(organ_idx, f"Label_{organ_idx}")
            
            # Create binary masks for this organ
            gt_binary = (gt_mask_vol == organ_idx).astype(float)
            pred_binary = (pred_mask_vol == organ_idx).astype(float)
            
            # Calculate DICE for this organ
            dice = dice_score_binary(gt_binary, pred_binary)
            dice_scores[organ_name] = dice
        
        # Calculate overall DICE (mean across all organs)
        if len(dice_scores) > 0:
            dice_scores['overall_dice'] = np.mean(list(dice_scores.values()))
        else:
            dice_scores['overall_dice'] = np.nan
        
        return dice_scores
    
    @staticmethod
    def calculate_surface_dice(
        gt_mask: Union[np.ndarray, str],
        pred_mask: Union[np.ndarray, str],
        organ_dictionary_path: Optional[str] = None,
        spacing_mm: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        tolerance_mm: float = 3.0
    ) -> Dict[str, float]:
        """
        Calculate Surface DICE score per organ for entire 3D volume.
        
        Surface DICE measures the overlap of surface boundaries rather than volumes.
        It's more sensitive to boundary accuracy than regular DICE.
        
        Parameters
        ----------
        gt_mask : np.ndarray or str
            Ground truth mask as numpy array or path to NIfTI file
        pred_mask : np.ndarray or str
            Predicted mask as numpy array or path to NIfTI file
        organ_dictionary_path : str, optional
            Path to JSON file containing organ dictionary mapping organ names to indices.
            If provided, results will include organ names. If None, results use label indices.
        spacing_mm : tuple of float, optional
            Voxel spacing in mm (x, y, z). Default is (1.0, 1.0, 1.0).
            Important: Use correct spacing for accurate Surface DICE calculation.
        tolerance_mm : float, optional
            Distance tolerance in mm for Surface DICE calculation. Default is 3.0 mm.
            This is the maximum distance from the surface that is considered acceptable.
        
        Returns
        -------
        Dict[str, float]
            Dictionary mapping organ_name (or "Label_{index}") -> Surface DICE score.
            Also includes 'overall_surface_dice' key with mean Surface DICE across all organs.
        """
        if not SURFACE_DICE_AVAILABLE:
            raise ImportError(
                "surface_distance package is required for Surface DICE calculation. "
                "Install with: pip install git+https://github.com/google-deepmind/surface-distance.git"
            )
        
        # Load masks if paths are provided
        if isinstance(gt_mask, str):
            gt_mask_vol = NIfTIHandler.load_nii_mask(gt_mask)
        else:
            gt_mask_vol = gt_mask.copy()
        
        if isinstance(pred_mask, str):
            pred_mask_vol = NIfTIHandler.load_nii_mask(pred_mask)
        else:
            pred_mask_vol = pred_mask.copy()
        
        # Check shapes
        if gt_mask_vol.shape != pred_mask_vol.shape:
            raise ValueError(
                f"Shape mismatch: GT {gt_mask_vol.shape} vs Pred {pred_mask_vol.shape}"
            )
        
        # Load organ dictionary if provided
        index_to_organ: Dict[int, str] = {}
        if organ_dictionary_path and os.path.exists(organ_dictionary_path):
            with open(organ_dictionary_path, 'r') as f:
                organ_dict: Dict[str, int] = json.load(f)
            index_to_organ = {v: k for k, v in organ_dict.items()}
        
        # Get all unique organ indices present in either mask
        unique_labels_gt = set(np.unique(gt_mask_vol).astype(int))
        unique_labels_pred = set(np.unique(pred_mask_vol).astype(int))
        unique_labels = sorted(unique_labels_gt.union(unique_labels_pred))
        # Remove background (0) if present
        unique_labels = [l for l in unique_labels if l > 0]
        
        # Calculate Surface DICE per organ
        surface_dice_scores = {}
        
        for organ_idx in unique_labels:
            organ_name = index_to_organ.get(organ_idx, f"Label_{organ_idx}")
            
            # Create binary masks for this organ
            gt_binary = (gt_mask_vol == organ_idx).astype(bool)
            pred_binary = (pred_mask_vol == organ_idx).astype(bool)
            
            # Handle edge cases
            if not gt_binary.any() and not pred_binary.any():
                # Both empty - perfect match
                surface_dice_scores[organ_name] = 1.0
            elif gt_binary.any() and not pred_binary.any():
                # GT exists but prediction is empty
                surface_dice_scores[organ_name] = 0.0
            elif not gt_binary.any() and pred_binary.any():
                # Prediction exists but GT is empty
                surface_dice_scores[organ_name] = 0.0
            else:
                # Both exist - calculate Surface DICE
                try:
                    surface_distances = compute_surface_distances(
                        gt_binary, 
                        pred_binary, 
                        spacing_mm
                    )
                    surface_dice = compute_surface_dice_at_tolerance(
                        surface_distances,
                        tolerance_mm
                    )
                    surface_dice_scores[organ_name] = surface_dice
                except Exception as e:
                    print(f"Warning: Could not calculate Surface DICE for {organ_name}: {e}")
                    surface_dice_scores[organ_name] = np.nan
        
        # Calculate overall Surface DICE (mean across all organs)
        valid_scores = [v for v in surface_dice_scores.values() if not np.isnan(v)]
        if len(valid_scores) > 0:
            surface_dice_scores['overall_surface_dice'] = np.mean(valid_scores)
        else:
            surface_dice_scores['overall_surface_dice'] = np.nan
        
        return surface_dice_scores
    
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
