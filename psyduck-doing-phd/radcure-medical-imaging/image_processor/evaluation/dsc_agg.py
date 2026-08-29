"""HECKTOR-style cohort-level aggregated Dice (DSCagg).

DSCagg is **not** the mean of per-patient Dice. Voxel TP/FP/FN are summed
over the whole cohort, then Dice is computed once:

    DSCagg = 2 * TP_total / (2 * TP_total + FP_total + FN_total)
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np


def binary_confusion_counts(
    pred_binary: np.ndarray,
    gt_binary: np.ndarray,
) -> Tuple[np.int64, np.int64, np.int64]:
    """
    Voxel TP/FP/FN for one subject, as int64.

    Predictions and labels are forced to boolean (no softmax / no extra
    threshold). This matches the existing test eval: ``mask == gtvp_index``.
    """
    pred_b = np.asarray(pred_binary).astype(bool, copy=False)
    gt_b = np.asarray(gt_binary).astype(bool, copy=False)
    if pred_b.shape != gt_b.shape:
        raise ValueError(f"Shape mismatch: pred {pred_b.shape} vs gt {gt_b.shape}")
    tp = np.int64(np.count_nonzero(pred_b & gt_b))
    fp = np.int64(np.count_nonzero(pred_b & ~gt_b))
    fn = np.int64(np.count_nonzero(~pred_b & gt_b))
    return tp, fp, fn


def dsc_agg_from_totals(
    tp_total: Union[int, np.integer],
    fp_total: Union[int, np.integer],
    fn_total: Union[int, np.integer],
) -> float:
    """
    HECKTOR DSCagg from cohort-level counts.

    Denominator 0 (no GTVp voxels and no predicted GTVp voxels in the whole
    cohort): return 1.0, matching ``SegmentationEvaluator.calculate_dice``
    when both binary masks are empty. Per-patient test Dice instead uses
    NaN when GT is empty (those cases are dropped from the *mean*, but their
    voxels still enter DSCagg if the prediction is non-empty).
    """
    tp = np.int64(tp_total)
    fp = np.int64(fp_total)
    fn = np.int64(fn_total)
    denom = np.int64(2) * tp + fp + fn
    if denom == 0:
        return 1.0
    return float((np.int64(2) * tp) / denom)
