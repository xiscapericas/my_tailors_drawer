"""Tests for HECKTOR-style aggregated Dice (DSCagg)."""

from __future__ import annotations

import unittest

import numpy as np

from image_processor.evaluation.dsc_agg import (
    binary_confusion_counts,
    dsc_agg_from_totals,
)


class TestDSCagg(unittest.TestCase):
    def test_counts_and_agg_not_equal_to_mean_dice(self):
        # Case A: large overlap. Case B: tiny overlap (low Dice, few voxels).
        gt_a = np.zeros((4, 4), dtype=bool)
        pred_a = np.zeros((4, 4), dtype=bool)
        gt_a[:2, :] = True
        pred_a[:2, :] = True
        tp_a, fp_a, fn_a = binary_confusion_counts(pred_a, gt_a)

        gt_b = np.zeros((4, 4), dtype=bool)
        pred_b = np.zeros((4, 4), dtype=bool)
        gt_b[0, 0] = True
        pred_b[0, 1] = True
        tp_b, fp_b, fn_b = binary_confusion_counts(pred_b, gt_b)

        dsc_agg = dsc_agg_from_totals(tp_a + tp_b, fp_a + fp_b, fn_a + fn_b)
        dice_a = 2 * tp_a / (2 * tp_a + fp_a + fn_a)
        dice_b = 2 * tp_b / (2 * tp_b + fp_b + fn_b)
        mean_dice = 0.5 * (dice_a + dice_b)
        self.assertGreater(dsc_agg, mean_dice)
        self.assertEqual(int(tp_a), 8)
        self.assertEqual(int(tp_b), 0)
        self.assertEqual(int(fp_b), 1)
        self.assertEqual(int(fn_b), 1)

    def test_both_empty_global_is_one(self):
        self.assertEqual(dsc_agg_from_totals(0, 0, 0), 1.0)

    def test_pred_empty_gt_nonempty(self):
        gt = np.array([[1, 1], [0, 0]], dtype=bool)
        pred = np.zeros_like(gt)
        tp, fp, fn = binary_confusion_counts(pred, gt)
        self.assertEqual((int(tp), int(fp), int(fn)), (0, 0, 2))
        self.assertEqual(dsc_agg_from_totals(tp, fp, fn), 0.0)

    def test_pred_nonempty_gt_empty(self):
        gt = np.zeros((2, 2), dtype=bool)
        pred = np.array([[1, 0], [0, 0]], dtype=bool)
        tp, fp, fn = binary_confusion_counts(pred, gt)
        self.assertEqual((int(tp), int(fp), int(fn)), (0, 1, 0))
        self.assertEqual(dsc_agg_from_totals(tp, fp, fn), 0.0)


if __name__ == "__main__":
    unittest.main()
