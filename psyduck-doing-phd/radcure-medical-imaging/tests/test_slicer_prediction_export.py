"""Tests for 3D Slicer prediction NIfTI export (CT-aligned, dataset.json IDs)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import SimpleITK as sitk

from image_processor.io.slicer_prediction_export import (
    export_prediction_aligned_to_ct,
    export_test_set_predictions_for_slicer,
    load_dataset_json_labels,
)


def _write_sitk(
    path: Path,
    arr_zyx: np.ndarray,
    *,
    origin=(0.0, 0.0, 0.0),
    spacing=(1.0, 1.0, 1.0),
) -> None:
    img = sitk.GetImageFromArray(arr_zyx)
    img.SetOrigin(origin)
    img.SetSpacing(spacing)
    sitk.WriteImage(img, str(path))


class TestSlicerPredictionExport(unittest.TestCase):
    def test_copies_ct_geometry_keeps_all_label_ids(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            ct = np.ones((4, 5, 6), dtype=np.int16)
            pred = np.zeros((4, 5, 6), dtype=np.uint16)
            pred[1, 2, 3] = 9
            pred[2, 2, 3] = 91
            pred[2, 3, 3] = 92
            _write_sitk(
                tmp / "ct.nii.gz",
                ct,
                origin=(10.0, 20.0, 30.0),
                spacing=(1.5, 1.5, 2.0),
            )
            _write_sitk(tmp / "pred.nii.gz", pred)
            dest = tmp / "out.nii.gz"
            export_prediction_aligned_to_ct(
                tmp / "pred.nii.gz",
                tmp / "ct.nii.gz",
                dest,
            )
            out = sitk.ReadImage(str(dest))
            ref = sitk.ReadImage(str(tmp / "ct.nii.gz"))
            self.assertEqual(out.GetSize(), ref.GetSize())
            self.assertEqual(out.GetOrigin(), ref.GetOrigin())
            self.assertEqual(out.GetSpacing(), ref.GetSpacing())
            arr = sitk.GetArrayFromImage(out)
            self.assertEqual(set(np.unique(arr).tolist()), {0, 9, 91, 92})
            np.testing.assert_array_equal(arr, pred)

    def test_resamples_when_pred_shape_differs(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            ct = np.zeros((4, 4, 4), dtype=np.int16)
            pred = np.zeros((2, 4, 4), dtype=np.uint16)
            pred[0, 1, 1] = 91
            _write_sitk(tmp / "ct.nii.gz", ct, spacing=(1.0, 1.0, 1.0))
            _write_sitk(tmp / "pred.nii.gz", pred, spacing=(1.0, 1.0, 2.0))
            dest = tmp / "out.nii.gz"
            export_prediction_aligned_to_ct(
                tmp / "pred.nii.gz",
                tmp / "ct.nii.gz",
                dest,
            )
            out = sitk.ReadImage(str(dest))
            self.assertEqual(tuple(out.GetSize()), (4, 4, 4))
            self.assertIn(91, set(sitk.GetArrayFromImage(out).ravel().tolist()))

    def test_batch_writes_dataset_json_labels(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            images = tmp / "imagesTs"
            preds = tmp / "pred"
            dest = tmp / "slicer"
            images.mkdir()
            preds.mkdir()
            vol = np.zeros((2, 3, 4), dtype=np.uint16)
            vol[0, 0, 0] = 91
            _write_sitk(images / "case_hek_A_0000.nii.gz", vol)
            _write_sitk(images / "case_hek_A_0001.nii.gz", vol.astype(np.float32))
            _write_sitk(preds / "case_hek_A.nii.gz", vol)
            dj = tmp / "dataset.json"
            labels = {"background": 0, "parotid_gland_left": 9, "GTVp": 91, "GTVn": 92}
            with open(dj, "w", encoding="utf-8") as f:
                json.dump({"labels": labels}, f)
            n = export_test_set_predictions_for_slicer(images, preds, dest, dj)
            self.assertEqual(n, 1)
            self.assertTrue((dest / "case_hek_A.nii.gz").is_file())
            self.assertEqual(load_dataset_json_labels(dest / "dataset_labels.json"), labels)


if __name__ == "__main__":
    unittest.main()
