"""Tests for Test 8.0 PET alignment and nnUNet case counting."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import nibabel as nib
import SimpleITK as sitk

from image_processor.io.pet_align import (
    build_pet_nnunet_channel,
    hecktor_slices_to_use,
    resample_pet_to_ct,
    sitk_to_xyz,
)
from image_processor.visualization.visualizer import _pet_display_slice
from nnunet_training.prepare_dataset import _channel_names_from_images, _count_training_cases
from pipelines.test8_0.build_dataset import hecktor_rows_from_case_map


def _write_sitk(path: Path, size, spacing, value=1.0) -> None:
    img = sitk.Image(size, sitk.sitkFloat32)
    img.SetSpacing(spacing)
    img.SetOrigin((0.0, 0.0, 0.0))
    arr = sitk.GetArrayFromImage(img)
    arr[:] = value
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(img)
    sitk.WriteImage(out, str(path))


class TestHecktorSlices(unittest.TestCase):
    def test_empty_mask_uses_all_z(self):
        mask = np.zeros((8, 8, 5), dtype=np.int32)
        self.assertEqual(hecktor_slices_to_use(mask, slice_expansion=5), list(range(5)))

    def test_expands_around_tumor(self):
        mask = np.zeros((8, 8, 20), dtype=np.int32)
        mask[:, :, 10] = 1
        slices = hecktor_slices_to_use(mask, slice_expansion=2)
        self.assertEqual(slices[0], 8)
        self.assertEqual(slices[-1], 12)
        self.assertEqual(len(slices), 5)


class TestPetAlign(unittest.TestCase):
    def test_resample_pet_to_ct_matches_ct_size(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            ct_path = tmp / "c__CT.nii.gz"
            pet_path = tmp / "c__PT.nii.gz"
            _write_sitk(ct_path, (16, 16, 8), (1.0, 1.0, 2.0), value=50.0)
            _write_sitk(pet_path, (8, 8, 4), (2.0, 2.0, 4.0), value=3.5)
            pet_on_ct = resample_pet_to_ct(pet_path, ct_path)
            ct = sitk.ReadImage(str(ct_path))
            self.assertEqual(pet_on_ct.GetSize(), ct.GetSize())
            self.assertEqual(pet_on_ct.GetSpacing(), ct.GetSpacing())
            arr = sitk_to_xyz(pet_on_ct)
            self.assertEqual(arr.shape, (16, 16, 8))
            self.assertAlmostEqual(float(np.nanmean(arr)), 3.5, delta=0.2)

    def test_build_pet_channel_shape_and_missing_file(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            ct_path = tmp / "CHUM-001__CT.nii.gz"
            pet_path = tmp / "CHUM-001__PT.nii.gz"
            mask_path = tmp / "CHUM-001.nii.gz"
            dest = tmp / "case_hek_CHUM_001_0001.nii.gz"
            _write_sitk(ct_path, (12, 12, 10), (1.0, 1.0, 1.0), value=1.0)
            _write_sitk(pet_path, (12, 12, 10), (1.0, 1.0, 1.0), value=4.0)
            mask = np.zeros((12, 12, 10), dtype=np.int16)
            mask[:, :, 4:7] = 1
            nib.save(nib.Nifti1Image(mask, np.eye(4)), str(mask_path))

            info = build_pet_nnunet_channel(
                pet_path, ct_path, mask_path, dest, expected_shape=None, slice_expansion=1
            )
            self.assertTrue(dest.is_file())
            self.assertEqual(info["n_slices"], 5)
            self.assertEqual(info["shape"][2], 5)

            with self.assertRaises(FileNotFoundError):
                build_pet_nnunet_channel(
                    tmp / "missing__PT.nii.gz",
                    ct_path,
                    mask_path,
                    tmp / "out_0001.nii.gz",
                )


class TestDatasetHelpers(unittest.TestCase):
    def test_count_training_cases_two_channels(self):
        with TemporaryDirectory() as td:
            images = Path(td) / "imagesTr"
            images.mkdir()
            (images / "case_a_0000.nii.gz").write_bytes(b"x")
            (images / "case_a_0001.nii.gz").write_bytes(b"x")
            (images / "case_b_0000.nii.gz").write_bytes(b"x")
            (images / "case_b_0001.nii.gz").write_bytes(b"x")
            self.assertEqual(_count_training_cases(str(images)), 2)
            self.assertEqual(
                _channel_names_from_images(str(images)), {0: "CT", 1: "PET"}
            )

    def test_hecktor_rows_drop_radcure(self):
        case_map = {
            "case_0001": {
                "cohort": "radcure",
                "case_id": "RADCURE-0001",
                "split": "Ts",
            },
            "case_hek_CHUM_001": {
                "cohort": "hecktor",
                "case_id": "CHUM-001",
                "split": "Tr",
            },
        }
        rows = hecktor_rows_from_case_map(case_map)
        self.assertEqual(list(rows), ["case_hek_CHUM_001"])


class TestPetDisplay(unittest.TestCase):
    def test_clips_hot_voxel(self):
        sl = np.array([[0.0, 1.0], [2.0, 1000.0]], dtype=np.float32)
        shown, vmax = _pet_display_slice(sl)
        self.assertLess(vmax, 1000.0)
        self.assertLessEqual(shown.max(), vmax)
        self.assertGreaterEqual(shown.min(), 0.0)


if __name__ == "__main__":
    unittest.main()
