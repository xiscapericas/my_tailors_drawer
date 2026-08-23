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
    slices_matching_test5_crop,
)
from image_processor.visualization.visualizer import _pet_display_slice, _volume_slice_for_display
from nnunet_training.prepare_dataset import (
    _channel_names_from_images,
    _count_training_cases,
    find_incomplete_nnunet_channel_cases,
)
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

    def test_slices_matching_test5_crop_finds_window(self):
        rng = np.random.default_rng(0)
        full = rng.random((16, 16, 20)).astype(np.float32)
        crop = full[:, :, 5:13].copy()
        slices = slices_matching_test5_crop(full, crop)
        self.assertEqual(slices, list(range(5, 13)))

    def test_build_pet_uses_reference_crop_when_mask_window_differs(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            ct_path = tmp / "c__CT.nii.gz"
            pet_path = tmp / "c__PT.nii.gz"
            mask_path = tmp / "c.nii.gz"
            ref_path = tmp / "test5_0000.nii.gz"
            dest = tmp / "out_0001.nii.gz"
            orig = np.zeros((12, 12, 16), dtype=np.float32)
            for z in range(16):
                orig[:, :, z] = float(z + 1)
            nib.save(nib.Nifti1Image(orig, np.eye(4)), str(ct_path))
            _write_sitk(pet_path, (12, 12, 16), (1.0, 1.0, 1.0), value=4.0)
            mask = np.zeros((12, 12, 16), dtype=np.int16)
            mask[:, :, 2:12] = 1
            nib.save(nib.Nifti1Image(mask, np.eye(4)), str(mask_path))
            crop = orig[:, :, 4:8]
            nib.save(nib.Nifti1Image(crop, np.eye(4)), str(ref_path))

            info = build_pet_nnunet_channel(
                pet_path,
                ct_path,
                mask_path,
                dest,
                expected_shape=tuple(crop.shape),
                reference_crop_ct=ref_path,
            )
            self.assertEqual(info["n_slices"], 4)
            self.assertEqual(info["crop_source"], "test5_ct")
            self.assertEqual(tuple(nib.load(str(dest)).shape), tuple(crop.shape))


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

    def test_incomplete_when_one_case_lacks_pet(self):
        with TemporaryDirectory() as td:
            images = Path(td) / "imagesTr"
            images.mkdir()
            (images / "case_a_0000.nii.gz").write_bytes(b"x")
            (images / "case_a_0001.nii.gz").write_bytes(b"x")
            (images / "case_b_0000.nii.gz").write_bytes(b"x")
            missing = find_incomplete_nnunet_channel_cases(str(images))
            self.assertEqual(missing, [("case_b", [1])])

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

    def test_resolve_test5_ct_finds_file_in_other_split_or_stem(self):
        from pipelines.test8_0.build_dataset import (
            candidate_stems,
            index_test5_ct_channels,
            resolve_test5_ct,
        )

        self.assertIn("case_hek_CHUM_015", candidate_stems("case_hek_CHUM_015", "CHUM-015"))
        self.assertIn("case_015", candidate_stems("case_hek_CHUM_015", "CHUM-015"))

        with TemporaryDirectory() as td:
            root = Path(td)
            (root / "imagesTr").mkdir(parents=True)
            (root / "labelsTr").mkdir(parents=True)
            ct = root / "imagesTr" / "case_015_0000.nii.gz"
            lbl = root / "labelsTr" / "case_015.nii.gz"
            ct.write_bytes(b"x")
            lbl.write_bytes(b"x")
            index = index_test5_ct_channels(root)
            stem, split, ct_p, lbl_p, src = resolve_test5_ct(
                root,
                stem="case_hek_CHUM_015",
                split="Tr",
                case_id="CHUM-015",
                index=index,
            )
            self.assertEqual(stem, "case_hek_CHUM_015")
            self.assertEqual(split, "Tr")
            self.assertEqual(src, "dataset650")
            self.assertEqual(ct_p, ct)
            self.assertEqual(lbl_p, lbl)

    def test_resolve_falls_back_to_radheck_output(self):
        from pipelines.test8_0.build_dataset import find_radheck_output_pair, resolve_test5_ct

        with TemporaryDirectory() as td:
            root = Path(td)
            (root / "imagesTr").mkdir(parents=True)
            cases = root / "RADHECK_1" / "cases" / "CHUM-017"
            img_dir = cases / "output" / "image"
            lbl_dir = cases / "output" / "labels"
            img_dir.mkdir(parents=True)
            lbl_dir.mkdir(parents=True)
            ct = img_dir / "case_017_0000.nii.gz"
            lbl = lbl_dir / "case_017_0000.nii.gz"
            ct.write_bytes(b"x")
            lbl.write_bytes(b"x")
            img, lab = find_radheck_output_pair(root / "RADHECK_1" / "cases", "CHUM-017")
            self.assertEqual(img, ct)
            stem, split, ct_p, lbl_p, src = resolve_test5_ct(
                root,
                stem="case_hek_CHUM_017",
                split="Tr",
                case_id="CHUM-017",
                index={},
                cases_root=root / "RADHECK_1" / "cases",
            )
            self.assertEqual(stem, "case_hek_CHUM_017")
            self.assertEqual(src, "radheck_cases_output")
            self.assertEqual(ct_p, ct)
            self.assertEqual(lbl_p, lbl)


class TestPetDisplay(unittest.TestCase):
    def test_clips_hot_voxel(self):
        sl = np.array([[0.0, 1.0], [2.0, 1000.0]], dtype=np.float32)
        shown, vmax = _pet_display_slice(sl)
        self.assertLess(vmax, 1000.0)
        self.assertLessEqual(shown.max(), vmax)
        self.assertGreaterEqual(shown.min(), 0.0)


class TestDisplayOrientation(unittest.TestCase):
    def test_display_is_180_from_legacy_rot90(self):
        vol = np.arange(24, dtype=np.int32).reshape(2, 3, 4)
        legacy = np.rot90(vol[:, :, 1])
        shown = _volume_slice_for_display(vol, 1, axis=2)
        np.testing.assert_array_equal(shown, np.rot90(legacy, k=2))


if __name__ == "__main__":
    unittest.main()
