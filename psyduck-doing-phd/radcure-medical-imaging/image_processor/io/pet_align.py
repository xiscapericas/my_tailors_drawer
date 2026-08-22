"""Align HECKTOR PET (SUV) onto the CT grid and the Test5 slice crop.

PET must keep SUV (no CT 1–99 percentile scaling). nnUNet channel files use an
identity affine, matching ``NIfTIHandler.save_as_nii``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import nibabel as nib
import SimpleITK as sitk

from image_processor.utils.image_processing import ImageProcessor

PathLike = Union[str, Path]

# Same default as CaseProcessor (HECKTOR crop around tumor slices)
DEFAULT_SLICE_EXPANSION = 5


def hecktor_slices_to_use(
    mask_vol: np.ndarray,
    slice_expansion: int = DEFAULT_SLICE_EXPANSION,
) -> List[int]:
    """Tumor-extent slice list used when writing Test5 HECKTOR nnUNet CT."""
    z = int(mask_vol.shape[2])
    non_zero = ImageProcessor.get_non_zero_slices(mask_vol)
    if len(non_zero) == 0:
        return list(range(z))
    start = max(int(min(non_zero)) - slice_expansion, 0)
    end = min(int(max(non_zero)) + slice_expansion, z - 1)
    return list(range(start, end + 1))


def sitk_to_xyz(image: sitk.Image) -> np.ndarray:
    """SimpleITK (z, y, x) array → nibabel-style (x, y, z)."""
    arr_zyx = sitk.GetArrayFromImage(image)
    return np.transpose(arr_zyx, (2, 1, 0))


def resample_pet_to_ct(path_pet: PathLike, path_ct: PathLike) -> sitk.Image:
    """Linear-resample PET onto the CT geometry (same-session PET/CT)."""
    ct = sitk.ReadImage(str(path_ct))
    pet = sitk.ReadImage(str(path_pet))
    return sitk.Resample(
        pet,
        ct,
        sitk.Transform(),
        sitk.sitkLinear,
        0.0,
        pet.GetPixelID(),
    )


def crop_xyz(volume_xyz: np.ndarray, slices: Sequence[int]) -> np.ndarray:
    """Keep ``volume_xyz[:, :, slices]`` (axis 2 = HECKTOR/nibabel z)."""
    idx = np.asarray(list(slices), dtype=int)
    return volume_xyz[:, :, idx]


def save_nnunet_channel(volume_xyz: np.ndarray, dest: PathLike) -> Path:
    """Write a float32 NIfTI with identity affine (Test5 nnUNet convention)."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    nii = nib.Nifti1Image(np.ascontiguousarray(volume_xyz.astype(np.float32)), np.eye(4))
    nib.save(nii, str(dest))
    return dest


def describe_nifti(path: PathLike) -> dict:
    """Geometry + intensity summary for exploration notebooks."""
    path = Path(path)
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img).astype(np.float64)
    finite = arr[np.isfinite(arr)]
    return {
        "path": str(path),
        "exists": path.is_file(),
        "sitk_size_xyz": tuple(int(x) for x in img.GetSize()),
        "spacing_xyz": tuple(float(x) for x in img.GetSpacing()),
        "origin_xyz": tuple(float(x) for x in img.GetOrigin()),
        "direction": tuple(float(x) for x in img.GetDirection()),
        "dtype": str(img.GetPixelIDTypeAsString()),
        "min": float(finite.min()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "mean": float(finite.mean()) if finite.size else None,
        "p99": float(np.percentile(finite, 99)) if finite.size else None,
    }


def build_pet_nnunet_channel(
    path_pet: PathLike,
    path_ct: PathLike,
    path_mask: PathLike,
    dest_0001: PathLike,
    expected_shape: Optional[Tuple[int, ...]] = None,
    slice_expansion: int = DEFAULT_SLICE_EXPANSION,
) -> dict:
    """
    Resample PET → CT, crop the Test5 tumor window, save ``*_0001.nii.gz``.

    Parameters
    ----------
    expected_shape
        If set (typically Test5 ``*_0000`` shape), cropped PET must match or
        this raises. Pass ``None`` to skip the check (Phase 1 explore).
    """
    path_pet = Path(path_pet)
    path_ct = Path(path_ct)
    path_mask = Path(path_mask)
    if not path_pet.is_file():
        raise FileNotFoundError(f"PET not found: {path_pet}")
    if not path_ct.is_file():
        raise FileNotFoundError(f"CT not found: {path_ct}")
    if not path_mask.is_file():
        raise FileNotFoundError(f"Mask not found: {path_mask}")

    mask_vol = nib.load(str(path_mask)).get_fdata().astype(np.int32)
    slices = hecktor_slices_to_use(mask_vol, slice_expansion=slice_expansion)
    pet_on_ct = resample_pet_to_ct(path_pet, path_ct)
    pet_xyz = sitk_to_xyz(pet_on_ct).astype(np.float32)
    cropped = crop_xyz(pet_xyz, slices)

    if expected_shape is not None and tuple(cropped.shape) != tuple(expected_shape):
        raise ValueError(
            "Cropped PET shape does not match Test5 CT channel:\n"
            f"  PET cropped: {tuple(cropped.shape)}\n"
            f"  expected:    {tuple(expected_shape)}\n"
            f"  n_slices={len(slices)} ct_full_z={pet_xyz.shape[2]}"
        )

    dest = save_nnunet_channel(cropped, dest_0001)
    return {
        "dest": str(dest),
        "shape": tuple(int(x) for x in cropped.shape),
        "n_slices": len(slices),
        "slice_start": int(slices[0]),
        "slice_end": int(slices[-1]),
        "suv_min": float(np.nanmin(cropped)),
        "suv_max": float(np.nanmax(cropped)),
        "suv_mean": float(np.nanmean(cropped)),
    }
