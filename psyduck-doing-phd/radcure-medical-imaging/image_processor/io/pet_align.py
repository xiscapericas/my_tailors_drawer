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


def _percentile_01(volume: np.ndarray) -> np.ndarray:
    """Same 1–99 percentile stretch as ``NIfTIHandler.load_nii_image``."""
    data = np.asarray(volume, dtype=np.float32)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.nan_to_num(data)
    low, high = np.percentile(finite, (1, 99))
    data = np.clip(data, low, high)
    data = (data - low) / (high - low + 1e-8)
    return np.nan_to_num(data)


def _best_z_start_3d(full_d: np.ndarray, crop_d: np.ndarray) -> Tuple[int, float]:
    zf, zc = full_d.shape[2], crop_d.shape[2]
    crop_n = crop_d - float(crop_d.mean())
    crop_norm = float(np.linalg.norm(crop_n)) + 1e-8
    best_s, best_corr = 0, -1.0
    for s in range(zf - zc + 1):
        w = full_d[:, :, s : s + zc]
        w = w - float(w.mean())
        corr = float(np.sum(w * crop_n) / ((np.linalg.norm(w) + 1e-8) * crop_norm))
        if corr > best_corr:
            best_corr, best_s = corr, s
    return best_s, best_corr


def _best_z_start_profile(full_xyz: np.ndarray, crop_xyz: np.ndarray) -> Tuple[int, float]:
    """z-window from per-slice mean (robust to in-plane rot/flip)."""
    full_p = np.mean(full_xyz, axis=(0, 1))
    crop_p = np.mean(crop_xyz, axis=(0, 1))
    zf, zc = int(full_p.shape[0]), int(crop_p.shape[0])
    crop_n = crop_p - float(crop_p.mean())
    crop_norm = float(np.linalg.norm(crop_n)) + 1e-8
    best_s, best_corr = 0, -1.0
    for s in range(zf - zc + 1):
        w = full_p[s : s + zc]
        w = w - float(w.mean())
        corr = float(np.dot(w, crop_n) / ((np.linalg.norm(w) + 1e-8) * crop_norm))
        if corr > best_corr:
            best_corr, best_s = corr, s
    return best_s, best_corr


def slices_matching_test5_crop(
    full_xyz: np.ndarray,
    crop_xyz: np.ndarray,
    *,
    min_corr: float = 0.5,
    strict: bool = False,
) -> Tuple[List[int], dict]:
    """
    Contiguous z-window of ``full_xyz`` whose block matches ``crop_xyz``.

    Tries 3D correlation, slice-mean profiles, and a z-flip of the original.
    If the best score is still below ``min_corr``, ``strict=True`` raises;
    otherwise the best window is kept so the build can continue.
    """
    if full_xyz.ndim != 3 or crop_xyz.ndim != 3:
        raise ValueError(f"Expected 3D volumes, got {full_xyz.shape} and {crop_xyz.shape}")
    if full_xyz.shape[0] != crop_xyz.shape[0] or full_xyz.shape[1] != crop_xyz.shape[1]:
        raise ValueError(
            "XY size mismatch between original CT and Test5 crop: "
            f"{full_xyz.shape[:2]} vs {crop_xyz.shape[:2]}"
        )
    zf = int(full_xyz.shape[2])
    zc = int(crop_xyz.shape[2])
    if zc == zf:
        return list(range(zf)), {"corr": 1.0, "method": "full", "z_flipped": False, "weak_match": False}
    if zc > zf:
        raise ValueError(
            f"Test5 CT has more slices ({zc}) than original CT ({zf})"
        )

    step = 8 if min(full_xyz.shape[0], full_xyz.shape[1]) >= 32 else 1
    full_n = _percentile_01(np.asarray(full_xyz, dtype=np.float32))
    crop_n = np.asarray(crop_xyz, dtype=np.float32)
    if float(np.nanmax(crop_n)) > 1.5:
        crop_n = _percentile_01(crop_n)
    full_d = full_n[::step, ::step, :]
    crop_d = crop_n[::step, ::step, :]

    candidates: List[Tuple[float, int, str, bool]] = []

    s, c = _best_z_start_3d(full_d, crop_d)
    candidates.append((c, s, "3d", False))
    s, c = _best_z_start_profile(full_n, crop_n)
    candidates.append((c, s, "profile", False))

    full_flip = full_n[:, :, ::-1]
    full_d_flip = full_flip[::step, ::step, :]
    s, c = _best_z_start_3d(full_d_flip, crop_d)
    candidates.append((c, zf - s - zc, "3d_zflip", True))
    s, c = _best_z_start_profile(full_flip, crop_n)
    candidates.append((c, zf - s - zc, "profile_zflip", True))

    crop_rev = crop_n[:, :, ::-1]
    crop_d_rev = crop_rev[::step, ::step, :]
    s, c = _best_z_start_3d(full_d, crop_d_rev)
    candidates.append((c, s, "3d_crop_zrev", False))

    best_corr, best_s, method, z_flipped = max(candidates, key=lambda t: t[0])
    best_s = int(np.clip(best_s, 0, zf - zc))
    meta = {
        "corr": float(best_corr),
        "method": method,
        "z_flipped": z_flipped,
        "weak_match": bool(best_corr < min_corr),
    }
    if best_corr < min_corr and strict:
        raise ValueError(
            "Could not locate Test5 CT crop inside original CT "
            f"(best corr={best_corr:.3f} < {min_corr}, z_full={zf}, z_crop={zc})"
        )
    return list(range(best_s, best_s + zc)), meta


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
    reference_crop_ct: Optional[PathLike] = None,
) -> dict:
    """
    Resample PET → original CT, crop to the Test5 CT window, save ``*_0001``.

    If ``reference_crop_ct`` is set (Test5 ``*_0000``), the z-crop is the
    contiguous window that matches that volume. Otherwise crop from the
    original mask with ``slice_expansion`` (explore / no Test5 CT).
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

    pet_on_ct = resample_pet_to_ct(path_pet, path_ct)
    pet_xyz = sitk_to_xyz(pet_on_ct).astype(np.float32)

    crop_source = "mask"
    match_meta: dict = {}
    if reference_crop_ct is not None:
        ref_path = Path(reference_crop_ct)
        if not ref_path.is_file():
            raise FileNotFoundError(f"Test5 CT crop not found: {ref_path}")
        orig_ct_xyz = nib.load(str(path_ct)).get_fdata().astype(np.float32)
        crop_xyz_ref = nib.load(str(ref_path)).get_fdata().astype(np.float32)
        slices, match_meta = slices_matching_test5_crop(orig_ct_xyz, crop_xyz_ref)
        crop_source = "test5_ct"
        if match_meta.get("weak_match"):
            print(
                f"  warning: weak Test5 CT match "
                f"(corr={match_meta['corr']:.3f}, method={match_meta['method']}); "
                "using best z-window so PET shape still matches"
            )
    else:
        mask_vol = nib.load(str(path_mask)).get_fdata().astype(np.int32)
        slices = hecktor_slices_to_use(mask_vol, slice_expansion=slice_expansion)

    cropped = crop_xyz(pet_xyz, slices)

    if expected_shape is not None and tuple(cropped.shape) != tuple(expected_shape):
        raise ValueError(
            "Cropped PET shape does not match Test5 CT channel:\n"
            f"  PET cropped: {tuple(cropped.shape)}\n"
            f"  expected:    {tuple(expected_shape)}\n"
            f"  n_slices={len(slices)} ct_full_z={pet_xyz.shape[2]} "
            f"crop_source={crop_source}"
        )

    dest = save_nnunet_channel(cropped, dest_0001)
    return {
        "dest": str(dest),
        "shape": tuple(int(x) for x in cropped.shape),
        "n_slices": len(slices),
        "slice_start": int(slices[0]),
        "slice_end": int(slices[-1]),
        "crop_source": crop_source,
        "match_corr": match_meta.get("corr") if reference_crop_ct is not None else None,
        "match_method": match_meta.get("method") if reference_crop_ct is not None else None,
        "suv_min": float(np.nanmin(cropped)),
        "suv_max": float(np.nanmax(cropped)),
        "suv_mean": float(np.nanmean(cropped)),
    }
