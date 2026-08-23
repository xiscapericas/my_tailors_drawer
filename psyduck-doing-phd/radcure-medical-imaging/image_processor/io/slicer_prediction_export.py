"""Export nnUNet predictions as NIfTI aligned to CT/PET for 3D Slicer.

Integer label IDs are left unchanged (dataset.json / nnUNet argmax). Only the
spatial header is taken from the case CT (``*_0000``), so the mask overlays
CT and PET in Slicer. Display rotation used in PDF figures is not applied.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
import SimpleITK as sitk

PathLike = Union[str, Path]


def load_dataset_json_labels(dataset_json_path: PathLike) -> dict:
    """Return the ``labels`` map from nnUNet ``dataset.json`` (name → int or list)."""
    path = Path(dataset_json_path)
    if not path.is_file():
        raise FileNotFoundError(f"dataset.json not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    labels = data.get("labels")
    if not isinstance(labels, dict) or not labels:
        raise ValueError(f"No labels dict in {path}")
    return labels


def write_dataset_labels_sidecar(
    dataset_json_path: PathLike,
    dest_dir: PathLike,
) -> Path:
    """Copy dataset.json label IDs next to exported predictions (Slicer lookup)."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    src = Path(dataset_json_path)
    labels = load_dataset_json_labels(src)
    out = dest_dir / "dataset_labels.json"
    payload = {
        "labels": labels,
        "source_dataset_json": str(src),
        "note": (
            "Voxel integers in the prediction NIfTIs match these IDs "
            "(full multiclass, not GTVp-only)."
        ),
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out


def _as_label_array(image: sitk.Image) -> np.ndarray:
    arr = sitk.GetArrayFromImage(image)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.rint(arr)
    return np.ascontiguousarray(arr.astype(np.uint16))


def export_prediction_aligned_to_ct(
    pred_path: PathLike,
    ct_path: PathLike,
    dest_path: PathLike,
    pet_path: Optional[PathLike] = None,
) -> Path:
    """
    Write a multiclass prediction NIfTI on the CT voxel grid and header.

    Same-size predictions keep voxel values (all organs). Different-size
    volumes are nearest-neighbour resampled onto CT. PET is only checked
    to match the CT grid; it is not written.
    """
    pred_path = Path(pred_path)
    ct_path = Path(ct_path)
    dest_path = Path(dest_path)
    if not pred_path.is_file():
        raise FileNotFoundError(f"Prediction not found: {pred_path}")
    if not ct_path.is_file():
        raise FileNotFoundError(f"CT not found: {ct_path}")

    ct = sitk.ReadImage(str(ct_path))
    pred = sitk.ReadImage(str(pred_path))

    if pet_path is not None:
        pet_path = Path(pet_path)
        if pet_path.is_file():
            pet = sitk.ReadImage(str(pet_path))
            if pet.GetSize() != ct.GetSize():
                print(
                    f"⚠️  PET and CT grids differ for overlay "
                    f"({pet_path.name} size={pet.GetSize()}, "
                    f"CT size={ct.GetSize()})"
                )

    if pred.GetSize() != ct.GetSize():
        pred = sitk.Resample(
            pred,
            ct,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            0,
            pred.GetPixelID(),
        )

    labels = _as_label_array(pred)
    out = sitk.GetImageFromArray(labels)
    out.CopyInformation(ct)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out, str(dest_path))
    return dest_path


def export_test_set_predictions_for_slicer(
    images_ts: PathLike,
    pred_dir: PathLike,
    dest_dir: PathLike,
    dataset_json_path: PathLike,
) -> int:
    """
    Export every ``{case}.nii.gz`` in ``pred_dir`` aligned to ``{case}_0000``.

    Returns the number of cases written.
    """
    images_ts = Path(images_ts)
    pred_dir = Path(pred_dir)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    write_dataset_labels_sidecar(dataset_json_path, dest_dir)

    n = 0
    for pred_path in sorted(pred_dir.glob("*.nii.gz")):
        name = pred_path.name
        if name.endswith("_0000.nii.gz") or name.endswith("_0001.nii.gz"):
            continue
        case_id = name[: -len(".nii.gz")]
        ct_path = images_ts / f"{case_id}_0000.nii.gz"
        pet_path = images_ts / f"{case_id}_0001.nii.gz"
        if not ct_path.is_file():
            print(f"⚠️  Skip {case_id}: no CT {ct_path.name}")
            continue
        export_prediction_aligned_to_ct(
            pred_path,
            ct_path,
            dest_dir / f"{case_id}.nii.gz",
            pet_path=pet_path if pet_path.is_file() else None,
        )
        n += 1
    return n
