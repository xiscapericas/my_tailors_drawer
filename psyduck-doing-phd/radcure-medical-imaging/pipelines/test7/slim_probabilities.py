#!/usr/bin/env python3
"""
Test7 — convert raw nnUNet ``.npz`` softmax dumps into slim cropped archives.

Each ``{case}.slim.npz`` stores float16 probabilities only near the tumor ROI:

  - bbox around dilated GT GTVp (fallback: high P(GTVp) / argmax)
  - p_gtvp + selected classes (GT overlap organs, GTVn, top-k by mean P)
  - deletes the raw ``.npz`` after a successful write (default)

Example:

  python -m pipelines.test7.slim_probabilities
  python -m pipelines.test7.slim_probabilities --keep-raw --top-k 8 --margin 12
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipelines.test7.paths import (
    load_organ_dict,
    pin_test7_env,
    probabilities_dir,
    work_root,
)
from pipelines.test7.prob_io import (
    SLIM_SUFFIX,
    align_probs_to_reference,
    bbox_from_mask,
    build_slim_from_full,
    find_raw_npz,
    load_probability_npz,
    save_slim_npz,
    select_class_indices,
    slim_path_for_case,
    tumor_roi_mask,
)


def _load_nifti_mask(path: Path) -> np.ndarray:
    import nibabel as nib

    return np.asanyarray(nib.load(str(path)).dataobj)


def _list_raw_cases(prob_dir: Path) -> list[str]:
    cases = []
    for p in sorted(prob_dir.glob("*.npz")):
        if p.name.endswith(SLIM_SUFFIX):
            continue
        if p.name.endswith(".npz.npz"):
            cases.append(p.name[: -len(".npz.npz")])
        else:
            cases.append(p.stem)
    return cases


def convert_case(
    case_id: str,
    raw_path: Path,
    gt_path: Path | None,
    organ_dict: dict,
    out_path: Path,
    *,
    dilate_iter: int,
    margin: int,
    top_k: int,
    pred_threshold: float,
    image_path: Path | None = None,
) -> dict:
    probs = load_probability_npz(raw_path)
    gtvp_index = int(organ_dict["GTVp"])
    if probs.shape[0] <= gtvp_index:
        raise ValueError(
            f"{case_id}: C={probs.shape[0]} but GTVp index={gtvp_index}"
        )

    gt = None
    if gt_path is not None and gt_path.is_file():
        gt = _load_nifti_mask(gt_path)

    # Align nnUNet .npz spatial axes to GT (or CT) nibabel order
    ref_shape = None
    if gt is not None:
        ref_shape = gt.shape
    elif image_path is not None and image_path.is_file():
        ref_shape = _load_nifti_mask(image_path).shape

    spatial_transpose = None
    if ref_shape is not None:
        try:
            probs, spatial_transpose = align_probs_to_reference(probs, ref_shape)
        except ValueError as e:
            raise ValueError(f"{case_id}: {e}") from e
        if spatial_transpose is not None:
            print(
                f"  {case_id}: transposed probs spatial {spatial_transpose} "
                f"→ {probs.shape[1:]} (match GT/CT)"
            )

    if gt is not None and gt.shape != probs.shape[1:]:
        raise ValueError(
            f"{case_id}: GT shape {gt.shape} != probs spatial {probs.shape[1:]} "
            "after align"
        )

    roi = tumor_roi_mask(
        gt,
        probs,
        gtvp_index,
        dilate_iter=dilate_iter,
        pred_threshold=pred_threshold,
    )
    bbox = bbox_from_mask(roi, margin=margin, full_shape=probs.shape[1:])
    x0, x1, y0, y1, z0, z1 = bbox
    probs_crop = probs[:, x0:x1, y0:y1, z0:z1]
    gt_crop = gt[x0:x1, y0:y1, z0:z1] if gt is not None else None

    class_indices = select_class_indices(
        probs_crop,
        gt_crop,
        gtvp_index,
        organ_dict,
        always_include=("GTVn",),
        top_k=top_k,
    )
    slim = build_slim_from_full(
        probs, gtvp_index, class_indices, bbox, case_id=case_id
    )
    raw_bytes = raw_path.stat().st_size
    save_slim_npz(
        out_path,
        slim,
        dilate_iter=np.int16(dilate_iter),
        margin=np.int16(margin),
        top_k=np.int16(top_k),
        spatial_transpose=np.asarray(
            spatial_transpose if spatial_transpose is not None else (-1, -1, -1),
            dtype=np.int16,
        ),
        roi_source=np.asarray(
            "gt_gtvp" if gt is not None and np.any(gt == gtvp_index) else "pred"
        ),
    )
    slim_bytes = out_path.stat().st_size
    return {
        "case_id": case_id,
        "bbox": list(bbox),
        "crop_shape": list(slim.crop_shape),
        "full_shape": list(slim.full_shape),
        "spatial_transpose": list(spatial_transpose)
        if spatial_transpose is not None
        else None,
        "n_classes_kept": int(len(class_indices)),
        "class_indices": [int(i) for i in class_indices],
        "raw_bytes": raw_bytes,
        "slim_bytes": slim_bytes,
        "compression_ratio": round(raw_bytes / slim_bytes, 2) if slim_bytes else None,
    }


def slim_all(
    work: Path,
    *,
    dilate_iter: int = 2,
    margin: int = 8,
    top_k: int = 5,
    pred_threshold: float = 0.3,
    keep_raw: bool = False,
    max_cases: int = 0,
) -> dict:
    paths = pin_test7_env(work)
    dataset = paths["dataset"]
    labels_ts = dataset / "labelsTs"
    images_ts = dataset / "imagesTs"
    hard_dir = Path(work) / "predictions" / "labelsTs_predicted"
    prob_dir = probabilities_dir(work)
    prob_dir.mkdir(parents=True, exist_ok=True)
    organ_dict = load_organ_dict(paths.get("organ"))

    cases = _list_raw_cases(prob_dir)
    if max_cases > 0:
        cases = cases[:max_cases]

    print("=" * 70)
    print("Test7 — slim_probabilities (crop + float16 + class subset)")
    print(f"  raw/slim dir: {prob_dir}")
    print(f"  cases:        {len(cases)}")
    print(f"  dilate/margin/top_k: {dilate_iter}/{margin}/{top_k}")
    print(f"  keep_raw:     {keep_raw}")
    print("=" * 70)

    results = []
    failed = []
    for case_id in cases:
        raw = find_raw_npz(prob_dir, case_id)
        if raw is None:
            continue
        out = slim_path_for_case(prob_dir, case_id)
        gt_path = labels_ts / f"{case_id}.nii.gz"
        image_path = images_ts / f"{case_id}_0000.nii.gz"
        if not image_path.is_file():
            # fallback: hard prediction geometry (same as CT after nnUNet export)
            alt = hard_dir / f"{case_id}.nii.gz"
            image_path = alt if alt.is_file() else image_path
        try:
            info = convert_case(
                case_id,
                raw,
                gt_path if gt_path.is_file() else None,
                organ_dict,
                out,
                dilate_iter=dilate_iter,
                margin=margin,
                top_k=top_k,
                pred_threshold=pred_threshold,
                image_path=image_path if image_path.is_file() else None,
            )
            results.append(info)
            print(
                f"  {case_id}: crop={info['crop_shape']} "
                f"K={info['n_classes_kept']} "
                f"{info['raw_bytes']//(1024*1024)}MB→{info['slim_bytes']//1024}KB "
                f"(×{info['compression_ratio']})"
            )
            if not keep_raw:
                raw.unlink()
        except Exception as e:
            print(f"  fail {case_id}: {e}")
            failed.append({"case_id": case_id, "error": str(e)})

    summary = {
        "n_ok": len(results),
        "n_failed": len(failed),
        "dilate_iter": dilate_iter,
        "margin": margin,
        "top_k": top_k,
        "keep_raw": keep_raw,
        "cases": results,
        "failed": failed,
    }
    with open(prob_dir / "slim_STATUS.json", "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"\nWrote {prob_dir / 'slim_STATUS.json'} ({len(results)} ok, {len(failed)} failed)")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test7: convert raw .npz probs → slim cropped float16"
    )
    parser.add_argument("--work-root", default=str(work_root()))
    parser.add_argument("--dilate", type=int, default=2)
    parser.add_argument("--margin", type=int, default=8, help="Voxels pad around ROI bbox")
    parser.add_argument("--top-k", type=int, default=12, help="Extra classes by mean P near GTVp (not whole crop)")
    parser.add_argument("--pred-threshold", type=float, default=0.3)
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep original nnUNet .npz after writing slim (default: delete)",
    )
    parser.add_argument("--max-cases", type=int, default=0)
    args = parser.parse_args()

    work = Path(args.work_root).expanduser().resolve()
    slim_all(
        work,
        dilate_iter=args.dilate,
        margin=args.margin,
        top_k=args.top_k,
        pred_threshold=args.pred_threshold,
        keep_raw=args.keep_raw,
        max_cases=args.max_cases,
    )
    print("\nNext:")
    print("  python -m pipelines.test7.region_tumor_probabilities_vs_dice_curves")
    print("  python -m pipelines.test7.probability_visualisation")


if __name__ == "__main__":
    main()
