"""Heuristic QC: score how likely a CT crop looks like human head/neck anatomy."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from skimage import measure

from image_processor.utils.image_processing import ImageProcessor


def _clamp01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _band_score(value: float, low: float, high: float, soft: float = 0.15) -> float:
    """1 inside [low, high], linear falloff outside over ``soft`` width."""
    if low <= value <= high:
        return 1.0
    if value < low:
        return _clamp01(1.0 - (low - value) / max(soft, 1e-6))
    return _clamp01(1.0 - (value - high) / max(soft, 1e-6))


def score_human_anatomy(
    ct: np.ndarray,
    tumor: np.ndarray,
    slices: Optional[Sequence[int]] = None,
    *,
    min_gtvp_voxels: int = 20,
    weights: Optional[Dict[str, float]] = None,
    fill_low: float = 0.075,
    fill_high: float = 0.55,
    fill_soft: float = 0.02,
    hard_fail_fill_below: float = 0.065,
) -> Dict[str, Any]:
    """
    Score likelihood that a volume crop is usable human H&N anatomy.

    Empirically (audit set): non-human FOVs had mean_patient_fill ≈ 0.04–0.06
    while usable cases were ≥ ≈ 0.07. Tumor voxels alone are a weak signal
    (labels can exist on non-anatomy), so fill / coherence dominate.

    Hard fail: if ``mean_patient_fill < hard_fail_fill_below``, score is capped
    and ``hard_fail`` is set so thresholding can discard regardless of tumor.
    """
    if slices is None:
        slices = list(range(ct.shape[2]))
    slices = list(slices)
    if not slices:
        return {
            "score": 0.0,
            "components": {},
            "metrics": {"error": "no_slices"},
            "reasons": ["no_slices"],
            "hard_fail": True,
        }

    ct_c = ct[:, :, slices]
    tu_c = tumor[:, :, slices]

    gtvp = int(np.sum(tu_c == 1))
    gtvn = int(np.sum(tu_c == 2))
    tumor_any = int(np.sum(tu_c > 0))

    # Tumor presence — useful but must not dominate (non-anatomy can still have labels)
    if gtvp >= min_gtvp_voxels:
        tumor_score = 1.0
    elif gtvp > 0:
        tumor_score = _clamp01(gtvp / float(min_gtvp_voxels))
    elif gtvn > 0:
        tumor_score = 0.35
    else:
        tumor_score = 0.0

    p1, p50, p99 = np.percentile(ct_c.astype(np.float64), (1, 50, 99))
    dyn = float(p99 - p1)
    intensity_score = _band_score(dyn, low=50.0, high=1e6, soft=50.0)
    if dyn < 1e-3:
        intensity_score = 0.0

    sample_idx = sorted(
        {
            slices[0],
            slices[len(slices) // 4],
            slices[len(slices) // 2],
            slices[(3 * len(slices)) // 4],
            slices[-1],
        }
    )

    fill_fracs: List[float] = []
    largest_cc_fracs: List[float] = []
    tumor_in_patient = 0
    tumor_total = 0

    for z in sample_idx:
        bg = ImageProcessor.head_mask_from_array(ct[:, :, z])
        patient = ~bg
        fill_fracs.append(float(np.mean(patient)))

        labels = measure.label(patient.astype(np.uint8))
        if labels.max() == 0:
            largest_cc_fracs.append(0.0)
        else:
            sizes = np.bincount(labels.ravel())
            sizes[0] = 0
            largest_cc_fracs.append(float(sizes.max()) / max(int(patient.sum()), 1))

        m = tumor[:, :, z] > 0
        tumor_total += int(m.sum())
        tumor_in_patient += int(np.sum(m & patient))

    mean_fill = float(np.mean(fill_fracs)) if fill_fracs else 0.0
    mean_cc = float(np.mean(largest_cc_fracs)) if largest_cc_fracs else 0.0
    tumor_inside_frac = (
        float(tumor_in_patient) / float(tumor_total) if tumor_total > 0 else 0.0
    )

    # Stricter fill band (audit: bad ~0.04–0.06, ok ≥~0.07)
    fill_score = _band_score(mean_fill, low=fill_low, high=fill_high, soft=fill_soft)
    coherence_score = _band_score(mean_cc, low=0.55, high=1.0, soft=0.25)
    tumor_inside_score = (
        _band_score(tumor_inside_frac, low=0.70, high=1.0, soft=0.25)
        if tumor_total > 0
        else 0.0
    )

    n_slices = len(slices)
    extent_score = _band_score(float(n_slices), low=8.0, high=200.0, soft=6.0)

    components = {
        "tumor": tumor_score,
        "tumor_inside_patient": tumor_inside_score,
        "intensity": intensity_score,
        "patient_fill": fill_score,
        "body_coherence": coherence_score,
        "slice_extent": extent_score,
    }
    # Fill/coherence dominate: tumor labels alone do not prove human FOV
    w = weights or {
        "tumor": 0.12,
        "tumor_inside_patient": 0.18,
        "intensity": 0.10,
        "patient_fill": 0.35,
        "body_coherence": 0.15,
        "slice_extent": 0.10,
    }
    wsum = sum(w.get(k, 0.0) for k in components) or 1.0
    score = float(sum(components[k] * w.get(k, 0.0) for k in components) / wsum)

    hard_fail = bool(mean_fill < hard_fail_fill_below)
    reasons: List[str] = []
    if hard_fail:
        reasons.append(
            f"hard_fail_low_patient_fill={mean_fill:.3f}<{hard_fail_fill_below}"
        )
        score = min(score, 0.35)

    if tumor_score < 0.5:
        reasons.append("weak_or_missing_tumor")
    if fill_score < 0.5 and not hard_fail:
        reasons.append(f"abnormal_patient_fill={mean_fill:.3f}")
    if coherence_score < 0.5:
        reasons.append(f"fragmented_body_mask_cc={mean_cc:.3f}")
    if tumor_inside_score < 0.5 and tumor_total > 0:
        reasons.append(f"tumor_outside_patient={tumor_inside_frac:.3f}")
    if intensity_score < 0.5:
        reasons.append(f"flat_intensity_dyn={dyn:.3f}")
    if extent_score < 0.5:
        reasons.append(f"few_slices={n_slices}")

    return {
        "score": score,
        "hard_fail": hard_fail,
        "components": components,
        "metrics": {
            "gtvp_voxels": gtvp,
            "gtvn_voxels": gtvn,
            "tumor_voxels": tumor_any,
            "n_slices": n_slices,
            "ct_p1": float(p1),
            "ct_p50": float(p50),
            "ct_p99": float(p99),
            "ct_dynamic_range": dyn,
            "mean_patient_fill": mean_fill,
            "mean_largest_cc_frac": mean_cc,
            "tumor_inside_patient_frac": tumor_inside_frac,
            "sample_z": sample_idx,
            "hard_fail_fill_below": hard_fail_fill_below,
        },
        "reasons": reasons,
    }


def apply_anatomy_threshold(
    score_result: Dict[str, Any],
    threshold: float = 0.70,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Return (keep, record).

    Discard if ``hard_fail`` or ``score < threshold``.
    """
    hard_fail = bool(score_result.get("hard_fail", False))
    score = float(score_result.get("score", 0.0))
    keep = (not hard_fail) and (score >= float(threshold))
    record = {
        **score_result,
        "threshold": float(threshold),
        "keep": keep,
        "decision": "keep" if keep else "discard",
    }
    return keep, record


def append_qc_log(
    log_path: str,
    *,
    case_id: str,
    convention: str,
    record: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Append one JSON line for a case QC decision (kept or discarded)."""
    os.makedirs(os.path.dirname(os.path.abspath(log_path)) or ".", exist_ok=True)
    row = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "case_id": case_id,
        "convention": convention,
        **record,
    }
    if extra:
        row["extra"] = extra
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=float) + "\n")


def write_discard_summary_csv(discard_records: List[Dict[str, Any]], csv_path: str) -> None:
    """Write a small CSV of discarded cases for quick review."""
    import csv

    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)
    fields = [
        "case_id",
        "convention",
        "score",
        "threshold",
        "hard_fail",
        "reasons",
        "gtvp_voxels",
        "gtvn_voxels",
        "mean_patient_fill",
        "tumor_inside_patient_frac",
        "ct_dynamic_range",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in discard_records:
            m = r.get("metrics") or {}
            w.writerow(
                {
                    "case_id": r.get("case_id", ""),
                    "convention": r.get("convention", ""),
                    "score": r.get("score", ""),
                    "threshold": r.get("threshold", ""),
                    "hard_fail": r.get("hard_fail", ""),
                    "reasons": ";".join(r.get("reasons") or []),
                    "gtvp_voxels": m.get("gtvp_voxels", ""),
                    "gtvn_voxels": m.get("gtvn_voxels", ""),
                    "mean_patient_fill": m.get("mean_patient_fill", ""),
                    "tumor_inside_patient_frac": m.get("tumor_inside_patient_frac", ""),
                    "ct_dynamic_range": m.get("ct_dynamic_range", ""),
                }
            )
