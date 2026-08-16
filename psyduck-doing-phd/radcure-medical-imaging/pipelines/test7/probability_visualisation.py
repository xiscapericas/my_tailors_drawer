#!/usr/bin/env python3
"""
Test7 — probability_visualisation

Full-slice layout (no tumor crop zoom):

  1. Original CT
  2. CT + GTVp ground truth only
  3. CT + soft multi-class overlay (alpha = P(class))

Colormap = same standard as Test6 / ``MedicalImageVisualizer``
(tab20 + GTVp red + GTVn magenta).

Prefers ``{case}.slim.npz`` (pasted into full volume at bbox); falls back to
raw full-volume ``.npz``.

Example:

  python -m pipelines.test7.probability_visualisation
  python -m pipelines.test7.probability_visualisation --max-cases 3 --max-slices 12
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from image_processor.visualization.visualizer import MedicalImageVisualizer, _get_cmap
from pipelines.test7.paths import (
    load_organ_dict,
    pin_test7_env,
    probabilities_dir,
    probability_viz_dir,
    work_root,
)
from pipelines.test7.prob_io import (
    SlimProbabilities,
    align_probs_to_reference,
    list_cases_with_probabilities,
    load_case_probabilities,
    load_probability_npz,
)

# Classes that drown soft overlays if painted at full strength
_SOFT_EXCLUDE = frozenset(
    {
        "background",
        "anatomical_region",
        "other-tissue",
    }
)


def standard_label_colors(organ_dict: Dict[str, int]) -> np.ndarray:
    """
    Test6 / MedicalImageVisualizer standard: tab20, GTVp=red, GTVn=magenta,
    replace other red-like entries with Set3.
    """
    max_idx = max(int(v) for v in organ_dict.values() if isinstance(v, int))
    colormap_size = max_idx + 1
    base_cmap = _get_cmap("tab20", colormap_size)
    colors = np.asarray(base_cmap(np.arange(colormap_size)), dtype=np.float32)
    colors[0, :] = [0.0, 0.0, 0.0, 0.0]
    MedicalImageVisualizer._apply_tumor_highlight_colors(
        colors, organ_dict, colormap_size
    )
    return colors


def _load_nifti(path: Path) -> np.ndarray:
    import nibabel as nib

    return np.asanyarray(nib.load(str(path)).dataobj)


def _get_slice(vol: np.ndarray, idx: int, axis: int) -> np.ndarray:
    if axis == 0:
        sl = vol[idx, :, :]
    elif axis == 1:
        sl = vol[:, idx, :]
    else:
        sl = vol[:, :, idx]
    return np.rot90(sl)


def _ct_to_rgb(ct_slice: np.ndarray) -> np.ndarray:
    """Normalize CT slice to float RGB in [0, 1]."""
    x = np.asarray(ct_slice, dtype=np.float32)
    lo, hi = np.percentile(x, (1.0, 99.0))
    if hi <= lo:
        lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        g = np.zeros_like(x, dtype=np.float32)
    else:
        g = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return np.stack([g, g, g], axis=-1)


def soft_overlay_on_ct(
    ct_slice: np.ndarray,
    probs_by_class: Dict[int, np.ndarray],
    colors: np.ndarray,
    *,
    alpha_scale: float = 0.65,
    min_prob: float = 0.12,
) -> np.ndarray:
    """
    Alpha-composite class colours over CT.

    Low-probability classes first, high last (so confident labels sit on top).
    ``alpha = P(class) * alpha_scale`` — CT remains visible where P is low.
    """
    rgb = _ct_to_rgb(ct_slice)
    # Paint low mean-P first so high-P (e.g. GTVp) wins conflicts
    order = sorted(
        probs_by_class.keys(),
        key=lambda c: float(np.asarray(probs_by_class[c]).mean()),
    )
    for cls in order:
        if cls <= 0 or cls >= len(colors):
            continue
        p = np.asarray(probs_by_class[cls], dtype=np.float32)
        if float(p.max()) < min_prob:
            continue
        a = np.clip(p * alpha_scale, 0.0, 1.0)
        a = np.where(p >= min_prob, a, 0.0)
        if not np.any(a > 0):
            continue
        col = colors[cls, :3].astype(np.float32)
        a3 = a[..., None]
        rgb = a3 * col + (1.0 - a3) * rgb
    return np.clip(rgb, 0.0, 1.0)


def slim_probs_to_full(
    slim: SlimProbabilities,
) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    """
    Expand slim crop channels into full-volume maps (zeros outside bbox).

    Returns ``(probs_by_label, p_gtvp_full)``.
    """
    probs_k, idxs = slim.channel_stack_crop()
    x0, x1, y0, y1, z0, z1 = slim.bbox
    full_shape = slim.full_shape
    out: Dict[int, np.ndarray] = {}
    for c, lab in enumerate(idxs):
        lab = int(lab)
        vol = np.zeros(full_shape, dtype=np.float32)
        vol[x0:x1, y0:y1, z0:z1] = np.asarray(probs_k[c], dtype=np.float32)
        out[lab] = vol
    p_gtvp = out.get(
        int(slim.gtvp_index),
        np.zeros(full_shape, dtype=np.float32),
    )
    return out, p_gtvp


def _pick_slices(
    n_slices: int,
    axis: int,
    max_slices: int,
    gtvp_gt: Optional[np.ndarray],
    p_gtvp: Optional[np.ndarray],
) -> List[int]:
    """Prefer slices with GTVp GT, else high mean P(GTVp)."""
    if max_slices <= 0 or max_slices >= n_slices:
        return list(range(n_slices))

    scores: List[Tuple[float, int]] = []
    for i in range(n_slices):
        score = 0.0
        if gtvp_gt is not None:
            score += 10.0 * float(_get_slice(gtvp_gt, i, axis).any())
            score += float(_get_slice(gtvp_gt, i, axis).mean())
        if p_gtvp is not None:
            score += float(_get_slice(p_gtvp, i, axis).mean())
        scores.append((score, i))
    scores.sort(reverse=True)
    return sorted(i for _, i in scores[:max_slices])


def _legend_patches(
    present_labels: List[int],
    colors: np.ndarray,
    index_to_organ: Dict[int, str],
):
    import matplotlib.patches as mpatches

    patches = []
    for lab in present_labels:
        if lab <= 0 or lab >= len(colors):
            continue
        name = index_to_organ.get(lab, f"label_{lab}")
        patches.append(mpatches.Patch(color=colors[lab, :3], label=f"{name} ({lab})"))
    return patches


def visualize_case(
    case_id: str,
    img: np.ndarray,
    gt: Optional[np.ndarray],
    probs_by_label: Dict[int, np.ndarray],
    p_gtvp_full: np.ndarray,
    organ_dict: Dict[str, int],
    out_pdf: Path,
    *,
    axis: int = 2,
    max_slices: int = 16,
    alpha_scale: float = 0.65,
    min_prob: float = 0.12,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    colors = standard_label_colors(organ_dict)
    index_to_organ = {int(v): k for k, v in organ_dict.items() if isinstance(v, int)}
    gtvp_idx = int(organ_dict["GTVp"])
    exclude_idx = {
        int(organ_dict[n]) for n in _SOFT_EXCLUDE if n in organ_dict
    }

    gtvp_gt = (gt == gtvp_idx).astype(np.uint8) if gt is not None else None
    n_slices = img.shape[axis]
    slices_to_show = _pick_slices(
        n_slices, axis, max_slices, gtvp_gt, p_gtvp_full
    )

    # Soft channels: all stored labels except background / bulk tissue classes
    soft_labels = sorted(
        lab
        for lab in probs_by_label
        if lab > 0 and lab not in exclude_idx
    )

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_pdf) as pdf:
        for slice_idx in slices_to_show:
            img_sl = _get_slice(img, slice_idx, axis)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            ax_ct, ax_gt, ax_soft = axes

            # 1) Original CT
            ax_ct.imshow(img_sl, cmap="gray")
            ax_ct.set_title(f"CT\n(slice {slice_idx})")
            ax_ct.axis("off")

            # 2) CT + GTVp GT only
            ax_gt.imshow(img_sl, cmap="gray")
            if gtvp_gt is not None:
                gtvp_sl = _get_slice(gtvp_gt, slice_idx, axis).astype(bool)
                if np.any(gtvp_sl):
                    overlay = np.zeros((*gtvp_sl.shape, 4), dtype=np.float32)
                    overlay[gtvp_sl, :3] = colors[gtvp_idx, :3]
                    overlay[gtvp_sl, 3] = 0.55
                    ax_gt.imshow(overlay)
            ax_gt.set_title("CT + GTVp (GT)")
            ax_gt.axis("off")

            # 3) Soft multi-class (alpha = P)
            probs_sl: Dict[int, np.ndarray] = {}
            for lab in soft_labels:
                psl = _get_slice(probs_by_label[lab], slice_idx, axis)
                if float(psl.max()) >= min_prob:
                    probs_sl[lab] = psl
            # Always include GTVp channel if present
            if gtvp_idx in probs_by_label:
                probs_sl[gtvp_idx] = _get_slice(
                    probs_by_label[gtvp_idx], slice_idx, axis
                )

            soft_rgb = soft_overlay_on_ct(
                img_sl,
                probs_sl,
                colors,
                alpha_scale=alpha_scale,
                min_prob=min_prob,
            )
            ax_soft.imshow(soft_rgb)
            ax_soft.set_title("Soft prediction\n(alpha = P(class))")
            ax_soft.axis("off")

            # Legend: every class with max P >= min_prob on this slice (+ GTVp)
            present = []
            for lab, psl in probs_sl.items():
                if lab == gtvp_idx or float(psl.max()) >= min_prob:
                    present.append(lab)
            present = sorted(set(present))
            # Put GTVp first in the legend
            if gtvp_idx in present:
                present = [gtvp_idx] + [x for x in present if x != gtvp_idx]

            patches = _legend_patches(present, colors, index_to_organ)
            if patches:
                fig.legend(
                    handles=patches,
                    loc="center left",
                    bbox_to_anchor=(1.01, 0.5),
                    fontsize=7,
                    title="organs (index)",
                )

            fig.suptitle(
                f"{case_id} — probability visualisation (full CT)",
                y=1.02,
            )
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test7: full-CT probability visualisation (Test6 colormap)"
    )
    parser.add_argument("--work-root", default=str(work_root()))
    parser.add_argument("--axis", type=int, default=2, help="0=sag, 1=cor, 2=ax")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument(
        "--max-slices",
        type=int,
        default=16,
        help="Max slices per case (0 = all; prefers GTVp / high-P slices)",
    )
    parser.add_argument("--alpha-scale", type=float, default=0.65)
    parser.add_argument(
        "--min-prob",
        type=float,
        default=0.12,
        help="Min P(class) to paint / list in legend",
    )
    args = parser.parse_args()

    work = Path(args.work_root).expanduser().resolve()
    paths = pin_test7_env(work)
    dataset = paths["dataset"]
    images_ts = dataset / "imagesTs"
    labels_ts = dataset / "labelsTs"
    prob_dir = probabilities_dir(work)
    out_dir = probability_viz_dir(work)
    out_dir.mkdir(parents=True, exist_ok=True)

    organ_dict = load_organ_dict(paths.get("organ"))
    if "GTVp" not in organ_dict:
        raise RuntimeError("Organ dictionary missing GTVp")

    cases = list_cases_with_probabilities(prob_dir)
    if not cases:
        raise FileNotFoundError(
            f"No .slim.npz / .npz probabilities in {prob_dir}\n"
            "Run: python -m pipelines.test7.predict_probabilities"
        )
    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    print("=" * 70)
    print("Test7 — probability_visualisation (full CT, Test6 colormap)")
    print(f"  cases:  {len(cases)}")
    print(f"  probs:  {prob_dir}")
    print(f"  output: {out_dir}")
    print("=" * 70)

    done = []
    failed = []
    for case_id in cases:
        image_path = images_ts / f"{case_id}_0000.nii.gz"
        gt_path = labels_ts / f"{case_id}.nii.gz"
        if not image_path.is_file():
            print(f"  skip {case_id}: missing image")
            failed.append(case_id)
            continue
        try:
            kind, payload = load_case_probabilities(prob_dir, case_id)
        except FileNotFoundError:
            print(f"  skip {case_id}: missing probabilities")
            failed.append(case_id)
            continue

        out_pdf = out_dir / f"{case_id}_probability.pdf"
        try:
            img = _load_nifti(image_path)
            gt = _load_nifti(gt_path) if gt_path.is_file() else None

            if kind == "slim":
                slim: SlimProbabilities = payload  # type: ignore[assignment]
                # Ensure bbox geometry matches CT
                if slim.full_shape != tuple(int(x) for x in img.shape):
                    print(
                        f"  NOTE {case_id}: slim full_shape {slim.full_shape} "
                        f"!= CT {img.shape} — still pasting by bbox"
                    )
                probs_by_label, p_gtvp = slim_probs_to_full(slim)
            else:
                probs = np.asarray(payload, dtype=np.float32)
                if gt is not None and probs.shape[1:] != gt.shape:
                    probs, _ = align_probs_to_reference(probs, gt.shape)
                elif probs.shape[1:] != img.shape:
                    probs, _ = align_probs_to_reference(probs, img.shape)
                gtvp_idx = int(organ_dict["GTVp"])
                probs_by_label = {
                    c: probs[c] for c in range(1, probs.shape[0])
                }
                p_gtvp = probs[gtvp_idx]

            visualize_case(
                case_id=case_id,
                img=img,
                gt=gt,
                probs_by_label=probs_by_label,
                p_gtvp_full=p_gtvp,
                organ_dict=organ_dict,
                out_pdf=out_pdf,
                axis=args.axis,
                max_slices=args.max_slices,
                alpha_scale=args.alpha_scale,
                min_prob=args.min_prob,
            )
            print(f"  wrote {out_pdf.name} [{kind}]")
            done.append(case_id)
        except Exception as e:
            print(f"  fail {case_id}: {e}")
            failed.append(case_id)

    meta = {
        "done": done,
        "failed": failed,
        "n_done": len(done),
        "colormap": "MedicalImageVisualizer / Test6 (tab20, GTVp=red, GTVn=magenta)",
        "layout": ["CT", "CT+GTVp GT", "soft alpha=P(class)"],
        "full_ct": True,
    }
    with open(out_dir / "STATUS.json", "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    print(f"\nDone: {len(done)} ok, {len(failed)} failed → {out_dir}")


if __name__ == "__main__":
    main()
