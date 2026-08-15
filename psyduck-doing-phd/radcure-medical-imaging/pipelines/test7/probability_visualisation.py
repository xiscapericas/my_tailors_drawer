#!/usr/bin/env python3
"""
Test7 — probability_visualisation

CT underlay + soft multi-class overlay on the **tumor crop**: same organ RGB
as the hard-label colormap, alpha = P(class) per voxel.

Prefers ``{case}.slim.npz``; falls back to raw full-volume ``.npz``.

Writes PDFs under ``{TEST7_WORK_ROOT}/predictions/labelsTs_probability_viz/``.

Example:

  python -m pipelines.test7.probability_visualisation
  python -m pipelines.test7.probability_visualisation --max-cases 3 --max-slices 12
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
    probability_viz_dir,
    work_root,
)
from pipelines.test7.prob_io import (
    SlimProbabilities,
    list_cases_with_probabilities,
    load_case_probabilities,
)


def _get_cmap(name: str, lut: Optional[int] = None):
    import matplotlib.pyplot as plt

    try:
        cmap = plt.colormaps[name]
        if lut is not None and hasattr(cmap, "resampled"):
            return cmap.resampled(int(lut))
        return cmap
    except (AttributeError, KeyError, TypeError):
        pass
    get_cmap = getattr(plt.cm, "get_cmap", None) or getattr(plt, "get_cmap", None)
    return get_cmap(name, lut) if lut is not None else get_cmap(name)


def build_label_colors(organ_dict: Dict[str, int], n_classes: int) -> np.ndarray:
    """RGBA colors aligned with MedicalImageVisualizer (GTVp red, GTVn magenta)."""
    colormap_size = max(n_classes, max(organ_dict.values()) + 1 if organ_dict else 1)
    base_cmap = _get_cmap("tab20", colormap_size)
    colors = base_cmap(np.arange(colormap_size)).astype(np.float32)
    colors[0, :] = [0.0, 0.0, 0.0, 0.0]

    gtvp_index = organ_dict.get("GTVp")
    gtvn_index = organ_dict.get("GTVn")
    reserved = {i for i in (gtvp_index, gtvn_index) if i is not None}

    if gtvp_index is not None and gtvp_index < colormap_size:
        colors[gtvp_index, :] = [1.0, 0.0, 0.0, 1.0]
    if gtvn_index is not None and gtvn_index < colormap_size:
        colors[gtvn_index, :] = [1.0, 0.0, 1.0, 1.0]

    red_threshold = 0.7
    for i in range(1, colormap_size):
        if i in reserved:
            continue
        if colors[i, 0] > red_threshold:
            alt = _get_cmap("Set3", colormap_size)(np.arange(colormap_size))
            colors[i, :3] = alt[i, :3]
            colors[i, 3] = 1.0
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


def soft_overlay_rgba_sparse(
    probs_by_class: Dict[int, np.ndarray],
    colors: np.ndarray,
    alpha_scale: float = 0.85,
    min_prob: float = 0.05,
) -> np.ndarray:
    """Composite soft labels: alpha = P(class). Painter order = ascending class id."""
    if not probs_by_class:
        raise ValueError("No probability channels to overlay")
    any_p = next(iter(probs_by_class.values()))
    h, w = any_p.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    for cls in sorted(probs_by_class):
        if cls <= 0 or cls >= len(colors):
            continue
        p = probs_by_class[cls]
        if float(p.max()) < min_prob:
            continue
        rgb = colors[cls, :3]
        a = np.clip(p * alpha_scale, 0.0, 1.0)
        m = a > min_prob * alpha_scale
        if not np.any(m):
            continue
        rgba[m, 0] = rgb[0]
        rgba[m, 1] = rgb[1]
        rgba[m, 2] = rgb[2]
        rgba[m, 3] = a[m]
    return rgba


def _slice_scores_gtvp(
    p_gtvp: np.ndarray, axis: int, n_slices: int
) -> List[Tuple[float, int]]:
    scores = []
    for i in range(n_slices):
        scores.append((float(_get_slice(p_gtvp, i, axis).mean()), i))
    scores.sort(reverse=True)
    return scores


def visualize_case_slim(
    case_id: str,
    img_full: np.ndarray,
    gt_full: Optional[np.ndarray],
    slim: SlimProbabilities,
    organ_dict: Dict[str, int],
    out_pdf: Path,
    axis: int = 2,
    max_slices: int = 0,
    alpha_scale: float = 0.85,
    show_gt: bool = True,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import ListedColormap, BoundaryNorm

    img = slim.crop_volume(img_full)
    gt = slim.crop_gt(gt_full) if gt_full is not None else None
    probs_k, idxs = slim.channel_stack_crop()
    p_gtvp = np.asarray(slim.p_gtvp, dtype=np.float32)

    max_label = max(int(idxs.max()) if len(idxs) else 1, max(organ_dict.values()))
    colors = build_label_colors(organ_dict, max_label + 1)
    index_to_organ = {v: k for k, v in organ_dict.items() if isinstance(v, int)}

    n_slices = img.shape[axis]
    if max_slices > 0 and n_slices > max_slices:
        scores = _slice_scores_gtvp(p_gtvp, axis, n_slices)
        slices_to_show = sorted(i for _, i in scores[:max_slices])
    else:
        slices_to_show = list(range(n_slices))

    cmap_size = len(colors)
    cmap_mask = ListedColormap(colors)
    norm_mask = BoundaryNorm(np.arange(-0.5, cmap_size + 0.5, 1), cmap_mask.N)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    x0, x1, y0, y1, z0, z1 = slim.bbox
    with PdfPages(out_pdf) as pdf:
        for slice_idx in slices_to_show:
            img_sl = _get_slice(img, slice_idx, axis)
            probs_by_class = {
                int(lab): _get_slice(probs_k[c], slice_idx, axis)
                for c, lab in enumerate(idxs)
            }
            overlay = soft_overlay_rgba_sparse(
                probs_by_class, colors, alpha_scale=alpha_scale
            )

            ncols = 3 if (show_gt and gt is not None) else 2
            fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
            if ncols == 2:
                ax_ct, ax_soft = axes
                ax_gt = None
            else:
                ax_ct, ax_gt, ax_soft = axes

            ax_ct.imshow(img_sl, cmap="gray")
            ax_ct.set_title(f"CT crop\n(slice {slice_idx})")
            ax_ct.axis("off")

            if ax_gt is not None:
                gt_sl = _get_slice(gt, slice_idx, axis)
                ax_gt.imshow(img_sl, cmap="gray")
                ax_gt.imshow(gt_sl, cmap=cmap_mask, norm=norm_mask, alpha=0.45)
                ax_gt.set_title("GT (hard, crop)")
                ax_gt.axis("off")

            ax_soft.imshow(img_sl, cmap="gray")
            ax_soft.imshow(overlay)
            ax_soft.set_title("Soft prediction\n(alpha = P(class))")
            ax_soft.axis("off")

            present = [
                lab
                for lab, p in probs_by_class.items()
                if lab > 0 and float(p.mean()) > 0.02 and lab < len(colors)
            ]
            patches = [
                mpatches.Patch(
                    color=colors[lab, :3],
                    label=f"{index_to_organ.get(lab, lab)} ({lab})",
                )
                for lab in present[:25]
            ]
            if patches:
                fig.legend(
                    handles=patches,
                    loc="center left",
                    bbox_to_anchor=(1.02, 0.5),
                    fontsize=7,
                )

            fig.suptitle(
                f"{case_id} — probability viz (bbox {x0}:{x1},{y0}:{y1},{z0}:{z1})",
                y=1.02,
            )
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def visualize_case_raw(
    case_id: str,
    img: np.ndarray,
    gt: Optional[np.ndarray],
    probs: np.ndarray,
    organ_dict: Dict[str, int],
    out_pdf: Path,
    axis: int = 2,
    max_slices: int = 0,
    alpha_scale: float = 0.85,
    show_gt: bool = True,
) -> None:
    """Fallback when only full-volume raw .npz is available."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import ListedColormap, BoundaryNorm

    colors = build_label_colors(organ_dict, probs.shape[0])
    index_to_organ = {v: k for k, v in organ_dict.items() if isinstance(v, int)}
    gtvp_idx = int(organ_dict.get("GTVp", 1))

    n_slices = min(img.shape[axis], probs.shape[axis + 1])
    if max_slices > 0 and n_slices > max_slices:
        scores = _slice_scores_gtvp(probs[gtvp_idx], axis, n_slices)
        slices_to_show = sorted(i for _, i in scores[:max_slices])
    else:
        slices_to_show = list(range(n_slices))

    cmap_mask = ListedColormap(colors)
    norm_mask = BoundaryNorm(np.arange(-0.5, len(colors) + 0.5, 1), cmap_mask.N)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_pdf) as pdf:
        for slice_idx in slices_to_show:
            img_sl = _get_slice(img, slice_idx, axis)
            probs_by_class = {
                c: _get_slice(probs[c], slice_idx, axis)
                for c in range(1, probs.shape[0])
            }
            overlay = soft_overlay_rgba_sparse(
                probs_by_class, colors, alpha_scale=alpha_scale
            )

            ncols = 3 if (show_gt and gt is not None) else 2
            fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
            axes = np.atleast_1d(axes)
            if ncols == 2:
                ax_ct, ax_soft = axes
                ax_gt = None
            else:
                ax_ct, ax_gt, ax_soft = axes

            ax_ct.imshow(img_sl, cmap="gray")
            ax_ct.set_title(f"CT\n(slice {slice_idx})")
            ax_ct.axis("off")
            if ax_gt is not None:
                gt_sl = _get_slice(gt, slice_idx, axis)
                ax_gt.imshow(img_sl, cmap="gray")
                ax_gt.imshow(gt_sl, cmap=cmap_mask, norm=norm_mask, alpha=0.45)
                ax_gt.set_title("GT (hard)")
                ax_gt.axis("off")
            ax_soft.imshow(img_sl, cmap="gray")
            ax_soft.imshow(overlay)
            ax_soft.set_title("Soft prediction\n(alpha = P(class))")
            ax_soft.axis("off")

            present = [
                c
                for c, p in probs_by_class.items()
                if float(p.mean()) > 0.02 and c < len(colors)
            ]
            patches = [
                mpatches.Patch(
                    color=colors[c, :3],
                    label=f"{index_to_organ.get(c, c)} ({c})",
                )
                for c in present[:25]
            ]
            if patches:
                fig.legend(
                    handles=patches,
                    loc="center left",
                    bbox_to_anchor=(1.02, 0.5),
                    fontsize=7,
                )
            fig.suptitle(f"{case_id} — probability visualisation (raw)", y=1.02)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test7: probability_visualisation")
    parser.add_argument("--work-root", default=str(work_root()))
    parser.add_argument("--axis", type=int, default=2, help="0=sag, 1=cor, 2=ax")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument(
        "--max-slices",
        type=int,
        default=16,
        help="Max slices per case (0 = all; default prefers high-P(GTVp) slices)",
    )
    parser.add_argument("--alpha-scale", type=float, default=0.85)
    parser.add_argument("--no-gt", action="store_true")
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
    cases = list_cases_with_probabilities(prob_dir)
    if not cases:
        raise FileNotFoundError(
            f"No .slim.npz / .npz probabilities in {prob_dir}\n"
            "Run: python -m pipelines.test7.predict_probabilities"
        )
    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    print("=" * 70)
    print("Test7 — probability_visualisation")
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
            gt = (
                None
                if args.no_gt or not gt_path.is_file()
                else _load_nifti(gt_path)
            )
            if kind == "slim":
                visualize_case_slim(
                    case_id=case_id,
                    img_full=img,
                    gt_full=gt,
                    slim=payload,  # type: ignore[arg-type]
                    organ_dict=organ_dict,
                    out_pdf=out_pdf,
                    axis=args.axis,
                    max_slices=args.max_slices,
                    alpha_scale=args.alpha_scale,
                    show_gt=not args.no_gt,
                )
            else:
                visualize_case_raw(
                    case_id=case_id,
                    img=img,
                    gt=gt,
                    probs=payload,  # type: ignore[arg-type]
                    organ_dict=organ_dict,
                    out_pdf=out_pdf,
                    axis=args.axis,
                    max_slices=args.max_slices,
                    alpha_scale=args.alpha_scale,
                    show_gt=not args.no_gt,
                )
            print(f"  wrote {out_pdf.name} [{kind}]")
            done.append(case_id)
        except Exception as e:
            print(f"  fail {case_id}: {e}")
            failed.append(case_id)

    meta = {"done": done, "failed": failed, "n_done": len(done)}
    with open(out_dir / "STATUS.json", "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    print(f"\nDone: {len(done)} ok, {len(failed)} failed → {out_dir}")


if __name__ == "__main__":
    main()
