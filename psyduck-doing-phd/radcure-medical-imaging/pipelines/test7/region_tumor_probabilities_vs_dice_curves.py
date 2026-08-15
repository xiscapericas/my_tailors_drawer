#!/usr/bin/env python3
"""
Test7 — region_tumor_probabilities_vs_dice_curves

For each organ/region that spatially overlaps GTVp in ground truth (dilated
neighborhood), plot:

  X = P(GTVp) − P(region)
  Y = 1 if voxel is GTVp in GT, else 0

Also tests the hypothesis:
  If P(GTVp) ≥ 0.80 — even when another class has higher probability —
  the voxel can still be called GTVp.

Outputs under ``{TEST7_WORK_ROOT}/region_tumor_probabilities_vs_dice_curves/``.

Example:

  python -m pipelines.test7.region_tumor_probabilities_vs_dice_curves
  python -m pipelines.test7.region_tumor_probabilities_vs_dice_curves --max-cases 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipelines.test7.paths import (
    competing_region_names,
    curves_dir,
    load_organ_dict,
    pin_test7_env,
    probabilities_dir,
    work_root,
)
from pipelines.test7.prob_io import (
    SlimProbabilities,
    list_cases_with_probabilities,
    load_case_probabilities,
)


def _load_nifti_mask(path: Path) -> np.ndarray:
    import nibabel as nib

    return np.asanyarray(nib.load(str(path)).dataobj)


def _dilate_binary(mask: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Cheap binary dilation via max-pooling neighborhood (no scipy required)."""
    if iterations <= 0:
        return mask.astype(bool)
    out = mask.astype(bool)
    for _ in range(iterations):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        neigh = (
            padded[1:-1, 1:-1, 1:-1]
            | padded[:-2, 1:-1, 1:-1]
            | padded[2:, 1:-1, 1:-1]
            | padded[1:-1, :-2, 1:-1]
            | padded[1:-1, 2:, 1:-1]
            | padded[1:-1, 1:-1, :-2]
            | padded[1:-1, 1:-1, 2:]
        )
        out = neigh
    return out


def discover_overlapping_regions(
    gt: np.ndarray,
    gtvp_idx: int,
    organ_dict: Dict[str, int],
    dilate_iter: int = 2,
) -> List[Tuple[str, int]]:
    """
    Regions whose GT voxels intersect a dilated GTVp mask.

    Excludes background / anatomical_region / other-tissue / GTVp / GTVn.
    """
    gtvp = gt == gtvp_idx
    if not np.any(gtvp):
        return []
    neighborhood = _dilate_binary(gtvp, iterations=dilate_iter)
    present = set(int(x) for x in np.unique(gt[neighborhood]))
    present.discard(0)
    present.discard(gtvp_idx)

    name_by_idx = {v: k for k, v in organ_dict.items() if isinstance(v, int)}
    allowed = set(competing_region_names(organ_dict))
    out: List[Tuple[str, int]] = []
    for idx in sorted(present):
        name = name_by_idx.get(idx)
        if name is None or name not in allowed:
            continue
        out.append((name, idx))
    return out


def _binned_positive_rate(
    x: np.ndarray, y: np.ndarray, n_bins: int = 21
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return bin centers, P(Y=1|bin), counts."""
    if len(x) == 0:
        empty = np.array([])
        return empty, empty, empty
    edges = np.linspace(-1.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    rates = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        if i < n_bins - 1:
            m = (x >= edges[i]) & (x < edges[i + 1])
        else:
            m = (x >= edges[i]) & (x <= edges[i + 1])
        counts[i] = int(m.sum())
        if counts[i] > 0:
            rates[i] = float(y[m].mean())
    return centers, rates, counts


def _binomial_pvalue(successes: int, n: int, p0: float = 0.5) -> float:
    """Two-sided exact binomial test; prefers scipy, else normal approx."""
    if n <= 0:
        return float("nan")
    try:
        from scipy.stats import binomtest

        return float(binomtest(successes, n, p=p0, alternative="two-sided").pvalue)
    except ImportError:
        pass
    # Normal approximation with continuity correction
    phat = successes / n
    se = np.sqrt(p0 * (1.0 - p0) / n)
    if se == 0:
        return float("nan")
    z = abs(phat - p0) / se
    # erfc for two-sided normal p
    from math import erfc, sqrt

    return float(erfc(z / sqrt(2.0)))


def analyze_case(
    case_id: str,
    gt: np.ndarray,
    probs: np.ndarray,
    organ_dict: Dict[str, int],
    dilate_iter: int,
    max_voxels_per_region: int,
    rng: np.random.Generator,
    *,
    available_class_indices: Optional[Sequence[int]] = None,
) -> Tuple[Dict[str, dict], List[dict]]:
    """
    Returns per-region summaries and voxel-row dicts (possibly subsampled).

    ``gt`` and ``probs`` must share the same spatial grid (full volume or crop).
    If ``available_class_indices`` is set (slim dumps), only those region
    channels are analyzed.
    """
    gtvp_idx = int(organ_dict["GTVp"])
    if probs.ndim != 4:
        raise ValueError(f"{case_id}: probs must be (C,X,Y,Z), got {probs.shape}")
    if gt.shape != probs.shape[1:]:
        raise ValueError(
            f"{case_id}: GT shape {gt.shape} != prob spatial {probs.shape[1:]}"
        )

    # Map label index → channel in probs (identity for raw full-C dumps)
    if available_class_indices is None:
        channel_for = {i: i for i in range(probs.shape[0])}
        p_gtvp = probs[gtvp_idx] if gtvp_idx < probs.shape[0] else None
    else:
        channel_for = {int(lab): i for i, lab in enumerate(available_class_indices)}
        if gtvp_idx not in channel_for:
            raise ValueError(f"{case_id}: slim dump missing GTVp channel")
        p_gtvp = probs[channel_for[gtvp_idx]]

    if p_gtvp is None:
        raise ValueError(
            f"{case_id}: probs has C={probs.shape[0]} but GTVp index={gtvp_idx}"
        )

    regions = discover_overlapping_regions(gt, gtvp_idx, organ_dict, dilate_iter)
    y_gtvp = (gt == gtvp_idx).astype(np.uint8)

    region_summaries: Dict[str, dict] = {}
    rows: List[dict] = []

    for name, r_idx in regions:
        if r_idx not in channel_for:
            # Slim dump did not keep this overlapping organ — skip
            continue
        ch = channel_for[r_idx]
        support = (gt == gtvp_idx) | (gt == r_idx)
        if not np.any(support):
            continue
        x = (p_gtvp - probs[ch])[support]
        y = y_gtvp[support]
        n = int(y.size)
        if n > max_voxels_per_region > 0:
            pick = rng.choice(n, size=max_voxels_per_region, replace=False)
            x_s, y_s = x[pick], y[pick]
        else:
            x_s, y_s = x, y

        centers, rates, counts = _binned_positive_rate(x_s, y_s)
        region_summaries[name] = {
            "region_index": r_idx,
            "n_voxels_support": n,
            "n_voxels_plotted": int(y_s.size),
            "mean_x": float(x_s.mean()) if y_s.size else None,
            "frac_gtvp": float(y_s.mean()) if y_s.size else None,
            "bin_centers": centers.tolist(),
            "bin_positive_rate": rates.tolist(),
            "bin_counts": counts.tolist(),
        }
        step = max(1, len(x_s) // 2000)
        for xi, yi in zip(x_s[::step], y_s[::step]):
            rows.append(
                {
                    "case_id": case_id,
                    "region": name,
                    "x_p_gtvp_minus_p_region": float(xi),
                    "y_is_gtvp_gt": int(yi),
                }
            )

    return region_summaries, rows


def analyze_slim_case(
    case_id: str,
    gt_full: np.ndarray,
    slim: SlimProbabilities,
    organ_dict: Dict[str, int],
    dilate_iter: int,
    max_voxels_per_region: int,
    rng: np.random.Generator,
) -> Tuple[Dict[str, dict], List[dict]]:
    gt_c = slim.crop_gt(gt_full)
    probs_k, idxs = slim.channel_stack_crop()
    return analyze_case(
        case_id,
        gt_c,
        probs_k,
        organ_dict,
        dilate_iter,
        max_voxels_per_region,
        rng,
        available_class_indices=idxs.tolist(),
    )


def hypothesis_gtvp_ge_threshold(
    gt: np.ndarray,
    probs: np.ndarray,
    gtvp_idx: int,
    threshold: float = 0.80,
    *,
    available_class_indices: Optional[Sequence[int]] = None,
) -> dict:
    """
    Among voxels where P(GTVp) ≥ threshold AND some other class has higher
    probability than GTVp, what fraction are true GTVp in GT?

    For slim dumps, ``other`` is max over **stored** non-GTVp channels only.
    """
    if available_class_indices is None:
        p_gtvp = probs[gtvp_idx]
        other = np.concatenate(
            [probs[:gtvp_idx], probs[gtvp_idx + 1 :]], axis=0
        )
    else:
        channel_for = {int(lab): i for i, lab in enumerate(available_class_indices)}
        if gtvp_idx not in channel_for:
            raise ValueError("GTVp channel missing from slim probabilities")
        p_gtvp = probs[channel_for[gtvp_idx]]
        other_chs = [
            probs[i]
            for lab, i in channel_for.items()
            if lab != gtvp_idx
        ]
        other = np.stack(other_chs, axis=0) if other_chs else np.zeros((0, *p_gtvp.shape))

    p_other_max = other.max(axis=0) if other.size else np.zeros_like(p_gtvp)
    mask = (p_gtvp >= threshold) & (p_other_max > p_gtvp)
    n = int(mask.sum())
    if n == 0:
        return {
            "threshold": threshold,
            "n_voxels": 0,
            "n_true_gtvp": 0,
            "precision_as_gtvp": None,
            "p_value_vs_0.5": None,
            "other_scope": "stored_classes"
            if available_class_indices is not None
            else "all_classes",
        }
    n_true = int(((gt == gtvp_idx) & mask).sum())
    prec = n_true / n
    return {
        "threshold": threshold,
        "n_voxels": n,
        "n_true_gtvp": n_true,
        "precision_as_gtvp": float(prec),
        "p_value_vs_0.5": _binomial_pvalue(n_true, n, p0=0.5),
        "other_scope": "stored_classes"
        if available_class_indices is not None
        else "all_classes",
    }


def hypothesis_slim(
    gt_full: np.ndarray,
    slim: SlimProbabilities,
    threshold: float = 0.80,
) -> dict:
    gt_c = slim.crop_gt(gt_full)
    probs_k, idxs = slim.channel_stack_crop()
    return hypothesis_gtvp_ge_threshold(
        gt_c,
        probs_k,
        slim.gtvp_index,
        threshold,
        available_class_indices=idxs.tolist(),
    )


def plot_region_curves(
    pooled: Dict[str, dict],
    out_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    if not pooled:
        return
    n = len(pooled)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    for ax, (name, summary) in zip(axes.ravel(), pooled.items()):
        c = np.asarray(summary["bin_centers"])
        r = np.asarray(summary["bin_positive_rate"])
        ax.plot(c, r, marker="o", ms=3)
        ax.axvline(0.0, color="gray", ls="--", lw=0.8)
        ax.axhline(0.5, color="gray", ls=":", lw=0.8)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("P(GTVp) − P(%s)" % name)
        ax.set_ylabel("P(GT is GTVp | bin)")
        ax.set_title(name)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def pool_region_summaries(
    per_case: Sequence[Dict[str, dict]],
) -> Dict[str, dict]:
    """Average binned positive rates across cases (count-weighted)."""
    # Collect bin counts / weighted rates per region
    centers_ref: Optional[np.ndarray] = None
    sum_count: Dict[str, np.ndarray] = {}
    sum_rate_w: Dict[str, np.ndarray] = {}
    n_cases: Dict[str, int] = defaultdict(int)

    for case_map in per_case:
        for name, s in case_map.items():
            c = np.asarray(s["bin_centers"], dtype=float)
            rates = np.asarray(s["bin_positive_rate"], dtype=float)
            counts = np.asarray(s["bin_counts"], dtype=float)
            if centers_ref is None:
                centers_ref = c
            if name not in sum_count:
                sum_count[name] = np.zeros_like(counts)
                sum_rate_w[name] = np.zeros_like(counts)
            valid = counts > 0
            sum_count[name][valid] += counts[valid]
            sum_rate_w[name][valid] += rates[valid] * counts[valid]
            n_cases[name] += 1

    pooled: Dict[str, dict] = {}
    if centers_ref is None:
        return pooled
    for name, counts in sum_count.items():
        rates = np.full_like(counts, np.nan)
        m = counts > 0
        rates[m] = sum_rate_w[name][m] / counts[m]
        pooled[name] = {
            "n_cases": n_cases[name],
            "bin_centers": centers_ref.tolist(),
            "bin_positive_rate": rates.tolist(),
            "bin_counts": counts.astype(int).tolist(),
        }
    return pooled


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test7: region_tumor_probabilities_vs_dice_curves"
    )
    parser.add_argument("--work-root", default=str(work_root()))
    parser.add_argument("--dilate", type=int, default=2, help="GTVp dilate iterations for overlap discovery")
    parser.add_argument(
        "--max-voxels-per-region",
        type=int,
        default=200_000,
        help="Subsample support voxels per region per case (0 = all)",
    )
    parser.add_argument("--max-cases", type=int, default=0, help="Limit cases (0 = all)")
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    work = Path(args.work_root).expanduser().resolve()
    paths = pin_test7_env(work)
    dataset = paths["dataset"]
    labels_ts = dataset / "labelsTs"
    prob_dir = probabilities_dir(work)
    out_dir = curves_dir(work)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not prob_dir.is_dir() or not list_cases_with_probabilities(prob_dir):
        raise FileNotFoundError(
            f"No probability dumps (.slim.npz or .npz) in {prob_dir}\n"
            "Run: python -m pipelines.test7.predict_probabilities\n"
            "  or: python -m pipelines.test7.slim_probabilities"
        )

    organ_dict = load_organ_dict(paths.get("organ"))
    gtvp_idx = int(organ_dict["GTVp"])
    rng = np.random.default_rng(args.seed)

    cases = list_cases_with_probabilities(prob_dir)
    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    all_rows: List[dict] = []
    per_case_summaries: List[Dict[str, dict]] = []
    hyp_rows: List[dict] = []
    case_region_json: Dict[str, dict] = {}

    print("=" * 70)
    print("Test7 — region_tumor_probabilities_vs_dice_curves")
    print(f"  cases:     {len(cases)}")
    print(f"  probs:     {prob_dir}")
    print(f"  labelsTs:  {labels_ts}")
    print(f"  output:    {out_dir}")
    print(f"  threshold: {args.threshold}")
    print("=" * 70)

    for case_id in cases:
        gt_path = labels_ts / f"{case_id}.nii.gz"
        if not gt_path.is_file():
            print(f"  skip {case_id}: missing GT {gt_path}")
            continue
        try:
            kind, payload = load_case_probabilities(prob_dir, case_id)
        except FileNotFoundError:
            print(f"  skip {case_id}: missing probabilities")
            continue
        gt = _load_nifti_mask(gt_path)
        try:
            if kind == "slim":
                slim = payload  # type: SlimProbabilities
                region_sum, rows = analyze_slim_case(
                    case_id,
                    gt,
                    slim,
                    organ_dict,
                    dilate_iter=args.dilate,
                    max_voxels_per_region=args.max_voxels_per_region,
                    rng=rng,
                )
                hyp = hypothesis_slim(gt, slim, args.threshold)
            else:
                probs = payload  # type: np.ndarray
                region_sum, rows = analyze_case(
                    case_id,
                    gt,
                    probs,
                    organ_dict,
                    dilate_iter=args.dilate,
                    max_voxels_per_region=args.max_voxels_per_region,
                    rng=rng,
                )
                hyp = hypothesis_gtvp_ge_threshold(
                    gt, probs, gtvp_idx, args.threshold
                )
        except ValueError as e:
            print(f"  skip {case_id}: {e}")
            continue

        hyp["case_id"] = case_id
        hyp["prob_source"] = kind
        hyp_rows.append(hyp)
        all_rows.extend(rows)
        per_case_summaries.append(region_sum)
        case_region_json[case_id] = {
            "regions": region_sum,
            "hypothesis": hyp,
            "prob_source": kind,
        }
        print(
            f"  {case_id} [{kind}]: regions={list(region_sum)} "
            f"hyp_n={hyp['n_voxels']} prec={hyp['precision_as_gtvp']}"
        )

    # Pooled curves
    pooled = pool_region_summaries(per_case_summaries)
    plot_region_curves(
        pooled,
        out_dir / "pooled_region_curves.png",
        title="Pooled: P(GT is GTVp) vs P(GTVp)−P(region)",
    )

    # Per-region figures
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    for name, summary in pooled.items():
        plot_region_curves(
            {name: summary},
            figures_dir / f"{name}_curve.png",
            title=f"{name}: P(GT is GTVp) vs P(GTVp)−P({name})",
        )

    if all_rows:
        pd.DataFrame(all_rows).to_csv(out_dir / "voxel_samples.csv", index=False)

    hyp_df = pd.DataFrame(hyp_rows)
    if not hyp_df.empty:
        hyp_df.to_csv(out_dir / "hypothesis_per_case.csv", index=False)
        # Aggregate hypothesis across cases (sum voxels)
        n = int(hyp_df["n_voxels"].sum())
        n_true = int(hyp_df["n_true_gtvp"].sum())
        agg = {
            "threshold": args.threshold,
            "n_voxels": n,
            "n_true_gtvp": n_true,
            "precision_as_gtvp": (n_true / n) if n else None,
            "p_value_vs_0.5": _binomial_pvalue(n_true, n, p0=0.5) if n else None,
            "n_cases": int(len(hyp_df)),
            "rule": (
                f"Call GTVp when P(GTVp)>={args.threshold} even if another "
                "class has higher probability"
            ),
        }
    else:
        agg = {"threshold": args.threshold, "n_voxels": 0}

    summary = {
        "n_cases_processed": len(case_region_json),
        "pooled_regions": list(pooled.keys()),
        "hypothesis_aggregate": agg,
        "dilate_iter": args.dilate,
        "gtvp_index": gtvp_idx,
        "definition": {
            "overlap": "GT labels intersecting dilated GTVp GT mask (within slim crop when using .slim.npz)",
            "x": "P(GTVp) - P(region)",
            "y": "1 if GT label == GTVp else 0",
            "support": "voxels where GT is GTVp or the competing region",
            "probs": "Prefer {case}.slim.npz (cropped float16); raw .npz supported as fallback",
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    with open(out_dir / "per_case_regions.json", "w") as f:
        json.dump(case_region_json, f, indent=2)
        f.write("\n")
    with open(out_dir / "pooled_curves.json", "w") as f:
        json.dump(pooled, f, indent=2)
        f.write("\n")

    print("\nHypothesis aggregate:")
    print(json.dumps(agg, indent=2))
    print(f"\nWrote outputs under {out_dir}")


if __name__ == "__main__":
    main()
