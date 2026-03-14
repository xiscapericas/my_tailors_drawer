"""
Temporary module for testing head_mask_from_array with keep_top_ratio
applied on the HORIZONTAL axis (columns) instead of the vertical axis (rows).
The main implementation now lives in image_processor.utils.image_processing
(ImageProcessor.head_mask_from_array) with the same fix and default keep_top_ratio=0.6.

Usage in a notebook:
    from _temp_head_mask_from_array_fixed import head_mask_from_array_fixed
    bg_mask = head_mask_from_array_fixed(img, keep_top_ratio=0.60)
"""

import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage import filters, morphology, measure, segmentation


def head_mask_from_array_fixed(
    img: np.ndarray,
    edge_pct: float = 90.0,
    roi_radius: float = 0.42,
    close_radius: int = 7,
    min_area: int = 2000,
    keep_top_ratio: float = 0.6,
    sigma: float = 1.0,
    do_split: bool = True,
    head_top_ratio: float = 0.5,
):
    """
    Same as ImageProcessor.head_mask_from_array but keep_top_ratio is applied
    along the HORIZONTAL axis (columns) instead of the vertical axis (rows).

    Original: cut = int(keep_top_ratio * h); mask[cut:, :] = False  (vertical)
    Fixed:    cut = int(keep_top_ratio * w); mask[:, cut:] = False  (horizontal, keep left portion)
    """
    from scipy.ndimage import gaussian_filter

    if not isinstance(img, np.ndarray):
        raise TypeError("img must be a numpy.ndarray")

    if img.ndim == 3:
        if img.shape[-1] in (1, 3, 4):
            img2d = img[..., 0]
        elif img.shape[0] in (1, 3, 4):
            img2d = img[0, ...]
        else:
            img2d = img.mean(axis=-1)
    elif img.ndim == 2:
        img2d = img
    else:
        raise ValueError(f"img must be 2D (H,W) or 3D; got shape {img.shape}")

    im = img2d.astype(np.float32)
    im = np.nan_to_num(im, nan=0.0, posinf=1.0, neginf=0.0)
    p1, p99 = np.percentile(im, (1, 99))
    denom = (p99 - p1) if (p99 - p1) > 1e-6 else 1e-6
    im = np.clip((im - p1) / denom, 0.0, 1.0).astype(np.float32)

    g = gaussian_filter(im, sigma=sigma)
    edges = filters.sobel(g)
    t = np.percentile(edges, edge_pct)
    rim = edges > t

    h, w = rim.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    r = roi_radius * min(h, w)
    rim = rim & ((X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2)

    rim = morphology.binary_closing(rim, morphology.disk(close_radius))
    mask = binary_fill_holes(rim)
    mask = morphology.remove_small_holes(mask, area_threshold=min_area)
    mask = morphology.remove_small_objects(mask, min_size=min_area)

    # FIXED: apply cut along HORIZONTAL axis (columns), keeping left portion
    cut = int(keep_top_ratio * w)
    mask[:, cut:] = False

    labels = measure.label(mask)
    if labels.max() > 0:
        props = measure.regionprops(labels)
        largest = max(props, key=lambda r: r.area).label
        head_mask = labels == largest
    else:
        head_mask = np.zeros_like(mask, dtype=bool)

    head_mask = binary_fill_holes(head_mask)
    head_mask = morphology.binary_closing(head_mask, morphology.disk(5))

    if do_split:
        head_mask = _split_head_from_bottom_fixed(
            im, head_mask, head_top_ratio=head_top_ratio
        )

    background = ~head_mask
    return background


def _split_head_from_bottom_fixed(
    img: np.ndarray,
    mask: np.ndarray,
    head_top_ratio: float = 0.7,
) -> np.ndarray:
    """Same as ImageProcessor._split_head_from_bottom (unchanged)."""
    from scipy.ndimage import gaussian_filter

    h, w = img.shape
    g = gaussian_filter(img, sigma=1.0)
    grad = filters.sobel(g)
    markers = np.zeros_like(mask, np.int32)

    upper = mask.copy()
    upper[int(head_top_ratio * h):, :] = False
    head_seed = morphology.binary_erosion(upper, morphology.disk(10))
    head_seed = morphology.remove_small_objects(head_seed, 1000)
    markers[head_seed] = 1

    outside = ~mask
    bottom_band = np.zeros_like(mask, bool)
    bottom_band[int(0.80 * h):, :] = True
    bg_seed = outside | bottom_band
    bg_seed = morphology.binary_dilation(bg_seed, morphology.disk(5))
    markers[bg_seed] = 2

    seg = segmentation.watershed(grad, markers=markers, mask=mask)
    head = seg == 1
    head = binary_fill_holes(head)
    head = morphology.binary_closing(head, morphology.disk(5))

    labels = measure.label(head)
    regs = measure.regionprops(labels)
    if not regs:
        return np.zeros_like(head, bool)
    cx = w / 2
    cands = [r for r in regs if r.centroid[0] < 0.8 * h] or regs
    best = min(cands, key=lambda r: (abs(r.centroid[1] - cx), -r.area))
    return labels == best.label
