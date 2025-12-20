"""Image processing utilities for head/background detection."""

import numpy as np
from scipy.ndimage import gaussian_filter, binary_fill_holes, distance_transform_edt
from skimage import filters, morphology, measure, segmentation
from typing import Optional


class ImageProcessor:
    """Image processing utilities for medical images."""
    
    @staticmethod
    def head_mask_from_array(
        img: np.ndarray,
        edge_pct: float = 90.0,
        roi_radius: float = 0.42,
        close_radius: int = 7,
        min_area: int = 2000,
        keep_top_ratio: float = 0.70,
        sigma: float = 1.0,
        do_split: bool = True,
        head_top_ratio: float = 0.5
    ) -> np.ndarray:
        """
        Generate background mask from 2D CT slice.
        
        Parameters
        ----------
        img : np.ndarray
            2D image array (H, W)
        edge_pct : float
            Percentile on edge magnitude
        roi_radius : float
            Fraction of min(H,W) for central ROI
        close_radius : int
            Morphological closing radius
        min_area : int
            Minimum area for removing small blobs
        keep_top_ratio : float
            Keep only top % of image
        sigma : float
            Gaussian smoothing sigma
        do_split : bool
            Whether to apply split_head_from_bottom
        head_top_ratio : float
            Ratio for head splitting
        
        Returns
        -------
        np.ndarray
            Background mask (True = background, False = head)
        """
        if not isinstance(img, np.ndarray):
            raise TypeError("img must be a numpy.ndarray")
        
        # Handle 3D arrays
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
        
        # Clean NaN/Inf
        im = np.nan_to_num(im, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Robust [0,1] normalization
        p1, p99 = np.percentile(im, (1, 99))
        denom = (p99 - p1) if (p99 - p1) > 1e-6 else 1e-6
        im = np.clip((im - p1) / denom, 0.0, 1.0).astype(np.float32)
        
        # Smooth + edges
        g = gaussian_filter(im, sigma=sigma)
        edges = filters.sobel(g)
        
        # Threshold edges by percentile
        t = np.percentile(edges, edge_pct)
        rim = edges > t
        
        # Central ROI gating
        h, w = rim.shape
        cy, cx = h // 2, w // 2
        Y, X = np.ogrid[:h, :w]
        r = roi_radius * min(h, w)
        rim = rim & ((X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2)
        
        # Close + fill holes + remove small artifacts
        rim = morphology.binary_closing(rim, morphology.disk(close_radius))
        mask = binary_fill_holes(rim)
        mask = morphology.remove_small_holes(mask, area_threshold=min_area)
        mask = morphology.remove_small_objects(mask, min_size=min_area)
        
        # Keep only the top part
        cut = int(keep_top_ratio * h)
        mask[cut:, :] = False
        
        # Keep largest connected component
        labels = measure.label(mask)
        if labels.max() > 0:
            props = measure.regionprops(labels)
            largest = max(props, key=lambda r: r.area).label
            head_mask = (labels == largest)
        else:
            head_mask = np.zeros_like(mask, dtype=bool)
        
        head_mask = binary_fill_holes(head_mask)
        head_mask = morphology.binary_closing(head_mask, morphology.disk(5))
        
        # Optional: custom split step
        if do_split:
            head_mask = ImageProcessor._split_head_from_bottom(
                im, head_mask, head_top_ratio=head_top_ratio
            )
        
        background = ~head_mask
        return background
    
    @staticmethod
    def _split_head_from_bottom(
        img: np.ndarray,
        mask: np.ndarray,
        head_top_ratio: float = 0.7
    ) -> np.ndarray:
        """
        Split head from bottom using watershed segmentation.
        
        Parameters
        ----------
        img : np.ndarray
            2D float image (normalized to [0,1])
        mask : np.ndarray
            Initial binary foreground mask
        head_top_ratio : float
            Ratio for head marker placement
        
        Returns
        -------
        np.ndarray
            Refined head mask
        """
        h, w = img.shape
        
        # Smooth and compute gradient
        g = gaussian_filter(img, sigma=1.0)
        grad = filters.sobel(g)
        
        # Build markers (0=unknown, 1=head, 2=background/bottom)
        markers = np.zeros_like(mask, np.int32)
        
        # Head marker: safe interior above midline
        upper = mask.copy()
        upper[int(head_top_ratio * h):, :] = False
        head_seed = morphology.binary_erosion(upper, morphology.disk(10))
        head_seed = morphology.remove_small_objects(head_seed, 1000)
        markers[head_seed] = 1
        
        # Background markers: outside & bottom band
        outside = ~mask
        bottom_band = np.zeros_like(mask, bool)
        bottom_band[int(0.80 * h):, :] = True
        bg_seed = outside | bottom_band
        bg_seed = morphology.binary_dilation(bg_seed, morphology.disk(5))
        markers[bg_seed] = 2
        
        # Watershed on gradient
        seg = segmentation.watershed(grad, markers=markers, mask=mask)
        head = (seg == 1)
        
        # Cleanup
        head = binary_fill_holes(head)
        head = morphology.binary_closing(head, morphology.disk(5))
        
        # Keep the most head-like component
        labels = measure.label(head)
        regs = measure.regionprops(labels)
        if not regs:
            return np.zeros_like(head, bool)
        cx = w / 2
        cands = [r for r in regs if r.centroid[0] < 0.8 * h] or regs
        best = min(cands, key=lambda r: (abs(r.centroid[1] - cx), -r.area))
        return labels == best.label
    
    @staticmethod
    def get_non_zero_slices(mask_3d: np.ndarray) -> list:
        """
        Get list of slice indices with non-zero values.
        
        Parameters
        ----------
        mask_3d : np.ndarray
            3D mask array
        
        Returns
        -------
        list
            List of slice indices with non-zero values
        """
        return [i for i in range(0, mask_3d.shape[2]) if np.sum(mask_3d[:, :, i]) > 0]

