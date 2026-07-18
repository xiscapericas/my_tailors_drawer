"""Image processing utilities for background and anatomical region detection."""

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
        keep_top_ratio: float = 0.6,
        sigma: float = 1.0,
        do_split: bool = True,
        head_top_ratio: float = 0.5
    ) -> np.ndarray:
        """
        Generate background mask from 2D CT slice.
        
        Detects patient outline (anatomical region: head or body) vs background;
        the exact body region is not distinguished. Caller maps to indices
        (e.g. 0 = background, 1 = anatomical_region).
        
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
            Fraction of image width to keep (horizontal axis); columns beyond this are set to background. Default 0.6.
        sigma : float
            Gaussian smoothing sigma
        do_split : bool
            Whether to apply split_head_from_bottom
        head_top_ratio : float
            Ratio for head splitting
        
        Returns
        -------
        np.ndarray
            Boolean mask: True = background (outside patient), False = anatomical region (patient outline).
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
        
        # Keep only the left portion (horizontal axis): columns beyond cut are background
        cut = int(keep_top_ratio * w)
        mask[:, cut:] = False
        
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
    def body_mask_from_intensity(
        img: np.ndarray,
        *,
        sigma: float = 1.0,
        min_area: int = 1500,
        max_fill: float = 0.55,
        fov_floor: float = 0.02,
    ) -> np.ndarray:
        """
        Patient-vs-background mask for H&N / shoulder axial CT.

        Same return convention as ``head_mask_from_array``:
        True = background, False = anatomical (patient).

        Design goals
        ------------
        - Body is usually **centered**; corners outside the reconstruction FOV
          are black and must not drive the threshold (naive Otsu on the full
          image often labels the entire FOV circle as "patient").
        - Works for **head and shoulders** (no left-crop, no head watershed).
        - Always leaves background: if fill is too high, raise the tissue
          threshold and prefer the connected component nearest the center.
        """
        if not isinstance(img, np.ndarray):
            raise TypeError("img must be a numpy.ndarray")
        if img.ndim != 2:
            raise ValueError(f"img must be 2D (H,W); got shape {img.shape}")

        im = np.nan_to_num(img.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
        p1, p99 = np.percentile(im, (1, 99))
        denom = (p99 - p1) if (p99 - p1) > 1e-6 else 1e-6
        im = np.clip((im - p1) / denom, 0.0, 1.0)
        g = gaussian_filter(im, sigma=sigma)

        h, w = g.shape
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0

        # 1) Reconstruction FOV = non-black interior (exclude corner void)
        fov = g > fov_floor
        fov = morphology.remove_small_objects(fov, min_size=min_area)
        if int(fov.sum()) < min_area:
            return np.ones((h, w), dtype=bool)

        fov_labels = measure.label(fov)
        if fov_labels.max() > 0:
            props = measure.regionprops(fov_labels)
            fov = fov_labels == max(props, key=lambda r: r.area).label

        fov_vals = g[fov]

        # 2) Tissue vs air *inside* FOV (not FOV vs corners)
        t = float(filters.threshold_otsu(fov_vals))
        patient = fov & (g > t)

        # If almost the whole FOV is "tissue", threshold was too low → bump
        fill_in_fov = float(patient.sum()) / float(max(int(fov.sum()), 1))
        fill_img = float(patient.mean())
        if fill_in_fov > 0.80 or fill_img > max_fill:
            t = max(t, float(np.percentile(fov_vals, 60)))
            patient = fov & (g > t)

        patient = binary_fill_holes(patient)
        patient = morphology.remove_small_objects(patient, min_size=min_area)
        patient = morphology.binary_closing(patient, morphology.disk(5))
        patient = binary_fill_holes(patient)
        patient = patient & fov

        # 3) Prefer a large component near the image center (body is centered)
        labels = measure.label(patient)
        if labels.max() == 0:
            return np.ones((h, w), dtype=bool)

        props = [r for r in measure.regionprops(labels) if r.area >= min_area]
        if not props:
            props = list(measure.regionprops(labels))

        def _center_score(r):
            ry, rx = r.centroid
            dist2 = (ry - cy) ** 2 + (rx - cx) ** 2
            return dist2 / (1.0 + r.area)  # smaller is better

        best = min(props, key=_center_score)
        patient = labels == best.label
        patient = binary_fill_holes(patient)

        # Final guard: never allow near-full-image anatomy
        if float(patient.mean()) > max_fill:
            # Erode until under cap or empty
            eroded = patient.copy()
            for _ in range(8):
                if float(eroded.mean()) <= max_fill:
                    break
                eroded = morphology.binary_erosion(eroded, morphology.disk(3))
            if int(eroded.sum()) >= min_area:
                patient = binary_fill_holes(eroded)
            else:
                # Keep only central core of current mask
                yy, xx = np.ogrid[:h, :w]
                core = ((yy - cy) ** 2 + (xx - cx) ** 2) <= (0.35 * min(h, w)) ** 2
                patient = patient & core
                patient = binary_fill_holes(patient)

        return ~patient

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

