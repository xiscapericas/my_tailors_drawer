"""Image processing utilities for background and anatomical region detection."""

import numpy as np
from scipy.ndimage import (
    gaussian_filter,
    binary_fill_holes,
    binary_closing as nd_binary_closing,
    distance_transform_edt,
)
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
        enforce_symmetry: bool = True,
        sagittal_flip_axis: int = 0,
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
        - Optional **sagittal (L/R) symmetry**: ``sagittal_flip_axis=0`` matches
          left-right after ``imshow(img.T)`` used in this project (not A/P).
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
        tissue = fov & (g > t)

        fill_in_fov = float(tissue.sum()) / float(max(int(fov.sum()), 1))
        fill_img = float(tissue.mean())
        if fill_in_fov > 0.80 or fill_img > max_fill:
            t = max(t, float(np.percentile(fov_vals, 60)))
            tissue = fov & (g > t)

        patient = tissue.copy()
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
            return dist2 / (1.0 + r.area)

        best = min(props, key=_center_score)
        patient = labels == best.label
        patient = binary_fill_holes(patient)

        # 4) Sagittal symmetry (patient L/R).
        # Notebook viz uses imshow(..., .T), so display left-right == array axis 0.
        # Using fliplr (axis 1) wrongly mirrors anterior/posterior → table/top artifacts.
        if enforce_symmetry:
            t_sym = float(np.percentile(fov_vals, 45))  # body vs darker FOV air
            soft_tissue = fov & (g > t_sym)
            patient = ImageProcessor._enforce_sagittal_symmetry(
                patient,
                tissue_candidate=soft_tissue,
                fov=fov,
                min_area=max(500, min_area // 3),
                flip_axis=sagittal_flip_axis,
            )

        # Final guard: never allow near-full-image anatomy
        if float(patient.mean()) > max_fill:
            eroded = patient.copy()
            for _ in range(8):
                if float(eroded.mean()) <= max_fill:
                    break
                eroded = morphology.binary_erosion(eroded, morphology.disk(3))
            if int(eroded.sum()) >= min_area:
                patient = binary_fill_holes(eroded)
            else:
                yy, xx = np.ogrid[:h, :w]
                core = ((yy - cy) ** 2 + (xx - cx) ** 2) <= (0.35 * min(h, w)) ** 2
                patient = patient & core
                patient = binary_fill_holes(patient)

        return ~patient

    @staticmethod
    def _enforce_sagittal_symmetry(
        patient: np.ndarray,
        *,
        tissue_candidate: np.ndarray,
        fov: np.ndarray,
        min_area: int = 500,
        flip_axis: int = 0,
    ) -> np.ndarray:
        """
        Left/right (sagittal) symmetry about the mid-plane.

        Parameters
        ----------
        flip_axis
            0 = ``flipud`` (L/R when volumes are shown with ``imshow(img.T)``),
            1 = ``fliplr`` (L/R when rows=Y, cols=X without transpose).

        Only adds mirrored voxels that:
        - fall inside FOV
        - look like body vs air (``tissue_candidate``)
        - lie near the current patient bbox (avoids painting table / top bar)
        """
        if flip_axis not in (0, 1):
            raise ValueError("flip_axis must be 0 or 1")

        mirrored = np.flip(patient, axis=flip_axis)

        # Restrict additions to a padded bbox of the current patient (no table invent)
        ys, xs = np.where(patient)
        h, w = patient.shape
        roi = np.zeros_like(patient, dtype=bool)
        if len(ys) > 0:
            pad = max(8, int(0.04 * max(h, w)))
            y0, y1 = max(0, int(ys.min()) - pad), min(h, int(ys.max()) + pad + 1)
            x0, x1 = max(0, int(xs.min()) - pad), min(w, int(xs.max()) + pad + 1)
            # Expand bbox on the flip axis so contralateral side is allowed
            if flip_axis == 0:
                # flipping rows: allow full height span of bbox, widen? actually L/R is rows
                # contralateral is the flipped row range — use full padded bbox flipped
                roi[y0:y1, x0:x1] = True
                roi = roi | np.flip(roi, axis=0)
            else:
                roi[y0:y1, x0:x1] = True
                roi = roi | np.flip(roi, axis=1)
        else:
            return patient

        fill = mirrored & tissue_candidate & fov & roi & (~patient)
        out = (patient | fill) & fov
        out = binary_fill_holes(out)
        out = morphology.binary_closing(out, morphology.disk(5))
        out = binary_fill_holes(out)
        out = morphology.remove_small_objects(out, min_size=min_area)

        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        labels = measure.label(out)
        if labels.max() == 0:
            return patient
        props = list(measure.regionprops(labels))
        best = min(
            props,
            key=lambda r: ((r.centroid[0] - cy) ** 2 + (r.centroid[1] - cx) ** 2)
            / (1.0 + r.area),
        )
        out = labels == best.label
        return binary_fill_holes(out)

    @staticmethod
    def enforce_anatomical_continuity(
        background_masks: list,
        *,
        min_area: int = 800,
        z_radius: int = 2,
    ) -> list:
        """
        Fill isolated empty / tiny anatomical slices using Z-neighbors.

        More aggressive than a single-neighbor patch:
        - weak slice ← OR of all strong slices within ±z_radius
        - then binary closing along Z (bridges 1–2 slice gaps)
        - does **not** re-apply a high min_area wipe after fill
        """
        if not background_masks:
            return []

        patients = []
        for m in background_masks:
            m = np.asarray(m)
            if m.dtype == bool:
                patients.append(~m)
            else:
                patients.append(m == 1)

        vol = np.stack(patients, axis=0).astype(bool)
        z, h, w = vol.shape
        areas = vol.reshape(z, -1).sum(axis=1).astype(np.int64)

        # Pass 1: fill weak slices from neighborhood OR
        for i in range(z):
            if areas[i] >= min_area:
                continue
            neigh = []
            for j in range(max(0, i - z_radius), min(z, i + z_radius + 1)):
                if j == i:
                    continue
                if areas[j] >= min_area:
                    neigh.append(vol[j])
            if not neigh:
                continue
            merged = np.zeros((h, w), dtype=bool)
            for n in neigh:
                merged |= n
            vol[i] = binary_fill_holes(merged)
            areas[i] = int(vol[i].sum())

        # Pass 2: closing along Z to bridge residual gaps
        if z >= 3:
            kz = 2 * z_radius + 1
            struct = np.zeros((kz, 1, 1), dtype=bool)
            struct[:, 0, 0] = True
            vol = nd_binary_closing(vol, structure=struct)
            for i in range(z):
                vol[i] = binary_fill_holes(vol[i])
                # light cleanup only — keep small filled bridges
                vol[i] = morphology.remove_small_objects(
                    vol[i], min_size=max(200, min_area // 4)
                )

        return [np.where(vol[i], 1, 0).astype(np.int32) for i in range(z)]

    @staticmethod
    def anatomical_region_masks_from_slices(
        ct: np.ndarray,
        slices: list,
        *,
        enforce_symmetry: bool = True,
        enforce_continuity: bool = True,
        min_area: int = 1500,
        sagittal_flip_axis: int = 0,
    ) -> list:
        """
        Full anatomical-region pipeline for a Z-crop:

        1. per-slice ``body_mask_from_intensity`` (+ optional L/R symmetry)
        2. optional Z continuity to avoid missing slices

        Returns list of int masks (0=background, 1=anatomical), one per slice index.
        """
        bg_bools = [
            ImageProcessor.body_mask_from_intensity(
                ct[:, :, z],
                min_area=min_area,
                enforce_symmetry=enforce_symmetry,
                sagittal_flip_axis=sagittal_flip_axis,
            )
            for z in slices
        ]
        masks = [np.where(bg, 0, 1).astype(np.int32) for bg in bg_bools]
        if enforce_continuity and len(masks) >= 2:
            # Lower bar so thin shoulder slices are not treated as "ok empty"
            masks = ImageProcessor.enforce_anatomical_continuity(
                masks,
                min_area=max(400, min_area // 3),
                z_radius=2,
            )
        return masks

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

