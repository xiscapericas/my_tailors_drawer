"""Visualization utilities for medical images and masks."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.backends.backend_pdf import PdfPages
from typing import Optional
from radcure_processor.io.nifti_handler import NIfTIHandler


class MedicalImageVisualizer:
    """Visualization utilities for medical images."""
    
    @staticmethod
    def make_label_cmap(mask: np.ndarray) -> tuple:
        """
        Create a discrete colormap and norm for a label mask.
        
        Parameters
        ----------
        mask : np.ndarray
            3D mask array
        
        Returns
        -------
        tuple
            (colormap, norm) tuple
        """
        max_label = int(mask.max())
        if max_label <= 0:
            max_label = 1
        
        base_cmap = plt.cm.get_cmap("tab20", max_label + 1)
        colors = base_cmap(np.arange(max_label + 1))
        colors[0, :] = [0.0, 0.0, 0.0, 0.0]  # transparent background
        
        cmap = ListedColormap(colors)
        boundaries = np.arange(-0.5, max_label + 1.5, 1)
        norm = BoundaryNorm(boundaries, cmap.N)
        
        return cmap, norm
    
    @staticmethod
    def show_two_niis_side_by_side(
        path_image: str,
        path_mask: str,
        axis: int = 2,
        save_pdf_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Visualize two NIfTI volumes slice-by-slice with 3 panels:
        Image / Mask / Overlay. Optionally save all slices to a single PDF.
        
        Parameters
        ----------
        path_image : str
            Path to image volume (.nii or .nii.gz)
        path_mask : str
            Path to mask volume (.nii or .nii.gz)
        axis : int
            Axis to slice along: 0 = sagittal, 1 = coronal, 2 = axial (default)
        save_pdf_path : str, optional
            If provided, saves all slice figures into one PDF
        show : bool
            If True, displays figures interactively
        """
        img_vol = NIfTIHandler.load_nii_image(path_image)
        mask_vol = NIfTIHandler.load_nii_mask(path_mask)
        
        cmap_mask, norm_mask = MedicalImageVisualizer.make_label_cmap(mask_vol)
        
        if img_vol.shape != mask_vol.shape:
            print("⚠️ Warning: volumes have different shapes:",
                  img_vol.shape, mask_vol.shape)
        
        num_slices = min(img_vol.shape[axis], mask_vol.shape[axis])
        
        def get_slice(vol, idx, axis):
            if axis == 0:
                sl = vol[idx, :, :]
            elif axis == 1:
                sl = vol[:, idx, :]
            elif axis == 2:
                sl = vol[:, :, idx]
            else:
                raise ValueError("axis must be 0, 1, or 2")
            return np.rot90(sl)
        
        pdf = PdfPages(save_pdf_path) if save_pdf_path else None
        
        try:
            for slice_idx in range(num_slices):
                img_slice = get_slice(img_vol, slice_idx, axis)
                mask_slice = get_slice(mask_vol, slice_idx, axis)
                
                fig, (ax_img, ax_mask, ax_overlay) = plt.subplots(1, 3, figsize=(15, 5))
                
                # Image panel
                ax_img.imshow(img_slice, cmap="gray")
                ax_img.set_title(f"Image (slice {slice_idx})")
                ax_img.axis("off")
                
                # Mask panel
                ax_mask.imshow(mask_slice, cmap=cmap_mask, norm=norm_mask)
                ax_mask.set_title("Mask (labels)")
                ax_mask.axis("off")
                
                # Overlay panel
                ax_overlay.imshow(img_slice, cmap="gray")
                ax_overlay.imshow(
                    mask_slice, cmap=cmap_mask, norm=norm_mask, alpha=0.4
                )
                ax_overlay.set_title("Overlay")
                ax_overlay.axis("off")
                
                fig.tight_layout()
                
                if pdf is not None:
                    pdf.savefig(fig, bbox_inches="tight")
                
                if show:
                    plt.show()
                else:
                    plt.close(fig)
                
                if show:
                    plt.close(fig)
        
        finally:
            if pdf is not None:
                pdf.close()

