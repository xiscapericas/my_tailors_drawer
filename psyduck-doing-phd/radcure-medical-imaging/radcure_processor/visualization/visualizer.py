"""Visualization utilities for medical images and masks."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.backends.backend_pdf import PdfPages
from typing import Optional, List
import SimpleITK as sitk
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
    def visualize_dicom_series(
        dicom_folder_path: str,
        slice_indices: Optional[List[int]] = None,
        axis: int = 0,
        save_path: Optional[str] = None,
        show: bool = True,
        window_level: Optional[tuple] = None
    ) -> np.ndarray:
        """
        Visualize a DICOM CT series directly from folder path.
        
        Parameters
        ----------
        dicom_folder_path : str
            Path to folder containing DICOM CT series
        slice_indices : List[int], optional
            Specific slice indices to visualize. If None, shows slices at 25%, 50%, 75%
        axis : int
            Axis to slice along: 0 = sagittal, 1 = coronal, 2 = axial (default)
        save_path : str, optional
            If provided, saves the figure to this path
        show : bool
            If True, displays the figure interactively
        window_level : tuple, optional
            (window, level) for windowing. If None, uses automatic windowing
        
        Returns
        -------
        np.ndarray
            The loaded CT volume as numpy array
        """
        # Load DICOM series
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(dicom_folder_path)
        
        if not series_ids:
            raise ValueError(f"No DICOM series found in {dicom_folder_path}")
        
        series_file_names = reader.GetGDCMSeriesFileNames(dicom_folder_path, series_ids[0])
        reader.SetFileNames(series_file_names)
        image = reader.Execute()
        
        # Convert to numpy array
        ct_array = sitk.GetArrayFromImage(image)
        print(f"Loaded DICOM series: shape {ct_array.shape}")
        
        # Determine slice indices if not provided
        if slice_indices is None:
            num_slices = ct_array.shape[axis]
            slice_indices = [
                num_slices // 4,
                num_slices // 2,
                3 * num_slices // 4
            ]
        
        # Apply windowing if specified
        if window_level is not None:
            window, level = window_level
            ct_array = np.clip(
                ct_array,
                level - window // 2,
                level + window // 2
            )
            ct_array = (ct_array - (level - window // 2)) / window
        
        # Create visualization
        num_slices_to_show = len(slice_indices)
        fig, axes = plt.subplots(1, num_slices_to_show, figsize=(5 * num_slices_to_show, 5))
        
        if num_slices_to_show == 1:
            axes = [axes]
        
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
        
        for idx, slice_idx in enumerate(slice_indices):
            if slice_idx >= ct_array.shape[axis]:
                print(f"Warning: slice index {slice_idx} exceeds volume size")
                continue
            
            slice_data = get_slice(ct_array, slice_idx, axis)
            
            axes[idx].imshow(slice_data, cmap='gray')
            axes[idx].set_title(f'Slice {slice_idx}/{ct_array.shape[axis]}')
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return ct_array
    
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

