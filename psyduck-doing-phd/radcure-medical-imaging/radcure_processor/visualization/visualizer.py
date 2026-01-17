"""Visualization utilities for medical images and masks."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.backends.backend_pdf import PdfPages
from typing import Optional, List, Dict
import SimpleITK as sitk
import json
import os
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
        show: bool = True,
        slice_indices: Optional[List[int]] = None
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
        slice_indices : List[int], optional
            Specific slice indices to visualize. If None, shows all slices.
        """
        img_vol = NIfTIHandler.load_nii_image(path_image)
        mask_vol = NIfTIHandler.load_nii_mask(path_mask)
        
        cmap_mask, norm_mask = MedicalImageVisualizer.make_label_cmap(mask_vol)
        
        if img_vol.shape != mask_vol.shape:
            print("⚠️ Warning: volumes have different shapes:",
                  img_vol.shape, mask_vol.shape)
        
        num_slices = min(img_vol.shape[axis], mask_vol.shape[axis])
        
        # Determine which slices to show
        if slice_indices is None:
            slices_to_show = list(range(num_slices))
        else:
            # Filter slice indices to valid range
            slices_to_show = [idx for idx in slice_indices if 0 <= idx < num_slices]
            if len(slices_to_show) != len(slice_indices):
                invalid = [idx for idx in slice_indices if idx not in slices_to_show]
                print(f"⚠️ Warning: Invalid slice indices {invalid} (valid range: 0-{num_slices-1})")
        
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
            for slice_idx in slices_to_show:
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
    
    @staticmethod
    def visualize_prediction_comparison(
        case_id: str,
        images_folder: str,
        labels_folder: str,
        predicted_labels_folder: str,
        organ_dictionary_path: str,
        axis: int = 2,
        save_pdf_path: Optional[str] = None,
        show: bool = True,
        slice_indices: Optional[List[int]] = None
    ) -> None:
        """
        Visualize CT image, ground truth mask, and predicted mask side by side.
        
        Parameters
        ----------
        case_id : str
            Case identifier (e.g., "case_0405")
        images_folder : str
            Path to folder containing images (e.g., "imagesTs")
        labels_folder : str
            Path to folder containing ground truth labels (e.g., "labelsTs")
        predicted_labels_folder : str
            Path to folder containing predicted labels (e.g., "labelsTs_predicted")
        organ_dictionary_path : str
            Path to JSON file containing organ dictionary mapping organ names to indices
        axis : int
            Axis to slice along: 0 = sagittal, 1 = coronal, 2 = axial (default)
        save_pdf_path : str, optional
            If provided, saves all slice figures into one PDF
        show : bool
            If True, displays figures interactively
        slice_indices : List[int], optional
            Specific slice indices to visualize. If None, shows all slices.
        """
        # Load organ dictionary
        if not os.path.exists(organ_dictionary_path):
            raise FileNotFoundError(
                f"Organ dictionary not found at {organ_dictionary_path}"
            )
        
        with open(organ_dictionary_path, 'r') as f:
            organ_dict: Dict[str, int] = json.load(f)
        
        # Invert dictionary: {organ_name: index} -> {index: organ_name}
        index_to_organ: Dict[int, str] = {v: k for k, v in organ_dict.items()}
        
        # Construct file paths
        # Images have _0000 suffix, labels don't
        image_path = os.path.join(images_folder, f"{case_id}_0000.nii.gz")
        label_path = os.path.join(labels_folder, f"{case_id}.nii.gz")
        predicted_path = os.path.join(predicted_labels_folder, f"{case_id}.nii.gz")
        
        # Check if files exist
        for path, name in [(image_path, "image"), (label_path, "ground truth"), 
                          (predicted_path, "predicted")]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name} file not found: {path}")
        
        # Load volumes
        img_vol = NIfTIHandler.load_nii_image(image_path)
        gt_mask_vol = NIfTIHandler.load_nii_mask(label_path)
        pred_mask_vol = NIfTIHandler.load_nii_mask(predicted_path)
        
        # Determine maximum label across both masks for unified colormap
        max_label = max(int(gt_mask_vol.max()), int(pred_mask_vol.max()))
        if max_label <= 0:
            max_label = 1
        
        # Find GTVp index if it exists
        gtvp_index = organ_dict.get('GTVp', None)
        if gtvp_index is not None:
            print(f"✓ Found GTVp at index {gtvp_index}")
        else:
            print("⚠️ Warning: GTVp not found in organ dictionary")
        
        # Ensure colormap has enough entries for all labels including GTVp
        # Use max of actual max_label and gtvp_index to ensure enough colors
        if gtvp_index is not None:
            colormap_size = max(max_label + 1, gtvp_index + 1)
        else:
            colormap_size = max_label + 1
        
        # Create unified colormap (same for both masks)
        # Use tab20 colormap but ensure GTVp is always RED
        base_cmap = plt.cm.get_cmap("tab20", colormap_size)
        colors = base_cmap(np.arange(colormap_size))
        colors[0, :] = [0.0, 0.0, 0.0, 0.0]  # transparent background
        
        # Force GTVp to be RED and ensure no other organ uses red
        if gtvp_index is not None and gtvp_index < colormap_size:
            # Set GTVp to pure red: [1.0, 0.0, 0.0, 1.0]
            colors[gtvp_index, :] = [1.0, 0.0, 0.0, 1.0]
            print(f"✓ Set GTVp (index {gtvp_index}) to RED color")
            
            # Ensure no other organ uses red (replace any red-like colors)
            red_threshold = 0.7  # Consider colors with R > 0.7 as "red-like"
            replaced_count = 0
            for i in range(1, colormap_size):
                if i != gtvp_index and colors[i, 0] > red_threshold:
                    # Replace with a different color from a different colormap
                    alt_cmap = plt.cm.get_cmap("Set3", colormap_size)
                    alt_colors = alt_cmap(np.arange(colormap_size))
                    # Use the alternative color but keep alpha
                    colors[i, :3] = alt_colors[i, :3]
                    colors[i, 3] = 1.0  # Keep full opacity
                    replaced_count += 1
            if replaced_count > 0:
                print(f"✓ Replaced {replaced_count} red-like colors to avoid conflict with GTVp")
        
        cmap_mask = ListedColormap(colors)
        boundaries = np.arange(-0.5, colormap_size + 0.5, 1)
        norm_mask = BoundaryNorm(boundaries, cmap_mask.N)
        
        # Check shapes
        if img_vol.shape != gt_mask_vol.shape:
            print("⚠️ Warning: image and ground truth have different shapes:",
                  img_vol.shape, gt_mask_vol.shape)
        if img_vol.shape != pred_mask_vol.shape:
            print("⚠️ Warning: image and predicted mask have different shapes:",
                  img_vol.shape, pred_mask_vol.shape)
        
        num_slices = min(img_vol.shape[axis], gt_mask_vol.shape[axis], 
                        pred_mask_vol.shape[axis])
        
        # Determine which slices to show
        if slice_indices is None:
            slices_to_show = list(range(num_slices))
        else:
            slices_to_show = [idx for idx in slice_indices if 0 <= idx < num_slices]
            if len(slices_to_show) != len(slice_indices):
                invalid = [idx for idx in slice_indices if idx not in slices_to_show]
                print(f"⚠️ Warning: Invalid slice indices {invalid} "
                      f"(valid range: 0-{num_slices-1})")
        
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
        
        # Get unique labels present in either mask (for overall legend - will be filtered per slice)
        unique_labels_gt = set(np.unique(gt_mask_vol).astype(int))
        unique_labels_pred = set(np.unique(pred_mask_vol).astype(int))
        unique_labels_all = sorted(unique_labels_gt.union(unique_labels_pred))
        # Remove background (0)
        unique_labels_all = [l for l in unique_labels_all if l > 0]
        
        pdf = PdfPages(save_pdf_path) if save_pdf_path else None
        
        try:
            for slice_idx in slices_to_show:
                img_slice = get_slice(img_vol, slice_idx, axis)
                gt_slice = get_slice(gt_mask_vol, slice_idx, axis)
                pred_slice = get_slice(pred_mask_vol, slice_idx, axis)
                
                # Get unique labels present in THIS specific slice
                unique_labels_slice_gt = set(np.unique(gt_slice).astype(int))
                unique_labels_slice_pred = set(np.unique(pred_slice).astype(int))
                unique_labels_slice = sorted(unique_labels_slice_gt.union(unique_labels_slice_pred))
                # Remove background (0) from legend
                unique_labels_slice = [l for l in unique_labels_slice if l > 0]
                
                # Create figure with 3 panels + space for legend
                fig = plt.figure(figsize=(18, 6))
                gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.4], hspace=0.3)
                
                ax_img = fig.add_subplot(gs[0])
                ax_gt = fig.add_subplot(gs[1])
                ax_pred = fig.add_subplot(gs[2])
                ax_legend = fig.add_subplot(gs[3])
                
                # Image panel
                ax_img.imshow(img_slice, cmap="gray")
                ax_img.set_title(f"CT Image\n(slice {slice_idx})", fontsize=12)
                ax_img.axis("off")
                
                # Ground truth panel
                ax_gt.imshow(gt_slice, cmap=cmap_mask, norm=norm_mask)
                ax_gt.set_title("Ground Truth Mask", fontsize=12)
                ax_gt.axis("off")
                
                # Predicted mask panel
                ax_pred.imshow(pred_slice, cmap=cmap_mask, norm=norm_mask)
                ax_pred.set_title("Predicted Mask", fontsize=12)
                ax_pred.axis("off")
                
                # Create legend with organ names - only for organs present in this slice
                ax_legend.axis("off")
                legend_patches = []
                seen_colors = set()  # Track colors to avoid duplicates
                
                for label_idx in unique_labels_slice:
                    organ_name = index_to_organ.get(label_idx, f"Label {label_idx}")
                    # Get color directly from our colors array to ensure GTVp is red
                    if label_idx < len(colors):
                        color = colors[label_idx]
                    else:
                        # Fallback: get from colormap
                        color = cmap_mask(norm_mask(label_idx))
                    
                    # Ensure GTVp is always red in legend (double-check)
                    if organ_name == 'GTVp' and gtvp_index is not None and label_idx == gtvp_index:
                        color = [1.0, 0.0, 0.0, 1.0]  # Pure red
                    
                    # Convert color to tuple for comparison (avoid duplicates)
                    color_tuple = tuple(color[:3])  # RGB only, ignore alpha
                    
                    # Skip if we've already seen this exact color (avoid duplicate entries)
                    if color_tuple not in seen_colors:
                        seen_colors.add(color_tuple)
                        patch = mpatches.Patch(facecolor=color, edgecolor='black', 
                                             linewidth=0.5, label=organ_name)
                        legend_patches.append(patch)
                
                if legend_patches:
                    ax_legend.legend(handles=legend_patches, loc='center left',
                                   fontsize=9, framealpha=0.9)
                    ax_legend.set_title("Organs", fontsize=11, pad=10)
                
                fig.suptitle(f"{case_id} - Slice {slice_idx}/{num_slices-1}", 
                           fontsize=14, y=0.98)
                fig.tight_layout(rect=[0, 0, 0.95, 0.96])
                
                if pdf is not None:
                    pdf.savefig(fig, bbox_inches="tight")
                
                if show:
                    plt.show()
                    plt.close(fig)
                else:
                    plt.close(fig)
        
        finally:
            if pdf is not None:
                pdf.close()

