"""Mask generation and combination utilities."""

import os
import numpy as np
import nibabel as nib
from typing import List, Dict, Tuple, Optional
from image_processor.conventions import TUMOR_LABEL_MODE_MERGED, TUMOR_LABEL_MODE_SEPARATE
from image_processor.utils.organ_dictionary import OrganDictionary
from image_processor.utils.image_processing import ImageProcessor
from image_processor.io.nifti_handler import NIfTIHandler


class MaskGenerator:
    """Generates combined masks from TotalSegmentator outputs and tumor masks."""
    
    def __init__(self, organ_dictionary: OrganDictionary):
        """
        Initialize mask generator.
        
        Parameters
        ----------
        organ_dictionary : OrganDictionary
            Organ dictionary instance
        """
        self.organ_dictionary = organ_dictionary
    
    def generate_background_array(
        self,
        non_zero_tumor_mask_expanded: List[int],
        nifti_output_path: str
    ) -> List[np.ndarray]:
        """
        Generate background/anatomical_region mask array for CT slices.
        
        Parameters
        ----------
        non_zero_tumor_mask_expanded : List[int]
            List of slice indices
        nifti_output_path : str
            Path to NIfTI CT file
        
        Returns
        -------
        List[np.ndarray]
            List of integer masks per slice: 0 = background, 1 = anatomical_region.
        """
        result = []
        nii_image = NIfTIHandler.load_nii_image(nifti_output_path)
        for slice_ind in non_zero_tumor_mask_expanded:
            img = nii_image[:, :, slice_ind]
            background_mask_bool = ImageProcessor.head_mask_from_array(img)
            # Map: background (True) -> 0, anatomical_region (False) -> 1
            slice_mask = np.where(background_mask_bool, 0, 1)
            result.append(slice_mask.astype(np.int32))
        return result
    
    def get_individual_segmentator_paths(
        self,
        main_total_segmentator_path: str
    ) -> List[str]:
        """
        Get all individual organ mask paths from TotalSegmentator output.
        
        Parameters
        ----------
        main_total_segmentator_path : str
            Base path to TotalSegmentator output
        
        Returns
        -------
        List[str]
            List of paths to individual organ mask files
        """
        individual_paths = []
        for sec_dir in os.listdir(main_total_segmentator_path):
            sec_path = os.path.join(main_total_segmentator_path, sec_dir)
            if os.path.isdir(sec_path):
                for file in os.listdir(sec_path):
                    if file.endswith('.nii.gz'):
                        individual_paths.append(os.path.join(sec_path, file))
        return individual_paths
    
    def get_ts_mask(
        self,
        total_s_example_path: str,
        non_zero_tumor_mask_expanded: List[int]
    ) -> np.ndarray:
        """
        Load and filter TotalSegmentator mask for specific slices.
        
        Parameters
        ----------
        total_s_example_path : str
            Path to TotalSegmentator mask file
        non_zero_tumor_mask_expanded : List[int]
            List of slice indices to keep
        
        Returns
        -------
        np.ndarray
            3D mask array filtered to specified slices
        """
        nii = nib.load(total_s_example_path)
        data = nii.get_fdata()
        data_of_interest = data[:, :, non_zero_tumor_mask_expanded]
        return data_of_interest
    
    def generate_combined_mask(
        self,
        non_zero_tumor_mask_expanded: List[int],
        background_array_int: List[np.ndarray],
        main_total_segmentator_path: str
    ) -> Tuple[List[np.ndarray], OrganDictionary]:
        """
        Generate combined mask from TotalSegmentator outputs.
        
        Parameters
        ----------
        non_zero_tumor_mask_expanded : List[int]
            List of slice indices
        background_array_int : List[np.ndarray]
            List of background masks
        main_total_segmentator_path : str
            Path to TotalSegmentator output folder
        
        Returns
        -------
        Tuple[List[np.ndarray], OrganDictionary]
            Combined mask array and updated organ dictionary
        """
        # Copy dictionary
        organs_dic = self.organ_dictionary.copy()
        
        # Get individual organ paths
        individual_paths = self.get_individual_segmentator_paths(
            main_total_segmentator_path
        )
        
        # Initialize combined mask from background array (0 = background, 1 = anatomical_region)
        combined_mask_array = [arr.copy() for arr in background_array_int]
        other_tissue_index = organs_dic.get('other-tissue')
        if other_tissue_index is None:
            other_tissue_index = self.organ_dictionary.add_organ('other-tissue')
            organs_dic['other-tissue'] = other_tissue_index

        print(f'Starting organ index at value {self.organ_dictionary.get_max_index() + 1}')

        # Process each organ mask (in reverse order)
        for organ_path in reversed(individual_paths):
            print(f'Checking: {organ_path}')

            # Get the mask filtered to slices of interest
            organ_mask = self.get_ts_mask(organ_path, non_zero_tumor_mask_expanded)

            # Extract organ name from path
            organ_name = os.path.basename(organ_path).replace('.nii.gz', '')

            # Only process if organ_mask has data
            if np.sum(organ_mask) > 0:
                # Get or assign organ index (organs start after background, anatomical_region, other-tissue)
                if organ_name in organs_dic:
                    organ_index = organs_dic[organ_name]
                    print(f'Organ {organ_name} found in dictionary with index {organ_index}')
                else:
                    organ_index = self.organ_dictionary.add_organ(organ_name)
                    organs_dic[organ_name] = organ_index
                    print(f'Organ {organ_name} added with index {organ_index}')

                # Add organ mask to combined mask
                print(f'Running for {organ_name} with organ index {organ_index}')
                for ind in range(len(combined_mask_array)):
                    tsmask = organ_mask[:, :, ind]
                    combined_mask_array[ind][tsmask == 1] = organ_index
            else:
                print(f'Excluding {organ_name} (no data)')
            print('-------------------')

        # Assign remaining anatomical_region (still 1) to other-tissue
        for ind in range(len(combined_mask_array)):
            combined_mask_array[ind][combined_mask_array[ind] == 1] = other_tissue_index

        # Update organ dictionary
        self.organ_dictionary.dictionary = organs_dic

        return combined_mask_array, self.organ_dictionary
    
    def update_combined_mask_with_tumor(
        self,
        tumor_mask_nifti_path: str,
        slices_to_use: List[int],
        combined_mask_array: List[np.ndarray],
        tumor_source_labels: Optional[List[int]] = None,
        tumor_label_mode: str = TUMOR_LABEL_MODE_MERGED,
        tumor_source_label_mapping: Optional[Dict[int, str]] = None,
    ) -> List[np.ndarray]:
        """
        Update combined mask with tumor annotations from NIfTI mask.

        Modes
        -----
        merged (default, Test1–3):
            Source labels in ``tumor_source_labels`` map to a single GTVp index.
        separate (Test4+):
            Each source label maps to its own organ (1→GTVp, 2→GTVn).

        Parameters
        ----------
        tumor_mask_nifti_path : str
            Path to tumor mask NIfTI (0=bg; 1=GTVp; 2=GTVn when separate)
        slices_to_use : List[int]
            Slice indices
        combined_mask_array : List[np.ndarray]
            Combined mask array to update
        tumor_source_labels : List[int], optional
            Used in merged mode only (e.g. [1] RADCURE, [1, 2] HECKTOR)
        tumor_label_mode : str
            ``merged`` or ``separate``
        tumor_source_label_mapping : Dict[int, str], optional
            Source value → organ name for separate mode (default {1: GTVp, 2: GTVn})

        Returns
        -------
        List[np.ndarray]
            Updated combined mask array
        """
        tumor_mask_vol = NIfTIHandler.load_nii_mask(tumor_mask_nifti_path)
        tumor_masks = tumor_mask_vol[:, :, slices_to_use]

        if tumor_label_mode == TUMOR_LABEL_MODE_SEPARATE:
            mapping = tumor_source_label_mapping or {1: "GTVp", 2: "GTVn"}
            target_indices = self.organ_dictionary.add_tumor_indices(separate_gtvp_gtvn=True)
            for source_label, organ_name in mapping.items():
                target_value = target_indices[organ_name]
                print(f"Using index {target_value} for {organ_name} (source label {source_label})")
                for ind in range(len(combined_mask_array)):
                    tumor_slice = tumor_masks[:, :, ind]
                    combined_mask_array[ind][tumor_slice == source_label] = target_value
            return combined_mask_array

        tumor_value = self.organ_dictionary.add_tumor_index()
        source_labels = tumor_source_labels if tumor_source_labels is not None else [1]
        print(f"Using index {tumor_value} for tumor (merged source labels: {source_labels})")
        for ind in range(len(combined_mask_array)):
            tumor_slice = tumor_masks[:, :, ind]
            combined_mask_array[ind][np.isin(tumor_slice, source_labels)] = tumor_value

        return combined_mask_array
    
    def generate_ct_images(
        self,
        nifti_output_path: str,
        non_zero_tumor_mask_expanded: List[int]
    ) -> List[np.ndarray]:
        """
        Generate CT image slices from NIfTI file.
        
        Parameters
        ----------
        nifti_output_path : str
            Path to NIfTI CT file
        non_zero_tumor_mask_expanded : List[int]
            List of slice indices
        
        Returns
        -------
        List[np.ndarray]
            List of 2D CT image slices
        """
        img_array = []
        nii_image = NIfTIHandler.load_nii_image(nifti_output_path)
        for slice_ind in non_zero_tumor_mask_expanded:
            img = nii_image[:, :, slice_ind]
            img_array.append(img)
        return img_array

