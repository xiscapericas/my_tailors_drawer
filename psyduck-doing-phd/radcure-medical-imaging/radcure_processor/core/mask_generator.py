"""Mask generation and combination utilities."""

import os
import numpy as np
import nibabel as nib
from typing import List, Dict, Tuple
from radcure_processor.utils.organ_dictionary import OrganDictionary
from radcure_processor.utils.image_processing import ImageProcessor
from radcure_processor.io.nifti_handler import NIfTIHandler


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
        Generate background mask array for CT slices.
        
        Parameters
        ----------
        non_zero_tumor_mask_expanded : List[int]
            List of slice indices
        nifti_output_path : str
            Path to NIfTI CT file
        
        Returns
        -------
        List[np.ndarray]
            List of background masks (1 = background, 0 = head)
        """
        background_array = []
        nii_image = NIfTIHandler.load_nii_image(nifti_output_path)
        
        for slice_ind in non_zero_tumor_mask_expanded:
            img = nii_image[:, :, slice_ind]
            background_mask = ImageProcessor.head_mask_from_array(img)
            # Convert to int (1 = background, 0 = head)
            background_array.append(background_mask.astype(int))
        
        return background_array
    
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
        
        # Initialize combined mask array with background
        combined_mask_array = background_array_int.copy()
        
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
                # Get or assign organ index
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
        
        # Update organ dictionary
        self.organ_dictionary.dictionary = organs_dic
        
        return combined_mask_array, self.organ_dictionary
    
    def update_combined_mask_with_tumor(
        self,
        tumor_contours: np.ndarray,
        slices_to_use: List[int],
        combined_mask_array: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Update combined mask with tumor (GTVp) annotations.
        
        Parameters
        ----------
        tumor_contours : np.ndarray
            3D tumor mask array
        slices_to_use : List[int]
            List of slice indices
        combined_mask_array : List[np.ndarray]
            Combined mask array to update
        
        Returns
        -------
        List[np.ndarray]
            Updated combined mask array with tumor
        """
        # Ensure tumor index exists
        tumor_value = self.organ_dictionary.add_tumor_index()
        
        # Get tumor masks for slices of interest
        tumor_masks = tumor_contours[:, :, slices_to_use]
        
        print(f'Using index {tumor_value} for tumor')
        for ind in range(len(combined_mask_array)):
            combined_mask = combined_mask_array[ind]
            # Rotate tumor mask to match orientation
            tumor_mask = np.rot90(tumor_masks[:, :, ind], k=1)
            combined_mask[tumor_mask == 1] = tumor_value
            combined_mask_array[ind] = combined_mask
        
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

