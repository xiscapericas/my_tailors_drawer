"""DICOM handling for CT and RTSTRUCT files."""

import os
import numpy as np
from rt_utils import RTStructBuilder
from typing import Dict, Optional


class DICOMHandler:
    """Handles DICOM operations for CT and RTSTRUCT files."""
    
    @staticmethod
    def load_mask(
        ct_folder_path: str,
        mask_path: str,
        get_all_struct: bool = False,
        structure_name: str = 'GTVp'
    ) -> Dict[str, np.ndarray]:
        """
        Load a binary mask (or multiple masks) from an RTSTRUCT file.
        
        Parameters
        ----------
        ct_folder_path : str
            Path to the folder containing the DICOM CT series
        mask_path : str
            Path to the RTSTRUCT DICOM file
        get_all_struct : bool
            Whether to load all available structures
        structure_name : str
            Name of the structure to extract if get_all_struct is False
        
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary where keys are structure names and values are 3D masks
        """
        # Build RTSTRUCT object
        rtstruct = RTStructBuilder.create_from(ct_folder_path, mask_path)
        print("Available structures:", rtstruct.get_roi_names())
        
        contours_dict = {}
        
        if get_all_struct:
            print('Loading all structures...')
            for roi_name in rtstruct.get_roi_names():
                mask = rtstruct.get_roi_mask_by_name(roi_name)
                contours_dict[roi_name] = mask
        else:
            print(f'Loading only: {structure_name}')
            mask = rtstruct.get_roi_mask_by_name(structure_name)
            contours_dict[structure_name] = mask
        
        return contours_dict
    
    @staticmethod
    def load_tumor_mask(ct_folder_path: str, mask_folder_path: str) -> np.ndarray:
        """
        Load tumor mask (GTVp) from RTSTRUCT folder.
        
        Parameters
        ----------
        ct_folder_path : str
            Path to CT DICOM folder
        mask_folder_path : str
            Path to RTSTRUCT folder
        
        Returns
        -------
        np.ndarray
            3D tumor mask array
        """
        files = [f for f in os.listdir(mask_folder_path) if not f.startswith(".")]
        if not files:
            raise FileNotFoundError(f"No RTSTRUCT file found in {mask_folder_path}")
        tumor_mask_path = os.path.join(mask_folder_path, files[0])
        
        contours_dict = DICOMHandler.load_mask(
            ct_folder_path,
            tumor_mask_path,
            get_all_struct=False,
            structure_name='GTVp'
        )
        return contours_dict['GTVp']

