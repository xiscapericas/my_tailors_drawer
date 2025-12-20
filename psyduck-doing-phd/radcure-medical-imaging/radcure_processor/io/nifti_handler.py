"""NIfTI file handling utilities."""

import os
import numpy as np
import nibabel as nib
from typing import List, Optional


class NIfTIHandler:
    """Handles NIfTI file operations."""
    
    @staticmethod
    def load_nii_image(path: str) -> np.ndarray:
        """
        Load a NIfTI image and normalize to [0, 1] using 1–99 percentiles.
        
        Parameters
        ----------
        path : str
            Path to NIfTI file
        
        Returns
        -------
        np.ndarray
            Normalized 3D image array
        """
        nii = nib.load(path)
        data = nii.get_fdata().astype(np.float32)
        
        if data.ndim != 3:
            raise ValueError(f"Expected 3D image, got shape {data.shape}")
        
        low, high = np.percentile(data, (1, 99))
        data = np.clip(data, low, high)
        data = (data - low) / (high - low + 1e-8)
        data = np.nan_to_num(data)
        
        return data
    
    @staticmethod
    def load_nii_mask(path: str) -> np.ndarray:
        """
        Load a NIfTI mask and return integer labels.
        
        Parameters
        ----------
        path : str
            Path to NIfTI mask file
        
        Returns
        -------
        np.ndarray
            Integer mask array
        """
        nii = nib.load(path)
        data = nii.get_fdata()
        
        if data.ndim != 3:
            raise ValueError(f"Expected 3D mask, got shape {data.shape}")
        
        return data.astype(int)
    
    @staticmethod
    def save_as_nii(
        img_array: List[np.ndarray],
        store_path: str,
        radcure_case_id: str,
        dtype_needed: bool = False
    ) -> str:
        """
        Save a list of 2D arrays as a NIfTI file.
        
        Parameters
        ----------
        img_array : List[np.ndarray]
            List of 2D arrays (slices)
        store_path : str
            Directory to save the file
        radcure_case_id : str
            Case ID for filename
        dtype_needed : bool
            If True, convert to uint8
        
        Returns
        -------
        str
            Path to saved file
        """
        os.makedirs(store_path, exist_ok=True)
        
        # Create affine
        affine = np.eye(4)
        
        # Stack and transpose
        img_volume = np.stack(img_array)
        volume_t = np.transpose(img_volume, (1, 2, 0))
        
        if dtype_needed:
            volume_t = volume_t.astype(np.uint8)
        
        # Create and save
        nii = nib.Nifti1Image(volume_t, affine)
        case_id = radcure_case_id.split('-')[1]
        file_name = f'case_{case_id}_0000.nii.gz'
        file_path = os.path.join(store_path, file_name)
        nib.save(nii, file_path)
        
        return file_path
    
    @staticmethod
    def convert_dicom_to_nifti(
        dicom_folder_ct_path: str,
        radcure_case_id: str,
        output_folder: str
    ) -> str:
        """
        Convert DICOM CT series to NIfTI format.
        
        Parameters
        ----------
        dicom_folder_ct_path : str
            Path to DICOM CT folder
        radcure_case_id : str
            Case ID
        output_folder : str
            Output folder for NIfTI file
        
        Returns
        -------
        str
            Path to created NIfTI file
        """
        import SimpleITK as sitk
        
        output_path = os.path.join(output_folder, f'{radcure_case_id}.nii.gz')
        
        # Read the DICOM series
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(dicom_folder_ct_path)
        series_file_names = reader.GetGDCMSeriesFileNames(
            dicom_folder_ct_path, series_ids[0]
        )
        reader.SetFileNames(series_file_names)
        image = reader.Execute()
        
        # Save as NIfTI
        sitk.WriteImage(image, output_path)
        print(f'Image saved in: {output_path}')
        
        return output_path

