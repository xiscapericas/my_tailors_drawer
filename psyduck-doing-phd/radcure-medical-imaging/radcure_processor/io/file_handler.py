"""File handling utilities for zip, DICOM paths, etc."""

import os
import zipfile
import shutil
from typing import Dict, List, Optional


class FileHandler:
    """Handles file operations like unzipping, path resolution, etc."""
    
    @staticmethod
    def unzip_file(zip_path: str, output_folder: str) -> bool:
        """
        Unzip a file to a specified output folder.
        
        Parameters
        ----------
        zip_path : str
            Path to zip file
        output_folder : str
            Output folder for extracted files
        
        Returns
        -------
        bool
            True if successful
        """
        os.makedirs(output_folder, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(output_folder)
        return True
    
    @staticmethod
    def get_dicom_path(local_folder: str, radcure_case_id: str) -> str:
        """
        Get the DICOM folder path for a RADCURE case.
        
        Parameters
        ----------
        local_folder : str
            Base folder containing the extracted case
        radcure_case_id : str
            Case ID (e.g., 'RADCURE-0005')
        
        Returns
        -------
        str
            Path to DICOM folder
        """
        base = os.path.join(local_folder, "RADCURE", radcure_case_id)
        entries = sorted(f for f in os.listdir(base) if not f.startswith("."))
        if not entries:
            raise FileNotFoundError(f"No visible DICOM subfolder in {base}")
        return os.path.join(base, entries[0]) + "/"
    
    @staticmethod
    def get_ct_and_mask_paths(dicom_folder_path: str) -> Dict[str, str]:
        """
        Get CT and mask folder paths from DICOM folder.
        
        Parameters
        ----------
        dicom_folder_path : str
            Path to DICOM folder
        
        Returns
        -------
        Dict[str, str]
            Dictionary with 'ct_path' and 'mask_path' keys
        """
        dicom_folders = sorted(
            f for f in os.listdir(dicom_folder_path)
            if not f.startswith('.')
        )
        if len(dicom_folders) < 2:
            raise ValueError(
                f"Expected at least 2 subfolders in {dicom_folder_path}, "
                f"found {dicom_folders}"
            )
        
        return {
            "ct_path": os.path.join(dicom_folder_path, dicom_folders[1]) + "/",
            "mask_path": os.path.join(dicom_folder_path, dicom_folders[0]) + "/",
        }
    
    @staticmethod
    def get_number_from_name(text: str) -> Optional[str]:
        """
        Extract number from filename (e.g., 'RADCURE-0005' -> '0005').
        
        Parameters
        ----------
        text : str
            Filename or text containing pattern
        
        Returns
        -------
        str or None
            Extracted number or None if not found
        """
        import re
        match = re.search(r"\d+-(\d+)", text)
        return match.group(1) if match else None
    
    @staticmethod
    def get_ct_slices_to_keep(
        non_zero_tumor_mask_expanded: List[int],
        dicom_folder_ct_path: str
    ) -> List[str]:
        """
        Get list of DICOM slice filenames to keep based on slice indices.
        
        Parameters
        ----------
        non_zero_tumor_mask_expanded : List[int]
            List of slice indices to keep
        dicom_folder_ct_path : str
            Path to CT DICOM folder
        
        Returns
        -------
        List[str]
            List of DICOM filenames to keep
        """
        slices_to_keep = []
        dicom_paths_sorted = sorted(os.listdir(dicom_folder_ct_path))
        for filename in dicom_paths_sorted:
            file_number = FileHandler.get_number_from_name(filename)
            if file_number and int(file_number) in non_zero_tumor_mask_expanded:
                slices_to_keep.append(filename)
        return slices_to_keep
    
    @staticmethod
    def cleanup_case_files(case_folder: str, zip_path: Optional[str] = None) -> None:
        """
        Clean up case files (folder and optionally zip file).
        
        Parameters
        ----------
        case_folder : str
            Path to case folder to delete
        zip_path : str, optional
            Path to zip file to delete
        """
        # Delete zip if it exists
        if zip_path and os.path.exists(zip_path):
            os.remove(zip_path)
            print(f"Deleted zip: {zip_path}")
        
        # Delete folder if it exists
        if os.path.exists(case_folder):
            shutil.rmtree(case_folder)
            print(f"Deleted folder: {case_folder}")

