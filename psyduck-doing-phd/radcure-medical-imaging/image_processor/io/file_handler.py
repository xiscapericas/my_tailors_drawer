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
    def _has_dcm_files(folder: str) -> bool:
        return any(
            f.lower().endswith(".dcm")
            for f in os.listdir(folder)
            if not f.startswith(".")
        )

    @staticmethod
    def _first_dicom_modality(folder: str) -> Optional[str]:
        """Read Modality from the first .dcm file in a folder."""
        try:
            import pydicom
        except ImportError:
            return None
        for name in sorted(os.listdir(folder)):
            if not name.lower().endswith(".dcm"):
                continue
            path = os.path.join(folder, name)
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
            mod = getattr(ds, "Modality", None)
            return str(mod).upper() if mod else None
        return None

    @staticmethod
    def _largest_ct_series_file_count(folder: str) -> int:
        """Return slice count of the largest DICOM series in folder (0 if none)."""
        try:
            import SimpleITK as sitk
        except ImportError:
            return 0
        reader = sitk.ImageSeriesReader()
        try:
            series_ids = reader.GetGDCMSeriesIDs(folder)
        except Exception:
            return 0
        if not series_ids:
            return 0
        best = 0
        for series_id in series_ids:
            try:
                n = len(reader.GetGDCMSeriesFileNames(folder, series_id))
            except Exception:
                continue
            best = max(best, n)
        return best

    @staticmethod
    def get_ct_and_mask_paths(dicom_folder_path: str) -> Dict[str, str]:
        """
        Get CT and RTSTRUCT folder paths from a RADCURE DICOM study folder.

        Uses DICOM Modality and series size — not alphabetical order (CT vs RTSTRUCT
        sort order was wrong and caused empty series_ids on convert).
        """
        root = dicom_folder_path.rstrip("/") + "/"
        subdirs = sorted(
            os.path.join(root, name)
            for name in os.listdir(root)
            if not name.startswith(".") and os.path.isdir(os.path.join(root, name))
        )
        if len(subdirs) < 2:
            raise ValueError(
                f"Expected at least 2 subfolders in {root}, "
                f"found {[os.path.basename(d) for d in subdirs]}"
            )

        ct_candidates: List[tuple] = []
        rtstruct_candidates: List[str] = []

        for folder in subdirs:
            folder_slash = folder.rstrip("/") + "/"
            modality = FileHandler._first_dicom_modality(folder)
            series_slices = FileHandler._largest_ct_series_file_count(folder)
            label = os.path.basename(folder.rstrip("/"))

            if modality == "CT" or series_slices >= 2:
                ct_candidates.append((series_slices, folder_slash, label, modality))
            if modality == "RTSTRUCT" or (
                modality not in ("CT",) and series_slices == 0 and FileHandler._has_dcm_files(folder)
            ):
                rtstruct_candidates.append(folder_slash)

        ct_path = None
        mask_path = None

        if ct_candidates:
            ct_candidates.sort(key=lambda x: x[0], reverse=True)
            ct_path = ct_candidates[0][1]
            print(
                f"Detected CT folder: {ct_candidates[0][2]} "
                f"(modality={ct_candidates[0][3]}, slices={ct_candidates[0][0]})"
            )

        if rtstruct_candidates:
            mask_path = rtstruct_candidates[0]
            print(f"Detected RTSTRUCT folder: {os.path.basename(mask_path.rstrip('/'))}")

        # Two folders: if only one side detected, assign the other by exclusion
        if len(subdirs) == 2:
            a, b = (d.rstrip("/") + "/" for d in subdirs)
            if ct_path and not mask_path:
                mask_path = b if ct_path.rstrip("/") == a.rstrip("/") else a
            elif mask_path and not ct_path:
                ct_path = b if mask_path.rstrip("/") == a.rstrip("/") else a

        if not ct_path or not mask_path:
            # Legacy fallback (alphabetical) with explicit warning
            names = sorted(
                f for f in os.listdir(root) if not f.startswith(".") and os.path.isdir(os.path.join(root, f))
            )
            print(
                "WARNING: Could not detect CT/RTSTRUCT by modality; "
                f"falling back to sorted order {names} (mask=first, ct=second)."
            )
            mask_path = os.path.join(root, names[0]) + "/"
            ct_path = os.path.join(root, names[1]) + "/"

        if ct_path.rstrip("/") == mask_path.rstrip("/"):
            raise ValueError(
                f"CT and RTSTRUCT resolved to the same folder under {root}. "
                f"Subfolders: {[os.path.basename(d) for d in subdirs]}"
            )

        return {"ct_path": ct_path, "mask_path": mask_path}
    
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

    @staticmethod
    def get_dicom_mask_file_path(mask_folder_path: str, mask_filename: Optional[str] = None) -> str:
        """
        Get the full path to the DICOM mask file (RTSTRUCT).
        
        Parameters
        ----------
        mask_folder_path : str
            Path to the mask folder containing the RTSTRUCT file
        mask_filename : str, optional
            Specific mask filename (e.g., '1-1.dcm'). If None, finds the first .dcm file.
        
        Returns
        -------
        str
            Full path to the DICOM mask file
        """
        if mask_filename:
            mask_file_path = os.path.join(mask_folder_path, mask_filename)
            if not os.path.exists(mask_file_path):
                raise FileNotFoundError(f"Mask file not found: {mask_file_path}")
            return mask_file_path
        
        # Find the first .dcm file in the mask folder
        files = [f for f in os.listdir(mask_folder_path) if not f.startswith(".")]
        dcm_files = [f for f in files if f.lower().endswith('.dcm')]
        
        if not dcm_files:
            raise FileNotFoundError(f"No DICOM file found in {mask_folder_path}")
        
        # If there's a file named '1-1.dcm', prefer it, otherwise use the first one
        preferred_file = '1-1.dcm'
        if preferred_file in dcm_files:
            mask_file_path = os.path.join(mask_folder_path, preferred_file)
        else:
            mask_file_path = os.path.join(mask_folder_path, dcm_files[0])
        
        return mask_file_path

