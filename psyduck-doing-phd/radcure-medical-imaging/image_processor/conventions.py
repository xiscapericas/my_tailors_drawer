"""Dataset conventions for RADCURE and HECKTOR."""

import os
from typing import Dict, List, Optional

# Convention names
RADCURE = "radcure"
HECKTOR = "hecktor"

# Tumor label modes (Test1–3 used merged; Test4+ can use separate)
TUMOR_LABEL_MODE_MERGED = "merged"
TUMOR_LABEL_MODE_SEPARATE = "separate"


def get_tumor_source_labels(convention: str) -> List[int]:
    """
    Return mask label values to merge into a single tumor index.

    - RADCURE: single GTVp label -> [1]
    - HECKTOR: GTVp (1) + GTVn (2) -> [1, 2]
    """
    if convention == HECKTOR:
        return [1, 2]
    return [1]


def get_tumor_source_label_mapping(convention: str) -> Dict[int, str]:
    """
    Map source tumor mask values to organ names (separate-tumor mode).

    Source encoding (RADCURE RTSTRUCT → aligned NIfTI, or HECKTOR mask file):
      1 = GTVp, 2 = GTVn
    """
    if convention not in (RADCURE, HECKTOR):
        raise ValueError(f"Unknown convention: {convention}")
    return {1: "GTVp", 2: "GTVn"}


def get_nnunet_case_number(case_id: str, convention: str) -> str:
    """
    Extract nnUNet-style case number from case_id for output filenames.

    - RADCURE: 'RADCURE-0005' -> '0005'
    - HECKTOR: 'CHUM-001' -> '001' (part after last '-')
    """
    if convention == RADCURE and "-" in case_id:
        return case_id.split("-", 1)[1]
    if convention == HECKTOR and "-" in case_id:
        return case_id.split("-")[-1]
    return case_id.replace("-", "_")


def get_hecktor_paths(case_folder: str, case_id: str) -> dict:
    """
    Return CT, PET, and mask paths for a HECKTOR case.

    HECKTOR layout:
      {case_folder}/{case_id}__CT.nii.gz
      {case_folder}/{case_id}__PT.nii.gz  (SUV PET; unused before Test 8.0)
      {case_folder}/{case_id}.nii.gz      (labels: GTVp=1, GTVn=2)
    """
    path_ct = os.path.join(case_folder, f"{case_id}__CT.nii.gz")
    path_pet = os.path.join(case_folder, f"{case_id}__PT.nii.gz")
    path_mask = os.path.join(case_folder, f"{case_id}.nii.gz")
    return {"path_ct": path_ct, "path_pet": path_pet, "path_mask": path_mask}
