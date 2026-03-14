"""Dataset conventions for RADCURE and HECKTOR."""

import os
from typing import List, Optional

# Convention names
RADCURE = "radcure"
HECKTOR = "hecktor"


def get_tumor_source_labels(convention: str) -> List[int]:
    """
    Return mask label values to merge into a single tumor index.

    - RADCURE: single GTVp label -> [1]
    - HECKTOR: GTVp (1) + GTVn (2) -> [1, 2]
    """
    if convention == HECKTOR:
        return [1, 2]
    return [1]


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
    Return CT and mask paths for a HECKTOR case.

    HECKTOR layout: {case_folder}/{case_id}__CT.nii.gz, {case_folder}/{case_id}.nii.gz
    """
    path_ct = os.path.join(case_folder, f"{case_id}__CT.nii.gz")
    path_mask = os.path.join(case_folder, f"{case_id}.nii.gz")
    return {"path_ct": path_ct, "path_mask": path_mask}
