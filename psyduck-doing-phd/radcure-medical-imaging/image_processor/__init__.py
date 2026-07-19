"""
Image processor - Dataset-agnostic medical image pipeline (RADCURE, HECKTOR)
with TotalSegmentator for tumor detection model training.
"""

from image_processor.core.case_processor import CaseProcessor, AnatomyQCRejected
from image_processor.core.dicom_handler import DICOMHandler
from image_processor.core.segmentator import TotalSegmentatorWrapper
from image_processor.core.mask_generator import MaskGenerator
from image_processor.conventions import (
    RADCURE,
    HECKTOR,
    TUMOR_LABEL_MODE_MERGED,
    TUMOR_LABEL_MODE_SEPARATE,
)

__version__ = "0.1.0"
__all__ = [
    "CaseProcessor",
    "AnatomyQCRejected",
    "DICOMHandler",
    "TotalSegmentatorWrapper",
    "MaskGenerator",
    "RADCURE",
    "HECKTOR",
    "TUMOR_LABEL_MODE_MERGED",
    "TUMOR_LABEL_MODE_SEPARATE",
]

