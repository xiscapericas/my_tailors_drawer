"""
RADCURE Processor - A Python package for processing RADCURE DICOM cases
with TotalSegmentator for tumor detection model training.
"""

from radcure_processor.core.case_processor import CaseProcessor
from radcure_processor.core.dicom_handler import DICOMHandler
from radcure_processor.core.segmentator import TotalSegmentatorWrapper
from radcure_processor.core.mask_generator import MaskGenerator

__version__ = "0.1.0"
__all__ = [
    "CaseProcessor",
    "DICOMHandler",
    "TotalSegmentatorWrapper",
    "MaskGenerator",
]

